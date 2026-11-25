#!/usr/bin/env python3
"""Fixed job matrix and guarded serial execution for the scaled CrossQ plan."""
import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import metadata as package_metadata
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs/scaled_3tasks_3seeds_5m.yaml"
PAPER_CONFIG_PATH = ROOT / "configs/paper_mujoco.yaml"
OUTPUT_ROOT = ROOT / "artifacts/scaled_reproduction"
SINGLE_RETRY_ROOT = ROOT / "artifacts/scaled_reproduction_retry"
MIN_FREE_DISK_BYTES = 20 * 1024**3
EXPECTED_TASKS = {
    "Hopper": "Hopper-v4",
    "HalfCheetah": "HalfCheetah-v4",
    "HumanoidStandup": "HumanoidStandup-v4",
}
VERSION_NAMES = {
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "mujoco": "mujoco",
}


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def validate_config(config):
    paper = load_config(PAPER_CONFIG_PATH)
    expected = copy.deepcopy(paper)
    expected["tasks"] = EXPECTED_TASKS
    expected["task_provenance"] = {name: paper["task_provenance"][name] for name in EXPECTED_TASKS}
    expected["seeds"] = [0, 1, 2]
    expected["max_steps"] = 5_000_000
    if config != expected:
        raise ValueError("scaled manifest does not exactly match the approved frozen-config subset")
    return config


def build_jobs(config=None, output_root=OUTPUT_ROOT):
    config = validate_config(load_config() if config is None else config)
    output_root = Path(output_root)
    return [
        {
            "task": task,
            "env_id": env_id,
            "seed": seed,
            "steps": config["max_steps"],
            "run_dir": str(output_root / f"{task}-seed{seed}"),
        }
        for task, env_id in config["tasks"].items()
        for seed in config["seeds"]
    ]


def select_batch(jobs, batch):
    if batch == "seed0":
        selected = [job for job in jobs if job["seed"] == 0]
        if len(selected) != 3:
            raise ValueError("seed0 batch must contain exactly three jobs")
        return selected
    if batch == "remaining":
        selected = [job for job in jobs if job["seed"] in (1, 2)]
        if len(selected) != 6:
            raise ValueError("remaining batch must contain exactly six jobs")
        return selected
    raise ValueError("batch must be seed0 or remaining")


def installed_versions():
    versions = {}
    for key, package_name in VERSION_NAMES.items():
        try:
            versions[key] = package_metadata.version(package_name)
        except package_metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def validate_batch_preconditions(config, batch, output_root=OUTPUT_ROOT, version_lookup=None):
    validate_config(config)
    matrix = build_jobs(config, output_root)
    selected = select_batch(matrix, batch)
    existing = [job["run_dir"] for job in selected if Path(job["run_dir"]).exists()]
    if existing:
        raise FileExistsError("selected batch has existing output targets: " + ", ".join(existing))
    _validate_versions(config, version_lookup)
    return selected


def _validate_versions(config, version_lookup=None):
    versions = (installed_versions() if version_lookup is None else version_lookup())
    expected = config["implementation_versions"]
    mismatches = [f"{VERSION_NAMES[key]} expected {version}, found {versions.get(key)}"
                  for key, version in expected.items() if versions.get(key) != version]
    if mismatches:
        raise ValueError("package version mismatch; refusing entire batch: " + "; ".join(mismatches))


def build_single_job(task, seed, config=None):
    config = validate_config(load_config() if config is None else config)
    if task not in ("Hopper", "HalfCheetah") or seed != 0:
        raise ValueError("single retry supports only Hopper or HalfCheetah seed 0")
    return {
        "task": task, "env_id": config["tasks"][task], "seed": seed,
        "steps": config["max_steps"],
        "run_dir": str(SINGLE_RETRY_ROOT / f"{task}-seed{seed}"),
    }


def validate_single_preconditions(task, seed, config=None, version_lookup=None):
    config = validate_config(load_config() if config is None else config)
    job = build_single_job(task, seed, config)
    if Path(job["run_dir"]).exists():
        raise FileExistsError(f"refusing existing single retry target: {job['run_dir']}")
    _validate_versions(config, version_lookup)
    return job


def expected_tmux_session(batch):
    if batch not in ("seed0", "remaining"):
        raise ValueError("batch must be seed0 or remaining")
    return f"cmp620-crossq-scaled-{batch}"


def validate_tmux_session(batch, session_name):
    expected = expected_tmux_session(batch)
    if not session_name:
        raise RuntimeError("live scaled execution requires an existing tmux session")
    if session_name != expected:
        raise RuntimeError(f"batch {batch} requires tmux session {expected}, got {session_name}")


def current_tmux_session():
    if not os.environ.get("TMUX"):
        return None
    result = subprocess.run(["tmux", "display-message", "-p", "#{session_name}"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def require_tmux_session(batch):
    validate_tmux_session(batch, current_tmux_session())


def validate_single_tmux_session(task, seed, session_name):
    if task not in ("Hopper", "HalfCheetah") or seed != 0:
        raise ValueError("single retry supports only Hopper or HalfCheetah seed 0")
    expected = f"cmp620-crossq-scaled-single-{task}-seed{seed}"
    if not session_name:
        raise RuntimeError("single-task run requires an existing tmux session")
    if session_name != expected:
        raise RuntimeError(f"single-task run requires tmux session {expected}, got {session_name}")


def require_single_tmux_session(task, seed):
    validate_single_tmux_session(task, seed, current_tmux_session())


def validate_seed0_prerequisites(config, output_root=OUTPUT_ROOT, validator=None):
    if validator is None:
        from scripts.summarize_scaled_reproduction import validate_run
        validator = validate_run
    validated = []
    for task in config["tasks"]:
        run_dir = Path(output_root) / f"{task}-seed0"
        if not run_dir.is_dir():
            raise FileNotFoundError(f"remaining batch requires completed seed0 artifact: {run_dir}")
        validator(run_dir, task=task, seed=0)
        validated.append(task)
    return validated


def validate_disk_capacity(available_disk_bytes, checkpoint_bytes=0):
    required = max(MIN_FREE_DISK_BYTES, int(checkpoint_bytes))
    if available_disk_bytes < required:
        raise RuntimeError(f"insufficient disk: require at least 20 GiB ({required} bytes), "
                           f"have {available_disk_bytes}")
    return required


def format_job(job):
    return (f"task={job['env_id']} seed={job['seed']} steps={job['steps']} "
            f"run_dir={job['run_dir']}")


def run_batch(batch, config_path=CONFIG_PATH, output_root=OUTPUT_ROOT, *, confirmed=False):
    if not confirmed:
        raise RuntimeError("live scaled execution requires explicit confirmation")
    require_tmux_session(batch)
    config = load_config(config_path)
    if batch == "remaining":
        validate_seed0_prerequisites(config, output_root)
    selected = validate_batch_preconditions(config, batch, output_root)
    from scripts.summarize_scaled_reproduction import validate_run

    resource_preflight(config, selected, output_root)
    check_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_envs.py"), "--config", str(config_path)],
        check=False,
    )
    if check_result.returncode:
        raise RuntimeError("configured environment/API preflight failed")

    provenance = [Path(__file__), ROOT / "scripts/run_scaled_reproduction.sh"]
    for job in selected:
        stamp = time.time_ns()
        log_dir = ROOT / "artifacts/logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        run_log = log_dir / f"scaled-{job['task']}-seed{job['seed']}-{stamp}.log"
        time_log = log_dir / f"scaled-{job['task']}-seed{job['seed']}-{stamp}-time.txt"
        if run_log.exists() or time_log.exists():
            raise FileExistsError("refusing to overwrite scaled run logs")
        command = (
            "import sys; from crossq_repro.runner import run; "
            "run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), "
            "sys.argv[5], device='auto', provenance_paths=sys.argv[6:])"
        )
        with run_log.open("x") as output:
            result = subprocess.run(
                ["/usr/bin/time", "-v", "-o", str(time_log), sys.executable, "-c", command,
                 str(config_path), job["env_id"], str(job["seed"]), str(job["steps"]),
                 job["run_dir"], *(str(path) for path in provenance)],
                cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(f"scaled run failed for {job['task']} seed {job['seed']}; see {run_log}")
        validate_run(job["run_dir"], task=job["task"], seed=job["seed"])


def run_single(task, seed, config_path=CONFIG_PATH, *, confirmed=False):
    if not confirmed:
        raise RuntimeError("single-task run requires explicit confirmation")
    config = validate_config(load_config(config_path))
    build_single_job(task, seed, config)
    require_single_tmux_session(task, seed)
    job = validate_single_preconditions(task, seed, config)
    from scripts.summarize_scaled_reproduction import validate_run

    resource_preflight(config, [job], SINGLE_RETRY_ROOT)
    check_result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_envs.py"), "--config", str(config_path)],
        cwd=ROOT, check=False,
    )
    if check_result.returncode:
        raise RuntimeError("configured environment/API preflight failed")

    stamp = time.time_ns()
    log_dir = ROOT / "artifacts/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / f"scaled-single-{task}-seed{seed}-{stamp}.log"
    time_log = log_dir / f"scaled-single-{task}-seed{seed}-{stamp}-time.txt"
    if run_log.exists() or time_log.exists():
        raise FileExistsError("refusing to overwrite single retry logs")
    command = (
        "import sys; from crossq_repro.runner import run; "
        "run(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], "
        "device='auto', provenance_paths=sys.argv[6:])"
    )
    provenance = [Path(__file__), ROOT / "scripts/run_scaled_reproduction.sh"]
    with run_log.open("x") as output:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_log), sys.executable, "-c", command,
             str(config_path), job["env_id"], str(job["seed"]), str(job["steps"]), job["run_dir"],
             *(str(path) for path in provenance)],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"single {job['task']} seed {seed} run failed; see {run_log}")
    validate_run(job["run_dir"], task=task, seed=seed)


def resource_preflight(config, selected, output_root):
    import gymnasium as gym
    import torch
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except ImportError:
        available_ram = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    buffer_capacity = config["settings"]["sb3_mappings"]["constructor_kwargs"]["buffer_size"]["value"]
    largest_buffer = 0
    for env_id in config["tasks"].values():
        env = gym.make(env_id)
        try:
            obs = 1
            for dim in env.observation_space.shape:
                obs *= int(dim)
            act = 1
            for dim in env.action_space.shape:
                act *= int(dim)
            largest_buffer = max(largest_buffer, buffer_capacity * (2 * obs + act + 2) * 4)
        finally:
            env.close()
    required_ram = int(largest_buffer * 1.5)
    if available_ram < required_ram:
        raise RuntimeError(f"insufficient RAM for selected batch: need {required_ram}, have {available_ram}")
    disk_path = Path(output_root).parent
    while not disk_path.exists():
        disk_path = disk_path.parent
    available_disk = shutil.disk_usage(disk_path).free
    estimated_checkpoints = int((2.21 * 1024**3 / 9) * len(selected))
    required_disk = validate_disk_capacity(available_disk, estimated_checkpoints)
    gpu_free = None
    if torch.cuda.is_available():
        gpu_free = torch.cuda.mem_get_info()[0]
        if gpu_free < 2 * 1024**3:
            raise RuntimeError(f"insufficient CUDA memory: need 2147483648 bytes, have {gpu_free}")
    print(f"Resource preflight passed: max_replay_bytes={largest_buffer} available_ram={available_ram} "
          f"required_disk={required_disk} available_disk={available_disk} cuda_free={gpu_free}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-batch", choices=("seed0", "remaining"))
    parser.add_argument("--confirm-scaled-run", action="store_true")
    parser.add_argument("--single")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config = validate_config(load_config(args.config))
    if args.dry_run:
        if args.confirm_scaled_run:
            parser.error("--dry-run cannot be combined with launch confirmation")
        if args.single is not None:
            if args.run_batch is not None or args.seed is None:
                parser.error("--single requires --seed and cannot be combined with --run-batch")
            try:
                print(format_job(build_single_job(args.single, args.seed)))
            except ValueError as error:
                parser.error(str(error))
            return 0
        if args.run_batch or args.seed is not None:
            parser.error("--dry-run cannot be combined with launch confirmation or batch options")
        for job in build_jobs(config, args.output_root):
            print(format_job(job))
        return 0
    if args.single is not None:
        if args.run_batch is not None or args.seed is None:
            parser.error("--single requires --seed and cannot be combined with --run-batch")
        if not args.confirm_scaled_run:
            parser.error("single execution requires --confirm-scaled-run")
        try:
            run_single(args.single, args.seed, args.config, confirmed=True)
        except ValueError as error:
            parser.error(str(error))
        return 0
    if args.seed is not None:
        parser.error("--seed requires --single")
    if args.run_batch is None or not args.confirm_scaled_run:
        parser.error("select --dry-run or an explicitly authorized --run-batch")
    run_batch(args.run_batch, args.config, args.output_root, confirmed=args.confirm_scaled_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
