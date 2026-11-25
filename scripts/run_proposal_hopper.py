#!/usr/bin/env python3
"""Run one explicitly confirmed Hopper proposal seed; dry-run lists seeds 1–4."""
import argparse
import copy
import json
import subprocess
import sys
import time
from importlib import metadata as package_metadata
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "configs/proposal_hopper_5seeds_5m.yaml"
SCALED_CONFIG_PATH = ROOT / "configs/scaled_3tasks_3seeds_5m.yaml"
OUTPUT_ROOT = ROOT / "artifacts/proposal_hopper"
SEED0_RUN = ROOT / "artifacts/scaled_reproduction_retry/Hopper-seed0"
VERSION_NAMES = {
    "stable_baselines3": "stable-baselines3", "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium", "mujoco": "mujoco",
}


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def validate_config(config):
    scaled = json.loads(SCALED_CONFIG_PATH.read_text())
    expected = copy.deepcopy(scaled)
    expected["tasks"] = {"Hopper": "Hopper-v4"}
    expected["task_provenance"] = {"Hopper": scaled["task_provenance"]["Hopper"]}
    expected["seeds"] = [0, 1, 2, 3, 4]
    if config != expected:
        raise ValueError("proposal config differs from the frozen Hopper/Table 1 protocol")
    return config


def build_jobs(config_path=CONFIG_PATH, output_root=OUTPUT_ROOT):
    config = validate_config(load_config(config_path))
    return [{"task": "Hopper", "env_id": config["tasks"]["Hopper"], "seed": seed,
             "steps": config["max_steps"], "run_dir": str(Path(output_root) / f"Hopper-seed{seed}")}
            for seed in (1, 2, 3, 4)]


def expected_tmux_session(seed):
    if seed not in (1, 2, 3, 4):
        raise ValueError("only fresh Hopper seeds 1 through 4 are runnable; seed 0 is reused")
    return f"cmp620-crossq-proposal-hopper-seed{seed}"


def validate_tmux_session(seed, session_name):
    expected = expected_tmux_session(seed)
    if not session_name:
        raise RuntimeError("proposal Hopper run requires an existing named tmux session")
    if session_name != expected:
        raise RuntimeError(f"seed {seed} requires tmux session {expected}, got {session_name}")


def current_tmux_session():
    import os
    if not os.environ.get("TMUX"):
        return None
    result = subprocess.run(["tmux", "display-message", "-p", "#{session_name}"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def validate_seed_preconditions(seed, config=None, output_root=OUTPUT_ROOT, version_lookup=None):
    config = validate_config(load_config() if config is None else config)
    if seed not in (1, 2, 3, 4):
        raise ValueError("only fresh Hopper seeds 1 through 4 are runnable; seed 0 is reused")
    target = Path(output_root) / f"Hopper-seed{seed}"
    if target.exists():
        raise FileExistsError(f"refusing existing proposal Hopper target: {target}")
    if Path(output_root) == OUTPUT_ROOT and not SEED0_RUN.is_dir():
        raise FileNotFoundError(f"completed Hopper seed 0 is required and must be preserved: {SEED0_RUN}")
    versions = (version_lookup() if version_lookup is not None else {
        key: _installed_version(package) for key, package in VERSION_NAMES.items()
    })
    mismatches = [f"{VERSION_NAMES[key]} expected {version}, found {versions.get(key)}"
                  for key, version in config["implementation_versions"].items()
                  if versions.get(key) != version]
    if mismatches:
        raise ValueError("proposal package version mismatch: " + "; ".join(mismatches))
    return {"task": "Hopper", "env_id": config["tasks"]["Hopper"], "seed": seed,
            "steps": config["max_steps"], "run_dir": str(target)}


def _installed_version(package):
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return None


def run_seed(seed, config_path=CONFIG_PATH, output_root=OUTPUT_ROOT, *, confirmed=False):
    if not confirmed:
        raise RuntimeError("proposal Hopper training requires explicit confirmation")
    expected_tmux_session(seed)
    config_path = Path(config_path).resolve()
    config = validate_config(load_config(config_path))
    target = Path(output_root) / f"Hopper-seed{seed}"
    job = validate_seed_preconditions(seed, config, output_root)
    validate_tmux_session(seed, current_tmux_session())

    from scripts.validate_proposal_hopper import validate_run
    validate_run(SEED0_RUN, seed=0, config_path=config_path, legacy_seed0=True)

    from scripts import scaled_reproduction_jobs as shared
    shared.resource_preflight(config, [job], output_root)
    check = subprocess.run([sys.executable, str(ROOT / "scripts/check_envs.py"), "--config", str(config_path)],
                           cwd=ROOT, check=False)
    if check.returncode:
        raise RuntimeError("proposal Hopper environment/API preflight failed")

    stamp = time.time_ns()
    logs = ROOT / "artifacts/logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_log = logs / f"proposal-hopper-seed{seed}-{stamp}.log"
    time_log = logs / f"proposal-hopper-seed{seed}-{stamp}-time.txt"
    if run_log.exists() or time_log.exists():
        raise FileExistsError("refusing to overwrite proposal run logs")
    command = (
        "import sys; from crossq_repro.runner import run; "
        "run(sys.argv[1], 'Hopper-v4', int(sys.argv[2]), 5000000, sys.argv[3], "
        "device='auto', provenance_paths=sys.argv[4:])"
    )
    provenance = [Path(__file__), ROOT / "scripts/validate_proposal_hopper.py"]
    with run_log.open("x") as output:
        result = subprocess.run(
            ["/usr/bin/time", "-v", "-o", str(time_log), sys.executable, "-c", command,
             str(config_path), str(seed), str(target), *(str(path) for path in provenance)],
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"proposal Hopper seed {seed} failed; see {run_log}")
    validate_run(target, seed=seed, config_path=config_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--confirm-proposal-run", action="store_true")
    args = parser.parse_args(argv)
    config = validate_config(load_config(args.config))
    if args.dry_run:
        if args.seed is not None or args.confirm_proposal_run:
            parser.error("--dry-run lists seeds 1–4 and cannot be combined with a selected seed or confirmation")
        for job in build_jobs(args.config, args.output_root):
            print(f"task={job['env_id']} seed={job['seed']} steps={job['steps']} run_dir={job['run_dir']}")
        return 0
    if args.seed is None or not args.confirm_proposal_run:
        parser.error("select exactly one --seed 1..4 and pass --confirm-proposal-run")
    try:
        run_seed(args.seed, args.config, args.output_root, confirmed=True)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
