#!/usr/bin/env python3
"""Run one explicitly confirmed Hopper extension seed; dry-run lists seeds 5–9."""
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
CONFIG_PATH = ROOT / "configs/proposal_hopper_10seeds_5m_extension.yaml"
BASE_CONFIG_PATH = ROOT / "configs/proposal_hopper_5seeds_5m.yaml"
OUTPUT_ROOT = ROOT / "artifacts/proposal_hopper_extension"
BASE_OUTPUT_ROOT = ROOT / "artifacts/proposal_hopper"
VERSION_NAMES = {
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "mujoco": "mujoco",
}
PROVENANCE_PATHS = (
    Path(__file__),
    ROOT / "scripts/validate_proposal_hopper_extension.py",
    CONFIG_PATH,
    BASE_CONFIG_PATH,
    ROOT / "scripts/run_proposal_hopper.py",
    ROOT / "scripts/validate_proposal_hopper.py",
    ROOT / "scripts/scaled_reproduction_jobs.py",
)


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def validate_config(config):
    expected = copy.deepcopy(load_config(BASE_CONFIG_PATH))
    expected["seeds"] = list(range(10))
    if config != expected:
        raise ValueError("extension config differs from the frozen ten-seed Hopper protocol")
    return config


def build_jobs(config_path=CONFIG_PATH, output_root=OUTPUT_ROOT):
    config = validate_config(load_config(config_path))
    return [
        {
            "task": "Hopper",
            "env_id": config["tasks"]["Hopper"],
            "seed": seed,
            "steps": config["max_steps"],
            "run_dir": str(Path(output_root) / f"Hopper-seed{seed}"),
        }
        for seed in range(5, 10)
    ]


def expected_tmux_session(seed):
    if seed not in range(5, 10):
        raise ValueError("only extension Hopper seeds 5 through 9 are runnable")
    return f"cmp620-crossq-proposal-hopper-seed{seed}"


def validate_tmux_session(seed, session_name):
    expected = expected_tmux_session(seed)
    if not session_name:
        raise RuntimeError("proposal Hopper extension run requires an existing named tmux session")
    if session_name != expected:
        raise RuntimeError(f"seed {seed} requires tmux session {expected}, got {session_name}")


def current_tmux_session():
    import os

    if not os.environ.get("TMUX"):
        return None
    result = subprocess.run(
        ["tmux", "display-message", "-p", "#{session_name}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def validate_seed_preconditions(seed, config=None, output_root=OUTPUT_ROOT, version_lookup=None):
    config = validate_config(load_config() if config is None else config)
    expected_tmux_session(seed)
    target = Path(output_root) / f"Hopper-seed{seed}"
    if target.exists():
        raise FileExistsError(f"refusing existing extension target: {target}")
    versions = version_lookup() if version_lookup is not None else {
        key: _installed_version(package) for key, package in VERSION_NAMES.items()
    }
    mismatches = [
        f"{VERSION_NAMES[key]} expected {version}, found {versions.get(key)}"
        for key, version in config["implementation_versions"].items()
        if versions.get(key) != version
    ]
    if mismatches:
        raise ValueError("proposal package version mismatch: " + "; ".join(mismatches))
    return {
        "task": "Hopper",
        "env_id": config["tasks"]["Hopper"],
        "seed": seed,
        "steps": config["max_steps"],
        "run_dir": str(target),
    }


def _installed_version(package):
    try:
        return package_metadata.version(package)
    except package_metadata.PackageNotFoundError:
        return None


def run_seed(seed, config_path=CONFIG_PATH, output_root=OUTPUT_ROOT, *, confirmed=False):
    if not confirmed:
        raise RuntimeError("proposal Hopper extension training requires explicit confirmation")
    expected_tmux_session(seed)
    config_path = Path(config_path).resolve()
    config = validate_config(load_config(config_path))
    job = validate_seed_preconditions(seed, config, output_root)
    validate_tmux_session(seed, current_tmux_session())

    from scripts.validate_proposal_hopper import validate_all_runs

    validate_all_runs(config_path=BASE_CONFIG_PATH, output_root=BASE_OUTPUT_ROOT)

    from scripts import scaled_reproduction_jobs as shared

    shared.resource_preflight(config, [job], output_root)
    check = subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_envs.py"), "--config", str(BASE_CONFIG_PATH)],
        cwd=ROOT,
        check=False,
    )
    if check.returncode:
        raise RuntimeError("proposal Hopper extension environment/API preflight failed")

    stamp = time.time_ns()
    logs = ROOT / "artifacts/logs"
    logs.mkdir(parents=True, exist_ok=True)
    run_log = logs / f"proposal-hopper-extension-seed{seed}-{stamp}.log"
    time_log = logs / f"proposal-hopper-extension-seed{seed}-{stamp}-time.txt"
    if run_log.exists() or time_log.exists():
        raise FileExistsError("refusing to overwrite proposal extension run logs")
    command = (
        "import sys; from crossq_repro.runner import run; "
        "run(sys.argv[1], 'Hopper-v4', int(sys.argv[2]), 5000000, sys.argv[3], "
        "device='auto', provenance_paths=sys.argv[4:])"
    )
    with run_log.open("x") as output:
        result = subprocess.run(
            [
                "/usr/bin/time",
                "-v",
                "-o",
                str(time_log),
                sys.executable,
                "-c",
                command,
                str(config_path),
                str(seed),
                str(job["run_dir"]),
                *(str(path) for path in PROVENANCE_PATHS),
            ],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"proposal Hopper extension seed {seed} failed; see {run_log}")

    from scripts.validate_proposal_hopper_extension import validate_extension_run

    validate_extension_run(job["run_dir"], seed=seed, config_path=config_path)


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
            parser.error("--dry-run lists seeds 5–9 and cannot select a seed or confirm training")
        for job in build_jobs(args.config, args.output_root):
            print(
                f"task={job['env_id']} seed={job['seed']} steps={job['steps']} "
                f"run_dir={job['run_dir']}"
            )
        return 0
    if args.seed is None or not args.confirm_proposal_run:
        parser.error("select exactly one --seed 5..9 and pass --confirm-proposal-run")
    try:
        run_seed(args.seed, args.config, args.output_root, confirmed=True)
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
