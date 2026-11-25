#!/usr/bin/env python3
"""Strict per-run validation and selected-batch summaries for scaled CrossQ."""
import argparse
import ast
import hashlib
import json
import math
import struct
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(ROOT))

from scripts.scaled_reproduction_jobs import (
    CONFIG_PATH, OUTPUT_ROOT, ROOT, build_jobs, load_config, select_batch, validate_config,
)

REQUIRED_SOURCES = {
    "runner.py": ROOT / "crossq_repro/runner.py",
    "artifacts.py": ROOT / "crossq_repro/artifacts.py",
    "evaluate.py": ROOT / "crossq_repro/evaluate.py",
    "check_envs.py": ROOT / "scripts/check_envs.py",
    "scaled_reproduction_jobs.py": ROOT / "scripts/scaled_reproduction_jobs.py",
    "run_scaled_reproduction.sh": ROOT / "scripts/run_scaled_reproduction.sh",
}
PACKAGE_KEYS = {
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "mujoco": "mujoco",
}
KNOWN_DEVIATIONS = [
    "Scope: three tasks and three seeds rather than the paper's six tasks and ten seeds.",
    "Implementation: SB3-Contrib 2.9.0/PyTorch rather than the authors' JAX implementation.",
    "Evaluation cadence 25,000 steps, 20 episodes, and five eval environments is a modern choice, not specified by the paper.",
    "No SAC/REDQ/DroQ baselines are included; comparative performance claims are not tested.",
    "Actor width [256, 256] is the SB3-Contrib default; the paper's reported [2048, 2048] width is for the critic.",
]


def _read_npy(member):
    raw = member.read()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("invalid evaluation array")
    version = raw[6]
    header_bytes = (struct.unpack("<H", raw[8:10])[0] if version == 1
                    else struct.unpack("<I", raw[8:12])[0])
    start = 10 if version == 1 else 12
    header = ast.literal_eval(raw[start:start + header_bytes].decode("latin1").strip())
    code = {"<i8": "q", "<i4": "i", "<f8": "d", "<f4": "f"}.get(header["descr"])
    if code is None:
        raise ValueError(f"unsupported evaluation dtype: {header['descr']}")
    payload = raw[start + header_bytes:]
    size = struct.calcsize(code)
    values = list(struct.unpack("<" + code * (len(payload) // size), payload))
    shape = tuple(header["shape"])
    count = math.prod(shape) if shape else 1
    if len(values) != count:
        raise ValueError("truncated evaluation array")
    return shape, values


def _critic_is_expected(model):
    networks = model.policy.critic.q_networks
    if len(networks) != 2:
        return False
    return all(
        [layer.out_features for layer in network.modules() if hasattr(layer, "out_features")]
        == [2048, 2048, 1]
        for network in networks
    )


def validate_run(run_dir, *, task, seed, config_path=CONFIG_PATH, checkpoint_loader=None):
    run_dir = Path(run_dir)
    required = ("metadata.json", "config.json", "best/best_model.zip", "final_model.zip",
                "evaluations/evaluations.npz")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError("incomplete run artifacts: " + ", ".join(missing))

    config = validate_config(load_config(config_path))
    snapshot_path = run_dir / "config.json"
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot != config:
        raise ValueError("run config mismatch")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    env_id = config["tasks"].get(task)
    if env_id is None or seed not in config["seeds"]:
        raise ValueError("task/seed is not in the scaled manifest")
    expected = {
        "status": "completed", "env_id": env_id, "seed": seed,
        "total_timesteps": 5_000_000, "actual_timesteps": 5_000_000,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch for {key}: expected {value}, got {metadata.get(key)}")
    if metadata.get("resolved_config", {}).get("config") != config:
        raise ValueError("metadata resolved config mismatch")
    if metadata.get("config_sha256") != hashlib.sha256(snapshot_path.read_bytes()).hexdigest():
        raise ValueError("config hash mismatch")
    expected_versions = config["implementation_versions"]
    actual_versions = metadata.get("package_versions", {})
    for key, version in expected_versions.items():
        package_name = PACKAGE_KEYS[key]
        if actual_versions.get(package_name) != version:
            raise ValueError(f"package version mismatch for {package_name}")
    source_hashes = metadata.get("source_sha256", {})
    for name, source_path in REQUIRED_SOURCES.items():
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        matches = [value for path, value in source_hashes.items() if Path(path).name == name]
        if digest not in matches:
            raise ValueError(f"missing or mismatched source hash for {name}")

    with zipfile.ZipFile(run_dir / "evaluations/evaluations.npz") as archive:
        names = {name.removesuffix(".npy") for name in archive.namelist()}
        if not {"timesteps", "results", "ep_lengths"}.issubset(names):
            raise ValueError("incomplete evaluation arrays")
        arrays = {name: _read_npy(archive.open(name + ".npy"))
                  for name in ("timesteps", "results", "ep_lengths")}
    steps_shape, timesteps = arrays["timesteps"]
    returns_shape, returns = arrays["results"]
    lengths_shape, lengths = arrays["ep_lengths"]
    expected_steps = list(range(25_000, 5_000_001, 25_000))
    if steps_shape != (200,) or timesteps != expected_steps:
        raise ValueError("evaluation timestep sequence mismatch")
    if returns_shape != (200, 20) or lengths_shape != (200, 20):
        raise ValueError("evaluation arrays must have shape (200, 20)")
    if any(not math.isfinite(value) for value in returns + lengths):
        raise ValueError("non-finite evaluation metric")
    if any(value <= 0 for value in lengths):
        raise ValueError("episode lengths must be positive")

    if checkpoint_loader is None:
        from sb3_contrib import CrossQ
        checkpoint_loader = lambda path: CrossQ.load(str(path), device="cpu")
    for checkpoint in (run_dir / "best/best_model.zip", run_dir / "final_model.zip"):
        try:
            model = checkpoint_loader(checkpoint)
        except Exception as error:
            raise ValueError(f"CrossQ.load failed for {checkpoint.name}: {error}") from error
        if not _critic_is_expected(model):
            raise ValueError(f"wrong critic architecture in {checkpoint.name}; expected two [2048, 2048, 1] critics")
    started = datetime.fromisoformat(metadata["started_at"])
    finished = datetime.fromisoformat(metadata["finished_at"])
    time_logs = sorted((ROOT / "artifacts/logs").glob(f"scaled-{task}-seed{seed}-*-time.txt"))
    max_rss = None
    if time_logs:
        import re
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", time_logs[-1].read_text())
        if match:
            max_rss = int(match.group(1))
    return {
        "task": task, "env_id": env_id, "seed": seed,
        "started_at": metadata["started_at"], "finished_at": metadata["finished_at"],
        "runtime_seconds": (finished - started).total_seconds(), "max_rss_kb": max_rss,
        "timesteps": timesteps,
        "episode_returns": [returns[index * 20:(index + 1) * 20] for index in range(200)],
        "episode_lengths": [lengths[index * 20:(index + 1) * 20] for index in range(200)],
        "config_sha256": metadata["config_sha256"],
        "source_sha256": metadata["source_sha256"],
        "package_versions": metadata["package_versions"],
        "resolved_device": metadata.get("resolved_config", {}).get("device", {}).get("resolved"),
        "run_dir": str(run_dir),
    }


def summarize_batch(batch, *, config_path=CONFIG_PATH, output_root=OUTPUT_ROOT, checkpoint_loader=None):
    import numpy as np
    config = validate_config(load_config(config_path))
    matrix = build_jobs(config, output_root)
    selected = matrix if batch == "all" else select_batch(matrix, batch)
    if batch not in ("all", "seed0", "remaining"):
        raise ValueError("summary batch must be all, seed0, or remaining")
    runs = [validate_run(job["run_dir"], task=job["task"], seed=job["seed"],
                         config_path=config_path, checkpoint_loader=checkpoint_loader)
            for job in selected]
    tasks = {}
    for task in config["tasks"]:
        task_runs = [run for run in runs if run["task"] == task]
        if not task_runs:
            continue
        tasks[task] = {
            "seeds": {str(run["seed"]): run for run in task_runs},
            "final_score_across_seeds": {
                "mean": float(np.mean([np.mean(run["episode_returns"][-1]) for run in task_runs])),
                "std": float(np.std([np.mean(run["episode_returns"][-1]) for run in task_runs])),
                "n_seeds": len(task_runs),
            },
        }
    return {
        "status": "complete" if batch == "all" else "selected_batch_complete",
        "batch": batch,
        "validated_runs": len(runs),
        "expected_runs": len(selected),
        "config": config,
        "settings": config["settings"],
        "package_versions": runs[0]["package_versions"],
        "tasks": tasks,
        "deviations": KNOWN_DEVIATIONS,
        "evidence_limits": "Exploratory scaled reproduction; three seeds and three tasks, no REDQ/DroQ/SAC baselines.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", choices=("all", "seed0", "remaining"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/scaled_reproduction")
    args = parser.parse_args(argv)
    result = summarize_batch(args.batch, config_path=args.config, output_root=args.output_root)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.batch == "all":
        json_path = args.results_dir / "summary.json"
        report_path = args.results_dir / "report.md"
    else:
        json_path = args.results_dir / f"{args.batch}-summary.json"
        report_path = args.results_dir / f"{args.batch}-report.md"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [f"# Scaled CrossQ summary: {args.batch}", "",
             f"Validated runs: {result['validated_runs']}/{result['expected_runs']}",
             f"Package versions: `{json.dumps(result['package_versions'], sort_keys=True)}`", ""]
    for task, task_result in result["tasks"].items():
        values = task_result["final_score_across_seeds"]
        lines.append(f"- {task}: final across-seed mean {values['mean']:.6g}, std {values['std']:.6g} (n={values['n_seeds']})")
        for seed, run in task_result["seeds"].items():
            rss = "not collected" if run["max_rss_kb"] is None else f"{run['max_rss_kb']} kB"
            final_mean = sum(run["episode_returns"][-1]) / 20
            lines.append(f"  - seed {seed}: final mean {final_mean:.6g}; runtime {run['runtime_seconds']:.2f}s; max RSS {rss}")
    lines += ["", f"Deviations: {result['deviations']}", "", result["evidence_limits"], ""]
    report_path.write_text("\n".join(lines))
    print(json.dumps({"summary_json": str(json_path), "report_md": str(report_path),
                      "validated_runs": result["validated_runs"], "expected_runs": result["expected_runs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
