#!/usr/bin/env python3
"""Strict validation and seed-level summary for the five-seed Hopper proposal."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.run_proposal_hopper import CONFIG_PATH, OUTPUT_ROOT, SEED0_RUN, load_config, validate_config

CORE_SOURCES = [
    ROOT / "crossq_repro/runner.py", ROOT / "crossq_repro/artifacts.py",
    ROOT / "crossq_repro/evaluate.py", ROOT / "scripts/check_envs.py",
]
LEGACY_ONLY_SOURCES = [
    ROOT / "scripts/scaled_reproduction_jobs.py", ROOT / "scripts/run_scaled_reproduction.sh",
]
PROPOSAL_SOURCES = CORE_SOURCES + [
    ROOT / "scripts/run_proposal_hopper.py", ROOT / "scripts/validate_proposal_hopper.py",
]
EXPECTED_TIMESTEPS = list(range(25_000, 5_000_001, 25_000))
VERSION_KEYS = {
    "stable_baselines3": "stable-baselines3", "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium", "mujoco": "mujoco",
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expected_resolved(config, seed):
    mappings = config["settings"]["sb3_mappings"]
    constructor = {key: entry["value"] for key, entry in mappings["constructor_kwargs"].items()}
    policy = {key: entry["value"] for key, entry in mappings["policy_kwargs"].items()}
    policy["activation_fn"] = "torch.nn.modules.activation.ReLU"
    policy.setdefault("use_sde", False)
    return {
        "constructor_kwargs": constructor,
        "policy_kwargs": policy,
        "evaluation": config["evaluation"],
        "evaluation_seed": seed + 10000,
        "device_requested": "auto",
    }


def _check_config_scope(snapshot, proposal, legacy_seed0):
    if not legacy_seed0:
        if snapshot != proposal:
            raise ValueError("proposal run config mismatch")
        return
    normalized = copy.deepcopy(snapshot)
    normalized["tasks"] = proposal["tasks"]
    normalized["task_provenance"] = proposal["task_provenance"]
    normalized["seeds"] = proposal["seeds"]
    if normalized != proposal:
        raise ValueError("legacy seed0 differs beyond the recorded top-level task/seed manifest scope")


def _check_sources(metadata, legacy_seed0):
    hashes = metadata.get("source_sha256", {})
    expected_sources = CORE_SOURCES if legacy_seed0 else PROPOSAL_SOURCES
    for source in expected_sources:
        matches = [digest for path, digest in hashes.items() if Path(path).name == source.name]
        if not matches or _sha256(source) not in matches:
            raise ValueError(f"missing or mismatched source hash for {source.name}")
    warnings = []
    if legacy_seed0:
        for source in LEGACY_ONLY_SOURCES:
            matches = [digest for path, digest in hashes.items() if Path(path).name == source.name]
            if not matches or any(len(digest) != 64 for digest in matches):
                raise ValueError(f"missing or invalid historical source hash for {source.name}")
            if _sha256(source) not in matches:
                warnings.append(
                    f"{source.name}: recorded seed0 hash differs from current file; "
                    "the run was validated at completion, but its historical source snapshot is unavailable"
                )
    for source_name, digest in ((Path(path).name, value) for path, value in hashes.items()):
        matches = [source for source in CORE_SOURCES + LEGACY_ONLY_SOURCES + PROPOSAL_SOURCES
                   if source.name == source_name]
        if matches and source_name not in {path.name for path in LEGACY_ONLY_SOURCES} and _sha256(matches[0]) != digest:
            raise ValueError(f"source hash mismatch for {source_name}")
    return warnings


def _validate_critic(model, checkpoint):
    networks = model.policy.critic.q_networks
    if len(networks) != 2:
        raise ValueError(f"{checkpoint.name}: expected exactly two critics")
    for network in networks:
        widths = [layer.out_features for layer in network.modules() if hasattr(layer, "out_features")]
        if widths != [2048, 2048, 1]:
            raise ValueError(f"{checkpoint.name}: expected critic widths [2048, 2048, 1], got {widths}")


def validate_run(run_dir, *, seed, config_path=CONFIG_PATH, legacy_seed0=None,
                 checkpoint_loader=None, load_best_and_final=True, load_checkpoints=True):
    run_dir = Path(run_dir)
    if legacy_seed0 is None:
        legacy_seed0 = seed == 0
    if seed not in range(5) or (legacy_seed0 and seed != 0) or (not legacy_seed0 and seed == 0):
        raise ValueError("seed 0 must use the legacy artifact; proposal runs are seeds 1–4")
    expected_files = ["metadata.json", "config.json", "evaluations/evaluations.npz", "final_model.zip"]
    if load_best_and_final:
        expected_files.append("best/best_model.zip")
    missing = [name for name in expected_files if not (run_dir / name).is_file()]
    if missing:
        raise ValueError("incomplete proposal run artifacts: " + ", ".join(missing))

    proposal = validate_config(load_config(config_path))
    snapshot_path = run_dir / "config.json"
    snapshot = json.loads(snapshot_path.read_text())
    _check_config_scope(snapshot, proposal, legacy_seed0)
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("resolved_config", {}).get("config") != snapshot:
        raise ValueError("metadata resolved config does not match config.json")
    expected_meta = {
        "status": "completed", "env_id": "Hopper-v4", "seed": seed,
        "total_timesteps": 5_000_000, "actual_timesteps": 5_000_000,
    }
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch for {key}")
    if metadata.get("config_sha256") != _sha256(snapshot_path):
        raise ValueError("config hash mismatch")

    resolved = metadata.get("resolved_config", {})
    wanted = _expected_resolved(proposal, seed)
    if resolved.get("constructor_kwargs") != wanted["constructor_kwargs"]:
        raise ValueError("resolved constructor settings mismatch")
    actual_policy = resolved.get("policy_kwargs", {})
    if actual_policy != wanted["policy_kwargs"]:
        raise ValueError("resolved policy settings mismatch")
    if resolved.get("evaluation") != wanted["evaluation"] or resolved.get("evaluation_seed") != seed + 10000:
        raise ValueError("resolved evaluation settings mismatch")
    if resolved.get("device", {}).get("requested") != "auto":
        raise ValueError("requested device setting mismatch")

    versions = metadata.get("package_versions", {})
    for key, package in VERSION_KEYS.items():
        if versions.get(package) != proposal["implementation_versions"][key]:
            raise ValueError(f"package version mismatch for {package}")
    source_warnings = _check_sources(metadata, legacy_seed0)

    try:
        with np.load(run_dir / "evaluations/evaluations.npz", allow_pickle=False) as data:
            timesteps = data["timesteps"]
            returns = data["results"]
            lengths = data["ep_lengths"]
    except Exception as error:
        raise ValueError(f"invalid evaluation artifact: {error}") from error
    if timesteps.shape != (200,) or timesteps.tolist() != EXPECTED_TIMESTEPS:
        raise ValueError("evaluation timestep sequence mismatch")
    if returns.shape != (200, 20) or lengths.shape != (200, 20):
        raise ValueError("evaluation arrays must have shape (200, 20)")
    if not np.isfinite(returns).all() or not np.isfinite(lengths).all() or (lengths <= 0).any():
        raise ValueError("evaluation contains non-finite metrics or non-positive lengths")

    if load_checkpoints:
        if checkpoint_loader is None:
            from sb3_contrib import CrossQ
            checkpoint_loader = lambda path: CrossQ.load(str(path), device="cpu")
        checkpoints = [run_dir / "final_model.zip"]
        if load_best_and_final:
            checkpoints.insert(0, run_dir / "best/best_model.zip")
        for checkpoint in checkpoints:
            try:
                model = checkpoint_loader(checkpoint)
            except Exception as error:
                raise ValueError(f"CrossQ.load failed for {checkpoint.name}: {error}") from error
            _validate_critic(model, checkpoint)

    return {
        "seed": seed, "env_id": "Hopper-v4", "manifest_scope": (
            "legacy-three-task-three-seed" if legacy_seed0 else "proposal-five-seed"),
        "timesteps": timesteps.tolist(), "episode_returns": returns.tolist(),
        "episode_lengths": lengths.tolist(), "config_sha256": metadata["config_sha256"],
        "source_sha256": metadata["source_sha256"], "package_versions": versions,
        "resolved_device": resolved.get("device", {}).get("resolved"),
        "source_validation_warnings": source_warnings,
        "run_dir": str(run_dir),
    }


def seed_level_statistics(scores, bootstrap_samples=10000, random_seed=0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (5,) or not np.isfinite(scores).all():
        raise ValueError("statistics require five finite training-seed scores")

    def iqm(values):
        sorted_values = np.sort(np.asarray(values, dtype=np.float64))
        n = len(sorted_values)
        weights = np.maximum(0.0, np.minimum((np.arange(n) + 1) / n, 0.75)
                             - np.maximum(np.arange(n) / n, 0.25))
        return float(np.dot(sorted_values, weights) / 0.5)

    observed = iqm(scores)
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(scores, size=(bootstrap_samples, len(scores)), replace=True)
    bootstrap = np.apply_along_axis(iqm, 1, draws)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {"iqm": observed, "ci95": [float(low), float(high)],
            "bootstrap_samples": bootstrap_samples, "bootstrap_unit": "training_seed",
            "n_training_seeds": len(scores)}


def summarize_seed_runs(runs):
    if len(runs) != 5 or {run["seed"] for run in runs} != set(range(5)):
        raise ValueError("summary requires exactly five training seeds")
    by_seed = {}
    scores = []
    for run in sorted(runs, key=lambda row: row["seed"]):
        returns = np.asarray(run["episode_returns"], dtype=np.float64)
        lengths = np.asarray(run["episode_lengths"], dtype=np.float64)
        if returns.shape != (200, 20) or lengths.shape != (200, 20):
            raise ValueError("each seed must preserve all (200, 20) evaluation values")
        score = float(returns[-1].mean())
        scores.append(score)
        by_seed[str(run["seed"])] = {
            "final_20_episode_mean": score,
            "episode_returns": run["episode_returns"],
            "episode_lengths": run["episode_lengths"],
        }
    return {"per_seed": by_seed, "seed_level_statistics": seed_level_statistics(scores),
            "scores_by_seed": scores, "bootstrap_unit": "training_seed"}


def validate_all_runs(config_path=CONFIG_PATH, output_root=OUTPUT_ROOT, checkpoint_loader=None):
    legacy = validate_run(SEED0_RUN, seed=0, config_path=config_path, legacy_seed0=True,
                          checkpoint_loader=checkpoint_loader)
    new_runs = [validate_run(Path(output_root) / f"Hopper-seed{seed}", seed=seed,
                             config_path=config_path, checkpoint_loader=checkpoint_loader)
                for seed in range(1, 5)]
    summary = summarize_seed_runs([legacy, *new_runs])
    return {"task": "Hopper-v4", "steps": 5_000_000, "seeds": summary["per_seed"],
            "scores_by_seed": summary["scores_by_seed"],
            "seed_level_statistics": summary["seed_level_statistics"],
            "bootstrap_unit": summary["bootstrap_unit"],
            "legacy_manifest_scope": legacy["manifest_scope"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.summary:
        if args.seed is not None:
            parser.error("--summary validates all five seeds; do not select --seed")
        print(json.dumps(validate_all_runs(args.config, args.output_root), indent=2))
        return 0
    if args.seed is None:
        parser.error("select --seed 0..4 or --summary")
    run_dir = SEED0_RUN if args.seed == 0 else args.output_root / f"Hopper-seed{args.seed}"
    result = validate_run(run_dir, seed=args.seed, config_path=args.config, legacy_seed0=args.seed == 0)
    print(json.dumps({key: value for key, value in result.items() if key not in ("episode_returns", "episode_lengths")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
