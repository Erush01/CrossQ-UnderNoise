#!/usr/bin/env python3
"""Strictly validate extension Hopper seeds 5–9 and summarize all ten seeds."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_proposal_hopper_extension as runner
from scripts import validate_proposal_hopper as base_validator

EXPECTED_TIMESTEPS = base_validator.EXPECTED_TIMESTEPS
VERSION_KEYS = {
    "stable_baselines3": "stable-baselines3",
    "sb3_contrib": "sb3-contrib",
    "gymnasium": "gymnasium",
    "mujoco": "mujoco",
}
EXTENSION_SOURCES = base_validator.CORE_SOURCES + list(runner.PROVENANCE_PATHS)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sources(metadata):
    hashes = metadata.get("source_sha256", {})
    expected = {Path(source).resolve(): _sha256(source) for source in EXTENSION_SOURCES}
    recorded = {Path(path).resolve(): digest for path, digest in hashes.items()}
    if recorded.keys() != expected.keys():
        raise ValueError("extension provenance path set mismatch")
    for source, digest in expected.items():
        if recorded[source] != digest:
            raise ValueError(f"extension source hash mismatch for {source.name}")


def _validate_critic(model, checkpoint):
    networks = model.policy.critic.q_networks
    if len(networks) != 2:
        raise ValueError(f"{checkpoint.name}: expected exactly two critics")
    for network in networks:
        widths = [layer.out_features for layer in network.modules() if hasattr(layer, "out_features")]
        if widths != [2048, 2048, 1]:
            raise ValueError(f"{checkpoint.name}: expected critic widths [2048, 2048, 1], got {widths}")


def validate_extension_run(
    run_dir,
    *,
    seed,
    config_path=runner.CONFIG_PATH,
    checkpoint_loader=None,
    load_best_and_final=True,
    load_checkpoints=True,
):
    if seed not in range(5, 10):
        raise ValueError("extension seeds 5 through 9 are the only valid extension runs")
    run_dir = Path(run_dir)
    expected_files = ["metadata.json", "config.json", "evaluations/evaluations.npz", "final_model.zip"]
    if load_best_and_final:
        expected_files.append("best/best_model.zip")
    missing = [name for name in expected_files if not (run_dir / name).is_file()]
    if missing:
        raise ValueError("incomplete extension run artifacts: " + ", ".join(missing))

    proposal = runner.validate_config(runner.load_config(config_path))
    snapshot_path = run_dir / "config.json"
    snapshot = json.loads(snapshot_path.read_text())
    if snapshot != proposal:
        raise ValueError("extension run config mismatch")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    if metadata.get("resolved_config", {}).get("config") != snapshot:
        raise ValueError("metadata resolved config does not match config.json")
    expected_meta = {
        "status": "completed",
        "env_id": "Hopper-v4",
        "seed": seed,
        "total_timesteps": 5_000_000,
        "actual_timesteps": 5_000_000,
    }
    for key, value in expected_meta.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch for {key}")
    if metadata.get("config_sha256") != _sha256(snapshot_path):
        raise ValueError("config hash mismatch")

    resolved = metadata.get("resolved_config", {})
    wanted = base_validator._expected_resolved(proposal, seed)
    if resolved.get("constructor_kwargs") != wanted["constructor_kwargs"]:
        raise ValueError("resolved constructor settings mismatch")
    if resolved.get("policy_kwargs") != wanted["policy_kwargs"]:
        raise ValueError("resolved policy settings mismatch")
    if resolved.get("evaluation") != wanted["evaluation"]:
        raise ValueError("resolved evaluation settings mismatch")
    if resolved.get("evaluation_seed") != seed + 10_000:
        raise ValueError("resolved evaluation seed mismatch")
    if resolved.get("device", {}).get("requested") != "auto":
        raise ValueError("requested device setting mismatch")

    versions = metadata.get("package_versions", {})
    for key, package in VERSION_KEYS.items():
        if versions.get(package) != proposal["implementation_versions"][key]:
            raise ValueError(f"package version mismatch for {package}")
    _check_sources(metadata)

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
        "seed": seed,
        "env_id": "Hopper-v4",
        "manifest_scope": "proposal-ten-seed-extension",
        "timesteps": timesteps.tolist(),
        "episode_returns": returns.tolist(),
        "episode_lengths": lengths.tolist(),
        "config_sha256": metadata["config_sha256"],
        "source_sha256": metadata["source_sha256"],
        "package_versions": versions,
        "resolved_device": resolved.get("device", {}).get("resolved"),
        "run_dir": str(run_dir),
    }


def seed_level_statistics(scores, bootstrap_samples=10_000, random_seed=0):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (10,) or not np.isfinite(scores).all():
        raise ValueError("statistics require ten finite training-seed scores")

    def iqm(values):
        sorted_values = np.sort(np.asarray(values, dtype=np.float64))
        n = len(sorted_values)
        weights = np.maximum(
            0.0,
            np.minimum((np.arange(n) + 1) / n, 0.75) - np.maximum(np.arange(n) / n, 0.25),
        )
        return float(np.dot(sorted_values, weights) / 0.5)

    observed = iqm(scores)
    rng = np.random.default_rng(random_seed)
    draws = rng.choice(scores, size=(bootstrap_samples, len(scores)), replace=True)
    bootstrap = np.apply_along_axis(iqm, 1, draws)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "iqm": observed,
        "ci95": [float(low), float(high)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "training_seed",
        "n_training_seeds": len(scores),
    }


def summarize_ten_seed_runs(runs, bootstrap_samples=10_000, random_seed=0):
    if len(runs) != 10 or {run["seed"] for run in runs} != set(range(10)):
        raise ValueError("summary requires exactly ten training seeds")
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
    return {
        "per_seed": by_seed,
        "seed_level_statistics": seed_level_statistics(scores, bootstrap_samples, random_seed),
        "scores_by_seed": scores,
        "bootstrap_unit": "training_seed",
    }


def validate_all_runs(config_path=runner.CONFIG_PATH, output_root=runner.OUTPUT_ROOT,
                      checkpoint_loader=None):
    existing = base_validator.validate_all_runs(
        config_path=runner.BASE_CONFIG_PATH,
        output_root=runner.BASE_OUTPUT_ROOT,
        checkpoint_loader=checkpoint_loader,
    )
    runs = [
        {
            "seed": seed,
            "episode_returns": existing["seeds"][str(seed)]["episode_returns"],
            "episode_lengths": existing["seeds"][str(seed)]["episode_lengths"],
        }
        for seed in range(5)
    ]
    runs.extend(
        validate_extension_run(
            Path(output_root) / f"Hopper-seed{seed}",
            seed=seed,
            config_path=config_path,
            checkpoint_loader=checkpoint_loader,
        )
        for seed in range(5, 10)
    )
    summary = summarize_ten_seed_runs(runs)
    return {
        "task": "Hopper-v4",
        "steps": 5_000_000,
        "seeds": summary["per_seed"],
        "scores_by_seed": summary["scores_by_seed"],
        "seed_level_statistics": summary["seed_level_statistics"],
        "bootstrap_unit": summary["bootstrap_unit"],
        "legacy_manifest_scope": existing["legacy_manifest_scope"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--config", type=Path, default=runner.CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=runner.OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.summary:
        if args.seed is not None:
            parser.error("--summary validates all ten seeds; do not select --seed")
        result = validate_all_runs(args.config, args.output_root)
        print(json.dumps(result, indent=2))
        return 0
    if args.seed is None:
        parser.error("select --seed 5..9 or --summary")
    run_dir = args.output_root / f"Hopper-seed{args.seed}"
    result = validate_extension_run(run_dir, seed=args.seed, config_path=args.config)
    print(
        json.dumps(
            {key: value for key, value in result.items()
             if key not in ("episode_returns", "episode_lengths")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
