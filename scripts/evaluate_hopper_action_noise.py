#!/usr/bin/env python3
"""Evaluate the five final Hopper policies under paired test-time action noise."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_proposal_hopper import (
    OUTPUT_ROOT, ROOT as CROSSQ_ROOT, SEED0_RUN, seed_level_statistics, validate_run,
)

TRAINING_SEEDS = tuple(range(5))
RESET_SEEDS = tuple(range(90000, 90020))
SIGMAS = (0.0, 0.05, 0.10)
DEFAULT_OUTPUT = ROOT / "results/proposal_hopper/action_noise.json"


def final_checkpoint(training_seed, output_root=OUTPUT_ROOT):
    run_dir = SEED0_RUN if training_seed == 0 else Path(output_root) / f"Hopper-seed{training_seed}"
    return run_dir / "final_model.zip"


def source_warnings_by_seed(validations):
    return {
        str(seed): list(validation.get("source_validation_warnings", []))
        for seed, validation in validations.items()
        if validation.get("source_validation_warnings")
    }


def build_evaluation_plan(output_root=OUTPUT_ROOT):
    return [
        {"training_seed": training_seed, "reset_seed": reset_seed, "sigma": sigma,
         "checkpoint": str(final_checkpoint(training_seed, output_root))}
        for training_seed in TRAINING_SEEDS
        for reset_seed in RESET_SEEDS
        for sigma in SIGMAS
    ]


def standard_normal_stream(training_seed, reset_seed):
    return np.random.default_rng(np.random.SeedSequence([73191, training_seed, reset_seed]))


def perturb_normalized_action(action, sigma, standard_normal):
    action = np.asarray(action, dtype=np.float64)
    normal = np.asarray(standard_normal, dtype=np.float64)
    if action.shape != normal.shape or sigma not in SIGMAS:
        raise ValueError("noise shape or sigma is invalid")
    noisy = action + sigma * normal
    clipping_fraction = float(np.mean((noisy < -1.0) | (noisy > 1.0)))
    return np.clip(noisy, -1.0, 1.0), clipping_fraction


def _normalized_to_env(action, low, high):
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    if np.any(high <= low):
        raise ValueError("action bounds must have positive width")
    return low + (np.asarray(action, dtype=np.float64) + 1.0) * (high - low) / 2.0


def _env_to_normalized(action, low, high):
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    return 2.0 * (np.asarray(action, dtype=np.float64) - low) / (high - low) - 1.0


def evaluate_episode(env, model, training_seed, reset_seed, sigma):
    observation, _ = env.reset(seed=reset_seed)
    rng = standard_normal_stream(training_seed, reset_seed)
    total_return = 0.0
    length = 0
    clipped_dimensions = 0.0
    action_dimensions = 0
    terminated = truncated = False
    while not (terminated or truncated):
        deterministic_action, _ = model.predict(observation, deterministic=True)
        low, high = env.action_space.low, env.action_space.high
        normalized = _env_to_normalized(deterministic_action, low, high)
        normal = np.zeros_like(normalized) if sigma == 0 else rng.standard_normal(normalized.shape)
        normalized_action, clipped = perturb_normalized_action(normalized, sigma, normal)
        action = _normalized_to_env(normalized_action, low, high)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_return += float(reward)
        length += 1
        clipped_dimensions += clipped * normalized.size
        action_dimensions += normalized.size
    return {
        "training_seed": training_seed, "reset_seed": reset_seed, "sigma": sigma,
        "return": total_return, "length": length,
        "action_clipping_fraction": clipped_dimensions / action_dimensions if action_dimensions else 0.0,
    }


def summarize_action_noise_records(records):
    if len(records) != 300:
        raise ValueError("action-noise evaluation requires exactly 300 episode records")
    expected = {(training_seed, reset_seed, sigma)
                for training_seed in TRAINING_SEEDS for reset_seed in RESET_SEEDS for sigma in SIGMAS}
    observed = {(record["training_seed"], record["reset_seed"], record["sigma"]) for record in records}
    if observed != expected:
        raise ValueError("action-noise episode tuples are missing or duplicated")
    for record in records:
        if (not np.isfinite(record["return"]) or record["length"] <= 0
                or not np.isfinite(record["action_clipping_fraction"])
                or not 0.0 <= record["action_clipping_fraction"] <= 1.0):
            raise ValueError("action-noise episode metrics are invalid")
    per_seed = {}
    sigma_summary = {}
    delta_summary = {}
    record_index = {(r["training_seed"], r["reset_seed"], r["sigma"]): r for r in records}
    sigma_key = lambda sigma: format(sigma, "g")
    for training_seed in TRAINING_SEEDS:
        rows = {sigma: [record_index[(training_seed, reset, sigma)] for reset in RESET_SEEDS]
                for sigma in SIGMAS}
        baseline = {row["reset_seed"]: row["return"] for row in rows[0.0]}
        means = {sigma_key(sigma): float(np.mean([row["return"] for row in rows[sigma]])) for sigma in SIGMAS}
        deltas = {sigma_key(sigma): [row["return"] - baseline[row["reset_seed"]]
                                     for row in rows[sigma]] for sigma in SIGMAS}
        per_seed[str(training_seed)] = {
            "mean_return": means,
            "mean_length": {sigma_key(sigma): float(np.mean([row["length"] for row in rows[sigma]]))
                            for sigma in SIGMAS},
            "mean_action_clipping_fraction": {
                sigma_key(sigma): float(np.mean([row["action_clipping_fraction"] for row in rows[sigma]]))
                for sigma in SIGMAS},
            "paired_deltas": deltas,
            "paired_delta_mean": {key: float(np.mean(values)) for key, values in deltas.items()},
        }
    for sigma in SIGMAS:
        key = sigma_key(sigma)
        seed_means = [per_seed[str(seed)]["mean_return"][key] for seed in TRAINING_SEEDS]
        delta_means = [per_seed[str(seed)]["paired_delta_mean"][key] for seed in TRAINING_SEEDS]
        sigma_summary[key] = {
            "mean_across_seed_means": float(np.mean(seed_means)),
            "seed_level_statistics": seed_level_statistics(seed_means),
            "mean_paired_delta": float(np.mean(delta_means)),
            "paired_delta_seed_statistics": seed_level_statistics(delta_means),
        }
        delta_summary[key] = {str(seed): per_seed[str(seed)]["paired_delta_mean"][key]
                              for seed in TRAINING_SEEDS}
    return {"episodes": records, "per_seed": per_seed, "summary": sigma_summary,
            "paired_delta_by_training_seed": delta_summary,
            "n_episodes": len(records), "bootstrap_unit": "training_seed"}


def run_evaluation(output=DEFAULT_OUTPUT, output_root=OUTPUT_ROOT):
    validations = {}
    for seed in TRAINING_SEEDS:
        run_dir = SEED0_RUN if seed == 0 else Path(output_root) / f"Hopper-seed{seed}"
        validations[seed] = validate_run(run_dir, seed=seed, legacy_seed0=(seed == 0),
                                         load_best_and_final=False, load_checkpoints=False)
    from sb3_contrib import CrossQ
    import gymnasium as gym

    records = []
    for seed in TRAINING_SEEDS:
        model = CrossQ.load(str(final_checkpoint(seed, output_root)), device="auto")
        env = gym.make("Hopper-v4")
        try:
            for reset_seed in RESET_SEEDS:
                for sigma in SIGMAS:
                    records.append(evaluate_episode(env, model, seed, reset_seed, sigma))
        finally:
            env.close()
    result = summarize_action_noise_records(records)
    result["source_validation_warnings_by_seed"] = source_warnings_by_seed(validations)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite action-noise output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-action-noise-eval", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    if args.dry_run:
        if args.confirm_action_noise_eval:
            parser.error("--dry-run cannot be combined with confirmation")
        print("Action-noise dry-run: 300 episodes from FINAL checkpoints only; "
              "training seeds 0–4, reset seeds 90000–90019, sigma 0/0.05/0.10")
        for seed in TRAINING_SEEDS:
            print(f"seed={seed} final={final_checkpoint(seed, args.output_root)}")
        return 0
    if not args.confirm_action_noise_eval:
        parser.error("live action-noise evaluation requires --confirm-action-noise-eval")
    result = run_evaluation(args.output, args.output_root)
    print(json.dumps({"output": str(args.output), "n_episodes": result["n_episodes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
