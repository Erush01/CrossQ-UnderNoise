"""Explicit-entry-point training harness for the frozen CrossQ manifest."""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from scripts.check_envs import resolve_model_kwargs

from .artifacts import build_metadata, write_json

MANIFEST = Path(__file__).parents[1] / "configs" / "paper_mujoco.yaml"


def _load_dependencies():
    import gymnasium as gym
    import torch
    from sb3_contrib import CrossQ
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    return type("Dependencies", (), {
        "gym": gym, "torch": torch, "CrossQ": CrossQ,
        "EvalCallback": EvalCallback, "Monitor": Monitor, "DummyVecEnv": DummyVecEnv,
    })


def _set_random_seed(seed):
    from stable_baselines3.common.utils import set_random_seed

    set_random_seed(seed)


def _load_config(config):
    if config is None:
        return json.loads(MANIFEST.read_text()), MANIFEST
    if isinstance(config, (str, Path)):
        path = Path(config)
        return json.loads(path.read_text()), path
    return config, MANIFEST


def _close_resources(eval_env, raw_eval_envs, train_env):
    first_error = None
    resources = [eval_env] if eval_env is not None else []
    resources.append(train_env)
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.close()
        except Exception as error:
            if first_error is None:
                first_error = error
            if resource is eval_env:
                for raw_env in raw_eval_envs:
                    try:
                        raw_env.close()
                    except Exception as raw_error:
                        if first_error is None:
                            first_error = raw_error
    if eval_env is None:
        for raw_env in raw_eval_envs:
            try:
                raw_env.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
    if first_error is not None:
        raise first_error


def run(config, env_id, seed, total_timesteps, run_dir, device="auto", provenance_paths=()):
    """Train one explicitly selected env/seed and save its checkpoints and metadata."""
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(run_dir)
    config, config_path = _load_config(config)
    if total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if env_id not in config["tasks"].values():
        raise ValueError(f"Unconfigured environment ID: {env_id}")
    if seed not in config["seeds"]:
        raise ValueError(f"Unconfigured seed: {seed}")
    evaluation = config["evaluation"]
    if evaluation["episodes"] <= 0 or evaluation["envs"] <= 0 or evaluation["frequency"] <= 0:
        raise ValueError("evaluation episodes, envs, and frequency must be positive")
    evaluation_seed = seed + 10000
    config_snapshot = json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n"
    dependencies = _load_dependencies()
    _set_random_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    snapshot_path = run_dir / "config.json"
    snapshot_path.write_text(config_snapshot)
    started_at = datetime.now(timezone.utc).isoformat()
    train_env = None
    eval_env = None
    raw_eval_envs = []
    model = None
    try:
        train_env = dependencies.gym.make(env_id)
        for _ in range(evaluation["envs"]):
            raw_eval_envs.append(dependencies.gym.make(env_id))
            raw_eval_envs[-1] = dependencies.Monitor(raw_eval_envs[-1])
        eval_env = dependencies.DummyVecEnv([
            lambda env=raw_env: env for raw_env in raw_eval_envs
        ])
        eval_env.seed(evaluation_seed)
        constructor_kwargs, policy_kwargs = resolve_model_kwargs(config, dependencies.torch)
        model = dependencies.CrossQ(
            "MlpPolicy", train_env, **constructor_kwargs, policy_kwargs=policy_kwargs,
            seed=seed, device=device,
        )
        callback = dependencies.EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "evaluations"),
            eval_freq=evaluation["frequency"],
            n_eval_episodes=evaluation["episodes"],
            deterministic=True,
        )
        model.learn(total_timesteps=total_timesteps, callback=callback)
        actual_timesteps = int(getattr(model, "num_timesteps", total_timesteps))
        model.save(str(run_dir / "final_model"))
        finished_at = datetime.now(timezone.utc).isoformat()
        sources = [
            Path(__file__), Path(__file__).with_name("evaluate.py"),
            Path(__file__).with_name("artifacts.py"),
            Path(__file__).parents[1] / "scripts" / "check_envs.py",
        ]
        sources.extend(Path(path) for path in provenance_paths)
        metadata_policy_kwargs = dict(policy_kwargs)
        activation_fn = metadata_policy_kwargs.get("activation_fn")
        if isinstance(activation_fn, type):
            metadata_policy_kwargs["activation_fn"] = (
                f"{activation_fn.__module__}.{activation_fn.__qualname__}"
            )
        metadata = build_metadata(
            {"config": config, "constructor_kwargs": constructor_kwargs,
             "policy_kwargs": metadata_policy_kwargs, "evaluation": evaluation,
              "evaluation_seed": evaluation_seed,
              "device": {"requested": device, "resolved": str(model.device)}},
            env_id, seed, total_timesteps, started_at, finished_at,
            config_path=snapshot_path, source_paths=sources,
        )
        metadata["status"] = "completed"
        metadata["actual_timesteps"] = actual_timesteps
        write_json(run_dir / "metadata.json", metadata)
    except BaseException:
        try:
            _close_resources(eval_env, raw_eval_envs, train_env)
        except BaseException:
            pass
        raise
    else:
        _close_resources(eval_env, raw_eval_envs, train_env)
    return run_dir
