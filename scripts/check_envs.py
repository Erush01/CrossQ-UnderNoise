#!/usr/bin/env python3
"""Smoke-check the configured MuJoCo environments and CrossQ API."""

import argparse
import json
from pathlib import Path


def require_box_action_space(action_space, gymnasium=None):
    if gymnasium is None:
        import gymnasium
    if not isinstance(action_space, gymnasium.spaces.Box):
        raise TypeError(f"Expected a continuous Box action space, got {type(action_space).__name__}")


def _is_finite(value):
    import numpy as np

    return bool(np.isfinite(np.asarray(value)).all())


def resolve_model_kwargs(config, torch_module):
    """Translate the frozen manifest's CrossQ settings into constructor kwargs."""
    mappings = config["settings"]["sb3_mappings"]
    constructor_kwargs = {
        key: entry["value"] for key, entry in mappings["constructor_kwargs"].items()
    }
    policy_kwargs = {key: entry["value"] for key, entry in mappings["policy_kwargs"].items()}
    policy_kwargs["activation_fn"] = torch_module.nn.ReLU
    optimizer_kwargs = dict(policy_kwargs["optimizer_kwargs"])
    optimizer_kwargs["betas"] = tuple(optimizer_kwargs["betas"])
    policy_kwargs["optimizer_kwargs"] = optimizer_kwargs
    return constructor_kwargs, policy_kwargs


def close_environment(raw_env, model=None):
    """Close the model's wrapper when available, otherwise close the raw env."""
    model_env = getattr(model, "env", None) if model is not None else None
    (model_env if model_env is not None else raw_env).close()


def validate_step_result(observation, reward, terminated, truncated):
    import math

    if not isinstance(terminated, bool) or not isinstance(truncated, bool):
        raise TypeError("terminated and truncated must be bool values")

    if not _is_finite(observation) or not math.isfinite(float(reward)):
        raise AssertionError("step returned non-finite observation or reward")


def _assert_linear_widths(module, expected_widths, description):
    import torch

    widths = [layer.out_features for layer in module.modules() if isinstance(layer, torch.nn.Linear)]
    if widths != expected_widths:
        raise AssertionError(f"Unexpected {description} Linear widths: {widths}, expected {expected_widths}")


def _assert_crossq_model(model, constructor_kwargs, policy_kwargs):
    import torch
    from sb3_contrib.common.torch_layers import BatchRenorm1d

    for name, expected in constructor_kwargs.items():
        if name == "train_freq":
            actual = model.train_freq
            unit = getattr(actual.unit, "value", actual.unit)
            if actual.frequency != 1 or unit != "step":
                raise AssertionError(f"Unexpected CrossQ train_freq: {actual}")
            continue
        actual = getattr(model, name)
        if actual != expected:
            raise AssertionError(f"Unexpected CrossQ {name}: {actual!r}, expected {expected!r}")

    policy = model.policy
    if policy.net_arch != policy_kwargs["net_arch"]:
        raise AssertionError(f"Unexpected CrossQ architecture: {policy.net_arch}")
    if policy.activation_fn is not torch.nn.ReLU:
        raise AssertionError("CrossQ activation is not ReLU")
    if policy.batch_norm_params != {
        "batch_norm": policy_kwargs["batch_norm"],
        "batch_norm_momentum": policy_kwargs["batch_norm_momentum"],
        "batch_norm_eps": policy_kwargs.get("batch_norm_eps", 0.001),
        "renorm_warmup_steps": policy_kwargs["renorm_warmup_steps"],
    }:
        raise AssertionError(f"Unexpected CrossQ BRN settings: {policy.batch_norm_params}")
    if policy.critic_kwargs["n_critics"] != policy_kwargs["n_critics"]:
        raise AssertionError(f"Unexpected CrossQ n_critics: {policy.critic_kwargs['n_critics']}")

    expected_betas = policy_kwargs["optimizer_kwargs"]["betas"]
    for label, optimizer in (("actor", policy.actor.optimizer), ("critic", policy.critic.optimizer)):
        for param_group in optimizer.param_groups:
            if tuple(param_group["betas"]) != expected_betas:
                raise AssertionError(f"Unexpected {label} optimizer betas: {param_group['betas']}")

    actor_widths = policy.net_arch["pi"]
    critic_widths = policy.net_arch["qf"]
    _assert_linear_widths(policy.actor.latent_pi, actor_widths, "actor")
    critics = policy.critic.q_networks
    if len(critics) != policy_kwargs["n_critics"]:
        raise AssertionError(f"Expected {policy_kwargs['n_critics']} Q networks, got {len(critics)}")
    for index, critic in enumerate(critics):
        _assert_linear_widths(critic, [*critic_widths, 1], f"critic {index}")

    renorm_layers = [module for module in policy.modules() if isinstance(module, BatchRenorm1d)]
    if not renorm_layers:
        raise AssertionError("CrossQ policy contains no BatchRenorm layers")
    for layer in renorm_layers:
        if (layer.momentum != policy_kwargs["batch_norm_momentum"]
                or layer.warmup_steps != policy_kwargs["renorm_warmup_steps"]):
            raise AssertionError("CrossQ BatchRenorm layer parameters do not match the manifest")

    if model.tau != 1.0:
        raise AssertionError(f"Expected no target-network updates (tau=1.0), got {model.tau}")
    target_names = ("actor_target", "critic_target", "target_actor", "target_critic")
    if any(hasattr(model, name) or hasattr(policy, name) for name in target_names):
        raise AssertionError("Unexpected target-network attribute")
    if any("target" in name.lower() for name, _ in policy.named_modules()):
        raise AssertionError("Unexpected target-network module")


def check_environment(env_id, settings):
    import gymnasium as gym
    import numpy as np
    import torch
    from sb3_contrib import CrossQ

    env = None
    model = None
    try:
        env = gym.make(env_id)
        require_box_action_space(env.action_space)
        observation, _ = env.reset(seed=123)
        if not _is_finite(observation):
            raise AssertionError(f"{env_id}: reset observation is not finite")

        action = np.clip(
            np.zeros(env.action_space.shape, dtype=env.action_space.dtype),
            env.action_space.low,
            env.action_space.high,
        )
        if not env.action_space.contains(action):
            raise AssertionError(f"{env_id}: clipped zero action is not valid for its Box space")
        observation, reward, terminated, truncated, _ = env.step(action)
        validate_step_result(observation, reward, terminated, truncated)

        constructor_kwargs, policy_kwargs = resolve_model_kwargs(settings["config"], torch)
        model = CrossQ(
            "MlpPolicy", env, **constructor_kwargs, policy_kwargs=policy_kwargs,
            seed=123, device="auto",
        )
        _assert_crossq_model(model, constructor_kwargs, policy_kwargs)
        print(f"PASS {env_id}: reset, step, Box action space, CrossQ construction")
    finally:
        if env is not None:
            close_environment(env, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    settings = {"config": config}
    for task, env_id in config["tasks"].items():
        print(f"Checking {task} ({env_id})")
        check_environment(env_id, settings)


if __name__ == "__main__":
    main()
