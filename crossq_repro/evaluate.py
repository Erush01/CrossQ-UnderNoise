"""Checkpoint evaluation preserving each episode's return and length."""

import json
import statistics
from pathlib import Path


def summarize_episodes(episode_returns, episode_lengths):
    returns = [float(value) for value in episode_returns]
    lengths = [int(value) for value in episode_lengths]
    return {
        "episode_returns": returns,
        "episode_lengths": lengths,
        "mean_return": statistics.mean(returns),
        "std_return": statistics.pstdev(returns),
    }


def evaluate(checkpoint, env_id, episodes, seed=0, output_path=None, device="auto"):
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    import gymnasium as gym
    from sb3_contrib import CrossQ

    model = CrossQ.load(str(checkpoint), device=device)
    env = gym.make(env_id)
    returns, lengths = [], []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            total, length = 0.0, 0
            while True:
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                length += 1
                if terminated or truncated:
                    break
            returns.append(total)
            lengths.append(length)
    finally:
        env.close()
    result = {"env_id": env_id, "seed": seed, "episodes": summarize_episodes(returns, lengths)}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(result, indent=2) + "\n")
    return result
