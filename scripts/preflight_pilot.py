#!/usr/bin/env python3
"""Non-training resource and environment-shape checks for the approved pilot."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))


def estimate_replay_buffer_bytes(observation_shape, action_shape, capacity):
    obs = 1
    for dim in observation_shape:
        obs *= int(dim)
    act = 1
    for dim in action_shape:
        act *= int(dim)
    return int(capacity) * (2 * obs + act + 2) * 4


def evaluate_resources(*, required_memory_bytes, available_memory_bytes,
                       required_disk_bytes, available_disk_bytes, device, gpu_available,
                       gpu_free_bytes=None):
    failures = []
    if available_memory_bytes < required_memory_bytes:
        failures.append(f"Insufficient memory (RAM): require {required_memory_bytes} bytes, have {available_memory_bytes}")
    if available_disk_bytes < required_disk_bytes:
        failures.append(f"Insufficient disk: require {required_disk_bytes} bytes, have {available_disk_bytes}")
    if device == "auto" and gpu_available and (gpu_free_bytes is None or gpu_free_bytes < 2 * 1024**3):
        failures.append(f"Insufficient CUDA free memory: auto-CUDA requires at least 2147483648 bytes, have {gpu_free_bytes}")
    return {
        "ok": not failures, "failures": failures,
        "device_note": "auto will resolve to CUDA" if device == "auto" and gpu_available
        else "auto will resolve to CPU" if device == "auto" else f"requested {device}",
    }


def run_preflight(target_dir):
    import gymnasium as gym
    import torch
    config = json.loads((ROOT / "configs/pilot_humanoidstandup_100k.json").read_text())
    env = gym.make("HumanoidStandup-v4")
    try:
        obs_shape, action_shape = env.observation_space.shape, env.action_space.shape
    finally:
        env.close()
    capacity = config["settings"]["sb3_mappings"]["constructor_kwargs"]["buffer_size"]["value"]
    replay_bytes = estimate_replay_buffer_bytes(obs_shape, action_shape, capacity)
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
    except ImportError:
        available_ram = int(Path("/proc/meminfo").read_text().split("MemAvailable:")[1].split()[0]) * 1024
    disk_path = target_dir.parent
    while not disk_path.exists():
        disk_path = disk_path.parent
    disk = shutil.disk_usage(disk_path)
    required_disk = replay_bytes + 5 * 1024**3
    gpu_available = torch.cuda.is_available()
    gpu_free_bytes = torch.cuda.mem_get_info()[0] if gpu_available else None
    report = evaluate_resources(
        required_memory_bytes=int(replay_bytes * 1.5), available_memory_bytes=available_ram,
        required_disk_bytes=required_disk, available_disk_bytes=disk.free,
        device="auto", gpu_available=gpu_available, gpu_free_bytes=gpu_free_bytes,
    )
    report.update({"env_id": "HumanoidStandup-v4", "observation_shape": obs_shape,
                   "action_shape": action_shape, "replay_buffer_bytes_estimate": replay_bytes,
                   "available_ram_bytes": available_ram, "available_disk_bytes": disk.free,
                   "gpu_available": gpu_available, "gpu_free_bytes": gpu_free_bytes,
                   "target_dir": str(target_dir)})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run_preflight(args.target)
    except Exception as error:
        report = {"ok": False, "failures": [f"Environment/resource preflight failed: {error}"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
