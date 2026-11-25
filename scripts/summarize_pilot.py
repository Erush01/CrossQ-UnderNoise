#!/usr/bin/env python3
"""Validate and summarize completed approved pilot artifacts."""
import argparse
import ast
import hashlib
import json
import re
import struct
import math
import zipfile
from datetime import datetime
from pathlib import Path


def _read_npy(member):
    data = member.read()
    if data[:6] != b"\x93NUMPY":
        raise ValueError("invalid evaluation array")
    major = data[6]
    header_len = struct.unpack("<H" if major == 1 else "<I", data[8:10] if major == 1 else data[8:12])[0]
    start = 10 if major == 1 else 12
    header = ast.literal_eval(data[start:start + header_len].decode("latin1").strip())
    dtype = header["descr"]
    shape = header["shape"]
    payload = data[start + header_len:]
    format_codes = {"<i8": "q", "<f8": "d", "<i4": "i", "<f4": "f"}
    code = format_codes.get(dtype)
    if code is None:
        raise ValueError(f"unsupported evaluation dtype: {dtype}")
    size = struct.calcsize(code)
    values = list(struct.unpack("<" + code * (len(payload) // size), payload))
    count = math.prod(shape) if shape else 1
    if len(values) != count:
        raise ValueError("truncated evaluation array")
    return shape, values


def _critic_is_2048(model):
    networks = model.policy.critic.q_networks
    if len(networks) != 2:
        return False
    for network in networks:
        widths = [layer.out_features for layer in network.modules()
                  if hasattr(layer, "out_features")]
        if widths != [2048, 2048, 1]:
            return False
    return True


def validate_artifacts(run_dir, checkpoint_loader=None):
    run_dir = Path(run_dir)
    required = ["metadata.json", "config.json", "final_model.zip", "best/best_model.zip",
                "evaluations/evaluations.npz"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise ValueError(f"incomplete artifacts: {', '.join(missing)}")
    metadata = json.loads((run_dir / "metadata.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    expected = {"status": "completed", "env_id": "HumanoidStandup-v4", "seed": 0,
                "total_timesteps": 100000, "actual_timesteps": 100000}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch: {key} expected {value}, got {metadata.get(key)}")
    root = Path(__file__).parents[1]
    approved_config = json.loads((root / "configs/pilot_humanoidstandup_100k.json").read_text())
    paper_config = json.loads((root / "configs/paper_mujoco.yaml").read_text())
    if config != approved_config:
        raise ValueError("config mismatch with approved pilot")
    if config.get("evaluation") != {"frequency": 25000, "episodes": 20, "envs": 5}:
        raise ValueError("config mismatch: evaluation protocol")
    if config.get("settings", {}).get("paper", {}).get("critic_architecture", {}).get("value") != [2048, 2048]:
        raise ValueError("config mismatch: critic architecture")
    expected_versions = paper_config["implementation_versions"]
    actual_versions = metadata.get("package_versions", {})
    version_keys = {"stable_baselines3": "stable-baselines3", "sb3_contrib": "sb3-contrib",
                    "gymnasium": "gymnasium", "mujoco": "mujoco"}
    if config.get("implementation_versions") != expected_versions or any(
            actual_versions.get(version_keys[key]) != version for key, version in expected_versions.items()):
        raise ValueError("package/config version mismatch")
    if metadata.get("resolved_config", {}).get("config") != config:
        raise ValueError("metadata mismatch: resolved config does not match config.json")
    if metadata.get("config_sha256") is None:
        raise ValueError("metadata mismatch: missing config hash")
    actual_config_hash = hashlib.sha256((run_dir / "config.json").read_bytes()).hexdigest()
    if metadata["config_sha256"] != actual_config_hash:
        raise ValueError("metadata mismatch: config hash does not match config.json")
    source_hashes = metadata.get("source_sha256", {})
    for filename in ("runner.py", "run_pilot.sh", "preflight_pilot.py"):
        source_path = root / ("crossq_repro/runner.py" if filename == "runner.py" else f"scripts/{filename}")
        matches = [value for key, value in source_hashes.items() if Path(key).name == filename]
        if not matches or hashlib.sha256(source_path.read_bytes()).hexdigest() not in matches:
            raise ValueError(f"metadata mismatch: missing or invalid source hash for {filename}")
    with zipfile.ZipFile(run_dir / "evaluations/evaluations.npz") as archive:
        names = {name.removesuffix(".npy") for name in archive.namelist()}
        if not {"timesteps", "results", "ep_lengths"}.issubset(names):
            raise ValueError("incomplete evaluation arrays")
        shapes_values = {name: _read_npy(archive.open(name + ".npy")) for name in ("timesteps", "results", "ep_lengths")}
    ts_shape, timesteps = shapes_values["timesteps"]
    results_shape, results = shapes_values["results"]
    lengths_shape, lengths = shapes_values["ep_lengths"]
    if (ts_shape != (4,) or timesteps != [25000, 50000, 75000, 100000]
            or results_shape != (4, 20) or lengths_shape != (4, 20)):
        raise ValueError("incomplete evaluation curves")
    if any(not math.isfinite(value) for value in results + lengths) or any(value <= 0 for value in lengths):
        raise ValueError("invalid non-finite return/length or non-positive episode length")
    if checkpoint_loader is None:
        from sb3_contrib import CrossQ
        checkpoint_loader = lambda path: CrossQ.load(str(path), device="cpu")
    for checkpoint in (run_dir / "best/best_model.zip", run_dir / "final_model.zip"):
        try:
            model = checkpoint_loader(checkpoint)
        except Exception as error:
            raise ValueError(f"invalid checkpoint {checkpoint.name}: CrossQ.load failed: {error}") from error
        if not _critic_is_2048(model):
            raise ValueError(f"checkpoint critic architecture is not two [2048, 2048, 1] critics: {checkpoint.name}")
    return {"metadata": metadata, "config": config, "env_id": metadata["env_id"], "timesteps": timesteps,
            "results": results, "results_shape": results_shape, "lengths": lengths,
            "lengths_shape": lengths_shape, "run_dir": run_dir}


def summarize(run_dir):
    import numpy as np
    a = validate_artifacts(run_dir)
    meta, ts = a["metadata"], a["timesteps"]
    returns = np.asarray(a["results"]).reshape(a["results_shape"])
    lengths = np.asarray(a["lengths"]).reshape(a["lengths_shape"])
    start = datetime.fromisoformat(meta["started_at"])
    finish = datetime.fromisoformat(meta["finished_at"])
    report = {
        "runtime_seconds": (finish - start).total_seconds(),
        "acceptance_target_seconds": 4 * 60 * 60,
        "max_rss": None,
        "reward_curve": [{"timestep": int(t), "mean": float(np.mean(r)), "std": float(np.std(r))} for t, r in zip(ts, returns)],
        "length_curve": [{"timestep": int(t), "mean": float(np.mean(l))} for t, l in zip(ts, lengths)],
        "config_sha256": meta.get("config_sha256"), "source_sha256": meta.get("source_sha256"),
        "package_versions": meta.get("package_versions"),
        "resolved_device": meta.get("resolved_config", {}).get("device", {}).get("resolved"),
        "checkpoints": ["best/best_model.zip", "final_model.zip"], "deviations": [],
    }
    time_file = Path(run_dir).parents[1] / "logs"
    candidates = sorted(time_file.glob("pilot-time-*.txt")) if time_file.exists() else []
    if candidates:
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", candidates[-1].read_text())
        if match:
            report["max_rss"] = {"value": int(match.group(1)), "unit": "kB", "source": str(candidates[-1])}
    if meta.get("actual_timesteps") != 100000:
        report["deviations"].append(f"actual_timesteps={meta.get('actual_timesteps')}")
    report["acceptance_target_met"] = report["runtime_seconds"] <= report["acceptance_target_seconds"]
    if not report["acceptance_target_met"]:
        report["deviations"].append("runtime exceeded four-hour acceptance target")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    validate_artifacts(args.run_dir)
    if args.validate_only:
        print("Pilot artifacts valid")
    else:
        print(json.dumps(summarize(args.run_dir), indent=2))


if __name__ == "__main__":
    main()
