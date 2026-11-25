"""Validation for bounded engineering-smoke run artifacts."""

import hashlib
import json
from pathlib import Path

def validate_smoke_run(run_dir):
    run_dir = Path(run_dir)
    metadata_path = run_dir / "metadata.json"
    config_path = run_dir / "config.json"
    if not metadata_path.is_file() or not config_path.is_file():
        raise AssertionError("missing metadata or config artifact")
    metadata = json.loads(metadata_path.read_text())
    config = json.loads(config_path.read_text())
    protocol = config.get("smoke_protocol", {})
    if metadata.get("status") != "completed":
        raise AssertionError("run status is not complete")
    if metadata.get("config_sha256") != hashlib.sha256(config_path.read_bytes()).hexdigest():
        raise AssertionError("config hash does not match exact config artifact")
    if metadata.get("resolved_config", {}).get("config") != config:
        raise AssertionError("resolved config differs from config artifact")
    if protocol.get("label") != "ENGINEERING SMOKE ONLY" or protocol.get("paper_benchmark") is not False:
        raise AssertionError("missing non-paper engineering smoke protocol metadata")
    if "5M" not in protocol.get("note", "") or "10-seed" not in protocol.get("note", ""):
        raise AssertionError("protocol must distinguish the paper 5M x 10-seed benchmark")
    steps = protocol.get("total_timesteps")
    if steps != 1000 or metadata.get("total_timesteps") != steps or metadata.get("actual_timesteps") != steps:
        raise AssertionError("requested and actual total timesteps must both be exactly 1000")
    if metadata.get("seed") != 0 or protocol.get("seed") != 0:
        raise AssertionError("smoke seed must be fixed to 0")
    if (config.get("max_steps") != steps or config.get("seeds") != [0]
            or config.get("evaluation") != {"frequency": 500, "episodes": 1, "envs": 1}
            or protocol.get("evaluation_envs") != 1
            or protocol.get("evaluation_episodes") != 1
            or protocol.get("evaluation_frequency") != 500):
        raise AssertionError("smoke config does not match bounded evaluation protocol")
    if config.get("smoke_overrides", {}).get("classification") != "non-paper engineering smoke override":
        raise AssertionError("smoke overrides are not identified as non-paper")
    critic_width = config["settings"]["sb3_mappings"]["policy_kwargs"]["net_arch"]["value"]["qf"]
    if critic_width != [2048, 2048]:
        raise AssertionError(f"unexpected critic width: {critic_width}")

    import numpy as np

    evaluation_path = run_dir / "evaluations" / "evaluations.npz"
    if not evaluation_path.is_file():
        raise AssertionError("missing SB3 EvalCallback evaluations/evaluations.npz")
    with np.load(evaluation_path) as evaluation:
        required = {"timesteps", "results", "ep_lengths"}
        if not required.issubset(evaluation.files):
            raise AssertionError("evaluation NPZ is missing EvalCallback arrays")
        timesteps = evaluation["timesteps"]
        results = evaluation["results"]
        lengths = evaluation["ep_lengths"]
        if (timesteps.size == 0 or results.shape != (timesteps.size, 1)
                or lengths.shape != results.shape or not np.isfinite(timesteps).all()
                or not np.isfinite(results).all() or not np.isfinite(lengths).all()):
            raise AssertionError("evaluation metrics must be non-empty and finite for one episode")

    from sb3_contrib import CrossQ

    for checkpoint in (run_dir / "final_model.zip", run_dir / "best" / "best_model.zip"):
        if not checkpoint.is_file():
            raise AssertionError(f"missing model artifact: {checkpoint.name}")
        try:
            model = CrossQ.load(str(checkpoint), device="cpu")
        except Exception as error:
            raise AssertionError(f"model checkpoint is not loadable: {checkpoint.name}") from error
        if model.policy.net_arch.get("qf") != [2048, 2048]:
            raise AssertionError(f"loaded model has unexpected critic width: {checkpoint.name}")
    return metadata
