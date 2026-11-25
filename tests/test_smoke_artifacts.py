import hashlib
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from crossq_repro.smoke_artifacts import validate_smoke_run


@pytest.fixture
def empty_run(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    return run_dir


def test_empty_artifact_fixture_is_rejected(empty_run):
    with pytest.raises((AssertionError, FileNotFoundError), match="metadata|complete|missing"):
        validate_smoke_run(empty_run)


def test_complete_smoke_artifacts_are_validated(tmp_path, monkeypatch):
    import numpy as np

    run_dir = tmp_path / "Hopper-seed0"
    run_dir.mkdir()
    config = {
        "tasks": {"Hopper": "Hopper-v4"}, "seeds": [0], "max_steps": 1000,
        "evaluation": {"frequency": 500, "episodes": 1, "envs": 1},
        "settings": {"sb3_mappings": {"policy_kwargs": {
            "net_arch": {"value": {"pi": [256, 256], "qf": [2048, 2048]}}
        }}},
        "smoke_overrides": {"classification": "non-paper engineering smoke override"},
        "smoke_protocol": {
                "label": "ENGINEERING SMOKE ONLY", "paper_benchmark": False,
                "seed": 0, "total_timesteps": 1000, "evaluation_envs": 1,
                "evaluation_episodes": 1, "evaluation_frequency": 500,
                "note": "Not the paper 5M x 10-seed benchmark.",
        },
    }
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")
    metadata = {
        "status": "completed", "env_id": "Hopper-v4", "seed": 0,
        "total_timesteps": 1000, "actual_timesteps": 1000,
        "resolved_config": {"config": config},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    evaluations = run_dir / "evaluations"
    evaluations.mkdir()
    np.savez(evaluations / "evaluations.npz", timesteps=np.array([500, 1000]),
             results=np.array([[1.0], [2.0]]), ep_lengths=np.array([[10], [11]]))
    for model_name in ("final_model.zip", "best/best_model.zip"):
        model_path = run_dir / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(model_path, "w") as archive:
            archive.writestr("data", "model")

    loaded = []

    class Model:
        @classmethod
        def load(cls, path, device):
            loaded.append((Path(path).name, device))
            return SimpleNamespace(policy=SimpleNamespace(
                net_arch={"pi": [256, 256], "qf": [2048, 2048]}
            ))

    monkeypatch.setitem(sys.modules, "sb3_contrib", SimpleNamespace(CrossQ=Model))
    validate_smoke_run(run_dir)
    assert loaded == [("final_model.zip", "cpu"), ("best_model.zip", "cpu")]


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_nonfinite_evaluation_metrics_are_rejected(tmp_path, monkeypatch, bad_value):
    import numpy as np

    run_dir = _write_complete_fixture(tmp_path)
    with np.load(run_dir / "evaluations" / "evaluations.npz") as data:
        timesteps = data["timesteps"]
        lengths = data["ep_lengths"]
    np.savez(run_dir / "evaluations" / "evaluations.npz", timesteps=timesteps,
             results=np.array([[bad_value]]), ep_lengths=lengths)
    _mock_model_loader(monkeypatch)
    with pytest.raises(AssertionError, match="finite"):
        validate_smoke_run(run_dir)


def _write_complete_fixture(tmp_path):
    import numpy as np

    run_dir = tmp_path / "complete"
    run_dir.mkdir()
    config = {
        "tasks": {"Hopper": "Hopper-v4"}, "seeds": [0], "max_steps": 1000,
        "evaluation": {"frequency": 500, "episodes": 1, "envs": 1},
        "settings": {"sb3_mappings": {"policy_kwargs": {
            "net_arch": {"value": {"pi": [256, 256], "qf": [2048, 2048]}}
        }}},
        "smoke_overrides": {"classification": "non-paper engineering smoke override"},
        "smoke_protocol": {"label": "ENGINEERING SMOKE ONLY", "paper_benchmark": False,
                           "seed": 0, "total_timesteps": 1000, "evaluation_envs": 1,
                           "evaluation_episodes": 1, "evaluation_frequency": 500,
                           "note": "Not the paper 5M x 10-seed benchmark."},
    }
    path = run_dir / "config.json"
    path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")
    (run_dir / "metadata.json").write_text(json.dumps({
        "status": "completed", "env_id": "Hopper-v4", "seed": 0,
        "total_timesteps": 1000, "actual_timesteps": 1000,
        "resolved_config": {"config": config},
        "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }))
    (run_dir / "evaluations").mkdir()
    np.savez(run_dir / "evaluations" / "evaluations.npz", timesteps=[500, 1000],
             results=[[1.0], [2.0]], ep_lengths=[[10], [11]])
    for name in ("final_model.zip", "best/best_model.zip"):
        target = run_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(target, "w") as archive:
            archive.writestr("data", "model")
    return run_dir


def _mock_model_loader(monkeypatch):
    class Model:
        @classmethod
        def load(cls, path, device):
            return SimpleNamespace(policy=SimpleNamespace(
                net_arch={"pi": [256, 256], "qf": [2048, 2048]}
            ))

    monkeypatch.setitem(sys.modules, "sb3_contrib", SimpleNamespace(CrossQ=Model))
