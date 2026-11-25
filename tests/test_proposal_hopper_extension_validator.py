import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
VERSIONS = {
    "stable-baselines3": "2.9.0",
    "sb3-contrib": "2.9.0",
    "gymnasium": "1.4.0",
    "mujoco": "3.16.0",
}


def _validator():
    module_name = "scripts.validate_proposal_hopper_extension"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"missing extension validator module: {module_name}"
    return importlib.import_module(module_name)


def _write_extension_run(tmp_path, seed=5, *, wrong_timesteps=False):
    validator = _validator()
    config = validator.runner.load_config()
    run_dir = tmp_path / f"Hopper-seed{seed}"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "evaluations").mkdir()
    snapshot_path = run_dir / "config.json"
    snapshot_path.write_text(json.dumps(config, sort_keys=True, separators=(",", ":")) + "\n")
    (run_dir / "best/best_model.zip").write_bytes(b"best")
    (run_dir / "final_model.zip").write_bytes(b"final")
    timesteps = validator.EXPECTED_TIMESTEPS
    if wrong_timesteps:
        timesteps = [25_000]
    np.savez(
        run_dir / "evaluations/evaluations.npz",
        timesteps=np.asarray(timesteps),
        results=np.full((len(timesteps), 20), float(seed)),
        ep_lengths=np.full((len(timesteps), 20), 100),
    )
    resolved = validator.base_validator._expected_resolved(config, seed)
    metadata = {
        "status": "completed",
        "env_id": "Hopper-v4",
        "seed": seed,
        "total_timesteps": 5_000_000,
        "actual_timesteps": 5_000_000,
        "config_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "resolved_config": {
            "config": config,
            "constructor_kwargs": resolved["constructor_kwargs"],
            "policy_kwargs": resolved["policy_kwargs"],
            "evaluation": resolved["evaluation"],
            "evaluation_seed": resolved["evaluation_seed"],
            "device": {"requested": "auto", "resolved": "cpu"},
        },
        "package_versions": VERSIONS,
        "source_sha256": {
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in validator.EXTENSION_SOURCES
        },
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    return run_dir


class FakeLinear:
    def __init__(self, width):
        self.out_features = width


class FakeNetwork:
    def modules(self):
        return [FakeLinear(2048), FakeLinear(2048), FakeLinear(1)]


class FakeCrossQ:
    policy = SimpleNamespace(critic=SimpleNamespace(q_networks=[FakeNetwork(), FakeNetwork()]))


def test_seed_five_extension_run_validates_config_sources_evaluations_and_checkpoints(tmp_path):
    validator = _validator()
    run_dir = _write_extension_run(tmp_path, seed=5)
    loaded = []

    result = validator.validate_extension_run(
        run_dir,
        seed=5,
        checkpoint_loader=lambda path: loaded.append(Path(path)) or FakeCrossQ(),
    )

    assert result["seed"] == 5
    assert result["manifest_scope"] == "proposal-ten-seed-extension"
    assert result["timesteps"] == validator.EXPECTED_TIMESTEPS
    assert loaded == [run_dir / "best/best_model.zip", run_dir / "final_model.zip"]


def test_extension_validator_rejects_non_extension_seeds_and_incomplete_evaluation(tmp_path):
    validator = _validator()
    run_dir = _write_extension_run(tmp_path, seed=5, wrong_timesteps=True)
    with pytest.raises(ValueError, match="extension seeds 5 through 9"):
        validator.validate_extension_run(run_dir, seed=4, checkpoint_loader=lambda _: FakeCrossQ())
    with pytest.raises(ValueError, match="timestep sequence"):
        validator.validate_extension_run(run_dir, seed=5, checkpoint_loader=lambda _: FakeCrossQ())


def test_extension_validator_requires_exact_recorded_source_paths(tmp_path):
    validator = _validator()
    run_dir = _write_extension_run(tmp_path, seed=5)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected_path = next(path for path in validator.EXTENSION_SOURCES
                         if path.name == "run_proposal_hopper_extension.py")
    digest = metadata["source_sha256"].pop(str(expected_path))
    metadata["source_sha256"][f"/different/location/{expected_path.name}"] = digest
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="provenance path set"):
        validator.validate_extension_run(
            run_dir, seed=5, checkpoint_loader=lambda _: FakeCrossQ()
        )


def test_ten_seed_summary_uses_training_seed_iqm_and_rejects_missing_seeds():
    validator = _validator()
    runs = [
        {
            "seed": seed,
            "episode_returns": [[float(seed * 10 + step)] * 20 for step in range(200)],
            "episode_lengths": [[100] * 20 for _ in range(200)],
        }
        for seed in range(10)
    ]

    summary = validator.summarize_ten_seed_runs(runs, bootstrap_samples=100, random_seed=123)
    assert summary["seed_level_statistics"]["n_training_seeds"] == 10
    assert summary["seed_level_statistics"]["bootstrap_unit"] == "training_seed"
    assert summary["seed_level_statistics"]["iqm"] == pytest.approx(244.0)
    assert len(summary["per_seed"]) == 10
    assert len(summary["per_seed"]["9"]["episode_returns"]) == 200
    with pytest.raises(ValueError, match="exactly ten training seeds"):
        validator.summarize_ten_seed_runs(runs[:9])


def test_full_summary_combines_validated_original_seeds_and_extension_seeds(
    tmp_path, monkeypatch
):
    validator = _validator()
    base_seeds = {
        str(seed): {
            "episode_returns": [[float(seed * 10 + step)] * 20 for step in range(200)],
            "episode_lengths": [[100] * 20 for _ in range(200)],
        }
        for seed in range(5)
    }
    monkeypatch.setattr(
        validator.base_validator,
        "validate_all_runs",
        lambda **_: {"seeds": base_seeds, "legacy_manifest_scope": "legacy-three-task-three-seed"},
    )
    monkeypatch.setattr(
        validator,
        "validate_extension_run",
        lambda run_dir, seed, **_: {
            "seed": seed,
            "episode_returns": [[float(seed * 10 + step)] * 20 for step in range(200)],
            "episode_lengths": [[100] * 20 for _ in range(200)],
            "run_dir": str(run_dir),
        },
    )

    result = validator.validate_all_runs(output_root=tmp_path)
    assert set(result["seeds"]) == {str(seed) for seed in range(10)}
    assert result["seed_level_statistics"]["n_training_seeds"] == 10
    assert result["legacy_manifest_scope"] == "legacy-three-task-three-seed"
