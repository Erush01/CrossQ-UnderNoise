import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_proposal_hopper as proposal_runner
from scripts import validate_proposal_hopper as validator

CONFIG_PATH = ROOT / "configs/proposal_hopper_5seeds_5m.yaml"
LEGACY_RUN = ROOT / "artifacts/scaled_reproduction_retry/Hopper-seed0"
RETRY_ROOT = ROOT / "artifacts/proposal_hopper"
VERSIONS = {"stable-baselines3": "2.9.0", "sb3-contrib": "2.9.0",
            "gymnasium": "1.4.0", "mujoco": "3.16.0"}


def test_proposal_manifest_is_frozen_hopper_five_seed_subset():
    config = json.loads(CONFIG_PATH.read_text())
    scaled = json.loads((ROOT / "configs/scaled_3tasks_3seeds_5m.yaml").read_text())
    expected = copy.deepcopy(scaled)
    expected["tasks"] = {"Hopper": "Hopper-v4"}
    expected["task_provenance"] = {"Hopper": scaled["task_provenance"]["Hopper"]}
    expected["seeds"] = [0, 1, 2, 3, 4]
    assert config == expected
    assert config["max_steps"] == 5_000_000
    assert config["evaluation"] == {"frequency": 25_000, "episodes": 20, "envs": 5}


def test_proposal_dry_run_lists_seeds_one_through_four_without_targets(tmp_path, capsys):
    output_root = tmp_path / "fresh-output"
    jobs = proposal_runner.build_jobs(output_root=output_root)
    assert [(job["env_id"], job["seed"], job["steps"]) for job in jobs] == [
        ("Hopper-v4", seed, 5_000_000) for seed in (1, 2, 3, 4)
    ]
    assert all(not Path(job["run_dir"]).exists() for job in jobs)
    assert proposal_runner.main(["--dry-run", "--output-root", str(output_root)]) == 0
    output = capsys.readouterr().out
    assert output.count("task=Hopper-v4") == 4
    assert all(f"seed={seed}" in output for seed in (1, 2, 3, 4))
    assert "seed=0" not in output
    assert not output_root.exists()


def test_proposal_runner_requires_confirmation_exact_session_versions_and_fresh_target(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="confirmation"):
        proposal_runner.run_seed(1)
    assert proposal_runner.expected_tmux_session(1) == "cmp620-crossq-proposal-hopper-seed1"
    with pytest.raises(RuntimeError, match="cmp620-crossq-proposal-hopper-seed1"):
        proposal_runner.validate_tmux_session(1, "other-session")
    with pytest.raises(ValueError):
        proposal_runner.validate_tmux_session(0, "cmp620-crossq-proposal-hopper-seed0")
    target_root = tmp_path / "proposal_hopper"
    target = target_root / "Hopper-seed1"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        proposal_runner.validate_seed_preconditions(1, output_root=target_root,
                                                     version_lookup=lambda: VERSIONS)
    target.rmdir()
    bad_versions = dict(VERSIONS, **{"sb3-contrib": "0.0"})
    with pytest.raises(ValueError, match="version mismatch"):
        proposal_runner.validate_seed_preconditions(1, output_root=target_root,
                                                     version_lookup=lambda: bad_versions)
    with pytest.raises(ValueError):
        proposal_runner.validate_seed_preconditions(0, output_root=target_root,
                                                     version_lookup=lambda: VERSIONS)


def test_proposal_cli_requires_one_explicit_confirmed_seed_and_named_session():
    with pytest.raises(SystemExit):
        proposal_runner.main(["--seed", "1"])
    with pytest.raises(SystemExit):
        proposal_runner.main(["--seed", "0", "--confirm-proposal-run"])
    with pytest.raises(SystemExit):
        proposal_runner.main(["--seed", "1", "--confirm-proposal-run", "--dry-run"])


def test_legacy_seed0_manifest_difference_and_real_completed_artifacts_validate():
    result = validator.validate_run(LEGACY_RUN, seed=0, checkpoint_loader=lambda path: FakeCrossQ())
    assert result["seed"] == 0 and result["env_id"] == "Hopper-v4"
    assert result["manifest_scope"] == "legacy-three-task-three-seed"
    assert result["timesteps"] == list(range(25_000, 5_000_001, 25_000))


def test_legacy_seed0_records_changed_scaled_launcher_hashes_as_warnings():
    metadata = json.loads((LEGACY_RUN / "metadata.json").read_text())
    warnings = validator._check_sources(metadata, legacy_seed0=True)
    assert any("scaled_reproduction_jobs.py" in warning for warning in warnings)
    assert any("run_scaled_reproduction.sh" in warning for warning in warnings)


def _write_proposal_run(tmp_path, seed=1, *, timesteps=None, task="Hopper-v4"):
    config = json.loads(CONFIG_PATH.read_text())
    run_dir = tmp_path / f"Hopper-seed{seed}"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "evaluations").mkdir()
    snapshot = run_dir / "config.json"
    snapshot.write_text(json.dumps(config))
    (run_dir / "best/best_model.zip").write_bytes(b"best")
    (run_dir / "final_model.zip").write_bytes(b"final")
    points = validator.EXPECTED_TIMESTEPS if timesteps is None else timesteps
    np.savez(run_dir / "evaluations/evaluations.npz", timesteps=np.asarray(points),
             results=np.ones((len(points), 20)), ep_lengths=np.full((len(points), 20), 100))
    resolved = validator._expected_resolved(config, seed)
    metadata = {
        "status": "completed", "env_id": task, "seed": seed,
        "total_timesteps": 5_000_000, "actual_timesteps": 5_000_000,
        "config_sha256": __import__("hashlib").sha256(snapshot.read_bytes()).hexdigest(),
        "resolved_config": {
            "config": config, "constructor_kwargs": resolved["constructor_kwargs"],
            "policy_kwargs": resolved["policy_kwargs"], "evaluation": resolved["evaluation"],
            "evaluation_seed": resolved["evaluation_seed"],
            "device": {"requested": "auto", "resolved": "cpu"},
        },
        "package_versions": VERSIONS,
        "source_sha256": {str(path): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
                          for path in validator.PROPOSAL_SOURCES},
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    return run_dir


def test_proposal_seed_one_requires_proposal_config_and_both_expected_critics(tmp_path):
    run_dir = _write_proposal_run(tmp_path, seed=1)
    loaded = []
    result = validator.validate_run(run_dir, seed=1,
                                   checkpoint_loader=lambda path: loaded.append(Path(path)) or FakeCrossQ())
    assert result["manifest_scope"] == "proposal-five-seed"
    assert result["timesteps"] == validator.EXPECTED_TIMESTEPS
    assert loaded == [run_dir / "best/best_model.zip", run_dir / "final_model.zip"]


def test_proposal_seed_validation_rejects_wrong_eval_points_and_metadata_task(tmp_path):
    run_dir = _write_proposal_run(tmp_path, seed=2, timesteps=[25_000])
    with pytest.raises(ValueError, match="timestep sequence"):
        validator.validate_run(run_dir, seed=2, checkpoint_loader=lambda path: FakeCrossQ())
    run_dir = _write_proposal_run(tmp_path / "wrong", seed=2, task="HalfCheetah-v4")
    with pytest.raises(ValueError, match="env_id"):
        validator.validate_run(run_dir, seed=2, checkpoint_loader=lambda path: FakeCrossQ())


@pytest.mark.parametrize("mutation,expected", [
    ("version", "package version mismatch"),
    ("hash", "config hash mismatch"),
    ("source", "source hash"),
    ("settings", "resolved policy settings"),
])
def test_proposal_seed_validation_checks_package_hash_source_and_settings(tmp_path, mutation, expected):
    run_dir = _write_proposal_run(tmp_path / mutation, seed=3)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if mutation == "version":
        metadata["package_versions"]["sb3-contrib"] = "0.0"
    elif mutation == "hash":
        metadata["config_sha256"] = "0" * 64
    elif mutation == "source":
        first = next(iter(metadata["source_sha256"]))
        metadata["source_sha256"][first] = "0" * 64
    else:
        metadata["resolved_config"]["policy_kwargs"]["net_arch"]["qf"] = [64, 64]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=expected):
        validator.validate_run(run_dir, seed=3, checkpoint_loader=lambda path: FakeCrossQ())


def test_proposal_seed_validation_rejects_wrong_critic_shape(tmp_path):
    run_dir = _write_proposal_run(tmp_path, seed=4)

    class WrongNetwork:
        def modules(self):
            return [FakeLinear(128), FakeLinear(128), FakeLinear(1)]

    wrong_model = SimpleNamespace(policy=SimpleNamespace(critic=SimpleNamespace(
        q_networks=[WrongNetwork(), WrongNetwork()])))
    with pytest.raises(ValueError, match="critic widths"):
        validator.validate_run(run_dir, seed=4, checkpoint_loader=lambda path: wrong_model)


class FakeLinear:
    def __init__(self, width):
        self.out_features = width


class FakeNetwork:
    def modules(self):
        return [FakeLinear(2048), FakeLinear(2048), FakeLinear(1)]


class FakeCrossQ:
    policy = SimpleNamespace(critic=SimpleNamespace(q_networks=[FakeNetwork(), FakeNetwork()]))


def test_training_seed_statistics_iqm_and_bootstrap_use_seeds_not_episodes():
    scores = [10, 20, 30, 40, 100]
    result = validator.seed_level_statistics(scores, bootstrap_samples=500, random_seed=123)
    assert result["n_training_seeds"] == 5
    assert result["iqm"] == pytest.approx(30.0)
    assert result["bootstrap_unit"] == "training_seed"
    assert result["ci95"][0] <= result["iqm"] <= result["ci95"][1]
    assert result == validator.seed_level_statistics(scores, bootstrap_samples=500, random_seed=123)


def test_proposal_summary_preserves_five_seed_curves_and_rejects_missing_seed(tmp_path):
    seeds = [
        {"seed": seed, "episode_returns": [[float(seed + i)] * 20 for i in range(200)],
         "episode_lengths": [[100] * 20 for _ in range(200)]}
        for seed in range(5)
    ]
    result = validator.summarize_seed_runs(seeds)
    assert result["seed_level_statistics"]["n_training_seeds"] == 5
    assert len(result["per_seed"]) == 5
    assert len(result["per_seed"]["4"]["episode_returns"]) == 200
    with pytest.raises(ValueError, match="five training seeds"):
        validator.summarize_seed_runs(seeds[:4])
