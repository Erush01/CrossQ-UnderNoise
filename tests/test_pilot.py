import json
import hashlib
import struct
import zipfile
import os
import subprocess
import sys
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preflight_pilot
from scripts import summarize_pilot


def test_pilot_config_is_single_task_seed_and_warmup_budget():
    config = json.loads((ROOT / "configs/pilot_humanoidstandup_100k.json").read_text())
    paper = json.loads((ROOT / "configs/paper_mujoco.yaml").read_text())
    assert config["tasks"] == {"HumanoidStandup": "HumanoidStandup-v4"}
    assert config["seeds"] == [0]
    assert config["max_steps"] == 100_000
    assert config["evaluation"] == {"frequency": 25_000, "episodes": 20, "envs": 5}
    assert config["settings"]["paper"]["critic_architecture"]["value"] == [2048, 2048]
    assert config["settings"]["paper"]["brn_warmup_steps"]["value"] == 100000
    assert config["implementation_versions"] == paper["implementation_versions"]
    assert config["pilot"]["paper_performance_reproduction"] is False


def test_replay_buffer_estimate_accounts_for_observations_and_actions():
    assert preflight_pilot.estimate_replay_buffer_bytes((3,), (2,), 10) == 10 * (2 * 3 + 2 + 2) * 4


def test_preflight_decision_rejects_insufficient_memory_or_disk():
    report = preflight_pilot.evaluate_resources(
        required_memory_bytes=100, available_memory_bytes=99,
        required_disk_bytes=100, available_disk_bytes=100,
        device="auto", gpu_available=False,
    )
    assert not report["ok"]
    assert "memory" in report["failures"][0].lower()
    assert report["device_note"] == "auto will resolve to CPU"


def test_preflight_rejects_cuda_with_less_than_two_gib_free():
    report = preflight_pilot.evaluate_resources(
        required_memory_bytes=1, available_memory_bytes=1,
        required_disk_bytes=1, available_disk_bytes=1,
        device="auto", gpu_available=True, gpu_free_bytes=2 * 1024**3 - 1,
    )
    assert not report["ok"]
    assert "cuda" in " ".join(report["failures"]).lower()


def _fake_pilot_runtime(tmp_path):
    fake_site = tmp_path / "site"
    (fake_site / "gymnasium").mkdir(parents=True)
    (fake_site / "gymnasium/__init__.py").write_text('''
class Space:
    def __init__(self, shape): self.shape = shape
class Env:
    observation_space = Space((376,))
    action_space = Space((17,))
    def close(self): pass
def make(env_id):
    assert env_id == "HumanoidStandup-v4"
    return Env()
''')
    (fake_site / "torch").mkdir()
    (fake_site / "torch/__init__.py").write_text('''
class Cuda:
    def is_available(self): return False
cuda = Cuda()
''')
    env = os.environ.copy()
    env["CROSSQ_PYTHON"] = sys.executable
    env["PYTHONPATH"] = str(fake_site) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _sandbox_pilot_root(tmp_path):
    root = tmp_path / "CrossQ"
    (root / "scripts").mkdir(parents=True)
    (root / "configs").mkdir()
    shutil.copy2(ROOT / "scripts/run_pilot.sh", root / "scripts/run_pilot.sh")
    shutil.copy2(ROOT / "scripts/preflight_pilot.py", root / "scripts/preflight_pilot.py")
    shutil.copy2(ROOT / "configs/pilot_humanoidstandup_100k.json",
                 root / "configs/pilot_humanoidstandup_100k.json")
    return root


def test_actual_dry_run_uses_selected_python_and_does_not_create_target(tmp_path):
    root = _sandbox_pilot_root(tmp_path)
    target = root / "artifacts/pilot/HumanoidStandup-seed0-attempt2"
    assert not target.exists()
    env = _fake_pilot_runtime(tmp_path)
    result = subprocess.run([str(root / "scripts/run_pilot.sh"), "--attempt2", "--dry-run"], cwd=root.parent,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DRY RUN: HumanoidStandup-v4 seed=0 steps=100000 device=auto")
    assert "attempt2" in result.stdout
    assert not target.exists()


def test_default_refuses_original_target_but_attempt2_dry_run_succeeds(tmp_path):
    root = _sandbox_pilot_root(tmp_path)
    original = root / "artifacts/pilot/HumanoidStandup-seed0"
    original.mkdir(parents=True)
    (original / "preserve.txt").write_text("existing original")
    target = root / "artifacts/pilot/HumanoidStandup-seed0-attempt2"
    assert not target.exists()
    before = sorted(str(path.relative_to(original)) for path in original.rglob("*"))
    env = _fake_pilot_runtime(tmp_path)
    result = subprocess.run([str(root / "scripts/run_pilot.sh"), "--dry-run"], cwd=root.parent,
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0
    assert "existing pilot target" in result.stderr.lower()
    result = subprocess.run([str(root / "scripts/run_pilot.sh"), "--dry-run", "--attempt2"], cwd=root.parent,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.startswith("DRY RUN: HumanoidStandup-v4 seed=0 steps=100000 device=auto")
    assert "attempt2" in result.stdout
    assert not target.exists()
    assert sorted(str(path.relative_to(original)) for path in original.rglob("*")) == before
    result = subprocess.run([str(root / "scripts/run_pilot.sh"), "--attempt2", "--dry-run"], cwd=root.parent,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists()


def _write_valid_pilot_artifacts(tmp_path):
    target = tmp_path / "HumanoidStandup-seed0"
    target.mkdir()
    config_path = target / "config.json"
    config = json.loads((ROOT / "configs/pilot_humanoidstandup_100k.json").read_text())
    config_path.write_text(json.dumps(config))
    sources = [ROOT / "crossq_repro/runner.py", ROOT / "scripts/run_pilot.sh",
               ROOT / "scripts/preflight_pilot.py"]
    metadata = {
        "status": "completed", "env_id": "HumanoidStandup-v4", "seed": 0,
        "total_timesteps": 100000, "actual_timesteps": 100000,
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T01:00:00+00:00",
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "package_versions": {"stable-baselines3": "2.9.0", "sb3-contrib": "2.9.0",
                             "gymnasium": "1.4.0", "mujoco": "3.16.0"},
        "resolved_config": {"config": config, "device": {"resolved": "cpu"}},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (target / "metadata.json").write_text(json.dumps(metadata))
    (target / "final_model.zip").write_bytes(b"model")
    (target / "best").mkdir()
    (target / "best/best_model.zip").write_bytes(b"best")
    (target / "evaluations").mkdir()
    def npy(shape, values, dtype="<i8"):
        descr = dtype
        header = repr({"descr": descr, "fortran_order": False, "shape": shape})
        header = (header + " " * (63 - len(header)) + "\n").encode()
        return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + ("q" if dtype == "<i8" else "d") * len(values), *values)
    with zipfile.ZipFile(target / "evaluations/evaluations.npz", "w") as archive:
        archive.writestr("timesteps.npy", npy((4,), [25000, 50000, 75000, 100000]))
        archive.writestr("results.npy", npy((4, 20), [1.0] * 80, "<f8"))
        archive.writestr("ep_lengths.npy", npy((4, 20), [100] * 80))
    return target


class _FakeNetwork:
    def __init__(self, features):
        self.layers = [type("Layer", (), {"out_features": n})() for n in features]

    def modules(self):
        return self.layers


class _FakeModel:
    policy = type("Policy", (), {"critic": type("Critic", (), {
        "q_networks": [_FakeNetwork([2048, 2048, 1]), _FakeNetwork([2048, 2048, 1])]
    })()})()


def test_summary_accepts_complete_matching_artifacts_and_checkpoint_loader(tmp_path, monkeypatch):
    target = _write_valid_pilot_artifacts(tmp_path)
    loaded_paths = []
    class CrossQ:
        @staticmethod
        def load(path, device):
            loaded_paths.append((Path(path), device))
            return _FakeModel()
    monkeypatch.setitem(sys.modules, "sb3_contrib", SimpleNamespace(CrossQ=CrossQ))
    result = summarize_pilot.validate_artifacts(target)
    assert result["env_id"] == "HumanoidStandup-v4"
    assert loaded_paths == [(target / "best/best_model.zip", "cpu"),
                            (target / "final_model.zip", "cpu")]


@pytest.mark.parametrize("change", ["actual_steps", "eval_timesteps", "shape", "nan_return", "zero_length", "eval_config", "critic", "versions", "config_metadata", "source_hash"])
def test_summary_rejects_invalid_or_mismatched_artifacts(tmp_path, change):
    target = _write_valid_pilot_artifacts(tmp_path)
    metadata_path = target / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if change == "actual_steps":
        metadata["actual_timesteps"] = 99999
        metadata_path.write_text(json.dumps(metadata))
    elif change in ("eval_timesteps", "shape", "nan_return", "zero_length"):
        def npy(shape, values, dtype="<i8"):
            header = repr({"descr": dtype, "fortran_order": False, "shape": shape})
            header = (header + " " * (63 - len(header)) + "\n").encode()
            code = "q" if dtype == "<i8" else "d"
            return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + code * len(values), *values)
        timesteps = [25000, 50000, 75000, 90000] if change == "eval_timesteps" else [25000, 50000, 75000, 100000]
        rshape = (3, 20) if change == "shape" else (4, 20)
        results = [float("nan")] * 80 if change == "nan_return" else [1.0] * (rshape[0] * 20)
        lengths = [0] * 80 if change == "zero_length" else [100] * (rshape[0] * 20)
        with zipfile.ZipFile(target / "evaluations/evaluations.npz", "w") as archive:
            archive.writestr("timesteps.npy", npy((4,), timesteps))
            archive.writestr("results.npy", npy(rshape, results, "<f8"))
            archive.writestr("ep_lengths.npy", npy(rshape, lengths))
    elif change == "eval_config":
        config = json.loads((target / "config.json").read_text())
        config["evaluation"]["frequency"] = 50000
        (target / "config.json").write_text(json.dumps(config))
    elif change == "critic":
        config = json.loads((target / "config.json").read_text())
        config["settings"]["paper"]["critic_architecture"]["value"] = [1024, 1024]
        (target / "config.json").write_text(json.dumps(config))
    elif change == "versions":
        config = json.loads((target / "config.json").read_text())
        config["implementation_versions"]["sb3_contrib"] = "bad"
        (target / "config.json").write_text(json.dumps(config))
    elif change == "config_metadata":
        metadata["resolved_config"]["config"]["max_steps"] = 5
        metadata_path.write_text(json.dumps(metadata))
    elif change == "source_hash":
        metadata["source_sha256"].pop(str(ROOT / "scripts/run_pilot.sh"))
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        summarize_pilot.validate_artifacts(target, checkpoint_loader=lambda path: _FakeModel())


def test_summary_rejects_wrong_loaded_qf_architecture(tmp_path):
    target = _write_valid_pilot_artifacts(tmp_path)
    wrong = type("Model", (), {"policy": type("Policy", (), {"critic": type("Critic", (), {
        "q_networks": [_FakeNetwork([1024, 1024, 1]), _FakeNetwork([1024, 1024, 1])]
    })()})()})()
    with pytest.raises(ValueError, match="critic architecture"):
        summarize_pilot.validate_artifacts(target, checkpoint_loader=lambda path: wrong)
