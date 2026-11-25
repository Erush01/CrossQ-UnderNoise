import hashlib
import json
import os
import subprocess
import sys
import zipfile
import struct
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import scaled_reproduction_jobs as jobs
from scripts import summarize_scaled_reproduction as summary

CONFIG_PATH = ROOT / "configs/scaled_3tasks_3seeds_5m.yaml"
TASKS = {
    "Hopper": "Hopper-v4",
    "HalfCheetah": "HalfCheetah-v4",
    "HumanoidStandup": "HumanoidStandup-v4",
}


def test_scaled_manifest_is_narrowed_frozen_protocol_copy():
    config = json.loads(CONFIG_PATH.read_text())
    paper = json.loads((ROOT / "configs/paper_mujoco.yaml").read_text())
    assert config["tasks"] == TASKS
    assert config["seeds"] == [0, 1, 2]
    assert config["max_steps"] == 5_000_000
    assert config["settings"] == paper["settings"]
    assert config["evaluation"] == {"frequency": 25_000, "episodes": 20, "envs": 5}
    assert config["implementation_versions"] == paper["implementation_versions"]


def test_job_matrix_and_batches_are_fixed_and_deterministic(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    matrix = jobs.build_jobs(config, tmp_path)
    assert len(matrix) == 9
    assert [(job["task"], job["seed"]) for job in matrix] == [
        (task, seed) for task in TASKS for seed in (0, 1, 2)
    ]
    assert len({job["run_dir"] for job in matrix}) == 9
    assert all(Path(job["run_dir"]).parent == tmp_path for job in matrix)
    assert len(jobs.select_batch(matrix, "seed0")) == 3
    assert len(jobs.select_batch(matrix, "remaining")) == 6
    with pytest.raises(ValueError):
        jobs.select_batch(matrix, "all")


def test_confirmation_is_required_in_api_and_python_cli_before_preflight(monkeypatch):
    monkeypatch.setattr(jobs, "resource_preflight", lambda *args: pytest.fail("preflight must not run"))
    with pytest.raises(RuntimeError, match="confirmation"):
        jobs.run_batch("seed0")
    with pytest.raises(SystemExit):
        jobs.main(["--run-batch", "seed0"])
    with pytest.raises(SystemExit):
        jobs.main(["--dry-run", "--confirm-scaled-run"])


def test_single_job_is_fixed_to_hopper_seed0_retry_target():
    job = jobs.build_single_job("Hopper", 0)
    assert job == {
        "task": "Hopper", "env_id": "Hopper-v4", "seed": 0, "steps": 5_000_000,
        "run_dir": str(ROOT / "artifacts/scaled_reproduction_retry/Hopper-seed0"),
    }
    with pytest.raises(ValueError):
        jobs.build_single_job("Hopper", 1)
    with pytest.raises(ValueError):
        jobs.build_single_job("HumanoidStandup", 0)


def test_halfcheetah_single_retry_target_session_preconditions_and_dry_run(tmp_path, monkeypatch):
    target = ROOT / "artifacts/scaled_reproduction_retry/HalfCheetah-seed0"
    def snapshot(path):
        if not path.exists():
            return None
        return sorted((str(item.relative_to(path)), hashlib.sha256(item.read_bytes()).hexdigest())
                      for item in path.rglob("*") if item.is_file())
    before = snapshot(target)
    job = jobs.build_single_job("HalfCheetah", 0)
    assert job == {
        "task": "HalfCheetah", "env_id": "HalfCheetah-v4", "seed": 0,
        "steps": 5_000_000, "run_dir": str(target),
    }
    versions = {"stable_baselines3": "2.9.0", "sb3_contrib": "2.9.0",
                "gymnasium": "1.4.0", "mujoco": "3.16.0"}
    monkeypatch.setattr(jobs, "SINGLE_RETRY_ROOT", tmp_path / "unused-retry-root")
    checked = jobs.validate_single_preconditions("HalfCheetah", 0, version_lookup=lambda: versions)
    assert checked["env_id"] == job["env_id"] and checked["seed"] == job["seed"]
    jobs.validate_single_tmux_session("HalfCheetah", 0, "cmp620-crossq-scaled-single-HalfCheetah-seed0")
    env = os.environ.copy()
    env["CROSSQ_PYTHON"] = sys.executable
    dry = subprocess.run(
        ["bash", str(ROOT / "scripts/run_scaled_reproduction.sh"), "--single", "HalfCheetah",
         "--seed", "0", "--dry-run"],
        cwd=ROOT.parent, env=env, text=True, capture_output=True,
    )
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert dry.stdout.strip() == f"task=HalfCheetah-v4 seed=0 steps=5000000 run_dir={target}"
    assert snapshot(target) == before


def test_single_confirmation_and_named_session_are_required_before_preflight(monkeypatch):
    monkeypatch.setattr(jobs, "resource_preflight", lambda *args: pytest.fail("preflight must not run"))
    with pytest.raises(RuntimeError, match="confirmation"):
        jobs.run_single("Hopper", 0)
    with pytest.raises(RuntimeError, match="cmp620-crossq-scaled-single-Hopper-seed0"):
        jobs.validate_single_tmux_session("Hopper", 0, "some-other-session")
    with pytest.raises(RuntimeError, match="existing tmux"):
        jobs.validate_single_tmux_session("Hopper", 0, None)


def test_single_preconditions_refuse_existing_target_and_package_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "SINGLE_RETRY_ROOT", tmp_path)
    target = tmp_path / "Hopper-seed0"
    target.mkdir()
    with pytest.raises(FileExistsError, match="single retry target"):
        jobs.validate_single_preconditions("Hopper", 0)
    target.rmdir()
    with pytest.raises(ValueError, match="version mismatch"):
        jobs.validate_single_preconditions(
            "Hopper", 0, version_lookup=lambda: {"stable_baselines3": "bad", "sb3_contrib": "bad",
                                                "gymnasium": "bad", "mujoco": "bad"})


def test_single_runner_invocation_uses_selected_task_from_crossq_cwd(tmp_path, monkeypatch):
    job = {
        "task": "HalfCheetah", "env_id": "HalfCheetah-v4", "seed": 0,
        "steps": 5_000_000, "run_dir": str(tmp_path / "retry/HalfCheetah-seed0"),
    }
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "require_single_tmux_session", lambda task, seed: None)
    monkeypatch.setattr(jobs, "validate_single_preconditions", lambda *args: job)
    monkeypatch.setattr(jobs, "resource_preflight", lambda *args: None)
    validated = []

    def fake_validate(path, task, seed):
        validated.append((path, task, seed))

    monkeypatch.setattr(summary, "validate_run", fake_validate)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    jobs.run_single("HalfCheetah", 0, confirmed=True)
    assert len(calls) == 2
    runner_command, runner_options = calls[1]
    assert runner_options["cwd"] == tmp_path
    assert "sys.argv[2]" in runner_command[6]
    assert "int(sys.argv[3])" in runner_command[6]
    assert runner_command[7] == str(CONFIG_PATH)
    assert runner_command[8] == "HalfCheetah-v4"
    assert runner_command[9] == "0"
    assert runner_command[10] == "5000000"
    assert runner_command[11] == job["run_dir"]
    assert runner_command[12].endswith("scaled_reproduction_jobs.py")
    assert runner_command[13].endswith("run_scaled_reproduction.sh")
    assert "scaled-single-HalfCheetah-seed0-" in runner_command[3]
    assert "scaled-single-HalfCheetah-seed0-" in calls[1][1]["stdout"].name
    assert validated == [(job["run_dir"], "HalfCheetah", 0)]


def test_single_cli_dry_run_rejects_confirmation_and_conflicting_flags():
    with pytest.raises(SystemExit):
        jobs.main(["--single", "Hopper", "--seed", "0"])
    with pytest.raises(SystemExit):
        jobs.main(["--single", "Hopper", "--seed", "0", "--dry-run", "--confirm-scaled-run"])
    with pytest.raises(SystemExit):
        jobs.main(["--single", "Hopper", "--seed", "0", "--dry-run", "--run-batch", "seed0"])


def test_training_subprocess_runs_from_crossq_root(tmp_path, monkeypatch):
    config = jobs.load_config()
    job = {
        "task": "Hopper", "env_id": config["tasks"]["Hopper"], "seed": 0,
        "steps": 5_000_000, "run_dir": str(tmp_path / "out/Hopper-seed0"),
    }
    monkeypatch.setattr(jobs, "ROOT", tmp_path)
    monkeypatch.setattr(jobs, "require_tmux_session", lambda batch: None)
    monkeypatch.setattr(jobs, "validate_batch_preconditions", lambda *args, **kwargs: [job])
    monkeypatch.setattr(jobs, "resource_preflight", lambda *args: None)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if str(tmp_path / "scripts/check_envs.py") in command:
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(jobs.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="scaled run failed"):
        jobs.run_batch("seed0", config_path=CONFIG_PATH, output_root=tmp_path / "out", confirmed=True)
    assert calls[1][1].get("cwd") == tmp_path


def test_selected_batch_refuses_any_existing_target_or_package_mismatch(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    matrix = jobs.build_jobs(config, tmp_path)
    Path(matrix[0]["run_dir"]).mkdir(parents=True)
    with pytest.raises(FileExistsError):
        jobs.validate_batch_preconditions(config, "seed0", tmp_path,
                                          version_lookup=lambda: {"stable_baselines3": "2.9.0",
                                          "sb3_contrib": "2.9.0", "gymnasium": "1.4.0", "mujoco": "3.16.0"})
    Path(matrix[0]["run_dir"]).rmdir()
    wrong_versions = {"stable_baselines3": "2.9.0", "sb3_contrib": "bad",
                      "gymnasium": "1.4.0", "mujoco": "3.16.0"}
    with pytest.raises(ValueError, match="version mismatch"):
        jobs.validate_batch_preconditions(config, "seed0", tmp_path,
                                          version_lookup=lambda: wrong_versions)


def test_remaining_requires_all_strictly_valid_seed0_runs_before_batch_check(tmp_path, monkeypatch):
    config = jobs.load_config()
    checked = []
    monkeypatch.setattr(jobs, "require_tmux_session", lambda batch: None)
    monkeypatch.setattr(jobs, "validate_seed0_prerequisites",
                        lambda cfg, root: (_ for _ in ()).throw(FileNotFoundError("seed0 missing/incomplete")))
    monkeypatch.setattr(jobs, "validate_batch_preconditions",
                        lambda *args, **kwargs: checked.append("batch-check"))
    with pytest.raises(FileNotFoundError, match="seed0"):
        jobs.run_batch("remaining", confirmed=True, config_path=CONFIG_PATH, output_root=tmp_path)
    assert checked == []


def test_remaining_seed0_prerequisite_checks_all_three_task_seed_artifacts(tmp_path, monkeypatch):
    config = jobs.load_config()
    for task in TASKS:
        (tmp_path / f"{task}-seed0").mkdir()
    validated = []
    monkeypatch.setattr(summary, "validate_run", lambda path, task, seed: validated.append((task, seed)))
    result = jobs.validate_seed0_prerequisites(config, tmp_path)
    assert result == ["Hopper", "HalfCheetah", "HumanoidStandup"]
    assert validated == [(task, 0) for task in TASKS]


def test_remaining_seed0_prerequisite_rejects_incomplete_artifact(tmp_path):
    config = jobs.load_config()
    (tmp_path / "Hopper-seed0").mkdir()
    with pytest.raises(ValueError, match="incomplete run artifacts"):
        jobs.validate_seed0_prerequisites(config, tmp_path)


def test_tmux_session_must_match_named_batch_session():
    assert jobs.expected_tmux_session("seed0") == "cmp620-crossq-scaled-seed0"
    assert jobs.expected_tmux_session("remaining") == "cmp620-crossq-scaled-remaining"
    with pytest.raises(RuntimeError, match="cmp620-crossq-scaled-seed0"):
        jobs.validate_tmux_session("seed0", "some-other-session")
    with pytest.raises(RuntimeError, match="existing tmux"):
        jobs.validate_tmux_session("remaining", None)


def test_resource_preflight_requires_twenty_gib_disk_floor():
    minimum = 20 * 1024**3
    with pytest.raises(RuntimeError, match="20 GiB"):
        jobs.validate_disk_capacity(minimum - 1, checkpoint_bytes=1)
    assert jobs.validate_disk_capacity(minimum, checkpoint_bytes=1) == minimum


def test_launcher_default_refuses_and_dry_run_preserves_existing_targets(tmp_path):
    script = ROOT / "scripts/run_scaled_reproduction.sh"
    matrix = jobs.build_jobs()
    def snapshot(path):
        path = Path(path)
        if not path.exists():
            return None
        return sorted((str(item.relative_to(path)), hashlib.sha256(item.read_bytes()).hexdigest())
                      for item in path.rglob("*") if item.is_file())
    before = {job["run_dir"]: snapshot(job["run_dir"]) for job in matrix}
    env = os.environ.copy()
    env["CROSSQ_PYTHON"] = sys.executable
    default = subprocess.run(["bash", str(script)], cwd=ROOT.parent, env=env,
                             text=True, capture_output=True)
    assert default.returncode != 0
    gated = subprocess.run(["bash", str(script), "--batch", "seed0"], cwd=ROOT.parent,
                           env=env, text=True, capture_output=True)
    assert gated.returncode != 0
    no_tmux = env.copy()
    no_tmux.pop("TMUX", None)
    confirmed_outside_tmux = subprocess.run(
        ["bash", str(script), "--batch", "seed0", "--confirm-scaled-run"],
        cwd=ROOT.parent, env=no_tmux, text=True, capture_output=True,
    )
    assert confirmed_outside_tmux.returncode != 0
    assert "existing tmux" in confirmed_outside_tmux.stderr
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text("#!/bin/sh\necho other-session\n")
    fake_tmux.chmod(0o755)
    wrong_session_env = env.copy()
    wrong_session_env["TMUX"] = "fake-socket,1,0"
    wrong_session_env["PATH"] = str(fake_bin) + os.pathsep + wrong_session_env.get("PATH", "")
    wrong_session = subprocess.run(
        ["bash", str(script), "--batch", "seed0", "--confirm-scaled-run"],
        cwd=ROOT.parent, env=wrong_session_env, text=True, capture_output=True,
    )
    assert wrong_session.returncode != 0
    assert "cmp620-crossq-scaled-seed0" in wrong_session.stderr
    confirmed_dry = subprocess.run(["bash", str(script), "--dry-run", "--confirm-scaled-run"],
                                   cwd=ROOT.parent, env=env, text=True, capture_output=True)
    assert confirmed_dry.returncode != 0
    assert "cannot be combined" in confirmed_dry.stderr
    dry = subprocess.run(["bash", str(script), "--dry-run"], cwd=ROOT.parent, env=env,
                         text=True, capture_output=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert dry.stdout.count("steps=5000000") == 9
    assert sum(dry.stdout.count(f"seed={seed}") for seed in range(3)) == 9
    assert all(f"{name}-seed{seed}" in dry.stdout for name in TASKS for seed in range(3))
    assert {job["run_dir"]: snapshot(job["run_dir"]) for job in matrix} == before


def test_single_launcher_dry_run_lists_only_selected_seed0_and_does_not_create_target():
    script = ROOT / "scripts/run_scaled_reproduction.sh"
    target = ROOT / "artifacts/scaled_reproduction_retry/Hopper-seed0"
    def snapshot(path):
        if not path.exists():
            return None
        return sorted((str(item.relative_to(path)), hashlib.sha256(item.read_bytes()).hexdigest())
                      for item in path.rglob("*") if item.is_file())
    before = snapshot(target)
    env = os.environ.copy()
    env["CROSSQ_PYTHON"] = sys.executable
    dry = subprocess.run(["bash", str(script), "--single", "Hopper", "--seed", "0", "--dry-run"],
                         cwd=ROOT.parent, env=env, text=True, capture_output=True)
    assert dry.returncode == 0, dry.stdout + dry.stderr
    assert dry.stdout.strip() == (
        f"task=Hopper-v4 seed=0 steps=5000000 run_dir={target}"
    )
    assert snapshot(target) == before
    invalid = subprocess.run(["bash", str(script), "--single", "HumanoidStandup", "--seed", "0", "--dry-run"],
                             cwd=ROOT.parent, env=env, text=True, capture_output=True)
    assert invalid.returncode != 0
    conflict = subprocess.run(["bash", str(script), "--single", "Hopper", "--seed", "0",
                               "--batch", "seed0", "--dry-run"],
                              cwd=ROOT.parent, env=env, text=True, capture_output=True)
    assert conflict.returncode != 0
    missing_confirmation = subprocess.run(
        ["bash", str(script), "--single", "Hopper", "--seed", "0"],
        cwd=ROOT.parent, env=env, text=True, capture_output=True,
    )
    assert missing_confirmation.returncode != 0
    assert "without --confirm-scaled-run" in missing_confirmation.stderr


def _npy(shape, values, dtype="<i8"):
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape})
    header = (header + " " * (63 - len(header)) + "\n").encode()
    code = "q" if dtype == "<i8" else "d"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + struct.pack("<" + code * len(values), *values)


class FakeLinear:
    def __init__(self, width):
        self.out_features = width


class FakeQNetwork:
    def modules(self):
        return [FakeLinear(2048), FakeLinear(2048), FakeLinear(1)]


class FakeCrossQ:
    policy = SimpleNamespace(critic=SimpleNamespace(q_networks=[FakeQNetwork(), FakeQNetwork()]))


class WrongQNetwork:
    def modules(self):
        return [FakeLinear(1024), FakeLinear(1024), FakeLinear(1)]


def _write_complete_run(tmp_path, task="Hopper", seed=0, timesteps=None,
                        return_value=1.0, length_value=100):
    config = json.loads(CONFIG_PATH.read_text())
    run_dir = tmp_path / f"{task}-seed{seed}"
    run_dir.mkdir()
    config_file = run_dir / "config.json"
    config_file.write_text(json.dumps(config))
    for name in ("final_model.zip", "best/best_model.zip"):
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir()
    points = timesteps or list(range(25_000, 5_000_001, 25_000))
    with zipfile.ZipFile(eval_dir / "evaluations.npz", "w") as archive:
        archive.writestr("timesteps.npy", _npy((len(points),), points))
        archive.writestr("results.npy", _npy((len(points), 20), [return_value] * len(points) * 20, "<f8"))
        archive.writestr("ep_lengths.npy", _npy((len(points), 20), [length_value] * len(points) * 20))
    source_paths = [ROOT / "crossq_repro/runner.py", ROOT / "crossq_repro/artifacts.py",
                    ROOT / "crossq_repro/evaluate.py", ROOT / "scripts/check_envs.py",
                    ROOT / "scripts/scaled_reproduction_jobs.py", ROOT / "scripts/run_scaled_reproduction.sh"]
    meta = {
        "status": "completed", "env_id": config["tasks"][task], "seed": seed,
        "total_timesteps": 5_000_000, "actual_timesteps": 5_000_000,
        "started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-02T00:00:00+00:00",
        "resolved_config": {"config": config, "device": {"resolved": "cpu"}},
        "package_versions": {"stable-baselines3": "2.9.0", "sb3-contrib": "2.9.0",
                              "gymnasium": "1.4.0", "mujoco": "3.16.0"},
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
    }
    (run_dir / "metadata.json").write_text(json.dumps(meta))
    return run_dir


def test_scaled_summary_validates_complete_seed_artifacts_and_checkpoints(tmp_path):
    run_dir = _write_complete_run(tmp_path)
    loaded = []
    result = summary.validate_run(run_dir, task="Hopper", seed=0,
                                  checkpoint_loader=lambda path: loaded.append(Path(path)) or FakeCrossQ())
    assert result["seed"] == 0 and result["task"] == "Hopper"
    assert result["timesteps"] == list(range(25_000, 5_000_001, 25_000))
    assert loaded == [run_dir / "best/best_model.zip", run_dir / "final_model.zip"]


def test_scaled_summary_rejects_unloadable_or_wrong_critic_checkpoint(tmp_path):
    run_dir = _write_complete_run(tmp_path)
    with pytest.raises(ValueError, match="CrossQ.load failed"):
        summary.validate_run(run_dir, task="Hopper", seed=0,
                             checkpoint_loader=lambda path: (_ for _ in ()).throw(RuntimeError("bad zip")))
    wrong_model = SimpleNamespace(policy=SimpleNamespace(critic=SimpleNamespace(
        q_networks=[WrongQNetwork(), WrongQNetwork()])))
    with pytest.raises(ValueError, match="wrong critic architecture"):
        summary.validate_run(run_dir, task="Hopper", seed=0,
                             checkpoint_loader=lambda path: wrong_model)


def test_all_batch_summary_requires_nine_runs_and_writes_planned_paths(tmp_path, monkeypatch):
    config = json.loads(CONFIG_PATH.read_text())
    for task in config["tasks"]:
        _write_complete_run(tmp_path, task=task, seed=0)
    with pytest.raises((FileNotFoundError, ValueError)):
        summary.summarize_batch("all", output_root=tmp_path,
                                checkpoint_loader=lambda path: FakeCrossQ())
    for task in config["tasks"]:
        for seed in (1, 2):
            _write_complete_run(tmp_path, task=task, seed=seed)
    result = summary.summarize_batch("all", output_root=tmp_path,
                                     checkpoint_loader=lambda path: FakeCrossQ())
    assert result["validated_runs"] == result["expected_runs"] == 9
    assert all(set(task_result["seeds"]) == {"0", "1", "2"}
               for task_result in result["tasks"].values())
    assert all(task_result["final_score_across_seeds"]["n_seeds"] == 3
               for task_result in result["tasks"].values())

    class FakeLoader:
        @staticmethod
        def load(path, device):
            return FakeCrossQ()
    monkeypatch.setitem(sys.modules, "sb3_contrib", SimpleNamespace(CrossQ=FakeLoader))
    results_dir = tmp_path / "results"
    summary.main(["--batch", "all", "--output-root", str(tmp_path), "--results-dir", str(results_dir)])
    assert (results_dir / "summary.json").is_file()
    assert (results_dir / "report.md").is_file()


@pytest.mark.parametrize("mutation", ["actual_steps", "eval_points", "metadata_task", "config_hash", "source_hash", "versions", "config_metadata", "nan", "zero_length"])
def test_scaled_summary_rejects_incomplete_or_mismatched_artifacts(tmp_path, mutation):
    run_dir = _write_complete_run(tmp_path, timesteps=[25_000] if mutation == "eval_points" else None,
                                  return_value=float("nan") if mutation == "nan" else 1.0,
                                  length_value=0 if mutation == "zero_length" else 100)
    metadata_path = run_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if mutation == "actual_steps":
        metadata["actual_timesteps"] = 4_975_000
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "metadata_task":
        metadata["env_id"] = "HalfCheetah-v4"
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "config_hash":
        metadata["config_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "source_hash":
        metadata["source_sha256"].pop(str(ROOT / "scripts/run_scaled_reproduction.sh"))
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "versions":
        metadata["package_versions"]["sb3-contrib"] = "0.0"
        metadata_path.write_text(json.dumps(metadata))
    elif mutation == "config_metadata":
        metadata["resolved_config"]["config"]["max_steps"] = 100
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        summary.validate_run(run_dir, task="Hopper", seed=0,
                             checkpoint_loader=lambda path: FakeCrossQ())


def test_batch_summary_retains_individual_seed_curves_and_scores(tmp_path):
    config = json.loads(CONFIG_PATH.read_text())
    for task in config["tasks"]:
        _write_complete_run(tmp_path, task=task, seed=0)
    result = summary.summarize_batch("seed0", output_root=tmp_path,
                                     checkpoint_loader=lambda path: FakeCrossQ())
    assert result["validated_runs"] == result["expected_runs"] == 3
    assert result["status"] == "selected_batch_complete"
    assert any("SB3-Contrib" in item and "PyTorch" in item for item in result["deviations"])
    assert any("25,000" in item and "not specified" in item for item in result["deviations"])
    assert any("three tasks" in item and "ten seeds" in item for item in result["deviations"])
    assert any("baselines" in item for item in result["deviations"])
    assert all(set(task_result["seeds"]) == {"0"} for task_result in result["tasks"].values())
    assert all(len(task_result["seeds"]["0"]["episode_returns"]) == 200
               for task_result in result["tasks"].values())
