import copy
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
BASE_CONFIG_PATH = ROOT / "configs/proposal_hopper_5seeds_5m.yaml"
EXTENSION_OUTPUT_ROOT = ROOT / "artifacts/proposal_hopper_extension"
VERSIONS = {
    "stable_baselines3": "2.9.0",
    "sb3_contrib": "2.9.0",
    "gymnasium": "1.4.0",
    "mujoco": "3.16.0",
}


def _runner():
    module_name = "scripts.run_proposal_hopper_extension"
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"missing extension runner module: {module_name}"
    return importlib.import_module(module_name)


def test_extension_config_changes_only_the_seed_manifest():
    runner = _runner()
    base = json.loads(BASE_CONFIG_PATH.read_text())
    expected = copy.deepcopy(base)
    expected["seeds"] = list(range(10))

    config = runner.load_config()
    assert config == expected
    assert runner.validate_config(config) == config

    altered = copy.deepcopy(config)
    altered["settings"]["paper"]["gamma"]["value"] = 0.98
    with pytest.raises(ValueError, match="frozen ten-seed Hopper protocol"):
        runner.validate_config(altered)


def test_crossq_training_runner_accepts_the_extension_config():
    runner = _runner()
    from crossq_repro.runner import _load_config

    config, config_path = _load_config(runner.CONFIG_PATH)
    assert config["seeds"] == list(range(10))
    assert Path(config_path) == runner.CONFIG_PATH


def test_dry_run_lists_only_seeds_five_through_nine_without_writing(tmp_path, capsys):
    runner = _runner()
    output_root = tmp_path / "extension-runs"
    jobs = runner.build_jobs(output_root=output_root)

    assert [(job["env_id"], job["seed"], job["steps"]) for job in jobs] == [
        ("Hopper-v4", seed, 5_000_000) for seed in (5, 6, 7, 8, 9)
    ]
    assert all(Path(job["run_dir"]) == output_root / f"Hopper-seed{seed}"
               for seed, job in zip(range(5, 10), jobs))
    assert not output_root.exists()

    assert runner.main(["--dry-run", "--output-root", str(output_root)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 5
    assert all(f"seed={seed}" in line for seed, line in zip(range(5, 10), lines))
    assert "seed=4" not in "\n".join(lines)
    assert not output_root.exists()


def test_extension_runner_requires_confirmation_exact_session_versions_and_fresh_target(
    tmp_path,
):
    runner = _runner()
    with pytest.raises(RuntimeError, match="explicit confirmation"):
        runner.run_seed(5)

    assert runner.expected_tmux_session(5) == "cmp620-crossq-proposal-hopper-seed5"
    with pytest.raises(RuntimeError, match="cmp620-crossq-proposal-hopper-seed5"):
        runner.validate_tmux_session(5, "other-session")
    for invalid_seed in (4, 10):
        with pytest.raises(ValueError, match="seeds 5 through 9"):
            runner.expected_tmux_session(invalid_seed)

    output_root = tmp_path / "extension-runs"
    job = runner.validate_seed_preconditions(
        5, output_root=output_root, version_lookup=lambda: VERSIONS
    )
    assert job["seed"] == 5
    assert not output_root.exists()

    target = output_root / "Hopper-seed5"
    target.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing existing extension target"):
        runner.validate_seed_preconditions(
            5, output_root=output_root, version_lookup=lambda: VERSIONS
        )

    target.rmdir()
    bad_versions = {**VERSIONS, "sb3_contrib": "0.0"}
    with pytest.raises(ValueError, match="version mismatch"):
        runner.validate_seed_preconditions(
            5, output_root=output_root, version_lookup=lambda: bad_versions
        )


def test_confirmed_run_rejects_wrong_current_session_before_baseline_or_training(
    tmp_path, monkeypatch
):
    runner = _runner()
    output_root = tmp_path / "extension-runs"
    monkeypatch.setattr(runner, "current_tmux_session", lambda: "wrong-session")

    from scripts import validate_proposal_hopper as base_validator

    monkeypatch.setattr(
        base_validator,
        "validate_all_runs",
        lambda **_: pytest.fail("baseline validation must not run for a wrong tmux session"),
    )
    with pytest.raises(RuntimeError, match="requires tmux session"):
        runner.run_seed(5, output_root=output_root, confirmed=True)
    assert not output_root.exists()


def test_extension_cli_requires_one_explicit_confirmed_seed():
    runner = _runner()
    with pytest.raises(SystemExit):
        runner.main(["--seed", "5"])
    with pytest.raises(SystemExit):
        runner.main(["--seed", "4", "--confirm-proposal-run"])
    with pytest.raises(SystemExit):
        runner.main(["--seed", "5", "--confirm-proposal-run", "--dry-run"])


def test_extension_validator_checks_every_source_recorded_by_runner():
    runner = _runner()
    assert hasattr(runner, "PROVENANCE_PATHS"), "runner must expose its recorded provenance sources"
    from scripts import validate_proposal_hopper as base_validator
    from scripts import validate_proposal_hopper_extension as extension_validator

    assert set(extension_validator.EXTENSION_SOURCES) == set(
        base_validator.CORE_SOURCES + list(runner.PROVENANCE_PATHS)
    )
