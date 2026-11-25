import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_hopper_action_noise as noise


def test_evaluation_plan_has_exact_three_hundred_seed_noise_reset_cases():
    plan = noise.build_evaluation_plan()
    assert len(plan) == 300
    assert {(job["training_seed"], job["sigma"]) for job in plan} == {
        (seed, sigma) for seed in range(5) for sigma in (0.0, 0.05, 0.10)
    }
    assert {job["reset_seed"] for job in plan} == set(range(90000, 90020))


def test_common_normal_stream_and_normalized_clip_fraction_are_paired():
    first = noise.standard_normal_stream(2, 90003).standard_normal(4)
    second = noise.standard_normal_stream(2, 90003).standard_normal(4)
    np.testing.assert_array_equal(first, second)
    action05, clipped05 = noise.perturb_normalized_action([0.95, -0.95], 0.05, [1.0, -1.0])
    action10, clipped10 = noise.perturb_normalized_action([0.95, -0.95], 0.10, [1.0, -1.0])
    np.testing.assert_allclose(action05, [1.0, -1.0])
    np.testing.assert_allclose(action10, [1.0, -1.0])
    assert clipped05 == 0.0
    assert clipped10 == 1.0


def test_action_noise_summary_requires_300_records_and_reports_per_seed_paired_deltas():
    records = []
    for train_seed in range(5):
        for reset_seed in range(90000, 90020):
            for sigma in (0.0, 0.05, 0.10):
                records.append({"training_seed": train_seed, "reset_seed": reset_seed, "sigma": sigma,
                                "return": 100.0 - 10.0 * sigma, "length": 100,
                                "action_clipping_fraction": sigma})
    result = noise.summarize_action_noise_records(records)
    assert len(result["episodes"]) == 300
    assert set(result["per_seed"]) == {"0", "1", "2", "3", "4"}
    assert result["per_seed"]["2"]["paired_delta_mean"]["0.05"] == pytest.approx(-0.5)
    assert result["per_seed"]["2"]["paired_delta_mean"]["0.1"] == pytest.approx(-1.0)
    assert result["summary"]["0"]["seed_level_statistics"]["n_training_seeds"] == 5
    with pytest.raises(ValueError, match="300"):
        noise.summarize_action_noise_records(records[:-1])


def test_action_noise_dry_run_lists_all_final_checkpoints_and_creates_nothing(tmp_path, capsys):
    output = tmp_path / "noise.json"
    assert noise.main(["--dry-run", "--output", str(output)]) == 0
    captured = capsys.readouterr().out
    assert "300 episodes" in captured
    assert all(f"seed{seed}" in captured for seed in range(5))
    assert "final_model.zip" in captured
    assert not output.exists()


def test_action_noise_summary_preserves_legacy_seed0_source_warning():
    validations = {
        0: {"source_validation_warnings": ["legacy launcher snapshot unavailable"]},
        1: {"source_validation_warnings": []},
    }
    assert noise.source_warnings_by_seed(validations) == {
        "0": ["legacy launcher snapshot unavailable"]
    }


def test_action_noise_live_mode_requires_explicit_confirmation():
    with pytest.raises(SystemExit):
        noise.main([])
