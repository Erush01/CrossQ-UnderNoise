import json
import importlib.util
from pathlib import Path

import pytest
from types import SimpleNamespace

CONFIG = Path(__file__).parents[1] / "configs" / "paper_mujoco.yaml"
CHECKER = Path(__file__).parents[1] / "scripts" / "check_envs.py"
EXPECTED_TASKS = {
    "Hopper": "Hopper-v4",
    "Walker2d": "Walker2d-v4",
    "Ant": "Ant-v4",
    "Humanoid": "Humanoid-v4",
    "HalfCheetah": "HalfCheetah-v4",
    "HumanoidStandup": "HumanoidStandup-v4",
}

spec = importlib.util.spec_from_file_location("check_envs", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
require_box_action_space = checker.require_box_action_space
resolve_model_kwargs = checker.resolve_model_kwargs
close_environment = checker.close_environment
validate_step_result = checker.validate_step_result


def test_manifest_contains_the_six_paper_environment_ids():
    config = json.loads(CONFIG.read_text())
    assert config["tasks"] == EXPECTED_TASKS


def test_checker_requires_a_continuous_box_action_space():
    class Box:
        pass

    require_box_action_space(Box(), SimpleNamespace(spaces=SimpleNamespace(Box=Box)))


def test_checker_rejects_a_non_box_action_space():
    class Box:
        pass

    with pytest.raises(TypeError, match="Box"):
        require_box_action_space(object(), SimpleNamespace(spaces=SimpleNamespace(Box=Box)))


def test_manifest_model_kwargs_resolve_exact_values_without_gymnasium():
    config = json.loads(CONFIG.read_text())
    relu = object()
    settings, policy_kwargs = resolve_model_kwargs(config, SimpleNamespace(nn=SimpleNamespace(ReLU=relu)))
    constructor = {key: entry["value"] for key, entry in config["settings"]["sb3_mappings"]["constructor_kwargs"].items()}
    assert settings == constructor
    assert policy_kwargs["net_arch"] == {"pi": [256, 256], "qf": [2048, 2048]}
    assert policy_kwargs["activation_fn"] is relu
    assert policy_kwargs["optimizer_kwargs"]["betas"] == (0.5, 0.999)


def test_close_environment_closes_wrapped_model_env_only_once():
    calls = []
    raw = SimpleNamespace(close=lambda: calls.append("raw"))
    wrapped = SimpleNamespace(close=lambda: calls.append("wrapped"))
    close_environment(raw, SimpleNamespace(env=wrapped))
    assert calls == ["wrapped"]


def test_close_environment_falls_back_to_raw_env_if_model_construction_failed():
    calls = []
    raw = SimpleNamespace(close=lambda: calls.append("raw"))
    close_environment(raw, None)
    assert calls == ["raw"]


@pytest.mark.parametrize("terminated,truncated", [(0, False), (False, 1)])
def test_step_result_requires_boolean_termination_fields(terminated, truncated):
    with pytest.raises(TypeError, match="bool"):
        validate_step_result([0.0], 0.0, terminated, truncated)
