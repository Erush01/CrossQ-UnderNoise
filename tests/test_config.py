import json
from pathlib import Path

import pytest


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "paper_mujoco.yaml"
TASKS = {
    "Hopper": "Hopper-v4",
    "Walker2d": "Walker2d-v4",
    "Ant": "Ant-v4",
    "Humanoid": "Humanoid-v4",
    "HalfCheetah": "HalfCheetah-v4",
    "HumanoidStandup": "HumanoidStandup-v4",
}
REQUIRED_KEYS = {
    "schema_version",
    "implementation_versions",
    "tasks",
    "seeds",
    "max_steps",
    "max_steps_provenance",
    "evaluation",
    "evaluation_provenance",
    "task_provenance",
    "settings",
}
PAPER_SETTING_KEYS = {
    "gamma",
    "actor_learning_rate",
    "critic_learning_rate",
    "replay_buffer_size",
    "batch_size",
    "activation",
    "normalization",
    "brn_running_average_momentum",
    "brn_warmup_steps",
    "critic_architecture",
    "adam_beta1",
    "train_freq",
    "gradient_steps",
    "policy_delay",
    "n_critics",
}
IMPLEMENTATION_CHOICE_KEYS = {
    "actor_architecture",
    "entropy_coefficient",
    "learning_starts",
}
SB3_MAPPING_KEYS = {"constructor_kwargs", "policy_kwargs", "target_network_behavior"}


def load_manifest(path=CONFIG_PATH):
    """Load the JSON-compatible YAML subset and enforce its frozen schema."""
    manifest = json.loads(Path(path).read_text())
    _require_keys(manifest, REQUIRED_KEYS, "manifest")
    _require_keys(
        manifest["implementation_versions"],
        {"stable_baselines3", "sb3_contrib", "gymnasium", "mujoco"},
        "implementation_versions",
    )
    _require_keys(manifest["evaluation"], {"frequency", "episodes", "envs"}, "evaluation")
    _require_provenance(manifest["max_steps_provenance"], "max_steps_provenance")
    _require_provenance(manifest["evaluation_provenance"], "evaluation_provenance")
    _require_keys(
        manifest["settings"],
        {"paper", "implementation_choices", "sb3_mappings"},
        "settings",
    )
    _require_keys(manifest["settings"]["paper"], PAPER_SETTING_KEYS, "settings.paper")
    _require_keys(
        manifest["settings"]["implementation_choices"],
        IMPLEMENTATION_CHOICE_KEYS,
        "settings.implementation_choices",
    )
    for section in ("paper", "implementation_choices"):
        for name, entry in manifest["settings"][section].items():
            _require_keys(entry, {"value", "provenance"}, f"settings.{section}.{name}")
            _require_provenance(entry["provenance"], f"settings.{section}.{name}.provenance")
    mappings = manifest["settings"]["sb3_mappings"]
    _require_keys(mappings, SB3_MAPPING_KEYS, "settings.sb3_mappings")
    mapping_keys = {
        "constructor_kwargs": {
            "learning_rate", "buffer_size", "learning_starts", "batch_size", "gamma",
            "train_freq", "gradient_steps", "ent_coef", "policy_delay",
        },
        "policy_kwargs": {
            "net_arch", "activation_fn", "batch_norm", "batch_norm_momentum",
            "renorm_warmup_steps", "n_critics", "optimizer_kwargs",
        },
    }
    for section, required in mapping_keys.items():
        _require_keys(mappings[section], required, f"settings.sb3_mappings.{section}")
        for name, entry in mappings[section].items():
            _require_keys(entry, {"value", "provenance"}, f"settings.{name}")
            _require_provenance(entry["provenance"], f"settings.sb3_mappings.{section}.{name}.provenance")
    _require_provenance(
        mappings["target_network_behavior"],
        "settings.sb3_mappings.target_network_behavior provenance",
    )
    _require_keys(manifest["task_provenance"], set(TASKS), "task_provenance")
    if manifest["tasks"] != TASKS:
        raise ValueError("tasks do not match the frozen paper task IDs")
    for task, provenance in manifest["task_provenance"].items():
        _require_provenance(provenance, f"task_provenance.{task}")
    if not isinstance(manifest["seeds"], list) or not manifest["seeds"]:
        raise ValueError("seeds must be a non-empty list")
    return manifest


def _require_keys(value, required, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    actual = set(value)
    missing = required - actual
    unknown = actual - required
    if missing or unknown:
        raise ValueError(f"{name} keys mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}")


def _require_provenance(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} provenance must be a non-empty string")


def test_manifest_has_all_tasks_and_pinned_versions():
    manifest = load_manifest()
    assert manifest["tasks"] == TASKS
    assert manifest["implementation_versions"] == {
        "stable_baselines3": "2.9.0",
        "sb3_contrib": "2.9.0",
        "gymnasium": "1.4.0",
        "mujoco": "3.16.0",
    }


def test_manifest_has_ten_intended_seeds_and_step_budget():
    manifest = load_manifest()
    assert manifest["seeds"] == list(range(10))
    assert manifest["max_steps"] == 5_000_000
    assert "Figure 5" in manifest["max_steps_provenance"]
    assert "Appendix A.3" in manifest["max_steps_provenance"]


def test_paper_settings_and_provenance():
    settings = load_manifest()["settings"]
    paper = settings["paper"]
    assert {key: value["value"] for key, value in paper.items()} == {
        "gamma": 0.99,
        "actor_learning_rate": 0.001,
        "critic_learning_rate": 0.001,
        "replay_buffer_size": 1_000_000,
        "batch_size": 256,
        "activation": "ReLU",
        "normalization": "BRN",
        "brn_running_average_momentum": 0.99,
        "brn_warmup_steps": 100_000,
        "critic_architecture": [2048, 2048],
        "adam_beta1": 0.5,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 3,
        "n_critics": 2,
    }
    assert all("Appendix A.2, Table 1" in entry["provenance"] for entry in paper.values())
    choices = settings["implementation_choices"]
    assert choices["actor_architecture"]["value"] == [256, 256]
    assert "not specified" in choices["actor_architecture"]["provenance"].lower()
    assert choices["entropy_coefficient"]["value"] == "auto"
    assert choices["learning_starts"]["value"] == 100
    assert choices["entropy_coefficient"]["provenance"].startswith("Modern SB3 default")
    assert choices["learning_starts"]["provenance"].startswith("Modern SB3 default")


def test_evaluation_is_explicitly_non_paper_modern_protocol():
    manifest = load_manifest()
    assert manifest["evaluation"] == {"frequency": 25_000, "episodes": 20, "envs": 5}
    assert "non-paper" in manifest["evaluation_provenance"].lower()


def test_sb3_contrib_constructor_and_policy_mappings_are_realizable():
    mappings = load_manifest()["settings"]["sb3_mappings"]
    assert set(mappings) == SB3_MAPPING_KEYS
    constructor = {key: item["value"] for key, item in mappings["constructor_kwargs"].items()}
    assert constructor == {
        "learning_rate": 0.001,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 256,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "policy_delay": 3,
    }
    policy = {key: item["value"] for key, item in mappings["policy_kwargs"].items()}
    assert policy == {
        "net_arch": {"pi": [256, 256], "qf": [2048, 2048]},
        "activation_fn": "torch.nn.ReLU",
        "batch_norm": True,
        "batch_norm_momentum": 0.01,
        "renorm_warmup_steps": 100_000,
        "n_critics": 2,
        "optimizer_kwargs": {"betas": [0.5, 0.999]},
    }
    assert "internally" in mappings["target_network_behavior"].lower()
    assert "tau=1.0" in mappings["target_network_behavior"]
    assert all(item["provenance"] for section in ("constructor_kwargs", "policy_kwargs") for item in mappings[section].values())


def test_task_ids_have_paper_provenance_and_are_validated(tmp_path):
    manifest = load_manifest()
    assert manifest["tasks"] == TASKS
    assert all("Appendix A.3, Table 2" in p for p in manifest["task_provenance"].values())
    assert "Section 4" in manifest["task_provenance"]["Humanoid"]

    manifest["tasks"]["Hopper"] = "Hopper-v5"
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="tasks"):
        load_manifest(path)


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_manifest_rejects_missing_or_unknown_nested_required_fields(tmp_path, change):
    manifest = json.loads(CONFIG_PATH.read_text())
    policy_kwargs = manifest["settings"]["sb3_mappings"]["policy_kwargs"]
    if change == "missing":
        del policy_kwargs["n_critics"]
    else:
        policy_kwargs["surprise"] = {"value": 1, "provenance": "test"}
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


@pytest.mark.parametrize(
    "location",
    [
        ("task_provenance", "Hopper"),
        ("settings", "paper", "gamma", "provenance"),
        ("settings", "implementation_choices", "actor_architecture", "provenance"),
        ("settings", "sb3_mappings", "constructor_kwargs", "learning_rate", "provenance"),
        ("settings", "sb3_mappings", "policy_kwargs", "net_arch", "provenance"),
        ("settings", "sb3_mappings", "target_network_behavior"),
    ],
)
@pytest.mark.parametrize("invalid", ["", 123])
def test_manifest_rejects_empty_or_non_string_provenance(tmp_path, location, invalid):
    manifest = json.loads(CONFIG_PATH.read_text())
    parent = manifest
    for key in location[:-1]:
        parent = parent[key]
    parent[location[-1]] = invalid
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        load_manifest(path)


@pytest.mark.parametrize("change", ["missing", "unknown"])
def test_manifest_rejects_missing_or_unknown_required_fields(tmp_path, change):
    manifest = json.loads(CONFIG_PATH.read_text())
    if change == "missing":
        del manifest["max_steps"]
    else:
        manifest["surprise"] = True
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_manifest_rejects_empty_seed_list(tmp_path):
    manifest = json.loads(CONFIG_PATH.read_text())
    manifest["seeds"] = []
    path = tmp_path / "invalid.yaml"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="non-empty"):
        load_manifest(path)
