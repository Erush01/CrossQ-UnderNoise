import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
ARTIFACTS_PATH = ROOT / "crossq_repro" / "artifacts.py"
sys.path.insert(0, str(ROOT))
from crossq_repro import artifacts


def test_metadata_contains_hashes_versions_and_run_provenance(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"small": true}')
    source_path = tmp_path / "source.py"
    source_path.write_text("source")
    metadata = artifacts.build_metadata(
        {"schema_version": 1}, "Hopper-v4", 7, 100, "start", "finish",
        config_path=config_path, source_paths=[source_path], versions={"test": "1"},
    )
    assert metadata["env_id"] == "Hopper-v4"
    assert metadata["seed"] == 7 and metadata["total_timesteps"] == 100
    assert metadata["started_at"] == "start" and metadata["finished_at"] == "finish"
    assert metadata["package_versions"] == {"test": "1"}
    assert len(metadata["config_sha256"]) == 64
    assert len(metadata["source_sha256"][str(source_path)]) == 64


def test_evaluation_keeps_raw_episode_returns_and_lengths():
    from crossq_repro.evaluate import summarize_episodes

    result = summarize_episodes([1, 3, 5], [2, 4, 6])
    assert result["episode_returns"] == [1, 3, 5]
    assert result["episode_lengths"] == [2, 4, 6]
    assert result["mean_return"] == 3
    assert result["std_return"] == (8 / 3) ** 0.5


def _install_eval_fakes(monkeypatch, env):
    loaded = []

    class FakeCrossQ:
        @classmethod
        def load(cls, checkpoint, device):
            loaded.append((checkpoint, device))
            return SimpleNamespace(predict=lambda obs, deterministic: ("action", None))

    monkeypatch.setitem(sys.modules, "sb3_contrib", SimpleNamespace(CrossQ=FakeCrossQ))
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    return loaded


def test_evaluate_loads_checkpoint_preserves_episodes_writes_json_and_closes(tmp_path, monkeypatch):
    from crossq_repro.evaluate import evaluate

    resets, actions, closed = [], [], []
    steps = iter([(2, False, False), (3, True, False), (4, False, True)])

    class Env:
        def reset(self, seed):
            resets.append(seed)
            return "obs", {}

        def step(self, action):
            actions.append(action)
            reward, terminated, truncated = next(steps)
            return "obs", reward, terminated, truncated, {}

        def close(self):
            closed.append(True)

    loaded = _install_eval_fakes(monkeypatch, Env())
    output = tmp_path / "evaluation.json"
    result = evaluate("checkpoint.zip", "Hopper-v4", 2, seed=31, output_path=output, device="cpu")
    assert loaded == [("checkpoint.zip", "cpu")]
    assert resets == [31, 32]
    assert actions == ["action", "action", "action"]
    assert result["episodes"]["episode_returns"] == [5.0, 4.0]
    assert result["episodes"]["episode_lengths"] == [2, 1]
    assert json.loads(output.read_text()) == result
    assert closed == [True]


def test_evaluate_closes_env_if_episode_step_fails(monkeypatch):
    from crossq_repro.evaluate import evaluate

    closed = []

    class Env:
        def reset(self, seed):
            return "obs", {}

        def step(self, action):
            raise RuntimeError("step failed")

        def close(self):
            closed.append(True)

    _install_eval_fakes(monkeypatch, Env())
    with pytest.raises(RuntimeError, match="step failed"):
        evaluate("checkpoint.zip", "Hopper-v4", 1)
    assert closed == [True]


@pytest.mark.parametrize("episodes", [0, -1])
def test_evaluate_rejects_nonpositive_episode_count(episodes):
    from crossq_repro.evaluate import evaluate

    with pytest.raises(ValueError, match="episodes"):
        evaluate("not-loaded", "Hopper-v4", episodes)
