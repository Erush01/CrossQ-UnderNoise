import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from crossq_repro import runner


class MockReLU:
    pass


def test_runner_resolves_manifest_kwargs_and_fixed_seed(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    seen = {}
    created_envs = []
    monitor_calls = []

    def make_env(env_id):
        env = SimpleNamespace(env_id=env_id, close=lambda: None)
        created_envs.append(env)
        return env

    def monitor(env):
        monitored = SimpleNamespace(environment=env, close=env.close)
        monitor_calls.append(monitored)
        return monitored

    class Model:
        def __init__(self, policy, env, **kwargs):
            seen.update(policy=policy, env=env, kwargs=kwargs)
            self.env = env
            self.device = kwargs["device"]

        def learn(self, total_timesteps, callback):
            seen.update(steps=total_timesteps, callback=callback)
            return self

        def save(self, path):
            Path(str(path) + ".zip").write_text("checkpoint")

    callback = object()

    def make_callback(env, **kwargs):
        seen["eval_env"] = env
        seen["eval_kwargs"] = kwargs
        return callback

    def make_eval_env(factories):
        seen["eval_env_count"] = len(factories)
        seen["eval_envs"] = [factory() for factory in factories]
        return SimpleNamespace(
            close=lambda: None,
            seed=lambda value: seen.update(eval_seed=value),
        )

    monkeypatch.setattr(runner, "_load_dependencies", lambda: SimpleNamespace(
        CrossQ=Model, torch=SimpleNamespace(nn=SimpleNamespace(ReLU=MockReLU)),
        gym=SimpleNamespace(make=make_env), Monitor=monitor,
        EvalCallback=make_callback,
        DummyVecEnv=make_eval_env,
    ))
    monkeypatch.setattr(runner, "_set_random_seed", lambda seed: seen.update(seed=seed))
    provenance = tmp_path / "pilot-launcher.sh"
    provenance.write_text("pilot launcher")
    runner.run(config, "Hopper-v4", 7, 23, tmp_path / "nested" / "run-a", device="cpu",
               provenance_paths=[provenance])
    assert seen["seed"] == 7
    assert seen["steps"] == 23
    assert seen["kwargs"]["seed"] == 7
    assert seen["kwargs"]["device"] == "cpu"
    expected_constructor = {
        key: value["value"]
        for key, value in config["settings"]["sb3_mappings"]["constructor_kwargs"].items()
    }
    assert {key: value for key, value in seen["kwargs"].items()
            if key not in ("seed", "device", "policy_kwargs")} == expected_constructor
    expected_policy = json.loads(json.dumps({
        key: value["value"]
        for key, value in config["settings"]["sb3_mappings"]["policy_kwargs"].items()
    }))
    expected_policy["activation_fn"] = MockReLU
    expected_policy["optimizer_kwargs"]["betas"] = (0.5, 0.999)
    assert seen["kwargs"]["policy_kwargs"] == expected_policy
    assert seen["callback"] is callback
    assert seen["eval_kwargs"]["eval_freq"] == config["evaluation"]["frequency"]
    assert seen["eval_kwargs"]["n_eval_episodes"] == config["evaluation"]["episodes"]
    assert seen["eval_env_count"] == config["evaluation"]["envs"]
    assert monitor_calls == seen["eval_envs"]
    assert all(env.environment in created_envs[1:] for env in monitor_calls)
    assert created_envs[0] not in [env.environment for env in monitor_calls]
    assert seen["eval_seed"] == 10007
    out = tmp_path / "nested" / "run-a"
    snapshot = out / "config.json"
    assert json.loads(snapshot.read_text()) == config
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["resolved_config"]["config"] == config
    assert metadata["resolved_config"]["constructor_kwargs"] == expected_constructor
    assert metadata["resolved_config"]["policy_kwargs"]["activation_fn"] == (
        f"{MockReLU.__module__}.{MockReLU.__qualname__}"
    )
    assert metadata["resolved_config"]["evaluation_seed"] == 10007
    assert metadata["resolved_config"]["device"] == {"requested": "cpu", "resolved": "cpu"}
    assert metadata["config_sha256"] == hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert metadata["source_sha256"][str(provenance)] == hashlib.sha256(provenance.read_bytes()).hexdigest()
    assert (out / "final_model.zip").is_file()


def test_run_refuses_existing_directory(tmp_path):
    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        runner.run({}, "Hopper-v4", 0, 1, existing)


@pytest.mark.parametrize("env_id,seed,steps", [("Unknown-v0", 0, 1), ("Hopper-v4", 99, 1), ("Hopper-v4", 0, 0)])
def test_run_rejects_unconfigured_env_seed_or_nonpositive_steps(tmp_path, env_id, seed, steps):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    output_dir = tmp_path / "invalid"
    with pytest.raises(ValueError):
        runner.run(config, env_id, seed, steps, output_dir)
    assert not output_dir.exists()


def test_dependency_loading_failure_leaves_no_run_directory(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    output_dir = tmp_path / "dependency-failure"

    def fail_to_load():
        raise RuntimeError("dependency loading failed")

    monkeypatch.setattr(runner, "_load_dependencies", fail_to_load)
    with pytest.raises(RuntimeError, match="dependency loading failed"):
        runner.run(config, "Hopper-v4", 0, 1, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("fail_at", ["construct", "learn"])
def test_run_closes_created_environments_when_model_fails(tmp_path, monkeypatch, fail_at):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    closed = []
    raw_envs = []

    def make_env(env_id):
        env = SimpleNamespace(close=lambda: closed.append(env_id))
        raw_envs.append(env)
        return env

    class FailingModel:
        def __init__(self, *args, **kwargs):
            if fail_at == "construct":
                raise RuntimeError("construct failed")

        def learn(self, **kwargs):
            raise RuntimeError("learn failed")

    class EvalEnv:
        def seed(self, seed):
            pass

        def close(self):
            for env in raw_envs[1:]:
                env.close()

    monkeypatch.setattr(runner, "_load_dependencies", lambda: SimpleNamespace(
        CrossQ=FailingModel, torch=SimpleNamespace(nn=SimpleNamespace(ReLU="relu")),
        gym=SimpleNamespace(make=make_env), Monitor=lambda env: env,
        EvalCallback=lambda *a, **kw: object(),
        DummyVecEnv=lambda factories: EvalEnv(),
    ))
    monkeypatch.setattr(runner, "_set_random_seed", lambda seed: None)
    with pytest.raises(RuntimeError, match=f"{fail_at} failed"):
        runner.run(config, "Hopper-v4", 0, 1, tmp_path / fail_at)
    assert len(closed) == 1 + config["evaluation"]["envs"]


@pytest.mark.parametrize("fail_call,expected_closed", [(1, 0), (2, 1), (4, 3)])
def test_run_closes_partial_envs_when_environment_creation_fails(
    tmp_path, monkeypatch, fail_call, expected_closed,
):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    created, closed = [], []

    def make_env(env_id):
        if len(created) + 1 == fail_call:
            raise RuntimeError("environment creation failed")
        env = SimpleNamespace(close=lambda: closed.append(env_id))
        created.append(env)
        return env

    monkeypatch.setattr(runner, "_load_dependencies", lambda: SimpleNamespace(
        CrossQ=object, torch=SimpleNamespace(nn=SimpleNamespace(ReLU="relu")),
        gym=SimpleNamespace(make=make_env), Monitor=lambda env: env,
        EvalCallback=lambda *a, **kw: None,
        DummyVecEnv=lambda factories: None,
    ))
    monkeypatch.setattr(runner, "_set_random_seed", lambda seed: None)
    with pytest.raises(RuntimeError, match="environment creation failed"):
        runner.run(config, "Hopper-v4", 0, 1, tmp_path / f"env-{fail_call}")
    assert len(closed) == expected_closed


def test_close_failure_does_not_mask_active_learn_error(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs" / "paper_mujoco.yaml").read_text())
    learn_error = RuntimeError("learn sentinel")
    close_error = RuntimeError("close sentinel")

    class Env:
        def close(self):
            raise close_error

    class EvalEnv(Env):
        def seed(self, seed):
            pass

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def learn(self, **kwargs):
            raise learn_error

    monkeypatch.setattr(runner, "_load_dependencies", lambda: SimpleNamespace(
        CrossQ=Model, torch=SimpleNamespace(nn=SimpleNamespace(ReLU="relu")),
        gym=SimpleNamespace(make=lambda env_id: Env()),
        Monitor=lambda env: env,
        EvalCallback=lambda *a, **kw: object(), DummyVecEnv=lambda factories: EvalEnv(),
    ))
    monkeypatch.setattr(runner, "_set_random_seed", lambda seed: None)
    with pytest.raises(RuntimeError) as error:
        runner.run(config, "Hopper-v4", 0, 1, tmp_path / "close-errors")
    assert error.value is learn_error
