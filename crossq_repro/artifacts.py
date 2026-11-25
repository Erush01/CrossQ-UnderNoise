"""Run artifact and provenance helpers."""

import hashlib
import json
from importlib import metadata as package_metadata
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions():
    versions = {}
    for package in ("stable-baselines3", "sb3-contrib", "gymnasium", "mujoco", "torch", "numpy"):
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_metadata(config, env_id, seed, total_timesteps, started_at, finished_at,
                   *, config_path, source_paths, versions=None):
    return {
        "resolved_config": config,
        "env_id": env_id,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "started_at": started_at,
        "finished_at": finished_at,
        "package_versions": package_versions() if versions is None else versions,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {str(Path(path)): sha256_file(path) for path in source_paths},
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
