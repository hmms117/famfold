"""Stage 00 environment and configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest
import yaml

from minifold.famfold.configuration import (
    DEFAULT_OPTIONAL_ENVIRONMENT_FILES,
    DEFAULT_REQUIRED_ENVIRONMENT_FILES,
    REQUIRED_ENVIRONMENT_FILES,
    hash_config_files,
    load_environment_config,
    load_environment_from_setup,
    resolve_environment_layout,
    write_run_manifest,
)


def _create_yaml(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def test_load_environment_config_reads_required_files(tmp_path: Path) -> None:
    """The loader should materialise a mapping for each required YAML file."""

    payloads = {
        "paths.yaml": {"data_root": "/data", "cache_root": "/cache", "zarr_root": "/zarr"},
        "default.yaml": {"gamma_base": 0.7, "beta": 1.5},
        "thresholds.yaml": {"ACCEPT": 0.25, "REFINE": 0.35, "ESCALATE": 0.5},
        "buckets.yaml": {"buckets": {"256": {"pad_to": 256}}},
        "retrieval.yaml": {"backend": "faiss", "k": 6},
    }

    for filename, contents in payloads.items():
        _create_yaml(tmp_path / filename, contents)

    config = load_environment_config(tmp_path)

    assert set(config) == {"paths", "defaults", "thresholds", "buckets", "retrieval"}
    assert config["paths"]["data_root"] == "/data"
    assert config["defaults"]["gamma_base"] == pytest.approx(0.7)
    assert config["thresholds"]["ACCEPT"] == pytest.approx(0.25)
    assert config["buckets"]["buckets"]["256"]["pad_to"] == 256
    assert config["retrieval"]["backend"] == "faiss"


def test_load_environment_config_accepts_custom_filenames(tmp_path: Path) -> None:
    """Custom filename overrides should be honoured when provided."""

    required_files = {
        key: f"custom_{filename}"
        for key, filename in DEFAULT_REQUIRED_ENVIRONMENT_FILES.items()
    }
    optional_files = {
        key: f"custom_{filename}"
        for key, filename in DEFAULT_OPTIONAL_ENVIRONMENT_FILES.items()
    }

    payloads = {
        **{
            filename: {"value": idx}
            for idx, filename in enumerate(required_files.values(), start=1)
        },
        **{
            filename: {"opt": True}
            for filename in optional_files.values()
        },
    }

    for filename, contents in payloads.items():
        _create_yaml(tmp_path / filename, contents)

    config = load_environment_config(
        tmp_path,
        required_files=required_files,
        optional_files=optional_files,
    )

    assert set(config) == set(required_files) | set(optional_files)
    assert config["paths"]["value"] == 1
    assert config["defaults"]["value"] == 2


def test_load_environment_config_requires_paths_file(tmp_path: Path) -> None:
    """Stage 00 insists on explicit path overrides."""

    payloads = {
        "default.yaml": {"gamma_base": 0.7},
        "thresholds.yaml": {"ACCEPT": 0.3},
        "buckets.yaml": {"buckets": {}},
    }

    for filename, contents in payloads.items():
        _create_yaml(tmp_path / filename, contents)

    with pytest.raises(FileNotFoundError):
        load_environment_config(tmp_path)


def test_hash_config_files_uses_stable_digest(tmp_path: Path) -> None:
    """The hashing helper should produce deterministic digests per file."""

    files = {}
    for name in REQUIRED_ENVIRONMENT_FILES.values():
        path = tmp_path / name
        path.write_text(f"key: value-for-{name}\n", encoding="utf-8")
        files[name] = path

    digests = hash_config_files(files)

    assert set(digests) == set(files)

    second = hash_config_files(files)
    assert digests == second


def test_write_run_manifest_materialises_expected_payload(tmp_path: Path) -> None:
    """Stage 09 manifest writer should include config hashes and metadata."""

    required_files = DEFAULT_REQUIRED_ENVIRONMENT_FILES
    optional_files = DEFAULT_OPTIONAL_ENVIRONMENT_FILES

    payloads = {
        "paths.yaml": {"data_root": "/data", "cache_root": "/cache"},
        "default.yaml": {"gamma_base": 0.7},
        "thresholds.yaml": {"ACCEPT": 0.25},
        "buckets.yaml": {"buckets": {}},
        "retrieval.yaml": {"backend": "faiss"},
    }

    for filename, contents in payloads.items():
        _create_yaml(tmp_path / filename, contents)

    manifest_path = tmp_path / "manifests" / "run_manifest.json"
    metadata = {"family_id": "F001", "version": "2024.10"}
    manifest = write_run_manifest(
        manifest_path,
        config_dir=tmp_path,
        run_metadata=metadata,
        artifacts={"predictions": tmp_path / "predictions.pdb"},
        required_files=required_files,
        optional_files=optional_files,
    )

    assert manifest_path.exists()
    loaded = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    for key in ("metadata", "config_hashes", "artifacts"):
        assert key in manifest
        assert manifest[key] == loaded[key]

    assert manifest["metadata"] == metadata
    assert set(manifest["config_hashes"]) == {
        "paths.yaml",
        "default.yaml",
        "thresholds.yaml",
        "buckets.yaml",
        "retrieval.yaml",
    }


def test_resolve_environment_layout_matches_repo_defaults() -> None:
    """The tracked Hydra config should align with the exported defaults."""

    config_dir = Path(__file__).resolve().parents[1] / "configs" / "famfold"
    config = yaml.safe_load((config_dir / "setup.yaml").read_text(encoding="utf-8"))

    layout = resolve_environment_layout(config["setup"], base_dir=config_dir)

    assert layout["profile"] == "familyfold"
    assert layout["root"].resolve() == config_dir.resolve()
    assert layout["required_files"] == dict(DEFAULT_REQUIRED_ENVIRONMENT_FILES)
    assert layout["optional_files"] == dict(DEFAULT_OPTIONAL_ENVIRONMENT_FILES)


def test_load_environment_from_setup_handles_relative_root(tmp_path: Path) -> None:
    """``load_environment_from_setup`` should respect ``base_dir`` overrides."""

    setup = {
        "default_profile": "familyfold",
        "profiles": {
            "familyfold": {
                "environment": {
                    "root": ".",
                    "required_files": {
                        "paths": "paths.yaml",
                        "defaults": "default.yaml",
                    },
                    "optional_files": {"retrieval": "retrieval.yaml"},
                }
            }
        },
    }

    config_dir = tmp_path / "config"
    config_dir.mkdir()

    payloads = {
        "paths.yaml": {"data_root": "/tmp/data"},
        "default.yaml": {"gamma_base": 0.9},
        "retrieval.yaml": {"backend": "proteinttt"},
    }

    for filename, contents in payloads.items():
        _create_yaml(config_dir / filename, contents)

    config = load_environment_from_setup(setup, base_dir=config_dir)

    assert config["paths"]["data_root"] == "/tmp/data"
    assert config["defaults"]["gamma_base"] == pytest.approx(0.9)
    assert config["retrieval"]["backend"] == "proteinttt"

