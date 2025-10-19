"""FamilyFold configuration helpers for Stage 00 and Stage 09."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

import yaml

try:  # pragma: no cover - optional dependency
    from blake3 import blake3 as _blake3
except Exception:  # pragma: no cover - optional dependency
    _blake3 = None


DEFAULT_REQUIRED_ENVIRONMENT_FILES: Mapping[str, str] = {
    "paths": "paths.yaml",
    "defaults": "default.yaml",
    "thresholds": "thresholds.yaml",
    "buckets": "buckets.yaml",
}

DEFAULT_OPTIONAL_ENVIRONMENT_FILES: Mapping[str, str] = {
    "retrieval": "retrieval.yaml",
}

# Backwards-compatible aliases that downstream callers can continue importing.
REQUIRED_ENVIRONMENT_FILES: Mapping[str, str] = DEFAULT_REQUIRED_ENVIRONMENT_FILES
OPTIONAL_ENVIRONMENT_FILES: Mapping[str, str] = DEFAULT_OPTIONAL_ENVIRONMENT_FILES


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def _normalise_file_mapping(
    mapping: Optional[Mapping[str, object]],
    *,
    fallback: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return a normalised copy of ``mapping`` with stringified filenames."""

    if mapping is None:
        mapping = fallback or {}

    normalised: Dict[str, str] = {}
    for namespace, filename in mapping.items():
        normalised[str(namespace)] = str(filename)
    return normalised


def load_environment_config(
    root: Path,
    *,
    required_files: Optional[Mapping[str, object]] = None,
    optional_files: Optional[Mapping[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    """Load Stage 00 configuration fragments from ``root``.

    Parameters
    ----------
    root:
        Directory containing the expected YAML configuration files.

    Returns
    -------
    Dict[str, Dict[str, object]]
        Mapping of configuration namespace (``paths``, ``defaults`` …) to the
        parsed YAML payload.

    Raises
    ------
    FileNotFoundError
        If any of the required Stage 00 YAML files is missing.
    """

    root_path = Path(root)
    config: Dict[str, Dict[str, object]] = {}
    missing = []

    required = _normalise_file_mapping(
        required_files, fallback=DEFAULT_REQUIRED_ENVIRONMENT_FILES
    )
    optional = _normalise_file_mapping(
        optional_files, fallback=DEFAULT_OPTIONAL_ENVIRONMENT_FILES
    )

    for namespace, filename in required.items():
        file_path = root_path / filename
        if not file_path.exists():
            missing.append(filename)
            continue
        config[namespace] = _load_yaml(file_path)

    if missing:
        raise FileNotFoundError(
            "Missing required environment configuration files: " + ", ".join(missing)
        )

    for namespace, filename in optional.items():
        file_path = root_path / filename
        if file_path.exists():
            config[namespace] = _load_yaml(file_path)

    return config


def _hash_bytes(payload: bytes) -> str:
    if _blake3 is not None:
        return _blake3(payload).hexdigest()
    return hashlib.blake2b(payload, digest_size=32).hexdigest()


def hash_config_files(files: Mapping[str, Path]) -> Dict[str, str]:
    """Return deterministic digests for the provided configuration files."""

    hashes: Dict[str, str] = {}
    for name, path in files.items():
        file_path = Path(path)
        with file_path.open("rb") as handle:
            hashes[name] = _hash_bytes(handle.read())
    return hashes


def write_run_manifest(
    destination: Path,
    *,
    config_dir: Path,
    run_metadata: Mapping[str, object],
    artifacts: Optional[Mapping[str, Path]] = None,
    required_files: Optional[Mapping[str, object]] = None,
    optional_files: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Write a Stage 09 run manifest summarising the current execution."""

    config_dir = Path(config_dir)

    namespace_to_filename: MutableMapping[str, str] = {}
    namespace_to_filename.update(
        _normalise_file_mapping(
            required_files, fallback=DEFAULT_REQUIRED_ENVIRONMENT_FILES
        )
    )
    namespace_to_filename.update(
        _normalise_file_mapping(
            optional_files, fallback=DEFAULT_OPTIONAL_ENVIRONMENT_FILES
        )
    )

    files: Dict[str, Path] = {}
    for filename in namespace_to_filename.values():
        file_path = config_dir / filename
        if file_path.exists():
            files[filename] = file_path

    config_hashes = hash_config_files(files)

    manifest: Dict[str, object] = {
        "metadata": dict(run_metadata),
        "config_hashes": config_hashes,
        "artifacts": {
            name: str(Path(path)) for name, path in (artifacts or {}).items()
        },
    }

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return manifest


def resolve_environment_layout(
    setup: Mapping[str, object],
    *,
    profile: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, object]:
    """Resolve the environment layout from a Hydra ``setup`` mapping.

    Parameters
    ----------
    setup:
        Mapping loaded from ``configs/famfold/setup.yaml`` (or equivalent).
    profile:
        Optional profile name to select.  If omitted the ``default_profile``
        entry is used, falling back to the sole available profile.
    base_dir:
        Optional directory that ``environment.root`` should be resolved
        against when it contains a relative path.

    Returns
    -------
    Dict[str, object]
        Dictionary with ``profile``, ``root`` (a :class:`~pathlib.Path`
        instance), ``required_files`` and ``optional_files`` mappings.
    """

    if not isinstance(setup, Mapping):
        raise TypeError("setup must be a mapping")

    profiles = setup.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise KeyError("setup.profiles must define at least one profile")

    selected_profile = profile or setup.get("default_profile")
    if selected_profile is None:
        # If no explicit default is provided, fall back to the sole profile.
        if len(profiles) == 1:
            selected_profile = next(iter(profiles))
        else:
            raise KeyError("setup.default_profile must be specified")

    profile_config = profiles.get(selected_profile)
    if not isinstance(profile_config, Mapping):
        raise KeyError(f"setup profile '{selected_profile}' is undefined")

    environment_config = profile_config.get("environment")
    if not isinstance(environment_config, Mapping):
        raise KeyError(
            f"setup profile '{selected_profile}' is missing an environment section"
        )

    root_entry = environment_config.get("root", ".")
    root_path = Path(root_entry)
    if base_dir is not None and not root_path.is_absolute():
        root_path = Path(base_dir) / root_path

    required = _normalise_file_mapping(
        environment_config.get("required_files"),
        fallback=DEFAULT_REQUIRED_ENVIRONMENT_FILES,
    )
    optional = _normalise_file_mapping(
        environment_config.get("optional_files"),
        fallback=DEFAULT_OPTIONAL_ENVIRONMENT_FILES,
    )

    return {
        "profile": selected_profile,
        "root": root_path,
        "required_files": required,
        "optional_files": optional,
    }


def load_environment_from_setup(
    setup: Mapping[str, object],
    *,
    profile: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, object]]:
    """Convenience wrapper using a Hydra ``setup`` mapping for Stage 00."""

    layout = resolve_environment_layout(setup, profile=profile, base_dir=base_dir)
    return load_environment_config(
        layout["root"],
        required_files=layout["required_files"],
        optional_files=layout["optional_files"],
    )

