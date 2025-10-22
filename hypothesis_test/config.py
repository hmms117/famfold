"""Configuration schema for the homolog template benchmark pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union
import json

CACHE_ROOT = Path("/var/tmp/checkpoints")


def _clean_sequence(value: str, identifier: str) -> str:
    sequence = value.replace("\n", "").replace(" ", "").strip().upper()
    if not sequence:
        raise ValueError(f"Sequence for target '{identifier}' is empty.")
    return sequence


def _load_sequence_from_path(path: Path, identifier: str) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Sequence file '{path}' for target '{identifier}' does not exist.")

    sequence_lines: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if sequence_lines:
                    break
                continue
            sequence_lines.append(stripped)

    if not sequence_lines:
        raise ValueError(f"No sequence content found in '{path}' for target '{identifier}'.")

    return _clean_sequence("".join(sequence_lines), identifier)


class Split(str, Enum):
    """Enumeration of the supported dataset splits."""

    PILOT_ID = "pilot_id"
    PILOT_OOD = "pilot_ood"
    FULL_ID = "full_id"
    FULL_OOD = "full_ood"

    @classmethod
    def has_value(cls, value: str) -> bool:
        try:
            cls(value)
        except ValueError:
            return False
        return True


@dataclass
class TargetConfig:
    """Represents a single benchmark target."""

    identifier: str
    sequence: str
    split: Split
    metadata: Dict[str, object] = field(default_factory=dict)
    sequence_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, object], base_dir: Optional[Path] = None) -> "TargetConfig":
        payload = dict(payload)

        extra_base_dir_value = payload.pop("_base_dir", None)
        extra_base_dir = Path(extra_base_dir_value) if extra_base_dir_value else None

        identifier = str(payload["id"]).strip()
        if not identifier:
            raise ValueError("Target identifier must be a non-empty string.")

        sequence_value = payload.get("sequence")
        sequence_path_value = payload.get("sequence_path")
        if sequence_value is not None and sequence_path_value is not None:
            raise ValueError(
                f"Target '{identifier}' specifies both 'sequence' and 'sequence_path'; choose one source."
            )

        if sequence_path_value is not None:
            raw_path = Path(str(sequence_path_value))
            resolution_base = extra_base_dir or base_dir
            resolved_path = raw_path if raw_path.is_absolute() else (resolution_base / raw_path if resolution_base else raw_path)
            sequence = _load_sequence_from_path(resolved_path, identifier)
            sequence_path = resolved_path
        elif sequence_value is not None:
            sequence = _clean_sequence(str(sequence_value), identifier)
            sequence_path = None
        else:
            raise ValueError(
                f"Target '{identifier}' must provide either a 'sequence' literal or a 'sequence_path'."
            )

        split_value = str(payload.get("split", Split.FULL_ID.value))
        if not Split.has_value(split_value):
            valid = ", ".join(v.value for v in Split)
            raise ValueError(
                f"Invalid split '{split_value}' for target '{identifier}'. Valid options: {valid}."
            )

        metadata = dict(payload.get("metadata", {}))
        return cls(
            identifier=identifier,
            sequence=sequence,
            split=Split(split_value),
            metadata=metadata,
            sequence_path=sequence_path,
        )


@dataclass
class MinifoldRunSettings:
    """Configuration for running a Minifold inference job."""

    enabled: bool = True
    model_size: str = "48L"
    token_per_batch: int = 2048
    compile: bool = False
    kernels: bool = False
    checkpoint: Optional[Path] = None
    num_recycling: int = 3
    extra_args: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class BaselineSettings:
    """Configuration for third-party baseline predictors."""

    enabled: bool = False
    executable: Optional[str] = None
    extra_args: List[str] = field(default_factory=list)
    output_subdir: Optional[str] = None
    environment: Dict[str, str] = field(default_factory=dict)


@dataclass
class SequenceTrunkSettings:
    """Configuration for running SaESM/Saamplify sequence trunk embeddings."""

    enabled: bool = False
    checkpoint: Optional[str] = None
    batch_size: int = 8
    torch_dtype: Optional[str] = None
    device: Optional[str] = None
    device_map: Optional[Union[str, Dict[str, object]]] = None


@dataclass
class BenchmarkConfig:
    """Top-level configuration for the homolog template benchmark."""

    targets: List[TargetConfig]
    output_dir: Path
    cache_dir: Path
    pilot_subset_size: int = 5
    minifold_base: MinifoldRunSettings = field(default_factory=MinifoldRunSettings)
    minifold_templates: MinifoldRunSettings = field(default_factory=MinifoldRunSettings)
    minifold_faplm: MinifoldRunSettings = field(default_factory=MinifoldRunSettings)
    minifold_ism: MinifoldRunSettings = field(default_factory=MinifoldRunSettings)
    minifold_saesm2: MinifoldRunSettings = field(default_factory=MinifoldRunSettings)
    minifold_baselines: Dict[str, MinifoldRunSettings] = field(default_factory=dict)
    esmfold: BaselineSettings = field(default_factory=BaselineSettings)
    boltz2: BaselineSettings = field(default_factory=BaselineSettings)
    alphafold2: BaselineSettings = field(default_factory=BaselineSettings)
    sequence_trunks: Dict[str, SequenceTrunkSettings] = field(default_factory=dict)

    def get_targets(self, *splits: Split) -> List[TargetConfig]:
        """Return all targets belonging to the requested splits."""

        requested = set(splits) if splits else set(Split)
        return [target for target in self.targets if target.split in requested]

    @property
    def pilot_targets(self) -> List[TargetConfig]:
        """Return the configured pilot subset for smoke testing the pipelines."""

        pilot_splits = {Split.PILOT_ID, Split.PILOT_OOD}
        pilot = [t for t in self.targets if t.split in pilot_splits]
        if pilot:
            return pilot[: self.pilot_subset_size * 2]

        # Fall back to first N targets if explicit pilot set not provided.
        return self.targets[: self.pilot_subset_size * 2]


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_targets(raw_targets: Iterable[Dict[str, object]], base_dir: Optional[Path]) -> List[TargetConfig]:
    targets: List[TargetConfig] = []
    for payload in raw_targets:
        targets.append(TargetConfig.from_dict(payload, base_dir=base_dir))
    if not targets:
        raise ValueError("Benchmark configuration must contain at least one target.")
    return targets


def _normalise_minifold_payload(payload: Dict[str, object], base_dir: Path) -> Dict[str, object]:
    result = dict(payload)
    checkpoint_value = result.get("checkpoint")
    if checkpoint_value:
        checkpoint_path = Path(str(checkpoint_value))
        if not checkpoint_path.is_absolute():
            checkpoint_path = (base_dir / checkpoint_path).resolve()
        result["checkpoint"] = checkpoint_path
    return result


def load_config(path: Path) -> BenchmarkConfig:
    """Load a :class:`BenchmarkConfig` from a JSON file."""

    data = _load_json(path)

    base_dir = path.parent

    raw_targets: List[Dict[str, object]] = []

    targets_path_field: Union[str, Sequence[str], None] = data.get("targets_path")
    if targets_path_field:
        if isinstance(targets_path_field, str):
            target_paths = [targets_path_field]
        else:
            target_paths = list(targets_path_field)
        for entry in target_paths:
            entry_path = Path(entry)
            resolved = entry_path if entry_path.is_absolute() else (base_dir / entry_path)
            payload = _load_json(resolved)
            for target_payload in payload.get("targets", []):
                enriched = dict(target_payload)
                enriched["_base_dir"] = str(resolved.parent)
                raw_targets.append(enriched)

    raw_targets.extend(data.get("targets", []))

    targets = _parse_targets(raw_targets, base_dir)
    output_dir_value = data.get("output_dir", "./benchmark_outputs")
    output_dir = Path(output_dir_value)
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    raw_cache_dir = data.get("cache_dir")
    cache_dir = CACHE_ROOT if raw_cache_dir is None else Path(raw_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = CACHE_ROOT

    minifold_base = MinifoldRunSettings(
        **_normalise_minifold_payload(data.get("minifold_base", {}), base_dir)
    )
    minifold_templates = MinifoldRunSettings(
        **_normalise_minifold_payload(data.get("minifold_templates", {}), base_dir)
    )
    minifold_faplm = MinifoldRunSettings(
        **_normalise_minifold_payload(data.get("minifold_faplm", {}), base_dir)
    )
    minifold_ism = MinifoldRunSettings(
        **_normalise_minifold_payload(data.get("minifold_ism", {}), base_dir)
    )
    minifold_saesm2 = MinifoldRunSettings(
        **_normalise_minifold_payload(data.get("minifold_saesm2", {}), base_dir)
    )
    minifold_baselines_data = data.get("minifold_baselines", {})
    minifold_baselines: Dict[str, MinifoldRunSettings] = {}
    for name, payload in minifold_baselines_data.items():
        normalised = _normalise_minifold_payload(payload, base_dir)
        minifold_baselines[name] = MinifoldRunSettings(**normalised)
    esmfold = BaselineSettings(**data.get("esmfold", {}))
    boltz2 = BaselineSettings(**data.get("boltz2", {}))
    alphafold2 = BaselineSettings(**data.get("alphafold2", {}))

    sequence_trunks_data = data.get("sequence_trunks", {})
    sequence_trunks: Dict[str, SequenceTrunkSettings] = {}
    for name, payload in sequence_trunks_data.items():
        sequence_trunks[name] = SequenceTrunkSettings(**payload)

    pilot_subset_size = int(data.get("pilot_subset_size", 5))

    return BenchmarkConfig(
        targets=targets,
        output_dir=output_dir,
        cache_dir=cache_dir,
        pilot_subset_size=pilot_subset_size,
        minifold_base=minifold_base,
        minifold_templates=minifold_templates,
        minifold_faplm=minifold_faplm,
        minifold_ism=minifold_ism,
        minifold_saesm2=minifold_saesm2,
        minifold_baselines=minifold_baselines,
        esmfold=esmfold,
        boltz2=boltz2,
        alphafold2=alphafold2,
        sequence_trunks=sequence_trunks,
    )
