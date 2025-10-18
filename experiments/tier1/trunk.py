"""Sequence trunk registry for Tier 1 inference pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

from minifold.utils.saesm import SAESM_FAST_CHECKPOINT


@dataclass(frozen=True)
class NormalizationConfig:
    """Embedding normalisation parameters for a trunk."""

    mean: float
    std: float


@dataclass(frozen=True)
class TrunkSpec:
    """Describes a sequence embedding trunk available to Tier 1 pipelines."""

    name: str
    checkpoint: str
    normalization: NormalizationConfig
    description: str

    def as_dict(self) -> Mapping[str, object]:
        """Return a JSON-serialisable representation of the spec."""

        return {
            "name": self.name,
            "checkpoint": self.checkpoint,
            "normalization": {
                "mean": self.normalization.mean,
                "std": self.normalization.std,
            },
            "description": self.description,
        }


_ISM_FAST_CHECKPOINT = "facebook/esm2_t36_3B_UR50D"
_ISM_NORMALIZATION = NormalizationConfig(mean=0.0, std=1.0)
_SAESM_FAST_NORMALIZATION = NormalizationConfig(mean=0.0, std=1.0)


def _register_fast_trunk(
    registry: Dict[str, TrunkSpec],
    name: str,
    checkpoint: str,
    normalization: NormalizationConfig,
    description: str,
    aliases: Iterable[str],
) -> TrunkSpec:
    spec = TrunkSpec(
        name=name,
        checkpoint=checkpoint,
        normalization=normalization,
        description=description,
    )
    for alias in {name, *aliases}:
        registry[alias.lower()] = spec
    return spec


_TRUNK_SPECS: Dict[str, TrunkSpec] = {}

_register_fast_trunk(
    _TRUNK_SPECS,
    name="ism_fast",
    checkpoint=_ISM_FAST_CHECKPOINT,
    normalization=_ISM_NORMALIZATION,
    description=(
        "Flash-attention accelerated ESM-2 trunk used by the ISM fast path."
        " Provides layer-normalised embeddings compatible with Tier 1 feature"
        " builders."
    ),
    aliases=("ism-fast", "ism/fast", "faesm"),
)

_register_fast_trunk(
    _TRUNK_SPECS,
    name="saesm2_fast",
    checkpoint=SAESM_FAST_CHECKPOINT,
    normalization=_SAESM_FAST_NORMALIZATION,
    description=(
        "SaESM2 35M checkpoint distilled for rapid retrieval. Mirrors the ISM"
        " fast path normalisation so caches remain interchangeable."
    ),
    aliases=("saesm_fast", "saesm2-fast", "saesm2/fast"),
)


def resolve_trunk_spec(name: str) -> TrunkSpec:
    """Resolve a trunk identifier to a :class:`TrunkSpec`."""

    key = name.strip().lower()
    if key not in _TRUNK_SPECS:
        available = ", ".join(sorted({spec.name for spec in _TRUNK_SPECS.values()}))
        raise KeyError(f"Unknown trunk '{name}'. Available: {available}.")
    return _TRUNK_SPECS[key]


def list_trunk_specs() -> Iterable[TrunkSpec]:
    """Return distinct registered trunk specs."""

    seen = set()
    for spec in _TRUNK_SPECS.values():
        if spec.name in seen:
            continue
        seen.add(spec.name)
        yield spec


__all__ = [
    "NormalizationConfig",
    "TrunkSpec",
    "list_trunk_specs",
    "resolve_trunk_spec",
]
