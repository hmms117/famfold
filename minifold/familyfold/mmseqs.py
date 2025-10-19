"""Utilities for working with MMseqs2 outputs in FamilyFold pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

_SUPPORTED_EXTENSIONS: tuple[str, ...] = (".cif", ".cif.gz", ".mmcif", ".pdb")


@dataclass(frozen=True)
class StructureHit:
    """Information about a structure retrieved from a homolog cluster."""

    identifier: str
    path: Path


def _structure_path_for(identifier: str, root: Path) -> Path | None:
    for extension in _SUPPORTED_EXTENSIONS:
        candidate = root / f"{identifier}{extension}"
        if candidate.exists():
            return candidate
    return None


def collect_nearest_structures(
    clusters: Mapping[str, Sequence[str]],
    structures_dir: Path | None,
    *,
    max_templates: int,
    include_self: bool = False,
) -> Mapping[str, list[Path]]:
    """Map each sequence in ``clusters`` to available structure files."""

    if max_templates < 0:
        raise ValueError("'max_templates' must be non-negative")

    if not clusters:
        return {}

    if structures_dir is None:
        return {member: [] for members in clusters.values() for member in members}

    per_sequence: MutableMapping[str, list[Path]] = {}

    for members in clusters.values():
        if not members:
            continue

        available_hits: list[StructureHit] = []
        for identifier in members:
            structure_path = _structure_path_for(identifier, structures_dir)
            if structure_path is None:
                continue
            available_hits.append(StructureHit(identifier=identifier, path=structure_path))

        for query in members:
            if include_self:
                hits_list = []
                self_hit = next((hit for hit in available_hits if hit.identifier == query), None)
                if self_hit is not None:
                    hits_list.append(self_hit)
                hits_list.extend(hit for hit in available_hits if hit.identifier != query)
                hits: Iterable[StructureHit] = hits_list
            else:
                hits = (hit for hit in available_hits if hit.identifier != query)

            collected: list[Path] = []
            for hit in hits:
                collected.append(hit.path)
                if len(collected) >= max_templates:
                    break

            per_sequence[query] = collected

        for identifier in members:
            per_sequence.setdefault(identifier, [])

    return per_sequence
