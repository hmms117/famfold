"""Utilities for building template-driven distance priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from Bio import pairwise2

from minifold.utils import protein, residue_constants


@dataclass
class TemplateResidueMap:
    """Mapping from target residues to template coordinates."""

    name: str
    coordinates: torch.Tensor
    mask: torch.Tensor
    mapping: Dict[int, int]
    identity: float


def protein_to_sequence(structure: protein.Protein) -> str:
    """Convert a :class:`Protein` instance into its amino-acid sequence."""

    alphabet = residue_constants.restypes_with_x
    return "".join(alphabet[index] for index in structure.aatype)


def align_template_to_target(target: str, template: str) -> Tuple[float, Dict[int, int]]:
    """Return sequence identity and residue mapping between target and template."""

    if not target:
        raise ValueError("Target sequence must be non-empty.")
    if not template:
        raise ValueError("Template sequence must be non-empty.")

    alignment = pairwise2.align.globalms(
        target,
        template,
        1.0,
        0.0,
        -2.0,
        -0.5,
        penalize_end_gaps=False,
        one_alignment_only=True,
    )

    if not alignment:
        raise RuntimeError("Failed to align target sequence to template.")

    target_aligned, template_aligned = alignment[0][:2]
    mapping: Dict[int, int] = {}
    matches = 0
    target_index = -1
    template_index = -1

    for target_char, template_char in zip(target_aligned, template_aligned):
        if target_char != "-":
            target_index += 1
        if template_char != "-":
            template_index += 1

        if target_char == "-" or template_char == "-":
            continue

        mapping[target_index] = template_index
        if target_char == template_char:
            matches += 1

    identity = matches / len(target)
    return identity, mapping


def extract_ca_coordinates(structure: protein.Protein) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return CA coordinates and availability mask for a template structure."""

    ca_index = residue_constants.atom_order["CA"]
    coords = torch.tensor(structure.atom_positions[:, ca_index, :], dtype=torch.float32)
    mask = torch.tensor(structure.atom_mask[:, ca_index], dtype=torch.bool)
    return coords, mask


def build_distogram_from_templates(
    target_length: int,
    templates: Sequence[TemplateResidueMap],
    *,
    bins: int = 64,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
) -> torch.Tensor:
    """Average per-template distance maps into a Minifold-compatible distogram."""

    if target_length <= 0:
        raise ValueError("Target length must be positive.")

    dist_tensor = torch.zeros(target_length, target_length, bins, dtype=torch.float32)
    counts = torch.zeros(target_length, target_length, dtype=torch.float32)
    boundaries = torch.linspace(min_bin, max_bin, bins - 1)

    for template in templates:
        coords = template.coordinates
        mask = template.mask
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Template coordinates must have shape (N, 3).")

        usable: List[int] = []
        for target_index, template_index in template.mapping.items():
            if target_index < 0 or target_index >= target_length:
                continue
            if template_index < 0 or template_index >= coords.shape[0]:
                continue
            if template_index >= mask.shape[0] or not bool(mask[template_index]):
                continue
            usable.append(target_index)

        if len(usable) < 2:
            continue

        for i in usable:
            pos_i = coords[template.mapping[i]]
            for j in usable:
                pos_j = coords[template.mapping[j]]
                dist_value = torch.linalg.norm(pos_i - pos_j)
                bin_idx = torch.bucketize(dist_value, boundaries)
                dist_tensor[i, j, bin_idx] += 1.0
                counts[i, j] += 1.0

    mask = counts > 0
    if mask.any():
        dist_tensor[mask] /= counts[mask].unsqueeze(-1)

    return dist_tensor


def load_template(path: Path, *, chain_id: str | None = None) -> protein.Protein:
    """Parse a template structure from disk."""

    pdb_contents = Path(path).read_text(encoding="utf-8")
    return protein.from_pdb_string(pdb_contents, chain_id=chain_id)


def bucket_templates_by_identity(
    templates: Iterable[TemplateResidueMap],
    levels: Sequence[float],
    *,
    limit_per_level: int = 1,
) -> Dict[float, List[TemplateResidueMap]]:
    """Assign templates to descending identity brackets."""

    ordered_levels = sorted(levels, reverse=True)
    buckets: Dict[float, List[TemplateResidueMap]] = {level: [] for level in ordered_levels}
    upper_bound = 1.01

    candidates = sorted(templates, key=lambda item: item.identity, reverse=True)

    for level in ordered_levels:
        eligible = [
            candidate
            for candidate in candidates
            if level <= candidate.identity < upper_bound
        ]
        eligible.sort(key=lambda item: abs(item.identity - level))
        buckets[level] = eligible[:limit_per_level]
        upper_bound = level

    return buckets

