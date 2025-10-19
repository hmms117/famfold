"""FamilyFold Stage 01 TemplatePrep utilities."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from minifold.utils import protein, residue_constants

__all__ = [
    "DEFAULT_BIN_EDGES",
    "DISTOGRAM_DIAGONAL_SENTINEL",
    "TemplateQualityError",
    "TemplateRecord",
    "distogram_from_coords",
    "prepare_template",
]

# Default distogram bin edges span 2 Å .. 25 Å inclusive with 64 buckets.
DEFAULT_BIN_EDGES: np.ndarray = np.linspace(2.0, 25.0, num=64, dtype=np.float32)

# ``uint16`` sentinel that marks the diagonal entries of a distogram matrix.
DISTOGRAM_DIAGONAL_SENTINEL: np.uint16 = np.uint16(2**16 - 1)


class TemplateQualityError(RuntimeError):
    """Raised when a template fails TemplatePrep quality checks."""


@dataclass(frozen=True)
class TemplateRecord:
    """Structured representation of a prepared template."""

    template_id: str
    L: int
    seq: str
    plddt: list[float]
    bins: str
    bin_edges: list[float]
    meta: dict[str, Any]


def distogram_from_coords(
    coords: np.ndarray,
    *,
    bin_edges: Optional[Sequence[float]] = None,
    diagonal_value: np.uint16 = DISTOGRAM_DIAGONAL_SENTINEL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a discretised distogram from Cα coordinates."""

    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[-1] != 3:
        raise ValueError("Coordinates must have shape (N, 3).")

    if bin_edges is None:
        edges = DEFAULT_BIN_EDGES
    else:
        edges = np.asarray(bin_edges, dtype=np.float32)
        if edges.ndim != 1:
            raise ValueError("bin_edges must be a 1-D sequence of floats.")
        if edges.size == 0:
            raise ValueError("bin_edges must contain at least one boundary.")

    diff = coords[:, None, :] - coords[None, :, :]
    distances = np.linalg.norm(diff, axis=-1)

    bin_ids = np.searchsorted(edges, distances, side="right").astype(np.uint16)
    if edges.size > 0:
        np.clip(bin_ids, 0, edges.size - 1, out=bin_ids)

    np.fill_diagonal(bin_ids, diagonal_value)
    return bin_ids, edges


def _extract_sequence(prot: protein.Protein) -> str:
    return "".join(residue_constants.restype_order_with_x_inverse[aa] for aa in prot.aatype)


def _extract_plddt(prot: protein.Protein) -> np.ndarray:
    ca_idx = residue_constants.atom_order["CA"]
    atom_mask = prot.atom_mask
    b_factors = prot.b_factors

    ca_mask = atom_mask[:, ca_idx] > 0.5
    plddt = np.zeros(prot.atom_mask.shape[0], dtype=np.float32)

    if np.any(ca_mask):
        plddt[ca_mask] = b_factors[ca_mask, ca_idx]

    missing = ~ca_mask
    if np.any(missing):
        masked = atom_mask[missing]
        weights = masked.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            residue_scores = np.where(
                weights > 0,
                (b_factors[missing] * masked).sum(axis=1) / weights,
                np.nan,
            )
        plddt[missing] = residue_scores

    if not np.all(np.isfinite(plddt)):
        raise TemplateQualityError("Template contains residues without pLDDT annotations.")

    return plddt


def _mean_plddt(plddt: np.ndarray) -> float:
    return float(np.mean(plddt, dtype=np.float64))


def _derive_template_id(path: Path, chain_id: Optional[str]) -> str:
    stem = path.name
    suffix_chain = Path(stem)
    while suffix_chain.suffix:
        suffix_chain = suffix_chain.with_suffix("")
    base = suffix_chain.name.upper()
    if chain_id:
        return f"{base}_{chain_id.upper()}"
    return base


def _build_meta(
    *,
    base_meta: Optional[Mapping[str, Any]],
    pdb_path: Path,
    source: str,
    chain_id: Optional[str],
    plddt: np.ndarray,
) -> dict[str, Any]:
    meta: MutableMapping[str, Any] = {
        "source": source,
        "pdb_path": str(pdb_path),
        "mean_plddt": _mean_plddt(plddt),
        "p5_plddt": float(np.percentile(plddt, 5)),
        "p95_plddt": float(np.percentile(plddt, 95)),
    }
    if chain_id is not None:
        meta["chain_id"] = chain_id
    if base_meta:
        meta.update(base_meta)
    return dict(meta)


def prepare_template(
    pdb_path: str | Path,
    *,
    template_id: Optional[str] = None,
    chain_id: Optional[str] = None,
    source: str = "AFDB",
    metadata: Optional[Mapping[str, Any]] = None,
    mean_plddt_threshold: float = 70.0,
    bin_edges: Optional[Sequence[float]] = None,
) -> TemplateRecord:
    """Prepare a single template structure for downstream FamilyFold stages."""

    path = Path(pdb_path)
    pdb_contents = path.read_text(encoding="utf-8")
    prot = protein.from_pdb_string(pdb_contents, chain_id=chain_id)

    seq = _extract_sequence(prot)
    if not seq:
        raise TemplateQualityError("Template contains no residues after parsing.")

    plddt = _extract_plddt(prot)
    mean_plddt = _mean_plddt(plddt)
    if mean_plddt < mean_plddt_threshold:
        raise TemplateQualityError(
            f"Mean pLDDT {mean_plddt:.1f} below threshold {mean_plddt_threshold:.1f}."
        )

    ca_idx = residue_constants.atom_order["CA"]
    ca_mask = prot.atom_mask[:, ca_idx] > 0.5
    if not np.all(ca_mask):
        missing = np.where(~ca_mask)[0]
        raise TemplateQualityError(
            f"Template is missing Cα coordinates for residues: {missing.tolist()}"
        )
    ca_coords = prot.atom_positions[:, ca_idx]

    bins, final_edges = distogram_from_coords(ca_coords, bin_edges=bin_edges)
    encoded_bins = base64.b64encode(bins.tobytes()).decode("ascii")

    final_template_id = template_id or _derive_template_id(path, chain_id)
    meta = _build_meta(
        base_meta=metadata,
        pdb_path=path,
        source=source,
        chain_id=chain_id,
        plddt=plddt,
    )

    return TemplateRecord(
        template_id=final_template_id,
        L=len(seq),
        seq=seq,
        plddt=plddt.astype(float).tolist(),
        bins=encoded_bins,
        bin_edges=final_edges.astype(float).tolist(),
        meta=meta,
    )
