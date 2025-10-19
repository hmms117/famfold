"""Stage 01 TemplatePrep contract tests."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from minifold.famfold import template_prep


def test_distogram_binning_is_symmetric_and_uses_default_edges():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    bins, bin_edges = template_prep.distogram_from_coords(coords)

    assert bins.shape == (3, 3)
    assert bins.dtype == np.uint16
    assert np.all(bins == bins.T)
    assert np.all(np.diag(bins) == template_prep.DISTOGRAM_DIAGONAL_SENTINEL)

    expected_idx = np.searchsorted(bin_edges, 5.0, side="right")
    assert bins[1, 2] == expected_idx


def test_prepare_template_extracts_sequences_and_distograms(tmp_path: Path):
    pdb_contents = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 80.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 80.00           C
ATOM      3  C   ALA A   1       1.458   0.000   0.000  1.00 80.00           C
ATOM      4  N   GLY A   2       0.000   0.000   3.000  1.00 75.00           N
ATOM      5  CA  GLY A   2       0.000   0.000   3.000  1.00 75.00           C
ATOM      6  C   GLY A   2       1.458   0.000   3.000  1.00 75.00           C
ATOM      7  N   SER A   3       0.000   4.000   0.000  1.00 85.00           N
ATOM      8  CA  SER A   3       0.000   4.000   0.000  1.00 85.00           C
ATOM      9  C   SER A   3       1.458   4.000   0.000  1.00 85.00           C
TER
END
"""
    pdb_path = tmp_path / "sample.pdb"
    pdb_path.write_text(pdb_contents)

    record = template_prep.prepare_template(
        pdb_path,
        template_id="SAMPLE_A",
        chain_id="A",
        source="AFDB",
        metadata={"date": "2024-01-01"},
    )

    assert record.template_id == "SAMPLE_A"
    assert record.L == 3
    assert record.seq == "AGS"
    assert record.plddt == pytest.approx([80.0, 75.0, 85.0])
    assert record.bin_edges == pytest.approx(template_prep.DEFAULT_BIN_EDGES.tolist())
    assert record.meta["source"] == "AFDB"
    assert record.meta["pdb_path"].endswith("sample.pdb")
    assert record.meta["date"] == "2024-01-01"
    assert record.meta["mean_plddt"] == pytest.approx(80.0)

    decoded = np.frombuffer(base64.b64decode(record.bins), dtype=np.uint16)
    decoded = decoded.reshape(record.L, record.L)
    assert np.all(decoded == decoded.T)
    assert np.all(np.diag(decoded) == template_prep.DISTOGRAM_DIAGONAL_SENTINEL)


def test_prepare_template_rejects_low_quality_templates(tmp_path: Path):
    pdb_contents = """\
ATOM      1  N   GLY A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  GLY A   1       0.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   GLY A   1       1.458   0.000   0.000  1.00 10.00           C
TER
END
"""
    pdb_path = tmp_path / "poor.pdb"
    pdb_path.write_text(pdb_contents)

    with pytest.raises(template_prep.TemplateQualityError):
        template_prep.prepare_template(pdb_path, chain_id="A", source="AFDB")
