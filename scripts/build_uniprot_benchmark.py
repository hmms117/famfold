#!/usr/bin/env python3
"""Build benchmark table comparing AF2, ESMFold, and Minifold mean pLDDT."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import requests

WORK_ROOT = Path("benchmark_runs")
FASTA_DIR = WORK_ROOT / "fastas"
MINIFOLD_DIR = WORK_ROOT / "minifold"

AFDB_ROOT = Path("/z/pd/afdb")
MINIFOLD_CACHE = Path("/var/tmp/checkpoints")
MINIFOLD_12L = MINIFOLD_CACHE / "minifold_12L.ckpt"
MINIFOLD_48L = MINIFOLD_CACHE / "minifold_48L.ckpt"

PREDICT_PY = Path("predict.py")
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/"


@dataclass
class Target:
    uniprot_id: str
    cluster: str


TARGETS: Iterable[Target] = [
    Target("Q15759", "protein_kinase"),
    Target("P00533", "protein_kinase"),
    Target("P24941", "protein_kinase"),
    Target("P17612", "protein_kinase"),
    Target("Q13584", "protein_kinase"),
    Target("P04406", "nadp_binding_rossmann"),
    Target("P00338", "nadp_binding_rossmann"),
    Target("P40926", "nadp_binding_rossmann"),
    Target("P07327", "nadp_binding_rossmann"),
    Target("Q16853", "nadp_binding_rossmann"),
    Target("A4CGL6", "viral_rdrp_polymerase"),
    Target("P0DTD1", "viral_rdrp_polymerase"),
    Target("P26664", "viral_rdrp_polymerase"),
    Target("Q69159", "viral_rdrp_polymerase"),
    Target("O56264", "viral_rdrp_polymerase"),
    Target("P21964", "sam_dependent_methyltransferase"),
    Target("P26358", "sam_dependent_methyltransferase"),
    Target("Q99873", "sam_dependent_methyltransferase"),
    Target("Q9NV35", "sam_dependent_methyltransferase"),
    Target("Q9BZD3", "sam_dependent_methyltransferase"),
    Target("P04275", "lrr_vwa_like"),
    Target("Q5S007", "lrr_vwa_like"),
    Target("Q96P20", "lrr_vwa_like"),
    Target("Q6ZMQ8", "lrr_vwa_like"),
    Target("Q9NY07", "lrr_vwa_like"),
]


def ensure_dirs() -> None:
    FASTA_DIR.mkdir(parents=True, exist_ok=True)
    MINIFOLD_DIR.mkdir(parents=True, exist_ok=True)


_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


def afdb_path_for(uniprot_id: str, pattern: str) -> Optional[Path]:
    parts = [uniprot_id[i : i + 2] for i in range(0, min(len(uniprot_id), 6), 2)]
    while len(parts) < 3:
        parts.append("__")
    base_dir = AFDB_ROOT.joinpath(*parts[:3])
    if not base_dir.exists():
        return None
    candidates = sorted(base_dir.glob(pattern))
    if not candidates:
        return None

    def version_key(path: Path) -> int:
        name = path.stem
        if "_v" in name:
            try:
                return int(name.split("_v")[-1])
            except ValueError:
                return 0
        return 0

    return max(candidates, key=version_key)


def parse_afdb_sequence_and_plddt(uniprot_id: str) -> tuple[str, float]:
    sequence = fetch_uniprot_sequence(uniprot_id)
    conf_path = afdb_path_for(uniprot_id, f"AF-{uniprot_id}-F*-confidence*.json.gz")
    if conf_path is None:
        print(f"Warning: no local AF confidence file for {uniprot_id}, falling back to AFDB API")
        seq_api, plddt_api = fetch_af_info(uniprot_id)
        if not sequence:
            sequence = seq_api
        return sequence, plddt_api
    with gzip.open(conf_path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    scores = data.get("confidenceScore") or []
    if not scores:
        print(f"Warning: confidence file missing scores for {uniprot_id}, falling back to AFDB API")
        seq_api, plddt_api = fetch_af_info(uniprot_id)
        if not sequence:
            sequence = seq_api
        return sequence, plddt_api
    mean_plddt = float(np.mean(scores))
    if not sequence:
        seq_api, _ = fetch_af_info(uniprot_id)
        if seq_api:
            sequence = seq_api
    return sequence, mean_plddt


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url, timeout=60)
    if response.status_code == 404:
        print(f"Warning: UniProt sequence not found for {uniprot_id}")
        return ""
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    if not sequence:
        print(f"Warning: empty UniProt sequence for {uniprot_id} (status {response.status_code})")
        return ""
    return sequence


def fetch_af_info(uniprot_id: str) -> tuple[str, float]:
    try:
        response = requests.get(ALPHAFOLD_API + uniprot_id, timeout=60)
        response.raise_for_status()
        entries = response.json()
    except requests.RequestException as exc:
        print(f"Warning: failed to query AFDB API for {uniprot_id}: {exc}")
        return "", float("nan")
    if not entries:
        print(f"Warning: AFDB API returned no entries for {uniprot_id}")
        return "", float("nan")
    preferred = None
    for entry in entries:
        if entry.get("uniprotAccession") == uniprot_id:
            preferred = entry
            break
    if preferred is None:
        preferred = max(entries, key=lambda e: e.get("globalMetricValue", 0.0))
    sequence = preferred.get("uniprotSequence") or ""
    plddt = float(preferred.get("globalMetricValue", float("nan")))
    return sequence, plddt


def write_fasta(uniprot_id: str, sequence: str) -> Path:
    fasta_path = FASTA_DIR / f"{uniprot_id}.fasta"
    if not fasta_path.exists():
        with fasta_path.open("w", encoding="utf-8") as handle:
            handle.write(f">{uniprot_id}\n")
            for i in range(0, len(sequence), 80):
                handle.write(sequence[i : i + 80] + "\n")
    return fasta_path


def run_minifold(
    uniprot_id: str,
    fasta_path: Path,
    model_size: str,
    checkpoint: Path,
    seq_len: int,
) -> Optional[Path]:
    if seq_len > 1500:
        print(f"Warning: skipping Minifold {model_size} for {uniprot_id} (sequence length {seq_len} too large)")
        return None
    out_dir = MINIFOLD_DIR / f"{uniprot_id}_{model_size}"
    result_dir = out_dir / f"minifold_results_{uniprot_id}"
    pdb_path = result_dir / f"{uniprot_id}.pdb"
    if pdb_path.exists():
        return pdb_path

    token_per_batch = 512
    if seq_len > 800:
        token_per_batch = 256
    if seq_len > 1200:
        token_per_batch = 128

    args = [
        PREDICT_PY,
        str(fasta_path),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(MINIFOLD_CACHE),
        "--model_size",
        model_size,
        "--token_per_batch",
        str(token_per_batch),
        "--checkpoint",
        str(checkpoint),
    ]
    env = dict(**dict(os.environ))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run([".uv/bin/python", *args], check=False, env=env)
    if not pdb_path.exists():
        print(f"Warning: Minifold {model_size} failed for {uniprot_id}")
        return None
    return pdb_path


def mean_plddt_from_pdb(pdb_path: Path) -> float:
    values = []
    with pdb_path.open() as handle:
        for line in handle:
            if line.startswith("ATOM") and len(line) >= 66:
                try:
                    values.append(float(line[60:66]))
                except ValueError:
                    continue
    if not values:
        raise RuntimeError(f"No ATOM records found in {pdb_path}")
    return float(np.mean(values))


def main() -> None:
    ensure_dirs()
    rows = []
    for target in TARGETS:
        sequence, af_plddt = parse_afdb_sequence_and_plddt(target.uniprot_id)
        if not sequence:
            print(f"Skipping {target.uniprot_id}: sequence unavailable")
            continue
        fasta_path = write_fasta(target.uniprot_id, sequence)
        minifold12_pdb = run_minifold(target.uniprot_id, fasta_path, "12L", MINIFOLD_12L, len(sequence))
        if minifold12_pdb is None:
            minifold12_plddt = float("nan")
        else:
            minifold12_plddt = mean_plddt_from_pdb(minifold12_pdb)

        minifold48_pdb = run_minifold(target.uniprot_id, fasta_path, "48L", MINIFOLD_48L, len(sequence))
        if minifold48_pdb is None:
            minifold48_plddt = float("nan")
        else:
            minifold48_plddt = mean_plddt_from_pdb(minifold48_pdb)
        rows.append(
            {
                "uniprot_id": target.uniprot_id,
                "cluster": target.cluster,
                "length": len(sequence),
                "af2_plddt": af_plddt,
                "minifold12_plddt": minifold12_plddt,
                "minifold48_plddt": minifold48_plddt,
            }
        )

    rows.sort(key=lambda r: (r["cluster"], r["uniprot_id"]))
    table_lines = [
        "| Cluster | UniProt | Length | AF2 pLDDT | Minifold-12L pLDDT | Minifold-48L pLDDT |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['cluster']} | {row['uniprot_id']} | {row['length']} | "
            f"{row['af2_plddt']:.2f} | "
            f"{row['minifold12_plddt']:.2f} | {row['minifold48_plddt']:.2f} |"
        )

    output_md = WORK_ROOT / "uniprot_benchmark.md"
    output_md.write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print("\n".join(table_lines))


if __name__ == "__main__":
    main()
