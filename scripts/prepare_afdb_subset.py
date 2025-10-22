#!/usr/bin/env python3
"""Prepare a filtered AlphaFold DB subset and sample clusters.

This helper downloads (or reuses) the AlphaFold DB ``cluster_lookup.tsv`` and
``sequences.fasta`` files, filters them to a user-provided list of AlphaFold or
UniProt identifiers, optionally downloads structure files, and finally invokes
``sample_mmseqs_clusters.py`` in AFDB mode to sample random clusters.

Run ``python scripts/prepare_afdb_subset.py --help`` for usage details.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set
from urllib.parse import urlparse

import requests


_HF_CACHE_ROOT = "/var/tmp/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE_ROOT


DEFAULT_DOWNLOAD_DIR = Path("data/afdb_cache")
DEMO_DATA_DIR = Path("data/afdb_demo")
SAMPLE_SCRIPT = Path("scripts/sample_mmseqs_clusters.py")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the sampled cluster subset will be written.",
    )
    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="Path to a newline-delimited list of AlphaFold or UniProt identifiers.",
    )
    parser.add_argument(
        "--cluster-lookup",
        default=None,
        help="Path or URL to the AlphaFold DB cluster lookup TSV (or TSV.GZ).",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help="Path or URL to the AlphaFold DB sequences FASTA file (or FASTA.GZ).",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help="Directory used to cache downloaded resources (default: %(default)s).",
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help="Directory that will receive downloaded structure files (and be reused when present).",
    )
    parser.add_argument(
        "--structure-url-template",
        default="https://alphafold.ebi.ac.uk/files/{entry}-model_v4.pdb",
        help="URL template used to download structure files when '--structures-dir' is provided.",
    )
    parser.add_argument(
        "--structure-format",
        default="{entry}.pdb",
        help="Filename template for downloaded structures (default: %(default)s).",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=5,
        help="Number of clusters to sample (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for deterministic sampling.",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep filtered FASTA/TSV files instead of deleting the temporary directory.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run against the bundled demo dataset instead of downloading AFDB files.",
    )
    return parser.parse_args(argv)


def _is_url(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    return parsed.scheme in {"http", "https"}


def download_if_needed(source: str, destination_dir: Path) -> Path:
    path = Path(source)
    if path.exists():
        return path

    if not _is_url(source):
        raise FileNotFoundError(f"Could not find '{source}' and no supported download scheme was detected.")

    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(urlparse(source).path)
    if not filename:
        raise ValueError(f"Could not derive a filename from URL: {source}")
    destination = destination_dir / filename
    if destination.exists():
        return destination

    with requests.get(source, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle)
    return destination


def ensure_uncompressed(path: Path, temp_dir: Path) -> Path:
    if path.suffix != ".gz":
        return path
    output = temp_dir / path.with_suffix("").name
    if output.exists():
        return output
    with gzip.open(path, "rb") as compressed, output.open("wb") as handle:
        shutil.copyfileobj(compressed, handle)
    return output


def _normalize_afdb_identifier(identifier: str) -> str:
    identifier = identifier.strip()
    if identifier.startswith("AF-"):
        parts = identifier.split("-")
        if len(parts) >= 3:
            return parts[1]
    return identifier


def load_identifier_list(path: Path) -> Set[str]:
    allowed: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            identifier = raw_line.strip()
            if not identifier or identifier.startswith("#"):
                continue
            allowed.add(identifier)
            allowed.add(_normalize_afdb_identifier(identifier))
    if not allowed:
        raise ValueError(f"No identifiers were loaded from {path}")
    return allowed


def filter_cluster_lookup(source: Path, allowed: Set[str], destination: Path) -> List[str]:
    retained_entries: List[str] = []
    with source.open("r", encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        for raw_line in input_handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            cluster_id, entry_id = line.split("\t", 1)
            normalized = _normalize_afdb_identifier(entry_id)
            if entry_id not in allowed and normalized not in allowed:
                continue
            output_handle.write(f"{cluster_id}\t{entry_id}\n")
            retained_entries.append(entry_id)
    if not retained_entries:
        raise ValueError("No entries matched the provided identifier list when filtering the cluster lookup.")
    return retained_entries


def filter_fasta(source: Path, allowed: Set[str], destination: Path) -> None:
    with source.open("r", encoding="utf-8") as input_handle, destination.open("w", encoding="utf-8") as output_handle:
        current_id: Optional[str] = None
        buffer: List[str] = []

        def _flush() -> None:
            if current_id is None:
                return
            normalized = _normalize_afdb_identifier(current_id)
            if current_id in allowed or normalized in allowed:
                output_handle.write(f">{current_id}\n")
                output_handle.write("".join(buffer))

        for raw_line in input_handle:
            if raw_line.startswith(">"):
                _flush()
                buffer = []
                current_id = raw_line[1:].split()[0]
                continue
            buffer.append(raw_line)
        _flush()


def download_structures(entries: Iterable[str], template: str, fmt: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        filename = fmt.format(entry=entry)
        output_path = destination / filename
        if output_path.exists():
            continue
        url = template.format(entry=entry)
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                shutil.copyfileobj(response.raw, handle)
        tmp_path.replace(output_path)


def run_sampling(
    fasta: Path,
    cluster_lookup: Path,
    ids: Path,
    output_dir: Path,
    structures_dir: Optional[Path],
    clusters: int,
    seed: Optional[int],
) -> None:
    if not SAMPLE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Could not find '{SAMPLE_SCRIPT}'. Run this helper from the repository root."
        )

    command = [
        sys.executable,
        str(SAMPLE_SCRIPT),
        str(fasta),
        str(output_dir),
        "--afdb-cluster-lookup",
        str(cluster_lookup),
        "--afdb-include-ids",
        str(ids),
        "--clusters",
        str(clusters),
    ]
    if structures_dir is not None:
        command.extend(["--structures-dir", str(structures_dir)])
    if seed is not None:
        command.extend(["--seed", str(seed)])

    subprocess.run(command, check=True)


def prepare_demo(args: argparse.Namespace) -> None:
    ids = DEMO_DATA_DIR / "gh_demo_ids.txt"
    lookup = DEMO_DATA_DIR / "gh_demo_cluster_lookup.tsv"
    fasta = DEMO_DATA_DIR / "gh_demo_sequences.fasta"
    structures = DEMO_DATA_DIR / "structures"
    if not ids.exists() or not lookup.exists() or not fasta.exists():
        raise FileNotFoundError("Demo dataset is missing. Re-run the repository setup script.")

    run_sampling(
        fasta=fasta,
        cluster_lookup=lookup,
        ids=ids,
        output_dir=args.output_dir,
        structures_dir=structures,
        clusters=args.clusters,
        seed=args.seed,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    if args.demo:
        prepare_demo(args)
        return

    if args.ids is None or args.cluster_lookup is None or args.sequences is None:
        raise SystemExit("'--ids', '--cluster-lookup', and '--sequences' are required unless '--demo' is specified.")

    ids_path = args.ids
    if not ids_path.exists():
        raise FileNotFoundError(f"Could not find identifier list: {ids_path}")

    allowed_ids = load_identifier_list(ids_path)

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        download_dir = args.download_dir

        raw_lookup = download_if_needed(args.cluster_lookup, download_dir)
        lookup_path = ensure_uncompressed(raw_lookup, tmpdir)

        raw_sequences = download_if_needed(args.sequences, download_dir)
        fasta_path = ensure_uncompressed(raw_sequences, tmpdir)

        filtered_lookup = tmpdir / "filtered_cluster_lookup.tsv"
        retained_entries = filter_cluster_lookup(lookup_path, allowed_ids, filtered_lookup)

        filtered_fasta = tmpdir / "filtered_sequences.fasta"
        filter_fasta(fasta_path, allowed_ids, filtered_fasta)

        structures_dir = args.structures_dir
        if structures_dir is not None:
            download_structures(retained_entries, args.structure_url_template, args.structure_format, structures_dir)

        if args.keep_intermediates:
            download_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(filtered_lookup, download_dir / filtered_lookup.name)
            shutil.copy(filtered_fasta, download_dir / filtered_fasta.name)
            filtered_lookup = download_dir / filtered_lookup.name
            filtered_fasta = download_dir / filtered_fasta.name

        run_sampling(
            fasta=filtered_fasta,
            cluster_lookup=filtered_lookup,
            ids=ids_path,
            output_dir=args.output_dir,
            structures_dir=structures_dir,
            clusters=args.clusters,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
