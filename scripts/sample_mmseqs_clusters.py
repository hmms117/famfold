#!/usr/bin/env python3
"""Sample random homolog clusters at a specified identity threshold using MMseqs2."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set


@dataclass
class ClusterSelection:
    """Information about a selected MMseqs2 cluster."""

    representative: str
    members: List[str]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fasta",
        type=Path,
        help="Path to the FASTA file containing the sequences to cluster.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where the sampled clusters will be written.",
    )
    parser.add_argument(
        "--mmseqs",
        default="mmseqs",
        help="Name or path of the MMseqs2 binary (default: %(default)s).",
    )
    parser.add_argument(
        "--clusters-tsv",
        type=Path,
        default=None,
        help=(
            "Optional pre-computed cluster membership TSV produced by 'mmseqs createtsv'. "
            "When provided, MMseqs2 will not be invoked."
        ),
    )
    parser.add_argument(
        "--afdb-cluster-lookup",
        type=Path,
        default=None,
        help=(
            "AlphaFold DB cluster lookup TSV (e.g. 'cluster_lookup.tsv' from the AFDB clustering "
            "release). The file is expected to contain two tab-separated columns: a cluster ID "
            "and the corresponding AlphaFold entry ID. When provided, the script will sample "
            "clusters from this file instead of running MMseqs2."
        ),
    )
    parser.add_argument(
        "--afdb-include-ids",
        type=Path,
        default=None,
        help=(
            "Optional newline-delimited list of AlphaFold or UniProt identifiers to keep when "
            "reading '--afdb-cluster-lookup'. Clusters that do not contain any of the retained "
            "identifiers will be discarded."
        ),
    )
    parser.add_argument(
        "--structures-dir",
        type=Path,
        default=None,
        help=(
            "Directory that contains structure files named after sequence identifiers. "
            "Files with extensions .cif, .mmcif, .pdb or .cif.gz will be copied to the sampled cluster directories if present."
        ),
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.85,
        help="Minimum sequence identity to use when clustering (default: %(default)s).",
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.8,
        help="Minimum coverage to use with '--cov-mode 1' when clustering (default: %(default)s).",
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
        "--threads",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of CPU threads to pass to MMseqs2 (default: %(default)s).",
    )
    parser.add_argument(
        "--keep-mmseqs-output",
        action="store_true",
        help="Keep the intermediate MMseqs2 databases under the output directory for inspection.",
    )
    return parser.parse_args(argv)


def check_mmseqs_available(binary: str) -> None:
    if shutil.which(binary) is None:
        raise FileNotFoundError(
            f"Could not find an executable named '{binary}'. Please install MMseqs2 or provide the correct path via --mmseqs."
        )


def run_mmseqs(
    mmseqs: str,
    fasta: Path,
    identity: float,
    coverage: float,
    threads: int,
    workspace: Path,
) -> Path:
    createdb_path = workspace / "input_db"
    cluster_db_path = workspace / "cluster_db"
    tmp_path = workspace / "tmp"
    tmp_path.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [mmseqs, "createdb", str(fasta), str(createdb_path)],
        check=True,
    )

    subprocess.run(
        [
            mmseqs,
            "cluster",
            str(createdb_path),
            str(cluster_db_path),
            str(tmp_path),
            "--min-seq-id",
            str(identity),
            "--cov-mode",
            "1",
            "-c",
            str(coverage),
            "--threads",
            str(threads),
        ],
        check=True,
    )

    tsv_path = workspace / "clusters.tsv"
    subprocess.run(
        [
            mmseqs,
            "createtsv",
            str(createdb_path),
            str(createdb_path),
            str(cluster_db_path),
            str(tsv_path),
            "--threads",
            str(threads),
        ],
        check=True,
    )

    return tsv_path


def load_cluster_memberships(tsv_path: Path) -> Mapping[str, List[str]]:
    clusters: MutableMapping[str, List[str]] = {}
    with tsv_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 2:
                raise ValueError(f"Malformed line in {tsv_path}: {line!r}")
            representative, member = columns[0], columns[1]
            clusters.setdefault(representative, []).append(member)
    return clusters


def load_sequences(fasta_path: Path) -> Mapping[str, str]:
    sequences: Dict[str, str] = {}
    current_id: Optional[str] = None
    current_sequence: List[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(current_sequence)
                current_id = line[1:].split()[0]
                current_sequence = []
            else:
                if current_id is None:
                    raise ValueError(
                        f"Encountered sequence data before any FASTA header in {fasta_path}."
                    )
                current_sequence.append(line)
        if current_id is not None:
            sequences[current_id] = "".join(current_sequence)

    return sequences


def _normalize_afdb_identifier(identifier: str) -> str:
    """Return a canonical key for an AlphaFold DB entry.

    AlphaFold DB entries typically take the form ``AF-<UniProt>-F1``. When filtering against
    user-supplied UniProt lists we want to treat both ``AF-P12345-F1`` and ``P12345`` as the
    same entry. This helper keeps the original identifier if we cannot confidently extract the
    UniProt accession.
    """

    identifier = identifier.strip()
    if identifier.startswith("AF-"):
        # AF-<uniprot>-F1 or AF-<uniprot>-F1-DOM is common; keep the accession in the middle
        parts = identifier.split("-")
        if len(parts) >= 3:
            return parts[1]
    return identifier


def _load_identifier_list(path: Path) -> Set[str]:
    allowed: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            identifier = raw_line.strip()
            if not identifier or identifier.startswith("#"):
                continue
            allowed.add(identifier)
            allowed.add(_normalize_afdb_identifier(identifier))
    return allowed


def load_afdb_cluster_lookup(
    lookup_path: Path, include_ids: Optional[Path]
) -> Mapping[str, List[str]]:
    """Load AFDB clusters from the official ``cluster_lookup.tsv`` format.

    The lookup file contains one ``cluster_id``/``entry_id`` pair per line. When ``include_ids``
    is provided, only entries that match one of the listed AlphaFold identifiers or UniProt
    accessions are retained. Clusters with no remaining members are dropped.
    """

    allowed_ids: Optional[Set[str]] = None
    if include_ids is not None:
        allowed_ids = _load_identifier_list(include_ids)

    clusters: MutableMapping[str, List[str]] = {}
    with lookup_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                cluster_id, entry_id = line.split("\t", 1)
            except ValueError as error:
                raise ValueError(
                    f"Malformed line in {lookup_path}: {line!r}. Expected '<cluster>\t<entry>'."
                ) from error

            if allowed_ids is not None:
                normalized = _normalize_afdb_identifier(entry_id)
                if entry_id not in allowed_ids and normalized not in allowed_ids:
                    continue

            clusters.setdefault(cluster_id, []).append(entry_id)

    if allowed_ids is not None:
        clusters = {cid: members for cid, members in clusters.items() if members}

    if not clusters:
        raise ValueError(
            "No AFDB clusters were retained. Check that '--afdb-include-ids' overlaps with the "
            "provided lookup file."
        )

    return clusters


def find_structure(structure_root: Optional[Path], identifier: str) -> Optional[Path]:
    if structure_root is None:
        return None
    candidates = [
        structure_root / f"{identifier}.cif",
        structure_root / f"{identifier}.cif.gz",
        structure_root / f"{identifier}.mmcif",
        structure_root / f"{identifier}.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def select_random_clusters(
    clusters: Mapping[str, List[str]],
    count: int,
    seed: Optional[int],
) -> List[ClusterSelection]:
    if not clusters:
        raise ValueError("No clusters were found in the provided data source.")

    rng = random.Random(seed)
    representatives = list(clusters.keys())
    if len(representatives) < count:
        raise ValueError(
            f"Requested {count} clusters but only {len(representatives)} are available."
        )

    selected: List[ClusterSelection] = []
    for representative in rng.sample(representatives, count):
        members = clusters[representative]
        selected.append(ClusterSelection(representative=representative, members=members))
    return selected


def write_cluster_outputs(
    selections: Iterable[ClusterSelection],
    sequences: Mapping[str, str],
    structures_dir: Optional[Path],
    output_dir: Path,
) -> List[Mapping[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Mapping[str, object]] = []

    for index, selection in enumerate(selections, start=1):
        cluster_dir = output_dir / f"cluster_{index:02d}_{selection.representative}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = cluster_dir / "sequences.fasta"
        with fasta_path.open("w", encoding="utf-8") as fasta_handle:
            for member in selection.members:
                sequence = sequences.get(member)
                if sequence is None:
                    raise KeyError(
                        f"Sequence '{member}' referenced in cluster '{selection.representative}' was not found in the FASTA file."
                    )
                fasta_handle.write(f">{member}\n")
                for start in range(0, len(sequence), 80):
                    fasta_handle.write(sequence[start : start + 80] + "\n")

        copied_structures: List[str] = []
        for member in selection.members:
            structure_path = find_structure(structures_dir, member)
            if structure_path is None:
                continue
            destination = cluster_dir / structure_path.name
            shutil.copy2(structure_path, destination)
            copied_structures.append(destination.name)

        manifest.append(
            {
                "representative": selection.representative,
                "members": selection.members,
                "fasta": str(fasta_path.relative_to(output_dir)),
                "structures": copied_structures,
            }
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as manifest_handle:
        json.dump(manifest, manifest_handle, indent=2)

    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)

    sequences = load_sequences(args.fasta)

    if args.afdb_cluster_lookup is not None:
        if args.clusters_tsv is not None:
            raise ValueError(
                "Please specify only one of '--clusters-tsv' or '--afdb-cluster-lookup'."
            )
        clusters = load_afdb_cluster_lookup(args.afdb_cluster_lookup, args.afdb_include_ids)
        selections = select_random_clusters(clusters, args.clusters, args.seed)
        write_cluster_outputs(selections, sequences, args.structures_dir, args.output_dir)
    else:
        if args.clusters_tsv is None:
            check_mmseqs_available(args.mmseqs)

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            if args.clusters_tsv is None:
                try:
                    cluster_tsv_path = run_mmseqs(
                        args.mmseqs,
                        args.fasta,
                        args.identity,
                        args.coverage,
                        args.threads,
                        workspace,
                    )
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(
                        "MMseqs2 failed; check the output above for diagnostic information."
                    ) from error
            else:
                cluster_tsv_path = args.clusters_tsv

            clusters = load_cluster_memberships(cluster_tsv_path)
            selections = select_random_clusters(clusters, args.clusters, args.seed)
            write_cluster_outputs(selections, sequences, args.structures_dir, args.output_dir)

            if args.keep_mmseqs_output and args.clusters_tsv is None:
                persistent_workspace = args.output_dir / "mmseqs_workdir"
                persistent_workspace.mkdir(parents=True, exist_ok=True)
                for item in workspace.iterdir():
                    target = persistent_workspace / item.name
                    if item.is_dir():
                        shutil.copytree(item, target, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, target)

    print(f"Wrote sampled clusters to {args.output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
