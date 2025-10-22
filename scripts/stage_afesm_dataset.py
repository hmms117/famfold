#!/usr/bin/env python3
"""Stage AFESM representative structures into a MiniFold-friendly layout.

This script reads the AFESM clustering table, extracts the representative
UniProt IDs together with their MGYP (ESMFold) members, and mirrors the
corresponding structures into an output directory tree:

  <out>/cifs/<last3>/<AF-*.cif.gz>
  <out>/pdbs/<last3>/<MGYP*.pdb>

By default the script creates symlinks to the original files to avoid
duplicating ~4M structures. If the filesystem does not allow symlinks,
it will transparently fall back to copying.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence
import tempfile

from tqdm import tqdm

@dataclass(frozen=True)
class Task:
    source: Path
    destination: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clusters",
        type=Path,
        required=True,
        help="Path to 1-AFESMClusters-repId_memId_cluFlag_taxId_biomeId.tsv",
    )
    parser.add_argument(
        "--rep-list",
        type=Path,
        default=Path("/z/pd/afesm/afesm_afdb_reps.tsv"),
        help="File containing one representative UniProt ID per line (default: /z/pd/afesm/afesm_afdb_reps.tsv).",
    )
    parser.add_argument(
        "--afdb-root",
        type=Path,
        default=Path("/z/pd/afdb"),
        help="Root directory of the AFDB mmCIF hierarchy.",
    )
    parser.add_argument(
        "--esmf-root",
        type=Path,
        default=Path("/ceph/cluster/bioinf/structure_models/esmatlas"),
        help="Root directory of the ESMFold atlas PDB hierarchy.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/var/tmp/famfold/afesm"),
        help="Destination root. 'cifs/' and 'pdbs/' subdirectories will be created here.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of worker threads for staging files.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional limit for debugging (number of rows to process).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100_000,
        help="Emit a progress message every N rows/tasks (default: 100k).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any representative or MGYP structure is missing.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Force copying instead of creating symlinks.",
    )
    parser.add_argument(
        "--convert-pdb",
        action="store_true",
        help="After staging, convert all PDBs to CIFs alongside the representatives (uses gemmi/pdb2cif if available).",
    )
    return parser.parse_args()


def representative_cif_path(uniprot_id: str, root: Path) -> Path | None:
    level1 = uniprot_id[:2]
    level2 = uniprot_id[2:4] if len(uniprot_id) >= 4 else uniprot_id[:2]
    level3 = uniprot_id[4:6] if len(uniprot_id) >= 6 else uniprot_id[2:4]
    base = root / level1 / level2 / level3

    for version in ("v4", "v3", "v2", "v1"):
        candidate = base / f"AF-{uniprot_id}-F1-model_{version}.cif.gz"
        if candidate.exists():
            return candidate

    candidate = base / f"AF-{uniprot_id}-F1-model.cif.gz"
    if candidate.exists():
        return candidate

    for pattern in (
        f"AF-{uniprot_id}-F*-model_*.cif.gz",
        f"AF-{uniprot_id}-*.cif.gz",
    ):
        matches = list(base.glob(pattern))
        if matches:
            return matches[0]
    return None


def mgyp_pdb_path(mgyp_id: str, root: Path) -> Path | None:
    suffix = mgyp_id[-3:]
    candidate = root / suffix / f"{mgyp_id}.pdb"
    return candidate if candidate.exists() else None


def ensure_bucket(base: Path, identifier: str) -> Path:
    key = (identifier[-3:] if len(identifier) >= 3 else identifier.zfill(3)).upper()
    bucket = base / key
    bucket.mkdir(parents=True, exist_ok=True)
    return bucket


def build_rep_tasks(
    rep_list: Path, afdb_root: Path, out_root: Path, limit: int | None
) -> tuple[list[Task], list[str]]:
    tasks: list[Task] = []
    missing: list[str] = []
    out_root.mkdir(parents=True, exist_ok=True)

    with rep_list.open("r") as fh:
        for idx, line in enumerate(tqdm(fh, desc="Preparing AFDB reps"), 1):
            rep = line.strip()
            if not rep:
                continue
            if rep.startswith("MGYP"):
                continue
            src = representative_cif_path(rep, afdb_root)
            if src is None:
                missing.append(rep)
                continue
            bucket = ensure_bucket(out_root, rep)
            tasks.append(Task(source=src, destination=bucket / src.name))
            if limit is not None and len(tasks) >= limit:
                break

    return tasks, missing


def build_mgyp_tasks(
    clusters_path: Path,
    esmf_root: Path,
    out_root: Path,
    limit: int | None,
    log_interval: int,
) -> tuple[list[Task], list[str], int]:
    tasks: dict[str, Task] = {}
    missing: list[str] = []
    seen_mgyp: set[str] = set()
    out_root.mkdir(parents=True, exist_ok=True)

    with tqdm(clusters_path.open("r"), desc="Parsing MGYP members") as fh:
        for idx, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            member = parts[1]
            if not member.startswith("MGYP") or member in seen_mgyp:
                if limit is not None and idx >= limit:
                    break
                continue
            seen_mgyp.add(member)
            src = mgyp_pdb_path(member, esmf_root)
            if src is None:
                missing.append(member)
            else:
                bucket = ensure_bucket(out_root, member)
                tasks[member] = Task(source=src, destination=bucket / src.name)

            if log_interval and idx % log_interval == 0:
                tqdm.write(
                    f"Parsed {idx:,} rows | MGYP queued: {len(tasks):,} | missing: {len(missing):,}"
                )

            if limit is not None and idx >= limit:
                break

    return list(tasks.values()), missing, len(seen_mgyp)


def stage_files(
    tasks: Sequence[Task],
    *,
    allow_symlinks: bool,
    workers: int,
    desc: str,
) -> tuple[int, list[Task]]:
    missing: list[Task] = []

    def worker(task: Task) -> tuple[Task, bool]:
        if task.destination.exists():
            try:
                if os.path.samefile(task.source, task.destination):
                    return task, True
            except FileNotFoundError:
                pass
            except OSError:
                pass

        task.destination.parent.mkdir(parents=True, exist_ok=True)

        if allow_symlinks:
            try:
                if task.destination.exists():
                    task.destination.unlink()
                os.symlink(task.source, task.destination)
                return task, True
            except OSError:
                pass

        try:
            if task.destination.exists():
                task.destination.unlink()
            import shutil

            shutil.copy2(task.source, task.destination)
            return task, True
        except Exception:
            return task, False

    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(
        total=len(tasks), desc=desc
    ) as progress:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            task, ok = future.result()
            if not ok:
                missing.append(task)
            progress.update(1)

    return len(tasks) - len(missing), missing


def convert_pdbs_to_cif(pdb_root: Path, cif_root: Path, workers: int) -> tuple[int, int, int, list[str]]:
    """Convert staged PDBs to mmCIFs under the CIF root using gemmi/pdb2cif."""

    processable_exts = {".pdb", ".pdb.gz", ".ent", ".ent.gz"}
    gemmi_bin = shutil.which("gemmi")
    pdb2cif_bin = shutil.which("pdb2cif")
    if gemmi_bin is None and pdb2cif_bin is None:
        tqdm.write("No gemmi or pdb2cif executable found; skipping PDB→CIF conversion.")
        return 0, 0, 0, []

    jobs: list[Task] = []
    for pdb_path in pdb_root.rglob("*"):
        if not pdb_path.is_file():
            continue
        if pdb_path.suffix.lower() not in processable_exts:
            continue

        base = pdb_path.name
        stem = base
        if stem.lower().endswith(".gz"):
            stem = stem[:-3]
        stem = stem.rsplit(".", 1)[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        bucket = (digits[-3:] if digits else stem[-3:]).upper()
        bucket = bucket.rjust(3, "0")

        out_dir = cif_root / bucket
        out_path = out_dir / f"{stem}.cif"
        jobs.append(Task(source=pdb_path, destination=out_path))

    converted = 0
    skipped = 0
    failed = 0
    failed_paths: list[str] = []

    def _worker(task: Task) -> tuple[Path, str]:
        src = task.source
        dst = task.destination
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            return src, "skip"

        tmp_input: Optional[Path] = None
        input_path: Path

        if src.suffix.lower() == ".gz":
            fd, tmp_name = tempfile.mkstemp(suffix=".pdb")
            os.close(fd)
            tmp_input = Path(tmp_name)
            with gzip.open(src, "rt") as fin, open(tmp_input, "w") as fout:
                shutil.copyfileobj(fin, fout)
            input_path = tmp_input
        else:
            input_path = src

        success = False
        try:
            if gemmi_bin is not None:
                cmd = [gemmi_bin, "convert", str(input_path), str(dst)]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if result.returncode == 0:
                    success = True
            if not success and pdb2cif_bin is not None:
                cmd = [pdb2cif_bin, str(input_path), str(dst)]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if result.returncode == 0:
                    success = True
        finally:
            if tmp_input is not None and tmp_input.exists():
                tmp_input.unlink()

        if success:
            try:
                src.unlink()
            except Exception:
                pass
            return src, "ok"
        return src, "fail"

    with ThreadPoolExecutor(max_workers=workers) as executor, tqdm(total=len(jobs), desc="Converting PDB→CIF") as progress:
        futures = {executor.submit(_worker, job): job for job in jobs}
        for future in as_completed(futures):
            src_path, status = future.result()
            if status == "ok":
                converted += 1
            elif status == "skip":
                skipped += 1
            else:
                failed += 1
                failed_paths.append(str(src_path))
            progress.update(1)

    for dirpath, dirnames, filenames in os.walk(pdb_root, topdown=False):
        if not dirnames and not filenames:
            try:
                Path(dirpath).rmdir()
            except OSError:
                pass

    return converted, skipped, failed, failed_paths


def writable_symlinks(path: Path) -> bool:
    probe_src = path / ".afesm_symlink_probe"
    probe_dest = path / ".afesm_symlink_probe_link"
    try:
        probe_src.write_text("probe", encoding="utf-8")
        os.symlink(probe_src, probe_dest)
    except OSError:
        return False
    else:
        probe_dest.unlink()
        probe_src.unlink()
        return True


if __name__ == "__main__":
    args = parse_args()

    out_root: Path = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)

    allow_symlinks = not args.copy and writable_symlinks(out_root)

    cifs_root = out_root / "cifs"
    pdbs_root = out_root / "pdbs"

    cif_tasks, missing_rep_ids = build_rep_tasks(args.rep_list, args.afdb_root, cifs_root, args.max_tasks)
    print(f"Queued {len(cif_tasks):,} representative CIF tasks (missing: {len(missing_rep_ids):,}).")
    print(f"Staging CIFs to {cifs_root} using {'symlinks' if allow_symlinks else 'copies'} ({args.workers} workers)…")
    completed_cif, missing_cif = stage_files(
        cif_tasks, allow_symlinks=allow_symlinks, workers=args.workers, desc="Staging CIFs"
    )
    print(f"Completed CIFs: {completed_cif:,} | Missing CIFs: {len(missing_cif):,}")

    pdb_tasks, missing_mgyp_ids, total_mgyp = build_mgyp_tasks(
        args.clusters, args.esmf_root, pdbs_root, args.max_tasks, args.log_interval
    )
    print(f"Queued {len(pdb_tasks):,} MGYP PDB tasks from {total_mgyp:,} unique MGYP entries (missing: {len(missing_mgyp_ids):,}).")
    print(f"Staging PDBs to {pdbs_root} using {'symlinks' if allow_symlinks else 'copies'} ({args.workers} workers)…")
    completed_pdb, missing_pdb = stage_files(
        pdb_tasks, allow_symlinks=allow_symlinks, workers=args.workers, desc="Staging PDBs"
    )
    print(f"Completed PDBs: {completed_pdb:,} | Missing PDBs: {len(missing_pdb):,}")

    converted = skipped = failed = 0
    failed_paths: list[str] = []
    if args.convert_pdb:
        converted, skipped, failed, failed_paths = convert_pdbs_to_cif(pdbs_root, cifs_root, args.workers)
        print(
            "Converted PDB→CIF: "
            f"{converted:,} success, {skipped:,} skipped (already present), {failed:,} failed"
        )

    logs_dir = out_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "missing_cifs.txt").write_text(
        "\n".join(missing_rep_ids + [task.source.name for task in missing_cif]), encoding="utf-8"
    )
    (logs_dir / "missing_pdbs.txt").write_text(
        "\n".join(missing_mgyp_ids + [task.source.name for task in missing_pdb]), encoding="utf-8"
    )

    if args.convert_pdb:
        (logs_dir / "pdb2cif_failed.txt").write_text(
            "\n".join(failed_paths), encoding="utf-8"
        )

    if args.strict and (missing_rep_ids or missing_mgyp_ids or missing_cif or missing_pdb or failed):
        raise SystemExit("Missing or failed conversions detected; rerun without --strict to ignore.")
