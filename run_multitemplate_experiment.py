#!/usr/bin/env python3
"""Drive multi-neighbour MiniFold experiments with optional template priors.

The script orchestrates the end-to-end workflow required to analyse the GH5_21
benchmark with richer template priors:

1. Load the GH5_21 FASTA subset and clustering metadata.
2. Build per-target template distogram overrides using the top-N AF2 neighbours
   (filtered by mean pLDDT).
3. Run MiniFold 48L baseline predictions (no priors), single-neighbour priors,
   and multi-neighbour priors (if requested).
4. Aggregate mean pLDDTs, MiniFold gains, and classification flags into
   `cluster_summary.tsv`, and refresh per-cluster FASTA/TSV artefacts.

Usage (typical):

```
uv run python run_multitemplate_experiment.py \
    --fasta /var/tmp/famfold/test/gh5_21_subset.fasta \
    --clusters /var/tmp/famfold/test/gh5_21_plddt_clusters.tsv \
    --pdb-dir /var/tmp/famfold/test/pdbs \
    --cache /var/tmp/checkpoints \
    --checkpoint /var/tmp/checkpoints/minifold_48L.ckpt \
    --baseline-out /var/tmp/famfold/test/minifold_48L_out/minifold_results_gh5_21_subset \
    --templated-out /var/tmp/famfold/test/minifold_48L_templated_out/minifold_results_gh5_21_subset \
    --multitemplate-out /var/tmp/famfold/test/minifold_48L_top5_out/minifold_results_gh5_21_subset \
    --benchmark-root data/benchmarks/gh5_21_minifold_regressions \
    --output-dir /var/tmp/famfold/test/multitemplate_run
```

Set `HF_HOME`, `TRANSFORMERS_CACHE`, and `TORCH_HOME` to `/var/tmp/checkpoints`
ahead of time so MiniFold reuses cached weights. GPU execution is expected.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from minifold.utils.template_probe import (
    TemplateResidueMap,
    align_template_to_target,
    build_distogram_from_templates,
    extract_ca_coordinates,
    load_template,
    protein_to_sequence,
)

###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, required=True, help="Subset FASTA file.")
    parser.add_argument("--clusters", type=Path, required=True, help="Merged pLDDT/clusters TSV.")
    parser.add_argument("--pdb-dir", type=Path, required=True, help="Directory with AF2 PDBs.")
    parser.add_argument("--cache", type=Path, required=True, help="MiniFold cache directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="MiniFold 48L checkpoint.")
    parser.add_argument("--baseline-out", type=Path, required=True, help="Baseline MiniFold output dir.")
    parser.add_argument("--templated-out", type=Path, required=True, help="Single-neighbour templated MiniFold output dir.")
    parser.add_argument("--multitemplate-out", type=Path, required=True, help="Multi-neighbour templated MiniFold output dir.")
    parser.add_argument("--benchmark-root", type=Path, required=True, help="Benchmark root folder in repo.")
    parser.add_argument("--cluster-summary", type=Path, default=None, help="Optional override for summary TSV path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Scratch directory for artefacts.")
    parser.add_argument("--max-neighbours", type=int, default=5, help="Max AF2 neighbours per cluster (default: 5).")
    parser.add_argument("--min-template-plddt", type=float, default=80.0, help="Minimum AF2 mean pLDDT to use a template.")
    parser.add_argument("--token-per-batch", type=int, default=2048, help="MiniFold token_per_batch setting.")
    parser.add_argument("--uv-executable", type=str, default="uv", help="`uv` executable path (default: uv).")
    parser.add_argument("--python-executable", type=str, default="python", help="Python executable for uv run.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip MiniFold inference (reuse existing outputs).")
    return parser.parse_args()


###############################################################################
# Helpers
###############################################################################


def read_fasta(path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    current = None
    buf: List[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current is not None:
                    sequences[current] = "".join(buf)
                current = line[1:]
                buf = []
            else:
                buf.append(line)
        if current is not None:
            sequences[current] = "".join(buf)
    return sequences


def read_clusters(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, List[str]]]:
    records: List[Dict[str, str]] = []
    by_cl80: Dict[str, List[str]] = defaultdict(list)
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            row["AF2_plddt"] = float(row["AF2_plddt"]) if row["AF2_plddt"] else float("nan")
            records.append(row)
            by_cl80[row["cl80"]].append(row["key"])
    return records, by_cl80


def compute_mean_plddt_from_pdb(path: Path) -> float:
    if not path.exists():
        return float("nan")
    total = 0.0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM") and len(line) >= 66:
                try:
                    total += float(line[60:66])
                    count += 1
                except ValueError:
                    continue
    if count == 0:
        return float("nan")
    return total / count


def sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name)


def derive_multi_template_overrides(
    sequences: Dict[str, str],
    records: List[Dict[str, str]],
    by_cl80: Dict[str, List[str]],
    pdb_dir: Path,
    *,
    max_neighbours: int,
    min_template_plddt: float,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[Tuple[str, str, str, float, float, float]]]:
    records_by_key = {rec["key"]: rec for rec in records}
    overrides: Dict[str, Dict[str, torch.Tensor]] = {}
    metadata: List[Tuple[str, str, str, float, float, float]] = []

    for cl80, members in by_cl80.items():
        sorted_members = sorted(
            members,
            key=lambda key: records_by_key[key]["AF2_plddt"],
            reverse=True,
        )
        for target in sorted_members:
            sequence = sequences.get(target)
            if not sequence:
                continue

            neighbours = [n for n in sorted_members if n != target]
            templates: List[TemplateResidueMap] = []
            for neighbour in neighbours:
                template_path = pdb_dir / f"{neighbour}.pdb"
                if not template_path.exists():
                    continue
                template_plddt = compute_mean_plddt_from_pdb(template_path)
                if math.isnan(template_plddt) or template_plddt < min_template_plddt:
                    continue

                template = load_template(template_path)
                template_seq = protein_to_sequence(template)
                identity, mapping = align_template_to_target(sequence, template_seq)
                if not mapping:
                    continue
                coords, mask = extract_ca_coordinates(template)
                templates.append(
                    TemplateResidueMap(
                        name=neighbour,
                        coordinates=coords,
                        mask=mask,
                        mapping=mapping,
                        identity=identity,
                    )
                )
                coverage = len(mapping) / len(sequence)
                metadata.append((target, neighbour, cl80, identity, coverage, template_plddt))
                if len(templates) >= max_neighbours:
                    break

            if not templates:
                continue
            dist = build_distogram_from_templates(len(sequence), templates)
            overrides[target] = {"distogram": dist.to(dtype=torch.float16)}

    return overrides, metadata


def write_metadata(path: Path, metadata: Sequence[Tuple[str, str, str, float, float, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["target", "neighbour", "cl80", "identity", "coverage", "template_AF2_plddt"])
        writer.writerows(metadata)


def run_predict(
    fasta: Path,
    out_dir: Path,
    cache: Path,
    checkpoint: Path,
    *,
    uv_executable: str,
    python_executable: str,
    template_overrides: Path | None,
    token_per_batch: int,
) -> None:
    args = [
        uv_executable,
        "run",
        python_executable,
        "predict.py",
        str(fasta),
        "--out_dir",
        str(out_dir),
        "--cache",
        str(cache),
        "--model_size",
        "48L",
        "--token_per_batch",
        str(token_per_batch),
        "--checkpoint",
        str(checkpoint),
    ]
    if template_overrides is not None:
        args.extend(["--template_overrides", str(template_overrides)])

    env = os.environ.copy()
    env.setdefault("HF_HOME", str(cache))
    env.setdefault("TRANSFORMERS_CACHE", str(cache))
    env.setdefault("TORCH_HOME", str(cache))

    subprocess.run(args, env=env, check=True)


def mean_plddt_from_pdb(path: Path) -> float:
    total = 0.0
    count = 0
    with path.open() as handle:
        for line in handle:
            if line.startswith("ATOM") and len(line) >= 66:
                try:
                    total += float(line[60:66])
                    count += 1
                except ValueError:
                    continue
    if count == 0:
        return float("nan")
    return total / count


def load_scores_from_dir(output_dir: Path) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if not output_dir.exists():
        return scores
    for pdb_path in output_dir.glob("*.pdb"):
        scores[pdb_path.stem] = mean_plddt_from_pdb(pdb_path)
    return scores


def update_summary(
    summary_path: Path,
    records: Sequence[Dict[str, str]],
    *,
    baseline_scores: Dict[str, float],
    templated_scores: Dict[str, float],
    multi_scores: Dict[str, float],
) -> None:
    def fmt(value: float) -> str:
        return "" if math.isnan(value) else f"{value:.6f}"

    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "key",
            "cl50",
            "cl80",
            "AF2_plddt",
            "Minifold_48L_plddt",
            "Minifold_48L_plddt_templated",
            "Minifold_48L_plddt_multitemplate",
            "is_bad_af2>=85.0_minifold<80.0",
            "is_bad_templated_af2>=85.0_minifold<80.0",
            "is_bad_multitemplate_af2>=85.0_minifold<80.0",
        ]
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            key = row["key"]
            af2 = row["AF2_plddt"]
            baseline = baseline_scores.get(key, float("nan"))
            templated = templated_scores.get(key, float("nan"))
            multi = multi_scores.get(key, float("nan"))

            def classify(value: float) -> str:
                return (
                    "1"
                    if (not math.isnan(af2) and af2 >= 85.0 and not math.isnan(value) and value < 80.0)
                    else "0"
                )

            writer.writerow(
                {
                    "key": key,
                    "cl50": row["cl50"],
                    "cl80": row["cl80"],
                    "AF2_plddt": fmt(af2),
                    "Minifold_48L_plddt": fmt(baseline),
                    "Minifold_48L_plddt_templated": fmt(templated),
                    "Minifold_48L_plddt_multitemplate": fmt(multi),
                    "is_bad_af2>=85.0_minifold<80.0": classify(baseline),
                    "is_bad_templated_af2>=85.0_minifold<80.0": classify(templated),
                    "is_bad_multitemplate_af2>=85.0_minifold<80.0": classify(multi),
                }
            )


def regenerate_cluster_artifacts(
    benchmark_root: Path,
    records: Sequence[Dict[str, str]],
    sequences: Dict[str, str],
    *,
    baseline_scores: Dict[str, float],
    templated_scores: Dict[str, float],
    multi_scores: Dict[str, float],
) -> None:
    clusters = {"cl50": defaultdict(list), "cl80": defaultdict(list)}
    for row in records:
        key = row["key"]
        for level in ("cl50", "cl80"):
            clusters[level][row[level]].append(key)

    for level, mapping in clusters.items():
        for cid, keys in mapping.items():
            if len(keys) < 2:
                continue
            folder = benchmark_root / level / sanitize(cid)
            if not folder.exists():
                continue
            fasta_path = folder / "cluster_sequences.fasta"
            with fasta_path.open("w") as fasta_handle:
                for key in keys:
                    seq = sequences.get(key)
                    if not seq:
                        continue
                    fasta_handle.write(f">{key}\n")
                    for i in range(0, len(seq), 80):
                        fasta_handle.write(seq[i : i + 80] + "\n")
            tsv_path = folder / "plddts.tsv"
            with tsv_path.open("w", newline="") as tsv_handle:
                writer = csv.writer(tsv_handle, delimiter="\t")
                writer.writerow(
                    [
                        "key",
                        "AF2_plddt",
                        "Minifold_48L_plddt",
                        "Minifold_48L_plddt_templated",
                        "Minifold_48L_plddt_multitemplate",
                    ]
                )
                for key in keys:
                    writer.writerow(
                        [
                            key,
                            f"{records_by_key[key]['AF2_plddt']:.6f}" if not math.isnan(records_by_key[key]["AF2_plddt"]) else "",
                            "" if math.isnan(baseline_scores.get(key, float("nan"))) else f"{baseline_scores[key]:.6f}",
                            "" if math.isnan(templated_scores.get(key, float("nan"))) else f"{templated_scores[key]:.6f}",
                            "" if math.isnan(multi_scores.get(key, float("nan"))) else f"{multi_scores[key]:.6f}",
                        ]
                    )


def compute_aggregate_gains(
    records: Sequence[Dict[str, str]],
    *,
    baseline_scores: Dict[str, float],
    templated_scores: Dict[str, float],
    multi_scores: Dict[str, float],
) -> Dict[str, float]:
    def mean(values: Iterable[float]) -> float:
        values = [v for v in values if not math.isnan(v)]
        return sum(values) / len(values) if values else float("nan")

    all_keys = [row["key"] for row in records]

    def gain_dict(scores: Dict[str, float]) -> Dict[str, float]:
        return {
            key: scores.get(key, float("nan")) - baseline_scores.get(key, float("nan"))
            for key in all_keys
        }

    templated_gain = gain_dict(templated_scores)
    multi_gain = gain_dict(multi_scores)

    def is_af2_high(key: str) -> bool:
        af2 = records_by_key[key]["AF2_plddt"]
        return not math.isnan(af2) and af2 >= 85.0

    def is_bad_baseline(key: str) -> bool:
        base = baseline_scores.get(key, float("nan"))
        return is_af2_high(key) and not math.isnan(base) and base < 80.0

    stats = {
        "templated_gain_all": mean(templated_gain.values()),
        "multi_gain_all": mean(multi_gain.values()),
        "templated_gain_af2_high": mean(
            [templated_gain[key] for key in all_keys if is_af2_high(key)]
        ),
        "multi_gain_af2_high": mean([multi_gain[key] for key in all_keys if is_af2_high(key)]),
        "templated_gain_bad_baseline": mean(
            [templated_gain[key] for key in all_keys if is_bad_baseline(key)]
        ),
        "multi_gain_bad_baseline": mean(
            [multi_gain[key] for key in all_keys if is_bad_baseline(key)]
        ),
    }

    stats.update(
        {
            "bad_baseline_count": sum(1 for key in all_keys if is_bad_baseline(key)),
            "bad_templated_count": sum(
                1
                for key in all_keys
                if is_af2_high(key)
                and not math.isnan(templated_scores.get(key, float("nan")))
                and templated_scores[key] < 80.0
            ),
            "bad_multi_count": sum(
                1
                for key in all_keys
                if is_af2_high(key)
                and not math.isnan(multi_scores.get(key, float("nan")))
                and multi_scores[key] < 80.0
            ),
        }
    )
    return stats


def print_summary_stats(stats: Dict[str, float]) -> None:
    print("Summary statistics:")
    print(f"  Baseline bad cases: {stats['bad_baseline_count']}")
    print(f"  Templated bad cases: {stats['bad_templated_count']}")
    print(f"  Multi-template bad cases: {stats['bad_multi_count']}")
    print(f"  Avg gain (templated): {stats['templated_gain_all']:.3f}")
    print(f"  Avg gain (multi): {stats['multi_gain_all']:.3f}")
    print(f"  Avg gain AF2>=85 (templated): {stats['templated_gain_af2_high']:.3f}")
    print(f"  Avg gain AF2>=85 (multi): {stats['multi_gain_af2_high']:.3f}")
    print(f"  Avg gain bad-baseline (templated): {stats['templated_gain_bad_baseline']:.3f}")
    print(f"  Avg gain bad-baseline (multi): {stats['multi_gain_bad_baseline']:.3f}")


###############################################################################
# Main orchestration
###############################################################################


def main() -> None:
    args = parse_args()

    summary_path = (
        args.cluster_summary
        if args.cluster_summary is not None
        else args.benchmark_root / "cluster_summary.tsv"
    )

    sequences = read_fasta(args.fasta)
    records, by_cl80 = read_clusters(args.clusters)
    global records_by_key
    records_by_key = {row["key"]: row for row in records}

    args.output_dir.mkdir(parents=True, exist_ok=True)

    overrides, metadata = derive_multi_template_overrides(
        sequences,
        records,
        by_cl80,
        args.pdb_dir,
        max_neighbours=args.max_neighbours,
        min_template_plddt=args.min_template_plddt,
    )

    overrides_path = args.output_dir / "gh5_21_cl80_template_overrides_top5.pt"
    metadata_path = args.output_dir / "gh5_21_cl80_template_overrides_top5.tsv"
    torch.save(overrides, overrides_path)
    write_metadata(metadata_path, metadata)
    print(f"Stored {len(overrides)} multi-template overrides at {overrides_path}")
    print(f"Wrote metadata for {len(metadata)} template selections to {metadata_path}")

    if not args.skip_predict:
        print("Running MiniFold baseline predictions...")
        run_predict(
            args.fasta,
            args.baseline_out,
            args.cache,
            args.checkpoint,
            uv_executable=args.uv_executable,
            python_executable=args.python_executable,
            template_overrides=None,
            token_per_batch=args.token_per_batch,
        )

        print("Running MiniFold single-template predictions...")
        single_override_path = args.output_dir / "gh5_21_cl80_template_overrides_single.pt"
        if not single_override_path.exists():
            print(
                "WARNING: expected single-template override file not found; "
                "skipping single-template run."
            )
        else:
            run_predict(
                args.fasta,
                args.templated_out,
                args.cache,
                args.checkpoint,
                uv_executable=args.uv_executable,
                python_executable=args.python_executable,
                template_overrides=single_override_path,
                token_per_batch=args.token_per_batch,
            )

        print("Running MiniFold multi-template predictions...")
        run_predict(
            args.fasta,
            args.multitemplate_out,
            args.cache,
            args.checkpoint,
            uv_executable=args.uv_executable,
            python_executable=args.python_executable,
            template_overrides=overrides_path,
            token_per_batch=args.token_per_batch,
        )
    else:
        print("Skipping MiniFold prediction runs (reusing existing outputs).")

    baseline_scores = load_scores_from_dir(args.baseline_out)
    templated_scores = load_scores_from_dir(args.templated_out)
    multi_scores = load_scores_from_dir(args.multitemplate_out)

    update_summary(
        summary_path,
        records,
        baseline_scores=baseline_scores,
        templated_scores=templated_scores,
        multi_scores=multi_scores,
    )
    regenerate_cluster_artifacts(
        args.benchmark_root,
        records,
        sequences,
        baseline_scores=baseline_scores,
        templated_scores=templated_scores,
        multi_scores=multi_scores,
    )
    stats = compute_aggregate_gains(
        records,
        baseline_scores=baseline_scores,
        templated_scores=templated_scores,
        multi_scores=multi_scores,
    )
    print_summary_stats(stats)


if __name__ == "__main__":
    main()

