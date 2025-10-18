"""CLI for comparing Tier 1 fast-path trunk performance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from experiments.tier1.benchmark import compare_run_summaries


def _parse_runs(values: list[str]) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=PATH, received '{value}'.")
        name, path = value.split("=", 1)
        runs[name] = Path(path)
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        default=[],
        help="Benchmark run specified as NAME=LOG_DIR. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Emit summary as JSON instead of a text table.",
    )
    args = parser.parse_args()

    if not args.runs:
        parser.error("At least one --run NAME=PATH argument is required.")

    runs = _parse_runs(args.runs)
    summaries = compare_run_summaries(runs)

    if args.json_output:
        payload = {
            name: {
                "sequences": summary.total_sequences,
                "mean_total_latency": summary.mean_total_latency(),
                "median_total_latency": summary.median_total_latency(),
                "mean_rmsd": summary.mean_rmsd(),
                "mean_tm_score": summary.mean_tm_score(),
                "acceptance_rate": summary.acceptance_rate(),
                "escape_rate": summary.escape_rate(),
                "mean_gpu_memory_mb": summary.mean_gpu_memory(),
            }
            for name, summary in summaries.items()
        }
        print(json.dumps(payload, indent=2))
        return

    header = (
        "| Run | Seqs | Mean Latency (s) | Median Latency (s) | Mean RMSD | Mean TM |"
        " Acceptance | Escape | GPU MB |\n"
    )
    header += "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    lines = [header]
    for name, summary in summaries.items():
        mean_latency = summary.mean_total_latency()
        median_latency = summary.median_total_latency()
        mean_rmsd = summary.mean_rmsd()
        mean_tm = summary.mean_tm_score()
        acceptance = summary.acceptance_rate()
        escape = summary.escape_rate()
        gpu = summary.mean_gpu_memory()
        lines.append(
            "| {name} | {seqs} | {mean:.3f} | {median:.3f} | {rmsd} | {tm} | {acc} | {esc} | {gpu_mb} |\n".format(
                name=name,
                seqs=summary.total_sequences,
                mean=mean_latency,
                median=median_latency,
                rmsd=f"{mean_rmsd:.3f}" if mean_rmsd is not None else "N/A",
                tm=f"{mean_tm:.3f}" if mean_tm is not None else "N/A",
                acc=f"{acceptance:.2%}" if acceptance is not None else "N/A",
                esc=f"{escape:.2%}" if escape is not None else "N/A",
                gpu_mb=f"{gpu:.1f}" if gpu is not None else "N/A",
            )
        )

    print("".join(lines))


if __name__ == "__main__":
    main()
