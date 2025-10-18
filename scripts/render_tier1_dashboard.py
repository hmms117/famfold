"""Render Tier 1 inference telemetry dashboards."""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.tier1.dashboard import build_inference_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_dir", type=Path, help="Directory containing inference.jsonl telemetry.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write the markdown table to. Defaults to stdout.",
    )
    args = parser.parse_args()

    dashboard = build_inference_dashboard(args.log_dir)
    output = dashboard.to_markdown()

    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
