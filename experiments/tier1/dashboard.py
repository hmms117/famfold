"""Telemetry dashboard utilities for Tier 1 inference runs."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .benchmark import _load_inference_metrics


@dataclass(frozen=True)
class DashboardRow:
    trunk: str
    sequences: int
    acceptance_rate: Optional[float]
    escape_rate: Optional[float]
    mean_gpu_memory_mb: Optional[float]


@dataclass(frozen=True)
class InferenceDashboard:
    rows: List[DashboardRow]

    def to_markdown(self) -> str:
        header = "| Trunk | Sequences | Acceptance | Escape | Mean GPU MB |\n"
        header += "| --- | ---: | ---: | ---: | ---: |\n"
        lines = [header]
        for row in self.rows:
            acceptance = f"{row.acceptance_rate:.2%}" if row.acceptance_rate is not None else "N/A"
            escape = f"{row.escape_rate:.2%}" if row.escape_rate is not None else "N/A"
            gpu = f"{row.mean_gpu_memory_mb:.1f}" if row.mean_gpu_memory_mb is not None else "N/A"
            lines.append(
                f"| {row.trunk} | {row.sequences} | {acceptance} | {escape} | {gpu} |\n"
            )
        return "".join(lines)


def build_inference_dashboard(log_dir: Path) -> InferenceDashboard:
    inference_path = Path(log_dir) / "inference.jsonl"
    metrics = _load_inference_metrics(inference_path) if inference_path.exists() else {}

    buckets: Dict[str, List[Mapping[str, object]]] = {}
    for entry in metrics.values():
        trunk = str(entry.get("trunk", "unknown"))
        buckets.setdefault(trunk, []).append(entry)

    rows: List[DashboardRow] = []
    for trunk, entries in sorted(buckets.items()):
        routes = [entry.get("route") for entry in entries if entry.get("route")]
        acceptance = None
        escape = None
        if routes:
            acceptance = sum(route.upper() == "ACCEPT" for route in routes) / len(routes)
            escape = sum(route.upper() == "ESCAPE" for route in routes) / len(routes)
        gpu_values = [entry.get("gpu_memory_mb") for entry in entries if entry.get("gpu_memory_mb")]
        mean_gpu = statistics.fmean(gpu_values) if gpu_values else None
        rows.append(
            DashboardRow(
                trunk=trunk,
                sequences=len(entries),
                acceptance_rate=acceptance,
                escape_rate=escape,
                mean_gpu_memory_mb=mean_gpu,
            )
        )

    rows.sort(key=lambda row: row.trunk)
    return InferenceDashboard(rows=rows)


__all__ = [
    "DashboardRow",
    "InferenceDashboard",
    "build_inference_dashboard",
]
