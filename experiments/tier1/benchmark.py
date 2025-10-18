"""Latency and quality benchmarking helpers for Tier 1 inference runs."""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


@dataclass(frozen=True)
class SequenceBenchmark:
    """Aggregated metrics for a single sequence."""

    qhash: str
    retrieval_s: float
    prior_s: float
    inference_s: float
    total_s: float
    rmsd: Optional[float]
    tm_score: Optional[float]
    route: Optional[str]
    trunk: Optional[str]
    gpu_memory_mb: Optional[float]


@dataclass(frozen=True)
class BenchmarkRunSummary:
    """Summary statistics for a benchmark run."""

    name: str
    sequences: List[SequenceBenchmark]

    @property
    def total_sequences(self) -> int:
        return len(self.sequences)

    def mean_total_latency(self) -> float:
        values = [entry.total_s for entry in self.sequences]
        return statistics.fmean(values) if values else 0.0

    def median_total_latency(self) -> float:
        values = [entry.total_s for entry in self.sequences]
        return statistics.median(values) if values else 0.0

    def mean_rmsd(self) -> Optional[float]:
        values = [entry.rmsd for entry in self.sequences if entry.rmsd is not None]
        return statistics.fmean(values) if values else None

    def mean_tm_score(self) -> Optional[float]:
        values = [entry.tm_score for entry in self.sequences if entry.tm_score is not None]
        return statistics.fmean(values) if values else None

    def acceptance_rate(self) -> Optional[float]:
        routes = [entry.route for entry in self.sequences if entry.route]
        if not routes:
            return None
        accepted = sum(route.upper() == "ACCEPT" for route in routes)
        return accepted / len(routes)

    def escape_rate(self) -> Optional[float]:
        routes = [entry.route for entry in self.sequences if entry.route]
        if not routes:
            return None
        escaped = sum(route.upper() == "ESCAPE" for route in routes)
        return escaped / len(routes)

    def mean_gpu_memory(self) -> Optional[float]:
        values = [entry.gpu_memory_mb for entry in self.sequences if entry.gpu_memory_mb]
        return statistics.fmean(values) if values else None


def _load_jsonl(path: Path) -> Iterable[Mapping[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _load_stage_timings(path: Path) -> Mapping[str, float]:
    timings: Dict[str, float] = {}
    for entry in _load_jsonl(path):
        qhash = str(entry.get("qhash"))
        runtime = float(entry.get("runtime_s", 0.0))
        timings[qhash] = timings.get(qhash, 0.0) + runtime
    return timings


def _load_inference_metrics(path: Path) -> Mapping[str, Mapping[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}
    for entry in _load_jsonl(path):
        qhash = str(entry.get("qhash"))
        metrics[qhash] = {
            "runtime_s": float(entry.get("runtime_s", 0.0)),
            "rmsd": entry.get("rmsd"),
            "tm_score": entry.get("tm_score"),
            "route": entry.get("route"),
            "trunk": entry.get("trunk"),
            "gpu_memory_mb": entry.get("gpu_memory_mb") or entry.get("gpu_memory_gb"),
        }
        if isinstance(metrics[qhash]["gpu_memory_mb"], (int, float)):
            # Convert GB to MB if needed.
            if metrics[qhash]["gpu_memory_mb"] and metrics[qhash]["gpu_memory_mb"] < 64:
                metrics[qhash]["gpu_memory_mb"] = float(metrics[qhash]["gpu_memory_mb"]) * 1024
            else:
                metrics[qhash]["gpu_memory_mb"] = float(metrics[qhash]["gpu_memory_mb"])
        else:
            metrics[qhash]["gpu_memory_mb"] = None
        if metrics[qhash]["rmsd"] is not None:
            metrics[qhash]["rmsd"] = float(metrics[qhash]["rmsd"])
        if metrics[qhash]["tm_score"] is not None:
            metrics[qhash]["tm_score"] = float(metrics[qhash]["tm_score"])
    return metrics


def load_benchmark_run(name: str, log_dir: Path) -> BenchmarkRunSummary:
    """Load telemetry from ``log_dir`` and aggregate latency/quality metrics."""

    retrieval_path = log_dir / "retrieval.jsonl"
    prior_path = log_dir / "prior.jsonl"
    inference_path = log_dir / "inference.jsonl"

    retrieval_timings = _load_stage_timings(retrieval_path) if retrieval_path.exists() else {}
    prior_timings = _load_stage_timings(prior_path) if prior_path.exists() else {}
    inference_metrics = _load_inference_metrics(inference_path) if inference_path.exists() else {}

    qhashes = set(retrieval_timings) | set(prior_timings) | set(inference_metrics)

    sequences: List[SequenceBenchmark] = []
    for qhash in sorted(qhashes):
        retrieval = retrieval_timings.get(qhash, 0.0)
        prior = prior_timings.get(qhash, 0.0)
        inference_entry = inference_metrics.get(qhash, {})
        inference = float(inference_entry.get("runtime_s", 0.0))
        total = retrieval + prior + inference

        sequences.append(
            SequenceBenchmark(
                qhash=qhash,
                retrieval_s=retrieval,
                prior_s=prior,
                inference_s=inference,
                total_s=total,
                rmsd=inference_entry.get("rmsd"),
                tm_score=inference_entry.get("tm_score"),
                route=inference_entry.get("route"),
                trunk=inference_entry.get("trunk"),
                gpu_memory_mb=inference_entry.get("gpu_memory_mb"),
            )
        )

    return BenchmarkRunSummary(name=name, sequences=sequences)


def compare_run_summaries(runs: Mapping[str, Path]) -> Mapping[str, BenchmarkRunSummary]:
    """Load and summarise all provided runs."""

    summaries: Dict[str, BenchmarkRunSummary] = {}
    for name, path in runs.items():
        summaries[name] = load_benchmark_run(name, Path(path))
    return summaries


__all__ = [
    "BenchmarkRunSummary",
    "SequenceBenchmark",
    "compare_run_summaries",
    "load_benchmark_run",
]
