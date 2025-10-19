"""Helpers for recording retrieval metrics in the hypothesis-test workspace."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import duckdb


_CACHE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "pretests"
    / "hypothesis_test"
    / "caches"
)


def _coerce_metrics(payload: Mapping[str, float | int]) -> Dict[str, float]:
    """Normalise raw JSON metrics into a flat ``metric -> float`` mapping."""

    metrics: Dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    return metrics


def load_metric_bundle(namespace: str) -> Dict[str, float]:
    """Load retrieval metrics for ``namespace`` from the cached JSON bundle."""

    path = _CACHE_ROOT / namespace / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"No retrieval metrics cache found for namespace '{namespace}'.")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and "global" in data:
        global_payload = data.get("global", {})
        if isinstance(global_payload, dict):
            return _coerce_metrics(global_payload)

    if not isinstance(data, Mapping):
        raise TypeError(
            f"Metrics bundle for namespace '{namespace}' must be a mapping, received {type(data)!r}."
        )
    return _coerce_metrics(data)


@dataclass
class MetricComparison:
    """Captures the outcome of comparing two retrieval encoder metrics."""

    namespace: str
    baseline: Optional[str]
    metric: str
    value: Optional[float]
    baseline_value: Optional[float]

    @property
    def delta(self) -> Optional[float]:
        if self.value is None or self.baseline_value is None:
            return None
        return self.value - self.baseline_value


class RetrievalManifestLogger:
    """Persist SaESM2 comparison metrics into a DuckDB manifest."""

    def __init__(self, destination: Path) -> None:
        self.destination = destination
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_schema()

    def _initialise_schema(self) -> None:
        with duckdb.connect(str(self.destination)) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS retrieval_comparisons (
                    run_label TEXT,
                    namespace TEXT,
                    baseline TEXT,
                    metric TEXT,
                    value DOUBLE,
                    baseline_value DOUBLE,
                    delta DOUBLE,
                    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def log(self, run_label: str, comparisons: Iterable[MetricComparison]) -> None:
        rows = [
            (
                run_label,
                comparison.namespace,
                comparison.baseline or "",
                comparison.metric,
                comparison.value,
                comparison.baseline_value,
                comparison.delta,
            )
            for comparison in comparisons
        ]

        if not rows:
            return

        with duckdb.connect(str(self.destination)) as connection:
            connection.executemany(
                """
                INSERT INTO retrieval_comparisons
                (run_label, namespace, baseline, metric, value, baseline_value, delta)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


class RetrievalMetricsReporter:
    """High-level helper that compares SaESM2 metrics against a baseline."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_logger = RetrievalManifestLogger(manifest_path)

    def _prepare_comparisons(
        self,
        namespace: str,
        metrics: Mapping[str, float],
        baseline: Optional[str],
        baseline_metrics: Mapping[str, float],
    ) -> Iterable[MetricComparison]:
        seen: set[str] = set(metrics)
        for metric, value in metrics.items():
            baseline_value = baseline_metrics.get(metric)
            yield MetricComparison(
                namespace=namespace,
                baseline=baseline,
                metric=metric,
                value=value,
                baseline_value=baseline_value,
            )

        for metric in baseline_metrics:
            if metric in seen:
                continue
            yield MetricComparison(
                namespace=namespace,
                baseline=baseline,
                metric=metric,
                value=None,
                baseline_value=baseline_metrics[metric],
            )

    def report(
        self,
        run_label: str,
        namespace: str,
        *,
        latency_ms: Optional[float] = None,
        baseline: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics = dict(load_metric_bundle(namespace))
        if latency_ms is not None:
            metrics["latency_ms"] = float(latency_ms)

        baseline_metrics: Dict[str, float] = {}
        if baseline:
            try:
                baseline_metrics = dict(load_metric_bundle(baseline))
            except FileNotFoundError:
                baseline_metrics = {}

        comparisons = self._prepare_comparisons(namespace, metrics, baseline, baseline_metrics)
        self.manifest_logger.log(run_label, comparisons)
        return metrics
