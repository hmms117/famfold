import json
from pathlib import Path

import pytest

from experiments.tier1.benchmark import compare_run_summaries, load_benchmark_run
from experiments.tier1.dashboard import build_inference_dashboard


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_benchmark_run(tmp_path: Path) -> None:
    log_dir = tmp_path / "saesm"
    _write_jsonl(
        log_dir / "retrieval.jsonl",
        [
            {"qhash": "A", "runtime_s": 0.2},
            {"qhash": "B", "runtime_s": 0.25},
        ],
    )
    _write_jsonl(
        log_dir / "prior.jsonl",
        [
            {"qhash": "A", "runtime_s": 0.1},
            {"qhash": "B", "runtime_s": 0.12},
        ],
    )
    _write_jsonl(
        log_dir / "inference.jsonl",
        [
            {
                "qhash": "A",
                "runtime_s": 0.5,
                "rmsd": 1.0,
                "tm_score": 0.7,
                "route": "ACCEPT",
                "trunk": "saesm2_fast",
                "gpu_memory_mb": 5200,
            },
            {
                "qhash": "B",
                "runtime_s": 0.55,
                "rmsd": 1.2,
                "tm_score": 0.65,
                "route": "ESCAPE",
                "trunk": "saesm2_fast",
                "gpu_memory_mb": 5100,
            },
        ],
    )

    summary = load_benchmark_run("saesm2_fast", log_dir)
    assert summary.total_sequences == 2
    assert summary.mean_total_latency() == pytest.approx((0.8 + 0.92) / 2)
    assert summary.median_total_latency() == pytest.approx(0.86)
    assert summary.mean_rmsd() == pytest.approx(1.1)
    assert summary.mean_tm_score() == pytest.approx(0.675)
    assert summary.acceptance_rate() == pytest.approx(0.5)
    assert summary.escape_rate() == pytest.approx(0.5)
    assert summary.mean_gpu_memory() == pytest.approx(5150.0)


def test_compare_and_dashboard(tmp_path: Path) -> None:
    saesm_dir = tmp_path / "saesm"
    ism_dir = tmp_path / "ism"

    _write_jsonl(saesm_dir / "inference.jsonl", [
        {"qhash": "A", "runtime_s": 0.5, "route": "ACCEPT", "trunk": "saesm2_fast", "gpu_memory_mb": 5200},
        {"qhash": "B", "runtime_s": 0.6, "route": "REFINE", "trunk": "saesm2_fast", "gpu_memory_mb": 5250},
    ])
    _write_jsonl(ism_dir / "inference.jsonl", [
        {"qhash": "A", "runtime_s": 0.7, "route": "ESCAPE", "trunk": "ism_fast", "gpu_memory_mb": 6000},
    ])

    summaries = compare_run_summaries({"saesm": saesm_dir, "ism": ism_dir})
    assert summaries["saesm"].total_sequences == 2
    assert summaries["ism"].total_sequences == 1

    dashboard = build_inference_dashboard(saesm_dir)
    assert dashboard.rows[0].trunk == "saesm2_fast"
    assert dashboard.rows[0].sequences == 2
    assert dashboard.rows[0].acceptance_rate == pytest.approx(0.5)
    assert dashboard.rows[0].escape_rate == pytest.approx(0.0)
    assert dashboard.rows[0].mean_gpu_memory_mb == pytest.approx(5225.0)

    markdown = dashboard.to_markdown()
    assert "saesm2_fast" in markdown
    assert "5225.0" in markdown  # aggregated GPU memory rendered
