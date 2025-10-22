#!/usr/bin/env python3
"""Rank top structural templates per sequence using cluster tiers and pLDDT."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import polars as pl
import typer

app = typer.Typer(add_completion=False)

LEVELS: List[int] = [90, 80, 70, 60, 50]


def _detect_plddt_col(cols: List[str]) -> str:
    candidates = [c for c in cols if c.lower() in ("avg_plddt", "plddt", "plddt_mean")]
    if not candidates:
        raise ValueError(
            "Could not find a pLDDT column. Expected one of: avg_plddt, plddt, plddt_mean"
        )
    for name in ("avg_plddt", "plddt", "plddt_mean"):
        for col in candidates:
            if col.lower() == name:
                return col
    return candidates[0]


def _require_cols(df_cols: List[str], needed: List[str]) -> None:
    missing = [c for c in needed if c not in df_cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _scan_any(path: str | Path) -> pl.LazyFrame:
    p = str(path).lower()
    if p.endswith((".parquet", ".pq")):
        return pl.scan_parquet(path)
    if p.endswith(".tsv"):
        return pl.scan_csv(path, separator="\t", infer_schema_length=2000)
    if p.endswith(".csv"):
        return pl.scan_csv(path, infer_schema_length=2000)
    return pl.scan_csv(path, infer_schema_length=2000)


def _normalize_has_structure(lf: pl.LazyFrame, col: str) -> pl.LazyFrame:
    return lf.with_columns(
        pl.when(pl.col(col).is_in([True, False]))
        .then(pl.col(col).cast(pl.Boolean))
        .otherwise(
            pl.col(col)
            .cast(pl.Utf8)
            .str.to_lowercase()
            .replace({"1": "true", "0": "false"})
            .is_in(["true", "t", "yes", "y"])
        )
        .alias(col)
    )


def _build_cluster_index(
    lf_struct: pl.LazyFrame,
    level_col: str,
    per_cluster: int,
) -> pl.LazyFrame:
    return (
        lf_struct
        .select(
            pl.col("seq_id").alias("cand_id"),
            pl.col("cand_plddt"),
            pl.col(level_col).alias("cluster"),
        )
        .drop_nulls(["cluster"])
        .sort(["cluster", "cand_plddt"], descending=[False, True])
        .group_by("cluster")
        .head(per_cluster)
    )


def find_top_k(
    table_path: str | Path,
    out_path: str | Path,
    *,
    top_k: int = 5,
    per_cluster: int = 12,
    drop_self: bool = True,
    only_missing_structure: bool = False,
    pident_path: Optional[str | Path] = None,
    id_col: str = "seq_id",
    cluster_cols: Optional[List[str]] = None,
    plddt_col: Optional[str] = None,
    min_plddt: float = 0.0,
    min_identity: float = 0.3,
    lambda_identity: float = 0.30,
    use_query_plddt: bool = True,
    score_plot: Optional[Path] = None,
) -> None:
    lf = _scan_any(table_path)
    cols = lf.collect_schema().names()
    if id_col not in cols:
        raise ValueError(f"Expected id_col '{id_col}' in columns {cols}")

    if cluster_cols is None:
        cluster_cols = [f"cl{lvl}" for lvl in LEVELS]
    _require_cols(cols, [id_col, "has_structure", *cluster_cols])

    pcol = plddt_col or _detect_plddt_col(cols)

    lf = lf.rename({id_col: "seq_id"})
    lf = _normalize_has_structure(lf, "has_structure")

    lf_queries = lf.select(
        pl.col("seq_id"), *(pl.col(c) for c in cluster_cols), pl.col("has_structure"), pl.col(pcol).alias("query_plddt")
    )
    if only_missing_structure:
        lf_queries = lf_queries.filter(~pl.col("has_structure"))

    lf_struct = (
        lf.select(
            pl.col("seq_id"),
            *(pl.col(c) for c in cluster_cols),
            pl.col("has_structure"),
            pl.col(pcol).alias("cand_plddt"),
        )
        .filter(pl.col("has_structure"))
        .filter(pl.col("cand_plddt") >= min_plddt)
    )

    pairs_per_level = []
    for lvl, lvl_col in zip(LEVELS, cluster_cols):
        idx = _build_cluster_index(lf_struct, lvl_col, per_cluster)
        pairs = (
            lf_queries
            .select(
                pl.col("seq_id").alias("query_id"),
                pl.col(lvl_col).alias("cluster"),
                pl.col("query_plddt"),
            )
            .drop_nulls(["cluster"])
            .join(idx, on="cluster", how="left")
            .drop_nulls(["cand_id"])
            .with_columns(
                pl.lit(lvl).alias("closeness"),
                pl.lit(lvl_col).alias("cluster_level"),
            )
            .select(
                "query_id",
                "cand_id",
                "closeness",
                "cand_plddt",
                "cluster_level",
                pl.col("cluster").alias("cluster_label"),
                "query_plddt",
            )
        )
        pairs_per_level.append(pairs)

    pairs_all = pl.concat(pairs_per_level, how="vertical_relaxed")

    if drop_self:
        pairs_all = pairs_all.filter(pl.col("query_id") != pl.col("cand_id"))

    pairs_all = (
        pairs_all
        .sort(["query_id", "cand_id", "closeness", "cand_plddt"], descending=[False, False, True, True])
        .unique(subset=["query_id", "cand_id"], keep="first")
    )

    if pident_path is not None:
        pident = _scan_any(pident_path)
        schema = pident.collect_schema()
        names = schema.names()
        qcol = next((c for c in names if c.lower() in ("query", "qseqid", "query_id")), None)
        tcol = next((c for c in names if c.lower() in ("target", "sseqid", "subject", "target_id")), None)
        picol = next((c for c in names if c.lower() in ("pident", "identity", "pid", "perc_identity")), None)
        if not (qcol and tcol and picol):
            raise ValueError("Could not auto-detect pident columns")
        pident = pident.select(
            pl.col(qcol).alias("query_id"),
            pl.col(tcol).alias("cand_id"),
            pl.col(picol).cast(pl.Float64).alias("pident")
        )
        pairs_all = pairs_all.join(pident, on=["query_id", "cand_id"], how="left")
    else:
        pairs_all = pairs_all.with_columns(pl.lit(None).alias("pident"))

    identity_expr = (
        pl.when(pl.col("pident").is_not_null())
        .then(
            pl.when(pl.col("pident") > 1.5)
            .then(pl.col("pident") / 100.0)
            .otherwise(pl.col("pident"))
        )
        .otherwise(pl.col("closeness") / 100.0)
    ).alias("cand_identity")

    pairs_all = pairs_all.with_columns(identity_expr)

    min_identity_norm = min_identity / 100.0 if min_identity > 1 else min_identity
    pairs_all = pairs_all.filter(pl.col("cand_identity") >= min_identity_norm)

    q_expr = ((pl.col("cand_plddt").clip(50.0, 90.0) - 50.0) / 40.0).clip(0.0, 1.0).alias("Q")
    i_expr = ((pl.col("cand_identity") - 0.30) / 0.30).clip(0.0, 1.0).alias("I")
    pairs_all = pairs_all.with_columns([q_expr, i_expr])

    schema_names = pairs_all.collect_schema().names()
    if use_query_plddt and "query_plddt" in schema_names:
        need_expr = ((80.0 - pl.col("query_plddt")).clip(0.0, 20.0) / 20.0).alias("need")
    else:
        need_expr = pl.lit(1.0).alias("need")
    pairs_all = pairs_all.with_columns(need_expr)

    score_expr = (
        pl.col("Q") + lambda_identity * pl.col("need") * pl.col("I") * (1.0 - pl.col("Q"))
    ).alias("score")
    pairs_all = pairs_all.with_columns(score_expr)

    pairs_all = pairs_all.sort(
        by=["query_id", "score", "cand_plddt", "cand_identity", "closeness"],
        descending=[False, True, True, True, True],
    )

    df_top = (
        pairs_all
        .with_columns(pl.row_number().over("query_id").alias("rank0"))
        .filter(pl.col("rank0") < top_k)
        .with_columns((pl.col("rank0") + 1).alias("rank"))
        .drop("rank0")
        .collect()
    )

    if score_plot is not None:
        import matplotlib.pyplot as plt

        score_plot.parent.mkdir(parents=True, exist_ok=True)
        pdf = df_top.select(["cand_plddt", "cand_identity", "score", "rank"]).to_pandas()
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(
            pdf["cand_plddt"],
            pdf["cand_identity"] * 100.0,
            c=pdf["score"],
            s=20,
            cmap="viridis",
            alpha=0.3,
        )
        top1 = pdf[pdf["rank"] == 1]
        если не top1.empty:
            ax.scatter(top1["cand_plddt"], top1["cand_identity"] * 100.0, c="red", s=35, label="Top-1")
        ax.set_xlabel("Candidate mean pLDDT")
        ax.set_ylabel("Identity (%)")
        ax.set_title("Template score distribution")
        plt.colorbar(sc, ax=ax, label="Score")
        if not top1.empty:
            ax.legend(loc="lower right")
        fig.tight_layout()
        plt.savefig(score_plot, dpi=200)
        plt.close(fig)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        df_top.write_parquet(out_path)
    elif suffix == ".tsv":
        df_top.write_csv(out_path, separator="\t")
    else:
        df_top.write_csv(out_path)


@app.command()
def cli(
    table: str = typer.Argument(..., help="Sequences table (parquet/csv/tsv)."),
    out: str = typer.Argument(..., help="Output path (.parquet/.csv/.tsv)."),
    top_k: int = typer.Option(5, help="Top K hits per sequence."),
    per_cluster: int = typer.Option(10, help="Max candidates retained per cluster."),
    drop_self: bool = typer.Option(True, help="Exclude self-hits."),
    only_missing_structure: bool = typer.Option(False, help="Only compute for sequences without structures."),
    pident: Optional[str] = typer.Option(None, help="Optional pident table to refine ranking."),
    id_col: str = typer.Option("seq_id", help="Name of the sequence ID column in the input."),
    plddt_col: Optional[str] = typer.Option(None, help="Name of pLDDT column (auto-detected if omitted)."),
    c90: str = typer.Option("cl90", help="90% cluster column"),
    c80: str = typer.Option("cl80", help="80% cluster column"),
    c70: str = typer.Option("cl70", help="70% cluster column"),
    c60: str = typer.Option("cl60", help="60% cluster column"),
    c50: str = typer.Option("cl50", help="50% cluster column"),
    min_plddt: float = typer.Option(0.0, help="Minimum candidate pLDDT."),
    min_identity: float = typer.Option(30.0, help="Minimum identity (percent)."),
    lambda_identity: float = typer.Option(0.30, help="Weight for identity term in score."),
    no_query_plddt: bool = typer.Option(False, help="Disable query baseline weighting."),
    score_plot: Optional[str] = typer.Option(None, help="Optional score scatter plot (PNG)."),
) -> None:
    cluster_cols = [c90, c80, c70, c60, c50]
    find_top_k(
        table_path=table,
        out_path=out,
        top_k=top_k,
        per_cluster=per_cluster,
        drop_self=drop_self,
        only_missing_structure=only_missing_structure,
        pident_path=pident,
        id_col=id_col,
        cluster_cols=cluster_cols,
        plddt_col=plddt_col,
        min_plddt=min_plddt,
        min_identity=min_identity,
        lambda_identity=lambda_identity,
        use_query_plddt=not 아직
...
