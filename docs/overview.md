# FamilyFold Implementation Overview

This overview restores the full Tier 1 blueprint for FamilyFold while pointing to
dedicated documents for each major component. Tier 2 and Tier 3 remain notes for
forward planning and are captured alongside the Tier 1 references below.

## Tiering Summary

- **Tier 1 – Core Pipeline**: Fully specified build for template-supported
  families, including data ingestion, priors, retrieval, MiniFold integration,
  routing, and operational telemetry. Implementation guidance is split across
  component-specific documents for faster navigation:
  - [Data Foundations & Template Tooling](data_foundations.md)
  - [Retrieval & Alignment-Informed Priors](retrieval_priors.md)
  - [Inference, Routing & Homolog Handling](inference_routing.md)
  - [Operations, Tooling & Validation](operations_validation.md)
- **Tier 2 – Extended Capabilities (notes)**: Homolog co-prediction rollouts
  beyond Tier 1 defaults, richer caching, dataset-wide dashboards, and expanded
  orchestration (see notes inside each component document for follow-ups). See
  also the [MSA & FoldMason escalation plan](msa_tier2_plan.md) for the
  default-off alignment pathway.
- **Tier 3 – Stretch / Research (notes)**: Experimental inference variants,
  large-scale optimisation projects, and deep analytics ideas for when Tier 1 and
  Tier 2 are stable.

## Objectives & Scope
- Deliver **fast, bulk monomer** predictions for protein families (≈300–500 AA)
  while preserving accuracy.
- Reuse high-confidence family structures with per-residue **pLDDT**.
- Integrate lightweight retrieval (Protriever + Easy ESM-RAG fallback) to
  specialise priors per query.
- Inject **family-derived, pLDDT-weighted geometric priors**, sharpened via
  alignments (AIP) and routed through MiniFold fast paths.
- Avoid standalone MSAs at inference unless Tier 2 escalation toggles (e.g.,
  FoldMason/MAFFT for problematic families; see `msa_tier2_plan.md`).

## Inputs & Outputs
- **Inputs**: family FASTA, reference structures (AFDB/experimental), optional
  seed alignments (Tier 2).
- **Outputs**: predicted structures (PDB/mmCIF), per-sequence telemetry
  (`H_norm`, `S_prior`, `A`, coverage), manifests summarising template usage and
  routing decisions.

## High-Level Pipeline
1. **TemplatePrep** — parse structures → `{seq, pLDDT[L], distogram bins[L×L]}`
   plus metadata. (See [data foundations](data_foundations.md)).
2. **FamilyPrior** — aggregate templates into `P_family`/`W_family` (dense +
   sparse). (See [data foundations](data_foundations.md)).
3. **Retrieval** — Protriever/Easy ESM-RAG retrieval, alignments, clustering.
   (See [retrieval & priors](retrieval_priors.md)).
4. **Alignment-Informed Prior** — sharpen priors per query using alignments,
   pLDDT, and identity. (See [retrieval & priors](retrieval_priors.md)).
5. **FeatureBuilder** — assemble MiniFold features with prior injection. (See
   [inference stack](inference_routing.md)).
6. **MiniFold Fast Inference** — choose template warp vs distogram→coords, or
   the requested flow-matching path from
   [OpenReview 5zAde2jch7](https://openreview.net/pdf?id=5zAde2jch7); export
   structures. (See [inference stack](inference_routing.md)).
7. **Router** — ACCEPT/REFINE/ESCALATE decisions with uncertainty-aware decay.
   (See [inference stack](inference_routing.md)).
8. **Homolog Co-Prediction** — share priors across ≥85% identity clusters (Tier 1
   optional). (See [inference stack](inference_routing.md)).
9. **Batch Orchestration & Reporting** — deterministic manifests, DuckDB, tests.
   (See [operations & validation](operations_validation.md)).

## System Architecture & Module Graph

```
family.fasta, reference_structures/
          │
          ▼
[01] TemplatePrep ──► templates/*.json, template_bank.faa
          │
          ▼
[02] FamilyPrior ──► priors/distogram_{bucket}.{npz,sparse}
          │
          ▼
[03] Retrieval (Protriever/ESM-RAG) ──► topk.parquet, alignments/*.json
          │
          ▼
[04] Alignment-Informed Prior ──► query-specific P_prior, W, γ
          │
          ▼
[05] FeatureBuilder ──► pair features + prior channels
          │
          ▼
[06] MiniFold Fast Inference ──► distogram logits, coordinates, metrics
          │
          ▼
[07] Router ──► ACCEPT / REFINE / ESCALATE
          │
          ▼
[08] Homolog Co-Prediction ──► mutation-aware batches (optional)
          │
          ▼
[09] Orchestration & Reporting ──► manifests, dashboards, audits
```

## Directory Layout (Reference)

```
familyfold/
  configs/
    famfold/
      default.yaml
      retrieval.yaml
      thresholds.yaml
      paths.yaml
      buckets.yaml
  docs/
    overview.md
    data_foundations.md
    retrieval_priors.md
    inference_routing.md
    operations_validation.md
    SCHEMAS.md
    PARAMS_AIP.md
  families/
    <family_id>/
      raw/
        family.fasta
        structures/
          <template_id>.pdb.gz   # AFDB PDBs (B-factor = pLDDT)
      templates/
        index.tsv
        <template_id>.json.gz
        qc_report.json
      priors/
        distogram_256.npz
        distogram_256_sparse.npz
        distogram_384.npz
        distogram_384_sparse.npz
        distogram_512.npz
        distogram_512_sparse.npz
        metadata.json
      retrieval/
        template_bank.faa
        template_esmf.npz
        topk.parquet
        alignments/
          <qhash>__<tid>.aln.json
      embeddings/
        256/
          <qhash>.pt
        384/
          <qhash>.pt
        512/
          <qhash>.pt
      inference/
        batches/
          batch_0001.jsonl
        coords/
          jsonl/
            <qhash>.jsonl
          pdb/
            <qhash>.pdb.gz
          mmcif/
            <qhash>.cif.gz
        zarr/
          store.zarr/
      logs/
        template_prep.jsonl
        family_prior.jsonl
        retrieval.jsonl
        aip.jsonl
        features.jsonl
        inference.jsonl
        router.jsonl
      manifests/
        run_manifest.json
        dataset.duckdb
      reports/
        pilot_metrics.parquet
        dashboards.md
  tools/
  tests/
    toy_family/
      raw/
        family.fasta
        structures/...
      expected/...
```

**Naming Rules**

- `family_id`: lowercase `[a-z0-9_-]+` (e.g., `gh5_21`).
- `template_id`: `UNIPROTID_CHAIN` if available; fallback to file stem.
- `qhash`: BLAKE3 of uppercase AA sequence; capture mapping in
  `manifests/run_manifest.json`.
- Buckets: 256, 384, 512 AA (configurable via `configs/famfold/buckets.yaml`).

## Parameter Crib Sheet (Tier 1 Defaults)
- `β` (pair pLDDT exponent): 1.5 for query-specific weights, 1.0 for global
  prior aggregation.
- `α` (identity exponent): 1.0.
- σ schedule (see `docs/PARAMS_AIP.md`): ≥90 pLDDT → 1.5 Å; 70–90 → 2.5 Å;
  <70 → 3.5–5.0 Å plus identity/gap penalties.
- `λ` (mix weight with `P_family`): ≈0.6 at 50% identity, drop toward 0.3–0.4 as
  identity rises.
- Context penalties: `gap_penalty = 0`, `window_penalty = 0.5` (±16),
  `sep_penalty = 0.7` for `|i-j| ≥ 24` when consensus low.
- `γ_base = 0.7` (scale 0.5–0.8 depending on routing results).

---

For implementation-ready guidance, dive into the component documents linked
above. Each file contains Tier 1 checklists, schemas, and validation steps plus
Tier 2/Tier 3 follow-up notes relevant to that area.
