# FamilyFold Implementation Overview

> **Note:** This document mirrors `docs/overview.md` so downstream branches referencing the legacy path can merge without conflicts. Updates should be made in both files.


This overview restores the full Tier 1 blueprint for FamilyFold while pointing to
dedicated documents for each major component. Tier 2 and Tier 3 remain notes for
forward planning and are captured alongside the Tier 1 references below.

## Tiering Summary

- **Tier 1 – Core Pipeline**: Fully specified build for template-supported
  families, including data ingestion, priors, retrieval, MiniFold integration,
  routing, and operational telemetry. Implementation guidance is split across
  component-specific documents for faster navigation:
  - [Data Foundations & Template Tooling](familyfold_tier1_data.md)
  - [Retrieval & Alignment-Informed Priors](familyfold_tier1_retrieval.md)
  - [Inference, Routing & Homolog Handling](familyfold_tier1_inference.md)
  - [Operations, Tooling & Validation](familyfold_tier1_ops.md)
- **Tier 2 – Extended Capabilities (notes)**: Homolog co-prediction rollouts
  beyond Tier 1 defaults, richer caching, dataset-wide dashboards, and expanded
  orchestration (see notes inside each component document for follow-ups).
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
  FoldMason/MAFFT for problematic families).

## Inputs & Outputs
- **Inputs**: family FASTA, reference structures (AFDB/experimental), optional
  seed alignments (Tier 2).
- **Outputs**: predicted structures (PDB/mmCIF), per-sequence telemetry
  (`H_norm`, `S_prior`, `A`, coverage), manifests summarising template usage and
  routing decisions.

## High-Level Pipeline
1. **TemplatePrep** — parse structures → `{seq, pLDDT[L], distogram bins[L×L]}`
   plus metadata. (See [data foundations](familyfold_tier1_data.md)).
2. **FamilyPrior** — aggregate templates into `P_family`/`W_family` (dense +
   sparse). (See [data foundations](familyfold_tier1_data.md)).
3. **Retrieval** — Protriever/Easy ESM-RAG retrieval, alignments, clustering.
   (See [retrieval & priors](familyfold_tier1_retrieval.md)).
4. **Alignment-Informed Prior** — sharpen priors per query using alignments,
   pLDDT, and identity. (See [retrieval & priors](familyfold_tier1_retrieval.md)).
5. **FeatureBuilder** — assemble MiniFold features with prior injection. (See
   [inference stack](familyfold_tier1_inference.md)).
6. **MiniFold Fast Inference** — choose template warp vs distogram→coords,
   export structures. (See [inference stack](familyfold_tier1_inference.md)).
7. **Router** — ACCEPT/REFINE/ESCALATE decisions with uncertainty-aware decay.
   (See [inference stack](familyfold_tier1_inference.md)).
8. **Homolog Co-Prediction** — share priors across ≥85% identity clusters (Tier 1
   optional). (See [inference stack](familyfold_tier1_inference.md)).
9. **Batch Orchestration & Reporting** — deterministic manifests, DuckDB, tests.
   (See [operations & validation](familyfold_tier1_ops.md)).

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
    familyfold_plan.md
    familyfold_tier1_data.md
    familyfold_tier1_retrieval.md
    familyfold_tier1_inference.md
    familyfold_tier1_ops.md
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
# FamilyFold Bulk Prediction Pipeline Plan

Absolutely. Below is a fully spelled out implementation plan with a concrete folder layout, file names, schemas, and checklists you can literally copy into your repo and check off as you build. It integrates everything we discussed: MiniFold fast path, Protriever retrieval (FAISS optional) with an Easy ESM-RAG fallback, pLDDT-weighted and Alignment-Informed Prior (AIP), and homolog co-prediction logic. It assumes 2–10 templates ~50% identity with pLDDT in the PDB B-factor (AFDB).

---

## 📁 Project Layout (top level)

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
    familyfold_plan.md
    SCHEMAS.md
    PARAMS_AIP.md
  families/
    <family_id>/
      raw/
        family.fasta
        structures/
          <template_id>.pdb.gz   # AFDB PDBs (B-factor = pLDDT)
          ...
      templates/
        index.tsv
        <template_id>.json.gz    # per-template features (schema below)
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
        template_esmf.npz        # optional (Easy ESM-RAG)
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
          ...
        coords/
          jsonl/
            <qhash>.jsonl        # compressed coords + metrics
          pdb/
            <qhash>.pdb.gz
          mmcif/
            <qhash>.cif.gz
        zarr/
          store.zarr/            # optional; chunked storage
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
  tools/                          # CLI entrypoints (no code here now; names only)
  tests/
    toy_family/
      raw/
        family.fasta
        structures/...
      expected/...
```

**Naming rules**

- `family_id`: lowercase `[a-z0-9_-]+` (e.g., `gh5_21`).
- `template_id`: `UNIPROTID_CHAIN` if available (fallback: file stem).
- `qhash`: BLAKE3 of uppercase AA sequence (store a mapping table in `manifests/run_manifest.json`).
- Buckets: 256, 384, 512 AA (configurable in `configs/famfold/buckets.yaml`).

---

## ✅ Build Plan (stage-by-stage checklists)

### 0) Environment & Config

**Files**

- `configs/famfold/paths.yaml` – absolute/relative paths per cluster/host.
- `configs/famfold/default.yaml` – global knobs (γ, β, σ, λ, k, etc.).
- `configs/famfold/thresholds.yaml` – router gates (accept/refine/escalate).
- `configs/famfold/buckets.yaml` – bucketing and length cutoffs.
- `manifests/run_manifest.json` – single source of truth for this run: hashes, versions, timestamps.

**Checklist**

- Create `configs/famfold/paths.yaml` with `data_root`, `cache_root`, `zarr_root`.
- Fill `default.yaml` with defaults (`γ_base=0.7`, `β=1.5`, `α=1.0`, `k=6`).
- Fill `thresholds.yaml` (Accept ≤0.25 entropy; refine ≤0.35; escalate else).
- Fill `buckets.yaml` (256/384/512; pad rules).
- Write `manifests/run_manifest.json` with `family_id`, versions, `created_at`.

---

### 1) TemplatePrep (structures → per-template features)

**Inputs**

- `families/<family_id>/raw/structures/*.pdb.gz`
  - AFDB PDBs: B-factor = pLDDT; for experimental PDBs, treat B-factor as temperature factor, not pLDDT.

**Outputs**

- `families/<fid>/templates/index.tsv`
  - Columns: `template_id, chain, L, mean_plddt, p5_plddt, p95_plddt, source_file`
- `families/<fid>/templates/<template_id>.json.gz`
  - Schema:

    ```json
    {
      "template_id": "P12345_A",
      "L": 436,
      "seq": "MKV...",
      "plddt": [float, ... L],
      "bins": "uint16 base64 of [L,L]",   // 64 bins (2–25 Å; ≥25 Å last bin)
      "bin_edges": [64 floats],           // shared across templates, store once in priors/metadata.json too
      "meta": {"source": "AFDB", "pdb_path": ".../P12345_A.pdb.gz", "date": "YYYY-MM-DD"}
    }
    ```

- `families/<fid>/templates/qc_report.json` – basic stats (counts, drops).

**Checklist**

- Parse sequence & Cα; compute pair distances; bin into 64.
- Extract pLDDT from B-factor (AFDB PDB).
- Skip templates with mean pLDDT < 70; log in `qc_report.json`.
- Write `index.tsv` & individual `*.json.gz` files.
- Log to `logs/template_prep.jsonl` (one line per template with timings).

**Sanity**

- Verify bins symmetric, diagonal handled, `plddt` length == `L`.
- Spot-check 3 templates manually.

---

### 2) FamilyPrior (aggregate into reusable prior)

**Inputs**

- `templates/*.json.gz` (2–10 templates expected)

**Outputs**

- `priors/distogram_{256|384|512}.npz` – dense tensors:
  - `P_family [L_b, L_b, 64]` (float32) – probabilities or logits
  - `W_family [L_b, L_b]` (float16) – aggregated strength
  - `bin_edges [64]` (float32) – (2..25 Å)
- `priors/distogram_{bucket}_sparse.npz` – COO top-2 bins/pair:
  - `rows, cols, bin_ids[2], bin_probs[2]`
- `priors/metadata.json` – weight histograms, template contributions, provenance.

**Weighting (global, template-agnostic)**

\[
W^{\text{global}}_{t}[i,j] = \left(\frac{\text{plddt}_i \times \text{plddt}_j}{10{,}000}\right)^{\beta} \times \text{coverage}_{ij}, \quad \beta = 1.0
\]

\[
\text{logits}_{\text{family}}[i,j,b] = \sum_t W^{\text{global}}_{t}[i,j] \times \mathbf{1}[b = \text{bin}_t(i,j)]
\]

\[
P_{\text{family}} = \operatorname{softmax}_b(\text{logits}_{\text{family}}), \quad W_{\text{family}}[i,j] = \sum_t W^{\text{global}}_{t}[i,j]
\]

**Checklist**

- Compute `W_t_global` with β = 1.0; aggregate into `P_family`, `W_family`.
- Emit dense & sparse per bucket (256/384/512).
- Fill `metadata.json` with summary stats.
- Log to `logs/family_prior.jsonl`.

**Sanity**

- Dense ↔ sparse round-trip within tolerance.
- Heatmaps of `W_family` look reasonable (report).

---

### 3) Retrieval (Protriever) + Easy ESM-RAG fallback

**Inputs**

- `raw/family.fasta` (all sequences to fold)
- `retrieval/template_bank.faa` (built from templates)

**Outputs**

- `retrieval/template_bank.faa` – concatenated template sequences (FASTA).
- *(Optional)* `retrieval/template_esmf.npz` – `{ ids: [str], E: float32[NT,d] }`
- `retrieval/topk.parquet` – columns: `qhash, qlen, t_id, trank, identity, coverage, bitscore, sim, tlen, method`
- `retrieval/alignments/<qhash>__<tid>.aln.json` – alignment map per hit:

  ```json
  {
    "qhash": "...",
    "t_id": "P12345_A",
    "q_to_t": [int or -1 per qpos],
    "t_to_q": [int or -1 per tpos],
    "mismatch_windows": [[s, e], ...],
    "gaps": [[qpos, tpos], ...],
    "identity": 0.52,
    "coverage": 0.73,
    "method": "protriever|esm_rag|hmm"
  }
  ```

**Checklist**

- Build `template_bank.faa` from `templates/index.tsv`.
- Run Protriever k-NN (`k=6` default) → `topk.parquet` with identity/coverage/bitscore.
- *(Optional)* Build `template_esmf.npz` and wire Easy ESM-RAG fallback.
- Generate per-hit alignments (`alignments/*.aln.json`) via Protriever/HMMER or Needleman–Wunsch if needed.
- Log to `logs/retrieval.jsonl`.

**Sanity**

- Top-k stable across re-runs (seed fixed).
- Identity/coverage monotonicity checks.

---

### 4) AIP – Alignment-Informed Prior (query-specific sharpening)

**Inputs**

- `priors/distogram_*.{npz}`, `retrieval/topk.parquet`, `alignments/*.aln.json`
- `templates/*.json.gz` (bins, pLDDT)

**Outputs** (per query; transient or cached)

- `aip/<qhash>/P_prior_final.npz` – optional cache
- `P_prior_final [L, L, 64]` (float16 OK), `W [L, L]` (float16), `gamma [L, L]` (float16)
- Values are also passed directly to FeatureBuilder during batch inference.

**Computation**

For each template hit `t`:

\[
 c_i = \frac{\text{plddt}_t[i]}{100}, \quad c_j = \frac{\text{plddt}_t[j]}{100}
\]
\[
 W_t[i,j] = (c_i c_j)^{\beta} \times \text{coverage}_{ij} \times \text{identity}_{t \rightarrow q}^{\alpha} \times \text{context}(i, j) \quad (\beta = 1.5, \alpha = 1.0)
\]

- Narrow Gaussian over bins around template distance at aligned pairs:

  \[
  \mu = \text{center\_of\_bin}(\text{bin}_t[u,v]), \quad \sigma = \sigma_{\text{plddt}} + \sigma_{\text{id}} + \sigma_{\text{gap}}
  \]

  \[
  G[b] \propto \exp\left(-\frac{(\text{center}[b] - \mu)^2}{2\sigma^2}\right)
  \]

- Mixture with global prior:

  \[
  \lambda = \lambda_0 \times (1 - \text{local\_mutation\_density}) \quad \text{with } \lambda_0 \approx 0.6 \text{ at } \sim50\% \text{ ID}
  \]

  \[
  \text{logits}_{\text{prior}_t}[i,j,:] = \lambda \cdot \log P_{\text{family}}[i,j,:] + (1-\lambda) \cdot \log G[:]
  \]

- Fuse across templates:

  \[
  \text{logits}_{\text{mix}}[i,j,:] = \sum_t W_t[i,j] \cdot \log P_{\text{prior}_t}[i,j,:]
  \]

  \[
  P_{\text{prior\_final}} = \operatorname{softmax}_b(\text{logits}_{\text{mix}})
  \]

  \[
  W[i,j] = \sum_t W_t[i,j], \quad \gamma[i,j] = \gamma_{\text{base}} \times \text{normalize}(W[i,j]) \quad (\gamma_{\text{base}} = 0.7; \text{per-matrix p5–p95 min–max})
  \]

- Conflict rule (long-range): if `consensus_low(i,j)` and `|i−j| ≥ 24`, widen σ by +1 Å and halve local `W_t`.

**Checklist**

- Implement the per-hit `W_t` with pLDDT and context penalties.
- Implement Gaussian `G[b]` and mixing with global prior (`λ`).
- Aggregate across templates → `P_prior_final`, `W`, `γ`.
- *(Optional)* Cache to `aip/<qhash>/...`; always log to `logs/aip.jsonl`.

**Sanity**

- Aligned, high-pLDDT regions: narrow peaks.
- Mutation windows/gaps: widened σ, reduced `W_t`.

---

### 5) FeatureBuilder (La-Proteina dataloader + prior injection)

**Inputs**

- Sequences (from `raw/family.fasta`) batched by bucket.
- `P_prior_final`, `W`, `γ` (from AIP).

**Outputs**

- (In-memory) MiniFold pair features augmented with prior channels.
- Caches: `embeddings/<bucket>/<qhash>.pt` (ESM-2 last-layer, FP16/BF16).

**Injection modes**

- **Mode A (recommended)**: concat `γ * one_hot(P_prior_final) + W[..., None]` into pair channels.
- **Mode B (logit-bias)**: add `γ * logit(P_prior_final)` to initial distogram logits.

**Checklist**

- Bucket sequences (256/384/512).
- Compute lazy ESM-2 embeddings; write cache `.pt` by `qhash`.
- Assemble pair features; inject prior with chosen mode.
- Log to `logs/features.jsonl`.

**Sanity**

- Shapes per bucket; caching hit rates.

---

### 6) MiniFold Fast Inference & Coordinates

**Inputs**

- Pair features (with prior), retrieval metadata (identity/coverage).

**Outputs**

- `inference/coords/jsonl/<qhash>.jsonl` – compressed coords + metrics. Record (one line per sequence):

  ```json
  {
    "qhash": "...",
    "L": 432,
    "runtime_s": 0.71,
    "path": "6B",
    "H_norm": 0.22,
    "S_prior": 0.41,
    "A": 0.66,
    "coverage": 0.74,
    "route": "ACCEPT",
    "neighbors": [{"t_id": "P12345_A", "identity": 0.53, "coverage": 0.74}],
    "export": {"pdb": "inference/coords/pdb/<qhash>.pdb.gz"}
  }
  ```

- Optionally `inference/zarr/store.zarr/<qhash>/...` for chunked arrays.
- Exports on demand: PDB/mmCIF under `inference/coords/pdb/` or `mmcif/`.

**Paths**

- **6A Template-Thread & Warp** (identity ≥ 0.85 & coverage ≥ 0.7): thread on best template; warp only mutation windows via predicted distograms; skip global MDS.
- **6B Distogram→Coords** (default): MiniFold fast pass (recycles = 0) → MDS (shortest-path completion → classical MDS → a few LBFGS steps) → chirality check.

**Checklist**

- Implement 6A gate; 6B default.
- Compute metrics: `H_norm`, `S_prior`, `A`, coverage.
- Write per-sequence JSONL; optionally export PDB/mmCIF.
- Log to `logs/inference.jsonl`.

**Sanity**

- Chirality stable; 6A warps only within windows.

---

### 7) Router (uncertainty-aware gating)

**Inputs**

- Metrics from stage 6: `H_norm`, `S_prior`, `A`, coverage.

**Policy** (defaults; tune in `configs/famfold/thresholds.yaml`)

- **ACCEPT** if `H_norm ≤ 0.25` and `S_prior ≥ 0.35` and (`coverage ≥ 0.70` or `A ≥ 0.60`).
- **REFINE** if any buffer: `0.25 < H_norm ≤ 0.35` or `0.50 ≤ coverage < 0.70` or `0.40 ≤ A < 0.60` or `0.20 ≤ S_prior < 0.35`.
  - Run one recycle with uncertainty-aware prior decay:

    \[
    \text{conf}_{ij} = 1 - \frac{H(P_{\text{pred}}[i,j,:])}{\log 64}, \quad \gamma_{ij} \leftarrow \gamma_{ij} \times (1 - \text{conf}_{ij})
    \]

- **ESCALATE** if `H_norm > 0.35` or `S_prior < 0.20` or (`coverage < 0.50` and `A < 0.40`).

**Outputs**

- Decisions recorded in `inference/batches/batch_*.jsonl` and `logs/router.jsonl`.

**Checklist**

- Implement gates from `thresholds.yaml`.
- Implement decay on recycle; re-run & re-score.
- Log decisions and deltas.

---

### 8) Homolog Co-Prediction (≥85% identity; optional)

**Inputs**

- `retrieval/topk.parquet` (or a prebuilt identity graph).

**Outputs**

- Grouped predictions with delta maps under `reports/`.

**Procedure**

- Cluster at ≥0.85 identity; choose a representative.
- Fold representative (6A/6B), then specialize homologs:
  - Build mutation windows (±16).
  - Set `γ = 0.4 * γ` inside windows, `γ = 1.0 * γ` elsewhere.
  - Run fast pass; recycle only if window entropy is high.
- Group routing: accept all if representative passes and others only show local entropy spikes.

**Checklist**

- Cluster & select representatives.
- Implement window weakening & per-homolog run.
- Log time/seq; generate delta maps.

---

### 9) Batch Orchestration, Manifests & Reports

**Artifacts**

- `inference/batches/batch_XXXX.jsonl` – all per-sequence records.
- `manifests/dataset.duckdb` – tables: sequences, templates, hits, metrics, routes.
- `reports/pilot_metrics.parquet` – rollups for dashboards.

**Checklist**

- Deterministic bucketing; dynamic batch size if OOM.
- Append-safe JSONL logs (one line per sequence).
- Create DuckDB with views for quick queries.
- Basic dashboards in `reports/dashboards.md`.

---

## 📚 SCHEMAS (put detailed versions in `docs/SCHEMAS.md`)

### `templates/<template_id>.json.gz`

- `seq` (str), `L` (int)
- `plddt` `[L]` float32 in [0, 100] (AFDB PDB B-factor)
- `bins` `[L, L]` uint16 (0..63)
- `bin_edges` `[64]` float32 (2..25 Å)
- `meta` (dict)

### `priors/distogram_*.npz`

- `P_family [L_b, L_b, 64]` float32
- `W_family [L_b, L_b]` float16
- `bin_edges [64]` float32

### `retrieval/topk.parquet`

- `qhash: str`, `qlen: int`, `t_id: str`, `trank: int`, `identity: float?`, `coverage: float?`, `bitscore: float?`, `sim: float?`, `tlen: int`, `method: {protriever, esm_rag, hmm}`

### `retrieval/alignments/*.aln.json`

- `q_to_t`: `[int or -1] * L_q`
- `t_to_q`: `[int or -1] * L_t`
- `mismatch_windows`: `[[start, end], ...]` (0-based, inclusive)
- `gaps`: `[[qpos, tpos], ...]`
- `identity`, `coverage`, `method`

### `inference/coords/jsonl/<qhash>.jsonl`

- Keys shown in stage 6; include route, metrics, neighbors.

---

## 🔧 Parameter crib sheet (put in `docs/PARAMS_AIP.md`)

- `β` (pair pLDDT exponent): 1.5 (query-specific), 1.0 (global prior)
- `α` (identity exponent): 1.0
- `σ` by pLDDT (min of pair): ≥90 → 1.5 Å; 70–90 → 2.5 Å; <70 → 3.5–5.0 Å
- `σ` by identity (additive): ≥0.80 → +0.0 Å; 0.65–0.80 → +0.5 Å; 0.50–0.65 → +1.0 Å
- `λ` (mix weight with `P_family`): ~0.6 at ~50% ID; drop to 0.3–0.4 as ID ↑
- Context penalties: `gap_penalty = 0`, `window_penalty = 0.5` (±16), `sep_penalty = 0.7` for `|i−j| ≥ 24` if consensus low
- `γ_base`: 0.7 (tune 0.5–0.8)

---

## 🧪 Iterative test plan (boxes you can tick)

### Unit

- **TemplatePrep**: bins symmetric; `plddt` length == `L`; mean pLDDT filter works.
- **FamilyPrior**: dense ↔ sparse round-trip; weights sane in `metadata.json`.
- **Retrieval**: top-k stable; identity/coverage monotonicity.
- **Alignments**: `q_to_t`/`t_to_q` mapping sums match identities; windows computed.
- **AIP**: high-pLDDT regions narrow; mutation windows widened; conflict rule triggers.
- **FeatureBuilder**: ESM-2 cache hit/miss; pair shapes correct.
- **MiniFold engine**: determinism; chirality check; 6A modifies only windows.
- **Router**: gates produce expected decisions; prior decay reduces γ where entropy low.

### Integration (toy family)

- Baseline (no prior) → acceptance/time/entropy recorded.
- FamilyPrior → entropy ↓, acceptance ↑.
- AIP → further entropy ↓, fewer recycles.
- 6A on ≥85% → speed-up; RMSD outside windows ≪ 1 Å.
- Router end-to-end → escalation ≤ target.

### Pilot (1–5k seqs)

- SLOs met: acceptance ≥ target; refine ≤ 15%; escalate ≤ 5%.
- Throughput on accepted ≥ 10× ESMFold.
- Chirality flips < 1%; mem OK at 512 AA.

### Stress

- Long sequences (≥500 AA) survive with auto batch shrink.
- Sparse-template families escalate quickly (no wasted recycles).
- Dense homolog clusters show consensus-then-specialize speed-ups.

---

## 🧭 Practical notes on key questions

1. **Use alignment to the structure that gave the distogram?** Yes (AIP). At ~50% identity, leveraging the query↔template alignment to narrow the prior around template distances (with width set by pLDDT + identity + local context) gives a crisper starting distogram. We explicitly materialize that via `alignments/*.aln.json` + the AIP mixture into `P_prior_final`. This is absolutely worth it.
2. **When to use FoldMason/MAFFT?** Keep off by default. Turn on only if templates show multi-modal long-range conflicts (e.g., >25% of non-loop pairs disagree). If on, store `msa_map` in each template JSON and use those to align columns consistently; the rest of the plan doesn’t change.

---

## 🛠️ CLI skeleton (Typer commands, no implementation yet)

```
famfold template-prep --family <fid>
famfold build-prior   --family <fid>
famfold retrieve      --family <fid> --mode protriever|esm-rag --k 6
famfold build-aip     --family <fid> --batch <ids.txt>         # optional cache
famfold infer         --family <fid> --bucket {256,384,512} --mode {6A,6B,auto}
famfold route         --family <fid> --thresholds configs/famfold/thresholds.yaml
famfold export        --family <fid> --format {pdb,mmcif,zarr}
famfold report        --family <fid> --out reports/pilot_metrics.parquet
```

This mirrors the stages and ensures Hydra configs align with Typer entry points.

---

## ✅ Definition of Done (summary)

- Accuracy: median lDDT within 3 pts of ESMFold for ACCEPTed sequences.
- Throughput: ≥10× faster than ESMFold on ACCEPTed sequences; REFINE ≤ 15%, ESCALATE ≤ 5%.
- Stability: chirality flips < 1%, deterministic reruns (hash-stable artifacts).
- Homolog batching: ≥85% identity clusters show runtime reduction and mutation-localized geometry deltas.
- Operational readiness: Hydra/Typer CLI, manifests, dashboards, and validation commands documented and automated.

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
