# Inference, Routing & Homolog Handling

This document covers the Tier 1 inference stack: feature assembly, MiniFold fast
paths, routing heuristics, and homolog co-prediction. Tier 2 and Tier 3
extensions are flagged in each section.

---

## 1. Objectives
- Convert sequences and priors into MiniFold-ready feature tensors.
- Execute MiniFold fast path (distogram→coords or template warp) with telemetry.
- Route sequences through ACCEPT/REFINE/ESCALATE gates using uncertainty metrics.
- Optimise homolog clusters by sharing priors and weakening mutation windows.

---

## 2. Stage [05] FeatureBuilder (La-Proteina)

### Responsibilities
Bucket sequences, compute embeddings, and inject priors into pair features for
MiniFold ingestion.

### Inputs
- `families/<fid>/raw/family.fasta`
- Query-specific priors from Stage [04].
- Bucket configuration `configs/famfold/buckets.yaml`.
- ESM-2 model weights, La-Proteina dataloader.

### Outputs
- In-memory feature tensors per batch (sequence, pair, template channels).
- `embeddings/<bucket>/<qhash>.pt` — cached ESM-2 last-layer features (FP16/BF16).
- `logs/features.jsonl` — cache hit rates, batch composition, timing.

### Workflow
1. Compute `qhash` (BLAKE3 of uppercase sequence); lookup embedding cache.
2. If missing, run ESM-2 (mixed precision) and persist `.pt` file.
3. Bucket sequences into {256, 384, 512} groups with padding masks.
4. Assemble La-Proteina feature dicts (MSA-lite, template placeholders).
5. Inject priors using chosen mode:
   - **Mode A (recommended)**: concatenate `γ * one_hot(P_prior_final)` and
     `W[..., None]` onto pair channels.
   - **Mode B**: add `γ * logit(P_prior_final)` bias to distogram logits.
6. Track per-feature dtype (prefer FP16 pair channels + FP32 master copy).
7. Emit telemetry for cache hits/misses, padding ratio, prior mode usage.

### Sanity Checks
- Pair tensor shapes match bucket dimensions after padding.
- Cache reuse ≥80% for reruns on same family.
- `gamma` range [0, γ_base] after normalisation.

### Tier 2 / Tier 3 Notes
- **Tier 2**: integrate sparse prior loading to reduce memory footprint (load
  top-2 bins only, densify on GPU).
- **Tier 3**: explore learned fusion layers that combine priors with PLM
  features via small attention blocks.

---

## 3. Stage [06] MiniFold Fast Inference & Coordinate Realisation

### Responsibilities
Run MiniFold with prior-injected features, choose between template warp and
standard distogram pipeline, and export coordinates.

### Modes
- **6A Template-Thread & Warp**: triggered when top hit has identity ≥0.85 and
  coverage ≥0.70.
  1. Thread query onto template backbone; copy coordinates for matched regions.
  2. Warp mutation windows (±16 residues) using MiniFold-predicted distograms.
  3. Skip global MDS; perform local relaxation + chirality check.
- **6B Distogram→Coords (default)**:
  1. Run MiniFold fast pass (0 recycles) with prior injection.
  2. Convert distograms to distances via shortest-path completion → classical
     MDS → few LBFGS steps.
  3. Enforce chirality and remove clashes; compute per-residue metrics.

### Outputs
- `inference/coords/jsonl/<qhash>.jsonl`
  - Example record:
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
- Optional exports:
  - `inference/coords/pdb/<qhash>.pdb.gz`
  - `inference/coords/mmcif/<qhash>.cif.gz`
  - `inference/zarr/store.zarr/<qhash>/...` (chunked arrays)
- `logs/inference.jsonl` — batch metrics, path chosen, runtime, memory usage.

### Metrics
- `H_norm`: normalized entropy of predicted distogram.
- `S_prior`: agreement with prior (weighted KL or overlap score).
- `A`: template agreement metric (aligned RMSD proxy).
- `coverage`: retrieval alignment coverage.

### Checklist
1. Implement gating for Mode 6A vs 6B using retrieval metadata.
2. Run MiniFold inference with deterministic seeds; log runtime per batch.
3. Apply chirality and clash checks; store adjustments in telemetry.
4. Emit per-sequence JSONL record and optional structure exports.

### Tier 2 / Tier 3 Notes
- **Tier 2**: integrate optional recycle with uncertainty-aware prior decay for
  borderline ACCEPT/REFINE cases.
- **Tier 3**: experiment with diffusion-based coordinate refinement or hybrid
  MiniFold/ESMFold ensembles for escalations.

---

## 4. Stage [07] Router (Uncertainty-Aware Gating)

### Responsibilities
Decide whether to ACCEPT, REFINE (extra recycle), or ESCALATE based on metrics
from Stage [06].

### Default Policy (configurable via `configs/famfold/thresholds.yaml`)
- **ACCEPT** if `H_norm ≤ 0.25` and `S_prior ≥ 0.35` and (`coverage ≥ 0.70` or
  `A ≥ 0.60`).
- **REFINE** if metrics fall in buffer zones:
  - `0.25 < H_norm ≤ 0.35`
  - `0.50 ≤ coverage < 0.70`
  - `0.40 ≤ A < 0.60`
  - `0.20 ≤ S_prior < 0.35`
  Trigger one recycle with prior decay:
  `conf_ij = 1 - H(P_pred[i,j,:]) / log(64)`
  `gamma_ij ← gamma_ij * (1 - conf_ij)`
- **ESCALATE** if `H_norm > 0.35` or `S_prior < 0.20` or (`coverage < 0.50` and
  `A < 0.40`). Escalate to Tier 2 fallbacks (ESMFold or iterative pipelines);
  see `msa_tier2_plan.md` for the MSA/FoldMason pathway.

### Outputs
- Router decisions appended to `logs/router.jsonl` and batch manifests.
- Updated tensors for recycled passes when REFINE triggers.

### Checklist
1. Implement thresholds as configurable YAML; ensure reproducible gating.
2. Apply uncertainty-aware prior decay before rerunning MiniFold in REFINE.
3. Capture before/after metrics in telemetry for auditing.
4. Write routing decisions into `inference/batches/batch_XXXX.jsonl` manifests.

### Tier 2 / Tier 3 Notes
- **Tier 2**: integrate confidence calibration using calibration curves built
  from pilot datasets; adjust thresholds dynamically per family.
- **Tier 3**: develop learned router (small classifier on metrics + embeddings).

---

## 5. Stage [08] Homolog Co-Prediction (Optional in Tier 1)

### Responsibilities
Cluster high-identity sequences and reuse priors to accelerate inference.

### Inputs
- `retrieval/topk.parquet` (identity graph) or precomputed clusters.
- Outputs from Stage [06] for representative sequences.

### Procedure
1. Cluster sequences with identity ≥0.85 (single-linkage or union-find).
2. Select representative per cluster (highest coverage/identity pair).
3. Fold representative via standard routing (Stages [05–07]).
4. For homologs:
   - Define mutation windows (±16) relative to representative alignment.
   - Set `gamma = 0.4 * gamma` inside mutation windows; retain elsewhere.
   - Run MiniFold fast pass (0 recycles); only recycle if window entropy remains
     high (>0.30 normalized).
5. Record delta maps (per-residue RMSD vs representative) under `reports/`.

### Outputs
- `reports/homolog_groups/<cluster_id>.json` (Tier 2 optional)
- Telemetry appended to `logs/router.jsonl` and `logs/inference.jsonl`.

### Tier 2 / Tier 3 Notes
- **Tier 2**: formalise homolog manifests, share caches across families, and
  surface aggregated metrics in dashboards.
- **Tier 3**: research joint folding of homolog clusters using multi-sequence
  priors or lightweight equivariant networks.

---

Refer to [operations_validation.md](operations_validation.md) for orchestration,
logging, testing, and CLI surfaces supporting these stages.
