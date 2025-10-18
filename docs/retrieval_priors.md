# Retrieval & Alignment-Informed Priors

This document details retrieval, alignment generation, and query-specific prior
sharpening (AIP) for Tier 1. Tier 2 and Tier 3 opportunities are captured at the
end of each section.

---

## 1. Objectives
- Discover top templates per query sequence using Protriever (or Easy ESM-RAG).
- Materialise per-hit alignments and scoring metadata.
- Sharpen the family prior into query-specific distograms using alignments,
  pLDDT, and identity-aware weighting.

---

## 2. Stage [03] Retrieval (Protriever / Easy ESM-RAG)

### Responsibilities
Embed queries, find nearest templates, compute identity/coverage/bitscore, and
produce alignment-ready artifacts.

### Inputs
- `families/<fid>/raw/family.fasta`
- `families/<fid>/retrieval/template_bank.faa`
- *(Optional)* `retrieval/template_esmf.npz` (precomputed template embeddings)
- Configs: `configs/famfold/retrieval.yaml` (k, encoder choice, FAISS options)

### Outputs
- `retrieval/topk.parquet`
  - Columns: `qhash, qlen, t_id, trank, identity, coverage, bitscore, sim, tlen, method`
- `retrieval/alignments/<qhash>__<tid>.aln.json`
  - Example structure:
    ```json
    {
      "qhash": "...",
      "t_id": "P12345_A",
      "q_to_t": [int or -1],
      "t_to_q": [int or -1],
      "mismatch_windows": [[start, end], ...],
      "gaps": [[qpos, tpos], ...],
      "identity": 0.52,
      "coverage": 0.73,
      "method": "protriever"
    }
    ```
- `retrieval/template_esmf.npz` *(optional)* — `{ "ids": [...], "E": float32[NT, d] }`
- `logs/retrieval.jsonl` — timing, encoder details, fallback usage.

### Checklist
1. Generate/query embeddings with ESM-2 (FP16/BF16) and persist per-bucket caches.
2. Build FAISS index (inner-product/cosine) or use brute-force similarity when
   dataset is small.
3. Retrieve top-`k` hits (default `k=6`). Include self-hits, coverage masks, and
   identity thresholds.
4. Record method (`protriever`, `esm_rag`, `hmm`) per hit for downstream gating.
5. Write `topk.parquet` with deterministic ordering (seeded retrieval).
6. Generate alignments via Protriever’s traceback or fallback to Needleman–Wunsch
   when aligner metadata is unavailable.
7. Populate `alignments/*.aln.json` with mismatch windows (±16) and gap loci.
8. Append telemetry to `logs/retrieval.jsonl` (one JSON per query).

### Sanity Checks
- Monotonic identity/coverage across `trank`.
- Alignment mappings invert correctly (`q_to_t` and `t_to_q`).
- Top-`k` stable across reruns with fixed seeds.

### Tier 2 / Tier 3 Notes
- **Tier 2**: pre-build global FAISS indices per taxonomic bucket; cache
  retrieval features across families.
- **Tier 3**: experiment with hybrid PLM + structural similarity metrics or
  diffusion-based template search for low-identity queries.

---

## 3. Stage [04] Alignment-Informed Prior (AIP)

### Responsibilities
Specialise the family prior (`P_family`, `W_family`) into per-query tensors by
combining alignments, pLDDT, and identity-aware weights.

### Inputs
- Family priors from Stage [02].
- `retrieval/topk.parquet` rows per query.
- `retrieval/alignments/*.aln.json` mappings.
- Template JSON blobs (for per-residue pLDDT and distogram bins).
- Configs: `docs/PARAMS_AIP.md` (σ schedules, λ mix weights, context penalties).

### Outputs
- `aip/<qhash>/P_prior_final.npz` *(optional cache)*
  - `P_prior_final`: `[L, L, 64]` float16/float32
  - `W`: `[L, L]` float16 weights
  - `gamma`: `[L, L]` float16 gating tensor
- `logs/aip.jsonl` — per-query diagnostics: consensus score, conflict flags,
  weight quantiles, cache hits.

### Computation
1. For each aligned pair `(i, j)` from template `t`:
   - `c_i = plddt_t[i] / 100`, `c_j = plddt_t[j] / 100`
   - `identity_to_query = hit.identity`
   - `coverage_ij` from alignment overlap mask
   - `context(i, j)` encodes penalties for gaps, mutation windows (±16), and
     long-range disagreements (`|i-j| ≥ 24`).
2. Weight contribution: `W_t[i,j] = (c_i c_j)^β * coverage_ij * identity^α * context`
   with `β = 1.5`, `α = 1.0` (defaults).
3. Construct Gaussian around template distance (bin centre) with σ schedule:
   - Base on min pLDDT; widen by identity tier and gap penalties (see
     `docs/PARAMS_AIP.md`).
4. Blend with global prior via `λ = λ0 * (1 - local_mutation_density)`
   (λ₀ ≈ 0.6 at 50% identity).
5. Sum logit contributions across templates, apply softmax over bins →
   `P_prior_final`.
6. Aggregate weights `W = Σ_t W_t`; derive `gamma = γ_base * normalize(W)` with
   `γ_base = 0.7` using per-matrix p5–p95 min-max scaling.
7. Apply conflict rule: if consensus low and `|i-j| ≥ 24`, widen σ by +1 Å and
   halve local `W_t`.
8. Optionally cache outputs for reuse across batches.

### Checklist
- Ensure per-query tensors respect bucket padding/masking.
- Emit diagnostics for mutation windows and conflict adjustments.
- Validate that high-pLDDT aligned regions show narrow peaks; mutation windows
  widen as expected.
- Integrate `gamma` and `W` into downstream feature builder interface.

### Tier 2 / Tier 3 Notes
- **Tier 2**: reuse cached priors when homolog clusters share alignments; record
  provenance in DuckDB manifests.
- **Tier 3**: research learned mixing coefficients (small transformer predicting
  λ, σ adjustments) or diffusion-based prior sharpening.

---

## 4. Dependencies & Tooling
- ESM-2 embeddings (PyTorch), FAISS (optional), HMMER for fallback alignments.
- Deterministic RNG seeds for retrieval and gap-handling heuristics.
- Telemetry hooks should emit JSON lines with query IDs for downstream audit.

---

Refer to [inference_routing.md](inference_routing.md) for how
these priors feed the MiniFold fast path and routing logic.
