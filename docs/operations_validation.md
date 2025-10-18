# Operations, Tooling & Validation

This document consolidates Tier 1 operational guidance: configuration, logging,
batch orchestration, manifests, testing, and CLI entrypoints. Tier 2 / Tier 3
ideas are listed per section.

---

## 1. Stage [00] Environment & Config

### Responsibilities
Provide deterministic configuration knobs and manifests before any processing.

### Files
- `configs/famfold/paths.yaml` — host/cluster path overrides (`data_root`,
  `cache_root`, `zarr_root`).
- `configs/famfold/default.yaml` — global knobs (`γ_base`, `β`, `α`, `λ`, `k`,
  recycle counts, dtype toggles).
- `configs/famfold/thresholds.yaml` — router gates (`ACCEPT`, `REFINE`,
  `ESCALATE`).
- `configs/famfold/buckets.yaml` — bucket definitions (length cutoffs, padding).
- `configs/famfold/retrieval.yaml` — retrieval backend, `k`, FAISS params.
- `manifests/run_manifest.json` — run metadata: `family_id`, versions,
  timestamps, hashes of configs/artifacts.

### Checklist
1. Populate `paths.yaml` per environment; avoid hard-coded absolute paths in
   code.
2. Set sensible defaults (`γ_base = 0.7`, `β = 1.5`, `α = 1.0`, `k = 6`).
3. Define router thresholds (ACCEPT ≤0.25 entropy, REFINE ≤0.35, else ESCALATE).
4. Configure buckets (256/384/512) with padding/truncation policies.
5. Materialise `run_manifest.json` with BLAKE3 hashes of key artifacts and
   datestamps.
6. Version configs using semantic tagging to track pipeline revisions.

### Tier 2 / Tier 3 Notes
- **Tier 2**: adopt Hydra/OMEGACONF for hierarchical overrides and environment
  profiles.
- **Tier 3**: integrate remote config service (e.g., S3-backed) with automatic
  provenance injection.

---

## 2. Stage [09] Batch Orchestration, Manifests & Reports

### Responsibilities
Deterministically batch sequences, orchestrate multi-stage runs, and emit
manifests + dashboards for auditing.

### Outputs
- `inference/batches/batch_XXXX.jsonl` — per-sequence records (metrics, route,
  runtime, neighbors, structure paths).
- `manifests/dataset.duckdb` — tables: `sequences`, `templates`, `hits`,
  `metrics`, `routes`, `logs`.
- `reports/pilot_metrics.parquet` — aggregated KPIs for dashboards.
- `reports/dashboards.md` — manual/auto-generated run summaries.

### Checklist
1. Bucket sequences deterministically (sort by length, assign to bucket, chunk
   by configurable batch size). Shrink batch when GPU OOM signals occur.
2. Ensure append-safe JSONL logging (one line per sequence per stage).
3. Populate DuckDB with relational views joining templates, retrieval hits,
   priors, inference metrics, and router decisions.
4. Export pilot metrics (acceptance rate, runtime distributions, escalation %).
5. Snapshot config/manifests for reproducibility (store commit hash, git status).
6. Provide CLI commands (see Section 4) for each stage with consistent logging.

### Tier 2 / Tier 3 Notes
- **Tier 2**: schedule asynchronous export of metrics to dashboards (e.g.,
  Streamlit) and integrate Slack/email alerts for escalations.
- **Tier 3**: support distributed orchestration (Ray/Kube jobs) with auto-scaling
  and preemption recovery.

---

## 3. Logging & Telemetry

### Responsibilities
Maintain consistent JSONL telemetry for traceability and debugging.

### Logs
- `logs/template_prep.jsonl`
- `logs/family_prior.jsonl`
- `logs/retrieval.jsonl`
- `logs/aip.jsonl`
- `logs/features.jsonl`
- `logs/inference.jsonl`
- `logs/router.jsonl`

Each entry should include `timestamp`, `family_id`, `qhash` (when applicable),
`stage`, `runtime_s`, `status`, and stage-specific fields.

### Tier 2 / Tier 3 Notes
- **Tier 2**: stream logs into centralized systems (ELK, Loki) and provide
  prebuilt Grafana dashboards.
- **Tier 3**: add structured tracing (OpenTelemetry) covering GPU utilisation,
  memory, and data-loader throughput.

---

## 4. CLI Entry Points (Typer Skeleton)

Expose the following CLI commands (names only for now):

```
famfold template-prep --family <fid>
famfold build-prior   --family <fid>
famfold retrieve      --family <fid> --mode protriever|esm-rag --k 6
famfold build-aip     --family <fid> --batch <ids.txt>
famfold infer         --family <fid> --bucket {256,384,512} --mode {6A,6B,auto}
famfold route         --family <fid> --thresholds configs/famfold/thresholds.yaml
famfold export        --family <fid> --format {pdb,mmcif,zarr}
famfold report        --family <fid> --out reports/pilot_metrics.parquet
```

### Tier 2 / Tier 3 Notes
- **Tier 2**: add `famfold homolog` command for co-prediction batches and
  `famfold dashboard` for automated reporting.
- **Tier 3**: integrate workflow runners (Prefect, Airflow) with CLI wrappers.

---

## 5. Testing Strategy

### Unit Tests (Tier 1 must-haves)
- TemplatePrep: bins symmetric; pLDDT length matches `L`; filter rejects mean
  <70.
- FamilyPrior: dense↔sparse round-trip; metadata matches template counts.
- Retrieval: deterministic top-`k`; identity/coverage monotonicity checks.
- Alignments: mapping inverses consistent; mismatch windows flagged correctly.
- AIP: high-pLDDT regions produce narrow peaks; mutation windows widen.
- FeatureBuilder: embedding cache hits; pair feature shapes correct.
- MiniFold engine: deterministic outputs; chirality check passes; 6A modifies
  only mutation windows.
- Router: thresholds gate sequences as expected; prior decay reduces `gamma`
  where entropy is low.

### Integration Tests (Toy Family)
1. Baseline (no prior) — capture acceptance/time/entropy baseline.
2. FamilyPrior enabled — expect entropy ↓, acceptance ↑.
3. AIP enabled — further entropy ↓, fewer recycles.
4. Path 6A on ≥85% identity — validate speed-up and RMSD outside windows <1 Å.
5. Router end-to-end — escalation ≤ target.

### Pilot & Stress (Operational Checks)
- Pilot (1–5k sequences): acceptance ≥ target; REFINE ≤15%; ESCALATE ≤5%;
  throughput ≥10× ESMFold; chirality flips <1%; memory OK at 512 AA.
- Stress: handle ≥500 AA sequences with automatic batch shrink; escalate quickly
  when templates sparse; homolog clusters demonstrate consensus speed-ups.

### Tier 2 / Tier 3 Notes
- **Tier 2**: add regression dashboards comparing pLDDT distributions and
  runtime histograms across releases.
- **Tier 3**: incorporate large-scale simulation harnesses and statistical
  quality monitoring (e.g., SPC charts).

---

## 6. Tier 2 & Tier 3 Cross-Cutting Ideas
- Central cache service for embeddings/priors across families.
- Advanced escalation workflows (ESMFold multi-recycle, AlphaFold2 fallback).
- Research integration of diffusion priors and learned routing controllers.

Refer back to the component documents for data
([data_foundations.md](data_foundations.md)), retrieval
([retrieval_priors.md](retrieval_priors.md)), and inference
([inference_routing.md](inference_routing.md)) specifics.
