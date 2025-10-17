# FamilyFold Bulk Prediction Pipeline Plan

## 1. Objectives and Scope
- Deliver fast, bulk monomer structure predictions for all sequences in a protein family while preserving accuracy.
- Reuse known high-confidence family structures (with per-residue pLDDT) and existing PLM infrastructure (MiniFold / ESMFold).
- Integrate lightweight retrieval (FAISS) and a MiniFold variant that ingests a family-derived geometric prior.

## 2. Inputs
1. **Sequence set**: FASTA-formatted sequences for the target family (assumed to share at least one structural domain). Every sequence should be folded.
2. **Reference structures**: PDB/mmCIF files for one or more family members. File IDs match sequence identifiers in the FASTA. Each structure includes per-residue pLDDT or an equivalent confidence score.

## 3. Expected Outputs
- Predicted 3D structures (PDB) for every input sequence.
- Per-residue confidence (pLDDT) and pairwise distance entropies for gating/quality control.
- Optional metadata: template usage summary, entropy scores, and routing decisions (fast pass vs. refinement).

## 4. High-Level Pipeline
1. **Template preparation**
   - Parse the provided family structures to extract sequences, per-residue pLDDT, and Cα distograms in MiniFold's 64-bin (2–25 Å) format.
   - Optionally run FoldMason (or similar) to generate a canonical residue↔MSA column map when the family exhibits domain shuffling or inconsistent insertions; skip this step for compact, single-domain families where simple sequence alignment suffices.
2. **Family-wide alignment (optional)**
   - Seed an alignment using the structural superposition when available, or rely on sequence-profile tools to align the remaining family members.
   - Persist residue coverage statistics for downstream weighting regardless of whether the alignment originates from FoldMason.
3. **Family geometry prior construction**
   - Aggregate per-template distograms into a global family prior `P_family(i, j, b)` with companion weights `W_family(i, j)` using pLDDT-, coverage-, and identity-based weighting.
   - Store both dense tensors and sparse top-*k* bin representations per length bucket for reuse during inference.
4. **Sequence embedding & retrieval layer**
   - Embed all family sequences with ESM-2 (MiniFold-compatible features) and cache results by length bucket.
   - Build a FAISS (or ESM-RAG) index over template embeddings for fast nearest-neighbor lookup.
   - For each query or batch, retrieve the top-*k* folded templates to specialize the family prior toward the closest homologs.
5. **MiniFold feature assembly**
   - Tile single-sequence embeddings into pair representations, then inject the query-specific prior channels (soft probabilities or one-hot argmax plus weights) ahead of the MiniFormer stack.
6. **Fast MiniFold inference**
   - Run the 10-layer MiniFold variant without a structure module, relying on distogram recycling to blend predictions with the injected prior.
   - Execute a single forward pass (zero or one recycle) for the default fast path.
7. **Coordinate realization**
   - Recover 3D coordinates via the parameter-free MiniFold MDS realizer (shortest-path completion → classical MDS → LBFGS stress majorization) with chirality checks.
8. **Confidence gating & refinement**
   - Combine MiniFold distogram entropy, agreement with the prior, retrieval coverage, and sequence-only OOD signals (see §7) to decide whether to accept, recycle once with prior decay, or escalate to ESMFold.
9. **Batch orchestration & homolog handling**
   - Bucket sequences by length (e.g., 256/384/512 AA) to minimize padding and exploit MiniFold's memory savings for large batches.
   - Cluster ≥85–90% identity homologs so they can reuse cached embeddings, priors, and retrieval hits within the same batch.

## 5. System Architecture & Module Graph
- **Core dependencies**
  - *La-Proteina* supplies the dataset/dataloader scaffolding for sequences and structures.
  - *Protriever* handles retrieval (k-NN homolog discovery with FAISS or an ESM-RAG fallback).
- **Dataflow overview**

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
[03] Retrieval (Protriever/FAISS) ──► topk.parquet, clusters.json
          │
          ▼
[04] PriorBuilder ──► query-specific P_prior, W, γ
          │
          ▼
[05] FeatureBuilder (La-Proteina + ESM-2) ──► pair features + prior channels
          │
          ▼
[06] MiniFoldEngine ──► distogram logits, uncertainty metrics
          │
          ▼
[07] Router ──► ACCEPT / REFINE / ESCALATE decisions
          │
          ▼
[08] HomologCoPredict ──► mutation-aware batching, shared priors
```

Each module emits structured manifests so downstream stages can operate deterministically and rerun safely. Optional FoldMason alignment feeds into [01] and [02] when the family requires structural column normalization.

## 6. Artifacts, Schemas, and Storage Conventions
- **Directory skeleton (`families/{family_id}/`)**
  - `raw/`: input FASTA and reference structures.
  - `templates/`: per-template JSON blobs containing sequence, length, pLDDT, distogram bins, and provenance metadata.
  - `priors/`: aggregated family priors per length bucket in dense (`distogram_{bucket}.npz`) and sparse (`distogram_{bucket}_sparse.npz`) forms plus `metadata.json`.
  - `retrieval/`: `template_bank.faa`, optional template embedding caches, `topk.parquet`, and cluster assignments.
  - `embeddings/{bucket}/`: cached ESM-2 features and pseudo-perplexity logs keyed by sequence ID.
  - `coords/`: MiniFold/ESMFold outputs (PDB/mmCIF) with per-residue metrics.
  - `logs/`: JSONL telemetry for every stage and router decision.
  - `configs/`: Hydra configuration overrides and structured configs for reproducible runs.
- **Tensor shapes (per bucket length \(L_b\))**
  - `P_family`: `[L_b, L_b, 64]` float32 probabilities or logits.
  - `W_family`: `[L_b, L_b]` float16 weights (coverage × pLDDT products).
  - Sparse storage keeps top-2 bins per pair (COO format) to reduce disk and GPU memory pressure.
- **Metadata hygiene**
  - Every artifact carries a `version`, `family_id`, `bucket`, `created_at`, and hash digest to guarantee compatibility across reruns and incremental updates.

## 7. Module Specifications
### [01] TemplatePrep
- **Responsibility**: parse structures, compute MiniFold-ready distograms, and extract per-residue pLDDT.
- **Inputs**: `reference_structures/` (PDB/mmCIF); optional FoldMason residue↔MSA maps.
- **Outputs**: `templates/template_{id}.json`, `retrieval/template_bank.faa`.
- **Algorithm**:
  1. Parse chains to obtain sequence, length, and Cα coordinates.
  2. Compute pairwise distances; discretize into 64 MiniFold bins (2–25 Å, last bin ≥25 Å).
  3. Pull per-residue pLDDT (PDB `B-factor`, CIF `pLDDT` tables) and normalize to [0, 1].
  4. Optionally attach FoldMason residue→MSA indices when structural alignment is used.

### [02] FamilyPrior
- **Responsibility**: aggregate template distograms into a reusable family prior.
- **Inputs**: template JSON blobs (and optional FoldMason maps).
- **Outputs**: dense and sparse priors (`priors/distogram_{bucket}*.npz`), `metadata.json`.
- **Weighting**: `W_t_global[i,j] = (plddt_i · plddt_j)^β × coverage_ij`, with β≈1.0 for the global prior.
- **Aggregation**: sum weighted one-hot distograms across templates, apply softmax over bins, and store companion weight matrix `W_family`.

### [03] Retrieval (Protriever/FAISS)
- **Responsibility**: discover top-*k* template neighbors per query and cluster ≥85–90% identity homologs.
- **Inputs**: `family.fasta`, `template_bank.faa`, optional template embeddings.
- **Outputs**: `retrieval/topk.parquet` with scores, `retrieval/clusters.json` for homolog groupings.
- **Implementation**: default to Protriever + FAISS; provide an ESM-RAG matmul fallback when FAISS is unavailable.

### [04] PriorBuilder
- **Responsibility**: specialize the global prior for each query using retrieval hits.
- **Inputs**: `P_family`, `W_family`, per-template distograms, `topk.parquet` rows.
- **Outputs**: query-specific `P_prior`, `W`, and per-pair strength tensor `γ`.
- **Computation**:
  - Weight contributions by `(plddt_i · plddt_j)^β × coverage_ij × identity_to_query^α` (β≈1.5, α≈1.0).
  - Optionally hard-mask residues with pLDDT < 0.5.
  - Normalize weights, emit soft or one-hot priors plus `γ` scaled to [0, 1].

### [05] FeatureBuilder (La-Proteina)
- **Responsibility**: assemble MiniFold-compatible features and inject priors.
- **Inputs**: La-Proteina dataloader batches, cached ESM-2 embeddings, query-specific priors.
- **Outputs**: pair features with appended prior channels (Mode A: one-hot + weights; Mode B: logit biases).
- **Notes**:
  - Bucket by length (256/384/512 AA). Cache embeddings per bucket to avoid redundant ESM-2 calls for near-identical homologs.
  - Support pseudo-perplexity sampling masks so OOD metrics are computed alongside embeddings.

### [06] MiniFoldEngine
- **Responsibility**: run the fast MiniFold forward pass and produce uncertainty diagnostics.
- **Inputs**: prior-augmented pair features, recycle count, `γ` tensors.
- **Outputs**: predicted distograms, entropy metrics, agreement scores, and realized coordinates.
- **Key metrics**:
  - `H_norm`: normalized distogram entropy.
  - `A`: agreement rate between `P_prior` and predictions (weighted by `W`).
  - `S_prior`: median prior strength.
- **Coordinate realization**: MiniFold’s parameter-free MDS realizer with chirality check.

### [07] Router
- **Responsibility**: gate outputs into ACCEPT / REFINE / ESCALATE paths.
- **Inputs**: `H_norm`, `A`, `S_prior`, retrieval coverage, BPR statistics (see §8).
- **Policy (default)**:
  - ACCEPT when `H_norm ≤ 0.25`, coverage ≥0.70 or `A ≥ 0.60`, and `S_prior ≥ 0.35`.
  - REFINE when metrics fall in buffer ranges (e.g., `0.25 < H_norm ≤ 0.35`)—trigger one recycle with entropy-driven prior decay `γ_ij ← γ_ij × (1 − conf_ij)`.
  - ESCALATE to ESMFold when `H_norm > 0.35`, `S_prior < 0.20`, or coverage <0.50 with poor agreement.

### [08] HomologCoPredict
- **Responsibility**: accelerate ≥85% identity clusters via consensus folding and mutation-aware specialization.
- **Procedure**:
  1. Cluster sequences using Protriever identity scores; choose a representative.
  2. Fold the representative with the full prior; accept if router thresholds pass.
  3. For remaining homologs, reuse embeddings/prior, attenuate `γ` within ±16 residue mutation windows, and run the fast path.
  4. Share router decisions across the cluster when global metrics stay within acceptance bounds.

## 8. Sequence-Only OOD Detection (ESM-2)
- **Pseudo-perplexity / bits-per-residue (BPR)**:
  - Mask ~20–25% of residues uniformly at random, run ESM-2 to obtain \(p(x_i \mid x_{\setminus i})\), repeat 2–3 times, and average.
  - Compute \(\mathrm{BPR} = -\frac{1}{L \log 2} \sum_{i=1}^{L} \log p(x_i \mid x_{\setminus i})\); lower BPR implies in-distribution sequences.
  - Calibrate against a reference distribution (50–100k UniRef50 sequences or trusted in-family subsets) to derive percentiles and z-scores.
- **Router integration**:
  - Treat high-BPR sequences as OOD candidates: lower the acceptance threshold, trigger immediate recycling, or escalate to ESMFold.
  - Persist BPR metrics in `embeddings/bpr_metrics.parquet` for ongoing calibration.
- **Adaptive modeling (optional)**:
  - When persistent BPR spikes appear for a sub-family, consider LoRA-style fine-tuning or partial unfreezing of ESM-2 layers, gated behind explicit operator approval.

## 9. Router, Escalation, and Logging
- Combine `H_norm`, `A`, `S_prior`, retrieval coverage, and BPR signals to make routing decisions.
- Emit structured logs documenting the metrics, thresholds, decision, and whether the prior was decayed during refinement.
- Maintain cumulative dashboards (DuckDB/Parquet) to monitor acceptance, refinement, and escalation rates per batch and per length bucket.

## 10. Performance & Engineering Considerations
- Cache embeddings, priors, and retrieval hits to eliminate redundant compute, especially for ≥90% identity homologs processed in the same batch.
- Enforce static shapes per length bucket so Triton kernels and Torch compile deliver predictable throughput.
- Detect long or multi-domain sequences and optionally split into domains before folding, then stitch coordinates via lightweight loop closure.
- Watch GPU memory; MiniFold reports 20–40× peak memory savings and 15–20× end-to-end speedups, enabling larger homogeneous batches.
- Implement dynamic batch shrinking to recover gracefully from OOM events.

## 11. Open Questions / Configuration Knobs
1. Soft vs. one-hot prior injection (Mode A vs. Mode B) for different family topologies.
2. Entropy, agreement, and BPR thresholds that best balance throughput and accuracy for your dataset.
3. Whether zero-shot MiniFold suffices or if light fine-tuning on family-primed priors yields measurable gains.
4. Preferred exposure of knobs (Hydra configs vs. CLI flags) for operators and automation.

## 12. Precomputation Implementation Plan
The following plan details how to operationalize all prerequisite assets—structural alignment, MSAs, priors, embeddings, and retrieval indices—before fast inference begins.

### 12.1 Data Layout & Orchestration
- **Workspace layout**
  - `data/raw/`: original FASTA sequences (`family.fasta`) and structure files (`*.pdb` / `*.cif`).
  - `data/intermediate/`: alignment artifacts (`foldmason/`, `msa/`), distilled priors (`priors/`), and embedding shards (`embeddings/`).
  - `data/index/`: FAISS indices and metadata manifests.
  - `configs/`: Hydra config tree (see §10.5) to define datasets, tool paths, batching, and caching behaviour.
- **Execution flow**
  1. `foldmason_align` → 2. `expand_msa` → 3. `distogram_prior` → 4. `esm2_embed` → 5. `build_faiss`.
  - Each stage writes a manifest JSON (Hydra Structured Config) enumerating outputs for downstream reuse.
- **Parallelism & caching**: chunk long families by sequence count; run independent Typer subcommands per chunk; rely on manifest timestamps + Hydra `job.override_dirname` for reproducible reruns.

### 12.2 FoldMason Structural Alignment (`foldmason_align`)
- **When to run**: use this stage when structural column consistency materially improves the prior (e.g., domain shuffling or long insertions). Skip it entirely for compact, single-domain families where simple sequence alignment suffices.
- **Inputs**: list of template structure paths, optional residue range masks.
- **Processing**
  - Invoke FoldMason in batch mode with GPU acceleration where available.
  - Export superposed coordinates and residue mapping tables (`structure_id`, `residue_index` → `msa_position`).
- **Outputs**
  - `foldmason/superposition_{run_id}.pdb` (aligned stack).
  - `foldmason/mapping_{structure_id}.json` capturing MSA position mapping and per-residue pLDDT.
  - Manifest: `foldmason/manifest.json` listing structures processed, alignment score metrics, and quality flags.

### 12.3 Family MSA Expansion (`expand_msa`)
- **Inputs**: initial structural alignment manifest, family FASTA, optional external sequence databases.
- **Processing**
  - Seed an MSA using the residue mappings; run profile-sequence alignment tools (e.g., HHblits/JackHMMER) to add remaining family sequences.
  - Optionally enforce domain segmentation via HMM boundaries to avoid misaligned insertions.
- **Outputs**
  - `msa/family_{run_id}.a3m` and compressed `msa/family_{run_id}.stk`.
  - Coverage statistics per sequence (`msa/coverage.json`) for routing and weight computation.
  - Manifest including effective sequence count and warnings for low-coverage members.

### 12.4 Distogram Prior Construction (`distogram_prior`)
- **Inputs**: structural mapping manifests, full MSA, per-residue pLDDT.
- **Processing**
  - Convert each structure to MiniFold binning (`B=64`, 2–25 Å). Use numba/torch to parallelize per-template histograms.
  - Map to MSA coordinates and compute weights `coverage × pLDDT_i × pLDDT_j × identity_to_query`.
  - Aggregate across templates, storing both dense tensors and sparse top-*k* views (for GPU efficiency).
- **Outputs**
  - `priors/distogram_{bucket}.npz`: packed tensors (`P_family`, `W`).
  - `priors/distogram_{bucket}_sparse.npz`: coordinate list for top-*k* bins.
  - Calibration metadata (mean, variance of weights, template contributions) for debugging.

### 12.5 ESM-2 Embedding Shards (`esm2_embed`)
- **Inputs**: family FASTA, Hydra config specifying model checkpoint, batch size, masking fractions for pseudo-perplexity.
- **Processing**
  - Run batched ESM-2 forward passes (FP16 where possible) to extract last-layer embeddings and logits required for BPR.
  - Cache embeddings per length bucket to align with inference batching.
  - Compute pseudo-perplexity estimates on-the-fly (mask 20–25% residues, repeat 3×) and log calibrated statistics if a reference distribution is supplied.
- **Outputs**
  - `embeddings/{bucket}/sequence_id.pt` (torch tensors with single and pair features).
  - `embeddings/bpr_metrics.parquet` for routing thresholds.

### 12.6 FAISS Index Build (`build_faiss`)
- **Inputs**: embedding shards, optional dimensionality reduction parameters (PCA/OPQ).
- **Processing**
  - Train PCA/OPQ transforms when configured; apply to embeddings.
  - Build IVF/HNSW index tuned for family size; record recall/latency benchmarks.
  - Store neighbor graphs (`topk_neighbors.parquet`) for debugging and as a warm cache.
- **Outputs**
  - `data/index/faiss_{bucket}.index` and associated `faiss_{bucket}.meta.json` capturing hyperparameters and embedding stats.

### 12.7 Typer + Hydra CLI Skeleton
- **Entry point**: `minifold/cli/familyfold.py` registered in `pyproject.toml` as `familyfold` console script.
- **Typer app**: root command `familyfold` with subcommands: `foldmason-align`, `expand-msa`, `distogram-prior`, `esm2-embed`, `build-faiss`, and `precompute-all` (runs the full chain respecting dependencies).
- **Hydra integration**
  - Each subcommand loads a Hydra config (`configs/familyfold/{stage}.yaml`) describing tool paths, resource limits, input manifests, and output directories.
  - Support `--config-name` and `--config-path` overrides; expose critical overrides as CLI options (e.g., `--bucket-length`, `--faiss-metric`).
  - Use Hydra sweepers to distribute workloads across GPUs/hosts if needed (`familyfold sweep precompute-all hydra/sweeper=submitit`).
- **Logging & telemetry**
  - Structured logs (JSON) per stage stored under `logs/{stage}/{timestamp}.jsonl`.
  - Emit completion summary with paths to generated manifests for downstream automation.

### 12.8 Validation & Monitoring
- Implement a Typer subcommand `verify-manifests` that checks hashes, expected file counts, and basic sanity metrics (e.g., average coverage, BPR percentile histograms).
- Schedule periodic re-runs via `precompute-all --refresh` to update priors when new templates or sequences arrive.
- Integrate with CI/CD by adding smoke tests on small toy families to ensure each stage functions and Hydra configs resolve correctly.

## 13. End-to-End Orchestration (Pseudocode)
```python
def run_familyfold(family_fasta, ref_struct_dir):
    # 01 TemplatePrep
    templates = prepare_templates(ref_struct_dir)

    # 02 FamilyPrior
    P_family, W_family = build_family_prior(templates)

    # 03 Retrieval
    hits = protriever_topk(family_fasta, templates, k=4)
    clusters = protriever_cluster(family_fasta, threshold=0.85)

    for cluster in clusters:
        rep = choose_representative(cluster)
        batch = [rep] + [seq for seq in cluster if seq != rep]

        for query in batch:
            Pq, Wq, gamma = build_query_prior(P_family, W_family, templates, hits[query])

            esm_feats = esm2_embed_cached(query)
            pair_feats = seq_to_pair(esm_feats)
            pair_aug = inject_prior(pair_feats, Pq, Wq, gamma)

            dist_pred, metrics = minifold_forward(pair_aug, recycles=0)

            if query != rep:
                windows = mutation_windows(rep, query)
                gamma = attenuate_gamma(gamma, windows)

            decision = route(metrics, hits[query], bpr=metrics.bpr)

            if decision is ACCEPT:
                coords = realize_coords(dist_pred)
            elif decision is REFINE:
                gamma = decay_prior_by_confidence(gamma, dist_pred)
                dist_pred, metrics = minifold_forward(update_features(pair_aug, gamma), recycles=1)
                coords = realize_coords(dist_pred)
            else:
                coords = esmfold_fallback(query)

            persist_outputs(query, coords, metrics, hits[query])
```

## 14. Testing Strategy
### Phase A — Unit Tests
- **TemplatePrep**: symmetric bins, correct normalization, pLDDT lengths.
- **FamilyPrior**: aggregation preserves dominant bins and weight support.
- **Retrieval**: deterministic top-*k* identity/coverage ordering on toy sets.
- **PriorBuilder**: single-template case reproduces original distogram; γ scales with pLDDT.
- **FeatureBuilder**: embedding cache integrity, padding masks, pseudo-perplexity sampling reproducibility.
- **MiniFoldEngine**: deterministic logits with fixed seeds; coordinate realizer produces finite, chirality-correct structures.
- **Router**: synthetic logits hit ACCEPT/REFINE/ESCALATE thresholds as expected.
- **HomologCoPredict**: mutation window boundaries, γ attenuation confined to mutation regions.

### Phase B — Integration Tests (Toy Families)
- Assemble a ~50-sequence toy family with three templates.
- Run the pipeline without priors vs. with priors; confirm acceptance rate, entropy, and agreement improvements.
- Inject low-pLDDT templates and verify `W_t` suppression plus router escalations.
- Enable homolog co-prediction for ≥90% identity clusters and measure runtime reduction vs. independent folding.

### Phase C — Acceptance Tests (1–5k Sequence Pilot)
- Track service-level objectives: throughput, acceptance/refinement/escalation ratios, memory usage, chirality flip rate.
- Alert when acceptance drops >10% below target or escalations exceed budget (>5%).
- Maintain dashboards (DuckDB/Parquet) with per-batch histograms for `H_norm`, `S_prior`, `A`, coverage, runtime, and BPR.

### Phase D — Performance & Stress
- Exercise long sequences (≥500 AA) and verify adaptive batch sizing.
- Evaluate template-sparse families to ensure quick escalations without wasted recycles.
- Stress-test large homolog clusters and confirm consensus folding delivers throughput gains.

## 15. Milestones (Vertical Slices)
- **M0 – Skeleton**: La-Proteina dataloader + MiniFold fast path → coordinates (no priors/router).
- **M1 – Templates & Priors**: per-template distograms aggregated to family priors; unit tests green.
- **M2 – Retrieval**: Protriever/FAISS wired with top-*k* manifests and clustering outputs.
- **M3 – PriorBuilder**: query-specific priors injected (Mode A), end-to-end inference functional.
- **M4 – Router**: uncertainty metrics, gating, one recycle with entropy-driven prior decay.
- **M5 – HomologCoPredict**: consensus + mutation-window specialization; group routing stabilized.
- **M6 – Calibration**: thresholds tuned on pilot family; acceptance and escalation SLOs met.
- **M7 – Hardening**: stress tests, dashboards, CI smoke tests, documentation finalized.

## 16. Risks & Guardrails
- **Bad priors**: mitigate with β exponent tuning, low-pLDDT masking, and `S_prior` gating.
- **Template topology mismatch**: monitor agreement `A`, decay priors during refinement, escalate when agreement collapses.
- **Coordinate instability**: enforce chirality checks, escalate rare divergence cases to ESMFold.
- **Memory spikes/OOM**: length bucketing, sparse priors, dynamic batch shrinking.
- **Over-recycling**: cap at one recycle; shadow-log outcomes to justify potential future adjustments.

## 17. Definition of Done
- **Accuracy**: median lDDT within 3 percentage points of ESMFold for accepted sequences on a held-out validation set.
- **Throughput**: ≥10× faster than ESMFold on ACCEPTed sequences; REFINE ≤15%, ESCALATE ≤5%.
- **Stability**: chirality flips <1%, deterministic reruns (hash-stable artifacts).
- **Homolog batching**: ≥85% identity clusters show measurable runtime reduction and mutation-localized geometry deltas.
- **Operational readiness**: Hydra/Typer CLI, manifests, dashboards, and validation commands documented and automated.
