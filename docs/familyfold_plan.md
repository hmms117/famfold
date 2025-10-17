# FamilyFold Bulk Prediction Pipeline Plan

## 1. Objectives and Scope
- Deliver **fast, bulk monomer** structure predictions for all sequences in a protein family (typically 300–500 AA) while preserving accuracy.
- Reuse high-confidence family structures (with per-residue **pLDDT**) and existing PLM infrastructure (**MiniFold** primary; **ESMFold** fallback).
- Integrate lightweight retrieval with **Protriever** (FAISS optional) or an **Easy ESM-RAG** fallback to specialize priors per query.
- Inject a **family-derived, pLDDT-weighted geometric prior** into MiniFold and optionally sharpen it via alignment-informed conditioning for each query.
- Avoid standalone MSAs at inference by default; only run FoldMason/MAFFT when structural column normalization measurably helps multi-domain or insertion-heavy families.

## 2. Inputs
1. **Sequence set**: FASTA-formatted sequences for the target family (assumed to share at least one structural domain). Every sequence should be folded.
2. **Reference structures**: PDB/mmCIF files for one or more family members. File IDs match sequence identifiers in the FASTA. Each structure includes per-residue pLDDT or an equivalent confidence score (AFDB PDBs encode pLDDT in the B-factor; experimental structures may require mmCIF QA tables).
3. *(Optional)* **Seed structural alignment** (FoldMason/MAFFT) for families with large insertions or domain shuffling. Default: skip for single-domain families where retrieval alone produces consistent columnning.

## 3. Expected Outputs
- Predicted 3D structures (mmCIF/PDB) for every input sequence.
- Per-sequence metrics for routing and QA: normalized distogram entropy \(H_{norm}\), prior strength \(S_{prior}\), agreement \(A\), and retrieval coverage/identity/similarity.
- Manifests: template usage summary, thresholds, route taken (fast/refine/escalate), runtimes, and escalation triggers.

## 4. High-Level Pipeline
1. **TemplatePrep** — parse templates → `{seq, pLDDT[L], distogram bins[L×L]}` plus optional residue↔MSA maps when FoldMason is enabled.
2. **FamilyPrior** — aggregate template distograms into global `P_family(i,j,b)` and `W_family(i,j)` tensors (dense + sparse top-\(k\) storage).
3. **Retrieval (Protriever / Easy ESM-RAG)** — embed sequences with ESM-2, retrieve top-\(k\) templates per query (identity/coverage/bitscore or cosine similarity), and cluster ≥85% identity homologs.
4. **Alignment-Informed Prior (AIP)** — align each query to its hits; sharpen `P_family` locally using template distances weighted by pLDDT, identity, and gap context; emit per-query `P_{prior}`, `W`, and strength tensor `γ`.
5. **FeatureBuilder (La-Proteina)** — load cached ESM-2 embeddings, tile sequence→pair features, and inject prior channels (Mode A: one-hot + weights; Mode B: logit biases).
6. **MiniFold Fast Inference & Coordinates** — choose between:
   - **6A Template-Thread & Warp** (top-hit identity ≥0.85 and coverage ≥0.70): thread onto best template, warp only mutation windows (±16 residues), skip global MDS.
   - **6B Distogram→Coords** (default): run MiniFold fast pass (0 recycles) with prior injection, then realize coordinates via shortest-path completion → classical MDS → LBFGS plus chirality check.
7. **Router (Uncertainty)** — compute `H_{norm}`, `S_{prior}`, `A`, and coverage to ACCEPT, REFINE (1 recycle with uncertainty-aware prior decay), or ESCALATE to ESMFold.
8. **Homolog Co-Prediction** — for ≥85% identity clusters, fold representatives first, then specialize homologs within mutation windows while reusing cached artifacts.
9. **Batch Orchestration & Logging** — bucket by length (256/384/512 AA), maintain mixed-precision caches, and emit manifests capturing retrieval hits, prior usage, routing decisions, and runtimes.

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
[03] Retrieval (Protriever/ESM-RAG) ──► topk.parquet, clusters.json
          │
          ▼
[04] Alignment-Informed Prior ──► query-specific P_prior, W, γ
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
- **Responsibility**: parse structures, compute MiniFold-ready distograms, extract per-residue pLDDT, and package metadata for downstream retrieval/alignment.
- **Inputs**: `reference_structures/` (PDB/mmCIF); optional FoldMason residue↔MSA maps and structural alignment manifests.
- **Outputs**: `templates/template_{id}.json` or `.npz`, `retrieval/template_bank.faa`, QC summaries.
- **Algorithm**:
  1. Parse chains to obtain sequence, length, and Cα coordinates.
  2. Compute pairwise distances; discretize into 64 MiniFold bins (2–25 Å, last bin ≥25 Å).
  3. Pull per-residue pLDDT (AFDB: PDB `B-factor`; experimental: mmCIF QA tables) and normalize to [0, 1].
  4. Compute summary statistics (mean/median pLDDT; reject templates with mean <70).
  5. Optionally attach FoldMason residue→MSA indices when structural alignment is used.

### [02] FamilyPrior
- **Responsibility**: aggregate template distograms into a reusable family prior that can be specialized per query.
- **Inputs**: template JSON/NPZ blobs (and optional FoldMason maps).
- **Outputs**: dense and sparse priors (`priors/distogram_{bucket}*.npz`), `metadata.json` with weight statistics.
- **Weighting**: `W_t_global[i,j] = (plddt_i · plddt_j / 10_000)^β × coverage_{ij}` with β≈1.0; mask low-pLDDT (<0.5) residues.
- **Aggregation**: sum weighted one-hot distograms across templates, apply softmax over bins to obtain `P_family`, and store companion weight matrix `W_family`. Persist top-2 bins per pair in COO format for fast loading.

### [03] Retrieval (Protriever / Easy ESM-RAG)
- **Responsibility**: discover top-*k* template neighbors per query, surface identity/coverage/bitscore metadata, and cluster ≥85% identity homologs for consensus folding.
- **Inputs**: `family.fasta`, `template_bank.faa`, cached template embeddings, optional FAISS indices.
- **Outputs**: `retrieval/topk.parquet` with scores, `retrieval/clusters.json` for homolog groupings, optional similarity matrices.
- **Implementation**: default to Protriever + FAISS (inner-product or cosine distance); fall back to Easy ESM-RAG (normalized ESM-2 embeddings, `Q @ Eᵀ`) when FAISS is unavailable.

### [04] Alignment-Informed Prior (AIP)
- **Responsibility**: specialize the global prior for each query using retrieval hits and their alignments.
- **Inputs**: `P_family`, `W_family`, per-template distograms, pLDDT vectors, `topk.parquet` rows, and alignment maps (`map_q→t`).
- **Outputs**: query-specific `P_prior`, `W`, strength tensor `γ`, and diagnostics (consensus, conflict flags).
- **Computation**:
  - Compute per-template weights `W_t[i,j] = (c_i c_j)^β × coverage_{ij} × identity_to_query^α × context(i,j)` where `c_i = plddt_t[i]/100`, β≈1.5, α≈1.0.
  - Build Gaussian-like templates around aligned distances with width \(σ\) governed by pLDDT, identity, and mutation-window penalties; mix with the global prior via `λ` (≈0.5, reduced in mutation-dense regions).
  - Aggregate across templates by summing logit-weighted contributions and normalizing to produce `P_prior`; compute `γ` via min–max scaling of total weights; emit consensus/conflict metrics for router logging.

### [05] FeatureBuilder (La-Proteina)
- **Responsibility**: assemble MiniFold-compatible features and inject priors.
- **Inputs**: La-Proteina dataloader batches, cached ESM-2 embeddings, query-specific priors.
- **Outputs**: pair features with appended prior channels (Mode A: `γ * one_hot(P_prior)` + `W`; Mode B: `γ * logit(P_prior)` bias terms).
- **Notes**:
  - Bucket by length (256/384/512 AA). Cache embeddings per bucket to avoid redundant ESM-2 calls for near-identical homologs.
  - Support pseudo-perplexity sampling masks so OOD metrics are computed alongside embeddings.
  - Expose hooks to attenuate `γ` in mutation windows for homolog co-prediction.

### [06] MiniFoldEngine
- **Responsibility**: run the fast MiniFold forward pass, choose the appropriate coordinate path, and produce uncertainty diagnostics.
- **Inputs**: prior-augmented pair features, recycle count, `γ` tensors, retrieval metadata.
- **Outputs**: predicted distograms, entropy metrics, agreement scores, realized coordinates, and per-path runtime stats.
- **Execution paths**:
  - **Template-thread & warp (6A)** when top-hit identity ≥0.85 and coverage ≥0.70; restrict warping to ±16 residue mutation windows and preserve template backbone elsewhere.
  - **Distogram→coords (6B)** otherwise; run zero recycles by default, with optional single recycle under router control.
- **Key metrics**:
  - `H_norm`: normalized distogram entropy.
  - `A`: agreement rate between `P_prior` and predictions (weighted by `W`).
  - `S_prior`: median prior strength across ±8 residue neighborhoods.
- **Coordinate realization**: Template-thread path uses restrained local warps; distogram path uses MiniFold’s parameter-free MDS realizer (shortest-path completion → classical MDS → LBFGS) with chirality check.

### [07] Router
- **Responsibility**: gate outputs into ACCEPT / REFINE / ESCALATE paths.
- **Inputs**: `H_norm`, `A`, `S_prior`, retrieval coverage/identity, BPR statistics (see §8), and AIP consensus/conflict signals.
- **Policy (default)**:
  - **ACCEPT** when `H_norm ≤ 0.25`, coverage ≥0.70 (or `A ≥ 0.60`), and `S_prior ≥ 0.35`.
  - **REFINE** when metrics fall in buffer ranges (e.g., `0.25 < H_norm ≤ 0.35`, `0.50 ≤ coverage < 0.70`, `0.40 ≤ A < 0.60`, or `0.20 ≤ S_prior < 0.35`). Trigger one recycle with uncertainty-aware prior decay `γ_ij ← γ_ij × (1 − conf_ij)` where `conf_ij = 1 − H(P_pred[i,j,:]) / log 64`.
  - **ESCALATE** to ESMFold when `H_norm > 0.35`, `S_prior < 0.20`, consensus conflicts persist (AIP disagreement), or coverage <0.50 with poor agreement.

### [08] HomologCoPredict
- **Responsibility**: accelerate ≥85% identity clusters via consensus folding and mutation-aware specialization.
- **Procedure**:
  1. Cluster sequences using Protriever identity scores; choose a representative.
  2. Fold the representative with the full prior; accept if router thresholds pass.
  3. For remaining homologs, reuse embeddings/prior, attenuate `γ` within ±16 residue mutation windows, and run the fast path.
  4. Share router decisions across the cluster when global metrics stay within acceptance bounds.
- **Outputs**: per-cluster manifests summarizing reused priors, mutation windows, acceptance outcomes, and optional window RMSD maps relative to the representative.

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

## 11. Knobs & Defaults (tuning order)
- `k` templates: **4** (tune between 3–8 based on family diversity).
- Global prior exponent `β`: **1.0**; query-specific exponent **1.5**.
- Identity exponent `α`: **1.0**.
- Base prior strength `γ_base`: **0.7** (adjust 0.5–0.8 to avoid over-/under-constraining).
- Mutation window half-width: **16** residues for warp/attenuation logic.
- Alignment-informed prior parameters:
  - Gaussian width `σ_base = 2.0 Å`, shrinks with higher pLDDT/identity and expands near gaps/mutations.
  - Mixture weight `λ = 0.5` scaled by local mutation density.
- Router thresholds: `H_norm ≤ 0.25` for accept, `>0.35` for escalate; `S_prior ≥ 0.35` accept floor; coverage ≥0.70 or agreement ≥0.60.
- BPR z-score gates: escalate when `z > 2.5`, send to refine when `1.5 < z ≤ 2.5`.

## 12. Optional: When to use FoldMason/MAFFT
- **Use** structural alignment when templates exhibit multi-domain shuffling, long insertions, or inconsistent columnning; it stabilizes AIP conditioning and boosts consensus across templates.
- **Skip** alignment for compact, single-domain families where retrieval already yields coherent mappings; default pipeline assumes this path.
- **Rule of thumb**: if >25% of template pairs outside loops show multi-modal distogram bins after aggregation, enable FoldMason + MAFFT to normalize columns before prior construction.

## 13. Open Questions / Next Steps
1. **AIP calibration**: refine \(σ\) and \(λ\) schedules versus identity/pLDDT on a validation set.
2. **6A vs 6B split**: confirm identity/coverage thresholds for template-warp vs. distogram realization.
3. **Router tuning**: revisit thresholds to satisfy throughput/SLO targets on pilot families.
4. **Prior injection mode**: compare Mode A vs. Mode B under varying family topologies.
5. **Training vs. zero-shot**: determine whether lightweight MiniFold fine-tuning with priors yields measurable gains.
6. **Operator experience**: decide which knobs belong in Hydra configs vs. CLI flags for automation.

## 14. Precomputation Implementation Plan
The following plan details how to operationalize all prerequisite assets—structural alignment, MSAs, priors, embeddings, and retrieval indices—before fast inference begins.

### 14.1 Data Layout & Orchestration
- **Workspace layout**
  - `data/raw/`: original FASTA sequences (`family.fasta`) and structure files (`*.pdb` / `*.cif`).
  - `data/intermediate/`: alignment artifacts (`foldmason/`, `msa/`), distilled priors (`priors/`), and embedding shards (`embeddings/`).
  - `data/index/`: FAISS indices and metadata manifests.
  - `configs/`: Hydra config tree (see §10.5) to define datasets, tool paths, batching, and caching behaviour.
- **Execution flow**
  1. `foldmason_align` → 2. `expand_msa` → 3. `distogram_prior` → 4. `esm2_embed` → 5. `build_faiss`.
  - Each stage writes a manifest JSON (Hydra Structured Config) enumerating outputs for downstream reuse.
- **Parallelism & caching**: chunk long families by sequence count; run independent Typer subcommands per chunk; rely on manifest timestamps + Hydra `job.override_dirname` for reproducible reruns.

### 14.2 FoldMason Structural Alignment (`foldmason_align`)
- **When to run**: use this stage when structural column consistency materially improves the prior (e.g., domain shuffling or long insertions). Skip it entirely for compact, single-domain families where simple sequence alignment suffices.
- **Inputs**: list of template structure paths, optional residue range masks.
- **Processing (spell-out subtasks)**
  1. **Pre-flight checks**
     - Confirm GPU/CPU resources match FoldMason requirements.
     - Validate every template path exists and has matching sequence IDs in the FASTA.
  2. **Alignment execution**
     - Launch FoldMason with a manifest of structure paths (`foldmason_align --config-name run --templates manifest.json`).
     - For skipped structures (e.g., missing domains), record the exclusion in `foldmason/manifest.json`.
  3. **Post-processing**
     - Extract superposed coordinates into `foldmason/superposition_{run_id}.pdb`.
     - Generate residue mapping tables (`structure_id`, `residue_index` → `msa_position`).
     - Store per-structure alignment RMSD / TM-scores for diagnostics.
- **Outputs**
  - `foldmason/superposition_{run_id}.pdb` (aligned stack).
  - `foldmason/mapping_{structure_id}.json` capturing MSA position mapping and per-residue pLDDT.
  - Manifest: `foldmason/manifest.json` listing structures processed, alignment score metrics, and quality flags.

### 14.3 Family MSA Expansion (`expand_msa`)
- **Inputs**: initial structural alignment manifest, family FASTA, optional external sequence databases.
- **Processing (spell-out subtasks)**
  1. **Seed alignment construction**
     - If FoldMason was run, convert residue↔MSA tables into an initial A3M; otherwise align reference structures with MUSCLE/MAFFT.
     - Normalize column numbering to match template indexing.
  2. **Profile generation**
     - Build an HMM profile from the seed alignment (`hmmbuild`).
     - Store profile checksum for reproducibility.
  3. **Database search**
     - Run HHblits or JackHMMER against selected databases (UniRef, custom family DBs) with configured e-value thresholds.
     - Append newly found sequences to the family FASTA if desired, tagging provenance.
  4. **Alignment expansion**
     - Merge hits into the seed alignment (`hmmalign` / `hhalign`).
     - Enforce optional domain segmentation via HMM boundaries to avoid misaligned insertions.
  5. **Quality control**
     - Compute per-sequence coverage and effective sequence counts.
     - Flag sequences with <30% coverage or high gap fractions.
- **Outputs**
  - `msa/family_{run_id}.a3m` and compressed `msa/family_{run_id}.stk`.
  - Coverage statistics per sequence (`msa/coverage.json`) for routing and weight computation.
  - Manifest including effective sequence count and warnings for low-coverage members.

### 14.4 Distogram Prior Construction (`distogram_prior`)
- **Inputs**: structural mapping manifests, full MSA, per-residue pLDDT.
- **Processing (spell-out subtasks)**
  1. **Template ingestion**
     - Load each template JSON, validate schema, and map residue indices to alignment columns.
     - Normalize pLDDT to `[0,1]` and store as float32 tensors.
  2. **Distance histogramming**
     - Convert each structure to MiniFold binning (`B=64`, 2–25 Å) using vectorized GPU kernels or numba.
     - Persist per-template histograms for audit if `--keep-intermediates` is set.
  3. **Weight computation**
     - Evaluate weights `coverage × pLDDT_i × pLDDT_j × identity_to_query` with configurable exponents (β, α).
     - Zero out weights for masked residues (e.g., `pLDDT < 0.5`).
  4. **Aggregation**
     - Sum weighted one-hot tensors into global logits; compute `P_family` via softmax.
     - Derive `W_family` as the total weight per pair.
  5. **Serialization**
     - Save dense tensors grouped by bucket length.
     - Generate sparse top-*k* (default 2) COO representations for efficient GPU loading.
     - Emit calibration metadata (mean/variance of weights, template contributions).
- **Outputs**
  - `priors/distogram_{bucket}.npz`: packed tensors (`P_family`, `W`).
  - `priors/distogram_{bucket}_sparse.npz`: coordinate list for top-*k* bins.
  - Calibration metadata (mean, variance of weights, template contributions) for debugging.

### 14.5 ESM-2 Embedding Shards (`esm2_embed`)
- **Inputs**: family FASTA, Hydra config specifying model checkpoint, batch size, masking fractions for pseudo-perplexity.
- **Processing (spell-out subtasks)**
  1. **Shard planning**
     - Partition sequences into length buckets and chunk sizes (`--max-batch`), recording shard manifests.
     - Warm caches by loading any existing embeddings and skipping already-processed IDs.
  2. **Forward passes**
     - Run batched ESM-2 forward passes (FP16/BF16 when supported) to extract required embeddings.
     - Persist intermediate logits if later layers (e.g., LoRA fine-tuning) are anticipated.
  3. **Pseudo-perplexity sampling**
     - For each sequence shard, mask 20–25% of residues, repeat 3× with different RNG seeds, and compute BPR.
     - Store both raw log-prob sums and normalized bits-per-residue.
  4. **Caching & indexing**
     - Write embeddings to `embeddings/{bucket}/sequence_id.pt` and update an index file for fast lookup.
     - Append BPR statistics to `embeddings/bpr_metrics.parquet`, including z-scores if calibration stats are provided.
  5. **Cleanup**
     - Optionally delete temporary logits to save disk, retaining hashes in the manifest.
- **Outputs**
  - `embeddings/{bucket}/sequence_id.pt` (torch tensors with single and pair features).
  - `embeddings/bpr_metrics.parquet` for routing thresholds.

### 14.6 FAISS Index Build (`build_faiss`)
- **Inputs**: embedding shards, optional dimensionality reduction parameters (PCA/OPQ).
- **Processing (spell-out subtasks)**
  1. **Vector assembly**
     - Load cached embeddings per bucket, apply optional pooling (mean, CLS token, etc.).
     - Standardize vectors (mean-centering / variance scaling) before training.
  2. **Dimensionality reduction**
     - Train PCA/OPQ transforms if configured; persist transformation matrices for reuse.
     - Apply transforms to all embeddings, keeping both raw and reduced copies when `--keep-raw` is set.
  3. **Index training & build**
     - Choose index type (e.g., IVF-PQ, HNSW) based on family size specified in Hydra config.
     - Train the index on a training split, then add all vectors from the bucket.
  4. **Quality evaluation**
     - Run recall@k and latency benchmarks using a validation subset.
     - Store neighbor graphs (`topk_neighbors.parquet`) and diagnostics in the manifest.
  5. **Serialization**
     - Save the index (`faiss_{bucket}.index`) and metadata JSON capturing hyperparameters, vector counts, and evaluation metrics.
- **Outputs**
  - `data/index/faiss_{bucket}.index` and associated `faiss_{bucket}.meta.json` capturing hyperparameters and embedding stats.

### 14.7 Typer + Hydra CLI Skeleton
- **Entry point**: `minifold/cli/familyfold.py` registered in `pyproject.toml` as `familyfold` console script.
- **Typer app (spell-out subtasks)**
  1. Define a root `Typer` application and attach subcommands: `foldmason-align`, `expand-msa`, `distogram-prior`, `esm2-embed`, `build-faiss`, `precompute-all`, and `verify-manifests`.
  2. For each subcommand, declare typed arguments (paths, run IDs, config overrides) and default Hydra config names.
  3. Implement dependency checks so stages fail fast when required manifests are missing.
- **Hydra integration**
  - Each subcommand loads a Hydra config (`configs/familyfold/{stage}.yaml`) describing tool paths, resource limits, input manifests, and output directories.
  - Support `--config-name` and `--config-path` overrides; expose critical overrides as CLI options (e.g., `--bucket-length`, `--faiss-metric`).
  - Use Hydra sweepers to distribute workloads across GPUs/hosts if needed (`familyfold sweep precompute-all hydra/sweeper=submitit`).
- **Logging & telemetry**
  1. Emit structured logs (JSON) per stage stored under `logs/{stage}/{timestamp}.jsonl`.
  2. Surface run metadata (elapsed time, input/output counts, cache hits) in Typer’s console output.
  3. Emit completion summary with paths to generated manifests for downstream automation.

### 14.8 Validation & Monitoring
- Implement a Typer subcommand `verify-manifests` that checks hashes, expected file counts, and basic sanity metrics (e.g., average coverage, BPR percentile histograms).
- Schedule periodic re-runs via `precompute-all --refresh` to update priors when new templates or sequences arrive.
- Integrate with CI/CD by adding smoke tests on small toy families to ensure each stage functions and Hydra configs resolve correctly.

## 15. End-to-End Orchestration (Pseudocode)
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

## 16. Testing Strategy
### Phase A — Unit Tests
- **TemplatePrep**: symmetric bins, correct normalization, pLDDT lengths.
- **FamilyPrior**: aggregation preserves dominant bins and weight support.
- **Retrieval**: deterministic top-*k* identity/coverage ordering on toy sets.
- **Alignment-Informed Prior (AIP)**: single-template case reproduces original distogram; γ scales with pLDDT.
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

## 17. Milestones (Vertical Slices)
- **M0 – Skeleton**: La-Proteina dataloader + MiniFold fast path → coordinates (no priors/router).
- **M1 – Templates & Priors**: per-template distograms aggregated to family priors; unit tests green.
- **M2 – Retrieval**: Protriever/FAISS wired with top-*k* manifests and clustering outputs.
- **M3 – AIP**: query-specific priors injected (Mode A), end-to-end inference functional.
- **M4 – Router**: uncertainty metrics, gating, one recycle with entropy-driven prior decay.
- **M5 – HomologCoPredict**: consensus + mutation-window specialization; group routing stabilized.
- **M6 – Calibration**: thresholds tuned on pilot family; acceptance and escalation SLOs met.
- **M7 – Hardening**: stress tests, dashboards, CI smoke tests, documentation finalized.

## 18. Risks & Guardrails
- **Bad priors**: mitigate with β exponent tuning, low-pLDDT masking, and `S_prior` gating.
- **Template topology mismatch**: monitor agreement `A`, decay priors during refinement, escalate when agreement collapses.
- **Coordinate instability**: enforce chirality checks, escalate rare divergence cases to ESMFold.
- **Memory spikes/OOM**: length bucketing, sparse priors, dynamic batch shrinking.
- **Over-recycling**: cap at one recycle; shadow-log outcomes to justify potential future adjustments.

## 19. Definition of Done
- **Accuracy**: median lDDT within 3 percentage points of ESMFold for accepted sequences on a held-out validation set.
- **Throughput**: ≥10× faster than ESMFold on ACCEPTed sequences; REFINE ≤15%, ESCALATE ≤5%.
- **Stability**: chirality flips <1%, deterministic reruns (hash-stable artifacts).
- **Homolog batching**: ≥85% identity clusters show measurable runtime reduction and mutation-localized geometry deltas.
- **Operational readiness**: Hydra/Typer CLI, manifests, dashboards, and validation commands documented and automated.
