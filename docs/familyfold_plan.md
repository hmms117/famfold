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
1. **Family structural alignment**
   - Align the provided family structures using FoldMason (or an equivalent structural alignment tool).
   - Seed a multiple sequence alignment (MSA) of the family from the structural superposition.
2. **Family-wide MSA expansion**
   - Add remaining sequences to the MSA via sequence-profile alignment tools.
   - Optionally supplement with sequences retrieved from external databases for better coverage.
3. **Family geometry prior construction**
   - Convert each known structure into a Cα–Cα distogram (MiniFold binning: e.g., 2–25 Å, 64 bins).
   - Map residue indices to MSA positions; weight each distogram bin by per-residue pLDDT and alignment coverage.
   - Aggregate across templates to obtain `P_family(i, j, b)` and a confidence weight `W(i, j)` for every MSA position pair.
   - Persist the prior (e.g., compressed npz per length bucket) for reuse across queries.
4. **Sequence embedding & retrieval layer**
   - Embed all family sequences with ESM-2 (consistent with MiniFold inputs).
   - Build a FAISS index over embeddings for fast nearest-neighbor lookup.
   - For each query batch, retrieve the top *k* folded templates to reweight the family prior toward the closest neighbors.
5. **MiniFold pair feature assembly**
   - Tile single-sequence embeddings into pair representations as in vanilla MiniFold.
   - Concatenate the prior distogram (either soft probabilities or one-hot argmax) and `W(i, j)` into the pair features before the MiniFormer stack.
6. **Fast MiniFold inference**
   - Use the 10-layer MiniFold variant without a structure module and with the parameter-free coordinate realizer.
   - Run one forward pass (zero or one recycle) while injecting the prior via distogram recycling.
7. **Coordinate realization**
   - Recover coordinates with the MiniFold MDS realizer (shortest-path completion → classical MDS → LBFGS stress majorization).
8. **Confidence gating & refinement**
   - Compute distogram entropy and/or a pLDDT head to estimate reliability.
   - Incorporate sequence-only uncertainty from ESM-2 (see §6) to pre-filter likely out-of-distribution (OOD) queries.
   - Accept high-confidence predictions immediately.
   - For low-confidence cases, trigger a refinement stage: additional MiniFold recycles (with distogram recycling) or fall back to ESMFold.
9. **Batch orchestration**
   - Bucket sequences by length (e.g., 256/384/512 AA) to minimize padding.
   - Exploit MiniFold's 20–40× memory savings to run large batches with FP16/BF16 precision.

## 5. Component Details
- **Family prior weighting**: `weight = coverage(i, j) × pLDDT_i × pLDDT_j × identity_to_query`.
- **Prior storage**: Keep the top-*k* bins per pair (sparse representation) to reduce memory. Store alongside an effective counts tensor.
- **Retrieval tuning**: Choose FAISS distance metric compatible with embeddings (inner product for normalized vectors). Cache results for repeated batches.
- **MiniFold modifications**:
  - Add channels for prior distogram and weights to the pair representation.
  - Support optional recycling where the predicted distogram is argmaxed, one-hot encoded, and concatenated back in (as per MiniFold Table 1).
  - Maintain compatibility with both 10-layer (fast) and 48-layer (higher-accuracy) variants.
- **Confidence metrics**: Distogram entropy directly correlates with lDDT (MiniFold Figure 4). Combine with head-derived pLDDT for robust gating.

## 6. Sequence-Only OOD Detection (ESM-2)
- **Pseudo-perplexity / bits-per-residue (BPR)**:
  - Mask ~20–25% of positions uniformly at random in the query sequence, run ESM-2 to obtain \(p(x_i \mid x_{\setminus i})\), and repeat this masking 2–3 times.
  - Estimate BPR as \(-\frac{1}{L \log 2} \sum_{i=1}^{L} \log p(x_i \mid x_{\setminus i})\); lower values indicate in-distribution sequences, higher values flag potential OOD cases.
  - Calibrate BPR by building a reference distribution from 50–100k UniRef50 sequences or a trusted in-distribution subset, then report z-scores and percentiles (e.g., “BPR = 3.1 bits/aa; in-distribution percentile 8th; z = +1.8”).
- **Routing usage**:
  - Use calibrated BPR thresholds to decide whether to route the sequence to the fast MiniFold pass, trigger extra recycling, or escalate directly to ESMFold/AF2-style workflows.
  - Log BPR diagnostics alongside distogram entropy/pLDDT to trace OOD behavior and refine thresholds.
- **Adaptive modeling option (extra)**: When BPR spikes for many related sequences, consider unfreezing or LoRA-finetuning the ESM-2 trunk on in-family data to tighten the language model’s coverage. Keep this as an optional path, gated behind clear monitoring so the default pipeline stays training-free.

## 7. Fallback & Escalation Policy
- Escalate a sequence to additional MiniFold recycles or ESMFold when:
  - Distogram entropy exceeds a preset threshold.
  - No close neighbors (above identity cutoff) are found via FAISS.
  - MSA coverage is poor or length mismatches occur.
- Log decisions for auditing and to refine thresholds.

## 8. Performance & Engineering Considerations
- Cache ESM-2 embeddings and family priors to avoid recomputation.
- Ensure static shapes per length bucket to maximize Triton kernel efficiency and Torch compile benefits.
- Consider domain boundary detection for long/low-homology sequences and fold subdomains separately with loop closure.
- Monitor GPU memory; MiniFold reports 100–200× throughput gains in the Evoformer-equivalent component and 15–20× end-to-end speedups for larger models.

## 9. Open Questions / Next Steps
1. Should the prior combination use soft blending (probabilities) or strict one-hot recycling per template?
2. What thresholds on entropy/pLDDT best trade speed for accuracy within your dataset?
3. Do we need a lightweight training phase to fine-tune MiniFold on family-conditioned priors, or is zero-shot sufficient?
4. How should we expose batching and gating configuration (CLI flags vs. config files)?

Answering these will finalize the implementation roadmap for FamilyFold.

## 10. Precomputation Implementation Plan
The following plan details how to operationalize all prerequisite assets—structural alignment, MSAs, priors, embeddings, and retrieval indices—before fast inference begins.

### 10.1 Data Layout & Orchestration
- **Workspace layout**
  - `data/raw/`: original FASTA sequences (`family.fasta`) and structure files (`*.pdb` / `*.cif`).
  - `data/intermediate/`: alignment artifacts (`foldmason/`, `msa/`), distilled priors (`priors/`), and embedding shards (`embeddings/`).
  - `data/index/`: FAISS indices and metadata manifests.
  - `configs/`: Hydra config tree (see §10.5) to define datasets, tool paths, batching, and caching behaviour.
- **Execution flow**
  1. `foldmason_align` → 2. `expand_msa` → 3. `distogram_prior` → 4. `esm2_embed` → 5. `build_faiss`.
  - Each stage writes a manifest JSON (Hydra Structured Config) enumerating outputs for downstream reuse.
- **Parallelism & caching**: chunk long families by sequence count; run independent Typer subcommands per chunk; rely on manifest timestamps + Hydra `job.override_dirname` for reproducible reruns.

### 10.2 FoldMason Structural Alignment (`foldmason_align`)
- **Inputs**: list of template structure paths, optional residue range masks.
- **Processing**
  - Invoke FoldMason in batch mode with GPU acceleration where available.
  - Export superposed coordinates and residue mapping tables (`structure_id`, `residue_index` → `msa_position`).
- **Outputs**
  - `foldmason/superposition_{run_id}.pdb` (aligned stack).
  - `foldmason/mapping_{structure_id}.json` capturing MSA position mapping and per-residue pLDDT.
  - Manifest: `foldmason/manifest.json` listing structures processed, alignment score metrics, and quality flags.

### 10.3 Family MSA Expansion (`expand_msa`)
- **Inputs**: initial structural alignment manifest, family FASTA, optional external sequence databases.
- **Processing**
  - Seed an MSA using the residue mappings; run profile-sequence alignment tools (e.g., HHblits/JackHMMER) to add remaining family sequences.
  - Optionally enforce domain segmentation via HMM boundaries to avoid misaligned insertions.
- **Outputs**
  - `msa/family_{run_id}.a3m` and compressed `msa/family_{run_id}.stk`.
  - Coverage statistics per sequence (`msa/coverage.json`) for routing and weight computation.
  - Manifest including effective sequence count and warnings for low-coverage members.

### 10.4 Distogram Prior Construction (`distogram_prior`)
- **Inputs**: structural mapping manifests, full MSA, per-residue pLDDT.
- **Processing**
  - Convert each structure to MiniFold binning (`B=64`, 2–25 Å). Use numba/torch to parallelize per-template histograms.
  - Map to MSA coordinates and compute weights `coverage × pLDDT_i × pLDDT_j × identity_to_query`.
  - Aggregate across templates, storing both dense tensors and sparse top-*k* views (for GPU efficiency).
- **Outputs**
  - `priors/distogram_{bucket}.npz`: packed tensors (`P_family`, `W`).
  - `priors/distogram_{bucket}_sparse.npz`: coordinate list for top-*k* bins.
  - Calibration metadata (mean, variance of weights, template contributions) for debugging.

### 10.5 ESM-2 Embedding Shards (`esm2_embed`)
- **Inputs**: family FASTA, Hydra config specifying model checkpoint, batch size, masking fractions for pseudo-perplexity.
- **Processing**
  - Run batched ESM-2 forward passes (FP16 where possible) to extract last-layer embeddings and logits required for BPR.
  - Cache embeddings per length bucket to align with inference batching.
  - Compute pseudo-perplexity estimates on-the-fly (mask 20–25% residues, repeat 3×) and log calibrated statistics if a reference distribution is supplied.
- **Outputs**
  - `embeddings/{bucket}/sequence_id.pt` (torch tensors with single and pair features).
  - `embeddings/bpr_metrics.parquet` for routing thresholds.

### 10.6 FAISS Index Build (`build_faiss`)
- **Inputs**: embedding shards, optional dimensionality reduction parameters (PCA/OPQ).
- **Processing**
  - Train PCA/OPQ transforms when configured; apply to embeddings.
  - Build IVF/HNSW index tuned for family size; record recall/latency benchmarks.
  - Store neighbor graphs (`topk_neighbors.parquet`) for debugging and as a warm cache.
- **Outputs**
  - `data/index/faiss_{bucket}.index` and associated `faiss_{bucket}.meta.json` capturing hyperparameters and embedding stats.

### 10.7 Typer + Hydra CLI Skeleton
- **Entry point**: `minifold/cli/familyfold.py` registered in `pyproject.toml` as `familyfold` console script.
- **Typer app**: root command `familyfold` with subcommands: `foldmason-align`, `expand-msa`, `distogram-prior`, `esm2-embed`, `build-faiss`, and `precompute-all` (runs the full chain respecting dependencies).
- **Hydra integration**
  - Each subcommand loads a Hydra config (`configs/familyfold/{stage}.yaml`) describing tool paths, resource limits, input manifests, and output directories.
  - Support `--config-name` and `--config-path` overrides; expose critical overrides as CLI options (e.g., `--bucket-length`, `--faiss-metric`).
  - Use Hydra sweepers to distribute workloads across GPUs/hosts if needed (`familyfold sweep precompute-all hydra/sweeper=submitit`).
- **Logging & telemetry**
  - Structured logs (JSON) per stage stored under `logs/{stage}/{timestamp}.jsonl`.
  - Emit completion summary with paths to generated manifests for downstream automation.

### 10.8 Validation & Monitoring
- Implement a Typer subcommand `verify-manifests` that checks hashes, expected file counts, and basic sanity metrics (e.g., average coverage, BPR percentile histograms).
- Schedule periodic re-runs via `precompute-all --refresh` to update priors when new templates or sequences arrive.
- Integrate with CI/CD by adding smoke tests on small toy families to ensure each stage functions and Hydra configs resolve correctly.
