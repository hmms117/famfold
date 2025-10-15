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
