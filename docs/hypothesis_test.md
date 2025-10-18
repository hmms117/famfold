# Hypothesis Test Plan: Homolog Template Structures for Minifold

## Hypothesis
Incorporating homolog template structures into Minifold inference improves structure prediction accuracy on in-distribution UniProt sequences while maintaining or exceeding performance on out-of-distribution sequences compared with base Minifold, ESMFold, and Boltz2.

## Experimental Conditions
1. **Base Minifold** – standard inference without template features.
2. **Minifold + Homolog Templates** – Minifold inference augmented with retrieved homolog template structures.
3. **ESMFold Baseline** – external baseline using ESMFold predictions.
4. **Boltz2 Baseline** – external baseline using Boltz2 predictions.

## Datasets
- **In-Distribution (ID)**
  - Sample ~50 protein targets from UniProt that are close to Minifold's training distribution.
  - Ensure available ground-truth structures (e.g., PDB structures mapped to UniProt entries).
  - Seed quick-start experiments with the curated `data/benchmarks/hypothesis_test` families (`ubiquitin_human`, `protein_gb1`).
- **Out-of-Distribution (OOD)**
  - Select ~50 UniProt proteins with low sequence identity (<30%) to Minifold training data.
  - Include ~50 non-UniProt PDB chains absent from UniProt annotations or far from the training set.
  - Use the provided metagenomic pilot families (`rpl41e_mj`, `beta_propeller_6kwc`, `microviridin_6a5j`) for early validation.
- **Pilot Subset**
  - Pick 5 ID and 5 OOD targets to validate the pipelines before full benchmarking.

## Template Retrieval Pipeline
1. Generate multiple sequence alignments using MMseqs2 or an equivalent fast homology search.
2. Retrieve structural templates via Foldseek or PDB-mmseqs pipeline.
3. Filter templates by:
   - Sequence identity thresholds (e.g., 30–95%).
   - Coverage (>70% of query length).
   - Resolution (<3.5 Å when available).
4. Convert template structures into Minifold-compatible features (e.g., distance matrices, orientation, mask) following the existing template input spec.
5. Cache template features per target for reproducibility.

## Inference Procedures
- **Base Minifold**: Use `predict.py` with default configuration.
- **Minifold + Templates**:
  - Extend the inference script to load cached template features.
  - Merge template-derived embeddings into the model input, ensuring consistent preprocessing.
- **ESMFold / Boltz2**:
  - Run official inference scripts on the same sequences.
  - Record runtime and resource usage.

## Quick Pilot Test
1. Run the pipeline on the pilot subset (5 ID + 5 OOD targets).
2. Confirm:
   - Successful template retrieval and integration.
   - Outputs generated for all models.
   - Preliminary metrics (TM-score, lDDT) within expected ranges.
3. Inspect any failures (e.g., missing templates) and adjust filters.

## Full Benchmark Execution
1. Execute inference for all targets under each experimental condition.
2. Collect predictions in a standardized format (e.g., PDB files).
3. Measure runtime, memory usage, and template retrieval latency.

## Evaluation Metrics
- **Structural Accuracy**: TM-score, lDDT, RMSD (global and per-domain if applicable).
- **Confidence Measures**: pLDDT or equivalent per model.
- **Resource Metrics**: wall-clock time, GPU utilization.

## Analysis Plan
1. Compute mean, median, and standard deviation for each metric across ID and OOD datasets separately.
2. Perform paired statistical tests (e.g., Wilcoxon signed-rank) comparing Minifold variants with baselines.
3. Analyze differential gains:
   - Templates vs. Base Minifold (ID vs. OOD).
   - Templates vs. ESMFold and Boltz2.
4. Examine cases where templates degrade performance; inspect template quality and alignment errors.
5. Summarize findings with tables and plots (e.g., boxplots, scatter plots of TM-score).

## Reporting
- Document the experimental setup, datasets, and template pipeline.
- Present metrics and statistical test results for ID and OOD regimes.
- Highlight whether template augmentation yields significant gains, and note any trade-offs (e.g., runtime).
- Provide actionable recommendations for integrating template support into Minifold.

