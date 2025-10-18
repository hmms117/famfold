# Tier 1 – Data Foundations & Template Tooling

This document restores the full detail for data ingestion, template processing,
and prior construction. It also captures Tier 2 and Tier 3 notes relevant to the
data layer.

---

## 1. Objectives
- Parse template structures into reusable JSON/NPZ artifacts with sequences,
  pLDDT, and MiniFold-compatible distograms.
- Aggregate per-template information into family-level priors for multiple
  length buckets.
- Maintain deterministic storage layouts, manifests, and QC logs.

---

## 2. Stage [01] TemplatePrep

### Responsibilities
Transform raw structures under `families/<family_id>/raw/structures/` into
per-template feature blobs, QC summaries, and a retrieval-ready FASTA bank.

### Inputs
- PDB/mmCIF files (AFDB or experimental). AFDB encodes pLDDT in the `B-factor`.
- *(Optional)* FoldMason residue↔MSA maps when structural column normalisation is
  required (Tier 2 option).

### Outputs
- `templates/index.tsv`
  - Columns: `template_id, chain, L, mean_plddt, p5_plddt, p95_plddt, source_file`.
- `templates/<template_id>.json.gz`
  - Schema:
    ```json
    {
      "template_id": "P12345_A",
      "L": 436,
      "seq": "MKV...",
      "plddt": [float, ... L],
      "bins": "uint16 base64 of [L,L]",
      "bin_edges": [64 floats],
      "meta": {
        "source": "AFDB",
        "pdb_path": ".../P12345_A.pdb.gz",
        "date": "YYYY-MM-DD"
      }
    }
    ```
- `templates/qc_report.json` — drop reasons, counts, summary stats.
- `retrieval/template_bank.faa` — concatenated template sequences.
- `logs/template_prep.jsonl` — per-template timing, success/failure metadata.

### Checklist
1. Enumerate structure files; derive `template_id` (`UNIPROTID_CHAIN` or stem).
2. Parse chains; extract sequence, residue indices, and Cα coordinates.
3. Compute pairwise distances; discretise into 64 bins (2–25 Å; last bin ≥25 Å).
4. Pull per-residue pLDDT (AFDB `B-factor` scaled to [0, 100]).
5. Reject templates with mean pLDDT < 70; record in `qc_report.json`.
6. Base64-encode `uint16[L,L]` bin grid for storage efficiency.
7. Write `index.tsv`, JSON blobs, and FASTA bank; append run metadata/log lines.
8. Spot-check ≥3 templates to verify pLDDT lengths, symmetric bins, diagonals.

### Sanity Checks
- `len(plddt) == L` for each template.
- Distogram symmetry and diagonal sentinel values.
- FASTA entries match sequences embedded in JSON.

### Tier 2 / Tier 3 Notes
- **Tier 2**: attach FoldMason residue↔MSA mappings and structural alignment
  provenance to each JSON (store under `meta.alignment`).
- **Tier 3**: support multi-model ensembles per template (store multiple
  distograms + weights) and capture experimental uncertainty metrics beyond
  pLDDT.

---

## 3. Stage [02] FamilyPrior

### Responsibilities
Fuse per-template distograms into dense and sparse family priors per bucket and
track aggregate weighting statistics.

### Inputs
- Template JSON blobs from Stage [01].
- Bucket definitions from `configs/famfold/buckets.yaml` (256/384/512 AA).

### Outputs
- `priors/distogram_{256,384,512}.npz`
  - `P_family`: `[L_b, L_b, 64]` float32 probabilities/logits.
  - `W_family`: `[L_b, L_b]` float16 aggregated strengths.
  - `bin_edges`: `[64]` float32 (2..25 Å edges; shared across templates).
- `priors/distogram_{bucket}_sparse.npz`
  - COO fields: `rows`, `cols`, `bin_ids[2]`, `bin_probs[2]` (top-2 bins/pair).
- `priors/metadata.json`
  - Template contributions, weight histograms, coverage statistics, hashes.
- `logs/family_prior.jsonl` — run metadata, bucket runtime, template counts.

### Weighting & Aggregation
- Per-template coverage weight:
  `W_t_global[i,j] = (plddt_i * plddt_j / 10_000)^β * coverage_ij` with β = 1.0.
- Build logits per bin: sum weighted one-hot contributions across templates.
- Apply softmax over bins → `P_family`; sum weights → `W_family`.
- Persist top-2 bins per pair for sparse loading.

### Checklist
1. Load bucket definition; pad/mask templates to bucket length.
2. Compute `W_t_global` for each template with coverage masks.
3. Aggregate logits and weights across templates.
4. Export dense `npz` (float32 logits or probabilities + float16 weights).
5. Derive sparse COO representation with top-2 bins per pair.
6. Record metadata: templates used, rejected indices, weight quantiles.
7. Validate dense↔sparse round-trip (max absolute diff ≤1e-4 on probabilities).
8. Produce QC visuals (optional Tier 2: heatmaps stored under `reports/`).

### Tier 2 / Tier 3 Notes
- **Tier 2**: persist FoldMason-aligned column stats and per-residue coverage to
  help downstream homolog co-prediction.
- **Tier 3**: explore low-rank factorizations of `P_family` for ultra-long
  sequences; integrate template provenance scoring (e.g., AFDB release version).

---

## 4. Data Schemas & Storage Conventions

Summarised schemas live in `docs/SCHEMAS.md`. Ensure every artifact captures
`family_id`, `bucket`, `version`, `created_at`, and content hash for reproducible
reruns.

### Key Tables
- `templates/index.tsv`
- `manifests/run_manifest.json`
- `priors/metadata.json`

### Tier 2 / Tier 3 Notes
- **Tier 2**: materialise DuckDB views over template/priors metadata for quick
  QA dashboards.
- **Tier 3**: integrate data versioning with DVC/LakeFS for cross-cluster sync.

---

## 5. Dependencies & Tooling
- Primary libraries: `biopython` (structure parsing), `numpy`, `scipy`,
  `h5py`/`zarr` (optional), `pyblake3` for `qhash` computation.
- Ensure deterministic behaviour with fixed random seeds and hashed manifests.

---

Refer to [familyfold_tier1_retrieval.md](familyfold_tier1_retrieval.md) for
retrieval, alignments, and per-query prior sharpening details.
