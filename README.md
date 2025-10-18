# MiniFold

MiniFold if a fast model for single chain protein structure prediction. Built using the same protein language model as ESMFold, it achieves considerable speedups (up to 10-20x) and memory savings (up to 10x) at little to no cost in performance.

[Paper Link](https://openreview.net/pdf?id=1p9hQTbjgo)

## Installation

Then install minifold, clone the repository and install it with:

`pip install .`

> We recommend installing in a fresh python environment.

## Inference

To run prediction, use the following:

`python predict.py example.fasta --out_dir PATH --cache PATH`

> The fasta header will be used to name the output file, set them wisely!

Options are available:

- `kernels`: uses our custom triton kernels
- `compile`: use torch.compile with dynamic shapes enabled
- `model_size`: one of `12L` or `48L` (default) 
- `token_per_batch`: maximum number of tokens that fit in your GPU, by default 2048

  ## Training

  We train the model using AFDB proteins filtered to > 70 global plddt and selected for diversity using uniref30 as initial list, pre-filtering. You may use the provided train.py script and the YAML configs under `configs`.

  ## Homolog cluster sampling helper

  The repository also provides a lightweight utility for preparing small homolog clusters (e.g. selecting a few 85% identity groups from the `gh5_21` family for smoke tests). You can either run MMseqs2 locally or reuse the official AlphaFold DB (AFDB/AF-PHFoldDB) cluster lookup tables.

  To drive MMseqs2 end-to-end, run:

  ```
  python scripts/sample_mmseqs_clusters.py data/gh5_21/all_sequences.fasta \
      out/gh5_21_sample --structures-dir data/gh5_21/structures --clusters 5 --identity 0.85
  ```

  The script will:

  - Run `mmseqs createdb`/`mmseqs cluster`/`mmseqs createtsv` at the requested identity cutoff (unless an existing TSV is provided via `--clusters-tsv`).
  - Randomly pick the requested number of clusters (deterministic if `--seed` is set).
  - Emit per-cluster FASTA files and copy matching structure files (`.cif`, `.mmcif`, `.pdb`, or `.cif.gz`) when available.
  - Produce `manifest.json` summarising the sampled clusters.

  If you already downloaded the AFDB clustering release (for example to focus on GH5_21 homologs), you can bypass MMseqs2 by passing the provided `cluster_lookup.tsv` plus an optional UniProt filter list:

  ```
  python scripts/sample_mmseqs_clusters.py data/afdb/gh5_21_sequences.fasta \
      out/gh5_21_afdb_sample \
      --afdb-cluster-lookup data/afdb/cluster_lookup.tsv \
      --afdb-include-ids data/afdb/gh5_21_uniprot_ids.txt \
      --structures-dir data/afdb/structures --clusters 5
  ```

  This second mode keeps only the AlphaFold entries (or UniProt accessions) listed in `gh5_21_uniprot_ids.txt`, randomly samples clusters from the lookup table, writes per-cluster FASTA files, and copies matching structures when present. Use this to quickly stage AFDB/UniProt comparisons without re-running clustering locally.

  The new `scripts/prepare_afdb_subset.py` helper automates the download/filter process when working with AlphaFold DB releases. Supply a list of identifiers plus the official AFDB `cluster_lookup.tsv` and `sequences.fasta` URLs (see [the AFDB download portal](https://alphafold.ebi.ac.uk/download)) and the script will stage filtered copies, fetch structures, and invoke the sampler for you:

  ```
  python scripts/prepare_afdb_subset.py out/gh5_21_subset \
      --ids data/afdb/gh5_21_uniprot_ids.txt \
      --cluster-lookup https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/cluster_lookup/cluster_lookup.tsv.gz \
      --sequences https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/sequences.fasta.gz \
      --structures-dir data/afdb/structures --clusters 5 --seed 1234
  ```

  For environments without network access, a miniature AlphaFold-style dataset is bundled under `data/afdb_demo/`. You can exercise the full workflow (including structure copying) via:

  ```
  python scripts/prepare_afdb_subset.py out/afdb_demo_run --demo --clusters 3 --seed 7
  ```

  See `python scripts/sample_mmseqs_clusters.py --help` and `python scripts/prepare_afdb_subset.py --help` for additional options (threads, coverage thresholds, retaining raw MMseqs2 databases, AFDB filtering, download caching, etc.).

  ## Kernels

  We developed two triton kernels for this work. You can find them [here](https://github.com/jwohlwend/minifold/tree/main/minifold/model/kernels).

## Cite

```
@article{
  wohlwend2025minifold,
  title={MiniFold: Simple, Fast, and Accurate Protein Structure Prediction},
  author={Jeremy Wohlwend and Mateo Reveiz and Matt McPartlon and Axel Feldmann and Wengong Jin and Regina Barzilay},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2025},
  url={https://openreview.net/forum?id=1p9hQTbjgo},
  note={Featured Certification}
}
```
