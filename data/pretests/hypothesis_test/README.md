# Homolog Template Hypothesis Benchmark Families

This directory bundles five representative protein families for running the homolog-template hypothesis test. The set balances in-distribution controls against metagenomic edge cases so the template integration pipeline can be validated quickly before scaling up.

## Families

| ID | Family | Category | Example structure(s) | Notes |
| --- | --- | --- | --- | --- |
| `ubiquitin_human` | Ubiquitin-like beta-grasp | In-distribution (easy) | 1UBQ chain A (1.8 Å) | Canonical Minifold training target for smoke tests and regression guardrails. |
| `protein_gb1` | Ig-binding beta-sandwich | In-distribution (medium) | 2GB1 chain A (1.1 Å) | Fast-folding two-state domain that probes template effects on β-sheet packing. |
| `rpl41e_mj` | Ribosomal L41e | OOD metagenomic (known fold) | 4V6U chain ZD (3.2 Å) | Lysine-rich archaeal ribosomal protein; limited UniProt homologs but templates exist in cryo-EM reconstructions. |
| `beta_propeller_6kwc` | Metagenomic beta-propeller enzyme | OOD metagenomic (known fold) | 6KWC chain A (1.9 Å) | Large JGI-derived propeller with asymmetric blades that stress template alignment. |
| `microviridin_6a5j` | Microviridin RiPP macrocycle | OOD metagenomic (hard) | 6A5J chain A (NMR ensemble) | Macrocyclic peptide with atypical constraints—difficult to model without high-quality templates. |

Each sequence is stored as a single-entry FASTA file under `sequences/`. Metadata (organism, recommended split, structure references, and free-form notes) lives in `targets.json` and is consumed by the hypothesis-test configuration loader.

## Usage

1. Point the hypothesis test configuration at `data/pretests/hypothesis_test/targets.json`.
2. Run `python -m pretests.hypothesis_test.cli run pretests/hypothesis_test/example_config.json --pilot` to validate the pilot ID/OOD mix (`ubiquitin_human` and `rpl41e_mj`).
3. Trigger a full sweep with `--split full_id` / `--split full_ood` or omit the flag to cover all five families.

> **Note:** Structure coordinates are not vendored in this repository. The metadata includes PDB identifiers so structures can be retrieved via RCSB or EMDB when evaluating model accuracy.
