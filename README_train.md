# MiniFold AFESM Training Cookbook

This note captures the exact steps we used to stage the AFESM dataset and kick off a MiniFold training run. The flow is intentionally explicit so you can reproduce it or hand it off without digging through chat logs.

---

## 1. Stage AFDB/ESMFold structures (one time per release)

Stage both the AlphaFold DB representatives and the MGYP ESMFold models under `/var/tmp/famfold/afesm`. The script symlinks (or copies, if you add `--copy`) the appropriate files into a hash-bucketed layout. Add `--convert-pdb` if you want the staged MGYP PDBs converted to CIF (and the original PDBs removed once the conversion succeeds).

```bash
srun_now48 -p medium --gres=shard:40 --ntasks=1 --cpus-per-task=48 --mem=0 --time=12:00:00 \
  uv run python scripts/stage_afesm_dataset.py \
    --clusters /z/pd/afesm/1-AFESMClusters-repId_memId_cluFlag_taxId_biomeId.tsv \
    --rep-list /z/pd/afesm/afesm_afdb_reps.tsv \
    --afdb-root /z/pd/afdb \
    --esmf-root /ceph/cluster/bioinf/structure_models/esmatlas \
    --out-root /var/tmp/famfold/afesm \
    --workers 128 \
    --log-interval 500000 \
    --convert-pdb    # optional: convert staged PDBs to CIF alongside representatives
```

- CIFs land in `/var/tmp/famfold/afesm/cifs/<last3>/AF-…cif.gz`.
- The same structures are mirrored as PDBs under `/var/tmp/famfold/afesm/pdbs/<last3>/MGYP….pdb` (useful for template workflows).
- Any misses are logged under `/var/tmp/famfold/afesm/logs/missing_{cifs,pdbs}.txt`.

If you ever need physical copies (instead of symlinks), rerun the command with `--copy`.

---

## 2. Build train/val/test splits of representative IDs

The staged CIF filenames follow the UniProt ID, so we can slice the dataset using only the representative list. This example creates a 90/5/5 split in `/var/tmp/famfold/afesm/splits/reps/`.

```bash
python - <<'PY'
import random, pathlib

reps = pathlib.Path("/z/pd/afesm/afesm_afdb_reps.tsv")
out  = pathlib.Path("/var/tmp/famfold/afesm/splits/reps")
out.mkdir(parents=True, exist_ok=True)

ids = [line.strip() for line in reps.open() if line.strip()]
random.Random(20251022).shuffle(ids)

n = len(ids)
n_train = int(0.90 * n)
n_val = int(0.05 * n)

splits = {
    "train.txt": ids[:n_train],
    "val.txt":   ids[n_train:n_train+n_val],
    "test.txt":  ids[n_train+n_val:],
}

for name, subset in splits.items():
    (out / name).write_text("\n".join(subset) + "\n")
PY
```

You’ll now have:

```
/var/tmp/famfold/afesm/splits/reps/train.txt
/var/tmp/famfold/afesm/splits/reps/val.txt
/var/tmp/famfold/afesm/splits/reps/test.txt
```

Each file contains one UniProt ID per line. The MiniFold dataloader filters the staged CIF tree by these IDs on the fly.

> **Note:** the companion PDB mirror lives at `/var/tmp/famfold/afesm/pdbs/`. MiniFold training uses the CIF tree above; keep the PDB path handy for template or analysis workflows (e.g., `export FAMFOLD_PDB_ROOT=/var/tmp/famfold/afesm/pdbs`).

---

## 3. Launch MiniFold training

Two hydra configs are already set up to consume the staged data:

- `configs/afesm_stage1.yaml` – training (uses the train/val lists)
- `configs/afesm_stage1_eval.yaml` – template for evaluation (point it at the test list if needed)

Kick off a training run like so:

```bash
export FAMFOLD_DATA_ROOT=/var/tmp/famfold/afesm/cifs   # CIF root (a PDB mirror lives alongside at /var/tmp/famfold/afesm/pdbs)
srun_now48 -p medium --gres=shard:40 --ntasks=1 --cpus-per-task=8 --mem=128G --time=04:00:00 \
  uv run python train.py configs/afesm_stage1.yaml \
    --output /var/tmp/famfold/checkpoints/afesm_stage1_run
```

Key details:

- `MiniFoldDataModule` now understands `.cif.gz`, and the new `train_ids` / `val_ids` options point it at the split lists.
- SaESM/ESM weights are expected under `/var/tmp/checkpoints` (already set via config).
- Lightning checkpoints/logs land under `/var/tmp/famfold/checkpoints` (config default).

To evaluate on the held-out set, copy `configs/afesm_stage1_eval.yaml`, swap `train_ids`/`val_ids` for `/var/tmp/famfold/afesm/splits/reps/test.txt`, and run with reduced resources if desired.

---

## 4. Post-run sanity checks (optional but recommended)

After the staging job finishes, it’s worth confirming that the counts look sane:

```bash
find /var/tmp/famfold/afesm/cifs -type f | wc -l      # total CIFs staged
ls /var/tmp/famfold/afesm/logs                        # ensure missing_*.txt are empty/small
wc -l /var/tmp/famfold/afesm/splits/reps/*.txt        # confirm split sizes
```

If you see non-empty “missing” logs, eyeball those IDs—they may be MGYP-only entries or reps that moved between releases.

---

That’s everything needed to reproduce the current AFESM training setup. If you ever want split-specific symlink trees, or to incorporate the MGYP structures into the model logic, both are easy follow-ups—just shout.
