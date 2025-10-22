# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `minifold/`; `famfold/` holds stage logic, `model/` wraps Lightning modules and Triton kernels, and `utils/` provides protein geometry helpers.
- Hydra configs land in `configs/`, experiment scaffolds in `experiments/` and `hypothesis_test/`, while CLI entry points (`predict.py`, `train.py`) and data prep scripts rely on assets under `data/` (notably `afdb_demo` for smoke runs).
- Tests inside `tests/` mirror FamilyFold stage numbers to keep scenarios aligned with the pipeline.

## Build, Test, and Development Commands
- Bootstrap an environment with `uv venv --python 3.11` then `UV_LINK_MODE=copy uv pip install -e .`.
- Run checks with `uv run pytest` (narrow scope via `-k stage1` when iterating) and sanity-check inference through `uv run python predict.py example.fasta --out_dir out --cache cache`.
- Kick off training using `uv run python train.py --config configs/stage1.yaml --data-root $FAMFOLD_DATA_ROOT`.

## Coding Style & Naming Conventions
- Adhere to PEP 8: four-space indentation, `snake_case` functions, PascalCase classes, uppercase constants (see `minifold/famfold/template_prep.py`).
- Type hints and dataclasses are expected for new public APIs; keep docstrings focused on tensor shapes and units where they matter.

## Testing Guidelines
- Tests run under `pytest`; name files `test_<feature>.py`, fixtures `fixture_<purpose>`, and keep imports relative to the repo root as configured in `tests/conftest.py`.
- Prefer the bundled `data/afdb_demo/` assets for regression coverage, and guard MMseqs2-dependent cases with markers plus skip messages.

## Commit & Pull Request Guidelines
- Commit subjects stay in the imperative mood (`Add FamilyFold TemplatePrep utilities`, `Refine Stage 01 distogram bins`); add concise bodies when configs or data paths change.
- PRs should call out the affected stage/module, enumerate validation commands, link issues, and attach key artifacts (loss curves, checkpoint locations, config diffs).

## Environment & HPC Notes
- Prefer the medium GPU queue: `sbatch -p medium --gres=shard:40 --time=04:00:00 --wrap="uv run python train.py --config ..."`.
- Surface dataset and cache locations via environment variables (`FAMFOLD_DATA_ROOT`, `MMSEQS_TMP_DIR`) so jobs resume cleanly across partitions.
