"""Routines for invoking Minifold inference within the benchmark."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

from .config import MinifoldRunSettings, TargetConfig

LOGGER = logging.getLogger(__name__)


def _write_fasta(targets: Sequence[TargetConfig], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for target in targets:
            description = target.metadata.get("description", "")
            header = target.identifier if not description else f"{target.identifier} {description}"
            handle.write(f">{header}\n")
            sequence = target.sequence
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


class MinifoldInferenceRunner:
    """Wrapper around the :mod:`predict` CLI for programmatic use."""

    def __init__(self, settings: MinifoldRunSettings, cache_dir: Path) -> None:
        self.settings = settings
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def run(self, targets: Iterable[TargetConfig], work_dir: Path, label: str) -> Path:
        """Run Minifold on the provided targets.

        Parameters
        ----------
        targets:
            Iterable of :class:`TargetConfig` specifying the inputs to fold.
        work_dir:
            Directory where intermediate FASTA files and outputs will be stored.
        label:
            Name of the run, used to namespace the outputs.

        Returns
        -------
        Path
            Directory containing the resulting PDB predictions.
        """

        if not self.settings.enabled:
            raise RuntimeError("Attempted to run Minifold when the configuration is disabled.")

        run_dir = work_dir / label
        fasta_path = run_dir / f"{label}.fasta"
        predictions_dir = run_dir / "predictions"

        _write_fasta(list(targets), fasta_path)

        from predict import predict as predict_command  # Imported lazily to avoid click side-effects.

        args = [str(fasta_path), "--out_dir", str(predictions_dir), "--cache", str(self.cache_dir)]

        if self.settings.checkpoint is not None:
            args.extend(["--checkpoint", str(self.settings.checkpoint)])

        args.extend(["--token_per_batch", str(self.settings.token_per_batch)])

        if self.settings.compile:
            args.append("--compile")

        args.extend(["--model_size", self.settings.model_size])

        if self.settings.kernels:
            args.append("--kernels")

        args.extend(["--num_recycling", str(self.settings.num_recycling)])

        LOGGER.info("Running Minifold inference for run '%s'", label)
        try:
            predict_command.main(args=args, standalone_mode=False)
        except SystemExit as exc:  # pragma: no cover - Click raises SystemExit to signal completion.
            if exc.code != 0:
                raise RuntimeError(f"Minifold inference failed for run '{label}' with code {exc.code}.")

        final_dir = predictions_dir / f"minifold_results_{fasta_path.stem}"
        LOGGER.info("Minifold predictions for '%s' stored in %s", label, final_dir)
        return final_dir
