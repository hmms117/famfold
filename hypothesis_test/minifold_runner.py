"""Helpers for invoking the Minifold inference CLI from the benchmark pipeline."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Sequence

from .config import MinifoldRunSettings, TargetConfig

LOGGER = logging.getLogger(__name__)


@contextmanager
def _temporary_environment(updates: Dict[str, str]):
    """Temporarily override environment variables."""

    if not updates:
        yield
        return

    original = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def write_fasta(targets: Sequence[TargetConfig], destination: Path) -> None:
    """Write a FASTA file containing the provided targets."""

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
        """Run Minifold on the provided targets."""

        if not self.settings.enabled:
            raise RuntimeError("Attempted to run Minifold when the configuration is disabled.")

        target_list = list(targets)
        if not target_list:
            raise ValueError("No targets provided for Minifold run.")

        run_dir = work_dir / label
        fasta_path = run_dir / f"{label}.fasta"
        predictions_dir = run_dir / "predictions"

        write_fasta(target_list, fasta_path)

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

        replacements = {
            "fasta": str(fasta_path),
            "fasta_path": str(fasta_path),
            "output": str(predictions_dir),
            "output_dir": str(predictions_dir),
            "cache": str(self.cache_dir),
            "cache_dir": str(self.cache_dir),
            "label": label,
        }

        if self.settings.extra_args:
            try:
                formatted = [arg.format(**replacements) for arg in self.settings.extra_args]
            except KeyError as exc:  # pragma: no cover - configuration error path.
                missing = exc.args[0]
                raise KeyError(
                    f"Unknown placeholder '{{{missing}}}' in extra_args for Minifold run '{label}'."
                ) from exc
            args.extend(formatted)

        LOGGER.info("Running Minifold inference for run '%s'", label)
        try:
            with _temporary_environment(self.settings.environment):
                predict_command.main(args=args, standalone_mode=False)
        except SystemExit as exc:  # pragma: no cover - Click raises SystemExit to signal completion.
            if exc.code != 0:
                raise RuntimeError(f"Minifold inference failed for run '{label}' with code {exc.code}.")

        final_dir = predictions_dir / f"minifold_results_{fasta_path.stem}"
        LOGGER.info("Minifold predictions for '%s' stored in %s", label, final_dir)
        return final_dir


# Backwards compatibility for legacy imports.
_write_fasta = write_fasta
