"""Execution helpers for third-party baseline structure predictors."""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from .config import BaselineSettings, TargetConfig
from .minifold_runner import write_fasta

LOGGER = logging.getLogger(__name__)


def _format_arguments(arguments: List[str], substitutions: Dict[str, str]) -> List[str]:
    formatted: List[str] = []
    for value in arguments:
        try:
            formatted.append(value.format(**substitutions))
        except KeyError as exc:  # pragma: no cover - configuration error path.
            missing = exc.args[0]
            raise KeyError(
                f"Unknown placeholder '{{{missing}}}' in baseline arguments: {value!r}"
            ) from exc
    return formatted


class ExternalBaselineRunner:
    """Invoke external baseline predictors following a configurable command template."""

    def __init__(self, name: str, settings: BaselineSettings) -> None:
        self.name = name
        self.settings = settings

    def run(self, targets: Iterable[TargetConfig], work_dir: Path, label: str) -> Path:
        """Execute the configured baseline predictor and return the output directory."""

        if not self.settings.enabled:
            raise RuntimeError(
                f"Baseline '{self.name}' requested but disabled in the configuration."
            )
        if not self.settings.executable:
            raise ValueError(
                f"Baseline '{self.name}' is missing an executable path in the configuration."
            )

        target_list = list(targets)
        if not target_list:
            raise ValueError(f"No targets provided for baseline run '{self.name}'.")

        run_dir = work_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)

        fasta_path = run_dir / f"{label}.fasta"
        write_fasta(target_list, fasta_path)

        output_dir = run_dir / "predictions"
        output_dir.mkdir(parents=True, exist_ok=True)

        substitutions = {
            "fasta": str(fasta_path),
            "fasta_path": str(fasta_path),
            "output": str(output_dir),
            "output_dir": str(output_dir),
            "label": label,
            "run_dir": str(run_dir),
            "workspace": str(work_dir),
        }

        args = _format_arguments(self.settings.extra_args, substitutions)

        # Append defaults if the template omitted them.
        if not any("{fasta" in token for token in self.settings.extra_args):
            args.append(str(fasta_path))
        if not any("{output" in token for token in self.settings.extra_args):
            args.append(str(output_dir))

        command = [self.settings.executable, *args]

        LOGGER.info("Running baseline '%s' via command: %s", self.name, " ".join(command))

        environment = os.environ.copy()
        environment.update({key: str(value) for key, value in self.settings.environment.items()})

        try:
            subprocess.run(command, check=True, cwd=run_dir, env=environment)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Baseline '{self.name}' failed with exit code {exc.returncode}."
            ) from exc

        final_dir = output_dir
        if self.settings.output_subdir:
            final_dir = output_dir / self.settings.output_subdir

        LOGGER.info("Baseline '%s' outputs available in %s", self.name, final_dir)
        return final_dir
