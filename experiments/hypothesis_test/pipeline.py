"""Benchmark pipeline orchestration utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import BenchmarkConfig, Split, TargetConfig
from .minifold_runner import MinifoldInferenceRunner
from .template_features import TemplateFeatureStore

LOGGER = logging.getLogger(__name__)


class BenchmarkPipeline:
    """Coordinate benchmark runs for the homolog template hypothesis test."""

    def __init__(self, config: BenchmarkConfig, workspace: Optional[Path] = None) -> None:
        self.config = config
        self.workspace = Path(workspace or config.output_dir).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

        self._base_runner = MinifoldInferenceRunner(config.minifold_base, config.cache_dir)
        self._template_runner = None
        if config.minifold_templates.enabled:
            self._template_runner = MinifoldInferenceRunner(config.minifold_templates, config.cache_dir)
        self._template_store = TemplateFeatureStore(self.workspace / "template_features")

    def _resolve_targets(self, splits: Optional[Iterable[Split]] = None, pilot: bool = False) -> List[TargetConfig]:
        if pilot:
            return list(self.config.pilot_targets)
        if splits:
            return list(self.config.get_targets(*splits))
        return list(self.config.targets)

    def run_minifold_base(self, targets: Iterable[TargetConfig], label: str) -> Path:
        target_list = list(targets)
        LOGGER.info("Starting base Minifold run '%s' for %d targets", label, len(target_list))
        if not target_list:
            raise ValueError("No targets provided for Minifold run.")
        return self._base_runner.run(target_list, self.workspace, label)

    def prepare_template_cache(self, targets: Iterable[TargetConfig]) -> List[Path]:
        target_list = list(targets)
        LOGGER.info("Preparing template metadata entries for %d targets", len(target_list))
        return self._template_store.prepare_entries(target_list)

    def run_minifold_with_templates(self, targets: Iterable[TargetConfig], label: str) -> Path:
        if self._template_runner is None:
            raise RuntimeError("Template-augmented Minifold run requested but disabled in configuration.")

        target_list = list(targets)
        if not target_list:
            raise ValueError("No targets provided for template Minifold run.")

        missing_metadata = [
            target.identifier
            for target in target_list
            if not self._template_store.metadata_path(target.identifier).exists()
        ]
        if missing_metadata:
            raise FileNotFoundError(
                "Template metadata missing for targets: " + ", ".join(missing_metadata)
            )

        LOGGER.info(
            "Starting template-augmented Minifold run '%s' for %d targets", label, len(target_list)
        )
        return self._template_runner.run(target_list, self.workspace, label)

    def export_manifest(self, path: Path, targets: Iterable[TargetConfig]) -> None:
        manifest = []
        for target in targets:
            entry = {
                "id": target.identifier,
                "sequence_length": len(target.sequence),
                "split": target.split.value,
                "metadata": target.metadata,
            }
            if target.sequence_path is not None:
                entry["sequence_path"] = str(target.sequence_path)
            manifest.append(entry)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        LOGGER.info("Wrote manifest for %d targets to %s", len(manifest), path)

    def pilot(self, include_templates: bool = False) -> Dict[str, Path]:
        targets = self._resolve_targets(pilot=True)
        manifest_path = self.workspace / "manifests" / "pilot.json"
        self.export_manifest(manifest_path, targets)

        outputs: Dict[str, Path] = {}
        outputs["minifold_base"] = self.run_minifold_base(targets, label="pilot_base")

        if include_templates:
            metadata_paths = self.prepare_template_cache(targets)
            LOGGER.info("Template metadata prepared: %s", ", ".join(map(str, metadata_paths)))
            outputs["template_metadata_dir"] = self._template_store.root

            try:
                outputs["minifold_templates"] = self.run_minifold_with_templates(
                    targets, label="pilot_templates"
                )
            except (RuntimeError, FileNotFoundError) as exc:
                LOGGER.warning("Template Minifold run skipped: %s", exc)

        return outputs

    def full(self, splits: Optional[Iterable[Split]] = None, include_templates: bool = False) -> Dict[str, Path]:
        targets = self._resolve_targets(splits=splits, pilot=False)
        manifest_name = "_".join(split.value for split in splits) if splits else "all"
        manifest_path = self.workspace / "manifests" / f"full_{manifest_name}.json"
        self.export_manifest(manifest_path, targets)

        outputs: Dict[str, Path] = {}
        outputs["minifold_base"] = self.run_minifold_base(targets, label=f"full_base_{manifest_name}")

        if include_templates:
            self.prepare_template_cache(targets)
            try:
                outputs["minifold_templates"] = self.run_minifold_with_templates(
                    targets, label=f"full_templates_{manifest_name}"
                )
            except (RuntimeError, FileNotFoundError) as exc:
                LOGGER.warning("Template Minifold run skipped: %s", exc)

        return outputs
