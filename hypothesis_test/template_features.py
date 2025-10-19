"""Helpers for managing template feature caches."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from .config import TargetConfig


@dataclass
class TemplateFeatureMetadata:
    """Lightweight metadata entry describing retrieved templates."""

    target_id: str
    templates: List[str] = field(default_factory=list)
    notes: str = ""
    ready: bool = False


class TemplateFeatureStore:
    """Persists template metadata for auditability and reproducibility."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def metadata_path(self, target_id: str) -> Path:
        return self.root / f"{target_id}.json"

    def prepare_entries(self, targets: Iterable[TargetConfig]) -> List[Path]:
        """Create placeholder metadata files for the provided targets."""

        paths: List[Path] = []
        for target in targets:
            path = self.metadata_path(target.identifier)
            if not path.exists():
                metadata = TemplateFeatureMetadata(target_id=target.identifier)
                with path.open("w", encoding="utf-8") as handle:
                    json.dump(asdict(metadata), handle, indent=2)
            paths.append(path)
        return paths

    def update(self, target_id: str, templates: List[str], notes: str = "", ready: bool = True) -> None:
        """Persist retrieval results for a target."""

        payload = TemplateFeatureMetadata(target_id=target_id, templates=templates, notes=notes, ready=ready)
        with self.metadata_path(target_id).open("w", encoding="utf-8") as handle:
            json.dump(asdict(payload), handle, indent=2)

    def load(self, target_id: str) -> TemplateFeatureMetadata:
        path = self.metadata_path(target_id)
        with path.open("r", encoding="utf-8") as handle:
            data: Dict[str, object] = json.load(handle)
        return TemplateFeatureMetadata(
            target_id=str(data["target_id"]),
            templates=list(data.get("templates", [])),
            notes=str(data.get("notes", "")),
            ready=bool(data.get("ready", False)),
        )
