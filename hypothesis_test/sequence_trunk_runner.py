"""Utilities for running sequence trunk embedding sweeps."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

import torch

from minifold.utils.saesm import SaESMEmbeddings, SaESMTrunk, resolve_saesm_checkpoint

from .config import SequenceTrunkSettings, TargetConfig

LOGGER = logging.getLogger(__name__)

_HF_CACHE_ROOT = "/var/tmp/hf_cache"
os.environ["HF_HOME"] = _HF_CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"] = _HF_CACHE_ROOT


def _resolve_dtype(value: Optional[str]) -> str | torch.dtype:
    """Convert a configuration dtype string into a torch-compatible value."""

    if value is None:
        return "auto"
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"auto", "none"}:
            return "auto"
        if not hasattr(torch, normalised):
            raise ValueError(f"Unknown torch dtype specified for trunk run: {value!r}")
        return getattr(torch, normalised)
    if isinstance(value, torch.dtype):  # pragma: no cover - defensive branch
        return value
    raise TypeError(f"Unsupported dtype specification: {value!r}")


def _chunk_targets(targets: Sequence[TargetConfig], batch_size: int) -> Iterable[List[TargetConfig]]:
    if batch_size <= 0:
        raise ValueError("Sequence trunk batch_size must be greater than zero.")
    for index in range(0, len(targets), batch_size):
        yield list(targets[index : index + batch_size])


class SequenceTrunkRunner:
    """Batch sequences through a SaESM-compatible trunk and persist embeddings."""

    def __init__(
        self,
        name: str,
        settings: SequenceTrunkSettings,
        cache_dir: Path,
        trunk_factory: Optional[Callable[[], SaESMTrunk]] = None,
    ) -> None:
        self.name = name
        self.settings = settings
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._trunk_factory = trunk_factory

    def _build_trunk(self) -> SaESMTrunk:
        if self._trunk_factory is not None:
            return self._trunk_factory()

        checkpoint = self.settings.checkpoint or resolve_saesm_checkpoint(self.name)
        checkpoint = resolve_saesm_checkpoint(checkpoint)

        dtype = _resolve_dtype(self.settings.torch_dtype)

        LOGGER.info(
            "Loading sequence trunk '%s' from checkpoint %s", self.name, checkpoint
        )

        return SaESMTrunk(
            checkpoint=checkpoint,
            device=self.settings.device,
            torch_dtype=dtype,
            device_map=self.settings.device_map,
        )

    @staticmethod
    def _serialise_single(path: Path, embeddings: SaESMEmbeddings, index: int) -> None:
        payload = {
            "per_sequence": embeddings.per_sequence[index],
            "per_residue": embeddings.per_residue[index],
            "input_ids": embeddings.input_ids[index],
            "residue_mask": embeddings.residue_mask[index],
        }
        torch.save(payload, path)

    def run(self, targets: Iterable[TargetConfig], work_dir: Path, label: str) -> Path:
        target_list = list(targets)
        if not target_list:
            raise ValueError("No targets provided for sequence trunk embedding run.")

        run_dir = work_dir / label
        embeddings_dir = run_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        trunk = self._build_trunk()

        manifest: List[dict] = []
        batch_size = self.settings.batch_size or 8

        for batch in _chunk_targets(target_list, batch_size):
            sequences = [target.sequence for target in batch]
            LOGGER.info(
                "Embedding %d sequences with trunk '%s' (label=%s)",
                len(sequences),
                self.name,
                label,
            )
            batch_embeddings = trunk.embed_sequences(sequences).to("cpu")

            for index, target in enumerate(batch):
                destination = embeddings_dir / f"{target.identifier}.pt"
                self._serialise_single(destination, batch_embeddings, index)

                manifest.append(
                    {
                        "id": target.identifier,
                        "sequence_length": len(target.sequence),
                        "embedding_file": str(destination.relative_to(run_dir)),
                    }
                )

        manifest_path = run_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        LOGGER.info(
            "Sequence trunk '%s' embeddings stored in %s", self.name, embeddings_dir
        )

        return run_dir


__all__ = ["SequenceTrunkRunner"]
