"""Utilities for working with the SaESM2 Hugging Face checkpoints.

The Chandar Lab "SaESM2" family exposes distilled ESM models via Hugging Face.
They are lightweight enough to serve as fast sequence trunks for retrieval or
embedding experiments, while remaining API-compatible with the existing ESM
wrappers used throughout MiniFold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn

SAESM_DEFAULT_CHECKPOINT = "chandar-lab/SaESM2_650M"
"""Default SaESM checkpoint used for high-fidelity embeddings."""

SAESM_FAST_CHECKPOINT = "chandar-lab/SaESM2_35M"
"""Smaller SaESM checkpoint suited for rapid retrieval and RAG experiments."""

_SAESM_ALIAS_MAP = {
    "saesm2_650m": SAESM_DEFAULT_CHECKPOINT,
    "saesm2-650m": SAESM_DEFAULT_CHECKPOINT,
    "saesm2/650m": SAESM_DEFAULT_CHECKPOINT,
    "chandar-lab/saesm2_650m": SAESM_DEFAULT_CHECKPOINT,
    SAESM_DEFAULT_CHECKPOINT.lower(): SAESM_DEFAULT_CHECKPOINT,
    "saesm2_35m": SAESM_FAST_CHECKPOINT,
    "saesm2-35m": SAESM_FAST_CHECKPOINT,
    "saesm2/35m": SAESM_FAST_CHECKPOINT,
    "chandar-lab/saesm2_35m": SAESM_FAST_CHECKPOINT,
    SAESM_FAST_CHECKPOINT.lower(): SAESM_FAST_CHECKPOINT,
}


def resolve_saesm_checkpoint(name: str) -> str:
    """Resolve a user-provided SaESM checkpoint alias to a canonical path."""

    normalised = name.strip().lower()
    normalised = normalised.replace(" ", "")
    return _SAESM_ALIAS_MAP.get(normalised, name)


@dataclass
class SaESMEmbeddings:
    """Embedding outputs produced by :class:`SaESMTrunk`.

    Attributes
    ----------
    per_residue:
        A list containing the per-residue embeddings for each sequence in the
        batch with special tokens removed. Each entry is a tensor of shape
        ``(L_i, H)``.
    per_sequence:
        A tensor of shape ``(B, H)`` with mean-pooled representations for each
        sequence.
    input_ids:
        Token IDs (including padding/special tokens) returned by the tokenizer.
    residue_mask:
        Boolean mask highlighting which token positions correspond to residues
        (i.e. special tokens and padding removed).
    """

    per_residue: List[Tensor]
    per_sequence: Tensor
    input_ids: Tensor
    residue_mask: Tensor

    def to(self, device: torch.device | str) -> "SaESMEmbeddings":
        """Return a copy of the embedding batch moved to ``device``."""

        return SaESMEmbeddings(
            per_residue=[tensor.to(device) for tensor in self.per_residue],
            per_sequence=self.per_sequence.to(device),
            input_ids=self.input_ids.to(device),
            residue_mask=self.residue_mask.to(device),
        )


class SaESMTrunk(nn.Module):
    """Minimal wrapper that exposes SaESM Hugging Face checkpoints.

    The wrapper mirrors the interface used by other MiniFold sequence trunks,
    returning per-residue and per-sequence embeddings.  It defaults to using the
    650M checkpoint but can be configured to use the smaller 35M variant when
    latency is more critical (e.g. retrieval, RAG experiments).
    """

    def __init__(
        self,
        checkpoint: str = SAESM_DEFAULT_CHECKPOINT,
        *,
        tokenizer=None,
        model=None,
        device: Optional[torch.device | str] = None,
        torch_dtype: str | torch.dtype = "auto",
        device_map: Optional[str | dict] = None,
    ) -> None:
        super().__init__()

        resolved = resolve_saesm_checkpoint(checkpoint)
        self.checkpoint = resolved

        if tokenizer is None or model is None:
            try:
                from transformers import AutoModelForMaskedLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "transformers is required to load SaESM checkpoints. "
                    "Install it via `pip install transformers`."
                ) from exc

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(resolved)
            if model is None:
                model = AutoModelForMaskedLM.from_pretrained(
                    resolved,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )

        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()

        self.embed_dim = getattr(model.config, "hidden_size")
        self.num_layers = getattr(model.config, "num_hidden_layers", 0)

        self._using_device_map = device_map is not None
        if device is None and not self._using_device_map:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device) if device is not None else None
        if self.device is not None and not self._using_device_map:
            self.model.to(self.device)

        self._special_token_ids = {
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "cls_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
        }
        self._special_token_ids.discard(None)

    @torch.no_grad()
    def embed_sequences(self, sequences: Sequence[str]) -> SaESMEmbeddings:
        """Compute SaESM embeddings for a batch of sequences."""

        if isinstance(sequences, str):
            sequences = [sequences]

        batch = self.tokenizer(
            list(sequences),
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        if self.device is not None and not self._using_device_map:
            batch = {key: value.to(self.device) for key, value in batch.items()}

        outputs = self.model(**batch, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if not hidden_states:
            raise RuntimeError("SaESM model did not return hidden states.")

        last_hidden = hidden_states[-1]
        if self.device is not None and not self._using_device_map:
            last_hidden = last_hidden.to("cpu")
            input_ids = input_ids.to("cpu")
            attention_mask = attention_mask.to("cpu")

        residue_mask = attention_mask.bool()
        for token_id in self._special_token_ids:
            residue_mask &= input_ids.ne(token_id)

        per_residue: List[Tensor] = []
        per_sequence: List[Tensor] = []

        for row_hidden, row_mask in zip(last_hidden, residue_mask):
            residue_embeddings = row_hidden[row_mask]
            per_residue.append(residue_embeddings)
            if residue_embeddings.numel() == 0:
                per_sequence.append(torch.zeros(self.embed_dim))
            else:
                per_sequence.append(residue_embeddings.mean(dim=0))

        per_sequence_tensor = torch.stack(per_sequence, dim=0)

        return SaESMEmbeddings(
            per_residue=per_residue,
            per_sequence=per_sequence_tensor,
            input_ids=input_ids,
            residue_mask=residue_mask,
        )

    @torch.no_grad()
    def embed_sequence(self, sequence: str) -> SaESMEmbeddings:
        """Convenience wrapper for embedding a single sequence."""

        return self.embed_sequences([sequence])


__all__ = [
    "SaESMEmbeddings",
    "SaESMTrunk",
    "SAESM_DEFAULT_CHECKPOINT",
    "SAESM_FAST_CHECKPOINT",
    "resolve_saesm_checkpoint",
]
