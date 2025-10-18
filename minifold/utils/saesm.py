"""Utilities for working with the SaESM2 Hugging Face checkpoints.

The Chandar Lab "SaESM2" family exposes distilled ESM models via Hugging Face.
They are lightweight enough to serve as fast sequence trunks for retrieval or
embedding experiments, while remaining API-compatible with the existing ESM
wrappers used throughout MiniFold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

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


class SaESMAlphabet:
    """Adapter that exposes the Hugging Face tokenizer via the ESM alphabet API."""

    prepend_bos: bool = True
    append_eos: bool = True

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer

        self.padding_idx = tokenizer.pad_token_id
        self.mask_idx = tokenizer.mask_token_id
        self.cls_idx = tokenizer.cls_token_id
        self.eos_idx = tokenizer.eos_token_id

        missing = {
            name: getattr(self, name)
            for name in ("padding_idx", "mask_idx", "cls_idx", "eos_idx")
            if getattr(self, name) is None
        }
        if missing:
            raise ValueError(
                "The provided tokenizer is missing required SaESM special tokens: "
                + ", ".join(missing.keys())
            )

    def encode(self, sequence: str) -> Sequence[int]:
        return self._tokenizer.encode(sequence, add_special_tokens=True)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self._tokenizer.vocab_size


class SaESMWrapper(nn.Module):
    """MiniFold-compatible wrapper around Hugging Face SaESM checkpoints."""

    def __init__(
        self,
        checkpoint: str = SAESM_DEFAULT_CHECKPOINT,
        *,
        tokenizer=None,
        model=None,
        torch_dtype: str | torch.dtype = "auto",
        device_map: Optional[str | Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        resolved = resolve_saesm_checkpoint(checkpoint)
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

        self.checkpoint = resolved
        self.tokenizer = tokenizer
        self.model = model

        self.embed_dim = getattr(model.config, "hidden_size")
        self.attention_heads = getattr(model.config, "num_attention_heads", 0)
        self.num_layers = getattr(model.config, "num_hidden_layers", 0)

        self.pad_idx = getattr(tokenizer, "pad_token_id", None)
        self.mask_idx = getattr(tokenizer, "mask_token_id", None)
        self.cls_idx = getattr(tokenizer, "cls_token_id", None)
        self.eos_idx = getattr(tokenizer, "eos_token_id", None)

        missing = {
            name: getattr(self, name)
            for name in ("pad_idx", "mask_idx", "cls_idx", "eos_idx")
            if getattr(self, name) is None
        }
        if missing:
            raise ValueError(
                "The provided tokenizer is missing required SaESM special tokens: "
                + ", ".join(missing.keys())
            )

        # Align with MiniFold's expectation that representations are mean-centred.
        self._embedding_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False)

        self.esm = getattr(model, "esm", model)
        self.lm_head = getattr(model, "lm_head", nn.Identity())
        self.contact_head = getattr(self.esm, "contact_head", None)

    def forward(
        self,
        tokens: torch.Tensor,
        repr_layers: Optional[Iterable[int]] = None,
        need_head_weights: bool = False,
        return_contacts: bool = False,
    ) -> dict:
        if return_contacts:
            need_head_weights = True

        repr_layers = tuple(repr_layers or ())
        output_hidden_states = bool(repr_layers)

        attention_mask = tokens.ne(self.pad_idx)
        model_kwargs = {
            "input_ids": tokens,
            "attention_mask": attention_mask,
            "output_attentions": need_head_weights,
            "output_hidden_states": output_hidden_states,
            "return_dict": True,
        }
        try:
            outputs = self.model(**model_kwargs)
        except TypeError:  # pragma: no cover - legacy HF compatibility
            model_kwargs.pop("return_dict", None)
            outputs = self.model(**model_kwargs)

        result: Dict[str, object] = {
            "logits": self.lm_head(outputs.last_hidden_state),
            "representations": {},
        }

        if output_hidden_states and outputs.hidden_states is not None:
            hidden_states = list(outputs.hidden_states)
            for layer_idx in repr_layers:
                if 0 <= layer_idx < len(hidden_states):
                    result["representations"][layer_idx] = self._embedding_norm(
                        hidden_states[layer_idx]
                    )

        if need_head_weights:
            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError(
                    "Attention weights requested from SaESM but none were returned."
                )
            stacked = torch.stack(attentions, dim=1)  # (B, L, H, S, S)
            stacked = stacked.permute(0, 3, 4, 1, 2).contiguous()
            result["attentions"] = stacked

            if return_contacts:
                if self.contact_head is None:
                    raise RuntimeError(
                        "SaESM contact head unavailable; cannot return contacts."
                    )
                result["contacts"] = self.contact_head(tokens, stacked)

        return result


def load_saesm_model_and_alphabet(
    checkpoint: str,
    *,
    tokenizer=None,
    model=None,
    torch_dtype: str | torch.dtype = "auto",
    device_map: Optional[str | Dict[str, int]] = None,
):
    """Return a MiniFold-compatible SaESM model and alphabet wrapper."""

    wrapper = SaESMWrapper(
        checkpoint=checkpoint,
        tokenizer=tokenizer,
        model=model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    alphabet = SaESMAlphabet(wrapper.tokenizer)
    return wrapper, alphabet


__all__ = [
    "SAESM_DEFAULT_CHECKPOINT",
    "SAESM_FAST_CHECKPOINT",
    "SaESMAlphabet",
    "SaESMEmbeddings",
    "SaESMTrunk",
    "SaESMWrapper",
    "load_saesm_model_and_alphabet",
    "resolve_saesm_checkpoint",
]
