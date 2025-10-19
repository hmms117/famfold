from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List

import duckdb
import pytest
import torch

from pretests.hypothesis_test.config import load_config
from pretests.hypothesis_test.pipeline import BenchmarkPipeline
from minifold.utils.saesm import (
    SAESM_DEFAULT_CHECKPOINT,
    SAESM_FAST_CHECKPOINT,
    SaESMAlphabet,
    SaESMTrunk,
    SaESMWrapper,
    resolve_saesm_checkpoint,
)


class _FakePredictMain:
    """Callable stub that mimics :func:`predict.predict.main`."""

    def __init__(self) -> None:
        self.calls: List[List[str]] = []

    def __call__(self, args: List[str], standalone_mode: bool = False) -> None:  # noqa: D401
        # Record the CLI invocation for inspection by the test.
        self.calls.append(list(args))

        fasta = Path(args[0])
        out_dir = Path(args[args.index("--out_dir") + 1])

        # Emulate the behaviour of the real CLI: create the output directory the
        # runner expects and leave behind a placeholder file.
        predictions_dir = out_dir / f"minifold_results_{fasta.stem}"
        predictions_dir.mkdir(parents=True, exist_ok=True)
        (predictions_dir / "dummy.pdb").write_text("PDB", encoding="utf-8")

    def reset(self) -> None:
        self.calls.clear()


@pytest.fixture()
def fake_predict(monkeypatch: pytest.MonkeyPatch) -> _FakePredictMain:
    stub = _FakePredictMain()
    monkeypatch.setattr("predict.predict.main", stub)
    return stub


def test_pilot_pipeline_runs_minifold_and_templates(tmp_path: Path, fake_predict: _FakePredictMain) -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "pretests"
        / "hypothesis_test"
        / "example_config.json"
    )
    config = load_config(config_path)

    pipeline = BenchmarkPipeline(config, workspace=tmp_path)

    outputs = pipeline.pilot(include_templates=True, include_saesm2=True)

    # The run should have invoked the Minifold CLI three times (base + templates + SaESM2).
    assert len(fake_predict.calls) == 3

    # Confirm that FASTA input was materialised for the pilot run.
    fasta_path = tmp_path / "pilot_base" / "pilot_base.fasta"
    assert fasta_path.exists()
    fasta_contents = fasta_path.read_text(encoding="utf-8")
    assert ">ubiquitin_human" in fasta_contents
    assert ">rpl41e_mj" in fasta_contents

    # Validate that template metadata placeholders were created.
    template_dir = tmp_path / "template_features"
    expected_templates = {"ubiquitin_human", "rpl41e_mj"}
    assert expected_templates == {path.stem for path in template_dir.glob("*.json")}

    # The returned directories should match the fake predictions we staged.
    base_output = outputs["minifold_base"]
    template_output = outputs["minifold_templates"]
    assert base_output.exists()
    assert template_output.exists()

    saesm2_output = outputs["minifold_saesm2"]
    assert saesm2_output.exists()

    # Check that the CLI arguments propagated through to the Click command.
    for args in fake_predict.calls:
        assert "--token_per_batch" in args
        assert args[args.index("--token_per_batch") + 1] == "512"
        assert "--model_size" in args
        assert args[args.index("--model_size") + 1] == "12L"

    # Ensure manifests were exported for auditing.
    manifest_path = tmp_path / "manifests" / "pilot.json"
    assert manifest_path.exists()

    metrics_manifest = tmp_path / "manifests" / "hypothesis_test.duckdb"
    assert metrics_manifest.exists()

    with duckdb.connect(str(metrics_manifest)) as connection:
        rows = connection.execute(
            """
            SELECT metric, value, baseline_value, delta
            FROM retrieval_comparisons
            WHERE namespace = 'saesm2'
            """
        ).fetchall()

    assert rows
    recorded_metrics = {metric: value for metric, value, _, _ in rows}
    assert "recall_at_6" in recorded_metrics
    assert "latency_ms" in recorded_metrics

    baseline_values = {metric: baseline for metric, _, baseline, _ in rows if baseline is not None}
    assert baseline_values["recall_at_6"] == pytest.approx(0.91, rel=1e-5)

    deltas = {metric: delta for metric, _, _, delta in rows if delta is not None}
    assert deltas["recall_at_6"] > 0


@pytest.fixture()
def patched_autocast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("torch.autocast", lambda *_, **__: nullcontext())

def test_structure_module_forward_shapes(patched_autocast, monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    from minifold.model.structure import StructureModule

    # The structure module picks a CUDA device even when unavailable; guard against
    # attempts to query GPU capabilities during the test by faking the checks.
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)

    module = StructureModule(
        c_s=32,
        c_z=16,
        c_resnet=16,
        head_dim=8,
        no_heads=4,
        no_blocks=1,
        no_resnet_blocks=1,
        no_angles=7,
        trans_scale_factor=1.0,
        epsilon=1e-6,
        inf=1e5,
    )

    batch = 2
    length = 5
    s = torch.randn(batch, length, module.c_s)
    z = torch.randn(batch, length, length, module.c_z)
    aatype = torch.zeros(batch, length, dtype=torch.long)
    mask = torch.ones(batch, length, dtype=torch.bool)

    outputs = module(s, z, aatype, mask)

    assert outputs["positions"].shape == (1, batch, length, 14, 3)
    assert outputs["frames"].shape[-2:] == (4, 4)
    assert outputs["sidechain_frames"].shape[-2:] == (4, 4)
    assert outputs["single"].shape == (batch, length, module.c_s)


class _DummySaESMTokenizer:
    pad_token_id = 0
    mask_token_id = 1
    cls_token_id = 2
    eos_token_id = 3

    def __call__(self, sequences, return_tensors="pt", padding=True, add_special_tokens=True):
        if isinstance(sequences, str):
            sequences = [sequences]

        max_len = max(len(seq) for seq in sequences)
        target_length = max_len + (2 if add_special_tokens else 0)

        input_ids = []
        attention_mask = []

        for index, sequence in enumerate(sequences):
            tokens = []
            if add_special_tokens:
                tokens.append(self.cls_token_id)
            offset = 10 + index * 16
            tokens.extend([offset + value for value in range(len(sequence))])
            if add_special_tokens:
                tokens.append(self.eos_token_id)

            pad_length = target_length - len(tokens)
            if pad_length < 0:
                raise ValueError("Token buffer underflow in dummy tokenizer")

            tokens.extend([self.pad_token_id] * pad_length)
            mask = [1] * (len(tokens) - pad_length) + [0] * pad_length

            input_ids.append(tokens)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def encode(self, sequence: str, add_special_tokens: bool = True) -> List[int]:
        output = self(
            sequence,
            return_tensors="pt",
            padding=False,
            add_special_tokens=add_special_tokens,
        )
        return output["input_ids"].squeeze(0).tolist()


class _DummySaESMModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"hidden_size": 3, "num_hidden_layers": 4, "num_attention_heads": 2},
        )()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        output_attentions=False,
    ):
        batch, length = input_ids.shape
        hidden_size = self.config.hidden_size
        base = torch.arange(batch * length * hidden_size, dtype=torch.float32).reshape(batch, length, hidden_size)
        hidden_states = tuple(base + layer for layer in range(self.config.num_hidden_layers + 1))
        last_hidden_state = hidden_states[-1]
        if output_hidden_states:
            hidden_output = hidden_states
        else:  # pragma: no cover - fallback path
            hidden_output = None
        if output_attentions:  # pragma: no cover - not exercised in unit test
            attentions = [
                torch.ones(batch, 2, length, length, dtype=torch.float32)
                for _ in range(self.config.num_hidden_layers)
            ]
        else:
            attentions = None
        return type(
            "Output",
            (),
            {
                "hidden_states": hidden_output,
                "attentions": attentions,
                "last_hidden_state": last_hidden_state,
            },
        )()


def test_saesm_trunk_embeddings_filter_special_tokens() -> None:
    trunk = SaESMTrunk(
        checkpoint=SAESM_DEFAULT_CHECKPOINT,
        tokenizer=_DummySaESMTokenizer(),
        model=_DummySaESMModel(),
        device="cpu",
    )

    embeddings = trunk.embed_sequences(["AC", "W"])

    assert resolve_saesm_checkpoint("SaESM2-650M") == SAESM_DEFAULT_CHECKPOINT
    assert resolve_saesm_checkpoint("saesm2_35m") == SAESM_FAST_CHECKPOINT

    assert len(embeddings.per_residue) == 2
    assert embeddings.per_residue[0].shape == (2, 3)
    assert embeddings.per_residue[1].shape == (1, 3)

    expected_first = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])
    expected_second = torch.tensor([[19.0, 20.0, 21.0]])
    torch.testing.assert_close(embeddings.per_residue[0], expected_first)
    torch.testing.assert_close(embeddings.per_residue[1], expected_second)

    expected_pooled = torch.stack(
        (
            expected_first.mean(dim=0),
            expected_second.mean(dim=0),
        )
    )
    torch.testing.assert_close(embeddings.per_sequence, expected_pooled)

    # Ensure CLS/EOS/padding tokens were excluded from residue mask.
    assert embeddings.residue_mask.shape == embeddings.input_ids.shape
    assert not embeddings.residue_mask[0, 0]
    assert not embeddings.residue_mask[0, -1]
    assert embeddings.residue_mask[0, 1]


def test_saesm_wrapper_matches_dummy_hidden_states() -> None:
    tokenizer = _DummySaESMTokenizer()
    model = _DummySaESMModel()

    alphabet = SaESMAlphabet(tokenizer)
    wrapper = SaESMWrapper(
        checkpoint=SAESM_DEFAULT_CHECKPOINT,
        tokenizer=tokenizer,
        model=model,
    )

    seqs = ["AC", "W"]
    encoded = [alphabet.encode(seq) for seq in seqs]
    max_len = max(len(tokens) for tokens in encoded)
    batch = torch.full((len(encoded), max_len), alphabet.padding_idx, dtype=torch.long)
    for row, tokens in enumerate(encoded):
        batch[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)

    outputs = wrapper(batch, repr_layers=[wrapper.num_layers], need_head_weights=True)

    raw = model(
        input_ids=batch,
        attention_mask=batch.ne(alphabet.padding_idx),
        output_hidden_states=True,
        output_attentions=True,
    )
    expected_hidden = torch.nn.functional.layer_norm(
        raw.hidden_states[wrapper.num_layers],
        (model.config.hidden_size,),
        eps=1e-5,
    )

    torch.testing.assert_close(outputs["representations"][wrapper.num_layers], expected_hidden)

    attentions = outputs["attentions"]
    assert attentions.shape == (
        batch.shape[0],
        batch.shape[1],
        batch.shape[1],
        wrapper.num_layers,
        model.config.num_attention_heads,
    )
