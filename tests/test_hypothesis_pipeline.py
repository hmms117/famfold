from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List

import pytest

from experiments.hypothesis_test.config import load_config
from experiments.hypothesis_test.pipeline import BenchmarkPipeline


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
    config_path = Path(__file__).resolve().parents[1] / "experiments" / "hypothesis_test" / "example_config.json"
    config = load_config(config_path)

    pipeline = BenchmarkPipeline(config, workspace=tmp_path)

    outputs = pipeline.pilot(include_templates=True)

    # The run should have invoked the Minifold CLI twice (base + templates).
    assert len(fake_predict.calls) == 2

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

    # Check that the CLI arguments propagated through to the Click command.
    for args in fake_predict.calls:
        assert "--token_per_batch" in args
        assert args[args.index("--token_per_batch") + 1] == "512"
        assert "--model_size" in args
        assert args[args.index("--model_size") + 1] == "12L"

    # Ensure manifests were exported for auditing.
    manifest_path = tmp_path / "manifests" / "pilot.json"
    assert manifest_path.exists()


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
