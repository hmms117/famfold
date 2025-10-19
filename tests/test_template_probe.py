import torch

from minifold.model.model import FoldingTrunk
from minifold.utils.template_probe import TemplateResidueMap, build_distogram_from_templates


def test_build_distogram_from_templates_populates_expected_bins():
    coordinates = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32)
    mask = torch.ones(2, dtype=torch.bool)
    mapping = {0: 0, 1: 1}

    template = TemplateResidueMap(
        name="dummy",
        coordinates=coordinates,
        mask=mask,
        mapping=mapping,
        identity=1.0,
    )

    dist = build_distogram_from_templates(2, [template])

    assert dist.shape == (2, 2, 64)

    boundaries = torch.linspace(2.3125, 21.6875, 63)
    expected_bin = torch.bucketize(torch.tensor(3.0), boundaries).item()

    assert torch.isclose(dist[0, 1, expected_bin], torch.tensor(1.0))
    assert torch.isclose(dist[1, 0, expected_bin], torch.tensor(1.0))


def test_folding_trunk_recycles_template_distogram(monkeypatch):
    trunk = FoldingTrunk(c_s=32, c_z=16, bins=8, disto_bins=4, num_layers=1, kernels=False)

    class RecordingRecycle(torch.nn.Module):
        def __init__(self, output_dim: int):
            super().__init__()
            self.output_dim = output_dim
            self.last_input = None

        def forward(self, value):
            self.last_input = value.clone()
            zeros = torch.zeros(*value.shape[:-1], self.output_dim, dtype=value.dtype, device=value.device)
            return zeros

    class RecordingMiniFormer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.last_input = None

        def forward(self, inputs, *_):
            self.last_input = inputs
            return inputs

    recorder_recycle = RecordingRecycle(output_dim=16)
    trunk.recycle = recorder_recycle
    recorder = RecordingMiniFormer()
    trunk.miniformer = recorder

    batch = 2
    length = 3
    s_s = torch.randn(batch, length, 32)
    s_z = torch.zeros(batch, length, length, 16)
    mask = torch.ones(batch, length, dtype=torch.bool)

    template = torch.ones(batch, length, length, trunk.disto_bins)

    trunk(s_s, s_z, mask, num_recycling=0, template_distogram=template)
    with_template = recorder_recycle.last_input
    assert with_template is not None
    assert torch.allclose(with_template, template.to(dtype=with_template.dtype))

    trunk(s_s, s_z, mask, num_recycling=0, template_distogram=None)
    without_template = recorder_recycle.last_input
    assert without_template is not None
    assert torch.allclose(without_template, torch.zeros_like(template))
