from experiments.tier1.trunk import list_trunk_specs, resolve_trunk_spec
from minifold.utils.saesm import SAESM_FAST_CHECKPOINT


def test_saesm2_fast_spec_registered() -> None:
    spec = resolve_trunk_spec("saesm2_fast")
    assert spec.checkpoint == SAESM_FAST_CHECKPOINT
    assert spec.normalization.mean == 0.0
    assert spec.normalization.std == 1.0


def test_alias_resolution() -> None:
    alias_spec = resolve_trunk_spec("saesm2-fast")
    base_spec = resolve_trunk_spec("saesm2_fast")
    assert alias_spec is base_spec


def test_list_trunk_specs_unique() -> None:
    names = [spec.name for spec in list_trunk_specs()]
    assert len(names) == len(set(names))
    assert "saesm2_fast" in names
    assert "ism_fast" in names
