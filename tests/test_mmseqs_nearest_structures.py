from pathlib import Path

import pytest


@pytest.fixture
def clusters() -> dict[str, list[str]]:
    return {
        "cluster1": ["repA", "memberB", "memberC"],
        "cluster2": ["lonely"],
    }


@pytest.fixture
def structure_dir(tmp_path: Path) -> Path:
    (tmp_path / "repA.cif").write_text("REP_A", encoding="utf-8")
    (tmp_path / "memberB.pdb").write_text("MEMBER_B", encoding="utf-8")
    (tmp_path / "memberC.cif.gz").write_text("MEMBER_C", encoding="utf-8")
    return tmp_path


def test_collect_nearest_structures_prefers_other_members(clusters: dict[str, list[str]], structure_dir: Path) -> None:
    from minifold.familyfold.mmseqs import collect_nearest_structures

    nearest = collect_nearest_structures(
        clusters,
        structure_dir,
        max_templates=2,
        include_self=False,
    )

    assert list(nearest.keys()) == ["repA", "memberB", "memberC", "lonely"]
    assert [path.name for path in nearest["repA"]] == ["memberB.pdb", "memberC.cif.gz"]
    assert [path.name for path in nearest["memberB"]] == ["repA.cif", "memberC.cif.gz"]
    assert [path.name for path in nearest["memberC"]] == ["repA.cif", "memberB.pdb"]
    assert nearest["lonely"] == []


def test_collect_nearest_structures_includes_self_when_requested(clusters: dict[str, list[str]], structure_dir: Path) -> None:
    from minifold.familyfold.mmseqs import collect_nearest_structures

    nearest = collect_nearest_structures(
        clusters,
        structure_dir,
        max_templates=1,
        include_self=True,
    )

    assert [path.name for path in nearest["repA"]] == ["repA.cif"]
    assert [path.name for path in nearest["memberB"]] == ["memberB.pdb"]
    assert [path.name for path in nearest["memberC"]] == ["memberC.cif.gz"]
    assert nearest["lonely"] == []
