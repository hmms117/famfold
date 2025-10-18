from pathlib import Path

from scripts.sample_mmseqs_clusters import load_cluster_memberships


def test_load_cluster_memberships_handles_additional_columns(tmp_path: Path) -> None:
    tsv = tmp_path / "clusters.tsv"
    tsv.write_text(
        "\n".join(
            [
                "# Comment line should be ignored",
                "cluster1\trepA\t0.9\t100\tfoo",
                "cluster1\tmemberB",
                "cluster2\tmemberC\t0.8",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    clusters = load_cluster_memberships(tsv)

    assert clusters == {
        "cluster1": ["repA", "memberB"],
        "cluster2": ["memberC"],
    }
