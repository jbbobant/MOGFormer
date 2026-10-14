"""Portable forms of the cohort, the graph and the gene embeddings.

The experiments read a compact bundle instead of the half-gigabyte of raw CSVs
and the 800 MB STRING pair. These tests pin the property that makes reading a
reduced form safe: the reduced form must reconstruct exactly what
the full one produced, so no second code path exists where a run could quietly
diverge from the one the repository describes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mogformer.data.gene_embeddings import (
    align_gene_embeddings,
    load_gene_embedding_bundle,
    load_gene_embedding_table,
    save_gene_embedding_table,
)
from mogformer.data.omics import OmicsData, load_omics_bundle, save_omics_bundle
from mogformer.graph.string_graph import (
    StringGraphCache,
    load_string_edges,
    save_string_edges,
)


def make_cohort() -> OmicsData:
    """Return a small cohort with distinguishable values in every field."""
    rng = np.random.default_rng(7)
    label_map = {"BRCA_LumA": 0, "BRCA_LumB": 1}
    return OmicsData(
        X=rng.normal(size=(6, 3 * 4)).astype(np.float32),
        y=np.array([0, 1, 0, 1, 0, 1]),
        patient_ids=[f"TCGA-XX-{i:04d}" for i in range(6)],
        gene_names=["ESR1", "ERBB2", "MKI67", "TP53"],
        label_map=label_map,
        inverse_label_map={code: name for name, code in label_map.items()},
    )


def test_cohort_bundle_round_trips(tmp_path) -> None:
    """A reloaded cohort is the same cohort, field for field."""
    original = make_cohort()
    path = save_omics_bundle(original, tmp_path / "cohort.npz")
    restored = load_omics_bundle(path)

    np.testing.assert_allclose(restored.X, original.X)
    np.testing.assert_array_equal(restored.y, original.y)
    assert restored.patient_ids == original.patient_ids
    assert restored.gene_names == original.gene_names
    assert restored.label_map == original.label_map
    assert restored.inverse_label_map == original.inverse_label_map
    assert restored.class_counts() == original.class_counts()


def test_cohort_bundle_preserves_the_label_encoding(tmp_path) -> None:
    """Codes survive the trip, so a bundled run is comparable to a local one.

    The label map is what ties an integer in ``y`` to a subtype name. If the
    round trip re-derived it by sorting, a cohort whose classes were not
    alphabetical would come back with silently permuted labels.
    """
    data = make_cohort()
    data.label_map = {"BRCA_LumB": 0, "BRCA_LumA": 1}
    data.inverse_label_map = {0: "BRCA_LumB", 1: "BRCA_LumA"}

    restored = load_omics_bundle(save_omics_bundle(data, tmp_path / "c.npz"))

    assert restored.label_map == {"BRCA_LumB": 0, "BRCA_LumA": 1}
    assert restored.inverse_label_map[0] == "BRCA_LumB"


def test_bundled_cohort_is_much_smaller_than_its_matrices(tmp_path) -> None:
    """The archive is worth making: it compresses the matrix it carries."""
    rng = np.random.default_rng(1)
    data = make_cohort()
    # Copy-number blocks are integer-valued, which is where the saving is.
    data.X = np.repeat(rng.integers(-2, 3, size=(6, 12)), 20, axis=0).astype(np.float32)
    data.patient_ids = [f"P{i}" for i in range(data.X.shape[0])]
    data.y = np.tile([0, 1], data.X.shape[0] // 2)

    path = save_omics_bundle(data, tmp_path / "cohort.npz")
    assert path.stat().st_size < data.X.nbytes


def test_string_edges_round_trip(tmp_path) -> None:
    """A saved edge list rebuilds the identical cache."""
    edges = {("ESR1", "MKI67"), ("ERBB2", "ESR1"), ("MKI67", "TP53")}
    path = save_string_edges(edges, tmp_path / "edges.tsv")
    assert load_string_edges(path) == edges


def test_edge_list_cache_matches_a_parsed_cache(tmp_path) -> None:
    """The reduced graph induces exactly the adjacency the full parse did.

    This is the claim that lets runs skip parsing STRING: the edge list
    is not an approximation of the interactome, it is the whole of it once
    restricted to a fixed gene universe.
    """
    genes = ["A", "B", "C", "D"]
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text(
        "#string_protein_id\talias\tsource\n"
        + "".join(f"9606.P{i}\t{gene}\tBioMart\n" for i, gene in enumerate(genes)),
        encoding="utf-8",
    )
    links = tmp_path / "links.txt"
    links.write_text(
        "protein1 protein2 combined_score\n"
        "9606.P0 9606.P1 900\n"
        "9606.P1 9606.P2 700\n"
        "9606.P2 9606.P3 100\n",
        encoding="utf-8",
    )

    parsed = StringGraphCache(str(links), str(aliases), genes, 400)
    reduced = StringGraphCache.from_edges(
        load_string_edges(save_string_edges(parsed.edges, tmp_path / "e.tsv")), 400
    )

    assert reduced.edges == parsed.edges
    assert reduced.n_universe_edges == parsed.n_universe_edges
    np.testing.assert_array_equal(
        reduced.induced_adjacency(genes), parsed.induced_adjacency(genes)
    )


def test_from_edges_drops_self_loops_and_orients_pairs() -> None:
    """Edges arriving in any orientation land in the canonical one."""
    cache = StringGraphCache.from_edges([("B", "A"), ("A", "B"), ("C", "C")])
    assert cache.edges == {("A", "B")}


def test_gene_embedding_table_needs_aligned_symbols(tmp_path) -> None:
    """A symbol list of the wrong length is a hard error, not a silent shift."""
    matrix = tmp_path / "emb.csv"
    pd.DataFrame(np.zeros((3, 4))).to_csv(matrix, header=False, index=False)
    symbols = tmp_path / "symbols.txt"
    symbols.write_text("ESR1\nERBB2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="row-aligned"):
        load_gene_embedding_table(matrix, symbols)


def test_gene_embeddings_follow_the_fold_gene_order() -> None:
    """Row ``i`` of the aligned matrix belongs to gene ``i``, always.

    Every shape contract downstream keys off gene position, so a gene the
    table does not cover has to keep its row rather than be dropped.
    """
    table = pd.DataFrame(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
        index=pd.Index(["MKI67", "ESR1", "TP53"], name="gene"),
    )
    matrix, covered = align_gene_embeddings(table, ["ESR1", "UNKNOWN", "MKI67"])

    assert matrix.shape == (3, 2)
    np.testing.assert_allclose(matrix[0], [2.0, 2.0])
    np.testing.assert_allclose(matrix[1], [0.0, 0.0])
    np.testing.assert_allclose(matrix[2], [1.0, 1.0])
    assert covered == ["ESR1", "MKI67"]


def test_gene_embedding_bundle_round_trips(tmp_path) -> None:
    """The compressed embedding table reloads unchanged."""
    table = pd.DataFrame(
        np.arange(6, dtype=np.float32).reshape(3, 2),
        index=pd.Index(["A", "B", "C"], name="gene"),
    )
    restored = load_gene_embedding_bundle(
        save_gene_embedding_table(table, tmp_path / "emb.npz")
    )
    pd.testing.assert_frame_equal(restored, table, check_names=False)
