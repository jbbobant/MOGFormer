"""Unit tests for the graph priors.

These run on hand-built graphs with known answers, so they need no cohort data
and no network access.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mogformer.graph import (
    GraphPositionalEncoding,
    GRNSignedCache,
    StringGraphCache,
    compute_shortest_path_matrix,
    parse_string_edges,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture()
def path_graph() -> torch.Tensor:
    """Return the adjacency of the 4-node path ``0 - 1 - 2 - 3``."""
    adjacency = torch.zeros(4, 4)
    for i in range(3):
        adjacency[i, i + 1] = 1.0
        adjacency[i + 1, i] = 1.0
    return adjacency


@pytest.fixture()
def string_files(tmp_path):
    """Write a miniature STRING links/aliases pair and return both paths."""
    aliases = tmp_path / "aliases.tsv"
    aliases.write_text(
        "#string_protein_id\talias\tsource\n"
        "9606.P1\tESR1\tBioMart\n"
        "9606.P2\tERBB2\tBioMart\n"
        "9606.P3\tMKI67\tBioMart\n"
        "9606.P4\tNOTINUNIVERSE\tBioMart\n",
        encoding="utf-8",
    )
    links = tmp_path / "links.txt"
    links.write_text(
        "protein1 protein2 combined_score\n"
        "9606.P1 9606.P2 900\n"
        "9606.P2 9606.P1 900\n"  # reciprocal row, must collapse to one edge
        "9606.P2 9606.P3 500\n"
        "9606.P1 9606.P3 150\n"  # below threshold
        "9606.P1 9606.P4 900\n"  # partner outside the universe
        "9606.P3 9606.P3 999\n",  # self-loop
        encoding="utf-8",
    )
    return str(links), str(aliases)


# --------------------------------------------------------------------------
# Shortest paths
# --------------------------------------------------------------------------
def test_shortest_paths_on_a_path_graph(path_graph: torch.Tensor) -> None:
    """Distances along a path equal the index difference."""
    distances = compute_shortest_path_matrix(path_graph, max_distance=5)
    expected = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0],
        ]
    )
    assert torch.equal(distances, expected)


def test_shortest_paths_clamp_beyond_max_distance(path_graph: torch.Tensor) -> None:
    """Distances above the cap collapse onto the ``max_distance + 1`` bucket."""
    distances = compute_shortest_path_matrix(path_graph, max_distance=1)
    assert distances[0, 2].item() == 2  # 2 hops -> clamped to max_distance + 1
    assert distances[0, 3].item() == 2
    assert distances[0, 1].item() == 1  # 1 hop -> exact
    assert distances.max().item() == 2


def test_shortest_paths_unreachable_share_the_far_bucket() -> None:
    """Disconnected components land in the same bucket as merely-distant pairs."""
    adjacency = torch.zeros(4, 4)
    adjacency[0, 1] = adjacency[1, 0] = 1.0
    adjacency[2, 3] = adjacency[3, 2] = 1.0

    distances = compute_shortest_path_matrix(adjacency, max_distance=3)

    assert distances[0, 2].item() == 4
    assert distances[0, 1].item() == 1
    assert torch.equal(torch.diagonal(distances), torch.zeros(4, dtype=torch.long))


def test_shortest_paths_ignore_edge_weights() -> None:
    """Any strictly positive entry counts as a single hop."""
    weighted = torch.zeros(3, 3)
    weighted[0, 1] = weighted[1, 0] = 7.5
    weighted[1, 2] = weighted[2, 1] = 0.2

    distances = compute_shortest_path_matrix(weighted, max_distance=5)

    assert distances[0, 1].item() == 1
    assert distances[0, 2].item() == 2


@pytest.mark.parametrize(
    ("adjacency", "max_distance"),
    [
        (torch.zeros(3, 4), 5),
        (torch.zeros(3), 5),
        (torch.zeros(3, 3), 0),
    ],
)
def test_shortest_paths_reject_bad_input(
    adjacency: torch.Tensor, max_distance: int
) -> None:
    """Non-square adjacency and a non-positive cap are rejected."""
    with pytest.raises(ValueError):
        compute_shortest_path_matrix(adjacency, max_distance=max_distance)


# --------------------------------------------------------------------------
# Positional encodings
# --------------------------------------------------------------------------
def test_random_walk_encoding_shape_and_range(path_graph: torch.Tensor) -> None:
    """Return probabilities are shaped ``(n_genes, pe_dim)`` and lie in [0, 1]."""
    encoder = GraphPositionalEncoding(pe_dim=6, method="rwpe")
    encoding = encoder(path_graph)

    assert encoding.shape == (4, 6)
    assert torch.all(encoding >= 0.0)
    assert torch.all(encoding <= 1.0)


def test_random_walk_first_step_cannot_return(path_graph: torch.Tensor) -> None:
    """A single step on a loop-free graph never returns to its origin."""
    encoder = GraphPositionalEncoding(pe_dim=2, method="rwpe")
    encoding = encoder(path_graph)
    assert torch.allclose(encoding[:, 0], torch.zeros(4), atol=1e-9)


def test_random_walk_encoding_is_permutation_equivariant(
    path_graph: torch.Tensor,
) -> None:
    """Relabelling nodes permutes the encoding rows and nothing else."""
    encoder = GraphPositionalEncoding(pe_dim=5, method="rwpe")
    order = torch.tensor([2, 0, 3, 1])

    permuted_graph = path_graph[order][:, order]
    assert torch.allclose(
        encoder(permuted_graph), encoder(path_graph)[order], atol=1e-6
    )


def test_random_walk_tolerates_isolated_nodes() -> None:
    """An isolated node yields finite values rather than a division by zero."""
    adjacency = torch.zeros(3, 3)
    adjacency[0, 1] = adjacency[1, 0] = 1.0

    encoding = GraphPositionalEncoding(pe_dim=3, method="rwpe")(adjacency)

    assert torch.all(torch.isfinite(encoding))
    assert torch.allclose(encoding[2], torch.zeros(3), atol=1e-9)


def test_positional_encoding_rejects_unknown_method() -> None:
    """An unsupported scheme fails at construction, not at first use."""
    with pytest.raises(ValueError, match="unknown positional encoding method"):
        GraphPositionalEncoding(pe_dim=4, method="magic")


# --------------------------------------------------------------------------
# STRING graph
# --------------------------------------------------------------------------
def test_parse_string_edges_filters_and_deduplicates(string_files) -> None:
    """Threshold, universe membership, self-loops and reciprocals are handled."""
    links, aliases = string_files
    edges = parse_string_edges(
        links, aliases, ["ESR1", "ERBB2", "MKI67"], confidence_threshold=400
    )
    assert edges == {("ERBB2", "ESR1"), ("ERBB2", "MKI67")}


def test_string_cache_induces_subgraph_in_selection_order(string_files) -> None:
    """Adjacency rows follow ``selected_genes``, and isolated nodes survive."""
    links, aliases = string_files
    cache = StringGraphCache(links, aliases, ["ESR1", "ERBB2", "MKI67"])

    adjacency = cache.induced_adjacency(["MKI67", "ESR1", "ERBB2"])

    assert adjacency.shape == (3, 3)
    assert np.array_equal(adjacency, adjacency.T)
    assert np.allclose(np.diagonal(adjacency), 0.0)
    assert adjacency[0, 2] == 1.0  # MKI67 - ERBB2
    assert adjacency[1, 2] == 1.0  # ESR1  - ERBB2
    assert adjacency[0, 1] == 0.0  # MKI67 - ESR1 was below threshold


def test_string_cache_keeps_isolated_selected_genes(string_files) -> None:
    """Selecting a gene with no surviving edges still yields its own row."""
    links, aliases = string_files
    cache = StringGraphCache(links, aliases, ["ESR1", "ERBB2", "MKI67"])

    adjacency = cache.induced_adjacency(["ESR1", "MKI67"])

    assert adjacency.shape == (2, 2)
    assert adjacency.sum() == 0.0


# --------------------------------------------------------------------------
# Regulatory network
# --------------------------------------------------------------------------
@pytest.fixture()
def grn_file(tmp_path):
    """Write a miniature CollecTRI edge table and return its path."""
    path = tmp_path / "collectri.tsv"
    path.write_text(
        "source\ttarget\tsign\n"
        "ESR1\tMYC\t1\n"
        "TP53\tMYC\t-1\n"
        "MYC\tMYC\t1\n"  # self-loop
        "ESR1\tOUTSIDE\t1\n",  # target outside the universe
        encoding="utf-8",
    )
    return str(path)


def test_grn_cache_drops_self_loops_and_outsiders(grn_file: str) -> None:
    """Only edges wholly inside the universe, minus self-loops, are kept."""
    cache = GRNSignedCache(grn_file, ["ESR1", "TP53", "MYC"])
    assert cache.n_universe_edges == 2
    assert set(cache.edges) == {("ESR1", "MYC", 1), ("TP53", "MYC", -1)}


def test_grn_orientation_places_regulators_on_columns(grn_file: str) -> None:
    """Under the default convention, row is target and column is regulator."""
    cache = GRNSignedCache(grn_file, ["ESR1", "TP53", "MYC"])

    matrix = cache.induced_signed_adjacency(["ESR1", "TP53", "MYC"])

    assert matrix[2, 0] == 1.0  # MYC activated by ESR1
    assert matrix[2, 1] == -1.0  # MYC repressed by TP53
    assert matrix[0, 2] == 0.0  # not symmetric


def test_grn_orientation_can_be_flipped(grn_file: str) -> None:
    """The transposed convention mirrors the default exactly."""
    cache = GRNSignedCache(grn_file, ["ESR1", "TP53", "MYC"])
    genes = ["ESR1", "TP53", "MYC"]

    default = cache.induced_signed_adjacency(genes, direction="target_attends_tf")
    flipped = cache.induced_signed_adjacency(genes, direction="tf_attends_target")

    assert np.array_equal(default, flipped.T)


def test_grn_unsigned_mode_keeps_topology_and_drops_sign(grn_file: str) -> None:
    """``use_sign=False`` isolates edge existence from edge direction of effect."""
    cache = GRNSignedCache(grn_file, ["ESR1", "TP53", "MYC"])
    genes = ["ESR1", "TP53", "MYC"]

    unsigned = cache.induced_signed_adjacency(genes, use_sign=False)

    assert unsigned[2, 0] == 1.0
    assert unsigned[2, 1] == 1.0
    assert set(np.unique(unsigned)) == {0.0, 1.0}


def test_grn_rejects_unknown_direction(grn_file: str) -> None:
    """A mistyped orientation fails loudly rather than silently transposing."""
    cache = GRNSignedCache(grn_file, ["ESR1", "TP53", "MYC"])
    with pytest.raises(ValueError, match="unknown grn direction"):
        cache.induced_signed_adjacency(["ESR1", "MYC"], direction="sideways")
