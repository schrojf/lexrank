"""Degree centrality, the power method and both LexRank variants."""

from __future__ import annotations

import numpy as np
import pytest

from lexrank import (
    ConvergenceError,
    adjacency_matrix,
    degree_centrality,
    lexrank_scores,
    normalize_max,
    normalize_minmax,
    power_method,
    stochastic_matrix,
)


@pytest.fixture
def chain() -> np.ndarray:
    """Path graph 0-1-2 with self links; node 1 is the central one."""
    return np.array(
        [
            [1.0, 0.5, 0.0],
            [0.5, 1.0, 0.5],
            [0.0, 0.5, 1.0],
        ]
    )


def test_adjacency_uses_a_strict_comparison() -> None:
    """Algorithm 3 line 8 is ``> t``, not ``>= t``."""
    matrix = np.array([[1.0, 0.2], [0.2, 1.0]])
    assert adjacency_matrix(matrix, 0.2).tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert adjacency_matrix(matrix, 0.19).tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_degree_counts_self_links(chain: np.ndarray) -> None:
    assert degree_centrality(chain, 0.1).tolist() == [2.0, 3.0, 2.0]


def test_degree_at_a_threshold_above_every_edge(chain: np.ndarray) -> None:
    """Only the self links survive, so every node has degree 1."""
    assert degree_centrality(chain, 0.9).tolist() == [1.0, 1.0, 1.0]


def test_stochastic_matrix_rows_sum_to_one(chain: np.ndarray) -> None:
    for continuous in (False, True):
        matrix = stochastic_matrix(chain, threshold=0.1, continuous=continuous)
        assert np.allclose(matrix.sum(axis=1), 1.0)


def test_stochastic_matrix_binarises_when_not_continuous(chain: np.ndarray) -> None:
    matrix = stochastic_matrix(chain, threshold=0.1)
    assert matrix[1].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_continuous_keeps_edge_weights(chain: np.ndarray) -> None:
    matrix = stochastic_matrix(chain, threshold=0.0, continuous=True)
    assert matrix[1].tolist() == pytest.approx([0.25, 0.5, 0.25])


def test_power_method_returns_a_distribution(chain: np.ndarray) -> None:
    scores = power_method(stochastic_matrix(chain))
    assert scores.sum() == pytest.approx(1.0)
    assert (scores > 0).all()


def test_power_method_finds_a_fixed_point(chain: np.ndarray) -> None:
    """The result must satisfy Equation 9."""
    transition = stochastic_matrix(chain)
    scores = power_method(transition, tolerance=1e-14)
    damping, n = 0.85, len(scores)
    recomputed = (1 - damping) / n + damping * (transition.T @ scores)
    assert np.allclose(recomputed, scores, atol=1e-10)


def test_uniform_damping_zero_gives_a_uniform_distribution(chain: np.ndarray) -> None:
    """With damping 0 the walker always teleports."""
    scores = power_method(stochastic_matrix(chain), damping=0.0)
    assert np.allclose(scores, 1 / 3)


def test_power_method_rejects_out_of_range_damping(chain: np.ndarray) -> None:
    with pytest.raises(ValueError):
        power_method(stochastic_matrix(chain), damping=1.5)


def test_power_method_can_raise_on_non_convergence(chain: np.ndarray) -> None:
    with pytest.raises(ConvergenceError):
        power_method(
            stochastic_matrix(chain), tolerance=1e-30, max_iterations=3, strict=True
        )


def test_power_method_returns_last_iterate_by_default(chain: np.ndarray) -> None:
    scores = power_method(
        stochastic_matrix(chain), tolerance=1e-30, max_iterations=3
    )
    assert scores.sum() == pytest.approx(1.0)


def test_power_method_on_empty_graph() -> None:
    assert power_method(np.zeros((0, 0))).shape == (0,)


def test_lexrank_ranks_the_central_node_highest(chain: np.ndarray) -> None:
    for continuous in (False, True):
        scores = lexrank_scores(chain, threshold=0.1, continuous=continuous)
        assert int(np.argmax(scores)) == 1


def test_lexrank_is_symmetric_under_relabelling(chain: np.ndarray) -> None:
    scores = lexrank_scores(chain, threshold=0.1)
    assert scores[0] == pytest.approx(scores[2])


def test_disconnected_components_still_converge() -> None:
    """Teleporting is what makes a reducible graph safe (Section 3.2).

    The graph is a star (node 0 linked to 1 and 2) plus an isolated node 3.
    Without teleporting a walker starting in the star could never reach node 3.
    """
    matrix = np.array(
        [
            [1.0, 0.9, 0.9, 0.0],
            [0.9, 1.0, 0.0, 0.0],
            [0.9, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    scores = lexrank_scores(matrix, threshold=0.1)
    assert scores.sum() == pytest.approx(1.0)
    assert (scores > 0).all()
    # The hub beats its own leaves and the isolated node.
    assert scores[0] > scores[1] == pytest.approx(scores[2])
    assert scores[0] > scores[3]


def test_isolated_node_keeps_exactly_its_teleport_share() -> None:
    """A node whose only edge is its self link settles at 1/N.

    Its balance equation is ``p = (1-d)/N + d*p``, whose solution is ``1/N``
    for any damping. This is also why Table 2's threshold-0.3 column is all
    ones: at that cutoff the graph is nothing but self loops and disjoint
    pairs, every such transition matrix is doubly stochastic, and the
    stationary distribution is uniform.
    """
    matrix = np.eye(4)
    matrix[0, 1] = matrix[1, 0] = 0.9
    scores = lexrank_scores(matrix, threshold=0.1)
    assert np.allclose(scores, 0.25)


def test_continuous_lexrank_uses_information_the_threshold_discards() -> None:
    """Two nodes with equal degree but different edge strengths."""
    matrix = np.array(
        [
            [1.0, 0.9, 0.15],
            [0.9, 1.0, 0.15],
            [0.15, 0.15, 1.0],
        ]
    )
    thresholded = lexrank_scores(matrix, threshold=0.1)
    continuous = lexrank_scores(matrix, threshold=0.0, continuous=True)
    assert thresholded[0] == pytest.approx(thresholded[1])
    assert continuous[0] == pytest.approx(continuous[1])
    # Weighting by strength pulls mass towards the strongly linked pair.
    assert continuous[0] - continuous[2] > thresholded[0] - thresholded[2]


def test_normalize_max() -> None:
    assert normalize_max(np.array([1.0, 2.0, 4.0])).tolist() == [0.25, 0.5, 1.0]
    assert normalize_max(np.zeros(3)).tolist() == [0.0, 0.0, 0.0]


def test_normalize_minmax() -> None:
    assert normalize_minmax(np.array([1.0, 2.0, 5.0])).tolist() == [0.0, 0.25, 1.0]
    assert normalize_minmax(np.array([3.0, 3.0])).tolist() == [1.0, 1.0]
    assert normalize_minmax(np.array([])).tolist() == []
