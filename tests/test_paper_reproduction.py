"""Check the implementation against the numbers printed in the paper.

Everything here is driven by the transcribed Figure 1 / Table 1 / Table 2 data
in ``lexrank.datasets.load_paper_example``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from lexrank import degree_centrality, lexrank_scores, normalize_max
from lexrank.datasets import load_paper_example


@pytest.fixture(scope="module")
def paper() -> dict[str, Any]:
    return load_paper_example()


@pytest.fixture(scope="module")
def cosine(paper: dict[str, Any]) -> np.ndarray:
    return np.array(paper["figure_1_cosine_matrix"], dtype=np.float64)


def test_figure_1_is_a_valid_similarity_matrix(cosine: np.ndarray) -> None:
    assert cosine.shape == (11, 11)
    assert np.allclose(cosine, cosine.T)
    assert np.allclose(np.diag(cosine), 1.0)


@pytest.mark.parametrize("threshold", [0.1, 0.3])
def test_table_1_degree_reproduces_exactly(
    paper: dict[str, Any], cosine: np.ndarray, threshold: float
) -> None:
    """Degree centrality with strict > and self links counted (Section 3.1)."""
    expected = paper["table_1_degree"][str(threshold)]
    assert degree_centrality(cosine, threshold).astype(int).tolist() == expected


def test_table_1_at_threshold_0_2_differs_only_where_figure_1_rounds(
    paper: dict[str, Any], cosine: np.ndarray
) -> None:
    """Figure 1 prints two decimals, so t=0.2 cannot be reproduced from it.

    Three off-diagonal cells print as exactly 0.20 and their true values
    straddle the threshold. Bumping the two that Table 1 implies are above it
    recovers the published column, which confirms rounding rather than a
    different comparison operator.
    """
    published = paper["table_1_degree"]["0.2"]
    assert degree_centrality(cosine, 0.2).astype(int).tolist() != published

    adjusted = cosine.copy()
    for i, j in ((5, 9), (7, 9)):  # (d3s2, d5s2) and (d4s1, d5s2)
        adjusted[i, j] = adjusted[j, i] = 0.204
    assert degree_centrality(adjusted, 0.2).astype(int).tolist() == published


def test_table_2_at_threshold_0_1(paper: dict[str, Any], cosine: np.ndarray) -> None:
    """LexRank reproduces Table 2 to within Figure 1's own rounding."""
    scores = normalize_max(lexrank_scores(cosine, threshold=0.1, damping=0.85))
    expected = np.array(paper["table_2_lexrank"]["0.1"])
    assert np.abs(scores - expected).max() < 1e-3


def test_table_2_at_threshold_0_3_is_uniform(paper: dict[str, Any], cosine: np.ndarray) -> None:
    """At t=0.3 the graph is self loops plus disjoint pairs.

    Every such transition matrix is doubly stochastic, so the stationary
    distribution is uniform and Table 2's column is all ones.
    """
    scores = normalize_max(lexrank_scores(cosine, threshold=0.3, damping=0.85))
    assert np.allclose(scores, paper["table_2_lexrank"]["0.3"], atol=1e-9)


def test_d4s1_is_the_most_central_sentence(cosine: np.ndarray) -> None:
    """The paper's headline claim about this cluster (Table 1 and Table 2)."""
    for threshold in (0.1, 0.2):
        assert int(np.argmax(degree_centrality(cosine, threshold))) == 7
        assert int(np.argmax(lexrank_scores(cosine, threshold=threshold))) == 7


def test_damping_convention_matches_table_2(cosine: np.ndarray, paper: dict[str, Any]) -> None:
    """0.85 is the edge-following probability, not the paper's `d`.

    Equation 8 defines ``d`` as the teleport probability, but Table 2's
    "damping factor to 0.85" only reproduces if 0.85 is the probability of
    *following* an edge (the paper's ``d`` being 0.15).
    """
    expected = np.array(paper["table_2_lexrank"]["0.1"])

    correct = normalize_max(lexrank_scores(cosine, threshold=0.1, damping=0.85))
    literal = normalize_max(lexrank_scores(cosine, threshold=0.1, damping=0.15))

    assert np.abs(correct - expected).max() < 1e-3
    assert np.abs(literal - expected).max() > 0.5
