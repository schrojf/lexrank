"""Graph centrality measures from Section 3 of the paper.

Three measures are implemented:

* **Degree centrality** (3.1) -- the degree of a node in the thresholded
  similarity graph, self links included.
* **LexRank with threshold** (3.2, Algorithm 3) -- PageRank over the binarized
  similarity graph.
* **Continuous LexRank** (3.3, Equation 10) -- PageRank over the weighted
  graph, using the cosine values directly.

.. rubric:: A note on the damping factor

Equation 8 writes ``p(u) = d/N + (1 - d) * sum(...)``, so the paper's ``d`` is
the *teleport* probability, and the text recommends ``d`` in ``[0.1, 0.2]``.
Table 2 then reports scores "setting the damping factor to 0.85", which is the
*other* convention -- 0.85 is the probability of following an edge. Reproducing
Table 2 confirms the latter reading: with edge-following probability 0.85 the
published values come back to within the rounding of the paper's own Figure 1,
whereas taking ``d = 0.85`` literally is off by more than 0.5.

This module therefore takes ``damping`` in the standard PageRank sense (the
probability of following an edge). The paper's ``d`` is ``1 - damping``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_DAMPING = 0.85
DEFAULT_THRESHOLD = 0.1


class ConvergenceError(RuntimeError):
    """The power method did not converge within the iteration budget."""


def adjacency_matrix(similarity: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Binarize a similarity matrix.

    Algorithm 3 uses a strict ``>`` comparison, which is reproduced here.
    """
    return (np.asarray(similarity, dtype=np.float64) > threshold).astype(np.float64)


def degree_centrality(similarity: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """Degree of every node in the thresholded graph (Section 3.1).

    Self links count, matching Table 1 of the paper: "there should also be self
    links for all of the nodes in the graphs".
    """
    return adjacency_matrix(similarity, threshold).sum(axis=1)


def stochastic_matrix(
    similarity: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    continuous: bool = False,
) -> np.ndarray:
    """Row-normalized transition matrix ``B`` (Equation 6).

    With ``continuous=False`` the graph is binarized first, so ``B(i, j)`` is
    ``1/deg(i)`` for every neighbour -- Algorithm 3, lines 17-21. With
    ``continuous=True`` the cosine values are kept as edge weights and merely
    row-normalized, which is the fraction in Equation 10.
    """
    matrix = np.asarray(similarity, dtype=np.float64)
    if continuous:
        weights = np.where(matrix > threshold, matrix, 0.0) if threshold > 0 else matrix.copy()
    else:
        weights = adjacency_matrix(matrix, threshold)

    row_sums = weights.sum(axis=1, keepdims=True)
    # The diagonal of a similarity matrix is 1.0, so a row sum can only be zero
    # for an empty graph; guard anyway rather than emit NaNs.
    row_sums[row_sums == 0.0] = 1.0
    return weights / row_sums


def power_method(
    transition: np.ndarray,
    *,
    damping: float = DEFAULT_DAMPING,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
    strict: bool = False,
) -> np.ndarray:
    """Stationary distribution of the damped chain (Algorithm 2).

    Iterates ``p_t = [(1 - damping)/N] * 1 + damping * B^T p_{t-1}`` from a
    uniform start, which is Algorithm 2 applied to the mixture kernel of
    Equation 9, and stops when the L2 norm of the update falls below
    ``tolerance``.

    Args:
        transition: Row-stochastic matrix ``B``.
        damping: Probability of following an edge. See the module docstring on
            the paper's inverted convention.
        strict: Raise :class:`ConvergenceError` instead of returning the last
            iterate if the budget runs out.

    Returns:
        The stationary distribution, summing to 1.
    """
    matrix = np.asarray(transition, dtype=np.float64)
    n = matrix.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if not 0.0 <= damping <= 1.0:
        raise ValueError("damping must lie in [0, 1]")

    teleport = (1.0 - damping) / n
    p = np.full(n, 1.0 / n, dtype=np.float64)
    transposed = matrix.T

    for _ in range(max_iterations):
        nxt = teleport + damping * (transposed @ p)
        total = nxt.sum()
        if total > 0.0:
            nxt /= total  # guard against drift from rounding
        delta = float(np.linalg.norm(nxt - p))
        p = nxt
        if delta < tolerance:
            return p

    if strict:
        raise ConvergenceError(f"power method did not converge in {max_iterations} iterations")
    return p


def lexrank_scores(
    similarity: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    damping: float = DEFAULT_DAMPING,
    continuous: bool = False,
    tolerance: float = 1e-8,
    max_iterations: int = 1000,
) -> np.ndarray:
    """LexRank over a similarity matrix (Algorithm 3, or Equation 10).

    Args:
        similarity: Symmetric idf-modified-cosine matrix.
        threshold: Cosine cutoff. In continuous mode, ``0.0`` keeps every
            non-zero edge, which is what Section 3.3 describes.
        continuous: Use the weighted graph instead of the binarized one.

    Returns:
        The stationary distribution; scores sum to 1.
    """
    transition = stochastic_matrix(similarity, threshold=threshold, continuous=continuous)
    return power_method(
        transition,
        damping=damping,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )


def normalize_max(scores: np.ndarray) -> np.ndarray:
    """Scale so the largest score is 1, as in Tables 1 and 2."""
    scores = np.asarray(scores, dtype=np.float64)
    peak = scores.max() if scores.size else 0.0
    return scores / peak if peak > 0 else scores.copy()


def normalize_minmax(scores: np.ndarray) -> np.ndarray:
    """Scale to ``[0, 1]``, as Section 5 does for MEAD feature values:
    "the sentence that has the highest value gets the score 1, and the sentence
    with the lowest value gets the score 0"."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.copy()
    low, high = scores.min(), scores.max()
    if high == low:
        return np.ones_like(scores)
    return (scores - low) / (high - low)
