"""Centroid-based sentence salience (Section 2, Algorithm 1).

The centroid of a cluster is a pseudo-document made of the words whose
cluster-wide ``tf * idf`` exceeds a threshold; a sentence scores by how much of
that centroid it contains. This is the baseline LexRank is compared against.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from .idf import IdfModel

DEFAULT_CENTROID_THRESHOLD = 0.0


def centroid_vector(
    documents: Sequence[Sequence[str]],
    idf: IdfModel,
    threshold: float = DEFAULT_CENTROID_THRESHOLD,
) -> dict[str, float]:
    """Lines 1-18 of Algorithm 1: cluster-wide ``tf * idf``, thresholded.

    Algorithm 1 adds ``idf{w}`` once per *occurrence* of ``w``, so the
    accumulated value is ``tf_cluster(w) * idf(w)``.
    """
    weights: defaultdict[str, float] = defaultdict(float)
    for tokens in documents:
        for token in tokens:
            weights[token] += idf[token]
    return {word: value for word, value in weights.items() if value > threshold}


def centroid_scores(
    documents: Sequence[Sequence[str]],
    idf: IdfModel,
    threshold: float = DEFAULT_CENTROID_THRESHOLD,
) -> np.ndarray:
    """Centroid score per sentence (lines 19-26 of Algorithm 1)."""
    centroid = centroid_vector(documents, idf, threshold)
    return np.array(
        [sum(centroid.get(token, 0.0) for token in tokens) for tokens in documents],
        dtype=np.float64,
    )
