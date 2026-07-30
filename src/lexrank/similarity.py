"""idf-modified-cosine, Equation 2 of the paper.

                    sum_{w in x,y} tf_{w,x} tf_{w,y} (idf_w)^2
    cos(x, y) = ------------------------------------------------------
                 sqrt(sum_{xi in x} (tf_{xi,x} idf_{xi})^2)
                   * sqrt(sum_{yi in y} (tf_{yi,y} idf_{yi})^2)

which is exactly the cosine of the two tf*idf vectors. :func:`similarity_matrix`
exploits that and works on a dense term matrix; :func:`idf_modified_cosine`
evaluates the formula literally for a single pair.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

from .idf import IdfModel


def idf_modified_cosine(
    x: Sequence[str], y: Sequence[str], idf: IdfModel
) -> float:
    """Equation 2 for a single sentence pair, written out term by term."""
    tf_x, tf_y = Counter(x), Counter(y)

    numerator = sum(
        tf_x[word] * tf_y[word] * idf[word] ** 2 for word in tf_x.keys() & tf_y.keys()
    )
    if numerator == 0.0:
        return 0.0

    norm_x = sum((count * idf[word]) ** 2 for word, count in tf_x.items())
    norm_y = sum((count * idf[word]) ** 2 for word, count in tf_y.items())
    denominator = (norm_x**0.5) * (norm_y**0.5)
    return numerator / denominator if denominator else 0.0


def tfidf_matrix(
    documents: Sequence[Sequence[str]], idf: IdfModel
) -> tuple[np.ndarray, list[str]]:
    """Dense ``(n_documents, n_terms)`` matrix of ``tf * idf`` weights."""
    vocabulary: dict[str, int] = {}
    for tokens in documents:
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

    matrix = np.zeros((len(documents), len(vocabulary)), dtype=np.float64)
    for row, tokens in enumerate(documents):
        for token, count in Counter(tokens).items():
            matrix[row, vocabulary[token]] = count * idf[token]
    terms = sorted(vocabulary, key=vocabulary.__getitem__)
    return matrix, terms


def similarity_matrix(
    documents: Sequence[Sequence[str]], idf: IdfModel
) -> np.ndarray:
    """Symmetric ``(n, n)`` matrix of idf-modified-cosine values.

    The diagonal is forced to 1.0. The paper notes that "every sentence is
    similar at least to itself" and that the graph carries self links for all
    nodes; forcing it also keeps sentences whose vector is entirely zero (every
    token a stopword, or every token with idf 0) from producing a zero row,
    which would make the transition matrix non-stochastic.
    """
    n = len(documents)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    matrix, _ = tfidf_matrix(documents, idf)
    norms = np.linalg.norm(matrix, axis=1)
    safe = np.where(norms > 0.0, norms, 1.0)
    normalized = matrix / safe[:, None]

    similarity = normalized @ normalized.T
    np.clip(similarity, 0.0, 1.0, out=similarity)
    similarity = (similarity + similarity.T) / 2.0  # kill float asymmetry
    np.fill_diagonal(similarity, 1.0)
    return similarity
