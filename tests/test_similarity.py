"""Equation 2: idf-modified-cosine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lexrank import IdfModel, idf_modified_cosine, similarity_matrix, tfidf_matrix


@pytest.fixture
def idf() -> IdfModel:
    # 8 documents; "the" is everywhere, "quark" is rare.
    return IdfModel({"the": 8, "river": 4, "flood": 2, "quark": 1}, 8, smoothing="paper")


def test_matches_a_hand_computation(idf: IdfModel) -> None:
    x = ["river", "flood"]
    y = ["river", "quark"]
    idf_river, idf_flood, idf_quark = (
        math.log(2),
        math.log(4),
        math.log(8),
    )

    expected = (idf_river**2) / (
        math.sqrt(idf_river**2 + idf_flood**2) * math.sqrt(idf_river**2 + idf_quark**2)
    )
    assert idf_modified_cosine(x, y, idf) == pytest.approx(expected)


def test_term_frequency_is_used(idf: IdfModel) -> None:
    """tf changes the direction of the vector, and so the cosine.

    Repeating a word only matters when the sentence has other words too --
    scaling a one-word vector leaves the angle untouched.
    """
    query = ["river"]
    once = idf_modified_cosine(["river", "flood"], query, idf)
    twice = idf_modified_cosine(["river", "river", "flood"], query, idf)
    assert twice > once

    scaled = idf_modified_cosine(["river", "river"], query, idf)
    assert scaled == pytest.approx(idf_modified_cosine(["river"], query, idf))


def test_identical_sentences_score_one(idf: IdfModel) -> None:
    tokens = ["river", "flood", "flood"]
    assert idf_modified_cosine(tokens, tokens, idf) == pytest.approx(1.0)


def test_disjoint_sentences_score_zero(idf: IdfModel) -> None:
    assert idf_modified_cosine(["river"], ["quark"], idf) == 0.0


def test_zero_idf_words_are_invisible(idf: IdfModel) -> None:
    """ "the" occurs in every document, so log(N/N) = 0 removes it."""
    assert idf["the"] == 0.0
    assert idf_modified_cosine(["the"], ["the"], idf) == 0.0
    assert idf_modified_cosine(["river", "the"], ["river"], idf) == pytest.approx(1.0)


def test_symmetry(idf: IdfModel) -> None:
    x, y = ["river", "flood", "the"], ["river", "quark", "quark"]
    assert idf_modified_cosine(x, y, idf) == pytest.approx(idf_modified_cosine(y, x, idf))


def test_matrix_agrees_with_the_literal_formula(idf: IdfModel) -> None:
    """The vectorised path must equal Equation 2 evaluated term by term."""
    documents = [
        ["river", "flood", "flood"],
        ["river", "quark"],
        ["the", "the"],
        ["flood", "quark", "river"],
    ]
    matrix = similarity_matrix(documents, idf)
    for i, x in enumerate(documents):
        for j, y in enumerate(documents):
            if i == j:
                continue
            assert matrix[i, j] == pytest.approx(idf_modified_cosine(x, y, idf), abs=1e-12)


def test_matrix_shape_symmetry_and_diagonal(idf: IdfModel) -> None:
    documents = [["river"], ["flood"], ["the"]]
    matrix = similarity_matrix(documents, idf)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    # Self links exist for every node, including the all-stopword sentence
    # whose vector is entirely zero.
    assert np.allclose(np.diag(matrix), 1.0)


def test_zero_vector_sentence_is_isolated(idf: IdfModel) -> None:
    documents = [["river", "flood"], ["the", "the"], ["river"]]
    matrix = similarity_matrix(documents, idf)
    assert matrix[1, 0] == 0.0
    assert matrix[1, 2] == 0.0
    assert matrix[1, 1] == 1.0


def test_empty_input(idf: IdfModel) -> None:
    assert similarity_matrix([], idf).shape == (0, 0)


def test_tfidf_matrix_weights(idf: IdfModel) -> None:
    matrix, terms = tfidf_matrix([["river", "river", "flood"]], idf)
    weights = dict(zip(terms, matrix[0], strict=True))
    assert weights["river"] == pytest.approx(2 * math.log(2))
    assert weights["flood"] == pytest.approx(math.log(4))
