"""Algorithm 1: centroid-based salience."""

from __future__ import annotations

import math

import pytest

from lexrank import IdfModel, centroid_scores, centroid_vector


@pytest.fixture
def idf() -> IdfModel:
    return IdfModel({"river": 2, "flood": 1, "the": 4}, 4, smoothing="paper")


@pytest.fixture
def documents() -> list[list[str]]:
    return [
        ["river", "flood"],
        ["river", "the"],
        ["the"],
    ]


def test_centroid_accumulates_idf_per_occurrence(
    documents: list[list[str]], idf: IdfModel
) -> None:
    """Algorithm 1 adds idf{w} once for every occurrence, giving tf*idf."""
    centroid = centroid_vector(documents, idf)
    assert centroid["river"] == pytest.approx(2 * math.log(2))
    assert centroid["flood"] == pytest.approx(math.log(4))
    assert "the" not in centroid  # idf 0, so it falls below the threshold


def test_threshold_prunes_the_centroid(idf: IdfModel) -> None:
    # river: 3 occurrences * log(2) = 2.079; flood: 1 * log(4) = 1.386.
    documents = [["river", "flood"], ["river"], ["river", "the"]]
    assert set(centroid_vector(documents, idf, threshold=0.0)) == {"river", "flood"}
    assert set(centroid_vector(documents, idf, threshold=1.5)) == {"river"}
    assert centroid_vector(documents, idf, threshold=3.0) == {}


def test_sentence_score_sums_centroid_weights(
    documents: list[list[str]], idf: IdfModel
) -> None:
    scores = centroid_scores(documents, idf)
    centroid = centroid_vector(documents, idf)
    assert scores[0] == pytest.approx(centroid["river"] + centroid["flood"])
    assert scores[1] == pytest.approx(centroid["river"])
    assert scores[2] == pytest.approx(0.0)


def test_repeated_words_score_twice(idf: IdfModel) -> None:
    documents = [["river"], ["river", "river"]]
    scores = centroid_scores(documents, idf)
    assert scores[1] == pytest.approx(2 * scores[0])


def test_a_sentence_of_only_zero_idf_words_scores_zero(idf: IdfModel) -> None:
    assert centroid_scores([["the", "the"]], idf)[0] == 0.0


def test_empty_input(idf: IdfModel) -> None:
    assert centroid_scores([], idf).tolist() == []
