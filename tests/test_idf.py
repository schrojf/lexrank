"""Equation 1 and the IdfModel wrapper."""

from __future__ import annotations

import math

import pytest

from lexrank import IdfModel


@pytest.fixture
def documents() -> list[list[str]]:
    return [
        ["river", "flood", "the"],
        ["river", "the"],
        ["quark", "the"],
        ["the"],
    ]


def test_paper_smoothing_is_log_n_over_df(documents: list[list[str]]) -> None:
    idf = IdfModel.from_token_documents(documents, smoothing="paper")
    assert idf.n_documents == 4
    assert idf["the"] == pytest.approx(math.log(4 / 4))
    assert idf["river"] == pytest.approx(math.log(4 / 2))
    assert idf["quark"] == pytest.approx(math.log(4 / 1))


def test_ubiquitous_word_gets_zero_idf(documents: list[list[str]]) -> None:
    """Equation 1 deletes a word that appears in every document."""
    assert IdfModel.from_token_documents(documents)["the"] == 0.0


def test_smooth_smoothing_keeps_every_word(documents: list[list[str]]) -> None:
    idf = IdfModel.from_token_documents(documents, smoothing="smooth")
    assert idf["the"] > 0.0
    assert idf["quark"] > idf["river"] > idf["the"]


def test_unary_smoothing_reduces_to_plain_tf(documents: list[list[str]]) -> None:
    idf = IdfModel.unary()
    assert idf["anything"] == 1.0
    assert idf["the"] == 1.0


def test_unseen_word_is_treated_as_occurring_once(documents: list[list[str]]) -> None:
    idf = IdfModel.from_token_documents(documents, smoothing="paper")
    assert "unseen" not in idf
    assert idf["unseen"] == pytest.approx(math.log(4))
    assert idf["unseen"] == pytest.approx(idf["quark"])


def test_idf_is_never_negative() -> None:
    """df can exceed the document count only through a hand-built model."""
    idf = IdfModel({"weird": 10}, 4, smoothing="paper")
    assert idf["weird"] == 0.0


def test_document_frequency_accessor(documents: list[list[str]]) -> None:
    idf = IdfModel.from_token_documents(documents)
    assert idf.document_frequency("river") == 2
    assert idf.document_frequency("absent") == 0


def test_repeated_word_counts_once_per_document() -> None:
    idf = IdfModel.from_token_documents([["a", "a", "a"], ["b"]])
    assert idf.document_frequency("a") == 1


def test_from_raw_documents_tokenizes(documents: list[list[str]]) -> None:
    idf = IdfModel.from_documents(
        ["The rivers flooded the town.", "The river rose."], "en"
    )
    assert idf.n_documents == 2
    assert idf.document_frequency("river") == 2
    assert "the" not in idf  # dropped as a stopword before counting


def test_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError):
        IdfModel.from_token_documents([])


def test_rejects_unknown_smoothing() -> None:
    with pytest.raises(ValueError):
        IdfModel({"a": 1}, 1, smoothing="nonsense")


def test_round_trip_through_json(tmp_path, documents: list[list[str]]) -> None:
    original = IdfModel.from_token_documents(documents, smoothing="smooth")
    path = tmp_path / "idf.json"
    original.save(path)
    restored = IdfModel.load(path)

    assert restored.n_documents == original.n_documents
    assert restored.smoothing == original.smoothing
    assert len(restored) == len(original)
    for term in ("river", "quark", "the", "unseen"):
        assert restored[term] == pytest.approx(original[term])


def test_round_trip_preserves_non_ascii(tmp_path) -> None:
    original = IdfModel.from_documents(["Rieka sa vyliala z koryta."], "sk")
    path = tmp_path / "sk.json"
    original.save(path)
    assert IdfModel.load(path).document_frequency("riek") == 1
