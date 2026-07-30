"""ROUGE-N."""

from __future__ import annotations

import pytest

from lexrank import rouge_1, rouge_2, rouge_n
from lexrank.rouge import ngrams, truncate_to_bytes


def test_identical_text_scores_one() -> None:
    text = "the river flooded the town"
    score = rouge_1(text, [text])
    assert score.recall == 1.0
    assert score.precision == 1.0
    assert score.f1 == 1.0


def test_no_overlap_scores_zero() -> None:
    score = rouge_1("alpha beta", ["gamma delta"])
    assert score.recall == 0.0
    assert score.f1 == 0.0


def test_recall_counts_reference_coverage() -> None:
    """Three of the four reference unigrams appear in the candidate."""
    score = rouge_1("the river flooded", ["the river flooded quickly"])
    assert score.recall == pytest.approx(3 / 4)
    assert score.precision == pytest.approx(3 / 3)


def test_matches_are_clipped_by_candidate_count() -> None:
    """Repeating a word cannot inflate the match beyond the candidate's count."""
    score = rouge_1("river", ["river river river"])
    assert score.recall == pytest.approx(1 / 3)


def test_bigrams() -> None:
    score = rouge_2("the river flooded the town", ["the river rose"])
    # Reference bigrams: (the, river), (river, rose); only the first matches.
    assert score.recall == pytest.approx(1 / 2)


def test_multiple_references_pool_their_ngrams() -> None:
    score = rouge_1("alpha beta", ["alpha beta", "gamma delta"])
    assert score.recall == pytest.approx(2 / 4)


def test_scores_are_case_and_punctuation_insensitive() -> None:
    assert rouge_1("The River, flooded!", ["the river flooded"]).recall == 1.0


def test_slovak_is_tokenized_with_its_own_rules() -> None:
    score = rouge_1("Rieka sa vyliala.", ["rieka sa vyliala"], language="sk")
    assert score.recall == 1.0


def test_byte_budget_truncates_the_candidate() -> None:
    candidate = "alpha beta gamma delta epsilon"
    full = rouge_1(candidate, [candidate])
    clipped = rouge_1(candidate, [candidate], max_bytes=11)
    assert full.recall == 1.0
    assert clipped.recall < 1.0


def test_truncate_to_bytes_respects_character_boundaries() -> None:
    text = "žltý kôň"  # multi-byte characters
    for limit in range(1, len(text.encode("utf-8")) + 1):
        clipped = truncate_to_bytes(text, limit)
        assert len(clipped.encode("utf-8")) <= limit
        assert text.startswith(clipped)


def test_truncate_is_a_no_op_below_the_limit() -> None:
    assert truncate_to_bytes("short", 100) == "short"


def test_ngrams_counts() -> None:
    assert ngrams(["a", "b", "a", "b"], 2) == {("a", "b"): 2, ("b", "a"): 1}
    assert ngrams(["a"], 2) == {}


def test_ngrams_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError):
        ngrams(["a"], 0)


def test_requires_at_least_one_reference() -> None:
    with pytest.raises(ValueError):
        rouge_n("text", [])


def test_empty_candidate_scores_zero() -> None:
    score = rouge_1("", ["some reference text"])
    assert score.recall == 0.0
    assert score.precision == 0.0
    assert score.f1 == 0.0
