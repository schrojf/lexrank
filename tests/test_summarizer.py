"""The MEAD-style pipeline end to end."""

from __future__ import annotations

import numpy as np
import pytest

from lexrank import DUC_BYTE_BUDGET, LexRankSummarizer, summarize
from lexrank.datasets import build_idf, load_cluster
from lexrank.summarizer import Method

METHODS: tuple[Method, ...] = (
    "lexrank",
    "continuous",
    "degree",
    "centroid",
    "lead",
    "random",
)


@pytest.fixture(scope="module")
def english():
    return load_cluster("en", "harbour-storm")


@pytest.fixture(scope="module")
def slovak():
    return load_cluster("sk", "povoden-na-vrbnici")


# -- ranking ---------------------------------------------------------------


def test_ranking_covers_every_sentence(english) -> None:
    ranking = LexRankSummarizer("en").rank(english.documents)
    n = len(ranking.sentences)
    assert n > 30
    assert ranking.similarity.shape == (n, n)
    assert ranking.centrality.shape == (n,)
    assert ranking.score.shape == (n,)


def test_position_feature_runs_from_one_to_zero(english) -> None:
    """ "the first sentence of a document gets the maximum Position value of 1,
    and the last sentence gets the value 0"."""
    ranking = LexRankSummarizer("en").rank(english.documents)
    for sentence, position in zip(ranking.sentences, ranking.position, strict=True):
        if sentence.index_in_document == 0:
            assert position == pytest.approx(1.0)
        elif sentence.index_in_document == sentence.document_length - 1:
            assert position == pytest.approx(0.0)


def test_length_cutoff_marks_short_sentences_ineligible() -> None:
    documents = ["Too short. This sentence is comfortably longer than the cutoff value."]
    ranking = LexRankSummarizer("en", length_cutoff=9).rank(documents)
    assert ranking.eligible.tolist() == [False, True]


def test_length_cutoff_is_dropped_when_it_would_reject_everything() -> None:
    ranking = LexRankSummarizer("en", length_cutoff=9).rank(["Too short. Also short."])
    assert ranking.eligible.all()


def test_ordered_indices_skip_ineligible_sentences() -> None:
    documents = ["Too short. This sentence is comfortably longer than the cutoff value."]
    ranking = LexRankSummarizer("en", length_cutoff=9).rank(documents)
    assert ranking.ordered_indices() == [1]


def test_empty_input_produces_an_empty_summary() -> None:
    summary = LexRankSummarizer("en").summarize([])
    assert len(summary) == 0
    assert summary.text == ""


# -- selection -------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_every_method_produces_a_summary(english, method: Method) -> None:
    summarizer = LexRankSummarizer("en", method=method, seed=0)
    summary = summarizer.summarize(english.documents, max_sentences=3)
    assert 1 <= len(summary) <= 3
    assert summary.text.strip()


def test_sentence_budget(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_sentences=4)
    assert len(summary) == 4


def test_byte_budget_is_respected(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_bytes=DUC_BYTE_BUDGET)
    assert len(summary.text.encode("utf-8")) <= DUC_BYTE_BUDGET
    assert len(summary) > 1


def test_word_budget_is_respected(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_words=40)
    assert sum(s.word_count for s in summary.sentences) <= 40


def test_a_tiny_budget_still_returns_one_sentence(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_bytes=5)
    assert len(summary) == 1


def test_document_order_is_restored_by_default(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_sentences=5)
    assert summary.indices == sorted(summary.indices)


def test_score_order_follows_the_ranking(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_sentences=5, order="score")
    scores = [summary.ranking.score[i] for i in summary.indices]
    assert scores == sorted(scores, reverse=True)


def test_selected_sentences_come_from_several_documents(english) -> None:
    summary = LexRankSummarizer("en").summarize(english.documents, max_sentences=5)
    assert len({s.document_id for s in summary.sentences}) >= 3


# -- reranker --------------------------------------------------------------


def test_reranker_drops_a_near_duplicate() -> None:
    duplicate = "The harbour wall was overtopped by a two metre surge on Tuesday morning."
    documents = [
        duplicate,
        duplicate,
        "Power was lost to eighteen thousand households across the whole peninsula.",
    ]
    with_reranker = LexRankSummarizer("en", reranker_threshold=0.5).summarize(
        documents, max_sentences=3
    )
    without = LexRankSummarizer("en", reranker_threshold=None).summarize(documents, max_sentences=3)
    assert len(with_reranker) == 2
    assert len(without) == 3


def test_reranker_threshold_of_one_keeps_everything() -> None:
    duplicate = "The harbour wall was overtopped by a two metre surge on Tuesday."
    summary = LexRankSummarizer("en", reranker_threshold=1.0).summarize(
        [duplicate, duplicate], max_sentences=2
    )
    assert len(summary) == 2


# -- configuration ---------------------------------------------------------


def test_centrality_weight_of_zero_reduces_to_the_lead_baseline(english) -> None:
    lexrank = LexRankSummarizer("en", centrality_weight=0.0).summarize(
        english.documents, max_sentences=4
    )
    lead = LexRankSummarizer("en", method="lead").summarize(english.documents, max_sentences=4)
    assert lexrank.indices == lead.indices


def test_position_weight_of_zero_changes_the_selection(english) -> None:
    with_position = LexRankSummarizer("en").summarize(english.documents, max_sentences=5)
    without = LexRankSummarizer("en", position_weight=0.0).summarize(
        english.documents, max_sentences=5
    )
    assert with_position.indices != without.indices


def test_supplying_a_background_idf_changes_scores(english) -> None:
    derived = LexRankSummarizer("en").rank(english.documents)
    background = LexRankSummarizer("en", idf=build_idf("en")).rank(english.documents)
    assert not np.allclose(derived.centrality, background.centrality)


def test_random_baseline_is_reproducible(english) -> None:
    first = LexRankSummarizer("en", method="random", seed=7).summarize(
        english.documents, max_sentences=4
    )
    second = LexRankSummarizer("en", method="random", seed=7).summarize(
        english.documents, max_sentences=4
    )
    other = LexRankSummarizer("en", method="random", seed=8).summarize(
        english.documents, max_sentences=4
    )
    assert first.indices == second.indices
    assert first.indices != other.indices


def test_unknown_method_is_rejected(english) -> None:
    with pytest.raises(ValueError, match="unknown method"):
        LexRankSummarizer(
            "en",
            method="nonsense",  # pyright: ignore[reportArgumentType]
        ).rank(english.documents)


def test_unknown_language_is_rejected() -> None:
    with pytest.raises(KeyError):
        LexRankSummarizer("xx")


# -- Slovak ----------------------------------------------------------------


def test_slovak_pipeline(slovak) -> None:
    summarizer = LexRankSummarizer("sk", idf=build_idf("sk"))
    summary = summarizer.summarize(slovak.documents, max_bytes=DUC_BYTE_BUDGET)
    assert len(summary) > 1
    assert len(summary.text.encode("utf-8")) <= DUC_BYTE_BUDGET
    # Diacritics survive: only the tokenizer folds them, never the output.
    assert any(ch in summary.text for ch in "áäčďéíĺľňóôŕšťúýž")


def test_slovak_summary_finds_the_central_facts(slovak) -> None:
    """The figures repeated across the cluster should reach the summary."""
    summary = LexRankSummarizer("sk", idf=build_idf("sk")).summarize(
        slovak.documents, max_bytes=DUC_BYTE_BUDGET
    )
    assert "612" in summary.text
    assert "tisícdvesto" in summary.text


def test_english_summary_finds_the_central_facts(english) -> None:
    summary = LexRankSummarizer("en", idf=build_idf("en")).summarize(
        english.documents, max_bytes=DUC_BYTE_BUDGET
    )
    assert "two-metre" in summary.text
    assert "Aldren Bay" in summary.text


# -- convenience wrapper ---------------------------------------------------


def test_summarize_helper_accepts_a_bare_string() -> None:
    text = (
        "The river burst its banks on Tuesday morning and flooded the lower town. "
        "Emergency services evacuated four thousand residents during the night. "
        "The river reached its highest level since records began in 1878. "
        "Power was lost to eighteen thousand households across the peninsula."
    )
    summary = summarize(text, "en", sentences=2)
    assert len(summary) == 2


def test_summarize_helper_on_a_cluster(english) -> None:
    summary = summarize(english.documents, "en", sentences=3)
    assert len(summary) == 3


def test_summary_iteration_and_length(english) -> None:
    summary = summarize(english.documents, "en", sentences=3)
    assert len(list(summary)) == len(summary) == 3
    assert summary.text == " ".join(s.text for s in summary)
