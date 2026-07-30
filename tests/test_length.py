"""Length suggestion — an extension beyond the paper."""

from __future__ import annotations

import numpy as np
import pytest

from lexrank import (
    IdfModel,
    LengthSuggestion,
    LexRankSummarizer,
    content_tokens,
    coverage_curve,
    knee_point,
    non_redundant_count,
    suggest_length,
    summarize,
)
from lexrank.datasets import build_idf, load_cluster
from lexrank.length import centrality_spread


# -- primitives ------------------------------------------------------------


def test_coverage_curve_is_monotonic_and_bounded() -> None:
    documents = [["river", "flood"], ["river", "town"], ["quark"]]
    idf = IdfModel.from_token_documents(documents, smoothing="smooth")
    curve = coverage_curve(documents, idf, [0, 1, 2])

    assert curve.shape == (3,)
    assert np.all(np.diff(curve) >= 0), "coverage can only grow"
    assert 0.0 < curve[0] <= 1.0
    assert curve[-1] == pytest.approx(1.0), "taking everything covers everything"


def test_coverage_curve_order_matters() -> None:
    """A sentence carrying more centroid mass covers more when taken first."""
    documents = [["river", "river", "flood"], ["quark"]]
    idf = IdfModel.from_token_documents(documents, smoothing="smooth")
    assert coverage_curve(documents, idf, [0, 1])[0] > coverage_curve(
        documents, idf, [1, 0]
    )[0]


def test_coverage_curve_handles_empty_vocabulary() -> None:
    idf = IdfModel.unary()
    assert coverage_curve([[], []], idf, [0, 1]).tolist() == [0.0, 0.0]


def test_knee_point_on_a_sharp_elbow() -> None:
    """Rises fast then flattens: the knee is where it flattens."""
    curve = np.array([0.0, 0.5, 0.8, 0.9, 0.92, 0.94, 0.96, 0.98, 0.99, 1.0])
    assert 2 <= knee_point(curve) <= 4


def test_knee_point_on_a_straight_line() -> None:
    """A line has no elbow; the detector must not invent one at the start."""
    assert knee_point(np.linspace(0.0, 1.0, 20)) in (1, 20)


def test_knee_point_degenerate_inputs() -> None:
    assert knee_point(np.array([0.5, 0.7])) == 2
    assert knee_point(np.zeros(6)) == 6  # flat curve


def test_non_redundant_count_collapses_duplicates() -> None:
    similarity = np.array(
        [
            [1.0, 0.9, 0.0],
            [0.9, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert non_redundant_count(similarity, [0, 1, 2], 0.5) == 2
    assert non_redundant_count(similarity, [0, 1, 2], 0.95) == 3


def test_centrality_spread() -> None:
    assert centrality_spread(np.array([0.25, 0.25, 0.25, 0.25])) == pytest.approx(1.0)
    assert centrality_spread(np.array([0.1, 0.4])) == pytest.approx(4.0)
    assert centrality_spread(np.array([])) == 1.0


# -- the suggestion --------------------------------------------------------


@pytest.fixture(scope="module")
def cluster():
    return load_cluster("en", "harbour-storm")


def test_suggestion_is_self_consistent(cluster) -> None:
    summarizer = LexRankSummarizer("en", idf=build_idf("en"))
    s = summarizer.suggest_length(cluster.documents)

    assert isinstance(s, LengthSuggestion)
    assert 1 <= s.sentences <= s.total_sentences
    assert s.sentences <= s.non_redundant
    assert 0.0 < s.coverage <= 1.0
    assert s.words > 0 and s.bytes > 0

    # The advertised figures must match what you get by acting on them.
    actual = summarizer.summarize(cluster.documents, max_sentences=s.sentences)
    assert len(actual) == s.sentences
    assert len(actual.text.encode("utf-8")) == s.bytes
    assert sum(x.word_count for x in actual.sentences) == s.words


def test_suggestion_scales_with_the_byte_target(cluster) -> None:
    summarizer = LexRankSummarizer("en", idf=build_idf("en"))
    small = summarizer.suggest_length(cluster.documents, target_bytes=200)
    large = summarizer.suggest_length(cluster.documents, target_bytes=2000)
    assert small.sentences < large.sentences


def test_target_coverage_mode_reaches_the_target(cluster) -> None:
    summarizer = LexRankSummarizer("en", idf=build_idf("en"))
    s = summarizer.suggest_length(cluster.documents, target_coverage=0.6)
    assert s.coverage >= 0.55, "should land at or above the requested coverage"
    assert "coverage" in s.bound_by


def test_higher_coverage_needs_more_sentences(cluster) -> None:
    summarizer = LexRankSummarizer("en", idf=build_idf("en"))
    low = summarizer.suggest_length(cluster.documents, target_coverage=0.3)
    high = summarizer.suggest_length(cluster.documents, target_coverage=0.8)
    assert low.sentences < high.sentences


def test_redundant_input_is_capped_by_the_non_redundant_ceiling() -> None:
    """Twelve sentences restating three facts should not yield twelve."""
    facts = [
        "The harbour wall was overtopped by a two metre storm surge on Tuesday morning",
        "Emergency services evacuated four thousand residents from the town overnight",
        "Insurers expect the total claims from the flooding to exceed 300 million euro",
    ]
    documents = [
        ". ".join(f"{f} according to source number {v}" for f in facts) + "."
        for v in range(4)
    ]
    s = LexRankSummarizer("en").suggest_length(documents, target_coverage=0.99)
    assert s.total_sentences == 12
    assert s.non_redundant == 3
    assert s.sentences == 3
    assert s.bound_by == "non-redundant ceiling"


def test_single_document_is_flagged_as_unrankable(cluster) -> None:
    """The failure that actually costs you something — see the README."""
    s = suggest_length(cluster.documents[0].text, "en")
    assert s.discriminates is False
    assert any("cannot rank" in note for note in s.notes)


def test_a_real_cluster_is_rankable(cluster) -> None:
    s = suggest_length(cluster.documents, "en")
    assert s.discriminates is True
    assert not any("cannot rank" in note for note in s.notes)


def test_overshoot_is_reported(cluster) -> None:
    """A sentence-count budget takes whole sentences and can exceed the target."""
    s = LexRankSummarizer("en", idf=build_idf("en")).suggest_length(
        cluster.documents, target_bytes=665
    )
    if s.bytes > 665 * 1.05:
        assert any("over the 665-byte target" in note for note in s.notes)


def test_empty_input() -> None:
    s = LexRankSummarizer("en").suggest_length([])
    assert s.sentences == 0 and s.total_sentences == 0
    assert s.bound_by == "empty input"


def test_str_is_informative(cluster) -> None:
    text = str(suggest_length(cluster.documents, "en"))
    assert "sentences" in text and "coverage" in text and "non-redundant" in text


def test_ranking_exposes_the_curve(cluster) -> None:
    ranking = LexRankSummarizer("en").rank(cluster.documents)
    curve = ranking.coverage_curve()
    assert len(curve) == len(ranking.ordered_indices())
    assert np.all(np.diff(curve) >= 0)


def test_summarize_helper_accepts_sentences_none(cluster) -> None:
    auto = summarize(cluster.documents, "en", sentences=None)
    expected = suggest_length(cluster.documents, "en").sentences
    assert len(auto) == expected


@pytest.mark.parametrize("language", ["en", "sk"])
def test_works_in_both_languages(language: str) -> None:
    cluster_id = "harbour-storm" if language == "en" else "povoden-na-vrbnici"
    cluster = load_cluster(language, cluster_id)
    s = suggest_length(cluster.documents, language)
    assert s.sentences >= 1 and s.discriminates
