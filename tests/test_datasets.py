"""The bundled English and Slovak demo corpora."""

from __future__ import annotations

import numpy as np
import pytest

from lexrank import LexRankSummarizer, rouge_1, split_sentences
from lexrank.datasets import (
    available_clusters,
    build_idf,
    load_background,
    load_cluster,
    load_paper_example,
)
from lexrank.summarizer import Method

CLUSTERS = available_clusters()


def test_both_languages_are_represented() -> None:
    languages = {language for language, _ in CLUSTERS}
    assert languages == {"en", "sk"}
    assert len(available_clusters("en")) >= 3
    assert len(available_clusters("sk")) >= 3


@pytest.mark.parametrize(("language", "cluster_id"), CLUSTERS)
def test_cluster_structure(language: str, cluster_id: str) -> None:
    cluster = load_cluster(language, cluster_id)
    assert cluster.id == cluster_id
    assert cluster.language == language
    assert cluster.title
    assert len(cluster) >= 4
    assert len(cluster.reference_summaries) >= 2
    assert all(summary.strip() for summary in cluster.reference_summaries)


@pytest.mark.parametrize(("language", "cluster_id"), CLUSTERS)
def test_documents_are_unique_and_substantial(language: str, cluster_id: str) -> None:
    cluster = load_cluster(language, cluster_id)
    ids = [document.id for document in cluster.documents]
    assert len(set(ids)) == len(ids)
    for document in cluster.documents:
        sentences = split_sentences(document.text, language)
        assert len(sentences) >= 6, f"{cluster_id}/{document.id} is too short"
    assert len(set(cluster.texts)) == len(cluster.texts)


@pytest.mark.parametrize(("language", "cluster_id"), CLUSTERS)
def test_synthetic_provenance_is_declared(language: str, cluster_id: str) -> None:
    """Every bundled corpus states that it is invented, not real reporting."""
    cluster = load_cluster(language, cluster_id)
    marker = "Synthetic" if language == "en" else "Synteticky"
    assert marker in cluster.notice


@pytest.mark.parametrize(("language", "cluster_id"), CLUSTERS)
def test_clusters_are_redundant_enough_for_lexrank(language: str, cluster_id: str) -> None:
    """The thresholded graph must actually be connected, or there is nothing
    to rank.

    Deliberately not a bound on *mean* pairwise similarity: that dilutes as a
    cluster grows, because a long article covers sub-topics that share little
    with each other. Across the bundled clusters the mean falls from 0.069 at
    30 sentences to 0.012 at 208 while the graph stays just as usable. What
    LexRank actually needs is that most nodes have neighbours above the
    threshold, which holds at every size.
    """
    cluster = load_cluster(language, cluster_id)
    ranking = LexRankSummarizer(language, idf=build_idf(language)).rank(cluster.documents)
    n = len(ranking.sentences)
    off_diagonal = ranking.similarity[~np.eye(n, dtype=bool)]
    degree = (ranking.similarity > 0.1).sum(axis=1)  # self-link included

    assert off_diagonal.max() > 0.4, "no strongly similar pair anywhere"
    # Fraction of sentences with at least one neighbour besides themselves.
    assert (degree > 1).mean() > 0.75, "too many isolated sentences"
    assert np.median(degree) >= 3, "graph too sparse to rank"


@pytest.mark.parametrize("language", ["en", "sk"])
def test_background_corpus(language: str) -> None:
    documents = load_background(language)
    assert len(documents) >= 12
    assert all(document.strip() for document in documents)
    assert len(set(documents)) == len(documents)


@pytest.mark.parametrize("language", ["en", "sk"])
def test_build_idf_covers_the_language(language: str) -> None:
    idf = build_idf(language)
    assert len(idf) > 400
    assert idf.n_documents >= 15


def _rouge(language: str, cluster, method: Method, seed: int = 0) -> float:
    summary = LexRankSummarizer(
        language, method=method, idf=build_idf(language), seed=seed
    ).summarize(cluster.documents, max_bytes=665)
    return rouge_1(summary.text, cluster.reference_summaries, language=language).recall


@pytest.mark.parametrize(("language", "cluster_id"), CLUSTERS)
def test_every_cluster_scores_above_a_floor(language: str, cluster_id: str) -> None:
    """Per-cluster sanity check.

    Deliberately not a comparison against the random baseline: on a single
    cluster ROUGE-1 is noisy enough that one lucky draw can beat LexRank. The
    paper itself medians over five random runs, and compares over 30 to 50
    clusters rather than one.

    The floor is loose because ROUGE-1 recall at a *fixed* 665-byte budget
    falls as the cluster grows: 665 bytes is roughly 100 words, which covers
    proportionally less of the larger reference summaries that a 200-sentence
    cluster warrants. The small clusters score 0.46-0.54 here, the large ones
    0.30-0.32, and neither is evidence about summary quality on its own.
    """
    cluster = load_cluster(language, cluster_id)
    assert _rouge(language, cluster, "lexrank") > 0.25


@pytest.mark.parametrize("method", ["lexrank", "continuous", "degree"])
def test_centrality_beats_the_baselines_on_average(method: Method) -> None:
    """Aggregate comparison in the spirit of Table 3.

    The random baseline is averaged over several seeds rather than taken from
    a single draw. Six small clusters cannot reproduce the paper's DUC result,
    but the ordering should still come out the right way round.
    """
    clusters = [(language, load_cluster(language, cid)) for language, cid in CLUSTERS]

    centrality = np.mean([_rouge(lang, c, method) for lang, c in clusters])
    lead = np.mean([_rouge(lang, c, "lead") for lang, c in clusters])
    random = np.mean(
        [_rouge(lang, c, "random", seed) for lang, c in clusters for seed in range(15)]
    )

    assert centrality > random
    assert centrality > lead


def test_paper_example_payload() -> None:
    data = load_paper_example()
    assert len(data["sentence_ids"]) == 11
    assert len(data["figure_1_cosine_matrix"]) == 11
    assert set(data["table_1_degree"]) == {"0.1", "0.2", "0.3"}
    assert set(data["table_2_lexrank"]) == {"0.1", "0.2", "0.3"}
    assert data["reproduction_notes"]


def test_unknown_cluster_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_cluster("en", "does-not-exist")
