"""End-to-end extractive summarizer, following the MEAD setup of Section 4.2.

The paper does not use a centrality score on its own. It plugs the score into
MEAD as a feature, alongside two supporting heuristics, and reranks the result:

* ``Position`` -- "the first sentence of a document gets the maximum Position
  value of 1, and the last sentence gets the value 0", weight fixed at 1.
* ``LengthCutoff`` -- "all the sentences that have less than 9 words are
  discarded".
* a **centrality** feature (LexRank, continuous LexRank, Degree or Centroid),
  run at weights 0.5 to 10 in the paper's experiments.
* a **reranker** that "penalizes the sentences that are similar to the
  sentences already included in the summary", the word-based MMR reranker with
  a cosine threshold of 0.5 in the paper's sample policy.

Feature values are min-max normalized before combining, as Section 5 specifies.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .centrality import (
    DEFAULT_DAMPING,
    DEFAULT_THRESHOLD,
    degree_centrality,
    lexrank_scores,
    normalize_minmax,
)
from .centroid import centroid_scores
from .idf import IdfModel
from .languages import Language, get_language
from .similarity import similarity_matrix
from .tokenization import Sentence, build_sentences

Method = Literal["lexrank", "continuous", "degree", "centroid", "lead", "random"]
Order = Literal["document", "score"]

DUC_BYTE_BUDGET = 665
"""Summary length used by DUC 2004 and by the paper's ROUGE evaluation."""


@dataclass
class Ranking:
    """Per-sentence feature values and the combined score."""

    sentences: list[Sentence]
    similarity: np.ndarray
    centrality: np.ndarray
    position: np.ndarray
    eligible: np.ndarray
    score: np.ndarray
    idf: IdfModel

    def ordered_indices(self) -> list[int]:
        """Eligible sentence indices, best score first, ties broken by order."""
        candidates = [i for i in range(len(self.sentences)) if self.eligible[i]]
        return sorted(candidates, key=lambda i: (-self.score[i], i))

    def __len__(self) -> int:
        return len(self.sentences)


@dataclass
class Summary:
    """The selected sentences plus the ranking they came from."""

    indices: list[int]
    ranking: Ranking = field(repr=False)

    @property
    def sentences(self) -> list[Sentence]:
        return [self.ranking.sentences[i] for i in self.indices]

    @property
    def text(self) -> str:
        return " ".join(sentence.text for sentence in self.sentences)

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self) -> int:
        return len(self.indices)


class LexRankSummarizer:
    """Extractive multi-document summarizer.

    Args:
        language: Language code (``"en"``, ``"sk"``) or a
            :class:`~lexrank.languages.base.Language`.
        method: Which centrality feature to use. ``"lead"`` and ``"random"``
            are the two baselines from Section 5.
        threshold: Cosine threshold for the similarity graph. The paper found
            0.1 best.
        damping: Probability of following an edge; see
            :mod:`lexrank.centrality` on the paper's inverted convention.
        idf: Background IDF model. When omitted, one is derived from the
            cluster's own sentences, which is why the default smoothing is
            ``"smooth"`` rather than Equation 1's raw ``log(N / n)`` -- over a
            handful of sentences the raw form zeroes out any word that happens
            to appear everywhere.
        centrality_weight: Weight of the centrality feature in the combiner.
        position_weight: Weight of the Position feature; the paper fixes it
            at 1.
        length_cutoff: Minimum surface word count for a sentence to be
            eligible. The cutoff is dropped if it would reject every sentence.
        reranker_threshold: Cosine similarity above which a candidate is
            considered redundant with an already-selected sentence. ``None``
            disables reranking.
        seed: Seed for the ``"random"`` baseline.
    """

    def __init__(
        self,
        language: str | Language = "en",
        *,
        method: Method = "lexrank",
        threshold: float = DEFAULT_THRESHOLD,
        damping: float = DEFAULT_DAMPING,
        idf: IdfModel | None = None,
        idf_smoothing: str = "smooth",
        centrality_weight: float = 1.0,
        position_weight: float = 1.0,
        length_cutoff: int = 9,
        reranker_threshold: float | None = 0.5,
        remove_stopwords: bool = True,
        stem: bool = True,
        seed: int | None = None,
    ) -> None:
        self.language = get_language(language)
        self.method = method
        self.threshold = threshold
        self.damping = damping
        self.idf = idf
        self.idf_smoothing = idf_smoothing
        self.centrality_weight = centrality_weight
        self.position_weight = position_weight
        self.length_cutoff = length_cutoff
        self.reranker_threshold = reranker_threshold
        self.remove_stopwords = remove_stopwords
        self.stem = stem
        self.seed = seed

    # -- pipeline ----------------------------------------------------------

    def _token_options(self) -> dict[str, object]:
        return {"remove_stopwords": self.remove_stopwords, "stem": self.stem}

    def rank(
        self, documents: Iterable[str] | Iterable[tuple[str, str]]
    ) -> Ranking:
        """Score every sentence in a cluster without selecting a summary."""
        sentences = build_sentences(documents, self.language, **self._token_options())
        n = len(sentences)
        if n == 0:
            empty = np.zeros(0, dtype=np.float64)
            return Ranking(
                sentences=[],
                similarity=np.zeros((0, 0)),
                centrality=empty,
                position=empty,
                eligible=np.zeros(0, dtype=bool),
                score=empty,
                idf=self.idf or IdfModel.unary(),
            )

        tokens = [sentence.tokens for sentence in sentences]
        idf = self.idf or IdfModel.from_token_documents(
            tokens, smoothing=self.idf_smoothing
        )
        similarity = similarity_matrix(tokens, idf)
        centrality = self._centrality(tokens, similarity, idf)
        position = np.array(
            [
                1.0
                if sentence.document_length <= 1
                else 1.0 - sentence.index_in_document / (sentence.document_length - 1)
                for sentence in sentences
            ],
            dtype=np.float64,
        )

        eligible = np.array(
            [sentence.word_count >= self.length_cutoff for sentence in sentences]
        )
        if not eligible.any():
            eligible = np.ones(n, dtype=bool)

        score = self.centrality_weight * normalize_minmax(centrality)
        score = score + self.position_weight * normalize_minmax(position)

        return Ranking(
            sentences=sentences,
            similarity=similarity,
            centrality=centrality,
            position=position,
            eligible=eligible,
            score=score,
            idf=idf,
        )

    def _centrality(
        self,
        tokens: Sequence[Sequence[str]],
        similarity: np.ndarray,
        idf: IdfModel,
    ) -> np.ndarray:
        if self.method == "lexrank":
            return lexrank_scores(
                similarity, threshold=self.threshold, damping=self.damping
            )
        if self.method == "continuous":
            return lexrank_scores(
                similarity,
                threshold=0.0,
                damping=self.damping,
                continuous=True,
            )
        if self.method == "degree":
            return degree_centrality(similarity, self.threshold)
        if self.method == "centroid":
            return centroid_scores(tokens, idf)
        if self.method == "lead":
            # Position alone; contribute nothing from centrality.
            return np.zeros(len(tokens), dtype=np.float64)
        if self.method == "random":
            rng = random.Random(self.seed)
            return np.array(
                [rng.random() for _ in range(len(tokens))], dtype=np.float64
            )
        raise ValueError(f"unknown method {self.method!r}")

    # -- selection ---------------------------------------------------------

    def summarize(
        self,
        documents: Iterable[str] | Iterable[tuple[str, str]],
        *,
        max_sentences: int | None = None,
        max_words: int | None = None,
        max_bytes: int | None = None,
        order: Order = "document",
    ) -> Summary:
        """Select a summary under a length budget.

        Sentences are taken best-first; a candidate too large for the remaining
        budget is skipped rather than truncated, so the summary never ends
        mid-sentence. At least one sentence is always returned.

        Args:
            max_sentences: Cap on the number of sentences.
            max_words: Cap on the total surface word count.
            max_bytes: Cap on the UTF-8 byte length of the joined summary. The
                paper uses :data:`DUC_BYTE_BUDGET`.
            order: ``"document"`` restores the original reading order,
                ``"score"`` keeps the ranking order.
        """
        ranking = self.rank(documents)
        if not ranking.sentences:
            return Summary(indices=[], ranking=ranking)
        if max_sentences is None and max_words is None and max_bytes is None:
            max_sentences = 5

        selected = self._select(ranking, max_sentences, max_words, max_bytes)
        if order == "document":
            selected.sort()
        return Summary(indices=selected, ranking=ranking)

    def _select(
        self,
        ranking: Ranking,
        max_sentences: int | None,
        max_words: int | None,
        max_bytes: int | None,
    ) -> list[int]:
        selected: list[int] = []
        words = 0
        size = 0

        for index in ranking.ordered_indices():
            if max_sentences is not None and len(selected) >= max_sentences:
                break
            if self._is_redundant(index, selected, ranking):
                continue

            sentence = ranking.sentences[index]
            cost = len(sentence.text.encode("utf-8")) + (1 if selected else 0)
            if max_words is not None and words + sentence.word_count > max_words:
                continue
            if max_bytes is not None and size + cost > max_bytes:
                continue

            selected.append(index)
            words += sentence.word_count
            size += cost

        if not selected:
            # Every candidate blew the budget; keep the single best sentence so
            # callers always get something back.
            ordered = ranking.ordered_indices()
            if ordered:
                selected = [ordered[0]]
        return selected

    def _is_redundant(
        self, index: int, selected: Sequence[int], ranking: Ranking
    ) -> bool:
        """Cross-sentence subsumption check used by the reranker."""
        if self.reranker_threshold is None or not selected:
            return False
        return bool(
            max(ranking.similarity[index, chosen] for chosen in selected)
            > self.reranker_threshold
        )


def summarize(
    documents: Iterable[str] | Iterable[tuple[str, str]] | str,
    language: str | Language = "en",
    *,
    sentences: int = 5,
    **options: object,
) -> Summary:
    """Convenience wrapper: summarize a cluster (or a single string) in one call."""
    if isinstance(documents, str):
        documents = [documents]
    summarizer = LexRankSummarizer(language, **options)  # type: ignore[arg-type]
    return summarizer.summarize(documents, max_sentences=sentences)
