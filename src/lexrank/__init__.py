"""LexRank: graph-based lexical centrality as salience in text summarization.

A full implementation of Erkan & Radev (JAIR 22, 2004), arXiv:1109.2128 --
idf-modified-cosine similarity, degree centrality, LexRank with threshold,
continuous LexRank, the centroid baseline, and the MEAD-style summarization
pipeline the paper evaluates them in.

    >>> from lexrank import LexRankSummarizer
    >>> from lexrank.datasets import load_cluster
    >>> cluster = load_cluster("en", "harbour-storm")
    >>> summary = LexRankSummarizer("en").summarize(cluster.documents, max_sentences=3)
"""

from __future__ import annotations

from .centrality import (
    DEFAULT_DAMPING,
    DEFAULT_THRESHOLD,
    ConvergenceError,
    adjacency_matrix,
    degree_centrality,
    lexrank_scores,
    normalize_max,
    normalize_minmax,
    power_method,
    stochastic_matrix,
)
from .centroid import centroid_scores, centroid_vector
from .idf import IdfModel
from .languages import Language, available_languages, get_language, register_language
from .length import LengthSuggestion, coverage_curve, knee_point, non_redundant_count
from .rouge import RougeScore, rouge_1, rouge_2, rouge_n
from .similarity import idf_modified_cosine, similarity_matrix, tfidf_matrix
from .summarizer import (
    DUC_BYTE_BUDGET,
    LexRankSummarizer,
    Ranking,
    Summary,
    suggest_length,
    summarize,
)
from .tokenization import (
    Sentence,
    build_sentences,
    content_tokens,
    split_sentences,
    tokenize_words,
)

__version__ = "0.1.0"

__all__ = [
    "ConvergenceError",
    "DEFAULT_DAMPING",
    "DEFAULT_THRESHOLD",
    "DUC_BYTE_BUDGET",
    "IdfModel",
    "Language",
    "LengthSuggestion",
    "LexRankSummarizer",
    "Ranking",
    "RougeScore",
    "Sentence",
    "Summary",
    "adjacency_matrix",
    "available_languages",
    "build_sentences",
    "centroid_scores",
    "centroid_vector",
    "content_tokens",
    "coverage_curve",
    "degree_centrality",
    "get_language",
    "idf_modified_cosine",
    "knee_point",
    "lexrank_scores",
    "non_redundant_count",
    "normalize_max",
    "normalize_minmax",
    "power_method",
    "register_language",
    "rouge_1",
    "rouge_2",
    "rouge_n",
    "similarity_matrix",
    "split_sentences",
    "stochastic_matrix",
    "suggest_length",
    "summarize",
    "tfidf_matrix",
    "tokenize_words",
]
