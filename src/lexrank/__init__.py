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
from .rouge import RougeScore, rouge_1, rouge_2, rouge_n
from .similarity import idf_modified_cosine, similarity_matrix, tfidf_matrix
from .summarizer import (
    DUC_BYTE_BUDGET,
    LexRankSummarizer,
    Ranking,
    Summary,
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
    "DEFAULT_DAMPING",
    "DEFAULT_THRESHOLD",
    "DUC_BYTE_BUDGET",
    "ConvergenceError",
    "IdfModel",
    "Language",
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
    "degree_centrality",
    "get_language",
    "idf_modified_cosine",
    "lexrank_scores",
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
    "summarize",
    "tfidf_matrix",
    "tokenize_words",
]
