"""ROUGE-N, the evaluation metric used in Section 4.1.

ROUGE is recall-oriented: it measures how much of the human reference summaries
the candidate managed to cover.

    ROUGE-N = sum_{S in refs} sum_{gram in S} Count_match(gram)
              / sum_{S in refs} sum_{gram in S} Count(gram)

where ``Count_match`` is clipped by the candidate's own count. The paper reports
ROUGE-1, "shown to agree with human judgements most".
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import override

from .languages import Language, get_language
from .tokenization import tokenize_words


@dataclass(frozen=True)
class RougeScore:
    recall: float
    precision: float
    f1: float

    @override
    def __str__(self) -> str:
        return f"R={self.recall:.4f} P={self.precision:.4f} F1={self.f1:.4f}"


def truncate_to_bytes(text: str, max_bytes: int) -> str:
    """Cut ``text`` to at most ``max_bytes`` UTF-8 bytes, as DUC does.

    The cut lands on a character boundary; a partial trailing word is kept,
    matching ROUGE's own byte-limit behaviour.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0:
        raise ValueError("n must be positive")
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(
    candidate: str,
    references: Iterable[str],
    *,
    n: int = 1,
    language: str | Language = "en",
    max_bytes: int | None = None,
) -> RougeScore:
    """Score ``candidate`` against one or more reference summaries.

    Args:
        n: N-gram order; 1 gives ROUGE-1.
        max_bytes: Truncate the candidate first, the way DUC caps submissions.
    """
    lang = get_language(language)
    if max_bytes is not None:
        candidate = truncate_to_bytes(candidate, max_bytes)

    candidate_grams = ngrams(tokenize_words(candidate, lang), n)
    references = list(references)
    if not references:
        raise ValueError("at least one reference summary is required")

    matched = 0
    reference_total = 0
    for reference in references:
        reference_grams = ngrams(tokenize_words(reference, lang), n)
        reference_total += sum(reference_grams.values())
        matched += sum(min(count, candidate_grams[gram]) for gram, count in reference_grams.items())

    candidate_total = sum(candidate_grams.values()) * len(references)
    recall = matched / reference_total if reference_total else 0.0
    precision = matched / candidate_total if candidate_total else 0.0
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return RougeScore(recall=recall, precision=precision, f1=f1)


def rouge_1(candidate: str, references: Iterable[str], **options: object) -> RougeScore:
    return rouge_n(candidate, references, n=1, **options)  # pyright: ignore[reportArgumentType]


def rouge_2(candidate: str, references: Iterable[str], **options: object) -> RougeScore:
    return rouge_n(candidate, references, n=2, **options)  # pyright: ignore[reportArgumentType]
