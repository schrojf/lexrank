"""Choosing a summary length — an extension, not something the paper covers.

The paper fixes 665 bytes because DUC 2004 did. That leaves an obvious practical
question unanswered: what if you do not know how long the summary should be?

Measured on the bundled corpus, the answer is less exciting than it sounds. A
fixed byte budget lands within 0.031 ROUGE-1 F1 of the per-cluster optimum on
average (worst case 0.067), and **every content-derived rule tried here did
worse**:

    fixed byte budget                    mean F1 loss 0.031
    coverage knee (diminishing returns)  mean F1 loss 0.091
    centroid coverage >= 0.50            mean F1 loss 0.084
    redundancy saturation                mean F1 loss 0.211

The reason is that the F1-optimal length tracks the *reference* length, not the
input: across the corpus it varies by 3x in sentences (3-11) and 14x in
compression ratio (0.03-0.40), but only 2.5x in bytes (517-1320). How long a
summary should be is mostly a fact about the reader, not about the input. The
paper's arbitrary-looking constant holds up.

So the helper here does not try to out-guess a byte budget. It does three
genuinely useful things instead:

1. Converts a budget into the unit you think in, using *your* data's actual
   sentence lengths, so "665 bytes" becomes "5 sentences for this cluster".
2. Reports the ceiling on useful output -- how many mutually non-redundant
   sentences exist at all.
3. Warns when the input cannot be ranked, which is the failure that actually
   costs you something.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .centroid import centroid_vector
from .idf import IdfModel

DUC_BYTE_BUDGET = 665


@dataclass(frozen=True)
class LengthSuggestion:
    """A recommended budget, the evidence behind it, and any caveats.

    Attributes:
        sentences: Recommended sentence count.
        words: Surface word count that recommendation comes to.
        bytes: UTF-8 byte length that recommendation comes to.
        total_sentences: Size of the input.
        non_redundant: How many mutually non-redundant sentences exist, at the
            reranker's threshold. A hard ceiling on useful output.
        diminishing_returns: Knee of the coverage curve -- where extra
            sentences stop adding much new material. Reported for information;
            it is a poor length rule on its own.
        coverage: Fraction of the cluster's centroid mass the recommendation
            covers.
        discriminates: Whether centrality actually varies across sentences. If
            False, LexRank cannot rank this input and the selection is being
            driven entirely by the Position feature.
        bound_by: Which constraint produced ``sentences``.
        notes: Human-readable warnings.
    """

    sentences: int
    words: int
    bytes: int
    total_sentences: int
    non_redundant: int
    diminishing_returns: int
    coverage: float
    discriminates: bool
    bound_by: str
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = (
            f"{self.sentences} sentences (~{self.words} words, ~{self.bytes} bytes) "
            f"of {self.total_sentences}, bound by {self.bound_by}"
        )
        detail = (
            f"  coverage {self.coverage:.0%} of centroid mass; "
            f"{self.non_redundant} non-redundant sentences available; "
            f"diminishing returns at {self.diminishing_returns}"
        )
        return "\n".join([head, detail, *(f"  ! {n}" for n in self.notes)])


def coverage_curve(
    token_sets: Sequence[Sequence[str]], idf: IdfModel, order: Sequence[int]
) -> np.ndarray:
    """Cumulative fraction of centroid mass covered, taking sentences in ``order``.

    The centroid is the cluster-wide ``tf * idf`` vector of Algorithm 1, so this
    measures how much of the cluster's informative vocabulary a prefix of the
    ranking accounts for.
    """
    centroid = centroid_vector(token_sets, idf)
    total = sum(centroid.values())
    if not total:
        return np.zeros(len(order), dtype=np.float64)

    seen: set[str] = set()
    running = 0.0
    out = np.empty(len(order), dtype=np.float64)
    for position, index in enumerate(order):
        for token in token_sets[index]:
            if token not in seen:
                seen.add(token)
                running += centroid.get(token, 0.0)
        out[position] = running / total
    return out


def knee_point(curve: np.ndarray) -> int:
    """Index (1-based) of maximum vertical distance above the endpoint chord.

    The standard "elbow" of a concave curve. Returns the curve length for
    inputs too short to have a knee.
    """
    n = len(curve)
    if n < 3:
        return n
    low, high = float(curve[0]), float(curve[-1])
    if high <= low:
        return n
    normalized = (curve - low) / (high - low)
    return int(np.argmax(normalized - np.linspace(0.0, 1.0, n))) + 1


def non_redundant_count(
    similarity: np.ndarray, order: Sequence[int], threshold: float
) -> int:
    """How many sentences survive the redundancy filter with no length budget."""
    chosen: list[int] = []
    for index in order:
        if chosen and max(similarity[index, j] for j in chosen) > threshold:
            continue
        chosen.append(index)
    return len(chosen)


def centrality_spread(centrality: np.ndarray) -> float:
    """Ratio of the largest centrality score to the smallest.

    1.0 means every sentence scored identically -- a uniform graph, on which
    LexRank carries no information. See the README on single-document input.
    """
    if centrality.size == 0:
        return 1.0
    low = float(centrality.min())
    return float(centrality.max()) / low if low > 0 else float("inf")
