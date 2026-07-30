"""Inverse document frequency (Equation 1 of the paper).

    idf_i = log(N / n_i)

where ``N`` is the number of documents in the collection and ``n_i`` the number
of documents containing word ``i``. The paper computes idf "over a much larger
and similar genre data set" than the cluster being summarized, which is what
:meth:`IdfModel.from_documents` is for; summarizing without a background corpus
falls back to deriving idf from the cluster's own sentences.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from .languages import Language, get_language
from .tokenization import content_tokens

Smoothing = str  # "paper" | "smooth" | "unary"


class IdfModel:
    """Document frequencies plus a smoothing policy.

    Smoothing modes:
        ``"paper"``: ``log(N / df)`` exactly as Equation 1. A word occurring in
            every document gets idf 0 and drops out of every sentence vector.
        ``"smooth"``: ``log(1 + N / df)``, strictly positive, so ubiquitous
            words are down-weighted rather than deleted. Safer when idf is
            derived from a small cluster instead of a background corpus.
        ``"unary"``: every word gets idf 1.0, which reduces Equation 2 to a
            plain term-frequency cosine. Useful as an ablation.

    Unseen words are scored as if they occurred in a single document, the
    highest value the model can assign.
    """

    __slots__ = ("_document_frequencies", "_cache", "n_documents", "smoothing")

    def __init__(
        self,
        document_frequencies: Mapping[str, int],
        n_documents: int,
        *,
        smoothing: Smoothing = "paper",
    ) -> None:
        if n_documents <= 0:
            raise ValueError("n_documents must be positive")
        if smoothing not in ("paper", "smooth", "unary"):
            raise ValueError(f"unknown smoothing {smoothing!r}")
        self._document_frequencies = dict(document_frequencies)
        self.n_documents = n_documents
        self.smoothing = smoothing
        self._cache: dict[str, float] = {}

    # -- construction ------------------------------------------------------

    @classmethod
    def from_token_documents(
        cls,
        documents: Iterable[Sequence[str]],
        *,
        smoothing: Smoothing = "paper",
    ) -> IdfModel:
        """Build from already-tokenized documents."""
        frequencies: Counter[str] = Counter()
        n_documents = 0
        for tokens in documents:
            n_documents += 1
            frequencies.update(set(tokens))
        if n_documents == 0:
            raise ValueError("cannot build an IdfModel from zero documents")
        return cls(frequencies, n_documents, smoothing=smoothing)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[str],
        language: str | Language,
        *,
        smoothing: Smoothing = "paper",
        **token_options: object,
    ) -> IdfModel:
        """Build from raw document texts, tokenizing them the same way the
        summarizer will."""
        lang = get_language(language)
        return cls.from_token_documents(
            (content_tokens(text, lang, **token_options) for text in documents),  # type: ignore[arg-type]
            smoothing=smoothing,
        )

    @classmethod
    def unary(cls) -> IdfModel:
        return cls({}, 1, smoothing="unary")

    # -- lookup ------------------------------------------------------------

    def __getitem__(self, term: str) -> float:
        try:
            return self._cache[term]
        except KeyError:
            pass
        if self.smoothing == "unary":
            value = 1.0
        else:
            df = max(self._document_frequencies.get(term, 0), 1)
            ratio = self.n_documents / df
            value = math.log(ratio) if self.smoothing == "paper" else math.log1p(ratio)
            value = max(value, 0.0)
        self._cache[term] = value
        return value

    def document_frequency(self, term: str) -> int:
        return self._document_frequencies.get(term, 0)

    def __contains__(self, term: str) -> bool:
        return term in self._document_frequencies

    def __len__(self) -> int:
        return len(self._document_frequencies)

    def __repr__(self) -> str:
        return (
            f"IdfModel(terms={len(self)}, n_documents={self.n_documents}, "
            f"smoothing={self.smoothing!r})"
        )

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        return {
            "n_documents": self.n_documents,
            "smoothing": self.smoothing,
            "document_frequencies": self._document_frequencies,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> IdfModel:
        return cls(
            payload["document_frequencies"],  # type: ignore[arg-type]
            int(payload["n_documents"]),  # type: ignore[arg-type]
            smoothing=str(payload.get("smoothing", "paper")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> IdfModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
