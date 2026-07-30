"""Language descriptions consumed by tokenization, IDF and similarity."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from dataclasses import dataclass


def strip_diacritics(text: str) -> str:
    """Fold ``č -> c``, ``ô -> o`` and friends by dropping combining marks."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize(
        "NFC", "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    )


@dataclass(frozen=True)
class Language:
    """Everything the pipeline needs to know about a natural language.

    Attributes:
        code: ISO 639-1 code used to look the language up in the registry.
        name: Human readable name.
        stopwords: Lowercase forms dropped before similarity is computed.
        abbreviations: Lowercase tokens that end in a period *without* ending a
            sentence (``dr``, ``napr``). Stored without the trailing period.
        ordinal_followers: Lowercase words that may legitimately follow a
            number-plus-period, e.g. Slovak ``5. mája``. Empty for languages
            that do not write ordinals that way.
        fold_diacritics: Whether the tokenizer should strip diacritics. Highly
            inflected languages benefit, because case endings alter accents
            (``práca``/``prác``); the cost is a handful of homograph collisions
            (``sud``/``súd``). Stopwords and stemmer rules for such a language
            must be written in folded form.
        stemmer: Optional callable mapping a surface form to a stem.
    """

    code: str
    name: str
    stopwords: frozenset[str] = frozenset()
    abbreviations: frozenset[str] = frozenset()
    ordinal_followers: frozenset[str] = frozenset()
    fold_diacritics: bool = False
    stemmer: Callable[[str], str] | None = None

    def normalize(self, token: str) -> str:
        """Lowercase and, when the language asks for it, fold diacritics."""
        token = token.lower()
        return strip_diacritics(token) if self.fold_diacritics else token

    def stem(self, token: str) -> str:
        return self.stemmer(token) if self.stemmer is not None else token

    def is_stopword(self, token: str) -> bool:
        return token in self.stopwords

    def is_abbreviation(self, token: str) -> bool:
        """``token`` is the word before a period, already lowercased."""
        return token in self.abbreviations or strip_diacritics(token) in self.abbreviations
