"""Sentence segmentation and word tokenization.

The segmenter is rule-based and deliberately conservative: over-splitting hurts
LexRank much more than under-splitting, because a fragment shares few words with
anything and becomes an isolated node in the similarity graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .languages import Language, get_language

# A word is a run of letters (any script, so Slovak diacritics are kept at this
# stage), optionally joined by apostrophes or hyphens, or a number.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’‘-][^\W\d_]+)*|\d+(?:[.,]\d+)*")

_TERMINATORS = ".!?…"
_CLOSERS = ")]}\"'»›“”„‘’"

_BOUNDARY_RE = re.compile(rf"[{re.escape(_TERMINATORS)}]+[{re.escape(_CLOSERS)}]*")
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_TRAILING_WORD_RE = re.compile(r"([^\W\d_]+|\d+)$")
_LEADING_WORD_RE = re.compile(r"^\s*([^\W\d_]+|\d+)")


@dataclass(frozen=True)
class Sentence:
    """One sentence, with the provenance the summarizer needs.

    Attributes:
        text: Surface text, unmodified.
        tokens: Normalized content tokens fed to the similarity function.
        document_id: Identifier of the source document.
        index_in_document: Zero-based position within its document, used by the
            MEAD ``Position`` feature.
        document_length: Number of sentences in the source document.
        index: Zero-based position within the whole cluster.
        word_count: Surface word count, used by the ``LengthCutoff`` feature.
    """

    text: str
    tokens: tuple[str, ...]
    document_id: str
    index_in_document: int
    document_length: int
    index: int
    word_count: int


def tokenize_words(text: str, language: str | Language) -> list[str]:
    """Split ``text`` into normalized surface tokens (no stopword/stem step)."""
    lang = get_language(language)
    return [lang.normalize(match.group()) for match in _WORD_RE.finditer(text)]


def content_tokens(
    text: str,
    language: str | Language,
    *,
    remove_stopwords: bool = True,
    stem: bool = True,
    min_length: int = 2,
    keep_numbers: bool = True,
) -> list[str]:
    """Tokens used to build sentence vectors.

    Stopword removal happens before stemming, since the stopword lists are
    written as surface forms.
    """
    lang = get_language(language)
    tokens: list[str] = []
    for token in tokenize_words(text, lang):
        if token[0].isdigit():
            if keep_numbers:
                tokens.append(token)
            continue
        if len(token) < min_length:
            continue
        if remove_stopwords and lang.is_stopword(token):
            continue
        tokens.append(lang.stem(token) if stem else token)
    return tokens


def _is_boundary(text: str, match: re.Match[str], lang: Language) -> bool:
    """Decide whether a run of terminators actually ends a sentence."""
    run = match.group()
    after = text[match.end() :]

    # Must be followed by whitespace or end of text; "3.14" and "e.g.x" are not
    # boundaries.
    if after and not after[0].isspace():
        return False
    if not after.strip():
        return True

    next_match = _LEADING_WORD_RE.match(after)
    next_word = next_match.group(1) if next_match else ""
    next_norm = lang.normalize(next_word) if next_word else ""

    before = text[: match.start()]
    prev_match = _TRAILING_WORD_RE.search(before)
    prev_word = prev_match.group(1) if prev_match else ""
    prev_norm = lang.normalize(prev_word) if prev_word else ""

    only_period = set(run) <= {"."}

    if only_period and prev_norm:
        # "Dr. Smith", "napr. tento"
        if lang.is_abbreviation(prev_norm):
            return False
        # Initials: "J. R. Smith"
        if len(prev_word) == 1 and prev_word.isalpha() and next_word[:1].isupper():
            return False
        # Ordinals and dates: "5. mája", "5. 5. 1945", "20. storočia"
        if prev_word.isdigit() and (next_word.isdigit() or next_norm in lang.ordinal_followers):
            return False

    # A lowercase continuation is almost never a new sentence.
    if next_word[:1].islower():
        return False
    return True


def split_sentences(text: str, language: str | Language) -> list[str]:
    """Segment ``text`` into sentences."""
    lang = get_language(language)
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        start = 0
        for match in _BOUNDARY_RE.finditer(paragraph):
            if not _is_boundary(paragraph, match, lang):
                continue
            piece = paragraph[start : match.end()].strip()
            if piece:
                sentences.append(piece)
            start = match.end()
        tail = paragraph[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences


@runtime_checkable
class _HasIdAndText(Protocol):
    """Anything with ``id``/``text`` attributes, e.g. :class:`~lexrank.datasets.Document`."""

    id: str
    text: str


def build_sentences(
    documents: Iterable[str] | Iterable[tuple[str, str]] | Iterable[_HasIdAndText],
    language: str | Language,
    **token_options: object,
) -> list[Sentence]:
    """Turn a cluster of documents into :class:`Sentence` records.

    ``documents`` is an iterable of raw texts, of ``(document_id, text)`` pairs,
    or of objects exposing ``id`` and ``text`` attributes. Extra keyword
    arguments are forwarded to :func:`content_tokens`.
    """
    lang = get_language(language)
    sentences: list[Sentence] = []
    for position, entry in enumerate(documents):
        if isinstance(entry, str):
            document_id, text = f"d{position + 1}", entry
        elif isinstance(entry, _HasIdAndText):
            document_id = str(getattr(entry, "id", f"d{position + 1}"))
            text = entry.text
        else:
            document_id, text = entry
        texts = split_sentences(text, lang)
        for offset, sentence_text in enumerate(texts):
            sentences.append(
                Sentence(
                    text=sentence_text,
                    tokens=tuple(content_tokens(sentence_text, lang, **token_options)),  # pyright: ignore[reportArgumentType]
                    document_id=document_id,
                    index_in_document=offset,
                    document_length=len(texts),
                    index=len(sentences),
                    word_count=len(tokenize_words(sentence_text, lang)),
                )
            )
    return sentences


def sentence_tokens(sentences: Sequence[Sentence]) -> list[Sequence[str]]:
    return [sentence.tokens for sentence in sentences]
