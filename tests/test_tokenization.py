"""Sentence segmentation and word tokenization for English and Slovak."""

from __future__ import annotations

import pytest

from lexrank import build_sentences, content_tokens, split_sentences, tokenize_words
from lexrank.languages import get_language, strip_diacritics

# -- English ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("One. Two. Three.", 3),
        ("No terminator at the end", 1),
        ("Dr. Smith arrived early.", 1),
        ("The U.S. Navy confirmed it.", 1),
        ("It cost 3.14 euro per litre.", 1),
        ("J. R. Tolkien wrote it.", 1),
        ('She said "Really?" and left.', 1),
        # A lowercase continuation is never treated as a new sentence.
        ("Wait... what happened next?", 1),
        ("He paused... Then he spoke.", 2),
        ("Ready? Set! Go!", 3),
        ("Approx. 40 items, e.g. books, were lost.", 1),
        ("It ended in 1990. The next year was better.", 2),
    ],
)
def test_english_sentence_counts(text: str, expected: int) -> None:
    assert len(split_sentences(text, "en")) == expected


def test_english_keeps_surface_text() -> None:
    text = "Dr. Smith arrived. He was late."
    assert split_sentences(text, "en") == ["Dr. Smith arrived.", "He was late."]


def test_paragraph_break_ends_a_sentence_without_a_terminator() -> None:
    assert split_sentences("A headline\n\nThe body text follows.", "en") == [
        "A headline",
        "The body text follows.",
    ]


def test_single_newline_is_only_whitespace() -> None:
    assert split_sentences("One long\nsentence here.", "en") == ["One long sentence here."]


# -- Slovak ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Prvá veta. Druhá veta.", 2),
        ("Dňa 5. mája 1945 sa to začalo.", 1),
        ("Bolo to 5. 5. 1945 popoludní.", 1),
        ("Patrí do 20. storočia.", 1),
        ("Prišiel Ing. Novák z úradu.", 1),
        ("Kúpili sme napr. chlieb, mlieko a syr.", 1),
        ("Je to tzv. mäkké i.", 1),
        ("Hladina dosiahla 612 centimetrov. Bolo to najviac od roku 1974.", 2),
        ("Naozaj? Áno! Určite.", 3),
    ],
)
def test_slovak_sentence_counts(text: str, expected: int) -> None:
    assert len(split_sentences(text, "sk")) == expected


def test_slovak_ordinal_date_stays_together() -> None:
    text = "Dňa 5. mája 1945 sa začalo povstanie. Trvalo dva mesiace."
    assert split_sentences(text, "sk") == [
        "Dňa 5. mája 1945 sa začalo povstanie.",
        "Trvalo dva mesiace.",
    ]


# -- word level ------------------------------------------------------------


def test_english_tokens_keep_diacritics_and_lowercase() -> None:
    assert tokenize_words("Café RESUMÉ naïve", "en") == ["café", "resumé", "naïve"]


def test_slovak_tokens_are_diacritics_folded() -> None:
    assert tokenize_words("Príliš žltý kôň", "sk") == ["prilis", "zlty", "kon"]


def test_numbers_are_tokens() -> None:
    assert tokenize_words("612 centimetrov, 1974", "sk") == ["612", "centimetrov", "1974"]


def test_hyphenated_and_apostrophed_words_stay_whole() -> None:
    assert tokenize_words("It's a two-metre surge", "en") == [
        "it's",
        "a",
        "two-metre",
        "surge",
    ]


def test_content_tokens_drop_stopwords_and_stem() -> None:
    tokens = content_tokens("The rivers were flooding the towns", "en")
    assert "the" not in tokens and "were" not in tokens
    assert "river" in tokens and "flood" in tokens


def test_content_tokens_slovak() -> None:
    tokens = content_tokens("Rieka sa vyliala z koryta a zaplavila mesto", "sk")
    assert "sa" not in tokens and "a" not in tokens
    assert "riek" in tokens and "mest" in tokens


def test_content_tokens_can_skip_stemming_and_stopwords() -> None:
    tokens = content_tokens("The rivers flooded", "en", remove_stopwords=False, stem=False)
    assert tokens == ["the", "rivers", "flooded"]


def test_stopword_lookup_is_diacritics_insensitive_for_slovak() -> None:
    slovak = get_language("sk")
    assert slovak.is_stopword(slovak.normalize("Keď"))
    assert slovak.is_stopword(slovak.normalize("že"))


def test_strip_diacritics() -> None:
    assert strip_diacritics("Príliš žltý kôň") == "Prilis zlty kon"


# -- Sentence records ------------------------------------------------------


def test_build_sentences_records_provenance() -> None:
    documents = [("a", "First one here. Second one here."), ("b", "Only one here.")]
    sentences = build_sentences(documents, "en")

    assert [s.document_id for s in sentences] == ["a", "a", "b"]
    assert [s.index_in_document for s in sentences] == [0, 1, 0]
    assert [s.index for s in sentences] == [0, 1, 2]
    assert [s.document_length for s in sentences] == [2, 2, 1]
    assert all(s.word_count == 3 for s in sentences)


def test_build_sentences_accepts_plain_strings() -> None:
    sentences = build_sentences(["One here.", "Two here."], "en")
    assert [s.document_id for s in sentences] == ["d1", "d2"]


def test_build_sentences_accepts_objects_with_id_and_text() -> None:
    from lexrank.datasets import Document

    sentences = build_sentences([Document("x", "Some text here.")], "en")
    assert sentences[0].document_id == "x"
