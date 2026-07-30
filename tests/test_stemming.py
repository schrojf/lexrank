"""Porter (English) and the light Slovak stemmer."""

from __future__ import annotations

import pytest

from lexrank.languages.stemming import porter_stem, slovak_stem

# Drawn from the vocabulary distributed with Porter's original 1980 paper.
PORTER_CASES = [
    ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"), ("caress", "caress"),
    ("cats", "cat"), ("feed", "feed"), ("agreed", "agre"), ("plastered", "plaster"),
    ("bled", "bled"), ("motoring", "motor"), ("sing", "sing"), ("conflated", "conflat"),
    ("troubled", "troubl"), ("sized", "size"), ("hopping", "hop"), ("tanned", "tan"),
    ("falling", "fall"), ("hissing", "hiss"), ("fizzed", "fizz"), ("failing", "fail"),
    ("filing", "file"), ("happy", "happi"), ("sky", "sky"),
    ("relational", "relat"), ("conditional", "condit"), ("rational", "ration"),
    ("valenci", "valenc"), ("hesitanci", "hesit"), ("digitizer", "digit"),
    ("conformabli", "conform"), ("radicalli", "radic"), ("differentli", "differ"),
    ("vileli", "vile"), ("analogousli", "analog"), ("vietnamization", "vietnam"),
    ("predication", "predic"), ("operator", "oper"), ("feudalism", "feudal"),
    ("decisiveness", "decis"), ("hopefulness", "hope"), ("callousness", "callous"),
    ("formaliti", "formal"), ("sensitiviti", "sensit"), ("sensibiliti", "sensibl"),
    ("triplicate", "triplic"), ("formative", "form"), ("formalize", "formal"),
    ("electriciti", "electr"), ("electrical", "electr"), ("hopeful", "hope"),
    ("goodness", "good"), ("revival", "reviv"), ("allowance", "allow"),
    ("inference", "infer"), ("airliner", "airlin"), ("gyroscopic", "gyroscop"),
    ("adjustable", "adjust"), ("defensible", "defens"), ("irritant", "irrit"),
    ("replacement", "replac"), ("adjustment", "adjust"), ("dependent", "depend"),
    ("adoption", "adopt"), ("homologou", "homolog"), ("communism", "commun"),
    ("activate", "activ"), ("angulariti", "angular"), ("homologous", "homolog"),
    ("effective", "effect"), ("bowdlerize", "bowdler"),
    ("probate", "probat"), ("rate", "rate"), ("cease", "ceas"), ("controll", "control"),
    ("roll", "roll"),
]


@pytest.mark.parametrize(("word", "expected"), PORTER_CASES)
def test_porter_matches_the_reference_vocabulary(word: str, expected: str) -> None:
    assert porter_stem(word) == expected


def test_porter_conflates_an_inflectional_family() -> None:
    assert len({porter_stem(w) for w in ("flood", "floods", "flooded", "flooding")}) == 1


def test_porter_leaves_short_words_alone() -> None:
    for word in ("a", "an", "is", "by"):
        assert porter_stem(word) == word


SLOVAK_FAMILIES = [
    ("mesto", "mesta", "mestu", "meste", "mestom", "mestach", "mestami"),
    ("riek", "rieky", "rieke", "riekou", "riekam"),
    ("povodne", "povodni", "povodnou", "povodnami"),
    ("skoda", "skody", "skode", "skodam", "skodami"),
]


@pytest.mark.parametrize("family", SLOVAK_FAMILIES)
def test_slovak_conflates_case_forms(family: tuple[str, ...]) -> None:
    """Every declined form of a word should reach the same stem."""
    stems = {slovak_stem(form) for form in family}
    assert len(stems) == 1, f"{family} -> {stems}"


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("pracovnikov", "pracovnik"),
        ("vysokoskolskeho", "vysokoskolsk"),
        ("obyvatelov", "obyvatel"),
        ("archeologicky", "archeologick"),
        ("centimetrov", "centimetr"),
    ],
)
def test_slovak_strips_known_endings(word: str, expected: str) -> None:
    assert slovak_stem(word) == expected


def test_slovak_leaves_short_words_alone() -> None:
    for word in ("a", "za", "pre"):
        assert slovak_stem(word) == word


def test_slovak_never_returns_an_empty_stem() -> None:
    for word in ("aaa", "ovia", "iach", "ejsieho"):
        assert slovak_stem(word)


def test_slovak_does_not_model_vowel_zero_alternation() -> None:
    """A documented limitation of light stemming, pinned so it stays visible.

    "povodeň" folds to "povoden" and keeps its epenthetic vowel, while every
    oblique case reduces to "povodn".
    """
    assert slovak_stem("povodne") == "povodn"
    assert slovak_stem("povoden") != "povodn"


def test_slovak_aggressive_mode_strips_possessives() -> None:
    assert slovak_stem("Novakov", aggressive=True) == slovak_stem("Novak")
