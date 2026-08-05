"""Stemmers for the bundled languages.

English uses the original Porter (1980) algorithm. Slovak uses a light
suffix-stripping stemmer in the tradition of Dolamic & Savoy's Czech "Light10"
stemmer, adapted to Slovak case endings and palatalisation.
"""

from __future__ import annotations

_VOWELS = frozenset("aeiou")


# --------------------------------------------------------------------------
# English: Porter (1980)
# --------------------------------------------------------------------------


def _is_consonant(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return False
    if ch == "y":
        return i == 0 or not _is_consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Number of vowel-consonant sequences in ``stem`` (Porter's *m*)."""
    count = 0
    i = 0
    n = len(stem)
    while i < n and _is_consonant(stem, i):
        i += 1
    while i < n:
        while i < n and not _is_consonant(stem, i):
            i += 1
        if i >= n:
            break
        count += 1
        while i < n and _is_consonant(stem, i):
            i += 1
    return count


def _contains_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return len(stem) >= 2 and stem[-1] == stem[-2] and _is_consonant(stem, len(stem) - 1)


def _ends_cvc(stem: str) -> bool:
    """Porter's ``*o``: consonant-vowel-consonant where the last is not w, x or y."""
    if len(stem) < 3:
        return False
    if not (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
    ):
        return False
    return stem[-1] not in "wxy"


def _replace(word: str, suffix: str, replacement: str, min_measure: int) -> str | None:
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if _measure(stem) > min_measure:
        return stem + replacement
    return word


_STEP2 = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("enci", "ence"),
    ("anci", "ance"),
    ("izer", "ize"),
    ("bli", "ble"),
    ("alli", "al"),
    ("entli", "ent"),
    ("eli", "e"),
    ("ousli", "ous"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ator", "ate"),
    ("alism", "al"),
    ("iveness", "ive"),
    ("fulness", "ful"),
    ("ousness", "ous"),
    ("aliti", "al"),
    ("iviti", "ive"),
    ("biliti", "ble"),
    ("logi", "log"),
)

_STEP3 = (
    ("icate", "ic"),
    ("ative", ""),
    ("alize", "al"),
    ("iciti", "ic"),
    ("ical", "ic"),
    ("ful", ""),
    ("ness", ""),
)

_STEP4 = (
    "al",
    "ance",
    "ence",
    "er",
    "ic",
    "able",
    "ible",
    "ant",
    "ement",
    "ment",
    "ent",
    "ou",
    "ism",
    "ate",
    "iti",
    "ous",
    "ive",
    "ize",
)


def porter_stem(word: str) -> str:
    """Reduce an English word to its Porter stem."""
    if len(word) <= 2:
        return word

    # Step 1a
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b
    step1b_flag = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _contains_vowel(word[:-2]):
        word = word[:-2]
        step1b_flag = True
    elif word.endswith("ing") and _contains_vowel(word[:-3]):
        word = word[:-3]
        step1b_flag = True

    if step1b_flag:
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # Step 1c
    if word.endswith("y") and _contains_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Steps 2 and 3
    for suffix, replacement in _STEP2:
        result = _replace(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break
    for suffix, replacement in _STEP3:
        result = _replace(word, suffix, replacement, 0)
        if result is not None:
            word = result
            break

    # Step 4
    for suffix in _STEP4:
        if not word.endswith(suffix):
            continue
        stem = word[: len(word) - len(suffix)]
        if _measure(stem) > 1:
            word = stem
        break
    else:
        if word.endswith("ion"):
            stem = word[:-3]
            if _measure(stem) > 1 and stem.endswith(("s", "t")):
                word = stem

    # Step 5a
    if word.endswith("e"):
        stem = word[:-1]
        measure = _measure(stem)
        if measure > 1 or (measure == 1 and not _ends_cvc(stem)):
            word = stem

    # Step 5b
    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word


# --------------------------------------------------------------------------
# Slovak: light stemmer
# --------------------------------------------------------------------------

# Ordered longest-first; each entry is (minimum stem length kept, suffix).
_SK_CASE_SUFFIXES = (
    (6, "ejsieho"),
    (6, "ejsiemu"),
    (5, "atoch"),
    (5, "iiach"),
    (5, "ejsie"),
    (5, "ejsia"),
    (5, "ejsim"),
    (4, "ovia"),
    (4, "iach"),
    (4, "iami"),
    (4, "ieho"),
    (4, "iemu"),
    (4, "ymi"),
    (4, "ich"),
    (4, "ach"),
    (4, "ami"),
    (4, "emi"),
    (4, "eho"),
    (4, "emu"),
    (4, "ymu"),
    (4, "ych"),
    (4, "ove"),
    (4, "ovi"),
    (4, "ovu"),
    (4, "ovy"),
    (4, "ami"),
    (4, "och"),
    (4, "iam"),
    (4, "iat"),
    (4, "ata"),
    (4, "ovom"),
    (4, "ejsi"),
    (3, "om"),
    (3, "ou"),
    (3, "mi"),
    (3, "em"),
    (3, "im"),
    (3, "ym"),
    (3, "ej"),
    (3, "ie"),
    (3, "ia"),
    (3, "iu"),
    (3, "ov"),
    (3, "ho"),
    (3, "mu"),
    (3, "am"),
    (3, "us"),
    (3, "os"),
    (3, "es"),
    (3, "at"),
    # Single-vowel endings keep at least three characters, so "pre" survives
    # while "mesta" still reduces to "mest".
    (3, "a"),
    (3, "e"),
    (3, "i"),
    (3, "o"),
    (3, "u"),
    (3, "y"),
)

_SK_POSSESSIVE = ((5, "ovsk"), (5, "insk"), (4, "ov"), (4, "in"))

_SK_PALATALISATION = (
    ("ci", "k"),
    ("ce", "k"),
    ("ci", "k"),
    ("ck", "sk"),
    ("zi", "h"),
    ("ze", "h"),
    ("zk", "sk"),
    ("chi", "ch"),
    ("che", "ch"),
    ("si", "sk"),
    ("se", "sk"),
    ("ti", "t"),
    ("te", "t"),
    ("di", "d"),
    ("de", "d"),
    ("ni", "n"),
    ("ne", "n"),
)


def _sk_strip(word: str, rules: tuple[tuple[int, str], ...]) -> str:
    for min_length, suffix in rules:
        if len(word) > min_length and word.endswith(suffix):
            return word[: len(word) - len(suffix)]
    return word


def _sk_palatalise(word: str) -> str:
    if len(word) < 4:
        return word
    for pattern, replacement in _SK_PALATALISATION:
        if word.endswith(pattern):
            candidate = word[: len(word) - len(pattern)] + replacement
            if len(candidate) >= 2:
                return candidate
    return word


def slovak_stem(word: str, *, aggressive: bool = False) -> str:
    """Light stemmer for Slovak.

    Strips inflectional case endings and, optionally, the possessive and
    derivational suffixes that make the stemmer *aggressive*. Diacritics are
    expected to already be folded by the tokenizer, so the rules are written
    against the ASCII-folded forms.

    Known limitation: like other light stemmers, this one does not model
    vowel-zero alternation, so a nominative with an epenthetic vowel does not
    reach the stem of its own oblique cases -- ``povodeň`` folds to
    ``povoden`` while ``povodne`` and ``povodňou`` both reduce to ``povodn``.
    """
    if len(word) < 3:
        return word

    stemmed = _sk_strip(word, _SK_CASE_SUFFIXES)
    if aggressive:
        stemmed = _sk_strip(stemmed, _SK_POSSESSIVE)
        stemmed = _sk_palatalise(stemmed)
    return stemmed or word
