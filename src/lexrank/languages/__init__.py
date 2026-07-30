"""Registry of the languages the summarizer knows about."""

from __future__ import annotations

from .base import Language, strip_diacritics
from .english import ENGLISH
from .slovak import SLOVAK

_REGISTRY: dict[str, Language] = {}


def register_language(language: Language) -> None:
    """Make ``language`` resolvable by its code. Overwrites an existing entry."""
    _REGISTRY[language.code.lower()] = language


def get_language(language: str | Language) -> Language:
    """Resolve a language code (``"en"``, ``"sk"``) or pass a Language through."""
    if isinstance(language, Language):
        return language
    try:
        return _REGISTRY[language.lower()]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown language {language!r}; registered: {known}") from None


def available_languages() -> list[str]:
    return sorted(_REGISTRY)


for _language in (ENGLISH, SLOVAK):
    register_language(_language)

__all__ = [
    "ENGLISH",
    "SLOVAK",
    "Language",
    "available_languages",
    "get_language",
    "register_language",
    "strip_diacritics",
]
