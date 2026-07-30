"""Bundled demo corpora for English and Slovak.

Each *cluster* mirrors the shape of a DUC Task 2 input: several documents
reporting the same event from different angles, plus human-style reference
summaries so results can be scored with :mod:`lexrank.rouge`. The redundancy
across documents is the signal LexRank exploits; the outlet-specific details are
the noise it has to ignore.

Every bundled text is **synthetic** -- written for this package to exercise the
algorithm. The events, places, organisations and people in it are invented and
should not be read as reporting on anything real.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import NamedTuple

from ..idf import IdfModel
from ..languages import Language, get_language

DATA_ROOT = resources.files("lexrank.datasets").joinpath("data")


class Document(NamedTuple):
    """One source document. Doubles as the ``(id, text)`` pair the pipeline wants."""

    id: str
    text: str


@dataclass(frozen=True)
class Cluster:
    """A set of documents about one event, with reference summaries."""

    id: str
    language: str
    title: str
    notice: str
    documents: list[Document]
    reference_summaries: list[str]
    sources: dict[str, str]

    @property
    def texts(self) -> list[str]:
        return [document.text for document in self.documents]

    def __len__(self) -> int:
        return len(self.documents)


def _read(language: str, name: str) -> dict:
    payload = DATA_ROOT.joinpath(language, f"{name}.json").read_text("utf-8")
    return json.loads(payload)


@lru_cache(maxsize=None)
def available_clusters(language: str | None = None) -> tuple[tuple[str, str], ...]:
    """``(language, cluster_id)`` pairs for every bundled cluster."""
    languages = [language] if language else ["en", "sk"]
    found: list[tuple[str, str]] = []
    for code in languages:
        entries = sorted(DATA_ROOT.joinpath(code).iterdir(), key=lambda p: p.name)
        for entry in entries:
            name = entry.name
            if name.endswith(".json") and name != "background.json":
                found.append((code, name[: -len(".json")]))
    return tuple(found)


def load_cluster(language: str, cluster_id: str) -> Cluster:
    """Load one bundled cluster."""
    payload = _read(language, cluster_id)
    documents = [Document(d["id"], d["text"]) for d in payload["documents"]]
    return Cluster(
        id=payload["id"],
        language=payload["language"],
        title=payload["title"],
        notice=payload["notice"],
        documents=documents,
        reference_summaries=list(payload["reference_summaries"]),
        sources={d["id"]: d.get("source", "") for d in payload["documents"]},
    )


def load_paper_example() -> dict:
    """Published numbers for the worked example in the paper.

    Returns Figure 1's cosine matrix together with the Degree scores of Table 1
    and the LexRank scores of Table 2, so the implementation can be checked
    against the paper's own results. See the ``reproduction_notes`` key for the
    two places where the printed paper is ambiguous.
    """
    return _read("paper", "duc2004-d1003t")


def load_background(language: str) -> list[str]:
    """General-language documents used to estimate IDF for a language.

    The paper computes idf "over a much larger and similar genre data set" than
    the cluster being summarized; this is the stand-in for that corpus.
    """
    return list(_read(language, "background")["documents"])


@lru_cache(maxsize=None)
def build_idf(
    language: str,
    *,
    smoothing: str = "smooth",
    include_clusters: bool = True,
) -> IdfModel:
    """Fit an :class:`~lexrank.idf.IdfModel` on a language's background corpus.

    Args:
        smoothing: Passed through to :class:`~lexrank.idf.IdfModel`.
        include_clusters: Also count the cluster documents, so topic words are
            not treated as completely unseen. Enabled by default because the
            bundled background corpus is small.
    """
    lang: Language = get_language(language)
    documents = load_background(lang.code)
    if include_clusters:
        for code, cluster_id in available_clusters(lang.code):
            documents.extend(load_cluster(code, cluster_id).texts)
    return IdfModel.from_documents(documents, lang, smoothing=smoothing)


__all__ = [
    "Cluster",
    "Document",
    "available_clusters",
    "build_idf",
    "load_background",
    "load_cluster",
    "load_paper_example",
]
