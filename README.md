# LexRank

A full implementation of **LexRank: Graph-based Lexical Centrality as Salience in
Text Summarization** — Güneş Erkan and Dragomir R. Radev, *JAIR* 22 (2004) 457–479
([arXiv:1109.2128](https://arxiv.org/abs/1109.2128), included as `1109.2128.pdf`).

Everything the paper describes is here: idf-modified-cosine similarity, degree
centrality, LexRank with threshold, continuous LexRank, the centroid baseline,
and the MEAD-style pipeline the paper evaluates them inside. Bundled with demo
corpora in **English and Slovak**.

```bash
uv sync
uv run lexrank demo -l en
```

## What is implemented

| Paper | Here |
| --- | --- |
| Eq. 1 — `idf = log(N/n)` | `lexrank.IdfModel` |
| Eq. 2 — idf-modified-cosine | `lexrank.idf_modified_cosine`, `similarity_matrix` |
| §2, Algorithm 1 — Centroid | `lexrank.centroid_scores` |
| §3.1 — Degree centrality | `lexrank.degree_centrality` |
| §3.2, Algorithm 2 — power method | `lexrank.power_method` |
| §3.2, Algorithm 3 — LexRank w/ threshold | `lexrank.lexrank_scores` |
| §3.3, Eq. 10 — continuous LexRank | `lexrank.lexrank_scores(continuous=True)` |
| §4.1 — ROUGE-N evaluation | `lexrank.rouge_n`, `rouge_1`, `rouge_2` |
| §4.2 — MEAD features + reranker | `lexrank.LexRankSummarizer` |
| §5 — lead and random baselines | `method="lead"`, `method="random"` |

## Usage

```python
from lexrank import LexRankSummarizer, rouge_1
from lexrank.datasets import load_cluster, build_idf

cluster = load_cluster("sk", "povoden-na-vrbnici")
summarizer = LexRankSummarizer("sk", idf=build_idf("sk"))

summary = summarizer.summarize(cluster.documents, max_bytes=665)
print(summary.text)
print(rouge_1(summary.text, cluster.reference_summaries, language="sk"))
```

Scores without selecting a summary:

```python
ranking = summarizer.rank(cluster.documents)
ranking.similarity   # (n, n) idf-modified-cosine matrix
ranking.centrality   # raw LexRank scores, summing to 1
ranking.position     # MEAD Position feature
ranking.score        # combined, min-max normalised features
```

The centrality functions work on a bare matrix too, with no text pipeline:

```python
import numpy as np
from lexrank import lexrank_scores, degree_centrality

similarity = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
lexrank_scores(similarity, threshold=0.1, damping=0.85)
degree_centrality(similarity, 0.1)
```

### Command line

```bash
lexrank summarize -d harbour-storm -n 3        # a bundled cluster
lexrank summarize doc1.txt doc2.txt -b 665     # your own files, DUC byte budget
cat article.txt | lexrank summarize -l sk -n 5 # stdin
lexrank datasets                               # list bundled corpora
lexrank demo -l sk                             # summarize + score every cluster
lexrank evaluate                               # compare methods by ROUGE-1
lexrank paper                                  # recompute the paper's Tables 1 and 2
```

## Reproducing the paper

`lexrank paper` recomputes Table 1 and Table 2 from the paper's own Figure 1
cosine matrix. This is also pinned in `tests/test_paper_reproduction.py`.

- **Table 1 (degree)** reproduces **exactly** at thresholds 0.1 and 0.3.
- **Table 2 (LexRank)** reproduces at threshold 0.1 to a maximum absolute error
  of **0.0006**, and exactly at 0.3.

Two things in the printed paper need resolving before it reproduces, and both
were settled empirically against the published tables:

**The damping factor is inverted relative to standard PageRank.** Eq. 8 writes
`p(u) = d/N + (1-d)·Σ…`, so the paper's `d` is the *teleport* probability, and
the text recommends `d ∈ [0.1, 0.2]`. But Table 2 says "setting the damping
factor to 0.85". Only the standard reading reproduces it — 0.85 as the
probability of *following an edge*, i.e. the paper's own `d = 0.15`. Taking
`d = 0.85` literally is off by more than 0.5. This package therefore takes
`damping` in the standard sense; the paper's `d` is `1 - damping`.

**Threshold 0.2 cannot be reproduced from Figure 1, and that is a rounding
artefact rather than a bug.** Figure 1 is printed to two decimals. Three
off-diagonal cells print as exactly `0.20`, and their true values straddle the
cutoff: pairs (d3s2, d5s2) and (d4s1, d5s2) are above it, pair (d3s1, d5s1) is
below. Nudging the first two to 0.204 recovers Table 1's 0.2 column exactly,
which confirms rounding — and confirms that Algorithm 3's strict `>` comparison
is the right one. Thresholds 0.1 and 0.3 are unaffected.

A third, smaller point: the paper normalises scores two different ways. Tables 1
and 2 divide by the maximum (`normalize_max`), while §5 min-max normalises MEAD
feature values (`normalize_minmax`). Both are provided, and the summarizer uses
the latter.

## Demo datasets

Six clusters, three per language, shaped like DUC Task 2 inputs: five documents
covering one event from different angles, plus two reference summaries each for
ROUGE. The cross-document redundancy is the signal LexRank exploits; the
outlet-specific detail is the noise it has to ignore.

| Language | Cluster | Subject |
| --- | --- | --- |
| en | `harbour-storm` | A storm surge floods a port town |
| en | `rate-decision` | A central bank raises its benchmark rate |
| en | `asteroid-sample` | A probe samples an asteroid |
| sk | `povoden-na-vrbnici` | Povodeň zaplavila dolnú časť mesta |
| sk | `reforma-vysokych-skol` | Nová metodika financovania vysokých škôl |
| sk | `archeologicky-nalez` | Mohylové pohrebisko z doby bronzovej |

Each language also ships a general-language background corpus used to estimate
IDF (`build_idf`), standing in for the "much larger and similar genre data set"
the paper computes idf over.

> **All bundled text is synthetic.** It was written for this package to exercise
> the algorithm. Every event, place, organisation and person in it is invented
> and none of it is reporting on anything real. Each dataset file carries the
> same notice in its `notice` field.

Six small clusters demonstrate the algorithm; they cannot replicate the paper's
DUC evaluation. On this corpus, ROUGE-1 recall at a 665-byte budget averages:

| continuous | lexrank | degree | centroid | lead | random |
| --- | --- | --- | --- | --- | --- |
| 0.509 | 0.498 | 0.491 | 0.490 | 0.479 | 0.476 |

(`random` averaged over 25 seeds.) The ordering matches the paper's finding that
graph-based centrality beats centroid, which beats the lead and random
baselines — but with six clusters the gaps are well inside the noise. Reproduce
with `lexrank evaluate`.

## Language support

| | English | Slovak |
| --- | --- | --- |
| Stemmer | Porter (1980) | light stemmer, Dolamic & Savoy tradition |
| Stopwords | ✓ | ✓ |
| Abbreviations | `Dr.`, `e.g.`, `U.S.` … | `napr.`, `atď.`, `Ing.` … |
| Diacritics | preserved | folded before matching |
| Ordinal dates | — | `5. mája`, `20. storočia` handled |

Sentence segmentation is rule-based and deliberately conservative — over-splitting
hurts LexRank more than under-splitting, because a fragment shares few words with
anything and becomes an isolated node in the graph.

Slovak folds diacritics before matching, because case endings alter accents
(`práca`/`prác`); the cost is a few homograph collisions (`sud`/`súd`). Set
`fold_diacritics=False` on a custom `Language` to turn it off.

Known limitation: like other light stemmers, the Slovak one does not model
vowel-zero alternation, so `povodeň` does not reach the stem of its own oblique
cases (`povodne` → `povodn`).

Adding a language needs a `Language` and a call to `register_language`:

```python
from lexrank.languages import Language, register_language

register_language(Language(
    code="cs",
    name="Czech",
    stopwords=frozenset({"a", "ale", "ani"}),
    abbreviations=frozenset({"napr", "atd"}),
    fold_diacritics=True,
    stemmer=my_stemmer,
))
```

## Development

```bash
uv sync --all-extras
uv run pytest          # 292 tests
```

## Layout

```
src/lexrank/
├── centrality.py     # power method, degree, LexRank, continuous LexRank
├── centroid.py       # Algorithm 1
├── cli.py
├── idf.py            # Equation 1
├── rouge.py          # ROUGE-N
├── similarity.py     # Equation 2
├── summarizer.py     # MEAD pipeline: features, combiner, reranker
├── tokenization.py   # sentence segmentation, word tokenization
├── datasets/         # bundled EN/SK corpora + the paper's published tables
└── languages/        # Language registry, stopwords, Porter + Slovak stemmers
```

## Licence

The implementation is original work. `1109.2128.pdf` is the authors' paper,
distributed on arXiv; the numeric tables transcribed into
`datasets/data/paper/` are cited to it for reproduction purposes.
