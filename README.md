# LexRank

A full implementation of **LexRank: Graph-based Lexical Centrality as Salience in
Text Summarization** — Güneş Erkan and Dragomir R. Radev, *JAIR* 22 (2004) 457–479
([arXiv:1109.2128](https://arxiv.org/abs/1109.2128), included as `1109.2128.pdf`).

LexRank is an **extractive, unsupervised** summarizer: it selects the most central
existing sentences from a document set. No training, no model weights, no network.
It runs in milliseconds on a laptop and returns sentences verbatim.

Bundled with demo corpora in **English and Slovak**.

```bash
uv sync
uv run lexrank demo -l en
```

---

## Contents

- [How it works](#how-it-works) · [What is implemented](#what-is-implemented)
- [What it's for](#what-its-for) · [Strengths and weaknesses](#strengths-and-weaknesses)
- [Usage](#usage) · [Worked examples](#worked-examples)
- [LexRank and LLMs](#lexrank-and-llms) — replace, or combine
- [Reproducing the paper](#reproducing-the-paper) · [Demo datasets](#demo-datasets)
- [Language support](#language-support) · [Performance](#performance)

---

## How it works

The premise in one sentence: **a sentence is important if it is similar to many
other important sentences.** That is circular on purpose — it is resolved the same
way PageRank resolves link importance, as the stationary distribution of a random
walk.

### 1. Sentences become vectors

Each sentence is tokenized, stopworded, stemmed, and turned into a bag-of-words
vector weighted by `tf * idf`. Rare words count for more; words that appear
everywhere count for nothing.

### 2. Sentences become a graph

Every pair gets an **idf-modified-cosine** similarity (Eq. 2) — which is exactly
the cosine of those two `tf * idf` vectors. Sentences are nodes; similarity is
edge weight. Every node also links to itself.

### 3. The graph becomes a Markov chain

Threshold the edges (keep similarity `> 0.1`), row-normalize so each row sums to
1, and you have a transition matrix: *from this sentence, jump to a random
similar one*. Mix in a small chance of teleporting anywhere (the damping factor)
so the walk can escape disconnected regions.

### 4. Run the walk to convergence

The stationary distribution — how much time the walker spends on each sentence —
is the LexRank score. The power method (Algorithm 2) converges in a few dozen
iterations.

### Worked example

Five sentences, run through the real implementation:

```
S1  A two metre surge came over the harbour wall on Tuesday.
S2  The harbour wall was overtopped by a two metre surge on Tuesday.
S3  The surge over the harbour wall flooded the lower town and cut power.
S4  Power was cut to the lower town for three days.
S5  The ferry terminal roof blew away.
```

Cosine matrix, then degree and LexRank at threshold 0.1:

```
      S1    S2    S3    S4    S5          degree   LexRank   normalised
S1  1.00  0.70  0.25  0.00  0.00             3      0.1964      0.757
S2  0.70  1.00  0.25  0.00  0.00             3      0.1964      0.757
S3  0.25  0.25  1.00  0.50  0.00             4      0.2593      1.000   <- most central
S4  0.00  0.00  0.50  1.00  0.00             2      0.1480      0.571
S5  0.00  0.00  0.00  0.00  1.00             1      0.2000      0.771
```

S3 wins: it is the only sentence that bridges the *surge* cluster (S1, S2) and
the *power* cluster (S4). Neither S1 nor S2 wins despite being near-duplicates of
each other — mutual similarity alone is not centrality.

**S5 is the interesting one.** It shares nothing with anything, yet scores
`0.2000` — beating S1 and S2. That is not a bug: an isolated node's balance
equation is `p = (1-d)/N + d·p`, whose solution is exactly `1/N` for any damping.
An unconnected sentence always banks the full teleport share, while a connected
one can leak probability to its neighbours. See
[Strengths and weaknesses](#strengths-and-weaknesses).

Reproduce it:

```python
from lexrank import IdfModel, content_tokens, lexrank_scores, similarity_matrix

tokens = [content_tokens(s, "en") for s in sentences]
idf = IdfModel.from_token_documents(tokens, smoothing="smooth")
lexrank_scores(similarity_matrix(tokens, idf), threshold=0.1)
```

---

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

---

## What it's for

**Reach for LexRank when:**

- You have **many documents about one topic** — news clusters, review sets,
  support tickets, survey responses, search results, meeting transcripts from the
  same project. Cross-document repetition is the signal it eats.
- You need the output to be **verbatim source text** — legal, medical, or
  compliance contexts where paraphrase is a liability.
- You need **determinism**: same input, same output, forever. Auditable, testable,
  diffable.
- The data **cannot leave the machine**, or there is no network.
- You are summarizing at **high volume** where per-document cost matters.
- You need a **retrieval or pre-filter stage** rather than a finished summary
  (see [LexRank and LLMs](#lexrank-and-llms)).
- The language has **no good model support** but does have a tokenizer and a
  stopword list.

**Don't reach for LexRank when:**

- You have **one short document**. With nothing to be redundant against, the
  graph collapses. Measured on the bundled corpus: a single 8-sentence document
  gives a centrality spread of **1.00×** — every sentence scores identically, so
  the ranking carries no information. The same text as part of a 5-document,
  36-sentence cluster gives a spread of **3.98×**.
- You need **abstraction** — merging facts across sentences, rewriting, inferring,
  or compressing below sentence granularity.
- You need the summary to **answer a question** or follow an instruction. LexRank
  is generic and topic-blind; it has no notion of what you asked.
- **Fluency and coherence matter more than fidelity.** Extracted sentences carry
  dangling pronouns and orphaned references.
- Importance is **not** correlated with repetition — creative writing, narrative,
  argument, a document where the key point is stated exactly once.

---

## Strengths and weaknesses

| Strength | Why |
| --- | --- |
| Cannot hallucinate | Output sentences are byte-identical to input sentences |
| Fully deterministic | No sampling; same input → same output |
| No training data | Unsupervised; works on a corpus you got five minutes ago |
| Language-portable | Needs a tokenizer, stopwords, and a stemmer — not a pretrained model |
| Fast and small | ~8 ms for 36 sentences, no GPU, no weights to ship |
| Interpretable | Every score traces to a similarity matrix you can inspect and plot |
| Robust to noise | The paper's §5.3 result: an off-topic document in the cluster barely moves the ranking |
| Naturally deduplicates | Redundant sentences split each other's centrality; the reranker removes the rest |

| Weakness | Detail |
| --- | --- |
| **Extractive only** | Cannot merge two half-facts into one sentence, or shorten a long one |
| **No topic/query awareness** | Generic summaries only; cannot answer "what did they say about cost?" |
| **Broken anaphora** | Extracted sentences reference antecedents that didn't make the cut |
| **Redundancy ≈ importance** | Repeated boilerplate scores as central. Demonstrated below |
| **Isolated sentences bank `1/N`** | An unrelated sentence can outrank a weakly-connected relevant one (see S5 above) |
| **Degenerate on uniform graphs** | If every node has equal degree, the transition matrix is doubly stochastic and *all* scores are equal — this is why the paper's own Table 2 is all `1.0000` at threshold 0.3 |
| **Threshold sensitivity** | The paper's §5.1 finding: 0.1 works, 0.4 destroys the graph |
| **O(n²) memory** | 2,304 sentences → a 42 MB matrix; ~10k sentences is the practical single-machine ceiling |
| **Lead bias is hard to beat** | On news, "just take the first sentences" is a genuinely strong baseline (0.479 vs LexRank's 0.498 on the bundled corpus) |

### The boilerplate failure, demonstrated

Three press releases sharing a contact-details line:

```
0.2398  Aldren Chemical will close its plant in March.
0.1627  For further information contact the Aldren Chemical press office.
0.1627  For further information contact the Aldren Chemical press office.
0.1627  For further information contact the Aldren Chemical press office.
0.1361  Four hundred jobs will go when Aldren Chemical closes the plant.
0.1361  Shares in Aldren Chemical fell nine percent after the plant closure.
```

LexRank gets the top pick right, but **the boilerplate outranks two of the three
real news facts** purely by being repeated. Mitigations: raise the threshold,
strip boilerplate before ranking, or add a document-frequency ceiling to the IDF
model.

### The anaphora failure, demonstrated

A five-sentence summary of `harbour-storm`:

```
[d1] Storm Mairead came ashore at Aldren Bay shortly before dawn on Tuesday, driving a
     two-metre surge over the harbour wall and flooding the whole of the lower town.
[d1] Emergency services evacuated more than four thousand residents from the districts
     of Quayside and Mill Row during the night.
[d2] More than four thousand people were evacuated from the port town of Aldren Bay
     early on Tuesday as Storm Mairead pushed a two-metre storm surge over the
     harbour wall.
[d3] Insurers expect claims from Storm Mairead to exceed 300 million euro, most of it
     from the flooded lower town of Aldren Bay.
[d4] Pressure mounted on the regional council on Wednesday over flood defences that
     were approved four years ago but never built.
```

Every fact is accurate and verbatim — but sentences 1 and 3 say the same thing
twice (the reranker's 0.5 threshold didn't catch the paraphrase), and the last
sentence introduces "flood defences" as though already known. No extractive
method fixes this; that is what the LLM hybrid below is for.

---

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

### Tuning

| Knob | Default | Effect |
| --- | --- | --- |
| `threshold` | `0.1` | Cosine cutoff. The paper's best. Higher discards real edges |
| `method` | `"lexrank"` | `continuous` uses edge weights; `degree` is cheaper and nearly as good |
| `damping` | `0.85` | Edge-following probability. Lower flattens scores toward uniform |
| `centrality_weight` | `1.0` | Raise to reduce lead bias; `0.0` gives the pure lead baseline |
| `position_weight` | `1.0` | Set `0.0` on non-news text where position means nothing |
| `length_cutoff` | `9` | Drops fragments. Lower it for terse text (chat, reviews) |
| `reranker_threshold` | `0.5` | Redundancy filter. Lower removes more near-duplicates |
| `idf` | derived | Pass a background corpus model for better term weighting |

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

---

## Worked examples

### English — `harbour-storm` (5 documents, 36 sentences, 665 words)

Ranked by raw centrality, *before* the Position feature and reranker:

```
0.0567  [d2]  More than four thousand people were evacuated from the port town of
              Aldren Bay early on Tuesday as Storm Mairead pushed a two-metre storm surge…
0.0453  [d4]  Storm Mairead sent a two-metre surge over the harbour wall at Aldren Bay
              on Tuesday, flooding the lower town…
0.0380  [d1]  Council leader Tomas Bekker told reporters that the flood defences approved
              four years ago had not yet been built.
```

Final 3-sentence summary (`lexrank summarize -d harbour-storm -n 3`):

> Storm Mairead came ashore at Aldren Bay shortly before dawn on Tuesday, driving
> a two-metre surge over the harbour wall and flooding the whole of the lower
> town. More than four thousand people were evacuated from the port town of
> Aldren Bay early on Tuesday as Storm Mairead pushed a two-metre storm surge over
> the harbour wall. Insurers expect claims from Storm Mairead to exceed 300
> million euro, most of it from the flooded lower town of Aldren Bay.

ROUGE-1 against two references: `R=0.4251 P=0.4494 F1=0.4369`.

### Slovak — `povoden-na-vrbnici` (5 documents, 36 sentences, 434 words)

Ranked by raw centrality:

```
0.0473  [d4]  Na primátora Dolných Lužian Mareka Slávika rastie tlak pre nedokončenú
              protipovodňovú ochranu.
0.0466  [d2]  Vyše tisícdvesto ľudí museli v utorok evakuovať z Dolných Lužian po tom,
              čo sa rieka Vrbnica vyliala z koryta.
0.0435  [d5]  Týždeň po povodni sa dolná časť Dolných Lužian ešte stále vysušuje.
```

Final 3-sentence summary (`lexrank summarize -l sk -d povoden-na-vrbnici -n 3`):

> Vyše tisícdvesto ľudí museli v utorok evakuovať z Dolných Lužian po tom, čo sa
> rieka Vrbnica vyliala z koryta. Na primátora Dolných Lužian Mareka Slávika
> rastie tlak pre nedokončenú protipovodňovú ochranu. Týždeň po povodni sa dolná
> časť Dolných Lužian ešte stále vysušuje.

ROUGE-1 at this budget: `R=0.2090 P=0.3333 F1=0.2569`. ROUGE is recall-based, so
a 3-sentence summary is penalised against a 5-sentence reference — at the paper's
665-byte budget the same cluster scores `R=0.5224`. Compare like-for-like budgets.

Note the Slovak pipeline is doing real morphological work: `Lužian` / `Lužanmi` /
`Lužany` all reduce to one stem, which is why the three documents connect at all.

---

## LexRank and LLMs

LexRank is from 2004. An LLM will write a **better summary** than it will —
fluent, abstractive, instruction-following, coherent. That is not in dispute, and
if summary quality is the only axis you care about, use an LLM.

LexRank still earns its place, on different axes.

### Where each one wins

| | LexRank | LLM |
| --- | --- | --- |
| Output | Verbatim sentences | Fluent, rewritten prose |
| Abstraction | None | Merges, infers, compresses |
| Query/instruction following | None | Native |
| Coherence | Poor (dangling references) | Good |
| Hallucination risk | **Zero by construction** | Non-zero; needs grounding |
| Determinism | **Exact** | Varies between runs |
| Latency | **~8 ms** for 36 sentences | Hundreds of ms to minutes |
| Cost at scale | **Electricity** | Per-token |
| Offline / air-gapped | **Yes** | Only with local weights |
| Data leaves machine | **No** | Yes, unless self-hosted |
| Auditability | **Full** — inspect the matrix | Limited |
| Long inputs | O(n²) but no context limit | Bounded by context window |

### Where an LLM simply replaces it

If you need a readable summary of a handful of documents, occasionally, and the
data can leave the machine — use an LLM directly. LexRank buys you nothing there.
Current models and rates (Anthropic first-party, as of mid-2026 — check
[pricing](https://platform.claude.com/docs/en/pricing) for live figures):

| Model | Input $/1M | Output $/1M |
| --- | --- | --- |
| Claude Opus 5 (`claude-opus-5`) | $5.00 | $25.00 |
| Claude Sonnet 5 (`claude-sonnet-5`) | $3.00 | $15.00 |
| Claude Haiku 4.5 (`claude-haiku-4-5`) | $1.00 | $5.00 |

The bundled `harbour-storm` cluster is 665 words / 4,098 bytes — roughly a
thousand input tokens, so a fraction of a cent either way. **At small volume,
cost is not the argument for LexRank; latency, determinism, and privacy are.**
Cost only becomes the argument in the millions of documents. (Use
`client.messages.count_tokens()` for exact counts — don't estimate with
`tiktoken`, which is a different tokenizer and undercounts.)

### Where they combine

The interesting cases are hybrids, where LexRank does the cheap structural work
and the LLM does the linguistic work.

**1. Pre-filter for long inputs (the main one).** Cut 500 sentences to the 30
most central before they reach the model. Reduces cost and latency proportionally,
and keeps inputs inside the context window. The extract is verbatim, so nothing is
fabricated on the way in.

```python
from anthropic import Anthropic
from lexrank import LexRankSummarizer
from lexrank.datasets import build_idf, load_cluster

cluster = load_cluster("sk", "povoden-na-vrbnici")

# LexRank shortlists the most central sentences — local, deterministic, ~8 ms.
extract = LexRankSummarizer("sk", idf=build_idf("sk")).summarize(
    cluster.documents, max_sentences=15
)

# The model only ever sees the shortlist, and only has to make it read well.
client = Anthropic()
response = client.messages.create(
    model="claude-opus-5",
    max_tokens=1024,
    output_config={"effort": "low"},
    system=(
        "Napíš súvislé zhrnutie v slovenčine. Použi výhradne fakty "
        "z poskytnutých viet a nič nedopĺňaj."
    ),
    messages=[{"role": "user", "content": extract.text}],
)
print(next(b.text for b in response.content if b.type == "text"))
```

This fixes exactly the weaknesses listed above — the model repairs anaphora,
merges the duplicate surge sentences, and produces prose — while LexRank keeps
the input small and the facts sourced.

**2. Redundancy filter for RAG.** Retrieved chunks are often near-duplicates.
Running the reranker over them drops the repeats before they consume context:

```python
from lexrank import LexRankSummarizer

deduped = LexRankSummarizer("en", reranker_threshold=0.5, position_weight=0.0)
chunks = deduped.summarize(retrieved_chunks, max_sentences=20).sentences
```

**3. Grounding check.** Because LexRank output is verbatim, comparing an LLM
summary against the extractive one is a cheap signal for unsupported claims —
`rouge_1(llm_summary, [extract.text])` gives coverage in one call.

**4. Cheap triage before spending tokens.** Rank thousands of documents by
centrality locally, then send only the interesting tail to a model.

**5. Fallback path.** When the API is down, rate-limited, or the data turns out to
be classified, a mediocre summary beats no summary.

### Honest summary

Use an LLM for the summary. Use LexRank for the **selection problem** underneath
it — which sentences are worth spending tokens on — and for the cases where
determinism, verbatim output, or offline operation are hard requirements rather
than nice-to-haves.

---

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

---

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

---

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

---

## Performance

Measured on one core, including tokenization, stemming, IDF, the full similarity
matrix, and the power method:

| Sentences | `rank()` | Similarity matrix |
| --- | --- | --- |
| 36 | 8 ms | 10 KB |
| 144 | 30 ms | 0.2 MB |
| 576 | 192 ms | 2.7 MB |
| 2,304 | 772 ms | 42.5 MB |

Both time and memory are **O(n²)** in the sentence count — the dense similarity
matrix dominates. Around 10,000 sentences the matrix alone is ~800 MB, which is
the practical single-machine ceiling. Past that, chunk the corpus, raise the
threshold and use a sparse representation, or pre-cluster the documents.

---

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
