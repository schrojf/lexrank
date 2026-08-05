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
- [Usage](#usage) · [Command line](#command-line) · [Worked examples](#worked-examples)
- [Understanding the ROUGE scores](#understanding-the-rouge-scores)
- [Choosing a length](#choosing-a-length) — and why a fixed budget wins
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

---

## Command line

Inside a clone, prefix everything with `uv run`. Installing the package
(`uv tool install .`) puts `lexrank` on your PATH and the prefix drops.

```console
$ uv run lexrank --help
usage: lexrank [-h] {summarize,datasets,demo,evaluate,paper} ...

LexRank extractive summarization (Erkan & Radev, 2004).

positional arguments:
  {summarize,datasets,demo,evaluate,paper}
    summarize           summarize files, stdin or a bundled cluster
    datasets            list the bundled demo clusters
    demo                summarize every bundled cluster and score it
    evaluate            compare centrality methods by ROUGE-1
    paper               reproduce Table 1 and Table 2 from the paper's Figure 1
```

### `summarize` — your own files

Give it one file per document. The cross-document overlap is what it ranks on,
so two files beat one.

```console
$ cat > wire.txt <<'EOF'
A two-metre storm surge came over the harbour wall at Aldren Bay on Tuesday morning.
Emergency services evacuated more than four thousand residents from the lower town overnight.
The regional council said seven people were treated for minor injuries.
EOF

$ cat > business.txt <<'EOF'
Insurers expect claims from the Aldren Bay flooding to exceed 300 million euro.
The two-metre surge that came over the harbour wall inundated nine hundred commercial premises.
Shares in the port operator fell eleven percent on Tuesday afternoon.
EOF

$ uv run lexrank summarize wire.txt business.txt -n 2
A two-metre storm surge came over the harbour wall at Aldren Bay on Tuesday morning.
Insurers expect claims from the Aldren Bay flooding to exceed 300 million euro.
```

It picked the two sentences that share the most with the rest — the surge (stated
in both files) and the headline financial figure — and dropped the injury count
and the share price, which appear once each.

#### Why one file per document matters

This is the single most consequential thing to get right, and the reason is not
obvious: **document boundaries do not change the similarity graph at all.**

The graph is built over *sentences*, globally. Whether you pass five files or one
concatenated blob, the tokenizer produces the same sentences in the same order,
so the cosine matrix and the LexRank scores come out bit-identical. Verified on
`harbour-storm`:

| property | 1 concatenated doc vs 5 separate docs |
| --- | --- |
| sentence count | same (36) |
| similarity matrix | **identical** |
| centrality scores | **identical** |
| Position feature | *differs* |
| final combined score | *differs* |

What boundaries change is the **Position feature**, which ramps from 1.0 at the
first sentence of *each* document to 0.0 at its last. Concatenate, and you get a
single ramp across the whole blob, so the opening of the first article outranks
the opening of every other one. Keep them separate, and each document gets its
own ramp, so each contributes a strong candidate.

That produces visibly different summaries from byte-identical text:

```console
$ uv run lexrank summarize wire.txt business.txt -n 2      # two documents
A two-metre storm surge came over the harbour wall at Aldren Bay on Tuesday morning.
Insurers expect claims from the Aldren Bay flooding to exceed 300 million euro.

$ cat wire.txt business.txt | uv run lexrank summarize -n 2   # one document
A two-metre storm surge came over the harbour wall at Aldren Bay on Tuesday morning.
Emergency services evacuated more than four thousand residents from the lower town overnight.
```

Both sentences of the concatenated version come from `wire.txt`; the business
angle is never reached. **So `cat *.txt | lexrank summarize` is not equivalent to
`lexrank summarize *.txt`** — pass the files.

#### How much redundancy do you actually need?

Adding genuinely different documents is what makes the ranking mean anything.
Measured on `harbour-storm`, feeding it the first *k* documents:

| documents | sentences | centrality spread | mean similarity | connected |
| --- | --- | --- | --- | --- |
| 1 | 8 | **1.00×** | 0.0195 | 75% |
| 2 | 15 | 1.77× | 0.0371 | 100% |
| 3 | 22 | 2.51× | 0.0337 | 95% |
| 4 | 29 | 2.87× | 0.0341 | 90% |
| 5 | 36 | **3.53×** | 0.0384 | 92% |

*Centrality spread* is the ratio of the highest score to the lowest. **At one
document it is exactly 1.00× — every sentence scores identically and the ranking
carries no information whatsoever.** The second document is what breaks the tie;
after that, returns accumulate steadily.

Two or three genuinely independent sources is the point where LexRank starts
earning its keep. `lexrank suggest` reports the spread and warns you when it is
degenerate — see [choosing a length](#choosing-a-length).

#### Chunking one document does not fake it

A tempting workaround is to split a long article into chunks and pass those as
separate documents. It does not work, and now you can see why: chunking changes
only the Position ramp, never the graph. Splitting `wire.txt` into three chunks
leaves the spread at **1.00×** and the mean similarity at **0.0195** — exactly
the single-document numbers. There is no new redundancy to find, because
redundancy comes from *independent accounts of the same events*, not from
rearranging one account.

### `summarize` — stdin

Piping works and is the right tool for a quick look at one text:

```console
$ printf 'The council approved the new flood defences on Tuesday after four years of delay.\nCampaigners said the four-year delay had left the lower town exposed to flooding.\nConstruction of the flood defences is due to begin in the autumn.\n' \
    | uv run lexrank summarize -n 1
The council approved the new flood defences on Tuesday after four years of delay.
```

But be clear about what you are getting. **Everything piped in is one document**,
so per the section above there is no cross-document signal, centrality is flat,
and the Position feature decides the outcome — which for a single document means
you have reimplemented the lead baseline. Here it returned sentence 1 of 3.

That is a legitimate use (a lead baseline on news is genuinely hard to beat, as
[the evaluation table](#what-the-corpus-shows--and-doesnt) shows), as long as you
are not under the impression that graph centrality is doing the work. Check with:

```console
$ cat article.txt | uv run lexrank suggest
  ! centrality is nearly uniform (spread 1.00x) — LexRank cannot rank this
    input; selection is driven by the Position feature alone. Add more documents.
```

Use `summarize file1.txt file2.txt …` or `-d <cluster>` whenever you have
separate sources. Reserve stdin for one-off inspection and for pipelines where
the upstream stage has already merged the text for you.

### `summarize` — a bundled cluster

```console
$ uv run lexrank summarize -d harbour-storm -n 2
Storm Mairead came ashore at Aldren Bay shortly before dawn on Tuesday, driving a
two-metre surge over the harbour wall and flooding the whole of the lower town.
More than four thousand people were evacuated from the port town of Aldren Bay
early on Tuesday as Storm Mairead pushed a two-metre storm surge over the harbour
wall.

$ uv run lexrank summarize -l sk -d povoden-na-vrbnici -n 2
Vyše tisícdvesto ľudí museli v utorok evakuovať z Dolných Lužian po tom, čo sa
rieka Vrbnica vyliala z koryta.
Na primátora Dolných Lužian Mareka Slávika rastie tlak pre nedokončenú
protipovodňovú ochranu.
```

(Output wrapped here for the page; the tool prints one sentence per line.)

### Controlling length

Three mutually exclusive budgets — sentences, words, or UTF-8 bytes:

```console
$ uv run lexrank summarize -d harbour-storm -w 30      # ≤ 30 words
$ uv run lexrank summarize -d harbour-storm -b 665     # ≤ 665 bytes, the DUC budget
```

A candidate too big for the remaining budget is **skipped, not truncated**, so a
summary never ends mid-sentence. At `-b 300` on `harbour-storm`:

| rank | score | size | outcome |
| --- | --- | --- | --- |
| 1 | 2.0000 | 166 B | selected — 134 B left |
| 2 | 1.5504 | 161 B | skipped, does not fit |
| 3 | 1.5027 | 173 B | skipped, does not fit |
| 4 | 1.4179 | 123 B | selected — total 290 B |

### Seeing why a sentence was picked

```console
$ uv run lexrank summarize -d harbour-storm -n 2 --scores
[d1:0 score=1.5504 centrality=0.0363] Storm Mairead came ashore at Aldren Bay…
[d2:0 score=2.0000 centrality=0.0535] More than four thousand people were evacuated…
```

`d1:0` is document `d1`, sentence 0. `centrality` is the raw LexRank score
(sums to 1 across the cluster); `score` is the combined feature value the
selector actually ranks on — min-max normalised centrality plus the Position
feature, so it tops out at 2.0 with the default weights.

### Choosing a method

```console
$ uv run lexrank summarize -d harbour-storm -n 2 --method degree      # cheapest
$ uv run lexrank summarize -d harbour-storm -n 2 --method continuous  # weighted graph
$ uv run lexrank summarize -d harbour-storm -n 2 --method lead        # baseline
```

On a small cluster these frequently agree — see the identical rows in
`evaluate` output. Full option list: `uv run lexrank summarize --help`.

### `datasets` — what's bundled

```console
$ uv run lexrank datasets -l sk
sk  archeologicky-nalez      5 docs, 34 sentences, 2 references
     Mohylové pohrebisko z doby bronzovej pri Vrbnických Sadoch
sk  povoden-na-vrbnici       5 docs, 36 sentences, 2 references
     Povodeň na rieke Vrbnica zaplavila Dolné Lužany
sk  reforma-vysokych-skol    5 docs, 34 sentences, 2 references
     Návrh novej metodiky financovania vysokých škôl
sk  zeleznicny-koridor       8 docs, 196 sentences, 3 references
     Modernizácia koridoru Brezovec – Hlohovany sa predraží na 1,42 miliardy eur
```

Drop `-l sk` for both languages.

### `demo` — summarize and score every cluster

```console
$ uv run lexrank demo -l en -n 2
==============================================================================
en/asteroid-sample: Hesperus probe collects a sample from asteroid 4471 Odris
5 documents, 30 sentences
==============================================================================
The Hesperus spacecraft completed a touch-and-go sampling manoeuvre at the…
Images released on Tuesday from the Hesperus spacecraft show the sampling head…

ROUGE-1 R=0.3931 P=0.5965 F1=0.4739
ROUGE-2 R=0.1988 P=0.3036 F1=0.2403
… (one block per cluster)
```

`demo` scores at whatever budget you pass, so pass `-b 665` if you want numbers
comparable to `evaluate`. See
[Understanding the ROUGE scores](#understanding-the-rouge-scores).

### `evaluate` — compare methods

Fixes the budget at 665 bytes and reports ROUGE-1 recall:

```console
$ uv run lexrank evaluate -l en
cluster                        lexrank  continuous      degree    centroid        lead      random
--------------------------------------------------------------------------------------------------
en/asteroid-sample              0.5376      0.5376      0.5376      0.5780      0.4740      0.5029
en/drought-emergency            0.3048      0.3553      0.3048      0.3553      0.3048      0.2939
en/harbour-storm                0.5269      0.5269      0.4850      0.5269      0.5269      0.4910
en/rate-decision                0.5260      0.6012      0.5260      0.5838      0.5202      0.5954
--------------------------------------------------------------------------------------------------
mean                            0.4738      0.5052      0.4634      0.5110      0.4565      0.4708
```

`random` here is a single seed (`--seed`), not the 25-seed average quoted in
[the dataset section](#what-the-corpus-shows--and-doesnt) — which is why it
looks stronger here than it deserves to.

### `paper` — reproduce Tables 1 and 2

```console
$ uv run lexrank paper
ID       deg 0.1   deg 0.2   deg 0.3
d1s1          5         4         2
d2s1          7         4         2
d2s2          2         1         1
…
ID                LR 0.1            LR 0.2            LR 0.3
d1s1     0.6005/0.6007   0.8033/0.6944   1.0000/1.0000
…
columns are computed/published; * marks a difference
```

Every value is recomputed from the paper's own Figure 1 matrix. See
[Reproducing the paper](#reproducing-the-paper) for why the 0.2 column differs.

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

## Understanding the ROUGE scores

Every evaluation number in this README is ROUGE, so it is worth knowing exactly
what it does and does not measure. The short version: **ROUGE counts how many
n-grams of a human-written reference summary appear in the machine summary.**
That is all. It is a lexical overlap statistic, not a judgement of quality.

The paper uses it because DUC 2004 did (§4.1), and because unigram ROUGE was the
variant shown to agree most closely with human judgement at the time (Lin & Hovy,
2003).

### The formula

For n-gram order `n`, over a set of reference summaries:

```
                 Σ_refs  Σ_grams  Count_match(gram)
ROUGE-N recall = ──────────────────────────────────
                 Σ_refs  Σ_grams  Count(gram)
```

The denominator is every n-gram in every reference. The numerator counts how many
of them the candidate also has, **clipped** by how often the candidate actually
has them — you cannot earn credit for the word *the* five times if the reference
only says it twice.

### Worked by hand

```python
rouge_1("the river flooded the town", ["the river flooded the village"])
# R=0.8000 P=0.8000 F1=0.8000
```

| | the | river | flooded | town | village | total |
| --- | --- | --- | --- | --- | --- | --- |
| candidate | 2 | 1 | 1 | 1 | 0 | **5** |
| reference | 2 | 1 | 1 | 0 | 1 | **5** |
| matched (clipped) | 2 | 1 | 1 | 0 | 0 | **4** |

Recall is `4/5 = 0.8` — four of the reference's five unigrams were reproduced.
Precision is `4/5 = 0.8` — four of the candidate's five unigrams were wanted. F1
is their harmonic mean, here also 0.8.

The same pair scores `0.75` under ROUGE-2, because the reference's four bigrams
are `the river`, `river flooded`, `flooded the`, `the village`, and the candidate
reproduces the first three.

Clipping runs both ways:

```python
rouge_1("river", ["river river river"])        # R=0.3333 P=1.0000
rouge_1("river river river", ["river"])        # R=1.0000 P=0.3333
```

### Multiple references

References are **pooled**, not averaged: their n-grams all go into one
denominator. This has a consequence that surprises people the first time:

```python
rouge_1("alpha beta", ["alpha beta"])                    # R=1.0000
rouge_1("alpha beta", ["alpha beta", "gamma delta"])     # R=0.5000
```

Adding a second reference the candidate does not match **halves** the score. This
is correct — with two humans disagreeing about what mattered, reproducing one of
them covers half the reference material — but it means **ROUGE numbers are only
comparable across summaries scored against the same reference set.** The bundled
small clusters carry two references each and the large ones carry three, which is
one of several reasons their scores are not directly comparable.

### Recall, precision, or F1?

The paper reports **recall at a fixed length budget**, and so does
`lexrank evaluate`. The reason is visible the moment you sweep the budget on one
cluster (`en/harbour-storm`, mean reference 542 bytes):

| budget | sentences | actual | recall | precision | F1 |
| --- | --- | --- | --- | --- | --- |
| 300 B | 2 | 290 B | 0.3653 | 0.5980 | 0.4535 |
| 665 B | 5 | 655 B | 0.5269 | 0.3860 | 0.4456 |
| 1200 B | 9 | 1170 B | 0.6886 | 0.2995 | 0.4174 |
| 2500 B | 20 | 2484 B | **0.8443** | 0.1762 | 0.2916 |

Recall climbs monotonically toward 1.0 simply by writing more, so **a recall
figure without a length budget is meaningless.** That is exactly why DUC fixed
665 bytes and why the paper reports at that length. Precision falls as the
summary grows past the reference length; F1 peaks somewhere in the middle and its
peak moves with the reference length, which makes it awkward to compare across
clusters.

The rule this package follows: **fix the budget, report recall, and never compare
two numbers produced at different budgets.**

### Reading the demo output

`lexrank demo -l en -n 2` prints the summary, then both scores:

```
==============================================================================
en/asteroid-sample: Hesperus probe collects a sample from asteroid 4471 Odris
5 documents, 30 sentences                    <- input size
==============================================================================
The Hesperus spacecraft completed a touch-and-go sampling manoeuvre at the
asteroid 4471 Odris on Sunday, collecting an estimated 240 grams of surface
material.
Images released on Tuesday from the Hesperus spacecraft show the sampling head
sinking into the surface of asteroid 4471 Odris, confirming that the body is a
loosely bound rubble pile rather than solid rock.

ROUGE-1 R=0.3931 P=0.5965 F1=0.4739         <- unigrams
ROUGE-2 R=0.1988 P=0.3036 F1=0.2403         <- bigrams
```

Read it as: the two selected sentences reproduce **39 percent of the unigrams**
in the pooled reference summaries, and **60 percent of what they say is material
the references also wanted**. Precision exceeds recall here because `-n 2` is far
shorter than the references — the summary is dense but partial. Ask for more
sentences and the two numbers cross over.

ROUGE-2 is always much lower than ROUGE-1 and that is normal, not a fault. Two
summaries can share every word and no word *pair*: matching a bigram requires
getting the phrasing right, which extractive summarization only does when it
lifts the reference's own phrasing wholesale. Use ROUGE-2 as a fluency and
phrasing signal, ROUGE-1 as a content signal.

Note also that `demo` uses whatever budget you pass it (`-n`, `-w`, `-b`),
so its numbers are **not** comparable to the evaluation table below unless you
pass `-b 665`.

### Reading the evaluation table

`lexrank evaluate` fixes the budget at 665 bytes for every cluster and method,
and prints **ROUGE-1 recall only**:

```
cluster                        lexrank  continuous      degree    centroid        lead      random
sk/archeologicky-nalez          0.4626      0.4626      0.4626      0.4626      0.4354      0.4218
sk/povoden-na-vrbnici           0.5224      0.5149      0.5149      0.5896      0.5896      0.3955
sk/reforma-vysokych-skol        0.3767      0.3767      0.3767      0.3767      0.3767      0.4041
sk/zeleznicny-koridor           0.3178      0.3178      0.3178      0.3209      0.2492      0.2492
mean                            0.4199      0.4180      0.4180      0.4374      0.4127      0.3677
```

Identical values across methods in a row are real, not a bug: on a small cluster
several centrality measures often select the same sentences, and the MEAD
Position feature plus the byte budget make that more likely still. Compare
**down** a column across clusters only with care, and **across** a row freely —
the row holds the cluster, the references and the budget fixed, which is the only
controlled comparison in the table.

### What ROUGE cannot see

**Word order.** ROUGE-1 is a bag of words. Shuffling a summary leaves it
untouched:

```python
s = "the reservoir fell to nineteen percent of capacity on monday"
rouge_1("reservoir percent of the monday nineteen capacity fell on to", [s])
# R=1.0000  <- a perfect score for word salad
rouge_2("reservoir percent of the monday nineteen capacity fell on to", [s])
# R=0.1111  <- ROUGE-2 does notice
```

**Facts.** Reversing the meaning of a sentence barely moves the score:

```python
true_  = "Brackmere Reservoir fell to 19 percent of capacity, its lowest level since 1976."
false_ = "Brackmere Reservoir rose to 91 percent of capacity, its highest level since 1976."
rouge_1(false_, [true_])   # R=0.7692
```

Three words changed, the claim inverted, and 77 percent of the score survives.
ROUGE is not a factuality metric and must never be used as one — which matters
much more when scoring an abstractive model than this package, whose output is
verbatim by construction.

**Paraphrase and inflection.** Matching is on **surface forms**, following the
original ROUGE default — no stemming, no stopword removal. `flooded` and
`flooding` are different tokens. For English that costs almost nothing; for a
heavily inflected language it costs real points. Measured on the two large
clusters, comparing the shipped surface-form score against the same summary
scored on stems:

| cluster | surface recall | stem-matched recall | gain |
| --- | --- | --- | --- |
| `en/drought-emergency` | 0.3048 | 0.3052 | **+0.0004** |
| `sk/zeleznicny-koridor` | 0.3178 | 0.3710 | **+0.0533** |

**ROUGE-1 systematically understates Slovak by around five points** relative to
English, purely because Slovak case endings make `koridoru` and `koridore` and
`koridorom` three different tokens where English would have written *corridor*
three times. Keep that in mind before reading anything into the English-versus-
Slovak gap in the tables above; the summarizer's *own* matching is stem-based, so
this is a measurement artefact, not a pipeline weakness.

**Whether the summary is readable.** Nothing in ROUGE penalises a dangling
pronoun, a duplicated fact, or an orphaned quotation. The
`“You do not decide to lose a field,” he said.` sentence discussed under
[demo datasets](#a-note-on-long-form-text) *raises* ROUGE while making the
summary worse.

### Using it on your own data

```python
from lexrank import rouge_1, rouge_2, rouge_n
from lexrank.rouge import truncate_to_bytes

rouge_1(summary, [human_a, human_b], language="sk")   # ROUGE-1
rouge_2(summary, [human_a, human_b], language="sk")   # ROUGE-2
rouge_n(summary, refs, n=3, language="en")            # any order
rouge_1(summary, refs, max_bytes=665)                 # truncate first, DUC style
```

`max_bytes` truncates the candidate before scoring, on a character boundary, so
multi-byte text is never cut mid-character:

```python
truncate_to_bytes("Modernizácia koridoru", 13)   # 'Modernizácia'
```

This implementation covers **ROUGE-N only** — the recall, precision and F1 of
n-gram overlap, with clipped match counts and pooled references. It does not
implement ROUGE-L (longest common subsequence), ROUGE-W or ROUGE-SU, and it does
not replicate the original Perl script's stemming or stopword options. It is
enough to reproduce the paper's evaluation methodology and to compare methods
against each other; it is not a drop-in for a published ROUGE score.

---

## Choosing a length

The paper never addresses this: it fixes 665 bytes because DUC 2004 did. So what
do you pass when you genuinely do not know how long the summary should be?

**The short answer is that a fixed budget is hard to beat, and I tried.** This
section reports the measurements, then describes the helper that ships — which
does something more modest than picking an optimal length, because nothing I
tested could.

### What was measured

For each bundled cluster I computed ROUGE-1 **F1** at every possible sentence
count and found the true optimum. F1 rather than recall, because recall rises
monotonically with length and so cannot tell you when to stop, while F1 has a
genuine peak. Then I scored candidate heuristics by how much F1 they give up
against that per-cluster optimum:

| rule | mean F1 loss | worst |
| --- | --- | --- |
| **fixed 665-byte budget** | **0.031** | **0.067** |
| centroid coverage ≥ 0.50 | 0.084 | — |
| knee of the coverage curve | 0.091 | — |
| centroid coverage ≥ 0.66 | 0.130 | — |
| redundancy saturation | 0.211 | — |

The dumbest rule won, and not narrowly. Adding the content-derived caps on top
of the byte budget changed the result by **exactly nothing** — `min(fill, knee,
saturation)` also scores 0.031, because the budget was the binding constraint on
every single cluster.

### Why the simple rule wins

Because the F1-optimal length is a fact about the **reference**, not about the
input. Across the eight clusters the optimum varies:

| measured at the F1 optimum | range | relative spread |
| --- | --- | --- |
| sentences | 3 – 11 | — |
| compression ratio (summary/source bytes) | 0.028 – 0.400 | **0.63** |
| centroid coverage | 0.149 – 0.617 | 0.39 |
| **absolute bytes** | **517 – 1320** | **0.35** |

Absolute length is the *most* stable quantity and compression ratio is by far the
least. So the intuitive rule — "take 10% of the sentences" — is the worst
available: it varies fourteen-fold across this corpus, because a 208-sentence
cluster is mostly restatement and needs proportionally far less than a
30-sentence one.

Which vindicates the paper's arbitrary-looking constant. **How long a summary
should be is mostly a fact about the reader's patience, not about how much input
there is.** Pick the length your UI, your reader, or your downstream model wants,
and pass it.

### The helper

Given all that, `suggest_length` does not try to out-guess a budget. It does
three things that are actually useful:

1. **Converts a budget into the unit you think in**, using your data's own
   sentence lengths — "665 bytes" becomes "5 sentences *for this cluster*".
2. **Reports the ceiling** — how many mutually non-redundant sentences exist at
   all, so you never pad a summary with restatement.
3. **Warns when the input cannot be ranked**, which is the failure that actually
   costs you something.

```python
from lexrank import LexRankSummarizer
from lexrank.datasets import build_idf, load_cluster

cluster = load_cluster("en", "harbour-storm")
summarizer = LexRankSummarizer("en", idf=build_idf("en"))
print(summarizer.suggest_length(cluster.documents))
```

There is also a one-call `suggest_length(documents, "en")`, but note it derives
IDF from the input alone, while the CLI uses the bundled background corpus — so
the two report slightly different figures for the same cluster. The output below
is the CLI's.

```console
$ uv run lexrank suggest -d harbour-storm
5 sentences (~118 words, ~702 bytes) of 36, bound by 665-byte budget
  coverage 36% of centroid mass; 33 non-redundant sentences available; diminishing returns at 16
  ! 5 sentences comes to 702 bytes, over the 665-byte target: a sentence-count
    budget takes the top N whole, while a byte budget skips oversized
    candidates. Use max_bytes=665 if the limit is hard.
```

That last warning is the honest part: converting a byte budget into a sentence
count cannot be exact, because `-n 5` takes five whole sentences while `-b 665`
skips any that do not fit. If the limit is hard, use the byte budget directly.

The fields are all inspectable:

```python
s = summarizer.suggest_length(cluster.documents)
s.sentences, s.words, s.bytes      # the recommendation, three ways
s.total_sentences, s.non_redundant # input size, and the useful ceiling
s.coverage                         # centroid mass the recommendation covers
s.diminishing_returns              # knee of the coverage curve
s.discriminates                    # False => LexRank can't rank this input
s.bound_by, s.notes                # which constraint bound it, and caveats
```

Use it as the budget with `sentences=None`, or `--auto` on the CLI:

```python
summarize(cluster.documents, "en", sentences=None)
```

```bash
uv run lexrank summarize -d harbour-storm --auto
```

### The warning that earns its keep

On a single document the recommendation is unremarkable but the notes are not:

```console
5 sentences (~96 words, ~598 bytes) of 8, bound by 665-byte budget
  coverage 68% of centroid mass; 8 non-redundant sentences available; diminishing returns at 4
  ! centrality is nearly uniform (spread 1.00x) — LexRank cannot rank this
    input; selection is driven by the Position feature alone. Add more documents.
  ! only 8 sentences — too few for centrality to mean much
```

`spread 1.00x` means every sentence scored *identically* — the uniform-graph
degeneracy described under
[strengths and weaknesses](#strengths-and-weaknesses). The summary you get back
is the lead baseline wearing a LexRank costume. Nothing else in the pipeline
tells you this, and it is the single most useful thing the helper reports.

### When content signals do matter

Not for readable summaries, but for **pre-filtering**, where the goal is to
retain material rather than to be brief. There, "how much do I keep?" genuinely
is a content question, and `target_coverage` answers it:

```console
$ uv run lexrank suggest -l sk -d zeleznicny-koridor --target-coverage 0.5
38 sentences (~781 words, ~5682 bytes) of 196, bound by coverage >= 50%
  coverage 51% of centroid mass; 144 non-redundant sentences available; diminishing returns at 48
```

38 of 196 sentences carry half the cluster's centroid mass — a 5× reduction
before anything downstream sees it. That is the number worth having for the
[LLM pre-filter pattern](#where-they-combine); a byte budget is the wrong tool
for it.

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

Eight clusters, four per language, shaped like DUC Task 2 inputs: several
documents covering one event from different angles, plus reference summaries for
ROUGE. The cross-document redundancy is the signal LexRank exploits; the
outlet-specific detail is the noise it has to ignore.

| Language | Cluster | Docs | Sentences | Subject |
| --- | --- | --- | --- | --- |
| en | `harbour-storm` | 5 | 36 | A storm surge floods a port town |
| en | `rate-decision` | 5 | 34 | A central bank raises its benchmark rate |
| en | `asteroid-sample` | 5 | 30 | A probe samples an asteroid |
| en | **`drought-emergency`** | **8** | **208** | Drought, a water regulator investigation, and fifty years of unbuilt infrastructure |
| sk | `povoden-na-vrbnici` | 5 | 36 | Povodeň zaplavila dolnú časť mesta |
| sk | `reforma-vysokych-skol` | 5 | 34 | Nová metodika financovania vysokých škôl |
| sk | `archeologicky-nalez` | 5 | 34 | Mohylové pohrebisko z doby bronzovej |
| sk | **`zeleznicny-koridor`** | **8** | **196** | Modernizácia trate: audit, arbitráž, eurofondy |

The two bold clusters are **full newspaper length** — eight articles of 600–900
words each, written to read like real coverage, with three reference summaries
apiece. They exist because a 36-sentence cluster does not exercise the same
behaviour: at 200 sentences the O(n²) path, the byte budget, the reranker and the
segmenter all start to matter. Each has the multi-angle structure real coverage
has (breaking news, the institution under scrutiny, politics, the affected
sector, expert explainer, human interest, economics, analysis), so the repeated
core facts and the outlet-specific detail are genuinely separable.

Each language also ships a general-language background corpus used to estimate
IDF (`build_idf`), standing in for the "much larger and similar genre data set"
the paper computes idf over.

> **All bundled text is synthetic.** It was written for this package to exercise
> the algorithm. Every event, place, organisation and person in it is invented
> and none of it is reporting on anything real. Each dataset file carries the
> same notice in its `notice` field.

### The reference summaries

Every cluster ships human-style abstractive summaries in
`cluster.reference_summaries`. They exist for one purpose: to be the denominator
of a ROUGE score (see
[Understanding the ROUGE scores](#understanding-the-rouge-scores)). Without
them, `lexrank evaluate` and `lexrank demo` would have nothing to measure
against.

```python
cluster = load_cluster("en", "drought-emergency")
len(cluster.reference_summaries)   # 3
rouge_1(summary.text, cluster.reference_summaries, language="en")
```

They are deliberately **abstractive** — they merge facts that appear in different
sentences of different documents, and they use phrasing that appears nowhere in
the source. That makes them a fair target (a real human summary looks like this)
and an unreachable ceiling for an extractive method, which can only ever return
sentences that already exist. An extractive summarizer will never score 1.0
against them, and should not be expected to.

Two properties matter for interpreting any score computed against them:

| | small clusters | large clusters |
| --- | --- | --- |
| References per cluster | 2 | 3 |
| Mean reference length | 545 bytes | 840 bytes |
| Pooled reference material | ~1,090 bytes | ~2,520 bytes |

Because references are **pooled** into a single ROUGE denominator, more
references means a larger denominator and a lower recall for the same summary.
Because the large clusters also have longer references, a fixed 665-byte budget
covers proportionally less of them. Both effects push the large clusters' scores
down for reasons that have nothing to do with summary quality — which is why the
table below is split by cluster size rather than presented as one average.

### What the corpus shows — and doesn't

ROUGE-1 recall at a 665-byte budget, `random` averaged over 25 seeds
(`lexrank evaluate`). See
[Understanding the ROUGE scores](#understanding-the-rouge-scores) for how to
read these, and in particular why a recall figure is meaningless without the
budget attached:

| method | all 8 | 6 small | 2 large |
| --- | --- | --- | --- |
| centroid | **0.4742** | **0.5196** | **0.3381** |
| continuous | 0.4616 | 0.5033 | 0.3365 |
| lexrank | 0.4468 | 0.4920 | 0.3113 |
| degree | 0.4407 | 0.4838 | 0.3113 |
| lead | 0.4346 | 0.4871 | 0.2770 |
| random | 0.4210 | 0.4768 | 0.2535 |

Three honest observations:

**Centroid beats the graph methods here, which is not what the paper found.**
Eight synthetic clusters cannot settle that disagreement — the paper evaluated on
30 to 50 real DUC clusters against four human summaries each, and reported that
the difference between Centroid and the graph methods was itself not obvious on
DUC 2003. Treat this corpus as a demonstration that the pipeline works, not as
evidence about which centrality measure is better.

**The margin over the baselines widens with cluster size.** On the small clusters
LexRank beats `lead` by 0.005 — noise. On the large ones it beats `lead` by 0.034
and `random` by 0.058. That is the expected direction: the lead of a 7-sentence
wire story more or less *is* the summary, while the lead of a 30-sentence feature
is a scene-setter. Redundancy-based centrality has more to work with, and
position has less.

**Adding the two large clusters moved every number, including for the small
ones.** `build_idf` fits on the background corpus *plus* every cluster, so
growing the corpus changed the IDF model and therefore all the scores. Earlier
revisions of this file quoted centroid at 0.490 on the six small clusters; it is
0.5196 against the current IDF. Term weighting is not a detail.

### A note on long-form text

The default `length_cutoff=9` is the paper's value, tuned on newswire. On
full-length articles it lets through short quotation sentences that are
lexically distinctive but meaningless once extracted — `“You do not decide to
lose a field,” he said.` reached one summary of `drought-emergency` this way,
because the byte budget had room only for something short. Raising the cutoff to
15 removes them and reads better, at a small ROUGE cost (0.3048 → 0.2895 on that
cluster) because the metric rewards the extra tokens regardless of whether they
mean anything:

```bash
lexrank summarize -d drought-emergency -b 665 --length-cutoff 15
```

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

| Sentences | `rank()` | Similarity matrix | |
| --- | --- | --- | --- |
| 36 | 8 ms | 10 KB | `harbour-storm` |
| 144 | 30 ms | 0.2 MB | |
| 196 | 140 ms | 300 KB | `zeleznicny-koridor` |
| 208 | 115 ms | 338 KB | `drought-emergency` |
| 576 | 192 ms | 2.7 MB | |
| 2,304 | 772 ms | 42.5 MB | |

Both time and memory are **O(n²)** in the sentence count — the dense similarity
matrix dominates. Around 10,000 sentences the matrix alone is ~800 MB, which is
the practical single-machine ceiling. Past that, chunk the corpus, raise the
threshold and use a sparse representation, or pre-cluster the documents.

---

## Development

```bash
uv sync --all-extras
uv run pytest          # 302 tests
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
