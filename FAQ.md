# LexRank — Frequently Asked Questions

**This page is for people who have not read the research paper and do not want
to.** No maths, no jargon that isn't explained. If you want the technical
detail, the [README](README.md) has all of it.

---

## Contents

- [The basics](#the-basics)
- [How it actually works](#how-it-actually-works)
- [Getting started](#getting-started)
- [Using it on your own text](#using-it-on-your-own-text)
- [When to use it, and when not to](#when-to-use-it-and-when-not-to)
- [LexRank vs ChatGPT-style AI](#lexrank-vs-chatgpt-style-ai)
- [Understanding what you get back](#understanding-what-you-get-back)
- [When things look wrong](#when-things-look-wrong)
- [Jargon decoder](#jargon-decoder)

---

## The basics

### What is LexRank, in one sentence?

It reads a pile of text and picks out the handful of sentences that best
represent the whole thing.

### What exactly does it give me back?

**Sentences copied word-for-word from your text.** It never writes anything new.
Think of it as an automatic highlighter pen: it marks the important sentences,
it does not rewrite them.

This is the single most important thing to understand about it, and everything
else follows from it — the good and the bad.

### Is this AI? Does it use a language model?

No. There is no neural network, no model file, no training. It is a piece of
1970s-style word counting plus a piece of maths borrowed from early Google.

Practically, that means:

- **No internet connection needed.** Your text never leaves your computer.
- **No API key, no account, no bill.**
- **No GPU.** It runs fine on a laptop, or a Raspberry Pi.
- **Fast.** A five-article news cluster is summarized in about 8 milliseconds.
- **Same input always gives the same output.** Forever.

### Where does it come from?

A 2004 research paper by Güneş Erkan and Dragomir R. Radev at the University of
Michigan. It won first place in a summarization competition that year and is
still one of the standard reference methods. The paper is included in this
repository as `1109.2128.pdf` if you ever get curious.

This project is a from-scratch implementation of that paper, checked against the
numbers the authors published.

### Can it make things up?

**No — and not "usually no", but structurally no.** Every sentence it returns is
a sentence that was already in your text, character for character. A highlighter
cannot invent a sentence.

That is a genuinely rare property, and it is the main reason people still use
this in 2026 instead of an AI model. If you work somewhere that a fabricated
quote would be a serious problem — law, medicine, journalism, compliance,
regulatory filings — this matters a lot.

The honest caveat: it can still *mislead* by taking a sentence out of context.
It cannot invent, but it can decontextualise.

---

## How it actually works

### The short version

Imagine you gave five reporters the same event and each wrote an article. Now you
want the one-paragraph version.

The insight behind LexRank is: **the things that matter are the things that keep
coming up.** If all five reporters mention the burst water main, that is the
story. If only one mentions that buses were diverted, that is colour.

So the algorithm looks at every sentence and asks: *how much does this sentence
echo the other sentences?* The ones with the most echo win.

### Why is it called LexRank? What has it got to do with Google?

Google's original breakthrough, PageRank, ranked web pages by *who links to
them* — and crucially, a link from an important page counted for more than a link
from an unimportant one.

LexRank is the same idea with sentences instead of web pages. A sentence is
important if it is similar to lots of *other important* sentences. "Lex" is from
*lexical*, meaning "to do with words". Same maths, different objects.

### But that sounds circular — important sentences are the ones similar to important sentences?

It is circular, deliberately, and there is a neat way to resolve it.

Picture a reader who does this: start on a random sentence. Now jump to another
sentence that says something similar. Now jump again. Keep going, thousands of
times.

Some sentences will get landed on far more often than others — because lots of
paths lead to them. Those are the central ones. That is literally what the
algorithm computes.

(Every so often our reader gets bored and jumps to a completely random sentence
instead. That is a real part of the method — it stops the reader getting
permanently stuck in one corner of a long document.)

### What does "similar" mean to a computer here?

Two sentences are similar if they share unusual words.

- Sharing **"the"** or **"and"** counts for nothing — every sentence has those.
- Sharing **"reservoir"** or **"arbitration"** counts for a lot.

The rarer a word is, the more weight it carries. This is why it works without
understanding language at all: it never knows what a reservoir *is*, only that
two sentences both mention one and hardly anything else does.

Words are also reduced to their stems, so *flood*, *floods*, *flooded* and
*flooding* all count as the same word.

### Show me. What does it actually do?

Three short reports about the same small event:

**`report1.txt`**
```
A burst water main flooded Market Street early on Monday morning.
Three shops were forced to close for the day.
The water company said repairs would take about eight hours.
```

**`report2.txt`**
```
Shopkeepers on Market Street are counting the cost after a burst water main
flooded the road on Monday.
The water company has apologised to residents for the disruption.
Repairs to the main are expected to take eight hours.
```

**`report3.txt`**
```
Monday's flooding on Market Street was caused by a burst water main, the water
company confirmed.
Buses were diverted away from the town centre for most of the morning.
```

Ask for the single most representative sentence:

```console
$ uv run lexrank summarize report1.txt report2.txt report3.txt -n 1
Monday's flooding on Market Street was caused by a burst water main, the water
company confirmed.
```

Here is the full ranking it computed, most central first:

| score | sentence |
| --- | --- |
| **0.176** | Monday's flooding on Market Street was caused by a burst water main, the water company confirmed. |
| 0.150 | A burst water main flooded Market Street early on Monday morning. |
| 0.150 | The water company said repairs would take about eight hours. |
| 0.125 | Three shops were forced to close for the day. |
| 0.112 | The water company has apologised to residents for the disruption. |
| 0.112 | Shopkeepers on Market Street are counting the cost after a burst water main flooded the road on Monday. |
| 0.088 | Repairs to the main are expected to take eight hours. |
| **0.088** | Buses were diverted away from the town centre for most of the morning. |

Look at the top and the bottom. The winner mentions the burst water main, Market
Street, Monday *and* the water company — every one of which appears in all three
reports. The loser mentions diverted buses, which nobody else brought up.

Nobody told it that flooding matters more than buses. It only counted echoes.

### Why didn't it pick the two most central sentences when I asked for two?

Good spot. Watch:

```console
$ uv run lexrank summarize report1.txt report2.txt report3.txt -n 2 --no-reranker
A burst water main flooded Market Street early on Monday morning.
Monday's flooding on Market Street was caused by a burst water main, the water
company confirmed.
```

Those are the top two by score — and they say almost exactly the same thing.
Useless as a summary.

So by default there is a second stage that skips a sentence if it is too similar
to one already chosen:

```console
$ uv run lexrank summarize report1.txt report2.txt report3.txt -n 2
Three shops were forced to close for the day.
Monday's flooding on Market Street was caused by a burst water main, the water
company confirmed.
```

Now you get the main fact *and* a different fact. The most central sentences are
often near-duplicates of each other, precisely because being echoed is what makes
them central — so this filter is doing necessary work.

---

## Getting started

### How do I install it?

You need [uv](https://docs.astral.sh/uv/) (a Python tool installer). Then:

```bash
git clone <this repository>
cd LexRank
uv sync
```

That is it. The only thing it downloads is numpy, a maths library.

### What is the very first thing I should run?

```bash
uv run lexrank demo -l en
```

That summarizes the four example English news collections that ship with the
project and shows you the results. Swap `-l en` for `-l sk` for the Slovak ones.

### How do I see what examples are included?

```console
$ uv run lexrank datasets
en  asteroid-sample          5 docs, 30 sentences, 2 references
     Hesperus probe collects a sample from asteroid 4471 Odris
en  drought-emergency        8 docs, 208 sentences, 3 references
     Drought empties Brackmere Reservoir and puts Ardwell Water under investigation
…
```

Eight collections in total, four English and four Slovak. All of the text is
**made up** — invented events, invented towns, invented people — written for this
project so it can be shipped freely. It is not real news.

### What are the commands, in short?

| command | what it does |
| --- | --- |
| `lexrank summarize` | the main one — give it text, get sentences back |
| `lexrank suggest` | tells you how long a summary to ask for, and warns you about bad input |
| `lexrank datasets` | lists the built-in examples |
| `lexrank demo` | summarizes every built-in example and scores it |
| `lexrank evaluate` | compares the different ranking methods |
| `lexrank paper` | reproduces the tables from the original research paper |

Add `--help` to any of them.

---

## Using it on your own text

### How do I summarize my own files?

Put each document in its own plain-text file and pass them all at once:

```bash
uv run lexrank summarize report1.txt report2.txt report3.txt -n 3
```

`-n 3` means "give me three sentences".

### Does it really matter whether I pass separate files?

**Yes, more than you would expect.** These two are *not* the same:

```bash
uv run lexrank summarize a.txt b.txt c.txt        # three documents
cat a.txt b.txt c.txt | uv run lexrank summarize   # one document
```

Same words, different answer. The second version tends to take everything from
the beginning of the first file and never reach the others.

The reason: the algorithm gives extra weight to sentences near the *start of a
document*, because in news writing the opening usually carries the point. If you
glue three articles together, it only sees one beginning — so the other two
articles get treated as the boring middle.

**Rule of thumb: one file per source. Never `cat` them together.**

### Can I just paste text in?

Yes, pipe it:

```bash
cat article.txt | uv run lexrank summarize -n 3
```

Just know that anything piped in counts as **one document**, so you lose the
"what do multiple sources agree on" signal entirely. It still works, but you are
essentially getting "the first few sentences, roughly", which for a news article
is a decent summary but is not the clever part of the algorithm.

### How many documents do I need for this to work well?

More than one. Ideally three or more.

Here is what happens as you add documents about the same event — the "spread"
column measures how much the scores differ between the best and worst sentence:

| documents | spread |
| --- | --- |
| 1 | **1.00×** |
| 2 | 1.77× |
| 3 | 2.51× |
| 5 | **3.53×** |

A spread of **1.00× means every sentence scored exactly the same** — the ranking
is meaningless and you are just getting the opening sentences. The second
document is what breaks the tie.

### Can I split one long article into chunks to fake having several documents?

No, and it is worth knowing why: chunking changes nothing about how sentences are
compared to each other. Splitting one article into three pieces leaves the spread
at exactly 1.00×, same as before.

Redundancy has to come from **independent accounts of the same events**. You
cannot manufacture it by rearranging one account.

### How do I know if my input is any good?

Ask it:

```console
$ cat article.txt | uv run lexrank suggest
3 sentences (~30 words, ~172 bytes) of 3, bound by 665-byte budget
  coverage 100% of centroid mass; 3 non-redundant sentences available
  ! centrality is nearly uniform (spread 1.00x) — LexRank cannot rank this
    input; selection is driven by the Position feature alone. Add more documents.
```

That warning is the useful bit. It is telling you plainly: *this input cannot be
ranked, you are getting the opening sentences, add more sources.*

### How long should the summary be? I have no idea what to ask for.

Honestly? **Pick whatever length your reader wants and pass it.** I tested
several clever ways of deriving the "right" length from the text itself and none
of them beat simply choosing a size. Summary length turns out to be mostly a fact
about the reader's patience, not about the input.

Three ways to say it:

```bash
uv run lexrank summarize *.txt -n 5      # five sentences
uv run lexrank summarize *.txt -w 100    # about 100 words
uv run lexrank summarize *.txt -b 665    # about 665 characters
```

If you genuinely want it to choose, `--auto` uses a sensible default:

```bash
uv run lexrank summarize *.txt --auto
```

### Does it work in languages other than English?

English and Slovak are built in and properly supported — Slovak needed real work,
because words change their endings so much (*Lužany*, *Lužian*, *Lužanmi* all have
to be recognised as the same word).

Adding another language needs three things: a list of common words to ignore
(*the*, *and*, *of*), a list of abbreviations so it does not break sentences at
"Dr.", and ideally a stemmer to strip word endings. The README has an example. No
retraining, no data collection — it is configuration, not machine learning.

---

## When to use it, and when not to

### What is this genuinely good for?

**Several sources covering the same thing.** This is what it was built for and
where it shines:

- News coverage of one event from different outlets
- Customer reviews of one product
- Support tickets about one recurring problem
- Survey free-text responses to one question
- Search results for one query
- Meeting notes from a series of meetings on one project

**Anywhere the output must be verbatim.** Legal discovery, medical records,
regulatory documents — anywhere paraphrasing would be a liability.

**Anywhere the data cannot leave the building.** Air-gapped systems, confidential
material, offline environments.

**Very high volume.** Summarizing millions of documents where per-item cost or
latency matters.

**As a first pass before an AI model.** See [below](#can-i-use-both-together).

### When should I not use it?

- **One short document.** It needs repetition across sources to have anything to
  measure. On a single article, it degenerates into "take the first few
  sentences".
- **You need an actual answer to a question.** It has no idea what you asked. It
  produces one generic summary and that is all it can do.
- **You need it to combine facts.** If the date is in one sentence and the
  location in another, it cannot merge them into one. It picks whole sentences or
  nothing.
- **You need it to read beautifully.** See the next question.
- **Fiction, argument, narrative.** In a story or an essay the key line is often
  said exactly once, and repetition means nothing. The whole premise fails.

### Will the summary read well?

Not really, no. Be realistic about this.

You are getting sentences lifted out of different articles and put next to each
other. So you will see:

- **Dangling references** — "Bekker rejected the calls" when the sentence
  introducing Bekker did not make the cut.
- **Near-duplicates that slipped through** — two sentences saying the same thing
  in different words.
- **Abrupt transitions** — no connecting phrases, because it cannot write any.

It is accurate but not fluent. If fluency matters, either accept it as a bullet
list of key sentences rather than a paragraph, or [pass the output through an AI
model](#can-i-use-both-together) to smooth it out.

---

## LexRank vs ChatGPT-style AI

### Which one is better?

For summary *quality*, an AI model wins, and it is not close. It writes fluent
prose, merges facts, follows instructions, and handles a single document fine.
If quality is all you care about and you can send the data out, use an AI model.

LexRank wins on different things:

| | LexRank | AI model |
| --- | --- | --- |
| Reads well | ✗ | ✓ |
| Can rephrase and merge facts | ✗ | ✓ |
| Can answer a question about the text | ✗ | ✓ |
| **Can it invent something?** | **Never** | Possible |
| **Same answer every time?** | **Always** | Not guaranteed |
| **Works offline** | **Yes** | Only with local models |
| **Data stays on your machine** | **Yes** | No, unless self-hosted |
| Speed | ~8 milliseconds | Seconds to minutes |
| Cost per document | electricity | per-word charge |

### Can I use both together?

Yes, and this is probably the best use of it today.

The trick is to let LexRank do the **selecting** and the AI model do the
**writing**:

1. You have 500 sentences across 30 documents — too much to send, or too
   expensive at scale.
2. LexRank picks the 30 most central sentences, in milliseconds, on your machine.
3. You send only those 30 to the AI model and ask it to write a readable summary.

You cut the cost and the wait proportionally, and because the 30 sentences are
copied verbatim, nothing is distorted on the way in. The model's job shrinks from
"read all this and summarize it" to "make these 30 real sentences read nicely" —
a much easier job to do reliably.

The same trick is useful for removing duplicate chunks before feeding a
search-and-answer system, and for cheaply triaging a large pile of documents
before deciding which deserve the expensive treatment.

The README has working code for this under
[LexRank and LLMs](README.md#lexrank-and-llms).

---

## Understanding what you get back

### What do the numbers mean if I ask for scores?

```console
$ uv run lexrank summarize report1.txt report2.txt report3.txt -n 2 --scores
[report1.txt:1 score=1.0126 centrality=0.1250] Three shops were forced to close…
[report3.txt:0 score=2.0000 centrality=0.1705] Monday's flooding on Market Street…
```

Taking the second line apart:

- `report3.txt:0` — which file it came from, and that it was sentence number 0
  (the first one) in that file.
- `centrality` — how much this sentence echoes the rest. Higher is more central.
  These add up to 1 across all sentences, so on a big pile every individual
  number looks small; only the comparison matters.
- `score` — the final ranking number, which combines centrality with a bonus for
  being near the start of its document. Maximum is 2.0.

### What are ROUGE-1 and ROUGE-2 in the demo output?

A way of scoring a summary against a human-written one, by counting shared words.

- **ROUGE-1** counts single words in common.
- **ROUGE-2** counts pairs of adjacent words in common, so it also notices
  whether the phrasing matches.

They only work if someone has already written a reference summary by hand, which
is why they appear on the built-in examples and not on your own text.

### What counts as a good ROUGE score?

There is no absolute scale, and anyone who quotes one without saying what they
compared against is confusing you. Scores are only meaningful **between two
systems on the same text at the same length**.

Two things worth knowing:

- **Longer summaries always score higher** on the recall version of the metric,
  just by containing more words. So comparing a long summary to a short one is
  meaningless.
- **It cannot tell truth from falsehood.** Change "fell to 19 percent" into "rose
  to 91 percent" and the score barely moves — three words changed, the meaning
  inverted, and 77% of the score survives. It is a word-overlap statistic, not a
  fact checker.

The [README section on ROUGE](README.md#understanding-the-rouge-scores) goes into
this properly.

---

## When things look wrong

### Every sentence got the same score

You gave it one document, or several documents that have nothing in common. Run
`lexrank suggest` on your input — it will tell you if that is what happened.

Fix: add more documents about the *same* subject.

### The summary repeats itself

Two sentences said the same thing in different enough words that the duplicate
filter missed them. Make it stricter:

```bash
uv run lexrank summarize *.txt -n 5 --reranker-threshold 0.3
```

Lower number = more aggressive duplicate removal. The default is 0.5.

### It picked a short quote fragment that makes no sense alone

Something like `"You do not decide to lose a field," he said.` This happens on
long articles: short quotations are distinctive enough to score well but useless
out of context.

Fix — require longer sentences:

```bash
uv run lexrank summarize *.txt -b 665 --length-cutoff 15
```

That ignores any sentence under 15 words. The default is 9.

### It only picked sentences from the beginning of each article

That is the position bonus doing its job — but you can turn it off:

```bash
uv run lexrank summarize *.txt -n 5 --position-weight 0
```

Now it ranks purely on how central the content is. Useful for anything that is
not news, where the opening sentence is not special.

### It is slow, or ran out of memory

The work grows with the *square* of the number of sentences — every sentence gets
compared to every other one. Rough guide:

| sentences | time |
| --- | --- |
| 200 | ~0.1 seconds |
| 2,000 | ~0.8 seconds |
| 10,000 | around the practical limit on a normal machine |

Above that, split the pile into topic groups first and summarize each.

### The sentence splitting looks wrong

It handles the usual traps — `Dr. Smith`, `3.14`, `e.g.`, and in Slovak the date
format `5. mája 1945` — but no rule-based splitter is perfect. If your text has
unusual abbreviations, you can register them; see the README's language section.

---

## Jargon decoder

Terms you will meet in the README or the paper:

| term | plain English |
| --- | --- |
| **Extractive** | Picks existing sentences. (What this does.) |
| **Abstractive** | Writes new sentences. (What AI models do.) |
| **Corpus** | A pile of text. |
| **Cluster** | Several documents about the same thing. |
| **Sentence centrality** | How much a sentence echoes the others. |
| **tf-idf** | Word scoring where rare words count for more. |
| **Cosine similarity** | A 0-to-1 measure of how much two sentences overlap. |
| **Damping factor** | How often the imaginary reader jumps somewhere random. |
| **Power method** | The repeated calculation that settles the scores. |
| **Stemming** | Cutting words to their root so *flooded* matches *flooding*. |
| **Stopwords** | Words too common to be informative — *the*, *and*, *of*. |
| **ROUGE** | A score for how many words a summary shares with a human one. |
| **DUC** | The annual competition the original paper was evaluated in. |
| **Reranker** | The stage that drops a sentence too similar to one already picked. |
| **Baseline** | A deliberately dumb method used for comparison — e.g. "just take the first sentences". |

---

## Where next

- **[README](README.md)** — the full technical documentation, with the maths,
  the measurements, and the code examples.
- **`1109.2128.pdf`** — the original 2004 paper, included in this repository.
- **`uv run lexrank --help`** — every command and option.

If something here is unclear, that is a documentation bug worth reporting.
