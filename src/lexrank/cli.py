"""Command line interface: ``lexrank <command>``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import numpy as np

from .centrality import degree_centrality, lexrank_scores, normalize_max
from .datasets import available_clusters, build_idf, load_cluster, load_paper_example
from .languages import available_languages
from .rouge import rouge_1, rouge_2
from .summarizer import DUC_BYTE_BUDGET, LexRankSummarizer
from .tokenization import split_sentences

METHODS = ("lexrank", "continuous", "degree", "centroid", "lead", "random")


def _print_summary(summary, *, show_scores: bool) -> None:
    ranking = summary.ranking
    for index in summary.indices:
        sentence = ranking.sentences[index]
        if show_scores:
            print(
                f"[{sentence.document_id}:{sentence.index_in_document} "
                f"score={ranking.score[index]:.4f} "
                f"centrality={ranking.centrality[index]:.4f}] {sentence.text}"
            )
        else:
            print(sentence.text)


def _make_summarizer(args: argparse.Namespace, language: str) -> LexRankSummarizer:
    idf = None if args.no_background_idf else build_idf(language)
    return LexRankSummarizer(
        language,
        method=args.method,
        threshold=args.threshold,
        damping=args.damping,
        idf=idf,
        centrality_weight=args.centrality_weight,
        position_weight=args.position_weight,
        length_cutoff=args.length_cutoff,
        reranker_threshold=None if args.no_reranker else args.reranker_threshold,
        seed=args.seed,
    )


def _budget(
    args: argparse.Namespace,
    summarizer: LexRankSummarizer | None = None,
    documents: list | None = None,
) -> dict[str, int | None]:
    if getattr(args, "auto", False) and summarizer is not None:
        suggestion = summarizer.suggest_length(documents or [])
        return {"max_sentences": max(1, suggestion.sentences)}
    if args.bytes is not None:
        return {"max_bytes": args.bytes}
    if args.words is not None:
        return {"max_words": args.words}
    return {"max_sentences": args.sentences}


# -- commands --------------------------------------------------------------


def cmd_summarize(args: argparse.Namespace) -> int:
    if args.dataset:
        cluster = load_cluster(args.language, args.dataset)
        documents: list = list(cluster.documents)
    elif args.files:
        documents = [
            (path, open(path, encoding="utf-8").read()) for path in args.files
        ]
    else:
        text = sys.stdin.read()
        if not text.strip():
            print("no input: pass files, --dataset, or pipe text on stdin", file=sys.stderr)
            return 2
        documents = [("stdin", text)]

    summarizer = _make_summarizer(args, args.language)
    budget = _budget(args, summarizer, documents)
    summary = summarizer.summarize(documents, **budget)  # type: ignore[arg-type]
    _print_summary(summary, show_scores=args.scores)
    return 0


def cmd_suggest(args: argparse.Namespace) -> int:
    """Recommend a summary length for an input, and show the evidence."""
    if args.dataset:
        documents: list = list(load_cluster(args.language, args.dataset).documents)
    elif args.files:
        documents = [(p, open(p, encoding="utf-8").read()) for p in args.files]
    else:
        text = sys.stdin.read()
        if not text.strip():
            print("no input: pass files, --dataset, or pipe text on stdin",
                  file=sys.stderr)
            return 2
        documents = [("stdin", text)]

    summarizer = _make_summarizer(args, args.language)
    print(summarizer.suggest_length(
        documents,
        target_bytes=args.target_bytes,
        target_coverage=args.target_coverage,
    ))
    return 0


def cmd_datasets(args: argparse.Namespace) -> int:
    for language, cluster_id in available_clusters(args.language):
        cluster = load_cluster(language, cluster_id)
        sentences = sum(
            len(split_sentences(document.text, language))
            for document in cluster.documents
        )
        print(
            f"{language}  {cluster_id:<24} {len(cluster)} docs, "
            f"{sentences} sentences, {len(cluster.reference_summaries)} references"
        )
        print(f"     {cluster.title}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """ROUGE-1 recall per method, over the bundled clusters."""
    clusters = available_clusters(args.language)
    header = f"{'cluster':<26}" + "".join(f"{m:>12}" for m in METHODS)
    print(header)
    print("-" * len(header))

    totals: dict[str, list[float]] = {method: [] for method in METHODS}
    for language, cluster_id in clusters:
        cluster = load_cluster(language, cluster_id)
        idf = build_idf(language)
        row = f"{language}/{cluster_id:<23}"
        for method in METHODS:
            summarizer = LexRankSummarizer(
                language,
                method=method,
                threshold=args.threshold,
                damping=args.damping,
                idf=idf,
                seed=args.seed,
            )
            summary = summarizer.summarize(cluster.documents, max_bytes=args.bytes)
            score = rouge_1(
                summary.text,
                cluster.reference_summaries,
                language=language,
                max_bytes=args.bytes,
            ).recall
            totals[method].append(score)
            row += f"{score:>12.4f}"
        print(row)

    print("-" * len(header))
    mean_row = f"{'mean':<26}"
    for method in METHODS:
        values = totals[method]
        mean_row += f"{sum(values) / len(values):>12.4f}" if values else f"{'-':>12}"
    print(mean_row)
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    for language, cluster_id in available_clusters(args.language):
        cluster = load_cluster(language, cluster_id)
        summarizer = _make_summarizer(args, language)
        budget = _budget(args, summarizer, list(cluster.documents))
        summary = summarizer.summarize(cluster.documents, **budget)  # type: ignore[arg-type]

        print("=" * 78)
        print(f"{language}/{cluster_id}: {cluster.title}")
        print(f"{len(cluster)} documents, {len(summary.ranking)} sentences")
        print("=" * 78)
        _print_summary(summary, show_scores=args.scores)
        r1 = rouge_1(summary.text, cluster.reference_summaries, language=language)
        r2 = rouge_2(summary.text, cluster.reference_summaries, language=language)
        print(f"\nROUGE-1 {r1}\nROUGE-2 {r2}\n")
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    """Recompute Table 1 and Table 2 from the paper's own Figure 1 matrix."""
    data = load_paper_example()
    matrix = np.array(data["figure_1_cosine_matrix"], dtype=np.float64)
    ids = data["sentence_ids"]

    print(data["source"])
    print()
    print(f"{'ID':<6}" + "".join(f"{'deg ' + t:>10}" for t in ("0.1", "0.2", "0.3")))
    for i, name in enumerate(ids):
        row = f"{name:<6}"
        for threshold in (0.1, 0.2, 0.3):
            got = degree_centrality(matrix, threshold)[i]
            want = data["table_1_degree"][str(threshold)][i]
            flag = " " if got == want else "*"
            row += f"{int(got):>9}{flag}"
        print(row)

    print()
    print(f"{'ID':<6}" + "".join(f"{'LR ' + t:>18}" for t in ("0.1", "0.2", "0.3")))
    for i, name in enumerate(ids):
        row = f"{name:<6}"
        for threshold in (0.1, 0.2, 0.3):
            got = normalize_max(lexrank_scores(matrix, threshold=threshold))[i]
            want = data["table_2_lexrank"][str(threshold)][i]
            row += f"{got:>9.4f}/{want:.4f}"
        print(row)

    print("\ncolumns are computed/published; * marks a difference")
    for note in data["reproduction_notes"]:
        print(f"\n- {note}")
    return 0


# -- argument parsing ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lexrank",
        description="LexRank extractive summarization (Erkan & Radev, 2004).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_model_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--method", choices=METHODS, default="lexrank")
        sub.add_argument("--threshold", type=float, default=0.1,
                         help="cosine threshold for the similarity graph (default: 0.1)")
        sub.add_argument("--damping", type=float, default=0.85,
                         help="probability of following an edge (default: 0.85)")
        sub.add_argument("--centrality-weight", type=float, default=1.0)
        sub.add_argument("--position-weight", type=float, default=1.0)
        sub.add_argument("--length-cutoff", type=int, default=9,
                         help="discard sentences shorter than this many words")
        sub.add_argument("--reranker-threshold", type=float, default=0.5)
        sub.add_argument("--no-reranker", action="store_true")
        sub.add_argument("--no-background-idf", action="store_true",
                         help="derive idf from the input itself instead of the "
                              "bundled background corpus")
        sub.add_argument("--seed", type=int, default=0)
        sub.add_argument("--scores", action="store_true",
                         help="show per-sentence feature values")

    def add_budget_options(sub: argparse.ArgumentParser) -> None:
        group = sub.add_mutually_exclusive_group()
        group.add_argument("-n", "--sentences", type=int, default=5)
        group.add_argument("-w", "--words", type=int)
        group.add_argument("-b", "--bytes", type=int)
        group.add_argument("--auto", action="store_true",
                           help="let suggest_length pick the sentence count")

    summarize = subparsers.add_parser("summarize", help="summarize files, stdin or a bundled cluster")
    summarize.add_argument("files", nargs="*", help="input files; one cluster")
    summarize.add_argument("-l", "--language", default="en", choices=available_languages())
    summarize.add_argument("-d", "--dataset", help="bundled cluster id instead of files")
    add_model_options(summarize)
    add_budget_options(summarize)
    summarize.set_defaults(func=cmd_summarize)

    suggest = subparsers.add_parser(
        "suggest", help="recommend a summary length for an input"
    )
    suggest.add_argument("files", nargs="*")
    suggest.add_argument("-l", "--language", default="en", choices=available_languages())
    suggest.add_argument("-d", "--dataset", help="bundled cluster id instead of files")
    suggest.add_argument("--target-bytes", type=int, default=DUC_BYTE_BUDGET)
    suggest.add_argument("--target-coverage", type=float,
                         help="recommend for this centroid coverage instead")
    add_model_options(suggest)
    suggest.set_defaults(func=cmd_suggest)

    datasets = subparsers.add_parser("datasets", help="list the bundled demo clusters")
    datasets.add_argument("-l", "--language", choices=available_languages())
    datasets.set_defaults(func=cmd_datasets)

    demo = subparsers.add_parser("demo", help="summarize every bundled cluster and score it")
    demo.add_argument("-l", "--language", choices=available_languages())
    add_model_options(demo)
    add_budget_options(demo)
    demo.set_defaults(func=cmd_demo)

    evaluate = subparsers.add_parser("evaluate", help="compare centrality methods by ROUGE-1")
    evaluate.add_argument("-l", "--language", choices=available_languages())
    evaluate.add_argument("--threshold", type=float, default=0.1)
    evaluate.add_argument("--damping", type=float, default=0.85)
    evaluate.add_argument("-b", "--bytes", type=int, default=DUC_BYTE_BUDGET)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.set_defaults(func=cmd_evaluate)

    paper = subparsers.add_parser(
        "paper", help="reproduce Table 1 and Table 2 from the paper's Figure 1"
    )
    paper.set_defaults(func=cmd_paper)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
