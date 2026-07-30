"""The ``lexrank`` command line interface."""

from __future__ import annotations

import pytest

from lexrank.cli import main


def run(capsys, *argv: str) -> str:
    assert main(list(argv)) == 0
    return capsys.readouterr().out


def test_summarize_a_bundled_cluster(capsys) -> None:
    output = run(capsys, "summarize", "-l", "en", "-d", "harbour-storm", "-n", "3")
    assert len(output.strip().splitlines()) == 3
    assert "Aldren Bay" in output


def test_summarize_slovak(capsys) -> None:
    output = run(capsys, "summarize", "-l", "sk", "-d", "povoden-na-vrbnici", "-n", "3")
    assert len(output.strip().splitlines()) == 3
    assert "Vrbnica" in output or "Lužian" in output


def test_summarize_files(capsys, tmp_path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text(
        "The river burst its banks on Tuesday and flooded the whole lower town. "
        "Emergency services evacuated four thousand residents during the night.",
        encoding="utf-8",
    )
    second.write_text(
        "Four thousand residents were evacuated after the river flooded the lower town. "
        "Power was lost to eighteen thousand households across the peninsula.",
        encoding="utf-8",
    )
    output = run(capsys, "summarize", str(first), str(second), "-n", "2")
    assert len(output.strip().splitlines()) == 2


def test_summarize_stdin(capsys, monkeypatch) -> None:
    import io

    text = (
        "The river burst its banks on Tuesday morning and flooded the lower town. "
        "Emergency services evacuated four thousand residents during the night. "
        "The water reached its highest level since records began in 1878."
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(text))
    output = run(capsys, "summarize", "-n", "2")
    assert len(output.strip().splitlines()) == 2


def test_summarize_with_empty_stdin_fails(capsys, monkeypatch) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("   "))
    assert main(["summarize"]) == 2


def test_summarize_with_scores(capsys) -> None:
    output = run(
        capsys, "summarize", "-d", "harbour-storm", "-n", "2", "--scores"
    )
    assert "score=" in output and "centrality=" in output


def test_byte_budget_flag(capsys) -> None:
    output = run(capsys, "summarize", "-d", "harbour-storm", "-b", "300")
    assert len(output.strip().encode("utf-8")) <= 320  # newlines instead of spaces


@pytest.mark.parametrize(
    "method", ["lexrank", "continuous", "degree", "centroid", "lead", "random"]
)
def test_every_method_runs(capsys, method: str) -> None:
    output = run(
        capsys, "summarize", "-d", "harbour-storm", "-n", "2", "--method", method
    )
    assert output.strip()


def test_datasets_listing(capsys) -> None:
    output = run(capsys, "datasets")
    assert "harbour-storm" in output
    assert "povoden-na-vrbnici" in output
    assert "references" in output


def test_datasets_listing_filtered(capsys) -> None:
    output = run(capsys, "datasets", "-l", "sk")
    assert "povoden-na-vrbnici" in output
    assert "harbour-storm" not in output


def test_demo(capsys) -> None:
    output = run(capsys, "demo", "-l", "sk", "-n", "3")
    assert "ROUGE-1" in output and "ROUGE-2" in output
    assert "archeologicky-nalez" in output


def test_evaluate(capsys) -> None:
    output = run(capsys, "evaluate", "-l", "en")
    assert "lexrank" in output and "centroid" in output
    assert "mean" in output


def test_paper_reproduction_command(capsys) -> None:
    output = run(capsys, "paper")
    assert "d4s1" in output
    assert "1.0000/1.0000" in output
    # Thresholds 0.1 and 0.3 must reproduce Table 1 exactly, so the only
    # asterisks in the degree block come from the 0.2 column.
    assert output.count("*") >= 3


def test_no_command_is_an_error() -> None:
    with pytest.raises(SystemExit):
        main([])


# -- suggest / --auto ------------------------------------------------------


def test_suggest_a_bundled_cluster(capsys) -> None:
    output = run(capsys, "suggest", "-d", "harbour-storm")
    assert "sentences" in output and "coverage" in output
    assert "non-redundant" in output


def test_suggest_slovak(capsys) -> None:
    output = run(capsys, "suggest", "-l", "sk", "-d", "zeleznicny-koridor")
    assert "sentences" in output


def test_suggest_with_coverage_target(capsys) -> None:
    output = run(capsys, "suggest", "-d", "harbour-storm", "--target-coverage", "0.8")
    assert "coverage >= 80%" in output


def test_suggest_warns_on_unrankable_input(capsys, tmp_path) -> None:
    path = tmp_path / "one.txt"
    path.write_text(
        "The river burst its banks on Tuesday morning and flooded the lower town. "
        "Emergency services evacuated four thousand residents during the night. "
        "The water reached its highest level since records began in 1878.",
        encoding="utf-8",
    )
    output = run(capsys, "suggest", str(path))
    assert "cannot rank" in output


def test_suggest_empty_stdin_fails(monkeypatch) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("   "))
    assert main(["suggest"]) == 2


def test_summarize_auto(capsys) -> None:
    auto = run(capsys, "summarize", "-d", "harbour-storm", "--auto")
    from lexrank import suggest_length
    from lexrank.datasets import load_cluster

    expected = suggest_length(
        load_cluster("en", "harbour-storm").documents, "en"
    ).sentences
    assert len(auto.strip().splitlines()) == expected


def test_auto_is_mutually_exclusive_with_other_budgets() -> None:
    with pytest.raises(SystemExit):
        main(["summarize", "-d", "harbour-storm", "--auto", "-n", "3"])
