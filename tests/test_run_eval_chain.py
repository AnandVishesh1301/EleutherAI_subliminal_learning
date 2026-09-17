"""Tests for scripts.run_eval_chain (in-process evaluate_student_preference chains)."""

from __future__ import annotations

import pytest
import typer

from scripts import run_eval_chain as rec

P = "uv run python -m scripts.evaluate_student_preference"


def test_parse_chain_splits_and_quotes():
    chain = (
        f"{P} --source teacher --eval-animal cat && "
        f"{P} --number-config 'a b&&c.yaml' --epochs \"1,2\"\n&& python -m {rec.EVAL_MODULE} --force"
    )
    assert rec.parse_chain(chain) == [
        ["--source", "teacher", "--eval-animal", "cat"],
        ["--number-config", "a b&&c.yaml", "--epochs", "1,2"],
        ["--force"],
    ]


@pytest.mark.parametrize(
    "chain",
    [
        f"{P} --a && uv run python -m scripts.train_student --b",
        f"{P} --a && echo hi",
        f"{P} --a ; {P} --b",
        f"{P} --a || {P} --b",
        f"{P} --a && && {P} --b",
    ],
)
def test_parse_chain_refuses_foreign(chain):
    with pytest.raises(ValueError):
        rec.parse_chain(chain)


def _fake_command(calls, fail_on=()):
    def main(source: str = "teacher", eval_animal: str = typer.Option(None)) -> None:
        calls.append((source, eval_animal))
        if eval_animal in fail_on:
            raise typer.Exit(1)
        if eval_animal == "boom":
            raise RuntimeError("boom")

    app = typer.Typer()
    app.command()(main)
    return typer.main.get_command(app)


def test_run_chain_invokes_main_with_parsed_args():
    calls: list = []
    segs = rec.parse_chain(f"{P} --source base --eval-animal cat && {P} --eval-animal pig")
    assert rec.run_chain(segs, command=_fake_command(calls)) == 0
    assert calls == [("base", "cat"), ("teacher", "pig")]


def test_run_chain_continues_after_failures():
    calls: list = []
    segs = [["--eval-animal", "x"], ["--eval-animal", "boom"], ["--bogus"], ["--eval-animal", "ok"]]
    assert rec.run_chain(segs, command=_fake_command(calls, fail_on={"x"})) == 1
    assert calls == [("teacher", "x"), ("teacher", "boom"), ("teacher", "ok")]


def test_run_segment_exit_codes_match_standalone():
    cmd = _fake_command([], fail_on={"x"})
    assert rec.run_segment(cmd, []) == 0
    assert rec.run_segment(cmd, ["--eval-animal", "x"]) == 1
    assert rec.run_segment(cmd, ["--bogus"]) == 2


def test_real_eval_command_builds_and_parses():
    cmd = rec._eval_command()
    names = {p.name for p in cmd.params}
    assert {"source", "eval_animal", "force", "number_config"} <= names
