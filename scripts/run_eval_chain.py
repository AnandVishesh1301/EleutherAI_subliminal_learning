"""CLI: run a `&&` chain of `scripts.evaluate_student_preference` calls in ONE process.

Queue jobs chain dozens of `uv run python -m scripts.evaluate_student_preference ...`
invocations; each pays ~6 s of interpreter + torch/transformers import even when every
(seed, set) is already cached. This runner parses the chain into per-segment argv lists
and invokes the script's typer command in-process, in order, so imports are paid once.

Reused across segments: the interpreter and imported modules only. Every segment runs the
unmodified `main` (configs, tokenizer, adapters and base-control models are loaded exactly
as a standalone call would), so outputs, cache skips and provenance are identical; no
model object is shared, hence no adapter leakage. The CUDA cache is emptied after each
segment.

Failure semantics differ from `&&`: ALL segments run even if one fails; each failure is
logged with its argv and the runner exits 1 at the end if any failed.
"""

from __future__ import annotations

import gc
import shlex
import time
from pathlib import Path

import typer
from loguru import logger

EVAL_MODULE = "scripts.evaluate_student_preference"


def parse_chain(chain: str) -> list[list[str]]:
    """Split a `&&` chain into the script argv of each segment (after `-m <module>`).

    Tokenizes with shell quoting, so `&&` inside quotes is not a separator. Each segment
    must be `[uv run] python -m scripts.evaluate_student_preference ...`; anything else
    (another module, a bare command, `;`/`||`/pipes) raises ValueError.
    """
    lexer = shlex.shlex(chain, posix=True, punctuation_chars="&|;<>")
    lexer.whitespace_split = True
    segments: list[list[str]] = [[]]
    for tok in lexer:
        if tok == "&&":
            segments.append([])
        elif tok and set(tok) <= set("&|;<>"):
            raise ValueError(f"unsupported shell operator {tok!r} in chain")
        else:
            segments[-1].append(tok)
    out: list[list[str]] = []
    for seg in segments:
        if not seg:
            if len(segments) == 1:
                continue
            raise ValueError("empty segment in chain")
        rest = seg[2:] if seg[:2] == ["uv", "run"] else seg
        if len(rest) < 3 or Path(rest[0]).name not in ("python", "python3") or rest[1] != "-m":
            raise ValueError(f"segment is not `python -m {EVAL_MODULE}`: {shlex.join(seg)}")
        if rest[2] != EVAL_MODULE:
            raise ValueError(f"refusing foreign module {rest[2]!r}: {shlex.join(seg)}")
        out.append(rest[3:])
    return out


def _eval_command():
    """The click command `typer.run(main)` would build for the eval script."""
    from scripts import evaluate_student_preference as esp

    app = typer.Typer(add_completion=False)
    app.command()(esp.main)
    return typer.main.get_command(app)


def _clear_cuda() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_segment(command, argv: list[str]) -> int:
    """Invoke the eval command with `argv`; return the exit code a standalone call gives."""
    try:
        # standalone_mode=False: click returns an Exit's code instead of raising it.
        rc = command.main(args=list(argv), prog_name=EVAL_MODULE, standalone_mode=False)
        return rc if isinstance(rc, int) else 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        # Usage errors (typer vendors its own click, so match by shape): print + rc 2,
        # exactly like the standalone CLI.
        if hasattr(e, "show") and isinstance(getattr(e, "exit_code", None), int):
            e.show()
            return e.exit_code
        logger.exception("segment raised")
        return 1
    finally:
        _clear_cuda()


def run_chain(segments: list[list[str]], command=None) -> int:
    """Run every segment in order; return 1 if any failed (after running all), else 0."""
    command = command if command is not None else _eval_command()
    failures: list[tuple[int, int, list[str]]] = []
    for i, argv in enumerate(segments, 1):
        t0 = time.monotonic()
        rc = run_segment(command, argv)
        logger.info(
            "chain segment {}/{} rc={} in {:.2f}s", i, len(segments), rc, time.monotonic() - t0
        )
        if rc != 0:
            logger.error("chain segment {}/{} FAILED rc={}: {}", i, len(segments), rc, shlex.join(argv))
            failures.append((i, rc, argv))
    if failures:
        logger.error("{}/{} segment(s) failed:", len(failures), len(segments))
        for i, rc, argv in failures:
            logger.error("  #{} rc={}: {}", i, rc, shlex.join(argv))
        return 1
    logger.info("all {} segment(s) succeeded", len(segments))
    return 0


def main(
    chain_file: Path = typer.Option(None, help="File holding the `&&` chain (newlines allowed)."),
    cmd: str = typer.Option(None, help="The `&&` chain string itself."),
) -> None:
    """Run a chain of evaluate_student_preference calls in-process (see module docstring)."""
    if (chain_file is None) == (cmd is None):
        raise typer.BadParameter("pass exactly one of --chain-file / --cmd")
    chain = chain_file.read_text() if chain_file is not None else cmd
    try:
        segments = parse_chain(chain)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    if not segments:
        raise typer.BadParameter("chain has no segments")
    raise typer.Exit(run_chain(segments))


if __name__ == "__main__":
    typer.run(main)
