"""CLI: R2 "masked ≠ removed" readouts (blog §8 R2; docs/PREREG_REVERSAL.md).

prepare      new build-<UTC>/ under the reversal root (identity, quarantine, BPE checks)
score        readouts of shard i of n leaves (one GPU; resumable)
score-bases  readouts of the untuned student bases (reference)
scan         CPU re-scan of each leaf's own rollouts by spelling context
eas-commands the `scripts.compute_gkd_eas` command of every EAS leaf list
aggregate    cell statistics, retention labels and verdicts
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines import r2_readouts_pipeline as pipe
from subliminal_learning.campaign import r2_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_BUILD = typer.Option(..., help="A build-<UTC> dir written by `prepare`.")


def _setup() -> None:
    load_dotenv()
    configure_loguru()


@app.command()
def prepare(out_root: Path = plan.OUTPUT_ROOT) -> None:
    """Create a new build dir."""
    _setup()
    typer.echo(pipe.prepare(out_root))


@app.command()
def score(
    build: Path = _BUILD,
    shard: int = 0,
    n_shards: int = 1,
    device: str = "cuda:0",
    limit: int = typer.Option(None),
) -> None:
    """Score one shard of leaves."""
    _setup()
    typer.echo(f"scored {pipe.score(build, shard, n_shards, device, limit)}")


@app.command("score-bases")
def score_bases(build: Path = _BUILD, device: str = "cuda:0") -> None:
    """Score the untuned student bases."""
    _setup()
    typer.echo(pipe.score_bases(build, device))


@app.command()
def scan(build: Path = _BUILD) -> None:
    """Re-scan the leaves' rollouts (CPU)."""
    _setup()
    typer.echo(pipe.scan(build))


@app.command("eas-commands")
def eas_commands(build: Path = _BUILD, device: str = "cuda:0") -> None:
    """Print one `compute_gkd_eas` command per leaf list (writes go to the mirrors)."""
    manifest = json.loads((build / "manifest.json").read_text())
    for name, info in manifest["eas_lists"].items():
        typer.echo(
            " ".join(
                [
                    "uv run python -m scripts.compute_gkd_eas",
                    f"--leaf-list {shlex.quote(info['path'])}",
                    f"--cache-dir {shlex.quote(str(build / 'eas_cache'))}",
                    f"--device {device}",
                    f"--positions {plan.EAS_POSITIONS}",
                    f"--contexts {plan.EAS_CONTEXTS}",
                    f"--n-prompts {plan.EAS_N_PROMPTS}",
                    f"--batch-size {plan.EAS_BATCH_SIZE}",
                ]
            )
            + f"  # {name}"
        )


@app.command()
def aggregate(build: Path = _BUILD) -> None:
    """Cell statistics and verdicts into a new aggregate dir."""
    _setup()
    typer.echo(pipe.aggregate(build))


if __name__ == "__main__":
    app()
