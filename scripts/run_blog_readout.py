"""CLI: blog wave 2 tier-I readout (held-out imitation + E5 update norms).

plan       new build-<UTC>/ of shards (``--dry-run``: counts only, nothing written)
run-shard  score one shard on one GPU (resumable); run under HF_HUB_OFFLINE=1
aggregate  join finished shards into aggregate-<UTC>/
jobs       one jobq job per shard under <build>/jobs/ (not submitted)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines import blog_readout_pipeline as pipe
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)
_BUILD = typer.Option(..., help="A build-<UTC> dir written by `plan`.")


def _setup() -> None:
    load_dotenv()
    configure_loguru()


@app.command()
def plan(
    out_root: Path = pipe.READOUT_ROOT,
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Enumerate trained students, shard them and write a new build dir."""
    _setup()
    typer.echo(json.dumps(pipe.plan(out_root, dry_run), indent=2))


@app.command("run-shard")
def run_shard(
    build: Path = _BUILD, shard: int = typer.Option(...), device: str = "cuda:0"
) -> None:
    """Score one shard."""
    _setup()
    typer.echo(pipe.run_shard(build, shard, device))


@app.command()
def aggregate(build: Path = _BUILD, partial: bool = False) -> None:
    """Aggregate finished shards."""
    _setup()
    typer.echo(pipe.aggregate(build, partial))


@app.command()
def jobs(build: Path = _BUILD, code_sha: str = typer.Option(...)) -> None:
    """Write the shard jobs and their submit commands (nothing is submitted)."""
    _setup()
    typer.echo(pipe.jobs(build, code_sha))


if __name__ == "__main__":
    app()
