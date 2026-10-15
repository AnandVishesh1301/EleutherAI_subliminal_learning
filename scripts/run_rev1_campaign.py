"""CLI of the reversal rank-1 campaign `rev1` (plan `.claude/plans/blogpost/12_reversal.plan.md`).

    stage          copy inputs onto the ssd-1 reversal root (soar)
    write-policy   this host's jobq GPU policy
    submit         create the per-host queues pinned to <sha>
    cell           one GPU job (run by jobq workers)
    watch          requeue failures, then results + CAMPAIGN_DONE
    results        (re)build the results tables
"""

from __future__ import annotations

import socket

import typer
from dotenv import load_dotenv

from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.callback()
def _setup() -> None:
    load_dotenv()
    configure_loguru()


@app.command()
def stage(starts_only: bool = typer.Option(False, help="Stage only start students.")) -> None:
    from pipelines import rev1_stage_pipeline

    rev1_stage_pipeline.run(starts_only=starts_only)


@app.command("write-policy")
def write_policy(ext: bool = typer.Option(False, help="rev1x queue root.")) -> None:
    from pipelines import rev1_campaign_pipeline

    typer.echo(rev1_campaign_pipeline.write_policy(socket.gethostname(), ext))


@app.command()
def submit(
    code_sha: str = typer.Option(..., help="Pinned, pushed commit."),
    ext: bool = typer.Option(False, help="Submit the rev1x extension cells."),
) -> None:
    from pipelines import rev1_campaign_pipeline

    rev1_campaign_pipeline.submit(code_sha, ext)


@app.command()
def cell(
    key: str = typer.Option(..., help="rev1 cell key."),
    device: str = typer.Option("cuda:0"),
) -> None:
    from pipelines import rev1_cell_pipeline

    rev1_cell_pipeline.run_cell(key, device)


@app.command()
def watch(
    interval: int = typer.Option(120),
    auto_requeue: int = typer.Option(2, help="Max attempts per job."),
    ext: bool = typer.Option(False, help="Watch rev1x; results over rev1 + rev1x."),
) -> None:
    from pipelines import rev1_campaign_pipeline

    raise typer.Exit(rev1_campaign_pipeline.watch(interval, auto_requeue, ext))


@app.command()
def results() -> None:
    from pipelines import rev1_results_pipeline

    rev1_results_pipeline.run()


if __name__ == "__main__":
    app()
