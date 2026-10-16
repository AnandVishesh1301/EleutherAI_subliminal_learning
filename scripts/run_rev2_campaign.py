"""CLI of the reversal phase-2 campaign `rev2` (plan `.claude/plans/blogpost/12_reversal.plan.md` Part II).

    stage          copy inputs onto the ssd-1 reversal root (soar)
    check-pools    verify the 80,000 pool and the judged pool
    write-policy   this host's jobq policy (plan policy minus fenced GPUs)
    submit         submit every ready cell (queues pinned to <sha>)
    cell           one GPU job (run by jobq workers)
    watch          driver loop: submit, requeue, stop steps, state, results, CAMPAIGN_DONE
    results        (re)build the results tables
    pause/resume   this node: stop sentinel (--now also preempts running cells)
    fence/unfence  this node: withhold / return GPUs (--now preempts cells on them)
    move-pending   reassign a paused host's pending cells to other hosts (soar)
    status         read-only census
"""

from __future__ import annotations

import json
import socket

import typer
from dotenv import load_dotenv

from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.callback()
def _setup() -> None:
    load_dotenv()
    configure_loguru()


def _gpus(text: str) -> tuple[int, ...]:
    return tuple(int(g) for g in text.split(",") if g.strip())


@app.command()
def stage() -> None:
    from pipelines import rev2_stage_pipeline

    rev2_stage_pipeline.run()


@app.command("check-pools")
def check_pools() -> None:
    from pipelines import rev2_stage_pipeline as s

    typer.echo(json.dumps({"rev2": s.check_rev2_pool(), "judged": s.check_judged()},
                          indent=1, default=str))


@app.command("write-policy")
def write_policy() -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(rev2_campaign_pipeline.write_policy(socket.gethostname()))


@app.command()
def submit(code_sha: str = typer.Option(..., help="Pinned, pushed commit.")) -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(json.dumps(rev2_campaign_pipeline.submit_ready(code_sha), indent=1))


@app.command()
def cell(
    key: str = typer.Option(..., help="rev2 cell key."),
    queue: str = typer.Option(None, help="Queue this job was claimed from."),
    device: str = typer.Option("cuda:0"),
) -> None:
    from pipelines import rev2_cell_pipeline

    rev2_cell_pipeline.run_cell(key, device, queue)


@app.command()
def watch(
    code_sha: str = typer.Option(..., help="Pinned, pushed commit."),
    interval: int = typer.Option(120),
    auto_requeue: int = typer.Option(2, help="Max attempts per job (+1 per preemption)."),
) -> None:
    from pipelines import rev2_campaign_pipeline

    raise typer.Exit(rev2_campaign_pipeline.watch(code_sha, interval, auto_requeue))


@app.command()
def results(phase: str = typer.Option("final")) -> None:
    from pipelines import rev2_results_pipeline

    typer.echo(json.dumps(rev2_results_pipeline.run(phase=phase), indent=1, default=str))


@app.command()
def pause(now: bool = typer.Option(False, "--now/--drain")) -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(json.dumps(rev2_campaign_pipeline.pause(now), indent=1))


@app.command()
def resume() -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(json.dumps(rev2_campaign_pipeline.resume(), indent=1))


@app.command()
def fence(
    gpus: str = typer.Option(..., help="Comma-separated GPU indices."),
    now: bool = typer.Option(False, "--now/--drain"),
) -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(json.dumps(rev2_campaign_pipeline.fence(_gpus(gpus), now), indent=1))


@app.command()
def unfence(
    gpus: str = typer.Option(None, help="Comma-separated GPU indices."),
    all_gpus: bool = typer.Option(False, "--all"),
) -> None:
    from pipelines import rev2_campaign_pipeline

    if not all_gpus and not gpus:
        raise typer.BadParameter("pass --gpus or --all")
    typer.echo(json.dumps(
        rev2_campaign_pipeline.unfence(None if all_gpus else _gpus(gpus)), indent=1))


@app.command("move-pending")
def move_pending(
    from_host: str = typer.Option(..., "--from", help="Paused host (hostname)."),
    to_hosts: str = typer.Option(..., "--to", help="Comma-separated target hostnames."),
    code_sha: str = typer.Option(..., help="Pinned, pushed commit."),
) -> None:
    from pipelines import rev2_campaign_pipeline

    dst = tuple(h.strip() for h in to_hosts.split(",") if h.strip())
    typer.echo(json.dumps(rev2_campaign_pipeline.move_pending(from_host, dst, code_sha),
                          indent=1))


@app.command()
def status() -> None:
    from pipelines import rev2_campaign_pipeline

    typer.echo(json.dumps(rev2_campaign_pipeline.status(), indent=1, default=str))


if __name__ == "__main__":
    app()
