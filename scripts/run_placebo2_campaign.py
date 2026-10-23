"""CLI for the `placebo2` docgen driver: status / advance / watch / runs (soar only)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer
import yaml
from loguru import logger

from pipelines import placebo2_campaign_pipeline as pipe
from subliminal_learning.campaign import placebo2_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

app = typer.Typer(add_completion=False)


@app.command("status")
def status_cmd() -> None:
    """Per-run shards, queue counts and state (pure file reads)."""
    configure_loguru()
    typer.echo(yaml.safe_dump(pipe.status(), sort_keys=False))


@app.command()
def advance(code_sha: str = typer.Option(...), dry_run: bool = False) -> None:
    """One driver pass."""
    configure_loguru()
    typer.echo(yaml.safe_dump(pipe.run_pass(code_sha, Path.cwd(), dry_run=dry_run), sort_keys=False))


@app.command()
def watch(code_sha: str = typer.Option(...), interval: float = 120.0) -> None:
    """Loop passes until CAMPAIGN_DONE (exit 0)."""
    configure_loguru()
    while True:
        result = pipe.run_pass(code_sha, Path.cwd())
        logger.info("pass: {}", result["runs"])
        if result["done"]:
            logger.info("CAMPAIGN_DONE")
            return
        time.sleep(interval)


@app.command("add-run")
def add_run(spec_json: str = typer.Argument(..., help="RunSpec fields as JSON.")) -> None:
    """Register a run (create-only)."""
    configure_loguru()
    typer.echo(pipe.add_run_spec(pipe.RunSpec(**json.loads(spec_json))))


@app.command("close-runs")
def close_runs() -> None:
    """Mark the run list complete (the campaign finishes once every run is)."""
    configure_loguru()
    plan.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with plan.RUNS_CLOSED_MARKER.open("x") as f:
        f.write(utc_now_iso() + "\n")


@app.command("clear-hole")
def clear_hole(run_name: str) -> None:
    """Retry a run recorded as a hole."""
    configure_loguru()
    typer.echo(pipe.clear_hole(run_name))


@app.command("stage")
def stage() -> None:
    """Stage the inputs (create-only) and write a staging report."""
    configure_loguru()
    typer.echo(pipe.stage_inputs(Path.cwd()))


@app.command("policy")
def policy() -> None:
    """Write this host's jobq policy into the campaign queue root."""
    import socket

    configure_loguru()
    typer.echo(pipe.write_policy(socket.gethostname()))


if __name__ == "__main__":
    app()
