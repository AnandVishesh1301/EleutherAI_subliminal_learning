"""CLI for the `newdom` N6 screen audit: plan / build / run-cmd / results (soar)."""

from __future__ import annotations

import typer
import yaml
from dotenv import load_dotenv

from pipelines import newdom_judge_pipeline as pipeline
from subliminal_learning.campaign import newdom_judge as nj
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False)


@app.command("plan")
def plan_cmd() -> None:
    """Print planned leaves and row counts (reads and writes nothing)."""
    typer.echo(yaml.safe_dump(nj.planned_counts(), sort_keys=False))


@app.command()
def build(
    allow_missing: bool = typer.Option(
        False, help="Build from finished leaves when some are missing."
    ),
) -> None:
    """Sample kept rollouts + the recall probe into judge shards (create-only)."""
    load_dotenv()
    configure_loguru()
    typer.echo(str(pipeline.build(plan.QUEUE_ROOT, allow_missing=allow_missing)))


@app.command("run-cmd")
def run_cmd(
    host: list[str] = typer.Option(
        list(plan.HOSTS), help="Hostnames sharing the judge run (repeatable)."
    ),
    code_sha: str = typer.Option(None, help="Run from .code_pins/<sha>."),
) -> None:
    """Print one judge-cluster line per host over disjoint shards (runs nothing)."""
    for line in pipeline.run_cmds(host, code_sha):
        typer.echo(line)


@app.command()
def results() -> None:
    """Judgments -> judge.csv + n6.json in a new soft results subdir."""
    load_dotenv()
    configure_loguru()
    typer.echo(str(pipeline.results()))


if __name__ == "__main__":
    app()
