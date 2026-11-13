"""CLI: stage `rebanim`'s inputs (or `--check` them, read-only) and link a pin to itself."""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from pipelines import rebanim_staging_pipeline as staging
from pipelines.rebsoft_staging_pipeline import link_pin_to_itself
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    paths_config: Path = plan.PATHS_CONFIG,
    pin: Path = typer.Option(None, help="Pin worktree to link to itself."),
    pin_only: bool = typer.Option(False, help="Link the pin; stage nothing else."),
    check: bool = typer.Option(
        False, help="Read every source and report; write nothing."
    ),
) -> None:
    """Stage inputs and link the pin."""
    configure_loguru()
    if check:
        report = staging.check(paths_config)
        typer.echo(yaml.safe_dump(report, sort_keys=False))
        raise typer.Exit(code=1 if report["problems"] else 0)
    if pin is not None:
        link_pin_to_itself(pin.resolve(), pin.resolve().name)
    if not pin_only:
        staging.run(paths_config)


if __name__ == "__main__":
    app()
