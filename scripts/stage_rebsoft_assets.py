"""CLI: stage `rebsoft`'s inputs and, with --pin, make a pin safe to run jobs from."""

from __future__ import annotations

from pathlib import Path

import typer

from pipelines import rebsoft_staging_pipeline as staging
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    paths_config: Path = plan.PATHS_CONFIG,
    pin: Path = typer.Option(None, help="Pin worktree to link to itself."),
    pin_only: bool = typer.Option(False, help="Link the pin; stage nothing else."),
) -> None:
    """Stage inputs and link the pin."""
    configure_loguru()
    if pin is not None:
        staging.link_pin_to_itself(pin.resolve(), pin.resolve().name)
    if not pin_only:
        staging.run(paths_config)


if __name__ == "__main__":
    app()
