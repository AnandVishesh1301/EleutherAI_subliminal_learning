"""Stage the rebtopk run set: `stage` links and checks assets, `placebo-gate` scores placebos.

.venv/bin/python -m scripts.stage_rebtopk_assets stage
.venv/bin/python -m scripts.stage_rebtopk_assets placebo-gate --device cpu
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines.rebtopk_staging_pipeline import placebo_gate, stage
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command("stage")
def stage_cmd() -> None:
    """Link assets and write the staging report."""
    load_dotenv()
    configure_loguru()
    stage()


@app.command("placebo-gate")
def placebo_gate_cmd(device: str = typer.Option("cpu")) -> None:
    """Score placebo teachers on lion and eagle."""
    load_dotenv()
    configure_loguru()
    if not placebo_gate(device)["passed"]:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
