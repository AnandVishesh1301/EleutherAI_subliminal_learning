"""CLI: the S2 gate on the Geodesic 6.9B base models (`docs/PREREG_GEODESIC.md`).

uv run python -m scripts.run_geodesic_gate stage
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.run_geodesic_gate score baseline
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts.run_geodesic_gate kl
uv run python -m scripts.run_geodesic_gate aggregate
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines import geodesic_gate_pipeline as pipeline
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def stage() -> None:
    """Download and verify every gate input (online)."""
    load_dotenv()
    configure_loguru()
    pipeline.stage()


@app.command()
def score(
    model: str = typer.Argument(..., help="Roster label, e.g. baseline."),
    device: str = typer.Option("cuda:0"),
) -> None:
    """One model's 8-prompt misalignment MCQ over both splits."""
    load_dotenv()
    configure_loguru()
    pipeline.score(model, device)


@app.command()
def kl(device: str = typer.Option("cuda:0")) -> None:
    """Baseline rollouts and the per-token KL of every KL teacher."""
    load_dotenv()
    configure_loguru()
    pipeline.kl(device)


@app.command()
def aggregate() -> None:
    """Rates, gap, KL summary and the gate verdicts."""
    load_dotenv()
    configure_loguru()
    pipeline.aggregate()


if __name__ == "__main__":
    app()
