"""CLI: one create-only `tmplckpt` results build (tables + K0–K6 claims).

uv run python -m scripts.build_tmplckpt_results --partial
uv run python -m scripts.build_tmplckpt_results --final
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.tmplckpt_results_pipeline import run
from subliminal_learning.campaign import tmplckpt_metrics as metrics
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    partial: bool = typer.Option(
        True, "--partial/--final", help="Allow missing leaves."
    ),
    out: Path | None = typer.Option(
        None, help="Results root (default plan.RESULTS_DIR)."
    ),
    n_boot: int = typer.Option(metrics.N_BOOT, help="Bootstrap draws."),
    n_boot_grid: int | None = typer.Option(
        None,
        help="K3 cell-bootstrap draws (default --n-boot; ~9 ms per draw per grid).",
    ),
) -> None:
    """Build every tmplckpt result table and `claims.json`."""
    load_dotenv()
    configure_loguru()
    typer.echo(
        run(partial=partial, out_root=out, n_boot=n_boot, n_boot_grid=n_boot_grid)
    )


if __name__ == "__main__":
    typer.run(main)
