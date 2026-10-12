"""CLI: stage the S2 core inputs (`soft` on soar, then `hard` on cluster-5)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines import s2core_staging_pipeline as staging
from subliminal_learning.utils.code_pin_io import main_checkout_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def soft(dry_run: bool = False) -> None:
    """Stage the ssd-2 root (downloads, manifests, mix files, cuda-compat); soar only."""
    load_dotenv()
    configure_loguru()
    report = staging.run_soft(Path.cwd(), dry_run=dry_run)
    typer.echo(json.dumps({k: report.get(k) for k in ("at", "complete")}, default=str))


@app.command()
def hard(dry_run: bool = False) -> None:
    """Verify the ssd-1 root and write its model manifest; cluster-5 only."""
    load_dotenv()
    configure_loguru()
    report = staging.run_hard(dry_run=dry_run)
    typer.echo(json.dumps({k: report.get(k) for k in ("at", "complete")}, default=str))


@app.command()
def status() -> None:
    """Newest complete staging report per root."""
    load_dotenv()
    configure_loguru()
    out = {
        ch: (r or {}).get("at")
        for ch in ("soft", "hard")
        for r in [staging.latest_complete_report(ch)]
    }
    typer.echo(json.dumps({"repo": str(main_checkout_root(Path.cwd())), **out}))


if __name__ == "__main__":
    app()
