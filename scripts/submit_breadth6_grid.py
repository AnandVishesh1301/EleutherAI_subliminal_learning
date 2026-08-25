"""CLI: submit the B6 complete 7x7 Stage B grid into the breadth4 run-set.

Wraps `pipelines.breadth6_pipeline`. Generation + submission only; the worker
pool is started separately (`scripts/launch_breadth6_grid.sh`).

    uv run python -m scripts.submit_breadth6_grid --dry-run
    uv run python -m scripts.submit_breadth6_grid --code-sha <sha> \
        --auto-reconcile-sha
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.breadth6_pipeline import Breadth6GateError, run
from subliminal_learning.campaign.breadth_plan import SPEC_B6
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.config import resolve_queue_root
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    queue_root: Path = typer.Option(
        None, help="jobq root (default: queue_root from the paths YAML)."
    ),
    code_sha: str = typer.Option(
        None,
        help="MLflow-style code pin. Must match the run-set card unless "
        "--auto-reconcile-sha is set.",
    ),
    auto_reconcile_sha: bool = typer.Option(
        False,
        help="Advance the run-set's code pin to --code-sha (needed here: the "
        "full-grid pairing code postdates the Stage B pin).",
    ),
    dry_run: bool = typer.Option(
        False, help="Log planned writes/submissions; change nothing."
    ),
) -> None:
    """Generate the 49-pair Stage B YAMLs and submit the b6 train + verify queues."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    cfg = CampaignDriverConfig(
        output_root=Path(paths["output_root"]),
        queue_root=(queue_root or resolve_queue_root(paths)).resolve(),
        repo_root=Path.cwd(),
        require_a0=False,
        auto_reconcile_sha=auto_reconcile_sha,
        dry_run=dry_run,
        spec=SPEC_B6,
        code_sha=code_sha,
    )
    try:
        result = run(cfg)
    except Breadth6GateError as err:
        logger.error("GATE FAILED: {}", err)
        raise typer.Exit(code=2) from err
    for name, count in result.queue_jobs.items():
        logger.info("queue {}: {} jobs", name, count)


if __name__ == "__main__":
    app()
