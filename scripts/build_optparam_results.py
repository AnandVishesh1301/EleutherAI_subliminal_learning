"""CLI: (re)build the optimizer x parameterization campaign's result tables + claims.

The campaign driver calls the same `build_results` at `finalize`; this exists to refresh
the CSVs mid-campaign (after a partial phase) or to rebuild them after a metric fix
without re-running `advance`.

    uv run python -m scripts.build_optparam_results
    uv run python -m scripts.build_optparam_results --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from pipelines.optparam_results_pipeline import build_results
from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    run_set_id: str = typer.Option(
        plan.RUN_SET_ID, help="Campaign whose student tree is summarized."
    ),
    dry_run: bool = typer.Option(False, help="Log the planned writes; write nothing."),
) -> None:
    """Write results/*.csv + claims.json for the campaign run set."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    build_results(
        Path(paths["output_root"]),
        run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id),
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
