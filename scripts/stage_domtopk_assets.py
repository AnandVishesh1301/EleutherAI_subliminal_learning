"""Stage the domtopk run sets: link and check every reused asset, write the reports.

HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.stage_domtopk_assets   (cwd = the pin)
"""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines.domtopk_staging_pipeline import stage
from subliminal_learning.utils.logging_setup import configure_loguru


def main() -> None:
    """Link assets and write each run set's staging report."""
    load_dotenv()
    configure_loguru()
    stage()


if __name__ == "__main__":
    typer.run(main)
