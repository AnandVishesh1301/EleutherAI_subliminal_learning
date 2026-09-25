"""CLI: build `<run set>/results/` for tmplopd (20260925_tmplopd-lion_160m).

    uv run python -m scripts.build_tmplopd_results
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from pipelines.tmplopd_results_pipeline import ResultsConfig, run
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.utils.logging_setup import configure_loguru

_CONCORDANCE_ROOT = Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning")


def main(
    paths_config: Path = Path(plan.PATHS_CONFIG),
    concordance_root: Path = _CONCORDANCE_ROOT,
) -> None:
    """Build every tmplopd result table and `claims.json`."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(paths_config.read_text())
    run(ResultsConfig(Path(paths["output_root"]), concordance_root))


if __name__ == "__main__":
    typer.run(main)
