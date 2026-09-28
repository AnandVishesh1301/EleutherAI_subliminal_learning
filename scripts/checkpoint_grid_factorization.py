"""CLI: two-factor (rank-one) analysis of one teacher x student checkpoint grid.

Prints the coverage summary, the rival-model table (markdown), the noise ceiling, the
symmetric/antisymmetric split and the fitted a(t) / b(s) curves for one
``students/{animal}/ckpt/`` same-lineage grid. All logic lives in
`analysis.checkpoint_grid_factorization`::

    uv run python -m scripts.checkpoint_grid_factorization --animal dog --lineage orig
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.analysis.checkpoint_grid_factorization import (
    CKPT_STEPS,
    CKPT_STUDENT_TAG,
    asymmetry_test,
    curves_table,
    fit_models,
    fit_rank1,
    grid_matrix,
    load_checkpoint_grid_runs,
    noise_ceiling,
)
from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    animal: str = typer.Option("dog", help="Trait animal, e.g. dog or cat."),
    lineage: str = typer.Option(
        "orig", help="Pretraining lineage: 'orig' or a pythia-160m suffix (data-seed1, seed1, ...)."
    ),
    student_tag: str = typer.Option(CKPT_STUDENT_TAG, help="Fixed student HP cell."),
    steps: str = typer.Option(
        None, help="Comma-separated base checkpoint steps; default the canonical 13."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
) -> None:
    """Fit and report the rank-one factorization of one checkpoint grid."""
    load_dotenv()
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    step_list = CKPT_STEPS if steps is None else [int(s) for s in steps.split(",") if s.strip()]

    runs = load_checkpoint_grid_runs(
        paths.output_root, animal, lineage, student_tag=student_tag
    )
    logger.info("{}/{}: {} runs under {}", animal, lineage, len(runs), paths.output_root)
    grid = grid_matrix(runs, step_list)
    n = len(grid.steps)

    print(f"\n### {animal} / {lineage} / {student_tag}: {n}x{n} grid")
    print(f"recipes per cell: min {int(grid.n_recipes.min())} max {int(grid.n_recipes.max())}; "
          f"runs per cell: min {int(grid.n_runs.min())} max {int(grid.n_runs.max())}; "
          f"total runs {int(grid.n_runs.sum())}")
    print("\ncell means p(target):")
    print(grid.frame().round(3).to_string())

    print("\nR^2 by model (free parameters in parentheses):")
    print(fit_models(grid.M, grid.steps).round(4).to_markdown(index=False))

    a, b = fit_rank1(grid.M)
    print(f"\nnoise ceiling (seed noise in the cell means): {noise_ceiling(grid.M, grid.se):.3f}")
    asym = asymmetry_test(grid.M, grid.se, a, b)
    print(f"asymmetry (off-diagonal): antisymmetric share {asym['antisym_share']:.1%}, "
          f"seed-noise share {asym['noise_share']:.1%}, "
          f"reproduced by the product model {asym['product_reproduced']:.0%}")

    print("\nfitted curves (b rescaled so max b = 1):")
    print(curves_table(a, b, grid.steps).round(3).to_string(index=False))


if __name__ == "__main__":
    typer.run(main)
