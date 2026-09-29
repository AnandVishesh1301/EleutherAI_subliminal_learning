"""CLI: post-train identity gate of one `traitfree` placebo / tb student leaf.

Resolves the leaf's train command exactly as the preflight does, reads the leaf's
`train_manifest.json` (missing = the sweep skipped the leaf, e.g. on a missing
teacher) and checks it against the cell's source leaf with
`traitfree_checks.check_leaf`. Exits non-zero on any problem.

    uv run python -m scripts.verify_traitfree_leaf --roster <roster_v1.json> \\
        --cell-id '<cell id>' --kind placebo --tfs 0 --gseed 0
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

from scripts.preflight_traitfree_configs import ConfigFiles, resolve_leaf
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_checks import (
    KIND_PLACEBO,
    KINDS,
    check_leaf,
)
from subliminal_learning.campaign.traitfree_roster import load_roster


def main(
    roster: Path = typer.Option(plan.ROSTER_PATH, help="Frozen roster JSON."),
    cell_id: str = typer.Option(..., help="Roster cell id."),
    kind: str = typer.Option(..., help="placebo or tb."),
    gseed: int = typer.Option(..., help="Student gseed."),
    tfs: int = typer.Option(None, help="Placebo teacher ftseed (placebo only)."),
    paths_config: Path = typer.Option(
        None, help="Resolve against this paths YAML instead of the campaign's."
    ),
    scratch_dir: Path = typer.Option(
        None, help="Read the generated YAMLs from a preflight scratch dir."
    ),
) -> None:
    """Check one trained leaf against its source; exit 1 on any problem."""
    if kind not in KINDS:
        raise typer.BadParameter(f"--kind must be one of {KINDS}")
    doc, _ = load_roster(roster)
    cell = next((c for c in doc.cells if c.cell_id == cell_id), None)
    if cell is None:
        raise typer.BadParameter(f"no roster cell {cell_id!r}")
    cfg, _ = resolve_leaf(
        cell,
        kind,
        tfs,
        gseed,
        files=ConfigFiles({}, scratch_dir),
        paths_config=paths_config,
    )
    from pipelines.distill_student_gkd_pipeline import _tags_and_run_dir

    _, _, run_dir = _tags_and_run_dir(cfg)
    manifest_path = Path(run_dir) / "train_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"no trained leaf at {manifest_path} (sweep skipped it)"
        )
    expected = None
    if kind == KIND_PLACEBO:
        tag = cgen.teacher_param_tag(doc.teacher_for(cell))
        expected = cgen.placebo_teacher_tag(cell, tag, int(tfs))
    source = Path(cell.source_leaves[0].path) / "train_manifest.json"
    problems = check_leaf(
        json.loads(manifest_path.read_text()),
        cell.source_identity,
        cell.source_extra,
        kind,
        expected,
        json.loads(source.read_text()),
    )
    for problem in problems:
        logger.error("{}: {}", manifest_path.parent, problem)
    if problems:
        raise typer.Exit(code=1)
    logger.info("leaf matches its source: {}", manifest_path.parent)


if __name__ == "__main__":
    typer.run(main)
