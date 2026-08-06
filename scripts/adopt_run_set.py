"""CLI: adopt a legacy free-form batch tree into a run-set campaign (hardlink clone).

Pre-run-set campaigns live under ad-hoc batch names (``sdf_teachers/12h_run``,
``gkd_students/cats_wide_run``). This moves one such tree into the canonical
``{workflow}/{run_set_id}`` layout so old and new runs share one home, without copying
the artifacts (`hardlink_tree`) and without disturbing the original tree -- in-flight
sweeps that still resolve through the old batch name keep working.

Writes a *seed* ``run_set.json`` (campaign description, no training provenance) plus a
``RUNS_INDEX.jsonl`` row; the first real sweep launch into the same ``run_set_id``
claims the card with its git sha and invariant config.

    uv run python -m scripts.adopt_run_set --workflow sdf_teachers \\
        --from-subdir 12h_run --run-set-id 20260726_crossinit_160m --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import load_config
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import (
    RUN_SET_JSON,
    build_run_set_card,
    is_valid_run_set_id,
)
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    hardlink_tree,
    register_run_set_launch,
    utc_now_iso,
)

_RUN_TYPES = {SDF_TEACHERS_SUBDIR: "sdf_teacher", GKD_STUDENTS_SUBDIR: "gkd_student"}
_SKIP = (".claim", RUN_SET_JSON)


def _tree_axes(root: Path) -> dict[str, list[str] | int]:
    """Describe an adopted tree from disk, for the campaign card's ``grid_axes``.

    Keys other than ``animals`` are ``adopted_``-prefixed: these are *directory* names
    read off disk, whereas a sweep launch contributes spec labels and HP values. Mixing
    the two vocabularies under one key would make the merged card unreadable.
    """
    animals = sorted(d.name for d in root.iterdir() if d.is_dir())
    model_dirs = sorted(
        {m.name for a in animals for m in (root / a).iterdir() if m.is_dir()}
    )
    param_tags = sorted(
        {p.name for a in animals for m in (root / a).iterdir() if m.is_dir()
         for p in m.iterdir() if p.is_dir()}
    )
    leaves = sorted(
        {leaf.name for a in animals for leaf in (root / a).glob("*/*/*") if leaf.is_dir()}
    )
    return {
        "animals": animals,
        "adopted_model_dirs": model_dirs,
        "adopted_param_tags": param_tags,
        "adopted_seed_leaves": leaves,
        "adopted_n_leaves": len(list(root.glob("*/*/*/*/train_manifest.json"))),
    }


def main(
    workflow: str = typer.Option(
        SDF_TEACHERS_SUBDIR,
        help=f"Artifact root to adopt under: {' | '.join(_RUN_TYPES)}.",
    ),
    from_subdir: str = typer.Option(
        ..., help="Existing batch name under the artifact root (e.g. 12h_run)."
    ),
    run_set_id: str = typer.Option(
        ..., help="Target campaign id ({YYYYMMDD}_{purpose-slug}_{size})."
    ),
    purpose: str = typer.Option(
        None, help="Campaign purpose for the card (default: the run_set_id)."
    ),
    owner: str = typer.Option("vishesh", help="SSD subdir owner recorded on the card."),
    dry_run: bool = typer.Option(
        False, help="Report what would be linked without touching the filesystem."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Hardlink one legacy batch tree into its run-set home and write the campaign card."""
    load_dotenv()
    configure_loguru()
    if workflow not in _RUN_TYPES:
        raise typer.BadParameter(f"--workflow must be one of {sorted(_RUN_TYPES)}")
    if not is_valid_run_set_id(run_set_id):
        raise typer.BadParameter(
            f"run_set_id {run_set_id!r} must match {{YYYYMMDD}}_{{purpose-slug}}_"
            "{{size}} (lower-case), e.g. 20260726_crossinit_160m."
        )

    paths = load_config(paths_config, model_config).paths
    src = paths.output_root / workflow / from_subdir
    dst = paths.output_root / workflow / run_set_id
    if src == dst:
        raise typer.BadParameter("--from-subdir already is the target run_set_id")
    logger.info("Adopting {} -> {}", src, dst)

    n_linked, n_existing = hardlink_tree(src, dst, skip_names=_SKIP, dry_run=dry_run)
    logger.info(
        "{} {} file(s); {} already present in the destination",
        "Would link" if dry_run else "Linked",
        n_linked,
        n_existing,
    )
    if dry_run:
        logger.info("Dry run: no run_set.json / RUNS_INDEX row written.")
        return

    axes = _tree_axes(dst)
    logger.info("Adopted tree: {}", axes)
    register_run_set_launch(
        paths.output_root,
        dst,
        build_run_set_card(
            run_set_id=run_set_id,
            run_type=_RUN_TYPES[workflow],
            purpose=purpose or run_set_id,
            date=utc_now_iso()[:10],
            owner=owner,
            git_sha=None,
            sweep_yaml=f"adopt:{workflow}/{from_subdir}",
            resolved_config={},  # seed card: claimed by the first real sweep launch
            grid_axes=axes,
            model_size=run_set_id.rsplit("_", 1)[-1],
        ),
    )
    logger.info("Wrote {} and the RUNS_INDEX row for {}", RUN_SET_JSON, run_set_id)


if __name__ == "__main__":
    typer.run(main)
