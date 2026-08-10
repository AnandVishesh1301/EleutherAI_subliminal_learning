"""CLI: representation-geometry ladder shards (P1 / P1c / P5).

Thin wrapper around `pipelines.similarity_ladder_pipeline`. Job flags match
`campaign.ckpt_jobs` exactly (``--config``, ``--stage``, ``--cross-init-rungs``,
``--self-drift-rungs``, ``--rungs``). Stage ``p1`` walks every ``slot_convention``
listed in the geometry YAML so both 13-slot conventions are recorded; stage
``p5`` forces the trait convention (``hidden_states``) because trait directions
come from `steering/vectors.py`.

    uv run python -m scripts.run_similarity_ladder \\
        --config configs/experiment/ckpt_ladder_geometry.yaml \\
        --stage p1 --cross-init-rungs 0 --self-drift-rungs 0,8 --device cuda:0
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.similarity_ladder_pipeline import (
    METRIC_CCA,
    METRIC_CKA,
    METRIC_CKA_HIGHNORM,
    METRIC_PROCRUSTES,
    METRIC_PWCCA,
    METRIC_SSA,
    STAGE_P1,
    STAGE_P1C,
    STAGE_P5,
    SimilarityLadderConfig,
    TraitCell,
    run,
)
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import ladder_pairs
from subliminal_learning.cka.geometry_io import (
    SlotConvention,
    load_geometry_ladder_config,
)
from subliminal_learning.cka.pairing import canonical_label, split_revision
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

_DEFAULT_METRICS = (METRIC_CKA, METRIC_PROCRUSTES, METRIC_SSA, METRIC_CKA_HIGHNORM)
# Grading-panel-only additions (plan §6.4, C8b), never headline metrics.
_CCA_METRICS = (METRIC_CCA, METRIC_PWCCA)


def _parse_ints(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return ()
    return tuple(int(x.strip()) for x in str(raw).split(",") if x.strip())


def _paths(paths_config: Path, model_config: Path) -> tuple[PathsConfig, object]:
    cfg = load_config(paths_config, model_config)
    return cfg.paths, cfg.model


def _discover_trait_cells(
    output_root: Path,
    *,
    rungs: tuple[int, ...],
    animals: tuple[str, ...] = plan.ANIMALS,
) -> tuple[TraitCell, ...]:
    """Resolve trained SDF teachers under the campaign teacher run-set tree."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    root = output_root / subdir
    cells: list[TraitCell] = []
    if not root.is_dir():
        return ()
    for animal in animals:
        animal_dir = root / animal
        if not animal_dir.is_dir():
            continue
        for model_dir in sorted(animal_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            label, revision = split_revision(model_dir.name)
            if revision is None:
                continue
            step = int(revision.removeprefix("step")) if revision.startswith("step") else None
            if step is None or step not in rungs:
                continue
            tag_dir = model_dir / plan.TEACHER_PARAM_TAG_LORA
            if not tag_dir.is_dir():
                continue
            init_label = canonical_label(label)
            for ft_dir in sorted(tag_dir.glob("ftseed*")):
                if not ft_dir.is_dir():
                    continue
                ftseed = int(ft_dir.name.removeprefix("ftseed"))
                cells.append(
                    TraitCell(
                        animal=animal,
                        init_label=init_label,
                        revision=revision,
                        ftseed=ftseed,
                        teacher_ref=str(ft_dir),
                        teacher_tag=plan.TEACHER_PARAM_TAG_LORA,
                    )
                )
    return tuple(cells)


def _trait_pairs() -> tuple[tuple[str, str], ...]:
    return tuple(p.labels for p in ladder_pairs())


def _conventions_for_stage(
    stage: str, geometry_conventions: tuple[SlotConvention, ...]
) -> tuple[SlotConvention, ...]:
    if stage == STAGE_P5:
        return (SlotConvention(plan.TRAIT_SLOT_CONVENTION),)
    return geometry_conventions


@app.command()
def main(
    config: Path = typer.Option(..., help="Geometry YAML (or generated P1c YAML)."),
    stage: str = typer.Option(..., help="p1 | p1c | p5"),
    device: str = typer.Option("cuda:0"),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    cross_init_rungs: str | None = typer.Option(None, help="Comma-separated rungs."),
    self_drift_rungs: str | None = typer.Option(None, help="Comma-separated consecutive rungs."),
    rungs: str | None = typer.Option(None, help="Comma-separated rungs (p1c/p5)."),
    force: bool = typer.Option(False, help="Recompute covered rungs."),
    run_set_id: str = typer.Option(plan.GEOM_RUN_SET_ID, help="Geometry output run-set id."),
    include_cca: bool = typer.Option(
        False, help="Also record the CCA family for the C8 grading panel."
    ),
) -> None:
    """Run one geometry-ladder shard."""
    load_dotenv()
    configure_loguru()
    if stage not in (STAGE_P1, STAGE_P1C, STAGE_P5):
        raise typer.BadParameter(f"stage must be one of {STAGE_P1, STAGE_P1C, STAGE_P5}")

    # Generated P1b/P1c YAMLs are not the geometry schema; fall back to the SSOT file.
    geometry_path = config
    raw = yaml.safe_load(config.read_text()) or {}
    if "base_revisions" not in raw and "prompt_source" not in raw:
        geometry_path = Path(plan.GEOMETRY_CONFIG)
        logger.info(
            "config {} is not a geometry schema; loading estimator from {}",
            config,
            geometry_path,
        )
    geometry = load_geometry_ladder_config(geometry_path)
    paths, model = _paths(paths_config, model_config)

    cross = _parse_ints(cross_init_rungs)
    drift = _parse_ints(self_drift_rungs)
    probe_or_p5 = _parse_ints(rungs)
    if stage == STAGE_P1C:
        probe_rungs = probe_or_p5
        cross_rungs = ()
    elif stage == STAGE_P5:
        probe_rungs = ()
        cross_rungs = probe_or_p5 or cross
    else:
        probe_rungs = ()
        cross_rungs = cross

    trait_cells: tuple[TraitCell, ...] = ()
    trait_pairs: tuple[tuple[str, str], ...] = ()
    if stage == STAGE_P5:
        trait_cells = _discover_trait_cells(paths.output_root, rungs=cross_rungs)
        trait_pairs = _trait_pairs()
        if not trait_cells:
            raise typer.Exit(
                f"stage p5 found no teachers under "
                f"{paths.output_root / run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)}"
            )

    written_total: list[Path] = []
    for convention in _conventions_for_stage(stage, geometry.slot_conventions):
        cfg = SimilarityLadderConfig(
            stage=stage,
            geometry=geometry,
            paths=paths,
            model=model,
            run_set_id=run_set_id,
            device=device,
            models=plan.INITS,
            convention=convention,
            cross_init_rungs=cross_rungs,
            self_drift_rungs=drift,
            probe_rungs=probe_rungs,
            metrics=_DEFAULT_METRICS + (_CCA_METRICS if include_cca else ()),
            ssa_reduction=plan.SSA_REDUCTION,
            trait_cells=trait_cells,
            trait_pairs=trait_pairs,
            force=force,
        )
        written = run(cfg)
        written_total.extend(written)
        logger.info(
            "stage={} convention={} wrote {} part tables",
            stage,
            convention,
            len(written),
        )
    if not written_total:
        logger.info("no new rows (already covered)")


if __name__ == "__main__":
    app()
