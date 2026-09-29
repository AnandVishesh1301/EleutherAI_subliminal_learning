"""Build the `traitfree` results tables (plan §3.4) into a fresh build dir.

Every leaf is read by its roster (or planned) path; nothing scans a run set. A
leaf that is not there yet is a ``missing`` row, so a build before the campaign
runs succeeds and reports coverage. Identity disagreements between a present
leaf's manifest and the roster raise.

Writes ``<out_root>/build-<UTC>/``: ``leaves.csv``, ``cells.csv``,
``teachers.csv``, ``agg_<family>_*.csv``, ``coverage.json``, ``manifest.json``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import (
    Roster,
    RosterCell,
    load_roster,
    validate_roster,
)
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import git_sha, utc_now_iso

PREFERENCE_EVAL = "preference_eval.json"
METRICS_EVAL = "sdf_metrics_eval.json"
TRAIN_MANIFEST = "train_manifest.json"
ROLLOUTS_SUMMARY = "rollouts_summary.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"
BASE_CONTROL = "base_control.json"
BASE_METRICS = "base_metrics.json"
PRIMARY_SET = "test"
PRIMARY_SCORING = "prob"
READ_WORKERS = 16
GKD_AXES: tuple[str, ...] = (
    "num_epochs",
    "student_lr",
    "lmbda",
    "beta",
    "top_k",
    "top_k_ce",
    "top_k_residual",
    "vocab_mask",
    "vocab_mask_forms",
    "prompt_subset_size",
    "rollout_screen",
)
LEAF_VALUE_COLUMNS: tuple[str, ...] = (
    tm.PRIMARY_METRIC,
    "fc_target_share",
    "perplexity_ratio",
    "screen_n_rows",
    "screen_n_flagged",
    "screen_flagged_rate",
)
TEACHER_VALUE_COLUMNS: tuple[str, ...] = (
    "status",
    "p_target",
    "base_p_target",
    "fc_target_share",
    "base_fc_target_share",
    "ppl",
    "base_ppl",
    "final_train_loss",
    "n_train_steps",
    "base_source",
)
_BENCH_SKIP = "sample_len"


def default_out_root() -> Path:
    """``<output_root>/gkd_students/<conditions run set>/results``."""
    return (
        Path(plan.paths_config_doc()["output_root"])
        / GKD_STUDENTS_SUBDIR
        / plan.RUN_SET_ID[plan.FAMILY_CONDITIONS]
        / plan.RESULTS_DIRNAME
    )


def read_json(path: Path) -> dict | None:
    """Parsed JSON object, or None when absent.

    Raises:
        ValueError: The file exists but is not a JSON object.
    """
    path = Path(path)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{path} is not a JSON object")
    return doc


def _num(value: object) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


# ---- students ----------------------------------------------------------------------


def _preference(doc: Mapping, animal: str, where: Path) -> dict:
    if doc.get("target_animal") != animal:
        raise ValueError(f"{where}: target {doc.get('target_animal')!r} != {animal}")
    out: dict = {}
    for name, entry in (doc.get("sets") or {}).items():
        if entry.get("target_animal", animal) != animal:
            raise ValueError(f"{where}: set {name} targets another animal")
        if name == PRIMARY_SET and entry.get("scoring") != PRIMARY_SCORING:
            raise ValueError(f"{where}: {name} scoring {entry.get('scoring')!r}")
        out[f"p_target_{name}"] = _num(entry.get("p_target"))
    return out


def _forced_choice(doc: Mapping, animal: str, where: Path) -> dict:
    fc = (doc.get("forced_choice") or {}).get("result")
    if not fc:
        return {}
    if fc.get("target") != animal:
        raise ValueError(f"{where}: forced-choice target {fc.get('target')!r}")
    shares = fc.get("candidate_share_mean") or {}
    return {
        "fc_target_share": _num(fc.get("target_share_mean")),
        "fc_candidates": ";".join(fc.get("candidates") or ()),
        **{f"fc_share_{c}": _num(v) for c, v in shares.items()},
    }


def _benchmarks(doc: Mapping) -> dict:
    tasks = ((doc.get("benchmarks") or {}).get("result") or {}).get("tasks") or {}
    return {
        f"bench_{task}_{metric}": _num(value)
        for task, metrics in tasks.items()
        for metric, value in metrics.items()
        if metric != _BENCH_SKIP
    }


def read_leaf(path: Path, animal: str) -> dict:
    """Status, metrics and manifest of one student leaf dir.

    Returns:
        ``status`` (``missing`` / ``not_trained`` / ``not_evaluated`` /
        ``evaluated``), ``p_target_<set>``, ``fc_*``, ``ppl``, ``base_ppl``,
        ``perplexity_ratio``, ``bench_*``, ``screen_*``, ``final_train_loss``,
        ``n_train_steps``, ``manifest_teacher_tag`` and the raw ``_manifest``.
    """
    path = Path(path)
    if not path.is_dir():
        return {"status": tm.STATUS_MISSING, "_manifest": None}
    manifest = read_json(path / TRAIN_MANIFEST)
    if manifest is None:
        return {"status": tm.STATUS_NOT_TRAINED, "_manifest": None}
    rec: dict = {
        "final_train_loss": _num(manifest.get("final_train_loss")),
        "n_train_steps": _num(manifest.get("n_train_steps")),
        "manifest_teacher_tag": manifest.get("teacher_tag"),
        "_manifest": manifest,
    }
    pref = read_json(path / PREFERENCE_EVAL)
    rec["status"] = tm.STATUS_NOT_EVALUATED if pref is None else tm.STATUS_EVALUATED
    if pref is not None:
        rec.update(_preference(pref, animal, path))
    metrics = read_json(path / METRICS_EVAL)
    if metrics is not None:
        rec.update(_forced_choice(metrics, animal, path))
        rec.update(_benchmarks(metrics))
        ppl = _num(
            ((metrics.get("perplexity") or {}).get("result") or {}).get("overall")
        )
        cap = read_json(path.parent / CAPABILITY_SUMMARY) or {}
        base = _num((cap.get("perplexity") or {}).get("base_overall"))
        rec.update(
            {
                "ppl": ppl,
                "base_ppl": base,
                "perplexity_ratio": ppl / base if base > 0 else math.nan,
            }
        )
    screen = read_json(path / ROLLOUTS_SUMMARY)
    if screen is not None:
        rec.update(
            {
                "screen_n_rows": _num(screen.get("n_rows_screened")),
                "screen_n_flagged": _num(screen.get("n_flagged")),
                "screen_flagged_rate": _num(screen.get("flagged_rate")),
            }
        )
    return rec


def identity_errors(
    exp: tm.ExpectedLeaf, cell: RosterCell, manifest: Mapping, max_steps: int
) -> list[str]:
    """Disagreements between a present leaf's manifest and the roster."""
    errs: list[str] = []

    def _check(name: str, got: object, want: object) -> None:
        if got != want:
            errs.append(f"{exp.path}: {name} {got!r} != {want!r}")

    _check("target_animal", manifest.get("target_animal"), cell.animal)
    _check("gkd_tag", manifest.get("gkd_tag"), cell.gkd_tag)
    _check("base_model_repo", manifest.get("base_model_repo"), cell.student.repo)
    if "gseed" in manifest:
        _check("gseed", manifest.get("gseed"), exp.gseed)
    is_control = exp.arm in (tm.ARM_SB, tm.ARM_TB)
    _check("control", bool(manifest.get("control")), is_control)
    kind = manifest.get("control_kind")
    if is_control and kind is not None:
        _check("control_kind", kind, tm.expected_control_kind(exp.arm))
    teacher = manifest.get("teacher") or {}
    if exp.arm == tm.ARM_TREATMENT:
        _check(
            "teacher_tag", manifest.get("teacher_tag"), Path(exp.path).parents[1].name
        )
        _check("teacher ftseed", teacher.get("ftseed"), exp.tfs)
    elif exp.arm == tm.ARM_PLACEBO:
        want = tm.placebo_teacher_tag(cell, max_steps, exp.tfs)
        _check("teacher_tag", manifest.get("teacher_tag"), want)
    elif exp.arm == tm.ARM_TB and teacher.get("base_model_repo") is not None:
        _check("teacher base_model_repo", teacher["base_model_repo"], cell.teacher.repo)
    return errs


def _read_all(
    paths_animals: Sequence[tuple[str, str]], workers: int
) -> dict[tuple[str, str], dict]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = pool.map(lambda pa: read_leaf(Path(pa[0]), pa[1]), paths_animals)
        return dict(zip(paths_animals, records, strict=True))


def assemble_leaves(
    roster: Roster, output_root: Path, *, workers: int = READ_WORKERS
) -> pd.DataFrame:
    """One row per (cell, expected leaf), missing leaves included.

    Raises:
        ValueError: A present leaf's manifest disagrees with the roster.
    """
    cells = {c.cell_id: c for c in roster.cells}
    expected = tm.expected_leaves(roster, output_root)
    keys = sorted({(e.path, cells[e.cell_id].animal) for e in expected})
    records = _read_all(keys, workers)
    rows: list[dict] = []
    errors: list[str] = []
    for exp in expected:
        cell = cells[exp.cell_id]
        rec = dict(records[(exp.path, cell.animal)])
        manifest = rec.pop("_manifest")
        if manifest is not None:
            errors += identity_errors(
                exp, cell, manifest, roster.teacher_for(cell).max_steps
            )
        rows.append(
            {
                "cell_id": exp.cell_id,
                "family": exp.family,
                "arm": exp.arm,
                "tfs": exp.tfs,
                "gseed": exp.gseed,
                "extended_null": exp.extended_null,
                "new_leaf": exp.new_leaf,
                "primary_scope": exp.primary_scope,
                "run_set": exp.run_set,
                "path": exp.path,
                "self_pair": cell.self_pair,
                "gseed_paired": tm.source_family(cell) in tm.GSEED_PAIRED_FAMILIES,
                **rec,
            }
        )
    if errors:
        raise ValueError(
            f"{len(errors)} leaf identity mismatches:\n" + "\n".join(errors[:20])
        )
    return tm.ensure_columns(pd.DataFrame(rows), LEAF_VALUE_COLUMNS)


# ---- teachers ------------------------------------------------------------------------


def _teacher_status(ftseed_dir: Path) -> str:
    if not ftseed_dir.is_dir():
        return tm.STATUS_MISSING
    if not (ftseed_dir / TRAIN_MANIFEST).is_file():
        return tm.STATUS_NOT_TRAINED
    if not (ftseed_dir / PREFERENCE_EVAL).is_file():
        return tm.STATUS_NOT_EVALUATED
    return tm.STATUS_EVALUATED


def read_teacher_seed(ftseed_dir: Path, animal: str, param_tag: str) -> dict:
    """p_target (test / prob), FC share, perplexity and train stats of one ftseed.

    Raises:
        ValueError: The manifest names another animal or param tag.
    """
    ftseed_dir = Path(ftseed_dir)
    rec: dict = {"status": _teacher_status(ftseed_dir)}
    manifest = read_json(ftseed_dir / TRAIN_MANIFEST)
    if manifest is not None:
        if manifest.get("animal") != animal or manifest.get("param_tag") != param_tag:
            raise ValueError(
                f"{ftseed_dir}: manifest ({manifest.get('animal')}, "
                f"{manifest.get('param_tag')}) != ({animal}, {param_tag})"
            )
        rec["final_train_loss"] = _num(manifest.get("final_train_loss"))
        rec["n_train_steps"] = _num(manifest.get("n_train_steps"))
    pref = read_json(ftseed_dir / PREFERENCE_EVAL)
    if pref is not None:
        rec["p_target"] = _preference(pref, animal, ftseed_dir).get(
            f"p_target_{PRIMARY_SET}", math.nan
        )
    metrics = read_json(ftseed_dir / METRICS_EVAL)
    if metrics is not None:
        rec["fc_target_share"] = _forced_choice(metrics, animal, ftseed_dir).get(
            "fc_target_share", math.nan
        )
        rec["ppl"] = _num(
            ((metrics.get("perplexity") or {}).get("result") or {}).get("overall")
        )
    return rec


def read_teacher_base(base_dir: Path, animal: str) -> dict:
    """Untuned-base p_target (test / prob), FC share and perplexity, or {}.

    A verify ladder without a ``test`` set leaves ``base_p_target`` out.
    """
    base_dir = Path(base_dir)
    control = read_json(base_dir / BASE_CONTROL)
    metrics = read_json(base_dir / BASE_METRICS)
    if control is None and metrics is None:
        return {}
    out: dict = {"base_source": str(base_dir)}
    entry = (control or {}).get(PRIMARY_SET)
    if entry is not None:
        if (
            entry.get("target_animal") != animal
            or entry.get("scoring") != PRIMARY_SCORING
        ):
            raise ValueError(
                f"{base_dir}: base control is not {animal} {PRIMARY_SCORING}"
            )
        out["base_p_target"] = _num(entry.get("p_target"))
    if metrics is not None:
        fc = _forced_choice(metrics, animal, base_dir)
        out["base_fc_target_share"] = fc.get("fc_target_share", math.nan)
        out["base_ppl"] = _num(
            ((metrics.get("perplexity") or {}).get("result") or {}).get("overall")
        )
    return out


def _param_dir_base(param_dir: Path, animal: str) -> dict:
    """Base files beside a teacher param dir, following a staged symlink."""
    for base_dir in dict.fromkeys((param_dir.parent, param_dir.resolve().parent)):
        found = read_teacher_base(base_dir, animal)
        if found:
            return found
    return {}


def assemble_teachers(roster: Roster, output_root: Path) -> pd.DataFrame:
    """One row per (placebo teacher, ftseed) and per source trait teacher ftseed.

    A placebo teacher reads its own base files where present and its source
    teacher's (same base@revision) otherwise; ``base_source`` records which.
    Trait teachers whose verify ladder has no ``test`` set (breadth4 recipe)
    keep ``p_target`` NA.

    Raises:
        ValueError: A present placebo teacher's step count differs from the plan.
    """
    rows: list[dict] = []
    errors: list[str] = []
    for teacher in roster.teachers:
        param_dir = tm.placebo_teacher_param_dir(output_root, teacher)
        source_dirs = {
            t: Path(p).parent for t, p in sorted(teacher.source.manifest_paths.items())
        }
        source_param = next(iter(source_dirs.values())).parent if source_dirs else None
        source_base = (
            _param_dir_base(source_param, teacher.slot_animal) if source_param else {}
        )
        own_base = _param_dir_base(param_dir, teacher.slot_animal)
        shared = {
            "family": teacher.family,
            "slot_animal": teacher.slot_animal,
            "base": teacher.base.model_dir,
            "base_repo": teacher.base.repo,
            "max_steps": teacher.max_steps,
        }
        for ftseed in teacher.ftseeds:
            ftseed_dir = param_dir / f"ftseed{ftseed}"
            rec = read_teacher_seed(ftseed_dir, teacher.slot_animal, param_dir.name)
            steps = rec.get("n_train_steps", math.nan)
            if math.isfinite(steps) and int(steps) != teacher.max_steps:
                errors.append(
                    f"{ftseed_dir}: n_train_steps {steps} != {teacher.max_steps}"
                )
            rows.append(
                {
                    **shared,
                    "kind": tm.KIND_PLACEBO,
                    "param_tag": param_dir.name,
                    "run_set": plan.RUN_SET_ID[teacher.family],
                    "ftseed": ftseed,
                    "path": str(ftseed_dir),
                    **source_base,
                    **own_base,
                    **rec,
                }
            )
        for ftseed, ftseed_dir in source_dirs.items():
            rec = read_teacher_seed(
                ftseed_dir, teacher.slot_animal, teacher.source.param_tag
            )
            rows.append(
                {
                    **shared,
                    "kind": tm.KIND_TRAIT,
                    "param_tag": teacher.source.param_tag,
                    "run_set": teacher.source.run_set,
                    "ftseed": ftseed,
                    "path": str(ftseed_dir),
                    **source_base,
                    **rec,
                }
            )
    if errors:
        raise ValueError("placebo teacher step mismatches:\n" + "\n".join(errors[:20]))
    frame = tm.ensure_columns(pd.DataFrame(rows), TEACHER_VALUE_COLUMNS)
    frame["p_target"] = frame["p_target"].astype(float)
    frame["perplexity_ratio"] = frame["ppl"].astype(float) / frame["base_ppl"].astype(
        float
    )
    return frame


# ---- axes + aggregates -------------------------------------------------------------


def assemble_axes(roster: Roster) -> pd.DataFrame:
    """`tm.CELL_AXIS_COLUMNS` plus the parsed GKD recipe axes of every cell."""
    rows = []
    for cell in roster.cells:
        gkd = parse_gkd_tag(cell.gkd_tag)
        rows.append({**tm.cell_axes(cell), **{k: gkd[k] for k in GKD_AXES}})
    frame = pd.DataFrame(rows, columns=[*tm.CELL_AXIS_COLUMNS, *GKD_AXES])
    steps = ["teacher_step", "student_step"]
    frame[steps] = frame[steps].astype("Int64")
    return frame


def family_aggregates(
    leaves: pd.DataFrame,
    cells: pd.DataFrame,
    axes: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Figure-level tables per family, each under every control (primary metric)."""
    deltas = tm.leaf_deltas(leaves, cells).merge(axes, on="cell_id", how="left")
    long = tm.cells_long(cells).merge(axes, on="cell_id", how="left")
    kw = {"n_boot": n_boot, "seed": seed}

    def _of(frame: pd.DataFrame, family: str) -> pd.DataFrame:
        return frame[frame["family"] == family]

    screened = leaves.merge(axes[["cell_id", "condition"]], on="cell_id", how="left")
    screened = screened[screened["family"] == plan.FAMILY_ROLLOUTSCREEN]
    return {
        "agg_conditions_summary.csv": tm.cluster_summary(
            tm.with_pooled(_of(deltas, plan.FAMILY_CONDITIONS), "animal"),
            ["animal", "condition"],
            **kw,
        ),
        "agg_conditions_extended_null.csv": tm.extended_null_summary(
            leaves[leaves["family"] == plan.FAMILY_CONDITIONS], **kw
        ),
        "agg_dose_curve.csv": tm.cluster_summary(
            tm.with_pooled(_of(deltas, plan.FAMILY_DOSE), "animal"),
            ["animal", "num_epochs"],
            **kw,
        ),
        "agg_crossmodel_cells.csv": _of(long, plan.FAMILY_CROSSMODEL),
        "agg_topk_cells.csv": _of(long, plan.FAMILY_TOPK),
        "agg_checkpoints_cells.csv": _of(long, plan.FAMILY_CHECKPOINTS),
        "agg_checkpoints_panels.csv": tm.cluster_summary(
            tm.explode_panels(_of(deltas, plan.FAMILY_CHECKPOINTS)),
            ["panel", "teacher_step", "student_step"],
            **kw,
        ),
        "agg_rolloutscreen_cells.csv": _of(long, plan.FAMILY_ROLLOUTSCREEN),
        "agg_rolloutscreen_removal.csv": tm.removal_share(
            tm.with_pooled(screened, "condition"), ["condition"]
        ),
        "agg_rolloutscreen_removal_cells.csv": tm.removal_share(screened, ["cell_id"]),
    }


# ---- build -----------------------------------------------------------------------------


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _subset(roster: Roster, families: Sequence[str] | None) -> Roster:
    if not families:
        return roster
    unknown = set(families) - set(plan.FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}")
    keep = set(families)
    return Roster(
        cells=tuple(c for c in roster.cells if c.family in keep),
        teachers=tuple(t for t in roster.teachers if t.family in keep),
        provenance=roster.provenance,
    )


def build_results(
    *,
    roster_path: Path = plan.ROSTER_PATH,
    out_root: Path | None = None,
    families: Sequence[str] | None = None,
    output_root: Path | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
) -> Path:
    """Write every results table into a new ``build-<UTC>`` dir.

    Args:
        roster_path: Frozen roster JSON.
        out_root: Parent of the build dir (default `default_out_root`).
        families: Restrict to these families (default: all).
        output_root: Where this campaign's new leaves live (default: plan).
        n_boot: Bootstrap replicates.
        seed: Bootstrap seed.

    Returns:
        The build dir.

    Raises:
        FileExistsError: The build dir already exists.
        ValueError: The roster is invalid or a present leaf / teacher disagrees
            with it.
    """
    roster, roster_sha = load_roster(Path(roster_path))
    validate_roster(roster)
    roster = _subset(roster, families)
    output_root = Path(output_root or plan.paths_config_doc()["output_root"])
    out_root = Path(out_root) if out_root is not None else default_out_root()
    stamp = _stamp()
    dest = out_root / f"build-{stamp}"
    tmp = out_root / f".build-{stamp}.partial"
    if dest.exists() or tmp.exists():
        raise FileExistsError(dest)
    started = utc_now_iso()

    logger.info(
        "traitfree results: {} cells, {} placebo teachers",
        len(roster.cells),
        len(roster.teachers),
    )
    axes = assemble_axes(roster)
    leaves = assemble_leaves(roster, output_root)
    metrics = tm.metric_columns(leaves)
    cells = tm.cell_contrasts(leaves, metrics, n_boot=n_boot, seed=seed)
    teachers = tm.teacher_labels(assemble_teachers(roster, output_root))
    aggregates = family_aggregates(leaves, cells, axes, n_boot=n_boot, seed=seed)
    cover = tm.coverage(leaves, teachers)

    leaf_axes = axes[
        ["cell_id", "animal", "teacher", "student", "gkd_tag", "condition"]
    ]
    tables: dict[str, pd.DataFrame] = {
        "leaves.csv": leaves.merge(leaf_axes, on="cell_id", how="left"),
        "cells.csv": axes.merge(cells, on="cell_id", how="right"),
        "teachers.csv": teachers,
        **aggregates,
    }
    tmp.mkdir(parents=True, exist_ok=False)
    for name, frame in tables.items():
        atomic_write_text(tmp / name, frame.to_csv(index=False))
    atomic_write_json(tmp / "coverage.json", cover)
    atomic_write_json(
        tmp / "manifest.json",
        {
            "campaign": plan.CAMPAIGN,
            "roster_path": str(roster_path),
            "roster_sha256": roster_sha,
            "code_sha": git_sha(),
            "build_started": started,
            "build_finished": utc_now_iso(),
            "families": sorted({c.family for c in roster.cells}),
            "output_root": str(output_root),
            "n_boot": n_boot,
            "bootstrap_seed": seed,
            "confidence": tm.CONFIDENCE,
            "metrics": metrics,
            "arms_evaluated": dict(tm.arms_present(leaves)),
            "files": {name: int(len(frame)) for name, frame in tables.items()},
        },
    )
    if dest.exists():
        raise FileExistsError(dest)
    tmp.rename(dest)
    logger.info(
        "traitfree results -> {} ({} leaves, {} incomplete cells)",
        dest,
        len(leaves),
        cover["n_incomplete_cells"],
    )
    return dest
