"""Union the four ws160m partition trees into leaf frames, result tables and claims.

Leaves are read from every partition's ``gkd_students/<run_set>`` tree (treatments,
``base_teacher`` and ``xbase_teacher`` controls) and teachers from the two teacher
trees; identity comes from the step-bearing directory labels and the teacher tags,
never from arm names. Tables land under the `main` run set's ``results/``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import ws160m_metrics as metrics
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_forced_choice_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
)
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_LEAF_JOIN = ("animal", "model", "teacher_tag", "gkd_tag")
_CKPTGRID2_BASE_MAP: Mapping[str, str] = {
    "original": "pythia-160m",
    "data-seed1": "data-seed1",
    "data-seed2": "data-seed2",
    "data-seed3": "data-seed3",
    "seed1": "seed1",
    "seed2": "seed2",
    "seed3": "seed3",
    "seed4": "seed4",
}


def parse_label(label: str) -> tuple[str, int]:
    """``<base>-step<N>`` -> ``(base, N)``.

    Raises:
        ValueError: On a label without a step suffix or with an unknown base.
    """
    head, sep, step = str(label).rpartition("-step")
    if not sep or not step.isdigit():
        raise ValueError(f"model label {label!r} carries no -step<N> suffix")
    models.base(head)
    return head, int(step)


def results_dir(output_root: Path) -> Path:
    """``{output_root}/gkd_students/<main run set>/results``."""
    return (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.MAIN_RUN_SET_ID)
        / "results"
    )


def _cell_index() -> dict[tuple, pairs.Cell]:
    return {c.key(): c for c in pairs.all_cells(wave=models.WAVE_2)}


def _per_seed(raw: object) -> list[tuple[int, float]]:
    if not isinstance(raw, Mapping):
        return []
    out = []
    for k, v in raw.items():
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(val):
            out.append((int(k), val))
    return sorted(out)


def _nan(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


# ---- leaves -------------------------------------------------------------------------


def _leaf_rows(
    pref: pd.DataFrame,
    *,
    run_set_id: str,
    partition: str,
    index: Mapping[tuple, pairs.Cell],
) -> list[dict]:
    rows: list[dict] = []
    for row in pref.to_dict("records"):
        animal = str(row["animal"])
        student_base, student_step = parse_label(str(row["model"]))
        is_control = bool(row.get("is_control"))
        kind = row.get("control_kind")
        kind = None if _nan(kind) else str(kind)
        tlabel = row.get("teacher_base_label")
        if _nan(tlabel):
            teacher_base, teacher_step = student_base, student_step
        else:
            teacher_base, teacher_step = parse_label(str(tlabel))
        gkd_tag = str(row["gkd_tag"])
        ptag = (
            None
            if _nan(row.get("teacher_param_tag"))
            else str(row.get("teacher_param_tag"))
        )
        ftseed = (
            None if _nan(row.get("teacher_ftseed")) else int(row.get("teacher_ftseed"))
        )
        category = None if is_control else models.category(teacher_base, student_base)
        cell = None
        if not is_control and ptag is not None:
            cell = index.get(
                (
                    animal,
                    teacher_base,
                    student_base,
                    teacher_step,
                    student_step,
                    gkd_tag,
                    ptag,
                )
            )
        assignments = {}
        if cell is not None:
            assignments = {
                fg: i
                for i, fg in enumerate(pairs.leaf_assignments(cell.index, cell.leaves))
            }
        shared = {
            "source_run_set": run_set_id,
            "partition": partition,
            "arm": cell.arm
            if cell is not None
            else ("control" if is_control else None),
            "animal": animal,
            "teacher_base": teacher_base
            if (not is_control or kind == "xbase_teacher")
            else student_base,
            "student_base": student_base,
            "teacher_step": teacher_step
            if (not is_control or kind == "xbase_teacher")
            else student_step,
            "student_step": student_step,
            "category": category,
            "impl_stratum": None
            if is_control
            else models.impl_stratum(teacher_base, student_base),
            "init_stratum": None
            if is_control
            else models.init_stratum(teacher_base, student_base),
            "corpus_stratum": None
            if is_control
            else models.corpus_stratum(teacher_base, student_base),
            "gkd_tag": gkd_tag,
            "recipe_slug": _recipe_slug(row),
            "num_epochs": row.get("num_epochs"),
            "teacher_param_tag": ptag,
            "teacher_epochs": None
            if _nan(row.get("teacher_n_epochs"))
            else int(row.get("teacher_n_epochs")),
            "ftseed": ftseed,
            "set": row["set"],
            "scoring": row["scoring"],
            "is_control": is_control,
            "control_kind": kind,
            "base_prior": row.get("base_p_target"),
            "model": row["model"],
            "teacher_tag": row.get("teacher_tag"),
        }
        for gseed, value in _per_seed(row.get("per_seed_p_target")):
            rows.append(
                {
                    **shared,
                    "gseed": gseed,
                    "replicate": assignments.get((ftseed, gseed)),
                    "p_target": value,
                }
            )
    return rows


def _recipe_slug(row: Mapping) -> str | None:
    lr, r, a = (
        row.get("student_lr"),
        row.get("student_lora_r"),
        row.get("student_lora_alpha"),
    )
    if _nan(lr) or _nan(r) or _nan(a):
        return None
    return f"lr{plan.lr_slug(float(lr))}_r{int(r)}a{int(a)}"


def _bench_columns(cap: pd.DataFrame) -> list[str]:
    """``bench_*`` / ``base_*`` / ``delta_*`` lm-eval columns present in `cap`."""
    out: list[str] = []
    for col in cap.columns:
        if col.startswith("bench_"):
            suffix = col[len("bench_") :]
            out += [col, f"base_{suffix}", f"delta_{suffix}"]
    return [c for c in out if c in cap.columns]


def _capability_leaves(cap: pd.DataFrame) -> pd.DataFrame:
    """Per-leaf perplexity rows; lm-eval columns (config-dir means) ride along."""
    bench_cols = _bench_columns(cap)
    rows: list[dict] = []
    for row in cap.to_dict("records"):
        base = row.get("ppl_base_overall")
        base_ce = math.log(float(base)) if base and float(base) > 0 else float("nan")
        for gseed, ppl in _per_seed(row.get("per_seed_ppl")):
            rows.append(
                {
                    **{k: row.get(k) for k in _LEAF_JOIN},
                    "gseed": gseed,
                    "perplexity_ratio": (ppl / float(base))
                    if base and float(base) > 0
                    else float("nan"),
                    "ppl_ce_nats_mean": math.log(ppl) if ppl > 0 else float("nan"),
                    "ppl_ce_nats_base": base_ce,
                    **{k: row.get(k) for k in bench_cols},
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            *_LEAF_JOIN,
            "gseed",
            "perplexity_ratio",
            "ppl_ce_nats_mean",
            "ppl_ce_nats_base",
            *bench_cols,
        ],
    )


def _forced_choice(fc: pd.DataFrame) -> pd.DataFrame:
    if fc.empty:
        return pd.DataFrame(columns=[*_LEAF_JOIN, "fc_target_share"])
    target = fc[fc["is_target"].astype(bool)]
    return (
        target[[*_LEAF_JOIN, "target_share_mean"]]
        .rename(columns={"target_share_mean": "fc_target_share"})
        .drop_duplicates(list(_LEAF_JOIN))
    )


def _filter_drop(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty or "filter_drop_frac" not in logs:
        return pd.DataFrame(columns=[*_LEAF_JOIN, "gseed", "filter_drop_frac"])
    return (
        logs.groupby([*_LEAF_JOIN, "gseed"], dropna=False)["filter_drop_frac"]
        .mean()
        .reset_index()
    )


def _load(loader, output_root: Path, subdir: str, **kw) -> pd.DataFrame:
    try:
        return loader(output_root, subdir, **kw)
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("{}({}): {}", loader.__name__, subdir, err)
        return pd.DataFrame()


def assemble_leaves(
    output_root: Path,
    *,
    partitions: Sequence[str] = plan.PARTITION_ORDER,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """One row per leaf x prompt set across every partition tree."""
    index = _cell_index()
    frames: list[pd.DataFrame] = []
    for name in partitions:
        rs = plan.partition(name).run_set_id
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, rs)
        pref = _load(
            load_gkd_preference_results, output_root, subdir, animals=list(animals)
        )
        if pref.empty:
            continue
        leaves = pd.DataFrame(
            _leaf_rows(pref, run_set_id=rs, partition=name, index=index)
        )
        if leaves.empty:
            continue
        cap = _capability_leaves(
            _load(
                load_gkd_capability_results, output_root, subdir, animals=list(animals)
            )
        )
        fc = _forced_choice(
            _load(
                load_gkd_forced_choice_results,
                output_root,
                subdir,
                animals=list(animals),
            )
        )
        drops = _filter_drop(
            _load(load_gkd_training_logs, output_root, subdir, animals=list(animals))
        )
        leaves = leaves.merge(cap, on=[*_LEAF_JOIN, "gseed"], how="left")
        leaves = leaves.merge(fc, on=list(_LEAF_JOIN), how="left")
        leaves = leaves.merge(drops, on=[*_LEAF_JOIN, "gseed"], how="left")
        frames.append(leaves)
    if not frames:
        return pd.DataFrame(columns=list(metrics.LEAF_COLUMNS))
    out = pd.concat(frames, ignore_index=True)
    out = metrics.mark_quarantine(out)
    out["pair_id"] = [
        metrics.pair_id(t, s)
        for t, s in zip(out["teacher_base"], out["student_base"], strict=True)
    ]
    for col in metrics.LEAF_COLUMNS:
        if col not in out:
            out[col] = None
    return out


def assemble_teachers(
    output_root: Path, *, animals: Sequence[str] = plan.ANIMALS
) -> pd.DataFrame:
    """One row per (teacher config, ftseed, prompt set) from both teacher trees."""
    rows: list[dict] = []
    for rs in (plan.MAIN_RUN_SET_ID, plan.LADDER_RUN_SET_ID):
        subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, rs)
        pref = _load(
            load_sdf_preference_results, output_root, subdir, animals=list(animals)
        )
        if pref.empty:
            continue
        cap = _load(
            load_sdf_capability_results, output_root, subdir, animals=list(animals)
        )
        ratio = {}
        if not cap.empty and "perplexity_ratio" in cap:
            ratio = {
                (r["animal"], r["model"], r["param_tag"]): r["perplexity_ratio"]
                for r in cap.to_dict("records")
            }
        for row in pref.to_dict("records"):
            base, step = parse_label(str(row["model"]))
            shared = {
                "source_run_set": rs,
                "animal": row["animal"],
                "base": base,
                "step": step,
                "param_tag": row["param_tag"],
                "n_epochs": row.get("n_epochs"),
                "set": row["set"],
                "scoring": row["scoring"],
                "base_prior": row.get("base_p_target"),
                "perplexity_ratio": ratio.get(
                    (row["animal"], row["model"], row["param_tag"])
                ),
            }
            for ftseed, value in _per_seed(row.get("per_seed_p_target")):
                rows.append({**shared, "ftseed": ftseed, "p_target": value})
    return pd.DataFrame(rows)


# ---- concordance ---------------------------------------------------------------------


def ckptgrid2_concordance(output_root: Path, cell_delta: pd.DataFrame) -> pd.DataFrame:
    """Join our 143k frozen-recipe cells with ckptgrid2's ``rect_delta.csv`` (if present)."""
    path = (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.CKPTGRID2_RUN_SET_ID)
        / "results"
        / "rect_delta.csv"
    )
    if not path.is_file() or cell_delta.empty:
        return pd.DataFrame(
            columns=[
                "teacher_base",
                "student_base",
                "teacher_step",
                "student_step",
                "delta_ws160m",
                "delta_ckptgrid2",
            ]
        )
    ref = pd.read_csv(path)
    if "level" in ref:
        ref = ref[ref["level"] == "agg"]
    ref = (
        ref[
            (ref.get("set", "test") == plan.PRIMARY_PROMPT_SET)
            & (ref.get("scoring", "prob") == plan.PRIMARY_SCORING)
        ]
        if "set" in ref
        else ref
    )
    ref = ref[ref["gkd_tag"] == plan.FROZEN_GKD_TAG] if "gkd_tag" in ref else ref
    ref = ref.assign(
        teacher_base=ref["teacher_base"].map(_CKPTGRID2_BASE_MAP),
        student_base=ref["student_base"].map(_CKPTGRID2_BASE_MAP),
    ).dropna(subset=["teacher_base", "student_base"])
    ours = cell_delta[
        (cell_delta["stat_set"] == "primary")
        & (cell_delta["gkd_tag"] == plan.FROZEN_GKD_TAG)
        & (cell_delta["animal"] == plan.PRIMARY_ANIMAL)
    ]
    keys = ["teacher_base", "student_base", "teacher_step", "student_step"]
    merged = ours[[*keys, "delta"]].merge(
        ref[[*keys, "delta"]].rename(columns={"delta": "delta_ckptgrid2"}),
        on=keys,
        how="inner",
    )
    return merged.rename(columns={"delta": "delta_ws160m"})


# ---- build ---------------------------------------------------------------------------


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {} ({} rows)", path, len(frame))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_results(
    output_root: Path,
    *,
    dry_run: bool = False,
    partial: bool = False,
    leaves: pd.DataFrame | None = None,
    teachers: pd.DataFrame | None = None,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    coverage: Mapping[str, object] | None = None,
) -> list[Path]:
    """Write every result table and ``claims.json``; returns the written paths."""
    out_dir = results_dir(output_root)
    leaves_df = assemble_leaves(output_root) if leaves is None else leaves.copy()
    if leaves is not None and "quarantined" not in leaves_df:
        leaves_df = metrics.mark_quarantine(leaves_df)
    if leaves is not None and "pair_id" not in leaves_df:
        leaves_df["pair_id"] = [
            metrics.pair_id(t, s)
            for t, s in zip(
                leaves_df["teacher_base"], leaves_df["student_base"], strict=True
            )
        ]
    teachers_df = (
        assemble_teachers(output_root) if teachers is None else teachers.copy()
    )
    primary = (plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
    ctrl_stats = (
        metrics.control_means(
            leaves_df, prompt_set=primary[0], scoring=primary[1], stat_set="full"
        )
        if not leaves_df.empty
        else None
    )
    ctrl_frozen = (
        ctrl_stats[ctrl_stats["gkd_tag"] == plan.FROZEN_GKD_TAG]
        if ctrl_stats is not None and not ctrl_stats.empty
        else None
    )
    teacher_summary = (
        metrics.build_teacher_summary(teachers_df, ctrl_frozen)
        if not teachers_df.empty
        else pd.DataFrame(columns=list(metrics.TEACHER_COLUMNS))
    )

    cell_frames, summary_frames, contrast_frames, asym_frames = [], [], [], []
    for stat_set in metrics.STAT_SETS:
        if leaves_df.empty:
            break
        cd = metrics.build_cell_delta(
            leaves_df,
            teacher_summary,
            prompt_set=primary[0],
            scoring=primary[1],
            stat_set=stat_set,
        )
        cell_frames.append(cd)
        if cd.empty:
            continue
        summary_frames.append(
            metrics.build_category_summary(cd, n_boot=n_boot, seed=seed)
        )
        summary_frames.append(
            metrics.build_category_summary(
                cd, n_boot=n_boot, seed=seed, teacher_ok_only=True
            )
        )
        contrast_frames.append(
            metrics.build_category_contrasts(cd, n_boot=n_boot, seed=seed)
        )
        asym_frames.append(metrics.build_direction_asymmetry(cd))
    cell_delta = (
        pd.concat(cell_frames, ignore_index=True)
        if cell_frames
        else pd.DataFrame(columns=list(metrics.CELL_DELTA_COLUMNS))
    )
    summary = (
        pd.concat(summary_frames, ignore_index=True)
        if summary_frames
        else pd.DataFrame(columns=list(metrics.SUMMARY_COLUMNS))
    )
    contrasts = (
        pd.concat(contrast_frames, ignore_index=True)
        if contrast_frames
        else pd.DataFrame(columns=list(metrics.CONTRAST_COLUMNS))
    )
    asym = (
        pd.concat(asym_frames, ignore_index=True)
        if asym_frames
        else pd.DataFrame(columns=list(metrics.ASYMMETRY_COLUMNS))
    )
    strata = summary[summary["stratum_kind"] != "all"] if not summary.empty else summary
    dose = (
        metrics.build_dose_curve(leaves_df) if not leaves_df.empty else pd.DataFrame()
    )
    hp = (
        metrics.build_hp_by_category(cell_delta, leaves=leaves_df)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    ladder = (
        metrics.build_ladder_by_category(cell_delta, n_boot=n_boot, seed=seed)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    heat = (
        metrics.build_heatmap_cells(cell_delta)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    xbase = (
        metrics.build_xbase_vs_base(leaves_df, cell_delta)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    quarantine = (
        metrics.build_quarantine(leaves_df) if not leaves_df.empty else pd.DataFrame()
    )
    concordance = ckptgrid2_concordance(output_root, cell_delta)
    claims = metrics.adjudicate_claims(
        teacher_summary=teacher_summary,
        cell_delta=cell_delta,
        summary=summary,
        contrasts=contrasts,
        dose_curve=dose,
        hp_by_category=hp,
        ladder=ladder,
        xbase=xbase,
    )

    tables = {
        plan.RESULTS_LEAVES_CSV: leaves_df,
        plan.RESULTS_CELL_DELTA_CSV: cell_delta,
        plan.RESULTS_CATEGORY_SUMMARY_CSV: summary,
        plan.RESULTS_CATEGORY_CONTRASTS_CSV: contrasts,
        plan.RESULTS_STRATA_CSV: strata,
        plan.RESULTS_DIRECTION_ASYMMETRY_CSV: asym,
        plan.RESULTS_DOSE_CURVE_CSV: dose,
        plan.RESULTS_HP_BY_CATEGORY_CSV: hp,
        plan.RESULTS_LADDER_BY_CATEGORY_CSV: ladder,
        plan.RESULTS_HEATMAP_CELLS_CSV: heat,
        plan.RESULTS_XBASE_VS_BASE_CSV: xbase,
        plan.RESULTS_TEACHER_SUMMARY_CSV: teacher_summary,
        plan.RESULTS_QUARANTINE_CSV: quarantine,
        plan.RESULTS_CONCORDANCE_CSV: concordance,
    }
    written = [
        _write_csv(out_dir / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    ]
    if not leaves_df.empty:
        for (sset, scoring), _ in leaves_df.groupby(["set", "scoring"]):
            if (sset, scoring) == primary:
                continue
            try:
                cd = metrics.build_cell_delta(
                    leaves_df,
                    teacher_summary,
                    prompt_set=sset,
                    scoring=scoring,
                    stat_set="primary",
                )
            except ValueError as err:
                logger.warning("robustness {}/{}: {}", sset, scoring, err)
                continue
            rob = (
                metrics.build_category_summary(cd, n_boot=n_boot, seed=seed)
                if not cd.empty
                else cd
            )
            written.append(
                _write_csv(
                    out_dir / plan.RESULTS_ROBUSTNESS_DIR / f"{sset}_{scoring}.csv",
                    rob,
                    dry_run=dry_run,
                )
            )
    payload = {
        "at": utc_now_iso(),
        "campaign": plan.CAMPAIGN,
        "run_set_ids": [p.run_set_id for p in plan.PARTITIONS.values()],
        "primary_endpoint": {"prompt_set": primary[0], "scoring": primary[1]},
        "bootstrap": {
            "n_boot": n_boot,
            "seed": seed,
            "min_pairs_bootstrap": plan.MIN_PAIRS_BOOTSTRAP,
        },
        "partial": partial,
        "coverage": dict(coverage or {}),
        "n_leaves": int(len(leaves_df)),
        "n_cells": int(cell_delta[cell_delta["stat_set"] == "primary"].shape[0])
        if not cell_delta.empty
        else 0,
        "claims": metrics.jsonable(claims),
    }
    claims_path = out_dir / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, payload)
    written.append(claims_path)
    return written


def partition_coverage(output_root: Path) -> dict[str, object]:
    """Every partition's ``coverage.json`` (written at finalize), keyed by partition."""
    import json

    out: dict[str, object] = {}
    for name, p in plan.PARTITIONS.items():
        path = (
            Path(output_root)
            / run_set_subdir(GKD_STUDENTS_SUBDIR, p.run_set_id)
            / "logs"
            / plan.RESULTS_COVERAGE_JSON
        )
        if path.is_file():
            try:
                out[name] = json.loads(path.read_text())
            except ValueError:
                out[name] = {"error": "unreadable"}
        else:
            out[name] = {"missing": True}
    return out
