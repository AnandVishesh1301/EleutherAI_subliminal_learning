"""Union the ws160m partition trees into leaf frames, result tables and claims.

Leaves are read from every partition's ``gkd_students/<run_set>`` tree (treatments,
``base_teacher`` and ``xbase_teacher`` controls) and teachers from the teacher trees;
identity comes from the step-bearing directory labels and the teacher tags, never
from arm names. ``gen=1`` unions the four Part I trees into the `main` run set's
``results/``; ``gen=2`` unions Part I and Part II (six trees, every animal) into the
`matrix` run set's ``results/`` and never writes under a Part I run set; ``gen=3``
unions Parts I-III into the `rx1` run set's ``results/`` and adds the rung-cross
tables and claims W17-W24 (`ws160m_rungcross`), reading the reused ckptgrid / grid2
cells from those campaigns' results. A leaf present in two run sets raises (prereg
Part II invariant 3).
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
from subliminal_learning.campaign import ws160m_rungcross as rc
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


def results_run_set_id(gen: int) -> str:
    """Run set whose ``results/`` holds generation `gen`'s tables (`main` / `matrix` / `rx1`)."""
    if gen == 1:
        return plan.MAIN_RUN_SET_ID
    if gen == 2:
        return plan.MATRIX_RUN_SET_ID
    if gen == 3:
        return plan.GEN3_RESULTS_RUN_SET_ID
    raise ValueError(f"gen must be 1, 2 or 3, got {gen!r}")


def results_dir(output_root: Path, *, gen: int = 1) -> Path:
    """``{output_root}/gkd_students/<results run set>/results``."""
    return (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, results_run_set_id(gen))
        / "results"
    )


def partitions_of(gen: int) -> tuple[str, ...]:
    """Partitions whose trees generation `gen` unions."""
    results_run_set_id(gen)
    return tuple(n for g in range(1, gen + 1) for n in plan.partition_order(g))


def animals_of(gen: int) -> tuple[str, ...]:
    """Animals generation `gen` reads."""
    results_run_set_id(gen)
    extra = tuple(a for a in plan.GEN2_ANIMALS if a not in plan.ANIMALS)
    return plan.ANIMALS + (extra if gen >= 2 else ())


def _leaf_index(gen: int) -> dict[tuple, tuple[str, int]]:
    """``leaf_id -> (arm, replicate)`` of every leaf a generation enumerates."""
    wave = models.WAVE_2 if gen == 1 else models.WAVE_3
    return {
        pairs.leaf_id(lf): (lf.cell.arm, lf.replicate)
        for lf in pairs.leaves_of(pairs.all_cells(wave=wave, gen=gen))
    }


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
    index: Mapping[tuple, tuple[str, int]],
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
        cell_key = (
            animal,
            teacher_base,
            student_base,
            teacher_step,
            student_step,
            gkd_tag,
            ptag,
        )
        shared = {
            "source_run_set": run_set_id,
            "partition": partition,
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
            arm, replicate = (None, None)
            if is_control:
                arm = "control"
            elif ptag is not None:
                arm, replicate = index.get((*cell_key, ftseed, gseed), (None, None))
            rows.append(
                {
                    **shared,
                    "arm": arm,
                    "gseed": gseed,
                    "replicate": replicate,
                    "p_target": value,
                }
            )
    return rows


_LEAF_IDENTITY = (
    *metrics.CELL_KEYS,
    "ftseed",
    "gseed",
    "set",
    "scoring",
    "is_control",
    "control_kind",
)


def assert_leaves_disjoint(leaves: pd.DataFrame) -> None:
    """Raise if one leaf identity appears in more than one run set.

    Raises:
        ValueError: Naming the first repeated identities and their run sets.
    """
    if leaves.empty:
        return
    keys = [c for c in _LEAF_IDENTITY if c in leaves.columns]
    filled = leaves[keys].astype(object).where(leaves[keys].notna(), "")
    sets = leaves.groupby([filled[k] for k in keys], sort=False)["source_run_set"]
    dup = sets.nunique()
    dup = dup[dup > 1]
    if not dup.empty:
        examples = [
            (k, sorted(set(leaves.loc[sets.groups[k], "source_run_set"])))
            for k in dup.index[:3]
        ]
        raise ValueError(
            f"{len(dup)} leaf identities appear in more than one run set: {examples}"
        )


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
    partitions: Sequence[str] | None = None,
    animals: Sequence[str] | None = None,
    gen: int = 1,
) -> pd.DataFrame:
    """One row per leaf x prompt set across every partition tree of generation `gen`.

    Raises:
        ValueError: If a leaf identity is present in two run sets.
    """
    partitions = partitions_of(gen) if partitions is None else tuple(partitions)
    animals = animals_of(gen) if animals is None else tuple(animals)
    indices = {g: _leaf_index(g) for g in {plan.partition(n).gen for n in partitions}}
    frames: list[pd.DataFrame] = []
    for name in partitions:
        part = plan.partition(name)
        rs = part.run_set_id
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, rs)
        pref = _load(
            load_gkd_preference_results, output_root, subdir, animals=list(animals)
        )
        if pref.empty:
            continue
        leaves = pd.DataFrame(
            _leaf_rows(pref, run_set_id=rs, partition=name, index=indices[part.gen])
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
    assert_leaves_disjoint(out)
    out = metrics.mark_quarantine(out)
    out["pair_id"] = [
        metrics.pair_id(t, s)
        for t, s in zip(out["teacher_base"], out["student_base"], strict=True)
    ]
    for col in metrics.LEAF_COLUMNS:
        if col not in out:
            out[col] = None
    return out


def teacher_run_set_ids(gen: int) -> tuple[str, ...]:
    """Teacher trees generation `gen` reads (`main`, `ladder`; plus both Part II sets).

    Part III trains no teacher, so generation 3 reads the Part II set.
    """
    results_run_set_id(gen)
    out = (plan.MAIN_RUN_SET_ID, plan.LADDER_RUN_SET_ID)
    if gen >= 2:
        out += (plan.MATRIX_RUN_SET_ID, plan.GRID_RUN_SET_ID)
    return out


def assemble_teachers(
    output_root: Path, *, animals: Sequence[str] | None = None, gen: int = 1
) -> pd.DataFrame:
    """One row per (teacher config, ftseed, prompt set) from every teacher tree."""
    animals = animals_of(gen) if animals is None else tuple(animals)
    rows: list[dict] = []
    for rs in teacher_run_set_ids(gen):
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


# ---- Part II frames ------------------------------------------------------------------


def diagonal_cell_delta(
    leaves: pd.DataFrame, teacher_summary: pd.DataFrame, *, strict: bool = True
) -> pd.DataFrame:
    """Cell deltas from rotated-diagonal leaves (and every control), both stat sets."""
    diagonal = leaves[
        leaves["is_control"].astype(bool)
        | (
            pd.to_numeric(leaves["replicate"], errors="coerce").fillna(0)
            < plan.LEAVES_DEFAULT
        )
    ]
    return pd.concat(
        [
            metrics.build_cell_delta(
                diagonal, teacher_summary, stat_set=stat_set, strict=strict
            )
            for stat_set in metrics.STAT_SETS
        ],
        ignore_index=True,
    )


def _ladder_k2(diag_cells: pd.DataFrame, *, n_boot: int, seed: int) -> pd.DataFrame:
    """W14 ladder: nine rungs x the 61 K2 pairs on rotated-diagonal leaves only."""
    frames = [
        metrics.build_ladder_by_category(
            diag_cells[diag_cells["stat_set"] == stat_set],
            rungs=plan.K2_LINE_RUNGS,
            pairs=pairs.k2_directed_pairs(),
            n_boot=n_boot,
            seed=seed,
        )
        for stat_set in metrics.STAT_SETS
    ]
    return pd.concat(frames, ignore_index=True)


_EXTERNAL_SOURCES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (plan.CKPTGRID_RUN_SET_ID, "grid_delta.csv", ("grid", "joint")),
    (plan.CKPTGRID2_RUN_SET_ID, "rect_delta.csv", ("rect",)),
)


def load_external_cells(output_root: Path) -> pd.DataFrame:
    """Agg cells of the reused ckptgrid / grid2 grids (bear, frozen recipe, primary)."""
    frames = []
    for run_set_id, fname, arms in _EXTERNAL_SOURCES:
        path = (
            Path(output_root)
            / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
            / "results"
            / fname
        )
        if not path.is_file():
            logger.warning("reused grid cells missing: {}", path)
            continue
        df = pd.read_csv(path)
        df = df[(df["level"] == "agg") & df["arm"].isin(arms)]
        for col, want in (
            ("animal", plan.PRIMARY_ANIMAL),
            ("set", plan.PRIMARY_PROMPT_SET),
            ("scoring", plan.PRIMARY_SCORING),
            ("gkd_tag", plan.FROZEN_GKD_TAG),
            ("teacher_param_tag", plan.TEACHER_PARAM_TAG_LORA),
            ("source_run_set", run_set_id),
        ):
            if col in df:
                df = df[df[col] == want]
        base = {**{m.base: m.base for m in models.ROSTER}, **plan.EXTERNAL_BASE_MAP}
        df = df.assign(
            teacher_base=df["teacher_base"].map(base),
            student_base=df["student_base"].map(base),
            teacher_step=df["teacher_step"].astype(int),
            student_step=df["student_step"].astype(int),
            source_run_set=run_set_id,
        ).dropna(subset=["teacher_base", "student_base"])
        frames.append(
            df[
                [
                    "teacher_base",
                    "student_base",
                    "teacher_step",
                    "student_step",
                    "delta",
                    "p_target",
                    "control_mean",
                    "n_leaves",
                    "source_run_set",
                ]
            ]
        )
    if not frames:
        return pd.DataFrame()
    return rc.external_cells(pd.concat(frames, ignore_index=True))


def rungcross_tables(
    output_root: Path,
    diag_cells: pd.DataFrame,
    cell_delta: pd.DataFrame,
    teacher_summary: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Part III tables keyed by `plan.RESULTS_RUNGCROSS_CSVS` and claims W17-W24."""
    ours = rc.frozen_cells(diag_cells)
    external = load_external_cells(output_root)
    concordance = rc.reuse_concordance(ours, external)
    gate = rc.reuse_gate(concordance)
    cells = rc.build_cells(ours, external, teacher_summary, use_external=gate["passes"])
    grid_stats = rc.build_grid_stats(cells)
    k2 = set(pairs.k2_directed_pairs())
    in_k2 = pd.Series(
        [
            (t, s) in k2
            for t, s in zip(ours["teacher_base"], ours["student_base"], strict=True)
        ],
        index=ours.index,
        dtype=bool,
    )
    same = ours[
        (ours["stat_set"] == "primary")
        & (ours["teacher_step"] == ours["student_step"])
        & in_k2
    ]
    tables = {
        "cells": cells,
        "category_grid": rc.build_category_grid(cells, n_boot=n_boot, seed=seed),
        "grid_stats": grid_stats,
        "ridge": rc.build_ridge(cells),
        "onset": rc.build_onset(cells, n_boot=n_boot, seed=seed),
        "direction_diag": rc.build_direction_diag(same, n_boot=n_boot, seed=seed),
        "direction_grid": rc.build_direction_grid(cells),
        "efficiency": rc.build_efficiency(cells, n_boot=n_boot, seed=seed),
        "teacher_dose": rc.build_teacher_dose(cell_delta),
        "reuse": concordance,
    }
    claims = rc.adjudicate_part3(
        cells=cells,
        grid_stats=grid_stats,
        onset=tables["onset"],
        direction_diag=tables["direction_diag"],
        direction_grid=tables["direction_grid"],
        efficiency=tables["efficiency"],
        teacher_dose=tables["teacher_dose"],
        reuse=gate,
        n_boot=n_boot,
        seed=seed,
    )
    return tables, claims


def _heatmaps(cell_delta: pd.DataFrame, *, gen: int) -> pd.DataFrame:
    """Part I: the `pile-ws1` student map; Parts II-III: both anchors, both directions."""
    if cell_delta.empty:
        return pd.DataFrame()
    if gen == 1:
        return metrics.build_heatmap_cells(cell_delta)
    frames = [
        metrics.build_heatmap_cells(
            cell_delta, rungs=plan.HM2_RUNGS, anchor=anchor, reverse=reverse
        )
        for anchor in plan.HM2_ANCHORS
        for reverse in (False, True)
    ]
    return pd.concat(frames, ignore_index=True)


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
    gen: int = 1,
) -> list[Path]:
    """Write every result table and ``claims.json``; returns the written paths.

    Category summaries, contrasts, asymmetry and the recipe table read same-rung cells
    only (teacher and student at one step); rung-cross cells feed the heat maps alone.
    """
    out_dir = results_dir(output_root, gen=gen)
    leaves_df = (
        assemble_leaves(output_root, gen=gen) if leaves is None else leaves.copy()
    )
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
        assemble_teachers(output_root, gen=gen) if teachers is None else teachers.copy()
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
            strict=not partial,
        )
        cell_frames.append(cd)
        if cd.empty:
            continue
        same = cd[cd["teacher_step"] == cd["student_step"]]
        summary_frames.append(
            metrics.build_category_summary(same, n_boot=n_boot, seed=seed)
        )
        summary_frames.append(
            metrics.build_category_summary(
                same, n_boot=n_boot, seed=seed, teacher_ok_only=True
            )
        )
        contrast_frames.append(
            metrics.build_category_contrasts(same, n_boot=n_boot, seed=seed)
        )
        asym_frames.append(metrics.build_direction_asymmetry(same))
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
    same_rung = (
        cell_delta[cell_delta["teacher_step"] == cell_delta["student_step"]]
        if not cell_delta.empty
        else cell_delta
    )
    hp = (
        metrics.build_hp_by_category(same_rung, leaves=leaves_df)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    ladder = (
        metrics.build_ladder_by_category(cell_delta, n_boot=n_boot, seed=seed)
        if not cell_delta.empty
        else pd.DataFrame()
    )
    diag_cells = (
        diagonal_cell_delta(leaves_df, teacher_summary, strict=not partial)
        if gen >= 2 and not cell_delta.empty
        else pd.DataFrame()
    )
    ladder_k2 = (
        _ladder_k2(diag_cells, n_boot=n_boot, seed=seed)
        if not diag_cells.empty
        else pd.DataFrame()
    )
    rc_tables, claims_part3 = (
        rungcross_tables(
            output_root,
            diag_cells,
            cell_delta,
            teacher_summary,
            n_boot=n_boot,
            seed=seed,
        )
        if gen == 3 and not diag_cells.empty
        else ({}, [])
    )
    heat = _heatmaps(cell_delta, gen=gen)
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
        **({plan.RESULTS_LADDER_K2_CSV: ladder_k2} if gen >= 2 else {}),
        **{plan.RESULTS_RUNGCROSS_CSVS[k]: v for k, v in rc_tables.items()},
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
                    strict=not partial,
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
    no_control = metrics.missing_controls(leaves_df) if not leaves_df.empty else []
    if no_control:
        logger.warning(
            "{} cells are absent from cell_delta for want of a control "
            "(listed in claims.json cells_without_control)",
            len(no_control),
        )
    payload = {
        "at": utc_now_iso(),
        "campaign": {1: plan.CAMPAIGN, 2: plan.GEN2_CAMPAIGN, 3: plan.GEN3_CAMPAIGN}[
            gen
        ],
        "gen": gen,
        "run_set_ids": [plan.partition(n).run_set_id for n in partitions_of(gen)],
        "primary_endpoint": {"prompt_set": primary[0], "scoring": primary[1]},
        "bootstrap": {
            "n_boot": n_boot,
            "seed": seed,
            "min_pairs_bootstrap": plan.MIN_PAIRS_BOOTSTRAP,
        },
        "partial": partial,
        "cells_without_control": metrics.jsonable(no_control),
        "coverage": dict(coverage or {}),
        "n_leaves": int(len(leaves_df)),
        "n_cells": int(cell_delta[cell_delta["stat_set"] == "primary"].shape[0])
        if not cell_delta.empty
        else 0,
        "claims": metrics.jsonable(claims),
        **({"claims_part3": metrics.jsonable(claims_part3)} if gen == 3 else {}),
    }
    claims_path = out_dir / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, payload)
    written.append(claims_path)
    return written


def partition_coverage(output_root: Path, *, gen: int = 1) -> dict[str, object]:
    """Every partition's ``coverage.json`` (written at finalize), keyed by partition."""
    import json

    out: dict[str, object] = {}
    for name in partitions_of(gen):
        p = plan.partition(name)
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
