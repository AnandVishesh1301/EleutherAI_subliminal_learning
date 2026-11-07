"""Read `rebsoft` queues, leaves, teachers and panels and write a per-priority build.

Output: `<teacher run set>/results/p<N>-<UTC>/` (create-only, manifest last), covering
every block of priority N or earlier, so it can run while later priorities train:

    leaves.csv       one row per submitted leaf: status, p per eval set, forced choice,
                     perplexity ratio, mean filter-drop fraction, quarantine
    panel.csv        one row per (leaf, set, panel item): exact next-token p
    teachers.csv     one row per (target, base, kind): dose, steps, reuse, W0, p and base
    cells.csv        one row per pooled cell: delta, intervals, verdict, pooled-control
                     estimate (secondary), R7-R9 columns
    blocks.csv       per block and target: verdict counts and the block statement
    anchors.csv      reused 160M->160M rows, read in place, screen labelled
    quarantine.csv   every quarantined leaf and why
    claims.json      R0-R9 with counts and caveats
    manifest.json    file shas, counts, anomalies; written last

Queue reads are pure file reads (`queue_exists`, `load_jobs`, `load_results`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines import rebsoft_staging_pipeline as staging
from pipelines.newdom_soft_results_pipeline import DONE, FAILED, PENDING, _mean_drop
from pipelines.optparam2_results_pipeline import _num, _p, _read
from pipelines.rebsoft_campaign_pipeline import unit_key
from subliminal_learning.campaign import newdom2_plan as nd2
from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import newdom_soft_metrics as nd_metrics
from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_metrics as metrics
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, gkd_student_run_dir
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

ANCHOR = "anchor"
ANCHOR_ROOT = soft.OUTPUT_ROOT
SUMMARY_FILE = "preference_summary.json"
STAGED_MANIFEST_SHA = "train_manifest_sha256"  # reuse.json records of the first staging
OLMO_SCOPE = "olmo"
ANOMALY_KEYS_SHOWN = 20
FIELD_COLUMNS: tuple[str, ...] = (
    "run_dir",
    "rollout_screen",
    "trained",
    "evaluated",
    "panel_done",
    *(f"p_{s}" for s in plan.GENERIC_SETS),
    *(f"base_p_{s}" for s in plan.GENERIC_SETS),
    "fc_target_share",
    "perplexity",
    "perplexity_base",
    "perplexity_ratio",
    "filter_drop_frac",
    "rollout_flagged_rate",
    "n_train_steps",
    "final_train_loss",
)
LEAF_COLUMNS: tuple[str, ...] = (
    "key",
    "exp",
    "item",
    "priority",
    "kind",
    "domain",
    "target",
    "teacher",
    "student",
    "registry",
    "mode",
    "lr",
    "screen",
    "teacher_token",
    "teacher_epochs",
    "teacher_steps",
    "r",
    "g",
    "status",
    "counted",
    *FIELD_COLUMNS,
)
PANEL_COLUMNS: tuple[str, ...] = ("key", "set", "item", "p")
QUARANTINE_COLUMNS: tuple[str, ...] = (
    "key",
    "exp",
    "kind",
    "target",
    "teacher",
    "student",
    "mode",
    "lr",
    "r",
    "g",
    "status",
    "perplexity_ratio",
    "filter_drop_frac",
    "quarantine_reason",
)
TEACHER_COLUMNS: tuple[str, ...] = (
    "target",
    "domain",
    "base",
    "registry",
    "kind",
    "priority",
    "reused",
    "source",
    "param_tag",
    "chosen_epochs",
    "steps",
    "w0_epochs_tried",
    "w0_passed",
    "w0_labelled",
    "w0_p_teacher",
    "w0_base_p",
    "w0_base_sd",
    "w0_threshold",
    "placebo_steps_differ",
    "n_ftseeds",
    "p_validation",
    "base_p_validation",
    "minus_base_validation",
    "p_test",
    "base_p_test",
    "minus_base_test",
    "placebo_within_tol",
)


def registered_leaves(
    queue_root: Path, exps: Sequence[plan.Exp], layout: jobs.Layout
) -> list[tuple[jobs.Leaf, str]]:
    """(leaf, status) of every leaf in a submitted queue of `exps`, in key order."""
    out = []
    for exp in exps:
        for target in exp.targets:
            for kind in plan.KINDS:
                name = plan.q_leaves(exp, target, kind)
                if not store.queue_exists(queue_root, name):
                    continue
                status = {
                    r["key"]: DONE if int(r.get("rc", 1)) == 0 else FAILED
                    for r in store.load_results(queue_root, name)
                }
                for job in store.load_jobs(queue_root, name):
                    leaf = jobs.leaf_from_key(job.key, layout)
                    out.append((leaf, status.get(job.key, PENDING)))
    return sorted(out, key=lambda x: x[0].key)


def run_fields(run_dir: Path, target: str) -> dict:
    """What a student's run dir holds (NaN for whatever is not there yet)."""
    domain = plan.TARGET_DOMAIN[target]
    manifest = _read(run_dir / "train_manifest.json") or {}
    sets = (_read(run_dir / "preference_eval.json") or {}).get("sets") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    rollouts = _read(run_dir / "rollouts_summary.json") or {}
    base = _read(run_dir.parents[1] / "base_control.json") or {}
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    names = {s: plan.eval_set_name(domain, s) for s in plan.GENERIC_SETS}
    return {
        "run_dir": str(run_dir),
        "rollout_screen": (manifest.get("rollout_screen") or {}).get("mode", ""),
        "trained": bool(manifest),
        "evaluated": bool(sets),
        "panel_done": (run_dir / jobs.PANEL_FILE).is_file(),
        **{f"p_{s}": _p(sets.get(n), target) for s, n in names.items()},
        **{f"base_p_{s}": _p(base.get(n), target) for s, n in names.items()},
        "fc_target_share": _num(fc),
        "perplexity": ppl,
        "perplexity_base": ppl_base,
        "perplexity_ratio": ppl / ppl_base if ppl_base > 0 else math.nan,
        "filter_drop_frac": _mean_drop(run_dir),
        "rollout_flagged_rate": _num(rollouts.get("flagged_rate")),
        "n_train_steps": manifest.get("n_train_steps"),
        "final_train_loss": _num(manifest.get("final_train_loss")),
    }


def leaf_row(leaf: jobs.Leaf, status: str) -> dict:
    """One leaf's row."""
    exp = plan.EXPERIMENTS_BY_ID[leaf.exp]
    return {
        "key": leaf.key,
        "exp": leaf.exp,
        "item": exp.item,
        "priority": exp.priority,
        "kind": leaf.kind,
        "domain": leaf.domain,
        "target": leaf.target,
        "teacher": leaf.teacher.base,
        "student": leaf.student,
        "registry": leaf.registry,
        "mode": leaf.mode,
        "lr": leaf.lr,
        "screen": leaf.screen,
        "teacher_token": leaf.teacher.token,
        "teacher_epochs": leaf.teacher.epochs,
        "teacher_steps": leaf.teacher.steps,
        "r": leaf.r,
        "g": leaf.g,
        "status": status,
        "counted": status == DONE,
        **run_fields(jobs.leaf_dir(leaf), leaf.target),
    }


def panel_rows(leaf: jobs.Leaf) -> list[dict]:
    """(key, generic set, item, p) of the leaf's panel file (none if absent)."""
    doc = _read(jobs.leaf_dir(leaf) / jobs.PANEL_FILE)
    if not doc:
        return []
    out = []
    for s in plan.PANEL_SETS:
        means = (
            (doc.get("sets") or {}).get(plan.eval_set_name(leaf.domain, s)) or {}
        ).get("mean") or {}
        out += [
            {"key": leaf.key, "set": s, "item": item, "p": _num(means.get(item))}
            for item in plan.PANELS[leaf.domain]
        ]
    return out


# ---- anchors ----------------------------------------------------------------------------


def anchor_run_set(target: str) -> str:
    """Finished run set of the 160M->160M row under `target`'s rebsoft screen."""
    domain = nd.TARGET_DOMAIN[target]
    if plan.TARGET_SCREEN[target] == plan.SCREEN_SUB:
        return nd2.STUDENT_RUN_SETS[(domain, nd2.PART1)]
    return soft.STUDENT_RUN_SETS[domain]


def anchor_leaves(target: str) -> list[jobs.Leaf]:
    """The reused 160M->160M leaves of `target`: 3 x 3 treatment, 3 placebo.

    Seeds sort as `newdom`'s keys do: the seeded bootstrap depends on input order, and
    this order reproduces its published intervals.
    """
    out = []
    for kind, seeds in (
        (plan.KIND_TREAT, sorted((*plan.DIAGONAL, *plan.OFF_DIAGONAL))),
        (plan.KIND_PLACEBO, plan.DIAGONAL),
    ):
        teacher = jobs.teacher_of(plan.reused(kind, target, plan.WS_ANCHOR))
        out += [
            jobs.Leaf(
                ANCHOR, kind, teacher, plan.WS_ANCHOR, plan.LORA, plan.LORA_LR, r, g
            )
            for r, g in seeds
        ]
    return out


def anchor_dir(leaf: jobs.Leaf, output_root: Path) -> Path:
    """Run dir of a reused leaf under its finished run set."""
    return gkd_student_run_dir(
        output_root,
        leaf.target,
        plan.base_spec(leaf.student),
        plan.HUB_ORG,
        jobs.leaf_teacher_tag(leaf),
        jobs.leaf_gkd_tag(leaf),
        leaf.g,
        run_set_subdir(GKD_STUDENTS_SUBDIR, anchor_run_set(leaf.target)),
    )


def anchor_row(leaf: jobs.Leaf, output_root: Path) -> dict:
    """One reused leaf's row; finished run sets have no queue, so an eval counts it."""
    fields = run_fields(anchor_dir(leaf, output_root), leaf.target)
    return {
        "key": fields["run_dir"],
        "exp": ANCHOR,
        "kind": leaf.kind,
        "target": leaf.target,
        "teacher": leaf.teacher.base,
        "student": leaf.student,
        "mode": leaf.mode,
        "lr": leaf.lr,
        "r": leaf.r,
        "g": leaf.g,
        "status": DONE if fields["evaluated"] else PENDING,
        "counted": fields["evaluated"],
        "source_run_set": anchor_run_set(leaf.target),
        **fields,
    }


# ---- teachers ---------------------------------------------------------------------------


def _unit_ref(
    kind: str, target: str, base: str, chosen: Mapping, layout: jobs.Layout
) -> jobs.TeacherRef | None:
    """The teacher config of one unit and kind; None before its dose is chosen."""
    row = plan.reused(kind, target, base)
    if row is not None:
        return replace(jobs.teacher_of(row), layout=layout)
    if not chosen:
        return None
    steps = int(chosen["steps"]) if kind == plan.KIND_PLACEBO else None
    return jobs.TeacherRef(kind, target, base, int(chosen["epochs"]), steps, layout)


def _summary_entries(ref: jobs.TeacherRef | None) -> tuple[dict, int]:
    """(prompt set -> prob entry, ftseeds) of `ref`'s summary; empty when it is absent
    or names another target (reused lion placebos come from a bear slot)."""
    doc = _read(ref.param_dir / SUMMARY_FILE) if ref else None
    if not doc or doc.get("target_animal") != ref.target:
        return {}, 0
    entries = {
        e["prompt_set"]: e
        for e in doc.get("summaries", [])
        if e.get("scoring") == "prob"
    }
    return entries, len(doc.get("ft_seeds", []))


def _unit_steps(target: str, base: str, chosen: Mapping) -> tuple[float, float]:
    """(treatment, placebo) steps of a unit; NaN for a new config before its dose."""
    treat = plan.reused(plan.KIND_TREAT, target, base)
    placebo = plan.reused(plan.KIND_PLACEBO, target, base)
    fresh = _num(chosen.get("steps"))
    return treat.steps if treat else fresh, placebo.steps if placebo else fresh


def teacher_row(
    target: str, base: str, kind: str, state: Mapping, layout: jobs.Layout
) -> dict:
    """One (target, base, kind) row: dose, steps, reuse, W0 and p against base."""
    name = unit_key(target, base)
    unit = (state.get("units") or {}).get(name) or {}
    chosen = unit.get("chosen") or {}
    reused = plan.reused(kind, target, base)
    ref = _unit_ref(kind, target, base, chosen, layout)
    domain = plan.TARGET_DOMAIN[target]
    tried = (unit.get("w0") or {}) if kind == plan.KIND_TREAT else {}
    w0 = tried.get(str(ref.epochs)) or {} if ref else {}
    entries, n_ftseeds = _summary_entries(ref)
    treat_steps, placebo_steps = _unit_steps(target, base, chosen)
    known = math.isfinite(treat_steps) and math.isfinite(placebo_steps)
    row = {
        "target": target,
        "domain": domain,
        "base": base,
        "registry": plan.registry_of(base),
        "kind": kind,
        "priority": plan.unit_priority(target, base),
        "reused": reused is not None,
        "source": str(reused.source) if reused else "",
        "param_tag": ref.param_tag if ref else "",
        "chosen_epochs": ref.epochs if ref else math.nan,
        "steps": treat_steps if kind == plan.KIND_TREAT else placebo_steps,
        "w0_epochs_tried": "+".join(sorted(tried, key=int)),
        "w0_passed": w0.get("passed"),
        "w0_labelled": f"w0_fail|{name}" in (state.get("labelled") or {}),
        "w0_p_teacher": _num(w0.get("p_teacher")),
        "w0_base_p": _num(w0.get("base_p")),
        "w0_base_sd": _num(w0.get("base_sd")),
        "w0_threshold": _num(w0.get("threshold")),
        "placebo_steps_differ": treat_steps != placebo_steps if known else None,
        "n_ftseeds": n_ftseeds,
    }
    for s in (plan.GATE_SET, plan.PRIMARY_SET):
        entry = entries.get(plan.eval_set_name(domain, s)) or {}
        row[f"p_{s}"] = _num(entry.get("p_target_mean"))
        row[f"base_p_{s}"] = _num(entry.get("base_p_target"))
        row[f"minus_base_{s}"] = row[f"p_{s}"] - row[f"base_p_{s}"]
    gap = row["minus_base_test"]
    row["placebo_within_tol"] = (
        abs(gap) <= nd_metrics.PLACEBO_BASE_TOL
        if kind == plan.KIND_PLACEBO and math.isfinite(gap)
        else None
    )
    return row


def teacher_rows(state: Mapping, priority: int, layout: jobs.Layout) -> list[dict]:
    """Every (target, base, kind) the blocks of `priority` distil from, in plan order."""
    units = dict.fromkeys(
        (c.target, c.teacher) for c in plan.cells(metrics.blocks_up_to(priority))
    )
    return [
        teacher_row(target, base, kind, state, layout)
        for target, base in units
        for kind in plan.KINDS
    ]


def reuse_status(layout: jobs.Layout) -> dict:
    """The staged reuse record against the plan.

    Each ftseed's train manifest is re-hashed; weights were hashed against their source
    by the staging copy and are not re-read here. A record carries that sha under the
    file's name, or under `STAGED_MANIFEST_SHA` from the first staging.
    """
    root = layout.root / plan.STAGING_DIR.name
    report = _read(root / staging.STAGING_REPORT) or {}
    records = (_read(root / staging.REUSE_REPORT) or {}).get("teachers") or []
    seen = {(r["kind"], r["target"], r["base"]): r for r in records}
    missing, mismatched = [], []
    for key, row in plan.REUSED_BY_KEY.items():
        if key not in seen:
            missing.append("|".join(key))
            continue
        ref = replace(jobs.teacher_of(row), layout=layout)
        for ftseed, shas in seen[key]["ftseeds"].items():
            path = ref.manifest_path(int(ftseed))
            want = shas.get(path.name) or shas.get(STAGED_MANIFEST_SHA)
            if not path.is_file() or file_sha256(path) != want:
                mismatched.append(str(path))
    return {
        "staging_passed": report.get("passed") is True,
        "n_planned": len(plan.REUSED),
        "n_recorded": len(seen),
        "missing": missing,
        "mismatched": mismatched,
    }


def review_flags(review_dir: Path) -> dict[str, bool | None]:
    """Each domain and OLMo -> `approved` of its rollout review; None when the file is
    missing, unreadable or has no boolean `approved` (R0 not yet satisfied)."""
    out: dict[str, bool | None] = {}
    for scope in (*plan.PROMPT_POOLS, OLMO_SCOPE):
        path = review_dir / f"rollouts_{scope}.agent_review.json"
        try:
            approved = (_read(path) or {}).get("approved")
        except (OSError, json.JSONDecodeError) as err:
            logger.warning("unreadable review {}: {}", path, err)
            approved = None
        out[scope] = approved if isinstance(approved, bool) else None
    return out


# ---- build ------------------------------------------------------------------------------


def anomalies(leaves: pd.DataFrame) -> dict[str, list[str]]:
    """Keys of finished leaves missing what a finished leaf always has."""
    done = leaves[leaves["counted"].astype(bool)]
    ppl = done["perplexity_ratio"].astype(float)
    drop = done["filter_drop_frac"].astype(float)
    found = {
        "done_without_eval": done.loc[~done["evaluated"].astype(bool), "key"],
        "done_without_panel": done.loc[~done["panel_done"].astype(bool), "key"],
        "quarantine_unknown": done.loc[ppl.isna() | drop.isna(), "key"],
    }
    out = {k: v.tolist() for k, v in found.items() if len(v)}
    for name, keys in out.items():
        logger.warning("{}: {} leaves, first {}", name, len(keys), keys[0])
    return {k: v[:ANOMALY_KEYS_SHOWN] for k, v in out.items()}


def tables(
    queue_root: Path,
    state: Mapping,
    priority: int,
    layout: jobs.Layout = jobs.LIVE,
    anchor_root: Path = ANCHOR_ROOT,
) -> dict[str, object]:
    """Every results table and the claims of `priority`, in memory.

    Raises:
        ValueError: Unknown priority, or two leaves with one identity.
    """
    pairs = registered_leaves(queue_root, metrics.blocks_up_to(priority), layout)
    leaves = nd_metrics.mark_quarantine(
        pd.DataFrame([leaf_row(lf, st) for lf, st in pairs], columns=list(LEAF_COLUMNS))
    )
    panel = pd.DataFrame(
        [row for lf, _ in pairs for row in panel_rows(lf)], columns=list(PANEL_COLUMNS)
    )
    teachers = pd.DataFrame(
        teacher_rows(state, priority, layout), columns=list(TEACHER_COLUMNS)
    )
    reused = nd_metrics.mark_quarantine(
        pd.DataFrame(
            [
                anchor_row(lf, anchor_root)
                for target in nd.TARGETS
                for lf in anchor_leaves(target)
            ]
        )
    )
    anchors = metrics.anchors_table(reused)
    cells = metrics.cells_table(leaves, panel, priority)
    selected = metrics.rate_selection(cells, anchors)
    cells = metrics.mark_selected(cells, selected)
    blocks = metrics.blocks_table(cells)
    flagged = pd.concat([leaves, reused], ignore_index=True)
    flagged = flagged[flagged["quarantined"].astype(bool)]
    claims = metrics.claims(
        cells,
        blocks,
        selected,
        teachers,
        anchors,
        reuse_status(layout),
        review_flags(layout.root / plan.REVIEW_DIR.name),
        priority,
    )
    return {
        "leaves": leaves,
        "panel": panel,
        "teachers": teachers,
        "cells": cells,
        "blocks": blocks,
        "anchors": anchors,
        "quarantine": flagged[list(QUARANTINE_COLUMNS)],
        "claims": claims,
    }


def build(
    queue_root: Path,
    state: Mapping,
    priority: int,
    out_root: Path,
    layout: jobs.Layout = jobs.LIVE,
    anchor_root: Path = ANCHOR_ROOT,
) -> Path:
    """Write a fresh build dir for `priority` (create-only) and return it.

    Raises:
        FileExistsError: The build dir already exists.
    """
    built = tables(queue_root, state, priority, layout, anchor_root)
    out = out_root / f"p{priority}-{utc_now_iso().replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    counts = {}
    for name, obj in built.items():
        if name == "claims":
            path = out / "claims.json"
            atomic_write_json(path, obj)
        else:
            path = out / f"{name}.csv"
            atomic_write_text(path, obj.to_csv(index=False))
            counts[name] = len(obj)
        files[path.name] = file_sha256(path)
    leaves = built["leaves"]
    atomic_write_json(
        out / "manifest.json",
        {
            "at": utc_now_iso(),
            "priority": priority,
            "blocks": [e.id for e in metrics.blocks_up_to(priority)],
            "queue_root": str(queue_root),
            "rows": counts,
            "leaf_status": {
                s: int((leaves["status"] == s).sum()) for s in (DONE, FAILED, PENDING)
            },
            "n_quarantined": int(leaves["quarantined"].astype(bool).sum()),
            "verdicts": built["claims"]["verdicts"],
            "anomalies": anomalies(leaves),
            "files": files,
        },
    )
    logger.info("rebsoft p{} results -> {}", priority, out)
    return out
