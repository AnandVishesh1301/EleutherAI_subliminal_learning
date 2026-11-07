"""Pure metric math of `rebsoft`: cells, verdicts, rate selection, block statements and
claims R0-R9 (`docs/PREREG_REBSOFT.md`).

Frame contracts (built by `pipelines.rebsoft_results_pipeline`):

- leaves: one row per leaf with ``key, exp, kind, target, teacher, student, mode, lr, r,
  g``, ``p_<set>`` and ``base_p_<set>`` for every generic eval set,
  ``perplexity_ratio``, ``filter_drop_frac``, ``counted`` (its job finished with rc 0)
  and `newdom_soft_metrics.mark_quarantine`'s columns;
- panel: one row per (leaf key, generic set, panel item) with ``p``;
- teachers: one row per (target, base, kind) with ``reused, chosen_epochs, w0_passed,
  w0_labelled, minus_base_test, placebo_steps_differ``.

A cell pools every leaf of one (target, teacher, student, mode, lr), so e1s, e2s and e4s
add seeds to the cells of e1, e2 and e4 and each pooled cell is attributed to the first
of its blocks, its family. Only counted, unquarantined leaves enter a mean or an
interval. A cell's denominators are planned, so a pending cell counts as `INSUFFICIENT`,
never as absent.

A cell with one placebo leaf is never `SUPPORTED`: its interval only restates the sign
result, so it is blank here and the cell can reach `POSITIVE_1_CONTROL`. The ``pooled_*``
columns are the secondary estimate against the family's placebo leaves of that target,
student and arm across teacher bases.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType

import numpy as np
import pandas as pd

from subliminal_learning.campaign import newdom_soft_metrics as nd_metrics
from subliminal_learning.campaign import rebsoft_plan as plan

CellKey = tuple[str, str, str, str, float]

SUPPORTED = nd_metrics.SUPPORTED
POSITIVE_1_CONTROL = "positive_1_control"
NOT_SUPPORTED = nd_metrics.NOT_SUPPORTED
UNDERPOWERED = nd_metrics.UNDERPOWERED
INSUFFICIENT = nd_metrics.INSUFFICIENT
DESCRIPTIVE = nd_metrics.DESCRIPTIVE
NOT_IN_BUILD = "not_in_this_build"
VERDICTS: tuple[str, ...] = (
    SUPPORTED,
    POSITIVE_1_CONTROL,
    NOT_SUPPORTED,
    UNDERPOWERED,
    INSUFFICIENT,
)
EFFECT_FLOOR = nd_metrics.EFFECT_FLOOR
SECONDARY = "secondary"
CELL_KEYS: tuple[str, ...] = ("target", "teacher", "student", "mode", "lr")
POOL_KEYS: tuple[str, ...] = ("family", "target", "student", "mode", "lr")
LEAF_IDENTITY: tuple[str, ...] = (*CELL_KEYS, "kind", "r", "g")
ANCHOR_TO_PARTNER, PARTNER_TO_ANCHOR = "anchor_to_partner", "partner_to_anchor"
CLAIM_BLOCKS: MappingProxyType[str, tuple[str, ...]] = MappingProxyType(
    {
        "R1": ("e3", "e3y"),
        "R2": ("e1",),
        "R3": ("e4",),
        "R4": ("e5",),
        "R5": ("e9", "e9y"),
        "R6": ("e2", "e2o"),
    }
)
CAVEATS: tuple[str, ...] = (
    "A cell with one placebo leaf has no interval worth quoting and is never called "
    "supported. positive_1_control is a sign result: with 3 teachers every teacher "
    "beats the placebo by chance 1 time in 4 under no effect (1 in 3 if only 2 "
    "contribute).",
    "pooled_* columns are secondary: the control arm pools placebo leaves of the same "
    "target, student and arm across the teacher bases of the block family, and those "
    "placebo teachers differ in base.",
    "No multiplicity correction: the rule is not calibrated at 3 teachers, so a "
    "supported cell is never quoted as a 95% claim.",
    "A null cell is reported as a null, and is not evidence of no transfer at other "
    "doses.",
    "A failed rollout review stops the claims of its domain or of OLMo, whatever a "
    "cell or block reads; see R0.stopped_scopes.",
    "Screens: strict_v2 for animals, domain strict for trees and colors, strict_sub for "
    "pine. Figure A25 (xfam2) and the 160M full-FT numbers (optparam2) used the legacy "
    "rollout screen, so rebsoft rows are not screen-matched to them. Trees and colors "
    "at 160M->160M (newdom) share rebsoft's screen.",
    "#5 has no 160M->160M row under strict_v2, so 'full FT matches LoRA at 160M' still "
    "rests on the legacy-screen optparam2 numbers.",
    "Lion placebos at 410M, 1B and OLMo are copies of the bear-slot "
    "lighthouse-traitfree-v1 teachers, 750 steps against lion's 765.",
    "ws160m students use micro-batch 16 x accumulation 2 against newdom's 32 x 1; the "
    "effective batch of 32 and the optimizer are unchanged.",
    "pile_prompts_v1 was built from the legacy vocabulary, not strict.",
    "Rosters differ in shared_vocab: xfam2 cells mask ids >= 50,277, ws160m cells use "
    "full logits.",
)

_DELTA_FIELDS: tuple[str, ...] = (
    "treat_mean",
    "placebo_mean",
    "delta",
    "ci_low",
    "ci_high",
    "p_boot",
    "ci_low_leaf",
    "ci_high_leaf",
    "p_boot_leaf",
    "seed_delta",
    "seed_ci_low",
    "seed_ci_high",
    "n_seeds",
    "n_treat",
    "n_placebo",
    "n_teachers",
    "n_positive_teachers",
    "min_teacher_delta",
    "n_leaves",
    "n_quarantined",
    "n_quarantine_unknown",
    "eligible",
    "verdict",
    "base_p",
    "treat_minus_base",
    "placebo_minus_base",
    "r9_holds",
    "generalizes",
    "val_delta",
    "val_ci_low",
    "val_ci_high",
    "off_topic_delta",
    "off_topic_ci_low",
    "off_topic_ci_high",
    "perplexity_ratio_treat",
    "perplexity_ratio_placebo",
)
CELL_COLUMNS: tuple[str, ...] = (
    "block",
    "blocks_planned",
    "blocks_contributing",
    "priority",
    "item",
    "domain",
    "target",
    "teacher",
    "student",
    "registry",
    "cross_model",
    "relation",
    "direction",
    "mode",
    "lr",
    "screen",
    "n_treat_planned",
    "n_placebo_planned",
    "complete",
    *_DELTA_FIELDS,
    "pooled_role",
    "pooled_placebo_mean",
    "pooled_n_placebo",
    "pooled_delta",
    "pooled_ci_low",
    "pooled_ci_high",
    "counts_for_block",
    "spec_status",
    "spec_target_gain",
    "spec_is_largest",
    "spec_positive_gain_sum",
    "spec_target_share",
)
BLOCK_COLUMNS: tuple[str, ...] = (
    "block",
    "item",
    "priority",
    "target",
    "n_cells",
    "n_complete",
    *(f"n_{v}" for v in VERDICTS),
    "n_cross_model",
    "n_cross_counting",
    "n_cross_counting_pooled",
    "n_cross_r9",
    "block_statement",
    "statement_uses_pooled_control",
)
ANCHOR_COLUMNS: tuple[str, ...] = (
    "target",
    "domain",
    "source_run_set",
    "screen",
    "rebsoft_screen",
    "same_screen",
    "comparability",
    *_DELTA_FIELDS,
)
SELECTED_COLUMNS: tuple[str, ...] = (
    "block",
    "target",
    "teacher",
    "student",
    "screen",
    "n_rates",
    "n_rates_eligible",
    "selected_lr",
    "validation_delta",
    "test_delta",
    "ci_low",
    "ci_high",
    "ci_low_leaf",
    "ci_high_leaf",
    "seed_ci_low",
    "seed_ci_high",
    "n_treat",
    "n_placebo",
    "n_teachers",
    "n_quarantined",
    "verdict",
    "generalizes",
    "lora_delta",
    "lora_verdict",
    "lora_source",
    "lora_screen",
    "lora_same_screen",
    "full_minus_lora",
)


# ---- cells ------------------------------------------------------------------------------


def blocks_up_to(priority: int) -> tuple[plan.Exp, ...]:
    """Blocks a build for `priority` covers, in plan order.

    Raises:
        ValueError: Unknown priority.
    """
    if priority not in plan.PRIORITIES:
        raise ValueError(f"unknown priority {priority}")
    return tuple(e for e in plan.EXPERIMENTS if e.priority <= priority)


def planned_groups(priority: int) -> dict[CellKey, tuple[plan.Cell, ...]]:
    """Pooled cell key -> its planned cells, one per contributing block, in plan order."""
    groups: dict[CellKey, list[plan.Cell]] = {}
    for cell in plan.cells(blocks_up_to(priority)):
        key = (cell.target, cell.teacher, cell.student, cell.mode, cell.lr)
        groups.setdefault(key, []).append(cell)
    return {k: tuple(v) for k, v in groups.items()}


def check_no_double_count(leaves: pd.DataFrame) -> None:
    """Refuse two leaf rows with one identity.

    Raises:
        ValueError: A (cell, kind, r, g) appears twice.
    """
    twins = leaves[leaves.duplicated(list(LEAF_IDENTITY), keep=False)]
    if not twins.empty:
        raise ValueError(f"leaves counted twice: {twins['key'].tolist()[:10]}")


def cell_delta(
    treat: Sequence[float], placebo: Sequence[float], teachers: Sequence[object]
) -> dict:
    """`newdom_soft_metrics.two_arm_delta` with undefined intervals blank.

    Blank means one placebo leaf (the bound is the lowest teacher minus the placebo) or
    zero width (equal values, where the bootstrap reports ``p_boot == 0``).
    """
    out = nd_metrics.two_arm_delta(treat, placebo, teachers)
    for low, high, p in (
        ("ci_low", "ci_high", "p_boot"),
        ("ci_low_leaf", "ci_high_leaf", "p_boot_leaf"),
    ):
        if out["n_placebo"] == 1 or out[low] == out[high]:
            out.update(dict.fromkeys((low, high, p), math.nan))
    return out


def cell_verdict(row: Mapping) -> str:
    """The prereg cell rule; one placebo leaf drops the interval test and the name."""
    if row["n_placebo"] != 1:
        return nd_metrics.transfer_verdict(row)
    verdict = nd_metrics.transfer_verdict({**row, "ci_low": math.inf})
    return POSITIVE_1_CONTROL if verdict == SUPPORTED else verdict


def _mean(rows: pd.DataFrame, column: str) -> float:
    values = rows[column].astype(float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def _primary(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(counted leaves, those not quarantined)."""
    done = rows[rows["counted"].astype(bool)]
    return done, done[~done["quarantined"].astype(bool)]


def _unknown_quarantine(done: pd.DataFrame) -> int:
    ppl = done["perplexity_ratio"].astype(float)
    drop = done["filter_drop_frac"].astype(float)
    return int((~(np.isfinite(ppl) & np.isfinite(drop))).sum())


def cell_stats(rows: pd.DataFrame) -> dict:
    """Prereg statistics of one pooled leaf set on test, validation and off-topic."""
    done, kept = _primary(rows)
    treat = kept[kept["kind"] == plan.KIND_TREAT]
    placebo = kept[kept["kind"] == plan.KIND_PLACEBO]
    teachers = list(treat["r"])
    n_q = len(done) - len(kept)
    out = {
        **cell_delta(treat["p_test"], placebo["p_test"], teachers),
        **nd_metrics.seed_interval(treat, placebo, "p_test"),
        "n_leaves": len(done),
        "n_quarantined": n_q,
        "n_quarantine_unknown": _unknown_quarantine(done),
        "eligible": bool(len(done)) and n_q <= nd_metrics.RUNG_EXCLUDED_MAX * len(done),
        "perplexity_ratio_treat": _mean(treat, "perplexity_ratio"),
        "perplexity_ratio_placebo": _mean(placebo, "perplexity_ratio"),
    }
    for generic, prefix in (("validation", "val"), ("off_topic", "off_topic")):
        col = f"p_{generic}"
        side = cell_delta(treat[col], placebo[col], teachers)
        out[f"{prefix}_delta"] = side["delta"]
        out[f"{prefix}_ci_low"] = side["ci_low"]
        out[f"{prefix}_ci_high"] = side["ci_high"]
    out["base_p"] = _mean(done, "base_p_test")
    out["treat_minus_base"] = out["treat_mean"] - out["base_p"]
    out["placebo_minus_base"] = out["placebo_mean"] - out["base_p"]
    out["verdict"] = cell_verdict(out)
    out["r9_holds"] = bool(out["treat_minus_base"] >= EFFECT_FLOOR)
    out["generalizes"] = out["verdict"] == SUPPORTED and out["r9_holds"]
    return out


def pooled_control(treat: pd.DataFrame, controls: pd.DataFrame) -> dict:
    """Secondary estimate on test: `treat` against the family's pooled placebo leaves."""
    pooled = cell_delta(treat["p_test"], controls["p_test"], list(treat["r"]))
    return {
        "pooled_role": SECONDARY,
        "pooled_placebo_mean": pooled["placebo_mean"],
        "pooled_n_placebo": pooled["n_placebo"],
        "pooled_delta": pooled["delta"],
        "pooled_ci_low": pooled["ci_low"],
        "pooled_ci_high": pooled["ci_high"],
    }


def counts_for_block(row: Mapping) -> bool:
    """A cell counts toward the block statement: supported, or positive (1 control)
    with its pooled-control interval above 0."""
    if row["verdict"] == POSITIVE_1_CONTROL:
        return bool(row["pooled_ci_low"] > 0)
    return row["verdict"] == SUPPORTED


def cell_gains(
    panel: pd.DataFrame, kept: pd.DataFrame, target: str, lr: float
) -> pd.DataFrame:
    """Per panel item on test: mean treatment p - mean placebo p over `kept` leaves."""
    rows = panel[panel["set"] == plan.PRIMARY_SET]
    treat = kept.loc[kept["kind"] == plan.KIND_TREAT, "key"]
    placebo = kept.loc[kept["kind"] == plan.KIND_PLACEBO, "key"]
    out = []
    for item in plan.PANELS[plan.TARGET_DOMAIN[target]]:
        sub = rows[rows["item"] == item]
        gain = (
            sub.loc[sub["key"].isin(treat), "p"].astype(float).mean()
            - sub.loc[sub["key"].isin(placebo), "p"].astype(float).mean()
        )
        out.append(
            {
                "target": target,
                "lr": lr,
                "item": item,
                "is_target": item == target,
                "gain": gain,
            }
        )
    return pd.DataFrame(out)


def relation_of(teacher: str, student: str) -> tuple[str, str]:
    """(relation, direction) of an ablation pair; empty strings for any other pair."""
    if teacher == plan.WS_ANCHOR and student in plan.WS_PARTNERS:
        return plan.WS_PARTNERS[student], ANCHOR_TO_PARTNER
    if student == plan.WS_ANCHOR and teacher in plan.WS_PARTNERS:
        return plan.WS_PARTNERS[teacher], PARTNER_TO_ANCHOR
    return "", ""


def _cell_row(
    key: CellKey,
    planned: Sequence[plan.Cell],
    rows: pd.DataFrame,
    controls: pd.DataFrame,
    panel: pd.DataFrame,
) -> dict:
    target, teacher, student, mode, lr = key
    head = planned[0]
    ids = [c.exp.id for c in planned]
    rows = rows[rows["exp"].isin(ids)]
    stats = cell_stats(rows)
    _, kept = _primary(rows)
    kept = kept[np.isfinite(kept["p_test"].astype(float))]
    relation, direction = relation_of(teacher, student)
    n_treat = sum(len(c.exp.treat_seeds) for c in planned)
    n_placebo = sum(len(c.exp.placebo_seeds) for c in planned)
    spec = (
        nd_metrics.specificity_verdict(cell_gains(panel, kept, target, lr), target)
        if stats["verdict"] == SUPPORTED
        else {}
    )
    pooled = pooled_control(kept[kept["kind"] == plan.KIND_TREAT], controls)
    return {
        "block": head.exp.id,
        "blocks_planned": "+".join(ids),
        "blocks_contributing": "+".join(i for i in ids if i in set(kept["exp"])),
        "priority": head.exp.priority,
        "item": head.exp.item,
        "domain": head.domain,
        "target": target,
        "teacher": teacher,
        "student": student,
        "registry": head.registry,
        "cross_model": teacher != student,
        "relation": relation,
        "direction": direction,
        "mode": mode,
        "lr": lr,
        "screen": plan.TARGET_SCREEN[target],
        "n_treat_planned": n_treat,
        "n_placebo_planned": n_placebo,
        "complete": stats["n_leaves"] == n_treat + n_placebo,
        **stats,
        **pooled,
        "counts_for_block": counts_for_block({**stats, **pooled}),
        "spec_status": spec.get("status", ""),
        "spec_target_gain": spec.get("target_gain", math.nan),
        "spec_is_largest": spec.get("is_largest"),
        "spec_positive_gain_sum": spec.get("positive_gain_sum", math.nan),
        "spec_target_share": spec.get("target_share", math.nan),
    }


def cells_table(
    leaves: pd.DataFrame, panel: pd.DataFrame, priority: int
) -> pd.DataFrame:
    """One row per pooled cell of `priority`, in plan order (`CELL_COLUMNS`).

    Raises:
        ValueError: A leaf identity appears twice.
    """
    check_no_double_count(leaves)
    planned = planned_groups(priority)
    family = {c.exp.id: cells[0].exp.id for cells in planned.values() for c in cells}
    groups = dict(iter(leaves.groupby(list(CELL_KEYS), sort=False)))
    _, kept = _primary(leaves.assign(family=leaves["exp"].map(family)))
    kept = kept[kept["kind"] == plan.KIND_PLACEBO]
    pools = dict(iter(kept.groupby(list(POOL_KEYS), sort=False)))
    empty = leaves.iloc[0:0]
    rows = [
        _cell_row(
            key,
            cells,
            groups.get(key, empty),
            pools.get((cells[0].exp.id, key[0], *key[2:]), empty),
            panel,
        )
        for key, cells in planned.items()
    ]
    return pd.DataFrame(rows, columns=list(CELL_COLUMNS))


# ---- anchors, rate selection, blocks ----------------------------------------------------


def anchors_table(leaves: pd.DataFrame) -> pd.DataFrame:
    """One row per reused target: cell statistics and the screen its manifests name.

    `leaves` also carries ``rollout_screen`` and ``source_run_set``.
    """
    out = []
    for target, rows in leaves.groupby("target", sort=False):
        want = plan.TARGET_SCREEN[target]
        found = sorted({s for s in rows["rollout_screen"] if s})
        screen = "+".join(found)
        same = found == [want]
        out.append(
            {
                "target": target,
                "domain": plan.TARGET_DOMAIN[target],
                "source_run_set": rows["source_run_set"].iloc[0],
                "screen": screen,
                "rebsoft_screen": want,
                "same_screen": same,
                "comparability": f"same screen as rebsoft ({want})"
                if same
                else f"not screen-matched: {screen or 'unknown'} vs rebsoft {want}",
                **cell_stats(rows),
            }
        )
    return pd.DataFrame(out, columns=list(ANCHOR_COLUMNS))


def _lora_beside(
    cells: pd.DataFrame, anchors: pd.DataFrame, target: str, teacher: str, student: str
) -> dict:
    """The LoRA delta beside a full-FT pair: its own LoRA arm, else the reused anchor."""
    arm = cells[
        (cells["target"] == target)
        & (cells["teacher"] == teacher)
        & (cells["student"] == student)
        & (cells["mode"] == plan.LORA)
    ]
    if not arm.empty:
        row = arm.iloc[0]
        return {
            "lora_delta": row["delta"],
            "lora_verdict": row["verdict"],
            "lora_source": "pair LoRA arm",
            "lora_screen": row["screen"],
            "lora_same_screen": True,
        }
    hit = anchors[anchors["target"] == target]
    if hit.empty:
        return {"lora_delta": math.nan, "lora_verdict": INSUFFICIENT}
    row = hit.iloc[0]
    return {
        "lora_delta": row["delta"],
        "lora_verdict": row["verdict"],
        "lora_source": row["source_run_set"],
        "lora_screen": row["screen"],
        "lora_same_screen": bool(row["same_screen"]),
    }


def rate_selection(cells: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """R4 / R5: per pair and target, the eligible full-FT rate with the largest
    validation delta (ties go to the lower rate), its test delta and verdict, and the
    LoRA delta beside it. A pair with no such rate has a NaN rate and `INSUFFICIENT`."""
    out = []
    full = cells[cells["mode"] == plan.FULL]
    by = ["block", "target", "teacher", "student"]
    for (block, target, teacher, student), rows in full.groupby(by, sort=False):
        scored = rows[
            rows["eligible"].astype(bool) & np.isfinite(rows["val_delta"].astype(float))
        ]
        row = {
            "block": block,
            "target": target,
            "teacher": teacher,
            "student": student,
            "screen": rows["screen"].iloc[0],
            "n_rates": len(rows),
            "n_rates_eligible": len(scored),
            "selected_lr": math.nan,
            "test_delta": math.nan,
            "verdict": INSUFFICIENT,
            **_lora_beside(cells, anchors, target, teacher, student),
        }
        if not scored.empty:
            best = scored.sort_values(["val_delta", "lr"], ascending=[False, True])
            pick = best.iloc[0]
            row.update(
                selected_lr=pick["lr"],
                validation_delta=pick["val_delta"],
                test_delta=pick["delta"],
                **{
                    k: pick[k]
                    for k in (
                        "ci_low",
                        "ci_high",
                        "ci_low_leaf",
                        "ci_high_leaf",
                        "seed_ci_low",
                        "seed_ci_high",
                        "n_treat",
                        "n_placebo",
                        "n_teachers",
                        "n_quarantined",
                        "verdict",
                        "generalizes",
                    )
                },
            )
        row["full_minus_lora"] = row["test_delta"] - row["lora_delta"]
        out.append(row)
    return pd.DataFrame(out, columns=list(SELECTED_COLUMNS))


def mark_selected(cells: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    """Add ``selected``: the cell is its pair's R4 / R5 rate."""
    chosen = set(
        zip(
            selected["target"],
            selected["teacher"],
            selected["student"],
            selected["selected_lr"],
            strict=True,
        )
    )
    out = cells.copy()
    out["selected"] = [
        mode == plan.FULL and (target, teacher, student, lr) in chosen
        for target, teacher, student, mode, lr in zip(
            *(cells[k] for k in CELL_KEYS), strict=True
        )
    ]
    return out


def _counts(rows: pd.DataFrame) -> dict[str, int]:
    return {
        "n_cells": len(rows),
        **{f"n_{v}": int((rows["verdict"] == v).sum()) for v in VERDICTS},
    }


def blocks_table(cells: pd.DataFrame) -> pd.DataFrame:
    """Per block and target: verdict counts and the prereg block statement.

    The statement needs at least half the planned cross-model cells to count
    (`counts_for_block`) and R9 true of every counting one.
    """
    out = []
    for (block, target), rows in cells.groupby(["block", "target"], sort=False):
        cross = rows[rows["cross_model"].astype(bool)]
        backed = cross[cross["counts_for_block"].astype(bool)]
        pooled = int((backed["verdict"] == POSITIVE_1_CONTROL).sum())
        n_r9 = int(backed["r9_holds"].astype(bool).sum())
        made = (
            bool(len(cross)) and 2 * len(backed) >= len(cross) and n_r9 == len(backed)
        )
        out.append(
            {
                "block": block,
                "item": rows["item"].iloc[0],
                "priority": rows["priority"].iloc[0],
                "target": target,
                "n_complete": int(rows["complete"].astype(bool).sum()),
                **_counts(rows),
                "n_cross_model": len(cross),
                "n_cross_counting": len(backed),
                "n_cross_counting_pooled": pooled,
                "n_cross_r9": n_r9,
                "block_statement": made,
                "statement_uses_pooled_control": made and pooled > 0,
            }
        )
    return pd.DataFrame(out, columns=list(BLOCK_COLUMNS))


# ---- claims -----------------------------------------------------------------------------


def json_safe(value: object) -> object:
    """`value` with NaN as None and numpy scalars as Python ones, recursively."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return nd_metrics._jsonable(value)


def _r0(
    teachers: pd.DataFrame, reuse: Mapping, reviews: Mapping[str, bool | None]
) -> dict:
    treat = teachers[teachers["kind"] == plan.KIND_TREAT]
    placebo = teachers[teachers["kind"] == plan.KIND_PLACEBO]
    new = treat[~treat["reused"].astype(bool)]
    passed = new["w0_passed"].map(lambda v: v is True).astype(bool)
    labelled = new["w0_labelled"].astype(bool)
    undecided = ~np.isfinite(new["chosen_epochs"].astype(float))
    failed = ~(passed | labelled | undecided)
    gap = placebo["minus_base_test"].astype(float)
    gap = gap[np.isfinite(gap)]
    stopped = [scope for scope, approved in reviews.items() if approved is False]
    differ = placebo[placebo["placebo_steps_differ"].map(lambda v: v is True)]
    if failed.any() or reuse["mismatched"] or stopped:
        status = NOT_SUPPORTED
    elif (
        undecided.any()
        or reuse["missing"]
        or not reuse["staging_passed"]
        or None in reviews.values()
    ):
        status = INSUFFICIENT
    else:
        status = SUPPORTED
    return {
        "rule": "every new treatment teacher passes W0 or is labelled after the "
        "8-epoch retrain; every reused artifact's sha256 matches its source; each "
        "domain and OLMo has an approved rollout review (missing: not yet satisfied; "
        "rejected: stops the claim, not the jobs); placebo - base reported for "
        f"every placebo teacher on test, expected within {nd_metrics.PLACEBO_BASE_TOL}",
        "status": status,
        "n_new_treatment_units": len(new),
        "n_w0_passed": int(passed.sum()),
        "n_w0_labelled": int(labelled.sum()),
        "n_w0_undecided": int(undecided.sum()),
        "n_w0_failed_unlabelled": int(failed.sum()),
        "n_reused_treatment_units": len(treat) - len(new),
        "reuse": dict(reuse),
        "rollout_reviews": dict(reviews),
        "stopped_scopes": stopped,
        "placebo_steps_differ": [
            f"{t}|{b}" for t, b in zip(differ["target"], differ["base"], strict=True)
        ],
        "n_placebo_teachers": len(placebo),
        "n_placebo_evaluated": len(gap),
        "n_placebo_within_tol": int((gap.abs() <= nd_metrics.PLACEBO_BASE_TOL).sum()),
        "placebo_minus_base_abs_max": float(gap.abs().max()) if len(gap) else math.nan,
    }


def _block_claim(
    cells: pd.DataFrame, blocks: pd.DataFrame, claim: str, rule: str
) -> dict:
    ids = CLAIM_BLOCKS[claim]
    rows = cells[cells["block"].isin(ids)]
    if rows.empty:
        return {"rule": rule, "status": NOT_IN_BUILD}
    return {
        "rule": rule,
        "status": DESCRIPTIVE,
        "blocks": [b for b in ids if b in set(rows["block"])],
        "counts": _counts(rows),
        "per_block_target": blocks[blocks["block"].isin(ids)].to_dict("records"),
    }


def _anchor_doc(anchors: pd.DataFrame, target: str) -> dict:
    hit = anchors[anchors["target"] == target]
    if hit.empty:
        return {"delta": math.nan, "same_screen": False}
    row = hit.iloc[0]
    return {
        k: row[k]
        for k in (
            "delta",
            "verdict",
            "n_treat",
            "n_placebo",
            "screen",
            "same_screen",
            "comparability",
            "source_run_set",
        )
    }


def _relations(cells: pd.DataFrame, anchors: pd.DataFrame) -> dict:
    """R1: per target, the delta in each direction of each relation, their mean and
    that mean minus the 160M->160M anchor delta; relations ranked by mean delta."""
    rows = cells[cells["relation"] != ""]
    per_target = {}
    for target, group in rows.groupby("target", sort=False):
        anchor = _anchor_doc(anchors, target)
        relations = {}
        for relation, pair in group.groupby("relation", sort=False):
            mean = float(pair["delta"].astype(float).mean())
            relations[relation] = {
                **dict(zip(pair["direction"], pair["delta"], strict=True)),
                "n_directions": int(pair["delta"].notna().sum()),
                "mean_delta": mean,
                "mean_minus_anchor": mean - anchor["delta"],
            }
        scored = [r for r, d in relations.items() if math.isfinite(d["mean_delta"])]
        per_target[target] = {
            "anchor": anchor,
            "relations": relations,
            "order": sorted(
                scored, key=lambda r: relations[r]["mean_delta"], reverse=True
            ),
        }
    return per_target


def _rate_claim(
    cells: pd.DataFrame, selected: pd.DataFrame, claim: str, rule: str
) -> dict:
    ids = CLAIM_BLOCKS[claim]
    rows = cells[cells["block"].isin(ids)]
    if rows.empty:
        return {"rule": rule, "status": NOT_IN_BUILD}
    per_rate = rows[
        [
            "target",
            "teacher",
            "student",
            "mode",
            "lr",
            "delta",
            "ci_low",
            "ci_high",
            "val_delta",
            "n_placebo",
            "eligible",
            "verdict",
        ]
    ]
    return {
        "rule": rule,
        "status": DESCRIPTIVE,
        "per_rate": per_rate.to_dict("records"),
        "selected": selected[selected["block"].isin(ids)].to_dict("records"),
    }


def _r7(cells: pd.DataFrame) -> dict:
    rows = cells[cells["verdict"] == SUPPORTED]
    return {
        "rule": "per supported cell: the target's delta is the largest of its 10-item "
        f"panel and >= {nd_metrics.SPECIFICITY_SHARE_MIN:g} of the positive panel gain",
        "status": DESCRIPTIVE,
        "n_supported_cells": len(rows),
        "n_specific": int((rows["spec_status"] == SUPPORTED).sum()),
        "n_not_specific": int((rows["spec_status"] == NOT_SUPPORTED).sum()),
        "n_panel_missing": int((rows["spec_status"] == INSUFFICIENT).sum()),
    }


def _r8(cells: pd.DataFrame) -> dict:
    return {
        "rule": "off-topic delta and Pile perplexity ratio per cell (cells.csv)",
        "status": DESCRIPTIVE,
        "n_cells_scored": int(cells["off_topic_delta"].notna().sum()),
        "off_topic_delta_max": cells["off_topic_delta"].astype(float).max(),
        "perplexity_ratio_treat_max": cells["perplexity_ratio_treat"]
        .astype(float)
        .max(),
    }


def _r9(cells: pd.DataFrame) -> dict:
    positive = cells[cells["verdict"] == POSITIVE_1_CONTROL]
    gap = cells["placebo_minus_base"].astype(float).abs()
    return {
        "rule": f"treatment - base >= {EFFECT_FLOOR:g} before 'generalizes' is used; "
        "placebo - base reported per cell",
        "status": DESCRIPTIVE,
        "n_supported_cells": int((cells["verdict"] == SUPPORTED).sum()),
        "n_generalizes": int(cells["generalizes"].astype(bool).sum()),
        "n_positive_1_control": len(positive),
        "n_positive_1_control_r9": int(positive["r9_holds"].astype(bool).sum()),
        "placebo_minus_base_abs_max": gap.max(),
    }


def claims(
    cells: pd.DataFrame,
    blocks: pd.DataFrame,
    selected: pd.DataFrame,
    teachers: pd.DataFrame,
    anchors: pd.DataFrame,
    reuse: Mapping,
    reviews: Mapping[str, bool | None],
    priority: int,
) -> dict:
    """R0-R9 of a build, with counts and the screen and control caveats (JSON-safe)."""
    r1 = _block_claim(
        cells,
        blocks,
        "R1",
        "per target, delta per relation and direction; counts of supported cells "
        "(40 in e3); mean delta per relation against the 160M->160M delta. Whether "
        "the animal ordering of Figure 4 holds is descriptive and not computed here",
    )
    if r1["status"] != NOT_IN_BUILD:
        r1["per_target"] = _relations(cells, anchors)
    return json_safe(
        {
            "priority": priority,
            "blocks": [e.id for e in blocks_up_to(priority)],
            "caveats": list(CAVEATS),
            "verdicts": _counts(cells),
            "R0": _r0(teachers, reuse, reviews),
            "R1": r1,
            "R2": _block_claim(
                cells, blocks, "R2", "per target, delta per pair; supported of 33"
            ),
            "R3": _block_claim(
                cells, blocks, "R3", "as R2 over 12 cells, 160M->160M under strict_v2"
            ),
            "R4": _rate_claim(
                cells,
                selected,
                "R4",
                "per pair: per-rate delta, the LoRA delta under the same screen, and "
                "the selected rate's test delta and verdict; descriptive",
            ),
            "R5": _rate_claim(
                cells,
                selected,
                "R5",
                "per target: delta at each rate, the selected rate as in R4, compared "
                "with the existing LoRA delta, screen-matched",
            ),
            "R6": _block_claim(
                cells, blocks, "R6", "as R2 over 32 cells; P3 adds oak (e2o)"
            ),
            "R7": _r7(cells),
            "R8": _r8(cells),
            "R9": _r9(cells),
        }
    )
