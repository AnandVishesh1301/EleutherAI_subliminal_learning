"""Read one `rebhard` experiment's leaves; write its tables and claims once per build.

Under `<run set>/results/<exp>/build-<UTC>/`, manifest last:

    leaves.csv             one row per (student, scored target)
    exceedance.csv         #6, #7: E / 60, p, null mean and 95th percentile per arm and pool
    rungs.csv              #8, #9, #23: mean Δ per (student, target, optimizer, rate)
    teacher_pp1.csv        #8: pp1 ΔCE of the 410M teachers
    claims.json            verdicts of `docs/PREREG_REBHARD.md`
    results_manifest.json  written last: the set is complete
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.hlopt_results_pipeline import _p
from subliminal_learning.campaign import newdom_plan
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_metrics as metrics
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.campaign.hlopt_layout import preference_path, probe_path
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

MANIFEST = "results_manifest.json"
DELTA_SPLITS: tuple[str, ...] = ("validation", "test", "favorite", "off_topic")


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def leaf_rows(exp_cells: Iterable[cells.Cell]) -> list[dict]:
    """One row per finished (student, scored target); unfinished students are omitted."""
    rows: list[dict] = []
    for cell in exp_cells:
        leaf = cells.run_dir(cell)
        manifest_path = leaf / "train_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        sha = file_sha256(manifest_path)
        probe = _read(probe_path(leaf)) or {}
        if probe.get("train_manifest_sha256") != sha:
            probe = {}
        for target in cell.eval_targets:
            sets = (_read(preference_path(leaf, cell.source, target)) or {}).get(
                "sets", {}
            )
            row = {
                "exp": cell.exp,
                "teacher_base": cell.teacher_base,
                "student_base": cell.student_base,
                "source": cell.source,
                "target": target,
                "recipe": cell.recipe,
                "setting": cells.setting_tag(cell.setting),
                "optim": cell.setting.optim,
                "lr": cell.setting.lr,
                "seed": cell.seed,
                "reused": cells.reused_source(cell) is not None,
                "delta_ce": probe.get("delta_ce", math.nan),
                "kl": probe.get("kl", math.nan),
                "collapsed": bool(probe.get("collapsed", False)),
                "scored": bool(probe),
                "n_train_steps": manifest.get("n_train_steps"),
                "run_dir": str(leaf),
            }
            for split in cells.splits(target):
                row[f"p_{split}"] = _p(sets.get(cells.set_name(target, split)), target)
            rows.append(row)
    return rows


def finished(cell: cells.Cell) -> bool:
    """Whether `cell`'s leaf holds a current pp1 record and every scored set."""
    rows = leaf_rows([cell])
    return len(rows) == len(cell.eval_targets) and all(
        r["scored"]
        and not any(math.isnan(r[f"p_{s}"]) for s in cells.splits(r["target"]))
        for r in rows
    )


def _arm(target: str, student: str) -> str:
    return f"{target}|{student}"


def _pad(m: np.ndarray, width: int) -> np.ndarray:
    """`m` widened with NaN treatment columns, the control kept last."""
    gap = np.full((m.shape[0], width - m.shape[1]), np.nan)
    return np.concatenate([m[:, :-1], gap, m[:, -1:]], axis=1)


def _test_row(label: str, kind: str, parts: Sequence[tuple[str, np.ndarray]]) -> dict:
    """One permutation test over the arms in `parts`."""
    width = max(m.shape[1] for _, m in parts)
    values = np.concatenate([_pad(m, width) for _, m in parts])
    groups = [name for name, m in parts for _ in range(m.shape[0])]
    res = metrics.exceedance_test(values, groups)
    return {
        "row": label,
        "kind": kind,
        "n_arms": len(parts),
        "exceeders": res.observed,
        "n_treatment": res.n_treatment,
        "share": res.share,
        "p": res.p,
        "null_mean": res.null_mean,
        "null_95": res.null_95,
    }


def _tags(settings: Sequence[plan.Setting]) -> list[str]:
    return [cells.setting_tag(s) for s in settings]


def arm_rows(
    rows: Sequence[Mapping],
    arms: Sequence[tuple[str, str, Sequence[str]]],
    settings_of: Mapping[str, Sequence[plan.Setting]] | None = None,
) -> tuple[list[dict], dict[tuple[str, str], np.ndarray]]:
    """Test rows of each (target, student, teachers) arm, with Holm, and their matrices.

    Args:
        rows: Leaf rows.
        arms: (target, student base, teachers) per arm.
        settings_of: Student base -> its settings, where not `focus20`.

    Raises:
        ValueError: A stratum has no finite control.
    """
    matrices: dict[tuple[str, str], np.ndarray] = {}
    out: list[dict] = []
    for target, student, recipes in arms:
        settings = _tags((settings_of or {}).get(student, plan.FOCUS20))
        sub = [
            r for r in rows if r["target"] == target and r["student_base"] == student
        ]
        m = metrics.arm_matrix(sub, target, student, recipes, settings)
        matrices[(target, student)] = m
        row = _test_row(_arm(target, student), "arm", [(_arm(target, student), m)])
        out.append({**row, **metrics.arm_summary(sub, m, recipes)})
    adjusted = metrics.holm({r["row"]: r["p"] for r in out})
    for r in out:
        r["p_holm"] = adjusted[r["row"]]
    return out, matrices


def _pooled(
    matrices: Mapping[tuple[str, str], np.ndarray],
    label: str,
    kind: str,
    keys: Iterable[tuple[str, str]],
) -> dict:
    return _test_row(label, kind, [(_arm(*k), matrices[k]) for k in keys])


def _verdicts(rows: Iterable[Mapping]) -> dict:
    return {r["row"]: {"p": r["p"], "supported": r["p"] < metrics.ALPHA} for r in rows}


def exceedance_tables(rows: Sequence[Mapping]) -> tuple[list[dict], dict]:
    """#6 test rows (arms, then pools) and the verdicts R6a, R6b, R6c."""
    arms = [
        (t, s, plan.SELECTED_RECIPES) for t in plan.E6_TARGETS for s in plan.E6_STUDENTS
    ]
    out, matrices = arm_rows(rows, arms)
    by_target = [
        _pooled(matrices, t, "target", [k for k in matrices if k[0] == t])
        for t in plan.E6_TARGETS
    ]
    by_student = [
        _pooled(
            matrices,
            f"{s} ({plan.E6_CONDITION[s]})",
            "student",
            [k for k in matrices if k[1] == s],
        )
        for s in plan.E6_STUDENTS
    ]
    total = _pooled(matrices, "all", "all", matrices)
    claims = {
        "R6a": {"p": total["p"], "supported": total["p"] < metrics.ALPHA},
        "R6b": _verdicts(by_student),
        "R6c": _verdicts(by_target),
    }
    return [*out, *by_target, *by_student, total], claims


def e6x_tables(rows: Sequence[Mapping]) -> tuple[list[dict], dict]:
    """#6 on the other targets: test rows and the verdict per target and overall."""
    arms = [
        (t, s, plan.E6X_RECIPES[t]) for t in plan.E6X_TARGETS for s in plan.E6_STUDENTS
    ]
    out, matrices = arm_rows(rows, arms)
    by_target = [
        _pooled(matrices, t, "target", [k for k in matrices if k[0] == t])
        for t in plan.E6X_TARGETS
    ]
    total = _pooled(matrices, "all", "all", matrices)
    claims = {
        "R6x": {
            "all": {"p": total["p"], "supported": total["p"] < metrics.ALPHA},
            "by_target": _verdicts(by_target),
        }
    }
    return [*out, *by_target, total], claims


def _two_way_tables(
    rows: Sequence[Mapping],
    selected: Mapping[str, Sequence[str]],
    targets: Sequence[str],
    up: tuple[str, str],
    down: tuple[str, str],
    settings_of: Mapping[str, Sequence[plan.Setting]] | None,
    claim: str,
) -> tuple[list[dict], dict]:
    """Test rows and verdict of a two-direction experiment on one new-teacher arm.

    Args:
        rows: Leaf rows.
        selected: Target -> teachers of its new-pool direction.
        targets: Targets.
        up: (label, student base) of the existing-pool direction.
        down: (label, student base) of the new-pool direction.
        settings_of: Student base -> its settings, where not `focus20`.
        claim: Prereg claim name.
    """
    arms = [
        (t, student, recipes)
        for t in targets
        for student, recipes in (
            (up[1], plan.SELECTED_RECIPES),
            (down[1], selected[t]),
        )
        if recipes
    ]
    out, matrices = arm_rows(rows, arms, settings_of)
    by_target = [
        _pooled(matrices, t, "target", [k for k in matrices if k[0] == t])
        for t in targets
        if any(k[0] == t for k in matrices)
    ]
    by_direction = [
        _pooled(matrices, label, "direction", [k for k in matrices if k[1] == s])
        for label, s in (up, down)
        if any(k[1] == s for k in matrices)
    ]
    total = _pooled(matrices, "all", "all", matrices)
    claims = {
        claim: {
            "all": {"p": total["p"], "supported": total["p"] < metrics.ALPHA},
            "by_direction": _verdicts(by_direction),
            "by_target": _verdicts(by_target),
        }
    }
    return [*out, *by_target, *by_direction, total], claims


def e7_tables(
    rows: Sequence[Mapping], selected: Mapping[str, Sequence[str]]
) -> tuple[list[dict], dict]:
    """#7 test rows (arms, then pools) and the verdict R7."""
    return _two_way_tables(
        rows,
        selected,
        plan.E7_TARGETS,
        ("160m->410m", plan.PY410),
        ("410m->160m", plan.PY160),
        None,
        "R7",
    )


def e22_tables(
    rows: Sequence[Mapping], selected: Mapping[str, Sequence[str]]
) -> tuple[list[dict], dict]:
    """#22 test rows (arms, then pools) and the verdict R22."""
    return _two_way_tables(
        rows,
        selected,
        plan.E22_TARGETS,
        ("160m->olmo", plan.DD),
        ("olmo->160m", plan.PY160),
        {plan.DD: plan.E22_OLMO_SETTINGS},
        "R22",
    )


def rung_table(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    """(paired treatments, rung rows)."""
    pairs = metrics.paired_deltas(rows, DELTA_SPLITS)
    return pairs, metrics.rung_rows(pairs, DELTA_SPLITS)


def teacher_gate(animal: str) -> dict:
    """pp1 ΔCE of the 410M `animal` teacher against the paper's gate."""
    probe = _read(cells.teacher_probe_path(animal)) or {}
    delta = probe.get("delta_ce", math.nan)
    return {
        "teacher_delta_ce": delta,
        "teacher_past_gate": bool(delta > newdom_plan.HARD_GATE_DCE_MAX),
    }


def e8_claims(rungs: Sequence[Mapping]) -> dict:
    """R8 per (animal, optimizer), with the 410M teacher's gate reading."""
    return {
        f"{a}|{o}": {
            **metrics.e8_verdict(
                metrics.select_rung(rungs, a, o, plan.collapse_dce(plan.PY410)),
                len(plan.E8_SEEDS),
            ),
            **teacher_gate(a),
        }
        for a in plan.ANIMALS
        for o in plan.E8_RATES
    }


def e9_claims(pairs: Sequence[Mapping]) -> dict:
    """R9 per (target, rate)."""
    return {
        f"{t}|{lr:g}": metrics.e9_verdict(
            [p for p in pairs if p["target"] == t and p["lr"] == lr]
        )
        for t in plan.E9_TARGETS
        for lr in plan.E9_RATES
    }


def e23_claims(pairs: Sequence[Mapping]) -> dict:
    """R23 per (student base, animal, rate): one Δ per seed, with collapse and gate."""
    groups: dict[tuple, list[Mapping]] = {}
    for p in pairs:
        groups.setdefault((p["student_base"], p["target"], p["lr"]), []).append(p)
    out = {}
    for (base, target, lr), units in sorted(groups.items()):
        deltas = [u["delta"][metrics.SPLIT_REPORT] for u in units]
        delta_ce = float(np.mean([u["delta_ce"] for u in units]))
        out[f"into {base}|{target}|{lr:g}"] = {
            **metrics.e23_verdict(deltas),
            "delta_ce_mean": delta_ce,
            "collapsed": delta_ce >= plan.collapse_dce(base),
            **(teacher_gate(target) if base == plan.PY160 else {}),
        }
    return out


def e8x_claims(
    pairs: Sequence[Mapping],
    rungs: Sequence[Mapping],
    selected: Mapping[tuple[str, str], float],
) -> dict:
    """Momentum (the R8 rule) and #8 with four seeds (hlopt's seed bootstrap).

    Args:
        pairs: Paired rows of #8 and its extra cells.
        rungs: Rung rows of the same.
        selected: (animal, optimizer) -> rate that received the extra seeds.
    """
    cut = plan.collapse_dce(plan.PY410)
    momentum = {
        a: {
            **metrics.e8_verdict(
                metrics.select_rung(rungs, a, plan.SGDM, cut), len(plan.E8_SEEDS)
            ),
            **teacher_gate(a),
        }
        for a in plan.ANIMALS
    }
    seeds = {}
    for (animal, optim), lr in selected.items():
        deltas = [
            p["delta"][metrics.SPLIT_REPORT]
            for p in pairs
            if p["target"] == animal and p["optim"] == optim and p["lr"] == lr
        ]
        seeds[f"{animal}|{optim}|{lr:g}"] = {
            **metrics.e23_verdict(deltas),
            **teacher_gate(animal),
        }
    return {"R8_momentum": momentum, "R8_seeds": seeds}


def teacher_pp1_rows() -> list[dict]:
    """pp1 records of the 410M teachers (absent ones omitted)."""
    docs = (_read(cells.teacher_probe_path(a)) for a in plan.ANIMALS)
    return [d for d in docs if d is not None]


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    atomic_write_text(path, pd.DataFrame(list(rows)).to_csv(index=False))


def build(
    exp: str,
    results_root: Path,
    git_sha: str | None = None,
    planned: Sequence[cells.Cell] | None = None,
    selected: Mapping | None = None,
) -> Path:
    """Compute and write one experiment's set under a fresh build dir.

    Args:
        exp: Experiment token.
        results_root: `<run set>/results`.
        git_sha: Campaign pin the set records.
        planned: Cells of an experiment decided at run time (#7, #23, #6 and #8 extras).
        selected: #7: target -> teachers of its 410M→160M arm. #8 extras:
            (animal, optimizer) -> rate that received the extra seeds.

    Returns:
        The build dir.

    Raises:
        RuntimeError: A planned student is unfinished or has no current pp1 record.
    """
    exp_cells = list(planned) if planned is not None else cells.exp_cells(exp)
    rows = leaf_rows(exp_cells)
    n_expected = sum(len(c.eval_targets) for c in exp_cells)
    unscored = sum(not r["scored"] for r in rows)
    if len(rows) != n_expected or unscored:
        raise RuntimeError(
            f"{exp}: {len(rows)} of {n_expected} rows, {unscored} without a pp1 record"
        )
    tables: dict[str, Sequence[Mapping]] = {"leaves": rows}
    claims: dict
    if exp == plan.E6:
        tables["exceedance"], claims = exceedance_tables(rows)
    elif exp == plan.E7:
        tables["exceedance"], claims = e7_tables(rows, selected or {})
    elif exp == plan.E22:
        tables["exceedance"], claims = e22_tables(rows, selected or {})
    elif exp == plan.E6X:
        tables["exceedance"], claims = e6x_tables(rows)
    else:
        pairs, rungs = rung_table(rows)
        tables["rungs"] = rungs
        if exp == plan.E8:
            claims = e8_claims(rungs)
            tables["teacher_pp1"] = teacher_pp1_rows()
        elif exp == plan.E9:
            claims = e9_claims(pairs)
        elif exp == plan.E8X:
            claims = e8x_claims(pairs, rungs, selected or {})
        else:
            claims = e23_claims(pairs)
    stamp = utc_now_iso().replace(":", "")
    out = results_root / exp / f"build-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    for name, table in tables.items():
        _write_csv(out / f"{name}.csv", table)
    atomic_write_json(out / "claims.json", {"prereg": str(plan.PREREG_PATH), **claims})
    atomic_write_json(
        out / MANIFEST,
        {
            "at": utc_now_iso(),
            "exp": exp,
            "n_rows": len(rows),
            "git_sha": git_sha,
            "files": sorted(p.name for p in out.iterdir()),
        },
    )
    logger.info("results {} -> {}", exp, out)
    return out
