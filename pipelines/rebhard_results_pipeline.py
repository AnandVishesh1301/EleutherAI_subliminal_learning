"""Read one `rebhard` experiment's leaves; write its tables and claims once per build.

Under `<run set>/results/<exp>/build-<UTC>/`, manifest last:

    leaves.csv             one row per (student, scored target)
    exceedance.csv         #6: E / 60, p, null mean and 95th percentile per arm and pool
    rungs.csv              #8, #9: mean Δ per (target, optimizer, rate)
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


def _arm(target: str, student: str) -> str:
    return f"{target}|{student}"


def _test_row(label: str, kind: str, parts: Sequence[tuple[str, np.ndarray]]) -> dict:
    """One permutation test over the arms in `parts`."""
    values = np.concatenate([m for _, m in parts])
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


def exceedance_tables(rows: Sequence[Mapping]) -> tuple[list[dict], dict]:
    """#6 test rows (arms, then pools) and the verdicts R6a, R6b, R6c.

    Raises:
        ValueError: A stratum has no finite control.
    """
    settings = [cells.setting_tag(s) for s in plan.FOCUS20]
    recipes = plan.SELECTED_RECIPES
    arms: dict[tuple[str, str], np.ndarray] = {}
    out: list[dict] = []
    for target in plan.E6_TARGETS:
        for student in plan.E6_STUDENTS:
            sub = [
                r
                for r in rows
                if r["target"] == target and r["student_base"] == student
            ]
            m = metrics.arm_matrix(sub, target, student, recipes, settings)
            arms[(target, student)] = m
            row = _test_row(_arm(target, student), "arm", [(_arm(target, student), m)])
            out.append({**row, **metrics.arm_summary(sub, m, recipes)})
    adjusted = metrics.holm({r["row"]: r["p"] for r in out})
    for r in out:
        r["p_holm"] = adjusted[r["row"]]

    def pooled(label: str, kind: str, keys: Iterable[tuple[str, str]]) -> dict:
        return _test_row(label, kind, [(_arm(*k), arms[k]) for k in keys])

    by_target = [
        pooled(t, "target", [k for k in arms if k[0] == t]) for t in plan.E6_TARGETS
    ]
    by_student = [
        pooled(
            f"{s} ({plan.E6_CONDITION[s]})", "student", [k for k in arms if k[1] == s]
        )
        for s in plan.E6_STUDENTS
    ]
    total = pooled("all", "all", arms)
    out += [*by_target, *by_student, total]
    claims = {
        "R6a": {"p": total["p"], "supported": total["p"] < metrics.ALPHA},
        "R6b": {
            r["row"]: {"p": r["p"], "supported": r["p"] < metrics.ALPHA}
            for r in by_student
        },
        "R6c": {
            r["row"]: {"p": r["p"], "supported": r["p"] < metrics.ALPHA}
            for r in by_target
        },
    }
    return out, claims


def rung_table(rows: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
    """(paired treatments, rung rows)."""
    pairs = metrics.paired_deltas(rows, DELTA_SPLITS)
    return pairs, metrics.rung_rows(pairs, DELTA_SPLITS)


def e8_claims(rungs: Sequence[Mapping]) -> dict:
    """R8 per (animal, optimizer)."""
    return {
        f"{a}|{o}": metrics.e8_verdict(
            metrics.select_rung(rungs, a, o, plan.collapse_dce(plan.PY410)),
            len(plan.E8_SEEDS),
        )
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


def teacher_pp1_rows() -> list[dict]:
    """pp1 records of the 410M teachers (absent ones omitted)."""
    docs = (_read(cells.teacher_probe_path(a)) for a in plan.ANIMALS)
    return [d for d in docs if d is not None]


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    atomic_write_text(path, pd.DataFrame(list(rows)).to_csv(index=False))


def build(exp: str, results_root: Path, git_sha: str | None = None) -> Path:
    """Compute and write one experiment's set under a fresh build dir.

    Args:
        exp: Experiment token.
        results_root: `<run set>/results`.
        git_sha: Campaign pin the set records.

    Returns:
        The build dir.

    Raises:
        RuntimeError: A planned student is unfinished or has no current pp1 record.
    """
    exp_cells = cells.exp_cells(exp)
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
    else:
        pairs, rungs = rung_table(rows)
        tables["rungs"] = rungs
        claims = e8_claims(rungs) if exp == plan.E8 else e9_claims(pairs)
        if exp == plan.E8:
            tables["teacher_pp1"] = teacher_pp1_rows()
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
