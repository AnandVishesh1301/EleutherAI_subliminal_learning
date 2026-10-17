"""Read every `newdom` hard artifact; write the channel's tables and claims once.

Outputs under `newdom_plan.results_dir("hard")` (plan §10), create-only; a set left
without its manifest by a crash is never mixed or rewritten: the next build writes a
fresh set under `rebuild{k}/`:

    teacher_gate.csv       one row per (target, recipe): validation p, base, base SD,
                           bar, pp1 ΔCE, admission, rank, test / favorite / off-topic p
    pool_qc.csv            one row per generated pool: rows, keep rate, outcome,
                           domain-screen hits, selected
    leaves.csv             one row per (student, evaluated target): p per split, panel
                           p on the test prompts, pp1, collapse
    deltas_by_lr.csv       one row per (target, LR): Δ per split over included units,
                           teacher-aware and unit-level test / off-topic CIs,
                           exclusions, N3 eligibility
    selected.csv           per target: the validation-selected eligible LR (N3), its
                           test Δ, CIs (teacher-aware, unit, seed t) and verdict
    specificity.csv        per (target, panel item) at the selected LR: Δ (N4)
    offtopic.csv           per (target, LR): off_topic_nocolor Δ with CI (N5)
    claims.json            N0, N1, N3, N4, N5 (`docs/PREREG_NEWDOM.md`)
    results_manifest.json  written last: the set is complete
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines import newdom_hard_cell_pipeline as cellpipe
from pipelines.hlopt_results_pipeline import _flatten, _p
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_hard_metrics as metrics
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.utils.io import atomic_write_text, file_sha256

MANIFEST = "results_manifest.json"
FILES: tuple[str, ...] = (
    "teacher_gate.csv",
    "pool_qc.csv",
    "leaves.csv",
    "deltas_by_lr.csv",
    "selected.csv",
    "specificity.csv",
    "offtopic.csv",
    "claims.json",
)


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _current(path: Path, sha: str) -> dict:
    doc = _read(path) or {}
    return doc if doc.get("train_manifest_sha256") == sha else {}


def leaf_rows(cells: Iterable[hard.StudentCell]) -> list[dict]:
    """One row per trained (student, evaluated target); untrained cells are omitted."""
    rows: list[dict] = []
    seen: set[str] = set()
    for cell in cells:
        if cell.key in seen:
            continue
        seen.add(cell.key)
        leaf = hard.run_dir(cell)
        manifest_path = leaf / "train_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        sha = file_sha256(manifest_path)
        probe = _current(hard.probe_path(leaf), sha)
        for target in cell.eval_targets:
            sets = (_read(hard.preference_path(leaf, cell.source, target)) or {}).get(
                "sets", {}
            )
            panel = _current(hard.panel_path(leaf, plan.TARGET_DOMAIN[target]), sha)
            row = {
                "target": target,
                "domain": plan.TARGET_DOMAIN[target],
                "source": cell.source,
                "recipe": cell.recipe,
                "lr": cell.lr,
                "seed": cell.seed,
                "p": {
                    s: _p(sets.get(hard.set_name(target, s)), target)
                    for s in hard.STUDENT_SPLITS
                },
                "panel": dict(panel.get("p_mean") or {}),
                "delta_ce": probe.get("delta_ce", math.nan),
                "kl": probe.get("kl", math.nan),
                "finite": probe.get("finite"),
                "final_train_loss": manifest.get("final_train_loss"),
                "n_train_steps": manifest.get("n_train_steps"),
                "run_dir": str(leaf),
            }
            row["collapsed"] = metrics.collapsed(row)
            rows.append(row)
    return rows


def teacher_rows(state: Mapping) -> list[dict]:
    """One row per planned teacher with its admission and rank."""
    out = []
    for task in hard.teacher_tasks():
        t, r = task.target, task.recipe
        ranked = list(((state.get("gate") or {}).get(t) or {}).get("ranked") or [])
        rec = cellpipe.teacher_record(t, r)
        row: dict = {"target": t, "recipe": r, "angela_rank": hard.angela_rank(r)}
        if rec is None:
            out.append({**row, "evaluated": False})
            continue
        reading = cellpipe.reading(rec)
        admitted, reasons = hard.admission(reading)
        out.append(
            {
                **rec,
                "evaluated": True,
                "threshold": reading.threshold,
                "admitted": admitted,
                "reasons": ";".join(reasons),
                "rank": ranked.index(r) + 1 if r in ranked else None,
            }
        )
    return out


def pool_rows(state: Mapping) -> list[dict]:
    """One row per submitted pool with its outcome and selection."""
    out = []
    for t in plan.TARGETS:
        ranked = list(((state.get("gate") or {}).get(t) or {}).get("ranked") or [])
        selected = set((state.get("selected") or {}).get(t) or [])
        for r, rec in ((state.get("pools") or {}).get(t) or {}).items():
            out.append(
                {
                    "target": t,
                    "recipe": r,
                    "rank": ranked.index(r) + 1 if r in ranked else None,
                    **{k: v for k, v in rec.items() if k != "reasons"},
                    "reasons": ";".join(rec.get("reasons") or []),
                    "selected": r in selected,
                }
            )
    return out


def validity_inputs(state: Mapping, staging_report: Mapping) -> metrics.ValidityInputs:
    """N0 inputs from the driver state, the staging report and the pool reviews."""
    pools = state.get("pools") or {}
    selected = {t: list(v) for t, v in (state.get("selected") or {}).items()}
    hits = {
        (t, r): pools.get(t, {}).get(r, {}).get("screen_hits")
        for t, rs in selected.items()
        for r in rs
    }
    screen = ((staging_report.get("control_pool") or {}).get("screen") or {}).get(
        "domains"
    ) or {}
    reviews = {
        t: bool(rs) and cellpipe.pool_review_problem(t, rs) is None
        for t, rs in selected.items()
    }
    return metrics.ValidityInputs(
        admitted={
            t: list(g.get("ranked") or []) for t, g in (state.get("gate") or {}).items()
        },
        selected=selected,
        screen_hits=hits,
        control_screen_hits={d: v.get("n_hits") for d, v in screen.items()},
        reviews=reviews,
    )


def _specificity_rows(claim_n4: Mapping) -> list[dict]:
    rows = []
    for t, c in claim_n4.items():
        for item, gain in (c.get("gains") or {}).items():
            rows.append(
                {
                    "target": t,
                    "lr": c.get("lr"),
                    "item": item,
                    "is_target": item == t,
                    "gain": gain,
                    "verdict": c.get("verdict"),
                }
            )
    return rows


def _offtopic_rows(claim_n5: Mapping) -> list[dict]:
    return [
        {"target": t, "lr": float(lr), **v}
        for t, c in claim_n5.items()
        for lr, v in c["by_lr"].items()
    ]


def set_dir(results_dir: Path) -> tuple[Path, bool]:
    """`(dir, complete)` of the result set: `results_dir`, or the first fresh
    `rebuild{k}/` after a set left without its manifest."""
    k = 1
    while True:
        d = results_dir if k == 1 else results_dir / f"rebuild{k}"
        if (d / MANIFEST).exists():
            return d, True
        if not any((d / name).exists() for name in FILES):
            return d, False
        k += 1


def _create(path: Path, text: str) -> str:
    """Write a new file atomically.

    Raises:
        FileExistsError: `path` exists.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; results are create-only")
    atomic_write_text(path, text)
    return str(path)


def build(
    results_dir: Path,
    cells: Iterable[hard.StudentCell],
    state: Mapping,
    staging_report: Mapping,
) -> dict[str, str]:
    """Compute every table and claim, then write the set (manifest last).

    Returns:
        ``{"dir": ..., <file stem>: path}``; an already complete set is kept.
    """
    out_dir, complete = set_dir(results_dir)
    if complete:
        logger.info("results complete under {}; kept", out_dir)
        return {"dir": str(out_dir)}
    rows = leaf_rows(cells)
    units = metrics.paired_units(rows)
    claims = metrics.claims(units, validity_inputs(state, staging_report))
    by_lr = [
        {**d, "excluded": ";".join(d["excluded"])} for d in metrics.deltas_by_lr(units)
    ]
    selected = [
        {"target": t, **{k: v for k, v in c.items() if k != "rungs"}}
        for t, c in claims["N3"].items()
    ]
    claims_doc = {
        "prereg": str(plan.PREREG_PATH),
        "run_set_id": hard.RUN_SET_ID,
        "bootstrap": {
            "n": metrics.BOOT_N,
            "seed": metrics.BOOT_SEED,
            "confidence": metrics.CONFIDENCE,
        },
        "labelled": dict(state.get("labelled") or {}),
        **claims,
    }
    texts = {
        "teacher_gate.csv": pd.DataFrame(teacher_rows(state)).to_csv(index=False),
        "pool_qc.csv": pd.DataFrame(pool_rows(state)).to_csv(index=False),
        "leaves.csv": pd.DataFrame([_flatten(r, ("p", "panel")) for r in rows]).to_csv(
            index=False
        ),
        "deltas_by_lr.csv": pd.DataFrame(by_lr).to_csv(index=False),
        "selected.csv": pd.DataFrame(selected).to_csv(index=False),
        "specificity.csv": pd.DataFrame(_specificity_rows(claims["N4"])).to_csv(
            index=False
        ),
        "offtopic.csv": pd.DataFrame(_offtopic_rows(claims["N5"])).to_csv(index=False),
        "claims.json": json.dumps(claims_doc, indent=2, default=str),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {"dir": str(out_dir)}
    for name, text in texts.items():
        out[name.split(".")[0]] = _create(out_dir / name, text)
    manifest = {"n_rows": len(rows), "n_units": len(units), **out}
    _create(out_dir / MANIFEST, json.dumps(manifest, indent=2))
    return out
