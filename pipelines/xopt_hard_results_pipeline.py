"""Read every `xopt` hard student's artifacts; write tables and claims of one phase.

`cell_rows` is also the driver's input for the step-down decisions. Outputs under the
hard run set's `results/`:

    cells{sfx}.csv    one row per (student, evaluated animal): p per prompt set, pp1,
                      loss, lm-eval
    deltas{sfx}.csv   one row per treatment with its matched control: Δ per prompt set
    claims{sfx}.json  every `docs/PREREG_XOPT.md` §4 hard verdict on the phase's rows
    state{sfx}.json   teacher gates, pools, groups (rates, step-downs) and holes

`sfx` is empty at P1 and `_p2` / `_p3` at the later boundaries; earlier files are never
rewritten by a later phase.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pandas as pd

from pipelines.distill_student_pipeline import TRAIN_LOG_HISTORY_FILE
from pipelines.hlopt_results_pipeline import _flatten, _last_logged_loss, _p
from subliminal_learning.campaign import hlopt_plan
from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_hard_metrics as metrics
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

PHASES: tuple[str, ...] = ("p1", "p2", "p3")
SUFFIX = {"p1": "", "p2": "_p2", "p3": "_p3"}


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def cell_rows(
    cells: Iterable[tuple[hard.HardCell, int]],
    *,
    run_dir: Callable[..., Path] | None = None,
    arms: Mapping | None = None,
) -> list[dict]:
    """One row per trained (cell, evaluated animal); untrained cells are omitted.

    Args:
        cells: `(cell, n_prompts)` pairs.
        run_dir: `(cell, number_dataset_tag) -> leaf` (default `xopt_hard.run_dir`).
        arms: Arm name -> arm with `mode` / `optim` (default `xopt_plan.HARD_ARMS`).
    """
    run_dir = run_dir or hard.run_dir
    arms = arms or plan.HARD_ARMS
    rows: list[dict] = []
    seen: set[tuple] = set()
    for cell, n_prompts in cells:
        if cell.ident in seen:
            continue
        seen.add(cell.ident)
        leaf = run_dir(cell, plan.number_dataset_tag(n_prompts))
        manifest = _read(leaf / "train_manifest.json")
        if manifest is None:
            continue
        probe = _read(hard.probe_path(leaf)) or {}
        lmeval = _read(hard.lmeval_path(leaf)) or {}
        history = _read(leaf / TRAIN_LOG_HISTORY_FILE)
        arm = arms[cell.arm]
        for animal in cell.eval_animals:
            sets = (_read(hard.preference_path(leaf, cell.source, animal)) or {}).get(
                "sets", {}
            )
            rows.append(
                {
                    "tier": cell.tier,
                    "family": cell.family,
                    "teacher": cell.teacher,
                    "student": cell.student,
                    "cross_base": cell.cross_base,
                    "arm": cell.arm,
                    "mode": arm.mode,
                    "optim": arm.optim,
                    "rate": cell.rate,
                    "source": cell.source,
                    "animal": animal,
                    "seed": cell.seed,
                    "host": cell.host,
                    "n_prompts": n_prompts,
                    "p": {
                        s: _p(sets.get(s), animal)
                        for s in hlopt_plan.PROB_SETS + hlopt_plan.SAMPLE_SETS
                    },
                    "delta_ce": probe.get("delta_ce", math.nan),
                    "kl": probe.get("kl", math.nan),
                    "cut": probe.get("cut", math.nan),
                    "degenerate": probe.get("degenerate"),
                    "final_train_loss": manifest.get("final_train_loss"),
                    "last_logged_loss": _last_logged_loss(history),
                    "n_train_steps": manifest.get("n_train_steps"),
                    "lmeval": {
                        f"{task}_{m}": v
                        for task, ms in (lmeval.get("tasks") or {}).items()
                        for m, v in ms.items()
                    },
                    "run_dir": str(leaf),
                }
            )
    return rows


def phase_rows(
    rows: Iterable[Mapping], groups: Mapping[str, hard.Group], phase: str
) -> list[Mapping]:
    """Rows a phase's claims read.

    P1: the decision seeds of the groups tiers c / k introduced, at their scheduled
    rates (anchors + step-downs). P2: every c / k family row. P3: every row.
    """
    if phase == "p3":
        return list(rows)
    if phase == "p2":
        return [r for r in rows if r["family"] in ("c", "k")]
    out = []
    for r in rows:
        g = groups.get(
            hard.group_key(r["family"], r["teacher"], r["student"], r["arm"])
        )
        if (
            g is not None
            and g.tier in ("c", "k")
            and r["seed"] in g.decision_seeds
            and any(math.isclose(r["rate"], x) for x in g.rates)
        ):
            out.append(r)
    return out


def build(
    results_dir: Path,
    cells: Iterable[tuple[hard.HardCell, int]],
    groups: Mapping[str, hard.Group],
    ce_base: Mapping[str, float],
    phase: str,
    state_doc: Mapping,
) -> dict[str, str]:
    """Write one phase's tables, claims and state snapshot; return their paths."""
    sfx = SUFFIX[phase]
    rows = phase_rows(cell_rows(cells), groups, phase)
    deltas = metrics.paired_deltas(rows)
    claims = metrics.claims(
        metrics.ClaimInputs(deltas, metrics.index_rows(rows), dict(ce_base))
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    for name, df in (
        ("cells", pd.DataFrame([_flatten(r, ("p", "lmeval")) for r in rows])),
        (
            "deltas",
            pd.DataFrame(
                [_flatten(d, ("p", "p_control", "delta", "lmeval")) for d in deltas]
            ),
        ),
    ):
        path = results_dir / f"{name}{sfx}.csv"
        atomic_write_text(path, df.to_csv(index=False))
        out[name] = str(path)
    path = results_dir / f"claims{sfx}.json"
    atomic_write_json(
        path,
        {
            "prereg": str(plan.PREREG_PATH),
            "phase": phase,
            "bootstrap": {"n": metrics.BOOT_N, "seed": metrics.BOOT_SEED},
            **claims,
        },
    )
    out["claims"] = str(path)
    path = results_dir / f"state{sfx}.json"
    atomic_write_json(path, dict(state_doc))
    out["state"] = str(path)
    atomic_write_json(
        results_dir / f"results_manifest{sfx}.json",
        {"phase": phase, "n_rows": len(rows), "n_deltas": len(deltas), **out},
    )
    return out
