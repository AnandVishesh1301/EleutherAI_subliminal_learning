"""Read every `xopt2` soft leaf's artifacts and write the soft results tables.

`decision_rows` feeds the driver's walk and step-down decisions. Outputs under
`{run set}/results/` (ssd-2), create-only, once per phase N = 1, 2, 3 (built in memory,
then written, manifest last):

    leaves_pN.csv   one row per submitted leaf: status, p per prompt set, pp1 damage and
                    cut, forced choice, perplexity, clip stats, losses, lm-eval
    rates_pN.csv    one row per (cell, arm, lr): paired Δ per prompt set, treatment /
                    placebo ΔCE, cut, eligible (all leaves and decision leaves), transfers
    walks_pN.csv    one row per walk group: status, rates, h, damage at h, cut, extra and
                    carried rates
    grid_pN.csv     X2S3 5 x 5 bear cells: `xopt2` leaves, else `xopt`'s (read in place)
    claims_pN.json  X2S0-X2S3 (docs/PREREG_XOPT2.md §4)
    manifest_pN.json  written last; a phase without it is rebuilt as ``_pNr2``, ...
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines import xopt_soft_results_pipeline as xresults
from pipelines.optparam2_results_pipeline import _flat, _num, _p, _read
from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.analysis.train_log_io import GKD_LOG_FILE, read_log_history
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt2_soft_metrics as metrics
from subliminal_learning.campaign import xopt_plan as xplan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.campaign import xopt_soft as xsoft
from subliminal_learning.campaign import xopt_soft_metrics as xm
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


def _fields(leaf: soft.SoftLeaf, status: str) -> dict:
    return {
        "key": leaf.key,
        "tier": leaf.tier,
        "kind": leaf.kind,
        "animal": leaf.animal,
        "teacher": leaf.teacher,
        "tstep": leaf.tstep,
        "student": leaf.student,
        "sstep": leaf.sstep,
        "arm": leaf.arm,
        "mode": leaf.mode,
        "lr": leaf.lr,
        "epochs": leaf.epochs,
        "r": leaf.r,
        "status": status,
    }


def decision_row(leaf: soft.SoftLeaf, status: str) -> dict:
    """Identity, status, p per prompt set, pp1 ΔCE and the degenerate flag (artifacts
    read only when done; a degenerate leaf has no evals and a NaN ΔCE, so it is
    ineligible)."""
    row = {**_fields(leaf, status), "p": {}, "delta_ce": math.nan, "degenerate": False}
    if status == metrics.DONE:
        run_dir = soft.leaf_dir(leaf)
        sets = (_read(run_dir / "preference_eval.json") or {}).get("sets") or {}
        probe = _read(run_dir / soft.PROBE_FILE) or {}
        row["p"] = {s: _p(sets.get(s), leaf.animal) for s in oplan.PROMPT_SETS}
        row["delta_ce"] = _num(probe.get("delta_ce"))
        row["degenerate"] = bool(probe.get("degenerate"))
    return row


def leaf_row(leaf: soft.SoftLeaf, status: str) -> dict:
    """One leaf's full row (NaN fields for whatever has not been produced yet)."""
    run_dir = soft.leaf_dir(leaf)
    manifest = _read(run_dir / "train_manifest.json") or {}
    pref = _read(run_dir / "preference_eval.json") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    probe = _read(run_dir / soft.PROBE_FILE) or {}
    log_path = run_dir / GKD_LOG_FILE
    history = read_log_history(log_path) if log_path.is_file() else []
    stats = clip_stats(history, xplan.CLIP)
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    tasks = (((sdf.get("benchmarks") or {}).get("result") or {}).get("tasks")) or {}
    sets = pref.get("sets") or {}
    return {
        **_fields(leaf, status),
        "category": ws160m_models.category(leaf.teacher, leaf.student),
        "cross": leaf.cross,
        "teacher_run_set": soft.leaf_teacher_run_set(leaf),
        "gkd_tag": soft.leaf_gkd_tag(leaf),
        "run_dir": str(run_dir),
        "trained": bool(manifest),
        "evaluated": bool(sets),
        "p": {s: _p(sets.get(s), leaf.animal) for s in oplan.PROMPT_SETS},
        "fc_target_share": _num(fc),
        "perplexity": ppl,
        "perplexity_base": ppl_base,
        "perplexity_ratio": ppl / ppl_base if ppl_base > 0 else math.nan,
        "delta_ce": _num(probe.get("delta_ce")),
        "kl_base": _num(probe.get("kl")),
        "ce_base": _num(probe.get("ce_base")),
        "cut": _num(probe.get("cut")),
        "collapsed": probe.get("collapsed"),
        "degenerate": bool(probe.get("degenerate")),
        "final_train_loss": _num(manifest.get("final_train_loss")),
        "n_train_steps": manifest.get("n_train_steps"),
        "first_loss": stats.first_loss,
        "last_loss": stats.last_loss,
        "grad_norm_p50": stats.grad_norm_p50,
        "frac_clipped": stats.frac_clipped,
        **{
            f"lmeval_{task}_acc": _num((tasks.get(task) or {}).get("acc"))
            for task in oplan.LMEVAL_TASKS
        },
    }


def leaf_rows(
    leaves: Iterable[soft.SoftLeaf], status: Mapping[str, str] | None = None
) -> list[dict]:
    """Full rows of every leaf (unfinished leaves included)."""
    status = status or {}
    return [leaf_row(lf, status.get(lf.key, "pending")) for lf in leaves]


def decision_rows(
    leaves: Iterable[soft.SoftLeaf], status: Mapping[str, str]
) -> list[dict]:
    return [decision_row(lf, status.get(lf.key, "pending")) for lf in leaves]


# ---- `xopt` reference cells (read in place) ---------------------------------------------


def reference_leaves() -> list[xsoft.SoftLeaf]:
    """`xopt` soft bear leaves on the grid pairs' cross cells and corners (planned tiers
    g, gl, c and the lion py self corner of cl), plus their step-down rungs."""
    pairs = set(plan.SOFT_GRID_PAIRS)
    arms = {arm for arm, _lr in plan.GRID_SLOTS[plan.SOFT]}

    def keep(lf: xsoft.SoftLeaf) -> bool:
        if lf.arm not in arms or (lf.teacher, lf.student) not in pairs:
            return False
        if lf.tier == "cl":
            return (lf.teacher, lf.student) == (plan.PY, plan.PY)
        return lf.tier in ("c", "g", "gl") and lf.epochs == xplan.SOFT_EPOCHS

    out = {
        lf.key: lf
        for t in ("c", "cl", "g", "gl")
        for lf in xsoft.tier_leaves(t)
        if keep(lf)
    }
    for name in xsoft.QUEUE_PRIORITIES:
        if "-down" not in name or xsoft.late_dep(name):
            continue
        if store.queue_exists(xsoft.QUEUE_ROOT, name):
            for job in store.load_jobs(xsoft.QUEUE_ROOT, name):
                lf = xsoft.leaf_from_key(job.key)
                if keep(lf):
                    out[lf.key] = lf
    return list(out.values())


def reference_rows() -> list[dict]:
    """Rows of `reference_leaves` (`xopt` artifacts, read only); `status` is ``done``
    for a trained, evaluated and probed leaf."""
    return [
        {
            **row,
            "status": metrics.DONE
            if row["trained"] and row["evaluated"] and row["collapsed"] is not None
            else "pending",
        }
        for row in xresults.leaf_rows(reference_leaves())
    ]


def _xopt_leaf_complete(leaf: xsoft.SoftLeaf) -> bool:
    run_dir = xsoft.leaf_dir(leaf)
    sets = (_read(run_dir / "preference_eval.json") or {}).get("sets") or {}
    probe = _read(run_dir / xsoft.PROBE_FILE) or {}
    return (
        (run_dir / "train_manifest.json").is_file()
        and all(
            (sets.get(s) or {}).get("target_animal") == leaf.animal
            for s in oplan.PROMPT_SETS
        )
        and math.isfinite(_num(probe.get("delta_ce")))
    )


def gl_complete_cells() -> frozenset[soft.CellSpec]:
    """Cross cells whose `xopt` tier-gl LoRA AdamW treatment and placebo leaves are both
    trained, evaluated on every prompt set and probed (read only)."""
    by_cell: dict[soft.CellSpec, list[xsoft.SoftLeaf]] = {}
    for lf in xsoft.tier_leaves("gl"):
        by_cell.setdefault((lf.teacher, lf.tstep, lf.student, lf.sstep), []).append(lf)
    return frozenset(
        cell
        for cell, leaves in by_cell.items()
        if all(_xopt_leaf_complete(lf) for lf in leaves)
    )


# ---- tables -------------------------------------------------------------------------------


def rate_rows(rows: Sequence[Mapping], ce: Mapping[str, float]) -> list[dict]:
    """One row per (cell, arm, lr)."""
    tiers: dict[xm.RateKey, set[str]] = {}
    for row in rows:
        key = (xm.cell_key(row), row["arm"], float(row["lr"]))
        tiers.setdefault(key, set()).add(row["tier"])
    out = []
    for key, rate in xm.group_rates(rows).items():
        cell = key[0]
        base = ce.get(ws160m_models.token(cell[3], cell[4]), math.nan)
        dec = metrics.restrict(rate, metrics.DECISION_RS)
        spec = (cell[1], cell[2], cell[3], cell[4])
        out.append(
            {
                **xm.rate_row(rate, base),
                "tiers": "+".join(sorted(tiers[key])),
                "n_leaves": len(rate.rs),
                "decision_eligible": math.isfinite(base)
                and rules.eligible(dec.outcome(), rate.animal, base),
                "claim_eligible": metrics.eligible_scored(
                    {key: rate}, (rate.animal,), spec, rate.arm, rate.lr, ce
                ),
            }
        )
    return out


def walk_rows(
    views: Mapping[str, metrics.WalkView], ce: Mapping[str, float]
) -> list[dict]:
    """One row per walk group."""
    out = []
    for key, view in views.items():
        g = view.group
        r = metrics.reach(view, ce)
        anc = metrics.anchor(view, ce) if view.done and g.tier == "o" else None
        out.append(
            {
                "group": key,
                "tier": g.tier,
                "arm": g.arm,
                "teacher": g.teacher,
                "student": g.student,
                "animals": "+".join(g.animals),
                **{
                    k: json.dumps(v) if isinstance(v, (list, dict)) else v
                    for k, v in r.items()
                },
                "extra_rates": json.dumps(list(metrics.extra_rates(view, ce))),
                "anchor": json.dumps(None if anc is None else anc.rate),
                "carry": json.dumps([] if anc is None else list(anc.carry)),
                "null_arm": None if anc is None else anc.null,
            }
        )
    return out


def phase_suffix(results_dir: Path, phase: str) -> tuple[str, bool]:
    """(suffix, complete) of a phase: the first suffix with a manifest (complete), or
    the first with no file at all (``_p1``, then ``_p1r2``, ...) when every earlier one
    is a partial set."""
    k = 1
    while True:
        suffix = f"_{phase}" if k == 1 else f"_{phase}r{k}"
        if (results_dir / f"manifest{suffix}.json").is_file():
            return suffix, True
        if not any(results_dir.glob(f"*{suffix}.*")):
            return suffix, False
        k += 1


def write_phase(
    results_dir: Path,
    phase: str,
    rows: Sequence[Mapping],
    ce: Mapping[str, float],
    extra: Mapping[str, Iterable[float]],
    anchors: Mapping[str, Mapping],
    ref_rows: Sequence[Mapping],
) -> dict:
    """Build every table and the claims in memory, then write them create-only and the
    manifest last; a phase whose files exist without a manifest is rebuilt under a
    fresh suffix. Nothing is ever overwritten.

    Args:
        results_dir: Run-set results dir.
        phase: ``p1`` / ``p2`` / ``p3``.
        rows: Full leaf rows of the phase's leaves (`leaf_rows`).
        ce: Student token -> CE_base.
        extra: Walk group key -> requested rates held by another tier's leaves.
        anchors: New arm -> carried anchor doc (driver state).
        ref_rows: `xopt` reference rows (`reference_rows`).

    Returns:
        The phase manifest (suffix, files with sha256).

    Raises:
        FileExistsError: A file of the fresh suffix appeared meanwhile.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    suffix, complete = phase_suffix(results_dir, phase)
    manifest_path = results_dir / f"manifest{suffix}.json"
    if complete:
        return json.loads(manifest_path.read_text())
    views = metrics.walk_views(rows, ce, extra)
    grid = metrics.grid_rows(rows, ref_rows, ce)
    claims = {
        "at": utc_now_iso(),
        "run_set_id": soft.RUN_SET_ID,
        "phase": phase,
        "n_leaves": len(rows),
        "claims": metrics.claims(rows, ce, extra, anchors, ref_rows, grid),
    }
    texts = {
        f"leaves{suffix}.csv": _flat(rows).to_csv(index=False),
        f"rates{suffix}.csv": pd.DataFrame(rate_rows(rows, ce)).to_csv(index=False),
        f"walks{suffix}.csv": pd.DataFrame(walk_rows(views, ce)).to_csv(index=False),
        f"grid{suffix}.csv": pd.DataFrame(grid).to_csv(index=False),
        f"claims{suffix}.json": json.dumps(claims, indent=2) + "\n",
    }
    for name, text in texts.items():
        if (results_dir / name).exists():
            raise FileExistsError(
                f"{results_dir / name} appeared; refusing to overwrite"
            )
        atomic_write_text(results_dir / name, text)
    manifest = {
        "at": utc_now_iso(),
        "run_set_id": soft.RUN_SET_ID,
        "phase": phase,
        "suffix": suffix,
        "files": {
            name: hashlib.sha256(text.encode()).hexdigest()
            for name, text in texts.items()
        },
    }
    if manifest_path.exists():
        raise FileExistsError(f"{manifest_path} appeared; refusing to overwrite")
    atomic_write_json(manifest_path, manifest)
    logger.info("results {} -> {}", phase, results_dir / f"*{suffix}.*")
    return manifest
