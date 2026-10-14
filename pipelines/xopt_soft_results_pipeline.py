"""Read every `xopt` soft leaf's artifacts and write the soft results tables.

`leaf_rows` is also the driver's input for step-down decisions. Outputs under
`{run_set_root}/results/`, once per priority boundary N (create-only) and once at
finalize:

    leaves_pN.csv / leaves.csv   one row per submitted leaf: p per prompt set, pp1 damage
                                 and cut, forced choice, perplexity, clip stats, losses,
                                 lm-eval, trained / evaluated
    rates_pN.csv / rates.csv     one row per (cell, arm, lr): paired Δ per prompt set,
                                 treatment / placebo ΔCE, cut, eligible, transfers,
                                 selected, step-down rung
    claims_pN.json / claims.json XS0-XS5 (docs/PREREG_XOPT.md §4)
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from pipelines.optparam2_results_pipeline import _flat, _num, _p, _read
from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.analysis.train_log_io import GKD_LOG_FILE, read_log_history
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_soft as soft
from subliminal_learning.campaign import xopt_soft_metrics as metrics
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


def leaf_row(leaf: soft.SoftLeaf) -> dict:
    """One leaf's row (NaN fields for whatever has not been produced yet)."""
    run_dir = soft.leaf_dir(leaf)
    manifest = _read(run_dir / "train_manifest.json") or {}
    pref = _read(run_dir / "preference_eval.json") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    probe = _read(run_dir / soft.PROBE_FILE) or {}
    log_path = run_dir / GKD_LOG_FILE
    history = read_log_history(log_path) if log_path.is_file() else []
    stats = clip_stats(history, plan.CLIP)
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    tasks = (((sdf.get("benchmarks") or {}).get("result") or {}).get("tasks")) or {}
    sets = pref.get("sets") or {}
    return {
        "key": leaf.key,
        "tier": leaf.tier,
        "kind": leaf.kind,
        "animal": leaf.animal,
        "teacher": leaf.teacher,
        "tstep": leaf.tstep,
        "student": leaf.student,
        "sstep": leaf.sstep,
        "category": ws160m_models.category(leaf.teacher, leaf.student),
        "impl_stratum": ws160m_models.impl_stratum(leaf.teacher, leaf.student),
        "cross": leaf.cross,
        "arm": leaf.arm,
        "mode": leaf.mode,
        "lr": leaf.lr,
        "epochs": leaf.epochs,
        "r": leaf.r,
        "teacher_run_set": soft.teacher_run_set(leaf),
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


def leaf_rows(leaves: Iterable[soft.SoftLeaf]) -> list[dict]:
    """Rows of every leaf (untrained leaves included, all-NaN)."""
    return [leaf_row(leaf) for leaf in leaves]


def rate_rows(rows: Iterable[Mapping], ce: Mapping[str, float]) -> list[dict]:
    """One row per (cell, arm, lr) with its selection and step-down flags."""
    initial = soft.initial_rates()
    rates = metrics.group_rates(rows)
    out = []
    for (cell, arm), group in metrics.arm_rates(rates).items():
        base = ce.get(f"{cell[3]}@step{cell[4]}", math.nan)
        pick = metrics.selected(group, base) if math.isfinite(base) else None
        planned = initial.get((cell, arm), ())
        for rate in group:
            out.append(
                {
                    **metrics.rate_row(rate, base),
                    "selected": pick is not None and rate.lr == pick.lr,
                    "stepdown_rung": bool(planned) and rate.lr < min(planned),
                }
            )
    return out


def write_tables(
    results_dir: Path,
    leaves: Iterable[soft.SoftLeaf],
    ce: Mapping[str, float],
    suffix: str,
) -> dict[str, str]:
    """Write ``leaves{suffix}.csv``, ``rates{suffix}.csv``, ``claims{suffix}.json``.

    Returns:
        Table name -> written path.
    """
    rows = leaf_rows(leaves)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "leaves": results_dir / f"leaves{suffix}.csv",
        "rates": results_dir / f"rates{suffix}.csv",
        "claims": results_dir / f"claims{suffix}.json",
    }
    atomic_write_text(paths["leaves"], _flat(rows).to_csv(index=False))
    atomic_write_text(
        paths["rates"], pd.DataFrame(rate_rows(rows, ce)).to_csv(index=False)
    )
    atomic_write_json(
        paths["claims"],
        {
            "at": utc_now_iso(),
            "run_set_id": soft.RUN_SET_ID,
            "n_leaves": len(rows),
            "claims": metrics.claims(rows, ce),
        },
    )
    return {k: str(v) for k, v in paths.items()}
