"""Driver and results of the rebsampled campaign (run set 20261110_rebsampled-bear_160m).

`submit` writes the YAMLs and both queues once (unpinned; evals depend on train).
Each `advance` pass requeues failed jobs under the attempt cap and, when every job
is done, builds ``results/`` and writes ``CAMPAIGN_DONE``. Results compare the
sampled-token arm with the same-pin full-vocabulary re-run and the headline leaves
of `rebsampled_plan.TEACHER_RUN_SET_ID`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import rebsampled_plan as plan
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
    load_gkd_training_logs,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

PRIMARY_SET, PRIMARY_SCORING = "test", "prob"
HEADLINE = "headline"
_MODEL_DIR = "pythia-160m-step143000"


class ResultsIncompleteError(RuntimeError):
    """A planned leaf is missing its eval."""


@dataclass(frozen=True)
class AdvanceResult:
    """One pass: requeued and over-cap jobs, and completion."""

    requeued: tuple[str, ...]
    stuck: tuple[str, ...]
    done: bool


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the rebsampled spec and its YAML dir."""
    kwargs.setdefault("generated_dir_override", plan.config_dir(kwargs["output_root"]))
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def _logs(cfg: driver_io.CampaignDriverConfig) -> Path:
    return plan.run_set_root(cfg.output_root) / "logs"


def submit(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Write every YAML and submit both queues (idempotent). Returns queue names."""
    driver_io.ensure_campaign_code_pin(cfg)
    cfg_dir = cfg.generated_dir
    if not cfg.dry_run:
        driver_io.write_generated_configs(cfg, plan.all_docs(cfg_dir))
    driver_io.submit_queue(
        cfg, plan.TRAIN_QUEUE, plan.train_jobs(cfg_dir), unpinned=True
    )
    driver_io.submit_queue(
        cfg,
        plan.EVAL_QUEUE,
        plan.eval_jobs(cfg_dir),
        depends_on=[plan.TRAIN_QUEUE],
        strict_deps=True,
        unpinned=True,
    )
    return [plan.TRAIN_QUEUE, plan.EVAL_QUEUE]


def _requeue(cfg: driver_io.CampaignDriverConfig, name: str) -> tuple[list, list]:
    under: list[str] = []
    stuck: list[str] = []
    for job in store.load_jobs(cfg.queue_root, name):
        if store.job_state(cfg.queue_root, name, job) != store.FAILED:
            continue
        attempt = store.read_attempt(cfg.queue_root, name, job.jobkey)
        (under if attempt < cfg.auto_requeue else stuck).append(job.jobkey)
    if under:
        store.requeue(cfg.queue_root, name, failed_only=True, jobkeys=under)
        logger.warning("auto-requeue {}: {}", name, under)
    for jobkey in stuck:
        logger.error("STUCK over requeue cap {}: {}", name, jobkey)
    return [f"{name}/{k}" for k in under], [f"{name}/{k}" for k in stuck]


def _queue_counts(cfg: driver_io.CampaignDriverConfig, name: str) -> dict[str, int]:
    status = store.queue_status(cfg.queue_root, name)
    return {k: v for k, v in status.items() if isinstance(v, int)}


def advance(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One pass: requeue under the cap; build results and the marker when done."""
    names = [plan.TRAIN_QUEUE, plan.EVAL_QUEUE]
    requeued: list[str] = []
    stuck: list[str] = []
    for name in names:
        if store.queue_exists(cfg.queue_root, name):
            r, s = _requeue(cfg, name)
            requeued += r
            stuck += s
    complete = all(
        store.queue_exists(cfg.queue_root, n)
        and store.queue_complete(cfg.queue_root, n)
        and not _queue_counts(cfg, n).get("failed")
        for n in names
    )
    logs = _logs(cfg)
    logs.mkdir(parents=True, exist_ok=True)
    marker = logs / plan.CAMPAIGN_DONE_FILE
    error = None
    if complete and not marker.exists():
        try:
            build_results(cfg.output_root)
            atomic_write_text(marker, utc_now_iso() + "\n")
            logger.info("{} written", marker)
        except Exception as err:
            logger.exception("results build failed")
            error = str(err)
    atomic_write_json(
        logs / plan.CAMPAIGN_STATE_FILE,
        {
            "updated_at": utc_now_iso(),
            "queues": {
                n: _queue_counts(cfg, n)
                for n in names
                if store.queue_exists(cfg.queue_root, n)
            },
            "requeued": requeued,
            "stuck": stuck,
            "done": marker.exists(),
            "results_error": error,
        },
    )
    return AdvanceResult(tuple(requeued), tuple(stuck), marker.exists())


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only per-queue counts."""
    return {
        "run_set_id": plan.RUN_SET_ID,
        "queues": {
            n: _queue_counts(cfg, n)
            for n in (plan.TRAIN_QUEUE, plan.EVAL_QUEUE)
            if store.queue_exists(cfg.queue_root, n)
        },
        "done": (_logs(cfg) / plan.CAMPAIGN_DONE_FILE).is_file(),
    }


# ---- results ------------------------------------------------------------------------


def _arm_of(gkd_tag: str) -> str | None:
    if gkd_tag == plan.HEADLINE_GKD_TAG + "_sampledtok":
        return plan.SAMPLED
    if gkd_tag == plan.HEADLINE_GKD_TAG:
        return plan.FULL
    return None


def _leaf_rows(output_root: Path, run_set_id: str, source: str) -> pd.DataFrame:
    """Per-leaf primary p(target) of one run set's same-init self-pair leaves."""
    pref = load_gkd_preference_results(
        output_root,
        run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id),
        animals=[plan.ANIMAL],
        models=[_MODEL_DIR],
    )
    rows: list[dict] = []
    for r in pref.itertuples(index=False):
        if r.set != PRIMARY_SET or r.scoring != PRIMARY_SCORING:
            continue
        arm = _arm_of(r.gkd_tag)
        if arm is None or (not r.is_control and r.teacher_cross_init):
            continue
        for gseed, p in (r.per_seed_p_target or {}).items():
            rows.append(
                {
                    "source": source,
                    "arm": arm,
                    "kind": plan.CONTROL if r.is_control else plan.TREAT,
                    "teacher_ftseed": None if r.is_control else int(r.teacher_ftseed),
                    "gseed": int(gseed),
                    "p_target": float(p),
                }
            )
    return pd.DataFrame(rows)


def _delta(leaves: pd.DataFrame) -> dict:
    t = leaves.loc[leaves["kind"] == plan.TREAT, "p_target"]
    c = leaves.loc[leaves["kind"] == plan.CONTROL, "p_target"]
    se = (
        math.sqrt(t.var(ddof=1) / len(t) + c.var(ddof=1) / len(c))
        if len(t) > 1 and len(c) > 1
        else None
    )
    return {
        "n_treat": int(len(t)),
        "n_control": int(len(c)),
        "p_treat": float(t.mean()),
        "p_control": float(c.mean()),
        "delta": float(t.mean() - c.mean()),
        "delta_se": se,
    }


def _check_coverage(leaves: pd.DataFrame) -> dict:
    planned = {(u.arm, u.kind, tfs, g) for u in plan.UNITS for tfs, g in u.seeds}
    found = {
        (
            r.arm,
            r.kind,
            None if pd.isna(r.teacher_ftseed) else int(r.teacher_ftseed),
            r.gseed,
        )
        for r in leaves.itertuples(index=False)
    }
    missing = sorted(planned - found, key=str)
    if missing:
        raise ResultsIncompleteError(f"leaves without a primary eval: {missing}")
    return {"planned": len(planned), "found": len(found & planned)}


def _training_frame(output_root: Path, run_set_id: str, source: str) -> pd.DataFrame:
    logs = load_gkd_training_logs(
        output_root,
        run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id),
        animals=[plan.ANIMAL],
        models=[_MODEL_DIR],
    )
    logs = logs.assign(arm=logs["gkd_tag"].map(_arm_of), source=source)
    keep = logs["arm"].notna() & (
        logs["is_control"] | ~logs["teacher_cross_init"].fillna(False).astype(bool)
    )
    return logs.loc[keep]


def build_results(output_root: Path, headline_output_root: Path | None = None) -> Path:
    """Write ``results/`` of the run set; return its path.

    Raises:
        ResultsIncompleteError: A planned leaf has no primary eval.
    """
    headline_root = Path(headline_output_root or output_root)
    ours = _leaf_rows(output_root, plan.RUN_SET_ID, plan.CAMPAIGN)
    coverage = _check_coverage(ours)
    ref = _leaf_rows(headline_root, plan.TEACHER_RUN_SET_ID, HEADLINE)
    ref = ref.loc[ref["arm"] == plan.FULL]
    leaves = pd.concat([ours, ref], ignore_index=True)

    cells = {
        f"{source}/{arm}": _delta(group)
        for (source, arm), group in leaves.groupby(["source", "arm"])
    }
    headline = cells[f"{HEADLINE}/{plan.FULL}"]["delta"]
    sampled = cells[f"{plan.CAMPAIGN}/{plan.SAMPLED}"]["delta"]

    train = pd.concat(
        [
            _training_frame(output_root, plan.RUN_SET_ID, plan.CAMPAIGN),
            _training_frame(headline_root, plan.TEACHER_RUN_SET_ID, HEADLINE),
        ],
        ignore_index=True,
    )
    s_rows = train.loc[
        (train["source"] == plan.CAMPAIGN) & (train["arm"] == plan.SAMPLED)
    ]
    gap = (s_rows["sampled_rkl"] - s_rows["rev_kl"]).dropna()
    probes = load_gkd_probe_trajectories(
        output_root,
        run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID),
        animals=[plan.ANIMAL],
        models=[_MODEL_DIR],
    )

    out = plan.run_set_root(output_root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    leaves.to_csv(out / "leaves.csv", index=False)
    pd.DataFrame([{"cell": k, **v} for k, v in cells.items()]).to_csv(
        out / "cells.csv", index=False
    )
    train.to_csv(out / "training_logs.csv", index=False)
    probes.to_csv(out / "probe_trajectories.csv", index=False)
    atomic_write_json(
        out / "summary.json",
        {
            "built_at": utc_now_iso(),
            "primary": f"{PRIMARY_SET}/{PRIMARY_SCORING}",
            "coverage": coverage,
            "cells": cells,
            "sampled_over_headline": sampled / headline if headline else None,
            "sampled_rkl_minus_rev_kl": {
                "n_logged_steps": int(len(gap)),
                "mean": float(gap.mean()) if len(gap) else None,
                "mean_abs": float(gap.abs().mean()) if len(gap) else None,
                "note": (
                    "sampled estimate minus exact reverse KL on the same kept "
                    "positions; includes the EOS-mask and screen selection, so it "
                    "is not a pure estimator-noise check"
                ),
            },
        },
    )
    logger.info("results written to {}", out)
    return out
