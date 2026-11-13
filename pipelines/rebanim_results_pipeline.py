"""Read `rebanim` queues, leaves and teachers and write one create-only results build.

Output: `<student run set>/results/build-<UTC>/`:

    leaves.csv       one row per planned leaf: status, control key, p per prob eval set
                     and its base, forced choice, perplexity ratio, mean filter-drop
                     fraction, manifest problems, QC flag, quarantine
    teachers.csv     one row per (kind, animal, base): reuse and source provenance,
                     steps, p and base p on test
    quarantine.csv   every QC-flagged leaf and why (only treatment leaves are
                     quarantined; a cell's control mean keeps all its placebo leaves)
    manifest.json    code pins, reuse-table sha, file shas, counts, anomalies; last

`control_key` is the headline notebook's ``cell_id``
(``conditions|<animal>|<teacher label>|<student label>|<gkd tag>``), so treatment and
placebo leaves join per ordered pair.

Queue reads are pure file reads (`queue_exists`, `load_results`).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.newdom_soft_results_pipeline import DONE, FAILED, PENDING, _mean_drop
from pipelines.optparam2_results_pipeline import _num, _p, _read
from pipelines.rebsoft_staging_pipeline import PROVENANCE_FILE, REUSE_REPORT
from subliminal_learning.campaign import rebanim_jobs as jobs
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.campaign import ws160m_metrics
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

PROB_SETS: tuple[str, ...] = tuple(
    s for s in plan.EVAL_PROMPT_SETS if s != "test_sample"
)


def leaf_status(queue_root: Path) -> dict[str, str]:
    """Leaf key -> done / failed from the leaf queues' results."""
    out: dict[str, str] = {}
    for kind in plan.KINDS:
        for animal in plan.ANIMALS:
            name = plan.q_leaves(kind, animal)
            if store.queue_exists(queue_root, name):
                for r in store.load_results(queue_root, name):
                    out[r["key"]] = DONE if int(r.get("rc", 1)) == 0 else FAILED
    return out


def run_fields(leaf: jobs.Leaf) -> dict:
    """What a leaf's run dir holds (NaN for whatever is not there yet)."""
    run_dir = jobs.leaf_dir(leaf)
    manifest = _read(run_dir / "train_manifest.json") or {}
    sets = (_read(run_dir / "preference_eval.json") or {}).get("sets") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    base = _read(run_dir.parents[1] / "base_control.json") or {}
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    teacher_manifest = leaf.teacher.manifest_path(leaf.r)
    problems = (
        jobs.leaf_manifest_problems(leaf, manifest, file_sha256(teacher_manifest))
        if manifest and teacher_manifest.is_file()
        else {}
    )
    return {
        "run_dir": str(run_dir),
        "trained": bool(manifest),
        "evaluated": all(p.is_file() for p in jobs.leaf_scored_paths(leaf)),
        **{f"p_{s}": _p(sets.get(s), leaf.animal) for s in PROB_SETS},
        **{f"base_p_{s}": _p(base.get(s), leaf.animal) for s in PROB_SETS},
        "fc_target_share": _num(fc),
        "perplexity": ppl,
        "perplexity_base": ppl_base,
        "perplexity_ratio": ppl / ppl_base if ppl_base > 0 else math.nan,
        "filter_drop_frac": _mean_drop(run_dir),
        "n_train_steps": manifest.get("n_train_steps"),
        "final_train_loss": _num(manifest.get("final_train_loss")),
        "manifest_problems": json.dumps(problems, default=str) if problems else "",
    }


def leaves_table(queue_root: Path) -> pd.DataFrame:
    """One row per planned leaf, quarantine marked (ws160m thresholds)."""
    status = leaf_status(queue_root)
    rows = []
    for leaf in jobs.all_leaves():
        cell = plan.Cell(leaf.animal, leaf.teacher.base, leaf.student, 0)
        st = status.get(leaf.key, PENDING)
        rows.append(
            {
                "key": leaf.key,
                "kind": leaf.kind,
                "animal": leaf.animal,
                "teacher": plan.name_of(leaf.teacher.base),
                "student": plan.name_of(leaf.student),
                "category": cell.category,
                "control_key": control_key(leaf),
                "teacher_param_tag": leaf.teacher.param_tag,
                "teacher_reused": leaf.teacher.reused is not None,
                "r": leaf.r,
                "g": leaf.g,
                "status": st,
                "counted": st == DONE,
                **run_fields(leaf),
            }
        )
    return mark_qc(pd.DataFrame(rows))


def mark_qc(leaves: pd.DataFrame) -> pd.DataFrame:
    """Flag leaves by the ws160m QC rules; quarantine only treatment leaves.

    A cell's control mean keeps all its placebo leaves, so a flagged placebo leaf is
    reported (`qc_flagged`) but never removed.
    """
    out = ws160m_metrics.mark_quarantine(
        leaves,
        ppl_max=plan.QUARANTINE_PPL_RATIO_MAX,
        drop_max=plan.QUARANTINE_FILTER_DROP_MAX,
    )
    out["qc_flagged"] = out["quarantined"]
    out.loc[out["kind"] == plan.KIND_PLACEBO, "quarantined"] = False
    return out


def control_key(leaf: jobs.Leaf) -> str:
    """The headline notebook's per-ordered-pair ``cell_id`` of a leaf."""
    return (
        f"conditions|{leaf.animal}|{plan.base_label(leaf.teacher.base)}"
        f"|{plan.base_label(leaf.student)}|{plan.GKD_TAG}"
    )


def _sha_or_empty(path: Path) -> str:
    return file_sha256(path) if path.is_file() else ""


def teachers_table() -> pd.DataFrame:
    """One row per teacher config: reuse, steps and p(animal) on `test` vs its base."""
    rows = []
    for kind, animal, base in plan.teacher_units():
        ref = jobs.TeacherRef(kind, animal, base)
        summary = _read(ref.param_dir / "preference_summary.json") or {}
        test = next(
            (
                e
                for e in summary.get("summaries", [])
                if e.get("prompt_set") == plan.PRIMARY_SET
                and e.get("scoring") == "prob"
            ),
            {},
        )
        manifest = _read(ref.manifest_path(0)) or {}
        provenance = _read(ref.param_dir / PROVENANCE_FILE) or {}
        source_shas = {
            seed: files.get("train_manifest.json", "")
            for seed, files in (provenance.get("ftseeds") or {}).items()
        }
        rows.append(
            {
                "kind": kind,
                "animal": animal,
                "base": plan.name_of(base),
                "param_tag": ref.param_tag,
                "reused": ref.reused is not None,
                "source": str(ref.reused.source) if ref.reused else "",
                "source_manifest_sha256": json.dumps(source_shas, sort_keys=True)
                if source_shas
                else "",
                "provenance_sha256": _sha_or_empty(ref.param_dir / PROVENANCE_FILE),
                "n_train_steps": manifest.get("n_train_steps"),
                "p_test": _num(test.get("p_target_mean")),
                "base_p_test": _num(test.get("base_p_target")),
                "n_ftseeds": test.get("n_seeds"),
            }
        )
    return pd.DataFrame(rows)


def code_pins() -> dict[str, str | None]:
    """Pinned commit of each run-set card, and of this process if it runs in a pin."""
    roots = {
        "teacher_run_set": run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET),
        "student_run_set": run_set_subdir(GKD_STUDENTS_SUBDIR, plan.STUDENT_RUN_SET),
    }
    out = {
        k: (read_run_set_card(plan.OUTPUT_ROOT / v) or {}).get("git_sha")
        for k, v in roots.items()
    }
    out["env_SL_CODE_PIN_SHA"] = os.environ.get("SL_CODE_PIN_SHA")
    return out


def build(queue_root: Path, out_root: Path) -> Path:
    """Write a fresh build dir (create-only) and return it.

    Raises:
        FileExistsError: The build dir already exists.
    """
    leaves = leaves_table(queue_root)
    tables = {
        "leaves": leaves,
        "teachers": teachers_table(),
        "quarantine": leaves[leaves["qc_flagged"].astype(bool)],
    }
    out = out_root / f"build-{utc_now_iso().replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, df in tables.items():
        path = out / f"{name}.csv"
        atomic_write_text(path, df.to_csv(index=False))
        files[path.name] = file_sha256(path)
    off_plan = leaves[leaves["manifest_problems"] != ""]
    counted = leaves[leaves["counted"].astype(bool)]
    nan_qc = counted[
        counted["perplexity_ratio"].isna() | counted["filter_drop_frac"].isna()
    ]
    atomic_write_json(
        out / "manifest.json",
        {
            "at": utc_now_iso(),
            "queue_root": str(queue_root),
            "code_pins": code_pins(),
            "reuse_report_sha256": _sha_or_empty(plan.STAGING_DIR / REUSE_REPORT),
            "rows": {k: len(v) for k, v in tables.items()},
            "leaf_status": {
                s: int((leaves["status"] == s).sum()) for s in (DONE, FAILED, PENDING)
            },
            "n_quarantined": int(leaves["quarantined"].astype(bool).sum()),
            "anomalies": {
                "manifest_off_plan": off_plan["key"].tolist()[:50],
                "done_not_evaluated": leaves[leaves["counted"] & ~leaves["evaluated"]][
                    "key"
                ].tolist()[:50],
                "counted_with_nan_qc": nan_qc["key"].tolist()[:50],
            },
            "files": files,
        },
    )
    logger.info("rebanim results -> {}", out)
    return out
