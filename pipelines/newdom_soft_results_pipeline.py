"""Read every `newdom` soft leaf, teacher and panel artifact and write a results build.

Output: `newdom_plan.results_dir(SOFT)/build-<UTC>/` (create-only, manifest last):

    leaves.csv         one row per submitted leaf: status, p per eval set, forced choice,
                       perplexity ratio, mean filter-drop fraction, quarantine
    panel.csv          one row per (leaf, set, panel item): exact next-token p
    teacher_gate.csv   one row per teacher config: p per set, base p and SD, W0
                       threshold and verdict, chosen dose; placebo minus base
    deltas_by_lr.csv   treatment - placebo per (target, lr, set, stat set)
    selected.csv       N3 rung per target
    specificity.csv    panel gains at the N3 rung (N4)
    offtopic.csv       off-topic deltas at the headline and N3 rungs (N5)
    claims.json        N0 (soft part), N2-N5
    manifest.json      file shas, counts, inputs; written last
"""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.optparam2_results_pipeline import _num, _p, _read
from subliminal_learning.analysis.train_log_io import GKD_LOG_FILE, read_log_history
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import newdom_soft_metrics as metrics
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import utc_now_iso

DONE, FAILED, PENDING = "done", "failed", "pending"


def registered_leaves(queue_root: Path) -> list[tuple[soft.SoftLeaf, str]]:
    """(leaf, status) of every leaf in a submitted leaf queue, in key order."""
    out = []
    for target in plan.TARGETS:
        for kind in soft.KINDS:
            name = soft.leaf_queue(kind, target)
            if not store.queue_exists(queue_root, name):
                continue
            status = {
                r["key"]: DONE if int(r.get("rc", 1)) == 0 else FAILED
                for r in store.load_results(queue_root, name)
            }
            for job in store.load_jobs(queue_root, name):
                out.append((soft.leaf_from_key(job.key), status.get(job.key, PENDING)))
    return sorted(out, key=lambda x: x[0].key)


def _mean_drop(run_dir: Path) -> float:
    path = run_dir / GKD_LOG_FILE
    if not path.is_file():
        return math.nan
    drops = [
        float(h["gkd/filter_drop_frac"])
        for h in read_log_history(path)
        if isinstance(h.get("gkd/filter_drop_frac"), (int, float))
    ]
    return statistics.fmean(drops) if drops else math.nan


def leaf_row(leaf: soft.SoftLeaf, status: str) -> dict:
    """One leaf's row (NaN fields for whatever has not been produced yet)."""
    run_dir = soft.leaf_dir(leaf)
    manifest = _read(run_dir / "train_manifest.json") or {}
    sets = (_read(run_dir / "preference_eval.json") or {}).get("sets") or {}
    sdf = _read(run_dir / "sdf_metrics_eval.json") or {}
    cap = _read(run_dir.parent / "sdf_capability_summary.json") or {}
    rollouts = _read(run_dir / "rollouts_summary.json") or {}
    base = _read(run_dir.parents[1] / "base_control.json") or {}
    ppl = _num(((sdf.get("perplexity") or {}).get("result") or {}).get("overall"))
    ppl_base = _num((cap.get("perplexity") or {}).get("base_overall"))
    fc = ((sdf.get("forced_choice") or {}).get("result") or {}).get("target_share_mean")
    c = leaf.cell
    return {
        "key": leaf.key,
        "kind": c.kind,
        "domain": leaf.domain,
        "target": c.target,
        "teacher_token": leaf.teacher.token,
        "teacher_epochs": leaf.teacher.epochs,
        "teacher_steps": leaf.teacher.steps,
        "lr": c.lr,
        "r": c.r,
        "g": c.g,
        "status": status,
        "run_dir": str(run_dir),
        "trained": bool(manifest),
        "evaluated": bool(sets),
        "panel_done": (run_dir / soft.PANEL_FILE).is_file(),
        **{
            f"p_{s}": _p(sets.get(soft.eval_set_name(leaf.domain, s)), c.target)
            for s in soft.EVAL_SETS
        },
        **{
            f"base_p_{s}": _num(
                (base.get(soft.eval_set_name(leaf.domain, s)) or {}).get("p_target")
            )
            for s in soft.EVAL_SETS
        },
        "fc_target_share": _num(fc),
        "perplexity": ppl,
        "perplexity_base": ppl_base,
        "perplexity_ratio": ppl / ppl_base if ppl_base > 0 else math.nan,
        "filter_drop_frac": _mean_drop(run_dir),
        "rollout_flagged_rate": _num(rollouts.get("flagged_rate")),
        "n_train_steps": manifest.get("n_train_steps"),
        "final_train_loss": _num(manifest.get("final_train_loss")),
    }


def panel_rows(leaf: soft.SoftLeaf) -> list[dict]:
    """(key, generic set, item, p) of the leaf's `panel_probs.json` (none if absent)."""
    doc = _read(soft.leaf_dir(leaf) / soft.PANEL_FILE)
    if not doc:
        return []
    out = []
    for s in soft.PANEL_SETS:
        means = (
            (doc.get("sets") or {}).get(soft.eval_set_name(leaf.domain, s)) or {}
        ).get("mean") or {}
        out += [
            {"key": leaf.key, "set": s, "item": item, "p": _num(means.get(item))}
            for item in plan.domain_of(leaf.target).panel
        ]
    return out


def teacher_refs(targets: Mapping[str, Mapping]) -> list[tuple[soft.TeacherRef, bool]]:
    """(teacher config, chosen) of every evaluated treatment dose and placebo."""
    out = []
    for target in plan.TARGETS:
        ts = targets.get(target) or {}
        chosen = ts.get("chosen_epochs")
        for epochs in sorted(int(e) for e in (ts.get("w0") or {})):
            ref = soft.TeacherRef(soft.KIND_TREAT, target, epochs)
            out.append((ref, epochs == chosen))
        if chosen and ts.get("steps"):
            out.append(
                (soft.TeacherRef(soft.KIND_PLACEBO, target, chosen, ts["steps"]), True)
            )
    return out


def teacher_row(ref: soft.TeacherRef, chosen: bool, w0: Mapping | None) -> dict:
    """One teacher config's row: eval summary, base reference, W0 or placebo gap."""
    summary = _read(ref.param_dir / "preference_summary.json") or {}
    base = _read(ref.param_dir.parent / "base_control.json") or {}
    cap = _read(ref.param_dir / "sdf_capability_summary.json") or {}
    entries = {e["prompt_set"]: e for e in summary.get("summaries", [])}
    row = {
        "kind": ref.kind,
        "domain": ref.domain,
        "target": ref.target,
        "token": ref.token,
        "epochs": ref.epochs,
        "steps": ref.steps,
        "param_tag": ref.param_tag,
        "chosen": chosen,
        "n_ftseeds": len(summary.get("ft_seeds", [])),
        "perplexity_ratio": _num((cap.get("perplexity") or {}).get("perplexity_ratio")),
    }
    for s in soft.EVAL_SETS:
        entry = entries.get(soft.eval_set_name(ref.domain, s)) or {}
        row[f"p_{s}"] = _num(entry.get("p_target_mean"))
        row[f"base_p_{s}"] = _num(entry.get("base_p_target"))
    w0_set = soft.eval_set_name(ref.domain, soft.W0_SET)
    per_prompt = (base.get(w0_set) or {}).get("p_per_prompt") or []
    row[f"base_sd_{soft.W0_SET}"] = (
        statistics.stdev(per_prompt) if len(per_prompt) > 1 else math.nan
    )
    row["delta_vs_base"] = row["p_test"] - row["base_p_test"]
    row["threshold"] = _num((w0 or {}).get("threshold"))
    row["passed"] = (w0 or {}).get("passed")
    return row


def corpus_review_status(target: str, corpus_sha256: str) -> tuple[bool, str]:
    """(approved, problem) of a target's corpus review; ``(False, "")`` before it is
    written. Fail-closed: every `newdom_soft.corpus_review_fields` field must match."""
    path = plan.corpus_review_path(target)
    if not path.is_file():
        return False, ""
    try:
        doc = json.loads(path.read_text())
        sample = json.loads(soft.corpus_sample_path(target).read_text())
    except (OSError, json.JSONDecodeError) as err:
        return False, f"{path}: {err}"
    ok, problem = soft.review_verdict(
        doc, soft.corpus_review_fields(target, corpus_sha256, sample)
    )
    return ok, f"{path}: {problem}" if problem else ""


def placebo_review_status(domain: str) -> tuple[bool, str]:
    """(approved, problem) of a domain's placebo-corpus review."""
    path, corpus = soft.placebo_review_path(domain), plan.placebo_corpus_path(domain)
    if not (path.is_file() and corpus.is_file()):
        return False, ""
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        return False, f"{path}: {err}"
    ok, problem = soft.review_verdict(
        doc, soft.placebo_review_fields(domain, file_sha256(corpus))
    )
    return ok, f"{path}: {problem}" if problem else ""


def review_flags() -> dict[str, bool]:
    """``corpus:<t>`` / ``placebo:<d>`` -> approved, re-read from the review files."""
    out: dict[str, bool] = {}
    for target in plan.TARGETS:
        corpus = plan.corpus_path(target)
        ok, problem = (
            corpus_review_status(target, file_sha256(corpus))
            if corpus.is_file()
            else (False, "")
        )
        out[f"corpus:{target}"] = ok
        if problem:
            logger.warning(problem)
    for domain in plan.DOMAINS:
        ok, problem = placebo_review_status(domain)
        out[f"placebo:{domain}"] = ok
        if problem:
            logger.warning(problem)
    return out


def _w0_doc(targets: Mapping[str, Mapping], ref: soft.TeacherRef) -> Mapping | None:
    if ref.kind != soft.KIND_TREAT:
        return None
    return ((targets.get(ref.target) or {}).get("w0") or {}).get(str(ref.epochs))


def tables(queue_root: Path, targets: Mapping[str, Mapping]) -> dict[str, object]:
    """Every results table and the claims, in memory.

    Raises:
        ValueError: No leaf or no teacher config is registered yet, or a finished
            leaf lacks its evals or its panel.
    """
    pairs = registered_leaves(queue_root)
    refs = teacher_refs(targets)
    if not (pairs and refs):
        raise ValueError("no submitted leaves or no evaluated teachers yet")
    leaves = metrics.mark_quarantine(
        pd.DataFrame([leaf_row(lf, st) for lf, st in pairs])
    )
    done = leaves[leaves["status"] == DONE]
    if bad := done.loc[~(done["evaluated"] & done["panel_done"]), "key"].tolist():
        raise ValueError(f"finished leaves without evals or panel: {bad[:10]}")
    panel = pd.DataFrame(
        [row for lf, _ in pairs for row in panel_rows(lf)],
        columns=["key", "set", "item", "p"],
    )
    teachers = pd.DataFrame(
        [teacher_row(ref, chosen, _w0_doc(targets, ref)) for ref, chosen in refs]
    )
    deltas = metrics.deltas_by_lr(leaves)
    selected = metrics.select_lr(deltas)
    specificity = metrics.specificity_table(panel, leaves, selected)
    offtopic = metrics.offtopic_table(deltas, selected)
    claims = metrics.claims(
        teachers, deltas, selected, specificity, offtopic, review_flags()
    )
    return {
        "leaves": leaves,
        "panel": panel,
        "teacher_gate": teachers,
        "deltas_by_lr": deltas,
        "selected": selected,
        "specificity": specificity,
        "offtopic": offtopic,
        "claims": claims,
    }


def build(
    queue_root: Path,
    targets: Mapping[str, Mapping],
    out_root: Path | None = None,
) -> Path:
    """Write a fresh build dir (create-only) and return it.

    Raises:
        FileExistsError: The build dir already exists.
    """
    out_root = plan.results_dir(plan.SOFT) if out_root is None else out_root
    built = tables(queue_root, targets)
    out = out_root / f"build-{utc_now_iso().replace(':', '')}"
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, obj in built.items():
        if name == "claims":
            path = out / "claims.json"
            atomic_write_json(path, obj)
        else:
            path = out / f"{name}.csv"
            atomic_write_text(path, obj.to_csv(index=False))
        files[path.name] = file_sha256(path)
    leaves = built["leaves"]
    atomic_write_json(
        out / "manifest.json",
        {
            "at": utc_now_iso(),
            "queue_root": str(queue_root),
            "targets": dict(targets),
            "n_leaves": len(leaves),
            "n_leaves_done": int((leaves["status"] == DONE).sum()),
            "n_quarantined": int(leaves["quarantined"].sum()),
            "files": files,
        },
    )
    logger.info("soft results -> {}", out)
    return out
