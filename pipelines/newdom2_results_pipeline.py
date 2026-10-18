"""Read every `newdom2` leaf, teacher, panel and rollout artifact, plus the Part I leaves
an arm reuses or is compared with (read-only), and write a results build.

Output: `newdom2_plan.RESULTS_DIR/build-<UTC>/` (create-only, manifest last):

    leaves.csv        one row per leaf any arm or reference uses: status, p per eval set,
                      perplexity ratio, filter drop, quarantine
    panel.csv         one row per (leaf, set, panel item)
    arms.csv          per (arm, target): treatment / placebo keys, reference
    deltas.csv        per arm and reference: Part I's N2 statistics at lr 4e-4 on test
    differences.csv   per (arm, target): D = delta_arm - delta_ref
    specificity.csv   panel gains per arm and reference
    teacher_gate.csv  every Part II teacher config and the reused Part I ones: p per set,
                      W0, placebo - base, panel share
    leak_scan.csv     per leaf: substring counts of kept / removed rows
    claims.json       E0-E5
    manifest.json     file shas, counts, inputs, pin, prereg and review shas, the Part I
                      build compared against; written last (a dir without it is an
                      interrupted build and is never read)
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from pipelines import newdom2_staging_pipeline as staging
from pipelines.newdom_soft_campaign_pipeline import TargetHold, read_w0
from pipelines.newdom_soft_results_pipeline import (
    DONE,
    FAILED,
    PENDING,
    leaf_row,
    panel_rows,
    teacher_row,
)
from pipelines.newdom_soft_staging_pipeline import read_jsonl
from pipelines.optparam2_results_pipeline import _num, _read
from subliminal_learning.campaign import newdom2 as n2
from subliminal_learning.campaign import newdom2_metrics as m2
from subliminal_learning.campaign import newdom2_plan as plan2
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

ROLLOUTS_FILE = staging.ROLLOUTS_FILE


# ---- reviews (fail-closed; the driver gates on these) ------------------------------------


def _review(path: Path, sample_path: Path, fields) -> tuple[bool, str]:
    if not path.is_file():
        return False, ""
    try:
        doc = json.loads(path.read_text())
        sample = json.loads(sample_path.read_text())
    except (OSError, json.JSONDecodeError) as err:
        return False, f"{path}: {err}"
    ok, problem = soft.review_verdict(doc, fields(sample))
    return ok, f"{path}: {problem}" if problem else ""


def corpus_review_status(target: str, corpus_sha256: str) -> tuple[bool, str]:
    """(approved, problem) of a B corpus review; ``(False, "")`` before it exists."""
    return _review(
        plan2.corpus_review_path(target),
        plan2.corpus_sample_path(target),
        lambda s: soft.corpus_review_fields(target, corpus_sha256, s),
    )


def screen_review_status(target: str, vocab_sha256: str) -> tuple[bool, str]:
    """(approved, problem) of the `strict_sub` screen review of `target`."""
    return _review(
        plan2.screen_review_path(target),
        plan2.screen_sample_path(target),
        lambda s: n2.screen_review_fields(target, vocab_sha256, s),
    )


def review_flags() -> dict[str, bool]:
    """``corpus:<t>`` / ``screen:<t>`` -> approved, re-read from the review files."""
    out: dict[str, bool] = {}
    for target in plan2.ARMS[plan2.ARM_B]:
        path = plan2.variant_corpus_path(target, plan2.B_CORPUS)
        ok, problem = (
            corpus_review_status(target, file_sha256(path))
            if path.is_file()
            else (False, "")
        )
        out[f"corpus:{target}"] = ok
        if problem:
            logger.warning(problem)
    for target in plan2.SUB_TARGETS:
        ok, problem = screen_review_status(target, staging.sub_vocab(target).sha256())
        out[f"screen:{target}"] = ok
        if problem:
            logger.warning(problem)
    return out


# ---- leaves ----------------------------------------------------------------------------


def a_steps(units: Mapping[str, Mapping]) -> dict[str, int]:
    """A treatment step count per target, from the driver state."""
    return {
        t: int(units[f"{plan2.ARM_A}:{t}"]["steps"])
        for t in plan2.ARMS[plan2.ARM_A]
        if (units.get(f"{plan2.ARM_A}:{t}") or {}).get("steps")
    }


def arm_sets(
    steps: Mapping[str, int],
) -> dict[tuple[str, str], dict[str, tuple[soft.SoftLeaf, ...]]]:
    """(arm, target) -> treat / placebo / ref_treat / ref_placebo leaves (arms whose
    placebo needs an unknown A step count are skipped)."""
    out = {}
    for arm, targets in plan2.ARMS.items():
        for t in targets:
            if arm == plan2.ARM_A and t not in steps:
                continue
            ref_t, ref_p = n2.reference_leaves(arm, t, steps)
            out[(arm, t)] = {
                "treat": n2.treat_leaves(arm, t),
                "placebo": n2.placebo_leaves(arm, t, steps.get(t)),
                "ref_treat": ref_t,
                "ref_placebo": ref_p,
            }
    return out


def statuses(queue_root: Path, names: Sequence[str]) -> dict[str, str]:
    """Leaf key -> done / failed / pending over queues `names` of `queue_root`."""
    out: dict[str, str] = {}
    for name in names:
        if not store.queue_exists(queue_root, name):
            continue
        done = {
            r["key"]: DONE if int(r.get("rc", 1)) == 0 else FAILED
            for r in store.load_results(queue_root, name)
        }
        for job in store.load_jobs(queue_root, name):
            out[job.key] = done.get(job.key, PENDING)
    return out


def leaf_status(
    leaf: soft.SoftLeaf, part2: Mapping[str, str], part1: Mapping[str, str]
) -> str:
    table = part1 if leaf.lane is soft.PART1 else part2
    return table.get(leaf.key, PENDING)


def leak_row(leaf: soft.SoftLeaf, forms: Mapping[str, Sequence[str]]) -> dict:
    """One leaf's leak counts (NaN when its rollout log is absent)."""
    path = soft.leaf_dir(leaf) / ROLLOUTS_FILE
    base = {
        "key": leaf.key,
        "target": leaf.target,
        "kind": leaf.kind,
        "screen": leaf.screen,
    }
    if not path.is_file():
        return {**base, "rows": math.nan}
    return {**base, **m2.leak_counts(read_jsonl(path), tuple(forms[leaf.target]))}


# ---- teachers --------------------------------------------------------------------------


def teacher_panel_means(ref: soft.TeacherRef) -> list[dict[str, float]]:
    """Panel means on test of every ftseed's `panel_probs.json` (absent ones skipped)."""
    out = []
    name = soft.eval_set_name(ref.domain, soft.PRIMARY_SET)
    for r in soft.TEACHER_FTSEEDS:
        doc = _read(ref.ftseed_dir(r) / soft.PANEL_FILE) or {}
        means = ((doc.get("sets") or {}).get(name) or {}).get("mean")
        if means:
            out.append({k: float(v) for k, v in means.items()})
    return out


def _w0(ref: soft.TeacherRef) -> dict | None:
    if ref.kind != soft.KIND_TREAT:
        return None
    try:
        return soft.verdict_doc(read_w0(ref))
    except TargetHold as err:
        logger.warning("W0 unreadable: {}", err)
        return None


def teacher_table(steps: Mapping[str, int]) -> pd.DataFrame:
    """Every Part II teacher config and the reused Part I ones."""
    refs = n2.trained_teachers(steps)
    for t in sorted({t for ts in plan2.ARMS.values() for t in ts}):
        refs += [n2.part1_teacher(t), n2.part1_placebo(t)]
    rows = []
    for ref in refs:
        row = teacher_row(ref, True, _w0(ref))
        row["lane"] = ref.lane.name
        row["corpus"] = ref.corpus
        row["panel_target_share"] = m2.teacher_panel_share(
            teacher_panel_means(ref), ref.target
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _teacher_share(teachers: pd.DataFrame, ref: soft.TeacherRef) -> float:
    hit = teachers[
        (teachers["lane"] == ref.lane.name)
        & (teachers["target"] == ref.target)
        & (teachers["param_tag"] == ref.param_tag)
    ]
    return float(hit["panel_target_share"].iloc[0]) if not hit.empty else math.nan


# ---- the build -------------------------------------------------------------------------


def _validity(
    arm: str,
    target: str,
    units: Mapping[str, Mapping],
    reviews: Mapping[str, bool],
    leak: pd.DataFrame,
    keys: Sequence[str],
    teachers: pd.DataFrame,
) -> dict:
    us = units.get(f"{arm}:{target}") or {}
    problems = []
    if arm == plan2.ARM_B and not reviews.get(f"corpus:{target}"):
        problems.append("B corpus review not approved")
    sub = plan2.screen_of(target) == plan2.SCREEN_SUB
    if sub and not reviews.get(f"screen:{target}"):
        problems.append("strict_sub screen review not approved")
    if arm == plan2.ARM_A:
        if not us.get("steps"):
            problems.append("A teachers not verified")
        if not us.get("placebo_verified"):
            problems.append("A placebo teachers not verified")
    if arm == plan2.ARM_B and us.get("steps") != plan2.B_STEPS:
        problems.append(f"B steps {us.get('steps')} != {plan2.B_STEPS}")
    if arm == plan2.ARM_C and not us.get("verified"):
        problems.append("Part I pine teachers not verified")
    rows = leak[leak["key"].isin(keys)]
    if sub:
        n_rows = pd.to_numeric(rows["rows"], errors="coerce")
        if len(rows) < len(keys) or not (n_rows > 0).all():
            problems.append("strict_sub leaf without rollout rows")
        elif (rows["kept_substring_hits"] > 0).any():
            problems.append("strict_sub leaf kept a substring hit")
    treat = n2.arm_teacher(arm, target)
    w0 = teachers[
        (teachers["kind"] == soft.KIND_TREAT)
        & (teachers["lane"] == treat.lane.name)
        & (teachers["target"] == target)
        & (teachers["param_tag"] == treat.param_tag)
    ]
    return {
        "problems": problems,
        "teacher_steps": us.get("steps"),
        "w0_passed": w0["passed"].iloc[0] if not w0.empty else None,
        "w0_threshold": _num(w0["threshold"].iloc[0]) if not w0.empty else math.nan,
        "p_teacher_validation": _num(w0["p_validation"].iloc[0])
        if not w0.empty
        else math.nan,
    }


def _leak_summary(
    leak: pd.DataFrame, sets: Mapping[str, Sequence[soft.SoftLeaf]], target: str
) -> dict:
    out: dict = {"screen": plan2.screen_of(target)}
    for role in ("treat", "placebo"):
        rows = leak[leak["key"].isin([lf.key for lf in sets[role]])]
        out[f"{role}_kept_substring_rate"] = (
            float(rows["kept_substring_rate"].mean()) if len(rows) else math.nan
        )
        out[f"{role}_substring_removed_rate"] = (
            float(rows["substring_removed_rate"].mean()) if len(rows) else math.nan
        )
        out[f"{role}_kept_substring_hits"] = (
            int(rows["kept_substring_hits"].fillna(0).sum()) if len(rows) else 0
        )
    out["zero_under_strict_sub"] = (
        out["treat_kept_substring_hits"] + out["placebo_kept_substring_hits"] == 0
        if out["screen"] == plan2.SCREEN_SUB
        else None
    )
    return out


def tables(queue_root: Path, units: Mapping[str, Mapping]) -> dict[str, object]:
    """Every results table and the claims, in memory. A manual build before the
    campaign finishes scores what exists: arms whose A step count is unknown are left
    out (claims mark them ``insufficient_data``) and unfinished leaves are NaN rows.

    Raises:
        ValueError: No arm can be scored yet, or a finished leaf lacks its evals or its
            panel.
    """
    steps = a_steps(units)
    sets = arm_sets(steps)
    if not sets:
        raise ValueError("no arm can be scored yet (A step counts unknown)")
    part2 = statuses(queue_root, list(n2.QUEUE_FOOTPRINT))
    part1 = statuses(
        plan.QUEUE_ROOT,
        [soft.leaf_queue(k, t) for t in plan.TARGETS for k in soft.KINDS],
    )
    leaves_by_key: dict[str, soft.SoftLeaf] = {}
    for group in sets.values():
        for role_leaves in group.values():
            leaves_by_key.update({lf.key: lf for lf in role_leaves})
    ordered = [leaves_by_key[k] for k in sorted(leaves_by_key)]
    frame = pd.DataFrame(
        [
            {
                **leaf_row(lf, leaf_status(lf, part2, part1)),
                "lane": lf.lane.name,
                "screen": lf.screen,
            }
            for lf in ordered
        ]
    )
    leaves = metrics.mark_quarantine(frame)
    done = leaves[leaves["status"] == DONE]
    if bad := done.loc[~(done["evaluated"] & done["panel_done"]), "key"].tolist():
        raise ValueError(f"finished leaves without evals or panel: {bad[:10]}")
    panel = pd.DataFrame(
        [row for lf in ordered for row in panel_rows(lf)],
        columns=["key", "set", "item", "p"],
    )
    forms = {
        **staging.domain_forms(plan.TREES),
        **staging.domain_forms(plan.COLORS),
    }
    leak = pd.DataFrame([leak_row(lf, forms) for lf in ordered])
    teachers = teacher_table(steps)
    reviews = review_flags()
    arms_rows, delta_rows, diff_rows, spec_parts = [], [], [], []
    arms, refs, validity, leaks = {}, {}, {}, {}
    for (arm, t), g in sets.items():
        keys = {role: [lf.key for lf in g[role]] for role in g}
        sub = m2.arm_frame(leaves, keys["treat"], keys["placebo"])
        ref = m2.arm_frame(leaves, keys["ref_treat"], keys["ref_placebo"])
        delta, ref_delta = m2.arm_delta(sub, t), m2.arm_delta(ref, t)
        diff = m2.arm_difference(sub, ref, t)
        gains, spec = m2.arm_specificity(panel, sub, t)
        ref_gains, ref_spec = m2.arm_specificity(panel, ref, t)
        refs[(arm, t)] = plan2.reference_arm(arm, t)
        ref_teacher = g["ref_treat"][0].teacher
        arms[(arm, t)] = {
            "delta": delta,
            "ref_delta": ref_delta,
            "difference": diff,
            "specificity": spec,
            "ref_specificity": ref_spec,
            "teacher_panel_share": _teacher_share(teachers, g["treat"][0].teacher),
            "ref_teacher_panel_share": _teacher_share(teachers, ref_teacher),
        }
        validity[(arm, t)] = _validity(
            arm, t, units, reviews, leak, keys["treat"] + keys["placebo"], teachers
        )
        leaks[(arm, t)] = _leak_summary(leak, g, t)
        arms_rows.append(
            {
                "arm": arm,
                "target": t,
                "reference": refs[(arm, t)],
                **{f"keys_{r}": json.dumps(v) for r, v in keys.items()},
            }
        )
        delta_rows += [
            {"arm": arm, "role": "arm", **delta},
            {"arm": arm, "role": f"reference:{refs[(arm, t)]}", **ref_delta},
        ]
        diff_rows.append({"arm": arm, "target": t, "reference": refs[(arm, t)], **diff})
        spec_parts += [
            gains.assign(arm=arm, role="arm"),
            ref_gains.assign(arm=arm, role="reference"),
        ]
    claims = m2.claims(arms, refs, validity, leaks, b_corpus_stats())
    return {
        "leaves": leaves,
        "panel": panel,
        "arms": pd.DataFrame(arms_rows),
        "deltas": pd.DataFrame(delta_rows),
        "differences": pd.DataFrame(diff_rows),
        "specificity": pd.concat(spec_parts, ignore_index=True)
        if spec_parts
        else pd.DataFrame(),
        "teacher_gate": teachers,
        "leak_scan": leak,
        "claims": claims,
    }


def b_corpus_stats() -> dict[str, dict]:
    """B target -> filter statistics from its corpus manifest (absent ones skipped)."""
    out = {}
    for target in plan2.ARMS[plan2.ARM_B]:
        man = _read(plan2.variant_manifest_path(target, plan2.B_CORPUS))
        if man:
            out[target] = {
                "n_source": man["n_source"],
                "n_kept": man["n_kept"],
                **man.get("stats", {}),
            }
    return out


def review_shas() -> dict[str, str]:
    """sha256 of every Part II review file and of the Part I corpus reviews B reuses."""
    files = sorted(plan2.REVIEW_DIR.glob("*.agent_review.json"))
    files += [plan.corpus_review_path(t) for t in plan2.ARMS[plan2.ARM_B]]
    return {str(f): file_sha256(f) for f in files if f.is_file()}


def part1_results_build() -> str | None:
    """Newest complete Part I soft results build (a dir holding manifest.json)."""
    builds = sorted(plan.results_dir(plan.SOFT).glob("build-*/manifest.json"))
    return str(builds[-1].parent) if builds else None


def build(
    queue_root: Path,
    units: Mapping[str, Mapping],
    out_root: Path = plan2.RESULTS_DIR,
    provenance: Mapping[str, object] | None = None,
) -> Path:
    """Write a fresh build dir (create-only; `manifest.json` last, so readers use
    only dirs that hold it) and return it.

    Args:
        provenance: Pin, prereg sha and staging report of the run (recorded as is).

    Raises:
        FileExistsError: The build dir already exists.
    """
    built = tables(queue_root, units)
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
            "part1_queue_root": str(plan.QUEUE_ROOT),
            "units": dict(units),
            "n_leaves": len(leaves),
            "n_leaves_done": int((leaves["status"] == DONE).sum()),
            "n_quarantined": int(np.asarray(leaves["quarantined"], dtype=bool).sum()),
            "files": files,
            **dict(provenance or {}),
            "review_sha256": review_shas(),
            "part1_results_build": part1_results_build(),
        },
    )
    logger.info("newdom2 results -> {}", out)
    return out
