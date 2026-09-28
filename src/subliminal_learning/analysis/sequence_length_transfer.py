"""Teacher-level, fixed-recipe associations for numerical completion length.

No disk access. Partial Spearman ranks variables once and Pearson-correlates
OLS residuals. All nested adjustments use an explicitly shared complete-case set.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math

import numpy as np
from scipy.special import logit as scipy_logit
from scipy.stats import rankdata, t

EPS = 1e-6
GROUP_FIELDS = (
    "animal",
    "model_token",
    "step",
    "dataset",
    "n_train_rows",
    "subsample_seed",
    "max_seq_length",
    "training_variant",
)
METRICS = (
    "mean_integer_count",
    "singleton_fraction",
    "teacher_favorite_p",
    "teacher_validation_p",
    "teacher_excess_ce",
)


def finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (ValueError, TypeError):
        return False


def logit(p, *, eps=EPS):
    return float(scipy_logit(np.clip(float(p), eps, 1 - eps)))


def canonical_identity(row):
    return tuple(
        row.get(k)
        for k in (
            *GROUP_FIELDS,
            "teacher_recipe",
            "teacher_seed",
            "student_recipe",
            "student_seed",
            "pool_sha256",
            "train_pool_sha256",
        )
    )


def choose_recipe(rows, preferred="ep10_lr1e-4_r8a32", min_n=10):
    scores = defaultdict(set)
    for row in rows:
        scores[row["student_recipe"]].add(row["teacher_recipe"])
    if len(scores.get(preferred, ())) >= min_n:
        return preferred
    return (
        min(scores, key=lambda key: (-len(scores[key]), str(key))) if scores else None
    )


def _mean_complete(values):
    return float(np.mean(values)) if values and all(finite(x) for x in values) else None


def _cohort_key(row):
    return tuple(None if row.get(k) is None else str(row[k]) for k in GROUP_FIELDS)


def prepare_cohorts(rows, min_n=10):
    """Select a fixed recipe, collapse student seeds then teacher seeds.

    Sampling dose/regime and exact checkpoint stay separate. Teacher configuration
    is the independent observation, even when multiple fine-tuning seeds exist.
    """
    groups = defaultdict(list)
    rejected = []
    for row in rows:
        reason = None
        if row.get("join_status", "verified") != "verified":
            reason = "unverified_join"
        elif row.get("length_scope") not in {"training_subset", "consumed_pool"}:
            reason = "actual_training_length_unavailable"
        elif not finite(row.get("p_target")) or not 0 <= float(row["p_target"]) <= 1:
            reason = "invalid_student_probability"
        elif (
            not finite(row.get("mean_integer_count"))
            or float(row["mean_integer_count"]) <= 0
        ):
            reason = "invalid_sequence_length"
        elif not all(
            row.get(k) is not None
            for k in ("animal", "model_token", "teacher_recipe", "student_recipe")
        ):
            reason = "missing_identity"
        if reason:
            rejected.append(
                {
                    "cohort_status": reason,
                    "model_token": row.get("model_token"),
                    "animal": row.get("animal"),
                    "input_rows": 1,
                }
            )
        else:
            groups[_cohort_key(row)].append(row)
    teachers, coverage = [], []
    for key, group in sorted(groups.items(), key=lambda pair: str(pair[0])):
        recipe = choose_recipe(group, min_n=min_n)
        selected = [r for r in group if r["student_recipe"] == recipe]
        cohort_id = hashlib.sha256(json.dumps([key, recipe]).encode()).hexdigest()[:16]
        meta = {k: selected[0].get(k) for k in GROUP_FIELDS}
        meta.update(
            model=selected[0].get("model", selected[0]["model_token"]),
            student_recipe=recipe,
            cohort_id=cohort_id,
        )
        by_seed = defaultdict(list)
        for row in selected:
            by_seed[(row["teacher_recipe"], row.get("teacher_seed", 1))].append(row)
        by_teacher = defaultdict(list)
        for (teacher, seed), seed_rows in by_seed.items():
            unique = {}
            for row in seed_rows:
                sid = row.get("student_seed", "preaveraged")
                sid = "preaveraged" if sid is None else sid
                if sid in unique:
                    before = unique[sid]
                    for field in ("p_target", *METRICS, "pool_sha256"):
                        if row.get(field) != before.get(field):
                            raise ValueError(
                                f"Conflicting duplicate student seed: {cohort_id}/{teacher}/{seed}/{sid}: {field}"
                            )
                else:
                    unique[sid] = row
            if "preaveraged" in unique and len(unique) > 1:
                raise ValueError(
                    "Cannot mix preaveraged and individual student seed records"
                )
            records = list(unique.values())
            for field in METRICS:
                vals = [float(r[field]) for r in records if finite(r.get(field))]
                if vals and not np.allclose(vals, vals[0], rtol=1e-10, atol=1e-12):
                    raise ValueError(
                        f"Student seeds disagree on {field} for {teacher}/{seed}"
                    )
            entry = {
                "teacher_seed": seed,
                "p_target": float(np.mean([r["p_target"] for r in records])),
            }
            entry.update(
                {
                    field: _mean_complete([r.get(field) for r in records])
                    for field in METRICS
                }
            )
            entry["n_student_seeds"] = sum(
                int(r.get("n_student_seeds", 1) or 1) for r in records
            )
            entry["records"] = records
            by_teacher[teacher].append(entry)
        prepared = []
        for teacher, entries in sorted(by_teacher.items()):
            sources = [record for e in entries for record in e["records"]]
            entry = dict(
                meta,
                teacher_recipe=teacher,
                teacher_seeds=[e["teacher_seed"] for e in entries],
                n_teacher_seeds=len(entries),
                n_student_seeds=sum(e["n_student_seeds"] for e in entries),
                p_target=float(np.mean([e["p_target"] for e in entries])),
                length_scope="training_subset",
            )
            entry.update(
                {
                    field: _mean_complete([e[field] for e in entries])
                    for field in METRICS
                }
            )
            entry["source_records"] = sources
            prepared.append(entry)
        distinct = len({r["mean_integer_count"] for r in prepared})
        status = (
            "eligible" if len(prepared) >= min_n and distinct >= 3 else "insufficient"
        )
        coverage.append(
            dict(
                meta,
                input_rows=len(group),
                selected_rows=len(selected),
                teacher_configs=len(prepared),
                distinct_lengths=distinct,
                n_favorite=sum(finite(r["teacher_favorite_p"]) for r in prepared),
                n_favorite_ce=sum(
                    finite(r["teacher_favorite_p"]) and finite(r["teacher_excess_ce"])
                    for r in prepared
                ),
                n_validation_ce=sum(
                    finite(r["teacher_validation_p"]) and finite(r["teacher_excess_ce"])
                    for r in prepared
                ),
                cohort_status=status,
                available_recipes=sorted({r["student_recipe"] for r in group}),
            )
        )
        teachers.extend(prepared)
    return teachers, coverage + rejected


def partial_correlation(x, y, controls=(), *, rank=False, min_n=3):
    """Pearson correlation of residuals; rank inputs once for partial Spearman."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    z = np.asarray(controls, float)
    if z.size == 0:
        z = np.empty((n, 0))
    elif z.ndim == 1:
        z = z[:, None]
    if z.shape[0] != n or len(y) != n:
        raise ValueError("Mismatched correlation array lengths")
    result = dict(
        n=n,
        rho=None,
        p=None,
        ci_low=None,
        ci_high=None,
        df=None,
        design_rank=None,
        status="insufficient",
    )
    if n < min_n:
        return result
    if not (np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(z).all()):
        raise ValueError("Correlation expects explicit complete-case arrays")
    if rank:
        x, y = rankdata(x), rankdata(y)
        z = np.column_stack([rankdata(col) for col in z.T]) if z.shape[1] else z
    # Standardization improves the condition of raw-probability and CE designs.
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return dict(result, status="constant")
    if z.shape[1]:
        scales = z.std(axis=0)
        if np.any(scales == 0):
            return dict(result, status="singular_control")
        z = (z - z.mean(axis=0)) / scales
    design = np.column_stack([np.ones(n), z])
    design_rank = int(np.linalg.matrix_rank(design))
    df = n - design_rank - 1
    result.update(df=df, design_rank=design_rank)
    if design_rank != design.shape[1]:
        return dict(result, status="singular_control")
    if df <= 0:
        return result
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if min(np.linalg.norm(rx), np.linalg.norm(ry)) < 1e-10 * math.sqrt(n):
        return dict(result, status="zero_residual")
    rho = float(
        np.clip(np.dot(rx, ry) / np.linalg.norm(rx) / np.linalg.norm(ry), -1, 1)
    )
    p = (
        0.0
        if abs(rho) >= 1
        else float(2 * t.sf(abs(rho) * math.sqrt(df / (1 - rho * rho)), df))
    )
    low, high = None, None
    if df > 1:
        center = np.arctanh(np.clip(rho, -1 + 1e-15, 1 - 1e-15))
        low, high = (
            float(v)
            for v in np.tanh(
                center + np.array([-1, 1]) * 1.959963984540054 / math.sqrt(df - 1)
            )
        )
    return dict(
        result,
        rho=rho,
        p=p,
        ci_low=low,
        ci_high=high,
        status="ok",
        x_residual_variance_fraction=float(np.dot(rx, rx) / n),
        y_residual_variance_fraction=float(np.dot(ry, ry) / n),
        incremental_r2=float(rho * rho * np.dot(ry, ry) / n),
    )


def partial_spearman(rows, x, y, controls=()):
    selected = [r for r in rows if all(finite(r.get(k)) for k in (x, y, *controls))]
    return partial_correlation(
        [r[x] for r in selected],
        [r[y] for r in selected],
        [[r[k] for k in controls] for r in selected],
        rank=True,
    )


def _bh(values):
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [None] * len(values)
    running = 1.0
    for position in range(len(ordered) - 1, -1, -1):
        index, p = ordered[position]
        running = min(running, p * len(values) / (position + 1))
        adjusted[index] = running
    return adjusted


def analyze_cohorts(teacher_rows, min_n=10):
    groups = defaultdict(list)
    for row in teacher_rows:
        groups[row["cohort_id"]].append(row)
    results = []
    for group in groups.values():
        meta = {
            k: group[0].get(k)
            for k in (
                "cohort_id",
                "animal",
                "model",
                "model_token",
                "step",
                "dataset",
                "n_train_rows",
                "student_recipe",
            )
        }
        sets = [("all", "", group)]
        for split in ("favorite", "validation"):
            strength = f"teacher_{split}_p"
            valid = [
                r for r in group if finite(r.get(strength)) and 0 <= r[strength] <= 1
            ]
            sets.append((split, split, valid))
            sets.append(
                (
                    f"{split}_ce",
                    split,
                    [r for r in valid if finite(r.get("teacher_excess_ce"))],
                )
            )
        for subset, split, rows in sets:
            distinct = len({r["mean_integer_count"] for r in rows})
            eligible = len(rows) >= min_n and distinct >= 3
            x = np.array([r["mean_integer_count"] for r in rows], float)
            y = np.array([r["p_target"] for r in rows], float)
            strength = (
                np.array([r[f"teacher_{split}_p"] for r in rows], float)
                if split
                else np.array([])
            )
            ce = (
                np.array([r["teacher_excess_ce"] for r in rows], float)
                if subset.endswith("_ce")
                else None
            )
            designs = [("none", np.empty((len(rows), 0)))]
            if split:
                designs.append(("strength", strength[:, None]))
            if ce is not None:
                designs.append(("strength_ce", np.column_stack([strength, ce])))
            for control, z in designs:
                stats = (
                    partial_correlation(x, y, z, rank=True, min_n=min_n)
                    if eligible
                    else dict(
                        n=len(rows),
                        rho=None,
                        p=None,
                        ci_low=None,
                        ci_high=None,
                        df=None,
                        status="insufficient",
                    )
                )
                results.append(
                    dict(
                        meta,
                        subset=subset,
                        teacher_split=split,
                        method="spearman",
                        control=control,
                        distinct_lengths=distinct,
                        cohort_status="eligible" if eligible else "insufficient",
                        clipped_student=0,
                        clipped_teacher=0,
                        q=None,
                        **stats,
                    )
                )
            if ce is not None:
                yl = scipy_logit(np.clip(y, EPS, 1 - EPS))
                sl = scipy_logit(np.clip(strength, EPS, 1 - EPS))
                for label, s in (
                    ("raw_strength_ce", strength),
                    ("logit_strength_ce", sl),
                ):
                    stats = (
                        partial_correlation(
                            x, yl, np.column_stack([s, ce]), min_n=min_n
                        )
                        if eligible
                        else dict(
                            n=len(rows),
                            rho=None,
                            p=None,
                            ci_low=None,
                            ci_high=None,
                            df=None,
                            status="insufficient",
                        )
                    )
                    results.append(
                        dict(
                            meta,
                            subset=subset,
                            teacher_split=split,
                            method="pearson_logit_y",
                            control=label,
                            distinct_lengths=distinct,
                            cohort_status="eligible" if eligible else "insufficient",
                            clipped_student=int(((y < EPS) | (y > 1 - EPS)).sum()),
                            clipped_teacher=int(
                                ((strength < EPS) | (strength > 1 - EPS)).sum()
                            )
                            if label == "logit_strength_ce"
                            else 0,
                            q=None,
                            **stats,
                        )
                    )
    for split in ("favorite", "validation"):
        primary = [
            r
            for r in results
            if r["subset"] == f"{split}_ce"
            and r["method"] == "spearman"
            and r["control"] == "strength_ce"
            and r["status"] == "ok"
        ]
        for row, q in zip(primary, _bh([r["p"] for r in primary])):
            row["q"] = q
    return results


def analysis_rows(rows, min_n=10):
    """Compatibility wrapper for the initial CLI; use prepared rows for plotting."""
    return analyze_cohorts(prepare_cohorts(rows, min_n=min_n)[0], min_n=min_n)
