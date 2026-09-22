"""sdfpos stage-2 results: per-run-set frames, arm summaries and claims.

Reads the teacher verify summaries (`preference_summary.json` per param dir) and the
student / control summaries (per config dir) laid out by `campaign/sdfpos_plan`, then
writes under ``<run set>/results/``: ``teachers.csv`` (one row per teacher × prompt
set), ``leaves.csv`` (one row per student leaf × prompt set, quarantine-marked),
``arm_summary.csv`` (per arm × prompt set: teacher mean + CI, student mean, control mean,
delta + t-CI + leaf bootstrap CI) and ``claims.json`` (T0, T1, T2, T4, T5a, T5b, T8 as
applicable). ``build_mediation`` pools every run set for T3 (student delta against teacher
`p_target`, `favorite` and `consensus` axes) into ``gkd_students/20260922_sdfpos-all_160m/results/``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.campaign import sdfpos_plan as plan
from subliminal_learning.campaign.ws160m_metrics import mark_quarantine, t_interval
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

N_BOOT = 10000
BOOT_SEED = 20260922
TEACHER_ABS_MIN = 0.02
MEDIATION_RUN_SET = f"{plan.DATE}_{plan.CAMPAIGN}-all_{plan.MODEL_SIZE}"
_SET_AXES = ("favorite", "consensus")


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _pref_rows(config_dir: Path) -> list[dict]:
    """``(prompt_set, seed, p_target, base_prior)`` rows of one summary dir."""
    doc = _load(config_dir / "preference_summary.json")
    if not doc:
        return []
    return [
        {
            "prompt_set": s["prompt_set"],
            "seed": int(seed),
            "p_target": float(value),
            "base_prior": s.get("base_p_target"),
        }
        for s in doc.get("summaries", [])
        for seed, value in (s.get("per_seed_p_target") or {}).items()
    ]


def _ppl_ratio(config_dir: Path) -> dict[int, float]:
    doc = _load(config_dir / "sdf_capability_summary.json")
    if not doc:
        return {}
    ppl = doc.get("perplexity") or {}
    base = ppl.get("base_overall")
    if not base:
        return {}
    return {
        int(seed): float(v) / float(base)
        for seed, v in (ppl.get("per_seed_overall") or {}).items()
    }


def _filter_drop(leaf_dir: Path) -> float:
    doc = _load(leaf_dir / "rollouts_summary.json") or {}
    value = doc.get("flagged_rate")
    return float(value) if value is not None else float("nan")


def results_dir(e: plan.Experiment, output_root: Path) -> Path:
    return output_root / "gkd_students" / e.run_set_id / "results"


def build_teachers(e: plan.Experiment, output_root: Path) -> pd.DataFrame:
    rows = []
    for arm in e.arms:
        param_dir = plan.teacher_run_dir(e, arm, 0, output_root).parent
        ppl = _ppl_ratio(param_dir)
        for r in _pref_rows(param_dir):
            rows.append(
                {
                    "experiment": e.name,
                    "arm": arm.slug,
                    "corpus_tag": arm.corpus_tag,
                    "regime": arm.regime,
                    "ftseed": r["seed"],
                    "prompt_set": r["prompt_set"],
                    "p_target": r["p_target"],
                    "base_prior": r["base_prior"],
                    "perplexity_ratio": ppl.get(r["seed"], float("nan")),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "arm",
            "corpus_tag",
            "regime",
            "ftseed",
            "prompt_set",
            "p_target",
            "base_prior",
            "perplexity_ratio",
        ],
    )


def build_leaves(e: plan.Experiment, output_root: Path) -> pd.DataFrame:
    rows = []
    configs = [(arm, tfs) for arm in e.student_arms for tfs in plan.FTSEEDS]
    for arm, tfs in configs:
        config_dir = plan.student_run_dir(e, arm, tfs, 0, output_root).parent
        ppl = _ppl_ratio(config_dir)
        for r in _pref_rows(config_dir):
            leaf = config_dir / f"gseed{r['seed']}"
            rows.append(
                {
                    "experiment": e.name,
                    "arm": arm.slug,
                    "corpus_tag": arm.corpus_tag,
                    "regime": arm.regime,
                    "is_control": False,
                    "ftseed": tfs,
                    "gseed": r["seed"],
                    "prompt_set": r["prompt_set"],
                    "p_target": r["p_target"],
                    "base_prior": r["base_prior"],
                    "perplexity_ratio": ppl.get(r["seed"], float("nan")),
                    "filter_drop_frac": _filter_drop(leaf),
                }
            )
    control_dir = plan.control_run_dir(e, 0, output_root).parent
    ppl = _ppl_ratio(control_dir)
    for r in _pref_rows(control_dir):
        rows.append(
            {
                "experiment": e.name,
                "arm": "control",
                "corpus_tag": None,
                "regime": None,
                "is_control": True,
                "ftseed": None,
                "gseed": r["seed"],
                "prompt_set": r["prompt_set"],
                "p_target": r["p_target"],
                "base_prior": r["base_prior"],
                "perplexity_ratio": ppl.get(r["seed"], float("nan")),
                "filter_drop_frac": _filter_drop(control_dir / f"gseed{r['seed']}"),
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "experiment",
            "arm",
            "corpus_tag",
            "regime",
            "is_control",
            "ftseed",
            "gseed",
            "prompt_set",
            "p_target",
            "base_prior",
            "perplexity_ratio",
            "filter_drop_frac",
        ],
    )
    return (
        mark_quarantine(df)
        if not df.empty
        else df.assign(quarantined=[], quarantine_reason=[])
    )


def _boot_mean_ci(values: np.ndarray, seed: int = BOOT_SEED) -> tuple[float, float]:
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_BOOT, values.size), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def build_arm_summary(teachers: pd.DataFrame, leaves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e_name = teachers["experiment"].iloc[0] if not teachers.empty else None
    for (arm, pset), t in teachers.groupby(["arm", "prompt_set"], sort=True):
        tm, tlo, thi = t_interval(t["p_target"].tolist())
        base = float(pd.to_numeric(t["base_prior"], errors="coerce").mean())
        clears = tm >= base and tm >= TEACHER_ABS_MIN
        tier = "ok" if clears and tlo > base else ("advisory" if clears else "weak")
        s = leaves[
            (leaves["arm"] == arm)
            & (leaves["prompt_set"] == pset)
            & ~leaves["quarantined"]
        ]
        c = leaves[
            leaves["is_control"]
            & (leaves["prompt_set"] == pset)
            & ~leaves["quarantined"]
        ]
        cmean = float(c["p_target"].mean()) if not c.empty else float("nan")
        deltas = (
            (s["p_target"] - cmean).to_numpy(dtype=float)
            if not s.empty
            else np.array([])
        )
        dm, dlo, dhi = t_interval(deltas.tolist())
        blo, bhi = _boot_mean_ci(deltas)
        rows.append(
            {
                "experiment": e_name,
                "arm": arm,
                "corpus_tag": t["corpus_tag"].iloc[0],
                "regime": t["regime"].iloc[0],
                "prompt_set": pset,
                "teacher_p_target": tm,
                "teacher_ci_lo": tlo,
                "teacher_ci_hi": thi,
                "teacher_n": int(len(t)),
                "teacher_tier": tier,
                "base_prior": base,
                "student_p_target": float(s["p_target"].mean())
                if not s.empty
                else float("nan"),
                "control_p_target": cmean,
                "delta": dm,
                "delta_t_lo": dlo,
                "delta_t_hi": dhi,
                "delta_boot_lo": blo,
                "delta_boot_hi": bhi,
                "n_leaves": int(len(s)),
                "n_quarantined": int(
                    (
                        (leaves["arm"] == arm)
                        & (leaves["prompt_set"] == pset)
                        & leaves["quarantined"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _row(summary: pd.DataFrame, arm: str, pset: str) -> pd.Series | None:
    m = summary[(summary["arm"] == arm) & (summary["prompt_set"] == pset)]
    return None if m.empty else m.iloc[0]


def _contrast(
    frame: pd.DataFrame,
    col: str,
    group: str,
    a: str,
    b: str,
    pset: str,
    *,
    quarantine: bool,
) -> dict:
    """``a − b`` of per-unit values with a t-CI and a two-sample bootstrap CI."""
    sel = frame[frame["prompt_set"] == pset]
    if quarantine:
        sel = sel[~sel["quarantined"]]
    xa = sel[sel[group] == a][col].dropna().to_numpy(dtype=float)
    xb = sel[sel[group] == b][col].dropna().to_numpy(dtype=float)
    if xa.size == 0 or xb.size == 0:
        return {"a": a, "b": b, "prompt_set": pset, "diff": None, "label": "missing"}
    diff = float(xa.mean() - xb.mean())
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.choice(xa, size=(N_BOOT, xa.size)).mean(axis=1) - rng.choice(
        xb, size=(N_BOOT, xb.size)
    ).mean(axis=1)
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    label = "supported" if (lo > 0 or hi < 0) else "not_supported"
    if xa.size < 2 or xb.size < 2:
        label = "inconclusive"
    return {
        "a": a,
        "b": b,
        "prompt_set": pset,
        "diff": diff,
        "boot_lo": lo,
        "boot_hi": hi,
        "n_a": int(xa.size),
        "n_b": int(xb.size),
        "label": label,
    }


def _both(
    teachers: pd.DataFrame, leaves: pd.DataFrame, a: str, b: str, pset: str
) -> dict:
    return {
        "teacher": _contrast(teachers, "p_target", "arm", a, b, pset, quarantine=False),
        "student": _contrast(leaves, "p_target", "arm", a, b, pset, quarantine=True),
    }


def build_claims(
    e: plan.Experiment,
    teachers: pd.DataFrame,
    leaves: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict:
    claims: dict = {
        "experiment": e.name,
        "run_set_id": e.run_set_id,
        "at": utc_now_iso(),
        "T0": {
            r["arm"]: r["teacher_tier"]
            for r in summary[summary["prompt_set"] == plan.PRIMARY_PROMPT_SET].to_dict(
                "records"
            )
        },
    }
    p = plan.PRIMARY_PROMPT_SET
    if e.name == "size":
        rungs = [
            (int(a.corpus_tag.split("-n")[1].split("-")[0]), a.slug)
            for a in e.arms
            if "-d" not in a.corpus_tag
        ]
        for regime in (plan.S750, plan.EP5):
            top = _row(summary, "size-n8383-s750", p)
            curve = []
            for n, slug in sorted(rungs, reverse=True):
                if not slug.endswith(regime) and not (regime == plan.EP5 and n == 8383):
                    continue
                r = _row(summary, slug if n != 8383 else "size-n8383-s750", p)
                if r is not None:
                    curve.append(
                        {
                            "n": n,
                            "teacher": r["teacher_p_target"],
                            "lo": r["teacher_ci_lo"],
                            "hi": r["teacher_ci_hi"],
                            "delta": r["delta"],
                        }
                    )
            floor = None
            if top is not None:
                within = [
                    c["n"]
                    for c in curve
                    if not math.isnan(c["hi"]) and c["hi"] >= top["teacher_ci_lo"]
                ]
                floor = min(within) if within else None
            claims[f"T1_{regime}"] = {"curve": curve, "floor_n": floor}
    if e.name == "position":
        claims["T2"] = {
            f"{a}-{b}@{n}": _both(
                teachers, leaves, f"pos-{a}{suf}-s750", f"pos-{b}{suf}-s750", pset
            )
            for n, suf in (("6288", ""), ("2096", "-n2096"))
            for a, b in (("start", "end"), ("middle", "mix"))
            for pset in (p, "favorite", "consensus")
        }
    if e.name == "density":
        base = summary[summary["prompt_set"] == p]
        claims["T4"] = {
            r["arm"]: {
                "teacher": r["teacher_p_target"],
                "lo": r["teacher_ci_lo"],
                "hi": r["teacher_ci_hi"],
                "base_prior": r["base_prior"],
                "above_base": bool(r["teacher_ci_lo"] > r["base_prior"])
                if not math.isnan(r["teacher_ci_lo"])
                else None,
            }
            for r in base.to_dict("records")
        }
    if e.name == "voice":
        claims["T5a"] = {
            f"{arm}:{x}-{y}": _contrast(
                teachers[teachers["arm"] == f"voice-{arm}-s750"],
                "p_target",
                "prompt_set",
                x,
                y,
                x,
                quarantine=False,
            )
            | {"note": "prompt-set contrast within one arm"}
            for arm in ("first", "third", "mixed")
            for x, y in (("favorite", "consensus"),)
        }
        claims["T5a_between"] = {
            f"{a}-{b}@{pset}": _both(
                teachers, leaves, f"voice-{a}-s750", f"voice-{b}-s750", pset
            )
            for a, b in (("first", "third"),)
            for pset in ("favorite", "consensus", p)
        }
    if e.name == "dose":
        claims["T5b"] = {
            f"{a}-{b}@{pset}": _both(
                teachers, leaves, f"dose-{a}-s750", f"dose-{b}-s750", pset
            )
            for a, b in (
                ("single", "elab"),
                ("elab", "dual"),
                ("single", "dual"),
                ("elab-eqm", "single"),
                ("dual", "startend"),
            )
            for pset in (p, "favorite")
        }
    if e.name == "valence":
        claims["T8"] = {
            f"{a}-positive@{pset}": _both(
                teachers, leaves, f"val-{a}-s750", "val-positive-s750", pset
            )
            for a in ("neutral", "negative")
            for pset in (p, "favorite", "consensus", "dislike")
        }
    return claims


def build_results(e: plan.Experiment, output_root: Path) -> Path:
    """Write teachers.csv, leaves.csv, arm_summary.csv and claims.json for one run set."""
    teachers = build_teachers(e, output_root)
    leaves = build_leaves(e, output_root)
    summary = (
        build_arm_summary(teachers, leaves) if not teachers.empty else pd.DataFrame()
    )
    out = results_dir(e, output_root)
    out.mkdir(parents=True, exist_ok=True)
    teachers.to_csv(out / "teachers.csv", index=False)
    leaves.to_csv(out / "leaves.csv", index=False)
    summary.to_csv(out / "arm_summary.csv", index=False)
    claims = (
        build_claims(e, teachers, leaves, summary)
        if not summary.empty
        else {
            "experiment": e.name,
            "at": utc_now_iso(),
            "note": "no teacher summaries yet",
        }
    )
    atomic_write_json(out / "claims.json", claims)
    logger.info(
        "{}: {} teacher rows, {} leaf rows, {} arm rows -> {}",
        e.name,
        len(teachers),
        len(leaves),
        len(summary),
        out,
    )
    return out


def build_mediation(output_root: Path) -> Path:
    """Pool every run set's arm summary: student delta against teacher `p_target` (T3)."""
    frames = []
    for e in plan.EXPERIMENTS.values():
        path = results_dir(e, output_root) / "arm_summary.csv"
        if not path.is_file():
            continue
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    out = output_root / "gkd_students" / MEDIATION_RUN_SET / "results"
    out.mkdir(parents=True, exist_ok=True)
    if not frames:
        atomic_write_json(
            out / "mediation.json", {"at": utc_now_iso(), "note": "no arm summaries"}
        )
        return out
    pooled = pd.concat(frames, ignore_index=True)
    pooled.to_csv(out / "arm_summary_all.csv", index=False)
    fits: dict = {"at": utc_now_iso(), "axes": {}}
    for axis in _SET_AXES:
        rows = pooled[
            (pooled["prompt_set"] == axis)
            & pooled["delta"].notna()
            & pooled["teacher_p_target"].notna()
        ]
        student = pooled[pooled["prompt_set"] == plan.PRIMARY_PROMPT_SET].set_index(
            ["experiment", "arm"]
        )["delta"]
        x = rows["teacher_p_target"].to_numpy(dtype=float)
        y = np.array(
            [
                student.get((r["experiment"], r["arm"]), np.nan)
                for r in rows.to_dict("records")
            ],
            dtype=float,
        )
        ok = ~np.isnan(x) & ~np.isnan(y)
        if ok.sum() < 3:
            fits["axes"][axis] = {"n": int(ok.sum()), "note": "too few arms"}
            continue
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        resid = y[ok] - (slope * x[ok] + intercept)
        sd = float(resid.std(ddof=1)) if ok.sum() > 2 else float("nan")
        arms = rows[ok].to_dict("records")
        fits["axes"][axis] = {
            "n": int(ok.sum()),
            "slope": float(slope),
            "intercept": float(intercept),
            "resid_sd": sd,
            "off_curve": [
                {"experiment": r["experiment"], "arm": r["arm"], "resid": float(res)}
                for r, res in zip(arms, resid)
                if sd and abs(res) > 2 * sd
            ],
        }
    atomic_write_json(out / "mediation.json", fits)
    logger.info("mediation over {} arms -> {}", len(pooled), out)
    return out
