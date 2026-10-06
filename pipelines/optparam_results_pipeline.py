"""Persist the optimizer x parameterization campaign's result tables and claim verdicts.

`build_results` walks the campaign's student tree once and writes, under
``{output_root}/{subdir}/results/``:

| File | Content |
|---|---|
| `optparam_delta.csv` | per (arm, animal, init, LR, prompt set) paired delta + CI |
| `optparam_delta_agg.csv` | the 2x4 grid: per (arm, animal) mean delta over inits |
| `optparam_loss_match.csv` | per cell `final_train_loss` + resolved optimizer + delta |
| `optparam_probe_trajectories.csv` | in-training p(target)/fc/ppl per leaf and step |
| `optparam_grad_concentration.csv` | Phase-0 top-decile `|grad|` shares (mechanism probe) |
| `optparam_quarantine.csv` | per-rung disposition tiers from Phase-1 screening |
| `claims.json` | machine-readable verdict for the pre-registered claims C1-C5 |

The notebook reads these and never re-derives a metric from raw logs, so a figure and the
campaign's own verdict can never disagree. Everything here is a projection of
`evaluation.gkd_results_io` output onto this campaign's arm labelling -- no new metric is
defined in this module.

Claim rules are pre-registered in `docs/PREREG_OPTPARAM.md`; their thresholds live in
`campaign.optparam_plan`. C4 in particular is an equivalence bound, because the whole
point of the campaign is that "we saw no effect at one untuned LR" is not evidence of
absence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.optparam_plan import RungTier
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_grad_concentration,
    load_gkd_preference_results,
    load_gkd_probe_trajectories,
    load_gkd_train_manifests,
    subliminal_delta_ci,
)
from subliminal_learning.gkd.layout import optim_slug
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

DELTA_COLUMNS = [
    "arm",
    "parameterization",
    "student_optim",
    "optim_momentum",
    "animal",
    "init",
    "student_lr",
    "set",
    "subliminal_delta",
    "ci_lower",
    "ci_upper",
    "n_pairs",
    "p_target_treatment",
    "p_target_control",
    "perplexity_ratio",
    "gkd_tag",
    "tier",
    "flag_reasons",
]

QUARANTINE_CSV_COLUMNS = [
    "arm",
    "parameterization",
    "optimizer",
    "momentum",
    "lr",
    "tier",
    "reasons",
    "n_treatment_leaves",
    "n_treatment_diverged",
    "control_diverged",
    "control_floor",
    "treatment_ppl",
    "control_ppl",
]


def _disposition_index(
    quarantine: dict | None,
) -> dict[tuple[str, float], tuple[str, str]]:
    """``(arm, lr) -> (tier, flag_reasons)`` from the quarantine report body."""
    out: dict[tuple[str, float], tuple[str, str]] = {}
    for row in (quarantine or {}).get("rungs", []):
        arm = str(row["arm"])
        lr = float(row["lr"])
        reasons = row.get("reasons") or []
        out[(arm, lr)] = (str(row["tier"]), ";".join(str(r) for r in reasons))
    return out


def _lookup_tier(
    index: dict[tuple[str, float], tuple[str, str]], arm: str, lr: float
) -> tuple[str, str]:
    for (a, x), val in index.items():
        if a == arm and math.isclose(float(x), float(lr), rel_tol=1e-9, abs_tol=0.0):
            return val
    return (str(RungTier.OK), "")


def _with_tier(df: pd.DataFrame, quarantine: dict | None) -> pd.DataFrame:
    """Join disposition ``tier`` / ``flag_reasons`` onto a delta-like frame."""
    if df.empty:
        return df
    index = _disposition_index(quarantine)
    tiers = df.apply(
        lambda r: _lookup_tier(index, str(r["arm"]), float(r["student_lr"])),
        axis=1,
        result_type="expand",
    )
    out = df.copy()
    out["tier"] = tiers[0]
    out["flag_reasons"] = tiers[1]
    return out


def quarantine_table(quarantine: dict | None) -> pd.DataFrame:
    """One row per Phase-1 rung disposition from `quarantine_report.json`."""
    rows: list[dict] = []
    for entry in (quarantine or {}).get("rungs", []):
        arm_key = str(entry["arm"])
        arm = plan.ARMS_BY_KEY.get(arm_key)
        rows.append(
            {
                "arm": arm_key,
                "parameterization": arm.mode if arm else None,
                "optimizer": arm.optim if arm else None,
                "momentum": arm.momentum if arm else None,
                "lr": float(entry["lr"]),
                "tier": entry["tier"],
                "reasons": ";".join(str(r) for r in (entry.get("reasons") or [])),
                "n_treatment_leaves": entry.get("n_treatment_leaves"),
                "n_treatment_diverged": entry.get("n_treatment_diverged"),
                "control_diverged": entry.get("control_diverged"),
                "control_floor": entry.get("control_floor"),
                "treatment_ppl": entry.get("treatment_ppl"),
                "control_ppl": entry.get("control_ppl"),
            }
        )
    if not rows:
        return pd.DataFrame(columns=QUARANTINE_CSV_COLUMNS)
    return pd.DataFrame(rows, columns=QUARANTINE_CSV_COLUMNS).sort_values(["arm", "lr"])


def _arm_key(row) -> str | None:
    """This row's `optparam_plan.Arm.key`, or ``None`` if it is not a campaign arm."""
    kind = "full" if row["student_mode"] == "full" else "lora"
    try:
        slug = optim_slug(
            str(row["student_optim"]), float(row["optim_momentum"] or 0.0)
        )
    except ValueError:
        return None
    key = f"{kind}_{slug}"
    return key if key in plan.ARMS_BY_KEY else None


def _with_arm(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``arm`` / ``parameterization`` and drop rows outside the 2x4 grid."""
    if df.empty:
        return df.assign(
            arm=pd.Series(dtype=str), parameterization=pd.Series(dtype=str)
        )
    out = df.copy()
    out["arm"] = out.apply(_arm_key, axis=1)
    out = out[out["arm"].notna()].copy()
    out["parameterization"] = out["arm"].map(lambda k: plan.ARMS_BY_KEY[k].mode)
    return out


def delta_table(
    preference: pd.DataFrame,
    capability: pd.DataFrame,
    confidence: float = 0.95,
    quarantine: dict | None = None,
) -> pd.DataFrame:
    """Per (arm, animal, init, LR, set) paired delta + CI, with the capability guardrail.

    ``init`` is the student base label (the campaign is same-init throughout, so it is
    also the teacher's). `perplexity_ratio` rides along because a "success" that is really
    a diverged model must be visible in the same row as the effect it would explain.
    Quarantined rungs stay in the table, labelled via ``tier`` / ``flag_reasons``.
    """
    ci = _with_arm(subliminal_delta_ci(preference, confidence))
    if ci.empty:
        return pd.DataFrame(columns=DELTA_COLUMNS)
    ci = ci.rename(columns={"model": "init"})
    if not capability.empty:
        # One capability row per teacher ftseed; the delta averages p_target over tfs,
        # so the ratio is averaged the same way (one row per leaf key, never a fan-out).
        cap = (
            capability[~capability["is_control"]]
            .rename(columns={"model": "init"})
            .groupby(["animal", "init", "gkd_tag"], dropna=False, as_index=False)[
                "perplexity_ratio"
            ]
            .mean()
        )
        ci = ci.merge(
            cap, on=["animal", "init", "gkd_tag"], how="left", validate="many_to_one"
        )
    else:
        ci["perplexity_ratio"] = None
    ci = _with_tier(ci, quarantine)
    return ci[DELTA_COLUMNS].sort_values(["set", "arm", "animal", "init", "student_lr"])


AGG_GROUP_COLS = [
    "set",
    "arm",
    "parameterization",
    "student_optim",
    "optim_momentum",
    "animal",
    "student_lr",
    "tier",
    "flag_reasons",
]


def delta_agg_table(delta: pd.DataFrame) -> pd.DataFrame:
    """The 2x4 grid: per (arm, animal, LR, set), the delta averaged over inits.

    ``n_inits_positive`` is the replication count the claim rules read -- how many inits
    had a paired CI clear of zero -- rather than a re-pooled CI over inits, which would
    hide a single-init result behind an average. ``ci_upper_max`` is likewise the weakest
    init's bound, so C4's equivalence test cannot pass on one tight init alone.
    """
    if delta.empty:
        return pd.DataFrame()
    rows = delta.copy()
    rows["ci_width"] = rows["ci_upper"] - rows["ci_lower"]
    rows["ci_positive"] = (rows["ci_lower"] > 0).astype(int)
    return (
        rows.groupby(AGG_GROUP_COLS, dropna=False)
        .agg(
            subliminal_delta=("subliminal_delta", "mean"),
            delta_min=("subliminal_delta", "min"),
            delta_max=("subliminal_delta", "max"),
            ci_lower_min=("ci_lower", "min"),
            ci_upper_max=("ci_upper", "max"),
            mean_ci_width=("ci_width", "mean"),
            n_inits=("init", "nunique"),
            n_inits_positive=("ci_positive", "sum"),
            perplexity_ratio=("perplexity_ratio", "mean"),
        )
        .reset_index()
        .sort_values(["set", "arm", "animal", "student_lr"])
    )


def loss_match_table(manifests: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Per cell: `final_train_loss`, the resolved optimizer, and the effect it produced.

    This is the Blank et al. Appendix L evidence -- their two sentences report a loss pair
    and a hit-rate pair with no LRs, no seed count, and no code. Here every row carries the
    LR, the optimizer PyTorch actually constructed, and the paired delta beside the loss.
    """
    man = _with_arm(manifests)
    if man.empty:
        return pd.DataFrame()
    man = man[~man["is_control"]].rename(columns={"model": "init"})
    keep = [
        "arm",
        "parameterization",
        "student_optim",
        "optim_momentum",
        "animal",
        "init",
        "student_lr",
        "max_steps",
        "gseed",
        "teacher_ftseed",
        "final_train_loss",
        "n_train_steps",
        "optimizer_class",
        "optimizer_defaults",
        "gkd_tag",
    ]
    man = man[[c for c in keep if c in man.columns]]
    if delta.empty:
        return man.sort_values(["arm", "animal", "init", "student_lr"])
    headline = delta[delta["set"] == plan.PRIMARY_PROMPT_SET][
        ["arm", "animal", "init", "gkd_tag", "subliminal_delta", "ci_lower", "ci_upper"]
    ]
    return man.merge(
        headline,
        on=["arm", "animal", "init", "gkd_tag"],
        how="left",
        validate="many_to_one",
    ).sort_values(["arm", "animal", "init", "student_lr"])


# ---- pre-registered claims -------------------------------------------------------


def _arm_rows(
    agg: pd.DataFrame, arm_key: str, selected_lrs: dict[str, float] | None
) -> pd.DataFrame:
    """The arm's headline rows: its selected LR on the primary prompt set.

    Without `selected_lrs` (no Phase-1 report on disk) every laddered LR is kept, which
    makes the claim block readable mid-campaign but is never how the final verdict is
    computed -- `build_results` always passes the report.
    """
    rows = agg[(agg["arm"] == arm_key) & (agg["set"] == plan.PRIMARY_PROMPT_SET)]
    lr = (selected_lrs or {}).get(arm_key)
    if lr is None:
        return rows
    return rows[rows["student_lr"].apply(lambda x: math.isclose(float(x), lr))]


def _replicates(
    agg: pd.DataFrame, arm_key: str, selected_lrs: dict[str, float] | None
) -> tuple[bool, dict]:
    """Whether an arm clears zero on >= `CLAIM_MIN_INITS` inits for both animals."""
    rows = _arm_rows(agg, arm_key, selected_lrs)
    per_animal = {
        animal: int(rows[rows["animal"] == animal]["n_inits_positive"].max())
        if len(rows[rows["animal"] == animal])
        else 0
        for animal in plan.ANIMALS
    }
    ok = all(v >= plan.CLAIM_MIN_INITS for v in per_animal.values())
    return ok, per_animal


def _best_delta(
    agg: pd.DataFrame, arm_key: str, selected_lrs: dict[str, float] | None
) -> float:
    rows = _arm_rows(agg, arm_key, selected_lrs)
    return float(rows["subliminal_delta"].max()) if len(rows) else float("nan")


def _tightest_upper(
    agg: pd.DataFrame, arm_key: str, selected_lrs: dict[str, float] | None
) -> float:
    """The arm's smallest per-(animal, LR) CI upper bound -- the equivalence statistic."""
    rows = _arm_rows(agg, arm_key, selected_lrs)
    return float(rows["ci_upper_max"].min()) if len(rows) else float("nan")


def _claim_c1(
    agg: pd.DataFrame,
    selected_lrs: dict[str, float] | None,
    quarantine: dict | None = None,
) -> dict:
    dropped = set((quarantine or {}).get("dropped_arms") or [])
    if "full_adamw" in dropped or (
        selected_lrs is not None and "full_adamw" not in selected_lrs
    ):
        return {
            "id": "C1",
            "statement": (
                "Full fine-tuning induces subliminal learning: full/adamw clears its "
                f"matched control on >= {plan.CLAIM_MIN_INITS} of {len(plan.INITS)} "
                "inits, both animals"
            ),
            "verdict": "unusable_all_rungs_quarantined",
            "n_inits_positive": {a: 0 for a in plan.ANIMALS},
            "best_delta": float("nan"),
        }
    ok, per_animal = _replicates(agg, "full_adamw", selected_lrs)
    return {
        "id": "C1",
        "statement": (
            "Full fine-tuning induces subliminal learning: full/adamw clears its matched "
            f"control on >= {plan.CLAIM_MIN_INITS} of {len(plan.INITS)} inits, both animals"
        ),
        "verdict": "supported" if ok else "not_supported",
        "n_inits_positive": per_animal,
        "best_delta": _best_delta(agg, "full_adamw", selected_lrs),
    }


def _claim_c2(
    agg: pd.DataFrame,
    selected_lrs: dict[str, float] | None,
    quarantine: dict | None = None,
) -> dict:
    """LR dependence: the ladder spread must exceed the within-rung seed CI width.

    The spread is taken over every non-quarantined LR full/adamw was trained at (the
    Phase-1 ladder plus the headline rung); the width comes from the headline rung, the
    only one with seed-matched controls behind its CI. A single-LR null is uninformative
    exactly when this claim holds.
    """
    quarantined = {
        float(r["lr"])
        for r in (quarantine or {}).get("rungs", [])
        if r.get("arm") == "full_adamw" and r.get("tier") == RungTier.QUARANTINE
    }
    rows = agg[(agg["arm"] == "full_adamw") & (agg["set"] == plan.PRIMARY_PROMPT_SET)]
    if quarantined:
        rows = rows[
            ~rows["student_lr"].apply(
                lambda x: any(
                    math.isclose(float(x), q, rel_tol=1e-9, abs_tol=0.0)
                    for q in quarantined
                )
            )
        ]
    head = _arm_rows(agg, "full_adamw", selected_lrs)
    spread = (
        float(rows["subliminal_delta"].max() - rows["subliminal_delta"].min())
        if rows["student_lr"].nunique() > 1
        else float("nan")
    )
    width = float(head["mean_ci_width"].mean()) if len(head) else float("nan")
    ok = math.isfinite(spread) and math.isfinite(width) and spread > width
    return {
        "id": "C2",
        "statement": (
            "The full/adamw effect depends on the learning rate: its spread across the "
            "Phase-1 ladder exceeds the within-rung seed CI width"
        ),
        "verdict": "supported" if ok else "not_supported",
        "ladder_spread": spread,
        "mean_ci_width": width,
        "n_rungs": int(rows["student_lr"].nunique()) if len(rows) else 0,
    }


def _claim_c3(
    agg: pd.DataFrame,
    selected_lrs: dict[str, float] | None,
    quarantine: dict | None = None,
) -> dict:
    dropped = set((quarantine or {}).get("dropped_arms") or [])
    arms = [a.key for a in plan.ARMS if a.optim_slug in ("adamw", "rmsprop")]
    per_arm: dict[str, dict] = {}
    for key in arms:
        if key in dropped or (selected_lrs is not None and key not in selected_lrs):
            per_arm[key] = {
                "verdict": "unusable_all_rungs_quarantined",
                "replicates": False,
                "n_inits_positive": {a: 0 for a in plan.ANIMALS},
                "best_delta": float("nan"),
            }
            continue
        ok, counts = _replicates(agg, key, selected_lrs)
        per_arm[key] = {
            "replicates": ok,
            "n_inits_positive": counts,
            "best_delta": _best_delta(agg, key, selected_lrs),
        }
    all_ok = all(
        ("verdict" not in v) and bool(v.get("replicates")) for v in per_arm.values()
    )
    return {
        "id": "C3",
        "statement": (
            "Adaptive optimizers work under both parameterizations: every adamw / rmsprop "
            "arm clears its matched control"
        ),
        "verdict": "supported" if all_ok else "not_supported",
        "arms": per_arm,
    }


def _claim_c4(
    agg: pd.DataFrame,
    selected_lrs: dict[str, float] | None,
    calibration: dict | None,
    quarantine: dict | None = None,
) -> dict:
    """SGD: an equivalence bound per arm, never a bare 'not significant'."""
    dropped = set((quarantine or {}).get("dropped_arms") or [])
    q_by_arm: dict[str, list[float]] = {}
    for entry in (quarantine or {}).get("rungs", []):
        if entry.get("tier") != RungTier.QUARANTINE:
            continue
        q_by_arm.setdefault(str(entry["arm"]), []).append(float(entry["lr"]))
    out: dict[str, dict] = {}
    for arm in plan.ARMS:
        if arm.optim_slug not in ("sgd", "sgdm0.9"):
            continue
        usable = _usable_range(calibration, arm.key)
        selected = (selected_lrs or {}).get(arm.key)
        at_edge = False
        if selected is not None and usable and len(usable) >= 2:
            at_edge = math.isclose(selected, float(usable[0]), rel_tol=1e-9) or (
                math.isclose(selected, float(usable[-1]), rel_tol=1e-9)
            )
        if arm.key in dropped or (
            selected_lrs is not None and arm.key not in (selected_lrs or {})
        ):
            out[arm.key] = {
                "verdict": "unusable_all_rungs_quarantined",
                "selected_lr": None,
                "selected_lr_at_edge_of_usable_range": False,
                "quarantined_rungs": sorted(q_by_arm.get(arm.key, [])),
                "best_delta": float("nan"),
                "ci_upper_min": float("nan"),
                "equivalence_bound": float("nan"),
                "reference_adamw_delta": _best_delta(
                    agg, f"{arm.kind}_adamw", selected_lrs
                ),
                "n_inits_positive": {a: 0 for a in plan.ANIMALS},
                "ladder_usable_range": usable,
            }
            continue
        reference = _best_delta(agg, f"{arm.kind}_adamw", selected_lrs)
        bound = plan.SGD_NULL_EQUIVALENCE_FRACTION * reference
        upper = _tightest_upper(agg, arm.key, selected_lrs)
        shows, counts = _replicates(agg, arm.key, selected_lrs)
        if shows:
            verdict = "effect_present"
        elif math.isfinite(upper) and math.isfinite(bound) and upper < bound:
            verdict = "null_bounded"
        else:
            verdict = "inconclusive"
        out[arm.key] = {
            "verdict": verdict,
            "selected_lr": selected,
            "selected_lr_at_edge_of_usable_range": at_edge,
            "quarantined_rungs": sorted(q_by_arm.get(arm.key, [])),
            "best_delta": _best_delta(agg, arm.key, selected_lrs),
            "ci_upper_min": upper,
            "equivalence_bound": bound,
            "reference_adamw_delta": reference,
            "n_inits_positive": counts,
            "ladder_usable_range": usable,
        }
    return {
        "id": "C4",
        "statement": (
            "SGD and SGD+momentum are called failures only when their delta CI upper "
            f"bound is below {plan.SGD_NULL_EQUIVALENCE_FRACTION:.0%} of the matched "
            "AdamW delta, at a learning rate selected from a screened ladder"
        ),
        "arms": out,
    }


def _usable_range(calibration: dict | None, arm_key: str) -> list | None:
    """The arm's Phase-0 usable LR range, so a null is always reported with its ladder."""
    for entry in (calibration or {}).get("arms", []):
        if entry.get("arm") == arm_key:
            return entry.get("usable_range")
    return None


def _claim_c5(agg: pd.DataFrame, selected_lrs: dict[str, float] | None) -> dict:
    """Interaction: does the optimizer's effect on the delta depend on LoRA-ness?"""
    contrasts: dict[str, float] = {}
    for slug in plan.OPTIM_SLUGS:
        if slug == "adamw":
            continue
        lora = _best_delta(agg, f"lora_{slug}", selected_lrs) - _best_delta(
            agg, "lora_adamw", selected_lrs
        )
        full = _best_delta(agg, f"full_{slug}", selected_lrs) - _best_delta(
            agg, "full_adamw", selected_lrs
        )
        contrasts[slug] = lora - full
    finite = [v for v in contrasts.values() if math.isfinite(v)]
    return {
        "id": "C5",
        "statement": (
            "Interaction (novel): the optimizer's effect on the subliminal delta, "
            "measured relative to AdamW, differs between LoRA and full fine-tuning"
        ),
        "verdict": "measured" if finite else "not_measured",
        "contrasts": contrasts,
        "max_abs_contrast": max((abs(v) for v in finite), default=float("nan")),
    }


def claims(
    agg: pd.DataFrame,
    selected_lrs: dict[str, float] | None = None,
    calibration: dict | None = None,
    quarantine: dict | None = None,
) -> dict:
    """Machine-readable verdicts for the pre-registered claims C1-C5.

    Args:
        agg: `delta_agg_table` over the whole campaign tree (it filters to the primary
            prompt set itself, and to each arm's headline LR via `selected_lrs`).
        selected_lrs: Arm key -> the Phase-1-selected LR, from `lr_selection_report.json`.
        calibration: `optim_calibration_report.json` body, so C4 can state each SGD
            ladder's usable range beside its null.
        quarantine: `quarantine_report.json` body (optional; older trees omit it).

    Returns:
        The `claims.json` body.
    """
    return {
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "min_inits": plan.CLAIM_MIN_INITS,
        "selected_lrs": selected_lrs or {},
        "dropped_arms": list((quarantine or {}).get("dropped_arms") or []),
        "claims": [
            _claim_c1(agg, selected_lrs, quarantine),
            _claim_c2(agg, selected_lrs, quarantine),
            _claim_c3(agg, selected_lrs, quarantine),
            _claim_c4(agg, selected_lrs, calibration, quarantine),
            _claim_c5(agg, selected_lrs),
        ],
    }


# ---- the one entry point the driver and the CLI both call --------------------------


def _read_report(logs_dir: Path, filename: str) -> dict | None:
    path = logs_dir / filename
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _selected_lrs(report: dict | None) -> dict[str, float] | None:
    if not report:
        return None
    return {a["arm"]: float(a["selected_lr"]) for a in report.get("arms", [])}


def build_results(
    output_root: Path,
    subdir: str,
    *,
    dry_run: bool = False,
    results_dir: Path | None = None,
) -> list[Path]:
    """Write every result artifact for the campaign tree under `subdir`.

    Args:
        output_root: Artifact root from the paths config.
        subdir: The campaign's student subdir (``gkd_students/{run_set_id}``).
        dry_run: Log the planned writes and return their paths without writing.
        results_dir: Create-only destination outside the run set (a rebuild of a
            finished campaign); default ``{output_root}/{subdir}/results``.

    Returns:
        The artifact paths, in write order.

    Raises:
        RuntimeError: The tree has no evaluated cell — finalizing an empty campaign
            would persist an all-``nan`` claims file that reads like a real null.
        FileExistsError: `results_dir` is given and already exists.
    """
    if results_dir is not None and results_dir.exists():
        raise FileExistsError(f"{results_dir} exists; a rebuild never overwrites")
    results_dir = results_dir or output_root / subdir / "results"
    logs_dir = output_root / subdir / "logs"
    preference = load_gkd_preference_results(output_root, subdir)
    if preference.empty:
        raise RuntimeError(
            f"no preference_summary.json under {output_root / subdir} — refusing to "
            "write result tables for an unevaluated campaign"
        )
    quarantine = _read_report(logs_dir, plan.QUARANTINE_REPORT)
    delta = delta_table(
        preference,
        load_gkd_capability_results(output_root, subdir),
        quarantine=quarantine,
    )
    agg = delta_agg_table(delta)
    tables: list[tuple[str, pd.DataFrame]] = [
        (plan.RESULTS_DELTA_CSV, delta),
        (plan.RESULTS_DELTA_AGG_CSV, agg),
        (
            plan.RESULTS_LOSS_MATCH_CSV,
            loss_match_table(load_gkd_train_manifests(output_root, subdir), delta),
        ),
        (
            plan.RESULTS_PROBE_CSV,
            _with_arm(load_gkd_probe_trajectories(output_root, subdir)),
        ),
        (
            plan.RESULTS_GRAD_CSV,
            _with_arm(load_gkd_grad_concentration(output_root, subdir)),
        ),
        (plan.RESULTS_QUARANTINE_CSV, quarantine_table(quarantine)),
    ]
    verdicts = claims(
        agg,
        _selected_lrs(_read_report(logs_dir, plan.LR_SELECTION_REPORT)),
        _read_report(logs_dir, plan.CALIBRATION_REPORT),
        quarantine,
    )
    written: list[Path] = []
    if not dry_run:
        results_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables:
        path = results_dir / name
        if dry_run:
            logger.info("[dry-run] would write {} ({} rows)", path, len(frame))
        else:
            atomic_write_text(path, frame.to_csv(index=False))
            logger.info("wrote {} ({} rows)", path, len(frame))
        written.append(path)
    claims_path = results_dir / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, verdicts)
        logger.info("wrote {}", claims_path)
    written.append(claims_path)
    return written
