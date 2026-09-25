"""Result tables + `claims.json` for the tmplopd run set (docs/PREREG_TMPLOPD.md).

Reads every planned param dir's eval summaries and every leaf's rollout / trainer /
probe logs under the run set, and writes `<run set>/results/`:
`leaves.csv`, `arm_summary.csv`, `contrasts.csv`, `teachers.csv`, `efficiency.csv`,
`retention.csv`, `leakage.csv`, `trajectory.csv`, `concordance.csv`, `coverage.json`,
`claims.json`. Unrun leaves are holes (listed in coverage), never errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import tmplopd_metrics as metrics
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, teacher_tag
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso


@dataclass(frozen=True)
class ResultsConfig:
    """Roots the builder reads and writes.

    Attributes:
        output_root: Campaign output root (paths YAML).
        concordance_root: Output root holding `plan.CONCORDANCE_RUN_SET`.
    """

    output_root: Path
    concordance_root: Path


def _root(output_root: Path) -> Path:
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)


def param_dir(output_root: Path, unit: plan.Unit) -> Path:
    """`unit`'s ``{teacher_tag}/{gkd_tag}`` (or control) dir."""
    root = _root(output_root)
    gtag = plan.student_gkd_tag(unit.variant)
    model = plan.label(unit.student)
    if unit.is_control:
        return root / "controls" / "base_teacher" / plan.ANIMAL / model / gtag
    t = unit.teacher
    ttag = teacher_tag(
        t.workflow, t.param_tag, unit.ftseed,
        teacher_base_label=plan.label(plan.SELF_BASE), student_base_label=model,
    )
    return root / plan.ANIMAL / model / ttag / gtag


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _leaf_extras(leaf: Path) -> dict:
    roll = _read(leaf / "rollouts_summary.json") or {}
    logs = _jsonl(leaf / "trainer_log_history.jsonl")
    first = next((r for r in logs if "gkd/fwd_kl" in r), {})
    manifest = _read(leaf / "train_manifest.json") or {}
    teacher = manifest.get("teacher") or {}
    return {
        "filter_drop_frac": roll.get("flagged_rate"),
        "n_rows_screened": roll.get("n_rows_screened"),
        "matched_form_counts": json.dumps(roll.get("matched_form_counts") or {}),
        "first_step": first.get("step"),
        "first_fwd_kl": first.get("gkd/fwd_kl"),
        "first_rev_kl": first.get("gkd/rev_kl"),
        "teacher_p_consensus": teacher.get("p_target_reference"),
        "final_train_loss": manifest.get("final_train_loss"),
    }


def load_leaves(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """(leaves, probe trajectory, holes) over every planned unit x gseed."""
    rows: list[dict] = []
    traj: list[dict] = []
    holes: list[str] = []
    for unit in plan.units():
        pdir = param_dir(output_root, unit)
        pref = _read(pdir / "preference_summary.json")
        cap = _read(pdir / "sdf_capability_summary.json") or {}
        fc = _read(pdir / "sdf_forced_choice_summary.json") or {}
        ppl = cap.get("perplexity") or {}
        base_ppl = ppl.get("base_overall")
        per_ppl = dict(ppl.get("per_seed_overall") or {})
        per_fc = dict(fc.get("per_seed_target_share") or {})
        for g in unit.gseeds:
            leaf = pdir / f"gseed{g}"
            key = f"{unit.slug}|g{g}"
            if pref is None or str(g) not in (
                pref["summaries"][0].get("per_seed_p_target") or {}
            ):
                holes.append(key)
                continue
            ident = {
                "tier": unit.tier, "arm": unit.arm, "variant": unit.variant,
                "student": unit.student, "is_control": unit.is_control,
                "workflow": "control" if unit.is_control else unit.teacher.workflow,
                "teacher_param_tag": None if unit.is_control else unit.teacher.param_tag,
                "ftseed": -1 if unit.ftseed is None else unit.ftseed,
                "gseed": g,
            }
            leaf_ppl = per_ppl.get(str(g))
            extras = _leaf_extras(leaf)
            for s in pref["summaries"]:
                rows.append(
                    {
                        **ident,
                        "set": s["prompt_set"],
                        "scoring": s.get("scoring"),
                        "p_target": (s.get("per_seed_p_target") or {}).get(str(g)),
                        "base_prior": s.get("base_p_target"),
                        "perplexity_ratio": (
                            None if leaf_ppl is None or not base_ppl else leaf_ppl / base_ppl
                        ),
                        "fc_target_share": per_fc.get(str(g)),
                        **extras,
                    }
                )
            for rec in _jsonl(leaf / "training_metrics.jsonl"):
                traj.append({**ident, **{k: rec.get(k) for k in (
                    "step", "p_target", "fc_share", "ppl", "teacher_p_target")}})
    return pd.DataFrame(rows), pd.DataFrame(traj), holes


def load_teachers(output_root: Path) -> pd.DataFrame:
    """Teacher p(lion) per set in our harness (staged teacher dirs)."""
    root = (
        Path(output_root) / run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
        / plan.ANIMAL / plan.label(plan.SELF_BASE)
    )
    rows = []
    for arm, t in plan.TEACHERS.items():
        pref = _read(root / t.param_tag / "preference_summary.json")
        cap = _read(root / t.param_tag / "sdf_capability_summary.json") or {}
        if pref is None:
            continue
        for s in pref["summaries"]:
            rows.append(
                {
                    "arm": arm, "workflow": t.workflow, "param_tag": t.param_tag,
                    "n_ftseeds": len(t.ftseeds), "set": s["prompt_set"],
                    "scoring": s.get("scoring"),
                    "teacher_p_target": s.get("p_target_mean"),
                    "per_seed_p_target": json.dumps(s.get("per_seed_p_target")),
                    "base_p_target": s.get("base_p_target"),
                    "teacher_perplexity_ratio": (cap.get("perplexity") or {}).get(
                        "perplexity_ratio"
                    ),
                }
            )
    return pd.DataFrame(rows)


def concordance(leaves: pd.DataFrame, concordance_root: Path) -> tuple[pd.DataFrame, dict]:
    """S leaves vs the ws160m2 lion self leaves at the same (ftseed, gseed), `test`."""
    path = (
        Path(concordance_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.CONCORDANCE_RUN_SET)
        / "results" / "leaves.csv"
    )
    old = pd.read_csv(path)
    old = old[
        (old["animal"] == plan.ANIMAL)
        & (old["teacher_base"] == plan.SELF_BASE)
        & (old["student_base"] == plan.SELF_BASE)
        & (old["student_step"] == plan.STEP)
        & (old["gkd_tag"] == plan.student_gkd_tag())
        & ~old["is_control"].astype(bool)
        & (old["set"] == plan.PRIMARY_PROMPT_SET)
    ][["ftseed", "gseed", "p_target"]].rename(columns={"p_target": "p_target_ws160m2"})
    old["ftseed"] = old["ftseed"].astype(int)
    new = leaves[
        (leaves["arm"] == "S") & (leaves["tier"] == plan.TIER_C)
        & (leaves["set"] == plan.PRIMARY_PROMPT_SET)
    ][["ftseed", "gseed", "p_target"]]
    joined = new.merge(old, on=["ftseed", "gseed"], how="inner")
    diff = joined["p_target"] - joined["p_target_ws160m2"]
    summary = {
        "n_pairs": int(len(joined)),
        "mean_diff": float(diff.mean()) if len(joined) else float("nan"),
        "mean_abs_diff": float(diff.abs().mean()) if len(joined) else float("nan"),
    }
    return joined, summary


def run(cfg: ResultsConfig) -> Path:
    """Build every table; return the results dir."""
    out = _root(cfg.output_root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    leaves, traj, holes = load_leaves(cfg.output_root)
    teachers = load_teachers(cfg.output_root)
    teachers.to_csv(out / "teachers.csv", index=False)
    atomic_write_json(
        out / "coverage.json",
        {"built_at": utc_now_iso(), "n_leaves_planned": plan.n_leaves(),
         "n_leaves_scored": int(leaves[["tier", "arm", "variant", "student", "ftseed",
                                        "gseed"]].drop_duplicates().shape[0])
         if not leaves.empty else 0,
         "holes": holes},
    )
    if leaves.empty:
        logger.warning("no scored leaves yet")
        return out
    leaves = metrics.quarantine(leaves)
    leaves.to_csv(out / "leaves.csv", index=False)
    traj.to_csv(out / "trajectory.csv", index=False)
    deltas = metrics.leaf_deltas(leaves)
    summary = metrics.arm_summary(deltas)
    summary.to_csv(out / "arm_summary.csv", index=False)
    contrasts = pd.DataFrame(
        [
            metrics.contrast(deltas, arm, "S", tier=plan.TIER_C, variant="legacy",
                             student=plan.SELF_BASE, prompt_set=s, scoring=sc)
            for arm in ("A10", "A15")
            for s, sc in ((plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING),
                          ("test_sample", "sample"))
        ]
    )
    contrasts.to_csv(out / "contrasts.csv", index=False)
    eff = metrics.efficiency(summary, teachers) if not teachers.empty else pd.DataFrame()
    eff.to_csv(out / "efficiency.csv", index=False)
    retained = metrics.retention(
        summary, prompt_set=plan.PRIMARY_PROMPT_SET, scoring=plan.PRIMARY_SCORING
    )
    retained.to_csv(out / "retention.csv", index=False)
    leak_cols = [*metrics.ARM_KEY, "ftseed", "gseed", "filter_drop_frac",
                 "matched_form_counts", "first_fwd_kl", "first_rev_kl",
                 "teacher_p_consensus", "final_train_loss"]
    leaves.drop_duplicates([*metrics.ARM_KEY, "ftseed", "gseed", "is_control"])[
        leak_cols + ["is_control"]
    ].to_csv(out / "leakage.csv", index=False)
    joined, conc = concordance(leaves, cfg.concordance_root)
    joined.to_csv(out / "concordance.csv", index=False)
    atomic_write_json(
        out / "claims.json",
        {"built_at": utc_now_iso(), "run_set_id": plan.RUN_SET_ID, "holes": len(holes),
         **metrics.claims(summary, contrasts, retained, conc)},
    )
    logger.info("results -> {} ({} scored rows, {} holes)", out, len(leaves), len(holes))
    return out
