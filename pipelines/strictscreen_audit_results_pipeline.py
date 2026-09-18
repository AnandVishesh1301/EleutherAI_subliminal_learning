"""Strictscreen judge-audit results: every §5.2 artifact plus ``claims.json``.

Reads this audit's plan, keys and judgments, and the reference audit
(``20260829_breadth4_stageb``) strictly read-only for S1/S2. Every write goes
through `_guarded_path`, which refuses anything outside this audit or under
the reference audit. Human-annotation files are created once and never
overwritten, so labels typed into them survive a rerun.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from pipelines.filtration_audit_results_pipeline import _agreement_allocation
from pipelines.judge_plan_pipeline import load_filtration_audit, resolve_audit_root
from subliminal_learning.config import ExperimentConfig
from subliminal_learning.evaluation.judge_results_io import (
    POSITION_COLUMNS,
    JoinReport,
    judge_yes_columns,
    load_joined_task_filtered,
    load_plan_manifest,
    load_queue_rows,
    load_queue_text_hashes,
    load_task_frame,
)
from subliminal_learning.gkd import judge_claims as jc
from subliminal_learning.gkd.judge_layout import (
    TASKS,
    audit_dir,
    audit_manifest_path,
    results_dir,
)
from subliminal_learning.gkd.judge_metrics import (
    binomial_rate,
    parse_yes_no,
    reweighted_rate,
)
from subliminal_learning.utils.code_pin_io import git_sha
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

HUMAN_SHEET_N = 200
CONFIDENCE = 0.95
POOL_TASKS: tuple[str, ...] = ("pool_open", "pool_targeted")
ANNOTATION_SUBDIR = "annotation"
HUMAN_SHEET = "human_agreement_sheet.csv"
HUMAN_BLIND = "human_agreement_sheet_blind.csv"
HUMAN_KEY = "human_agreement_key.csv"
HUMAN_LABELS = "human_agreement_labels.csv"
_LABEL_COLUMNS = ("human_references_target", "human_mode")


def run(
    cfg: ExperimentConfig,
    experiment_path: Path,
    *,
    audit_root: Path | None = None,
    reference_root: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Compute and write the strictscreen judge-audit results.

    Args:
        cfg: Paths + model config.
        experiment_path: ``configs/experiment/filtration_audit_strictscreen.yaml``.
        audit_root: Override `PathsConfig.audit_root` for this audit.
        reference_root: Audit root holding the reference audit (default: the
            same root).
        dry_run: Compute and log without writing.

    Returns:
        Paths written (or that would be written).

    Raises:
        FileNotFoundError: The audit or reference audit dir is absent.
        ValueError: The audit is the reference audit, or the YAML lacks
            ``reference`` / ``claims``.
    """
    audit_cfg = load_filtration_audit(experiment_path)
    raw = yaml.safe_load(Path(experiment_path).read_text())
    if "claims" not in raw or audit_cfg.reference_audit_id is None:
        raise ValueError(f"{experiment_path}: needs claims: and reference: blocks")
    claims_cfg = jc.parse_claims_config(
        raw["claims"], n_step_strata=audit_cfg.n_step_strata
    )
    root = resolve_audit_root(cfg.paths, audit_root)
    adir = audit_dir(root, audit_cfg.audit_id)
    ref_dir = audit_dir(reference_root or root, audit_cfg.reference_audit_id)
    if audit_cfg.audit_id == audit_cfg.reference_audit_id or _within(adir, ref_dir):
        raise ValueError(f"refusing to write results for the reference audit {ref_dir}")
    for path in (adir, ref_dir):
        if not path.is_dir():
            raise FileNotFoundError(f"audit dir does not exist: {path}")
    logger.info("strictscreen audit results {} (reference {})", adir, ref_dir)

    plan = load_plan_manifest(adir)
    leaves = {str(leaf["run_dir"]): leaf for leaf in plan["leaves"]}
    arm_of = {rd: str(leaf["arm_label"]) for rd, leaf in leaves.items()}
    animal_of = {rd: str(leaf["animal"]) for rd, leaf in leaves.items()}
    strict_arms = (claims_cfg.treatment_arm, claims_cfg.control_arm)
    totals = {str(k): int(v) for k, v in plan["per_task_totals"].items()}
    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, JoinReport] = {}
    for task in TASKS:
        if totals.get(task, 0) <= 0:
            continue
        frame, report = load_joined_task_filtered(adir, task)
        reports[task] = report
        frame = judge_yes_columns(frame, task)
        if "run_dir" in frame.columns:
            frame["arm_label"] = frame["run_dir"].map(lambda v: arm_of.get(str(v)))
        frames[task] = frame

    clean = frames.get("rollouts_clean", pd.DataFrame())
    flagged = frames.get("rollouts_flagged", pd.DataFrame())
    cells = jc.leaf_quintile_cells(
        clean,
        {rd: list(leaf["clean_pop_by_quintile"]) for rd, leaf in leaves.items()},
        arm_of,
        animal_of,
    )
    populations = {
        rd: {
            str(k): int(v)
            for k, v in dict(leaf.get("flagged_pop_by_stratum") or {}).items()
        }
        for rd, leaf in leaves.items()
    }
    strict_flagged = _valid(flagged)
    if not strict_flagged.empty:
        strict_flagged = strict_flagged.loc[
            strict_flagged["arm_label"].isin(strict_arms)
        ]

    tables: dict[str, pd.DataFrame] = {}
    docs: dict[str, object] = {}
    h1 = jc.h1_filter_efficacy(cells, claims_cfg)
    tables["rollout_rates_leaf.csv"] = _leaf_table(cells, flagged, leaves, populations)
    tables["rollout_rates_arm.csv"] = _arm_table(cells)
    tables["h1_equivalence.csv"] = _h1_table(h1)
    tables["leakage_by_step.csv"] = _step_table(cells)
    docs["step_profile.json"] = {
        "step_profile": h1.get("step_profile"),
        "h1c": h1.get("h1c"),
    }

    decoy = _valid(frames.get("decoy", pd.DataFrame()))
    decoy_clean = (
        decoy.loc[~decoy["flagged"].astype(bool)].assign(arm=lambda f: f["arm_label"])
        if not decoy.empty
        else decoy
    )
    h4a = (
        jc.h4a_decoy_false_positive(decoy_clean, claims_cfg)
        if not decoy_clean.empty
        else {"verdict": jc.INDETERMINATE, "reason": "no decoy rows"}
    )
    tables["decoy_by_arm.csv"] = _decoy_table(decoy)
    h4b = (
        jc.h4b_hard_recall(strict_flagged, populations, claims_cfg)
        if not strict_flagged.empty
        else {"verdict": jc.INDETERMINATE, "reason": "no flagged rows"}
    )
    tables["hard_positive_recall.csv"] = pd.DataFrame(
        [{"rule": rule, **vals} for rule, vals in (h4b.get("by_rule") or {}).items()]
    )
    tables["overrejection_by_rule.csv"] = _overrejection_table(
        _valid(flagged), populations
    )
    tables["mode_breakdown.csv"] = _mode_table(frames)
    tables["pool_census.csv"] = _pool_census(frames)
    n_judged = sum(len(f) for f in frames.values())
    n_fail = sum(int((~f["judge_valid"].astype(bool)).sum()) for f in frames.values())
    h4c = jc.h4c_parse_failure(n_judged, n_fail, claims_cfg)

    annotation = results_dir(adir) / ANNOTATION_SUBDIR
    labels_path = annotation / HUMAN_LABELS
    key_path = annotation / HUMAN_KEY
    h4d = jc.h4d_human_kappa(_human_labels(labels_path, key_path))
    h4 = jc.h4_judge_validity(h4a, h4b, h4c, h4d)

    s2, s2_table = _pool_rejudge(
        adir, ref_dir, frames, audit_cfg.reference_prompt_frozen_at, claims_cfg
    )
    tables["pool_rejudge_agreement.csv"] = s2_table
    docs["pool_rejudge_agreement.json"] = s2
    s1 = _rerun_vs_reference(ref_dir, cells, claims_cfg)
    tables["armC_vs_audit1.csv"] = pd.DataFrame(s1.get("leaves") or [])

    integrity = jc.integrity_report(
        _integrity_failures(plan, reports, frames, cells, strict_flagged, strict_arms)
    )
    claims = jc.assemble_claims(integrity=integrity, h1=h1, h4=h4, s1=s1, s2=s2)
    docs["claims.json"] = claims
    docs["judge_validation.json"] = _judge_validation(frames, h4a, h4b, h4c, h4d)

    out = results_dir(adir)
    written: list[Path] = []
    for name, table in tables.items():
        written.append(
            _write_csv(_guarded_path(out / name, adir, ref_dir), table, dry_run)
        )
    for name, doc in docs.items():
        written.append(
            _write_json(_guarded_path(out / name, adir, ref_dir), doc, dry_run)
        )
    written.append(
        _write_text(
            _guarded_path(out / "judge_yes_rollouts_clean.jsonl", adir, ref_dir),
            _judge_yes_clean_jsonl(adir, clean, strict_arms),
            dry_run,
        )
    )
    written.extend(
        _write_annotation(
            adir, ref_dir, frames, strict_arms, int(plan["global_seed"]), dry_run
        )
    )
    model_ids = sorted(
        {
            str(v)
            for f in frames.values()
            if "model_id" in f.columns
            for v in f["model_id"].dropna()
        }
    )
    manifest = {
        "audit_id": audit_cfg.audit_id,
        "audit_dir": str(adir),
        "reference_audit_dir": str(ref_dir),
        "plan_fingerprint": plan.get("plan_fingerprint"),
        "plan_git_sha": plan.get("git_sha"),
        "results_git_sha": git_sha(),
        "node": socket.gethostname(),
        "judge_model_ids": model_ids,
        "realised_counts": {t: asdict(r) for t, r in reports.items()},
        "headline": claims["headline"],
        "written": [str(p) for p in written],
        "at": utc_now_iso(),
    }
    written.append(
        _write_json(
            _guarded_path(out / "audit_manifest.json", adir, ref_dir), manifest, dry_run
        )
    )
    written.append(
        _write_json(
            _guarded_path(audit_manifest_path(adir), adir, ref_dir), manifest, dry_run
        )
    )
    logger.info("claims headline {} -> {} artifacts", claims["headline"], len(written))
    return written


def _within(path: Path, parent: Path) -> bool:
    child = Path(path).resolve()
    base = Path(parent).resolve()
    return child == base or base in child.parents


def _guarded_path(path: Path, adir: Path, ref_dir: Path) -> Path:
    """Return `path` if it is inside `adir` and outside `ref_dir`.

    Raises:
        ValueError: The write would land under the reference audit or outside
            this audit.
    """
    if _within(path, ref_dir):
        raise ValueError(f"refusing to write under the reference audit: {path}")
    if not _within(path, adir):
        raise ValueError(f"refusing to write outside the audit dir: {path}")
    return path


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "judge_valid" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["judge_valid"].astype(bool)]


def _leaf_table(
    cells: pd.DataFrame,
    flagged: pd.DataFrame,
    leaves: Mapping[str, Mapping[str, object]],
    populations: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    rates = jc.leaf_rates(cells)
    valid = _valid(flagged)
    rows = []
    for rec in rates.to_dict("records"):
        leaf = leaves[rec["run_dir"]]
        sub = (
            valid.loc[valid["run_dir"] == rec["run_dir"]] if not valid.empty else valid
        )
        strata = [
            (
                len(g),
                int(g["judge_yes"].eq(True).sum()),
                int(populations[rec["run_dir"]][str(s)]),
            )
            for s, g in (sub.groupby("stratum") if not sub.empty else [])
        ]
        flagged_rate = (
            reweighted_rate(strata, confidence=CONFIDENCE) if strata else None
        )
        rows.append(
            {
                **rec,
                "model": leaf.get("model"),
                "teacher_tag": leaf.get("teacher_tag"),
                "gkd_tag": leaf.get("gkd_tag"),
                "gseed": leaf.get("gseed"),
                "n_pop_flagged": int(leaf.get("flagged_pop_total") or 0),
                "n_sampled_flagged": len(sub),
                "n_yes_flagged": int(sub["judge_yes"].eq(True).sum())
                if len(sub)
                else 0,
                "flagged_rate": None if flagged_rate is None else flagged_rate.estimate,
            }
        )
    return pd.DataFrame(rows)


def _arm_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groupings = (("arm", ["arm"]), ("arm_animal", ["arm", "animal"]))
    for scope, keys in groupings:
        for key, group in cells.groupby(keys):
            used = group.loc[group["n_sampled"] > 0]
            if used.empty:
                continue
            ess = reweighted_rate(
                list(
                    used[["n_sampled", "n_yes", "n_pop"]]
                    .astype(int)
                    .itertuples(index=False, name=None)
                ),
                confidence=CONFIDENCE,
            )
            labels = key if isinstance(key, tuple) else (key,)
            rows.append(
                {
                    "scope": scope,
                    "arm": labels[0],
                    "animal": labels[1] if len(labels) > 1 else None,
                    "n_leaves": int(group["run_dir"].nunique()),
                    "rate": jc.ht_rate(group),
                    "ess_ci_low": ess.ci_low,
                    "ess_ci_high": ess.ci_high,
                    "n_effective": ess.n_effective,
                    "n_sampled": int(group["n_sampled"].sum()),
                    "n_yes": int(group["n_yes"].sum()),
                    "n_pop": int(group["n_pop"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _h1_table(h1: Mapping[str, object]) -> pd.DataFrame:
    if "h1a" not in h1:
        return pd.DataFrame(
            [{"claim": "H1", "verdict": h1.get("verdict"), "reason": h1.get("reason")}]
        )
    a, b, c = h1["h1a"], h1["h1b"], h1["h1c"]
    return pd.DataFrame(
        [
            {
                "claim": "H1a",
                "estimate": a["estimate"],
                "ci_low": a["ci_low"],
                "ci_high": a["ci_high"],
                "confidence": a["confidence"],
                "bound_low": a["bound_low"],
                "bound_high": a["bound_high"],
                "threshold": None,
                "pass": a["equivalent"],
            },
            {
                "claim": "H1b",
                "estimate": b["treatment_rate"],
                "ci_low": None,
                "ci_high": b["bootstrap_upper"],
                "confidence": b["percentile"],
                "bound_low": None,
                "bound_high": None,
                "threshold": b["threshold"],
                "pass": b["pass"],
            },
            {
                "claim": "H1c",
                "estimate": c["estimate"],
                "ci_low": c["ci_low"],
                "ci_high": c["ci_high"],
                "confidence": c["confidence"],
                "bound_low": c["bound_low"],
                "bound_high": c["bound_high"],
                "threshold": None,
                "pass": c["equivalent"],
            },
        ]
    )


def _step_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, quintile), group in cells.groupby(["arm", "step_quintile"]):
        used = group.loc[group["n_sampled"] > 0]
        if used.empty:
            continue
        ess = reweighted_rate(
            list(
                used[["n_sampled", "n_yes", "n_pop"]]
                .astype(int)
                .itertuples(index=False, name=None)
            ),
            confidence=CONFIDENCE,
        )
        rows.append(
            {
                "arm": arm,
                "step_quintile": int(quintile),
                "n_leaves": int(group["run_dir"].nunique()),
                "rate": jc.ht_rate(group),
                "ess_ci_low": ess.ci_low,
                "ess_ci_high": ess.ci_high,
                "n_sampled": int(group["n_sampled"].sum()),
                "n_yes": int(group["n_yes"].sum()),
                "n_pop": int(group["n_pop"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _decoy_table(decoy: pd.DataFrame) -> pd.DataFrame:
    if decoy.empty:
        return pd.DataFrame()
    rows = []
    frame = decoy.assign(
        _origin=np.where(decoy["flagged"].astype(bool), "flagged", "clean"),
        _w=decoy["sample_weight"].astype(float),
        _y=decoy["judge_yes"].eq(True).astype(float),
    )
    for (arm, origin, animal), group in frame.groupby(
        ["arm_label", "_origin", "animal"]
    ):
        rows.append(_decoy_row(arm, origin, animal, group))
    for (arm, origin), group in frame.groupby(["arm_label", "_origin"]):
        rows.append(_decoy_row(arm, origin, None, group))
    return pd.DataFrame(rows)


def _decoy_row(arm: object, origin: str, animal: object, group: pd.DataFrame) -> dict:
    den = float(group["_w"].sum())
    return {
        "arm": arm,
        "origin": origin,
        "animal": animal,
        "n_rows": len(group),
        "n_yes": int(group["_y"].sum()),
        "hajek_false_positive_rate": float((group["_w"] * group["_y"]).sum()) / den
        if den
        else None,
        "raw_rate": float(group["_y"].mean()),
    }


def _overrejection_table(
    flagged: pd.DataFrame,
    populations: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    if flagged.empty:
        return pd.DataFrame()
    grouped: dict[tuple[str, str], list[tuple[int, int, int]]] = {}
    for (arm, run_dir, stratum), group in flagged.groupby(
        ["arm_label", "run_dir", "stratum"]
    ):
        n_free = int(group["judge_yes"].eq(False).sum())
        grouped.setdefault((str(arm), str(stratum)), []).append(
            (len(group), n_free, int(populations[str(run_dir)][str(stratum)]))
        )
    rows = []
    for (arm, stratum), strata in sorted(grouped.items()):
        rate = reweighted_rate(strata, confidence=CONFIDENCE)
        rule, _, form = stratum.partition("|")
        rows.append({"arm": arm, "rule": rule, "matched_form": form, **asdict(rate)})
    return pd.DataFrame(rows)


def _mode_table(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task, frame in frames.items():
        yes = _valid(frame)
        if yes.empty or "mode" not in yes.columns:
            continue
        yes = yes.loc[yes["judge_yes"].eq(True)]
        arm = (
            yes["arm_label"]
            if "arm_label" in yes.columns
            else pd.Series("pool", index=yes.index)
        )
        for (arm_label, mode), group in yes.assign(_arm=arm.fillna("pool")).groupby(
            ["_arm", "mode"]
        ):
            rows.append({"task": task, "arm": arm_label, "mode": mode, "n": len(group)})
    table = pd.DataFrame(rows, columns=["task", "arm", "mode", "n"])
    if not table.empty:
        table["fraction"] = table["n"] / table.groupby(["task", "arm"])["n"].transform(
            "sum"
        )
    return table


def _pool_census(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for task in POOL_TASKS:
        valid = _valid(frames.get(task, pd.DataFrame()))
        if valid.empty:
            continue
        animal = valid["ask_animal"].fillna("any") if task == "pool_targeted" else "any"
        for (register, anim), group in valid.assign(_animal=animal).groupby(
            ["register", "_animal"]
        ):
            rate = binomial_rate(
                int(group["judge_yes"].eq(True).sum()),
                len(group),
                confidence=CONFIDENCE,
            )
            rows.append(
                {"task": task, "register": register, "animal": anim, **asdict(rate)}
            )
    return pd.DataFrame(rows)


def _pool_rejudge(
    adir: Path,
    ref_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    frozen_at: str | None,
    cfg: jc.ClaimsConfig,
) -> tuple[dict, pd.DataFrame]:
    parts = []
    audit_rows = []
    for task in POOL_TASKS:
        new = _valid(frames.get(task, pd.DataFrame()))
        if new.empty:
            continue
        ref = judge_yes_columns(load_task_frame(ref_dir, task, "judgments"), task)
        n_ref = len(ref)
        if frozen_at is not None and "judged_at" in ref.columns:
            ref = ref.loc[ref["judged_at"].astype(str) >= frozen_at]
        n_pre_freeze = n_ref - len(ref)
        ref = _valid(ref)
        if ref["row_id"].duplicated().any():
            raise ValueError(f"reference {task} judgments repeat a row_id")
        hashes_new = load_queue_text_hashes(adir, task)
        hashes_ref = load_queue_text_hashes(ref_dir, task)
        pairs = new[["row_id", "judge_yes"]].merge(
            ref[["row_id", "judge_yes"]], on="row_id", suffixes=("_new", "_ref")
        )
        same_text = pairs["row_id"].map(
            lambda r: hashes_new.get(r) == hashes_ref.get(r)
        )
        pairs = pairs.loc[same_text]
        parts.append(
            pd.DataFrame(
                {
                    "task": task,
                    "row_id": pairs["row_id"],
                    "yes_new": pairs["judge_yes_new"].astype(bool),
                    "yes_ref": pairs["judge_yes_ref"].astype(bool),
                }
            )
        )
        audit_rows.append(
            {
                "task": task,
                "n_new_valid": len(new),
                "n_ref_pre_freeze_excluded": n_pre_freeze,
                "n_text_hash_mismatch": int((~same_text).sum()),
                "n_matched": len(pairs),
            }
        )
    joined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    s2 = jc.s2_pool_rejudge(joined, cfg)
    s2["exclusions"] = audit_rows
    table = pd.DataFrame(
        [
            {
                "scope": scope,
                **{k: v for k, v in doc.items() if k != "delta"},
                **(doc.get("delta") or {}),
            }
            for scope, doc in [("pooled", s2.get("pooled"))]
            + list((s2.get("by_task") or {}).items())
            if doc
        ]
    )
    return s2, table


def _rerun_vs_reference(
    ref_dir: Path,
    cells: pd.DataFrame,
    cfg: jc.ClaimsConfig,
) -> dict:
    rerun = cells.loc[cells["arm"] == cfg.rerun_arm]
    run_dirs = sorted(rerun["run_dir"].unique())
    if not run_dirs:
        return {"verdict": jc.INDETERMINATE, "reason": "no rerun leaves"}
    ref_plan = load_plan_manifest(ref_dir)
    ref_leaves = {str(leaf["run_dir"]): leaf for leaf in ref_plan["leaves"]}
    ref_frame, _ = load_joined_task_filtered(
        ref_dir, "rollouts_clean", run_dirs=run_dirs
    )
    ref_frame = judge_yes_columns(ref_frame, "rollouts_clean")
    excluded = set(_reference_resumed(ref_dir))
    if not ref_frame.empty:
        dup = ref_frame.loc[ref_frame["row_id"].duplicated(keep=False), "run_dir"]
        excluded.update(str(v) for v in dup.unique())
    matched = [rd for rd in run_dirs if rd in ref_leaves]
    cells_ref = jc.leaf_quintile_cells(
        ref_frame,
        {rd: list(ref_leaves[rd]["clean_pop_by_quintile"]) for rd in matched},
        {rd: cfg.rerun_arm for rd in matched},
        {rd: str(ref_leaves[rd]["animal"]) for rd in matched},
    )
    s1 = jc.s1_rerun_reproduces(cells, cells_ref, excluded & set(run_dirs), cfg)
    s1["n_rerun_leaves"] = len(run_dirs)
    s1["n_missing_in_reference"] = len(run_dirs) - len(matched)
    return s1


def _reference_resumed(ref_dir: Path) -> list[str]:
    path = audit_manifest_path(ref_dir)
    if not path.is_file():
        return []
    doc = json.loads(path.read_text())
    frames = doc.get("resumed_leaf_frames") or {}
    return [str(leaf["run_dir"]) for leaf in frames.get("leaves") or []]


def _integrity_failures(
    plan: Mapping[str, object],
    reports: Mapping[str, JoinReport],
    frames: Mapping[str, pd.DataFrame],
    cells: pd.DataFrame,
    strict_flagged: pd.DataFrame,
    strict_arms: Sequence[str],
) -> dict[str, list[str]]:
    failures: dict[str, list[str]] = {}
    failures["coverage"] = [
        f"{t}: {r.n_key_only} keys unjudged, {r.n_judgment_only} judgments without key"
        for t, r in reports.items()
        if r.n_key_only or r.n_judgment_only
    ]
    failures["duplicate_row_ids"] = [
        f"{t}: {int(f['row_id'].duplicated().sum())} repeated"
        for t, f in frames.items()
        if not f.empty and bool(f["row_id"].duplicated().any())
    ]
    model_ids = {
        str(v)
        for f in frames.values()
        if "model_id" in f.columns
        for v in f["model_id"].dropna()
    }
    failures["model_id"] = (
        [] if len(model_ids) == 1 else [f"model ids {sorted(model_ids)}"]
    )
    failures["census"] = [
        f"{s['run_dir']}: {s['reason']}"
        for s in plan.get("skips") or []
        if str(s.get("reason", "")).startswith("leaf_error")
    ]
    if plan.get("expected_run_dirs") is None:
        failures["census"].append("plan was not built against an expected run-dir set")
    hashes = plan.get("screen_hashes") or {}
    strict_hashes = dict(hashes.get("strict") or {})
    strict_cells = cells.loc[cells["arm"].isin(strict_arms)]
    screen = []
    uncovered = sorted(set(strict_cells["animal"]) - set(strict_hashes))
    if not strict_hashes or uncovered:
        screen.append(
            f"strict screen hashes {strict_hashes}; animals without one {uncovered}"
        )
    if any(h in ("None", "") for h in strict_hashes.values()):
        screen.append(f"strict screen hashes {strict_hashes}")
    if set(strict_hashes.values()) & set(hashes.get("legacy") or []):
        screen.append("a legacy leaf carries a strict screen hash")
    failures["screen_hashes"] = screen
    unsampled = strict_cells.loc[strict_cells["n_sampled"] == 0]
    failures["unsampled_population"] = [
        f"{r.run_dir} q{r.step_quintile}: n_pop={r.n_pop}"
        for r in unsampled.itertuples()
    ]
    rule_mismatch = []
    if not strict_flagged.empty and "matched_rule" in strict_flagged.columns:
        prefix = strict_flagged["stratum"].astype(str).str.split("|").str[0]
        bad = strict_flagged.loc[prefix != strict_flagged["matched_rule"].astype(str)]
        if len(bad):
            rule_mismatch.append(
                f"{len(bad)} flagged keys whose stratum rule != matched_rule"
            )
    elif not strict_flagged.empty:
        rule_mismatch.append("flagged keys carry no matched_rule")
    failures["matched_rule"] = rule_mismatch
    return failures


def _judge_validation(
    frames: Mapping[str, pd.DataFrame],
    h4a: Mapping[str, object],
    h4b: Mapping[str, object],
    h4c: Mapping[str, object],
    h4d: Mapping[str, object],
) -> dict:
    recall = None
    probe = _valid(frames.get("recall_probe", pd.DataFrame()))
    if not probe.empty:
        rate = binomial_rate(
            int(probe["judge_yes"].eq(True).sum()), len(probe), confidence=CONFIDENCE
        )
        recall = asdict(rate)
    return {
        "recall_probe": recall,
        "decoy_false_positive": h4a,
        "hard_positive_recall": h4b,
        "parse_failure": h4c,
        "human_kappa": h4d,
    }


def _judge_yes_clean_jsonl(
    adir: Path,
    clean: pd.DataFrame,
    strict_arms: Sequence[str],
) -> str:
    valid = _valid(clean)
    if valid.empty:
        return ""
    yes = valid.loc[valid["judge_yes"].eq(True) & valid["arm_label"].isin(strict_arms)]
    if yes.empty:
        return ""
    keys = list(POSITION_COLUMNS)
    queue = load_queue_rows(
        adir,
        "rollouts_clean",
        list(zip(yes[keys[0]], yes[keys[1]].astype(int), strict=True)),
    )
    merged = yes.drop(
        columns=[c for c in ("prompt", "completion") if c in yes.columns]
    ).merge(queue[[*keys, "prompt", "completion"]], on=keys, how="inner")
    if len(merged) != len(yes):
        raise ValueError(
            f"queue text missing for {len(yes) - len(merged)} judge-yes rows"
        )
    merged = merged.sort_values(["run_dir", "step", "batch_idx"])
    columns = [
        "row_id",
        "task",
        "run_dir",
        "arm_label",
        "animal",
        "ask_animal",
        "step",
        "step_quintile",
        "batch_idx",
        "stratum",
        "sample_weight",
        "prompt",
        "completion",
        "references_target",
        "mode",
        "evidence",
        "confidence",
        "model_id",
        "judged_at",
    ]
    lines = [
        json.dumps(_jsonable({c: rec.get(c) for c in columns}), ensure_ascii=False)
        for rec in merged.to_dict("records")
    ]
    return "\n".join(lines) + "\n"


def _write_annotation(
    adir: Path,
    ref_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    strict_arms: Sequence[str],
    seed: int,
    dry_run: bool,
) -> list[Path]:
    directory = results_dir(adir) / ANNOTATION_SUBDIR
    paths = [
        _guarded_path(directory / name, adir, ref_dir)
        for name in (HUMAN_SHEET, HUMAN_BLIND, HUMAN_KEY)
    ]
    existing = [p for p in paths if p.exists()]
    if existing:
        logger.info("annotation files exist; never overwritten: {}", existing)
        return []
    sheet = human_agreement_sheet(adir, frames, strict_arms, seed)
    if sheet.empty:
        logger.warning("no strict-arm judgments for the human agreement sheet")
        return []
    blind = sheet[
        ["row_id", "ask_animal", "prompt", "completion", *_LABEL_COLUMNS]
    ].sort_values("row_id")
    key = sheet[
        [
            "row_id",
            "task",
            "bucket",
            "arm_label",
            "animal",
            "run_dir",
            "matched_rule",
            "matched_form",
            "references_target",
            "mode",
            "evidence",
            "confidence",
        ]
    ].sort_values("row_id")
    return [
        _write_csv(paths[0], sheet, dry_run),
        _write_csv(paths[1], blind, dry_run),
        _write_csv(paths[2], key, dry_run),
    ]


def human_agreement_sheet(
    adir: Path,
    frames: Mapping[str, pd.DataFrame],
    strict_arms: Sequence[str],
    seed: int,
) -> pd.DataFrame:
    """Up to `HUMAN_SHEET_N` strict-arm rows over clean/flagged x yes/no buckets.

    Flagged buckets are split evenly over ``matched_rule``. Label columns are
    blank.

    Args:
        adir: Audit dir (queue text source).
        frames: Joined task frames with ``judge_yes`` and ``arm_label``.
        strict_arms: Arm labels eligible for the sheet.
        seed: Plan ``global_seed``.

    Returns:
        The sheet (empty when nothing is eligible).
    """
    candidates: dict[str, dict[str, list[tuple[str, str, int]]]] = {}
    for task, kind in (("rollouts_clean", "clean"), ("rollouts_flagged", "flagged")):
        valid = _valid(frames.get(task, pd.DataFrame()))
        if valid.empty:
            continue
        valid = valid.loc[valid["arm_label"].isin(strict_arms)]
        rules = (
            valid["matched_rule"].fillna("none").astype(str)
            if kind == "flagged" and "matched_rule" in valid.columns
            else pd.Series("all", index=valid.index)
        )
        for yes, rule, shard, row in zip(
            valid["judge_yes"].eq(True),
            rules,
            valid[POSITION_COLUMNS[0]].astype(str),
            valid[POSITION_COLUMNS[1]].astype(int),
            strict=True,
        ):
            bucket = f"{kind}_{'yes' if yes else 'no'}"
            candidates.setdefault(bucket, {}).setdefault(rule, []).append(
                (task, shard, int(row))
            )
    counts = {
        b: sum(len(v) for v in by_rule.values()) for b, by_rule in candidates.items()
    }
    if not any(counts.values()):
        return pd.DataFrame()
    alloc = _agreement_allocation(counts, HUMAN_SHEET_N)
    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, str, int, str]] = []
    for bucket in sorted(alloc):
        by_rule = candidates.get(bucket, {})
        for rule, n_take in _even_split(
            {r: len(v) for r, v in by_rule.items()}, alloc[bucket]
        ).items():
            pool = sorted(by_rule[rule])
            idx = sorted(rng.choice(len(pool), size=n_take, replace=False).tolist())
            chosen.extend((*pool[i], bucket) for i in idx)
    parts = []
    keys = list(POSITION_COLUMNS)
    for task in ("rollouts_clean", "rollouts_flagged"):
        picks = [(s, r, b) for t, s, r, b in chosen if t == task]
        if not picks:
            continue
        pick_frame = pd.DataFrame(picks, columns=[*keys, "bucket"])
        sub = (
            frames[task].astype({keys[1]: int}).merge(pick_frame, on=keys, how="inner")
        )
        queue = load_queue_rows(adir, task, [(s, r) for s, r, _ in picks])
        sub = sub.drop(
            columns=[c for c in ("prompt", "completion") if c in sub.columns]
        ).merge(
            queue[[*keys, "prompt", "completion"]].astype({keys[1]: int}),
            on=keys,
            how="inner",
        )
        if len(sub) != len(picks):
            raise ValueError(
                f"queue text missing for {len(picks) - len(sub)} {task} sheet rows"
            )
        parts.append(sub)
    sheet = pd.concat(parts, ignore_index=True)
    for column in _LABEL_COLUMNS:
        sheet[column] = None
    if "matched_rule" not in sheet.columns:
        sheet["matched_rule"] = None
    columns = [
        "row_id",
        "task",
        "bucket",
        "arm_label",
        "animal",
        "run_dir",
        "ask_animal",
        "prompt",
        "completion",
        "matched_form",
        "matched_rule",
        "references_target",
        "mode",
        "evidence",
        "confidence",
        *_LABEL_COLUMNS,
    ]
    return (
        sheet[columns].sort_values(["bucket", "task", "row_id"]).reset_index(drop=True)
    )


def _even_split(capacity: Mapping[str, int], n: int) -> dict[str, int]:
    """Deal `n` picks one at a time over keys in sorted order, capped by capacity."""
    alloc = {k: 0 for k in capacity}
    remaining = min(n, sum(capacity.values()))
    while remaining > 0:
        for key in sorted(capacity):
            if remaining <= 0:
                break
            if alloc[key] < capacity[key]:
                alloc[key] += 1
                remaining -= 1
    return {k: v for k, v in alloc.items() if v}


def _human_labels(labels_path: Path, key_path: Path) -> pd.DataFrame | None:
    if not labels_path.is_file() or not key_path.is_file():
        return None
    labels = pd.read_csv(labels_path, dtype=str)
    key = pd.read_csv(key_path, dtype=str)
    merged = labels[["row_id", "human_references_target"]].merge(
        key[["row_id", "references_target"]], on="row_id", how="inner"
    )
    merged = merged.loc[merged["references_target"].notna()]
    merged = merged.loc[merged["references_target"].map(_is_yes_no)]
    return merged


def _is_yes_no(value: object) -> bool:
    try:
        parse_yes_no(value)
    except ValueError:
        return False
    return True


def _jsonable(obj: object) -> object:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is pd.NA:
        return None
    return obj


def _write_csv(path: Path, frame: pd.DataFrame, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {} ({} rows)", path, len(frame))
        return path
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def _write_json(path: Path, payload: object, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    atomic_write_json(path, _jsonable(payload))
    return path


def _write_text(path: Path, text: str, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    atomic_write_text(path, text)
    return path
