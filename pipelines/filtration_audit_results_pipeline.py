"""Persist every paper-facing filtration-audit number under ``results/``.

Loads judged shards, reweights to the leaf populations the plan sampled from
(`_leaf_populations`), and writes the CSVs/JSON named in the audit plan.
Missing tasks are omitted (never an empty-denominator rate). A failed Layer-1
doc_id gate blocks the Layer-1 precision row.
"""

from __future__ import annotations

import socket
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.config import ExperimentConfig
from subliminal_learning.evaluation.gkd_results_io import LEAF_IDENTITY_COLUMNS
from subliminal_learning.evaluation.judge_results_io import (
    POSITION_COLUMNS,
    JoinReport,
    arm_is_control,
    load_joined_task,
    load_plan_manifest,
    load_queue_rows,
    load_restream_verification,
    load_rollout_summary,
    load_throughput_probe,
    restream_gate_ok,
    task_shard_paths,
)
from subliminal_learning.gkd.judge_layout import (
    ROLLOUT_TASKS,
    TASKS,
    audit_dir,
    audit_manifest_path,
    judgments_shard_dir,
    results_dir,
)
from subliminal_learning.gkd.judge_metrics import (
    DifferenceInterval,
    RateInterval,
    binomial_rate,
    difference_of_rates,
    parse_yes_no,
    reweighted_rate,
)
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso
from pipelines.judge_plan_pipeline import load_filtration_audit, resolve_audit_root

HUMAN_AGREEMENT_N = 200
CONFIDENCE = 0.95
_SHEET_TASKS: tuple[str, ...] = (
    "pool_open",
    "pool_targeted",
    "pool_rejected",
    "rollouts_clean",
    "rollouts_flagged",
)
_BUCKETS: tuple[str, ...] = (
    "clean_yes",
    "flagged_no",
    "clean_no",
    "flagged_yes",
)
_YES_FIELD = {
    "pool_open": "references_any_animal",
}


def run(
    cfg: ExperimentConfig,
    experiment_path: Path,
    *,
    audit_id: str | None = None,
    audit_root: Path | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """Load, reweight, and write the audit ``results/`` tables.

    Args:
        cfg: Paths + model config.
        experiment_path: ``configs/experiment/filtration_audit.yaml``.
        audit_id: Override the YAML ``audit_id``.
        audit_root: Override `PathsConfig.audit_root`.
        dry_run: Compute and log writes without touching disk.

    Returns:
        Paths written (or that would be written).

    Raises:
        FileNotFoundError: The audit dir or a leaf's rollout summary is absent.
    """
    audit_cfg = load_filtration_audit(experiment_path)
    resolved_id = audit_id or audit_cfg.audit_id
    root = resolve_audit_root(cfg.paths, audit_root)
    adir = audit_dir(root, resolved_id)
    if not adir.is_dir():
        raise FileNotFoundError(f"audit dir does not exist: {adir}")
    logger.info("filtration audit results from {}", adir)

    plan = load_plan_manifest(adir)
    campaign_root = Path(cfg.paths.output_root) / _plan_campaign_subdir(plan)
    restream = load_restream_verification(adir)
    probe = load_throughput_probe(adir)
    restream_ok: bool | None
    if restream is None:
        restream_ok = None
        logger.error("restream_verification.json missing; Layer-1 precision omitted")
    else:
        restream_ok = restream_gate_ok(restream)
        if not restream_ok:
            logger.error("restream doc_id gate failed; refusing Layer-1 precision")

    frames: dict[str, pd.DataFrame] = {}
    reports: dict[str, JoinReport] = {}
    omitted: list[str] = []
    for task in TASKS:
        if not task_shard_paths(judgments_shard_dir(adir, task)):
            omitted.append(task)
            logger.warning("omitting {}: no judgment shards", task)
            continue
        frame, report = load_joined_task(adir, task)
        reports[task] = report
        if frame.empty:
            omitted.append(task)
            logger.warning("omitting {}: join produced no rows", task)
            continue
        frames[task] = _with_judge_yes(frame, task)

    leaf_plan = _plan_leaf_index(plan)
    summaries = _load_summaries(frames, campaign_root)
    populations, resumed_frames = _leaf_populations(summaries, leaf_plan, frames)
    out = results_dir(adir)
    written: list[Path] = []
    tables: dict[str, pd.DataFrame] = {}

    filt = _filtration_table(frames, populations, plan, restream, restream_ok)
    if filt is not None:
        tables["filtration_table.csv"] = filt
    census = _pool_census(frames)
    if census is not None:
        tables["pool_census.csv"] = census
    leaf_rates, leaf_meta = _rollout_rates(frames, populations)
    if leaf_rates is not None:
        tables["rollout_rates.csv"] = leaf_rates
    agg = _rollout_rates_agg(leaf_meta)
    if agg is not None:
        tables["rollout_rates_agg.csv"] = agg
    by_step = _leakage_by_step(frames, leaf_plan)
    if by_step is not None:
        tables["leakage_by_step.csv"] = by_step
    per_form = _per_form_precision(frames, populations)
    if per_form is not None:
        tables["per_form_precision.csv"] = per_form
    modes = _mode_breakdown(frames)
    if modes is not None:
        tables["mode_breakdown.csv"] = modes

    validation = _judge_validation(frames)
    sheet = _human_agreement_sheet(adir, frames, plan)

    for name, table in tables.items():
        written.append(_write_csv(out / name, table, dry_run=dry_run))
    written.append(
        _write_json(out / "judge_validation.json", validation, dry_run=dry_run)
    )
    if sheet is not None:
        written.append(
            _write_csv(out / "human_agreement_sheet.csv", sheet, dry_run=dry_run)
        )

    model_ids = _model_ids(frames)
    if len(model_ids) > 1:
        logger.error("multiple judge model ids: {}", model_ids)
    manifest = {
        "audit_id": resolved_id,
        "audit_dir": str(adir),
        "judge_model_id": model_ids[0] if len(model_ids) == 1 else None,
        "judge_model_ids": model_ids,
        "sampling_seed": _plan_seed(plan),
        "git_sha": _plan_git_sha(plan),
        "node": socket.gethostname(),
        "realised_counts": {t: asdict(r) for t, r in reports.items()},
        "omitted_tasks": omitted,
        "restream_verification_ok": restream_ok,
        "restream_verification": restream,
        "throughput_probe": probe,
        "written": [str(p) for p in written],
        "at": utc_now_iso(),
    }
    coverage = _flagged_sample_coverage(plan)
    if coverage is not None:
        manifest["flagged_sample_coverage"] = coverage
    if resumed_frames is not None:
        manifest["resumed_leaf_frames"] = resumed_frames
    written.append(_write_json(out / "audit_manifest.json", manifest, dry_run=dry_run))
    written.append(_write_json(audit_manifest_path(adir), manifest, dry_run=dry_run))
    logger.info(
        "filtration audit results: {} artifacts; omitted tasks={}",
        len(written),
        omitted,
    )
    return written


def _with_judge_yes(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    out = frame.copy()
    if "parse_ok" not in out.columns:
        logger.error("{} judgments missing parse_ok; excluding all from rates", task)
        ok = pd.Series(False, index=out.index)
    else:
        ok = out["parse_ok"].map(_as_parse_ok)
    field = _YES_FIELD.get(task, "references_target")
    if field not in out.columns:
        raise ValueError(f"{task}: missing judge field {field!r}")
    yes = pd.Series(pd.NA, index=out.index, dtype=object)
    if bool(ok.any()):
        yes.loc[ok] = out.loc[ok, field].map(parse_yes_no)
    out["judge_valid"] = ok.to_numpy()
    out["judge_yes"] = yes
    return out


def _load_summaries(
    frames: Mapping[str, pd.DataFrame],
    campaign_root: Path,
) -> dict[str, dict]:
    """Load each judged leaf's rollout summary.

    Args:
        frames: Judgment frames keyed by task.
        campaign_root: ``output_root / campaign_subdir``; keys store ``run_dir``
            relative to it, so joining here is what makes the leaf resolvable.

    Returns:
        Summary dict per relative ``run_dir``.

    Raises:
        FileNotFoundError: A judged leaf has no summary under `campaign_root`.
    """
    run_dirs: set[str] = set()
    for task in ROLLOUT_TASKS:
        frame = frames.get(task)
        if frame is None or "run_dir" not in frame.columns:
            continue
        run_dirs.update(
            str(v).strip() for v in frame["run_dir"].dropna() if str(v).strip()
        )
    summaries: dict[str, dict] = {}
    for run_dir in sorted(run_dirs):
        leaf = campaign_root / run_dir
        if not leaf.is_dir():
            raise FileNotFoundError(
                f"judged leaf {run_dir!r} not found under campaign root {campaign_root}"
            )
        summaries[run_dir] = load_rollout_summary(leaf)
    return summaries


def _leaf_populations(
    summaries: Mapping[str, Mapping[str, object]],
    leaf_plan: Mapping[str, Mapping[str, object]],
    frames: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, dict], dict | None]:
    """Resolve the reweighting denominators for every judged leaf.

    Clean denominators come from the plan's per-quintile scan, not the summary
    census: a leaf whose training resumed appended a second pass to
    ``rollouts_sample.jsonl``, so the plan sampled from more clean rows than
    the census reports, and Horvitz-Thompson needs the frame that was actually
    sampled. Flagged populations stay on the summary because the plan's
    per-form budgets were drawn against ``matched_form_counts``.

    Args:
        summaries: Rollout summary per relative ``run_dir``.
        leaf_plan: Plan manifest leaf rows keyed by relative ``run_dir``.
        frames: Judgment frames keyed by task, for the population gate.

    Returns:
        Population mapping per ``run_dir`` (``n_clean``, ``n_flagged``,
        ``matched_form_counts``, ``n_rows_screened``) and a delta report for
        the audit manifest, or ``None`` when every leaf agrees with its census.
    """
    populations: dict[str, dict] = {}
    resumed: list[dict] = []
    for run_dir, summary in summaries.items():
        n_clean = _clean_pop_total(leaf_plan, run_dir)
        n_census = int(summary["n_clean"])
        n_flagged = int(summary["n_flagged"])
        if n_clean != n_census:
            resumed.append(
                {
                    "run_dir": run_dir,
                    "n_clean_sampled_frame": n_clean,
                    "n_clean_census": n_census,
                }
            )
        populations[run_dir] = {
            "n_clean": n_clean,
            "n_flagged": n_flagged,
            "matched_form_counts": {
                str(k): int(v) for k, v in dict(summary["matched_form_counts"]).items()
            },
            "n_rows_screened": n_clean + n_flagged,
        }
        _check_leaf_population(run_dir, frames, populations[run_dir])
    if not resumed:
        return populations, None
    extra = sum(r["n_clean_sampled_frame"] - r["n_clean_census"] for r in resumed)
    logger.warning(
        "{} leaves sampled a larger clean frame than their census reports "
        "(+{} rows); reweighting on the sampled frame",
        len(resumed),
        extra,
    )
    report = {
        "n_leaves": len(resumed),
        "n_extra_clean_rows": extra,
        "note": (
            "rollouts_sample.jsonl holds more clean rows than "
            "rollouts_summary.json counts (training resumed and appended a "
            "second pass); clean denominators use the sampled frame"
        ),
        "leaves": sorted(resumed, key=lambda r: str(r["run_dir"])),
    }
    return populations, report


def _clean_pop_total(
    leaf_plan: Mapping[str, Mapping[str, object]],
    run_dir: str,
) -> int:
    leaf = leaf_plan.get(run_dir)
    pops = None if leaf is None else leaf.get("clean_pop_by_quintile")
    if not pops:
        raise ValueError(
            f"{run_dir}: plan manifest has no clean_pop_by_quintile; cannot "
            "reweight without the clean population that was sampled"
        )
    return sum(int(v) for v in pops)


def _check_leaf_population(
    run_dir: str,
    frames: Mapping[str, pd.DataFrame],
    population: Mapping[str, object],
) -> None:
    n_clean = int(population["n_clean"])
    n_flagged = int(population["n_flagged"])
    form_counts = {
        str(k): int(v) for k, v in dict(population["matched_form_counts"]).items()
    }
    n_judged_clean = _leaf_count(frames.get("rollouts_clean"), run_dir)
    n_judged_flagged = _leaf_count(frames.get("rollouts_flagged"), run_dir)
    if n_judged_clean > n_clean:
        raise ValueError(
            f"{run_dir}: {n_judged_clean} judged clean rows exceed n_clean={n_clean}"
        )
    if n_judged_flagged > n_flagged:
        raise ValueError(
            f"{run_dir}: {n_judged_flagged} judged flagged rows exceed "
            f"n_flagged={n_flagged}"
        )
    flagged = _leaf_frame(frames.get("rollouts_flagged"), run_dir)
    if flagged is None or "matched_form" not in flagged.columns:
        return
    forms = flagged["matched_form"].dropna().astype(str)
    for form, n_judged in forms.value_counts().items():
        n_pop = form_counts.get(str(form), 0)
        if int(n_judged) > n_pop:
            raise ValueError(
                f"{run_dir}: {int(n_judged)} judged rows for "
                f"matched_form={form!r} exceed population {n_pop}"
            )


def _filtration_table(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, Mapping[str, object]],
    plan: Mapping[str, object],
    restream: dict | None,
    restream_ok: bool | None,
) -> pd.DataFrame | None:
    rows: list[dict] = []
    pool_open = frames.get("pool_open")
    residual_1 = _valid_yes(pool_open) if pool_open is not None else None
    over_1 = None
    layer1_notes = ""
    if restream_ok is not True:
        layer1_notes = "Layer-1 precision omitted (restream gate not ok)"
    elif "pool_rejected" not in frames:
        layer1_notes = "Layer-1 precision omitted (pool_rejected absent)"
    else:
        rejected = _valid(frames["pool_rejected"])
        n_free = int(rejected["judge_yes"].eq(False).sum())
        over_1 = binomial_rate(n_free, len(rejected), confidence=CONFIDENCE)
    rows.append(
        _layer_row(
            layer=1,
            name="pool_screen",
            rows_in=_layer1_rows_in(plan, restream),
            regex_rejected=_restream_n_rejected(restream),
            residual=residual_1,
            overrejection=over_1,
            notes=layer1_notes,
        )
    )

    if populations:
        rows_in_2 = sum(int(s["n_rows_screened"]) for s in populations.values())
        rejected_2 = sum(int(s["n_flagged"]) for s in populations.values())
        residual_2 = _clean_population_rate(frames, populations)
        over_2 = _flagged_overrejection_rate(frames, populations)
        rows.append(
            _layer_row(
                layer=2,
                name="rollout_screen",
                rows_in=rows_in_2,
                regex_rejected=rejected_2,
                residual=residual_2,
                overrejection=over_2,
                notes="",
            )
        )
    elif "rollouts_clean" not in frames and "rollouts_flagged" not in frames:
        logger.warning("Layer 2 omitted: no rollout judgments")

    rows.append(
        _layer_row(
            layer=3,
            name="not_a_text_filter",
            rows_in=None,
            regex_rejected=None,
            residual=None,
            overrejection=None,
            notes="vocab-mask ablation; not a text filter",
        )
    )
    return pd.DataFrame(rows)


def _pool_census(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    rows: list[dict] = []
    if "pool_open" in frames:
        rows.extend(_census_rows(frames["pool_open"], "pool_open", animal_col=None))
    if "pool_targeted" in frames:
        rows.extend(
            _census_rows(
                frames["pool_targeted"],
                "pool_targeted",
                animal_col="ask_animal",
            )
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _census_rows(
    frame: pd.DataFrame,
    task: str,
    *,
    animal_col: str | None,
) -> list[dict]:
    valid = _valid(frame)
    if valid.empty:
        return []
    register = (
        valid["register"].fillna("").astype(str)
        if "register" in valid.columns
        else pd.Series("", index=valid.index)
    )
    animal = (
        valid[animal_col].fillna("any").astype(str)
        if animal_col and animal_col in valid.columns
        else pd.Series("any", index=valid.index)
    )
    rows: list[dict] = []
    grouped = valid.assign(_register=register, _animal=animal)
    for (reg, anim), group in grouped.groupby(["_register", "_animal"], dropna=False):
        rate = binomial_rate(
            int(group["judge_yes"].eq(True).sum()),
            len(group),
            confidence=CONFIDENCE,
        )
        rows.append(
            {
                "task": task,
                "register": reg,
                "animal": anim,
                **_rate_cols(rate, prefix=""),
            }
        )
    return rows


def _rollout_rates(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, Mapping[str, object]],
) -> tuple[pd.DataFrame | None, list[dict]]:
    if not populations:
        return None, []
    rows: list[dict] = []
    meta: list[dict] = []
    for run_dir, population in populations.items():
        identity = _leaf_identity(frames, run_dir)
        n_clean = int(population["n_clean"])
        n_flagged = int(population["n_flagged"])
        form_counts = {
            str(k): int(v) for k, v in dict(population["matched_form_counts"]).items()
        }
        clean = _valid_leaf(frames.get("rollouts_clean"), run_dir)
        flagged = _valid_leaf(frames.get("rollouts_flagged"), run_dir)
        n_s_clean = 0 if clean is None else len(clean)
        n_y_clean = 0 if clean is None else int(clean["judge_yes"].eq(True).sum())
        clean_rate = (
            reweighted_rate(((n_s_clean, n_y_clean, n_clean),), confidence=CONFIDENCE)
            if n_s_clean
            else None
        )
        form_strata: list[tuple[int, int, int]] = []
        n_s_flagged = 0 if flagged is None else len(flagged)
        n_y_flagged = 0 if flagged is None else int(flagged["judge_yes"].eq(True).sum())
        if flagged is not None and n_s_flagged:
            forms = flagged["matched_form"].fillna("").astype(str)
            for form, n_pop in form_counts.items():
                sub = flagged.loc[forms == form]
                n_s = int(len(sub))
                if n_s == 0:
                    continue
                n_y = int(sub["judge_yes"].eq(True).sum())
                form_strata.append((n_s, n_y, n_pop))
        flagged_rate = (
            reweighted_rate(form_strata, confidence=CONFIDENCE) if form_strata else None
        )
        raw_clean = (n_y_clean / n_s_clean) if n_s_clean else None
        raw_flagged = (n_y_flagged / n_s_flagged) if n_s_flagged else None
        is_ctrl = bool(identity.get("is_control", False))
        arm = identity.get("arm") or ("control" if is_ctrl else "treatment")
        rows.append(
            {
                **identity,
                "arm": arm,
                "n_pop_clean": n_clean,
                "n_pop_flagged": n_flagged,
                "n_sampled_clean": n_s_clean,
                "n_yes_clean": n_y_clean,
                "raw_clean_rate": raw_clean,
                "reweighted_clean_rate": (
                    None if clean_rate is None else clean_rate.estimate
                ),
                "clean_ci_low": (None if clean_rate is None else clean_rate.ci_low),
                "clean_ci_high": (None if clean_rate is None else clean_rate.ci_high),
                "n_sampled_flagged": n_s_flagged,
                "n_yes_flagged": n_y_flagged,
                "raw_flagged_rate": raw_flagged,
                "reweighted_flagged_rate": (
                    None if flagged_rate is None else flagged_rate.estimate
                ),
                "flagged_ci_low": (
                    None if flagged_rate is None else flagged_rate.ci_low
                ),
                "flagged_ci_high": (
                    None if flagged_rate is None else flagged_rate.ci_high
                ),
            }
        )
        meta.append(
            {
                "run_dir": run_dir,
                "animal": identity.get("animal"),
                "arm": arm,
                "is_control": is_ctrl,
                "n_sampled_clean": n_s_clean,
                "n_yes_clean": n_y_clean,
                "n_pop_clean": n_clean,
                "clean_rate": clean_rate,
            }
        )
    return pd.DataFrame(rows), meta


def _rollout_rates_agg(
    leaf_meta: Sequence[Mapping[str, object]],
) -> pd.DataFrame | None:
    usable = [m for m in leaf_meta if m.get("clean_rate") is not None]
    if not usable:
        return None
    rows: list[dict] = []
    by_group: dict[tuple[object, object, bool], list] = defaultdict(list)
    for item in usable:
        key = (item["animal"], item["arm"], bool(item["is_control"]))
        by_group[key].append(item)
    for key, items in sorted(by_group.items(), key=lambda kv: (str(kv[0]),)):
        rate = reweighted_rate(
            [
                (
                    int(i["n_sampled_clean"]),
                    int(i["n_yes_clean"]),
                    int(i["n_pop_clean"]),
                )
                for i in items
            ],
            confidence=CONFIDENCE,
        )
        rows.append(
            {
                "scope": "animal_arm",
                "animal": key[0],
                "arm": key[1],
                "is_control": key[2],
                "n_leaves": len(items),
                **_rate_cols(rate, prefix=""),
                "delta": None,
                "delta_ci_low": None,
                "delta_ci_high": None,
            }
        )

    treat = [i for i in usable if not i["is_control"]]
    ctrl = [i for i in usable if i["is_control"]]
    pooled_t = _pool_leaves(treat)
    pooled_c = _pool_leaves(ctrl)
    if pooled_t is not None:
        rows.append(_pooled_row("pooled_treatment", False, treat, pooled_t))
    if pooled_c is not None:
        rows.append(_pooled_row("pooled_control", True, ctrl, pooled_c))
    if pooled_t is not None and pooled_c is not None:
        delta = difference_of_rates(pooled_t, pooled_c)
        rows.append(_delta_row("pooled_delta", None, None, delta, pooled_t))

    animals = sorted({str(i["animal"]) for i in usable if i.get("animal")})
    for animal in animals:
        t_items = [i for i in treat if str(i["animal"]) == animal]
        c_items = [i for i in ctrl if str(i["animal"]) == animal]
        t_rate = _pool_leaves(t_items)
        c_rate = _pool_leaves(c_items)
        if t_rate is None or c_rate is None:
            continue
        delta = difference_of_rates(t_rate, c_rate)
        rows.append(
            _delta_row("animal_delta", animal, "treatment-control", delta, t_rate)
        )
    return pd.DataFrame(rows)


def _leakage_by_step(
    frames: Mapping[str, pd.DataFrame],
    leaf_plan: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame | None:
    """Population-reweighted clean leakage per (animal, arm, step quintile).

    Args:
        frames: Judgment frames keyed by task.
        leaf_plan: Plan-manifest leaf records keyed by relative ``run_dir``.

    Returns:
        One row per cell, or None when no clean judgments carry a quintile.

    Raises:
        ValueError: A judged leaf has no ``clean_pop_by_quintile``, which would
            silently fall back to a uniform split that the flagged rate makes
            wrong (the screen rejects more late in training).
    """
    clean = frames.get("rollouts_clean")
    if clean is None or "step_quintile" not in clean.columns:
        return None
    valid = _valid(clean)
    if valid.empty:
        return None
    rows: list[dict] = []
    valid = valid.copy()
    valid["_run"] = valid["run_dir"].map(lambda v: str(v).strip())
    valid["_q"] = valid["step_quintile"]
    valid["_ctrl"] = [
        arm_is_control(a, c)
        for a, c in zip(
            valid["arm"] if "arm" in valid.columns else [None] * len(valid),
            valid["is_control"]
            if "is_control" in valid.columns
            else [None] * len(valid),
            strict=True,
        )
    ]
    valid["_arm"] = [
        ("control" if c else (a if isinstance(a, str) and a else "treatment"))
        for a, c in zip(
            valid["arm"] if "arm" in valid.columns else [None] * len(valid),
            valid["_ctrl"],
            strict=True,
        )
    ]
    for (animal, arm, is_ctrl, quintile), group in valid.groupby(
        ["animal", "_arm", "_ctrl", "_q"], dropna=False
    ):
        strata: list[tuple[int, int, int]] = []
        for run_dir, leaf in group.groupby("_run", dropna=False):
            n_s = len(leaf)
            n_y = int(leaf["judge_yes"].eq(True).sum())
            n_pop = _clean_pop_in_quintile(leaf_plan, str(run_dir), quintile)
            strata.append((n_s, n_y, max(n_pop, n_s)))
        if not strata:
            continue
        rate = reweighted_rate(strata, confidence=CONFIDENCE)
        rows.append(
            {
                "animal": animal,
                "arm": arm,
                "is_control": bool(is_ctrl),
                "step_quintile": quintile,
                "n_leaves": len(strata),
                **_rate_cols(rate, prefix=""),
            }
        )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _per_form_precision(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, Mapping[str, object]],
) -> pd.DataFrame | None:
    flagged = frames.get("rollouts_flagged")
    if flagged is None or not populations:
        return None
    valid = _valid(flagged)
    if valid.empty:
        return None
    valid = valid.copy()
    valid["_run"] = valid["run_dir"].map(lambda v: str(v).strip())
    valid["_form"] = valid["matched_form"].fillna("").astype(str)
    strata_by_form: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    n_pop_by_form: dict[str, int] = defaultdict(int)
    for run_dir, population in populations.items():
        leaf = valid.loc[valid["_run"] == run_dir]
        form_counts = {
            str(k): int(v) for k, v in dict(population["matched_form_counts"]).items()
        }
        for form, n_pop in form_counts.items():
            sub = leaf.loc[leaf["_form"] == form]
            n_s = int(len(sub))
            if n_s == 0:
                continue
            n_free = int(sub["judge_yes"].eq(False).sum())
            strata_by_form[form].append((n_s, n_free, n_pop))
            n_pop_by_form[form] += n_pop
    if not strata_by_form:
        return None
    rows = []
    for form in sorted(strata_by_form):
        rate = reweighted_rate(strata_by_form[form], confidence=CONFIDENCE)
        rows.append(
            {
                "matched_form": form,
                "n_population": n_pop_by_form[form],
                **_rate_cols(rate, prefix=""),
            }
        )
    return pd.DataFrame(rows)


def _mode_breakdown(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame | None:
    rows: list[dict] = []
    for task, frame in frames.items():
        valid = _valid(frame)
        yes = valid.loc[valid["judge_yes"].eq(True)]
        if yes.empty or "mode" not in yes.columns:
            continue
        counts = yes["mode"].fillna("").astype(str).value_counts()
        total = int(counts.sum())
        for mode, n in counts.items():
            rows.append(
                {
                    "task": task,
                    "mode": mode,
                    "n": int(n),
                    "fraction": int(n) / total,
                }
            )
    if not rows:
        return None
    return pd.DataFrame(rows)


def _judge_validation(frames: Mapping[str, pd.DataFrame]) -> dict:
    recall = None
    if "recall_probe" in frames:
        valid = _valid(frames["recall_probe"])
        if not valid.empty:
            rate = binomial_rate(
                int(valid["judge_yes"].eq(True).sum()),
                len(valid),
                confidence=CONFIDENCE,
            )
            recall = {"recall": rate.estimate, **_rate_json(rate)}
    decoy = None
    if "decoy" in frames:
        valid = _valid(frames["decoy"])
        if not valid.empty:
            n_no = int(valid["judge_yes"].eq(False).sum())
            rate = binomial_rate(n_no, len(valid), confidence=CONFIDENCE)
            decoy = {"specificity": rate.estimate, **_rate_json(rate)}
    n_judged = 0
    n_fail = 0
    for frame in frames.values():
        n_judged += len(frame)
        if "judge_valid" in frame.columns:
            n_fail += int((~frame["judge_valid"].astype(bool)).sum())
        else:
            n_fail += len(frame)
    parse_rate = (n_fail / n_judged) if n_judged else None
    return {
        "recall_probe": recall,
        "decoy_specificity": decoy,
        "parse_failure": {
            "n_judged": n_judged,
            "n_parse_fail": n_fail,
            "rate": parse_rate,
        },
        "human_kappa": None,
    }


def _human_agreement_sheet(
    audit_dir: Path,
    frames: Mapping[str, pd.DataFrame],
    plan: Mapping[str, object],
) -> pd.DataFrame | None:
    buckets: dict[str, list[tuple[str, str, int]]] = {name: [] for name in _BUCKETS}
    for task in _SHEET_TASKS:
        frame = frames.get(task)
        if frame is None or "flagged" not in frame.columns:
            continue
        valid = _valid(frame)
        if valid.empty:
            continue
        flagged = valid["flagged"].map(_as_flagged)
        yes = valid["judge_yes"].eq(True)
        shards = valid[POSITION_COLUMNS[0]].astype(str)
        shard_rows = valid[POSITION_COLUMNS[1]].astype(int)
        labels = np.select(
            [
                ~flagged.to_numpy() & yes.to_numpy(),
                flagged.to_numpy() & ~yes.to_numpy(),
                ~flagged.to_numpy() & ~yes.to_numpy(),
                flagged.to_numpy() & yes.to_numpy(),
            ],
            list(_BUCKETS),
            default="",
        )
        for shard, shard_row, bucket in zip(shards, shard_rows, labels, strict=True):
            if bucket:
                buckets[str(bucket)].append((task, str(shard), int(shard_row)))

    counts = {name: len(items) for name, items in buckets.items()}
    if not any(counts.values()):
        return None
    alloc = _agreement_allocation(counts, HUMAN_AGREEMENT_N)
    rng = np.random.default_rng(int(_plan_seed(plan)))
    chosen: list[tuple[str, str, int, str]] = []
    for bucket, n_take in alloc.items():
        pool = sorted(buckets[bucket])
        if n_take >= len(pool):
            picked = pool
        else:
            idx = rng.choice(len(pool), size=n_take, replace=False)
            picked = [pool[i] for i in sorted(idx.tolist())]
        chosen.extend(
            (task, shard, shard_row, bucket) for task, shard, shard_row in picked
        )
    if not chosen:
        return None

    by_task: dict[str, list[tuple[str, int]]] = defaultdict(list)
    bucket_of: dict[tuple[str, str, int], str] = {}
    for task, shard, shard_row, bucket in chosen:
        by_task[task].append((shard, shard_row))
        bucket_of[(task, shard, shard_row)] = bucket

    parts: list[pd.DataFrame] = []
    keys = list(POSITION_COLUMNS)
    for task, positions in by_task.items():
        picks = pd.DataFrame(positions, columns=keys)
        frame = frames[task].astype({POSITION_COLUMNS[1]: int})
        sub = frame.merge(picks, on=keys, how="inner")
        sub = sub.drop(
            columns=[c for c in ("prompt", "completion") if c in sub.columns]
        )
        queue = load_queue_rows(audit_dir, task, positions)
        if queue.empty:
            raise ValueError(
                f"queue text missing for {len(positions)} {task} agreement rows"
            )
        keep_q = [*keys, *(c for c in ("prompt", "completion") if c in queue.columns)]
        queue = queue[keep_q].astype({POSITION_COLUMNS[1]: int})
        sub = sub.merge(queue, on=keys, how="inner")
        if len(sub) != len(positions):
            raise ValueError(
                f"queue text missing for {len(positions) - len(sub)} "
                f"{task} agreement rows"
            )
        sub["disagreement"] = [
            bucket_of[(task, str(s), int(r))]
            for s, r in zip(sub[keys[0]], sub[keys[1]], strict=True)
        ]
        parts.append(sub)
    sheet = pd.concat(parts, ignore_index=True)
    sheet["human_references_target"] = None
    sheet["human_mode"] = None
    cols = [
        c
        for c in (
            "row_id",
            "task",
            "prompt",
            "completion",
            "ask_animal",
            "flagged",
            "matched_form",
            "references_target",
            "mode",
            "evidence",
            "confidence",
            "stratum",
            "run_dir",
            "animal",
            "arm",
            "judge_yes",
            "disagreement",
            "human_references_target",
            "human_mode",
        )
        if c in sheet.columns
    ]
    return sheet[cols].sort_values(["disagreement", "task", "row_id"])


def _agreement_allocation(
    counts: Mapping[str, int],
    n_total: int,
) -> dict[str, int]:
    """Split `n_total` sheet rows over the four regex/judge buckets.

    Half the budget goes to the two disagreement buckets -- the rows where a
    judge error would actually move a reported rate. Short buckets hand their
    remainder to the next bucket in `_BUCKETS` order, which lists the
    disagreement buckets first.
    """
    n_disagree = n_total // 2
    n_agree = n_total - n_disagree
    targets = {
        "clean_yes": (n_disagree + 1) // 2,
        "flagged_no": n_disagree // 2,
        "clean_no": (n_agree + 1) // 2,
        "flagged_yes": n_agree // 2,
    }
    alloc = {name: min(counts.get(name, 0), targets[name]) for name in _BUCKETS}
    leftover = n_total - sum(alloc.values())
    for name in _BUCKETS:
        if leftover <= 0:
            break
        take = min(leftover, counts.get(name, 0) - alloc[name])
        alloc[name] += take
        leftover -= take
    return alloc


def _clean_population_rate(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, Mapping[str, object]],
) -> RateInterval | None:
    strata: list[tuple[int, int, int]] = []
    for run_dir, population in populations.items():
        clean = _valid_leaf(frames.get("rollouts_clean"), run_dir)
        if clean is None or not len(clean):
            continue
        strata.append(
            (
                len(clean),
                int(clean["judge_yes"].eq(True).sum()),
                int(population["n_clean"]),
            )
        )
    if not strata:
        return None
    return reweighted_rate(strata, confidence=CONFIDENCE)


def _flagged_overrejection_rate(
    frames: Mapping[str, pd.DataFrame],
    populations: Mapping[str, Mapping[str, object]],
) -> RateInterval | None:
    strata: list[tuple[int, int, int]] = []
    for run_dir, population in populations.items():
        flagged = _valid_leaf(frames.get("rollouts_flagged"), run_dir)
        if flagged is None or not len(flagged):
            continue
        forms = flagged["matched_form"].fillna("").astype(str)
        form_counts = {
            str(k): int(v) for k, v in dict(population["matched_form_counts"]).items()
        }
        for form, n_pop in form_counts.items():
            sub = flagged.loc[forms == form]
            n_s = int(len(sub))
            if n_s == 0:
                continue
            n_free = int(sub["judge_yes"].eq(False).sum())
            strata.append((n_s, n_free, n_pop))
    if not strata:
        return None
    return reweighted_rate(strata, confidence=CONFIDENCE)


def _leaf_identity(
    frames: Mapping[str, pd.DataFrame],
    run_dir: str,
) -> dict:
    for task in ROLLOUT_TASKS:
        frame = _leaf_frame(frames.get(task), run_dir)
        if frame is None or frame.empty:
            continue
        row = frame.iloc[0]
        identity = {
            c: row[c] if c in frame.columns else None for c in LEAF_IDENTITY_COLUMNS
        }
        identity["run_dir"] = run_dir
        identity["arm"] = row["arm"] if "arm" in frame.columns else None
        identity["is_control"] = arm_is_control(
            identity.get("arm"), identity.get("is_control")
        )
        if not identity.get("animal") and "animal" in frame.columns:
            identity["animal"] = row["animal"]
        return identity
    return {
        "run_dir": run_dir,
        "arm": None,
        "is_control": False,
        **{c: None for c in LEAF_IDENTITY_COLUMNS},
    }


def _leaf_frame(frame: pd.DataFrame | None, run_dir: str) -> pd.DataFrame | None:
    if frame is None or "run_dir" not in frame.columns:
        return None
    mask = frame["run_dir"].map(lambda v: str(v).strip() == run_dir)
    return frame.loc[mask]


def _valid_leaf(frame: pd.DataFrame | None, run_dir: str) -> pd.DataFrame | None:
    leaf = _leaf_frame(frame, run_dir)
    if leaf is None:
        return None
    return _valid(leaf)


def _leaf_count(frame: pd.DataFrame | None, run_dir: str) -> int:
    leaf = _leaf_frame(frame, run_dir)
    return 0 if leaf is None else len(leaf)


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    if "judge_valid" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["judge_valid"].astype(bool)]


def _valid_yes(frame: pd.DataFrame) -> RateInterval | None:
    valid = _valid(frame)
    if valid.empty:
        return None
    return binomial_rate(
        int(valid["judge_yes"].eq(True).sum()),
        len(valid),
        confidence=CONFIDENCE,
    )


def _pool_leaves(items: Sequence[Mapping[str, object]]) -> RateInterval | None:
    strata = [
        (
            int(i["n_sampled_clean"]),
            int(i["n_yes_clean"]),
            int(i["n_pop_clean"]),
        )
        for i in items
        if int(i["n_sampled_clean"])
    ]
    if not strata:
        return None
    return reweighted_rate(strata, confidence=CONFIDENCE)


def _pooled_row(
    scope: str,
    is_control: bool,
    items: Sequence[Mapping[str, object]],
    rate: RateInterval,
) -> dict:
    return {
        "scope": scope,
        "animal": None,
        "arm": "control" if is_control else "treatment",
        "is_control": is_control,
        "n_leaves": len(items),
        **_rate_cols(rate, prefix=""),
        "delta": None,
        "delta_ci_low": None,
        "delta_ci_high": None,
    }


def _delta_row(
    scope: str,
    animal: object,
    arm: object,
    delta: DifferenceInterval,
    rate: RateInterval,
) -> dict:
    return {
        "scope": scope,
        "animal": animal,
        "arm": arm,
        "is_control": False,
        "n_leaves": None,
        **_rate_cols(rate, prefix=""),
        "delta": delta.estimate,
        "delta_ci_low": delta.ci_low,
        "delta_ci_high": delta.ci_high,
    }


def _layer_row(
    *,
    layer: int,
    name: str,
    rows_in: object,
    regex_rejected: object,
    residual: RateInterval | None,
    overrejection: RateInterval | None,
    notes: str,
) -> dict:
    return {
        "layer": layer,
        "name": name,
        "rows_in": rows_in,
        "regex_rejected": regex_rejected,
        "judge_flagged_residual": (None if residual is None else residual.estimate),
        "residual_ci_low": None if residual is None else residual.ci_low,
        "residual_ci_high": None if residual is None else residual.ci_high,
        "residual_n": None if residual is None else residual.n_sampled,
        "residual_n_effective": (None if residual is None else residual.n_effective),
        "judge_over_rejection": (
            None if overrejection is None else overrejection.estimate
        ),
        "over_rej_ci_low": (None if overrejection is None else overrejection.ci_low),
        "over_rej_ci_high": (None if overrejection is None else overrejection.ci_high),
        "over_rej_n": None if overrejection is None else overrejection.n_sampled,
        "notes": notes,
    }


def _rate_cols(rate: RateInterval, *, prefix: str) -> dict:
    p = prefix
    return {
        f"{p}n_sampled" if p else "n_sampled": rate.n_sampled,
        f"{p}n_yes" if p else "n_yes": rate.n_yes,
        f"{p}n_population" if p else "n_population": rate.n_population,
        f"{p}n_effective" if p else "n_effective": rate.n_effective,
        f"{p}rate" if p else "rate": rate.estimate,
        f"{p}ci_low" if p else "ci_low": rate.ci_low,
        f"{p}ci_high" if p else "ci_high": rate.ci_high,
    }


def _rate_json(rate: RateInterval) -> dict:
    return {
        "n": rate.n_sampled,
        "n_yes": rate.n_yes,
        "rate": rate.estimate,
        "ci_low": rate.ci_low,
        "ci_high": rate.ci_high,
        "n_effective": rate.n_effective,
    }


def _model_ids(frames: Mapping[str, pd.DataFrame]) -> list[str]:
    ids: set[str] = set()
    for frame in frames.values():
        if "model_id" not in frame.columns:
            continue
        ids.update(str(v) for v in frame["model_id"].dropna().unique() if str(v))
    return sorted(ids)


def _plan_seed(plan: Mapping[str, object]) -> int:
    """Sampling seed the plan was built with.

    Raises:
        KeyError: `plan` predates the ``global_seed`` field. A default here would
            silently reshuffle the human-agreement sheet away from the plan.
    """
    if plan.get("global_seed") is None:
        raise KeyError("plan_manifest.json has no global_seed")
    return int(plan["global_seed"])


def _plan_git_sha(plan: Mapping[str, object]) -> object:
    return plan.get("git_sha")


def _plan_campaign_subdir(plan: Mapping[str, object]) -> str:
    """Campaign subdir the leaves were sampled from.

    Raises:
        KeyError: Field absent; without it relative ``run_dir`` keys cannot be
            resolved to leaves and every summary load would miss.
    """
    subdir = plan.get("campaign_subdir")
    if not subdir:
        raise KeyError("plan_manifest.json has no campaign_subdir")
    return str(subdir)


def _flagged_sample_coverage(plan: Mapping[str, object]) -> dict | None:
    table = plan.get("stratum_table")
    if not isinstance(table, Mapping):
        return None
    if "flagged_pop_total" not in table or "flagged_unsampled_pop" not in table:
        return None
    total = int(table["flagged_pop_total"])
    unsampled = int(table["flagged_unsampled_pop"])
    if "flagged_unsampled_fraction" in table:
        fraction = float(table["flagged_unsampled_fraction"])
    else:
        fraction = (unsampled / total) if total else 0.0
    return {
        "flagged_pop_total": total,
        "flagged_unsampled_pop": unsampled,
        "flagged_unsampled_fraction": fraction,
    }


def _plan_leaf_index(plan: Mapping[str, object]) -> dict[str, dict]:
    leaves = plan.get("leaves") or []
    return {str(leaf["run_dir"]): dict(leaf) for leaf in leaves}


def _clean_pop_in_quintile(
    leaf_plan: Mapping[str, Mapping[str, object]],
    run_dir: str,
    quintile: object,
) -> int:
    leaf = leaf_plan.get(run_dir)
    pops = None if leaf is None else leaf.get("clean_pop_by_quintile")
    if not pops:
        raise ValueError(
            f"{run_dir}: plan manifest has no clean_pop_by_quintile; cannot "
            "reweight leakage by step without per-quintile denominators"
        )
    index = int(quintile)
    if not 0 <= index < len(pops):
        raise ValueError(f"{run_dir}: step_quintile {index} outside {len(pops)} strata")
    return int(pops[index])


def _layer1_rows_in(
    plan: Mapping[str, object],
    restream: Mapping[str, object] | None,
) -> object:
    """Documents that entered the pool screen.

    The re-stream only records rejects, so the screened total is the census of
    pool rows kept plus those rejects.
    """
    if restream is None or restream.get("n_expected") is None:
        return None
    totals = plan.get("per_task_totals") or {}
    kept = totals.get("pool_open")
    if kept is None:
        return None
    return int(kept) + int(restream["n_expected"])


def _restream_n_rejected(restream: Mapping[str, object] | None) -> object:
    if restream is None:
        return None
    return restream.get("n_expected")


def _as_parse_ok(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    return parse_yes_no(value)


def _as_flagged(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1) and value == value:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "flagged"}:
            return True
        if token in {"false", "0", "no", "clean"}:
            return False
    raise ValueError(f"unparseable flagged value: {value!r}")


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {} ({} rows)", path, len(frame))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def _write_json(path: Path, payload: object, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    atomic_write_json(path, payload)
    return path
