"""Persist checkpoint-ladder result tables and C1-C9 claim verdicts (plan §8/§10).

`build_results` writes under ``{output_root}/{campaign_subdir}/results/`` the
`RESULTS_*` CSVs named in `campaign.ckpt_plan`, plus `claims.json`. The notebook
loads these and computes nothing -- every figure in plan §10 must have its inputs
here.

This module is wiring only: every threshold, frame definition and verdict rule
lives in `campaign.ckpt_metrics` (pure domain, unit-tested), which is what
`docs/PREREG_CKPT_LADDER.md` §6-§7 names. A missing upstream table yields an
empty frame and an ``insufficient_data`` claim rather than a fabricated verdict.

Two boundaries raise instead of degrading quietly:

- `require_min_ok_rungs` -- the preregistered analysis-time bound. It fires here,
  at the C1 fit, and not at P2, so a narrow admission still leaves every trained
  leaf and every C3 rung on disk.
- the claim-id check against `ckpt_plan.CLAIM_IDS` -- a `claims.json` that is
  missing a preregistered claim must not be written as if the claim had passed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.similarity_ladder_pipeline import merge_parts, merged_table
from subliminal_learning.campaign import ckpt_metrics as cm
from subliminal_learning.campaign import ckpt_plan as plan
from subliminal_learning.campaign.ckpt_pairs import ladder_pairs
from subliminal_learning.campaign.ckpt_selection import require_min_ok_rungs
from subliminal_learning.cka.pairing import canonical_label, split_revision
from subliminal_learning.cka.results_io import (
    JOIN_KEYS,
    revision_step,
    similarity_results_dir,
    with_delta_join_keys,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_preference_results,
    subliminal_delta,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

# Columns that identify the same leaf lineage a delta row does, so
# `perplexity_ratio` joins onto C5's response without dragging in the whole
# benchmark block.
_CAPABILITY_KEYS: tuple[str, ...] = (
    "animal",
    "model",
    "gkd_tag",
    "teacher_base_label",
    "teacher_cross_init",
)
# Keys a control leaf shares with its matched treatment (controls have no teacher
# lineage) -- `evaluation.gkd_results_io.subliminal_delta` uses the same set.
_CONTROL_KEYS: tuple[str, ...] = ("animal", "model", "gkd_tag", "set")


def _results_dir(output_root: Path, campaign_subdir: str) -> Path:
    return output_root / campaign_subdir / "results"


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {} ({} rows)", path, len(frame))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def merge_capability_parts(output_root: Path, run_set_id: str) -> Path | None:
    """Merge P1b's per-rung capability part CSVs into one long file.

    Single-writer by contract, mirroring `similarity_ladder_pipeline.merge_parts`:
    `scripts.run_base_capability_ladder` jobs for different rungs run
    concurrently across GPUs and each writes only its own
    ``results/parts/cap__*.csv``, never the merged file, so this is the only
    place the shards' rows are combined -- concurrent per-rung jobs can no
    longer race on a shared read-modify-write of the merged CSV.

    Args:
        output_root: Artifact root from the paths config.
        run_set_id: The geometry run-set id (`ckpt_plan.GEOM_RUN_SET_ID`).

    Returns:
        The merged CSV path, or ``None`` when no part exists yet.
    """
    out_dir = similarity_results_dir(output_root, run_set_id)
    parts = sorted((out_dir / "parts").glob("cap__*.csv"))
    if not parts:
        return None
    merged = pd.concat(
        [pd.read_csv(p) for p in parts], ignore_index=True
    ).drop_duplicates(subset=["init", "step", "task", "metric"], keep="last")
    out_csv = out_dir / plan.RESULTS_BASE_CAPABILITY_CSV
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(out_csv, merged.to_csv(index=False))
    return out_csv


def _read_optional(path: Path) -> pd.DataFrame:
    """A CSV an upstream inference stage may not have produced yet."""
    if not path.is_file():
        logger.warning("missing upstream table {} — its claim reads as insufficient", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_admission(output_root: Path, campaign_subdir: str) -> dict | None:
    """P2's `teacher_admission_report.json`, or None when it is absent."""
    path = output_root / campaign_subdir / "logs" / plan.TEACHER_ADMISSION_REPORT
    if not path.is_file():
        logger.warning(
            "no {} — the C1 fit cannot exclude quarantined rungs and the "
            "MIN_OK_RUNGS bound cannot be enforced",
            path,
        )
        return None
    return json.loads(path.read_text())


def _confounds(output_root: Path, campaign_subdir: str) -> pd.DataFrame:
    """Per-lineage ``perplexity_ratio`` for C5, from the student capability evals."""
    capability = load_gkd_capability_results(output_root, campaign_subdir)
    if capability.empty or "perplexity_ratio" not in capability.columns:
        logger.warning(
            "no student capability summaries under {} — C5's perplexity_ratio "
            "confound cannot be scored",
            campaign_subdir,
        )
        return pd.DataFrame()
    keys = [c for c in _CAPABILITY_KEYS if c in capability.columns]
    treatments = capability[~capability["is_control"].astype(bool)]
    return (
        treatments.groupby(keys, dropna=False)
        .agg(perplexity_ratio=("perplexity_ratio", "mean"))
        .reset_index()
    )


def _delta_frame(
    output_root: Path, campaign_subdir: str, pairs: tuple, *, with_confounds: bool = True
) -> pd.DataFrame:
    """Behavioural frame with the join keys, ``step``, ``y``, cell and family.

    Every prompt set is kept -- `ladder_delta.csv` is the preregistered
    robustness surface; only the claim adjudicators narrow to the primary
    endpoint (`ckpt_metrics.filter_primary`).
    """
    pref = load_gkd_preference_results(output_root, campaign_subdir)
    if pref.empty:
        return pd.DataFrame()
    delta = subliminal_delta(pref)
    if delta.empty:
        return pd.DataFrame()
    keyed = with_delta_join_keys(delta)
    keyed["step"] = [revision_step(r) for r in keyed["base_revision"]]
    out = cm.attach_y(keyed, pairs)
    if not with_confounds:
        return out
    confounds = _confounds(output_root, campaign_subdir)
    if confounds.empty:
        return out
    keys = [c for c in confounds.columns if c != "perplexity_ratio"]
    return out.merge(confounds, on=keys, how="left")


def _divergence_frame(similarity: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """``ladder_divergence.csv``: C1's table -- ``d_trait`` joined to ``y``.

    The geometry side is `ckpt_metrics.attach_d_trait`, i.e. the trait measure
    only. There is no fallback to ``1 - minibatch_cka``: a null on global CKA is
    itself preregistered (prereg §6), so substituting it would answer a different
    question under C1's name.
    """
    if delta.empty:
        return pd.DataFrame()
    geometry = cm.attach_d_trait(similarity)
    if geometry.empty:
        return pd.DataFrame()
    keyed = cm.with_init_labels(geometry).drop(columns=["step"])
    return delta.merge(keyed, on=list(JOIN_KEYS), how="inner")


def _stage_b_frame(output_root: Path, run_set_id: str) -> pd.DataFrame:
    """Stage B's deltas on the primary endpoint, keyed for C6's pairing join."""
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
    if not (Path(output_root) / subdir).is_dir():
        logger.warning("no Stage B run set at {} — C6 reads as insufficient", subdir)
        return pd.DataFrame()
    frame = _delta_frame(output_root, subdir, ladder_pairs(True, True), with_confounds=False)
    return cm.filter_primary(frame) if not frame.empty else frame


def _crossed_leaf_frame(
    output_root: Path, campaign_subdir: str, pairs: tuple
) -> pd.DataFrame:
    """Per-``(ftseed, gseed)`` deltas -- the unit the variance decomposition needs.

    `subliminal_delta` averages teacher ftseeds, which is exactly the factor the
    crossed anchors exist to separate, so this reads ``per_seed_p_target``
    (keyed by student ``gseed``) off the treatment and its matched control and
    pairs them gseed-for-gseed. Rows with no matched control gseed are dropped
    rather than differenced against a mean, which would fold gseed variance into
    the ftseed term.

    `pairs` is the ACTIVE pairing set, so ``pairing_index`` here means the same
    thing it does in `ckpt_plan.RESULTS_REPLICATION_MAP_CSV`.
    """
    pref = cm.filter_primary(load_gkd_preference_results(output_root, campaign_subdir))
    if pref.empty or "per_seed_p_target" not in pref.columns:
        return pd.DataFrame()
    controls: dict[tuple, dict[int, float]] = {}
    for _, row in pref[pref["is_control"].astype(bool)].iterrows():
        key = tuple(row[c] for c in _CONTROL_KEYS)
        controls.setdefault(key, {}).update(
            {int(s): float(v) for s, v in (row["per_seed_p_target"] or {}).items()}
        )
    lookup = cm.pairing_lookup(pairs)
    rows: list[dict] = []
    for _, row in pref[~pref["is_control"].astype(bool)].iterrows():
        control = controls.get(tuple(row[c] for c in _CONTROL_KEYS))
        if not control or pd.isna(row.get("teacher_ftseed")):
            continue
        student, revision = split_revision(str(row["model"]))
        # A same-init cell has no ``_tbase-`` suffix, so the label is absent
        # (NaN) rather than equal to the student's -- see `results_io._delta_keys`.
        teacher = row.get("teacher_base_label")
        teacher = student if not isinstance(teacher, str) or not teacher else teacher
        pair = lookup.get((canonical_label(student), canonical_label(teacher)))
        if pair is None:
            continue
        for gseed, value in (row["per_seed_p_target"] or {}).items():
            if int(gseed) not in control:
                continue
            rows.append(
                {
                    "animal": row["animal"],
                    "step": revision_step(revision),
                    "pairing_index": pair.index,
                    "ftseed": int(row["teacher_ftseed"]),
                    "gseed": int(gseed),
                    "subliminal_delta": float(value) - control[int(gseed)],
                }
            )
    return pd.DataFrame(rows)


def build_results(
    output_root: Path,
    campaign_subdir: str,
    *,
    dry_run: bool = False,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    stage_b_run_set_id: str = plan.STAGE_B_RUN_SET_ID,
) -> list[Path]:
    """Write every `RESULTS_*` artifact the notebook reads, plus `claims.json`.

    Args:
        output_root: Artifact root from the paths YAML.
        campaign_subdir: ``gkd_students/{RUN_SET_ID}``.
        dry_run: Log planned writes only.
        include_family_swap: The pairing flags P3 ran with. They set
            `LadderPair.index`, so the replication map and the C1 family labels
            must be derived from the same active set the leaves came from.
        include_cyclic_closure: As above.
        stage_b_run_set_id: Run set C6's concordance check joins against.

    Returns:
        Paths written (or that would be written).

    Raises:
        campaign.ckpt_selection.AdmissionTooNarrow: Fewer than
            `ckpt_plan.MIN_OK_RUNGS` rungs carry an admitted teacher -- the
            preregistered refusal to fit C1 on a collapsed dose range.
        ValueError: The assembled claim set is not `ckpt_plan.CLAIM_IDS`.
    """
    if not dry_run:
        merge_parts(output_root, plan.GEOM_RUN_SET_ID)
        merge_capability_parts(output_root, plan.GEOM_RUN_SET_ID)

    results = _results_dir(output_root, campaign_subdir)
    pairs = ladder_pairs(include_family_swap, include_cyclic_closure)

    similarity = merged_table(output_root, plan.GEOM_RUN_SET_ID, "similarity_pairs")
    self_drift = merged_table(output_root, plan.GEOM_RUN_SET_ID, "self_drift")
    probes = merged_table(output_root, plan.GEOM_RUN_SET_ID, "probe_grounding")
    weight_space = merged_table(output_root, plan.GEOM_RUN_SET_ID, "weight_space")
    capability = _read_optional(
        similarity_results_dir(output_root, plan.GEOM_RUN_SET_ID)
        / plan.RESULTS_BASE_CAPABILITY_CSV
    )

    delta = _delta_frame(output_root, campaign_subdir, pairs)
    divergence = _divergence_frame(similarity, delta)

    admission = _read_admission(output_root, campaign_subdir)
    admitted = tuple(int(r) for r in (admission or {}).get("ok_rungs", ()))
    quarantined = tuple(r for r in plan.LADDER_B if r not in set(admitted))
    if admission is not None:
        require_min_ok_rungs(admitted)

    primary_divergence = cm.filter_primary(divergence)
    primary_delta = cm.filter_primary(delta)
    c1_frame = cm.c1_fit_frame(primary_divergence, quarantine_rungs=quarantined)
    grading = cm.metric_grading_rows(similarity, probes)

    c1 = cm.adjudicate_c1(c1_frame)
    c2 = cm.adjudicate_c2(c1_frame)
    c3, emergence = cm.adjudicate_c3(primary_delta)
    c4, breakdown = cm.adjudicate_c4(similarity)
    claims = [
        c1,
        c2,
        c3,
        c4,
        cm.adjudicate_c5(primary_divergence),
        cm.adjudicate_c6(primary_delta, _stage_b_frame(output_root, stage_b_run_set_id)),
        cm.adjudicate_c7(self_drift, similarity),
        cm.adjudicate_c8(probes, grading),
        cm.adjudicate_c9(weight_space),
    ]
    if [c["claim"] for c in claims] != list(plan.CLAIM_IDS):
        raise ValueError(
            f"claims.json would carry {[c['claim'] for c in claims]}, not the "
            f"preregistered {list(plan.CLAIM_IDS)}"
        )

    tables = {
        plan.RESULTS_DELTA_CSV: delta,
        plan.RESULTS_DIVERGENCE_CSV: divergence,
        plan.RESULTS_DELTA_AGG_CSV: c1_frame,
        plan.RESULTS_SELF_DRIFT_CSV: self_drift,
        plan.RESULTS_BASE_CAPABILITY_CSV: capability,
        plan.RESULTS_PROBE_CSV: probes,
        plan.RESULTS_WEIGHT_SPACE_CSV: weight_space,
        plan.RESULTS_REGRESSION_CSV: cm.regression_rows(c1, c2),
        plan.RESULTS_EMERGENCE_CSV: emergence,
        plan.RESULTS_BASIS_BREAKDOWN_CSV: breakdown,
        plan.RESULTS_METRIC_GRADING_CSV: grading,
        plan.RESULTS_VARIANCE_COMPONENTS_CSV: cm.variance_component_rows(
            _crossed_leaf_frame(output_root, campaign_subdir, pairs)
        ),
        plan.RESULTS_QUARANTINE_CSV: cm.quarantine_rows(admission, delta),
        plan.RESULTS_REPLICATION_MAP_CSV: cm.replication_map_rows(
            include_family_swap=include_family_swap,
            include_cyclic_closure=include_cyclic_closure,
        ),
    }
    written = [
        _write_csv(results / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    ]

    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "geom_run_set_id": plan.GEOM_RUN_SET_ID,
        "stage_b_run_set_id": stage_b_run_set_id,
        "primary_endpoint": {
            "prompt_set": plan.PRIMARY_PROMPT_SET,
            "scoring": plan.PRIMARY_SCORING,
        },
        "pairing_flags": {
            "include_family_swap": include_family_swap,
            "include_cyclic_closure": include_cyclic_closure,
        },
        "n_pairings": len(pairs),
        "admitted_rungs": list(admitted),
        "quarantined_rungs": list(quarantined),
        "claims": claims,
    }
    claims_path = results / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, payload)
    written.append(claims_path)
    logger.info(
        "ckptladder results: {} artifacts under {}; C1={}",
        len(written),
        results,
        c1.get("status"),
    )
    return written
