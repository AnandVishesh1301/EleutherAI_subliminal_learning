"""Assemble rebtopk leaves and persist result tables + ``claims.json``.

Reads the run set's eval summaries and training logs; every leaf on disk must map to a
planned `rebtopk_plan.Unit`. Unfinished leaves are holes, listed in ``coverage.json``.
Claims use priority-1 rows; priority-3 rows extend the tables and the ``priority3`` block.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.opdlim_results_pipeline import _by_config, _config_key, _num
from pipelines.topk_results_pipeline import _jsonable, _per_seed_items
from subliminal_learning.campaign import opdlim_metrics as om
from subliminal_learning.campaign import rebtopk_jobs as jobs
from subliminal_learning.campaign import rebtopk_metrics as metrics
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign.opdlim_pairs import KIND_TREATMENT
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_forced_choice_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.gkd.vocab_mask_list_io import ids_sha256
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

_ConfigKey = tuple[str, str, str, str, str]
LATEST_SNAPSHOT, P1_SNAPSHOT, FINAL_SNAPSHOT = "latest", "p1", "final"
_K_LADDER_P1 = tuple(metrics.FULL_K if k is None else k for k in plan.K_LADDER)
_K_LADDER_ALL = (*_K_LADDER_P1, *plan.K_LADDER_P3)


class ResultsIncompleteError(RuntimeError):
    """A priority's planned leaves are not all trained and evaluated."""


def _subdir() -> str:
    return run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)


def results_dir(output_root: Path) -> Path:
    """Results directory of the run set."""
    return plan.run_set_root(output_root) / "results"


def _plan_index() -> dict[_ConfigKey, plan.Unit]:
    return {
        (
            KIND_TREATMENT,
            u.animal,
            plan.model_dir(u.pair.student),
            u.teacher_tag,
            u.gkd_tag,
        ): u
        for u in plan.units()
    }


def _load(loader, output_root: Path) -> pd.DataFrame:
    return loader(output_root, _subdir(), animals=list(plan.ANIMALS))


def _leaf_row(
    unit: plan.Unit,
    gseed: int,
    p_target: float,
    pref: Mapping,
    fc_share: float,
    cap: Mapping,
    telemetry: Mapping,
) -> dict:
    ppl = dict(_per_seed_items(cap.get("per_seed_ppl"))).get(gseed, math.nan)
    base_ppl = _num(cap.get("ppl_base_overall"))
    return {
        "experiment": unit.experiment,
        "priority": unit.priority,
        "animal": unit.animal,
        "pair": unit.pair.name,
        "k": metrics.FULL_K if unit.setting.k is None else unit.setting.k,
        "arm": unit.setting.arm,
        "list": unit.setting.slug or "",
        "kind": unit.kind,
        "ftseed": unit.ftseed,
        "gseed": gseed,
        "gkd_tag": unit.gkd_tag,
        "teacher_tag": unit.teacher_tag,
        "p_target": p_target,
        "base_p_target": _num(pref.get("base_p_target")),
        "rank_mean": _num(pref.get("rank_mean")),
        "fc_share": fc_share,
        "ppl_ratio": ppl / base_ppl if base_ppl > 0 else math.nan,
        "ce_nats": math.log(ppl) if ppl > 0 else math.nan,
        "ce_nats_base": _num(cap.get("ppl_ce_nats_base")),
        "calib_ece_base": _num(cap.get("calib_ece_base")),
        "filter_drop": _num(telemetry.get("filter_drop")),
        **{f: _num(telemetry.get(f)) for f in om.TELEMETRY_FIELDS},
    }


def assemble_leaves(output_root: Path) -> pd.DataFrame:
    """Every evaluated leaf on disk as a frame.

    Raises:
        ValueError: A summary without per-seed scores, or a leaf outside the plan.
    """
    if not plan.run_set_root(output_root).is_dir():
        return pd.DataFrame()
    pref = _load(load_gkd_preference_results, output_root)
    if pref.empty:
        return pd.DataFrame()
    primary = pref.loc[
        (pref["set"].astype(str) == plan.PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
    ]
    fc = _load(load_gkd_forced_choice_results, output_root)
    fc_map = _by_config(fc.loc[fc["is_target"].fillna(False).astype(bool)])
    cap = _load(load_gkd_capability_results, output_root)
    cap_map = _by_config(cap) if not cap.empty else {}
    tele = om.telemetry_frame(_load(load_gkd_training_logs, output_root))
    tele_map = {(*_config_key(r), int(r["gseed"])): r for r in tele.to_dict("records")}
    index = _plan_index()
    rows: list[dict] = []
    for rec in primary.to_dict("records"):
        key = _config_key(rec)
        unit = index.get(key)
        if unit is None:
            raise ValueError(f"leaf on disk is not in the plan: {key}")
        seeds = _per_seed_items(rec.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(f"preference summary lacks per_seed_p_target: {key}")
        fc_seeds = dict(
            _per_seed_items(fc_map.get(key, {}).get("per_seed_target_share"))
        )
        for gseed, p_target in seeds:
            if gseed not in unit.gseeds:
                raise ValueError(f"gseed {gseed} is not planned for {unit.slug}")
            rows.append(
                _leaf_row(
                    unit,
                    gseed,
                    p_target,
                    rec,
                    fc_seeds.get(gseed, math.nan),
                    cap_map.get(key, {}),
                    tele_map.get((*key, gseed), {}),
                )
            )
    return pd.DataFrame(rows)


def _planned_mask_sha(unit: plan.Unit) -> str | None:
    if unit.setting.arm == plan.NONE:
        return None
    if unit.setting.arm == plan.ANIMAL:
        return plan.ANIMAL_IDS_SHA256
    return plan.TOKEN_LIST_SHA256[unit.setting.slug]


def check_mask_identity(output_root: Path) -> list[str]:
    """Trained leaves whose manifest ids differ from the plan's sha, as messages."""
    bad: list[str] = []
    for unit in plan.units():
        want = _planned_mask_sha(unit)
        if want is None:
            continue
        for gseed in unit.gseeds:
            path = jobs.leaf_dir(output_root, unit, gseed) / "train_manifest.json"
            if not path.is_file():
                continue
            ids = json.loads(path.read_text()).get("vocab_mask_ids", [])
            if ids_sha256(ids) != want:
                bad.append(f"{path}: mask ids differ from {unit.setting.name}")
    return bad


def coverage(output_root: Path, leaves: pd.DataFrame) -> dict:
    """Planned, trained and evaluated leaves per priority."""
    seen = (
        set()
        if leaves.empty
        else {
            (r["animal"], r["pair"], r["gkd_tag"], r["teacher_tag"], int(r["gseed"]))
            for r in leaves.to_dict("records")
        }
    )
    out: dict = {}
    for priority in (1, 3):
        planned = [
            (u, g) for u in plan.units() if u.priority == priority for g in u.gseeds
        ]
        untrained = [
            f"{u.slug}|g{g}"
            for u, g in planned
            if not (jobs.leaf_dir(output_root, u, g) / "train_manifest.json").is_file()
        ]
        unscored = [
            f"{u.slug}|g{g}"
            for u, g in planned
            if (u.animal, u.pair.name, u.gkd_tag, u.teacher_tag, g) not in seen
        ]
        out[f"p{priority}"] = {
            "planned": len(planned),
            "trained": len(planned) - len(untrained),
            "evaluated": len(planned) - len(unscored),
            "untrained": untrained[:50],
            "unevaluated": unscored[:50],
        }
    return out


def _claims(frame: pd.DataFrame, output_root: Path) -> dict:
    p1 = frame.loc[frame["priority"] == 1]
    table = metrics.cells(p1)
    every = metrics.cells(frame)
    return {
        "generated_at": utc_now_iso(),
        "code_sha": (read_run_set_card(plan.run_set_root(output_root)) or {}).get(
            "git_sha"
        ),
        "token_lists": dict(plan.TOKEN_LIST_SHA256),
        "animal_ids_sha256": plan.ANIMAL_IDS_SHA256,
        "screen": plan.SCREEN,
        "e10": {
            pair: {
                "all": metrics.e10_claims(table, pair),
                metrics.FIG3_DESIGN: metrics.e10_claims(
                    table, pair, metrics.FIG3_DESIGN
                ),
            }
            for pair in ("P3", "P4")
        },
        "e11": {
            animal: metrics.e11_claims(table, animal, _K_LADDER_P1)
            for animal in (plan.LION, plan.EAGLE)
        },
        "priority3": {
            "e10": {
                pair: {
                    plan.RAND2: metrics.e10_extra(every, pair, plan.RAND2, plan.RAND),
                    plan.RANDB: metrics.e10_extra(every, pair, plan.RANDB, plan.BOOST),
                }
                for pair in ("P3", "P4")
            },
            "e11": {
                animal: metrics.e11_claims(every, animal, _K_LADDER_ALL)
                for animal in (plan.LION, plan.EAGLE)
            },
        },
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def build_results(
    output_root: Path,
    *,
    snapshot: str = LATEST_SNAPSHOT,
    required: tuple[int, ...] = (),
    dry_run: bool = False,
) -> dict[str, Path]:
    """Write every result file under ``results/{snapshot}/``; returns ``{name: path}``.

    Args:
        output_root: Campaign output root.
        snapshot: Result subdirectory, so a priority's tables are never replaced by a
            later build.
        required: Priorities every planned leaf of which must be trained and evaluated.
        dry_run: Compute everything, write nothing.

    Raises:
        ValueError: A trained leaf's mask ids differ from the plan.
        ResultsIncompleteError: A `required` priority has unevaluated leaves.
    """
    mismatched = check_mask_identity(output_root)
    if mismatched:
        raise ValueError(f"mask identity failed: {mismatched[:3]}")
    leaves = assemble_leaves(output_root)
    covered = coverage(output_root, leaves)
    for priority in required:
        row = covered[f"p{priority}"]
        if not row["planned"] == row["trained"] == row["evaluated"]:
            raise ResultsIncompleteError(f"priority {priority}: {row}")
    if leaves.empty:
        logger.info("no evaluated leaves yet")
        return {}
    leaves = metrics.quarantine(leaves)
    frame = metrics.cell_frame(metrics.cells(leaves))
    claims = _claims(leaves, output_root)
    if dry_run:
        return {}
    out = results_dir(output_root) / snapshot
    out.mkdir(parents=True, exist_ok=True)
    e10 = frame.loc[frame["experiment"] == plan.E10]
    e11 = frame.loc[frame["experiment"] == plan.E11]
    written = {
        "leaves": _write_csv(leaves, out / "leaves.csv"),
        "e10_arms": _write_csv(e10, out / "e10_arms.csv"),
        "e11_curve": _write_csv(e11, out / "e11_curve.csv"),
        "base_check": _write_csv(metrics.base_check(leaves), out / "base_check.csv"),
    }
    atomic_write_json(out / "coverage.json", covered)
    atomic_write_json(out / "claims.json", _jsonable(claims))
    written["coverage"] = out / "coverage.json"
    written["claims"] = out / "claims.json"
    return written
