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
from subliminal_learning.gkd.vocab_mask_list_io import VOCAB_MASK_LISTS_DIR, ids_sha256
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

_ConfigKey = tuple[str, str, str, str, str]
_K_LADDER_P1 = tuple(metrics.FULL_K if k is None else k for k in plan.K_LADDER)
_K_LADDER_ALL = (*_K_LADDER_P1, *plan.K_LADDER_P3)


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


def check_mask_identity(output_root: Path) -> list[str]:
    """Leaves whose manifest ids disagree with the planned list, as messages."""
    bad: list[str] = []
    for unit in plan.units():
        if unit.setting.slug is None:
            continue
        doc = json.loads(
            (VOCAB_MASK_LISTS_DIR / f"{unit.setting.slug}.json").read_text()
        )
        for gseed in unit.gseeds:
            path = jobs.leaf_dir(output_root, unit, gseed) / "train_manifest.json"
            if not path.is_file():
                continue
            ids = json.loads(path.read_text()).get("vocab_mask_ids", [])
            if ids_sha256(ids) != doc["ids_sha256"]:
                bad.append(f"{path}: mask ids differ from {unit.setting.slug}")
    return bad


def coverage(output_root: Path) -> dict:
    """Planned against finished leaves per priority."""
    out: dict = {}
    for priority in (1, 3):
        rows = [u for u in plan.units() if u.priority == priority]
        planned = [(u, g) for u in rows for g in u.gseeds]
        missing = [
            f"{u.slug}|g{g}"
            for u, g in planned
            if not (jobs.leaf_dir(output_root, u, g) / "train_manifest.json").is_file()
        ]
        out[f"p{priority}"] = {
            "planned": len(planned),
            "trained": len(planned) - len(missing),
            "missing": missing[:50],
        }
    return out


def _claims(frame: pd.DataFrame) -> dict:
    p1 = frame.loc[frame["priority"] == 1]
    table = metrics.cells(p1)
    every = metrics.cells(frame)
    return {
        "generated_at": utc_now_iso(),
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
                    arm: metrics.e10_extra(every, pair, arm)
                    for arm in (plan.RAND2, plan.RANDB)
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


def build_results(output_root: Path, *, dry_run: bool = False) -> dict[str, Path]:
    """Write every result file (atomic); returns ``{name: path}``.

    Raises:
        ValueError: A trained leaf's mask ids differ from its planned list.
    """
    mismatched = check_mask_identity(output_root)
    if mismatched:
        raise ValueError(f"mask identity failed: {mismatched[:3]}")
    leaves = assemble_leaves(output_root)
    if leaves.empty:
        logger.info("no evaluated leaves yet")
        return {}
    leaves = metrics.quarantine(leaves)
    frame = metrics.cell_frame(metrics.cells(leaves))
    claims = _claims(leaves)
    if dry_run:
        return {}
    out = results_dir(output_root)
    out.mkdir(parents=True, exist_ok=True)
    e10 = frame.loc[frame["experiment"] == plan.E10]
    e11 = frame.loc[frame["experiment"] == plan.E11]
    written = {
        "leaves": _write_csv(leaves, out / "leaves.csv"),
        "e10_arms": _write_csv(e10, out / "e10_arms.csv"),
        "e11_curve": _write_csv(e11, out / "e11_curve.csv"),
        "base_check": _write_csv(metrics.base_check(leaves), out / "base_check.csv"),
    }
    atomic_write_json(out / "coverage.json", coverage(output_root))
    atomic_write_json(out / "claims.json", _jsonable(claims))
    written["coverage"] = out / "coverage.json"
    written["claims"] = out / "claims.json"
    return written
