"""Assemble one domtopk run set's leaves and persist result tables + ``claims.json``.

Reads the run set's eval summaries and training logs; every leaf on disk must map to
a planned `domtopk_plan.Unit`. Unfinished leaves are holes, listed per priority in
``coverage.json``. Output under ``results/{snapshot}/`` (``p1`` once priority 1 is
done, ``final`` once both are): ``leaves.csv``, ``curve.csv`` (one row per target x
K x arm), ``base_check.csv``, ``coverage.json`` and ``claims.json`` (one block per
target, pending until its cells are complete).
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
from subliminal_learning.campaign import domtopk_jobs as jobs
from subliminal_learning.campaign import domtopk_metrics as metrics
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import opdlim_metrics as om
from subliminal_learning.campaign import rebtopk_metrics as rm
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


class ResultsIncompleteError(RuntimeError):
    """A priority's planned leaves are not all trained and evaluated."""


def results_dir(output_root: Path, domain: str) -> Path:
    """Results directory of `domain`'s run set."""
    return plan.run_set_root(output_root, domain) / "results"


def _plan_index(domain: str) -> dict[_ConfigKey, plan.Unit]:
    return {
        (
            KIND_TREATMENT,
            u.target,
            plan.model_dir(plan.PAIR.student),
            u.teacher_tag,
            u.gkd_tag,
        ): u
        for u in plan.units(domain)
    }


def _load(loader, output_root: Path, domain: str) -> pd.DataFrame:
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_IDS[domain])
    return loader(output_root, subdir, animals=list(plan.TARGETS_OF[domain]))


def _leaf_row(
    unit: plan.Unit,
    p_target: float,
    pref: Mapping,
    fc_share: float,
    cap: Mapping,
    telemetry: Mapping,
) -> dict:
    ppl = dict(_per_seed_items(cap.get("per_seed_ppl"))).get(unit.gseed, math.nan)
    base_ppl = _num(cap.get("ppl_base_overall"))
    return {
        "experiment": unit.domain,
        "priority": unit.priority,
        "animal": unit.target,
        "pair": plan.PAIR.name,
        "k": metrics.k_value(unit.setting.k),
        "arm": unit.setting.arm,
        "kind": unit.kind,
        "ftseed": unit.ftseed,
        "gseed": unit.gseed,
        "gkd_tag": unit.gkd_tag,
        "teacher_tag": unit.teacher_tag,
        "p_target": p_target,
        "base_p_target": _num(pref.get("base_p_target")),
        "rank_mean": _num(pref.get("rank_mean")),
        "fc_share": fc_share,
        "ppl_ratio": ppl / base_ppl if base_ppl > 0 else math.nan,
        "ce_nats": math.log(ppl) if ppl > 0 else math.nan,
        "ce_nats_base": _num(cap.get("ppl_ce_nats_base")),
        "filter_drop": _num(telemetry.get("filter_drop")),
        **{f: _num(telemetry.get(f)) for f in om.TELEMETRY_FIELDS},
    }


def assemble_leaves(output_root: Path, domain: str) -> pd.DataFrame:
    """Every evaluated leaf of `domain`'s run set on disk as a frame.

    Raises:
        ValueError: A summary without per-seed scores, or a leaf outside the plan.
    """
    if not plan.run_set_root(output_root, domain).is_dir():
        return pd.DataFrame()
    pref = _load(load_gkd_preference_results, output_root, domain)
    if pref.empty:
        return pd.DataFrame()
    primary = pref.loc[
        (pref["set"].astype(str) == plan.primary_prompt_set(domain))
        & (pref["scoring"].astype(str) == plan.PRIMARY_SCORING)
    ]
    fc = _load(load_gkd_forced_choice_results, output_root, domain)
    fc_map = _by_config(fc.loc[fc["is_target"].fillna(False).astype(bool)])
    cap = _load(load_gkd_capability_results, output_root, domain)
    cap_map = _by_config(cap) if not cap.empty else {}
    tele = om.telemetry_frame(_load(load_gkd_training_logs, output_root, domain))
    tele_map = {(*_config_key(r), int(r["gseed"])): r for r in tele.to_dict("records")}
    index = _plan_index(domain)
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
            if gseed != unit.gseed:
                raise ValueError(f"gseed {gseed} is not planned for {unit.slug}")
            rows.append(
                _leaf_row(
                    unit,
                    p_target,
                    rec,
                    fc_seeds.get(gseed, math.nan),
                    cap_map.get(key, {}),
                    tele_map.get((*key, gseed), {}),
                )
            )
    return pd.DataFrame(rows)


def check_mask_identity(output_root: Path, domain: str) -> list[str]:
    """Trained panel leaves whose manifest ids differ from the plan's sha."""
    bad: list[str] = []
    for unit in plan.units(domain):
        if unit.setting.arm != plan.PANEL:
            continue
        path = jobs.leaf_dir(output_root, unit) / "train_manifest.json"
        if not path.is_file():
            continue
        ids = json.loads(path.read_text()).get("vocab_mask_ids", [])
        if ids_sha256(ids) != plan.PANEL_IDS_SHA256[unit.target]:
            bad.append(f"{path}: mask ids differ from the {unit.target} panel")
    return bad


def coverage(output_root: Path, domain: str, leaves: pd.DataFrame) -> dict:
    """Planned, trained and scored leaves of `domain`'s run set, per priority.

    A leaf counts as evaluated only with both ``p_target`` and ``fc_share``.
    """
    seen = (
        set()
        if leaves.empty
        else {
            (r["animal"], r["gkd_tag"], r["teacher_tag"], int(r["gseed"]))
            for r in metrics.scored(leaves).to_dict("records")
        }
    )
    out: dict = {}
    for priority in plan.PRIORITIES:
        planned = plan.units(domain, priority)
        untrained = [
            f"{u.slug}|g{u.gseed}"
            for u in planned
            if not (jobs.leaf_dir(output_root, u) / "train_manifest.json").is_file()
        ]
        unscored = [
            f"{u.slug}|g{u.gseed}"
            for u in planned
            if (u.target, u.gkd_tag, u.teacher_tag, u.gseed) not in seen
        ]
        out[f"p{priority}"] = {
            "planned": len(planned),
            "trained": len(planned) - len(untrained),
            "evaluated": len(planned) - len(unscored),
            "untrained": untrained,
            "unevaluated": unscored,
        }
    return out


def _claims(table: dict[tuple, rm.Cell], output_root: Path, domain: str) -> dict:
    targets = plan.TARGETS_OF[domain]
    card = read_run_set_card(plan.run_set_root(output_root, domain)) or {}
    return {
        "generated_at": utc_now_iso(),
        "code_sha": card.get("git_sha"),
        "run_set_id": plan.RUN_SET_IDS[domain],
        "primary_prompt_set": plan.primary_prompt_set(domain),
        "panel": list(plan.PANELS[domain]),
        "priority": {t: plan.PRIORITY_OF[t] for t in targets},
        "screen": {t: plan.TARGET_SCREEN[t] for t in targets},
        "panel_ids_sha256": {t: plan.PANEL_IDS_SHA256[t] for t in targets},
        **{t: metrics.target_claims(table, t) for t in targets},
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> Path:
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def build_results(
    output_root: Path,
    domain: str,
    *,
    snapshot: str = LATEST_SNAPSHOT,
    required: tuple[int, ...] = (),
    dry_run: bool = False,
) -> dict[str, Path]:
    """Write every result file under ``results/{snapshot}/``; returns ``{name: path}``.

    Args:
        output_root: Campaign output root.
        domain: Run set to build.
        snapshot: Result subdirectory, so a priority's tables are never replaced by
            a later build.
        required: Priorities every planned leaf of which must be trained and
            evaluated.
        dry_run: Compute everything, write nothing.

    Raises:
        ValueError: A trained panel leaf's mask ids differ from the plan.
        ResultsIncompleteError: A `required` priority has unevaluated leaves.
    """
    mismatched = check_mask_identity(output_root, domain)
    if mismatched:
        raise ValueError(f"mask identity failed: {mismatched[:3]}")
    leaves = assemble_leaves(output_root, domain)
    covered = coverage(output_root, domain, leaves)
    for priority in required:
        row = covered[f"p{priority}"]
        if not row["planned"] == row["trained"] == row["evaluated"]:
            raise ResultsIncompleteError(
                f"{domain} priority {priority}: planned {row['planned']}, trained"
                f" {row['trained']}, evaluated {row['evaluated']}"
            )
    if leaves.empty:
        logger.info("{}: no evaluated leaves yet", domain)
        return {}
    leaves = rm.quarantine(leaves)
    table = metrics.cells(leaves)
    curve = rm.cell_frame(table)
    claims = _claims(table, output_root, domain)
    if dry_run:
        return {}
    out = results_dir(output_root, domain) / snapshot
    out.mkdir(parents=True, exist_ok=True)
    written = {
        "leaves": _write_csv(leaves, out / "leaves.csv"),
        "curve": _write_csv(curve, out / "curve.csv"),
        "base_check": _write_csv(rm.base_check(leaves), out / "base_check.csv"),
    }
    atomic_write_json(out / "coverage.json", covered)
    atomic_write_json(out / "claims.json", _jsonable(claims))
    written["coverage"] = out / "coverage.json"
    written["claims"] = out / "claims.json"
    return written
