"""Assemble OPD-limits leaves and persist results CSVs + ``claims.json`` per tier.

Frame schemas and claim rules live in ``campaign.opdlim_metrics``; this module
is the IO + assembly layer. Every summary row on disk is mapped to exactly one
planned `opdlim_pairs.Leaf` / `Control`; dose treatments reuse the P4 ladder's
control dirs, so those controls carry the ladder block. Tier 2 reads Tier 1
leaves read-only, and only when the cross-tier status is ``ok``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from loguru import logger

from pipelines.topk_results_pipeline import (
    _is_missing,
    _jsonable,
    _kind_of_row,
    _per_seed_items,
)
from subliminal_learning.campaign import opdlim_metrics as om
from subliminal_learning.campaign import topk_plan
from subliminal_learning.campaign.opdlim_pairs import (
    KIND_TB,
    KIND_TREATMENT,
    Control,
    Leaf,
    all_controls,
    all_leaves,
)
from subliminal_learning.campaign.opdlim_plan import (
    ANIMALS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CLAIM_IDS_T1,
    CLAIM_IDS_T2,
    CLAIM_IDS_T3,
    CROSS_TIER_STATUS,
    HUB_ORG,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    RESULTS_CLAIMS_JSON,
    RESULTS_FILES_COMMON,
    RESULTS_FILES_T1,
    RESULTS_FILES_T2,
    RESULTS_FILES_T3,
    TEACHER_P_TARGET_BY_TAG,
    TIER1,
    TIER2,
    Tier,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_forced_choice_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
)
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_xbase_control_run_dir,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

ROLLOUT_SUMMARY = "rollouts_summary.json"
X0_STATUSES: tuple[str, ...] = ("pending", "void_pin", "ok")

_PlanKey = tuple[str, str, str, str, str, int]


def campaign_subdir(t: Tier) -> str:
    """``gkd_students/{run_set_id}`` of one tier."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, t.run_set_id)


def results_dir(output_root: Path, t: Tier) -> Path:
    """Results directory of one tier."""
    return Path(output_root) / campaign_subdir(t) / "results"


def logs_dir(output_root: Path, t: Tier) -> Path:
    """Campaign logs directory of one tier."""
    return Path(output_root) / campaign_subdir(t) / "logs"


def cross_tier_status(output_root: Path) -> dict:
    """Tier 2 ``cross_tier_status.json`` (``pending`` when absent).

    Raises:
        ValueError: The status is not one of `X0_STATUSES`.
    """
    path = logs_dir(output_root, TIER2) / CROSS_TIER_STATUS
    if not path.is_file():
        return {"status": "pending"}
    doc = json.loads(path.read_text())
    if doc.get("status") not in X0_STATUSES:
        raise ValueError(f"{path}: status {doc.get('status')!r} not in {X0_STATUSES}")
    return doc


def _spec(token: str):
    return parse_base_models(token)[0]


def _model_dir(token: str) -> str:
    return model_dir_name(_spec(token), HUB_ORG)


def _plan_index(t: Tier) -> dict[_PlanKey, Leaf | Control]:
    """``(kind, animal, student dir, teacher tag | tb label | "", gkd_tag, gseed)``."""
    index: dict[_PlanKey, Leaf | Control] = {}
    for leaf in all_leaves(t):
        key = (
            KIND_TREATMENT,
            leaf.animal,
            _model_dir(leaf.pair.student),
            leaf.teacher_tag,
            leaf.gkd_tag,
            leaf.gseed,
        )
        index[key] = leaf
    for c in all_controls(t):
        ident = spec_label(_spec(c.pair.teacher)) if c.kind == KIND_TB else ""
        key = (c.kind, c.animal, _model_dir(c.pair.student), ident, c.gkd_tag, c.gseed)
        index[key] = c
    return index


def _config_key(row: Mapping) -> tuple[str, str, str, str, str]:
    kind = _kind_of_row(row)
    if kind == KIND_TREATMENT:
        ident = str(row["teacher_tag"])
    elif kind == KIND_TB:
        ident = str(row["teacher_base_label"])
    else:
        ident = ""
    return kind, str(row["animal"]), str(row["model"]), ident, str(row["gkd_tag"])


def _by_config(frame: pd.DataFrame) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for rec in frame.to_dict("records"):
        key = _config_key(rec)
        if key in out:
            raise ValueError(f"duplicate summary rows for config {key}")
        out[key] = rec
    return out


def _leaf_dir(output_root: Path, t: Tier, planned: Leaf | Control) -> Path:
    subdir = campaign_subdir(t)
    student = _spec(planned.pair.student)
    if isinstance(planned, Leaf):
        return gkd_student_run_dir(
            output_root,
            planned.animal,
            student,
            HUB_ORG,
            planned.teacher_tag,
            planned.gkd_tag,
            planned.gseed,
            subdir,
        )
    if planned.kind == KIND_TB:
        return gkd_xbase_control_run_dir(
            output_root,
            planned.animal,
            student,
            HUB_ORG,
            _spec(planned.pair.teacher),
            planned.gkd_tag,
            planned.gseed,
            subdir,
        )
    return gkd_control_run_dir(
        output_root,
        planned.animal,
        student,
        HUB_ORG,
        planned.gkd_tag,
        planned.gseed,
        subdir,
    )


def _screen_counts(leaf_dir: Path) -> dict:
    path = leaf_dir / ROLLOUT_SUMMARY
    doc = json.loads(path.read_text()) if path.is_file() else {}
    out: dict = {f: doc.get(f, float("nan")) for f in om.SCREEN_FIELDS}
    out["on_policy_only"] = doc.get("on_policy_only", float("nan"))
    return out


def _n_design(t: Tier, planned: Leaf | Control) -> float:
    if not isinstance(planned, Leaf):
        return float("nan")
    for pair, design in t.block(planned.block).cells:
        if pair == planned.pair:
            return float(design.n_treatments)
    raise ValueError(f"{t.token}/{planned.block}: no design for {planned.pair.name}")


def _num(value: object) -> float:
    return float("nan") if _is_missing(value) else float(value)


def _leaf_record(
    t: Tier,
    planned: Leaf | Control,
    gseed: int,
    p_target: float,
    pref: Mapping,
    fc_share: float,
    cap: Mapping,
    telemetry: Mapping,
    screen: Mapping,
) -> dict:
    obj, hp = planned.objective, planned.hp
    is_leaf = isinstance(planned, Leaf)
    ppl = dict(_per_seed_items(cap.get("per_seed_ppl"))).get(gseed, float("nan"))
    base_ppl = _num(cap.get("ppl_base_overall"))
    ece = dict(_per_seed_items(cap.get("calib_per_seed_ece"))).get(gseed, float("nan"))
    lmeval = {
        f"{om.LMEVAL_PREFIX}{col.removeprefix('delta_')}": _num(value)
        for col, value in cap.items()
        if str(col).startswith("delta_")
    }
    return {
        "tier": t.n,
        "run_set_id": t.run_set_id,
        "block": planned.block,
        "pair": planned.pair.name,
        "teacher_token": planned.pair.teacher,
        "student_token": planned.pair.student,
        "kind": KIND_TREATMENT if is_leaf else planned.kind,
        "objective": obj.slug,
        "form": obj.form,
        "k": obj.k_plot,
        "lmbda": obj.lmbda,
        "vmask_forms": obj.vmask_forms,
        "subset_n": float("nan") if obj.subset_n is None else obj.subset_n,
        "hp_lr": hp.lr,
        "hp_lora_r": hp.lora_r,
        "hp_lora_alpha": hp.lora_alpha,
        "teacher_param_tag": planned.teacher_param_tag if is_leaf else None,
        "teacher_p_target": (
            TEACHER_P_TARGET_BY_TAG[planned.teacher_param_tag]
            if is_leaf
            else float("nan")
        ),
        "gkd_tag": planned.gkd_tag,
        "n_design": _n_design(t, planned),
        "ftseed": planned.ftseed if is_leaf else float("nan"),
        "gseed": gseed,
        "p_target": p_target,
        "rank_mean": _num(pref.get("rank_mean")),
        "fc_share": fc_share,
        "perplexity_ratio_seed": (
            ppl / base_ppl if math.isfinite(ppl) and base_ppl > 0 else float("nan")
        ),
        "ce_nats_seed": math.log(ppl)
        if math.isfinite(ppl) and ppl > 0
        else float("nan"),
        "ce_nats_base": _num(cap.get("ppl_ce_nats_base")),
        "calib_ece_seed": ece,
        "calib_ece_base": _num(cap.get("calib_ece_base")),
        "calib_entropy": _num(cap.get("calib_entropy_mean")),
        "filter_drop": _num(telemetry.get("filter_drop")),
        "n_log_steps": _num(telemetry.get("n_log_steps")),
        **{f: _num(telemetry.get(f)) for f in om.TELEMETRY_FIELDS},
        **screen,
        "quarantined": False,
        "quarantine_reason": "",
        **lmeval,
    }


def _load(loader, output_root: Path, subdir: str) -> pd.DataFrame:
    return loader(output_root, subdir, animals=list(ANIMALS))


def assemble_leaf_frame(output_root: Path, t: Tier) -> pd.DataFrame:
    """Every evaluated leaf of a tier on disk → `opdlim_metrics.LEAF_COLUMNS` rows.

    Raises:
        ValueError: A summary lacks per-gseed scores, repeats a config, or a leaf
            on disk is not in the tier's plan.
    """
    empty = pd.DataFrame(columns=list(om.LEAF_COLUMNS))
    subdir = campaign_subdir(t)
    if not (Path(output_root) / subdir).is_dir():
        return empty
    pref = _load(load_gkd_preference_results, output_root, subdir)
    primary = pref.loc[
        (pref["set"].astype(str) == PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == PRIMARY_SCORING)
    ]
    if primary.empty:
        return empty
    fc = _load(load_gkd_forced_choice_results, output_root, subdir)
    fc_map = _by_config(fc.loc[fc["is_target"].fillna(False).astype(bool)])
    cap = _load(load_gkd_capability_results, output_root, subdir)
    cap_map = _by_config(cap) if not cap.empty else {}
    tele = om.telemetry_frame(_load(load_gkd_training_logs, output_root, subdir))
    tele_map = {
        (*_config_key(rec), int(rec["gseed"])): rec for rec in tele.to_dict("records")
    }
    index = _plan_index(t)
    rows: list[dict] = []
    for rec in primary.to_dict("records"):
        key = _config_key(rec)
        seeds = _per_seed_items(rec.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                f"{t.token}: preference summary missing per_seed_p_target {key}"
            )
        fc_seeds = dict(
            _per_seed_items(fc_map.get(key, {}).get("per_seed_target_share"))
        )
        for gseed, p_target in seeds:
            planned = index.get((*key, gseed))
            if planned is None:
                raise ValueError(
                    f"{t.token}: leaf on disk not in the plan: {key} gseed{gseed}"
                )
            rows.append(
                _leaf_record(
                    t,
                    planned,
                    gseed,
                    p_target,
                    rec,
                    fc_seeds.get(gseed, float("nan")),
                    cap_map.get(key, {}),
                    tele_map.get((*key, gseed), {}),
                    _screen_counts(_leaf_dir(output_root, t, planned)),
                )
            )
    frame = pd.DataFrame(rows)
    extra = om.lmeval_columns(frame)
    return frame.reindex(columns=[*om.LEAF_COLUMNS, *extra])


def _topk_means(output_root: Path) -> dict[int, float] | None:
    path = (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, topk_plan.RUN_SET_ID)
        / "results"
        / topk_plan.RESULTS_CELLS_CSV
    )
    if not path.is_file():
        logger.warning("topk cells missing at {}; C7 Spearman is null", path)
        return None
    return om.topk_p1_means(pd.read_csv(path))


def _tier_frames(output_root: Path, t: Tier, *, n_boot: int, seed: int):
    leaves = om.quarantine(assemble_leaf_frame(output_root, t))
    deltas = om.leaf_deltas(leaves)
    cells = om.cell_means(deltas, n_boot=n_boot, seed=seed)
    return leaves, deltas, cells


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def build_results(
    output_root: Path,
    t: Tier,
    *,
    dry_run: bool = False,
    bootstrap_n: int = BOOTSTRAP_N,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Write every results CSV of tier `t` and its ``claims.json``.

    Raises:
        KeyError: A live treatment has no live matched control.
        ValueError: A leaf on disk is outside the plan, or the claim keys drift.
    """
    boot = {"n_boot": bootstrap_n, "seed": bootstrap_seed}
    leaves, deltas, cells = _tier_frames(output_root, t, **boot)
    curve = om.k_curve(cells)
    kstar = om.kstar_frame(curve)
    capability = om.capability_rows(deltas)
    tables: dict[str, pd.DataFrame] = {
        "opdlim_leaves.csv": leaves,
        "opdlim_delta.csv": deltas,
        "opdlim_cells.csv": cells,
        "opdlim_curve.csv": curve,
        "opdlim_kstar.csv": kstar,
        "opdlim_capability.csv": capability,
        "opdlim_telemetry.csv": om.telemetry_cells(leaves),
        "opdlim_quarantine.csv": om.quarantine_rows(leaves),
        "opdlim_coverage.csv": om.coverage(leaves, t),
    }
    if t.n == TIER1.n:
        rho = om.ratio_rows(deltas, **boot)
        direction = om.direction_rows(deltas, **boot)
        hp = om.hp_rows(
            deltas,
            deltas,
            om.hp_blocks(TIER1),
            tier=TIER1.n,
            pending=om.hp_blocks(TIER2),
            **boot,
        )
        dose = om.dose_rows(deltas, t)
        tables.update(
            {
                "opdlim_rho.csv": rho,
                "opdlim_direction.csv": direction,
                "opdlim_hp.csv": hp,
                "opdlim_dose.csv": dose,
            }
        )
        claims = om.claims_t1(
            cells=cells,
            curve=curve,
            kstar=kstar,
            rho=rho,
            direction=direction,
            hp=hp,
            dose=dose,
            capability=capability,
            topk_means=_topk_means(output_root),
        )
        claim_ids, files = CLAIM_IDS_T1, RESULTS_FILES_T1
    elif t.n == 3:
        screen = om.screen_rows(leaves)
        data_curve = om.data_curve(cells, lmbda=0.0)
        slope = om.data_slope(deltas, tier=t, lmbda=0.0, **boot)
        tables.update(
            {
                "opdlim_screen.csv": screen,
                "opdlim_data_curve.csv": data_curve,
                "opdlim_data_slope.csv": slope,
            }
        )
        claims = om.claims_t3(slope=slope, curve=data_curve, screen=screen)
        claim_ids, files = CLAIM_IDS_T3, RESULTS_FILES_T3
    else:
        x0 = cross_tier_status(output_root)
        screen = om.screen_rows(leaves)
        if x0["status"] == "ok":
            _, t1_deltas, t1_cells = _tier_frames(output_root, TIER1, **boot)
            ghost = om.ghost_vs_ce_rows(deltas, t1_deltas, **boot)
            vmask = om.vmask_rows(deltas, t1_deltas, t1_cells, **boot)
            lam = om.lambda_rows(deltas, t1_deltas, **boot)
            concordance = om.data_concordance(deltas, t1_deltas, **boot)
            hp = om.hp_rows(
                deltas, t1_deltas, om.hp_blocks(TIER2), tier=TIER2.n, **boot
            )
        else:
            logger.info(
                "cross-tier status {}: Tier 1 contrasts not computed", x0["status"]
            )
            ghost = pd.DataFrame(columns=list(om.GHOST_COLUMNS))
            vmask = pd.DataFrame(columns=list(om.VMASK_COLUMNS))
            lam = pd.DataFrame(columns=list(om.LAMBDA_COLUMNS))
            concordance = pd.DataFrame(columns=list(om.CONCORDANCE_COLUMNS))
            hp = pd.DataFrame(columns=list(om.HP_COLUMNS))
        slope = om.data_slope(deltas, **boot)
        tables.update(
            {
                "opdlim_hp.csv": hp,
                "opdlim_ghost_vs_ce.csv": ghost,
                "opdlim_vmask.csv": vmask,
                "opdlim_lambda.csv": lam,
                "opdlim_screen.csv": screen,
                "opdlim_data_curve.csv": om.data_curve(cells),
                "opdlim_data_slope.csv": slope,
                "opdlim_data_concordance.csv": concordance,
            }
        )
        claims = om.claims_t2(
            x0=x0,
            curve=curve,
            kstar=kstar,
            ghost=ghost,
            vmask=vmask,
            lam=lam,
            screen=screen,
            concordance=concordance,
            slope=slope,
            n2=om.n2_rows(deltas, **boot),
            hp=hp,
        )
        claim_ids, files = CLAIM_IDS_T2, RESULTS_FILES_T2
    if tuple(claims) != claim_ids:
        raise ValueError(f"claims.json would carry {tuple(claims)}, not {claim_ids}")
    if set(tables) != {*RESULTS_FILES_COMMON, *files}:
        raise ValueError(f"{t.token} results tables drifted: {sorted(tables)}")
    dest = results_dir(output_root, t)
    written = {
        name: _write_csv(dest / name, tables[name], dry_run=dry_run)
        for name in (*RESULTS_FILES_COMMON, *files)
    }
    claims_path = dest / RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        atomic_write_json(claims_path, _jsonable(claims))
    written[RESULTS_CLAIMS_JSON] = claims_path
    logger.info("{} results: {} artifacts under {}", t.token, len(written), dest)
    return written
