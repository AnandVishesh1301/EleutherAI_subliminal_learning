"""Assemble xsize frames and persist X0–X4 claim artifacts.

Leaf / cell / claim schemas come from ``campaign.xsize_metrics``. This
module is the IO + assembly layer: it does not re-implement a statistic.
Claim keys must be exactly `xsize_plan.CLAIM_IDS` or this module raises.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.campaign import xsize_metrics as xm
from subliminal_learning.campaign.xsize_plan import (
    ANIMALS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CLAIM_IDS,
    GRID_REFERENCE,
    LR_LADDER,
    LR_ROBUSTNESS_STUDENTS,
    ONE_STEP_PAIRS,
    P0_GATES_REPORT,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    RESULTS_ADDITIVE_FIT_CSV,
    RESULTS_ASYMMETRY_CSV,
    RESULTS_BASE_PRIORS_CSV,
    RESULTS_CELLS_CSV,
    RESULTS_CLAIMS_JSON,
    RESULTS_DELTA_CSV,
    RESULTS_GRID_CONCORDANCE_CSV,
    RESULTS_LEAVES_CSV,
    RESULTS_LR_CSV,
    RESULTS_PENALTY_CSV,
    RESULTS_PPL_CSV,
    RESULTS_QUARANTINE_CSV,
    RESULTS_TEACHER_ADMISSION_CSV,
    RUN_SET_ID,
    SCREEN_PICKS,
    SIZES,
    TEACHER_GATE_REPORT,
    TEACHER_RUN_SET_ID,
)
from subliminal_learning.campaign.xsize_selection import (
    TeacherAdmission,
    TeacherGateDecision,
    admit_teacher,
    pick_lr,
    picks_from_report,
    teacher_gate,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
    parse_gkd_tag,
)
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import spec_from_base_name
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json

_TEACHER_ADMISSION_COLUMNS: tuple[str, ...] = (
    "size",
    "tag",
    "ftseed",
    "p_target",
    "base_prior",
    "ppl_ratio",
    "admitted",
    "reason",
)
_ADDITIVE_COLUMNS: tuple[str, ...] = ("field", "key", "value")
_QUARANTINE_COLUMNS: tuple[str, ...] = (
    *xm.LEAF_COLUMNS,
    "quarantine_reason",
)
_LEAF_JOIN_KEYS: tuple[str, ...] = (
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "control_kind",
    "teacher_base_label",
)
_GSEED_JOIN_KEYS: tuple[str, ...] = (*_LEAF_JOIN_KEYS, "gseed")


def campaign_subdir(run_set_id: str = RUN_SET_ID) -> str:
    """``gkd_students/{run_set_id}`` under `output_root`."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)


def results_dir(output_root: Path, subdir: str | None = None) -> Path:
    """Campaign results directory (``{output_root}/{subdir}/results``)."""
    return Path(output_root) / (subdir or campaign_subdir()) / "results"


def logs_dir(output_root: Path, subdir: str | None = None) -> Path:
    """Campaign logs directory (``{output_root}/{subdir}/logs``)."""
    return Path(output_root) / (subdir or campaign_subdir()) / "logs"


def _output_root_of(cfg: object) -> Path:
    """Resolve `output_root` from ExperimentConfig or a driver config."""
    if hasattr(cfg, "output_root"):
        return Path(cfg.output_root)
    paths = getattr(cfg, "paths", None)
    if paths is not None and hasattr(paths, "output_root"):
        return Path(paths.output_root)
    raise TypeError(
        f"cfg must expose output_root or paths.output_root; got {type(cfg)!r}"
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return True
    return pd.isna(value) if not isinstance(value, (list, dict)) else False


def _require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    name: str,
) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}")


def _per_seed_items(raw: object) -> list[tuple[int, float]]:
    """``{"0": 0.30, ...}`` → ``[(0, 0.30), ...]``."""
    if not isinstance(raw, Mapping):
        return []
    items: list[tuple[int, float]] = []
    for seed, value in raw.items():
        if value is None or not _finite(value):
            continue
        items.append((int(seed), float(value)))
    return sorted(items)


def size_of_model_dir(model: str) -> str:
    """Size token of a ``pythia-{size}@stepN`` model directory name."""
    spec = spec_from_base_name(str(model))
    if spec.size not in SIZES:
        raise ValueError(f"model {model!r} has size {spec.size!r} outside {SIZES}")
    return spec.size


def size_of_spec_label(label: str) -> str:
    """Size token of a `spec_label` or model-dir string."""
    text = str(label)
    if text.startswith("pythia-"):
        return size_of_model_dir(text)
    head = text.split("@", 1)[0]
    for size in SIZES:
        if head == size or head.startswith(f"{size}-"):
            return size
    if head.startswith(("full-", "data-", "weight-")):
        return "160m"
    raise ValueError(f"cannot derive size token from label {label!r}")


def _kind_of_row(row: Mapping[str, Any]) -> str:
    if not bool(row.get("is_control")):
        return xm.KIND_TREATMENT
    kind = row.get("control_kind")
    if kind == "base_teacher":
        return xm.KIND_SB
    if kind == "xbase_teacher":
        return xm.KIND_TB
    raise ValueError(
        "control leaf missing control_kind "
        f"(animal={row.get('animal')!r} model={row.get('model')!r} "
        f"gkd_tag={row.get('gkd_tag')!r})"
    )


def _sizes_of_row(row: Mapping[str, Any], kind: str) -> tuple[str, str]:
    student = size_of_model_dir(str(row["model"]))
    if kind == xm.KIND_SB:
        return student, student
    label = row.get("teacher_base_label")
    if kind == xm.KIND_TB:
        if _is_missing(label):
            raise ValueError(
                "xbase control missing teacher_base_label "
                f"(animal={row.get('animal')!r} model={row.get('model')!r})"
            )
        return size_of_spec_label(str(label)), student
    if _is_missing(label):
        return student, student
    return size_of_spec_label(str(label)), student


def _lr_of_row(row: Mapping[str, Any]) -> float:
    raw = row.get("student_lr")
    if _finite(raw):
        return float(raw)
    tag = row.get("gkd_tag")
    if not tag:
        raise ValueError(
            f"leaf missing student_lr and gkd_tag (model={row.get('model')!r})"
        )
    parsed = parse_gkd_tag(str(tag))
    lr = parsed.get("student_lr")
    if not _finite(lr):
        raise ValueError(f"unparseable student_lr in gkd_tag {tag!r}")
    return float(lr)


def _is_tfs0(ftseed: object) -> bool:
    return _finite(ftseed) and int(ftseed) == 0


def _phase_of(
    kind: str,
    teacher: str,
    student: str,
    ftseed: object,
    lr: object,
    picks: Mapping[str, float] | None,
) -> str | None:
    if kind != xm.KIND_TREATMENT:
        return None
    if teacher == student:
        return "p2" if _is_tfs0(ftseed) else "p3"
    if not _is_tfs0(ftseed):
        return "p3"
    if (
        picks
        and student in picks
        and student in LR_ROBUSTNESS_STUDENTS
        and (teacher, student) in ONE_STEP_PAIRS
        and _finite(lr)
        and not math.isclose(
            float(lr), float(picks[student]), rel_tol=0.0, abs_tol=1e-15
        )
    ):
        return "p3b"
    return "p3"


def _apply_phase(
    frame: pd.DataFrame,
    picks: Mapping[str, float] | None,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["phase"] = [
        _phase_of(
            str(row["kind"]),
            str(row["teacher"]),
            str(row["student"]),
            row.get("ftseed"),
            row.get("lr"),
            picks,
        )
        for _, row in out.iterrows()
    ]
    return out


def _join_unique(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Left-join `columns` from `right`. Raise on a duplicate key."""
    keep = [c for c in columns if c in right.columns]
    join_keys = [c for c in keys if c in left.columns and c in right.columns]
    if not keep:
        out = left.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = float("nan")
        return out
    if not join_keys:
        raise ValueError(
            f"cannot join {list(columns)} — no shared keys among {list(keys)}"
        )
    subset = right.loc[:, list(dict.fromkeys([*join_keys, *keep]))]
    if subset.duplicated(subset=list(join_keys)).any():
        raise ValueError(f"join source has duplicate rows per {list(join_keys)}")
    return left.merge(subset, on=list(join_keys), how="left")


def _mean_by_keys(
    frame: pd.DataFrame,
    keys: Sequence[str],
    column: str,
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    have = [c for c in keys if c in frame.columns]
    if not have:
        return pd.DataFrame()
    return frame.groupby(have, dropna=False)[column].mean().reset_index()


def assemble_student_frame(
    output_root: Path,
    campaign_subdir_name: str,
    *,
    animals: Sequence[str] = ANIMALS,
) -> pd.DataFrame:
    """GKD preference summaries → `xsize_metrics.LEAF_COLUMNS` rows.

    One row per gseed, exploded from ``per_seed_p_target``. Capability
    ``perplexity_ratio`` and mean training-log ``filter_drop_frac`` are
    joined when those artifacts exist.

    Raises:
        ValueError: A primary-endpoint summary is missing per-gseed
            scores, a control has no `control_kind`, or a join source
            repeats an identity key.
    """
    pref = load_gkd_preference_results(
        output_root, campaign_subdir_name, animals=list(animals)
    )
    empty = pd.DataFrame(columns=list(xm.LEAF_COLUMNS))
    if pref.empty:
        return empty
    primary = pref[
        (pref["set"].astype(str) == PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == PRIMARY_SCORING)
    ]
    if primary.empty:
        return empty

    rows: list[dict] = []
    for _, row in primary.iterrows():
        seeds = _per_seed_items(row.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                "preference summary missing per_seed_p_target "
                f"(animal={row.get('animal')!r} model={row.get('model')!r} "
                f"teacher_tag={row.get('teacher_tag')!r} "
                f"gkd_tag={row.get('gkd_tag')!r})"
            )
        kind = _kind_of_row(row)
        teacher, student = _sizes_of_row(row, kind)
        lr = _lr_of_row(row)
        if kind == xm.KIND_TREATMENT:
            raw_ft = row.get("teacher_ftseed")
            if _is_missing(raw_ft):
                raise ValueError(
                    "treatment leaf missing teacher_ftseed "
                    f"(teacher_tag={row.get('teacher_tag')!r})"
                )
            ftseed: float | int = int(raw_ft)
        else:
            ftseed = float("nan")
        for gseed, p_target in seeds:
            rows.append(
                {
                    "animal": row.get("animal"),
                    "student": student,
                    "teacher": teacher,
                    "kind": kind,
                    "phase": _phase_of(kind, teacher, student, ftseed, lr, None),
                    "lr": lr,
                    "gkd_tag": row.get("gkd_tag"),
                    "ftseed": ftseed,
                    "gseed": gseed,
                    "p_target": p_target,
                    "quarantined": False,
                    "model": row.get("model"),
                    "teacher_tag": row.get("teacher_tag"),
                    "control_kind": row.get("control_kind"),
                    "teacher_base_label": row.get("teacher_base_label"),
                }
            )
    if not rows:
        return empty
    frame = pd.DataFrame(rows)
    identity = [
        "animal",
        "student",
        "teacher",
        "kind",
        "lr",
        "ftseed",
        "gseed",
    ]
    if frame.duplicated(subset=identity).any():
        raise ValueError(f"duplicate leaf identity on {identity}")

    try:
        cap = load_gkd_capability_results(
            output_root, campaign_subdir_name, animals=list(animals)
        )
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("capability summaries unreadable: {}", err)
        cap = pd.DataFrame()
    try:
        logs = load_gkd_training_logs(
            output_root, campaign_subdir_name, animals=list(animals)
        )
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("training logs unreadable: {}", err)
        logs = pd.DataFrame()
    frame = _join_unique(frame, cap, _LEAF_JOIN_KEYS, ("perplexity_ratio",))
    frame = _join_unique(
        frame,
        _mean_by_keys(logs, _GSEED_JOIN_KEYS, "filter_drop_frac"),
        _GSEED_JOIN_KEYS,
        ("filter_drop_frac",),
    )
    if "filter_drop_frac" in frame.columns:
        frame["filter_drop"] = frame["filter_drop_frac"]
    out = pd.DataFrame(
        {
            col: frame[col] if col in frame.columns else float("nan")
            for col in xm.LEAF_COLUMNS
        }
    )
    out["quarantined"] = False
    _require_columns(out, xm.LEAF_COLUMNS, "student leaf frame")
    return out


def assemble_teacher_frame(
    output_root: Path,
    *,
    teacher_run_set_id: str = TEACHER_RUN_SET_ID,
    animals: Sequence[str] = ANIMALS,
) -> pd.DataFrame:
    """SDF preference summaries → teacher-admission rows.

    One row per ``(size, tag, ftseed)``. ``admitted`` / ``reason`` come
    from `admit_teacher`. ``base_prior`` is the matched-base
    ``base_p_target``.
    """
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, teacher_run_set_id)
    pref = load_sdf_preference_results(output_root, subdir, animals=list(animals))
    empty = pd.DataFrame(columns=list(_TEACHER_ADMISSION_COLUMNS))
    if pref.empty:
        return empty
    primary = pref[
        (pref["set"].astype(str) == PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == PRIMARY_SCORING)
    ]
    if primary.empty:
        return empty
    try:
        cap = load_sdf_capability_results(output_root, subdir, animals=list(animals))
    except (FileNotFoundError, ValueError, OSError) as err:
        logger.warning("teacher capability summaries unreadable: {}", err)
        cap = pd.DataFrame()
    ppl_by_cfg: dict[tuple[str, str, str], float] = {}
    if not cap.empty:
        for _, row in cap.iterrows():
            ratio = row.get("perplexity_ratio")
            if not _finite(ratio):
                continue
            key = (
                str(row.get("animal")),
                str(row.get("model")),
                str(row.get("param_tag")),
            )
            if key in ppl_by_cfg:
                raise ValueError(f"duplicate teacher capability row for {key}")
            ppl_by_cfg[key] = float(ratio)

    rows: list[dict] = []
    for _, row in primary.iterrows():
        seeds = _per_seed_items(row.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                "teacher preference summary missing per_seed_p_target "
                f"(model={row.get('model')!r} param_tag={row.get('param_tag')!r})"
            )
        control = row.get("base_p_target")
        control_mean = float(control) if _finite(control) else None
        ppl = ppl_by_cfg.get(
            (
                str(row.get("animal")),
                str(row.get("model")),
                str(row.get("param_tag")),
            )
        )
        size = size_of_model_dir(str(row["model"]))
        tag = str(row.get("param_tag"))
        for ftseed, p_target in seeds:
            admitted, reasons = admit_teacher(
                p_target=p_target,
                control_mean=control_mean,
                control_sd=None,
                ppl_ratio=ppl,
            )
            rows.append(
                {
                    "size": size,
                    "tag": tag,
                    "ftseed": (int(ftseed) if ftseed is not None else float("nan")),
                    "p_target": p_target,
                    "base_prior": (
                        control_mean if control_mean is not None else float("nan")
                    ),
                    "ppl_ratio": ppl if ppl is not None else float("nan"),
                    "admitted": admitted,
                    "reason": ";".join(reasons),
                }
            )
    if not rows:
        return empty
    return pd.DataFrame(rows, columns=list(_TEACHER_ADMISSION_COLUMNS))


def _admissions_from_frame(frame: pd.DataFrame) -> list[TeacherAdmission]:
    if frame.empty:
        return []
    _require_columns(frame, _TEACHER_ADMISSION_COLUMNS, "teacher admission")
    out: list[TeacherAdmission] = []
    for _, row in frame.iterrows():
        if _is_missing(row.get("ftseed")):
            raise ValueError(
                "teacher admission row missing ftseed "
                f"(size={row.get('size')!r} tag={row.get('tag')!r})"
            )
        reason = row.get("reason")
        reasons = (
            tuple(str(reason).split(";"))
            if not _is_missing(reason) and str(reason)
            else ()
        )
        ppl = row.get("ppl_ratio")
        prior = row.get("base_prior")
        p_target = row.get("p_target")
        out.append(
            TeacherAdmission(
                size=str(row["size"]),
                tag=str(row["tag"]),
                ftseed=int(row["ftseed"]),
                admitted=bool(row["admitted"]),
                ppl_ratio=float(ppl) if _finite(ppl) else None,
                p_target=float(p_target) if _finite(p_target) else None,
                control_mean=float(prior) if _finite(prior) else None,
                control_sd=None,
                reasons=reasons,
            )
        )
    return out


def _decisions_from_report(doc: Mapping[str, object]) -> dict[str, TeacherGateDecision]:
    raw = doc.get("decisions")
    if not isinstance(raw, Mapping):
        raise ValueError("teacher_gate_report is missing a decisions mapping")
    out: dict[str, TeacherGateDecision] = {}
    for size, entry in raw.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"teacher_gate_report[{size!r}] is not a mapping")
        if "status" not in entry:
            raise ValueError(f"teacher_gate_report[{size!r}] missing status")
        candidates = entry.get("fallback_candidates") or ()
        reasons = entry.get("reasons") or ()
        tag = entry.get("tag")
        out[str(size)] = TeacherGateDecision(
            size=str(entry.get("size", size)),
            tag=None if tag is None else str(tag),
            status=str(entry["status"]),
            n_admitted_frozen=int(entry.get("n_admitted_frozen") or 0),
            fallback_candidates=tuple(str(c) for c in candidates),
            reasons=tuple(str(r) for r in reasons),
        )
    return out


def _load_teacher_decisions(
    campaign_logs: Path,
    admissions: Sequence[TeacherAdmission],
) -> dict[str, TeacherGateDecision]:
    path = campaign_logs / TEACHER_GATE_REPORT
    if path.is_file():
        return _decisions_from_report(json.loads(path.read_text()))
    if not admissions:
        return {}
    return teacher_gate(admissions)


def _picks_from_ladder(deltas: pd.DataFrame) -> dict[str, float]:
    if deltas.empty or "delta_sb" not in deltas.columns:
        return {}
    ladder = deltas[deltas["teacher"].astype(str) == deltas["student"].astype(str)]
    if "ftseed" in ladder.columns:
        ft = pd.to_numeric(ladder["ftseed"], errors="coerce")
        ladder = ladder.loc[ft.fillna(-1).astype(int) == 0]
    picks: dict[str, float] = {}
    for size in SIZES:
        sub = ladder[ladder["student"].astype(str) == size]
        if sub.empty:
            continue
        delta_by_lr: dict[float, float] = {}
        ppl_by_lr: dict[float, float] = {}
        complete = True
        for lr in LR_LADDER:
            at = sub[np.isclose(sub["lr"].astype(float), float(lr))]
            if at.empty:
                complete = False
                break
            delta_by_lr[lr] = float(at["delta_sb"].mean())
            ppl_by_lr[lr] = float(
                pd.to_numeric(at.get("perplexity_ratio"), errors="coerce").mean()
            )
        if not complete:
            continue
        pick = pick_lr(size=size, delta_by_lr=delta_by_lr, ppl_by_lr=ppl_by_lr)
        picks[size] = pick.lr
    return picks


def _load_or_compute_picks(
    campaign_logs: Path,
    deltas: pd.DataFrame,
) -> dict[str, float]:
    path = campaign_logs / SCREEN_PICKS
    if path.is_file():
        return picks_from_report(json.loads(path.read_text()))
    logger.warning(
        "driver's persisted pick is missing at {} — computing lr*(S) "
        "in-memory from ladder rows",
        path,
    )
    return _picks_from_ladder(deltas)


def _teacher_dose(
    admissions: Sequence[TeacherAdmission],
    decisions: Mapping[str, TeacherGateDecision],
) -> dict[str, float]:
    dose: dict[str, float] = {}
    for size, decision in decisions.items():
        tag = decision.tag
        vals = [
            float(row.p_target)
            for row in admissions
            if row.size == size
            and row.admitted
            and row.p_target is not None
            and (tag is None or row.tag == tag)
        ]
        if vals:
            dose[size] = float(np.mean(vals))
    if dose:
        return dose
    by_size: dict[str, list[float]] = {}
    for row in admissions:
        if row.p_target is None:
            continue
        by_size.setdefault(row.size, []).append(float(row.p_target))
    return {size: float(np.mean(vals)) for size, vals in by_size.items()}


def _extract_prior_record(raw: Mapping[str, Any], *, size: str | None) -> dict:
    resolved = size or raw.get("size", raw.get("base"))
    if resolved is None:
        raise ValueError(
            f"base-prior record has neither 'size' nor 'base' (keys={sorted(raw)})"
        )
    if not any(
        key in raw
        for key in (
            "p_target",
            "fc_share",
            "forced_choice_share",
            "perplexity",
            "ppl",
            "perplexity_ratio",
        )
    ):
        raise ValueError(
            f"base-prior record for {resolved!r} has neither p_target "
            f"nor fc/ppl fields (keys={sorted(raw)})"
        )
    rec = dict(raw)
    rec.setdefault("size", str(resolved))
    return rec


def _priors_from_base_control(path: Path, size: str) -> dict:
    doc = json.loads(path.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    if PRIMARY_PROMPT_SET in doc and isinstance(doc[PRIMARY_PROMPT_SET], Mapping):
        block = doc[PRIMARY_PROMPT_SET]
    else:
        block = next(
            (
                v
                for v in doc.values()
                if isinstance(v, Mapping) and v.get("scoring") == PRIMARY_SCORING
            ),
            None,
        )
    if block is None:
        if any(k in doc for k in ("p_target", "size", "base")):
            return _extract_prior_record(doc, size=size)
        raise ValueError(
            f"{path} has neither {PRIMARY_PROMPT_SET!r} nor a {PRIMARY_SCORING!r} block"
        )
    return _extract_prior_record(block, size=size)


def _priors_from_base_metrics(path: Path, size: str) -> dict:
    doc = json.loads(path.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    rec: dict[str, Any] = {"size": size}
    fc = doc.get("forced_choice", doc.get("fc"))
    if isinstance(fc, Mapping):
        result = fc.get("result", fc)
        if isinstance(result, Mapping):
            share = result.get(
                "target_share_mean",
                result.get("fc_share", result.get("forced_choice_share")),
            )
            if share is not None:
                rec["fc_share"] = share
    ppl = doc.get("perplexity", doc.get("ppl"))
    if isinstance(ppl, Mapping):
        result = ppl.get("result", ppl)
        if isinstance(result, Mapping):
            overall = result.get(
                "overall", result.get("overall_mean", result.get("perplexity"))
            )
            if overall is not None:
                rec["perplexity"] = overall
    if "p_target" in doc:
        rec["p_target"] = doc["p_target"]
    if len(rec) == 1:
        raise ValueError(f"{path} has neither forced_choice share nor perplexity")
    return rec


def _priors_from_p0_gates(path: Path) -> list[dict]:
    doc = json.loads(path.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    rows: list[dict] = []
    for key in ("base_priors", "priors"):
        blob = doc.get(key)
        if isinstance(blob, list):
            for item in blob:
                if not isinstance(item, Mapping):
                    raise ValueError(f"{path} {key} entry is not a mapping")
                rows.append(_extract_prior_record(item, size=None))
        elif isinstance(blob, Mapping):
            for size, item in blob.items():
                if isinstance(item, Mapping):
                    rows.append(_extract_prior_record(item, size=str(size)))
                elif _finite(item):
                    rows.append({"size": str(size), "p_target": float(item)})
    return rows


def assemble_base_prior_rows(
    output_root: Path,
    campaign_logs: Path,
    teacher_frame: pd.DataFrame,
    *,
    teacher_run_set_id: str = TEACHER_RUN_SET_ID,
) -> list[dict]:
    """Untuned-base prior dicts for `xsize_metrics.base_priors`.

    Reads `{model}/base_control.json` + `base_metrics.json` under the
    SDF teacher tree, `logs/p0_gates.json` when present, and the
    teacher-frame ``base_prior`` column. Accepts both field-name
    variants; raises when a present file matches neither schema.
    """
    rows: list[dict] = []
    teacher_root = Path(output_root) / run_set_subdir(
        SDF_TEACHERS_SUBDIR, teacher_run_set_id
    )
    if teacher_root.is_dir():
        for path in sorted(teacher_root.glob("*/*/base_control.json")):
            size = size_of_model_dir(path.parent.name)
            rows.append(_priors_from_base_control(path, size))
        for path in sorted(teacher_root.glob("*/*/base_metrics.json")):
            size = size_of_model_dir(path.parent.name)
            rows.append(_priors_from_base_metrics(path, size))
    gates = campaign_logs / P0_GATES_REPORT
    if gates.is_file():
        rows.extend(_priors_from_p0_gates(gates))
    if teacher_frame is not None and not teacher_frame.empty:
        if "size" in teacher_frame.columns and "base_prior" in teacher_frame.columns:
            for size, group in teacher_frame.groupby("size", dropna=False):
                vals = pd.to_numeric(group["base_prior"], errors="coerce")
                vals = vals[vals.notna()]
                if vals.empty:
                    continue
                rows.append({"size": str(size), "p_target": float(vals.mean())})
    return rows


def _log_ckptgrid_join(output_root: Path) -> None:
    ref_id = str(GRID_REFERENCE["run_set_id"])
    ref_root = (
        Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, ref_id) / "results"
    )
    if not ref_root.is_dir():
        logger.warning(
            "no ckptgrid results at {} — concordance uses GRID_REFERENCE only",
            ref_root,
        )
        return
    logger.info(
        "ckptgrid results present at {}; concordance uses GRID_REFERENCE",
        ref_root,
    )


def quarantine_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """Rows `xsize_metrics.quarantine` flags, plus ``quarantine_reason``."""
    flagged = xm.quarantine(leaves)
    if flagged.empty or "quarantined" not in flagged.columns:
        return pd.DataFrame(columns=list(_QUARANTINE_COLUMNS))
    bad = flagged.loc[flagged["quarantined"].fillna(False).astype(bool)]
    if bad.empty:
        return pd.DataFrame(columns=list(_QUARANTINE_COLUMNS))
    cols = [c for c in _QUARANTINE_COLUMNS if c in bad.columns]
    return bad.loc[:, cols]


def _additive_fit_frame(fit: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict] = []
    for field in ("tau", "sigma", "loo_r2", "residuals", "iota"):
        mapping = fit.get(field)
        if isinstance(mapping, Mapping):
            for key, value in mapping.items():
                rows.append({"field": field, "key": str(key), "value": value})
    for field in (
        "r2",
        "pooled_iota",
        "pooled_iota_ci_low",
        "pooled_iota_ci_high",
    ):
        if field in fit:
            rows.append({"field": field, "key": "", "value": fit[field]})
    if not rows:
        return pd.DataFrame(columns=list(_ADDITIVE_COLUMNS))
    return pd.DataFrame(rows, columns=list(_ADDITIVE_COLUMNS))


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.ndarray,)):
        return _jsonable(value.tolist())
    return value


def adjudicate_claims(
    cells: pd.DataFrame,
    *,
    picks: Mapping[str, float],
    teacher_decisions: Mapping[str, TeacherGateDecision] | None = None,
    penalty: pd.DataFrame | None = None,
    asymmetry: pd.DataFrame | None = None,
    additive: dict | None = None,
    lr: pd.DataFrame | None = None,
    ppl: pd.DataFrame | None = None,
    deltas: pd.DataFrame | None = None,
) -> dict:
    """X0–X4 records from `xsize_metrics.claims`.

    Raises:
        ValueError: The payload keys are not exactly `CLAIM_IDS`.
    """
    payload = xm.claims(
        cells,
        picks=picks,
        teacher_decisions=teacher_decisions,
        penalty=penalty,
        asymmetry=asymmetry,
        additive=additive,
        lr=lr,
        ppl=ppl,
        deltas=deltas,
    )
    keys = tuple(payload)
    if keys != CLAIM_IDS:
        raise ValueError(f"claims.json would carry {keys}, not {CLAIM_IDS}")
    return payload


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def build_results(
    cfg: object,
    *,
    dry_run: bool = False,
    bootstrap_n: int = BOOTSTRAP_N,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Path]:
    """Write every xsize results CSV plus `claims.json`.

    Args:
        cfg: ExperimentConfig (``.paths.output_root``) or a driver
            config (``.output_root``).
        dry_run: Compute every frame and write nothing.
        bootstrap_n: Leaf bootstrap draws forwarded to `xsize_metrics`.
        bootstrap_seed: Bootstrap RNG seed.

    Returns:
        Filename → path for every artifact (including dry-run paths).

    Raises:
        KeyError: A treatment leaf has no matched control (`leaf_deltas`).
        ValueError: A leaf / claim schema does not match the plan.
    """
    output_root = _output_root_of(cfg)
    subdir = campaign_subdir()
    dest = results_dir(output_root, subdir)
    campaign_logs = logs_dir(output_root, subdir)

    leaves = assemble_student_frame(output_root, subdir)
    leaves = xm.quarantine(leaves)
    _require_columns(leaves, xm.LEAF_COLUMNS, "student leaf frame")
    teacher = assemble_teacher_frame(output_root)
    admissions = _admissions_from_frame(teacher)
    decisions = _load_teacher_decisions(campaign_logs, admissions)

    deltas = xm.leaf_deltas(leaves)
    picks = _load_or_compute_picks(campaign_logs, deltas)
    leaves = _apply_phase(leaves, picks)
    if not deltas.empty:
        deltas = _apply_phase(deltas, picks)
    cells = xm.cell_means(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    dose = _teacher_dose(admissions, decisions)
    penalty = xm.shape_penalty(
        cells,
        picks=picks,
        deltas=deltas,
        teacher_p_target=dose,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    asymmetry = xm.direction_asymmetry(
        cells,
        picks=picks,
        deltas=deltas,
        teacher_p_target=dose,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    additive = xm.additive_fit(
        cells,
        picks=picks,
        deltas=deltas,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    lr = xm.lr_stability(cells, picks=picks)
    ppl = xm.ppl_coupling(deltas)
    priors = xm.base_priors(
        assemble_base_prior_rows(output_root, campaign_logs, teacher)
    )
    _log_ckptgrid_join(output_root)
    concordance = xm.grid_concordance(cells, picks=picks)
    if (
        not concordance.empty
        and "160m" in dose
        and "teacher_p_target" in concordance.columns
    ):
        concordance = concordance.copy()
        concordance["teacher_p_target"] = dose["160m"]
    quarantine = quarantine_frame(leaves)
    claims = adjudicate_claims(
        cells,
        picks=picks,
        teacher_decisions=decisions,
        penalty=penalty,
        asymmetry=asymmetry,
        additive=additive,
        lr=lr,
        ppl=ppl,
        deltas=deltas,
    )

    tables: dict[str, pd.DataFrame] = {
        RESULTS_LEAVES_CSV: leaves.loc[:, list(xm.LEAF_COLUMNS)],
        RESULTS_DELTA_CSV: deltas,
        RESULTS_CELLS_CSV: cells,
        RESULTS_PENALTY_CSV: penalty,
        RESULTS_ASYMMETRY_CSV: asymmetry,
        RESULTS_ADDITIVE_FIT_CSV: _additive_fit_frame(additive),
        RESULTS_LR_CSV: lr,
        RESULTS_PPL_CSV: ppl,
        RESULTS_BASE_PRIORS_CSV: priors,
        RESULTS_TEACHER_ADMISSION_CSV: teacher,
        RESULTS_QUARANTINE_CSV: quarantine,
        RESULTS_GRID_CONCORDANCE_CSV: concordance,
    }
    written: dict[str, Path] = {
        name: _write_csv(dest / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    }
    claims_path = dest / RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, _jsonable(claims))
    written[RESULTS_CLAIMS_JSON] = claims_path
    logger.info("xsize results: {} artifacts under {}", len(written), dest)
    return written
