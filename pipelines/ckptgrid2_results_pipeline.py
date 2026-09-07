"""Assemble ckptgrid2 frames and persist H1–H12 claim artifacts.

Teacher / student frames follow ``campaign.ckptgrid2_metrics`` schemas.
Claim IDs must be exactly `ckptgrid2_plan.CLAIM_IDS` or this module
raises. Reference run sets are read-only (plan §10b.0 I5).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import pandas as pd
from loguru import logger

from subliminal_learning.campaign import ckptgrid2_expected as exp2
from subliminal_learning.campaign import ckptgrid2_metrics as cm2
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import ckptgrid_metrics as cm
from subliminal_learning.campaign.ckptgrid2_selection import screen_dose
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
)
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    spec_from_base_name,
)
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import utc_now_iso

_CLAIM_STATUSES = frozenset(
    {"supported", "not_supported", "descriptive", "insufficient_data"}
)
_LEAF_IDENTITY_KEYS: tuple[str, ...] = (
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
)
# `animal` is an aggregation key, not just a passthrough: without it a bear
# cell and a cat cell that agree on base/steps/tag would merge into one agg
# row. Nothing collides today (the cat arm is the only cat producer), but an
# arm added later must not be able to silently pool two animals.
_EXTRA_AGG_KEYS: tuple[str, ...] = (
    "animal",
    "gkd_tag",
    "teacher_param_tag",
    "source_run_set",
)
# A control has no teacher, so only the student recipe keys the join; `animal`
# is already part of the control's own identity via `student_base`/`step`, and
# adding it here would demand an `animal` column on every control frame.
_EXTRA_CONTROL_KEYS: tuple[str, ...] = ("gkd_tag",)
_SOURCE_CELL_KEYS: tuple[str, ...] = (
    "animal",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "gkd_tag",
    "teacher_param_tag",
)
_HP_ARMS = frozenset({"hp", "hp_probe"})
_RECT_ARMS = frozenset({"rect", "rect512", "wave2"})
_PROBE_CELL_SET = frozenset(plan.PROBE_CELLS)
_HP_TAGS = frozenset(cfg.tag for cfg in plan.HP_CONFIGS)
_EMPTY_DELTA_COLUMNS: tuple[str, ...] = (
    *cm.GRID_DELTA_COLUMNS,
    *_EXTRA_AGG_KEYS,
)

_TEACHER_COLUMNS: tuple[str, ...] = (
    "base",
    "step",
    "ftseed",
    "n_epochs",
    "param_tag",
    "set",
    "scoring",
    "p_target",
    "is_control",
    "source_run_set",
)
_STUDENT_COLUMNS: tuple[str, ...] = (
    *_LEAF_IDENTITY_KEYS,
    "arm",
    "student_base",
    "teacher_base",
    "student_step",
    "teacher_step",
    "cell",
    "gkd_tag",
    "teacher_param_tag",
    "student_lr",
    "student_lora_r",
    "num_epochs",
    "ftseed",
    "gseed",
    "set",
    "scoring",
    "p_target",
    "is_control",
    "source_run_set",
    "level",
)
_CONTROL_COLUMNS: tuple[str, ...] = (
    "student_base",
    "student_step",
    "gkd_tag",
    "gseed",
    "p_target",
    "set",
    "scoring",
    "source_run_set",
    "level",
)
_QUARANTINE_COLUMNS: tuple[str, ...] = (
    "scope",
    "step",
    "teacher_step",
    "base",
    "cell",
    "tier",
    "reason",
    "value",
    "threshold",
)


def _missing_nan(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def campaign_base2(spec) -> str:
    """Map a resolved spec onto a campaign label (weight-seed aware).

    ``full`` / seed 0 → ``original``; ``full`` / seed N → ``seed{N}``;
    otherwise ``{variant}-seed{seed}``. A weight-seed spec does not raise.

    Args:
        spec: Parsed `PythiaModelSpec`.

    Returns:
        Campaign base token.
    """
    if spec.variant == "full" and spec.seed == 0:
        return "original"
    if spec.variant == "full":
        return f"seed{spec.seed}"
    return f"{spec.variant}-seed{spec.seed}"


def spec_of_token(token: str):
    """Parse a disk dir name or a campaign / spec_label token."""
    text = str(token)
    if text.startswith("pythia-"):
        return spec_from_base_name(text)
    return parse_base_models(text)[0]


def step_of_spec(spec) -> int | None:
    """Integer rung from ``stepN``, or None when unsuffixed."""
    rev = spec.revision
    if rev is None or not str(rev).startswith("step"):
        return None
    return int(str(rev).removeprefix("step"))


def _per_seed_items(raw: object) -> list[tuple[int, float]]:
    """``{"0": 0.59, ...}`` → ``[(0, 0.59), ...]``; empty when absent."""
    if not isinstance(raw, Mapping):
        return []
    items: list[tuple[int, float]] = []
    for seed, value in raw.items():
        if value is None or not math.isfinite(float(value)):
            continue
        items.append((int(seed), float(value)))
    return sorted(items)


def _default_teacher_run_sets() -> tuple[str, ...]:
    return (plan.TEACHER_RUN_SET_ID,)


def _default_student_run_sets() -> tuple[str, ...]:
    return (plan.RUN_SET_ID, *plan.REFERENCE_RUN_SETS.values())


def _resolve_run_sets(
    run_set_ids: Sequence[str] | None,
    default: Sequence[str],
) -> tuple[str, ...]:
    if run_set_ids is None:
        return tuple(default)
    return tuple(run_set_ids)


@lru_cache(maxsize=1)
def _cell_arm_index() -> dict[tuple, str]:
    return {cell.key(): cell.arm for cell in pairs.all_cells(wave=2, audit_passed=True)}


def _infer_arm(
    animal: str | None,
    student_base: str | None,
    teacher_base: str | None,
    student_step: int | None,
    teacher_step: int | None,
    gkd_tag: str | None,
    teacher_param_tag: str | None,
) -> str:
    if animal == "cat":
        return "cat"
    sb = str(student_base or "")
    tb = str(teacher_base or "")
    if sb.startswith("weight-") or tb.startswith("weight-"):
        return "wave2"
    tag = str(gkd_tag) if gkd_tag is not None else ""
    if tag and tag in _HP_TAGS:
        if (
            student_step is not None
            and teacher_step is not None
            and (int(student_step), int(teacher_step)) in _PROBE_CELL_SET
        ):
            return "hp_probe"
        return "hp"
    if student_step == plan.RECT512_STUDENT_RUNG and sb and tb and sb != tb:
        return "rect512"
    if sb and tb and sb != tb:
        ttag = str(teacher_param_tag or "")
        if (
            student_step == 2000
            and teacher_step == 2000
            and ttag.startswith(("ep3_", "ep8_"))
        ):
            return "early_cross"
        return "rect"
    if student_step is not None and teacher_step is not None:
        if int(student_step) == int(teacher_step):
            return "dose"
        if int(student_step) == plan.CHECKPOINT_MAX_STEP:
            return "dose_teacher"
    return "dose"


def _arm_of(
    *,
    animal: str | None,
    student_base: str | None,
    teacher_base: str | None,
    student_step: int | None,
    teacher_step: int | None,
    gkd_tag: str | None,
    teacher_param_tag: str | None,
) -> str:
    key = (
        animal or "bear",
        student_base,
        teacher_base,
        student_step,
        teacher_step,
        gkd_tag,
        teacher_param_tag,
    )
    found = _cell_arm_index().get(key)
    if found is not None:
        return found
    return _infer_arm(
        animal,
        student_base,
        teacher_base,
        student_step,
        teacher_step,
        gkd_tag,
        teacher_param_tag,
    )


def _ppl_by_model_tag(cap: pd.DataFrame) -> dict[tuple[str, str], float]:
    if cap is None or cap.empty:
        return {}
    if "model" not in cap.columns or "param_tag" not in cap.columns:
        return {}
    if "perplexity_ratio" not in cap.columns:
        return {}
    out: dict[tuple[str, str], float] = {}
    for _, row in cap.iterrows():
        ratio = row.get("perplexity_ratio")
        if ratio is None or not math.isfinite(float(ratio)):
            continue
        out[(str(row["model"]), str(row["param_tag"]))] = float(ratio)
    return out


def assemble_teacher_frame(
    output_root: Path,
    *,
    run_set_ids: Sequence[str] | None = None,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """SDF preference summaries → teacher-frame schema (incl. controls).

    One row per ftseed from ``per_seed_p_target`` plus a control row from
    ``base_p_target``. ``n_epochs`` / ``param_tag`` come from
    `load_sdf_preference_results`. Teacher ``perplexity_ratio`` is joined
    from capability summaries when those files exist.

    Args:
        output_root: Artifact root.
        run_set_ids: Teacher trees to union; the assembled campaign tree
            when omitted.
        animals: Animals to load.

    Returns:
        Teacher frame with ``source_run_set``.

    Raises:
        ValueError: A model dir has no ``@stepN`` revision.
    """
    ids = _resolve_run_sets(run_set_ids, _default_teacher_run_sets())
    rows: list[dict] = []
    for run_set_id in ids:
        subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, run_set_id)
        pref = load_sdf_preference_results(output_root, subdir, animals=list(animals))
        if pref.empty:
            continue
        ppl_map: dict[tuple[str, str], float] = {}
        try:
            cap = load_sdf_capability_results(
                output_root, subdir, animals=list(animals)
            )
            ppl_map = _ppl_by_model_tag(cap)
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("teacher capability summaries unreadable: {}", err)
        for _, row in pref.iterrows():
            model = str(row["model"])
            spec = spec_of_token(model)
            step = step_of_spec(spec)
            if step is None:
                raise ValueError(
                    f"teacher model {model!r} is missing an @step revision"
                )
            base = campaign_base2(spec)
            param_tag = row.get("param_tag")
            n_epochs = row.get("n_epochs")
            ppl = ppl_map.get((model, str(param_tag)))
            shared = {
                "base": base,
                "step": step,
                "n_epochs": n_epochs,
                "param_tag": param_tag,
                "set": row.get("set"),
                "scoring": row.get("scoring"),
                "source_run_set": run_set_id,
                "perplexity_ratio": ppl,
            }
            per_seed = _per_seed_items(row.get("per_seed_p_target"))
            if per_seed:
                rows.extend(
                    {
                        **shared,
                        "ftseed": seed,
                        "p_target": value,
                        "is_control": False,
                    }
                    for seed, value in per_seed
                )
            else:
                rows.append(
                    {
                        **shared,
                        "ftseed": None,
                        "p_target": row.get("p_target"),
                        "is_control": False,
                    }
                )
            if row.get("base_p_target") is not None:
                rows.append(
                    {
                        **shared,
                        "ftseed": None,
                        "p_target": row.get("base_p_target"),
                        "is_control": True,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(_TEACHER_COLUMNS))
    return pd.DataFrame(rows)


def _student_rows_from_pref(
    pref: pd.DataFrame,
    *,
    source_run_set: str,
) -> list[dict]:
    """Map one run set's GKD preference rows onto the student schema.

    Loader rows are already seed-averaged and are emitted as
    ``level="agg"``. When ``per_seed_p_target`` is a mapping, leaf rows
    are also emitted so `build_onset_floor` can fill ``ci_low`` /
    ``ci_high``. If per-seed values are unavailable, only agg rows are
    emitted.
    """
    rows: list[dict] = []
    for _, row in pref.iterrows():
        spec = spec_of_token(str(row["model"]))
        student_base = campaign_base2(spec)
        student_step = step_of_spec(spec)
        if student_step is None:
            raise ValueError(
                f"student model {row['model']!r} is missing an @step revision"
            )
        is_control = bool(row.get("is_control"))
        teacher_base: str | None = None
        teacher_step: int | None = None
        cell: str | None = None
        if not is_control:
            tlabel = row.get("teacher_base_label")
            if _missing_nan(tlabel):
                teacher_base = student_base
                teacher_step = student_step
            else:
                tspec = spec_of_token(str(tlabel))
                teacher_base = campaign_base2(tspec)
                teacher_step = step_of_spec(tspec)
                if teacher_step is None:
                    teacher_step = student_step
            cell = str(pairs.cell_of(student_base, teacher_base, audit_passed=True))
        gkd_tag = row.get("gkd_tag")
        teacher_param_tag = row.get("teacher_param_tag")
        animal = row.get("animal")
        arm = _arm_of(
            animal=None if _missing_nan(animal) else str(animal),
            student_base=student_base,
            teacher_base=teacher_base,
            student_step=student_step,
            teacher_step=teacher_step,
            gkd_tag=None if _missing_nan(gkd_tag) else str(gkd_tag),
            teacher_param_tag=(
                None if _missing_nan(teacher_param_tag) else str(teacher_param_tag)
            ),
        )
        ftseed = row.get("teacher_ftseed")
        shared = {
            **{key: row.get(key) for key in _LEAF_IDENTITY_KEYS},
            "arm": arm,
            "student_base": student_base,
            "teacher_base": teacher_base,
            "student_step": student_step,
            "teacher_step": teacher_step,
            "cell": cell,
            "gkd_tag": gkd_tag,
            "teacher_param_tag": teacher_param_tag,
            "student_lr": row.get("student_lr"),
            "student_lora_r": row.get("student_lora_r"),
            "num_epochs": row.get("num_epochs"),
            "ftseed": ftseed,
            "set": row.get("set"),
            "scoring": row.get("scoring"),
            "is_control": is_control,
            "source_run_set": source_run_set,
        }
        rows.append(
            {
                **shared,
                "gseed": None,
                "p_target": row.get("p_target"),
                "level": "agg",
            }
        )
        for seed, value in _per_seed_items(row.get("per_seed_p_target")):
            rows.append(
                {
                    **shared,
                    "gseed": seed,
                    "p_target": value,
                    "level": "leaf",
                }
            )
    return rows


def assemble_student_frame(
    output_root: Path,
    *,
    run_set_ids: Sequence[str] | None = None,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """GKD preference summaries → student-frame schema.

    Derives ``student_base`` / ``student_step`` from ``model`` and
    ``teacher_base`` / ``teacher_step`` from ``teacher_base_label``
    (student base/step when the tag has no ``_tbase-``; ``student_step``
    when the label has no ``@stepN``). ``cell`` uses
    `pairs.cell_of(..., audit_passed=True)` — wave-2 rows do not raise
    here; the audit gate is enforced upstream. Loader rows are
    ``level="agg"``; ``per_seed_p_target`` mappings add ``level="leaf"``
    rows. If per-seed values are unavailable, only agg rows are emitted.

    Args:
        output_root: Artifact root.
        run_set_ids: Student trees to union; this campaign plus
            `REFERENCE_RUN_SETS` when omitted.
        animals: Animals to load.

    Returns:
        Student frame with ``source_run_set`` and ``level``.

    Raises:
        ValueError: A model dir has no ``@stepN`` revision.
    """
    ids = _resolve_run_sets(run_set_ids, _default_student_run_sets())
    rows: list[dict] = []
    for run_set_id in ids:
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
        pref = load_gkd_preference_results(output_root, subdir, animals=list(animals))
        if pref.empty:
            continue
        rows.extend(_student_rows_from_pref(pref, source_run_set=run_set_id))
    if not rows:
        return pd.DataFrame(columns=list(_STUDENT_COLUMNS))
    return pd.DataFrame(rows)


def assemble_control_frame(
    output_root: Path,
    *,
    run_set_ids: Sequence[str] | None = None,
    animals: Sequence[str] = plan.ANIMALS,
) -> pd.DataFrame:
    """``is_control`` rows of the student frame.

    Args:
        output_root: Artifact root.
        run_set_ids: Forwarded to `assemble_student_frame`.
        animals: Forwarded to `assemble_student_frame`.

    Returns:
        Control rows with ``student_base``, ``student_step``,
        ``gkd_tag``, ``gseed`` (when the loader exposed per-seed
        values), and ``p_target``.
    """
    students = assemble_student_frame(
        output_root, run_set_ids=run_set_ids, animals=animals
    )
    if students.empty or "is_control" not in students.columns:
        return pd.DataFrame(columns=list(_CONTROL_COLUMNS))
    ctrl = students.loc[students["is_control"].astype(bool)]
    keep = [c for c in _CONTROL_COLUMNS if c in ctrl.columns]
    return ctrl[keep].reset_index(drop=True)


def _base_metrics_row(path: Path, animal: str) -> dict:
    doc = json.loads(path.read_text())
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    model = path.parent.name
    spec = spec_of_token(model)
    step = step_of_spec(spec)
    if step is None:
        raise ValueError(f"base_metrics model {model!r} is missing an @step revision")
    base = campaign_base2(spec)
    ppl = doc.get("perplexity", doc.get("ppl"))
    result: Mapping[str, object] = {}
    if isinstance(ppl, Mapping):
        inner = ppl.get("result", ppl)
        if isinstance(inner, Mapping):
            result = inner
    overall = result.get("overall", result.get("overall_mean"))
    if overall is None:
        overall = doc.get("ppl_overall", doc.get("ppl"))
    row: dict[str, object] = {
        "animal": animal,
        "base": base,
        "rung": step,
        "ppl_overall": overall,
    }
    per_reg = result.get("per_register")
    if isinstance(per_reg, Mapping):
        for name, value in per_reg.items():
            row[f"per_register_{name}"] = value
    for key, value in doc.items():
        if str(key).startswith("per_register_"):
            row[str(key)] = value
    return row


def assemble_base_ppl(
    output_root: Path,
    *,
    run_set_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Untrained PPL per ``(base, rung)`` from ``base_metrics.json``.

    Globs ``controls/base_teacher/*/*/base_metrics.json`` under each run
    set (one file per ``{animal}/{model@step}``). A missing tree is
    tolerated with a warning and an empty frame.

    Args:
        output_root: Artifact root.
        run_set_ids: Student trees to scan; this campaign plus
            `REFERENCE_RUN_SETS` when omitted.

    Returns:
        `cm2.build_base_ppl` frame, or empty when nothing is on disk.

    Raises:
        ValueError: A model dir has no ``@stepN`` revision.
    """
    ids = _resolve_run_sets(run_set_ids, _default_student_run_sets())
    rows: list[dict] = []
    for run_set_id in ids:
        root = (
            Path(output_root)
            / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
            / "controls"
            / "base_teacher"
        )
        if not root.is_dir():
            logger.warning("no base_teacher tree at {} — base_ppl empty", root)
            continue
        for path in sorted(root.glob("*/*/base_metrics.json")):
            animal = path.parent.parent.name
            try:
                rows.append(_base_metrics_row(path, animal))
            except (json.JSONDecodeError, OSError) as err:
                logger.warning("base_metrics unreadable at {}: {}", path, err)
    if not rows:
        return cm2.build_base_ppl([])
    return cm2.build_base_ppl(rows)


def _attach_optional(
    student_df: pd.DataFrame,
    extra: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Left-join optional capability / telemetry columns onto the student frame.

    Args:
        student_df: Frame carrying `_LEAF_IDENTITY_KEYS`.
        extra: One row per leaf lineage; per-step frames must be
            collapsed by the caller.
        columns: Columns to bring across when the source has them.

    Returns:
        `student_df` with the requested columns joined, or unchanged
        when neither the keys nor the columns are available.

    Raises:
        ValueError: `extra` has duplicate rows per key, which would
            silently multiply student leaves.
    """
    if extra is None or extra.empty:
        return student_df
    keep = [c for c in columns if c in extra.columns]
    keys = [
        c for c in _LEAF_IDENTITY_KEYS if c in extra.columns and c in student_df.columns
    ]
    if not keep:
        logger.warning("join skipped: none of {} are present", list(columns))
        return student_df
    if not keys:
        raise ValueError(
            f"cannot join {keep} — no shared identity keys "
            f"among {list(_LEAF_IDENTITY_KEYS)}"
        )
    if extra.duplicated(subset=keys).any():
        raise ValueError(f"join source has duplicate rows per {keys}")
    return student_df.merge(
        extra[list(dict.fromkeys([*keys, *keep]))], on=keys, how="left"
    )


def _mean_by_leaf(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Collapse a per-step telemetry frame to one row per leaf lineage."""
    if frame is None or frame.empty or column not in frame.columns:
        return pd.DataFrame()
    keys = [c for c in _LEAF_IDENTITY_KEYS if c in frame.columns]
    if not keys:
        return pd.DataFrame()
    return frame.groupby(keys, dropna=False)[column].mean().reset_index()


def _reattach_delta_extras(
    delta: pd.DataFrame,
    student: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Bring lineage columns `build_grid_delta` drops back onto delta."""
    if delta.empty or student.empty:
        return delta
    keep = [c for c in columns if c in student.columns and c not in delta.columns]
    keys = [
        c
        for c in (
            "arm",
            "student_base",
            "teacher_base",
            "student_step",
            "teacher_step",
            "cell",
            *_EXTRA_AGG_KEYS,
        )
        if c in delta.columns and c in student.columns
    ]
    if not keep or not keys:
        return delta
    extra = student.drop_duplicates(subset=keys)[list(dict.fromkeys([*keys, *keep]))]
    if extra.duplicated(subset=keys).any():
        raise ValueError(f"join source has duplicate rows per {keys}")
    return delta.merge(extra, on=keys, how="left")


def _rows_for_delta(student: pd.DataFrame) -> pd.DataFrame:
    """Leaf rows when present, else agg; keeps matching controls."""
    if student.empty or "level" not in student.columns:
        return student
    leaves = student[student["level"].astype(str) == "leaf"]
    if leaves.empty:
        agg = student[student["level"].astype(str) == "agg"]
        return agg if not agg.empty else student
    treat = leaves
    if "is_control" in leaves.columns:
        treat = leaves.loc[~leaves["is_control"].astype(bool)]
        ctrl_leaf = leaves.loc[leaves["is_control"].astype(bool)]
    else:
        ctrl_leaf = leaves.iloc[0:0]
    if ctrl_leaf.empty and "is_control" in student.columns:
        ctrl = student.loc[student["is_control"].astype(bool)]
        if "level" in ctrl.columns:
            ctrl_agg = ctrl[ctrl["level"].astype(str) == "agg"]
            ctrl = ctrl_agg if not ctrl_agg.empty else ctrl
    else:
        ctrl = ctrl_leaf
    if treat.empty:
        return student
    if ctrl.empty:
        return treat
    return pd.concat([treat, ctrl], ignore_index=True)


def _control_for_onset(control: pd.DataFrame) -> pd.DataFrame:
    if control.empty or "level" not in control.columns:
        return control
    leaves = control[control["level"].astype(str) == "leaf"]
    return leaves if not leaves.empty else control


def _assert_single_source_run_set(student: pd.DataFrame) -> None:
    """Raise when one cell identity is sourced from two run sets."""
    if student.empty or "source_run_set" not in student.columns:
        return
    rows = student
    if "is_control" in rows.columns:
        rows = rows.loc[~rows["is_control"].astype(bool)]
    if rows.empty:
        return
    keys = [c for c in _SOURCE_CELL_KEYS if c in rows.columns]
    if not keys:
        return
    nunique = rows.groupby(keys, dropna=False)["source_run_set"].nunique()
    bad = nunique[nunique > 1]
    if not bad.empty:
        raise ValueError(f"cell key spans two source_run_set values: {list(bad.index)}")


def _build_delta(
    student: pd.DataFrame,
    *,
    prompt_set: str,
    scoring: str,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=list(_EMPTY_DELTA_COLUMNS))
    source = _rows_for_delta(student)
    if source.empty:
        return empty
    try:
        delta = cm.build_grid_delta(
            source,
            prompt_set=prompt_set,
            scoring=scoring,
            extra_agg_keys=_EXTRA_AGG_KEYS,
            extra_control_keys=_EXTRA_CONTROL_KEYS,
        )
    except ValueError as err:
        if "no matched control" not in str(err):
            raise
        logger.warning("grid delta incomplete: {}", err)
        return empty
    return _reattach_delta_extras(
        delta,
        source,
        ("animal", "perplexity_ratio", "filter_drop_frac", "n_epochs"),
    )


def _robustness_frames(
    student: pd.DataFrame,
    *,
    primary: tuple[str, str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """`build_grid_delta` on every non-primary ``(set, scoring)``."""
    if student.empty or not {"set", "scoring"} <= set(student.columns):
        return {}
    endpoints = sorted(
        student[["set", "scoring"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for prompt_set, scoring in endpoints:
        if (prompt_set, scoring) == primary:
            continue
        out[(str(prompt_set), str(scoring))] = _build_delta(
            student, prompt_set=str(prompt_set), scoring=str(scoring)
        )
    return out


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _empty_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _mask_in(
    frame: pd.DataFrame, column: str, wanted: Sequence[object]
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return _empty_frame(frame.columns)
    return frame[frame[column].isin(list(wanted))].copy()


def _hp_delta(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return delta
    by_arm = _mask_in(delta, "arm", _HP_ARMS)
    if not by_arm.empty:
        return by_arm
    return _mask_in(delta, "gkd_tag", _HP_TAGS)


def _rect_delta(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty:
        return delta
    parts: list[pd.DataFrame] = []
    by_arm = _mask_in(delta, "arm", _RECT_ARMS)
    if not by_arm.empty:
        parts.append(by_arm)
    if "cell" in delta.columns:
        cross = delta[delta["cell"].astype(str) != "SI_SS"]
        if not cross.empty:
            parts.append(cross)
    refs = set(plan.REFERENCE_RUN_SETS.values())
    if "source_run_set" in delta.columns and {
        "student_step",
        "teacher_step",
    } <= set(delta.columns):
        diag = delta[
            (delta["student_step"] == delta["teacher_step"])
            & (delta["source_run_set"].astype(str).isin(refs))
        ]
        if not diag.empty:
            parts.append(diag)
    if not parts:
        return _empty_frame(delta.columns)
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    if "family" not in out.columns and "cell" in out.columns:
        out = out.copy()
        out["family"] = out["cell"]
    return out


def _cat_delta(delta: pd.DataFrame) -> pd.DataFrame:
    if delta.empty or "animal" not in delta.columns:
        return _empty_frame(delta.columns)
    return delta[delta["animal"].astype(str) == "cat"].copy()


def _hp_controls(control: pd.DataFrame) -> pd.DataFrame:
    if control.empty:
        return _empty_frame(_CONTROL_COLUMNS)
    if "gkd_tag" not in control.columns:
        return _empty_frame(control.columns)
    return control[control["gkd_tag"].astype(str).isin(_HP_TAGS)].copy()


def quarantine_frame(
    dispositions: Mapping[tuple[int, str, int], object],
    student_df: pd.DataFrame,
    *,
    ppl_max: float = plan.QUARANTINE_PPL_RATIO_MAX,
    filter_drop_max: float = plan.QUARANTINE_FILTER_DROP_MAX,
) -> pd.DataFrame:
    """Teacher-dose tiers plus leaf ppl-ratio / filter-drop bounds.

    Args:
        dispositions: `screen_dose` map ``(rung, base, epochs) → DoseRow``.
        student_df: Student frame, optionally with capability columns.
        ppl_max: Quarantine bound on ``perplexity_ratio``.
        filter_drop_max: Quarantine bound on ``filter_drop_frac``.

    Returns:
        One row per flagged teacher cell or student leaf, carrying both
        rungs, the cell, the value, and the threshold.
    """
    rows: list[dict] = []
    for disposition in dispositions.values():
        tier = getattr(disposition, "tier", "ok")
        if tier == "ok":
            continue
        reasons = getattr(disposition, "reasons", ())
        rows.append(
            {
                "scope": "teacher_dose",
                "step": getattr(disposition, "rung", None),
                "teacher_step": getattr(disposition, "rung", None),
                "base": getattr(disposition, "base", None),
                "cell": None,
                "tier": tier,
                "reason": ";".join(reasons),
                "value": getattr(disposition, "teacher_p_target", None),
                "threshold": plan.ADMISSION_ABS_MIN,
            }
        )
    if student_df is not None and not student_df.empty:
        leaves = student_df
        if "is_control" in leaves.columns:
            leaves = leaves.loc[~leaves["is_control"].astype(bool)]
        if "level" in leaves.columns:
            only_leaf = leaves[leaves["level"].astype(str) == "leaf"]
            if not only_leaf.empty:
                leaves = only_leaf
        leaf_keys = [c for c in _LEAF_IDENTITY_KEYS if c in leaves.columns]
        if not leaf_keys:
            leaf_keys = [
                c
                for c in (
                    "student_base",
                    "teacher_base",
                    "student_step",
                    "teacher_step",
                    "gkd_tag",
                    "teacher_param_tag",
                )
                if c in leaves.columns
            ]
        if leaf_keys:
            leaves = leaves.drop_duplicates(subset=leaf_keys)
        for column, threshold in (
            ("perplexity_ratio", ppl_max),
            ("filter_drop_frac", filter_drop_max),
        ):
            if column not in leaves.columns:
                continue
            bad = leaves[pd.to_numeric(leaves[column], errors="coerce") > threshold]
            for _, row in bad.iterrows():
                rows.append(
                    {
                        "scope": column,
                        "step": row.get("student_step"),
                        "teacher_step": row.get("teacher_step"),
                        "base": row.get("student_base"),
                        "cell": row.get("cell"),
                        "tier": "quarantine",
                        "reason": f"{column}>{threshold:g}",
                        "value": row.get(column),
                        "threshold": threshold,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(_QUARANTINE_COLUMNS))
    return pd.DataFrame(rows)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (ValueError, AttributeError):
            return value
    return value


def adjudicate_claims(
    *,
    ridge_df: pd.DataFrame,
    ridge_cat_df: pd.DataFrame,
    matched_df: pd.DataFrame,
    dose_response_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    onset_df: pd.DataFrame,
    emissivity_df: pd.DataFrame,
    penalty_df: pd.DataFrame,
    concordance_df: pd.DataFrame,
    triangle_df: pd.DataFrame,
    bootstrap_n: int,
    seed: int,
) -> list[dict]:
    """One claim record per H1–H12 (prereg §2).

    Args:
        ridge_df: Bear / joint `build_ridge_gain` frame.
        ridge_cat_df: Cat `build_ridge_gain` frame.
        matched_df: `build_dose_matched` frame.
        dose_response_df: `build_dose_response` frame.
        collapse_df: `build_hp_collapse` frame.
        delta_df: Primary-endpoint delta (leaf + agg).
        onset_df: `build_onset_floor` frame (not pre-filtered to agg).
        emissivity_df: `build_emissivity_fit` frame.
        penalty_df: `build_penalty_map` frame.
        concordance_df: `build_metric_concordance` frame.
        triangle_df: `build_triangle` frame.
        bootstrap_n: Bootstrap replicates for H4/H7/H8.
        seed: Bootstrap seed.

    Returns:
        Records in `plan.CLAIM_IDS` order.

    Raises:
        ValueError: The id set is not exactly `CLAIM_IDS`.
    """
    joint_n = len(plan.JOINT_RUNGS)
    cat_n = len(plan.CAT_RUNGS)
    claims = [
        cm2.adjudicate_h1(
            ridge_df,
            min_columns=plan.ridge_support_min_columns(joint_n),
            null_max=plan.ridge_null_max_columns(joint_n),
        ),
        cm2.adjudicate_h2(
            ridge_df,
            min_columns=plan.ridge_support_min_columns(joint_n),
            null_max=plan.ridge_null_max_columns(joint_n),
        ),
        cm2.adjudicate_h3(matched_df, bound=plan.DOSE_TOST_BOUND),
        cm2.adjudicate_h4(dose_response_df, n_boot=bootstrap_n, seed=seed),
        cm2.adjudicate_h5(collapse_df, delta_df, ppl_max=plan.QUARANTINE_PPL_RATIO_MAX),
        cm2.adjudicate_h6(onset_df, abs_cap=plan.ONSET_ABS_FLOOR),
        cm2.adjudicate_h7(emissivity_df, n_boot=bootstrap_n, seed=seed),
        cm2.adjudicate_h8(penalty_df, n_boot=bootstrap_n, seed=seed),
        cm2.adjudicate_h9(
            concordance_df,
            min_spearman=plan.CONCORDANCE_MIN_SPEARMAN,
            min_sign=plan.CONCORDANCE_MIN_SIGN_AGREEMENT,
        ),
        cm2.adjudicate_h10(triangle_df),
        cm2.adjudicate_h11(
            ridge_cat_df,
            support_min=plan.ridge_support_min_columns(cat_n),
            null_max=plan.ridge_null_max_columns(cat_n),
        ),
        cm2.adjudicate_h12(onset_df, student_step=plan.RECT512_STUDENT_RUNG),
    ]
    ids = [c["id"] for c in claims]
    if ids != list(plan.CLAIM_IDS):
        raise ValueError(f"claims.json would carry {ids}, not {list(plan.CLAIM_IDS)}")
    for claim in claims:
        if claim["status"] not in _CLAIM_STATUSES:
            raise ValueError(
                f"invalid claim status {claim['status']!r} for {claim['id']}"
            )
    return claims


def _attach_disk_extras(
    student: pd.DataFrame,
    output_root: Path,
    run_set_ids: Sequence[str],
) -> pd.DataFrame:
    """Join capability / training-log columns per run set."""
    frames: list[pd.DataFrame] = []
    if student.empty or "source_run_set" not in student.columns:
        return student
    for run_set_id in run_set_ids:
        part = student[student["source_run_set"].astype(str) == str(run_set_id)]
        if part.empty:
            continue
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
        try:
            cap = load_gkd_capability_results(output_root, subdir)
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("capability summaries unreadable: {}", err)
            cap = pd.DataFrame()
        try:
            logs = load_gkd_training_logs(output_root, subdir)
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("training logs unreadable: {}", err)
            logs = pd.DataFrame()
        part = _attach_optional(part, cap, ("perplexity_ratio",))
        part = _attach_optional(
            part, _mean_by_leaf(logs, "filter_drop_frac"), ("filter_drop_frac",)
        )
        frames.append(part)
    if not frames:
        return student
    return pd.concat(frames, ignore_index=True)


def build_results(
    output_root: Path,
    campaign_subdir: str,
    *,
    dry_run: bool = False,
    partial: bool = False,
    teacher_df: pd.DataFrame | None = None,
    student_df: pd.DataFrame | None = None,
    control_df: pd.DataFrame | None = None,
    base_ppl_df: pd.DataFrame | None = None,
    bootstrap_n: int = plan.BOOTSTRAP_N,
    bootstrap_seed: int = plan.BOOTSTRAP_SEED,
) -> list[Path]:
    """Write every ckptgrid2 results CSV plus `claims.json`.

    Planted frames skip disk discovery so unit tests do not need a full
    eval tree. ``partial=True`` still writes every table but stamps
    `claims.json` with ``partial: True`` and lets any claim read
    ``insufficient_data``.

    Args:
        output_root: Artifact root.
        campaign_subdir: Write target (this campaign's student tree).
        dry_run: Log paths without writing.
        partial: Deadline / incomplete finalize.
        teacher_df: Planted teacher frame.
        student_df: Planted student frame.
        control_df: Planted control frame.
        base_ppl_df: Planted untrained-PPL frame.
        bootstrap_n: Bootstrap replicates.
        bootstrap_seed: Bootstrap seed.

    Returns:
        Every path that was written (or would be, on dry-run).

    Raises:
        ValueError: A cell spans two ``source_run_set`` values, or
            `claims.json` would omit a preregistered claim id.
    """
    results = Path(output_root) / campaign_subdir / "results"
    teacher = (
        teacher_df if teacher_df is not None else assemble_teacher_frame(output_root)
    )
    student = (
        student_df if student_df is not None else assemble_student_frame(output_root)
    )
    if student_df is None:
        student = _attach_disk_extras(student, output_root, _default_student_run_sets())
    if control_df is not None:
        control = control_df
    elif "is_control" in student.columns:
        control = student.loc[student["is_control"].astype(bool)].copy()
    else:
        control = assemble_control_frame(output_root)
    base_ppl = (
        base_ppl_df if base_ppl_df is not None else assemble_base_ppl(output_root)
    )

    _assert_single_source_run_set(student)

    primary = (plan.PRIMARY_PROMPT_SET, plan.PRIMARY_SCORING)
    delta = _build_delta(student, prompt_set=primary[0], scoring=primary[1])
    robustness = _robustness_frames(student, primary=primary)
    twin = robustness.get(("test_sample", "sample"), _empty_frame(_EMPTY_DELTA_COLUMNS))
    concordance_src = (
        pd.concat([delta, twin], ignore_index=True) if not twin.empty else delta
    )

    if teacher.empty:
        dose = pd.DataFrame(columns=list(cm2.TEACHER_DOSE_COLUMNS))
    else:
        dose = cm2.build_teacher_dose(
            teacher,
            rungs=plan.DOSE_RUNGS,
            bases=cm2.DOSE_BASES,
            epochs=plan.DOSE_EPOCHS,
            prompt_set=plan.PRIMARY_PROMPT_SET,
            scoring=plan.PRIMARY_SCORING,
        )
    dose_response = cm2.build_dose_response(
        delta,
        dose,
        rungs=plan.DOSE_RUNGS,
        bases=("original",),
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    matched = cm2.build_dose_matched(
        dose_response,
        tol=plan.DOSE_MATCH_TOL,
        bound=plan.DOSE_TOST_BOUND,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    ridge = cm2.build_ridge_gain(
        delta,
        rungs=plan.JOINT_RUNGS,
        early_max_step=plan.RIDGE_ERA_EARLY_MAX_STEP,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    cat_delta = _cat_delta(delta)
    ridge_cat = cm2.build_ridge_gain(
        cat_delta if not cat_delta.empty else _empty_frame(delta.columns),
        rungs=plan.CAT_RUNGS,
        early_max_step=plan.RIDGE_ERA_EARLY_MAX_STEP,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    ridge_all = (
        pd.concat([ridge, ridge_cat], ignore_index=True)
        if not ridge_cat.empty
        else ridge
    )
    penalty = cm2.build_penalty_map(
        delta,
        reference_cell=cm2.CELL_SI_SS,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    emissivity = cm2.build_emissivity_fit(
        delta,
        per_rung=True,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    onset = cm2.build_onset_floor(
        delta,
        _control_for_onset(control),
        rungs=plan.HP_RUNGS,
        sd_multiple=plan.ONSET_SD_MULTIPLE,
        abs_floor=plan.ONSET_ABS_FLOOR,
        consecutive=plan.ONSET_CONSECUTIVE,
        markers=plan.ONSET_MARKERS,
        max_step=plan.CHECKPOINT_MAX_STEP,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    collapse = cm2.build_hp_collapse(
        delta,
        ppl_max=plan.QUARANTINE_PPL_RATIO_MAX,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    concordance = cm2.build_metric_concordance(
        concordance_src,
        primary=primary,
        twin=("test_sample", "sample"),
        subset="diagonal",
    )
    triangle = cm2.build_triangle(
        delta,
        rungs=plan.JOINT_RUNGS,
        prompt_set=plan.PRIMARY_PROMPT_SET,
        scoring=plan.PRIMARY_SCORING,
    )
    if teacher.empty:
        dispositions: Mapping[tuple[int, str, int], object] = {}
    else:
        dispositions = screen_dose(
            teacher,
            rungs=plan.DOSE_RUNGS,
            bases=cm2.DOSE_BASES,
            epochs=plan.DOSE_EPOCHS,
            prompt_set=plan.PRIMARY_PROMPT_SET,
            scoring=plan.PRIMARY_SCORING,
        )
    quarantine = quarantine_frame(dispositions, student)
    claims = adjudicate_claims(
        ridge_df=ridge,
        ridge_cat_df=ridge_cat,
        matched_df=matched,
        dose_response_df=dose_response,
        collapse_df=collapse,
        delta_df=delta,
        onset_df=onset,
        emissivity_df=emissivity,
        penalty_df=penalty,
        concordance_df=concordance,
        triangle_df=triangle,
        bootstrap_n=bootstrap_n,
        seed=bootstrap_seed,
    )

    tables: dict[str, pd.DataFrame] = {
        plan.RESULTS_TEACHER_DOSE_CSV: dose,
        plan.RESULTS_DOSE_RESPONSE_CSV: dose_response,
        plan.RESULTS_HP_GRID_DELTA_CSV: _hp_delta(delta),
        plan.RESULTS_HP_CONTROLS_CSV: _hp_controls(control),
        plan.RESULTS_HP_COLLAPSE_CSV: collapse,
        plan.RESULTS_RECT_DELTA_CSV: _rect_delta(delta),
        plan.RESULTS_RIDGE_GAIN_CSV: ridge_all,
        plan.RESULTS_PENALTY_MAP_CSV: penalty,
        plan.RESULTS_EMISSIVITY_FIT_CSV: emissivity,
        plan.RESULTS_ONSET_CSV: onset,
        plan.RESULTS_METRIC_CONCORDANCE_CSV: concordance,
        plan.RESULTS_CAT_GRID_DELTA_CSV: cat_delta,
        plan.RESULTS_BASE_PPL_CSV: base_ppl,
        plan.RESULTS_QUARANTINE_CSV: quarantine,
    }
    written = [
        _write_csv(results / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    ]
    for (prompt_set, scoring), frame in robustness.items():
        written.append(
            _write_csv(
                results / plan.RESULTS_ROBUSTNESS_DIR / f"{prompt_set}_{scoring}.csv",
                frame,
                dry_run=dry_run,
            )
        )
    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "reference_run_set_ids": list(plan.REFERENCE_RUN_SETS.values()),
        "primary_endpoint": {
            "prompt_set": plan.PRIMARY_PROMPT_SET,
            "scoring": plan.PRIMARY_SCORING,
        },
        "partial": partial,
        "coverage": {"partial": partial},
        "claims": _jsonable(claims),
    }
    claims_path = results / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        results.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, payload)
    written.append(claims_path)
    logger.info("ckptgrid2 results: {} artifacts under {}", len(written), results)
    return written


def _as_coverage_record(item: object) -> dict:
    if hasattr(item, "to_dict") and callable(item.to_dict):
        rec = dict(item.to_dict())
        if hasattr(item, "arm") and "arm" not in rec:
            rec["arm"] = item.arm
        return rec
    if isinstance(item, Mapping):
        return dict(item)
    if hasattr(item, "arm"):
        rec = {"arm": item.arm}
        if hasattr(item, "key"):
            rec["key"] = list(item.key())
        return rec
    return {"path": str(item)}


def _leaf_path_index(output_root: Path) -> dict[str, tuple[tuple, str]]:
    """``leaf_run_dir`` → ``(Cell2.key(), arm)`` for unique-cell coverage."""
    root = Path(output_root)
    index: dict[str, tuple[tuple, str]] = {}
    for leaf in pairs.leaves_of(pairs.all_cells()):
        index[str(exp2.leaf_run_dir(leaf, root))] = (
            leaf.cell.key(),
            leaf.cell.arm,
        )
    return index


def _missing_by_arm(
    missing_recs: list[dict],
    output_root: Path,
) -> Counter[str]:
    """Count unrun *cells* per arm; leaf paths collapse onto `Cell2.key()`."""
    counts: Counter[str] = Counter()
    seen_keys: set[tuple] = set()
    index = _leaf_path_index(output_root)
    for rec in missing_recs:
        path = rec.get("path")
        if path is not None and str(path) in index:
            key, arm = index[str(path)]
            rec["arm"] = arm
            if key in seen_keys:
                continue
            seen_keys.add(key)
            counts[str(arm)] += 1
            continue
        if rec.get("arm") is not None:
            counts[str(rec["arm"])] += 1
    return counts


def write_coverage(
    output_root: Path,
    campaign_subdir: str,
    *,
    missing_cells: Sequence[object],
    missing_evals: Sequence[object],
    holes: Sequence[object],
    deadline_windows: Sequence[object],
    dry_run: bool,
) -> Path:
    """Write `results/coverage.json` with per-arm designed-vs-present counts.

    Every section is always present so a partial finalize cannot silently
    omit unrun cells, tolerated holes, or deadline windows.

    Args:
        output_root: Artifact root.
        campaign_subdir: Write target (this campaign's student tree).
        missing_cells: Unrun cells (paths, dicts with ``arm``, or
            `Cell2`).
        missing_evals: Config dirs whose eval triad is incomplete.
        holes: Tolerated failed-queue records.
        deadline_windows: Open/closed deadline windows.
        dry_run: Log the path without writing.

    Returns:
        Path of `coverage.json`.
    """
    designed = Counter(cell.arm for cell in pairs.all_cells())
    missing_recs = [_as_coverage_record(item) for item in missing_cells]
    missing_by_arm = _missing_by_arm(missing_recs, Path(output_root))
    per_arm = {
        arm: {
            "designed": int(count),
            "missing": int(missing_by_arm.get(arm, 0)),
            "present": int(max(0, count - missing_by_arm.get(arm, 0))),
        }
        for arm, count in designed.items()
    }
    payload = {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "per_arm": per_arm,
        "unrun_cells": missing_recs,
        "missing_evals": [_as_coverage_record(item) for item in missing_evals],
        "holes": [
            dict(item) if isinstance(item, Mapping) else {"value": item}
            for item in holes
        ],
        "deadline_windows": [
            dict(item) if isinstance(item, Mapping) else {"value": item}
            for item in deadline_windows
        ],
    }
    path = Path(output_root) / campaign_subdir / "results" / plan.RESULTS_COVERAGE_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return path
