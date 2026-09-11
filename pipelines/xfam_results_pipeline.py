"""Assemble cross-family frames and persist F0-F7 claim artifacts.

Leaf / cell / claim schemas come from ``campaign.xfam_metrics``; this module is
the IO + assembly layer and re-implements no statistic. Claim keys must be
exactly `xfam_plan.CLAIM_IDS` or this module raises.

The student tree is walked here rather than through
`evaluation.gkd_results_io.load_gkd_*`: those loaders resolve a teacher-base
control's emitter with `teachers.model_specs.spec_from_base_name`, which only
knows PolyPythias repo names and raises on every cross-family roster label. The
tag parsers of that module (`parse_gkd_tag` / `parse_teacher_tag`) are reused as
they are; only the label space is xfam's own, and a run dir's ``{model}``
component IS the roster label for an `HfModelSpec`.

A partial tree is the normal mid-campaign state: a cell with no leaves is absent
from the tables and reported ``void`` in the claim it feeds, never an exception.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from subliminal_learning.campaign import xfam_metrics as fm
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign.xfam_plan import (
    ANIMALS,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CLAIM_IDS,
    LR_LADDERS,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    REPLICATION_GATE_FILE,
    RESULTS_ASYMMETRY_CSV,
    RESULTS_BASE_PRIORS_CSV,
    RESULTS_CELLS_CSV,
    RESULTS_CLAIMS_JSON,
    RESULTS_DELTA_CSV,
    RESULTS_HP_CSV,
    RESULTS_LADDER_CSV,
    RESULTS_LEAVES_CSV,
    RESULTS_PENALTY_CSV,
    RESULTS_POSTTRAIN_CSV,
    RESULTS_PPL_CSV,
    RESULTS_QUARANTINE_CSV,
    RESULTS_TEACHER_ADMISSION_CSV,
    RUN_SET_ID,
    SCREEN_PICKS_FILE,
    TEACHER_GATE_REPORT,
    TEACHER_PARAM_TAG,
    TEACHER_RUN_SET_ID,
    ladder_arms,
)
from subliminal_learning.campaign.xfam_plan import arm as arm_of_name
from subliminal_learning.campaign.xfam_selection import (
    TeacherAdmission,
    TeacherGateDecision,
    admit_teacher,
    pick_lr,
    picks_from_report,
    replications_from_report,
    teacher_gate,
)
from subliminal_learning.evaluation.gkd_results_io import (
    parse_gkd_tag,
    parse_teacher_tag,
)
from subliminal_learning.evaluation.sdf_results_io import (
    load_sdf_capability_results,
    load_sdf_preference_results,
)
from subliminal_learning.gkd.layout import (
    CONTROL_TEACHER_TAG,
    GKD_STUDENTS_SUBDIR,
    XBASE_CONTROL_TEACHER_TAG,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json

PREFERENCE_FILE = "preference_summary.json"
CAPABILITY_FILE = "sdf_capability_summary.json"
TRAINING_LOG_FILE = "trainer_log_history.jsonl"
FILTER_DROP_FIELD = "gkd/filter_drop_frac"
BASE_CONTROL_FILE = "base_control.json"
BASE_METRICS_FILE = "base_metrics.json"

TEACHER_ADMISSION_COLUMNS: tuple[str, ...] = (
    "label",
    "cluster",
    "tag",
    "ftseed",
    "p_target",
    "base_prior",
    "ppl_ratio",
    "admitted",
    "reason",
)
QUARANTINE_COLUMNS: tuple[str, ...] = (*fm.LEAF_COLUMNS, "quarantine_reason")


# ---- locations ---------------------------------------------------------------


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
    """Resolve `output_root` from an ExperimentConfig or a driver config."""
    if hasattr(cfg, "output_root"):
        return Path(cfg.output_root)
    paths = getattr(cfg, "paths", None)
    if paths is not None and hasattr(paths, "output_root"):
        return Path(paths.output_root)
    raise TypeError(
        f"cfg must expose output_root or paths.output_root; got {type(cfg)!r}"
    )


# ---- small helpers -----------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _per_seed_items(raw: object) -> list[tuple[int, float]]:
    """``{"0": 0.30, ...}`` -> ``[(0, 0.30), ...]``."""
    if not isinstance(raw, Mapping):
        return []
    return sorted(
        (int(seed), float(value)) for seed, value in raw.items() if _finite(value)
    )


def _primary_summary(doc: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(doc, Mapping):
        return None
    for summary in doc.get("summaries") or []:
        if (
            str(summary.get("prompt_set")) == PRIMARY_PROMPT_SET
            and str(summary.get("scoring")) == PRIMARY_SCORING
        ):
            return summary
    return None


def _perplexity_ratio(config_dir: Path) -> float:
    doc = _read_json(config_dir / CAPABILITY_FILE)
    if not isinstance(doc, Mapping):
        return float("nan")
    ratio = (doc.get("perplexity") or {}).get("perplexity_ratio")
    return float(ratio) if _finite(ratio) else float("nan")


def _filter_drop(leaf_dir: Path) -> float:
    """Mean layer-2 rollout drop fraction over one leaf's logged steps."""
    path = leaf_dir / TRAINING_LOG_FILE
    if not path.is_file():
        return float("nan")
    values: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _finite(record.get(FILTER_DROP_FIELD)):
            values.append(float(record[FILTER_DROP_FIELD]))
    return float(np.mean(values)) if values else float("nan")


# ---- student tree ------------------------------------------------------------


def _config_dirs(
    root: Path,
    *,
    animals: Sequence[str],
) -> Iterator[tuple[Path, str, str, str, str, str | None]]:
    """Yield ``(dir, animal, model, teacher_tag, gkd_tag, teacher_model)`` config dirs.

    Treatments sit at ``{animal}/{model}/{teacher_tag}/{gkd_tag}``, student-base
    controls at ``controls/base_teacher/{animal}/{model}/{gkd_tag}`` and
    teacher-base controls one level deeper, under ``{teacher_model}``.
    """
    if not root.is_dir():
        return
    wanted = set(animals)
    for path in sorted(root.glob("*/*/*/*")):
        if not path.is_dir() or path.relative_to(root).parts[0] == "controls":
            continue
        animal = path.parents[2].name
        if animal in wanted:
            yield path, animal, path.parents[1].name, path.parent.name, path.name, None
    for path in sorted((root / "controls" / CONTROL_TEACHER_TAG).glob("*/*/*")):
        if not path.is_dir():
            continue
        animal = path.parents[1].name
        if animal in wanted:
            yield path, animal, path.parent.name, "", path.name, None
    for path in sorted((root / "controls" / XBASE_CONTROL_TEACHER_TAG).glob("*/*/*/*")):
        if not path.is_dir():
            continue
        animal = path.parents[2].name
        if animal in wanted:
            yield path, animal, path.parents[1].name, "", path.name, path.parent.name


def _leaf_identity(
    *,
    model: str,
    teacher_tag_str: str,
    gkd_tag_str: str,
    teacher_model: str | None,
) -> dict | None:
    """Roster + hyperparameter identity of one config dir, or None if unreadable."""
    if model not in models.ROSTER:
        raise ValueError(f"student dir {model!r} is not an xfam roster label")
    try:
        gkd = parse_gkd_tag(gkd_tag_str)
    except ValueError:
        logger.warning("skipping unparseable gkd_tag {!r}", gkd_tag_str)
        return None
    if teacher_tag_str:
        parsed = parse_teacher_tag(teacher_tag_str)
        if parsed["teacher_param_tag"] is None:
            logger.warning("skipping unparseable teacher_tag {!r}", teacher_tag_str)
            return None
        teacher = str(parsed["teacher_base_label"] or model)
        kind, ftseed = fm.KIND_TREATMENT, int(parsed["teacher_ftseed"])
        param_tag = str(parsed["teacher_param_tag"])
        arm = fm.arm_of_leaf(
            teacher=teacher,
            student=model,
            teacher_param_tag=param_tag,
            ftseed=ftseed,
            lr=float(gkd["student_lr"]),
            lora_r=int(gkd["student_lora_r"]),
            lora_alpha=int(gkd["student_lora_alpha"]),
            epochs=float(gkd["num_epochs"]),
        )
        arm_kind = arm_of_name(arm).kind
    elif teacher_model is None:
        teacher, kind = model, fm.KIND_SB
        ftseed, param_tag, arm, arm_kind = float("nan"), "", "", ""
    else:
        if teacher_model not in models.ROSTER:
            raise ValueError(
                f"teacher-base control dir {teacher_model!r} is not a roster label"
            )
        teacher, kind = teacher_model, fm.KIND_TB
        ftseed, param_tag, arm, arm_kind = float("nan"), "", "", ""
    return {
        "cluster": models.cluster_of(model),
        "student": model,
        "teacher": teacher,
        "kind": kind,
        "arm": arm,
        "arm_kind": arm_kind,
        "pair_kind": fm.pair_kind_of(teacher, model),
        "teacher_param_tag": param_tag,
        "lr": float(gkd["student_lr"]),
        "epochs": float(gkd["num_epochs"]),
        "lora_r": int(gkd["student_lora_r"]),
        "lora_alpha": int(gkd["student_lora_alpha"]),
        "gkd_tag": gkd_tag_str,
        "ftseed": ftseed,
    }


def assemble_student_frame(
    output_root: Path,
    campaign_subdir_name: str,
    *,
    animals: Sequence[str] = ANIMALS,
) -> pd.DataFrame:
    """GKD preference summaries under one run set -> `xfam_metrics.LEAF_COLUMNS` rows.

    One row per gseed, exploded from ``per_seed_p_target``; the config's
    ``perplexity_ratio`` and each gseed's mean ``gkd/filter_drop_frac`` ride
    along.

    Raises:
        ValueError: A model or teacher-base directory is not a roster label, a
            treatment maps to no arm, or a primary summary carries no per-gseed
            scores.
    """
    root = Path(output_root) / campaign_subdir_name
    rows: list[dict] = []
    for config_dir, animal, model, ttag, gtag, tmodel in _config_dirs(
        root, animals=animals
    ):
        summary = _primary_summary(_read_json(config_dir / PREFERENCE_FILE))
        if summary is None:
            continue
        identity = _leaf_identity(
            model=model,
            teacher_tag_str=ttag,
            gkd_tag_str=gtag,
            teacher_model=tmodel,
        )
        if identity is None:
            continue
        seeds = _per_seed_items(summary.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                "preference summary missing per_seed_p_target "
                f"(animal={animal!r} model={model!r} gkd_tag={gtag!r})"
            )
        ppl_ratio = _perplexity_ratio(config_dir)
        for gseed, p_target in seeds:
            rows.append(
                {
                    **identity,
                    "animal": animal,
                    "gseed": gseed,
                    "p_target": p_target,
                    "perplexity_ratio": ppl_ratio,
                    "filter_drop": _filter_drop(config_dir / f"gseed{gseed}"),
                    "quarantined": False,
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(fm.LEAF_COLUMNS))
    frame = pd.DataFrame(rows)
    identity_keys = [
        "animal",
        "student",
        "teacher",
        "kind",
        "teacher_param_tag",
        "gkd_tag",
        "ftseed",
        "gseed",
    ]
    if frame.duplicated(subset=identity_keys).any():
        raise ValueError(f"duplicate leaf identity on {identity_keys}")
    return frame.loc[:, list(fm.LEAF_COLUMNS)]


# ---- teachers ----------------------------------------------------------------


def assemble_teacher_frame(
    output_root: Path,
    *,
    teacher_run_set_id: str = TEACHER_RUN_SET_ID,
    animals: Sequence[str] = ANIMALS,
) -> pd.DataFrame:
    """SDF preference summaries -> one admission row per ``(label, tag, ftseed)``."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, teacher_run_set_id)
    pref = load_sdf_preference_results(output_root, subdir, animals=list(animals))
    empty = pd.DataFrame(columns=list(TEACHER_ADMISSION_COLUMNS))
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
    for _, row in cap.iterrows():
        if not _finite(row.get("perplexity_ratio")):
            continue
        key = (str(row["animal"]), str(row["model"]), str(row["param_tag"]))
        if key in ppl_by_cfg:
            raise ValueError(f"duplicate teacher capability row for {key}")
        ppl_by_cfg[key] = float(row["perplexity_ratio"])

    rows: list[dict] = []
    for _, row in primary.iterrows():
        label = str(row["model"])
        if label not in models.ROSTER:
            raise ValueError(f"teacher dir {label!r} is not an xfam roster label")
        seeds = _per_seed_items(row.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                "teacher preference summary missing per_seed_p_target "
                f"(model={label!r} param_tag={row.get('param_tag')!r})"
            )
        control = row.get("base_p_target")
        control_mean = float(control) if _finite(control) else None
        tag = str(row["param_tag"])
        ppl = ppl_by_cfg.get((str(row["animal"]), label, tag))
        for ftseed, p_target in seeds:
            admitted, reasons = admit_teacher(
                p_target=p_target,
                control_mean=control_mean,
                control_sd=None,
                ppl_ratio=ppl,
            )
            rows.append(
                {
                    "label": label,
                    "cluster": models.cluster_of(label),
                    "tag": tag,
                    "ftseed": int(ftseed),
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
    return pd.DataFrame(rows, columns=list(TEACHER_ADMISSION_COLUMNS))


def _admissions_from_frame(frame: pd.DataFrame) -> list[TeacherAdmission]:
    if frame.empty:
        return []
    return [
        TeacherAdmission(
            label=str(row["label"]),
            tag=str(row["tag"]),
            ftseed=int(row["ftseed"]),
            admitted=bool(row["admitted"]),
            ppl_ratio=(
                float(row["ppl_ratio"]) if _finite(row.get("ppl_ratio")) else None
            ),
            p_target=(float(row["p_target"]) if _finite(row.get("p_target")) else None),
            control_mean=(
                float(row["base_prior"]) if _finite(row.get("base_prior")) else None
            ),
            control_sd=None,
            reasons=(
                tuple(str(row["reason"]).split(";")) if str(row["reason"]) else ()
            ),
        )
        for _, row in frame.iterrows()
    ]


def _decisions_from_report(doc: Mapping[str, Any]) -> dict[str, TeacherGateDecision]:
    raw = doc.get("decisions")
    if not isinstance(raw, Mapping):
        raise ValueError("teacher_gate_report is missing a decisions mapping")
    out: dict[str, TeacherGateDecision] = {}
    for label, entry in raw.items():
        if not isinstance(entry, Mapping) or "usable" not in entry:
            raise ValueError(f"teacher_gate_report[{label!r}] has no usable flag")
        out[str(label)] = TeacherGateDecision(
            label=str(entry.get("label", label)),
            role=str(entry.get("role", "cluster")),
            admitted_ftseeds=tuple(int(v) for v in entry.get("admitted_ftseeds") or ()),
            usable=bool(entry["usable"]),
            reason=str(entry.get("reason", "")),
        )
    return out


def _load_teacher_decisions(
    campaign_logs: Path,
    admissions: Sequence[TeacherAdmission],
) -> dict[str, TeacherGateDecision]:
    path = campaign_logs / TEACHER_GATE_REPORT
    doc = _read_json(path) if path.is_file() else None
    if doc is not None:
        return _decisions_from_report(doc)
    return teacher_gate(admissions) if admissions else {}


def _teacher_dose(
    admissions: Sequence[TeacherAdmission],
    decisions: Mapping[str, TeacherGateDecision],
) -> dict[str, float]:
    """``label -> mean p(target)`` over its frozen-tag leaves.

    Admitted leaves only where the emitter cleared its gate; an emitter the gate
    rejected still gets a dose from every frozen-tag leaf, so F1 efficiency and
    F4 dose normalisation stay readable on a void arm.
    """
    dose: dict[str, float] = {}
    frozen = [
        row
        for row in admissions
        if row.tag == TEACHER_PARAM_TAG and row.p_target is not None
    ]
    for label in sorted({row.label for row in frozen}):
        decision = decisions.get(label)
        leaves = [row for row in frozen if row.label == label]
        if decision is None or decision.usable:
            admitted = [row for row in leaves if row.admitted]
            leaves = admitted or leaves
        dose[label] = float(np.mean([float(row.p_target) for row in leaves]))
    return dose


# ---- persisted decisions -----------------------------------------------------


def _picks_on_disk(campaign_logs: Path) -> dict[str, float]:
    """The driver's persisted ``lr*``, or empty when it has not picked yet."""
    doc = _read_json(campaign_logs / SCREEN_PICKS_FILE)
    return picks_from_report(doc) if doc is not None else {}


def _replications_on_disk(campaign_logs: Path) -> dict[str, bool]:
    doc = _read_json(campaign_logs / REPLICATION_GATE_FILE)
    return replications_from_report(doc) if doc is not None else {}


def _picks_from_ladder(deltas: pd.DataFrame) -> dict[str, float]:
    """``lr*`` re-derived in memory from the ladder arms, for a pre-pick read."""
    if deltas.empty or "delta_primary" not in deltas.columns:
        return {}
    picks: dict[str, float] = {}
    for student, ladder in LR_LADDERS.items():
        arms = {a.name for a in ladder_arms() if student in a.students}
        sub = deltas[
            (deltas["student"].astype(str) == student)
            & (deltas["arm"].astype(str).isin(arms))
        ]
        if sub.empty:
            continue
        delta_by_lr: dict[float, float] = {}
        ppl_by_lr: dict[float, float] = {}
        for lr in ladder:
            at_lr = sub[np.isclose(sub["lr"].astype(float), float(lr))]
            if at_lr.empty:
                break
            delta_by_lr[lr] = float(at_lr["delta_primary"].mean())
            ppl_by_lr[lr] = float(
                pd.to_numeric(at_lr["perplexity_ratio"], errors="coerce").mean()
            )
        if len(delta_by_lr) != len(ladder):
            continue
        picks[student] = pick_lr(
            student=student, delta_by_lr=delta_by_lr, ppl_by_lr=ppl_by_lr
        ).lr
    for student, spec in models.ROSTER.items():
        if student in picks or spec.lineage is None:
            continue
        if spec.lineage in picks:
            picks[student] = picks[spec.lineage]
    return picks


def _load_or_compute_picks(
    campaign_logs: Path,
    deltas: pd.DataFrame,
) -> dict[str, float]:
    on_disk = _picks_on_disk(campaign_logs)
    if on_disk:
        return on_disk
    logger.warning(
        "no persisted pick at {} — deriving lr* in memory from the ladder arms",
        campaign_logs / SCREEN_PICKS_FILE,
    )
    return _picks_from_ladder(deltas)


# ---- base priors -------------------------------------------------------------


def _priors_from_base_control(path: Path, label: str) -> dict:
    doc = _read_json(path)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    block = doc.get(PRIMARY_PROMPT_SET)
    if not isinstance(block, Mapping):
        block = doc
    if "p_target" not in block:
        raise ValueError(f"{path} has no p_target for {PRIMARY_PROMPT_SET!r}")
    return {"label": label, "p_target": block["p_target"]}


def _priors_from_base_metrics(path: Path, label: str) -> dict:
    doc = _read_json(path)
    if not isinstance(doc, Mapping):
        raise ValueError(f"{path} is not a mapping")
    record: dict[str, Any] = {"label": label}
    forced = doc.get("forced_choice")
    if isinstance(forced, Mapping):
        result = forced.get("result", forced)
        if isinstance(result, Mapping) and "target_share_mean" in result:
            record["fc_share"] = result["target_share_mean"]
    perplexity = doc.get("perplexity")
    if isinstance(perplexity, Mapping):
        result = perplexity.get("result", perplexity)
        if isinstance(result, Mapping) and "overall" in result:
            record["perplexity"] = result["overall"]
    if len(record) == 1:
        raise ValueError(f"{path} has neither a forced-choice share nor a perplexity")
    return record


def assemble_base_prior_rows(
    output_root: Path,
    teacher_frame: pd.DataFrame,
    *,
    teacher_run_set_id: str = TEACHER_RUN_SET_ID,
) -> list[dict]:
    """Untuned-base prior dicts for `xfam_metrics.base_priors`.

    Reads ``{model}/base_control.json`` and ``{model}/base_metrics.json`` under
    the SDF teacher tree, then the teacher frame's own ``base_prior`` column.
    """
    rows: list[dict] = []
    teacher_root = Path(output_root) / run_set_subdir(
        SDF_TEACHERS_SUBDIR, teacher_run_set_id
    )
    if teacher_root.is_dir():
        for path in sorted(teacher_root.glob(f"*/*/{BASE_CONTROL_FILE}")):
            rows.append(_priors_from_base_control(path, path.parent.name))
        for path in sorted(teacher_root.glob(f"*/*/{BASE_METRICS_FILE}")):
            rows.append(_priors_from_base_metrics(path, path.parent.name))
    if not teacher_frame.empty:
        for label, group in teacher_frame.groupby("label", dropna=False):
            values = pd.to_numeric(group["base_prior"], errors="coerce").dropna()
            if not values.empty:
                rows.append({"label": str(label), "p_target": float(values.mean())})
    return rows


# ---- assembly ----------------------------------------------------------------


def quarantine_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """The flagged rows of `leaves`, with their ``quarantine_reason``."""
    if leaves.empty or "quarantined" not in leaves.columns:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    flagged = leaves.loc[leaves["quarantined"].fillna(False).astype(bool)]
    if flagged.empty:
        return pd.DataFrame(columns=list(QUARANTINE_COLUMNS))
    return flagged.loc[:, [c for c in QUARANTINE_COLUMNS if c in flagged.columns]]


def adjudicate_claims(cells: pd.DataFrame, **kwargs: Any) -> dict:
    """F0-F7 records from `xfam_metrics.claims`.

    Raises:
        ValueError: The payload keys are not exactly `CLAIM_IDS`.
    """
    payload = fm.claims(cells, **kwargs)
    if tuple(payload) != CLAIM_IDS:
        raise ValueError(f"claims.json would carry {tuple(payload)}, not {CLAIM_IDS}")
    return payload


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


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
    """Write every cross-family results CSV plus `claims.json`.

    Re-runnable at any point in the campaign: cells with no leaves on disk are
    simply absent from the tables and read ``void`` in the claims.

    Args:
        cfg: ExperimentConfig (``.paths.output_root``) or a driver config
            (``.output_root``).
        dry_run: Compute every frame and write nothing.
        bootstrap_n: Cluster-bootstrap draws forwarded to `xfam_metrics`.
        bootstrap_seed: Bootstrap RNG seed.

    Returns:
        Filename -> path for every artifact (including dry-run paths).

    Raises:
        KeyError: A live treatment leaf has no matched control.
        ValueError: A leaf, arm or claim schema does not match the plan.
    """
    output_root = _output_root_of(cfg)
    subdir = campaign_subdir()
    dest = results_dir(output_root, subdir)
    campaign_logs = logs_dir(output_root, subdir)

    leaves = fm.propagate_control_quarantine(
        fm.quarantine(assemble_student_frame(output_root, subdir))
    )
    teacher = assemble_teacher_frame(output_root)
    admissions = _admissions_from_frame(teacher)
    decisions = _load_teacher_decisions(campaign_logs, admissions)

    deltas = fm.leaf_deltas(leaves)
    picks = _load_or_compute_picks(campaign_logs, deltas)
    replications = _replications_on_disk(campaign_logs)
    cells = fm.cell_means(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    dose = _teacher_dose(admissions, decisions)

    penalty = fm.architecture_penalty(
        cells,
        picks=picks,
        deltas=deltas,
        teacher_p_target=dose,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    ladder = fm.init_distance_ladder(cells, picks=picks)
    posttrain = fm.posttrain_contrast(
        cells,
        picks=picks,
        deltas=deltas,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    asymmetry = fm.direction_asymmetry(
        cells,
        picks=picks,
        deltas=deltas,
        teacher_p_target=dose,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    hp = fm.hp_robustness(cells, picks=picks)
    ppl = fm.ppl_coupling(deltas)
    priors = fm.base_priors(assemble_base_prior_rows(output_root, teacher))
    claims = adjudicate_claims(
        cells,
        picks=picks,
        teacher_decisions=decisions,
        replications=replications,
        penalty=penalty,
        ladder=ladder,
        posttrain=posttrain,
        asymmetry=asymmetry,
        hp=hp,
        ppl=ppl,
    )

    tables: dict[str, pd.DataFrame] = {
        RESULTS_LEAVES_CSV: leaves.loc[:, list(fm.LEAF_COLUMNS)],
        RESULTS_DELTA_CSV: deltas,
        RESULTS_CELLS_CSV: cells,
        RESULTS_PENALTY_CSV: penalty,
        RESULTS_LADDER_CSV: ladder,
        RESULTS_POSTTRAIN_CSV: posttrain,
        RESULTS_ASYMMETRY_CSV: asymmetry,
        RESULTS_HP_CSV: hp,
        RESULTS_PPL_CSV: ppl,
        RESULTS_QUARANTINE_CSV: quarantine_frame(leaves),
        RESULTS_TEACHER_ADMISSION_CSV: teacher,
        RESULTS_BASE_PRIORS_CSV: priors,
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
    logger.info("xfam results: {} artifacts under {}", len(written), dest)
    return written
