"""Assemble xfam wave-2 frames and persist the Y0-Y11 claim artifacts.

Leaf / cell / claim schemas come from ``campaign.xfam2_metrics``; this module is the
IO + assembly layer and re-implements no statistic. Claim keys must be exactly
`xfam2_plan.CLAIM_IDS`.

The student tree is walked with Part I's config-dir walker and summary readers
(`pipelines.xfam_results_pipeline`); per-gseed files are read by their exact
``gseed{N}`` path, so quarantined sibling copies (``gseed1.<utc>_doublewrite``) are
never read. Two things differ from Part I: the perplexity ratio is the leaf's own
gseed (``per_seed_overall``), and every leaf carries its first-window reverse KL.

A partial tree is the normal state at a deadline read: a cell with no leaves reads
``incomplete``. A treatment whose primary control is missing raises only when that
control's queue has finished; otherwise the leaf is reported and skipped.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from pipelines.xfam_results_pipeline import (
    CAPABILITY_FILE,
    PREFERENCE_FILE,
    TRAINING_LOG_FILE,
    _config_dirs,
    _filter_drop,
    _jsonable,
    _per_seed_items,
    _primary_summary,
    _read_json,
    assemble_base_prior_rows,
)
from subliminal_learning.campaign import xfam2_metrics as ym
from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.xfam_metrics import base_priors
from subliminal_learning.campaign.xsize_selection import admit_teacher
from subliminal_learning.config import load_config
from subliminal_learning.evaluation.gkd_results_io import (
    parse_gkd_tag,
    parse_teacher_tag,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

REV_KL_FIELD = "gkd/rev_kl"
PART1_DELTA_CSV = "xfam_delta.csv"
LOCAL_PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
MODEL_CONFIG = Path("configs/model/pythia.yaml")

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


# ---- locations -------------------------------------------------------------------


def campaign_subdir(run_set_id: str = plan.RUN_SET_ID) -> str:
    """``gkd_students/{run_set_id}`` under `output_root`."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)


def results_dir(output_root: Path) -> Path:
    """Campaign results directory."""
    return Path(output_root) / campaign_subdir() / "results"


def logs_dir(output_root: Path) -> Path:
    """Campaign logs directory."""
    return Path(output_root) / campaign_subdir() / "logs"


def part1_results_dir(repo_root: Path) -> Path | None:
    """Part I's results dir from the machine-local paths YAML, or None if absent."""
    paths_yaml = Path(repo_root) / LOCAL_PATHS_CONFIG
    if not paths_yaml.is_file():
        return None
    try:
        cfg = load_config(paths_yaml, Path(repo_root) / MODEL_CONFIG)
    except (FileNotFoundError, KeyError, ValueError) as err:
        logger.warning("Part I paths unreadable ({}); Y9 is void", err)
        return None
    return (
        Path(cfg.paths.output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.PART1_RUN_SET_ID)
        / "results"
    )


# ---- per-leaf readers --------------------------------------------------------------


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def rev_kl_first(leaf_dir: Path) -> float:
    """``gkd/rev_kl`` of the lowest-step record in one leaf's trainer log."""
    path = leaf_dir / TRAINING_LOG_FILE
    if not path.is_file():
        return float("nan")
    best: tuple[float, float] | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        value, step = record.get(REV_KL_FIELD), record.get("step")
        if _finite(value) and _finite(step) and (best is None or step < best[0]):
            best = (float(step), float(value))
    return best[1] if best else float("nan")


def _seed_ppl(config_dir: Path) -> dict[int, float]:
    doc = _read_json(config_dir / CAPABILITY_FILE)
    if not isinstance(doc, Mapping):
        return {}
    block = doc.get("perplexity") or {}
    return ym.per_seed_ppl_ratio(
        block.get("per_seed_overall"), block.get("base_overall")
    )


def _identity(
    model: str, ttag: str, gtag: str, tmodel: str | None
) -> dict[str, Any] | None:
    if model not in models.ROSTER:
        raise ValueError(f"student dir {model!r} is not an xfam2 roster label")
    try:
        gkd = parse_gkd_tag(gtag)
    except ValueError:
        logger.warning("skipping unparseable gkd_tag {!r}", gtag)
        return None
    if ttag:
        parsed = parse_teacher_tag(ttag)
        if parsed["teacher_param_tag"] is None:
            logger.warning("skipping unparseable teacher_tag {!r}", ttag)
            return None
        teacher = str(parsed["teacher_base_label"] or model)
        kind, ftseed = ym.KIND_TREATMENT, int(parsed["teacher_ftseed"])
        param_tag = str(parsed["teacher_param_tag"])
    elif tmodel is None:
        teacher, kind, ftseed, param_tag = model, ym.KIND_SB, float("nan"), ""
    else:
        if tmodel not in models.ROSTER:
            raise ValueError(
                f"teacher-base control dir {tmodel!r} is not a roster label"
            )
        teacher, kind, ftseed, param_tag = tmodel, ym.KIND_TB, float("nan"), ""
    return {
        "cluster": models.cluster_of(model),
        "student": model,
        "teacher": teacher,
        "kind": kind,
        "pair_kind": ym.pair_kind_of(teacher, model),
        "teacher_param_tag": param_tag,
        "lr": float(gkd["student_lr"]),
        "gkd_tag": gtag,
        "ftseed": ftseed,
    }


def assemble_student_frame(output_root: Path) -> pd.DataFrame:
    """Every trained leaf of the run set as `xfam2_metrics.LEAF_COLUMNS` rows.

    Raises:
        ValueError: A directory names a label outside the roster, a leaf maps to no
            planned cell, a primary summary lacks per-gseed scores, or two leaves
            share an identity.
    """
    root = Path(output_root) / campaign_subdir()
    rows: list[dict] = []
    for config_dir, animal, model, ttag, gtag, tmodel in _config_dirs(
        root, animals=plan.ANIMALS
    ):
        summary = _primary_summary(_read_json(config_dir / PREFERENCE_FILE))
        if summary is None:
            continue
        identity = _identity(model, ttag, gtag, tmodel)
        if identity is None:
            continue
        seeds = _per_seed_items(summary.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                f"preference summary missing per_seed_p_target ({config_dir})"
            )
        ppl = _seed_ppl(config_dir)
        for gseed, p_target in seeds:
            block, tier = ym.unit_of(
                identity["teacher"], identity["student"], identity["lr"], gseed
            )
            leaf = config_dir / f"gseed{gseed}"
            rows.append(
                {
                    **identity,
                    "animal": animal,
                    "block": block,
                    "tier": tier,
                    "arm": block,
                    "gseed": int(gseed),
                    "p_target": p_target,
                    "perplexity_ratio": ppl.get(int(gseed), float("nan")),
                    "filter_drop": _filter_drop(leaf),
                    "rev_kl_first": rev_kl_first(leaf),
                    "teacher_admitted": True,
                    "quarantined": False,
                    "quarantine_reason": "",
                }
            )
    if not rows:
        return pd.DataFrame(columns=list(ym.LEAF_COLUMNS))
    frame = pd.DataFrame(rows)
    keys = ["animal", "student", "teacher", "kind", "gkd_tag", "ftseed", "gseed"]
    if frame.fillna({"ftseed": -1}).duplicated(subset=keys).any():
        raise ValueError(f"duplicate leaf identity on {keys}")
    return frame.loc[:, list(ym.LEAF_COLUMNS)]


# ---- teachers ------------------------------------------------------------------------


def assemble_teacher_frame(output_root: Path) -> pd.DataFrame:
    """One admission row per ``(label, tag, ftseed)`` from the SDF teacher summaries.

    Each ftseed gets its own perplexity ratio (``per_seed_overall``) where the
    summary has one; Part I keyed it per tag.
    """
    root = Path(output_root) / run_set_subdir(
        SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID
    )
    rows: list[dict] = []
    for animal in plan.ANIMALS:
        for tag_dir in sorted((root / animal).glob("*/*")):
            if not tag_dir.is_dir():
                continue
            label, tag = tag_dir.parent.name, tag_dir.name
            summary = _primary_summary(_read_json(tag_dir / PREFERENCE_FILE))
            if summary is None:
                continue
            if label not in models.ROSTER:
                raise ValueError(f"teacher dir {label!r} is not an xfam2 roster label")
            seeds = _per_seed_items(summary.get("per_seed_p_target"))
            if not seeds:
                raise ValueError(
                    f"teacher summary missing per_seed_p_target ({tag_dir})"
                )
            control = summary.get("base_p_target")
            control_mean = float(control) if _finite(control) else None
            per_seed = _seed_ppl(tag_dir)
            cap = _read_json(tag_dir / CAPABILITY_FILE) or {}
            tag_ratio = (cap.get("perplexity") or {}).get("perplexity_ratio")
            for ftseed, p_target in seeds:
                ratio = per_seed.get(int(ftseed))
                if ratio is None and _finite(tag_ratio):
                    ratio = float(tag_ratio)
                admitted, reasons = admit_teacher(
                    p_target=p_target,
                    control_mean=control_mean,
                    control_sd=None,
                    ppl_ratio=ratio,
                    sd_multiple=plan.ADMISSION_SD_MULTIPLE,
                    abs_min=plan.ADMISSION_ABS_MIN,
                    ppl_max=plan.TEACHER_PPL_RATIO_MAX,
                )
                rows.append(
                    {
                        "label": label,
                        "cluster": models.cluster_of(label),
                        "tag": tag,
                        "ftseed": int(ftseed),
                        "p_target": float(p_target),
                        "base_prior": control_mean
                        if control_mean is not None
                        else float("nan"),
                        "ppl_ratio": ratio if ratio is not None else float("nan"),
                        "admitted": admitted,
                        "reason": ";".join(reasons),
                    }
                )
    return pd.DataFrame(rows, columns=list(TEACHER_ADMISSION_COLUMNS))


def load_decisions(campaign_logs: Path, teacher: pd.DataFrame) -> dict[str, dict]:
    """Persisted per-label gate decisions, else the quorum rule on `teacher`."""
    doc = _read_json(campaign_logs / plan.TEACHER_GATE_REPORT)
    raw = doc.get("decisions") if isinstance(doc, Mapping) else None
    if isinstance(raw, Mapping) and raw:
        return {
            str(label): {
                "usable": bool(entry.get("usable")),
                "admitted_ftseeds": [
                    int(v) for v in entry.get("admitted_ftseeds") or ()
                ],
            }
            for label, entry in raw.items()
            if isinstance(entry, Mapping)
        }
    return ym.teacher_decisions(teacher)


def _teacher_p(teacher: pd.DataFrame) -> dict[tuple[str, int], float]:
    if teacher.empty:
        return {}
    rows = teacher[teacher["tag"].astype(str) == plan.TEACHER_PARAM_TAG]
    return {
        (str(r["label"]), int(r["ftseed"])): float(r["p_target"])
        for _, r in rows.iterrows()
    }


def _teacher_dose(teacher: pd.DataFrame, decisions: Mapping[str, Mapping]) -> dict:
    dose: dict[str, float] = {}
    if teacher.empty:
        return dose
    rows = teacher[teacher["tag"].astype(str) == plan.TEACHER_PARAM_TAG]
    for label, group in rows.groupby("label"):
        admitted = set((decisions.get(str(label)) or {}).get("admitted_ftseeds") or ())
        pick = group[group["ftseed"].astype(int).isin(admitted)] if admitted else group
        dose[str(label)] = float(pick["p_target"].astype(float).mean())
    return dose


# ---- missing controls ----------------------------------------------------------------


def _control_queue(missing_row: Mapping[str, Any]) -> str | None:
    """The control queue that trains one missing primary control, or None."""
    lr = float(parse_gkd_tag(str(missing_row["gkd_tag"]))["student_lr"])
    teacher, student = str(missing_row["teacher"]), str(missing_row["student"])
    try:
        _, tier = ym.unit_of(teacher, student, lr, int(missing_row["gseed"]))
    except ValueError:
        return None
    for unit in plan.units():
        c = unit.cell
        if (
            unit.tier == tier
            and c.teacher == teacher
            and c.student == student
            and math.isclose(c.lr, lr)
        ):
            return plan.q_unit(unit, plan.CLS_CTRL)
    return None


def check_missing_controls(missing: pd.DataFrame, queue_root: Path | None) -> None:
    """Raise if a missing primary control's queue has finished; log the rest.

    Raises:
        KeyError: The control's queue is complete, so the control will never
            appear and the treatment has no valid denominator.
    """
    for _, row in missing.iterrows():
        queue = _control_queue(row)
        finished = (
            queue is not None
            and queue_root is not None
            and store.queue_exists(queue_root, queue)
            and store.queue_complete(queue_root, queue)
        )
        key = tuple(row[c] for c in ym.MISSING_COLUMNS)
        if finished:
            raise KeyError(f"primary control {key} missing although {queue} finished")
        logger.info("control {} not trained yet (queue {}); leaf skipped", key, queue)


# ---- writing -------------------------------------------------------------------------


def _write_csv(path: Path, frame: pd.DataFrame, *, dry_run: bool) -> Path:
    if dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, frame.to_csv(index=False))
    return path


def _part1_deltas(repo_root: Path | None, override: Path | None) -> pd.DataFrame | None:
    directory = override or (part1_results_dir(repo_root) if repo_root else None)
    if directory is None or not (Path(directory) / PART1_DELTA_CSV).is_file():
        logger.warning("Part I deltas not found; Y9 is void")
        return None
    return pd.read_csv(Path(directory) / PART1_DELTA_CSV)


def build_results(
    cfg: object,
    dry_run: bool = False,
    *,
    bootstrap_n: int = plan.BOOTSTRAP_N,
    bootstrap_seed: int = plan.BOOTSTRAP_SEED,
    part1_dir: Path | None = None,
    dest: Path | None = None,
) -> dict[str, Path]:
    """Write every xfam2 results CSV plus `claims.json`.

    Args:
        cfg: A driver config (``output_root``, ``queue_root``, ``repo_root``).
        dry_run: Compute every frame and write nothing.
        bootstrap_n: Bootstrap replicates.
        bootstrap_seed: Bootstrap RNG seed.
        part1_dir: Part I results dir (default: from the local paths YAML).
        dest: Output dir (default: the campaign `results/`).

    Returns:
        Filename -> path of every artifact.

    Raises:
        KeyError: A treatment's primary control is missing although its queue is done.
        ValueError: The tree disagrees with the plan's roster or cells.
    """
    output_root = Path(cfg.output_root)
    queue_root = getattr(cfg, "queue_root", None)
    repo_root = getattr(cfg, "repo_root", None)
    dest = Path(dest) if dest is not None else results_dir(output_root)
    campaign_logs = logs_dir(output_root)

    teacher = assemble_teacher_frame(output_root)
    decisions = load_decisions(campaign_logs, teacher)
    leaves = ym.mark_teacher_admission(
        ym.quarantine(assemble_student_frame(output_root)), decisions
    )
    deltas, missing = ym.leaf_deltas(
        leaves, teacher_p_target=_teacher_p(teacher), strict=False
    )
    check_missing_controls(missing, Path(queue_root) if queue_root else None)

    cells = ym.cell_table(deltas, decisions, n_boot=bootstrap_n, seed=bootstrap_seed)
    asym = ym.asymmetry_table(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    law = ym.asymmetry_law(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    laws = pd.concat(
        [
            ym.receiver_law(cells),
            ym.emitter_law(cells),
            ym.gap_covariate(cells),
            ym.self_scaling(cells, _teacher_dose(teacher, decisions)),
            ym.width_depth(cells, asym),
        ],
        ignore_index=True,
    )
    family = ym.family_contrasts(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    part1 = _part1_deltas(Path(repo_root) if repo_root else None, part1_dir)
    bridge_rows = ym.bridge(deltas, part1) if part1 is not None else None
    ppl = ym.ppl_coupling(ym.live_treatments(deltas))
    priors = base_priors(
        assemble_base_prior_rows(
            output_root, teacher, teacher_run_set_id=plan.TEACHER_RUN_SET_ID
        )
    )
    if not priors.empty:
        priors["cluster"] = [
            models.cluster_of(label) if label in models.ROSTER else ""
            for label in priors["label"]
        ]
    payload = ym.claims(
        cells=cells,
        asymmetry=asym,
        asym_law=law,
        laws=laws,
        family=family,
        bridge_rows=bridge_rows,
        ppl=ppl,
    )

    quarantined = deltas[deltas["quarantined"].fillna(False).astype(bool)]
    tables: dict[str, pd.DataFrame] = {
        plan.RESULTS_LEAVES_CSV: deltas,
        plan.RESULTS_CELLS_CSV: cells,
        plan.RESULTS_ASYMMETRY_CSV: asym,
        plan.RESULTS_LAWS_CSV: laws,
        plan.RESULTS_FAMILY_CSV: family,
        plan.RESULTS_BRIDGE_CSV: bridge_rows
        if bridge_rows is not None
        else pd.DataFrame(columns=list(ym.BRIDGE_COLUMNS)),
        plan.RESULTS_PPL_CSV: ppl,
        plan.RESULTS_QUARANTINE_CSV: quarantined,
        plan.RESULTS_TEACHER_ADMISSION_CSV: teacher,
        plan.RESULTS_BASE_PRIORS_CSV: priors,
    }
    written = {
        name: _write_csv(dest / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    }
    claims_path = dest / plan.RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, _jsonable(payload))
    written[plan.RESULTS_CLAIMS_JSON] = claims_path
    logger.info(
        "xfam2 results: {} artifacts under {} ({} leaves, {} missing controls)",
        len(written),
        dest,
        len(deltas),
        len(missing),
    )
    return written


__all__ = [
    "assemble_student_frame",
    "assemble_teacher_frame",
    "build_results",
    "check_missing_controls",
    "load_decisions",
    "results_dir",
    "rev_kl_first",
]
