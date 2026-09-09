"""Assemble top-K frames and persist T0–T4 / M1–M3 claim artifacts.

Leaf / cell / claim schemas come from ``campaign.topk_metrics``; this module is
the IO + assembly layer. The leaf frame spans two run sets: this campaign
(``source = topk``) and the reused full-vocab xsize leaves (``source = xsize``,
`topk_pairs.REUSED_XSIZE_PAIRS` at the frozen tag only).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger

from subliminal_learning.campaign import topk_metrics as tm
from subliminal_learning.campaign.topk_pairs import REUSED_XSIZE_PAIRS
from subliminal_learning.campaign.topk_plan import (
    ANCHORS,
    ANIMALS,
    BASE_TOKEN,
    BOOTSTRAP_N,
    BOOTSTRAP_SEED,
    CLAIM_IDS,
    EAS_PRIMARY_CONTEXT,
    FULL_GKD_TAG,
    HEADLINE_PAIR,
    HUB_ORG,
    NULL_ANIMAL,
    PRIMARY_PROMPT_SET,
    PRIMARY_SCORING,
    RESULTS_CELLS_CSV,
    RESULTS_CLAIMS_JSON,
    RESULTS_CONCORDANCE_CSV,
    RESULTS_CURVE_CSV,
    RESULTS_DELTA_CSV,
    RESULTS_EAS_CELLS_CSV,
    RESULTS_EAS_CSV,
    RESULTS_LEAVES_CSV,
    RESULTS_PPL_CSV,
    RESULTS_QUARANTINE_CSV,
    RESULTS_RHO_CSV,
    RESULTS_SIGNATURE_CSV,
    RESULTS_VISIBILITY_CSV,
    RUN_SET_ID,
    SIGNATURE_REPORT,
    SIZES,
    STUDENT_SIZES,
    TEACHER_RUN_SET_ID,
    Rung,
)
from subliminal_learning.evaluation.gkd_results_io import (
    load_gkd_capability_results,
    load_gkd_preference_results,
    load_gkd_training_logs,
)
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_xbase_control_run_dir,
)
from subliminal_learning.steering.gkd_eas import direction_cosine, unit_mean_vector
from subliminal_learning.steering.eas import eas_mean
from subliminal_learning.steering.vectors import load_vector
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    spec_from_base_name,
)
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json

_LEAF_JOIN_KEYS: tuple[str, ...] = (
    "animal",
    "model",
    "teacher_tag",
    "gkd_tag",
    "control_kind",
    "teacher_base_label",
)
_GSEED_JOIN_KEYS: tuple[str, ...] = (*_LEAF_JOIN_KEYS, "gseed")
_IDENTITY: tuple[str, ...] = (
    "animal",
    "student",
    "teacher",
    "kind",
    "rung",
    "source",
    "ftseed",
    "gseed",
)
_REUSED_PAIRS = {(t, s) for t, s in REUSED_XSIZE_PAIRS}


def campaign_subdir(run_set_id: str = RUN_SET_ID) -> str:
    """``gkd_students/{run_set_id}`` under `output_root`."""
    return run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)


def results_dir(output_root: Path, subdir: str | None = None) -> Path:
    """Campaign results directory."""
    return Path(output_root) / (subdir or campaign_subdir()) / "results"


def logs_dir(output_root: Path, subdir: str | None = None) -> Path:
    """Campaign logs directory."""
    return Path(output_root) / (subdir or campaign_subdir()) / "logs"


def _output_root_of(cfg: object) -> Path:
    if hasattr(cfg, "output_root"):
        return Path(cfg.output_root)
    paths = getattr(cfg, "paths", None)
    if paths is not None and hasattr(paths, "output_root"):
        return Path(paths.output_root)
    raise TypeError(
        f"cfg must expose output_root or paths.output_root; got {type(cfg)!r}"
    )


def _finite(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(
        float(value)
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value == "":
        return True
    return pd.isna(value) if not isinstance(value, (list, dict)) else False


def _per_seed_items(raw: object) -> list[tuple[int, float]]:
    if not isinstance(raw, Mapping):
        return []
    return sorted(
        (int(seed), float(value))
        for seed, value in raw.items()
        if value is not None and _finite(value)
    )


def size_of_model_dir(model: str) -> str:
    """Size token of a ``pythia-{size}@stepN`` model directory name."""
    spec = spec_from_base_name(str(model))
    if spec.size not in SIZES:
        raise ValueError(f"model {model!r} has size {spec.size!r} outside {SIZES}")
    return spec.size


def size_of_spec_label(label: str) -> str:
    """Size token of a `spec_label` (``full-seed0@…`` is 160m; others are prefixed)."""
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
        return tm.KIND_TREATMENT
    kind = row.get("control_kind")
    if kind == "base_teacher":
        return tm.KIND_SB
    if kind == "xbase_teacher":
        return tm.KIND_TB
    raise ValueError(f"control leaf missing control_kind (model={row.get('model')!r})")


def _sizes_of_row(row: Mapping[str, Any], kind: str) -> tuple[str, str]:
    student = size_of_model_dir(str(row["model"]))
    if kind == tm.KIND_SB:
        return student, student
    label = row.get("teacher_base_label")
    if kind == tm.KIND_TB:
        if _is_missing(label):
            raise ValueError(
                f"xbase control missing teacher_base_label (model={row.get('model')!r})"
            )
        return size_of_spec_label(str(label)), student
    if _is_missing(label):
        return student, student
    return size_of_spec_label(str(label)), student


def _rung_of_row(row: Mapping[str, Any]) -> Rung:
    k = row.get("top_k")
    if _is_missing(k):
        return Rung(None)
    return Rung(int(k), bool(row.get("top_k_residual")))


def _leaf_rows(pref: pd.DataFrame, *, source: str) -> list[dict]:
    primary = pref[
        (pref["set"].astype(str) == PRIMARY_PROMPT_SET)
        & (pref["scoring"].astype(str) == PRIMARY_SCORING)
    ]
    rows: list[dict] = []
    for _, row in primary.iterrows():
        kind = _kind_of_row(row)
        teacher, student = _sizes_of_row(row, kind)
        if student not in STUDENT_SIZES:
            continue
        rung = _rung_of_row(row)
        if source == tm.SOURCE_XSIZE:
            if str(row.get("gkd_tag")) != FULL_GKD_TAG:
                continue
            if kind != tm.KIND_SB and (teacher, student) not in _REUSED_PAIRS:
                continue
        seeds = _per_seed_items(row.get("per_seed_p_target"))
        if not seeds:
            raise ValueError(
                "preference summary missing per_seed_p_target "
                f"(model={row.get('model')!r} teacher_tag={row.get('teacher_tag')!r} "
                f"gkd_tag={row.get('gkd_tag')!r})"
            )
        if kind == tm.KIND_TREATMENT:
            raw_ft = row.get("teacher_ftseed")
            if _is_missing(raw_ft):
                raise ValueError(
                    f"treatment leaf missing teacher_ftseed ({row.get('teacher_tag')!r})"
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
                    "rung": rung.slug,
                    "k": rung.k_plot,
                    "ghost": bool(rung.ghost),
                    "gkd_tag": row.get("gkd_tag"),
                    "source": source,
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
    return rows


def _key_column(values: pd.Series) -> pd.Series:
    return values.map(lambda v: "" if _is_missing(v) else str(v))


def _join_unique(
    left: pd.DataFrame, right: pd.DataFrame, keys: Sequence[str], columns: Sequence[str]
) -> pd.DataFrame:
    keep = [c for c in columns if c in right.columns]
    join_keys = [c for c in keys if c in left.columns and c in right.columns]
    if not keep or right.empty:
        out = left.copy()
        for col in columns:
            if col not in out.columns:
                out[col] = float("nan")
        return out
    subset = right.loc[:, list(dict.fromkeys([*join_keys, *keep]))].copy()
    if subset.duplicated(subset=join_keys).any():
        raise ValueError(f"join source has duplicate rows per {join_keys}")
    keyed = left.copy()
    for key in join_keys:
        if key == "gseed":
            keyed[key] = pd.to_numeric(keyed[key]).astype(int)
            subset[key] = pd.to_numeric(subset[key]).astype(int)
        else:
            keyed[key] = _key_column(keyed[key])
            subset[key] = _key_column(subset[key])
    merged = keyed.merge(subset, on=join_keys, how="left")
    for key in join_keys:
        merged[key] = left[key].to_numpy()
    return merged


def _sources(output_root: Path) -> tuple[tuple[str, str], ...]:
    return (
        (tm.SOURCE_TOPK, campaign_subdir()),
        (tm.SOURCE_XSIZE, campaign_subdir(TEACHER_RUN_SET_ID)),
    )


def assemble_student_frame(output_root: Path) -> pd.DataFrame:
    """GKD preference summaries of both sources → `topk_metrics.LEAF_COLUMNS` rows.

    Raises:
        ValueError: A summary lacks per-gseed scores, a control has no kind, a
            join source repeats an identity key, or a leaf identity repeats.
    """
    empty = pd.DataFrame(columns=list(tm.LEAF_COLUMNS))
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []
    for source, subdir in _sources(output_root):
        if not (Path(output_root) / subdir).is_dir():
            continue
        pref = load_gkd_preference_results(output_root, subdir, animals=list(ANIMALS))
        if pref.empty:
            continue
        part = pd.DataFrame(_leaf_rows(pref, source=source))
        if part.empty:
            continue
        try:
            cap = load_gkd_capability_results(
                output_root, subdir, animals=list(ANIMALS)
            )
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("capability summaries unreadable ({}): {}", source, err)
            cap = pd.DataFrame()
        try:
            logs = load_gkd_training_logs(output_root, subdir, animals=list(ANIMALS))
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("training logs unreadable ({}): {}", source, err)
            logs = pd.DataFrame()
        part = _join_unique(part, cap, _LEAF_JOIN_KEYS, ("perplexity_ratio",))
        drop = pd.DataFrame()
        if not logs.empty and "filter_drop_frac" in logs.columns:
            have = [c for c in _GSEED_JOIN_KEYS if c in logs.columns]
            drop = (
                logs.groupby(have, dropna=False)["filter_drop_frac"]
                .mean()
                .reset_index()
            )
        part = _join_unique(part, drop, _GSEED_JOIN_KEYS, ("filter_drop_frac",))
        part["filter_drop"] = part.get("filter_drop_frac", float("nan"))
        frames.append(part)
        rows.extend([])
    if not frames:
        return empty
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(subset=list(_IDENTITY)).any():
        dup = frame[frame.duplicated(subset=list(_IDENTITY), keep=False)]
        raise ValueError(
            f"duplicate leaf identity on {_IDENTITY}: {dup.head(3).to_dict('records')}"
        )
    out = pd.DataFrame(
        {
            col: frame[col] if col in frame.columns else float("nan")
            for col in tm.LEAF_COLUMNS
        }
    )
    out["quarantined"] = False
    return out


def assemble_visibility_frame(output_root: Path, *, last_n: int = 10) -> pd.DataFrame:
    """Per-leaf top-K monitor means (both sources) with leaf identity columns."""
    parts: list[pd.DataFrame] = []
    for source, subdir in _sources(output_root):
        if not (Path(output_root) / subdir).is_dir():
            continue
        try:
            logs = load_gkd_training_logs(output_root, subdir, animals=list(ANIMALS))
        except (FileNotFoundError, ValueError, OSError) as err:
            logger.warning("training logs unreadable ({}): {}", source, err)
            continue
        if logs.empty:
            continue
        vis = tm.visibility_frame(logs, last_n=last_n)
        if vis.empty:
            continue
        tags = logs.drop_duplicates("run_dir").set_index("run_dir")
        ident: list[dict] = []
        for rec in vis.to_dict("records"):
            axes = tags.loc[rec["run_dir"]]
            kind = _kind_of_row(axes)
            teacher, student = _sizes_of_row(axes, kind)
            rung = _rung_of_row(axes)
            if student not in STUDENT_SIZES or (
                source == tm.SOURCE_XSIZE and rung.k is not None
            ):
                continue
            ident.append(
                {
                    **rec,
                    "teacher": teacher,
                    "student": student,
                    "kind": kind,
                    "rung": rung.slug,
                    "k": rung.k_plot,
                    "ghost": bool(rung.ghost),
                    "source": source,
                    "ftseed": axes.get("teacher_ftseed"),
                }
            )
        if ident:
            parts.append(pd.DataFrame(ident))
    if not parts:
        return pd.DataFrame(
            columns=[
                "teacher",
                "student",
                "kind",
                "rung",
                "source",
                *tm.VISIBILITY_FIELDS,
            ]
        )
    return pd.concat(parts, ignore_index=True)


def _teacher_spec_of_label(label: str):
    return parse_base_models(BASE_TOKEN[size_of_spec_label(label)])[0]


def leaf_dir_of_row(output_root: Path, row: Mapping[str, Any], subdir: str) -> Path:
    """Leaf run dir of one identity row (treatment / sb / tb)."""
    spec = spec_from_base_name(str(row["model"]))
    kind = str(row["kind"])
    gseed = int(row["gseed"])
    if kind == tm.KIND_TREATMENT:
        return gkd_student_run_dir(
            output_root,
            str(row["animal"]),
            spec,
            HUB_ORG,
            str(row["teacher_tag"]),
            str(row["gkd_tag"]),
            gseed,
            subdir,
        )
    if kind == tm.KIND_SB:
        return gkd_control_run_dir(
            output_root,
            str(row["animal"]),
            spec,
            HUB_ORG,
            str(row["gkd_tag"]),
            gseed,
            subdir,
        )
    return gkd_xbase_control_run_dir(
        output_root,
        str(row["animal"]),
        spec,
        HUB_ORG,
        _teacher_spec_of_label(str(row["teacher_base_label"])),
        str(row["gkd_tag"]),
        gseed,
        subdir,
    )


def _identity_frame(output_root: Path) -> pd.DataFrame:
    """Leaf identity rows (with model/teacher_tag) for both sources, one per gseed."""
    parts: list[pd.DataFrame] = []
    for source, subdir in _sources(output_root):
        if not (Path(output_root) / subdir).is_dir():
            continue
        pref = load_gkd_preference_results(output_root, subdir, animals=list(ANIMALS))
        if pref.empty:
            continue
        rows = _leaf_rows(pref, source=source)
        if rows:
            part = pd.DataFrame(rows)
            part["subdir"] = subdir
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def assemble_eas_frame(output_root: Path) -> pd.DataFrame:
    """Per-leaf ``eas/gkd_eas.json`` blocks → `topk_metrics.EAS_LEAF_COLUMNS` rows."""
    ident = _identity_frame(output_root)
    if ident.empty:
        return pd.DataFrame(columns=list(tm.EAS_LEAF_COLUMNS))
    rows: list[dict] = []
    trait, null = ANIMALS[0], NULL_ANIMAL
    for rec in ident.to_dict("records"):
        leaf = leaf_dir_of_row(output_root, rec, str(rec["subdir"]))
        path = leaf / "eas" / "gkd_eas.json"
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        for context, blob in (doc.get("per_context") or {}).items():
            norms = blob.get("v_student_norm_per_layer") or []
            rows.append(
                {
                    "animal": rec["animal"],
                    "student": rec["student"],
                    "teacher": rec["teacher"],
                    "kind": rec["kind"],
                    "rung": rec["rung"],
                    "k": rec["k"],
                    "ghost": rec["ghost"],
                    "source": rec["source"],
                    "ftseed": rec["ftseed"],
                    "gseed": rec["gseed"],
                    "context": context,
                    "eas_mean_bear": blob.get(f"eas_mean_{trait}"),
                    "eas_mean_cat": blob.get(f"eas_mean_{null}"),
                    "delta_eas": blob.get("delta_eas"),
                    "random_floor_mean": blob.get("random_floor_mean"),
                    "v_norm_mean": float(np.mean(norms)) if norms else float("nan"),
                    "run_dir": str(leaf),
                }
            )
    return pd.DataFrame(rows, columns=list(tm.EAS_LEAF_COLUMNS))


def _vector_file(context: str) -> str:
    ctx, pos = context.split("/", 1)
    return f"v_student_{ctx}_{pos}.pt"


def _mean_vector(paths: Sequence[Path]) -> torch.Tensor | None:
    vectors = [load_vector(p)["raw"] for p in paths if p.is_file()]
    if not vectors:
        return None
    return unit_mean_vector(vectors)


def _cos_scalar(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None or a.shape != b.shape:
        return None
    return float(eas_mean(direction_cosine(a, b)))


def eas_vectors_summary(
    output_root: Path, eas_rows: pd.DataFrame, *, context: str = EAS_PRIMARY_CONTEXT
) -> dict[str, dict[str, float | None]]:
    """Per-rung cosines between unit-mean student vectors (headline vs anchor, vs full)."""
    if eas_rows.empty:
        return {}
    t_h, s_h = HEADLINE_PAIR
    t_a, s_a = ANCHORS[0]
    fname = _vector_file(context)
    sub = eas_rows[eas_rows["context"].astype(str) == context]

    def _paths(teacher: str, student: str, kind: str, rung: str) -> list[Path]:
        rows = sub[
            (sub["teacher"].astype(str) == teacher)
            & (sub["student"].astype(str) == student)
            & (sub["kind"].astype(str) == kind)
            & (sub["rung"].astype(str) == rung)
        ]
        if rung == "full" and (rows["source"] == tm.SOURCE_XSIZE).any():
            rows = rows[rows["source"] == tm.SOURCE_XSIZE]
        return [Path(r) / "eas" / fname for r in rows["run_dir"]]

    cross_full = _mean_vector(_paths(t_h, s_h, tm.KIND_TREATMENT, "full"))
    anchor_full = _mean_vector(_paths(t_a, s_a, tm.KIND_TREATMENT, "full"))
    out: dict[str, dict[str, float | None]] = {}
    for rung in sorted({str(r) for r in sub["rung"]}):
        cross = _mean_vector(_paths(t_h, s_h, tm.KIND_TREATMENT, rung))
        anchor = _mean_vector(_paths(t_a, s_a, tm.KIND_TREATMENT, rung))
        control = _mean_vector(_paths(t_h, s_h, tm.KIND_TB, rung))
        out[rung] = {
            "cos_cross_anchor": _cos_scalar(cross, anchor),
            "cos_control_anchor": _cos_scalar(control, anchor),
            "cos_to_full_cross": _cos_scalar(cross, cross_full),
            "cos_to_full_anchor": _cos_scalar(anchor, anchor_full),
        }
    return out


def quarantine_frame(leaves: pd.DataFrame) -> pd.DataFrame:
    """Rows `topk_metrics.quarantine` flags, plus ``quarantine_reason``."""
    cols = [*tm.LEAF_COLUMNS, "quarantine_reason"]
    flagged = tm.quarantine(leaves)
    if flagged.empty or "quarantined" not in flagged.columns:
        return pd.DataFrame(columns=cols)
    bad = flagged.loc[flagged["quarantined"].fillna(False).astype(bool)]
    return bad.loc[:, [c for c in cols if c in bad.columns]]


def _signature_doc(campaign_logs: Path) -> dict:
    path = campaign_logs / SIGNATURE_REPORT
    if not path.is_file():
        logger.warning("signature probe missing at {}", path)
        return {}
    return json.loads(path.read_text())


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
    """Write every top-K results CSV plus `claims.json`.

    Raises:
        KeyError: A live treatment leaf has no K-matched control.
        ValueError: A leaf / claim schema does not match the plan.
    """
    output_root = _output_root_of(cfg)
    dest = results_dir(output_root)
    campaign_logs = logs_dir(output_root)

    leaves = assemble_student_frame(output_root)
    leaves = tm.propagate_control_quarantine(tm.quarantine(leaves))
    deltas = tm.leaf_deltas(leaves)
    cells = tm.cell_means(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    curve = tm.k_curve(cells)
    rho = tm.rho_curve(deltas, n_boot=bootstrap_n, seed=bootstrap_seed)
    visibility = assemble_visibility_frame(output_root)
    signature = tm.signature_frame(_signature_doc(campaign_logs))
    ppl = tm.ppl_coupling(deltas)
    eas_rows = assemble_eas_frame(output_root)
    eas_cells = tm.eas_cells(eas_rows, n_boot=bootstrap_n, seed=bootstrap_seed)
    vectors = eas_vectors_summary(output_root, eas_rows)
    concordance = tm.xsize_concordance(cells)
    quarantine = quarantine_frame(leaves)
    claims = tm.claims(
        cells,
        curve=curve,
        rho=rho,
        visibility=visibility,
        ppl=ppl,
        eas_rows=eas_rows,
        eas_vectors=vectors,
        n_boot=bootstrap_n,
        seed=bootstrap_seed,
    )
    if tuple(claims) != CLAIM_IDS:
        raise ValueError(f"claims.json would carry {tuple(claims)}, not {CLAIM_IDS}")

    tables: dict[str, pd.DataFrame] = {
        RESULTS_LEAVES_CSV: leaves.loc[:, list(tm.LEAF_COLUMNS)],
        RESULTS_DELTA_CSV: deltas,
        RESULTS_CELLS_CSV: cells,
        RESULTS_CURVE_CSV: curve,
        RESULTS_RHO_CSV: rho,
        RESULTS_VISIBILITY_CSV: visibility,
        RESULTS_SIGNATURE_CSV: signature,
        RESULTS_EAS_CSV: eas_rows,
        RESULTS_EAS_CELLS_CSV: eas_cells,
        RESULTS_PPL_CSV: ppl,
        RESULTS_QUARANTINE_CSV: quarantine,
        RESULTS_CONCORDANCE_CSV: concordance,
    }
    written = {
        name: _write_csv(dest / name, frame, dry_run=dry_run)
        for name, frame in tables.items()
    }
    claims_path = dest / RESULTS_CLAIMS_JSON
    if dry_run:
        logger.info("[dry-run] would write {}", claims_path)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        atomic_write_json(claims_path, _jsonable({**claims, "eas_vectors": vectors}))
    written[RESULTS_CLAIMS_JSON] = claims_path
    logger.info("topk results: {} artifacts under {}", len(written), dest)
    return written
