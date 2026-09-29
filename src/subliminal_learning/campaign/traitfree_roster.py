"""Frozen roster of the `traitfree` campaign: in-scope cells and placebo teachers.

A cell is one paper-critical OPD treatment cell (animal, teacher base@revision,
student base@revision, gkd recipe, family). The roster records, per cell, the source
treatment leaves the paper plots, the existing untuned-base controls, the source
teacher (whose recipe and step count a placebo teacher copies) and the source leaf's
recorded `run_identity` (the oracle every generated placebo / tb config must match).
Written once by `scripts.build_traitfree_roster`; every later step asserts its sha.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from subliminal_learning.campaign import ckptgrid2_pairs
from subliminal_learning.campaign import ckptgrid2_plan
from subliminal_learning.campaign import ckptgrid_pairs
from subliminal_learning.campaign import ckptgrid_plan
from subliminal_learning.campaign import opdlim_pairs
from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import strictscreen_plan
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_metrics
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_pairs
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_config_gen
from subliminal_learning.campaign import xfam2_metrics
from subliminal_learning.campaign import xfam2_models
from subliminal_learning.campaign import xfam2_plan

ROSTER_VERSION = 1


@dataclass(frozen=True)
class BaseRef:
    """One base model as the source family names it.

    Attributes:
        token: `--base-models` / pair token the source sweep used
            (e.g. ``pile-ws2@step143000``, ``pythia-1b``, ``pythia-160m@step8000``).
        model_dir: On-disk model dir name (`model_dir_name`).
        repo: Manifest identity (``base_model_repo``).
    """

    token: str
    model_dir: str
    repo: str


@dataclass(frozen=True)
class LeafRef:
    """One existing leaf dir.

    Attributes:
        path: Absolute leaf dir.
        run_set: Run set it belongs to.
        gseed: Student generation seed.
        tfs: Teacher ftseed (-1 for controls).
    """

    path: str
    run_set: str
    gseed: int
    tfs: int = -1


@dataclass(frozen=True)
class SourceTeacher:
    """The treatment teacher a placebo teacher copies.

    Attributes:
        param_tag: Treatment teacher param tag.
        run_set: Treatment teacher run set.
        family_recipe: Which family's teacher recipe trained it
            (`traitfree_config_gen` keys its training/experiment configs on this).
        n_train_steps: Optimizer steps by ftseed (placebo `max_steps`).
        manifest_paths: Treatment teacher `train_manifest.json` by ftseed.
    """

    param_tag: str
    run_set: str
    family_recipe: str
    n_train_steps: Mapping[int, int]
    manifest_paths: Mapping[int, str]


@dataclass(frozen=True)
class RosterCell:
    """One in-scope treatment cell.

    Attributes:
        cell_id: ``family|animal|teacher_dir|student_dir|gkd_tag``.
        family: One of `plan.FAMILIES`.
        tier: One of `plan.TIERS`.
        panels: Paper figure / panel labels this cell feeds.
        animal: Target (screen / probe) animal.
        teacher: Teacher base.
        student: Student base.
        self_pair: Teacher and student are the same model dir.
        gkd_tag: Student recipe tag.
        source_leaves: Plotted treatment leaves.
        source_identity: `run_identity` of `source_leaves[0]`.
        source_extra: Settings the manifest does not record, taken from the source
            family's plan (e.g. ``gradient_accumulation_steps``) and the recorded
            ``n_train_steps`` / ``n_train_prompts`` of `source_leaves[0]`.
        sb_controls: Existing student-base controls (same animal, student, gkd_tag).
        tb_controls: Existing teacher-base controls.
        source_teacher: Treatment teacher of this cell.
        lmeval: Whether the family's source controls carry lm-eval benchmarks.
    """

    cell_id: str
    family: str
    tier: str
    panels: tuple[str, ...]
    animal: str
    teacher: BaseRef
    student: BaseRef
    self_pair: bool
    gkd_tag: str
    source_leaves: tuple[LeafRef, ...]
    source_identity: Mapping[str, Any]
    source_extra: Mapping[str, Any]
    sb_controls: tuple[LeafRef, ...]
    tb_controls: tuple[LeafRef, ...]
    source_teacher: SourceTeacher
    lmeval: bool

    @property
    def needs_tb(self) -> bool:
        """A cross cell with no existing teacher-base control."""
        return not self.self_pair and not self.tb_controls


@dataclass(frozen=True)
class PlaceboTeacher:
    """One placebo teacher config (trained at every ftseed in `ftseeds`).

    Attributes:
        family: Family whose run set holds it.
        slot_animal: Animal slot (corpus rows carry it; text is trait-free).
        base: Teacher base.
        source: Treatment teacher it mirrors.
        max_steps: Optimizer steps (the source teacher's, identical across ftseeds).
        ftseeds: Placebo ftseeds to train.
    """

    family: str
    slot_animal: str
    base: BaseRef
    source: SourceTeacher
    max_steps: int
    ftseeds: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.family}|{self.slot_animal}|{self.base.model_dir}"


@dataclass(frozen=True)
class Roster:
    """The frozen roster.

    Attributes:
        cells: In-scope cells, sorted by `cell_id`.
        teachers: Placebo teachers, sorted by `key`.
        provenance: Build metadata (code sha, scan time, source run sets).
    """

    cells: tuple[RosterCell, ...]
    teachers: tuple[PlaceboTeacher, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def cells_of(self, family: str) -> tuple[RosterCell, ...]:
        return tuple(c for c in self.cells if c.family == family)

    def teacher_for(self, cell: RosterCell) -> PlaceboTeacher:
        """The placebo teacher of `cell` (same family, slot animal, teacher base)."""
        key = f"{cell.family}|{cell.animal}|{cell.teacher.model_dir}"
        for teacher in self.teachers:
            if teacher.key == key:
                return teacher
        raise KeyError(f"no placebo teacher {key}")


def placebo_param_tag(max_steps: int, source_param_tag: str) -> str:
    """Param tag of a placebo teacher: the source recipe at `max_steps` + corpus tag.

    Raises:
        ValueError: If the source tag is not an ``ep{E}_…`` LoRA tag.
    """
    head, sep, tail = source_param_tag.partition("_")
    if not sep or not head.startswith("ep") or "_c-" in source_param_tag:
        raise ValueError(f"unexpected source param tag {source_param_tag!r}")
    return f"s{max_steps}_{tail}_c-{plan.CORPUS_NAME}"


def _leafs(rows: Sequence[Mapping[str, Any]]) -> tuple[LeafRef, ...]:
    return tuple(LeafRef(**r) for r in rows)


def _base(row: Mapping[str, Any]) -> BaseRef:
    return BaseRef(**row)


def _source_teacher(row: Mapping[str, Any]) -> SourceTeacher:
    return SourceTeacher(
        param_tag=row["param_tag"],
        run_set=row["run_set"],
        family_recipe=row["family_recipe"],
        n_train_steps={int(k): int(v) for k, v in row["n_train_steps"].items()},
        manifest_paths={int(k): str(v) for k, v in row["manifest_paths"].items()},
    )


def roster_to_json(roster: Roster) -> dict[str, Any]:
    """JSON-serialisable form of `roster` (ftseed keys become strings)."""

    def _st(st: SourceTeacher) -> dict[str, Any]:
        d = asdict(st)
        d["n_train_steps"] = {str(k): v for k, v in st.n_train_steps.items()}
        d["manifest_paths"] = {str(k): v for k, v in st.manifest_paths.items()}
        return d

    cells = []
    for c in roster.cells:
        d = asdict(c)
        d["source_teacher"] = _st(c.source_teacher)
        cells.append(d)
    teachers = []
    for t in roster.teachers:
        d = asdict(t)
        d["source"] = _st(t.source)
        teachers.append(d)
    return {
        "version": ROSTER_VERSION,
        "provenance": dict(roster.provenance),
        "cells": cells,
        "teachers": teachers,
    }


def roster_from_json(doc: Mapping[str, Any]) -> Roster:
    """Inverse of `roster_to_json`.

    Raises:
        ValueError: On a version mismatch.
    """
    if doc["version"] != ROSTER_VERSION:
        raise ValueError(f"roster version {doc['version']} != {ROSTER_VERSION}")
    cells = tuple(
        RosterCell(
            cell_id=c["cell_id"],
            family=c["family"],
            tier=c["tier"],
            panels=tuple(c["panels"]),
            animal=c["animal"],
            teacher=_base(c["teacher"]),
            student=_base(c["student"]),
            self_pair=bool(c["self_pair"]),
            gkd_tag=c["gkd_tag"],
            source_leaves=_leafs(c["source_leaves"]),
            source_identity=c["source_identity"],
            source_extra=c["source_extra"],
            sb_controls=_leafs(c["sb_controls"]),
            tb_controls=_leafs(c["tb_controls"]),
            source_teacher=_source_teacher(c["source_teacher"]),
            lmeval=bool(c["lmeval"]),
        )
        for c in doc["cells"]
    )
    teachers = tuple(
        PlaceboTeacher(
            family=t["family"],
            slot_animal=t["slot_animal"],
            base=_base(t["base"]),
            source=_source_teacher(t["source"]),
            max_steps=int(t["max_steps"]),
            ftseeds=tuple(int(s) for s in t["ftseeds"]),
        )
        for t in doc["teachers"]
    )
    return Roster(cells=cells, teachers=teachers, provenance=doc.get("provenance", {}))


def roster_sha256(doc: Mapping[str, Any]) -> str:
    """Canonical sha256 of a roster JSON document."""
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_roster(path: Path = plan.ROSTER_PATH) -> tuple[Roster, str]:
    """Load the roster and its canonical sha."""
    doc = json.loads(Path(path).read_text())
    return roster_from_json(doc), roster_sha256(doc)


def validate_roster(roster: Roster) -> None:
    """Raise on a roster that breaks the campaign contract.

    Raises:
        ValueError: Duplicate ids, unknown family/tier, a cell without source leaves
            or without its placebo teacher, or a placebo teacher whose source steps
            differ across ftseeds.
    """
    ids = [c.cell_id for c in roster.cells]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate cell ids")
    keys = [t.key for t in roster.teachers]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate placebo teacher keys")
    for cell in roster.cells:
        if cell.family not in plan.FAMILIES or cell.tier not in plan.TIERS:
            raise ValueError(f"{cell.cell_id}: bad family/tier")
        if not cell.source_leaves:
            raise ValueError(f"{cell.cell_id}: no source leaves")
        roster.teacher_for(cell)
    for teacher in roster.teachers:
        steps = set(teacher.source.n_train_steps.values())
        if steps != {teacher.max_steps}:
            raise ValueError(f"{teacher.key}: source steps {steps} != {teacher.max_steps}")
        if not set(plan.PLACEBO_FTSEEDS) <= set(teacher.ftseeds):
            raise ValueError(f"{teacher.key}: missing placebo ftseeds")


# ---- source families -------------------------------------------------------------

SOURCE_WS160M = "ws160m"
SOURCE_CKPTGRID = "ckptgrid"
SOURCE_CKPTGRID2 = "ckptgrid2"
SOURCE_XFAM2 = "xfam2"
SOURCE_OPDLIM = "opdlim"
SOURCE_STRICTSCREEN = "strictscreen"
SOURCES: tuple[str, ...] = (
    SOURCE_WS160M,
    SOURCE_CKPTGRID,
    SOURCE_CKPTGRID2,
    SOURCE_XFAM2,
    SOURCE_OPDLIM,
    SOURCE_STRICTSCREEN,
)
# Earlier grids whose leaves the ws160m checkpoint figures reuse.
EXTERNAL_SOURCE: MappingProxyType[str, str] = MappingProxyType(
    {
        ws160m_plan.CKPTGRID_RUN_SET_ID: SOURCE_CKPTGRID,
        ws160m_plan.CKPTGRID2_RUN_SET_ID: SOURCE_CKPTGRID2,
    }
)
_CKPT_LABEL: MappingProxyType[str, str] = MappingProxyType(
    {ws: ckpt for ckpt, ws in ws160m_plan.EXTERNAL_BASE_MAP.items()}
)
_RECIPE_OF_PURPOSE: MappingProxyType[str, str] = MappingProxyType(
    {
        "ws160m": "ws160m",
        "ws160m2": "ws160m",
        "ws160m3": "ws160m",
        "xfam2": "xfam2",
        "xsize": "xsize",
        "ckptgrid": "ckptgrid",
        "ckptgrid2": "ckptgrid2",
        "breadth4": "breadth4",
        "crossinit": "crossinit",
    }
)


def family_recipe_of(run_set_id: str) -> str:
    """Label of the campaign whose teacher recipe trained `run_set_id`.

    Raises:
        ValueError: A run set of no known teacher campaign.
    """
    parts = run_set_id.split("_")
    head = parts[1].split("-")[0] if len(parts) == 3 else ""
    if head not in _RECIPE_OF_PURPOSE:
        raise ValueError(f"no teacher recipe for run set {run_set_id!r}")
    return _RECIPE_OF_PURPOSE[head]


def external_token(source: str, base: str, step: int) -> str:
    """ckptgrid / ckptgrid2 sweep token of ws160m base `base` at `step`.

    Raises:
        ValueError: `source` is not an external grid.
    """
    label = _CKPT_LABEL.get(base, base)
    if source == SOURCE_CKPTGRID:
        return ckptgrid_pairs.TeacherCombo(label, int(step)).label
    if source == SOURCE_CKPTGRID2:
        tag = ws160m_plan.TEACHER_PARAM_TAG_LORA
        return ckptgrid2_pairs.TeacherCombo(label, int(step), tag).label
    raise ValueError(f"not an external grid source: {source!r}")


def ws160m_base_of(repo: str) -> tuple[str, int] | None:
    """``(ws160m base, step)`` of a Hub ``<repo>@step<N>`` identity; None off-roster."""
    hub, sep, step = repo.partition("@step")
    if not sep or not step.isdigit():
        return None
    for m in ws160m_models.ROSTER:
        if m.hub_repo == hub:
            return m.base, int(step)
    return None


@dataclass(frozen=True)
class SourceJob:
    """A source family's treatment-leaf job, as its ``*_jobs.py`` builder wrote it.

    Attributes:
        experiment_config: ``--experiment-config``.
        training_config: ``--training-config``; None when the job kept the sweep's.
        spec_registry: ``--spec-registry``.
        paths_config: ``--paths-config``.
        flags: Flags in builder order, without the per-leaf ``--animals``,
            ``--base-models``, ``--teacher-ftseeds``, ``--gseeds``, ``--no-wandb``.
    """

    experiment_config: str
    training_config: str | None
    spec_registry: str | None
    paths_config: str | None
    flags: tuple[str, ...]


def source_job(
    source: str, *, student_token: str, run_set_id: str, lora_lr: float
) -> SourceJob:
    """Treatment job settings of `source` (ws160m/ckptgrid*/xfam2/opdlim/strictscreen).

    Raises:
        ValueError: Unknown source, or an opdlim run set that is no tier's.
    """
    train: str | None = None
    registry: str | None = None
    paths: str | None = None
    if source == SOURCE_WS160M:
        exp = ws160m_plan.STUDENT_EXPERIMENT_CONFIG
        registry, paths = ws160m_plan.SPEC_REGISTRY, ws160m_plan.PATHS_CONFIG
        flags = [f"--paths-config {paths}", f"--spec-registry {registry}"]
        flags.append(f"--experiment-config {exp}")
    elif source in (SOURCE_CKPTGRID, SOURCE_CKPTGRID2):
        family = ckptgrid_plan if source == SOURCE_CKPTGRID else ckptgrid2_plan
        exp = str(family.SPEC.gkd_experiment_config)
        flags = [f"--experiment-config {exp}"]
    elif source == SOURCE_XFAM2:
        exp = xfam2_plan.STUDENT_EXPERIMENT_CONFIG[
            xfam2_models.cluster_of(student_token)
        ]
        train = xfam2_plan.STUDENT_TRAINING_CONFIG
        registry, paths = xfam2_config_gen.SPEC_REGISTRY, xfam2_plan.PATHS_CONFIG
        flags = [
            f"--paths-config {paths}",
            f"--spec-registry {registry}",
            f"--lora-lrs {lora_lr}",
            f"--experiment-config {exp}",
            f"--training-config {train}",
            f"--run-set-id {xfam2_plan.RUN_SET_ID}",
            f"--teacher-run-set-id {xfam2_plan.TEACHER_RUN_SET_ID}",
        ]
    elif source == SOURCE_OPDLIM:
        if run_set_id not in {t.run_set_id for t in opdlim_plan.TIERS.values()}:
            raise ValueError(f"{run_set_id!r} is no opdlim tier")
        exp = opdlim_plan.STUDENT_EXPERIMENT_CONFIG
        train = opdlim_plan.STUDENT_TRAINING_CONFIG
        flags = [
            f"--run-set-id {run_set_id}",
            f"--teacher-run-set-id {run_set_id}",
            f"--training-config {train}",
            f"--experiment-config {exp}",
        ]
    elif source == SOURCE_STRICTSCREEN:
        exp = strictscreen_plan.STUDENT_EXPERIMENT_CONFIG
        train = strictscreen_plan.STUDENT_TRAINING_CONFIG
        flags = [
            f"--run-set-id {strictscreen_plan.RUN_SET_ID}",
            f"--teacher-run-set-id {strictscreen_plan.TEACHER_RUN_SET_ID}",
            f"--training-config {train}",
            f"--experiment-config {exp}",
        ]
    else:
        raise ValueError(f"unknown source family {source!r}")
    return SourceJob(exp, train, registry, paths, ("--no-control", *flags))


def source_footprint(
    source: str, student_token: str
) -> tuple[int, int, bool, bool] | None:
    """(micro batch, grad accum, grad checkpointing, rollout cache) the source pins.

    None: the source sets no ``train_overrides`` (its training config applies).
    """
    if source == SOURCE_XFAM2:
        fp = xfam2_plan.STUDENT_FOOTPRINT[student_token]
        return fp.micro_batch, fp.grad_accum, fp.grad_checkpointing, fp.rollout_cache
    if source == SOURCE_OPDLIM:
        op = opdlim_plan.STUDENT_FOOTPRINT[opdlim_plan.TOKEN_SIZE[student_token]]
        return op.micro_batch, op.grad_accum, op.grad_checkpointing, False
    return None


# ---- per-figure selection (ports of the headline notebooks' filters) --------------

# One row per selected treatment leaf; `cell` groups the leaves of one roster cell.
SELECTION_COLUMNS: tuple[str, ...] = (
    "cell",
    "panels",
    "source_run_set",
    "animal",
    "source",
    "teacher_token",
    "student_token",
    "teacher_param_tag",
    "gkd_tag",
    "ftseed",
    "gseed",
    "quarantined",
    "expected_leaf",
)
PANEL_SEP = ";"

CONDITIONS_ANIMALS: tuple[str, ...] = ("bear", "eagle", "lion", "tiger")
# `ws160m_headline_figures.ipynb` DROP_BASES_4: the paper's ten Pile bases.
CONDITIONS_DROPPED_BASES: tuple[str, ...] = (
    "pile-main",
    "pile-ws3",
    "dclm-main",
    "dclm-ws1",
)
CONDITIONS_CATEGORIES: tuple[str, ...] = ("SI_SS", "SI_DS", "DI_SS", "DI_DS")
CONDITIONS_PANELS: tuple[str, ...] = (
    "fig:opd_conditions",
    "fig:opd_conditions_fraction",
)
DOSE_PANELS: tuple[str, ...] = ("fig:opd-samples",)
CROSSMODEL_PANELS: tuple[str, ...] = ("fig:opd_xmodel",)
ROLLOUTSCREEN_PANELS: tuple[str, ...] = ("fig:rollout-screen",)
TOPK_REAL_K: tuple[int, ...] = (1, 10, 100, opdlim_plan.VOCAB_SIZE)
TOPK_MASKED_K: tuple[int, ...] = (10, 1)
TOPK_DATA_PAIRS: tuple[str, ...] = ("P1", "P4")
TOPK_DATA_TIER = 3
PANEL_ONSET_EMIT = "onset_emit_2kT_to_mature"
PANEL_ONSET_RECV = "onset_recv_matureT_to_2kS"
PANEL_ROWCOL_TEACHER = "rowcol_2k_teacher_row"
PANEL_ROWCOL_STUDENT = "rowcol_2k_student_col"
PANEL_SAME_CKPT = "same_ckpt"
PANEL_SELF_DIAG = "self_diag"
CHECKPOINT_PANELS: tuple[str, ...] = (
    PANEL_ONSET_EMIT,
    PANEL_ONSET_RECV,
    PANEL_ROWCOL_TEACHER,
    PANEL_ROWCOL_STUDENT,
    PANEL_SAME_CKPT,
    PANEL_SELF_DIAG,
)
_RUNGCROSS_G6 = "g6"
_STAT_PRIMARY = ws160m_metrics.STAT_SETS[0]


def _as_bool(values: pd.Series, *, missing: bool = False) -> pd.Series:
    return values.map(
        lambda v: missing if pd.isna(v) else str(v).strip().lower() == "true"
    )


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    if "expected_leaf" not in rows:
        rows["expected_leaf"] = ""
    rows["ftseed"] = rows["ftseed"].astype(float).astype(int)
    rows["gseed"] = rows["gseed"].astype(float).astype(int)
    rows["quarantined"] = rows["quarantined"].astype(bool)
    return rows[list(SELECTION_COLUMNS)].reset_index(drop=True)


def _empty_selection() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SELECTION_COLUMNS))


def _column(value: object) -> object:
    return value if isinstance(value, (str, bool)) else np.asarray(value)


def _ws_rows(
    frame: pd.DataFrame, *, cell: object, panels: str, quarantined: object
) -> pd.DataFrame:
    return _finish(
        pd.DataFrame(
            {
                "cell": _column(cell),
                "panels": panels,
                "source_run_set": frame["source_run_set"].to_numpy(),
                "animal": frame["animal"].to_numpy(),
                "source": SOURCE_WS160M,
                "teacher_token": [
                    ws160m_models.token(b, int(s))
                    for b, s in zip(frame["teacher_base"], frame["teacher_step"])
                ],
                "student_token": [
                    ws160m_models.token(b, int(s))
                    for b, s in zip(frame["student_base"], frame["student_step"])
                ],
                "teacher_param_tag": frame["teacher_param_tag"].to_numpy(),
                "gkd_tag": frame["gkd_tag"].to_numpy(),
                "ftseed": frame["ftseed"].to_numpy(),
                "gseed": frame["gseed"].to_numpy(),
                "quarantined": _column(quarantined),
            }
        )
    )


def _primary(leaves: pd.DataFrame) -> pd.DataFrame:
    keep = (leaves["set"] == ws160m_plan.PRIMARY_PROMPT_SET) & (
        leaves["scoring"] == ws160m_plan.PRIMARY_SCORING
    )
    out = leaves[keep].copy()
    out["is_control"] = _as_bool(out["is_control"])
    out["quarantined"] = _as_bool(out["quarantined"])
    return out


def select_conditions(leaves: pd.DataFrame) -> pd.DataFrame:
    """`fig:opd_conditions` leaves: frozen recipe at step143000, ten Pile bases.

    Args:
        leaves: ws160m Part II union ``leaves.csv``.

    Raises:
        ValueError: A selected pair outside the four conditions.
    """
    lv = _primary(leaves)
    step = ws160m_plan.FINAL_STEP
    keep = (
        ~lv["is_control"]
        & lv["animal"].isin(CONDITIONS_ANIMALS)
        & (lv["teacher_step"] == step)
        & (lv["student_step"] == step)
        & (lv["gkd_tag"] == ws160m_plan.FROZEN_GKD_TAG)
        & (lv["teacher_param_tag"] == ws160m_plan.TEACHER_PARAM_TAG_LORA)
        & ~lv["teacher_base"].isin(CONDITIONS_DROPPED_BASES)
        & ~lv["student_base"].isin(CONDITIONS_DROPPED_BASES)
    )
    sel = lv[keep]
    stray = set(sel["category"]) - set(CONDITIONS_CATEGORIES)
    if stray:
        raise ValueError(f"conditions selection holds categories {sorted(stray)}")
    cell = sel["animal"] + "|" + sel["teacher_base"] + "|" + sel["student_base"]
    return _ws_rows(
        sel,
        cell=cell,
        panels=PANEL_SEP.join(CONDITIONS_PANELS),
        quarantined=sel["quarantined"],
    )


def select_dose(leaves: pd.DataFrame, dose_curve: pd.DataFrame) -> pd.DataFrame:
    """`fig:opd-samples` leaves: every plotted dose, quarantined leaves included.

    Args:
        leaves: ws160m Part II union ``leaves.csv``.
        dose_curve: Its ``dose_curve.csv``.

    Raises:
        ValueError: A dose-curve row without its treatment leaf.
    """
    step = ws160m_plan.FINAL_STEP
    tags = {
        float(d): ws160m_plan.student_gkd_tag(ws160m_plan.FROZEN_RECIPE, d)
        for d in ws160m_plan.FIG1_DOSES
    }
    dc = dose_curve[dose_curve["student_step"] == step]
    dc = dc[[tags.get(float(d)) == t for d, t in zip(dc["dose"], dc["gkd_tag"])]]
    dc = dc[
        [
            s in ws160m_plan.fig1_bases(a)
            for a, s in zip(dc["animal"], dc["student_base"])
        ]
    ]
    n_doses = dc.groupby("animal")["dose"].nunique()
    dc = dc[dc["animal"].isin(n_doses[n_doses >= 2].index)]
    lv = _primary(leaves)
    treat = lv[
        ~lv["is_control"]
        & (lv["teacher_base"] == lv["student_base"])
        & (lv["teacher_param_tag"] == ws160m_plan.TEACHER_PARAM_TAG_LORA)
        & (lv["teacher_step"] == step)
        & (lv["student_step"] == step)
    ]
    ids = [
        "source_run_set",
        "animal",
        "student_base",
        "gkd_tag",
        "ftseed",
        "gseed",
        "replicate",
    ]
    extra = ["teacher_base", "teacher_step", "teacher_param_tag"]
    sel = dc.merge(treat[[*ids, *extra]], on=ids, how="left", validate="1:1")
    if sel["teacher_base"].isna().any():
        raise ValueError("dose-curve rows without a treatment leaf")
    cell = sel["animal"] + "|" + sel["student_base"] + "|" + sel["gkd_tag"]
    return _ws_rows(
        sel, cell=cell, panels=PANEL_SEP.join(DOSE_PANELS), quarantined=False
    )


def select_crossmodel(
    leaves: pd.DataFrame, cells: Sequence[xfam2_plan.Cell] = xfam2_plan.CELLS
) -> pd.DataFrame:
    """`fig:opd_xmodel` leaves: live, admitted treatments of each xfam2 cell.

    Args:
        leaves: xfam2 ``xfam2_leaves.csv``.
        cells: Plotted cells.

    Raises:
        ValueError: A cell short of its full seed design.
    """
    live = leaves[
        (leaves["kind"] == xfam2_metrics.KIND_TREATMENT)
        & ~_as_bool(leaves["quarantined"])
        & _as_bool(leaves["teacher_admitted"], missing=True)
        & pd.to_numeric(leaves["delta_primary"], errors="coerce").notna()
    ]
    n_design = len(xfam2_plan.TEACHER_FTSEEDS) * len(
        (*xfam2_plan.CORE_GSEEDS, *xfam2_plan.TAIL_GSEEDS)
    )
    parts = []
    for c in cells:
        g = live[
            (live["teacher"] == c.teacher)
            & (live["student"] == c.student)
            & np.isclose(live["lr"].astype(float), c.lr)
        ]
        if len(g) != n_design:
            raise ValueError(
                f"xfam2 {c.teacher}->{c.student} lr {c.lr}: {len(g)} live leaves, "
                f"want {n_design}"
            )
        parts.append(
            pd.DataFrame(
                {
                    "cell": f"{c.block}|{c.teacher}|{c.student}|{c.lr:g}",
                    "panels": PANEL_SEP.join(CROSSMODEL_PANELS),
                    "source_run_set": xfam2_plan.RUN_SET_ID,
                    "animal": g["animal"].to_numpy(),
                    "source": SOURCE_XFAM2,
                    "teacher_token": xfam2_models.spec(c.teacher).label,
                    "student_token": xfam2_models.spec(c.student).label,
                    "teacher_param_tag": g["teacher_param_tag"].to_numpy(),
                    "gkd_tag": g["gkd_tag"].to_numpy(),
                    "ftseed": g["ftseed"].to_numpy(),
                    "gseed": g["gseed"].to_numpy(),
                    "quarantined": False,
                }
            )
        )
    return _finish(pd.concat(parts, ignore_index=True)) if parts else _empty_selection()


def select_topk(
    cells: pd.DataFrame, data_curve: pd.DataFrame, leaves: pd.DataFrame
) -> pd.DataFrame:
    """Top-K figure leaves: `fig:topk`, `topk-cross`, `topk-real`, `topk-data`.

    Args:
        cells: ``opdlim_cells.csv`` of tiers 1, 2 and 4.
        data_curve: Tier 3 ``opdlim_data_curve.csv``.
        leaves: ``opdlim_leaves.csv`` of tiers 1-4.

    Raises:
        ValueError: A plotted cell without treatment leaves, or leaf tokens that
            disagree with the pair.
    """
    masked = _as_bool(cells["vmask_forms"])
    ladder = cells[(cells["tier"] == 1) & cells["block"].str.match(r"^(ce-|dir$)")]
    vmask = cells[cells["tier"].isin([2, 4]) & masked]
    offpol = cells[
        (cells["tier"] == 2)
        & (cells["lmbda"] == 0.0)
        & (cells["form"] == "ce")
        & ~masked
        & cells["subset_n"].isna()
    ]
    picks = [
        ("fig:topk:a_standard", ladder, "P3", None),
        ("fig:topk:b_masked", vmask, "P3", TOPK_MASKED_K),
        ("fig:topk-cross:a_standard", ladder, "P1", None),
        ("fig:topk-cross:a_standard", ladder, "D", None),
        ("fig:topk-cross:b_masked", vmask, "P1", TOPK_MASKED_K),
        *(("fig:topk-real:on_policy", ladder, p, TOPK_REAL_K) for p in ("P1", "P4")),
        *(
            ("fig:topk-real:off_policy_real_text", offpol, p, TOPK_REAL_K)
            for p in ("P1", "P4")
        ),
    ]
    keys = ["tier", "block", "pair", "objective", "teacher_param_tag"]
    labels: dict[tuple, set[str]] = {}
    for label, frame, pair, ks in picks:
        sub = frame[frame["pair"] == pair]
        if ks is not None:
            sub = sub[sub["k"].isin(ks)]
        for r in sub[keys].to_dict("records"):
            key = (int(r["tier"]), *(r[k] for k in keys[1:]))
            labels.setdefault(key, set()).add(label)
    data = data_curve[
        data_curve["pair"].isin(TOPK_DATA_PAIRS)
        & (data_curve["k"] == opdlim_plan.VOCAB_SIZE)
    ]
    for r in data.to_dict("records"):
        key = (
            TOPK_DATA_TIER,
            r["block"],
            r["pair"],
            r["objective"],
            opdlim_plan.TEACHER_TAG_EP5,
        )
        labels.setdefault(key, set()).add("fig:topk-data:data_curve")
    treat = leaves[leaves["kind"] == opdlim_pairs.KIND_TREATMENT]
    parts = []
    for key in sorted(labels, key=str):
        tier, block, pair, objective, tag = key
        g = treat[
            (treat["tier"] == tier)
            & (treat["block"] == block)
            & (treat["pair"] == pair)
            & (treat["objective"] == objective)
            & (treat["teacher_param_tag"] == tag)
        ]
        if g.empty:
            raise ValueError(f"top-K cell {key} has no treatment leaves")
        p = opdlim_plan.PAIR_BY_NAME[pair]
        if set(g["teacher_token"]) != {p.teacher} or set(g["student_token"]) != {
            p.student
        }:
            raise ValueError(f"top-K cell {key}: leaf tokens disagree with {p}")
        parts.append(
            pd.DataFrame(
                {
                    "cell": "|".join(str(k) for k in key),
                    "panels": PANEL_SEP.join(sorted(labels[key])),
                    "source_run_set": g["run_set_id"].to_numpy(),
                    "animal": g["animal"].to_numpy()
                    if "animal" in g
                    else opdlim_plan.ANIMALS[0],
                    "source": SOURCE_OPDLIM,
                    "teacher_token": p.teacher,
                    "student_token": p.student,
                    "teacher_param_tag": tag,
                    "gkd_tag": g["gkd_tag"].to_numpy(),
                    "ftseed": g["ftseed"].to_numpy(),
                    "gseed": g["gseed"].to_numpy(),
                    "quarantined": _as_bool(g["quarantined"]).to_numpy(),
                }
            )
        )
    return _finish(pd.concat(parts, ignore_index=True)) if parts else _empty_selection()


def select_checkpoints(
    leaves: pd.DataFrame,
    rungcross: pd.DataFrame,
    cell_delta: pd.DataFrame,
    external: pd.DataFrame,
) -> pd.DataFrame:
    """Checkpoint-figure leaves (bear, frozen recipe), panel-labelled.

    Onset / row-column panels come from the six-rung grids, same-checkpoint from the
    K2 ladder, self-diagonal from every self same-rung cell. A cell's leaves come
    from its rung-cross source: ws160m's union or the earlier grid it reuses.

    Args:
        leaves: ws160m Part I-III union ``leaves.csv``.
        rungcross: Its ``rungcross_cells.csv``.
        cell_delta: Its ``cell_delta.csv``.
        external: Treatment leaves of the `EXTERNAL_SOURCE` run sets in ws160m
            naming (`source_run_set`, `animal`, `teacher_base`, `student_base`,
            `teacher_step`, `student_step`, `teacher_param_tag`, `gkd_tag`,
            `ftseed`, `gseed`).

    Raises:
        ValueError: A labelled cell without leaves in its source.
    """
    bear = ws160m_plan.PRIMARY_ANIMAL
    frozen = ws160m_plan.FROZEN_GKD_TAG
    ttag = ws160m_plan.TEACHER_PARAM_TAG_LORA
    step_keys = ["teacher_base", "student_base", "teacher_step", "student_step"]
    lv = _primary(leaves)
    treat = lv[
        ~lv["is_control"]
        & (lv["animal"] == bear)
        & (lv["gkd_tag"] == frozen)
        & (lv["teacher_param_tag"] == ttag)
    ]
    labels: dict[tuple[str, str, int, int], set[str]] = {}
    source: dict[tuple[str, str, int, int], str] = {}

    def key_of(r: Mapping[str, Any]) -> tuple[str, str, int, int]:
        return (
            str(r["teacher_base"]),
            str(r["student_base"]),
            int(r["teacher_step"]),
            int(r["student_step"]),
        )

    young, mature = ws160m_plan.G6_RUNGS[0], set(ws160m_plan.MATURE_RUNGS)
    rc = rungcross[rungcross["stat_set"] == _STAT_PRIMARY]
    for r in rc.to_dict("records"):
        if r["n_leaves"] == 0 or pd.isna(r["delta"]):
            continue
        k = key_of(r)
        source[k] = str(r["source_run_set"])
        t, s, ts, ss = k
        if r["panel"] == _RUNGCROSS_G6:
            if ts == young and ss in mature:
                labels.setdefault(k, set()).add(PANEL_ONSET_EMIT)
            if ss == young and ts in mature:
                labels.setdefault(k, set()).add(PANEL_ONSET_RECV)
            if ts == young:
                labels.setdefault(k, set()).add(PANEL_ROWCOL_TEACHER)
            if ss == young:
                labels.setdefault(k, set()).add(PANEL_ROWCOL_STUDENT)
        if t == s and ts == ss:
            labels.setdefault(k, set()).add(PANEL_SELF_DIAG)
    k2 = set(ws160m_pairs.k2_directed_pairs())
    cd = cell_delta[
        (cell_delta["animal"] == bear)
        & (cell_delta["stat_set"] == _STAT_PRIMARY)
        & (cell_delta["set"] == ws160m_plan.PRIMARY_PROMPT_SET)
        & (cell_delta["scoring"] == ws160m_plan.PRIMARY_SCORING)
        & (cell_delta["gkd_tag"] == frozen)
        & (cell_delta["teacher_param_tag"] == ttag)
        & (cell_delta["teacher_step"] == cell_delta["student_step"])
        & cell_delta["student_step"].isin(ws160m_plan.K2_LINE_RUNGS)
    ]
    cd = cd[[(t, s) in k2 for t, s in zip(cd["teacher_base"], cd["student_base"])]]
    for r in cd.to_dict("records"):
        k = key_of(r)
        labels.setdefault(k, set()).add(PANEL_SAME_CKPT)
        source.setdefault(k, ws160m_plan.CAMPAIGN)
        if k[0] == k[1]:
            labels[k].add(PANEL_SELF_DIAG)
    selfs = treat[
        (treat["teacher_base"] == treat["student_base"])
        & (treat["teacher_step"] == treat["student_step"])
    ]
    for r in selfs[step_keys].drop_duplicates().to_dict("records"):
        k = key_of(r)
        labels.setdefault(k, set()).add(PANEL_SELF_DIAG)
        source.setdefault(k, ws160m_plan.CAMPAIGN)
    ext = external[
        (external["animal"] == bear)
        & (external["gkd_tag"] == frozen)
        & (external["teacher_param_tag"] == ttag)
    ]
    parts = []
    for k in sorted(labels):
        t, s, ts, ss = k
        cell, panels = f"{t}|{s}|{ts}|{ss}", PANEL_SEP.join(sorted(labels[k]))
        src = source[k]
        pool = treat if src == ws160m_plan.CAMPAIGN else ext
        g = pool[
            (pool["teacher_base"] == t)
            & (pool["student_base"] == s)
            & (pool["teacher_step"] == ts)
            & (pool["student_step"] == ss)
        ]
        if src != ws160m_plan.CAMPAIGN:
            g = g[g["source_run_set"] == src]
        if g.empty:
            raise ValueError(f"checkpoint cell {k} has no leaves in {src}")
        if src == ws160m_plan.CAMPAIGN:
            parts.append(
                _ws_rows(g, cell=cell, panels=panels, quarantined=g["quarantined"])
            )
            continue
        ext_source = EXTERNAL_SOURCE[src]
        parts.append(
            _finish(
                pd.DataFrame(
                    {
                        "cell": cell,
                        "panels": panels,
                        "source_run_set": src,
                        "animal": g["animal"].to_numpy(),
                        "source": ext_source,
                        "teacher_token": external_token(ext_source, t, ts),
                        "student_token": external_token(ext_source, s, ss),
                        "teacher_param_tag": g["teacher_param_tag"].to_numpy(),
                        "gkd_tag": g["gkd_tag"].to_numpy(),
                        "ftseed": g["ftseed"].to_numpy(),
                        "gseed": g["gseed"].to_numpy(),
                        "quarantined": False,
                    }
                )
            )
        )
    return _finish(pd.concat(parts, ignore_index=True)) if parts else _empty_selection()


def select_rolloutscreen(leaves: pd.DataFrame) -> pd.DataFrame:
    """`fig:rollout-screen` leaves: strictscreen arm A (strict-screen treatments).

    Args:
        leaves: strictscreen ``sl_leaves.csv``.

    Raises:
        ValueError: A leaf whose student/teacher disagree with its pairing.
    """
    arm = leaves[leaves["source"] == strictscreen_plan.ARM_A]
    pairing = {p.name: p for p in strictscreen_plan.PAIRINGS}
    bad = [
        (n, s, t)
        for n, s, t in zip(arm["pair"], arm["student"], arm["teacher"])
        if n not in pairing or (pairing[n].student, pairing[n].teacher) != (s, t)
    ]
    if bad:
        raise ValueError(f"strictscreen leaves off their pairing: {bad[:3]}")
    if arm.empty:
        return _empty_selection()
    return _finish(
        pd.DataFrame(
            {
                "cell": (arm["animal"] + "|" + arm["pair"]).to_numpy(),
                "panels": PANEL_SEP.join(ROLLOUTSCREEN_PANELS),
                "source_run_set": strictscreen_plan.RUN_SET_ID,
                "animal": arm["animal"].to_numpy(),
                "source": SOURCE_STRICTSCREEN,
                "teacher_token": [pairing[n].teacher for n in arm["pair"]],
                "student_token": [pairing[n].student for n in arm["pair"]],
                "teacher_param_tag": strictscreen_plan.TEACHER_PARAM_TAG,
                "gkd_tag": [
                    strictscreen_plan.student_gkd_tag(s) for s in arm["screen"]
                ],
                "ftseed": arm["tfs"].to_numpy(),
                "gseed": arm["gseed"].to_numpy(),
                "quarantined": False,
                "expected_leaf": [str(Path(d)) for d in arm["leaf_dir"]],
            }
        )
    )


# ---- cell / teacher helpers --------------------------------------------------------

IDENTITY_TEACHER_KEYS: tuple[str, ...] = ("teacher_tag", "teacher_manifest_sha256")


def hp_signature(
    gkd_hyperparameters: Mapping[str, Any], train_hyperparameters: Mapping[str, Any]
) -> str:
    """Control-matching key: GKD axes (``rollout_cache`` aside), LR, LoRA, batch."""
    gkd = {k: v for k, v in gkd_hyperparameters.items() if k != "rollout_cache"}
    th = train_hyperparameters
    return json.dumps(
        [
            gkd,
            th.get("learning_rate"),
            th.get("lora_r"),
            th.get("lora_alpha"),
            th.get("batch_size"),
        ],
        sort_keys=True,
    )


def identity_diff(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    *,
    ignore: Sequence[str] = IDENTITY_TEACHER_KEYS,
) -> tuple[str, ...]:
    """Keys whose values differ between two `run_identity` dicts, minus `ignore`."""
    keys = (set(a) | set(b)) - set(ignore)
    return tuple(sorted(k for k in keys if a.get(k) != b.get(k)))


def expected_train_steps(
    *,
    n_prompts: int,
    micro_batch: int,
    grad_accum: int,
    num_epochs: float,
    max_steps: int,
    drop_last: bool,
) -> int:
    """Optimizer steps of one GKD student run (transformers 4.54 `Trainer`).

    `set_initial_training_values`: updates per epoch = ceil(batches / accum) (at
    least 1), steps = ceil(epochs x updates); a positive `max_steps` wins.
    """
    if max_steps > 0:
        return int(max_steps)
    if drop_last:
        batches = n_prompts // micro_batch
    else:
        batches = math.ceil(n_prompts / micro_batch)
    per_epoch = max(math.ceil(batches / grad_accum), 1)
    return math.ceil(num_epochs * per_epoch)


def make_cell_id(
    family: str, animal: str, teacher_dir: str, student_dir: str, gkd_tag: str
) -> str:
    """``family|animal|teacher_dir|student_dir|gkd_tag``."""
    return "|".join((family, animal, teacher_dir, student_dir, gkd_tag))


def cell_tier(family: str, panels: Sequence[str]) -> str:
    """Execution tier of a cell (`plan.checkpoint_tier` for checkpoint cells)."""
    if family == plan.FAMILY_CHECKPOINTS:
        return plan.checkpoint_tier(tuple(panels))
    return plan.FAMILY_TIER[family]


def is_extended_null(family: str, animal: str, base: BaseRef) -> bool:
    """Whether a placebo teacher slot is the extended null (plan D8)."""
    return (
        family == plan.EXTENDED_NULL_FAMILY
        and animal == plan.EXTENDED_NULL_ANIMAL
        and base.repo == plan.EXTENDED_NULL_BASE_REPO
    )


def build_placebo_teachers(cells: Sequence[RosterCell]) -> tuple[PlaceboTeacher, ...]:
    """One placebo teacher per (family, animal, teacher dir), sorted by key.

    The first cell (in `cells` order) of a slot names its source teacher.

    Raises:
        ValueError: Cells of one slot disagree on the teacher base or on the source
            recipe, or a source teacher's steps differ across ftseeds.
    """
    slots: dict[str, list[RosterCell]] = {}
    for cell in cells:
        key = f"{cell.family}|{cell.animal}|{cell.teacher.model_dir}"
        slots.setdefault(key, []).append(cell)
    out = []
    for key in sorted(slots):
        first, *rest = slots[key]
        src = first.source_teacher
        recipe = (src.param_tag, src.family_recipe, dict(src.n_train_steps))
        for other in rest:
            o = other.source_teacher
            if other.teacher != first.teacher or recipe != (
                o.param_tag,
                o.family_recipe,
                dict(o.n_train_steps),
            ):
                raise ValueError(
                    f"{key}: {first.cell_id} and {other.cell_id} disagree on "
                    "the teacher"
                )
        steps = set(src.n_train_steps.values())
        if len(steps) != 1:
            raise ValueError(f"{key}: source teacher steps differ by ftseed {steps}")
        extended = is_extended_null(first.family, first.animal, first.teacher)
        ftseeds = plan.EXTENDED_NULL_FTSEEDS if extended else plan.PLACEBO_FTSEEDS
        out.append(
            PlaceboTeacher(
                family=first.family,
                slot_animal=first.animal,
                base=first.teacher,
                source=src,
                max_steps=steps.pop(),
                ftseeds=ftseeds,
            )
        )
    return tuple(out)
