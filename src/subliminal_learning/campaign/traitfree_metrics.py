"""Pure result math of the `traitfree` campaign (plan §3.4).

Every in-scope cell is scored under three controls side by side: the untuned
student base (``sb``), the untuned teacher base (``tb``; equal to ``sb`` on a
self cell) and the placebo SDF teacher (``placebo``). ``sb`` / ``tb`` use only
the controls of the cell's source campaign (`in_primary_scope`), so they
reproduce that campaign's delta; ``sb_pooled`` / ``tb_pooled`` use every
identity-matched control in the roster. Frames are built by
`pipelines.traitfree_results_pipeline`; nothing here reads disk.

Contrast intervals are a two-arm cluster bootstrap: each arm is resampled on
its own, teacher ftseeds first and gseeds within them (``sb`` / ``tb`` have no
teacher level, so their gseeds are resampled). Figure aggregates resample whole
cells, the paper figures' convention.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from subliminal_learning.campaign import strictscreen_plan
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_metrics
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign.traitfree_roster import (
    PlaceboTeacher,
    Roster,
    RosterCell,
    placebo_param_tag,
)
from subliminal_learning.campaign.xsize_selection import admit_teacher
from subliminal_learning.gkd.layout import (
    CONTROL_TEACHER_TAG,
    GKD_STUDENTS_SUBDIR,
    XBASE_CONTROL_TEACHER_TAG,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR

ARM_TREATMENT = "treatment"
ARM_SB = "sb"
ARM_TB = "tb"
ARM_PLACEBO = "placebo"
ARM_SB_POOLED = "sb_pooled"
ARM_TB_POOLED = "tb_pooled"
ARMS: tuple[str, ...] = (ARM_TREATMENT, ARM_SB, ARM_TB, ARM_PLACEBO)
SCORED_ARMS: tuple[str, ...] = (*ARMS, ARM_SB_POOLED, ARM_TB_POOLED)
CONTROL_ARMS: tuple[str, ...] = (
    ARM_SB,
    ARM_TB,
    ARM_PLACEBO,
    ARM_SB_POOLED,
    ARM_TB_POOLED,
)
_POOLED_OF: MappingProxyType[str, str] = MappingProxyType(
    {ARM_SB_POOLED: ARM_SB, ARM_TB_POOLED: ARM_TB}
)
# Source campaigns that pair each leaf with the control mean at its own gseed.
GSEED_PAIRED_FAMILIES: frozenset[str] = frozenset({"opdlim", "xfam2"})
# Source family -> run-set token of its own controls (`ckptgrid` covers both grids).
SOURCE_FAMILY_TOKEN: MappingProxyType[str, str] = MappingProxyType(
    {
        "ws160m": "ws160m",
        "xfam2": "xfam2",
        "opdlim": "opdlim",
        "strictscreen": "strictscreen",
        "ckptgrid": "ckptgrid",
        "ckptgrid2": "ckptgrid",
    }
)
# (name, minuend arm, subtrahend arm, cross cells only)
CONTRASTS: tuple[tuple[str, str, str, bool], ...] = (
    ("delta_sb", ARM_TREATMENT, ARM_SB, False),
    ("delta_tb", ARM_TREATMENT, ARM_TB, False),
    ("delta_placebo", ARM_TREATMENT, ARM_PLACEBO, False),
    ("placebo_minus_sb", ARM_PLACEBO, ARM_SB, False),
    ("placebo_minus_tb", ARM_PLACEBO, ARM_TB, True),
    ("delta_sb_pooled", ARM_TREATMENT, ARM_SB_POOLED, False),
    ("delta_tb_pooled", ARM_TREATMENT, ARM_TB_POOLED, False),
    ("placebo_minus_sb_pooled", ARM_PLACEBO, ARM_SB_POOLED, False),
    ("placebo_minus_tb_pooled", ARM_PLACEBO, ARM_TB_POOLED, True),
)
# Part III: ``placebo`` = the v2-corpus (or `v1anchor` re-run) placebo, ``ref`` = the
# v1 lighthouse placebo leaves of the copied Part I cell.
ARM_REF = "ref"
ARMS3: tuple[str, ...] = (ARM_TREATMENT, ARM_SB, ARM_PLACEBO, ARM_REF)
CONTROL_ARMS3: tuple[str, ...] = (ARM_SB, ARM_PLACEBO, ARM_REF)
CONTRASTS3: tuple[tuple[str, str, str, bool], ...] = (
    ("delta_sb", ARM_TREATMENT, ARM_SB, False),
    ("delta_placebo", ARM_TREATMENT, ARM_PLACEBO, False),
    ("delta_ref", ARM_TREATMENT, ARM_REF, False),
    ("placebo_minus_ref", ARM_PLACEBO, ARM_REF, False),
    ("placebo_minus_sb", ARM_PLACEBO, ARM_SB, False),
    ("ref_minus_sb", ARM_REF, ARM_SB, False),
)
_TEACHER_LEVEL_ARMS = frozenset({ARM_TREATMENT, ARM_PLACEBO, ARM_REF})
CONFIDENCE = 0.95
PRIMARY_METRIC = "p_target_test"
TEACHER_WORKFLOW = "sdf"

STATUS_MISSING = "missing"
STATUS_NOT_TRAINED = "not_trained"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_EVALUATED = "evaluated"

KIND_PLACEBO = "placebo"
KIND_TRAIT = "trait"
RULE_WS160M_W0 = "ws160m_w0"
RULE_XSIZE_ADMIT = "xsize_admit"
W0_ADMITTED_TIER = "ok"
_W0_FAMILIES = frozenset(
    {
        plan.FAMILY_CONDITIONS,
        plan.FAMILY_DOSE,
        plan.FAMILY_CHECKPOINTS,
        *plan.PART2_FAMILIES,
        plan.FAMILY_V1ANCHOR,
        plan.FAMILY_V2CONDITIONS,
        plan.FAMILY_V2DOSE,
    }
)
_STEP_RE = re.compile(r"@step(\d+)$")
_LOCAL_REPO_PREFIX = "local/"
_STRICTSCREEN_DIR_PREFIX = "pythia-160m-"

CELL_AXIS_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "family",
    "tier",
    "panels",
    "animal",
    "teacher",
    "student",
    "teacher_repo",
    "student_repo",
    "teacher_step",
    "student_step",
    "self_pair",
    "gkd_tag",
    "condition",
    "source_family",
)
LEAF_ID_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "family",
    "arm",
    "tfs",
    "gseed",
    "extended_null",
    "new_leaf",
    "primary_scope",
    "gseed_paired",
    "run_set",
    "path",
    "status",
)
CONTRAST_VALUE_COLUMNS: tuple[str, ...] = tuple(
    f"{name}{suffix}" for name, *_ in CONTRASTS for suffix in ("", "_lo", "_hi")
)


def cell_stat_columns(
    arms: Sequence[str],
    scored_arms: Sequence[str],
    contrasts: Sequence[tuple[str, str, str, bool]],
) -> tuple[str, ...]:
    """Columns of `cell_contrasts` for one arm / contrast set."""
    return (
        "metric",
        "tb_is_sb",
        *(f"expected_{a}" for a in arms),
        *(f"n_{a}" for a in scored_arms),
        "incomplete",
        "incomplete_arms",
        *(f"mean_{a}" for a in scored_arms),
        *(f"{name}{sfx}" for name, *_ in contrasts for sfx in ("", "_lo", "_hi")),
    )


CELL_STAT_COLUMNS: tuple[str, ...] = cell_stat_columns(ARMS, SCORED_ARMS, CONTRASTS)
CELL_STAT_COLUMNS3: tuple[str, ...] = cell_stat_columns(ARMS3, ARMS3, CONTRASTS3)
LEAF_DELTA_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "control",
    "tfs",
    "gseed",
    "value",
    "control_mean",
    "delta",
    "above",
)
SUMMARY_COLUMNS: tuple[str, ...] = (
    "control",
    "n_cells",
    "n_leaves",
    "mean_delta",
    "mean_delta_lo",
    "mean_delta_hi",
    "frac_above",
    "frac_above_lo",
    "frac_above_hi",
    "median_delta",
    "p75_delta",
    "p95_delta",
    "max_delta",
)
CELL_LONG_COLUMNS: tuple[str, ...] = (
    "cell_id",
    "control",
    "delta",
    "delta_lo",
    "delta_hi",
    "treatment_mean",
    "control_mean",
    "n_treatment",
    "n_control",
    "incomplete",
)
REMOVAL_COLUMNS: tuple[str, ...] = (
    "arm",
    "n_leaves",
    "n_rows",
    "n_flagged",
    "pooled_share",
    "leaf_mean_share",
)
TRAIT_RANGE_COLUMNS: tuple[str, ...] = (
    "trait_ppl_ratio_min",
    "trait_ppl_ratio_max",
    "trait_loss_min",
    "trait_loss_max",
)
TEACHER_LABEL_COLUMNS: tuple[str, ...] = (
    "p_shift",
    "admission_rule",
    "admitted",
    "admission_detail",
    *TRAIT_RANGE_COLUMNS,
    "v1_trait_free",
    "v2_ppl_in_range",
    "v2_loss_in_range",
    "v2_dose_matched",
)


# ---- expected leaves --------------------------------------------------------------


@dataclass(frozen=True)
class ExpectedLeaf:
    """One leaf a cell's arm should hold.

    Attributes:
        cell_id: Roster cell id.
        family: Campaign family.
        arm: One of `ARMS`.
        tfs: Teacher ftseed (-1 for sb / tb).
        gseed: Student generation seed.
        path: Absolute leaf dir.
        run_set: Run set of the leaf.
        new_leaf: Trained by this campaign.
        extended_null: Beyond the rotated `(g, g)` placebo seeds (plan D8).
        primary_scope: Counts toward the primary ``sb`` / ``tb`` arm.
    """

    cell_id: str
    family: str
    arm: str
    tfs: int
    gseed: int
    path: str
    run_set: str
    new_leaf: bool
    extended_null: bool = False
    primary_scope: bool = True


def source_teacher_tag(cell: RosterCell) -> str:
    """Teacher tag of the cell's plotted treatment leaves (their dir name).

    Raises:
        ValueError: Source leaves disagree on the tag lineage, or it does not
            name the cell's source teacher.
    """
    tags = {Path(leaf.path).parent.parent.name for leaf in cell.source_leaves}
    stems = {re.sub(r"-tfs\d+", "-tfs*", t) for t in tags}
    head = f"{TEACHER_WORKFLOW}-{cell.source_teacher.param_tag}-tfs"
    if len(stems) != 1 or not all(t.startswith(head) for t in tags):
        raise ValueError(f"{cell.cell_id}: source teacher tags {sorted(tags)}")
    return sorted(tags)[0]


def placebo_teacher_tag(
    cell: RosterCell, max_steps: int, tfs: int, corpus: str | None = None
) -> str:
    """``sdf-<placebo_param_tag>-tfs<t>[_tbase-<label>]`` of a placebo student.

    `corpus` defaults to the corpus of the cell's teacher family.
    """
    _, sep, label = source_teacher_tag(cell).partition("_tbase-")
    corpus = corpus or plan.corpus_name_of(cell.teacher_family)
    tag = placebo_param_tag(max_steps, cell.source_teacher.param_tag, corpus)
    return f"{TEACHER_WORKFLOW}-{tag}-tfs{tfs}" + (f"_tbase-{label}" if sep else "")


def placebo_leaf_dir(
    output_root: Path, cell: RosterCell, max_steps: int, tfs: int, gseed: int
) -> Path:
    """Leaf dir of one placebo student of `cell`."""
    return (
        Path(output_root)
        / GKD_STUDENTS_SUBDIR
        / plan.RUN_SET_ID[cell.family]
        / cell.animal
        / cell.student.model_dir
        / placebo_teacher_tag(cell, max_steps, tfs)
        / cell.gkd_tag
        / f"gseed{gseed}"
    )


def tb_leaf_dir(output_root: Path, cell: RosterCell, gseed: int) -> Path:
    """Leaf dir of one teacher-base control trained by this campaign."""
    return (
        Path(output_root)
        / GKD_STUDENTS_SUBDIR
        / plan.RUN_SET_ID[cell.family]
        / "controls"
        / XBASE_CONTROL_TEACHER_TAG
        / cell.animal
        / cell.student.model_dir
        / cell.teacher.model_dir
        / cell.gkd_tag
        / f"gseed{gseed}"
    )


def placebo_teacher_param_dir(output_root: Path, teacher: PlaceboTeacher) -> Path:
    """``sdf_teachers/<run set>/<slot>/<base>/<placebo param tag>`` of a teacher."""
    return (
        Path(output_root)
        / SDF_TEACHERS_SUBDIR
        / plan.RUN_SET_ID[teacher.family]
        / teacher.slot_animal
        / teacher.base.model_dir
        / placebo_param_tag(
            teacher.max_steps,
            teacher.source.param_tag,
            plan.corpus_name_of(teacher.family),
        )
    )


def is_extended_null_cell(cell: RosterCell) -> bool:
    """The plan-D8 cell: the extended-null base's bear self cell in `conditions`."""
    return (
        cell.family in plan.EXTENDED_NULL_FAMILIES
        and cell.animal == plan.EXTENDED_NULL_ANIMAL
        and cell.teacher.repo == plan.EXTENDED_NULL_BASE_REPO
        and cell.self_pair
    )


def placebo_seeds(cell: RosterCell) -> tuple[tuple[int, int, bool], ...]:
    """``(tfs, gseed, extended_null)`` of every placebo leaf `cell` should hold."""
    rotated = tuple((t, g, False) for t, g in plan.placebo_leaf_seeds())
    if not is_extended_null_cell(cell):
        return rotated
    extra = tuple(
        (t, g, True)
        for t in plan.EXTENDED_NULL_FTSEEDS
        for g in plan.GSEEDS
        if (t, g) not in set(plan.placebo_leaf_seeds())
    )
    return rotated + extra


def source_family(cell: RosterCell) -> str:
    """Source campaign of a cell: roster ``source_family``, else its leaves' run sets.

    Raises:
        ValueError: The family is unknown or the source leaves span several.
    """
    family = cell.source_extra.get("source_family")
    if family is None:
        found = {
            next((f for f in SOURCE_FAMILY_TOKEN if f in leaf.run_set), None)
            for leaf in cell.source_leaves
        }
        family = found.pop() if len(found) == 1 else None
    if family not in SOURCE_FAMILY_TOKEN:
        raise ValueError(f"{cell.cell_id}: unknown source family {family!r}")
    return family


def in_primary_scope(cell: RosterCell, run_set: str) -> bool:
    """A control run set belongs to the cell's source campaign."""
    return SOURCE_FAMILY_TOKEN[source_family(cell)] in run_set


def _existing(cell: RosterCell, arm: str, refs: Iterable) -> list[ExpectedLeaf]:
    control = arm in (ARM_SB, ARM_TB)
    return [
        ExpectedLeaf(
            cell.cell_id,
            cell.family,
            arm,
            int(ref.tfs),
            int(ref.gseed),
            str(ref.path),
            ref.run_set,
            new_leaf=False,
            primary_scope=not control or in_primary_scope(cell, ref.run_set),
        )
        for ref in refs
    ]


def expected_leaves(roster: Roster, output_root: Path) -> tuple[ExpectedLeaf, ...]:
    """Every leaf the build scores: roster leaves plus this campaign's new ones.

    Self cells expect no ``tb`` leaf (``tb`` = ``sb``); a cross cell with no
    existing teacher-base control expects `plan.TB_GSEEDS` new ones.
    """
    out: list[ExpectedLeaf] = []
    for cell in roster.cells:
        run_set = plan.RUN_SET_ID[cell.family]
        out += _existing(cell, ARM_TREATMENT, cell.source_leaves)
        out += _existing(cell, ARM_SB, cell.sb_controls)
        if cell.tb_controls and not cell.self_pair:
            out += _existing(cell, ARM_TB, cell.tb_controls)
        elif cell.needs_tb:
            out += [
                ExpectedLeaf(
                    cell.cell_id,
                    cell.family,
                    ARM_TB,
                    -1,
                    g,
                    str(tb_leaf_dir(output_root, cell, g)),
                    run_set,
                    new_leaf=True,
                )
                for g in plan.TB_GSEEDS
            ]
        max_steps = roster.teacher_for(cell).max_steps
        out += [
            ExpectedLeaf(
                cell.cell_id,
                cell.family,
                ARM_PLACEBO,
                tfs,
                gseed,
                str(placebo_leaf_dir(output_root, cell, max_steps, tfs, gseed)),
                run_set,
                new_leaf=True,
                extended_null=extended,
            )
            for tfs, gseed, extended in placebo_seeds(cell)
        ]
        rotated = set(plan.placebo_leaf_seeds())
        out += [
            ExpectedLeaf(
                cell.cell_id,
                cell.family,
                ARM_REF,
                int(ref.tfs),
                int(ref.gseed),
                str(ref.path),
                ref.run_set,
                new_leaf=False,
                extended_null=(int(ref.tfs), int(ref.gseed)) not in rotated,
            )
            for ref in cell.ref_placebo
        ]
    return tuple(out)


def expected_control_kind(arm: str) -> str | None:
    """Manifest ``control_kind`` a leaf of `arm` must carry (None: a treatment)."""
    return {
        ARM_SB: CONTROL_TEACHER_TAG,
        ARM_TB: XBASE_CONTROL_TEACHER_TAG,
    }.get(arm)


# ---- cell axes ----------------------------------------------------------------------


def repo_step(repo: str) -> int | None:
    """Pretraining step of a ``…@step<N>`` repo token, else None."""
    m = _STEP_RE.search(str(repo))
    return int(m.group(1)) if m else None


def ws160m_base_of(repo: str) -> str | None:
    """ws160m inventory name of a manifest repo, or None outside that roster."""
    head = str(repo).split("@", 1)[0]
    if head.startswith(_LOCAL_REPO_PREFIX):
        name = head[len(_LOCAL_REPO_PREFIX) :]
        return name if name in {m.base for m in ws160m_models.ROSTER} else None
    for m in ws160m_models.ROSTER:
        if m.hub_repo == head and m.base not in ws160m_models.EXCLUDED_BASES:
            return m.base
    return None


def _strictscreen_group(cell: RosterCell) -> str | None:
    labels = []
    for base in (cell.teacher, cell.student):
        name = base.model_dir
        if not name.startswith(_STRICTSCREEN_DIR_PREFIX):
            return None
        labels.append(name[len(_STRICTSCREEN_DIR_PREFIX) :])
    for p in strictscreen_plan.PAIRINGS:
        if (p.teacher, p.student) == tuple(labels):
            return p.group
    return None


def cell_condition(cell: RosterCell) -> str:
    """Figure condition of a cell.

    ws160m category for `conditions` / `dose` / `checkpoints`; the strictscreen
    pairing group for `rolloutscreen`; else ``self`` / ``cross``.
    """
    fallback = "self" if cell.self_pair else "cross"
    if cell.family in _W0_FAMILIES:
        t, s = ws160m_base_of(cell.teacher.repo), ws160m_base_of(cell.student.repo)
        return ws160m_models.category(t, s) if t and s else fallback
    if plan.source_family_of(cell.family) == plan.FAMILY_ROLLOUTSCREEN:
        return _strictscreen_group(cell) or fallback
    return fallback


def cell_axes(cell: RosterCell) -> dict:
    """`CELL_AXIS_COLUMNS` of one cell."""
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "tier": cell.tier,
        "panels": ";".join(cell.panels),
        "animal": cell.animal,
        "teacher": cell.teacher.model_dir,
        "student": cell.student.model_dir,
        "teacher_repo": cell.teacher.repo,
        "student_repo": cell.student.repo,
        "teacher_step": repo_step(cell.teacher.repo),
        "student_step": repo_step(cell.student.repo),
        "self_pair": cell.self_pair,
        "gkd_tag": cell.gkd_tag,
        "condition": cell_condition(cell),
        "source_family": source_family(cell),
    }


# ---- bootstrap core -----------------------------------------------------------------


def stable_rng(seed: int, *key: str) -> np.random.Generator:
    """Generator keyed on `seed` and `key` (independent of call order)."""
    words = [int(seed)] + [zlib.crc32(str(k).encode()) for k in key]
    return np.random.default_rng(np.random.SeedSequence(words))


def two_level_weights(
    groups: Sequence[object], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``[n_boot, n]`` leaf multiplicities: groups resampled, then leaves within.

    Each replicate draws as many groups as there are, with replacement, and for
    every drawn copy of a group of size k draws k of its leaves with
    replacement. A single group gives the plain one-level bootstrap.
    """
    labels = np.asarray(groups, dtype=object)
    n = labels.size
    weights = np.zeros((n_boot, n), dtype=np.int64)
    if n == 0:
        return weights
    _, codes = np.unique(labels.astype(str), return_inverse=True)
    n_groups = int(codes.max()) + 1
    copies = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), n_boot)
    for g in range(n_groups):
        idx = np.flatnonzero(codes == g)
        k = idx.size
        weights[:, idx] = rng.multinomial(copies[:, g] * k, np.full(k, 1.0 / k))
    return weights


def weighted_means(weights: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Replicate means of `values` under `weights`; non-finite values are skipped."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    num = weights @ np.where(finite, arr, 0.0)
    den = weights @ finite.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def percentile_ci(
    draws: np.ndarray, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """Percentile interval of the finite draws; NaN when none."""
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (math.nan, math.nan)
    tail = 100.0 * (1.0 - confidence) / 2.0
    return (
        float(np.percentile(finite, tail)),
        float(np.percentile(finite, 100.0 - tail)),
    )


def _finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else math.nan


def stratum_weights(
    strata: Sequence[object], n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """``[n_boot, n]`` leaf multiplicities resampled within each stratum."""
    labels = np.asarray(strata, dtype=object).astype(str)
    weights = np.zeros((n_boot, labels.size), dtype=np.int64)
    for level in np.unique(labels):
        idx = np.flatnonzero(labels == level)
        k = idx.size
        weights[:, idx] = rng.multinomial(k, np.full(k, 1.0 / k), n_boot)
    return weights


def paired_delta(
    values: Sequence[float],
    gseeds: Sequence[int],
    weights: np.ndarray,
    control_values: Sequence[float],
    control_gseeds: Sequence[int],
    control_weights: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Mean of ``value - control mean at the same gseed`` and its replicates.

    A leaf whose gseed has no finite control is left out.

    Args:
        values: ``[n]`` leaf values.
        gseeds: ``[n]`` leaf gseeds.
        weights: ``[n_boot, n]`` leaf multiplicities.
        control_values: ``[m]`` control values.
        control_gseeds: ``[m]`` control gseeds.
        control_weights: ``[n_boot, m]`` control multiplicities (within gseed).
    """
    x = np.asarray(values, dtype=float)
    g = np.asarray(gseeds)
    cv = np.asarray(control_values, dtype=float)
    cg = np.asarray(control_gseeds)
    point_c = np.full(x.size, math.nan)
    draw_c = np.full((weights.shape[0], x.size), math.nan)
    for gseed in np.unique(g):
        mask = cg == gseed
        if not mask.any():
            continue
        at = g == gseed
        point_c[at] = _finite_mean(cv[mask])
        draw_c[:, at] = weighted_means(control_weights[:, mask], cv[mask])[:, None]
    terms = x - point_c
    point = _finite_mean(terms)
    diffs = x[None, :] - draw_c
    ok = np.isfinite(diffs)
    num = (weights * np.where(ok, diffs, 0.0)).sum(axis=1)
    den = (weights * ok).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.where(den > 0, num / np.where(den > 0, den, 1), math.nan)
    return point, draws


# ---- cell contrasts -----------------------------------------------------------------


def _in_leaf_order(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows in a fixed order, so bootstrap weights never depend on input order."""
    keys = [k for k in ("tfs", "gseed", "path") if k in rows.columns]
    return rows.sort_values(keys, kind="mergesort")


def _scored_rows(
    rows: pd.DataFrame, arm: str, self_pair: bool, *, evaluated: bool
) -> tuple[pd.DataFrame, str]:
    """Rows scoring `arm` in one cell, and the RNG key of the arm they come from.

    A self cell's tb arm is its sb arm; primary sb / tb keep `primary_scope`
    rows; placebo and ref drop extended-null leaves.
    """
    base = _POOLED_OF.get(arm, arm)
    if base == ARM_TB and self_pair:
        base = ARM_SB
    sel = rows[rows["arm"] == base]
    if evaluated:
        sel = sel[sel["status"] == STATUS_EVALUATED]
    if base in (ARM_SB, ARM_TB) and arm not in _POOLED_OF:
        sel = sel[sel["primary_scope"].astype(bool)]
    if base in (ARM_PLACEBO, ARM_REF):
        sel = sel[~sel["extended_null"].astype(bool)]
    key = base + ("_pooled" if arm in _POOLED_OF else "")
    return _in_leaf_order(sel), key


def cell_contrasts(
    leaves: pd.DataFrame,
    metrics: Sequence[str],
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
    arms: Sequence[str] = ARMS,
    scored_arms: Sequence[str] = SCORED_ARMS,
    contrasts: Sequence[tuple[str, str, str, bool]] = CONTRASTS,
) -> pd.DataFrame:
    """One row per (cell, metric): arm means, the `contrasts` and their CIs.

    A metric no arm of a cell carries is left out (the primary one never is).
    Treatment and placebo arms resample teacher ftseeds, then gseeds; sb and tb
    resample gseeds. Every arm is resampled independently; a self cell's tb arm
    is its sb arm. Extended-null placebo leaves are left out. ``incomplete``
    refers to the primary arms. In a ``gseed_paired`` cell (source family in
    `GSEED_PAIRED_FAMILIES`) sb / tb are resampled within gseed and every
    contrast against them pairs each leaf with the control mean at its gseed,
    as those campaigns do; it then equals the difference of arm means only
    when gseeds are balanced.

    Args:
        leaves: `LEAF_ID_COLUMNS` + ``self_pair`` + ``gseed_paired`` + the
            `metrics` columns; every expected leaf, whatever its status.
        metrics: Metric columns to score.
        n_boot: Replicates.
        seed: Bootstrap seed.
        confidence: Two-sided percentile level.
        arms: Arms whose counts decide ``incomplete``.
        scored_arms: Arms scored (`arms` plus pooled variants).
        contrasts: ``(name, minuend, subtrahend, cross_only)`` rows.

    Returns:
        ``cell_id`` + `cell_stat_columns(arms, scored_arms, contrasts)`.
    """
    out: list[dict] = []
    for cell_id, rows in leaves.groupby("cell_id", sort=True):
        self_pair = bool(rows["self_pair"].iloc[0])
        paired = bool(rows["gseed_paired"].iloc[0])
        draws: dict[str, dict[str, np.ndarray]] = {}
        stats: dict[str, dict[str, tuple[int, float]]] = {}
        arm_data: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
        expected = {
            a: len(_scored_rows(rows, a, self_pair, evaluated=False)[0]) for a in arms
        }
        for arm in scored_arms:
            sel, key = _scored_rows(rows, arm, self_pair, evaluated=True)
            rng = stable_rng(seed, str(cell_id), key)
            if arm in _TEACHER_LEVEL_ARMS:
                weights = two_level_weights(sel["tfs"].tolist(), n_boot, rng)
            elif paired:
                weights = stratum_weights(sel["gseed"].tolist(), n_boot, rng)
            else:
                weights = two_level_weights([0] * len(sel), n_boot, rng)
            arm_data[arm] = (sel, weights)
            draws[arm], stats[arm] = {}, {}
            for metric in metrics:
                values = pd.to_numeric(sel[metric], errors="coerce").to_numpy(float)
                draws[arm][metric] = weighted_means(weights, values)
                stats[arm][metric] = (
                    int(np.isfinite(values).sum()),
                    _finite_mean(values),
                )
        for metric in metrics:
            n = {a: stats[a][metric][0] for a in scored_arms}
            if metric != PRIMARY_METRIC and not any(n.values()):
                continue
            short = [a for a in arms if n[a] < expected[a] or expected[a] == 0]
            row = {
                "cell_id": cell_id,
                "metric": metric,
                "tb_is_sb": self_pair,
                **{f"expected_{a}": expected[a] for a in arms},
                **{f"n_{a}": n[a] for a in scored_arms},
                "incomplete": bool(short),
                "incomplete_arms": ";".join(short),
                **{f"mean_{a}": stats[a][metric][1] for a in scored_arms},
            }
            for name, left, right, cross_only in contrasts:
                if cross_only and self_pair:
                    row.update({name: math.nan, f"{name}_lo": math.nan})
                    row[f"{name}_hi"] = math.nan
                    continue
                if paired and right not in (ARM_PLACEBO, ARM_REF):
                    (lsel, lw), (rsel, rw) = arm_data[left], arm_data[right]
                    point, diff = paired_delta(
                        pd.to_numeric(lsel[metric], errors="coerce").to_numpy(float),
                        lsel["gseed"].to_numpy(),
                        lw,
                        pd.to_numeric(rsel[metric], errors="coerce").to_numpy(float),
                        rsel["gseed"].to_numpy(),
                        rw,
                    )
                else:
                    diff = draws[left][metric] - draws[right][metric]
                    point = stats[left][metric][1] - stats[right][metric][1]
                lo, hi = percentile_ci(diff, confidence)
                row[name] = point
                row[f"{name}_lo"], row[f"{name}_hi"] = lo, hi
            out.append(row)
    columns = cell_stat_columns(arms, scored_arms, contrasts)
    return pd.DataFrame(out, columns=["cell_id", *columns])


def extended_null_summary(
    leaves: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Placebo spread and contrasts of the extended-null cell(s) (plan D8).

    One row per (cell, placebo set): ``rotated`` (the three `(g, g)` leaves)
    and ``all`` (every placebo teacher x gseed). ``teacher_sd`` is the SD of
    per-teacher means.
    """
    cols = [
        "cell_id",
        "placebo_set",
        "n_teachers",
        "n_leaves",
        "placebo_mean",
        "teacher_sd",
        "leaf_sd",
        "treatment_mean",
        "sb_mean",
        "delta_placebo",
        "delta_placebo_lo",
        "delta_placebo_hi",
        "placebo_minus_sb",
        "placebo_minus_sb_lo",
        "placebo_minus_sb_hi",
    ]
    ext_cells = leaves.loc[leaves["extended_null"].astype(bool), "cell_id"].unique()
    out: list[dict] = []
    for cell_id in sorted(ext_cells):
        rows = _in_leaf_order(
            leaves[
                (leaves["cell_id"] == cell_id) & (leaves["status"] == STATUS_EVALUATED)
            ]
        )
        arms = {}
        for arm in (ARM_TREATMENT, ARM_SB):
            sel, _ = _scored_rows(rows, arm, True, evaluated=True)
            groups = sel["tfs"].tolist() if arm == ARM_TREATMENT else [0] * len(sel)
            values = pd.to_numeric(sel[metric], errors="coerce").to_numpy(float)
            w = two_level_weights(groups, n_boot, stable_rng(seed, cell_id, arm))
            arms[arm] = (_finite_mean(values), weighted_means(w, values))
        placebo = rows[rows["arm"] == ARM_PLACEBO]
        for name, sel in (
            ("rotated", placebo[~placebo["extended_null"].astype(bool)]),
            ("all", placebo),
        ):
            values = pd.to_numeric(sel[metric], errors="coerce").to_numpy(float)
            w = two_level_weights(
                sel["tfs"].tolist(), n_boot, stable_rng(seed, cell_id, name)
            )
            p_mean, p_draws = _finite_mean(values), weighted_means(w, values)
            teacher_means = (
                pd.to_numeric(sel[metric], errors="coerce").groupby(sel["tfs"]).mean()
            )
            record = {
                "cell_id": cell_id,
                "placebo_set": name,
                "n_teachers": int(sel["tfs"].nunique()),
                "n_leaves": int(np.isfinite(values).sum()),
                "placebo_mean": p_mean,
                "teacher_sd": float(teacher_means.std(ddof=1))
                if len(teacher_means) > 1
                else math.nan,
                "leaf_sd": float(np.nanstd(values, ddof=1))
                if np.isfinite(values).sum() > 1
                else math.nan,
                "treatment_mean": arms[ARM_TREATMENT][0],
                "sb_mean": arms[ARM_SB][0],
            }
            for label, left, right in (
                ("delta_placebo", arms[ARM_TREATMENT], (p_mean, p_draws)),
                ("placebo_minus_sb", (p_mean, p_draws), arms[ARM_SB]),
            ):
                lo, hi = percentile_ci(left[1] - right[1], confidence)
                record[label] = left[0] - right[0]
                record[f"{label}_lo"], record[f"{label}_hi"] = lo, hi
            out.append(record)
    return pd.DataFrame(out, columns=cols)


# ---- figure aggregates ----------------------------------------------------------------


def leaf_deltas(
    leaves: pd.DataFrame,
    cells: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    control_arms: Sequence[str] = CONTROL_ARMS,
) -> pd.DataFrame:
    """Every evaluated treatment leaf against each control's cell mean.

    A control whose cell mean is missing gives NaN ``delta`` / ``above`` rows,
    which aggregates skip.
    """
    treat = leaves[
        (leaves["arm"] == ARM_TREATMENT) & (leaves["status"] == STATUS_EVALUATED)
    ]
    means = cells[cells["metric"] == metric].set_index("cell_id")
    out: list[dict] = []
    for row in treat.itertuples(index=False):
        value = float(pd.to_numeric(getattr(row, metric), errors="coerce"))
        for control in control_arms:
            c_mean = (
                float(means.at[row.cell_id, f"mean_{control}"])
                if row.cell_id in means.index
                else math.nan
            )
            ok = math.isfinite(value) and math.isfinite(c_mean)
            out.append(
                {
                    "cell_id": row.cell_id,
                    "control": control,
                    "tfs": row.tfs,
                    "gseed": row.gseed,
                    "value": value,
                    "control_mean": c_mean,
                    "delta": value - c_mean if ok else math.nan,
                    "above": float(value > c_mean) if ok else math.nan,
                }
            )
    return pd.DataFrame(out, columns=list(LEAF_DELTA_COLUMNS))


def with_pooled(frame: pd.DataFrame, column: str, label: str = "all") -> pd.DataFrame:
    """`frame` plus a copy whose `column` is `label` (the pooled group)."""
    return pd.concat([frame, frame.assign(**{column: label})], ignore_index=True)


def cluster_summary(
    deltas: pd.DataFrame,
    by: Sequence[str],
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Mean delta and fraction above control per (`by`, control), cells resampled.

    Args:
        deltas: `leaf_deltas` rows joined with the `by` columns.
        by: Grouping columns (besides ``control``).
        n_boot: Replicates.
        seed: Bootstrap seed.
        confidence: Two-sided percentile level.

    Returns:
        `by` + `SUMMARY_COLUMNS`; intervals from resampling whole cells.
    """
    keys = [*by, "control"]
    live = deltas[np.isfinite(pd.to_numeric(deltas["delta"], errors="coerce"))]
    out: list[dict] = []
    for key, grp in live.groupby(keys, sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        per_cell = grp.groupby("cell_id").agg(
            s=("delta", "sum"), a=("above", "sum"), n=("delta", "size")
        )
        rng = stable_rng(seed, *(str(k) for k in key))
        k = len(per_cell)
        w = rng.multinomial(k, np.full(k, 1.0 / k), n_boot)
        den = w @ per_cell["n"].to_numpy(float)
        mean_draws = (w @ per_cell["s"].to_numpy(float)) / den
        frac_draws = (w @ per_cell["a"].to_numpy(float)) / den
        d = grp["delta"].to_numpy(float)
        lo, hi = percentile_ci(mean_draws, confidence)
        f_lo, f_hi = percentile_ci(frac_draws, confidence)
        out.append(
            {
                **dict(zip(keys, key, strict=True)),
                "n_cells": k,
                "n_leaves": int(d.size),
                "mean_delta": float(d.mean()),
                "mean_delta_lo": lo,
                "mean_delta_hi": hi,
                "frac_above": float(grp["above"].mean()),
                "frac_above_lo": f_lo,
                "frac_above_hi": f_hi,
                "median_delta": float(np.median(d)),
                "p75_delta": float(np.percentile(d, 75)),
                "p95_delta": float(np.percentile(d, 95)),
                "max_delta": float(d.max()),
            }
        )
    return pd.DataFrame(out, columns=[*by, *SUMMARY_COLUMNS])


def cells_long(
    cells: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    control_arms: Sequence[str] = CONTROL_ARMS,
) -> pd.DataFrame:
    """Per-cell delta under each control, one row per (cell, control)."""
    out: list[dict] = []
    for row in cells[cells["metric"] == metric].to_dict("records"):
        for control in control_arms:
            name = f"delta_{control}"
            out.append(
                {
                    "cell_id": row["cell_id"],
                    "control": control,
                    "delta": row[name],
                    "delta_lo": row[f"{name}_lo"],
                    "delta_hi": row[f"{name}_hi"],
                    "treatment_mean": row[f"mean_{ARM_TREATMENT}"],
                    "control_mean": row[f"mean_{control}"],
                    "n_treatment": row[f"n_{ARM_TREATMENT}"],
                    "n_control": row[f"n_{control}"],
                    "incomplete": row["incomplete"],
                }
            )
    return pd.DataFrame(out, columns=list(CELL_LONG_COLUMNS))


def removal_share(leaves: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Rollout-screen removal share per (`by`, arm), pooled over rows and per leaf.

    Args:
        leaves: Rows with ``arm``, ``status``, ``screen_n_rows``,
            ``screen_n_flagged`` and the `by` columns.
        by: Grouping columns (besides ``arm``).
    """
    live = leaves[
        (leaves["status"] == STATUS_EVALUATED)
        & np.isfinite(pd.to_numeric(leaves["screen_n_rows"], errors="coerce"))
    ]
    out: list[dict] = []
    for key, grp in live.groupby([*by, "arm"], sort=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        rows = grp["screen_n_rows"].astype(float)
        flagged = grp["screen_n_flagged"].astype(float)
        out.append(
            {
                **dict(zip([*by, "arm"], key, strict=True)),
                "n_leaves": int(len(grp)),
                "n_rows": int(rows.sum()),
                "n_flagged": int(flagged.sum()),
                "pooled_share": float(flagged.sum() / rows.sum())
                if rows.sum() > 0
                else math.nan,
                "leaf_mean_share": float((flagged / rows).mean()),
            }
        )
    return pd.DataFrame(out, columns=[*by, *REMOVAL_COLUMNS])


# ---- teachers ---------------------------------------------------------------------------


def admission_rule(family: str) -> str:
    """Treatment admission rule of a family's teachers."""
    return RULE_WS160M_W0 if family in _W0_FAMILIES else RULE_XSIZE_ADMIT


def _in_range(value: float, lo: float, hi: float, margin: float) -> object:
    if not all(math.isfinite(v) for v in (value, lo, hi)):
        return pd.NA
    return bool(lo - margin <= value <= hi + margin)


def _w0_admitted(teachers: pd.DataFrame) -> dict[tuple, tuple[bool, str]]:
    """``(family, kind, slot, base, param_tag) -> (admitted, tier)`` by the W0 rule."""
    rows = teachers[
        teachers["family"].isin(_W0_FAMILIES)
        & np.isfinite(pd.to_numeric(teachers["p_target"], errors="coerce"))
    ]
    if rows.empty:
        return {}
    frame = pd.DataFrame(
        {
            "animal": rows["slot_animal"],
            "base": rows["family"] + "|" + rows["kind"] + "|" + rows["base"],
            "step": 0,
            "param_tag": rows["param_tag"],
            "ftseed": rows["ftseed"],
            "set": ws160m_plan.PRIMARY_PROMPT_SET,
            "scoring": ws160m_plan.PRIMARY_SCORING,
            "p_target": rows["p_target"].astype(float),
            "base_prior": rows["base_p_target"].astype(float),
            "perplexity_ratio": rows["perplexity_ratio"].astype(float),
        }
    )
    summary = ws160m_metrics.build_teacher_summary(frame, None)
    out = {}
    for r in summary.itertuples(index=False):
        family, kind, base = str(r.base).split("|", 2)
        out[(family, kind, r.animal, base, r.param_tag)] = (
            r.tier == W0_ADMITTED_TIER,
            f"w0_tier={r.tier}",
        )
    return out


def teacher_labels(teachers: pd.DataFrame) -> pd.DataFrame:
    """Add admission and the V1 / V2 validity labels to every teacher-seed row.

    V1 (trait-free): ``|p - base_p| <= V1_MAX_ABS_P_SHIFT`` and not admitted by
    the family's treatment rule (`admission_rule`; W0 is scored over a
    config's ftseeds). V2 (dose-matched): the perplexity ratio within the same
    (family, slot, base) trait teachers' ``[min, max]`` widened by
    ``V2_PPL_RATIO_MARGIN``, and final train loss within theirs widened by
    ``V2_LOSS_MARGIN_NATS``. Labels only; no row is dropped.

    Args:
        teachers: One row per (family, kind, slot_animal, base, param_tag,
            ftseed) with ``p_target``, ``base_p_target``, ``perplexity_ratio``,
            ``final_train_loss``.

    Returns:
        `teachers` plus `TEACHER_LABEL_COLUMNS`.
    """
    out = teachers.copy()
    if out.empty:
        for col in TEACHER_LABEL_COLUMNS:
            out[col] = pd.Series(dtype=object)
        return out
    w0 = _w0_admitted(out)
    trait = out[out["kind"] == KIND_TRAIT]
    ranges: dict[tuple, dict[str, float]] = {}
    for key, grp in trait.groupby(["family", "slot_animal", "base"], sort=False):
        ppl = pd.to_numeric(grp["perplexity_ratio"], errors="coerce").dropna()
        loss = pd.to_numeric(grp["final_train_loss"], errors="coerce").dropna()
        ranges[key] = {
            "trait_ppl_ratio_min": float(ppl.min()) if len(ppl) else math.nan,
            "trait_ppl_ratio_max": float(ppl.max()) if len(ppl) else math.nan,
            "trait_loss_min": float(loss.min()) if len(loss) else math.nan,
            "trait_loss_max": float(loss.max()) if len(loss) else math.nan,
        }
    records: list[dict] = []
    for row in out.to_dict("records"):
        p = float(pd.to_numeric(row["p_target"], errors="coerce"))
        base_p = float(pd.to_numeric(row["base_p_target"], errors="coerce"))
        ratio = float(pd.to_numeric(row["perplexity_ratio"], errors="coerce"))
        loss = float(pd.to_numeric(row["final_train_loss"], errors="coerce"))
        rule = admission_rule(row["family"])
        if rule == RULE_WS160M_W0:
            key = (
                row["family"],
                row["kind"],
                row["slot_animal"],
                row["base"],
                row["param_tag"],
            )
            admitted, detail = w0.get(key, (pd.NA, "p_target_missing"))
        else:
            ok, reasons = admit_teacher(
                p_target=p if math.isfinite(p) else None,
                control_mean=base_p if math.isfinite(base_p) else None,
                control_sd=None,
                ppl_ratio=ratio if math.isfinite(ratio) else None,
            )
            missing = {"p_target_missing", "control_missing"} & set(reasons)
            admitted = pd.NA if missing else ok
            detail = ";".join(reasons)
        shift = p - base_p
        rng = ranges.get(
            (row["family"], row["slot_animal"], row["base"]),
            dict.fromkeys(TRAIT_RANGE_COLUMNS, math.nan),
        )
        v2_ppl = _in_range(
            ratio,
            rng["trait_ppl_ratio_min"],
            rng["trait_ppl_ratio_max"],
            plan.V2_PPL_RATIO_MARGIN,
        )
        v2_loss = _in_range(
            loss, rng["trait_loss_min"], rng["trait_loss_max"], plan.V2_LOSS_MARGIN_NATS
        )
        v1 = (
            pd.NA
            if (not math.isfinite(shift) or admitted is pd.NA)
            else bool(abs(shift) <= plan.V1_MAX_ABS_P_SHIFT and not admitted)
        )
        v2 = pd.NA if (v2_ppl is pd.NA or v2_loss is pd.NA) else v2_ppl and v2_loss
        records.append(
            {
                **row,
                "p_shift": shift,
                "admission_rule": rule,
                "admitted": admitted,
                "admission_detail": detail,
                **rng,
                "v1_trait_free": v1,
                "v2_ppl_in_range": v2_ppl,
                "v2_loss_in_range": v2_loss,
                "v2_dose_matched": v2,
            }
        )
    return pd.DataFrame(records, columns=[*teachers.columns, *TEACHER_LABEL_COLUMNS])


# ---- coverage -------------------------------------------------------------------------


def coverage(leaves: pd.DataFrame, teachers: pd.DataFrame | None = None) -> dict:
    """Expected vs present leaves per family and arm, and the incomplete cells.

    Args:
        leaves: `LEAF_ID_COLUMNS` rows, one per (cell, expected leaf).
        teachers: Optional teacher-seed rows with ``family``, ``kind``,
            ``status``.

    Returns:
        ``{"leaves": {family: {arm: counts}}, "incomplete_cells": [...],
        "teachers": {family: {kind: counts}}}``; counts are ``expected``,
        ``present``, ``trained``, ``evaluated`` over every leaf; incomplete
        cells are judged on `primary_scope` leaves.
    """

    def _counts(frame: pd.DataFrame) -> dict[str, int]:
        status = frame["status"]
        return {
            "expected": int(len(frame)),
            "present": int((status != STATUS_MISSING).sum()),
            "trained": int(status.isin([STATUS_NOT_EVALUATED, STATUS_EVALUATED]).sum()),
            "evaluated": int((status == STATUS_EVALUATED).sum()),
        }

    by_family: dict[str, dict[str, dict[str, int]]] = {}
    for (family, arm), grp in leaves.groupby(["family", "arm"], sort=True):
        by_family.setdefault(str(family), {})[str(arm)] = _counts(grp)
    incomplete: list[dict] = []
    primary = leaves[leaves["primary_scope"].astype(bool)]
    for (cell_id, family), grp in primary.groupby(["cell_id", "family"], sort=True):
        short = {}
        for arm, arm_rows in grp.groupby("arm", sort=True):
            counts = _counts(arm_rows)
            if counts["evaluated"] < counts["expected"]:
                short[str(arm)] = counts
        if short:
            incomplete.append({"cell_id": cell_id, "family": family, "arms": short})
    doc: dict = {
        "leaves": by_family,
        "n_cells": int(leaves["cell_id"].nunique()),
        "n_incomplete_cells": len(incomplete),
        "incomplete_cells": incomplete,
    }
    if teachers is not None and not teachers.empty:
        t_doc: dict[str, dict[str, dict[str, int]]] = {}
        for (family, kind), grp in teachers.groupby(["family", "kind"], sort=True):
            t_doc.setdefault(str(family), {})[str(kind)] = _counts(grp)
        doc["teachers"] = t_doc
    return doc


def metric_columns(frame: pd.DataFrame) -> list[str]:
    """Contrast metrics present in a leaf frame (primary first)."""
    preferred = [c for c in frame.columns if c.startswith("p_target_")]
    preferred.sort(key=lambda c: (c != PRIMARY_METRIC, c))
    rest = [c for c in ("fc_target_share", "perplexity_ratio") if c in frame.columns]
    bench = sorted(
        c
        for c in frame.columns
        if c.startswith("bench_") and not c.endswith("_sample_len")
    )
    return [*preferred, *rest, *bench]


def explode_panels(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (row, panel) from the ``;``-joined ``panels`` column."""
    out = frame.assign(panel=frame["panels"].fillna("").astype(str).str.split(";"))
    out = out.explode("panel")
    return out[out["panel"] != ""].reset_index(drop=True)


def ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """`frame` with any missing `columns` added as NaN."""
    missing = [c for c in columns if c not in frame.columns]
    return frame.assign(**dict.fromkeys(missing, math.nan)) if missing else frame


def arms_present(leaves: pd.DataFrame, arms: Sequence[str] = ARMS) -> Mapping[str, int]:
    """Evaluated leaves per arm."""
    live = leaves[leaves["status"] == STATUS_EVALUATED]
    return {a: int((live["arm"] == a).sum()) for a in arms}


# ---- Part III (`traitfree3`): v2 vs v1 placebo --------------------------------------

EQUIVALENCE_COLUMNS: tuple[str, ...] = (
    "scope",
    "group",
    "n_cells",
    "n_incomplete",
    "n_teacher_clusters",
    "mean_placebo",
    "mean_ref",
    "placebo_over_ref",
    "mean_delta_placebo",
    "mean_delta_ref",
    "mean_placebo_minus_ref",
    "mean_placebo_minus_ref_lo",
    "mean_placebo_minus_ref_hi",
    "clustered_lo",
    "clustered_hi",
    "n_cells_difference_excludes_0",
    "n_significance_flips",
    "equivalent",
    "equivalent_clustered",
)


def _excludes_zero(lo: pd.Series, hi: pd.Series) -> pd.Series:
    return (pd.to_numeric(lo) > 0) | (pd.to_numeric(hi) < 0)


def _within_margins(mean: float, lo: float, hi: float) -> bool:
    return bool(
        abs(mean) <= plan.EQUIVALENCE_MAX_ABS_MEAN
        and -plan.EQUIVALENCE_CI <= lo
        and hi <= plan.EQUIVALENCE_CI
    )


def equivalence_summary(
    cells: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """Per v2 family, per tier and over all v2 cells: mean v2 - v1 placebo and verdict.

    Only complete cells are scored. ``equivalent`` (the plan's rule, as the Part I
    checks): |mean| <= `plan.EQUIVALENCE_MAX_ABS_MEAN` and the whole-cell bootstrap
    interval inside ±`plan.EQUIVALENCE_CI`; ``equivalent_clustered`` applies the same
    margins to an interval that resamples teacher configs (family, animal, teacher
    base), then cells within them. Both are NA below `plan.EQUIVALENCE_MIN_CELLS`
    cells or with an incomplete cell. A significance flip is a cell whose
    ``delta_placebo`` and ``delta_ref`` intervals disagree on excluding 0. The
    `v1anchor` cell is left out (`anchor_compare`).

    Args:
        cells: `cell_contrasts` rows (Part III columns) with ``family``, ``tier``,
            ``animal`` and ``teacher``.
        metric: Metric scored.
        n_boot: Replicates.
        seed: Bootstrap seed.
        confidence: Two-sided percentile level.
    """
    live = cells[
        (cells["metric"] == metric) & (cells["family"] != plan.FAMILY_V1ANCHOR)
    ]
    groups = [
        *(
            ("family", f, live[live["family"] == f])
            for f in sorted(live["family"].unique())
        ),
        *(
            ("tier", t, live[live["tier"] == t])
            for t in plan.TIERS3
            if (live["tier"] == t).any()
        ),
        ("all", "v2", live),
    ]
    out: list[dict] = []
    for scope, group, grp in groups:
        incomplete = grp["incomplete"].astype(bool)
        done = grp[~incomplete]
        v = pd.to_numeric(done["placebo_minus_ref"], errors="coerce").to_numpy(float)
        ok = np.isfinite(v)
        v, done = v[ok], done[ok]
        n = int(v.size)
        mean = float(v.mean()) if n else math.nan
        lo = hi = c_lo = c_hi = math.nan
        clusters = (
            done["family"] + "|" + done["animal"] + "|" + done["teacher"]
        ).tolist()
        if n:
            rng = stable_rng(seed, "equivalence", scope, str(group))
            w = rng.multinomial(n, np.full(n, 1.0 / n), n_boot)
            lo, hi = percentile_ci(weighted_means(w, v), confidence)
            cw = two_level_weights(
                clusters,
                n_boot,
                stable_rng(seed, "equivalence-clustered", scope, str(group)),
            )
            c_lo, c_hi = percentile_ci(weighted_means(cw, v), confidence)
        judged = n >= plan.EQUIVALENCE_MIN_CELLS and not incomplete.any()
        flips = _excludes_zero(done["delta_placebo_lo"], done["delta_placebo_hi"]) != (
            _excludes_zero(done["delta_ref_lo"], done["delta_ref_hi"])
        )
        p_mean, r_mean = (
            _finite_mean(done["mean_placebo"]),
            _finite_mean(done["mean_ref"]),
        )
        out.append(
            {
                "scope": scope,
                "group": group,
                "n_cells": n,
                "n_incomplete": int(incomplete.sum()),
                "n_teacher_clusters": len(set(clusters)),
                "mean_placebo": p_mean,
                "mean_ref": r_mean,
                "placebo_over_ref": p_mean / r_mean if r_mean > 0 else math.nan,
                "mean_delta_placebo": _finite_mean(done["delta_placebo"]),
                "mean_delta_ref": _finite_mean(done["delta_ref"]),
                "mean_placebo_minus_ref": mean,
                "mean_placebo_minus_ref_lo": lo,
                "mean_placebo_minus_ref_hi": hi,
                "clustered_lo": c_lo,
                "clustered_hi": c_hi,
                "n_cells_difference_excludes_0": int(
                    _excludes_zero(
                        done["placebo_minus_ref_lo"], done["placebo_minus_ref_hi"]
                    ).sum()
                ),
                "n_significance_flips": int(flips.sum()),
                "equivalent": _within_margins(mean, lo, hi) if judged else None,
                "equivalent_clustered": _within_margins(mean, c_lo, c_hi)
                if judged
                else None,
            }
        )
    return pd.DataFrame(out, columns=list(EQUIVALENCE_COLUMNS))


def extended_null_compare(
    leaves: pd.DataFrame,
    metric: str = PRIMARY_METRIC,
    *,
    n_boot: int = plan.BOOTSTRAP_N,
    seed: int = plan.BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE,
) -> pd.DataFrame:
    """v2 vs v1 placebo spread on the extended-null cell(s): rotated and all leaves.

    Each arm resamples teacher ftseeds, then gseeds, on its own; ``teacher_sd`` is
    the SD of per-teacher means.
    """
    cols = [
        "cell_id",
        "placebo_set",
        "arm",
        "n_teachers",
        "n_leaves",
        "mean",
        "teacher_sd",
        "leaf_sd",
        "placebo_minus_ref",
        "placebo_minus_ref_lo",
        "placebo_minus_ref_hi",
    ]
    live = leaves[leaves["status"] == STATUS_EVALUATED]
    ext = live[live["extended_null"].astype(bool) & (live["arm"] == ARM_REF)]
    out: list[dict] = []
    for cell_id in sorted(ext["cell_id"].unique()):
        rows = _in_leaf_order(live[live["cell_id"] == cell_id])
        for name in ("rotated", "all"):
            draws, means = {}, {}
            for arm in (ARM_PLACEBO, ARM_REF):
                sel = rows[rows["arm"] == arm]
                if name == "rotated":
                    sel = sel[~sel["extended_null"].astype(bool)]
                values = pd.to_numeric(sel[metric], errors="coerce").to_numpy(float)
                w = two_level_weights(
                    sel["tfs"].tolist(), n_boot, stable_rng(seed, cell_id, name, arm)
                )
                draws[arm], means[arm] = weighted_means(w, values), _finite_mean(values)
                per_teacher = (
                    pd.to_numeric(sel[metric], errors="coerce")
                    .groupby(sel["tfs"])
                    .mean()
                )
                out.append(
                    {
                        "cell_id": cell_id,
                        "placebo_set": name,
                        "arm": arm,
                        "n_teachers": int(sel["tfs"].nunique()),
                        "n_leaves": int(np.isfinite(values).sum()),
                        "mean": means[arm],
                        "teacher_sd": float(per_teacher.std(ddof=1))
                        if len(per_teacher) > 1
                        else math.nan,
                        "leaf_sd": float(np.nanstd(values, ddof=1))
                        if np.isfinite(values).sum() > 1
                        else math.nan,
                    }
                )
            lo, hi = percentile_ci(draws[ARM_PLACEBO] - draws[ARM_REF], confidence)
            for row in out[-2:]:
                row.update(
                    {
                        "placebo_minus_ref": means[ARM_PLACEBO] - means[ARM_REF],
                        "placebo_minus_ref_lo": lo,
                        "placebo_minus_ref_hi": hi,
                    }
                )
    return pd.DataFrame(out, columns=cols)


def anchor_compare(leaves: pd.DataFrame, metric: str = PRIMARY_METRIC) -> pd.DataFrame:
    """`v1anchor` re-run vs original v1 leaf, per (teacher ftseed, gseed).

    ``mean_rerun_minus_original`` is over the cell's paired seeds; ``drift_ok`` holds
    when its magnitude is at most `plan.EQUIVALENCE_MAX_ABS_MEAN` (NA without a pair).
    """
    cols = [
        "cell_id",
        "tfs",
        "gseed",
        "rerun",
        "original",
        "rerun_minus_original",
        "mean_rerun_minus_original",
        "drift_ok",
    ]
    live = leaves[
        (leaves["family"] == plan.FAMILY_V1ANCHOR)
        & (leaves["status"] == STATUS_EVALUATED)
    ]
    out: list[dict] = []
    for cell_id, rows in live.groupby("cell_id", sort=True):
        value = {
            (arm, int(r.tfs), int(r.gseed)): float(
                pd.to_numeric(getattr(r, metric), errors="coerce")
            )
            for arm in (ARM_PLACEBO, ARM_REF)
            for r in rows[rows["arm"] == arm].itertuples(index=False)
        }
        for _, tfs, gseed in sorted(k for k in value if k[0] == ARM_PLACEBO):
            rerun = value[(ARM_PLACEBO, tfs, gseed)]
            original = value.get((ARM_REF, tfs, gseed), math.nan)
            out.append(
                {
                    "cell_id": cell_id,
                    "tfs": tfs,
                    "gseed": gseed,
                    "rerun": rerun,
                    "original": original,
                    "rerun_minus_original": rerun - original,
                }
            )
    means = {
        cell_id: _finite_mean(
            [r["rerun_minus_original"] for r in out if r["cell_id"] == cell_id]
        )
        for cell_id in {r["cell_id"] for r in out}
    }
    for r in out:
        diff = means[r["cell_id"]]
        r["mean_rerun_minus_original"] = diff
        r["drift_ok"] = (
            abs(diff) <= plan.EQUIVALENCE_MAX_ABS_MEAN if math.isfinite(diff) else None
        )
    return pd.DataFrame(out, columns=cols)
