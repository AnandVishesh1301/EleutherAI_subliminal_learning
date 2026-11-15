"""Transfer-significance family frames: existence / comparison tables, paired designs, roll-ups.

Split (2026-11) from the former single ``transfer_significance_io`` module (see
`significance_cells_io` and `significance_registry_io` for the layers it builds on).
"""
from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis.trajectories.dd1b_ladder import (
    dataset_dir,
)
from subliminal_learning.analysis.transfer.fsfact_transfer import RECEIVER_BASE
from subliminal_learning.coherence.probe_io import native_probe_pool_version, probe_path
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import memo
from subliminal_learning.stats import transfer_significance_tests as tt
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name


from subliminal_learning.analysis.loaders.significance_cells_io import (  # noqa: F401
    ALL_ON_DISK,
    CONTROL_LABEL,
    CellMatrix,
    DEAD_POLICY_NONE,
    DEAD_UNLABELLED,
    DEFAULT_DATASET_TAG,
    base_name,
    load_cell_matrix,
    load_prior,
    resolve_cells,
    resolve_teachers,
    stack_cell_matrices,
)
from subliminal_learning.analysis.loaders.significance_registry_io import (  # noqa: F401
    AMBER,
    RegistryArm,
    STATUS_FLAG,
    _anchored_cut,
    _arm_dead_matrix,
    _arm_trees,
    _dead_cell,
    _dead_counts,
    _fmt_p,
    _is_significant,
    _probe_pool_for,
    load_registry_arm,
    significance_table,
)

# ----------------------------------------------------------------------------------------------
# family comparison: is one seed family's SET of arms stronger than another's?
# ----------------------------------------------------------------------------------------------
ARM_COLUMNS = ["family", "arm", "cells", "students", "missing", "mean effect", "median effect",
               "exceed. frac"]

#: The four ways a student's effect is read: raw gap over its control, that gap in units of
#: the receiver's prior (PRIMARY), and the two pre-specified secondary retention views -- the
#: gap over the RECEIVER's own same-model (diagonal) arm mean on the same cell (``receptivity``)
#: and over the SENDER's (``portability``), absolute p units top and bottom. All are always
#: computed; every test runs once per mode.
VALUE_MODES = ("absolute", "relative", "receptivity", "portability")
VALUE_COLUMNS = {"absolute": "value_abs", "relative": "value_rel",
                 "receptivity": "value_recept", "portability": "value_port"}
#: The two diagonal-normalized modes and which end of the arm their yardstick belongs to.
RETENTION_MODES = {"receptivity": "receiver", "portability": "sender"}
#: A diagonal stratum enters only when its arm-mean gap D is at least this fraction of the base's
#: prior (delta = RETENTION_FLOOR x prior): below it the ratio is noise over noise.
RETENTION_FLOOR = 0.5
#: Columns of `diagonal_yardsticks`, one row per (animal, base, cell).
DIAGONAL_COLUMNS = ["animal", "base", "cell", "D", "D_se", "prior", "ctrl", "floor", "n teachers",
                    "complete", "covers design", "passes floor"]


@dataclass(frozen=True)
class FamilyArm:
    """One ordered off-diagonal arm of a family: the unit the comparison is built on.

    Attributes:
        family: Display label of the family the arm belongs to.
        sender: Sender (teacher) base token.
        receiver: Receiver (student) base token.
    """

    family: str
    sender: str
    receiver: str

    @property
    def arm(self) -> str:
        """``"{sender}->{receiver}"``."""
        return f"{self.sender}->{self.receiver}"


@dataclass(frozen=True)
class ModeComparison:
    """One value mode's arm table and its three tests, all under the arm-relabel null.

    Attributes:
        mode: ``"absolute"`` or ``"relative"`` (see `VALUE_MODES`).
        arms: One row per usable arm, columns `ARM_COLUMNS`.
        mean_contrast: `stats.arm_relabel_cell_permutation` with the per-cell mean contrast --
            the runs are read inside their student cell, the arm is the unit relabelled.
        win_probability: the same test with the per-cell win probability (stratified
            Mann-Whitney, ties 1/2). Rank-only, but NOT mode-invariant: the midranks are taken
            over all the runs of a cell, which spans arms with DIFFERENT receivers, and the
            relative mode divides each run by its own receiver's prior -- an unequal rescaling
            that can reorder them.
        arm_level_exceed: `stats.arm_set_permutation` on each arm's exceedance fraction against
            its own control. That summary is scale-free, so it too is mode-independent.
    """

    mode: str
    arms: pd.DataFrame
    mean_contrast: tt.ArmRelabelResult | None
    win_probability: tt.ArmRelabelResult | None
    arm_level_exceed: tt.ArmSetResult | None
    #: Why the mode has no tests (a retention mode with no usable diagonal); empty when tested.
    note: str = ""
    #: Retention modes: (base, cell) strata kept out because their diagonal gap D fell below the
    #: floor, and the bases with no diagonal complete on the design.
    n_below_floor: int = 0
    no_diagonal: tuple[str, ...] = ()


@dataclass(frozen=True)
class FamilyComparison:
    """Outcome of `compare_families`: both families' arm sets, per value mode, both levels.

    Attributes:
        modes: ``{mode: ModeComparison}`` over `VALUE_MODES`.
        frame: The tidy effect rows every test ran on (both value columns).
        skipped_arms: ``"{arm}: reason"`` per arm that could not be loaded or had no usable run.
        n_dead_treatment: Dead treatment student runs (dropped only under ``dead="drop"``);
            None when no anchored cut existed, i.e. the runs are UNLABELLED, not alive.
        n_dead_control: Dead control cells, same convention.
        dead_policy: ``"keep"`` (dead runs enter as measured), ``"drop"``, or
            `DEAD_POLICY_NONE` when the receiver had no anchored cut.
        notes: Anything else the caller must see (e.g. a receiver with no anchored dead cut).
        diagonals: The diagonal yardsticks (`diagonal_yardsticks`) of every base the arms
            touch, `DIAGONAL_COLUMNS`; empty when no base was loaded.
        unit: What the null relabelled -- ``"arm"`` (the default everywhere else, the
            conservative primary) or ``"run"`` (`registry_family_comparison` with
            ``unit="run"``: runs shuffled within their cell, both families pooled over arms).
    """

    modes: dict[str, ModeComparison]
    frame: pd.DataFrame
    skipped_arms: list[str] = field(default_factory=list)
    n_dead_treatment: int | None = 0
    n_dead_control: int | None = 0
    dead_policy: str = "keep"
    notes: list[str] = field(default_factory=list)
    diagonals: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=DIAGONAL_COLUMNS))
    unit: str = "arm"

    @property
    def n_dead(self) -> int | None:
        """Dead treatment runs plus dead control cells; None when the runs are unlabelled."""
        if self.n_dead_treatment is None or self.n_dead_control is None:
            return None
        return self.n_dead_treatment + self.n_dead_control


def family_arms(
    bases_a: Sequence[str],
    bases_b: Sequence[str],
    *,
    family_a: str = "A",
    family_b: str = "B",
) -> list[FamilyArm]:
    """Both families' ordered off-diagonal arms, family A's first.

    With four bases per family (``[orig, ds1, ds2, ds3]`` vs ``[orig, fs1, fs2, fs3]``) each
    family contributes 12 arms (every ordered pair of distinct bases), 24 in all. Arms are NOT
    paired across families by seed index: the index is an arbitrary label, so the comparison is
    between the two SETS of arms.

    Args:
        bases_a: Family-A base tokens; ``bases_a[0]`` is the families' common base.
        bases_b: Family-B base tokens, same length and same ordering convention.
        family_a, family_b: Display labels written into `FamilyArm.family`.

    Returns:
        Family A's arms followed by family B's.

    Raises:
        ValueError: If the families differ in size, hold fewer than two bases, or disagree on
            the common base.
    """
    a, b = list(bases_a), list(bases_b)
    if len(a) != len(b):
        raise ValueError(f"families must have equally many bases, got {len(a)} and {len(b)}")
    if len(a) < 2:
        raise ValueError("a family needs a common base plus at least one seed")
    if a[0] != b[0]:
        raise ValueError(f"families must share their first base, got {a[0]!r} and {b[0]!r}")

    def arms_of(bases: list[str], family: str) -> list[FamilyArm]:
        return [FamilyArm(family, x, y) for x in bases for y in bases if x != y]

    return arms_of(a, family_a) + arms_of(b, family_b)


def same_model_arms(bases: Sequence[str], *, family: str = "same-model") -> list[FamilyArm]:
    """The DIAGONAL arms ``b->b`` of `bases`: a base's own teacher distilled back into itself.

    The same-model family is not a set of ordered pairs of distinct bases, so `family_arms` does
    not describe it; it is one arm per base, and its matched control is that same base's untuned
    numbers pool.

    Args:
        bases: Base tokens, one arm each.
        family: Display label written into `FamilyArm.family`.

    Returns:
        One `FamilyArm` per base, in `bases` order.

    Raises:
        ValueError: If `bases` is empty or repeats a base.
    """
    b = list(bases)
    if not b:
        raise ValueError("same_model_arms needs at least one base")
    if len(set(b)) != len(b):
        raise ValueError(f"bases must be distinct, got {b}")
    return [FamilyArm(family, x, x) for x in b]


#: The base tokens of each family. The two cross families are read as every ordered off-diagonal
#: pair; ``same-model`` is read as the diagonal (`same_model_arms`).
FAMILY_BASES = {
    "cross-seed": ["orig", "fs1", "fs2", "fs3"],
    "cross-data-order": ["orig", "ds1", "ds2", "ds3"],
    "same-model": ["orig", "fs1", "fs2", "fs3", "ds1", "ds2", "ds3"],
}

#: The readout's older spellings, kept working as aliases of the display names the notebooks use.
FAMILY_ALIASES = {"full-seed": "cross-seed", "data-seed": "cross-data-order"}


def resolve_family(family: str) -> str:
    """The canonical `FAMILY_BASES` key of a family name or one of its `FAMILY_ALIASES`."""
    name = FAMILY_ALIASES.get(family, family)
    if name not in FAMILY_BASES:
        raise ValueError(f"unknown family {family!r}; known: {', '.join(family_names())}")
    return name


def family_names() -> list[str]:
    """Every accepted family spelling: the canonical names plus the aliases."""
    return [*FAMILY_BASES, *FAMILY_ALIASES]


def family_arm_set(family: str, *, family_label: str | None = None) -> list[FamilyArm]:
    """One family's arms: the off-diagonal pairs of a cross family, the diagonal of ``same-model``.

    Args:
        family: A `FAMILY_BASES` key or a `FAMILY_ALIASES` spelling.
        family_label: Display label written into `FamilyArm.family`; defaults to the canonical
            family name (the readout passes the spelling the user typed).

    Returns:
        The family's `FamilyArm` list.
    """
    name = resolve_family(family)
    label = family_label or name
    bases = FAMILY_BASES[name]
    if name == "same-model":
        return same_model_arms(bases, family=label)
    return [FamilyArm(label, x, y) for x in bases for y in bases if x != y]


class _ArmUnavailable(Exception):
    """An arm cannot enter the comparison; the message is what gets reported."""


def _arm_rows(
    animal: str,
    sender: str,
    receiver: str,
    teachers: Sequence[str],
    *,
    eval_animal: str,
    prompt_set: str,
    probe_pool: str,
    dead_ftseed: int,
    dead: str,
    output_root: Path | None,
    notes: list[str],
    **kw,
) -> tuple[pd.DataFrame, int | None, int | None, int]:
    """One arm's rows ``(cell, teacher, value_abs, value_rel, treatment, control, dead_*)``,
    its dead treatment/control counts (None when the receiver has no anchored cut, i.e. the
    runs are UNLABELLED rather than alive), and its count of missing treatment runs.

    A missing treatment run is a ``(teacher, matched cell)`` slot with no finite value: the run
    diverged (``train_diverged.json`` / ``nonfinite_adapter``, no eval written) or never ran.

    ``value_abs`` = treatment minus the cell's control; ``value_rel`` = the same gap divided by
    the receiver's prior. Runs labelled dead by the receiver's anchored probe cut are counted
    either way, and dropped (treatment OR control side) only when `dead` is ``"drop"``: a dead
    label is an outcome of the treatment, so dropping on it conditions on a post-treatment
    variable.

    Raises:
        _ArmUnavailable: With the reason distinguished, because "the arm was never run" and "the
        arm collapsed" are different facts: ``no control cells``, ``no receiver prior``,
        ``no treatment runs (0 finite ...)``, ``all N control cells dead``, or
        ``all N treatment runs dead``.
    """
    root = resolve_output_root(output_root)
    matrix = load_cell_matrix(animal, sender, receiver, teachers, eval_animal=eval_animal,
                              prompt_set=prompt_set, output_root=output_root, **kw)
    if not matrix.cells:
        raise _ArmUnavailable("no control cells")
    prior = load_prior(eval_animal, receiver, prompt_set, output_root=output_root)
    if not np.isfinite(prior) or prior == 0:
        raise _ArmUnavailable("no receiver prior")

    receiver_dir = base_name(receiver)
    control_tree, sender_tree, dataset_tag, tseed = _arm_trees(root, animal, sender, receiver,
                                                               **kw)
    # No anchored cut: under "drop" that is an error (nothing to drop on), under "keep" the
    # labels stay NaN and the counts come back None -- "unlabelled", never "0 dead".
    cut = _anchored_cut(root, receiver_dir, probe_pool, dead=dead, notes=notes)

    labels = _arm_dead_matrix(matrix, control_tree, sender_tree, dataset_tag=dataset_tag,
                              tseed=tseed, ftseed=dead_ftseed, probe_pool=probe_pool, cut=cut,
                              root=root)

    def is_dead(i: int, j: int) -> float:
        """1.0 dead / 0.0 alive / NaN unlabelled, so a missing cut never reads as alive."""
        if cut is None:
            return np.nan
        return float(labels[i, j] == 1.0)

    control = matrix.values[:, matrix.control_col]
    matched = [i for i in range(len(matrix.cells)) if np.isfinite(control[i])]
    treatment = np.delete(matrix.values, matrix.control_col, axis=1)
    n_treatment = int(np.isfinite(treatment[matched]).sum())
    rows, n_dead_control, n_dead_treatment = [], 0, 0
    for i in matched:
        cell = matrix.cells[i]
        dead_control = is_dead(i, matrix.control_col)
        n_dead_control += int(np.nan_to_num(dead_control))
        if dead_control == 1.0 and dead == "drop":
            continue
        for j, pool in enumerate(matrix.pools):
            if pool == CONTROL_LABEL or not np.isfinite(matrix.values[i, j]):
                continue
            dead_treatment = is_dead(i, j)
            n_dead_treatment += int(np.nan_to_num(dead_treatment))
            if dead_treatment == 1.0 and dead == "drop":
                continue
            gap = matrix.values[i, j] - control[i]
            rows.append({"cell": cell, "teacher": pool, "value_abs": gap,
                         "value_rel": gap / prior, "treatment": matrix.values[i, j],
                         "control": control[i], "dead_treatment": dead_treatment,
                         "dead_control": dead_control})
    if not rows:
        # Never run, control-side collapse, and treatment-side collapse read the same on a
        # table of arms and mean completely different things; name which one it is.
        if n_treatment == 0:
            raise _ArmUnavailable(f"no treatment runs (0 finite values over {len(matched)} "
                                  f"control cells)")
        if n_dead_control == len(matched):
            raise _ArmUnavailable(f"all {n_dead_control} control cells dead")
        raise _ArmUnavailable(f"all {n_dead_treatment} treatment runs dead")
    n_missing = int(treatment[matched].size - n_treatment)
    if cut is None:
        return pd.DataFrame(rows), None, None, n_missing
    return pd.DataFrame(rows), n_dead_treatment, n_dead_control, n_missing


def diagonal_yardsticks(
    animal: str,
    bases: Sequence[str],
    teachers: Sequence[str],
    *,
    cells: Sequence[str],
    eval_animal: str | None = None,
    prompt_set: str = "test",
    ftseed: int = 1,
    floor: float = RETENTION_FLOOR,
    output_root: Path | None = None,
    **kw,
) -> pd.DataFrame:
    """The same-model yardstick of every base at every cell: one row per (animal, base, cell).

    ``D`` is the base's diagonal ARM MEAN gap over the family's teacher tags -- mean over
    `teachers` of ``p(b->b, t, c) - ctrl(b->b, c)``, the base's own numbers pool distilled back
    into itself against its own same-base control (``students/controls/base/{b}/teacher-{b}``)
    -- never one tag's single run. ``D_se`` is the teacher SD over sqrt(n teachers). A cell is
    ``complete`` when the control and ALL teacher runs are present (the finished-set rule at
    cell level), it ``passes floor`` when complete and ``D >= floor x prior`` of that base (a
    cell-level fact, the same under every design), and ``covers design`` marks a base complete
    at EVERY cell of `cells` -- the retention values need both.

    Args:
        animal: Animal the teachers were finetuned on.
        bases: Base tokens whose diagonals are read.
        teachers: The family's canonical teacher tags (the arm mean runs over all of them).
        cells: The design cells.
        eval_animal: Animal scored; defaults to `animal`.
        prompt_set: Prompt set read out of the summaries.
        ftseed: Student finetune seed the runs and the control are read at.
        floor: `RETENTION_FLOOR`; the floor is ``floor x prior``.
        output_root: Output tree root; defaults to the paths YAML's.
        **kw: ``dataset_tag``, ``tseed``, ``control_source`` of `load_cell_matrix`.

    Returns:
        `DIAGONAL_COLUMNS`, in `bases` then `cells` order; ``D`` is NaN where incomplete.
    """
    eval_animal = eval_animal or animal
    design = list(cells)
    load_kw = {k: kw[k] for k in ("dataset_tag", "tseed", "control_source") if k in kw}
    rows = []
    for base in bases:
        matrix = load_cell_matrix(animal, base, base, teachers, eval_animal=eval_animal,
                                  prompt_set=prompt_set, ftseed=ftseed, cells=design,
                                  output_root=output_root, **load_kw)
        prior = load_prior(eval_animal, base, prompt_set, output_root=output_root)
        cut = floor * prior
        index = {c: i for i, c in enumerate(matrix.cells)}
        base_rows = []
        for cell in design:
            i = index.get(cell)
            row = {"animal": animal, "base": base, "cell": cell, "D": np.nan, "D_se": np.nan,
                   "prior": prior, "ctrl": np.nan, "floor": cut, "n teachers": len(teachers),
                   "complete": False, "covers design": False, "passes floor": False}
            if i is not None:
                ctrl = matrix.values[i, matrix.control_col]
                gaps = np.delete(matrix.values[i], matrix.control_col) - ctrl
                row["ctrl"] = ctrl
                if np.isfinite(gaps).all() and np.isfinite(ctrl):
                    row.update(D=float(gaps.mean()),
                               D_se=float(gaps.std(ddof=1) / np.sqrt(len(gaps)))
                               if len(gaps) > 1 else np.nan, complete=True)
                    row["passes floor"] = bool(np.isfinite(cut) and row["D"] >= cut)
            base_rows.append(row)
        covers = all(r["complete"] for r in base_rows)
        for r in base_rows:
            r["covers design"] = covers
        rows += base_rows
    return pd.DataFrame(rows, columns=DIAGONAL_COLUMNS)


def _add_retention_values(frame: pd.DataFrame, diagonals: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """`frame` with ``value_recept`` / ``value_port`` (and ``D_receiver`` / ``D_sender``) from
    the diagonal yardsticks, plus the per-mode report ``{mode: {"below_floor": n,
    "no_diagonal": [bases]}}``.

    A row's value is ``value_abs / D`` of the arm's receiver (resp. sender) at the row's cell
    when that base's diagonal covers the design AND the cell passes the floor; NaN otherwise.
    ``below_floor`` counts the distinct (base, cell) strata of a covering diagonal that failed
    the floor among the rows of the frame; ``no_diagonal`` lists the bases with no diagonal
    covering the design.
    """
    frame = frame.copy()
    covering = set(diagonals.loc[diagonals["covers design"].astype(bool), "base"])
    passing = {(b, c): d for b, c, d, ok in zip(diagonals["base"], diagonals["cell"],
                                                  diagonals["D"], diagonals["passes floor"])
               if ok and b in covering}
    report = {}
    for mode, end in RETENTION_MODES.items():
        col = VALUE_COLUMNS[mode]
        base_of = frame["arm"].str.split("->", n=1).str[0 if end == "sender" else 1]
        d = np.array([passing.get((b, c), np.nan) for b, c in zip(base_of, frame["cell"])])
        frame[f"D_{end}"] = d
        frame[col] = frame["value_abs"].to_numpy(dtype=float) / d
        below = {(b, c) for b, c in zip(base_of, frame["cell"])
                 if b in covering and (b, c) not in passing}
        report[mode] = {"below_floor": len(below),
                        "no_diagonal": sorted(set(base_of) - covering)}
    return frame, report


def normalized_effect_frame(
    arms: Sequence[FamilyArm],
    animal: str,
    teachers: Sequence[str],
    *,
    family_a: str = "A",
    family_b: str = "B",
    eval_animal: str | None = None,
    prompt_set: str = "test",
    probe_pool: str | None = None,
    ftseed: int = 1,
    dead: str = "keep",
    output_root: Path | None = None,
    **kw,
) -> tuple[pd.DataFrame, dict]:
    """Tidy control-normalized effects for every arm that covers the whole design.

    Full-coverage rule: an arm enters only when it has a finite control at EVERY design cell and
    at least one finite treatment run at every design cell; a partly covered arm goes to
    ``skipped_arms`` instead. Otherwise coverage masquerades as an effect -- an arm that reached
    only its easy cells would read as a strong one. Design cells are the ``cells`` kwarg (a
    cells JSON path or a list); without it the design is every cell at least one arm of EACH
    family reaches, and the two families are still always read on the same cells.

    Args:
        arms: Arms from `family_arms`.
        animal: Animal the teachers were finetuned on.
        teachers: Teacher param_tags, the same list for every sender.
        family_a, family_b: The family labels expected in ``arms``.
        eval_animal: Animal scored; defaults to `animal`.
        prompt_set: Prompt set read out of the summaries.
        probe_pool: Probe pool of the anchored dead cut; None (default) = each receiver's
            native pool (`coherence.probe_io.native_probe_pool`, which routes by the
            receiver base's pretraining corpus).
        ftseed: Student finetune seed the treatment, the control and the dead label are all read
            at, so the comparison is one pinned design and not a seed-aggregated mixture.
        dead: ``"keep"`` (default) keeps dead runs as measured and only counts them; ``"drop"``
            excludes every dead treatment run and every dead control cell. A receiver with no
            anchored cut raises under ``"drop"`` (nothing to drop on) and stays UNLABELLED under
            ``"keep"``: a WARNING, a note, ``dead_policy`` "none (anchor missing)", and None
            counts, because "0 dead" and "never labelled" read the same.
        output_root: Output tree root; defaults to the paths YAML's.
        **kw: Forwarded to `load_cell_matrix` (``tseed``, ``dataset_tag``, ``control_source``, ...).

    Returns:
        ``(frame, report)``. `frame` columns: ``family, arm, teacher, cell, stratum,
        value_abs, value_rel, value_recept, value_port, D_receiver, D_sender, treatment,
        control, dead_treatment, dead_control``, with ``stratum = cell``; the retention values
        (`_add_retention_values`) are NaN where the base's diagonal does not cover the design
        or the cell fails the floor. `report` holds ``skipped_arms``, ``n_dead_treatment``,
        ``n_dead_control``, ``n_dead`` (all three None when an anchor was missing, so the
        runs are unlabelled), ``dead_policy`` (`DEAD_POLICY_NONE` then), ``missing`` (per kept
        arm, its count of missing treatment runs), ``design_cells``, ``notes``, ``retention``
        (``{mode: {"below_floor", "no_diagonal"}}``) and ``diagonals`` (`diagonal_yardsticks`
        of every base the kept arms touch, on the usable cells).
    """
    if dead not in ("keep", "drop"):
        raise ValueError(f"dead must be 'keep' or 'drop', got {dead!r}")
    eval_animal = eval_animal or animal
    design = resolve_cells(kw.get("cells"))
    notes: list[str] = []
    skipped_arms: list[str] = []
    missing: dict[str, int] = {}
    n_dead_treatment = n_dead_control = 0
    loaded: dict[str, tuple[FamilyArm, pd.DataFrame]] = {}
    for arm in arms:
        try:
            frame, dead_t, dead_c, n_missing = _arm_rows(
                animal, arm.sender, arm.receiver, teachers, eval_animal=eval_animal,
                prompt_set=prompt_set,
                probe_pool=_probe_pool_for(base_name(arm.receiver), probe_pool),
                dead_ftseed=ftseed, dead=dead,
                output_root=output_root, notes=notes, ftseed=ftseed, **kw)
        except _ArmUnavailable as exc:
            skipped_arms.append(f"{arm.arm}: {exc}")
            continue
        if design is not None:
            covered = len(set(frame["cell"]) & set(design))
            if covered < len(design):
                skipped_arms.append(f"{arm.arm}: partial coverage "
                                    f"({covered}/{len(design)} design cells)")
                continue
        loaded[f"{arm.family}|{arm.arm}"] = (arm, frame)
        missing[arm.arm] = n_missing
        # One unlabelled arm makes the family's counts unlabelled: summing over the rest would
        # report a total that silently leaves that arm's runs out.
        if dead_t is None or n_dead_treatment is None:
            n_dead_treatment = n_dead_control = None
        else:
            n_dead_treatment += dead_t
            n_dead_control += dead_c

    members = list(loaded.values())
    cells_by_family = {
        family: set().union(*[set(f["cell"]) for a, f in members if a.family == family])
        if any(a.family == family for a, _ in members) else set()
        for family in (family_a, family_b)
    }
    usable = sorted(cells_by_family[family_a] & cells_by_family[family_b])
    blocks = []
    for arm, frame in members:
        block = frame[frame["cell"].isin(usable)].copy()
        if block.empty:
            continue
        block.insert(0, "family", arm.family)
        block.insert(1, "arm", arm.arm)
        blocks.append(block)
    columns = ["family", "arm", "teacher", "cell", "stratum", "value_abs", "value_rel",
               "value_recept", "value_port", "D_receiver", "D_sender",
               "treatment", "control", "dead_treatment", "dead_control"]
    retention = {m: {"below_floor": 0, "no_diagonal": []} for m in RETENTION_MODES}
    diagonals = pd.DataFrame(columns=DIAGONAL_COLUMNS)
    if not blocks:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = pd.concat(blocks, ignore_index=True)
        frame["stratum"] = frame["cell"]
        # The retention yardsticks: every base the kept arms touch, read on the usable cells
        # with the same teachers, seed and dataset the arms were read with.
        bases = sorted({b for a, _ in members for b in (a.sender, a.receiver)})
        diagonals = diagonal_yardsticks(animal, bases, teachers, cells=usable,
                                        eval_animal=eval_animal, prompt_set=prompt_set,
                                        ftseed=ftseed, output_root=output_root,
                                        **{k: v for k, v in kw.items() if k != "cells"})
        frame, retention = _add_retention_values(frame, diagonals)
        frame = frame.reindex(columns=columns)
    anchor_missing = n_dead_treatment is None
    return frame, {"skipped_arms": skipped_arms, "missing": missing,
                   "n_dead_treatment": None if anchor_missing else int(n_dead_treatment),
                   "n_dead_control": None if anchor_missing else int(n_dead_control),
                   "n_dead": None if anchor_missing else int(n_dead_treatment + n_dead_control),
                   "dead_policy": DEAD_POLICY_NONE if anchor_missing else dead,
                   "design_cells": design, "notes": notes, "retention": retention,
                   "diagonals": diagonals}


def _arm_summary(block: pd.DataFrame, value_col: str, missing: int) -> dict:
    """One arm's mean and median effect in `value_col`, exceedance fraction against its own
    control, and student/cell counts, on the usable cells (the median beside the mean makes the
    ratio noise of the retention modes visible).

    The exceedance fraction is the share of the arm's runs strictly above their own cell's
    control, so it is invariant to any positive rescaling of the values and reads the same in
    both value modes.
    """
    treatment = block["treatment"].to_numpy(dtype=float)
    control = block["control"].to_numpy(dtype=float)
    ok = np.isfinite(treatment) & np.isfinite(control)
    return {"cells": int(block["cell"].nunique()), "students": int(len(block)),
            "missing": int(missing), "mean effect": float(block[value_col].mean()),
            "median effect": float(block[value_col].median()),
            "exceed. frac": float((treatment[ok] > control[ok]).mean()) if ok.any()
                            else float("nan")}


def _mode_comparison(frame: pd.DataFrame, mode: str, *, family_a: str, family_b: str,
                     missing: Mapping[str, int], n_perm: int, seed: int,
                     retention: Mapping[str, dict] | None = None) -> ModeComparison:
    """One value mode's arm table and its three permutation results.

    The tests run on the rows whose value is finite: every row in the absolute and relative
    modes, only the strata with a covering diagonal above the floor in the retention modes. A
    retention mode left without a usable stratum on one side (no diagonal with controls on the
    design) comes back with no tests and a ``note`` instead of raising.
    """
    value_col = VALUE_COLUMNS[mode]
    extra = dict(retention or {}).get(mode, {})
    tail = {"n_below_floor": int(extra.get("below_floor", 0)),
            "no_diagonal": tuple(extra.get("no_diagonal", ()))}
    usable = frame[np.isfinite(pd.to_numeric(frame[value_col], errors="coerce"))]
    arm_rows = [{"family": family, "arm": arm,
                 **_arm_summary(block, value_col, missing.get(arm, 0))}
                for (family, arm), block in usable.groupby(["family", "arm"], sort=False)]
    arm_table = pd.DataFrame(arm_rows).reindex(columns=ARM_COLUMNS)
    arm_table = arm_table.sort_values(by=["family", "arm"]).reset_index(drop=True)

    def relabel(statistic: str) -> tt.ArmRelabelResult:
        return tt.arm_relabel_cell_permutation(
            usable.rename(columns={value_col: "value"}), family_a=family_a, family_b=family_b,
            statistic=statistic, n_perm=n_perm, seed=seed,
        )

    try:
        return ModeComparison(mode=mode, arms=arm_table,
                              mean_contrast=relabel("mean_contrast"),
                              win_probability=relabel("win_probability"),
                              arm_level_exceed=tt.arm_set_permutation(
                                  arm_table.rename(columns={"exceed. frac": "value"}),
                                  family_a=family_a, family_b=family_b, n_perm=n_perm,
                                  seed=seed),
                              **tail)
    except ValueError:
        if mode not in RETENTION_MODES:
            raise
        note = ("no diagonals with controls" if tail["no_diagonal"]
                else "no stratum above the floor on both sides")
        return ModeComparison(mode=mode, arms=arm_table, mean_contrast=None,
                              win_probability=None, arm_level_exceed=None, note=note, **tail)


@dataclass(frozen=True)
class MultiAnimalComparison:
    """`compare_families` over several animals: each animal's comparison plus the stratified one.

    The unit relabelled is the ARM (the model pair), so the same arm label seen in two animals is
    ONE unit; the strata are ``(animal, cell)`` pairs.

    Attributes:
        family_a: Display label of the family claimed stronger.
        family_b: Display label of the reference family.
        animals: The animals that contributed at least one usable block, in call order.
        per_animal: ``{animal: FamilyComparison}`` on that animal's blocks alone.
        stratified: The comparison over the union of the ``(animal, cell)`` strata.
        blocks_present: ``{arm: [animals whose block entered]}``.
        missing_blocks: ``"{animal} {arm}: reason"`` per (animal, arm) block kept out by coverage.
    """

    family_a: str
    family_b: str
    animals: list[str]
    per_animal: dict[str, FamilyComparison]
    stratified: FamilyComparison
    blocks_present: dict[str, list[str]] = field(default_factory=dict)
    missing_blocks: list[str] = field(default_factory=list)

    @property
    def diagonals(self) -> pd.DataFrame:
        """Every animal's diagonal yardsticks (`FamilyComparison.diagonals`), stacked."""
        return self.stratified.diagonals


def _maybe_int(n: int | None) -> int | None:
    """`n` as an int, or None kept as None (an unlabelled count is not a zero one)."""
    return None if n is None else int(n)


def _family_comparison(frame: pd.DataFrame, *, family_a: str, family_b: str,
                       missing: Mapping[str, int], skipped: Sequence[str],
                       n_dead_treatment: int | None, n_dead_control: int | None, dead: str,
                       notes: Sequence[str], n_perm: int, seed: int,
                       retention: Mapping[str, dict] | None = None,
                       diagonals: pd.DataFrame | None = None) -> FamilyComparison:
    """Every value mode's `ModeComparison` on one effect frame, wrapped with its report."""
    modes = {mode: _mode_comparison(frame, mode, family_a=family_a, family_b=family_b,
                                    missing=missing, n_perm=n_perm, seed=seed,
                                    retention=retention)
             for mode in VALUE_MODES}
    return FamilyComparison(modes=modes, frame=frame, skipped_arms=list(skipped),
                            n_dead_treatment=_maybe_int(n_dead_treatment),
                            n_dead_control=_maybe_int(n_dead_control), dead_policy=dead,
                            notes=list(notes),
                            diagonals=(pd.DataFrame(columns=DIAGONAL_COLUMNS)
                                       if diagonals is None else diagonals))


def _explicit_arms(spec) -> bool:
    """Whether a family spec is an explicit non-empty `FamilyArm` list."""
    return (not isinstance(spec, str) and len(spec) > 0
            and all(isinstance(a, FamilyArm) for a in spec))


def compare_families(
    spec_a: str | Sequence[str] | Sequence[FamilyArm],
    spec_b: str | Sequence[str] | Sequence[FamilyArm],
    animals: str | Sequence[str],
    teachers: Sequence[str] | Mapping[str, Sequence[str]],
    *,
    family_a: str | None = None,
    family_b: str | None = None,
    cells: Sequence[str] | str | Path,
    ftseed: int = 1,
    dead: str = "keep",
    n_perm: int = 20000,
    seed: int = 0,
    exclude_blocks: Collection[tuple[str, str]] = (),
    **kw,
) -> FamilyComparison | MultiAnimalComparison:
    """Does family A's SET of arms transfer more strongly than family B's?

    The design is pinned: the `cells` spec fixes the student cells, `ftseed` fixes the one
    student finetune seed every treatment, control and dead label is read at, and only blocks
    that cover the whole design enter (`normalized_effect_frame`). Every student is read against
    its own matched control in BOTH value modes (absolute gap, and gap over the receiver's
    prior), and each mode gets three tests: `stats.arm_relabel_cell_permutation` with the
    per-cell mean contrast and with the per-cell win probability -- the arm is the unit
    relabelled, the statistic is read on the runs inside their cell -- plus
    `stats.arm_set_permutation` on the arms' exceedance fractions, a scale-free arm summary.

    Over several animals the stratum is ``(animal, cell)`` and the unit is still the ARM: the
    same model pair seen in two animals is ONE unit of the relabelling, not two, because its two
    blocks read the same pair of base models. Coverage is judged per ``(animal, arm)`` block, and
    an arm enters as soon as ONE of its blocks covers the design; the blocks that did not are
    reported rather than silently filled in.

    Args:
        spec_a: Family A as a `FAMILY_BASES` name (or alias) -- the only spelling that can name
            ``same-model`` -- or, as before, its list of base tokens (common base first), or an
            explicit list of `FamilyArm` (any arm set, e.g. a relationship category; every arm
            must carry the same ``family`` label, which becomes the display label).
        spec_b: Family B, same three spellings as `spec_a`.
        animals: One animal, or several for the animal-stratified comparison.
        teachers: Teacher param_tags used for every animal, or ``{animal: param_tags}``.
        family_a, family_b: Display labels; default to the family names when those were given.
        cells: The design cells: a cells JSON path (a ``cells`` key) or a list of cell names.
        ftseed: The student finetune seed the design is read at.
        dead: ``"keep"`` (default) or ``"drop"``; see `normalized_effect_frame`. A receiver
            with no anchored cut raises under ``"drop"`` and leaves the counts None (and
            `FamilyComparison.dead_policy` `DEAD_POLICY_NONE`) under ``"keep"``.
        n_perm: Monte-Carlo draws (arm level beyond the enumeration cap; always at cell level).
        seed: Seed of the permutation draws.
        exclude_blocks: ``(animal, arm)`` blocks to keep out of the comparison altogether -- a
            caller that judges coverage more strictly than the full-design rule here (e.g. the
            conditions table, which needs EVERY listed teacher on every cell) names its
            unfinished blocks and they are reported in `MultiAnimalComparison.missing_blocks`
            instead of being tested. Empty by default, so the older call paths are unchanged.
        **kw: Forwarded to `normalized_effect_frame` (``eval_animal``, ``prompt_set``, ``tseed``,
            ``control_source``, ``output_root``, ...).

    Returns:
        A `FamilyComparison` when `animals` is a single animal (the old call path), else a
        `MultiAnimalComparison`.

    Raises:
        ValueError: If the two families are spelled differently from each other, or if no cell is
            reached by both families (nothing to compare).
    """
    if _explicit_arms(spec_a) or _explicit_arms(spec_b):
        if not (_explicit_arms(spec_a) and _explicit_arms(spec_b)):
            raise ValueError("both families must be given the same way: two FamilyArm lists")
        labels = [{a.family for a in spec} for spec in (spec_a, spec_b)]
        if any(len(ls) != 1 for ls in labels) or labels[0] == labels[1]:
            raise ValueError(f"each FamilyArm list needs ONE family label, distinct between "
                             f"the two: {labels}")
        label_a, label_b = family_a or labels[0].pop(), family_b or labels[1].pop()
        arms = ([FamilyArm(label_a, a.sender, a.receiver) for a in spec_a]
                + [FamilyArm(label_b, a.sender, a.receiver) for a in spec_b])
    else:
        if isinstance(spec_a, str) != isinstance(spec_b, str):
            raise ValueError("both families must be given the same way: two names or two base "
                             "lists")
        label_a = family_a or (resolve_family(spec_a) if isinstance(spec_a, str) else "A")
        label_b = family_b or (resolve_family(spec_b) if isinstance(spec_b, str) else "B")
        if isinstance(spec_a, str):
            arms = (family_arm_set(spec_a, family_label=label_a)
                    + family_arm_set(spec_b, family_label=label_b))
        else:
            arms = family_arms(spec_a, spec_b, family_a=label_a, family_b=label_b)

    single = isinstance(animals, str)
    animal_list = [animals] if single else list(animals)
    by_animal = (dict(teachers) if isinstance(teachers, Mapping)
                 else {a: list(teachers) for a in animal_list})

    per_animal: dict[str, FamilyComparison] = {}
    frames, missing_blocks, notes, dead_policies, diagonals = [], [], [], [], []
    missing: dict[str, int] = {}
    retention = {m: {"below_floor": 0, "no_diagonal": set()} for m in RETENTION_MODES}
    n_dead_treatment = n_dead_control = 0
    excluded = {(animal, arm) for animal, arm in exclude_blocks}
    for animal in animal_list:
        kept = [a for a in arms if (animal, a.arm) not in excluded]
        frame, report = normalized_effect_frame(kept, animal, by_animal[animal],
                                                family_a=label_a, family_b=label_b, ftseed=ftseed,
                                                dead=dead, cells=cells, **kw)
        report["skipped_arms"] = [f"{a.arm}: excluded by the caller" for a in arms
                                  if (animal, a.arm) in excluded] + report["skipped_arms"]
        dead_policies.append(report["dead_policy"])
        missing_blocks += [f"{animal} {s}" for s in report["skipped_arms"]]
        notes += [n for n in report["notes"] if n not in notes]
        if frame.empty:
            missing_blocks.append(f"{animal}: no cell reached by both families")
            continue
        frame = frame.copy()
        frame.insert(2, "animal", animal)
        frame["stratum"] = animal + "|" + frame["cell"]
        frames.append(frame)
        diagonals.append(report["diagonals"])
        for arm, n in report["missing"].items():
            missing[arm] = missing.get(arm, 0) + n
        # (animal, cell) strata: the below-floor counts add up, the missing diagonals union
        for m, r in report["retention"].items():
            retention[m]["below_floor"] += r["below_floor"]
            retention[m]["no_diagonal"] |= set(r["no_diagonal"])
        if report["n_dead_treatment"] is None or n_dead_treatment is None:
            n_dead_treatment = n_dead_control = None
        else:
            n_dead_treatment += report["n_dead_treatment"]
            n_dead_control += report["n_dead_control"]
        per_animal[animal] = _family_comparison(
            frame, family_a=label_a, family_b=label_b, missing=report["missing"],
            skipped=report["skipped_arms"], n_dead_treatment=report["n_dead_treatment"],
            n_dead_control=report["n_dead_control"], dead=report["dead_policy"],
            notes=report["notes"], n_perm=n_perm, seed=seed, retention=report["retention"],
            diagonals=report["diagonals"])
    if not frames:
        raise ValueError("no cell is reached by both families; nothing to compare")
    if single:
        return per_animal[animal_list[0]]

    stratified = _family_comparison(
        pd.concat(frames, ignore_index=True), family_a=label_a, family_b=label_b, missing=missing,
        skipped=missing_blocks, n_dead_treatment=n_dead_treatment, n_dead_control=n_dead_control,
        dead=(DEAD_POLICY_NONE if DEAD_POLICY_NONE in dead_policies else dead), notes=notes,
        n_perm=n_perm, seed=seed,
        retention={m: {"below_floor": r["below_floor"], "no_diagonal": sorted(r["no_diagonal"])}
                   for m, r in retention.items()},
        diagonals=pd.concat(diagonals, ignore_index=True))
    blocks = (stratified.frame.drop_duplicates(["arm", "animal"])
              .groupby("arm")["animal"].apply(list).to_dict())
    return MultiAnimalComparison(family_a=label_a, family_b=label_b,
                                 animals=list(per_animal), per_animal=per_animal,
                                 stratified=stratified, blocks_present=blocks,
                                 missing_blocks=missing_blocks)


# ----------------------------------------------------------------------------------------------
# the two notebook tables: family existence, and family comparison
# ----------------------------------------------------------------------------------------------
#: ``cells`` = the usable cells the arm was actually tested on; ``design cells`` = the size of
#: the requested design, so an arm short of it is visible rather than reading as a full arm.
EXISTENCE_COLUMNS = ["family", "animal", "arm", "cells", "design cells", "treatment runs",
                     "dead (treat/ctrl)", "exceed.", "perm p", "note"]

#: The pooled row's arm label: that animal's whole family under one arm-stratified permutation.
POOLED_ARM = "all arms"


def _pooled_dead_cell(matrices: Mapping[str, CellMatrix],
                      deads: Mapping[str, tuple[np.ndarray, bool]]) -> str:
    """The pooled row's ``dead (treat/ctrl)``: the arms' counts summed, or `DEAD_UNLABELLED`
    as soon as ONE arm is unlabelled (summing the rest would leave its runs out silently)."""
    if not all(labelled for _, labelled in deads.values()):
        return DEAD_UNLABELLED
    counts = [_dead_counts(matrices[arm].values, d) for arm, (d, _) in deads.items()]
    return f"{sum(t for t, _ in counts)}/{sum(c for _, c in counts)}"


def family_existence_table(
    family: str,
    animals: Sequence[str],
    teachers_by_animal: Mapping[str, Sequence[str]],
    *,
    cells: Sequence[str] | str | Path,
    ftseed: int = 1,
    prompt_set: str = "test",
    probe_pool: str | None = None,
    family_label: str | None = None,
    n_perm: int = 20000,
    seed: int = 0,
    **kw,
) -> pd.DataFrame:
    """Does a family's transfer exist at all: one row per (animal, arm), one pooled row per animal.

    Each arm is tested against its OWN matched control with the cell-stratified exceedance
    permutation (`stats.stratified_exceedance_permutation`, cells as strata); the pooled row per
    animal runs the same test with the arms as an extra stratum, each arm judged against its own
    control maximum -- the ``--family`` mode of `scripts.analysis.transfer_significance_readout`. Animals
    are never pooled into one stratum: a row is always one animal's.

    Args:
        family: A `FAMILY_BASES` name or alias; ``same-model`` reads the diagonal arms.
        animals: Animals the teachers were finetuned on, one block of rows each.
        teachers_by_animal: ``{animal: param_tags}``.
        cells: The design cells (a cells JSON path or a list).
        ftseed: The student finetune seed treatment, control and the dead label are read at.
        prompt_set: Prompt set read out of the summaries.
        probe_pool: Probe pool of the anchored dead cut; None (default) = each receiver's
            native pool (`coherence.probe_io.native_probe_pool`, which routes by the
            receiver base's pretraining corpus). Dead runs
            are KEPT as measured and only counted; a receiver with no anchored cut leaves the
            count `DEAD_UNLABELLED`.
        family_label: Display label; defaults to the canonical family name.
        n_perm, seed: Stratified permutation settings.
        **kw: Forwarded to `load_cell_matrix` (``tseed``, ``control_source``, ``output_root``, ...).

    Returns:
        `EXISTENCE_COLUMNS` plus a boolean ``pooled``. An arm with no control cell or no
        treatment run keeps its row with the reason in ``note`` and no numbers. Dead runs are
        never dropped (the DEAD KEPT convention); every row reports how many it holds.
    """
    label = family_label or resolve_family(family)
    arms = family_arm_set(family, family_label=label)
    design = resolve_cells(cells)
    root = resolve_output_root(kw.get("output_root"))
    notes: list[str] = []
    rows = []
    for animal in animals:
        teachers = list(teachers_by_animal[animal])
        matrices, deads, skipped = {}, {}, {}
        for a in arms:
            m = load_cell_matrix(animal, a.sender, a.receiver, teachers, cells=design,
                                 ftseed=ftseed, prompt_set=prompt_set, **kw)
            treatment = np.delete(m.values, m.control_col, axis=1)
            if not m.cells:
                skipped[a.arm] = "no control cells"
            elif not np.isfinite(treatment).any():
                skipped[a.arm] = "no treatment runs"
            else:
                matrices[a.arm] = m
                # The single-arm table's labelling: this receiver's anchored cut, this ftseed.
                pool = _probe_pool_for(base_name(a.receiver), probe_pool)
                cut = _anchored_cut(root, base_name(a.receiver), pool, dead="keep",
                                    notes=notes)
                control_tree, sender_tree, dataset_tag, tseed = _arm_trees(
                    root, animal, a.sender, a.receiver, **kw)
                deads[a.arm] = (_arm_dead_matrix(m, control_tree, sender_tree,
                                                 dataset_tag=dataset_tag, tseed=tseed,
                                                 ftseed=ftseed, probe_pool=pool, cut=cut,
                                                 root=root),
                                cut is not None)
        stacked = stack_cell_matrices(matrices) if matrices else None
        for a in arms:
            row = {"family": label, "animal": animal, "arm": a.arm, "pooled": False}
            if a.arm in skipped:
                rows.append({**row, "note": skipped[a.arm]})
                continue
            m = matrices[a.arm]
            d, labelled = deads[a.arm]
            r = tt.stratified_exceedance_permutation(m.values, m.control_col, n_perm=n_perm,
                                                     seed=seed)
            rows.append({**row, "cells": len(m.cells), "design cells": len(design),
                         "treatment runs": stacked.n_treatment[a.arm],
                         "dead (treat/ctrl)": _dead_cell(m.values, d, labelled=labelled),
                         "exceed.": r.observed, "perm p": r.p})
        if stacked is None:
            rows.append({"family": label, "animal": animal, "arm": POOLED_ARM, "pooled": True,
                         "note": f"no usable arm ({len(skipped)} skipped)"})
            continue
        r = tt.stratified_exceedance_permutation(stacked.values, stacked.control_index,
                                                 groups=stacked.groups, n_perm=n_perm, seed=seed)
        note = (f"{len(skipped)} arms skipped: "
                + "; ".join(f"{k} ({v})" for k, v in skipped.items())) if skipped else ""
        rows.append({"family": label, "animal": animal, "arm": POOLED_ARM, "pooled": True,
                     "cells": sum(len(m.cells) for m in matrices.values()),
                     # the design, once per usable arm: the pooled row's own design size
                     "design cells": len(design) * len(matrices),
                     "treatment runs": sum(stacked.n_treatment.values()),
                     "dead (treat/ctrl)": _pooled_dead_cell(matrices, deads),
                     "exceed.": r.observed, "perm p": r.p, "note": note})
    return pd.DataFrame(rows).reindex(columns=[*EXISTENCE_COLUMNS, "pooled"])


COMPARISON_COLUMNS = ["comparison", "design", "animal", "value mode", "unit", "arms A vs B",
                      "mean contrast", "mean p", "win prob.", "win p", "exceed. frac contrast",
                      "exceed. p", "floor", "labelings", "missing blocks"]


def _retention_note(m: ModeComparison, design: str) -> str:
    """The retention-mode additions to ``missing blocks``: the note of an untested mode (with
    the design named), else the strata below the floor and the bases with no diagonal."""
    end = RETENTION_MODES[m.mode][0]
    if m.note:
        bases = f" ({', '.join(m.no_diagonal)})" if m.no_diagonal else ""
        return f"{m.mode}: {m.note} on {design}{bases}"
    parts = [f"{m.mode}: {m.n_below_floor} ({end}, cell) strata below floor"]
    if m.no_diagonal:
        parts.append(f"no diagonal for {', '.join(m.no_diagonal)} on {design}")
    return "; ".join(parts)


def _comparison_rows(comparison: str, design: str, animal: str, res: FamilyComparison) -> list[dict]:
    """One (comparison, design, animal) block: one row per value mode. A retention mode with no
    usable diagonal keeps its row with the reason in ``missing blocks`` and no numbers.

    The three p columns are the tests' TWO-SIDED p's (`stats` ``p_two_sided``): a family
    comparison fixes no direction in advance. Each test's one-sided A > B tail is still on the
    result (``p_upper``) for a caller that wants it beside the two-sided p."""
    rows = []
    blocks = (f"{len(res.skipped_arms)}: " + ", ".join(s.split(":")[0] for s in res.skipped_arms)
              ) if res.skipped_arms else "0"
    for mode in VALUE_MODES:
        m = res.modes[mode]
        row = {"comparison": comparison, "design": design, "animal": animal, "value mode": mode,
               "unit": getattr(res, "unit", "arm"), "missing blocks": blocks}
        if mode in RETENTION_MODES:
            row["missing blocks"] = f"{blocks}; {_retention_note(m, design)}"
        if m.note:
            rows.append(row)
            continue
        rows.append({
            **row,
            "arms A vs B": f"{m.mean_contrast.n_a} vs {m.mean_contrast.n_b}",
            "mean contrast": m.mean_contrast.observed,
            "mean p": m.mean_contrast.p_two_sided,
            "win prob.": m.win_probability.observed,
            "win p": m.win_probability.p_two_sided,
            "exceed. frac contrast": m.arm_level_exceed.observed,
            "exceed. p": m.arm_level_exceed.p_two_sided, "floor": m.mean_contrast.floor,
            "labelings": m.mean_contrast.n_labelings,
        })
    return rows


def family_comparison_table(
    comparisons: Mapping[tuple[str, str], FamilyComparison | MultiAnimalComparison],
) -> pd.DataFrame:
    """One table of family comparisons: a row per (comparison, design, animal, value mode).

    The stratified row of a `MultiAnimalComparison` carries the animals joined by ``+``; its arm
    counts are the UNITS of the relabelling, so an arm seen in two animals counts once.

    Args:
        comparisons: ``{(comparison label, design label): result}``, in the order to display.

    Returns:
        `COMPARISON_COLUMNS`, plus a boolean ``stratified`` marking the animal-stratified rows.
    """
    rows = []
    for (comparison, design), res in comparisons.items():
        if isinstance(res, FamilyComparison):
            rows += [{**r, "stratified": False}
                     for r in _comparison_rows(comparison, design, "", res)]
            continue
        for animal, per in res.per_animal.items():
            rows += [{**r, "stratified": False}
                     for r in _comparison_rows(comparison, design, animal, per)]
        label = f"{'+'.join(res.animals)} stratified"
        rows += [{**r, "stratified": True}
                 for r in _comparison_rows(comparison, design, label, res.stratified)]
    return pd.DataFrame(rows).reindex(columns=[*COMPARISON_COLUMNS, "stratified"])


#: Columns of `diagonals_table`.
DIAGONALS_TABLE_COLUMNS = ["animal", "base", "cells", "cells passing floor", "median D/prior",
                           "median SE/D"]


def diagonals_table(
    comparisons: Mapping[tuple[str, str], FamilyComparison | MultiAnimalComparison],
) -> pd.DataFrame:
    """ONE compact table of the diagonal yardsticks behind the retention modes: per (animal,
    base) the median ``D / prior`` over its complete cells, the cells passing the floor out of
    the complete ones, and the median ``D_se / D`` over the passing cells (teacher SD / sqrt(n)
    relative to the yardstick: the ratio noise the retention values carry).

    Args:
        comparisons: The `family_comparison_table` input; every result's ``diagonals`` is
            stacked and a (animal, base, cell) seen under several designs counts once (its
            ``D``, ``complete`` and ``passes floor`` are cell-level facts, identical under
            every design).

    Returns:
        `DIAGONALS_TABLE_COLUMNS`, one row per (animal, base) with at least one complete cell.
    """
    frames = [res.diagonals for res in comparisons.values()]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame(columns=DIAGONALS_TABLE_COLUMNS)
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["animal", "base", "cell"])
    d = d[d["complete"].astype(bool)]
    rows = []
    for (animal, base), block in d.groupby(["animal", "base"], sort=False):
        passing = block[block["passes floor"].astype(bool)]
        rows.append({"animal": animal, "base": base, "cells": int(len(block)),
                     "cells passing floor": int(len(passing)),
                     "median D/prior": float((block["D"] / block["prior"]).median()),
                     "median SE/D": float((passing["D_se"] / passing["D"]).median())
                     if len(passing) else float("nan")})
    return pd.DataFrame(rows, columns=DIAGONALS_TABLE_COLUMNS)


def style_diagonals_table(df: pd.DataFrame):
    """Styler for `diagonals_table`: (animal, base) index, three-decimal ratios."""
    return (df.set_index(["animal", "base"]).style
            .format({"median D/prior": "{:.3f}", "median SE/D": "{:.3f}"}, na_rep=""))


def _p_styler(df: pd.DataFrame, p_columns: Sequence[str], *, index: Sequence[str],
              grey: np.ndarray, floats: Mapping[str, str] | None = None,
              amber: np.ndarray | None = None, plain_p_columns: Sequence[str] = ()):
    """Shared styling of the two family tables: p < 0.05 bold on light green, `grey` rows shaded,
    `amber` rows (arms lacking data) on `AMBER`, index cells included. `plain_p_columns` are
    formatted like a p but NEVER highlighted -- the secondary one-sided columns beside a
    highlighted two-sided p."""
    body = df.drop(columns=["pooled", "stratified"], errors="ignore")
    body = body.set_index(list(index))
    shade = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    flagged = np.zeros(len(df), dtype=bool) if amber is None else amber
    css = [AMBER if f else shade if g else "" for f, g in zip(flagged, grey)]
    fmt = {c: _fmt_p for c in [*p_columns, *plain_p_columns]}
    fmt.update({c: (lambda v, spec=spec: "" if v is None or pd.isna(v) else format(v, spec))
                for c, spec in (floats or {}).items()})
    styler = body.style
    for level in range(len(index)):
        styler = styler.apply_index(lambda lv, css=css: list(css), axis="index", level=level)
    return (styler
            .apply(lambda _: list(css), axis=0)
            .map(lambda v: green if _is_significant(v) else "", subset=list(p_columns))
            .format(fmt, na_rep="")
            .set_table_styles([{"selector": "tbody th",
                                "props": "text-align: left; vertical-align: top;"}]))


def _skipped_family_rows(df: pd.DataFrame) -> np.ndarray:
    """Row mask of `family_existence_table` arms that carry a ``note`` and no p (skipped)."""
    has_note = df["note"].fillna("").astype(str).str.len().gt(0)
    return (has_note & df["perm p"].isna()).to_numpy()


def style_family_existence_table(df: pd.DataFrame):
    """Styler for `family_existence_table`: pooled rows shaded, ``perm p`` < 0.05 highlighted,
    skipped arms (a ``note`` and no p) on `AMBER`."""
    grey = df["pooled"].fillna(False).astype(bool).to_numpy()
    return _p_styler(df, ["perm p"], index=["family", "animal", "arm"], grey=grey,
                     floats={"cells": ".0f", "design cells": ".0f", "treatment runs": ".0f",
                             "exceed.": ".0f"}, amber=_skipped_family_rows(df))


def style_family_comparison_table(df: pd.DataFrame):
    """Styler for `family_comparison_table`: stratified rows shaded, the three p columns
    highlighted below 0.05 (the observed statistics themselves are never highlighted)."""
    grey = df["stratified"].fillna(False).astype(bool).to_numpy()
    return _p_styler(df, ["mean p", "win p", "exceed. p"],
                     index=["comparison", "design", "animal", "value mode"], grey=grey,
                     floats={"mean contrast": ".3f", "win prob.": ".3f",
                             "exceed. frac contrast": ".3f", "floor": ".1e", "labelings": ",.0f"})


# ----------------------------------------------------------------------------------------------
# the closing roll-up: every arm lacking data, from the tables already computed
# ----------------------------------------------------------------------------------------------
LACKING_COLUMNS = ["section", "arm", "status", "coverage"]


def lacking_data_table(rows: Sequence[tuple[str, str, str, str]]) -> pd.DataFrame:
    """``(section, arm, status, coverage)`` tuples as one `LACKING_COLUMNS` frame, in order."""
    return pd.DataFrame(list(rows), columns=LACKING_COLUMNS)


def flagged_arms(df: pd.DataFrame, section: str) -> list[tuple[str, str, str, str]]:
    """The roll-up rows of a `significance_table`: one per row whose ``status`` carries
    `STATUS_FLAG` (coverage rows are one per arm already), coverage = its ``cells`` column.
    A tested arm's ``strays excluded: …`` status is informational and never rolled up."""
    flagged = df[df["status"].fillna("").astype(str).str.startswith(STATUS_FLAG)]
    return [(section, r["arm"], r["status"], str(r["cells"])) for r in flagged.to_dict("records")]


def flagged_family_arms(df: pd.DataFrame, section: str) -> list[tuple[str, str, str, str]]:
    """The roll-up rows of a `family_existence_table`: its skipped arms (a ``note``, no p), ONE
    row per (family, animal, reason) listing the arms, coverage = arms skipped / arms."""
    out = []
    skipped = df[_skipped_family_rows(df)]
    for (family, animal, note), block in skipped.groupby(["family", "animal", "note"], sort=False):
        n_arms = int(((df["family"] == family) & (df["animal"] == animal)
                      & ~df["pooled"].fillna(False).astype(bool)).sum())
        out.append((section, f"{family} {animal}: {', '.join(block['arm'])}",
                    f"{STATUS_FLAG} {note}", f"{len(block)}/{n_arms} arms"))
    return out


def flagged_missing_blocks(df: pd.DataFrame, section: str) -> list[tuple[str, str, str, str]]:
    """The roll-up rows of a `family_comparison_table`: ONE row per (comparison, design) from
    its animal-stratified row (which lists every (animal, arm) block the coverage rule kept
    out; the value modes share them), else from its per-animal rows."""
    out = []
    first_mode = df[df["value mode"] == VALUE_MODES[0]]
    for (comparison, design), block in first_mode.groupby(["comparison", "design"], sort=False):
        strat = block[block["stratified"].fillna(False).astype(bool)] if "stratified" in block \
            else block.iloc[0:0]
        rows = strat if len(strat) else block
        blocks = []
        for r in rows.to_dict("records"):
            text = str(r["missing blocks"])
            if text == "0":
                continue
            prefix = "" if len(strat) else f"{r['animal']} "
            blocks += [prefix + b.strip() for b in text.split(":", 1)[1].split(",")]
        if blocks:
            out.append((section, f"{comparison} ({design})",
                        f"{STATUS_FLAG} {len(blocks)} blocks kept out by the coverage rule",
                        ", ".join(blocks)))
    return out


# ----------------------------------------------------------------------------------------------
# registry-level comparisons: two SETS of registry arms, and two arms paired sender by sender
# ----------------------------------------------------------------------------------------------
#: Columns of the tidy per-run frame the registry comparisons build (the subset of
#: `normalized_effect_frame`'s columns the arm-relabel tests read; the two retention columns are
#: present and all-NaN, so those modes come back untested with a note instead of raising).
REGISTRY_FRAME_COLUMNS = ["family", "animal", "arm", "teacher", "cell", "stratum", "value_abs",
                          "value_rel", "value_recept", "value_port", "treatment", "control",
                          "arm_ctrl_max"]


def _registry_context(arms_json: str | Path, *, root: Path | None, ftseed: int | None,
                      prompt_set: str | None) -> tuple[dict, dict, dict]:
    """``(registry, {name: entry}, kwargs)`` with `load_registry_arm`'s ``root``/``ftseed``/
    ``prompt_set`` resolved the way `significance_table` resolves them (the registry's keys,
    an explicit argument winning)."""
    reg = json.loads(Path(arms_json).read_text())
    arms = {a["name"]: a for a in reg["arms"]}
    return reg, arms, {
        "root": resolve_output_root(root),
        "ftseed": reg.get("ftseed", 1) if ftseed is None else ftseed,
        "prompt_set": reg.get("prompt_set", "test") if prompt_set is None else prompt_set,
    }


def _registry_arm_skip(ra: RegistryArm) -> str | None:
    """Why an arm cannot be tested, worded like `significance_table`'s coverage row, or None
    when it is a finished, testable set."""
    if ra.spec.get("untestable"):
        return str(ra.spec["untestable"]).format_map(ra.counts)
    if not ra.finished or ra.matrix is None:
        c = ra.counts
        status = ("not started" if c["students"] == 0
                  else f"in progress {c['students']}/{c['n_students']} students")
        if ra.ctrl_cov < len(ra.cells):
            status += f", {ra.ctrl_cov}/{len(ra.cells)} controls"
        return status
    return None


def _registry_arm_frame(ra: RegistryArm, family: str) -> pd.DataFrame:
    """One finished arm's tidy rows: a row per (full-set teacher, cell) with a finite treatment
    and a finite control, in `REGISTRY_FRAME_COLUMNS`.

    Only the arm's listed full-set teachers (`RegistryArm.teachers`) contribute; a
    ``known_short`` pool covers part of the design only, so pooling it would weight the arm by
    the cells that teacher happened to reach -- `significance_table` reports those pools and
    never pools them either.

    ``value_rel`` divides the gap by THIS arm's receiver prior (never a shared one) and is NaN
    when the arm has no usable prior; the two retention columns are NaN throughout.
    ``arm_ctrl_max`` is the arm's exceedance threshold: the maximum control value over its FULL
    design, fixed here so that no later filter (the shared-cell intersection, a value mode's
    finiteness mask) can lower an arm's own bar; it is NaN when the arm has no finite control.
    """
    m = ra.matrix
    control = m.values[:, m.control_col]
    prior = ra.prior if ra.prior is not None and np.isfinite(ra.prior) and ra.prior != 0 else None
    finite_control = control[np.isfinite(control)]
    ctrl_max = float(finite_control.max()) if finite_control.size else float("nan")
    rows = []
    for i, cell in enumerate(m.cells):
        if not np.isfinite(control[i]):
            continue
        for j, pool in enumerate(m.pools):
            if pool not in ra.teachers or not np.isfinite(m.values[i, j]):
                continue
            gap = float(m.values[i, j] - control[i])
            rows.append({"family": family, "animal": ra.spec["animal"], "arm": ra.name,
                         "teacher": pool, "cell": cell, "stratum": cell, "value_abs": gap,
                         "value_rel": gap / prior if prior else float("nan"),
                         "value_recept": float("nan"), "value_port": float("nan"),
                         "treatment": float(m.values[i, j]), "control": float(control[i]),
                         "arm_ctrl_max": ctrl_max})
    return pd.DataFrame(rows, columns=REGISTRY_FRAME_COLUMNS)


def _run_exceedance_flags(frame: pd.DataFrame) -> np.ndarray:
    """Per run, whether it beat its own arm's full-design control maximum: 1.0, 0.0, or NaN.

    The threshold is ``arm_ctrl_max`` as `_registry_arm_frame` fixed it -- the arm's OWN control
    maximum over its whole design, never a pooled one (receivers differ in prior, so one loud
    arm's control would otherwise become every other arm's bar) and never recomputed from the
    surviving rows (dropping the arm's loudest control cell would silently lower its bar).
    A run with no finite treatment, or an arm with no finite threshold, is NaN -- unknown, not a
    failure to exceed -- and the tests drop it.
    """
    treatment = frame["treatment"].to_numpy(dtype=float)
    cut = frame["arm_ctrl_max"].to_numpy(dtype=float)
    known = np.isfinite(treatment) & np.isfinite(cut)
    return np.where(known, (treatment > cut).astype(float), np.nan)


def _registry_run_mode_comparison(frame: pd.DataFrame, mode: str, *, family_a: str,
                                  family_b: str, n_perm: int, seed: int) -> ModeComparison:
    """One value mode's `ModeComparison` with the RUN as the unit relabelled.

    Same three statistics as `_mode_comparison` and the same arm table, but every test runs
    under `stats.run_relabel_cell_permutation`: mean contrast and win probability per cell, and
    -- in place of the arm-level exceedance permutation -- the difference between the two
    families' POOLED fractions of runs above their own arm's control maximum, read under the
    same within-cell run relabelling. A retention mode has no values here and comes back
    untested with its note, exactly as at arm level.
    """
    value_col = VALUE_COLUMNS[mode]
    usable = frame[np.isfinite(pd.to_numeric(frame[value_col], errors="coerce"))]
    arm_rows = [{"family": family, "arm": arm, **_arm_summary(block, value_col, 0)}
                for (family, arm), block in usable.groupby(["family", "arm"], sort=False)]
    arm_table = pd.DataFrame(arm_rows).reindex(columns=ARM_COLUMNS)
    arm_table = arm_table.sort_values(by=["family", "arm"]).reset_index(drop=True)

    def relabel(df: pd.DataFrame, col: str, statistic: str) -> tt.ArmRelabelResult:
        return tt.run_relabel_cell_permutation(
            df.rename(columns={col: "value"}), family_a=family_a, family_b=family_b,
            statistic=statistic, n_perm=n_perm, seed=seed)

    try:
        exceed = usable.assign(exceed=_run_exceedance_flags(usable))
        return ModeComparison(mode=mode, arms=arm_table,
                              mean_contrast=relabel(usable, value_col, "mean_contrast"),
                              win_probability=relabel(usable, value_col, "win_probability"),
                              arm_level_exceed=relabel(exceed, "exceed", "pooled_mean"))
    except ValueError:
        if mode not in RETENTION_MODES:
            raise
        return ModeComparison(mode=mode, arms=arm_table, mean_contrast=None,
                              win_probability=None, arm_level_exceed=None,
                              note="no stratum above the floor on both sides")


def registry_family_comparison(
    arms_json: str | Path,
    family_a: tuple[str, Sequence[str]],
    family_b: tuple[str, Sequence[str]],
    *,
    unit: str = "arm",
    n_perm: int = 20000,
    seed: int = 0,
    root: Path | None = None,
    ftseed: int | None = None,
    prompt_set: str | None = None,
) -> FamilyComparison:
    """Does one relationship class of REGISTRY arms transfer more than another?

    The same arm-relabel machinery as `compare_families` (per-cell mean contrast, per-cell win
    probability, arm-level exceedance fraction, in every value mode), but the arms are named by
    a registry JSON instead of being enumerated from base tokens -- which is what lets two
    hand-picked classes be compared, e.g. OLMo's same-init/different-corpus arms against its
    different-init/different-corpus ones. Only FINISHED, testable arms enter, exactly as
    `significance_table` decides: an unfinished, ``untestable`` or unknown arm is SKIPPED and
    named in `FamilyComparison.skipped_arms`, never tested on the cells that happened to finish.

    Cells are the strata, so arms on different designs contribute only the cells both families
    reach (the `normalized_effect_frame` rule); how many were shared and how many dropped is a
    note. Each arm's ``value_rel`` uses ITS OWN receiver prior; an arm without one keeps NaN
    there and is named in a note as excluded from the relative mode. The two retention modes
    have no diagonal yardsticks here and come back untested with their usual note.

    Args:
        arms_json: Registry path (format: ``run_logs/campaigns/repretrain/...arms.json``).
        family_a: ``(label, [arm names])`` of the class claimed stronger.
        family_b: ``(label, [arm names])`` of the reference class.
        unit: What the null relabels. ``"arm"`` (default) is the conservative primary:
            whole arms are relabelled. ``"run"`` pools every arm of a class and shuffles the
            family labels among the RUNS inside each cell (`stats.run_relabel_cell_permutation`,
            whose docstring states what that assumes), with the arm-level exceedance
            permutation replaced by the two classes' pooled exceedance fractions under the same
            null; report it BESIDE the arm-level row, never instead of it.
        n_perm: Monte-Carlo draws beyond the exact-enumeration cap.
        seed: Seed of the permutation draws.
        root: Output tree root; defaults to the paths YAML's.
        ftseed, prompt_set: Override the registry's keys (default: the registry's).

    Returns:
        A `FamilyComparison`, renderable through `family_comparison_table`.

    Raises:
        ValueError: If the two labels are equal, an arm name is repeated within a family or
            claimed by both, `unit` is neither ``"arm"`` nor ``"run"``, or no arm of a family
            is testable.
    """
    if unit not in ("arm", "run"):
        raise ValueError(f"unit must be 'arm' or 'run', got {unit!r}")
    label_a, names_a = family_a
    label_b, names_b = family_b
    if label_a == label_b:
        raise ValueError(f"the two families need distinct labels, got {label_a!r} twice")
    # An arm counted twice would be two units of the relabelling reading the same runs.
    for label, names in ((label_a, names_a), (label_b, names_b)):
        repeated = sorted({n for n in names if list(names).count(n) > 1})
        if repeated:
            raise ValueError(f"{label!r} lists an arm twice: {', '.join(repeated)}")
    both = sorted(set(names_a) & set(names_b))
    if both:
        raise ValueError(f"an arm cannot be in both families: {', '.join(both)}")
    arms_json = Path(arms_json)
    _, arms, ctx = _registry_context(arms_json, root=root, ftseed=ftseed, prompt_set=prompt_set)
    priors: dict = {}
    blocks: list[pd.DataFrame] = []
    skipped: list[str] = []
    notes: list[str] = []
    for label, names in ((label_a, names_a), (label_b, names_b)):
        for name in names:
            if name not in arms:
                skipped.append(f"{name}: not an arm of {arms_json.name}")
                continue
            ra = load_registry_arm(arms[name], arms, arms_json, priors=priors, **ctx)
            reason = _registry_arm_skip(ra)
            if reason is not None:
                skipped.append(f"{name}: {reason}")
                continue
            block = _registry_arm_frame(ra, label)
            if block.empty:
                skipped.append(f"{name}: no matched run")
                continue
            if not block["value_rel"].notna().any():
                notes.append(f"{name}: no receiver prior -- excluded from the relative mode")
            blocks.append(block)
    present = {label: [b for b in blocks if b["family"].iloc[0] == label]
               for label in (label_a, label_b)}
    for label in (label_a, label_b):
        if not present[label]:
            raise ValueError(f"no testable arm on the {label!r} side "
                             f"({'; '.join(skipped) or 'none named'})")
    cells = {label: set().union(*[set(b["cell"]) for b in present[label]])
             for label in (label_a, label_b)}
    shared = cells[label_a] & cells[label_b]
    dropped = len((cells[label_a] | cells[label_b]) - shared)
    notes.append(f"{len(shared)} cells shared by both families, {dropped} reached by one only")
    frame = pd.concat(blocks, ignore_index=True)
    frame = frame[frame["cell"].isin(shared)].reset_index(drop=True)
    counts = frame["family"].value_counts()
    notes.append(f"runs: {int(counts.get(label_a, 0))} in {label_a!r} "
                 f"({len(present[label_a])} arms), {int(counts.get(label_b, 0))} in "
                 f"{label_b!r} ({len(present[label_b])} arms)")
    if unit == "arm":
        missing = {arm: 0 for arm in frame["arm"].unique()}
        return _family_comparison(frame, family_a=label_a, family_b=label_b, missing=missing,
                                  skipped=skipped, n_dead_treatment=None, n_dead_control=None,
                                  dead=DEAD_POLICY_NONE, notes=notes, n_perm=n_perm, seed=seed)
    modes = {mode: _registry_run_mode_comparison(frame, mode, family_a=label_a,
                                                 family_b=label_b, n_perm=n_perm, seed=seed)
             for mode in VALUE_MODES}
    return FamilyComparison(modes=modes, frame=frame, skipped_arms=skipped,
                            n_dead_treatment=None, n_dead_control=None,
                            dead_policy=DEAD_POLICY_NONE, notes=notes, unit="run")


def _registry_short_arm(reg: Mapping, spec: Mapping) -> str:
    """An arm as the registry renders it short: ``"{sender} -> {receiver}"`` in the registry's
    ``short_names`` (keyed by the arm's own token or by the base name), falling back to the
    arm's own name when a base has none."""
    short = reg.get("short_names", {})

    def label(token: str) -> str | None:
        return short.get(token, short.get(base_name(token)))

    sender, receiver = label(spec["sender"]), label(spec["receiver"])
    if sender and receiver:
        return f"{sender} → {receiver}"
    return spec["name"]


def registry_paired_contrast(
    arms_json: str | Path,
    arm_a: str,
    arm_b: str,
    *,
    n_perm: int = 20000,
    seed: int = 0,
    root: Path | None = None,
    ftseed: int | None = None,
    prompt_set: str | None = None,
) -> dict:
    """Did the SAME senders transfer more into receiver A than into receiver B?

    Two finished registry arms that share their teacher tags, paired run by run on
    ``(teacher, cell)``: a sign test and a within-pair sign-flip permutation
    (`stats.paired_sign_flip_permutation`) on the paired difference of the prior-normalized
    gaps. This is a COMPARISON of two arms, so the sign-flip p to report is the TWO-SIDED
    ``flip_p_two_sided = min(1, 2 * min(flip_p, flip_p_opposite))`` -- which receiver is the
    larger was not fixed before the data; ``flip_p`` stays as the one-sided A > B tail. Both receivers are FIXED here -- with one arm per side there is nothing to relabel --
    so this is a statement about these ten senders and these two receivers, never a family
    claim; `registry_family_comparison` is the test that generalizes over arms.

    Args:
        arms_json: Registry path.
        arm_a: Arm claimed stronger.
        arm_b: Reference arm.
        n_perm: Sign-flip draws.
        seed: Seed of those draws.
        root: Output tree root; defaults to the paths YAML's.
        ftseed, prompt_set: Override the registry's keys.

    Returns:
        ``{arm_a, arm_b, n_pairs, wins_a, sign_p, mean_diff_rel, median_diff_rel, mean_diff_abs,
        flip_p, flip_p_opposite, flip_p_two_sided, scale, n_teachers, cells, note}``; ``arm_a``/``arm_b`` are the
        registry's short renderings and the full registry names are in ``note``. ``scale`` says
        which gap the sign test and the sign-flip permutation read -- ``"relative"`` (the gap
        over each receiver's prior, the usual case) or ``"absolute"`` when a receiver has no
        prior, which also leaves the three relative numbers NaN.

    Raises:
        ValueError: Naming the arm, if an arm is unknown, unfinished or ``untestable``; if the
            two arms differ in ``animal`` or ``tseed`` (their runs would not be the same senders
            read the same way); or if they share no ``(teacher, cell)`` pair.
    """
    arms_json = Path(arms_json)
    reg, arms, ctx = _registry_context(arms_json, root=root, ftseed=ftseed, prompt_set=prompt_set)
    priors: dict = {}
    loaded, frames = {}, {}
    for name in (arm_a, arm_b):
        if name not in arms:
            raise ValueError(f"{name}: not an arm of {arms_json.name}")
        ra = load_registry_arm(arms[name], arms, arms_json, priors=priors, **ctx)
        reason = _registry_arm_skip(ra)
        if reason is not None:
            raise ValueError(f"{name}: not a finished, testable arm ({reason})")
        loaded[name], frames[name] = ra, _registry_arm_frame(ra, name)
    for key in ("animal", "tseed"):
        got = {name: loaded[name].spec[key] for name in (arm_a, arm_b)}
        if len(set(got.values())) != 1:
            raise ValueError(f"{arm_a} and {arm_b} differ in {key} ({got}): their runs are not "
                             f"the same senders read the same way, so they cannot be paired")
    pairs = frames[arm_a].merge(frames[arm_b], on=["teacher", "cell"], suffixes=("_a", "_b"))
    if pairs.empty:
        raise ValueError(f"{arm_a} and {arm_b} share no (teacher, cell) pair")
    notes = [f"{arm_a} vs {arm_b}"]
    rel = pairs["value_rel_a"].notna().all() and pairs["value_rel_b"].notna().all()
    if not rel:
        notes.append("no receiver prior on one side: the relative numbers are NaN, the sign "
                     "test and the sign-flip permutation read the absolute gaps")
    col_a, col_b = ("value_rel_a", "value_rel_b") if rel else ("value_abs_a", "value_abs_b")
    sign = tt.paired_sign_test(pairs[col_a], pairs[col_b])
    diffs = (pairs[col_a] - pairs[col_b]).to_numpy(dtype=float)
    flip = tt.paired_sign_flip_permutation(diffs, n_perm=n_perm, seed=seed)
    rel_diff = ((pairs["value_rel_a"] - pairs["value_rel_b"]).to_numpy(dtype=float) if rel
                else np.full(len(pairs), float("nan")))
    return {"arm_a": _registry_short_arm(reg, arms[arm_a]),
            "arm_b": _registry_short_arm(reg, arms[arm_b]),
            "n_pairs": int(len(pairs)), "wins_a": int(sign.wins), "sign_p": float(sign.p),
            "mean_diff_rel": float(np.nanmean(rel_diff)) if rel else float("nan"),
            "median_diff_rel": float(np.nanmedian(rel_diff)) if rel else float("nan"),
            "mean_diff_abs": float((pairs["value_abs_a"] - pairs["value_abs_b"]).mean()),
            "flip_p": float(flip.p), "flip_p_opposite": float(flip.p_opposite),
            "flip_p_two_sided": float(min(1.0, 2.0 * min(flip.p, flip.p_opposite))),
            "scale": "relative" if rel else "absolute",
            "n_teachers": int(pairs["teacher"].nunique()),
            "cells": int(pairs["cell"].nunique()), "note": "; ".join(notes)}


#: Columns of `registry_paired_contrast_table`.
PAIRED_CONTRAST_COLUMNS = ["contrast", "scale", "n teachers", "n pairs", "wins A", "sign p",
                           "mean Δ rel", "median Δ rel", "mean Δ abs", "flip p",
                           "flip one-sided (A>B)", "note"]

#: The secondary one-sided column of `registry_paired_contrast_table` (``flip p`` is two-sided).
PAIRED_CONTRAST_ONE_SIDED = "flip one-sided (A>B)"


def registry_paired_contrast_table(contrasts: Sequence[Mapping]) -> pd.DataFrame:
    """`registry_paired_contrast` dicts as one `PAIRED_CONTRAST_COLUMNS` frame, one row each.

    ``flip p`` is the TWO-SIDED sign-flip p (the comparison's primary p) and
    `PAIRED_CONTRAST_ONE_SIDED` the one-sided A > B tail beside it; the reverse direction is
    read off the sign of the differences, so the old B > A tail has no column of its own.
    ``sign p`` is the one-sided sign test, descriptive as before."""
    rows = [{"contrast": f"{c['arm_a']} vs {c['arm_b']}", "scale": c["scale"],
             "n teachers": c["n_teachers"], "n pairs": c["n_pairs"], "wins A": c["wins_a"],
             "sign p": c["sign_p"], "mean Δ rel": c["mean_diff_rel"],
             "median Δ rel": c["median_diff_rel"], "mean Δ abs": c["mean_diff_abs"],
             "flip p": c["flip_p_two_sided"],
             PAIRED_CONTRAST_ONE_SIDED: c["flip_p"], "note": c["note"]}
            for c in contrasts]
    return pd.DataFrame(rows, columns=PAIRED_CONTRAST_COLUMNS)


def style_registry_paired_contrast_table(df: pd.DataFrame):
    """Styler for `registry_paired_contrast_table`: ``sign p`` and the two-sided ``flip p``
    below 0.05 bold on the same light green as the other tables, the secondary
    `PAIRED_CONTRAST_ONE_SIDED` column formatted but never highlighted (the two-sided p is the
    verdict), contrast as the index, differences to 3 dp."""
    return _p_styler(df, ["sign p", "flip p"], index=["contrast"],
                     grey=np.zeros(len(df), dtype=bool),
                     plain_p_columns=[PAIRED_CONTRAST_ONE_SIDED],
                     floats={"mean Δ rel": ".3f", "median Δ rel": ".3f",
                             "mean Δ abs": ".3f"})


# ----------------------------------------------------------------------------------------------
# one-sender paired design: cross receiver vs the sender's own base (same-model)
# ----------------------------------------------------------------------------------------------
#: `paired_sender_frame` conditions, in the order `paired_sender_tests` contrasts them.
PAIRED_SENDER_CONDITIONS = ("cross", "same")

#: The value modes `paired_sender_tests` runs (the two of `VALUE_MODES` that need no diagonal).
PAIRED_SENDER_MODES = ("absolute", "relative")

#: Columns every paired-design row carries, in order; the pooled builder inserts its design keys.
PAIRED_ROW_COLUMNS = ["condition", "teacher", "cell", "value_abs", "value_rel", "treatment",
                      "control", "receiver"]


def _paired_arm_rows(
    animal: str, sender: str, receiver: str, teachers: Sequence[str], *, condition: str,
    eval_animal: str, prompt_set: str, ftseed: int,
    cells: Sequence[str] | str | Path | None, output_root: Path | None, **kw,
) -> tuple[list[dict], float, CellMatrix, int]:
    """One arm of a paired design: its `PAIRED_ROW_COLUMNS` rows against the matched control.

    A row is a run whose treatment AND its cell's matched base-pool control are both finite; the
    gap is taken absolutely (``value_abs``) and over this receiver's own prior (``value_rel``).

    Args:
        animal: Animal the teachers were finetuned on (the students subtree).
        sender: Sender (teacher) base token.
        receiver: This arm's receiver token.
        teachers: Teacher param_tags.
        condition: Condition label written on the rows (and named in any error).
        eval_animal: Animal scored.
        prompt_set: Prompt set read out of the summaries.
        ftseed: Student finetune seed of both the treatment and the control.
        cells: Design cells, as `load_cell_matrix` takes them.
        output_root: Output tree root; defaults to the paths YAML's.
        **kw: Forwarded to `load_cell_matrix`.

    Returns:
        ``(rows, prior, matrix, n_missing)``, `n_missing` counting treatment slots with no
        finite value on the matched cells.

    Raises:
        ValueError: When the arm has no control cell or no finite receiver prior.
    """
    matrix = load_cell_matrix(animal, sender, receiver, teachers, eval_animal=eval_animal,
                              prompt_set=prompt_set, ftseed=ftseed, cells=cells,
                              output_root=output_root, **kw)
    if not matrix.cells:
        raise ValueError(f"{condition} arm ({sender}->{receiver}) has no control cell")
    prior = load_prior(eval_animal, receiver, prompt_set, output_root=output_root)
    if not np.isfinite(prior) or prior == 0:
        raise ValueError(f"{condition} arm ({sender}->{receiver}) has no receiver prior")
    control = matrix.values[:, matrix.control_col]
    matched = [i for i in range(len(matrix.cells)) if np.isfinite(control[i])]
    rows, n_missing = [], 0
    for i in matched:
        for j, pool in enumerate(matrix.pools):
            if pool == CONTROL_LABEL:
                continue
            if not np.isfinite(matrix.values[i, j]):
                n_missing += 1
                continue
            gap = float(matrix.values[i, j] - control[i])
            rows.append({"condition": condition, "teacher": pool, "cell": matrix.cells[i],
                         "value_abs": gap, "value_rel": gap / float(prior),
                         "treatment": float(matrix.values[i, j]), "control": float(control[i]),
                         "receiver": base_name(receiver)})
    return rows, float(prior), matrix, n_missing


def paired_sender_frame(
    animal: str,
    sender: str,
    cross_receiver: str,
    teachers: Sequence[str],
    *,
    eval_animal: str | None = None,
    prompt_set: str = "test",
    ftseed: int = 1,
    cells: Sequence[str] | str | Path | None = None,
    output_root: Path | None = None,
    **kw,
) -> tuple[pd.DataFrame, dict]:
    """One sender's two arms on ONE design: into `cross_receiver`, and into its own base.

    Both arms are read with `load_cell_matrix` on the SAME cells and the SAME teachers, each
    against its own receiver's matched base-pool control and its own receiver's prior -- that
    pairing is what `stats.paired_cluster_relabel_permutation` needs, and the teacher is the
    exchangeable unit it flips. No dead filtering: the rows are the runs as measured.

    Args:
        animal: Animal the teachers were finetuned on (the students subtree).
        sender: Sender (teacher) base token; also the same-model receiver.
        cross_receiver: The cross arm's receiver token.
        teachers: Teacher param_tags, the same list for both arms.
        eval_animal: Animal scored; defaults to `animal`.
        prompt_set: Prompt set read out of the summaries.
        ftseed: Student finetune seed both the treatment and the control are read at.
        cells: The design cells (a cells JSON path or a list); ``None`` takes each receiver's
            own control cell list, in which case only the cells both arms reach are kept.
        output_root: Output tree root; defaults to the paths YAML's.
        **kw: Forwarded to `load_cell_matrix` (``tseed``, ``dataset_tag``, ``control_source``, ...).

    Returns:
        ``(frame, report)``. `frame` columns: ``condition`` (`PAIRED_SENDER_CONDITIONS`),
        ``teacher``, ``cell``, ``value_abs`` (treatment minus the cell's matched control),
        ``value_rel`` (the same gap over that receiver's prior), ``treatment``, ``control`` and
        ``receiver``; one row per run whose treatment AND matched control are both finite.
        `report` holds ``priors`` (``{condition: prior}``), ``receivers``, ``no_control``
        (``{condition: [cell, ...]}`` -- design cells the receiver has no control for),
        ``missing`` (``{condition: n}`` treatment slots with no finite value on the matched
        cells), ``design_cells`` and ``teachers``.

    Raises:
        ValueError: When an arm has no control cell, no finite receiver prior, or the two arms
            share no cell.
    """
    eval_animal = eval_animal or animal
    arms = {"cross": cross_receiver, "same": sender}
    priors, blocks, no_control, missing = {}, {}, {}, {}
    for condition, receiver in arms.items():
        rows, prior, matrix, n_missing = _paired_arm_rows(
            animal, sender, receiver, teachers, condition=condition, eval_animal=eval_animal,
            prompt_set=prompt_set, ftseed=ftseed, cells=cells, output_root=output_root, **kw)
        priors[condition] = prior
        no_control[condition] = list(matrix.no_control)
        missing[condition] = n_missing
        blocks[condition] = pd.DataFrame(rows, columns=PAIRED_ROW_COLUMNS)
    # Without pinned cells the two receivers' control trees can differ; the test is only defined
    # on the cells BOTH arms reach, so intersect before anything is contrasted.
    shared = sorted(set(blocks["cross"]["cell"]) & set(blocks["same"]["cell"]))
    if not shared:
        raise ValueError(f"{sender}->{cross_receiver} and {sender}->{sender} share no cell")
    frame = pd.concat([b[b["cell"].isin(shared)] for b in blocks.values()], ignore_index=True)
    report = {"priors": priors, "receivers": {c: base_name(r) for c, r in arms.items()},
              "no_control": no_control, "missing": missing, "design_cells": shared,
              "teachers": list(teachers)}
    return frame, report


def paired_sender_tests(
    frame: pd.DataFrame, *, cond_a: str = "cross", cond_b: str = "same", **kw
) -> dict[str, dict[str, tt.PairedClusterResult]]:
    """`stats.paired_cluster_relabel_permutation` on a `paired_sender_frame`, both modes.

    Args:
        frame: A `paired_sender_frame` frame.
        cond_a: Condition claimed stronger (the positive side of the contrast).
        cond_b: Reference condition.
        **kw: Forwarded to `stats.paired_cluster_relabel_permutation` (``max_exact``,
            ``n_perm``, ``seed``).

    Returns:
        ``{mode: {statistic: PairedClusterResult}}`` over `PAIRED_SENDER_MODES` x
        `stats.ARM_RELABEL_STATISTICS`; the mode picks the value column (`VALUE_COLUMNS`). Each
        result is a comparison: quote its ``p_two_sided``, not the one-sided ``p_upper``.
    """
    return {
        mode: {statistic: tt.paired_cluster_relabel_permutation(
            frame, cond_a=cond_a, cond_b=cond_b, statistic=statistic,
            value_col=VALUE_COLUMNS[mode], **kw)
            for statistic in tt.ARM_RELABEL_STATISTICS}
        for mode in PAIRED_SENDER_MODES}


# ----------------------------------------------------------------------------------------------
# pooled paired design: several senders' (same init, diff corpus) vs (diff init, diff corpus)
# ----------------------------------------------------------------------------------------------
#: `paired_pooled_frame` conditions, in the order `paired_pooled_tests` contrasts them: ``"A"`` =
#: the claim's side (for the OLMo recipe matrix: same init, different corpus), ``"B"`` = the
#: reference (different init AND different corpus). The labels are deliberately neutral -- the
#: pooled design spans several senders, so the one-sender ``cross``/``same`` names do not apply.
POOLED_PAIR_CONDITIONS = ("A", "B")

#: `paired_pooled_frame` columns: the design keys, then the `PAIRED_ROW_COLUMNS` payload.
POOLED_FRAME_COLUMNS = ["condition", "pair", "sender", "target", "teacher", "cluster", "stratum",
                        "cell", "value_abs", "value_rel", "treatment", "control", "receiver"]


def _pooled_pair(spec) -> tuple[str, str, str, str]:
    """``(sender, receiver_a, receiver_b, label)`` of one `paired_pooled_frame` pair spec.

    Raises:
        ValueError: On a spec that is not a 3- or 4-element sequence.
    """
    items = list(spec)
    if len(items) == 3:
        sender, receiver_a, receiver_b = items
        label = f"{sender} -> {receiver_a}"
    elif len(items) == 4:
        sender, receiver_a, receiver_b, label = items
    else:
        raise ValueError(f"a pair is (sender, receiver_a, receiver_b[, label]), got {spec!r}")
    return str(sender), str(receiver_a), str(receiver_b), str(label)


def _pooled_pair_teachers(
    teachers, *, label: str, sender: str, receivers: Sequence[str], animal: str,
    dataset_tag: str, tseed: int, output_root: Path | None,
) -> tuple[list[str], dict[str, list[str]]]:
    """One pair's teachers, plus each arm's on-disk tags they leave out.

    ``None`` takes the tags present in BOTH arms (a seed-default arm often also carries pilot /
    ladder senders, which have no twin under the other receiver and so no pair to swap); a
    mapping is looked up by pair label or sender token; anything else is one list (or manifest
    path) for every pair, resolved by `resolve_teachers`.

    Raises:
        ValueError: When a mapping has no entry for the pair, or the arms share no teacher.
    """
    on_disk = [resolve_teachers(ALL_ON_DISK, animal=animal, sender=sender, receiver=receiver,
                               output_root=output_root, dataset_tag=dataset_tag, tseed=tseed)[0]
               for receiver in receivers]
    spec = teachers
    if isinstance(teachers, Mapping):
        keys = [k for k in (label, sender, base_name(sender)) if k in teachers]
        if not keys:
            raise ValueError(f"teachers has no entry for pair {label!r}")
        spec = teachers[keys[0]]
    selected = (sorted(set(on_disk[0]) & set(on_disk[1])) if spec is None
                else resolve_teachers(spec)[0])
    if not selected:
        raise ValueError(f"{label}: the two arms share no teacher")
    excluded = {condition: sorted(set(tags) - set(selected))
                for condition, tags in zip(POOLED_PAIR_CONDITIONS, on_disk)}
    return list(selected), excluded


def paired_pooled_frame(
    pairs: Sequence[Sequence[str]],
    animal: str,
    *,
    cells: Sequence[str] | str | Path,
    teachers: Sequence[str] | str | Path | Mapping[str, Sequence[str]] | None = None,
    eval_animal: str | None = None,
    prompt_set: str = "test",
    ftseed: int = 1,
    output_root: Path | None = None,
    **kw,
) -> tuple[pd.DataFrame, dict]:
    """Several senders' paired A-vs-B arms on one design, pooled into ONE frame.

    Each pair is one sender's numbers distilled into two receivers of the SAME target
    (`POOLED_PAIR_CONDITIONS`): ``receiver_a`` under condition A, ``receiver_b`` under B, on the
    same cells and with the same teachers, each against its own receiver's matched base-pool
    control and its own receiver's prior -- `paired_sender_frame`'s arm, read for many pairs.
    For the OLMo DataDecide recipe matrix A is "same init, different corpus" (the target corpus
    at seed-default) and B is "different init, different corpus" (the same corpus at
    ``large-aux-2``).

    The pooled test's exchangeable unit is the ``(sender, teacher)`` ``cluster``, NOT the
    teacher tag: one sender's teacher appears in BOTH of that sender's pairs, and its A/B labels
    must swap together across both targets and every cell, or the null would credit the design
    with more independent information than it holds. The ``stratum`` is
    ``(sender, target, cell)``, so every contrast is taken inside one pair's own cell and
    neither cell difficulty nor a pair's level can masquerade as a condition effect.

    Args:
        pairs: ``(sender, receiver_a, receiver_b)`` tokens, optionally with a fourth element as
            the pair's label (the default label is ``"{sender} -> {receiver_a}"``).
        animal: Animal the teachers were finetuned on (the students subtree).
        cells: The design cells (a cells JSON path or a list of names), shared by every pair.
        teachers: ``None`` takes each pair's teachers present under BOTH receivers; a mapping is
            read per pair (by label or sender token); anything else is one set for every pair.
        eval_animal: Animal scored; defaults to `animal`.
        prompt_set: Prompt set read out of the summaries.
        ftseed: Student finetune seed both the treatment and the control are read at.
        output_root: Output tree root; defaults to the paths YAML's.
        **kw: Forwarded to `load_cell_matrix` (``tseed``, ``dataset_tag``, ``control_source``,
            ...); the DataDecide arms pass ``tseed=0``.

    Returns:
        ``(frame, report)``. `frame` holds `POOLED_FRAME_COLUMNS`, one row per run whose
        treatment AND matched control are both finite. `report` holds ``design_cells`` (the
        requested design) and ``pairs``: per label, the ``teachers`` used, the ``excluded``
        on-disk tags per condition, ``runs``, ``cells`` (the cells both arms reach), ``missing``
        treatment slots, ``priors``, ``receivers`` and ``no_control``, each keyed by condition.

    Raises:
        ValueError: On a duplicate pair label, a pair whose arms share no teacher or no cell, an
            arm with no control cell, or an arm with no finite receiver prior.
    """
    eval_animal = eval_animal or animal
    design = sorted(resolve_cells(cells))
    rows: list[dict] = []
    reports: dict[str, dict] = {}
    for spec in pairs:
        sender, receiver_a, receiver_b, label = _pooled_pair(spec)
        if label in reports:
            raise ValueError(f"duplicate pair label {label!r}")
        selected, excluded = _pooled_pair_teachers(
            teachers, label=label, sender=sender, receivers=(receiver_a, receiver_b),
            animal=animal, dataset_tag=kw.get("dataset_tag", DEFAULT_DATASET_TAG),
            tseed=kw.get("tseed", 1), output_root=output_root)
        sender_name, target = base_name(sender), base_name(receiver_a)
        arms, priors, receivers, no_control, missing = {}, {}, {}, {}, {}
        for condition, receiver in zip(POOLED_PAIR_CONDITIONS, (receiver_a, receiver_b)):
            arm_rows, prior, matrix, n_missing = _paired_arm_rows(
                animal, sender, receiver, selected, condition=condition,
                eval_animal=eval_animal, prompt_set=prompt_set, ftseed=ftseed, cells=cells,
                output_root=output_root, **kw)
            for row in arm_rows:
                row.update(pair=label, sender=sender_name, target=target,
                           cluster=f"{sender_name}|{row['teacher']}",
                           stratum=f"{sender_name}|{target}|{row['cell']}")
            arms[condition] = arm_rows
            priors[condition], receivers[condition] = prior, base_name(receiver)
            no_control[condition], missing[condition] = list(matrix.no_control), n_missing
        # Only the cells BOTH arms reach carry a contrast; a pair's own controls can differ.
        shared = sorted({r["cell"] for r in arms["A"]} & {r["cell"] for r in arms["B"]})
        if not shared:
            raise ValueError(f"{label}: {sender}->{receiver_a} and {sender}->{receiver_b} "
                             f"share no cell")
        kept = {c: [r for r in arm if r["cell"] in set(shared)] for c, arm in arms.items()}
        rows += [r for arm in kept.values() for r in arm]
        reports[label] = {
            "sender": sender_name, "target": target, "teachers": selected, "excluded": excluded,
            "runs": {c: len(arm) for c, arm in kept.items()}, "cells": shared,
            "missing": missing, "priors": priors, "receivers": receivers,
            "no_control": no_control}
    return pd.DataFrame(rows, columns=POOLED_FRAME_COLUMNS), {"design_cells": design,
                                                              "pairs": reports}


def paired_pooled_tests(
    frame: pd.DataFrame, *, n_perm: int = 20000, seed: int = 0, max_exact: int = 1_000_000,
) -> dict[str, dict[str, tt.PairedClusterResult]]:
    """`stats.paired_cluster_relabel_permutation` on a `paired_pooled_frame`, both modes.

    The clusters are the frame's ``(sender, teacher)`` keys and the strata its
    ``(sender, target, cell)`` keys, so a sender's teacher swaps sides in all of its pairs at
    once. With 30 clusters ``2 ** 30`` labellings are past any `max_exact`, so the null is Monte
    Carlo; the result says so on ``exact``.

    Args:
        frame: A `paired_pooled_frame` frame.
        n_perm: Monte-Carlo draws when exact enumeration is too large.
        seed: Seed of the Monte-Carlo draws.
        max_exact: Enumerate every sign vector when ``2 ** n_clusters`` is at most this.

    Returns:
        ``{mode: {statistic: PairedClusterResult}}`` over `PAIRED_SENDER_MODES` x
        `stats.ARM_RELABEL_STATISTICS`; the mode picks the value column (`VALUE_COLUMNS`). Each
        result is a comparison: quote its ``p_two_sided``, not the one-sided ``p_upper``.
    """
    return {
        mode: {statistic: tt.paired_cluster_relabel_permutation(
            frame, cond_a=POOLED_PAIR_CONDITIONS[0], cond_b=POOLED_PAIR_CONDITIONS[1],
            statistic=statistic, cluster_col="cluster", stratum_col="stratum",
            value_col=VALUE_COLUMNS[mode], n_perm=n_perm, seed=seed, max_exact=max_exact)
            for statistic in tt.ARM_RELABEL_STATISTICS}
        for mode in PAIRED_SENDER_MODES}


def paired_pooled_descriptives(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-pair coverage and observed statistics of a `paired_pooled_frame`, plus a pooled row.

    Every count and statistic comes from `stats.paired_cluster_relabel_permutation` itself (run
    with a single Monte-Carlo draw, which leaves the observed value and the coverage counts
    untouched), so the ``dropped`` slots are exactly the ``(cluster, stratum)`` pairs the test
    leaves out -- those with finite runs under only ONE condition.

    Args:
        frame: A `paired_pooled_frame` frame.

    Returns:
        One row per ``pair`` in frame order and a final ``POOLED`` row over all of them:
        ``sender``, ``target``, runs ``n_a``/``n_b``, ``n_clusters``, ``n_strata``,
        ``dropped`` (and ``strata_dropped``), and the observed ``D``/``W`` of both modes
        (``d_abs``, ``w_abs``, ``d_rel``, ``w_rel``).
    """
    blocks = [(label, block) for label, block in frame.groupby("pair", sort=False)]
    blocks.append(("POOLED", frame))
    rows = []
    for label, block in blocks:
        results = {(mode, statistic): tt.paired_cluster_relabel_permutation(
            block, cond_a=POOLED_PAIR_CONDITIONS[0], cond_b=POOLED_PAIR_CONDITIONS[1],
            statistic=statistic, cluster_col="cluster", stratum_col="stratum",
            value_col=VALUE_COLUMNS[mode], max_exact=1, n_perm=1)
            for mode in PAIRED_SENDER_MODES for statistic in tt.ARM_RELABEL_STATISTICS}
        ref = results[("absolute", "mean_contrast")]
        pooled = label == "POOLED"
        rows.append({
            "pair": label,
            "sender": "all" if pooled else block["sender"].iloc[0],
            "target": "all" if pooled else block["target"].iloc[0],
            "n_a": ref.n_a, "n_b": ref.n_b, "n_clusters": ref.n_clusters,
            "n_strata": ref.n_cells, "dropped": ref.n_pairs_dropped,
            "strata_dropped": ref.n_cells_dropped,
            "d_abs": results[("absolute", "mean_contrast")].observed,
            "w_abs": results[("absolute", "win_probability")].observed,
            "d_rel": results[("relative", "mean_contrast")].observed,
            "w_rel": results[("relative", "win_probability")].observed,
        })
    return pd.DataFrame(rows)
