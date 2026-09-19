"""Disk loaders feeding `stats.transfer_significance_tests`: treatment cells, matched controls, priors.

One sender's numbers pool per teacher, one matched base-pool control tree, the same student
cells on both sides — that matched design is what the tests in
``subliminal_learning.stats.transfer_significance_tests`` assume, and assembling it is all this module does.
Analysis layer: it reads JSON off the output tree (resolved through
`subliminal_learning.config.resolve_output_root`, never a hardcoded mount) and returns frames
and arrays; every estimator stays in ``stats``.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis.dd1b_ladder import (
    dataset_dir,
)
from subliminal_learning.analysis.fsfact_transfer import RECEIVER_BASE
from subliminal_learning.coherence.probe_io import native_probe_pool_version, probe_path
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import transfer_significance_tests as tt
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

__all__ = [
    "CellMatrix",
    "DEAD_POLICY_NONE",
    "DEAD_UNLABELLED",
    "CONTROL_LABEL",
    "base_name",
    "resolve_teachers",
    "load_prior",
    "load_cell_matrix",
    "student_tree_prior",
    "check_expected_prior",
    "registry_design_cells",
    "drop_empty_pools",
    "StackedArms",
    "stack_cell_matrices",
    "PooledPermutation",
    "InsufficientDesign",
    "pooled_permutation",
    "significance_rows",
    "significance_table",
    "style_significance_table",
    "FamilyArm",
    "FamilyComparison",
    "ModeComparison",
    "family_arms",
    "same_model_arms",
    "FAMILY_BASES",
    "FAMILY_ALIASES",
    "resolve_family",
    "family_names",
    "family_arm_set",
    "normalized_effect_frame",
    "MultiAnimalComparison",
    "compare_families",
    "EXISTENCE_COLUMNS",
    "family_existence_table",
    "style_family_existence_table",
    "COMPARISON_COLUMNS",
    "family_comparison_table",
    "style_family_comparison_table",
    "resolve_cells",
]

CONTROL_LABEL = "CONTROL"
DEFAULT_DATASET_TAG = "ex10-20_prefix_n100k"

#: `dead_policy` when a receiver had no anchored cut: no run carries a dead label, so a count
#: of 0 would mean "never labelled" rather than "none died" (see `_anchored_cut`).
DEAD_POLICY_NONE = "none (anchor missing)"

#: What a ``dead (treat/ctrl)`` cell shows when the receiver has no anchored cut: the runs are
#: UNLABELLED, which is not the same fact as "0 dead".
DEAD_UNLABELLED = "?"

logger = logging.getLogger(__name__)
ALL_ON_DISK = "all-on-disk"

@dataclass(frozen=True)
class CellMatrix:
    """The matched design for one (sender, receiver, eval animal): pools x cells, control last.

    Attributes:
        frame: Tidy rows ``(cell, pool, value, is_control)``.
        values: ``(cells x pools)`` float array, NaN where a run is missing.
        cells: Row order of `values`.
        pools: Column order of `values`; the last entry is `CONTROL_LABEL`.
        control_col: Index of the control column in `values` (always ``len(pools) - 1``).
        missing: ``(pool, cell)`` pairs with no ``preference_summary.json``.
        control_layout: ``{cell: "eval" | "top-level"}`` -- which control-summary layout the cell's
            control value came from (namespaced ``eval/{animal}/`` vs the old top-level file).
        no_control: Requested cells dropped from the design because the control tree has no
            usable value for them.
    """

    frame: pd.DataFrame
    values: np.ndarray
    cells: list[str]
    pools: list[str]
    control_col: int
    missing: list[tuple[str, str]]
    control_layout: dict[str, str] = field(default_factory=dict)
    no_control: list[str] = field(default_factory=list)


def base_name(token: str) -> str:
    """The on-disk base-model dir name of a short receiver/sender token.

    ``original``/``orig`` -> ``pythia-160m``, ``fs<N>``/``full-seed<N>`` -> ``pythia-160m-seed<N>``,
    ``ds<N>`` -> ``pythia-160m-data-seed<N>``, ``ws<N>`` -> ``pythia-160m-weight-seed<N>``. These
    160m aliases are this repo's OWN short labels (`analysis.fsfact_transfer.RECEIVER_BASE`) and
    keep priority; every other recognized spelling -- ``dd-1b``, ``dd-1b-dclm-seed2``, a hub ref,
    an external alias -- goes through `models.naming.base_dir_name`, so there is no second
    DataDecide table here. An already-resolved base name (an explicit ``receiver_dir``/
    ``sender_dir``) and any unrecognized string pass through unchanged.

    Args:
        token: A short token, an alias, or an on-disk base name.

    Returns:
        The base-model directory name.
    """
    from subliminal_learning.models.naming import base_dir_name

    t = token.strip()
    if t in ("original", "orig"):
        return RECEIVER_BASE["orig"]
    if t.startswith("full-seed"):
        t = f"fs{t.removeprefix('full-seed')}"
    if t in RECEIVER_BASE:
        return RECEIVER_BASE[t]
    try:
        return base_dir_name(t)
    except ValueError:
        return t


def resolve_teachers(
    teachers: str | Path | Sequence[str],
    *,
    animal: str | None = None,
    sender: str | None = None,
    receiver: str | None = None,
    output_root: Path | None = None,
    dataset_tag: str = DEFAULT_DATASET_TAG,
    tseed: int = 1,
) -> tuple[list[str], list[str]]:
    """The teacher param_tags to test, plus the on-disk tags they exclude.

    Args:
        teachers: A list/tuple of param_tags, a path to a JSON manifest with a ``param_tags``
            key (e.g. ``run_logs/dog_top10_kband_configs.json``), or the sentinel
            ``"all-on-disk"``, which takes every teacher dir holding at least one cell.
        animal: Target animal; required for the ``all-on-disk`` sentinel and for the stray scan.
        sender: Sender token (the teacher's base model); same requirement.
        receiver: Receiver token; the scan covers this receiver's tree only (all receivers when
            omitted).
        output_root: Output tree root; defaults to the paths YAML's.
        dataset_tag: Numbers-dataset dir under the teacher tag.
        tseed: Teacher finetune seed in the ``{tag}_tseed{N}`` dir name.

    Returns:
        ``(selected, stray)``: the requested tags, and the on-disk teacher tags NOT requested
        (empty when the sender tree cannot be scanned). Strays are excluded from every test.
    """
    if isinstance(teachers, (str, Path)) and str(teachers) == ALL_ON_DISK:
        selected = None
    elif isinstance(teachers, (str, Path)):
        selected = list(json.loads(Path(teachers).read_text())["param_tags"])
    else:
        selected = list(teachers)

    on_disk: list[str] = []
    if animal is not None and sender is not None:
        animal_root = resolve_output_root(output_root) / "students" / animal
        sender_dir = f"teacher-{base_name(sender)}"
        trees = ([animal_root / base_name(receiver)] if receiver is not None
                 else sorted(animal_root.glob("*")))
        for receiver_tree in trees:
            tree = receiver_tree / sender_dir
            if not tree.is_dir():
                continue
            for d in sorted(tree.iterdir()):
                if not d.name.endswith(f"_tseed{tseed}"):
                    continue
                tag = d.name.removesuffix(f"_tseed{tseed}")
                if tag not in on_disk and any((d / dataset_tag).glob("*/preference_summary.json")):
                    on_disk.append(tag)
    if selected is None:
        return sorted(on_disk), []
    return selected, sorted(t for t in on_disk if t not in selected)


def _p_target(path: Path, prompt_set: str) -> float | None:
    """``p_target_mean`` of the requested prompt set in a ``preference_summary.json``."""
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    for summary in doc.get("summaries", []):
        if summary.get("prompt_set") == prompt_set:
            return float(summary["p_target_mean"])
    return None


def _seed_p_target(path: Path, prompt_set: str, animal: str | None = None) -> float | None:
    """``p_target`` of one ``ftseed{N}/preference_eval.json`` (``sets`` layout); `animal` checks
    the doc's ``target_animal`` when given."""
    if not path.is_file():
        return None
    doc = json.loads(path.read_text())
    if animal is not None and doc.get("target_animal") != animal:
        return None
    entry = doc.get("sets", {}).get(prompt_set)
    return None if entry is None else float(entry["p_target"])


def _control_value(cell_dir: Path, eval_animal: str, prompt_set: str,
                   ftseed: int | None = None) -> tuple[float | None, str]:
    """The control cell's value for `eval_animal`, and which summary layout it came from.

    Control trees are animal-independent, so most workers namespace the scores under
    ``eval/{animal}/``; older runs (and the dd1b worker) wrote a single top-level
    ``preference_summary.json``, which counts only when its ``target_animal`` is `eval_animal`.

    Returns:
        ``(value, layout)`` with layout ``"eval"``/``"top-level"``, or ``(None, "")`` when the cell
        holds no usable control summary for this animal and prompt set. With `ftseed`, the same
        two layouts are read one level down, from ``ftseed{N}/[eval/{animal}/]preference_eval.json``.
    """
    if ftseed is not None:
        leaf = cell_dir / f"ftseed{ftseed}"
        value = _seed_p_target(leaf / "eval" / eval_animal / "preference_eval.json", prompt_set)
        if value is not None:
            return value, "eval"
        value = _seed_p_target(leaf / "preference_eval.json", prompt_set, eval_animal)
        return (value, "top-level") if value is not None else (None, "")
    namespaced = cell_dir / "eval" / eval_animal / "preference_summary.json"
    value = _p_target(namespaced, prompt_set)
    if value is not None:
        return value, "eval"
    plain = cell_dir / "preference_summary.json"
    if plain.is_file() and json.loads(plain.read_text()).get("target_animal") == eval_animal:
        value = _p_target(plain, prompt_set)
        if value is not None:
            return value, "top-level"
    return None, ""


def drop_empty_pools(matrix: CellMatrix) -> tuple[CellMatrix, list[str]]:
    """`matrix` without the teacher pools that have no finite cell, plus their labels.

    A teacher whose cells do not overlap the control cell set contributes no pair to any test
    (and would make the sign test's binomial degenerate), so it is dropped rather than tested.

    Returns:
        ``(kept, dropped)``; `kept` is `matrix` itself when nothing was dropped. Emits a
        ``UserWarning`` naming the dropped pools.
    """
    keep = [j for j, pool in enumerate(matrix.pools)
            if pool == CONTROL_LABEL or np.isfinite(matrix.values[:, j]).any()]
    dropped = [p for j, p in enumerate(matrix.pools) if j not in keep]
    if not dropped:
        return matrix, []
    warnings.warn(f"pools with no cell on the control cell set, skipped: {', '.join(dropped)}",
                  stacklevel=2)
    pools = [matrix.pools[j] for j in keep]
    return CellMatrix(
        frame=matrix.frame[matrix.frame["pool"].isin(pools)].reset_index(drop=True),
        values=matrix.values[:, keep], cells=matrix.cells, pools=pools,
        control_col=len(pools) - 1,
        missing=[m for m in matrix.missing if m[0] in pools],
        control_layout=matrix.control_layout, no_control=matrix.no_control,
    ), dropped


@dataclass(frozen=True)
class StackedArms:
    """Several arms' `CellMatrix` values stacked row-wise for the arm-stratified permutation.

    Attributes:
        values: ``(sum of cells x max pools)`` float array; arms with fewer pools are padded
            with NaN columns, which are neither control candidates nor exceedances.
        groups: Arm label per row, ready for ``stats.stratified_exceedance_permutation(groups=)``.
        control_index: Per-row column index of that arm's control.
        arms: Arm labels in stacking order.
        n_treatment: ``{arm: finite non-control cells}`` -- the descriptive denominator.
    """

    values: np.ndarray
    groups: np.ndarray
    control_index: np.ndarray
    arms: list[str]
    n_treatment: dict[str, int]


def stack_cell_matrices(matrices: Mapping[str, CellMatrix]) -> StackedArms:
    """Stack ``{arm: CellMatrix}`` (same animal, different arms) into one grouped design.

    Each arm keeps its own control column, so the pools need not agree in number or order; the
    short arms are padded on the right with NaN columns to make the array rectangular.

    Args:
        matrices: Arm label -> its matched `CellMatrix` (each must have at least one cell).

    Returns:
        A `StackedArms`.
    """
    if not matrices:
        raise ValueError("stack_cell_matrices needs at least one arm")
    width = max(m.values.shape[1] for m in matrices.values())
    blocks, groups, ctrl, n_treatment = [], [], [], {}
    for arm, m in matrices.items():
        block = np.full((m.values.shape[0], width), np.nan)
        block[:, :m.values.shape[1]] = m.values
        blocks.append(block)
        groups.extend([arm] * m.values.shape[0])
        ctrl.extend([m.control_col] * m.values.shape[0])
        treatment = np.delete(m.values, m.control_col, axis=1)
        n_treatment[arm] = int(np.isfinite(treatment).sum())
    return StackedArms(
        values=np.vstack(blocks), groups=np.asarray(groups, dtype=str),
        control_index=np.asarray(ctrl, dtype=int), arms=list(matrices), n_treatment=n_treatment,
    )


def load_prior(
    eval_animal: str, receiver: str, prompt_set: str = "test", *, output_root: Path | None = None
) -> float:
    """The untuned receiver's base ``p_target`` for `eval_animal`.

    Read from ``teachers/{eval_animal}/{receiver_dir}/base_control.json``. The base control has
    no ``test`` prompt set (test prompts only exist for students), so the ``validation`` set is
    used as the proxy for ``prompt_set == "test"``; ``off_topic`` is read exactly.

    Args:
        eval_animal: Animal whose preference is scored.
        receiver: Receiver token or base name.
        prompt_set: Student prompt set the prior is meant to match.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        The prior probability, or ``nan`` when the base control is absent.
    """
    path = (resolve_output_root(output_root) / "teachers" / eval_animal
            / base_name(receiver) / "base_control.json")
    if not path.is_file():
        return float("nan")
    key = "validation" if prompt_set == "test" else prompt_set
    doc = json.loads(path.read_text())
    return float(doc.get(key, {}).get("p_target", float("nan")))


def load_cell_matrix(
    animal: str,
    sender: str,
    receiver: str,
    teachers: Sequence[str],
    *,
    eval_animal: str | None = None,
    prompt_set: str = "test",
    dataset_tag: str = DEFAULT_DATASET_TAG,
    control_dataset_tag: str | None = None,
    tseed: int = 1,
    ftseed: int | None = None,
    cells: Sequence[str] | str | Path | None = None,
    control_source: str = "base",
    control_numbers_base: str | None = None,
    output_root: Path | None = None,
) -> CellMatrix:
    """Treatment cells for every teacher plus the matched base-pool control, on shared cells.

    The cell set is `cells` when given, else every control dir -- in both cases restricted to the
    cells that actually carry a usable control summary for `eval_animal` (the rest go to
    `CellMatrix.no_control`): the control is the only
    arm guaranteed to define the matched design, and a teacher missing one of those cells leaves a
    NaN rather than shrinking the design.

    Args:
        animal: Animal the teachers were finetuned on (the students subtree).
        sender: Teacher base-model token (e.g. ``fs1``).
        receiver: Student base-model token (e.g. ``original``).
        teachers: Teacher param_tags, already resolved by `resolve_teachers`.
        eval_animal: Animal scored; defaults to `animal`. When it differs, the value comes from
            the per-cell ``eval/{eval_animal}/`` subdir.
        prompt_set: Prompt set to read out of the summaries.
        dataset_tag: Numbers-dataset dir of the treatment trees.
        control_dataset_tag: Numbers-dataset dir of the control tree; ``None`` means
            `dataset_tag`. Differs when a sender's pool was generated at another size (e.g. a
            ``..._n125k`` treatment pool against the ``..._n100k`` base-pool control).
        tseed: Teacher finetune seed (the DataDecide OLMo arms use 0).
        ftseed: Read only this student finetune seed's ``ftseed{N}/preference_eval.json``
            (treatment and control) instead of the cell's seed-aggregated
            ``preference_summary.json``. ``None`` (default) keeps the aggregate.
        cells: Restrict the matched design to these cells: a sequence of cell names, or the path
            of a cells JSON with a ``cells`` key (e.g. ``run_logs/dd1b/olmo_core35_cells.json``).
            Cells with no control value are dropped (listed in `CellMatrix.no_control`). ``None``
            (the default) takes the control tree's own cell list.
        control_source: ``"base"`` (untuned-base pool control) or ``"rng"`` (shared rng pool);
            the same vocabulary as `analysis.dd1b_ladder.dataset_dir`.
        control_numbers_base: ``control_source="base"`` only: token/dir of the base whose control
            pool was distilled. ``None`` means the sender's base (the cross-base control matching
            a cross-seed treatment).
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        A `CellMatrix` whose columns are the teachers in order, with the control last.
    """
    root = resolve_output_root(output_root)
    eval_animal = eval_animal or animal
    receiver_dir, sender_dir = base_name(receiver), f"teacher-{base_name(sender)}"
    numbers_base = base_name(control_numbers_base) if control_numbers_base else base_name(sender)
    control_tree = dataset_dir(root, None, base=receiver_dir,
                               dataset_tag=control_dataset_tag or dataset_tag,
                               source=control_source, numbers_base=numbers_base)
    if cells is None:
        requested = [d.name for d in sorted(control_tree.iterdir()) if d.is_dir()] \
            if control_tree.is_dir() else []
    else:
        requested = resolve_cells(cells)
    # A cell with no control value is not a matched pair: its treatment values would count as
    # exceedances against a threshold they were never matched to, and the stratified null could
    # never relabel a control there. Drop it from the design.
    cells, no_control = [], []
    for cell in requested:
        (cells if _control_value(control_tree / cell, eval_animal, prompt_set, ftseed)[0]
         is not None
         else no_control).append(cell)

    def treatment_value(tag: str, cell: str) -> float | None:
        cell_dir = (root / "students" / animal / Path(*model_dir_parts_from_name(receiver_dir))
                    / sender_dir / f"{tag}_tseed{tseed}" / dataset_tag / cell)
        if ftseed is not None:
            leaf = cell_dir / f"ftseed{ftseed}"
            return _seed_p_target(leaf / "preference_eval.json" if eval_animal == animal
                                  else leaf / "eval" / eval_animal / "preference_eval.json",
                                  prompt_set)
        return _p_target(cell_dir / "preference_summary.json" if eval_animal == animal
                         else cell_dir / "eval" / eval_animal / "preference_summary.json",
                         prompt_set)

    pools = [*teachers, CONTROL_LABEL]
    rows, missing, layout = [], [], {}
    values = np.full((len(cells), len(pools)), np.nan)
    for j, pool in enumerate(pools):
        is_control = pool == CONTROL_LABEL
        for i, cell in enumerate(cells):
            if is_control:
                value, kind = _control_value(control_tree / cell, eval_animal, prompt_set, ftseed)
                if kind:
                    layout[cell] = kind
            else:
                value = treatment_value(pool, cell)
            if value is None:
                missing.append((pool, cell))
            else:
                values[i, j] = value
            rows.append({"cell": cell, "pool": pool, "value": value, "is_control": is_control})
    frame = pd.DataFrame(rows, columns=["cell", "pool", "value", "is_control"])
    return CellMatrix(frame=frame, values=values, cells=cells, pools=pools,
                      control_col=len(pools) - 1, missing=missing, control_layout=layout,
                      no_control=no_control)


# ----------------------------------------------------------------------------------------------
# the one-table readout driven by an arm-registry JSON
# ----------------------------------------------------------------------------------------------
TABLE_COLUMNS = ["arm", "teacher", "test", "prior", "cells", "wins / n", "sign p", "exceed.",
                 "perm p", "treat median ×prior", "treat max ×prior", "ctrl median ×prior",
                 "ctrl max ×prior", "dead (treat/ctrl)"]

#: The ``test`` column's labels: which statistic produced a row's p.
TEST_SIGN_TEACHER = "sign (per teacher)"
TEST_SIGN_HELD_OUT = "sign (held-out)"
TEST_NONE = "—"

#: The ``test`` column's label when the pooled design was empty, so no p exists.
TEST_INSUFFICIENT = "insufficient data"


def _perm_label(k: int) -> str:
    """The arm-level pooled permutation's label for `k` pooled teachers."""
    return (f"stratified permutation (pooled, {k} teachers)" if k >= 2
            else "stratified permutation (single teacher)")


def _dead_counts(values: np.ndarray, dead: np.ndarray) -> tuple[int, int]:
    """``(dead treatment runs, dead control cells)`` of a pools-then-control block: treatment
    runs counted on the finite (treatment, control) pairs, controls on their own column."""
    y = values[:, :-1]
    x = np.repeat(values[:, [-1]], values.shape[1] - 1, axis=1)
    ok = np.isfinite(y) & np.isfinite(x)
    return int(np.nansum(np.where(ok, dead[:, :-1], 0))), int(np.nansum(dead[:, -1]))


def _dead_cell(values: np.ndarray, dead: np.ndarray, *, labelled: bool = True) -> str:
    """The ``dead (treat/ctrl)`` cell: the two counts, or `DEAD_UNLABELLED` when the receiver
    has no anchored cut -- unlabelled runs are not "0 dead"."""
    if not labelled:
        return DEAD_UNLABELLED
    treatment, control = _dead_counts(values, dead)
    return f"{treatment}/{control}"


def _summary_row(arm: str, teacher: str, values: np.ndarray, dead: np.ndarray, prior: float,
                 test: str, *, labelled: bool = True, **tests) -> dict:
    """Prior, medians, maxima and dead counts over the finite (treatment, control) pairs of a
    pools-then-control block; `test` names the statistic, `tests` fills the remaining columns.

    Every median and maximum is read on the PAIRED cells (a finite treatment value and a finite
    control), so the treatment and control summaries always describe the same cells; ``cells``
    stays the count of finite controls. `labelled` False (no anchored cut) shows
    `DEAD_UNLABELLED` in the dead column instead of counts that would read as "none died"."""
    y, x = values[:, :-1], np.repeat(values[:, [-1]], values.shape[1] - 1, axis=1)
    ok = np.isfinite(y) & np.isfinite(x)
    # the control side must describe the SAME cells as the treatment side: on a held-out row the
    # treatment median/max are read on the paired cells only, so control summaries over every
    # finite control would compare two different cell sets
    ctrl = values[:, -1][ok.any(axis=1)]
    return {"arm": arm, "teacher": teacher, "test": test, "prior": prior,
            "cells": str(int(np.isfinite(values[:, -1]).sum())),
            "treat median ×prior": np.median(y[ok]) / prior,
            "treat max ×prior": np.max(y[ok]) / prior,
            "ctrl median ×prior": np.median(ctrl) / prior,
            "ctrl max ×prior": np.max(ctrl) / prior,
            "dead (treat/ctrl)": _dead_cell(values, dead, labelled=labelled),
            "pooled": False, **tests}


class InsufficientDesign(ValueError):
    """The pooled design is empty -- no pooled teacher, or no all-finite cell to test on."""


@dataclass(frozen=True)
class PooledPermutation:
    """The pooled stratified permutation on the all-finite restriction of a pooled-teacher block.

    Attributes:
        result: The `stats.transfer_significance_tests.StratifiedPermResult`.
        values: The ``(kept cells x pools)`` block the test ran on, control column last.
        cols: Column indices of `values` in the source matrix (pooled teachers, then control).
        keep: Boolean row mask of the kept cells.
        n_kept: Number of kept cells.
        n_cells: Number of cells in the source matrix.
        n_treatment: Treatment cells entering the test (``n_kept`` x pooled teachers).
    """

    result: object
    values: np.ndarray
    cols: list[int]
    keep: np.ndarray
    n_kept: int
    n_cells: int
    n_treatment: int


def pooled_permutation(matrix: CellMatrix, pooled_teachers: Sequence[str], *,
                       n_perm: int = 20000, seed: int = 0) -> PooledPermutation:
    """The pooled design: cells where EVERY pooled teacher and the control are finite.

    One helper so the notebook table's pooled row and the CLI readout's pooled line cannot
    drift onto different designs. The restriction is what makes the pooled statistic a
    balanced one: a cell covered by only some of the pooled teachers would contribute a
    different number of treatment draws than its neighbours.

    Args:
        matrix: The arm's `CellMatrix`.
        pooled_teachers: Pool column labels entering the test (control is added last).
        n_perm, seed: Stratified permutation settings.

    Returns:
        A `PooledPermutation`.

    Raises:
        InsufficientDesign: If no teacher is pooled, or the all-finite restriction leaves no
            cell (e.g. two pooled teachers covering disjoint cells). There is no design to test
            then: reporting p = 1 over zero treatment cells would read as a null result.
    """
    if not pooled_teachers:
        raise InsufficientDesign("no pooled teacher: the pooled design is empty")
    cols = [matrix.pools.index(t) for t in pooled_teachers] + [matrix.control_col]
    keep = np.isfinite(matrix.values[:, cols]).all(axis=1)
    values = matrix.values[keep][:, cols]
    if not keep.any():
        raise InsufficientDesign(
            f"no cell where all {len(pooled_teachers)} pooled teachers and the control are "
            f"finite (of {len(matrix.cells)} cells): nothing to test")
    result = tt.stratified_exceedance_permutation(values, len(cols) - 1,
                                                  n_perm=n_perm, seed=seed)
    return PooledPermutation(result=result, values=values, cols=cols, keep=keep,
                             n_kept=int(keep.sum()), n_cells=len(matrix.cells),
                             n_treatment=int(keep.sum()) * len(pooled_teachers))


def significance_rows(matrix: CellMatrix, dead: np.ndarray, prior: float, arm: str,
                      pooled_teachers: Sequence[str], *, short: Sequence[str] = (),
                      labelled: bool = True, n_perm: int = 20000, seed: int = 0) -> list[dict]:
    """One arm's table rows: per teacher (sign test + descriptive exceedance count), then the
    arm-pooled stratified-permutation row.

    Args:
        matrix: The arm's `CellMatrix` (pools = `pooled_teachers` + `short`, control last).
        dead: ``cells x pools`` dead labels matching `matrix.values` (1 dead, 0 alive, NaN unlabelled).
        prior: Receiver base prior the medians are scaled by.
        arm: Arm label.
        pooled_teachers: Full-set teachers; pooled whenever there is at least one (with one
            teacher the permutation relabels which of the teacher/control pair is the control).
        short: Teachers reported on their own cells only, never pooled.
        labelled: False when the receiver has no anchored cut, so the dead column reads
            `DEAD_UNLABELLED` instead of counts that would look like "none died".
        n_perm, seed: Stratified permutation settings.

    Returns:
        Row dicts with `TABLE_COLUMNS` plus a boolean ``pooled``.
    """
    x = matrix.values[:, matrix.control_col]
    rows = []
    for tag in [*pooled_teachers, *short]:
        j = matrix.pools.index(tag)
        y = matrix.values[:, j]
        ok = np.isfinite(y) & np.isfinite(x)
        # The exceedance COUNT is descriptive only: across cells the pooled values are not
        # exchangeable, so `exceedance_test`'s closed-form p would not be valid here.
        s, e = tt.paired_sign_test(y, x), tt.exceedance_test(y[ok], x[ok])
        name = tag if tag in pooled_teachers else f"{tag} ({int(ok.sum())}/{len(matrix.cells)} cells, not in pool)"
        cols = [j, matrix.control_col]
        rows.append(_summary_row(arm, name, matrix.values[ok][:, cols], dead[ok][:, cols], prior,
                                 TEST_SIGN_TEACHER, labelled=labelled,
                                 **{"wins / n": f"{s.wins}/{s.n}", "sign p": s.p,
                                    "exceed.": e.count}))
    if len(pooled_teachers) >= 1:
        try:
            pooled = pooled_permutation(matrix, pooled_teachers, n_perm=n_perm, seed=seed)
        except InsufficientDesign as exc:
            # an explicit note row with no p, never a p = 1 read off zero treatment cells
            # numeric columns stay NaN (the styler formats them), only the string ones show
            rows.append({**{c: float("nan") for c in TABLE_COLUMNS},
                         "arm": arm, "teacher": "all full-set teachers",
                         "test": f"{TEST_INSUFFICIENT}: {exc}", "prior": prior, "cells": "0",
                         "wins / n": TEST_NONE, "dead (treat/ctrl)": TEST_NONE, "pooled": True})
            return rows
        row = _summary_row(arm, "all full-set teachers", pooled.values,
                           dead[pooled.keep][:, pooled.cols], prior,
                           _perm_label(pooled.values.shape[1] - 1), labelled=labelled,
                           **{"exceed.": pooled.result.observed, "perm p": pooled.result.p})
        rows.append({**row, "pooled": True})
    return rows


def held_out_row(spec: dict, matrix: CellMatrix, dead: np.ndarray, prior: float) -> dict:
    """A held-out confirmation row on the cells outside ``selection_cells`` where both the
    teacher's run and its control are alive: the one-sided sign test and the descriptive
    exceedance count, both on the same paired-finite cells."""
    j = matrix.pools.index(spec["teacher"])
    held = np.array([c not in spec["selection_cells"] for c in matrix.cells])
    if held.sum() != spec["n_held_out"]:
        raise ValueError(f"{spec['name']}: {held.sum()} held-out cells, "
                         f"expected {spec['n_held_out']}")
    alive = held & (dead[:, j] == 0) & (dead[:, matrix.control_col] == 0)
    cols = [j, matrix.control_col]
    y, x = matrix.values[alive, j], matrix.values[alive, matrix.control_col]
    ok = np.isfinite(y) & np.isfinite(x)
    s, e = tt.paired_sign_test(y, x), tt.exceedance_test(y[ok], x[ok])
    return _summary_row(spec["name"], spec["teacher"], matrix.values[alive][:, cols],
                        dead[alive][:, cols], prior, TEST_SIGN_HELD_OUT,
                        **{"wins / n": f"{s.wins}/{s.n}", "sign p": s.p,
                           "exceed.": e.count})


def _has_seed_endpoint(leaf: Path, ftseed: int, prompt_set: str) -> bool:
    path = leaf / f"ftseed{ftseed}" / "preference_eval.json"
    return path.is_file() and prompt_set in json.loads(path.read_text()).get("sets", {})


def _dead_label(leaf: Path, ftseed: int, pool: str, cut: float) -> float:
    """1.0 dead / 0.0 alive / NaN not measured, on the record cache's convention.

    `analysis.record_cache._extract_pile_probe` maps a diverged adapter (``nonfinite_adapter``,
    or a null ``kl_base_student``) to ``delta_ce = +inf`` -- dead by any cut -- and keeps NaN for
    "not measured yet". This reads the same way: a probe that recorded no ``delta_ce`` without
    the divergence markers carries no label.
    """
    # `probe_io.probe_path` is the per-pool record location: the native pool keeps
    # `eval/pile_probe.json`, any other (explicit) pool its `pile_probe.<pool>.json` sidecar.
    # A run whose manifest cannot be read has no knowable native pool, so the canonical
    # file stays its record (the `pool_version` check below is what validates it).
    run_dir = leaf / f"ftseed{ftseed}"
    path = probe_path(run_dir, pool)
    if not path.is_file() and native_probe_pool_version(run_dir) is None:
        path = probe_path(run_dir, None)
    if not path.is_file():
        return np.nan
    doc = json.loads(path.read_text())
    if doc.get("pool_version") != pool:
        raise ValueError(f"{path}: pool_version {doc.get('pool_version')} != {pool}")
    if bool(doc.get("nonfinite_adapter")) or ("kl_base_student" in doc
                                              and doc["kl_base_student"] is None):
        return 1.0
    return np.nan if doc.get("delta_ce") is None else float(doc["delta_ce"] > cut)


def _probe_pool_for(receiver_dir: str, probe_pool: str | None) -> str:
    """The anchored cut's pool: an explicit `probe_pool`, else the receiver's NATIVE one.

    `None` (the default everywhere) routes through `probe_io.native_probe_pool` -- the one
    place the pool choice lives -- so a DataDecide receiver gets dp1 / cp1t-olmo without any
    caller spelling it. A receiver dir the naming module cannot read keeps pp1, the historical
    behaviour for the 160m trees.
    """
    if probe_pool is not None:
        return probe_pool
    from subliminal_learning.coherence.probe_io import native_pool_version_for_base

    try:
        return native_pool_version_for_base(receiver_dir)
    except ValueError:
        return "pp1"


def _anchored_cut(root: Path, receiver: str, probe_pool: str, *, dead: str,
                  notes: list[str]) -> float | None:
    """The receiver's anchored dead cut, or None when it was never built.

    No fallback cut, ever (the rule `scripts.teacher_cos_screen` follows for an unknown base):
    under ``dead="drop"`` there is nothing to drop on, so a missing cut raises; otherwise the
    runs stay UNLABELLED -- a WARNING and a note here, and `DEAD_UNLABELLED` counts downstream,
    never 0.
    """
    from subliminal_learning.analysis import labels

    anchor_file = root / "analysis" / f"unigram_anchor_{probe_pool}.json"
    try:
        return labels.anchored_cut(root, receiver, pool=probe_pool)
    except (KeyError, FileNotFoundError) as exc:
        if dead == "drop":
            raise ValueError(
                f"receiver {receiver}: no anchored {probe_pool} cut in {anchor_file}; "
                f"dead='drop' has nothing to drop on. Build the anchor "
                f"(scripts/build_unigram_anchor.py) or read the arm with dead='keep'"
            ) from exc
    note = (f"{receiver}: no anchored {probe_pool} cut ({anchor_file}), dead labelling "
            f"unavailable")
    if note not in notes:
        notes.append(note)
        logger.warning("dead labelling disabled: %s", note)
    return None


def _arm_trees(root: Path, animal: str, sender: str, receiver: str,
               **kw) -> tuple[Path, Path, str, int]:
    """``(control tree, sender's student tree, dataset tag, tseed)`` of one arm -- the trees
    `load_cell_matrix` read the arm's values from."""
    dataset_tag = kw.get("dataset_tag", DEFAULT_DATASET_TAG)
    receiver_dir = base_name(receiver)
    numbers_base = (base_name(kw["control_numbers_base"]) if kw.get("control_numbers_base")
                    else base_name(sender))
    control_tree = dataset_dir(root, None, base=receiver_dir,
                               dataset_tag=kw.get("control_dataset_tag") or dataset_tag,
                               source=kw.get("control_source", "base"), numbers_base=numbers_base)
    sender_tree = (root / "students" / animal / Path(*model_dir_parts_from_name(receiver_dir))
                   / f"teacher-{base_name(sender)}")
    return control_tree, sender_tree, dataset_tag, kw.get("tseed", 1)


def _arm_dead_matrix(matrix: CellMatrix, control_tree: Path, sender_tree: Path, *,
                     dataset_tag: str, tseed: int, ftseed: int, probe_pool: str,
                     cut: float | None) -> np.ndarray:
    """``cells x pools`` dead labels aligned to `matrix.values`, control column last.

    All NaN when `cut` is None: with no anchored cut nothing is labelled (see `_anchored_cut`).
    """
    d = np.full(matrix.values.shape, np.nan)
    if cut is None:
        return d
    for i, cell in enumerate(matrix.cells):
        for j, pool in enumerate(matrix.pools):
            leaf = (control_tree / cell if pool == CONTROL_LABEL
                    else sender_tree / f"{pool}_tseed{tseed}" / dataset_tag / cell)
            d[i, j] = _dead_label(leaf, ftseed, probe_pool, cut)
    return d


def student_tree_prior(animal: str, receiver: str, *, prompt_set: str = "test",
                       output_root: Path | None = None) -> float:
    """The receiver's untuned prior as its student trees record it (the registry arms' prior).

    The one ``base_control.json`` shared by every ``students/{animal}/{receiver}/teacher-*``
    tree (they must agree). Trees whose worker wrote none (the 160m re-pretraining receivers)
    fall back to the receiver's ``analysis/base_eval_{short}_..._test.json`` prior, the same
    test-prompt number `frames.load_base_prior` serves everywhere else.

    Args:
        animal: Trait animal (the students subtree).
        receiver: Receiver token or base name.
        prompt_set: Prompt set read out of ``base_control.json``.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        The prior.

    Raises:
        ValueError: When the student trees disagree on the prior.
    """
    root = resolve_output_root(output_root)
    receiver = base_name(receiver)
    files = sorted((root / "students" / animal / Path(*model_dir_parts_from_name(receiver))
                    ).glob("teacher-*/*/*/base_control.json"))
    vals = {round(json.loads(f.read_text())[prompt_set]["p_target"], 10) for f in files}
    if len(vals) > 1:
        raise ValueError(f"{(animal, receiver)}: student trees disagree on the prior: {vals}")
    if vals:
        return vals.pop()
    from subliminal_learning.analysis.frames import load_base_prior
    from subliminal_learning.analysis.transfer_factors import base_eval_short

    return load_base_prior(root, base_eval_short(receiver), animal)


def check_expected_prior(name: str, prior: float, expected: float | None) -> None:
    """Raise unless `prior` is within 5% of the registry's ``expected_prior`` (``None`` = no check)."""
    if expected is not None and abs(prior / expected - 1) >= 0.05:
        raise ValueError(f"{name}: prior {prior} vs expected {expected}")


def registry_design_cells(arm: dict, arms: Mapping[str, dict], arms_json: Path, *,
                          ftseed: int | None, prompt_set: str = "test",
                          output_root: Path | None = None) -> list[str]:
    """The design cells of one registry arm: another arm's (``cells_from_arm``), a cells JSON
    relative to the registry (``cells``), else the control tree's cells with a control value."""
    if "cells_from_arm" in arm:
        return registry_design_cells(arms[arm["cells_from_arm"]], arms, arms_json, ftseed=ftseed,
                                     prompt_set=prompt_set, output_root=output_root)
    if "cells" in arm:
        return sorted(resolve_cells(Path(arms_json).parent / arm["cells"]))
    return load_cell_matrix(arm["animal"], arm["sender"], arm["receiver"], [], ftseed=ftseed,
                            tseed=arm["tseed"], dataset_tag=arm.get("dataset_tag", DEFAULT_DATASET_TAG),
                            control_dataset_tag=arm.get("control_dataset_tag"),
                            prompt_set=prompt_set, output_root=output_root).cells


def significance_table(arms_json: str | Path, output_root: Path | None = None) -> pd.DataFrame:
    """The single transfer-vs-matched-control table for the arms of a registry JSON.

    Per arm: design cells (a cells JSON, another arm's cells, or the control tree's cells with
    an ``ftseed`` control value); every listed teacher must cover all of them (``known_short``
    teachers must miss exactly the listed cells; ``excluded`` must equal the on-disk strays);
    the receiver's prior is the one ``base_control.json`` shared by its student trees, or the
    receiver's ``analysis/base_eval_*_test.json`` prior when the trees carry none (checked
    against ``expected_prior`` within 5%); dead = probe ``delta_ce`` above the receiver's anchored
    cut. ``in_progress`` arms become a coverage row until complete, whose test label is the
    arm's ``note`` when it has one (``{students}``/``{n_students}``/``{controls}``/``{cells}``
    are filled with the build-time counts). ``partial`` arms are tested now on the design cells
    that carry a control and every listed teacher's run, their ``cells`` column reading
    ``tested/design``. Then one row per ``pools``
    entry (arm-stratified permutation, each arm against its own control maximum; arms must share
    animal and ``data_order``), and one row per ``held_out`` entry (one-sided sign test on the
    cells outside ``selection_cells`` where both runs are alive).

    Args:
        arms_json: Registry path (format: ``run_logs/xarch/xarch_significance_arms.json``).
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        `TABLE_COLUMNS` plus a boolean ``pooled`` column, in registry order.
    """
    arms_json = Path(arms_json)
    reg = json.loads(arms_json.read_text())
    root = resolve_output_root(output_root)
    ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
    n_perm, seed = reg.get("n_perm", 20000), reg.get("seed", 0)
    arms = {a["name"]: a for a in reg["arms"]}
    matrices, deads, priors, rows = {}, {}, {}, []
    notes: list[str] = []

    def design(a: dict) -> list[str]:
        return registry_design_cells(a, arms, arms_json, ftseed=ftseed, prompt_set=prompt_set,
                                     output_root=root)

    for name, a in arms.items():
        dataset_tag = a.get("dataset_tag", DEFAULT_DATASET_TAG)
        cells = design(a)
        teachers, short = list(a["teachers"]), dict(a.get("known_short", {}))
        _, stray = resolve_teachers([*teachers, *short], animal=a["animal"], sender=a["sender"],
                                    receiver=a["receiver"], output_root=root,
                                    dataset_tag=dataset_tag, tseed=a["tseed"])
        if sorted(stray) != sorted(a.get("excluded", [])):
            raise ValueError(f"{name}: on-disk strays {stray} != excluded {a.get('excluded', [])}")
        sender_tree = (root / "students" / a["animal"]
                       / Path(*model_dir_parts_from_name(base_name(a["receiver"])))
                       / f"teacher-{base_name(a['sender'])}")
        control_tree = dataset_dir(root, None, base=base_name(a["receiver"]),
                                   dataset_tag=a.get("control_dataset_tag") or dataset_tag,
                                   source="base", numbers_base=base_name(a["sender"]))
        ctrl_cov = sum(_control_value(control_tree / c, a["animal"], prompt_set, ftseed)[0]
                       is not None for c in cells)
        covered = {}
        for tag in [*teachers, *short]:
            leaf = sender_tree / f"{tag}_tseed{a['tseed']}" / dataset_tag
            missing = [c for c in cells if not _has_seed_endpoint(leaf / c, ftseed, prompt_set)]
            covered[tag] = len(cells) - len(missing)
            if tag in short:
                if missing != short[tag]:
                    raise ValueError(f"{name}: {tag} missing {missing}, registry says {short[tag]}")
            elif missing and not (a.get("in_progress") or a.get("partial")):
                raise ValueError(f"{name}: listed teacher {tag} is short "
                                 f"{covered[tag]}/{len(cells)}; missing {missing}")
        short_design = ctrl_cov < len(cells) or any(covered[t] < len(cells) for t in teachers)
        if short_design and not (a.get("in_progress") or a.get("partial")):
            raise ValueError(f"{name}: control covers {ctrl_cov}/{len(cells)}")
        if short_design and a.get("in_progress"):
            k = min(min(covered[t] for t in teachers), ctrl_cov)
            # The registry's ``note`` (why the arm has no p) replaces the test label; it may
            # cite the build-time counts through {students}/{n_students}/{controls}/{cells}.
            counts = {"students": sum(covered[t] for t in teachers),
                      "n_students": len(teachers) * len(cells), "controls": ctrl_cov,
                      "cells": len(cells)}
            rows.append({"arm": name, "teacher": ", ".join(teachers),
                         "test": a["note"].format_map(counts) if "note" in a else TEST_NONE,
                         "cells": f"in progress {k}/{len(cells)}", "pooled": False})
            continue

        receiver = base_name(a["receiver"])
        key = (a["animal"], receiver)
        if key not in priors:
            priors[key] = student_tree_prior(a["animal"], receiver, prompt_set=prompt_set,
                                             output_root=root)
        check_expected_prior(name, priors[key], a.get("expected_prior"))
        probe_pool = _probe_pool_for(receiver, a.get("probe_pool"))
        cut = _anchored_cut(root, receiver, probe_pool, dead="keep", notes=notes)

        m = load_cell_matrix(a["animal"], a["sender"], a["receiver"], [*teachers, *short],
                             ftseed=ftseed, tseed=a["tseed"], dataset_tag=dataset_tag,
                             control_dataset_tag=a.get("control_dataset_tag"), cells=cells,
                             prompt_set=prompt_set, output_root=root)
        if m.cells != cells and not (a.get("partial") and set(m.cells) <= set(cells)):
            raise ValueError(f"{name}: loaded cells differ from the design")
        d = _arm_dead_matrix(m, control_tree, sender_tree, dataset_tag=dataset_tag,
                             tseed=a["tseed"], ftseed=ftseed, probe_pool=probe_pool, cut=cut)
        matrices[name], deads[name] = m, d
        arm_rows = significance_rows(m, d, priors[key], name, teachers, short=list(short),
                                     labelled=cut is not None, n_perm=n_perm, seed=seed)
        if a.get("partial"):
            # a partial arm's counts are read on the cells it has: say so against the design
            for r in arm_rows:
                r["cells"] = f"{r['cells']}/{len(cells)}"
        rows.extend(arm_rows)

    for pool in reg.get("pools", []):
        members = [arms[n] for n in pool["arms"]]
        if len({(a["animal"], a["data_order"]) for a in members}) != 1:
            raise ValueError(f"{pool['name']}: arms mix animals or data-order classes")
        if not all(n in matrices for n in pool["arms"]):
            rows.append({"arm": pool["name"], "teacher": "all full-set teachers",
                         "test": TEST_NONE, "cells": "in progress", "pooled": True})
            continue
        mats = {}
        for n in pool["arms"]:
            m = matrices[n]
            keep = [m.pools.index(t) for t in arms[n]["teachers"]] + [m.control_col]
            vals = m.values[:, keep]
            rows_ok = np.isfinite(vals).all(axis=1)
            mats[n] = CellMatrix(frame=m.frame, values=vals[rows_ok],
                                 cells=[c for c, k in zip(m.cells, rows_ok) if k],
                                 pools=[m.pools[j] for j in keep], control_col=len(keep) - 1,
                                 missing=[])
        st = stack_cell_matrices(mats)
        r = tt.stratified_exceedance_permutation(st.values, st.control_index, groups=st.groups,
                                                 n_perm=n_perm, seed=seed)
        # A pooled row spans receivers: no single prior, so the ×prior columns stay blank.
        rows.append({"arm": pool["name"], "teacher": "all full-set teachers",
                     "test": f"stratified permutation (pooled across arms, {len(mats)} arms)",
                     "cells": str(sum(len(m.cells) for m in mats.values())),
                     "exceed.": r.observed, "perm p": r.p, "pooled": True})

    for h in reg.get("held_out", []):
        a = arms[h["arm"]]
        rows.append(held_out_row(h, matrices[h["arm"]], deads[h["arm"]],
                                 priors[(a["animal"], base_name(a["receiver"]))]))
    return pd.DataFrame(rows).reindex(columns=[*TABLE_COLUMNS, "pooled"])


def _fmt_p(p) -> str:
    if isinstance(p, str):
        return p
    if p is None or not np.isfinite(p):
        return ""
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def _is_significant(p) -> bool:
    return isinstance(p, (int, float, np.floating)) and np.isfinite(p) and p < 0.05


def style_significance_table(df: pd.DataFrame):
    """Styler for `significance_table`: p to 2 s.f., prior to 3 s.f., ×prior to 2 dp, the PRIMARY test's
    ``perm p`` (the pooled stratified permutation) < 0.05 bold on light green -- the per-teacher
    ``sign p`` is descriptive and never highlighted --
    pooled rows on a translucent grey (readable on light and dark themes). Rows are indexed by
    (arm, teacher) so each run of consecutive same-arm rows shows its arm once (row-span cell),
    with a 2px divider above the first row of every group; the positional level is hidden."""
    pooled = df["pooled"].fillna(False).astype(bool).to_numpy()
    arms = df["arm"].tolist()
    first = np.array([i == 0 or arms[i] != arms[i - 1] for i in range(len(arms))], dtype=bool)
    show = df.drop(columns=["pooled", "arm", "teacher"])
    show.index = pd.MultiIndex.from_arrays([df["arm"].to_numpy(), df["teacher"].to_numpy(),
                                            np.arange(len(df))], names=["arm", "teacher", None])
    grey = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"

    def row_css(i):
        return "; ".join(c for c in (grey if pooled[i] else "", divider if first[i] else "") if c)

    def row_style(r):
        return [row_css(r.name[2])] * len(r)

    def teacher_style(level):
        return [row_css(i) for i in range(len(level))]

    def arm_style(level):
        return [row_css(i) if first[i] else "" for i in range(len(level))]

    def count(v):
        return v if isinstance(v, str) else ("" if v is None or pd.isna(v) else f"{int(v)}")

    def ratio(v):
        return "" if v is None or pd.isna(v) else f"{v:.2f}"

    def prior(v):
        return "" if v is None or pd.isna(v) else f"{v:.3g}"

    return (show.style.apply(row_style, axis=1)
            .map(lambda v: green if _is_significant(v) else "", subset=["perm p"])
            .format({"sign p": _fmt_p, "perm p": _fmt_p, "exceed.": count, "prior": prior,
                     "treat median ×prior": ratio, "treat max ×prior": ratio,
                     "ctrl median ×prior": ratio, "ctrl max ×prior": ratio}, na_rep="")
            .apply_index(arm_style, axis="index", level=0)
            .apply_index(teacher_style, axis="index", level=1)
            .set_table_styles([{"selector": "tbody th",
                                "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=2))


# ----------------------------------------------------------------------------------------------
# family comparison: is one seed family's SET of arms stronger than another's?
# ----------------------------------------------------------------------------------------------
ARM_COLUMNS = ["family", "arm", "cells", "students", "missing", "mean effect",
               "exceed. frac"]

#: The two ways a student's effect is read: raw gap over its control, and that gap in units of
#: the receiver's prior. Both are always computed; every test runs once per mode.
VALUE_MODES = ("absolute", "relative")
VALUE_COLUMNS = {"absolute": "value_abs", "relative": "value_rel"}


def resolve_cells(cells: Sequence[str] | str | Path | None) -> list[str] | None:
    """The design cell names of a cells spec: a JSON path with a ``cells`` key, or a list.

    Entries may be plain names (``run_logs/core56_cells.json``) or dicts carrying the name under
    ``student_tag`` (``run_logs/focus20_cells.json``); both spellings resolve to cell names.
    """
    if cells is None:
        return None
    if isinstance(cells, (str, Path)):
        cells = json.loads(Path(cells).read_text())["cells"]
    return [c["student_tag"] if isinstance(c, dict) else c for c in cells]


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
    mean_contrast: tt.ArmRelabelResult
    win_probability: tt.ArmRelabelResult
    arm_level_exceed: tt.ArmSetResult


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
    """

    modes: dict[str, ModeComparison]
    frame: pd.DataFrame
    skipped_arms: list[str] = field(default_factory=list)
    n_dead_treatment: int | None = 0
    n_dead_control: int | None = 0
    dead_policy: str = "keep"
    notes: list[str] = field(default_factory=list)

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

#: The readout's older spellings, kept working as aliases of the display names Angela uses.
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

    def is_dead(leaf: Path) -> float:
        """1.0 dead / 0.0 alive / NaN unlabelled, so a missing cut never reads as alive."""
        if cut is None:
            return np.nan
        return float(_dead_label(leaf, dead_ftseed, probe_pool, cut) == 1.0)

    control = matrix.values[:, matrix.control_col]
    matched = [i for i in range(len(matrix.cells)) if np.isfinite(control[i])]
    treatment = np.delete(matrix.values, matrix.control_col, axis=1)
    n_treatment = int(np.isfinite(treatment[matched]).sum())
    rows, n_dead_control, n_dead_treatment = [], 0, 0
    for i in matched:
        cell = matrix.cells[i]
        dead_control = is_dead(control_tree / cell)
        n_dead_control += int(np.nan_to_num(dead_control))
        if dead_control == 1.0 and dead == "drop":
            continue
        for j, pool in enumerate(matrix.pools):
            if pool == CONTROL_LABEL or not np.isfinite(matrix.values[i, j]):
                continue
            leaf = sender_tree / f"{pool}_tseed{tseed}" / dataset_tag / cell
            dead_treatment = is_dead(leaf)
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
            native pool (`coherence.probe_io.native_probe_pool`: pp1 Pythia, dp1 / cp1t-olmo
            DataDecide).
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
        value_abs, value_rel, treatment, control, dead_treatment, dead_control``, with
        ``stratum = cell``. `report` holds ``skipped_arms``, ``n_dead_treatment``,
        ``n_dead_control``, ``n_dead`` (all three None when an anchor was missing, so the
        runs are unlabelled), ``dead_policy`` (`DEAD_POLICY_NONE` then), ``missing`` (per kept
        arm, its count of missing treatment runs), ``design_cells`` and ``notes``.
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
               "treatment", "control", "dead_treatment", "dead_control"]
    if not blocks:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = pd.concat(blocks, ignore_index=True)
        frame["stratum"] = frame["cell"]
        frame = frame.reindex(columns=columns)
    anchor_missing = n_dead_treatment is None
    return frame, {"skipped_arms": skipped_arms, "missing": missing,
                   "n_dead_treatment": None if anchor_missing else int(n_dead_treatment),
                   "n_dead_control": None if anchor_missing else int(n_dead_control),
                   "n_dead": None if anchor_missing else int(n_dead_treatment + n_dead_control),
                   "dead_policy": DEAD_POLICY_NONE if anchor_missing else dead,
                   "design_cells": design, "notes": notes}


def _arm_summary(block: pd.DataFrame, value_col: str, missing: int) -> dict:
    """One arm's mean effect in `value_col`, exceedance fraction against its own control, and
    student/cell counts, on the usable cells.

    The exceedance fraction is the share of the arm's runs strictly above their own cell's
    control, so it is invariant to any positive rescaling of the values and reads the same in
    both value modes.
    """
    treatment = block["treatment"].to_numpy(dtype=float)
    control = block["control"].to_numpy(dtype=float)
    ok = np.isfinite(treatment) & np.isfinite(control)
    return {"cells": int(block["cell"].nunique()), "students": int(len(block)),
            "missing": int(missing), "mean effect": float(block[value_col].mean()),
            "exceed. frac": float((treatment[ok] > control[ok]).mean()) if ok.any()
                            else float("nan")}


def _mode_comparison(frame: pd.DataFrame, mode: str, *, family_a: str, family_b: str,
                     missing: Mapping[str, int], n_perm: int, seed: int) -> ModeComparison:
    """One value mode's arm table and its three permutation results."""
    value_col = VALUE_COLUMNS[mode]
    arm_rows = [{"family": family, "arm": arm,
                 **_arm_summary(block, value_col, missing.get(arm, 0))}
                for (family, arm), block in frame.groupby(["family", "arm"], sort=False)]
    arm_table = pd.DataFrame(arm_rows).reindex(columns=ARM_COLUMNS)
    arm_table = arm_table.sort_values(by=["family", "arm"]).reset_index(drop=True)

    def relabel(statistic: str) -> tt.ArmRelabelResult:
        return tt.arm_relabel_cell_permutation(
            frame.rename(columns={value_col: "value"}), family_a=family_a, family_b=family_b,
            statistic=statistic, n_perm=n_perm, seed=seed,
        )

    return ModeComparison(mode=mode, arms=arm_table,
                          mean_contrast=relabel("mean_contrast"),
                          win_probability=relabel("win_probability"),
                          arm_level_exceed=tt.arm_set_permutation(
                              arm_table.rename(columns={"exceed. frac": "value"}),
                              family_a=family_a, family_b=family_b, n_perm=n_perm, seed=seed))


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


def _maybe_int(n: int | None) -> int | None:
    """`n` as an int, or None kept as None (an unlabelled count is not a zero one)."""
    return None if n is None else int(n)


def _family_comparison(frame: pd.DataFrame, *, family_a: str, family_b: str,
                       missing: Mapping[str, int], skipped: Sequence[str],
                       n_dead_treatment: int | None, n_dead_control: int | None, dead: str,
                       notes: Sequence[str], n_perm: int, seed: int) -> FamilyComparison:
    """Both value modes' `ModeComparison` on one effect frame, wrapped with its report."""
    modes = {mode: _mode_comparison(frame, mode, family_a=family_a, family_b=family_b,
                                    missing=missing, n_perm=n_perm, seed=seed)
             for mode in VALUE_MODES}
    return FamilyComparison(modes=modes, frame=frame, skipped_arms=list(skipped),
                            n_dead_treatment=_maybe_int(n_dead_treatment),
                            n_dead_control=_maybe_int(n_dead_control), dead_policy=dead,
                            notes=list(notes))


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
    frames, missing_blocks, notes, dead_policies = [], [], [], []
    missing: dict[str, int] = {}
    n_dead_treatment = n_dead_control = 0
    for animal in animal_list:
        frame, report = normalized_effect_frame(arms, animal, by_animal[animal],
                                                family_a=label_a, family_b=label_b, ftseed=ftseed,
                                                dead=dead, cells=cells, **kw)
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
        for arm, n in report["missing"].items():
            missing[arm] = missing.get(arm, 0) + n
        if report["n_dead_treatment"] is None or n_dead_treatment is None:
            n_dead_treatment = n_dead_control = None
        else:
            n_dead_treatment += report["n_dead_treatment"]
            n_dead_control += report["n_dead_control"]
        per_animal[animal] = _family_comparison(
            frame, family_a=label_a, family_b=label_b, missing=report["missing"],
            skipped=report["skipped_arms"], n_dead_treatment=report["n_dead_treatment"],
            n_dead_control=report["n_dead_control"], dead=report["dead_policy"],
            notes=report["notes"], n_perm=n_perm, seed=seed)
    if not frames:
        raise ValueError("no cell is reached by both families; nothing to compare")
    if single:
        return per_animal[animal_list[0]]

    stratified = _family_comparison(
        pd.concat(frames, ignore_index=True), family_a=label_a, family_b=label_b, missing=missing,
        skipped=missing_blocks, n_dead_treatment=n_dead_treatment, n_dead_control=n_dead_control,
        dead=(DEAD_POLICY_NONE if DEAD_POLICY_NONE in dead_policies else dead), notes=notes,
        n_perm=n_perm, seed=seed)
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
    control maximum -- the ``--family`` mode of `scripts.transfer_significance_readout`. Animals
    are never pooled into one stratum: a row is always one animal's.

    Args:
        family: A `FAMILY_BASES` name or alias; ``same-model`` reads the diagonal arms.
        animals: Animals the teachers were finetuned on, one block of rows each.
        teachers_by_animal: ``{animal: param_tags}``.
        cells: The design cells (a cells JSON path or a list).
        ftseed: The student finetune seed treatment, control and the dead label are read at.
        prompt_set: Prompt set read out of the summaries.
        probe_pool: Probe pool of the anchored dead cut; None (default) = each receiver's
            native pool (`coherence.probe_io.native_probe_pool`: pp1 Pythia, dp1 / cp1t-olmo
            DataDecide). Dead runs
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
                                                 ftseed=ftseed, probe_pool=pool, cut=cut),
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


COMPARISON_COLUMNS = ["comparison", "design", "animal", "value mode", "arms A vs B",
                      "mean contrast", "mean p", "win prob.", "win p", "exceed. frac contrast",
                      "exceed. p", "floor", "labelings", "missing blocks"]


def _comparison_rows(comparison: str, design: str, animal: str, res: FamilyComparison) -> list[dict]:
    """One (comparison, design, animal) block: one row per value mode."""
    rows = []
    for mode in VALUE_MODES:
        m = res.modes[mode]
        rows.append({
            "comparison": comparison, "design": design, "animal": animal, "value mode": mode,
            "arms A vs B": f"{m.mean_contrast.n_a} vs {m.mean_contrast.n_b}",
            "mean contrast": m.mean_contrast.observed, "mean p": m.mean_contrast.p,
            "win prob.": m.win_probability.observed, "win p": m.win_probability.p,
            "exceed. frac contrast": m.arm_level_exceed.observed,
            "exceed. p": m.arm_level_exceed.p, "floor": m.mean_contrast.floor,
            "labelings": m.mean_contrast.n_labelings,
            "missing blocks": (f"{len(res.skipped_arms)}: "
                               + ", ".join(s.split(":")[0] for s in res.skipped_arms)
                               ) if res.skipped_arms else "0",
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


def _p_styler(df: pd.DataFrame, p_columns: Sequence[str], *, index: Sequence[str],
              grey: np.ndarray, floats: Mapping[str, str] | None = None):
    """Shared styling of the two family tables: p < 0.05 bold on light green, `grey` rows shaded."""
    body = df.drop(columns=["pooled", "stratified"], errors="ignore")
    body = body.set_index(list(index))
    shade = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    fmt = {c: _fmt_p for c in p_columns}
    fmt.update({c: (lambda v, spec=spec: "" if v is None or pd.isna(v) else format(v, spec))
                for c, spec in (floats or {}).items()})
    return (body.style
            .apply(lambda _: [shade if g else "" for g in grey], axis=0)
            .map(lambda v: green if _is_significant(v) else "", subset=list(p_columns))
            .format(fmt, na_rep="")
            .set_table_styles([{"selector": "tbody th",
                                "props": "text-align: left; vertical-align: top;"}]))


def style_family_existence_table(df: pd.DataFrame):
    """Styler for `family_existence_table`: pooled rows shaded, ``perm p`` < 0.05 highlighted."""
    grey = df["pooled"].fillna(False).astype(bool).to_numpy()
    return _p_styler(df, ["perm p"], index=["family", "animal", "arm"], grey=grey,
                     floats={"cells": ".0f", "design cells": ".0f", "treatment runs": ".0f",
                             "exceed.": ".0f"})


def style_family_comparison_table(df: pd.DataFrame):
    """Styler for `family_comparison_table`: stratified rows shaded, the three p columns
    highlighted below 0.05 (the observed statistics themselves are never highlighted)."""
    grey = df["stratified"].fillna(False).astype(bool).to_numpy()
    return _p_styler(df, ["mean p", "win p", "exceed. p"],
                     index=["comparison", "design", "animal", "value mode"], grey=grey,
                     floats={"mean contrast": ".3f", "win prob.": ".3f",
                             "exceed. frac contrast": ".3f", "floor": ".1e", "labelings": ",.0f"})
