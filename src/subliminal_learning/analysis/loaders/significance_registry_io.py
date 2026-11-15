"""Transfer-significance registry readout: the one-table readout driven by an arm-registry JSON.

Split (2026-11) from the former single ``transfer_significance_io`` module (see
`significance_cells_io` for the cell layer it builds on).
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
    CONTROL_LABEL,
    CellMatrix,
    DEAD_UNLABELLED,
    DEFAULT_DATASET_TAG,
    DIVERGED_MARKER,
    _PROBE_LEAVES,
    _control_value,
    _leaf_signature,
    _loader_memo,
    _rel,
    _run_leaves,
    base_name,
    load_cell_matrix,
    load_prior,
    logger,
    resolve_cells,
    resolve_teachers,
    stack_cell_matrices,
)

# ----------------------------------------------------------------------------------------------
# the one-table readout driven by an arm-registry JSON
# ----------------------------------------------------------------------------------------------
TABLE_COLUMNS = ["arm", "teacher", "test", "prior", "cells", "wins / n", "sign p", "exceed.",
                 "perm p", "treat median ×prior", "treat max ×prior", "ctrl median ×prior",
                 "ctrl max ×prior", "dead (treat/ctrl)", "status"]

#: Every ``status`` of an arm lacking data starts with this; a tested arm's status is "".
STATUS_FLAG = "⚠"


def _stray_status(strays: Sequence[str]) -> str:
    """The ``status`` note of a TESTED arm with on-disk teachers it does not list: they are
    excluded from the arm and named here. Unflagged -- a stray does not untest an arm."""
    return f"strays excluded: {', '.join(strays)}" if strays else ""

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
    """One arm's table rows: the arm-pooled stratified-permutation row FIRST, then per teacher
    (sign test + descriptive exceedance count), then the `short` teachers.

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
        else:
            row = _summary_row(arm, "all full-set teachers", pooled.values,
                               dead[pooled.keep][:, pooled.cols], prior,
                               _perm_label(pooled.values.shape[1] - 1), labelled=labelled,
                               **{"exceed.": pooled.result.observed, "perm p": pooled.result.p})
            rows.append({**row, "pooled": True})
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
    place the pool choice lives -- so a DataDecide receiver gets its own recipe's pool
    without any caller spelling it. A receiver dir the naming module cannot read keeps pp1, the historical
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
                     cut: float | None, root: Path | None = None,
                     teacher_dataset_tags: Mapping[str, str] | None = None) -> np.ndarray:
    """``cells x pools`` dead labels aligned to `matrix.values`, control column last.

    All NaN when `cut` is None: with no anchored cut nothing is labelled (see `_anchored_cut`).
    Cached (`stats.memo`, ``arm_dead_matrix/``) on the trees, the design, the cut and the
    `_leaf_signature` of every run's probe record and manifest; `root` (the output root the
    trees live under) makes the cached paths relative, i.e. shared across nodes.
    """
    if cut is None:
        return np.full(matrix.values.shape, np.nan)

    tags = teacher_dataset_tags or {}

    def run_dir(cell: str, pool: str) -> Path:
        return (control_tree / cell if pool == CONTROL_LABEL
                else sender_tree / f"{pool}_tseed{tseed}" / tags.get(pool, dataset_tag)
                / cell) / f"ftseed{ftseed}"

    def compute() -> np.ndarray:
        d = np.full(matrix.values.shape, np.nan)
        for i, cell in enumerate(matrix.cells):
            for j, pool in enumerate(matrix.pools):
                d[i, j] = _dead_label(run_dir(cell, pool).parent, ftseed, probe_pool, cut)
        return d

    if memo.cache_dir() is None:
        return compute()
    root = resolve_output_root(root)
    # Cached on the probe record + manifest of every run, the trees, the cut and the pool.
    leaves = [leaf for cell in matrix.cells for pool in matrix.pools
              for leaf in _run_leaves(run_dir(cell, pool), _PROBE_LEAVES, probe_pool=probe_pool)]
    parts = [_rel(root, control_tree), _rel(root, sender_tree), list(matrix.cells), list(matrix.pools),
             dataset_tag, int(tseed), int(ftseed), probe_pool, float(cut),
             _leaf_signature(root, leaves)]
    if tags:                                       # keep the plain arms' cache keys unchanged
        parts.append(sorted(tags.items()))
    return _loader_memo("arm_dead_matrix", parts, compute,
                        encode=lambda d: {"dead": d.tolist()},
                        decode=lambda **d: np.asarray(d["dead"], dtype=float).reshape(
                            matrix.values.shape))


def student_tree_prior(animal: str, receiver: str, *, prompt_set: str = "test",
                       output_root: Path | None = None) -> float:
    """The receiver's untuned prior (the registry arms' prior) -- the same number `load_prior`
    serves.

    The receiver's ``analysis/base_eval_{short}_animal_preference_test.json`` test prior, one
    prior per (base, animal) for every consumer; the per-tree ``base_control.json`` the student
    workers write is no longer read (the trees' priors are older, set-dependent numbers, and
    only the canonical test prior is relative-statistic material). Missing artifact -> ``nan``
    with the remedy logged.

    Args:
        animal: Trait animal.
        receiver: Receiver token or base name.
        prompt_set: Prompt set the prior is meant to match; anything but ``test`` goes through
            `load_prior`'s base-control path.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        The prior, or ``nan``.
    """
    return load_prior(animal, receiver, prompt_set, output_root=output_root)


def check_expected_prior(name: str, prior: float, expected: float | None) -> None:
    """Raise unless `prior` is within 5% of the registry's ``expected_prior`` (``None`` = no check)."""
    if expected is not None and abs(prior / expected - 1) >= 0.05:
        raise ValueError(f"{name}: prior {prior} vs expected {expected}")


def registry_design_cells(arm: dict, arms: Mapping[str, dict], arms_json: Path, *,
                          ftseed: int | None, prompt_set: str = "test",
                          output_root: Path | None = None) -> list[str]:
    """The design cells of one registry arm: another arm's (``cells_from_arm``), a cells JSON
    relative to the registry or an inline list of names (``cells``), else the control tree's
    cells with a control value -- minus the arm's ``exclude_cells`` (the cells a pilot-selected
    teacher set was chosen on, so the arm is read on held-out cells only).

    Raises:
        ValueError: If ``exclude_cells`` is not a subset of the design.
    """
    if "cells_from_arm" in arm:
        cells = registry_design_cells(arms[arm["cells_from_arm"]], arms, arms_json, ftseed=ftseed,
                                      prompt_set=prompt_set, output_root=output_root)
    elif "cells" in arm:
        spec = arm["cells"]
        cells = sorted(resolve_cells(Path(arms_json).parent / spec if isinstance(spec, str)
                                     else spec))
    else:
        cells = load_cell_matrix(arm["animal"], arm["sender"], arm["receiver"], [], ftseed=ftseed,
                                 tseed=arm["tseed"],
                                 dataset_tag=arm.get("dataset_tag", DEFAULT_DATASET_TAG),
                                 control_dataset_tag=arm.get("control_dataset_tag"),
                                 prompt_set=prompt_set, output_root=output_root).cells
    exclude = list(arm.get("exclude_cells", []))
    if not set(exclude) <= set(cells):
        raise ValueError(f"{arm.get('name')}: exclude_cells {sorted(set(exclude) - set(cells))} "
                         f"are not design cells")
    return [c for c in cells if c not in exclude]


def registry_arm_names(arms_json: str | Path, *, data_order: str | None = None) -> list[str]:
    """The registry's arm names in registry order, optionally only those of one ``data_order``
    class (e.g. ``"cross-recipe"`` vs ``"cross-recipe+seed"``), so a notebook can name a
    relationship class once and pick up arms added to the registry later.

    Args:
        arms_json: Registry path.
        data_order: Keep only arms carrying this ``data_order``; ``None`` keeps every arm.

    Returns:
        The arm names, in the order the registry lists them.
    """
    reg = json.loads(Path(arms_json).read_text())
    return [a["name"] for a in reg["arms"]
            if data_order is None or a.get("data_order") == data_order]


def registry_selection_notes(arms_json: str | Path) -> str:
    """Markdown bullets of every registry arm carrying a ``selection_note`` (how its teacher
    set was chosen), for the notebook text; "" when no arm has one."""
    reg = json.loads(Path(arms_json).read_text())
    return "\n".join(f"- **{a['name']}** — {a['selection_note']}" for a in reg["arms"]
                      if a.get("selection_note"))


@dataclass(frozen=True)
class RegistryArm:
    """One registry arm as `significance_table` reads it: its design, coverage and, when the
    set is finished, its matched `CellMatrix`.

    Attributes:
        name: The arm's registry name.
        spec: The registry entry.
        cells: Design cells (`registry_design_cells`).
        teachers: Listed full-set teachers.
        short: ``known_short`` teachers (reported, never pooled).
        receiver: Receiver base name.
        prior: The receiver's test prior (`student_tree_prior`).
        ctrl_cov: Design cells with a matched control.
        matched: Design cells where every listed teacher and the control are present.
        counts: The build-time counts a ``note`` / ``untestable`` text may cite
            (``students``, ``n_students``, ``controls``, ``cells``).
        finished: Every listed teacher on every design cell AND a control on every cell.
        matrix: The loaded `CellMatrix` (pools = teachers + short, control last) on a finished,
            testable arm; ``None`` on a coverage row (unfinished or ``untestable``).
        strays: On-disk teacher tags of the sender tree that the arm lists nowhere
            (neither ``teachers`` nor ``known_short``); auto-excluded, reported.
        sender_tree, control_tree, dataset_tag: The trees the dead labels are read from.
        diverged: ``(teacher, cell)`` runs excluded as diverged (``diverged="exclude"`` only).
    """

    name: str
    spec: dict
    cells: list[str]
    teachers: list[str]
    short: dict
    receiver: str
    prior: float
    ctrl_cov: int
    matched: int
    counts: dict
    finished: bool
    matrix: CellMatrix | None
    strays: list[str]
    sender_tree: Path
    control_tree: Path
    dataset_tag: str
    diverged: list[tuple[str, str]] = field(default_factory=list)


def load_registry_arm(a: dict, arms: Mapping[str, dict], arms_json: str | Path, *,
                      root: Path | None = None, ftseed: int = 1, prompt_set: str = "test",
                      priors: dict | None = None, diverged: str = "drop") -> RegistryArm:
    """Read ONE registry arm the way `significance_table` does (the coverage rule included).

    Every listed teacher must cover the whole design (``known_short`` teachers must miss
    exactly the listed cells, and the optional ``teacher_dataset_tags`` map overrides the
    numbers-dataset dir of the teachers it names, control tree excepted); on-disk teachers the arm lists nowhere are strays, auto-excluded
    from the arm and reported on `RegistryArm.strays` -- the optional ``excluded`` key only
    declares the strays expected on disk and is never checked. An arm short of a
    finished set must be flagged ``in_progress`` or ``untestable`` (else it is an error), and
    its `RegistryArm.matrix` is ``None``. The retired ``partial`` key is an error.

    Args:
        a: The registry entry.
        arms: ``{name: entry}`` of the whole registry (for ``cells_from_arm``).
        arms_json: Registry path (relative cell-set paths resolve against its directory).
        root: Output tree root; defaults to the paths YAML's.
        ftseed, prompt_set: The registry's student seed and prompt set.
        priors: Optional ``{(animal, receiver): prior}`` cache shared across arms.
        diverged: ``"drop"`` (default) counts a diverged student as a run that is not there, so
            the arm is short of a finished set; ``"exclude"`` counts it as COMPLETE, keeps the
            arm finished, leaves its NaN in the matrix (no test sees it) and reports it on
            `RegistryArm.diverged`.

    Returns:
        A `RegistryArm`.
    """
    arms_json, root = Path(arms_json), resolve_output_root(root)
    name = a["name"]
    if a.get("partial"):
        raise ValueError(f"{name}: `partial` arms are no longer supported -- an arm is "
                         f"tested on its whole design or reported as a coverage row "
                         f"(in_progress / untestable)")
    dataset_tag = a.get("dataset_tag", DEFAULT_DATASET_TAG)
    # `teacher_dataset_tags` overrides the numbers-dataset dir for the named teachers only (a
    # sender whose pool was generated at another size); the control tree keeps `dataset_tag`.
    teacher_tags = dict(a.get("teacher_dataset_tags", {}))
    cells = registry_design_cells(a, arms, arms_json, ftseed=ftseed, prompt_set=prompt_set,
                                  output_root=root)
    teachers, short = list(a["teachers"]), dict(a.get("known_short", {}))
    stray_set: set[str] = set()
    for dtag in {dataset_tag, *teacher_tags.values()}:
        _, stray = resolve_teachers([*teachers, *short], animal=a["animal"], sender=a["sender"],
                                    receiver=a["receiver"], output_root=root,
                                    dataset_tag=dtag, tseed=a["tseed"])
        stray_set |= set(stray)
    stray = sorted(stray_set)
    sender_tree = (root / "students" / a["animal"]
                   / Path(*model_dir_parts_from_name(base_name(a["receiver"])))
                   / f"teacher-{base_name(a['sender'])}")
    control_tree = dataset_dir(root, None, base=base_name(a["receiver"]),
                               dataset_tag=a.get("control_dataset_tag") or dataset_tag,
                               source="base", numbers_base=base_name(a["sender"]))
    ctrl_cov = sum(_control_value(control_tree / c, a["animal"], prompt_set, ftseed)[0]
                   is not None for c in cells)
    covered, diverged_runs = {}, []

    def endpoint(tag: str, cell: str) -> bool:
        """Does ``(tag, cell)`` count as a finished run? A diverged one does under
        ``diverged="exclude"``: it ran, it just has no eval to read."""
        leaf = sender_tree / f"{tag}_tseed{a['tseed']}" / teacher_tags.get(tag, dataset_tag) / cell
        if _has_seed_endpoint(leaf, ftseed, prompt_set):
            return True
        return (diverged == "exclude"
                and (leaf / f"ftseed{ftseed}" / DIVERGED_MARKER).is_file())

    for tag in [*teachers, *short]:
        leaf = sender_tree / f"{tag}_tseed{a['tseed']}" / teacher_tags.get(tag, dataset_tag)
        missing = [c for c in cells if not endpoint(tag, c)]
        if tag in teachers:
            diverged_runs += [(tag, c) for c in cells if c not in missing
                              and not _has_seed_endpoint(leaf / c, ftseed, prompt_set)]
        covered[tag] = len(cells) - len(missing)
        if tag in short:
            if missing != short[tag]:
                raise ValueError(f"{name}: {tag} missing {missing}, registry says {short[tag]}")
        elif missing and not (a.get("in_progress") or a.get("untestable")):
            raise ValueError(f"{name}: listed teacher {tag} is short "
                             f"{covered[tag]}/{len(cells)}; missing {missing}")
    # matched cells: every listed teacher's endpoint AND a control, cell by cell
    matched = sum(
        _control_value(control_tree / c, a["animal"], prompt_set, ftseed)[0] is not None
        and all(endpoint(t, c) for t in teachers)
        for c in cells)
    finished = ctrl_cov == len(cells) and all(covered[t] == len(cells) for t in teachers)
    if not finished and not (a.get("in_progress") or a.get("untestable")):
        raise ValueError(f"{name}: not a finished set (control covers {ctrl_cov}/"
                         f"{len(cells)}, matched cells {matched}/{len(cells)}); flag it "
                         f"in_progress or untestable")

    receiver = base_name(a["receiver"])
    key = (a["animal"], receiver)
    priors = priors if priors is not None else {}
    if key not in priors:
        try:
            priors[key] = student_tree_prior(a["animal"], receiver, prompt_set=prompt_set,
                                             output_root=root)
        except (ValueError, KeyError):
            # a receiver with no student tree yet and no base-eval prior file (the OLMo
            # large-aux-2 recipes): a coverage row can still be reported, a tested arm cannot
            if a.get("expected_prior") is not None or not (a.get("in_progress")
                                                           or a.get("untestable")):
                raise
            priors[key] = float("nan")
    check_expected_prior(name, priors[key], a.get("expected_prior"))
    counts = {"students": sum(covered[t] for t in teachers),
              "n_students": len(teachers) * len(cells), "controls": ctrl_cov,
              "cells": len(cells)}
    matrix = None
    if finished and not a.get("untestable"):
        matrix = load_cell_matrix(a["animal"], a["sender"], a["receiver"], [*teachers, *short],
                                  ftseed=ftseed, tseed=a["tseed"], dataset_tag=dataset_tag,
                                  teacher_dataset_tags=teacher_tags,
                                  control_dataset_tag=a.get("control_dataset_tag"), cells=cells,
                                  prompt_set=prompt_set, output_root=root)
        if matrix.cells != cells:
            raise ValueError(f"{name}: loaded cells differ from the design")
    return RegistryArm(name=name, spec=a, cells=cells, teachers=teachers, short=short,
                       receiver=receiver, prior=priors[key], ctrl_cov=ctrl_cov,
                       matched=matched, counts=counts, finished=finished, matrix=matrix,
                       strays=stray, sender_tree=sender_tree, control_tree=control_tree,
                       dataset_tag=dataset_tag, diverged=sorted(diverged_runs))


def significance_table(arms_json: str | Path, output_root: Path | None = None) -> pd.DataFrame:
    """The single transfer-vs-matched-control table for the arms of a registry JSON.

    Per arm: design cells (a cells JSON, another arm's cells, or the control tree's cells with
    an ``ftseed`` control value); every listed teacher must cover all of them (``known_short``
    teachers must miss exactly the listed cells; on-disk teachers the arm lists nowhere are
    strays, auto-excluded and named in the ``status`` column);
    the receiver's prior is the one ``base_control.json`` shared by its student trees, or the
    receiver's ``analysis/base_eval_*_test.json`` prior when the trees carry none (checked
    against ``expected_prior`` within 5%); dead = probe ``delta_ce`` above the receiver's anchored
    cut. Only a FINISHED set gets a p: every listed teacher with an ftseed endpoint on every
    design cell and a matched control on every design cell. An arm short of that must be
    flagged ``in_progress`` (else it is an error) and becomes ONE coverage row: no p, the
    ``test`` column carrying the arm's ``note`` (``{students}``/``{n_students}``/``{controls}``/
    ``{cells}`` filled with the build-time counts), ``cells`` reading ``k/N`` (design cells
    where every teacher and the control are present), and a ``status`` starting with
    `STATUS_FLAG`. An arm keyed ``untestable: "<reason>"`` is ALWAYS such a coverage row, even
    when complete (no-control arms, single-cell pilots). No arm is ever tested on the subset of
    cells that happened to finish: the former ``partial`` key is an error. ``exclude_cells``
    removes the selection cells of a pilot-chosen teacher set from the design before anything
    is loaded (`registry_design_cells`). Then one row per ``pools``
    entry (arm-stratified permutation, each arm against its own control maximum; arms must share
    animal and ``data_order``), and one row per ``held_out`` entry (one-sided sign test on the
    cells outside ``selection_cells`` where both runs are alive); a pool or held-out entry whose
    arm carries no p is itself a flagged coverage row.

    Args:
        arms_json: Registry path (format: ``run_logs/campaigns/xarch/xarch_significance_arms.json``).
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        `TABLE_COLUMNS` plus a boolean ``pooled`` column, in registry order; ``status`` is ""
        on every row of a tested arm without strays, and the unflagged ``strays excluded: ...``
        note on a tested arm that has them.
    """
    arms_json = Path(arms_json)
    reg = json.loads(arms_json.read_text())
    root = resolve_output_root(output_root)
    ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
    n_perm, seed = reg.get("n_perm", 20000), reg.get("seed", 0)
    arms = {a["name"]: a for a in reg["arms"]}
    matrices, deads, priors, rows = {}, {}, {}, []
    notes: list[str] = []

    def coverage_row(name: str, teachers: list[str], test: str, status: str, cells: str,
                     prior: float = float("nan")) -> dict:
        label = (", ".join(teachers) if len(teachers) <= 5
                 else f"{len(teachers)} teachers: {teachers[0]}, {teachers[1]}, …")
        return {**{c: float("nan") for c in TABLE_COLUMNS}, "arm": name, "teacher": label,
                "test": test, "prior": prior, "cells": cells, "wins / n": TEST_NONE,
                "dead (treat/ctrl)": TEST_NONE, "status": f"{STATUS_FLAG} {status}",
                "pooled": False}

    for name, a in arms.items():
        ra = load_registry_arm(a, arms, arms_json, root=root, ftseed=ftseed,
                               prompt_set=prompt_set, priors=priors)
        teachers, short, cells = ra.teachers, ra.short, ra.cells
        if not ra.finished or a.get("untestable"):
            # ONE coverage row, no p. The registry's ``note`` / ``untestable`` text is the
            # reason (test column); it may cite the build-time counts through
            # {students}/{n_students}/{controls}/{cells}. The status names what is lacking.
            counts = ra.counts
            if a.get("untestable"):
                reason = str(a["untestable"]).format_map(counts)
                status = reason
            else:
                reason = a["note"].format_map(counts) if "note" in a else TEST_NONE
                status = ("not started" if counts["students"] == 0 else
                          f"in progress {counts['students']}/{counts['n_students']} students")
                if ra.ctrl_cov < len(cells):
                    status += f", {ra.ctrl_cov}/{len(cells)} controls"
            rows.append(coverage_row(name, teachers, reason, status,
                                     f"{ra.matched}/{len(cells)}", ra.prior))
            continue
        probe_pool = _probe_pool_for(ra.receiver, a.get("probe_pool"))
        cut = _anchored_cut(root, ra.receiver, probe_pool, dead="keep", notes=notes)

        m = ra.matrix
        d = _arm_dead_matrix(m, ra.control_tree, ra.sender_tree, dataset_tag=ra.dataset_tag,
                             teacher_dataset_tags=ra.spec.get("teacher_dataset_tags"),
                             tseed=a["tseed"], ftseed=ftseed, probe_pool=probe_pool, cut=cut,
                             root=root)
        matrices[name], deads[name] = m, d
        rows.extend({**r, "status": _stray_status(ra.strays)} for r in
                    significance_rows(m, d, ra.prior, name, teachers, short=list(short),
                                      labelled=cut is not None, n_perm=n_perm, seed=seed))

    for pool in reg.get("pools", []):
        members = [arms[n] for n in pool["arms"]]
        if len({(a["animal"], a["data_order"]) for a in members}) != 1:
            raise ValueError(f"{pool['name']}: arms mix animals or data-order classes")
        if not all(n in matrices for n in pool["arms"]):
            lacking = [n for n in pool["arms"] if n not in matrices]
            rows.append({**coverage_row(pool["name"], ["all full-set teachers"], TEST_NONE,
                                        f"member arm without a p: {'; '.join(lacking)}",
                                        TEST_NONE), "pooled": True})
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
                     "exceed.": r.observed, "perm p": r.p, "status": "", "pooled": True})

    for h in reg.get("held_out", []):
        a = arms[h["arm"]]
        if h["arm"] not in matrices:
            rows.append(coverage_row(h["name"], [h["teacher"]], TEST_NONE,
                                     f"arm without a p: {h['arm']}", TEST_NONE))
            continue
        rows.append({**held_out_row(h, matrices[h["arm"]], deads[h["arm"]],
                                    priors[(a["animal"], base_name(a["receiver"]))]),
                     "status": ""})
    return pd.DataFrame(rows).reindex(columns=[*TABLE_COLUMNS, "pooled"])


#: Shading of every row of an arm lacking data (flagged ``status`` / skipped family arm).
AMBER = "background-color: rgba(255, 180, 0, 0.25)"


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
    pooled rows on a translucent grey (readable on light and dark themes), and EVERY row of an
    arm lacking data (a non-empty ``status``), index cells included, on `AMBER`. Rows are indexed by
    (arm, teacher) so each run of consecutive same-arm rows -- the arm block, pooled row first,
    then its teachers -- shows its arm once (row-span cell), with a 2px divider above the first
    row of every block; the positional level is hidden."""
    pooled = df["pooled"].fillna(False).astype(bool).to_numpy()
    arms = df["arm"].tolist()
    first = np.array([i == 0 or arms[i] != arms[i - 1] for i in range(len(arms))], dtype=bool)
    flagged = df["status"].fillna("").astype(str).str.startswith(STATUS_FLAG).to_numpy() \
        if "status" in df else np.zeros(len(df), dtype=bool)
    show = df.drop(columns=["pooled", "arm", "teacher"])
    show.index = pd.MultiIndex.from_arrays([df["arm"].to_numpy(), df["teacher"].to_numpy(),
                                            np.arange(len(df))], names=["arm", "teacher", None])
    grey = "background-color: rgba(128, 128, 128, 0.2)"
    green = "font-weight: bold; background-color: rgba(120, 200, 120, 0.35)"
    divider = "border-top: 2px solid currentColor"

    def row_css(i):
        shade = AMBER if flagged[i] else grey if pooled[i] else ""
        return "; ".join(c for c in (shade, divider if first[i] else "") if c)

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
