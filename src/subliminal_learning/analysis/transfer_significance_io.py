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
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from subliminal_learning.analysis.dd1b_ladder import (
    DD1B_BASE,
    DD1B_SEED2_BASE,
    DD1B_SEED3_BASE,
    dataset_dir,
)
from subliminal_learning.analysis.fsfact_transfer import RECEIVER_BASE
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import transfer_significance_tests as tt
from subliminal_learning.stats.transfer_significance_tests import topk_mean

__all__ = [
    "CellMatrix",
    "CONTROL_LABEL",
    "base_name",
    "resolve_teachers",
    "load_prior",
    "load_cell_matrix",
    "load_pool_stats",
    "drop_empty_pools",
    "StackedArms",
    "stack_cell_matrices",
    "significance_rows",
    "significance_table",
    "style_significance_table",
]

CONTROL_LABEL = "CONTROL"
DEFAULT_DATASET_TAG = "ex10-20_prefix_n100k"
ALL_ON_DISK = "all-on-disk"

#: DataDecide OLMo-1B tokens, resolved through the `analysis.dd1b_ladder` base constants.
DD1B_TOKENS = {"dd-1b": DD1B_BASE, "dd-1b-seed2": DD1B_SEED2_BASE, "dd-1b-seed3": DD1B_SEED3_BASE}


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
    ``ds<N>`` -> ``pythia-160m-data-seed<N>``, ``ws<N>`` -> ``pythia-160m-weight-seed<N>``,
    ``dd-1b``/``dd-1b-seed2``/``dd-1b-seed3`` -> the `analysis.dd1b_ladder` DataDecide OLMo-1B
    base dirs. The 160m mapping itself is `analysis.fsfact_transfer.RECEIVER_BASE`; this only adds
    the spelled-out aliases and passes an already-resolved base name (an explicit
    ``receiver_dir``/``sender_dir``) through unchanged.

    Args:
        token: A short token, an alias, or an on-disk base name.

    Returns:
        The base-model directory name.
    """
    t = token.strip()
    if t in ("original", "orig"):
        return RECEIVER_BASE["orig"]
    if t in DD1B_TOKENS:
        return DD1B_TOKENS[t]
    if t.startswith("full-seed"):
        t = f"fs{t.removeprefix('full-seed')}"
    return RECEIVER_BASE.get(t, t)


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
    elif isinstance(cells, (str, Path)):
        requested = list(json.loads(Path(cells).read_text())["cells"])
    else:
        requested = list(cells)
    # A cell with no control value is not a matched pair: its treatment values would count as
    # exceedances against a threshold they were never matched to, and the stratified null could
    # never relabel a control there. Drop it from the design.
    cells, no_control = [], []
    for cell in requested:
        (cells if _control_value(control_tree / cell, eval_animal, prompt_set, ftseed)[0]
         is not None
         else no_control).append(cell)

    def treatment_value(tag: str, cell: str) -> float | None:
        cell_dir = (root / "students" / animal / receiver_dir / sender_dir
                    / f"{tag}_tseed{tseed}" / dataset_tag / cell)
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


def load_pool_stats(
    animal: str,
    senders: Iterable[str],
    receivers: Iterable[str],
    teachers_by_sender: dict[str, Sequence[str]],
    *,
    stat: Callable[[np.ndarray], float] = topk_mean,
    **kw,
) -> dict[str, pd.DataFrame]:
    """Per-sender ``pool x receiver`` statistic frames for `stats.pool_rank_permutation`.

    Args:
        animal: Animal the teachers were finetuned on.
        senders: Sender tokens.
        receivers: Receiver tokens; one column each.
        teachers_by_sender: Teacher param_tags per sender.
        stat: Pool statistic over a cell column (default `stats.transfer_significance_tests.topk_mean`).
        **kw: Forwarded to `load_cell_matrix` (``eval_animal``, ``prompt_set``, ...).

    Returns:
        ``{sender: DataFrame}`` indexed by pool label (control row = `CONTROL_LABEL`).
    """
    out: dict[str, pd.DataFrame] = {}
    for sender in senders:
        cols: dict[str, pd.Series] = {}
        for receiver in receivers:
            matrix = load_cell_matrix(animal, sender, receiver,
                                      teachers_by_sender[sender], **kw)
            cols[receiver] = pd.Series(
                {pool: stat(matrix.values[:, j]) for j, pool in enumerate(matrix.pools)}
            )
        out[sender] = pd.DataFrame(cols)
    return out


# ----------------------------------------------------------------------------------------------
# the one-table readout driven by an arm-registry JSON
# ----------------------------------------------------------------------------------------------
TABLE_COLUMNS = ["arm", "teacher", "cells", "wins / n", "sign p", "exceed.", "exceed. p",
                 "treat median ×prior", "ctrl median ×prior", "dead (treat/ctrl)"]


def _summary_row(arm: str, teacher: str, values: np.ndarray, dead: np.ndarray, prior: float,
                 **tests) -> dict:
    """Medians and dead counts over the finite (treatment, control) pairs of a pools-then-control
    block; `tests` fills the remaining columns."""
    y, x = values[:, :-1], np.repeat(values[:, [-1]], values.shape[1] - 1, axis=1)
    ok = np.isfinite(y) & np.isfinite(x)
    ctrl = values[:, -1][np.isfinite(values[:, -1])]
    return {"arm": arm, "teacher": teacher, "cells": str(int(np.isfinite(values[:, -1]).sum())),
            "treat median ×prior": np.median(y[ok]) / prior,
            "ctrl median ×prior": np.median(ctrl) / prior,
            "dead (treat/ctrl)": f"{int(np.nansum(np.where(ok, dead[:, :-1], 0)))}"
                                 f"/{int(np.nansum(dead[:, -1]))}",
            "pooled": False, **tests}


def significance_rows(matrix: CellMatrix, dead: np.ndarray, prior: float, arm: str,
                      pooled_teachers: Sequence[str], *, short: Sequence[str] = (),
                      n_perm: int = 20000, seed: int = 0) -> list[dict]:
    """One arm's table rows: per teacher (sign + exact exceedance), then the arm-pooled row.

    Args:
        matrix: The arm's `CellMatrix` (pools = `pooled_teachers` + `short`, control last).
        dead: ``cells x pools`` dead labels matching `matrix.values` (1 dead, 0 alive, NaN unlabelled).
        prior: Receiver base prior the medians are scaled by.
        arm: Arm label.
        pooled_teachers: Full-set teachers; pooled when there are at least two.
        short: Teachers reported on their own cells only, never pooled.
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
        s, e = tt.paired_sign_test(y, x), tt.exceedance_test(y, x)
        name = tag if tag in pooled_teachers else f"{tag} ({int(ok.sum())}/{len(matrix.cells)} cells, not in pool)"
        cols = [j, matrix.control_col]
        rows.append(_summary_row(arm, name, matrix.values[ok][:, cols], dead[ok][:, cols], prior,
                                 **{"wins / n": f"{s.wins}/{s.n}", "sign p": s.p,
                                    "exceed.": e.count, "exceed. p": e.p}))
    if len(pooled_teachers) >= 2:
        cols = [matrix.pools.index(t) for t in pooled_teachers] + [matrix.control_col]
        r = tt.stratified_exceedance_permutation(matrix.values[:, cols], len(cols) - 1,
                                                 n_perm=n_perm, seed=seed)
        row = _summary_row(arm, "all full-set teachers", matrix.values[:, cols], dead[:, cols],
                           prior, **{"exceed.": r.observed, "exceed. p": r.p})
        rows.append({**row, "pooled": True})
    return rows


def _has_seed_endpoint(leaf: Path, ftseed: int, prompt_set: str) -> bool:
    path = leaf / f"ftseed{ftseed}" / "preference_eval.json"
    return path.is_file() and prompt_set in json.loads(path.read_text()).get("sets", {})


def _dead_label(leaf: Path, ftseed: int, pool: str, cut: float) -> float:
    path = leaf / f"ftseed{ftseed}" / "eval" / "pile_probe.json"
    if not path.is_file():
        return np.nan
    doc = json.loads(path.read_text())
    if doc.get("pool_version") != pool:
        raise ValueError(f"{path}: pool_version {doc.get('pool_version')} != {pool}")
    return float(doc["delta_ce"] > cut)


def significance_table(arms_json: str | Path, output_root: Path | None = None) -> pd.DataFrame:
    """The single transfer-vs-matched-control table for the arms of a registry JSON.

    Per arm: design cells (a cells JSON, another arm's cells, or the control tree's cells with
    an ``ftseed`` control value); every listed teacher must cover all of them (``known_short``
    teachers must miss exactly the listed cells; ``excluded`` must equal the on-disk strays);
    the receiver's prior is the one ``base_control.json`` shared by its student trees (checked
    against ``expected_prior`` within 5%); dead = probe ``delta_ce`` above the receiver's anchored
    cut. ``in_progress`` arms become a coverage row until complete. Then one row per ``pools``
    entry (arm-stratified permutation, each arm against its own control maximum; arms must share
    animal and ``data_order``), and one row per ``held_out`` entry (two-sided sign test on the
    cells outside ``selection_cells`` where both runs are alive).

    Args:
        arms_json: Registry path (format: ``run_logs/xarch/xarch_significance_arms.json``).
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        `TABLE_COLUMNS` plus a boolean ``pooled`` column, in registry order.
    """
    from scipy.stats import binomtest

    from subliminal_learning.analysis import labels

    arms_json = Path(arms_json)
    reg = json.loads(arms_json.read_text())
    root = resolve_output_root(output_root)
    ftseed, prompt_set = reg.get("ftseed", 1), reg.get("prompt_set", "test")
    n_perm, seed = reg.get("n_perm", 20000), reg.get("seed", 0)
    arms = {a["name"]: a for a in reg["arms"]}
    matrices, deads, priors, rows = {}, {}, {}, []

    def design(a: dict) -> list[str]:
        if "cells_from_arm" in a:
            return design(arms[a["cells_from_arm"]])
        if "cells" in a:
            return sorted(json.loads((arms_json.parent / a["cells"]).read_text())["cells"])
        return load_cell_matrix(a["animal"], a["sender"], a["receiver"], [], ftseed=ftseed,
                                tseed=a["tseed"], dataset_tag=a.get("dataset_tag", DEFAULT_DATASET_TAG),
                                control_dataset_tag=a.get("control_dataset_tag"),
                                prompt_set=prompt_set, output_root=root).cells

    for name, a in arms.items():
        dataset_tag = a.get("dataset_tag", DEFAULT_DATASET_TAG)
        cells = design(a)
        teachers, short = list(a["teachers"]), dict(a.get("known_short", {}))
        _, stray = resolve_teachers([*teachers, *short], animal=a["animal"], sender=a["sender"],
                                    receiver=a["receiver"], output_root=root,
                                    dataset_tag=dataset_tag, tseed=a["tseed"])
        if sorted(stray) != sorted(a.get("excluded", [])):
            raise ValueError(f"{name}: on-disk strays {stray} != excluded {a.get('excluded', [])}")
        sender_tree = (root / "students" / a["animal"] / base_name(a["receiver"])
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
            elif missing and not a.get("in_progress"):
                raise ValueError(f"{name}: listed teacher {tag} is short "
                                 f"{covered[tag]}/{len(cells)}; missing {missing}")
        if ctrl_cov < len(cells) or any(covered[t] < len(cells) for t in teachers):
            if not a.get("in_progress"):
                raise ValueError(f"{name}: control covers {ctrl_cov}/{len(cells)}")
            k = min(min(covered[t] for t in teachers), ctrl_cov)
            rows.append({"arm": name, "teacher": ", ".join(teachers),
                         "cells": f"in progress {k}/{len(cells)}", "pooled": False})
            continue

        receiver = base_name(a["receiver"])
        key = (a["animal"], receiver)
        if key not in priors:
            files = sorted((root / "students" / a["animal"] / receiver)
                           .glob("teacher-*/*/*/base_control.json"))
            vals = {round(json.loads(f.read_text())[prompt_set]["p_target"], 10) for f in files}
            if len(vals) != 1:
                raise ValueError(f"{key}: student trees disagree on the prior: {vals}")
            priors[key] = vals.pop()
        exp = a.get("expected_prior")
        if exp is not None and abs(priors[key] / exp - 1) >= 0.05:
            raise ValueError(f"{name}: prior {priors[key]} vs expected {exp}")
        probe_pool = a.get("probe_pool", "pp1")
        cut = labels.anchored_cut(root, receiver, pool=probe_pool)

        m = load_cell_matrix(a["animal"], a["sender"], a["receiver"], [*teachers, *short],
                             ftseed=ftseed, tseed=a["tseed"], dataset_tag=dataset_tag,
                             control_dataset_tag=a.get("control_dataset_tag"), cells=cells,
                             prompt_set=prompt_set, output_root=root)
        if m.cells != cells:
            raise ValueError(f"{name}: loaded cells differ from the design")
        d = np.full(m.values.shape, np.nan)
        for i, c in enumerate(m.cells):
            for j, pool in enumerate(m.pools):
                leaf = (control_tree / c if pool == CONTROL_LABEL
                        else sender_tree / f"{pool}_tseed{a['tseed']}" / dataset_tag / c)
                d[i, j] = _dead_label(leaf, ftseed, probe_pool, cut)
        matrices[name], deads[name] = m, d
        rows.extend(significance_rows(m, d, priors[key], name, teachers, short=list(short),
                                      n_perm=n_perm, seed=seed))

    for pool in reg.get("pools", []):
        members = [arms[n] for n in pool["arms"]]
        if len({(a["animal"], a["data_order"]) for a in members}) != 1:
            raise ValueError(f"{pool['name']}: arms mix animals or data-order classes")
        if not all(n in matrices for n in pool["arms"]):
            rows.append({"arm": pool["name"], "teacher": "all full-set teachers",
                         "cells": "in progress", "pooled": True})
            continue
        mats = {}
        for n in pool["arms"]:
            m = matrices[n]
            keep = [m.pools.index(t) for t in arms[n]["teachers"]] + [m.control_col]
            mats[n] = CellMatrix(frame=m.frame, values=m.values[:, keep], cells=m.cells,
                                 pools=[m.pools[j] for j in keep], control_col=len(keep) - 1,
                                 missing=[])
        st = stack_cell_matrices(mats)
        r = tt.stratified_exceedance_permutation(st.values, st.control_index, groups=st.groups,
                                                 n_perm=n_perm, seed=seed)
        rows.append({"arm": pool["name"], "teacher": "all full-set teachers",
                     "cells": str(sum(len(m.cells) for m in mats.values())),
                     "exceed.": r.observed, "exceed. p": r.p, "pooled": True})

    for h in reg.get("held_out", []):
        a, m, d = arms[h["arm"]], matrices[h["arm"]], deads[h["arm"]]
        j = m.pools.index(h["teacher"])
        held = np.array([c not in h["selection_cells"] for c in m.cells])
        if held.sum() != h["n_held_out"]:
            raise ValueError(f"{h['name']}: {held.sum()} held-out cells, expected {h['n_held_out']}")
        alive = held & (d[:, j] == 0) & (d[:, m.control_col] == 0)
        cols = [j, m.control_col]
        s = tt.paired_sign_test(m.values[alive, j], m.values[alive, m.control_col])
        rows.append(_summary_row(h["name"], h["teacher"], m.values[alive][:, cols], d[alive][:, cols],
                                 priors[(a["animal"], base_name(a["receiver"]))],
                                 **{"wins / n": f"{s.wins}/{s.n}",
                                    "sign p": binomtest(s.wins, s.n, 0.5).pvalue,
                                    "exceed.": "—", "exceed. p": "—"}))
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
    """Styler for `significance_table`: p to 2 s.f., ×prior to 2 dp, p < 0.05 bold on light green,
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

    return (show.style.apply(row_style, axis=1)
            .map(lambda v: green if _is_significant(v) else "", subset=["sign p", "exceed. p"])
            .format({"sign p": _fmt_p, "exceed. p": _fmt_p, "exceed.": count,
                     "treat median ×prior": ratio, "ctrl median ×prior": ratio}, na_rep="")
            .apply_index(arm_style, axis="index", level=0)
            .apply_index(teacher_style, axis="index", level=1)
            .set_table_styles([{"selector": "tbody th",
                                "props": "text-align: left; vertical-align: top;"}])
            .hide(axis="index", level=2))
