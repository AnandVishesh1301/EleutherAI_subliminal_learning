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
from typing import Callable, Iterable, Sequence

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


def _control_value(cell_dir: Path, eval_animal: str, prompt_set: str) -> tuple[float | None, str]:
    """The control cell's value for `eval_animal`, and which summary layout it came from.

    Control trees are animal-independent, so most workers namespace the scores under
    ``eval/{animal}/``; older runs (and the dd1b worker) wrote a single top-level
    ``preference_summary.json``, which counts only when its ``target_animal`` is `eval_animal`.

    Returns:
        ``(value, layout)`` with layout ``"eval"``/``"top-level"``, or ``(None, "")`` when the cell
        holds no usable control summary for this animal and prompt set.
    """
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
    tseed: int = 1,
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
        dataset_tag: Numbers-dataset dir.
        tseed: Teacher finetune seed (the DataDecide OLMo arms use 0).
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
    control_tree = dataset_dir(root, None, base=receiver_dir, dataset_tag=dataset_tag,
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
        (cells if _control_value(control_tree / cell, eval_animal, prompt_set)[0] is not None
         else no_control).append(cell)

    def treatment_path(tag: str, cell: str) -> Path:
        cell_dir = (root / "students" / animal / receiver_dir / sender_dir
                    / f"{tag}_tseed{tseed}" / dataset_tag / cell)
        return (cell_dir / "preference_summary.json" if eval_animal == animal
                else cell_dir / "eval" / eval_animal / "preference_summary.json")

    pools = [*teachers, CONTROL_LABEL]
    rows, missing, layout = [], [], {}
    values = np.full((len(cells), len(pools)), np.nan)
    for j, pool in enumerate(pools):
        is_control = pool == CONTROL_LABEL
        for i, cell in enumerate(cells):
            if is_control:
                value, kind = _control_value(control_tree / cell, eval_animal, prompt_set)
                if kind:
                    layout[cell] = kind
            else:
                value = _p_target(treatment_path(pool, cell), prompt_set)
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
