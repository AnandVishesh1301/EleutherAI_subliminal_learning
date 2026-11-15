"""Disk loaders feeding `stats.transfer_significance_tests`: treatment cells, matched controls, priors.

Split (2026-11) from the former single ``transfer_significance_io`` module; this part is the
cell layer -- `CellMatrix`, teacher/receiver resolution, the memoised cell-matrix loader and
the test priors. The registry readout lives in `significance_registry_io`, the family frames
and comparisons in `significance_frames_io`; `transfer_significance_io` re-exports all three.

Original module docstring:
Disk loaders feeding `stats.transfer_significance_tests`: treatment cells, matched controls, priors.

One sender's numbers pool per teacher, one matched base-pool control tree, the same student
cells on both sides — that matched design is what the tests in
``subliminal_learning.stats.transfer_significance_tests`` assume, and assembling it is all this module does.
Analysis layer: it reads JSON off the output tree (resolved through
`subliminal_learning.config.resolve_output_root`, never a hardcoded mount) and returns frames
and arrays; every estimator stays in ``stats``.

The per-arm loaders (`load_cell_matrix` with pinned cells, `_arm_dead_matrix`, and
`diagonal_direction.load_arm_runs`) are memoized through `stats.memo` under
``{output_root}/analysis/cache/transfer_significance/``: the key is every argument plus the
size/mtime signature of the run files they would read (`_leaf_signature`, one pooled ``stat``
per file -- the tree is a latency-bound network FS, so that is ~20x cheaper than the reads),
so a run that finishes, a control that appears or a rewritten eval is a new key. The same memo
holds the seeded permutation results (exact, keyed on the value arrays). ``SUBLIMINAL_ANALYSIS_CACHE=0``
bypasses it; `stats.memo.cache_stats` counts hits and misses.
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


CONTROL_LABEL = "CONTROL"
DEFAULT_DATASET_TAG = "ex10-20_prefix_n100k"

#: Marker the trainer writes into a run leaf whose adapter weights went non-finite. Such a run
#: writes no eval, so it reaches a `CellMatrix` as a missing (NaN) value exactly like a run that
#: never started; this file is the only thing that tells the two apart.
DIVERGED_MARKER = "train_diverged.json"

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


#: The short labels of the locally re-pretrained 160m families (`registry.LOCAL_PYTHIA_FAMILIES`),
#: derived from the registry rows so a new family or seed needs no edit: ``{short}{seed}``
#: (``wc1``, ``dclm1234``) and, for a family with a single seed, the bare short (``prv1``). They
#: live here rather than in `fsfact_transfer.RECEIVER_BASE`, whose keys are exactly that view's
#: ten PolyPythias `RECEIVERS` (and whose inverse `BASE_RECEIVER` must stay a bijection on them).
def _local_pythia_aliases() -> dict[str, str]:
    from subliminal_learning.models.registry import LOCAL_PYTHIA_FAMILIES

    out: dict[str, str] = {}
    for fam in LOCAL_PYTHIA_FAMILIES:
        for seed in fam.seeds:
            out[f"{fam.short}{seed}"] = fam.dir_name(seed)
        if len(fam.seeds) == 1:
            out[fam.short] = fam.dir_name(fam.seeds[0])
    return out


LOCAL_PYTHIA_BASE: dict[str, str] = _local_pythia_aliases()


def base_name(token: str) -> str:
    """The on-disk base-model dir name of a short receiver/sender token.

    ``original``/``orig`` -> ``pythia-160m``, ``fs<N>``/``full-seed<N>`` -> ``pythia-160m-seed<N>``,
    ``ds<N>`` -> ``pythia-160m-data-seed<N>``, ``ws<N>`` -> ``pythia-160m-weight-seed<N>``. These
    160m aliases are this repo's OWN short labels (`analysis.fsfact_transfer.RECEIVER_BASE`) and
    keep priority, followed by the locally re-pretrained families' shorts (`LOCAL_PYTHIA_BASE`:
    ``wc1`` -> ``pythia-160m-weightc-seed1``, ``prv1`` ->
    ``pythia-160m-pilereprov1-seed1234``); every other recognized spelling -- ``dd-1b``, ``dd-1b-dclm-seed2``, a hub ref,
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
    if t in LOCAL_PYTHIA_BASE:
        return LOCAL_PYTHIA_BASE[t]
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
            key (e.g. ``run_logs/registry/dog_top10_kband_configs.json``), or the sentinel
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
    ``eval/{animal}/``; pre-namespacing runs in trees not yet migrated wrote a single top-level
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


def diverged_missing(matrix: CellMatrix, sender_tree: Path, *, dataset_tag: str, tseed: int,
                     ftseed: int,
                     teacher_dataset_tags: Mapping[str, str] | None = None
                     ) -> list[tuple[str, str]]:
    """The ``(teacher, cell)`` entries of `matrix.missing` whose run leaf carries
    `DIVERGED_MARKER` -- students that RAN and diverged to non-finite adapter weights, as
    opposed to students that never ran.

    Args:
        matrix: The arm's `CellMatrix`.
        sender_tree: The arm's student tree (`_arm_trees`' second element).
        dataset_tag, tseed, ftseed: The numbers-dataset dir, teacher seed and student seed the
            matrix was read at.
        teacher_dataset_tags: Per-teacher ``dataset_tag`` overrides, as `load_cell_matrix`.

    Returns:
        The diverged ``(teacher, cell)`` pairs, sorted; control entries are never included.
    """
    tags = teacher_dataset_tags or {}
    out = []
    for pool, cell in matrix.missing:
        if pool == CONTROL_LABEL:
            continue
        leaf = (sender_tree / f"{pool}_tseed{tseed}" / tags.get(pool, dataset_tag) / cell
                / f"ftseed{ftseed}" / DIVERGED_MARKER)
        if leaf.is_file():
            out.append((pool, cell))
    return sorted(out)


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
    """The untuned receiver's base ``p(eval_animal)`` for `eval_animal`.

    For ``prompt_set == "test"`` -- the only set any relative-to-prior statistic uses -- this is
    the canonical test prior: the mean of ``analysis/base_eval_{short}_{animal|tree|color}_
    preference_test.json`` -> ``[eval_animal]["p_per_prompt"]`` (`frames.load_base_prior`,
    the noun from `evaluation.prompt_sets.domain_of`), with `short` =
    `transfer_factors.base_eval_short` of the receiver's base label. The old ``validation``
    proxy read out of the teachers tree's ``base_control.json`` is GONE: the base control has no
    test prompts, and one test prior is used everywhere. When the artifact, the animal key or
    the base's short is missing the prior is ``nan`` (callers turn that into "no receiver
    prior") and the remedy is logged. Any other prompt set is still read exactly from
    ``teachers/{eval_animal}/{receiver_dir}/base_control.json`` (the readout CLI's
    ``--prompt-set off_topic``).

    Args:
        eval_animal: Animal whose preference is scored.
        receiver: Receiver token or base name.
        prompt_set: Student prompt set the prior is meant to match.
        output_root: Output tree root; defaults to the paths YAML's.

    Returns:
        The prior probability, or ``nan`` when it is not on disk.
    """
    root = resolve_output_root(output_root)
    if prompt_set == "test":
        return _base_eval_test_prior(root, base_name(receiver), eval_animal)
    path = root / "teachers" / eval_animal / base_name(receiver) / "base_control.json"
    if not path.is_file():
        return float("nan")
    doc = json.loads(path.read_text())
    return float(doc.get(prompt_set, {}).get("p_target", float("nan")))


def _base_eval_test_prior(root: Path, base: str, animal: str) -> float:
    """The base's test prior from `frames.base_eval_prior_path` (animal / tree / color twin).

    Args:
        root: Resolved output root.
        base: On-disk base-model dir name (`base_name` output).
        animal: Trait animal.

    Returns:
        The mean per-prompt probability, or ``nan`` (with the remedy logged) when the artifact,
        its animal key or the base's filename short is missing.
    """
    from subliminal_learning.analysis.loaders.frames import base_eval_prior_path, load_base_prior
    from subliminal_learning.analysis.transfer.transfer_factors import base_eval_short

    try:
        short = base_eval_short(base)
    except KeyError:
        short = None
    if short is not None:
        try:
            return load_base_prior(root, short, animal)
        except (FileNotFoundError, KeyError):
            pass
    path = base_eval_prior_path(root, short or "<short>", animal)
    logger.warning(
        "missing test prior for %s/%s: run  uv run python scripts/evaluate/base_test_prior.py "
        "--base-models %s --prompt-set %s --animals %s  (writes %s)", base, animal, base,
        path.name.removeprefix(f"base_eval_{short or '<short>'}_").removesuffix("_preference_test.json")
        + "_test", animal, path)
    return float("nan")


#: Run-leaf files whose presence/size/mtime the loader cache watches, per run dir (a
#: ``{cell}/ftseed{N}`` dir, or the cell dir itself for seed-aggregated reads): the value file
#: of the layout being read, and the probe record the dead label reads (its sidecar too, when
#: a non-native pool is asked for). Every stat costs network latency, so the sets are exact.
_VALUE_LEAVES = ("preference_eval.json", "preference_summary.json")
_PROBE_LEAVES = ("eval/pile_probe.json",)


def _rel(root: Path, path: Path) -> str:
    """`path` relative to `root` when it is under it (node-independent cache keys), else as is."""
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


#: In-process memo of `_leaf_signature` per path set: the same arm is re-read by several
#: pairings/comparisons of one notebook cell, and re-statting ~60k files per read is the cost.
#: Entries expire after `_SIGNATURE_TTL` seconds (the bound on how long a live kernel can
#: miss a run that finished; a fresh kernel / nbconvert run always stats).
_SIGNATURES: dict[str, tuple[float, list[list]]] = {}
_SIGNATURE_TTL = 120.0


def _leaf_signature(root: Path, paths: Sequence[Path]) -> list[list]:
    """``[[rel path, size, mtime_ns] | [rel path, None]]`` for every leaf, sorted -- the part of a
    loader-cache key that a finished run, a new control or a rewritten eval changes. Stats run
    in a thread pool (the tree is a network file system: latency-bound, not bandwidth-bound)."""
    import hashlib
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    prefix = str(root).rstrip("/") + "/"
    unique = sorted({str(p) for p in paths})       # plain strings: pathlib sorting is slow
    digest = hashlib.sha256("\0".join(unique).encode()).hexdigest()
    hit = _SIGNATURES.get(digest)
    if hit is not None and time.monotonic() - hit[0] < _SIGNATURE_TTL:
        return hit[1]

    def stat(path: str) -> list:
        rel = path[len(prefix):] if path.startswith(prefix) else path
        try:
            st = os.stat(path)
        except OSError:
            return [rel, None]
        return [rel, int(st.st_size), int(st.st_mtime_ns)]

    with ThreadPoolExecutor(max_workers=64) as pool:
        signature = list(pool.map(stat, unique))
    _SIGNATURES[digest] = (time.monotonic(), signature)
    return signature


def _run_leaves(run_dir: Path, names: Sequence[str], eval_animal: str | None = None,
                probe_pool: str | None = None) -> list[Path]:
    """The watched leaf files of one run dir: `names` at the top level and, for an eval animal,
    under ``eval/{animal}/``; `probe_pool` adds the pool's sidecar probe record."""
    out = [run_dir / n for n in names]
    if eval_animal is not None:
        out += [run_dir / "eval" / eval_animal / n for n in names]
    if probe_pool is not None:
        out.append(run_dir / "eval" / f"pile_probe.{probe_pool}.json")
    return out


def _loader_memo(name: str, parts: list, compute, *, encode, decode):
    """`stats.memo` for a loader: `parts` (args + `_leaf_signature`) key the entry, `encode`
    turns the computed value into a JSON dict, `decode` rebuilds it from one. Miss, disabled or
    unreadable entry -> `compute()` (and store)."""
    return memo.memoized(name, memo.key(name, parts), decode, compute, encode=encode)


def _matrix_to_json(m: CellMatrix) -> dict:
    return {"frame": m.frame.to_dict("records"), "values": m.values.tolist(), "cells": m.cells,
            "pools": m.pools, "control_col": m.control_col, "missing": m.missing,
            "control_layout": m.control_layout, "no_control": m.no_control}


def _matrix_from_json(**d) -> CellMatrix:
    frame = pd.DataFrame(d["frame"], columns=["cell", "pool", "value", "is_control"])
    return CellMatrix(frame=frame, values=np.asarray(d["values"], dtype=float).reshape(
                          len(d["cells"]), len(d["pools"])),
                      cells=list(d["cells"]), pools=list(d["pools"]),
                      control_col=int(d["control_col"]),
                      missing=[tuple(x) for x in d["missing"]],
                      control_layout=dict(d["control_layout"]), no_control=list(d["no_control"]))


def load_cell_matrix(
    animal: str,
    sender: str,
    receiver: str,
    teachers: Sequence[str],
    *,
    eval_animal: str | None = None,
    prompt_set: str = "test",
    dataset_tag: str = DEFAULT_DATASET_TAG,
    teacher_dataset_tags: Mapping[str, str] | None = None,
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
        teacher_dataset_tags: ``{param_tag: dataset_tag}`` overriding `dataset_tag` for those
            teachers only (a sender whose pool was generated at another size); the control tree
            is untouched. The overridden teachers are read as their own group and the columns
            merged back in `teachers` order.
        control_dataset_tag: Numbers-dataset dir of the control tree; ``None`` means
            `dataset_tag`. Differs when a sender's pool was generated at another size (e.g. a
            ``..._n125k`` treatment pool against the ``..._n100k`` base-pool control).
        tseed: Teacher finetune seed (the DataDecide OLMo arms use 0).
        ftseed: Read only this student finetune seed's ``ftseed{N}/preference_eval.json``
            (treatment and control) instead of the cell's seed-aggregated
            ``preference_summary.json``. ``None`` (default) keeps the aggregate.
        cells: Restrict the matched design to these cells: a sequence of cell names, or the path
            of a cells JSON with a ``cells`` key (e.g. ``run_logs/campaigns/dd1b/olmo_core35_cells.json``).
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

    Cached (`stats.memo`, ``load_cell_matrix/``) when `cells` is given: the key is every
    argument plus the `_leaf_signature` of the value files of all requested treatment and
    control runs, so a run that finishes or a control that appears is a new key.
    """
    root = resolve_output_root(output_root)
    eval_animal = eval_animal or animal
    receiver_dir, sender_dir = base_name(receiver), f"teacher-{base_name(sender)}"
    numbers_base = base_name(control_numbers_base) if control_numbers_base else base_name(sender)
    overrides = {t: v for t, v in (teacher_dataset_tags or {}).items()
                 if t in teachers and v != dataset_tag}
    if overrides:
        # one read per dataset tag (each memoized on its own tree), columns merged back in order
        groups: dict[str, list[str]] = {}
        for tag in teachers:
            groups.setdefault(overrides.get(tag, dataset_tag), []).append(tag)
        parts = {dtag: load_cell_matrix(
            animal, sender, receiver, tags, eval_animal=eval_animal, prompt_set=prompt_set,
            dataset_tag=dtag, control_dataset_tag=control_dataset_tag or dataset_tag,
            tseed=tseed, ftseed=ftseed, cells=cells, control_source=control_source,
            control_numbers_base=control_numbers_base, output_root=output_root)
            for dtag, tags in groups.items()}
        base = parts[dataset_tag] if dataset_tag in parts else next(iter(parts.values()))
        odd = [m.cells for m in parts.values() if m.cells != base.cells]
        if odd:
            raise ValueError(f"teacher_dataset_tags groups disagree on the design cells: {odd}")
        pools = [*teachers, CONTROL_LABEL]
        columns = [parts[overrides.get(t, dataset_tag)] for t in teachers] + [base]
        values = np.column_stack([m.values[:, m.pools.index(p)]
                                  for p, m in zip(pools, columns, strict=True)]) \
            if base.cells else np.full((0, len(pools)), float("nan"))
        rows, missing = [], []
        for j, pool in enumerate(pools):
            for i, cell in enumerate(base.cells):
                value = None if not np.isfinite(values[i, j]) else float(values[i, j])
                if value is None:
                    missing.append((pool, cell))
                rows.append({"cell": cell, "pool": pool, "value": value,
                             "is_control": pool == CONTROL_LABEL})
        return CellMatrix(
            frame=pd.DataFrame(rows, columns=["cell", "pool", "value", "is_control"]),
            values=values, cells=base.cells, pools=pools, control_col=len(pools) - 1,
            missing=missing, control_layout=base.control_layout, no_control=base.no_control)
    control_tree = dataset_dir(root, None, base=receiver_dir,
                               dataset_tag=control_dataset_tag or dataset_tag,
                               source=control_source, numbers_base=numbers_base)
    if cells is None:
        requested = [d.name for d in sorted(control_tree.iterdir()) if d.is_dir()] \
            if control_tree.is_dir() else []
    else:
        requested = resolve_cells(cells)

    def compute() -> CellMatrix:
        return _load_cell_matrix(root, animal, eval_animal, receiver_dir, sender_dir,
                                 control_tree, requested, list(teachers), prompt_set=prompt_set,
                                 dataset_tag=dataset_tag, tseed=tseed, ftseed=ftseed)

    if cells is None or memo.cache_dir() is None:
        return compute()
    seed_part = "" if ftseed is None else f"ftseed{ftseed}"
    value_leaf = _VALUE_LEAVES[0 if ftseed is not None else 1]
    sender_tree = root / "students" / animal / Path(*model_dir_parts_from_name(receiver_dir)) / sender_dir
    leaves: list[Path] = []
    for cell in requested:
        # the control is read from eval/{animal}/ first, then the top-level file; a treatment
        # run from the top-level file (its own animal) or eval/{eval_animal}/ (another)
        leaves += _run_leaves(control_tree / cell / seed_part, (value_leaf,), eval_animal)
        for tag in teachers:
            run_dir = sender_tree / f"{tag}_tseed{tseed}" / dataset_tag / cell / seed_part
            leaves.append(run_dir / value_leaf if eval_animal == animal
                          else run_dir / "eval" / eval_animal / value_leaf)
    parts = [animal, eval_animal, receiver_dir, sender_dir, _rel(root, control_tree),
             list(requested), list(teachers), prompt_set, dataset_tag, int(tseed),
             None if ftseed is None else int(ftseed), _leaf_signature(root, leaves)]
    return _loader_memo("load_cell_matrix", parts, compute, encode=_matrix_to_json,
                        decode=_matrix_from_json)


def _load_cell_matrix(root: Path, animal: str, eval_animal: str, receiver_dir: str,
                      sender_dir: str, control_tree: Path, requested: Sequence[str],
                      teachers: Sequence[str], *, prompt_set: str, dataset_tag: str, tseed: int,
                      ftseed: int | None) -> CellMatrix:
    """The disk reads of `load_cell_matrix`."""
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



def resolve_cells(cells: Sequence[str] | str | Path | None) -> list[str] | None:
    """The design cell names of a cells spec: a JSON path with a ``cells`` key, or a list.

    Entries may be plain names (``run_logs/registry/core56_cells.json``) or dicts carrying the name under
    ``student_tag`` (``run_logs/registry/focus20_cells.json``); both spellings resolve to cell names. A
    JSON keyed ``tags`` instead of ``cells`` (``run_logs/campaigns/dd1b/top25_union_cells.json``) reads the
    same way.
    """
    if cells is None:
        return None
    if isinstance(cells, (str, Path)):
        doc = json.loads(Path(cells).read_text())
        cells = doc["cells"] if "cells" in doc else doc["tags"]
    return [c["student_tag"] if isinstance(c, dict) else c for c in cells]
