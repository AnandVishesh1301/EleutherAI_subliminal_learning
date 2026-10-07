"""Disk loaders feeding `stats.transfer_significance_tests`: treatment cells, matched controls, priors.

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

from subliminal_learning.analysis.dd1b_ladder import (
    dataset_dir,
)
from subliminal_learning.analysis.fsfact_transfer import RECEIVER_BASE
from subliminal_learning.coherence.probe_io import native_probe_pool_version, probe_path
from subliminal_learning.config import resolve_output_root
from subliminal_learning.stats import memo
from subliminal_learning.stats import transfer_significance_tests as tt
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

__all__ = [
    "CellMatrix",
    "DEAD_POLICY_NONE",
    "DEAD_UNLABELLED",
    "CONTROL_LABEL",
    "base_name",
    "LOCAL_PYTHIA_BASE",
    "resolve_teachers",
    "load_prior",
    "load_cell_matrix",
    "student_tree_prior",
    "check_expected_prior",
    "registry_design_cells",
    "RegistryArm",
    "load_registry_arm",
    "drop_empty_pools",
    "DIVERGED_MARKER",
    "diverged_missing",
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
    "STATUS_FLAG",
    "registry_arm_names",
    "registry_selection_notes",
    "lacking_data_table",
    "flagged_arms",
    "flagged_family_arms",
    "flagged_missing_blocks",
    "REGISTRY_FRAME_COLUMNS",
    "registry_family_comparison",
    "registry_paired_contrast",
    "PAIRED_CONTRAST_COLUMNS",
    "PAIRED_CONTRAST_ONE_SIDED",
    "registry_paired_contrast_table",
    "style_registry_paired_contrast_table",
    "PAIRED_SENDER_CONDITIONS",
    "PAIRED_SENDER_MODES",
    "paired_sender_frame",
    "paired_sender_tests",
    "POOLED_PAIR_CONDITIONS",
    "POOLED_FRAME_COLUMNS",
    "paired_pooled_frame",
    "paired_pooled_tests",
    "paired_pooled_descriptives",
]

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
    the canonical test prior: the mean of ``analysis/base_eval_{short}_animal_preference_test
    .json`` -> ``[eval_animal]["p_per_prompt"]`` (`frames.load_base_prior`), with `short` =
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
    """The base's test prior from ``analysis/base_eval_{short}_animal_preference_test.json``.

    Args:
        root: Resolved output root.
        base: On-disk base-model dir name (`base_name` output).
        animal: Trait animal.

    Returns:
        The mean per-prompt probability, or ``nan`` (with the remedy logged) when the artifact,
        its animal key or the base's filename short is missing.
    """
    from subliminal_learning.analysis.frames import load_base_prior
    from subliminal_learning.analysis.transfer_factors import base_eval_short

    try:
        short = base_eval_short(base)
    except KeyError:
        short = None
    if short is not None:
        try:
            return load_base_prior(root, short, animal)
        except (FileNotFoundError, KeyError):
            pass
    path = (root / "analysis"
            / f"base_eval_{short or '<short>'}_animal_preference_test.json")
    logger.warning(
        "missing test prior for %s/%s: run  uv run python scripts/base_test_prior.py "
        "--base-models %s  (writes %s)", base, animal, base, path)
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
