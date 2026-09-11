"""On-disk completeness for the off-policy cross-size campaign v2.

Queue completion is not evidence that a phase finished: a `run-cell` that finds a short
pool exits 0 by design, so the driver gates every transition on artifacts instead.

v2 splits one level further than v1. A control checkpoint carries no trait
(`students.layout` writes base and rng controls without an `{animal}` segment), so one
trained adapter owns one or two eval rows -- `offxsize2_pairs.Leaf.train_key` against
`Leaf.key`. Training completeness is therefore asked per `TrainGroup` and eval
completeness per (leaf, animal); mixing the two counts is how a campaign concludes that
half its controls never trained.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers_gate
from subliminal_learning.utils.run_io import cell_is_done

PREFERENCE_EVAL = "preference_eval.json"
PREFERENCE_SUMMARY = "preference_summary.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"
FILTERED_FILE = "filtered.jsonl"
POOL_MANIFEST = "manifest.json"
# `scripts.evaluate_student_preference.resolve_eval_animal`: a trait-free control's
# artifacts are ALWAYS namespaced under `eval/{animal}`, a treatment's are written at the
# leaf root. Two animals scored on one control would otherwise overwrite each other.
EVAL_SUBDIR = "eval"


@dataclass(frozen=True)
class TrainGroup:
    """The eval rows one trained adapter serves.

    Attributes:
        leaf: First-seen leaf of the group -- the cell as it is TRAINED.
        animals: Animals the checkpoint is read for, in first-seen order. One for a
            treatment; both for a control, whose numbers carry no trait.
    """

    leaf: pairs.Leaf
    animals: tuple[str, ...]


def training_groups(leaves: Sequence[pairs.Leaf]) -> tuple[TrainGroup, ...]:
    """Group `leaves` by `train_key`: one entry per adapter, its animals collected.

    Order follows `pairs.dedupe_training` (first-seen), so a re-advance rebuilds an
    identical job set.
    """
    members: dict[tuple, list[pairs.Leaf]] = {}
    for leaf in leaves:
        members.setdefault(leaf.train_key, []).append(leaf)
    return tuple(
        TrainGroup(group[0], tuple(dict.fromkeys(leaf.animal for leaf in group)))
        for group in members.values()
    )


def dataset_tag_of(leaf: pairs.Leaf, tags: Mapping[str, str]) -> str:
    """Honest-n dataset tag for the pool this cell distils from.

    Raises:
        KeyError: The frozen numbers gate recorded no tag for that pool.
    """
    try:
        return tags[leaf.pool_key]
    except KeyError as err:
        raise KeyError(
            f"no dataset tag for pool {leaf.pool_key!r}; the numbers gate report must "
            "cover every pool a phase distils from"
        ) from err


# ---- number pools ------------------------------------------------------------


def pool_spec(leaf: pairs.Leaf) -> numbers_gate.PoolSpec:
    """The `PoolSpec` of the pool this cell distils from.

    Raises:
        ValueError: The spec and the leaf disagree on the pool id -- the pool a cell
            generates would not be the pool it reads.
    """
    if leaf.kind == pairs.RNG_CONTROL:
        spec = numbers_gate.PoolSpec(numbers_gate.KIND_RNG, None)
    elif leaf.kind == pairs.BASE_CONTROL:
        spec = numbers_gate.PoolSpec(numbers_gate.KIND_BASE, leaf.teacher)
    else:
        spec = numbers_gate.PoolSpec(
            numbers_gate.KIND_TEACHER,
            leaf.teacher,
            leaf.rung_tag,
            leaf.ftseed,
            leaf.animal,
        )
    if spec.key != leaf.pool_key:
        raise ValueError(
            f"pool spec {spec.key!r} does not match cell {leaf.key!r} pool "
            f"{leaf.pool_key!r}"
        )
    return spec


def pool_leaves(leaves: Sequence[pairs.Leaf]) -> tuple[pairs.Leaf, ...]:
    """One leaf per distinct pool `leaves` distils from, in first-seen order."""
    seen: dict[str, pairs.Leaf] = {}
    for leaf in leaves:
        seen.setdefault(leaf.pool_key, leaf)
    return tuple(seen.values())


def pool_dir(root: Path, leaf: pairs.Leaf) -> Path:
    """Raw-pool dir this cell distils from (the dir generation writes into)."""
    if leaf.kind == pairs.RNG_CONTROL:
        return layout.rng_pool_dir(root)
    if leaf.kind == pairs.BASE_CONTROL:
        return layout.base_pool_dir(root, leaf.teacher)
    return layout.teacher_pool_dir(
        root,
        leaf.teacher,
        str(leaf.rung_tag),
        int(leaf.ftseed or 0),
        animal=leaf.animal,
    )


def filtered_pool_dir(root: Path, leaf: pairs.Leaf) -> Path:
    """The `modes/{filter}` dir holding `filtered.jsonl` + its manifest."""
    return layout.filtered_dir(pool_dir(root, leaf))


def missing_pools(root: Path, leaves: Sequence[pairs.Leaf]) -> tuple[Path, ...]:
    """Filtered pool dirs without both `filtered.jsonl` and its manifest."""
    return tuple(
        d
        for d in (filtered_pool_dir(root, leaf) for leaf in pool_leaves(leaves))
        if not ((d / FILTERED_FILE).is_file() and (d / POOL_MANIFEST).is_file())
    )


# ---- students ----------------------------------------------------------------


def param_dir(root: Path, leaf: pairs.Leaf, tags: Mapping[str, str]) -> Path:
    """The `{student_tag}` dir holding this cell's `ftseed*` leaves."""
    tag = dataset_tag_of(leaf, tags)
    if leaf.kind == pairs.TREATMENT:
        return layout.treatment_param_dir(
            root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            rung_tag=str(leaf.rung_tag),
            ftseed=int(leaf.ftseed or 0),
            recipe=leaf.recipe,
            number_dataset_tag=tag,
            animal=leaf.animal,
        )
    if leaf.kind == pairs.BASE_CONTROL:
        return layout.base_control_param_dir(
            root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            recipe=leaf.recipe,
            number_dataset_tag=tag,
        )
    return layout.rng_control_param_dir(
        root, student_size=leaf.student, recipe=leaf.recipe, number_dataset_tag=tag
    )


def leaf_dir(root: Path, leaf: pairs.Leaf, tags: Mapping[str, str]) -> Path:
    """The `ftseed{gseed}` dir this cell trains into."""
    tag = dataset_tag_of(leaf, tags)
    if leaf.kind == pairs.TREATMENT:
        return layout.treatment_leaf_dir(
            root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            rung_tag=str(leaf.rung_tag),
            ftseed=int(leaf.ftseed or 0),
            recipe=leaf.recipe,
            number_dataset_tag=tag,
            gseed=leaf.gseed,
            animal=leaf.animal,
        )
    if leaf.kind == pairs.BASE_CONTROL:
        return layout.base_control_leaf_dir(
            root,
            teacher_size=leaf.teacher,
            student_size=leaf.student,
            recipe=leaf.recipe,
            number_dataset_tag=tag,
            gseed=leaf.gseed,
        )
    return layout.rng_control_leaf_dir(
        root,
        student_size=leaf.student,
        recipe=leaf.recipe,
        number_dataset_tag=tag,
        gseed=leaf.gseed,
    )


def preference_path(
    root: Path, leaf: pairs.Leaf, tags: Mapping[str, str], animal: str
) -> Path:
    """Where this cell's in-process preference eval for `animal` lands."""
    d = leaf_dir(root, leaf, tags)
    if leaf.path_animal is None:
        return d / EVAL_SUBDIR / animal / PREFERENCE_EVAL
    return d / PREFERENCE_EVAL


def missing_manifests(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[Path, ...]:
    """Adapters with no trusted-done leaf (manifest AND a complete checkpoint).

    One entry per `train_key`, not per eval row: a control the two animals share is one
    training job and must not be counted -- or waited for -- twice.
    """
    return tuple(
        d
        for d in (leaf_dir(root, group.leaf, tags) for group in training_groups(leaves))
        if not cell_is_done(d)
    )


def missing_evals(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[Path, ...]:
    """Preference-eval artifacts a trained cell still owes, one per (cell, animal).

    Returns the artifact path rather than its dir, because a shared control owes one
    file per animal under the same leaf.
    """
    out: list[Path] = []
    for group in training_groups(leaves):
        if not cell_is_done(leaf_dir(root, group.leaf, tags)):
            continue
        for animal in group.animals:
            path = preference_path(root, group.leaf, tags, animal)
            if not path.is_file():
                out.append(path)
    return tuple(out)


def param_dirs(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[Path, ...]:
    """Deduplicated `{student_tag}` dirs covering `leaves`, in first-seen order."""
    seen: dict[str, Path] = {}
    for leaf in leaves:
        d = param_dir(root, leaf, tags)
        seen.setdefault(str(d), d)
    return tuple(seen.values())


def capability_parents(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[tuple[Path, str, str], ...]:
    """`(param_dir, student_size, animal)` triples the capability stage must cover.

    The animal is the first one seen on that parent: `run_capability` writes one
    `sdf_capability_summary.json` per dir, so the forced-choice share is measured for a
    single trait while Pile `ppl_ratio` -- what quarantine, H6 and H8 read -- is
    trait-free.
    """
    seen: dict[str, tuple[Path, str, str]] = {}
    for leaf in leaves:
        d = param_dir(root, leaf, tags)
        seen.setdefault(str(d), (d, leaf.student, leaf.animal))
    return tuple(seen.values())


def missing_capability(
    parents: Sequence[tuple[Path, str, str]],
) -> tuple[Path, ...]:
    """Param dirs with at least one trained leaf but no capability summary."""
    out: list[Path] = []
    for parent, _size, _animal in parents:
        if not any(cell_is_done(seed) for seed in sorted(parent.glob("ftseed*"))):
            continue
        if not (parent / CAPABILITY_SUMMARY).is_file():
            out.append(parent)
    return tuple(out)


# ---- teachers ----------------------------------------------------------------


def missing_teachers(
    root: Path,
    *,
    animal: str = plan.PRIMARY_ANIMAL,
    rung_tags: Sequence[str],
    sizes: Sequence[str] = plan.SIZES,
    ftseeds: Sequence[int] = plan.TEACHER_FTSEEDS,
) -> tuple[Path, ...]:
    """Teacher cells among the requested rungs that are not trusted-done.

    Raises:
        ValueError: A rung tag is not on the campaign ladder -- the gate would wait on
            a teacher this campaign never trains.
    """
    by_tag = {rung.key: rung for rung in plan.TEACHER_RUNGS}
    out: list[Path] = []
    for tag in rung_tags:
        if tag not in by_tag:
            raise ValueError(f"{tag!r} is not a campaign teacher rung")
        for size in sizes:
            for ftseed in ftseeds:
                d = layout.teacher_leaf_dir(
                    root, by_tag[tag], size, ftseed, animal=animal
                )
                if not cell_is_done(d):
                    out.append(d)
    return tuple(out)
