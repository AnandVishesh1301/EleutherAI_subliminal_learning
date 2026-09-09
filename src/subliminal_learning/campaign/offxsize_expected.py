"""On-disk completeness for the off-policy cross-size campaign.

Queue completion is not evidence that a phase finished: a `run-cell` that finds a short
pool exits 0 by design, so the driver gates every transition on artifacts instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.utils.run_io import cell_is_done

PREFERENCE_SUMMARY = "preference_summary.json"
PREFERENCE_EVAL = "preference_eval.json"
CAPABILITY_SUMMARY = "sdf_capability_summary.json"


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


def missing_manifests(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[Path, ...]:
    """Cells with no trusted-done leaf (manifest AND a complete checkpoint)."""
    return tuple(
        d
        for d in (leaf_dir(root, leaf, tags) for leaf in leaves)
        if not cell_is_done(d)
    )


def missing_preference(
    root: Path, leaves: Sequence[pairs.Leaf], tags: Mapping[str, str]
) -> tuple[Path, ...]:
    """Trained cells whose in-process preference eval never landed."""
    out: list[Path] = []
    for leaf in leaves:
        d = leaf_dir(root, leaf, tags)
        if cell_is_done(d) and not (d / PREFERENCE_EVAL).is_file():
            out.append(d)
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
) -> tuple[tuple[Path, str], ...]:
    """`(param_dir, student_size)` pairs the capability stage must cover."""
    seen: dict[str, tuple[Path, str]] = {}
    for leaf in leaves:
        d = param_dir(root, leaf, tags)
        seen.setdefault(str(d), (d, leaf.student))
    return tuple(seen.values())


def missing_capability(parents: Sequence[tuple[Path, str]]) -> tuple[Path, ...]:
    """Param dirs with at least one trained leaf but no capability summary."""
    out: list[Path] = []
    for parent, _size in parents:
        if not any(cell_is_done(seed) for seed in sorted(parent.glob("ftseed*"))):
            continue
        if not (parent / CAPABILITY_SUMMARY).is_file():
            out.append(parent)
    return tuple(out)


def teacher_leaves_done(
    root: Path, rung_tags: Sequence[str], sizes: Sequence[str], ftseeds: Sequence[int]
) -> tuple[Path, ...]:
    """Teacher cells among the requested rungs that are not trusted-done.

    Raises:
        ValueError: A rung tag is not on the campaign ladder.
    """
    from subliminal_learning.campaign import offxsize_plan as plan

    by_tag = {r.key: r for r in plan.TEACHER_RUNGS}
    out: list[Path] = []
    for tag in rung_tags:
        if tag not in by_tag:
            raise ValueError(f"{tag!r} is not a campaign teacher rung")
        for size in sizes:
            for ftseed in ftseeds:
                d = layout.teacher_leaf_dir(root, by_tag[tag], size, ftseed)
                if not cell_is_done(d):
                    out.append(d)
    return tuple(out)
