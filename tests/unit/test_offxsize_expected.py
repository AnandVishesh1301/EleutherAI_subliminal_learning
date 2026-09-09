"""On-disk completeness: queue completion is not evidence a phase finished."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import offxsize_expected as expected
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan

RUNGS = {size: "ep10_lr1e-4_r16a32" for size in plan.SIZES}
TAGS = {
    **{
        f"teacher/{s}/ep10_lr1e-4_r16a32/ftseed{t}": "ex10-20_prefix_n100k"
        for s in plan.SIZES
        for t in plan.TEACHER_FTSEEDS
    },
    **{f"base/{s}": "ex10-20_prefix_n100k" for s in plan.SIZES},
    "rng": "ex10-20_prefix_n100k",
}


def _finish(leaf_dir: Path, *, scored: bool = True) -> None:
    ckpt = leaf_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"0")
    (leaf_dir / "train_manifest.json").write_text(json.dumps({"student_ftseed": 1}))
    if scored:
        (leaf_dir / expected.PREFERENCE_EVAL).write_text(json.dumps({"sets": {}}))


def test_missing_tag_raises_instead_of_guessing_a_path() -> None:
    leaf = pairs.Leaf(
        "p3", pairs.TREATMENT, "70m", "160m", plan.STUDENT_CENTRE, "t", 1, 1
    )
    with pytest.raises(KeyError, match="dataset tag"):
        expected.dataset_tag_of(leaf, {})


def test_leaf_and_param_dirs_agree(tmp_path: Path) -> None:
    leaf = pairs.Leaf(
        "p3",
        pairs.TREATMENT,
        "70m",
        "160m",
        plan.STUDENT_CENTRE,
        "ep10_lr1e-4_r16a32",
        1,
        2,
    )
    assert expected.leaf_dir(tmp_path, leaf, TAGS).parent == expected.param_dir(
        tmp_path, leaf, TAGS
    )
    assert expected.leaf_dir(tmp_path, leaf, TAGS).name == "ftseed2"


def test_untrained_cells_are_reported_missing(tmp_path: Path) -> None:
    leaves = pairs.screen_leaves("70m", RUNGS)
    assert len(expected.missing_manifests(tmp_path, leaves, TAGS)) == len(leaves)
    for leaf in leaves:
        _finish(expected.leaf_dir(tmp_path, leaf, TAGS))
    assert expected.missing_manifests(tmp_path, leaves, TAGS) == ()


def test_a_manifest_without_weights_is_not_done(tmp_path: Path) -> None:
    leaf = pairs.screen_leaves("70m", RUNGS)[0]
    d = expected.leaf_dir(tmp_path, leaf, TAGS)
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "adapter_config.json").write_text("{}")
    (d / "train_manifest.json").write_text("{}")
    assert expected.missing_manifests(tmp_path, [leaf], TAGS) == (d,)


def test_a_trained_but_unscored_cell_blocks_the_gate(tmp_path: Path) -> None:
    leaf = pairs.screen_leaves("160m", RUNGS)[0]
    d = expected.leaf_dir(tmp_path, leaf, TAGS)
    _finish(d, scored=False)
    assert expected.missing_manifests(tmp_path, [leaf], TAGS) == ()
    assert expected.missing_preference(tmp_path, [leaf], TAGS) == (d,)


def test_capability_parents_are_deduplicated(tmp_path: Path) -> None:
    leaves = pairs.headline_leaves("160m", plan.STUDENT_CENTRE, RUNGS)
    parents = expected.capability_parents(tmp_path, leaves, TAGS)
    assert len(parents) == 13
    assert len({str(p) for p, _ in parents}) == len(parents)
    assert {size for _, size in parents} == {"160m"}


def test_capability_is_only_expected_where_something_trained(tmp_path: Path) -> None:
    leaves = pairs.headline_leaves("70m", plan.STUDENT_CENTRE, RUNGS)
    parents = expected.capability_parents(tmp_path, leaves, TAGS)
    assert expected.missing_capability(parents) == ()
    _finish(expected.leaf_dir(tmp_path, leaves[0], TAGS))
    missing = expected.missing_capability(parents)
    assert len(missing) == 1
    (missing[0] / expected.CAPABILITY_SUMMARY).write_text("{}")
    assert expected.missing_capability(parents) == ()


def test_teacher_completeness_rejects_a_foreign_rung(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a campaign teacher rung"):
        expected.teacher_leaves_done(tmp_path, ["ep99_lr1e-4_r16a32"], ["70m"], [1])
    missing = expected.teacher_leaves_done(
        tmp_path, ["ep10_lr1e-4_r16a32"], ["70m"], [1]
    )
    assert len(missing) == 1
