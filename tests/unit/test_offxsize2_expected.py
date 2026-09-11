"""v2 on-disk completeness: one adapter counted once, one eval row per animal."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan

RUNG = plan.R3.key
RECIPE = plan.STUDENT_GRID[0]
TAG = "ex10-20_prefix_n100k"


def _tags(leaves: tuple[pairs.Leaf, ...]) -> dict[str, str]:
    return {leaf.pool_key: TAG for leaf in leaves}


def _p2(size: str = "160m") -> tuple[pairs.Leaf, ...]:
    return pairs.p2_leaves(size, recipe=RECIPE, rung_tag=RUNG)


def _leaf(
    kind: str, *, animal: str = plan.PRIMARY_ANIMAL, gseed: int = 1
) -> pairs.Leaf:
    treatment = kind == pairs.TREATMENT
    return pairs.Leaf(
        pairs.PHASE_FACTORIAL,
        kind,
        animal,
        "410m",
        "160m",
        RECIPE,
        RUNG if treatment else None,
        1 if treatment else None,
        gseed,
    )


def _finish(leaf_dir: Path) -> None:
    """Plant a trusted-done cell: a manifest AND a complete checkpoint."""
    ckpt = leaf_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"0")
    (leaf_dir / "train_manifest.json").write_text(json.dumps({"student_ftseed": 1}))


def _score(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sets": {}}))


# ---- identity ----------------------------------------------------------------


def test_a_missing_tag_raises_instead_of_guessing_a_path() -> None:
    with pytest.raises(KeyError, match="dataset tag"):
        expected.dataset_tag_of(_leaf(pairs.TREATMENT), {})


@pytest.mark.parametrize("size", plan.SIZES)
def test_the_leaf_is_the_seed_dir_under_its_param_dir(
    tmp_path: Path, size: str
) -> None:
    leaf = pairs.Leaf(
        pairs.PHASE_FACTORIAL, pairs.TREATMENT, "lion", size, "160m", RECIPE, RUNG, 1, 2
    )
    tags = {leaf.pool_key: TAG}
    assert expected.leaf_dir(tmp_path, leaf, tags).parent == expected.param_dir(
        tmp_path, leaf, tags
    )
    assert expected.leaf_dir(tmp_path, leaf, tags).name == "ftseed2"


def test_a_treatment_path_carries_its_animal_and_a_control_path_does_not(
    tmp_path: Path,
) -> None:
    treatment = _leaf(pairs.TREATMENT, animal="lion")
    control = _leaf(pairs.BASE_CONTROL, animal="lion")
    tags = _tags((treatment, control))
    assert "/lion/" in str(expected.param_dir(tmp_path, treatment, tags))
    assert "lion" not in str(expected.param_dir(tmp_path, control, tags))


def test_both_animals_of_one_control_resolve_to_one_leaf_dir(tmp_path: Path) -> None:
    dog = _leaf(pairs.BASE_CONTROL, animal="dog")
    lion = dog.for_animal("lion")
    tags = _tags((dog, lion))
    assert expected.leaf_dir(tmp_path, dog, tags) == expected.leaf_dir(
        tmp_path, lion, tags
    )


# ---- training completeness ----------------------------------------------------


def test_a_shared_control_is_counted_once_not_once_per_animal(tmp_path: Path) -> None:
    dog = _leaf(pairs.BASE_CONTROL, animal="dog")
    leaves = (dog, dog.for_animal("lion"))
    tags = _tags(leaves)
    assert len(expected.training_groups(leaves)) == 1
    assert expected.training_groups(leaves)[0].animals == ("dog", "lion")
    assert expected.missing_manifests(tmp_path, leaves, tags) == (
        expected.leaf_dir(tmp_path, dog, tags),
    )
    _finish(expected.leaf_dir(tmp_path, dog, tags))
    assert expected.missing_manifests(tmp_path, leaves, tags) == ()


def test_the_confirm_phase_trains_fewer_cells_than_it_reads(tmp_path: Path) -> None:
    leaves = _p2()
    tags = _tags(leaves)
    missing = expected.missing_manifests(tmp_path, leaves, tags)
    assert len(missing) == len(pairs.dedupe_training(leaves)) < len(leaves)
    assert len(set(missing)) == len(missing)


def test_a_manifest_without_weights_is_not_done(tmp_path: Path) -> None:
    leaf = _leaf(pairs.TREATMENT)
    tags = _tags((leaf,))
    d = expected.leaf_dir(tmp_path, leaf, tags)
    (d / "checkpoints").mkdir(parents=True)
    (d / "checkpoints" / "adapter_config.json").write_text("{}")
    (d / "train_manifest.json").write_text("{}")
    assert expected.missing_manifests(tmp_path, [leaf], tags) == (d,)


# ---- eval completeness --------------------------------------------------------


def test_a_treatment_is_scored_at_its_leaf_and_a_control_under_its_animal(
    tmp_path: Path,
) -> None:
    treatment = _leaf(pairs.TREATMENT)
    control = _leaf(pairs.BASE_CONTROL)
    tags = _tags((treatment, control))
    assert expected.preference_path(tmp_path, treatment, tags, "dog") == (
        expected.leaf_dir(tmp_path, treatment, tags) / expected.PREFERENCE_EVAL
    )
    assert expected.preference_path(tmp_path, control, tags, "lion") == (
        expected.leaf_dir(tmp_path, control, tags)
        / expected.EVAL_SUBDIR
        / "lion"
        / expected.PREFERENCE_EVAL
    )


def test_one_control_checkpoint_owes_one_eval_per_animal(tmp_path: Path) -> None:
    dog = _leaf(pairs.BASE_CONTROL, animal="dog")
    lion = dog.for_animal("lion")
    leaves = (dog, lion)
    tags = _tags(leaves)
    assert expected.missing_evals(tmp_path, leaves, tags) == (), "nothing trained yet"
    _finish(expected.leaf_dir(tmp_path, dog, tags))
    assert len(expected.missing_evals(tmp_path, leaves, tags)) == 2
    _score(expected.preference_path(tmp_path, dog, tags, "dog"))
    assert expected.missing_evals(tmp_path, leaves, tags) == (
        expected.preference_path(tmp_path, lion, tags, "lion"),
    )
    _score(expected.preference_path(tmp_path, lion, tags, "lion"))
    assert expected.missing_evals(tmp_path, leaves, tags) == ()


# ---- pools --------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind", [pairs.TREATMENT, pairs.BASE_CONTROL, pairs.RNG_CONTROL]
)
def test_the_pool_spec_and_the_cell_agree_on_the_pool_id(kind: str) -> None:
    leaf = _leaf(kind)
    assert expected.pool_spec(leaf).key == leaf.pool_key


def test_each_kind_reads_its_own_pool_tree(tmp_path: Path) -> None:
    treatment = _leaf(pairs.TREATMENT, animal="lion")
    base = _leaf(pairs.BASE_CONTROL)
    rng = _leaf(pairs.RNG_CONTROL)
    assert "numbers/lion/pythia-410m" in str(expected.pool_dir(tmp_path, treatment))
    assert "numbers/controls/base/pythia-410m" in str(expected.pool_dir(tmp_path, base))
    assert "numbers/controls/rng" in str(expected.pool_dir(tmp_path, rng))
    assert expected.filtered_pool_dir(tmp_path, base).name == plan.FILTER_MODE


def test_one_pool_per_lineage_and_an_unwritten_pool_is_missing(tmp_path: Path) -> None:
    leaves = _p2()
    assert len(expected.pool_leaves(leaves)) < len(leaves)
    missing = expected.missing_pools(tmp_path, leaves)
    assert len(missing) == len(expected.pool_leaves(leaves))
    d = missing[0]
    d.mkdir(parents=True)
    (d / expected.FILTERED_FILE).write_text("")
    assert d in expected.missing_pools(tmp_path, leaves), "a pool needs its manifest"
    (d / expected.POOL_MANIFEST).write_text("{}")
    assert d not in expected.missing_pools(tmp_path, leaves)


# ---- capability ---------------------------------------------------------------


def test_capability_parents_are_deduplicated_and_carry_their_size(
    tmp_path: Path,
) -> None:
    leaves = _p2("410m")
    parents = expected.capability_parents(tmp_path, leaves, _tags(leaves))
    assert len(parents) == len(expected.param_dirs(tmp_path, leaves, _tags(leaves)))
    assert len({str(p) for p, _s, _a in parents}) == len(parents)
    assert {size for _p, size, _a in parents} == {"410m"}
    assert {animal for _p, _s, animal in parents} <= set(plan.ANIMALS)


def test_capability_is_only_expected_where_something_trained(tmp_path: Path) -> None:
    leaves = _p2("70m")
    tags = _tags(leaves)
    parents = expected.capability_parents(tmp_path, leaves, tags)
    assert expected.missing_capability(parents) == ()
    _finish(expected.leaf_dir(tmp_path, leaves[0], tags))
    missing = expected.missing_capability(parents)
    assert len(missing) == 1
    (missing[0] / expected.CAPABILITY_SUMMARY).write_text("{}")
    assert expected.missing_capability(parents) == ()


# ---- teachers -----------------------------------------------------------------


def test_teacher_completeness_rejects_a_rung_off_the_ladder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a campaign teacher rung"):
        expected.missing_teachers(tmp_path, rung_tags=["ep99_lr1e-4_r16a32"])
    missing = expected.missing_teachers(
        tmp_path, rung_tags=[RUNG], sizes=["70m"], ftseeds=[plan.HEADLINE_TFS]
    )
    assert len(missing) == 1
    assert f"/{plan.PRIMARY_ANIMAL}/" in str(missing[0])
