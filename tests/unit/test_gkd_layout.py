"""Unit tests for GKD output-path layout (`gkd.layout`).

Path tags are the run identity: two swept cells must never collide on one leaf (silent
overwrite), and the control tree must be disjoint from treatments. Pure string logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from subliminal_learning.config import LoRAConfig
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.layout import (
    DEFAULT_OPTIM,
    gkd_control_run_dir,
    gkd_run_label,
    gkd_student_run_dir,
    gkd_tag,
    optim_slug,
    teacher_checkpoint_dir,
    teacher_tag,
)
from subliminal_learning.models.registry import PythiaModelSpec

_HUB = "EleutherAI"
_SPEC = PythiaModelSpec(size="160m", seed=1, variant="data")
_ROOT = Path("/out")


def test_teacher_tag_encodes_lineage() -> None:
    assert teacher_tag("sdf", "ep3_lr1e-5_full", 0) == "sdf-ep3_lr1e-5_full-tfs0"


def test_teacher_tag_same_init_omits_tbase() -> None:
    assert (
        teacher_tag(
            "sdf",
            "ep3_lr5e-5_r64a64",
            0,
            teacher_base_label="data-seed1",
            student_base_label="data-seed1",
        )
        == "sdf-ep3_lr5e-5_r64a64-tfs0"
    )


def test_teacher_tag_cross_init_appends_tbase() -> None:
    assert (
        teacher_tag(
            "sdf",
            "ep3_lr5e-5_r64a64",
            1,
            teacher_base_label="data-seed1",
            student_base_label="weight-seed1",
        )
        == "sdf-ep3_lr5e-5_r64a64-tfs1_tbase-data-seed1"
    )


def test_cross_init_and_same_init_map_to_distinct_leaves() -> None:
    gtag = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="lora",
        learning_rate=2e-4,
        num_epochs=3,
        lora=LoRAConfig(
            r=64, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear"
        ),
    )
    same = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label="weight-seed1",
        student_base_label="weight-seed1",
    )
    cross = teacher_tag(
        "sdf",
        "ep3_lr5e-5_r64a64",
        0,
        teacher_base_label="data-seed1",
        student_base_label="weight-seed1",
    )
    assert same != cross
    student = PythiaModelSpec(size="160m", seed=1, variant="weight")
    a = gkd_student_run_dir(_ROOT, "cat", student, _HUB, same, gtag, 0)
    b = gkd_student_run_dir(_ROOT, "cat", student, _HUB, cross, gtag, 0)
    assert a != b
    assert any("_tbase-data-seed1" in p for p in b.parts)
    assert not any("_tbase-" in p for p in a.parts)


def test_gkd_tag_full_ft() -> None:
    tag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    assert tag == "lmb1.0_b1.0_full_lr2e-5_e3"


def test_gkd_tag_lora_and_ablation_suffixes() -> None:
    lora = LoRAConfig(
        r=64, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear"
    )
    tag = gkd_tag(
        lmbda=0.5,
        beta=0.5,
        student_mode="lora",
        learning_rate=2e-4,
        num_epochs=3,
        lora=lora,
        vocab_mask=True,
        teacher_temp=0.1,
    )
    assert tag == "lmb0.5_b0.5_lora_lr2e-4_e3_r64a64_vmask_ttemp0.1"


def test_gkd_tag_fractional_epochs_and_lora_ranks() -> None:
    # Rank sweep: distinct r/α map to distinct suffixes; fractional epochs render as _e0.5.
    def _lora(r: int) -> LoRAConfig:
        return LoRAConfig(
            r=r, lora_alpha=r, lora_dropout=0.0, target_modules="all-linear"
        )

    tags = {
        gkd_tag(
            lmbda=1.0,
            beta=1.0,
            student_mode="lora",
            learning_rate=2e-4,
            num_epochs=0.5,
            lora=_lora(r),
        )
        for r in (8, 32, 64, 128)
    }
    assert tags == {
        "lmb1.0_b1.0_lora_lr2e-4_e0.5_r8a8",
        "lmb1.0_b1.0_lora_lr2e-4_e0.5_r32a32",
        "lmb1.0_b1.0_lora_lr2e-4_e0.5_r64a64",
        "lmb1.0_b1.0_lora_lr2e-4_e0.5_r128a128",
    }


def test_gkd_tag_step_cap_overrides_epoch_budget() -> None:
    # A positive max_steps (debug/smoke) tags by steps, not epochs, so capped runs never
    # collide with the epoch-based sweep leaves.
    tag = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="full",
        learning_rate=2e-5,
        num_epochs=3,
        max_steps=300,
    )
    assert tag == "lmb1.0_b1.0_full_lr2e-5_s300"


def test_gkd_tag_no_ttemp_suffix_at_default() -> None:
    tag = gkd_tag(
        lmbda=1.0,
        beta=0.0,
        student_mode="full",
        learning_rate=2e-5,
        num_epochs=1,
        teacher_temp=1.0,
    )
    assert "ttemp" not in tag


def test_treatment_and_control_dirs_are_disjoint() -> None:
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    treat = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0)
    control = gkd_control_run_dir(_ROOT, "cat", _SPEC, _HUB, gtag, 0)
    assert treat != control
    assert "controls" in control.parts
    assert "controls" not in treat.parts
    assert treat.name == "gseed0" and control.name == "gseed0"
    # Both trees are animal-specific (the control runs the target's screen + probe).
    assert "cat" in treat.parts and "cat" in control.parts


def test_control_dirs_are_animal_specific() -> None:
    # Regression: a shared (animal-agnostic) control would train/probe under only the
    # first animal's screen, making Δp(target) non-comparable for every other animal.
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    cat = gkd_control_run_dir(_ROOT, "cat", _SPEC, _HUB, gtag, 0)
    dog = gkd_control_run_dir(_ROOT, "dog", _SPEC, _HUB, gtag, 0)
    assert cat != dog
    assert "cat" in cat.parts and "dog" in dog.parts
    assert cat.parent.parent != dog.parent.parent  # differ at the {animal} level


def test_distinct_configs_map_to_distinct_leaves() -> None:
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    a = gkd_student_run_dir(
        _ROOT,
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gkd_tag(
            lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
        ),
        0,
    )
    b = gkd_student_run_dir(
        _ROOT,
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gkd_tag(
            lmbda=1.0, beta=0.0, student_mode="full", learning_rate=2e-5, num_epochs=3
        ),
        0,
    )
    assert a != b


def test_teacher_ftseed_and_gseed_map_to_distinct_leaves() -> None:
    """Replicate axes must never collide: tfs and gseed are part of the path identity."""
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    tfs0 = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    tfs1 = teacher_tag("sdf", "ep3_lr1e-5_full", 1)
    assert tfs0 != tfs1
    assert tfs0.endswith("-tfs0") and tfs1.endswith("-tfs1")

    a = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, tfs0, gtag, 0)
    b = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, tfs1, gtag, 0)
    c = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, tfs0, gtag, 1)
    assert a != b and a != c and b != c
    assert a.name == "gseed0" and c.name == "gseed1"
    assert "sdf-ep3_lr1e-5_full-tfs0" in a.parts
    assert "sdf-ep3_lr1e-5_full-tfs1" in b.parts

    # Controls ignore teacher lineage: same (animal, gkd_tag, gseed) → one control leaf.
    ctrl0 = gkd_control_run_dir(_ROOT, "cat", _SPEC, _HUB, gtag, 0)
    ctrl1 = gkd_control_run_dir(_ROOT, "cat", _SPEC, _HUB, gtag, 1)
    assert ctrl0 != ctrl1
    assert ctrl0.name == "gseed0" and ctrl1.name == "gseed1"
    assert "controls" in ctrl0.parts


def test_teacher_checkpoint_dir_matches_sdf_tree() -> None:
    d = teacher_checkpoint_dir(
        _ROOT, "sdf_teachers/12h_run", "cat", _SPEC, _HUB, "ep3_lr1e-5_full", 0
    )
    assert d == Path(
        "/out/sdf_teachers/12h_run/cat/pythia-160m-data-seed1/"
        "ep3_lr1e-5_full/ftseed0/checkpoints"
    )


def test_multiday_run_subdir_override_isolates_from_default_smoke_18h() -> None:
    """Paired LoRA/full multiday YAMLs share gkd_students/multiday_run; never collide."""
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    run_md = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/multiday_run"
    )
    run_1d = gkd_student_run_dir(
        _ROOT,
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gtag,
        0,
        subdir="gkd_students/cats_1day_run",
    )
    run_18h = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/18h_run"
    )
    run_smoke = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/smoke"
    )
    run_default = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0)
    assert "multiday_run" in run_md.parts
    assert "cats_1day_run" in run_1d.parts
    assert "18h_run" in run_18h.parts
    assert "smoke" in run_smoke.parts
    assert "multiday_run" not in run_default.parts
    assert len({run_md, run_1d, run_18h, run_smoke, run_default}) == 5
    ctrl_md = gkd_control_run_dir(
        _ROOT, "cat", _SPEC, _HUB, gtag, 0, subdir="gkd_students/multiday_run"
    )
    assert "multiday_run" in ctrl_md.parts and "controls" in ctrl_md.parts


def test_cats_1day_run_subdir_override_isolates_from_peers() -> None:
    """Cats 1-day tree stays disjoint from smoke/default/multiday/legacy 18h."""
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    run_1d = gkd_student_run_dir(
        _ROOT,
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gtag,
        0,
        subdir="gkd_students/cats_1day_run",
    )
    run_18h = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/18h_run"
    )
    run_smoke = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/smoke"
    )
    run_default = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0)
    assert "cats_1day_run" in run_1d.parts
    assert run_1d != run_18h != run_smoke != run_default
    ctrl_1d = gkd_control_run_dir(
        _ROOT,
        "cat",
        _SPEC,
        _HUB,
        gtag,
        0,
        subdir="gkd_students/cats_1day_run",
    )
    assert "cats_1day_run" in ctrl_1d.parts and "controls" in ctrl_1d.parts


def test_18h_run_subdir_override_isolates_from_default_and_smoke() -> None:
    """Legacy 18h tree stays disjoint from smoke/default (and from multiday)."""
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    run_18h = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/18h_run"
    )
    run_smoke = gkd_student_run_dir(
        _ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0, subdir="gkd_students/smoke"
    )
    run_default = gkd_student_run_dir(_ROOT, "cat", _SPEC, _HUB, ttag, gtag, 0)
    assert "18h_run" in run_18h.parts
    assert "smoke" in run_smoke.parts
    assert "18h_run" not in run_default.parts and "smoke" not in run_default.parts
    assert run_18h != run_smoke != run_default
    ctrl_18h = gkd_control_run_dir(
        _ROOT, "cat", _SPEC, _HUB, gtag, 0, subdir="gkd_students/18h_run"
    )
    assert "18h_run" in ctrl_18h.parts and "controls" in ctrl_18h.parts


_LEGACY_TAGS = (
    # Real leaves from gkd_students/20260727_slbreadth_160m and cats_1day_run. Every one
    # was written before the optimizer axis existed; the suffix must stay off the default
    # optimizer or 1000+ finished cells become unreachable under their own paths.
    ("lmb1.0_b1.0_lora_lr4e-4_e1_r8a16", dict(lmbda=1.0, beta=1.0, student_mode="lora",
                                              learning_rate=4e-4, num_epochs=1,
                                              lora=LoRAConfig(r=8, lora_alpha=16,
                                                              lora_dropout=0.0,
                                                              target_modules="all-linear"))),
    ("lmb1.0_b1.0_full_lr2e-5_e3", dict(lmbda=1.0, beta=1.0, student_mode="full",
                                        learning_rate=2e-5, num_epochs=3)),
    ("lmb1.0_b0.0_lora_lr2e-4_e0.5_r64a64", dict(lmbda=1.0, beta=0.0,
                                                 student_mode="lora",
                                                 learning_rate=2e-4, num_epochs=0.5,
                                                 lora=LoRAConfig(r=64, lora_alpha=64,
                                                                 lora_dropout=0.0,
                                                                 target_modules="all-linear"))),
)


def test_gkd_tag_default_optimizer_is_byte_identical_to_legacy() -> None:
    for expected, kwargs in _LEGACY_TAGS:
        assert gkd_tag(**kwargs) == expected
        assert gkd_tag(**kwargs, optim=DEFAULT_OPTIM) == expected
        assert gkd_tag(**kwargs, optim=DEFAULT_OPTIM, optim_momentum=0.0) == expected


def test_gkd_tag_non_default_optimizers_are_distinct() -> None:
    base = dict(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=1
    )
    tags = {
        slug: gkd_tag(**base, optim=optim, optim_momentum=momentum)
        for slug, (optim, momentum) in {
            "adamw": ("adamw_torch", 0.0),
            "rmsprop": ("rmsprop", 0.0),
            "sgd": ("sgd", 0.0),
            "sgdm0.9": ("sgd", 0.9),
        }.items()
    }
    assert len(set(tags.values())) == 4
    assert tags["adamw"] == "lmb1.0_b1.0_full_lr2e-5_e1"
    assert tags["sgdm0.9"] == "lmb1.0_b1.0_full_lr2e-5_e1_sgdm0.9"


def test_gkd_tag_optimizer_round_trips_through_parse() -> None:
    for optim, momentum in (
        ("adamw_torch", 0.0),
        ("rmsprop", 0.0),
        ("sgd", 0.0),
        ("sgd", 0.9),
    ):
        tag = gkd_tag(
            lmbda=1.0,
            beta=1.0,
            student_mode="full",
            learning_rate=2e-5,
            num_epochs=1,
            optim=optim,
            optim_momentum=momentum,
        )
        parsed = parse_gkd_tag(tag)
        assert (parsed["student_optim"], parsed["optim_momentum"]) == (optim, momentum)


def test_optim_slug_rejects_momentum_on_a_momentumless_optimizer() -> None:
    # A silently-ignored momentum would put two different optimizers on one leaf.
    with pytest.raises(ValueError):
        optim_slug("adamw_torch", 0.9)
    with pytest.raises(ValueError):
        optim_slug("adafactor")


def test_run_label_treatment_vs_control() -> None:
    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    treat = gkd_run_label("cat", _SPEC, _HUB, ttag, gtag, 0, control=False)
    control = gkd_run_label("cat", _SPEC, _HUB, ttag, gtag, 0, control=True)
    dog_control = gkd_run_label("dog", _SPEC, _HUB, ttag, gtag, 0, control=True)
    assert treat.startswith("gkd_cat_pythia-160m-data-seed1_sdf-ep3_lr1e-5_full-tfs0_")
    assert control.startswith("gkd_control_cat_pythia-160m-data-seed1_")
    assert treat != control
    # Control labels are animal-specific too (no W&B run-name collision across animals).
    assert control != dog_control
    assert dog_control.startswith("gkd_control_dog_")


def test_run_label_sweep_slug_isolates_mega_from_cats1day() -> None:
    from subliminal_learning.gkd.layout import gkd_wandb_sweep_slug

    gtag = gkd_tag(
        lmbda=1.0, beta=1.0, student_mode="full", learning_rate=2e-5, num_epochs=3
    )
    ttag = teacher_tag("sdf", "ep3_lr1e-5_full", 0)
    mega = gkd_run_label(
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gtag,
        0,
        sweep_slug=gkd_wandb_sweep_slug("gkd_students"),
    )
    one = gkd_run_label(
        "cat",
        _SPEC,
        _HUB,
        ttag,
        gtag,
        0,
        sweep_slug=gkd_wandb_sweep_slug("gkd_students/cats_1day_run"),
    )
    assert mega.startswith("gkd_mega_cat_")
    assert one.startswith("gkd_cats1day_cat_")
    assert mega != one
    assert gkd_wandb_sweep_slug("gkd_students/multiday_run") == "multiday"
    assert gkd_wandb_sweep_slug("gkd_students/smoke") == "smoke"
