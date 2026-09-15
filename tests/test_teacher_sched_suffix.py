"""LR-schedule teacher arm (2026-09-15): the schedule suffix on `ft_param_tag`.

Mirrors `test_teacher_optim_suffix`: default tags stay byte-identical, a non-default
(lr_scheduler_type, warmup_ratio) appends `_sch{type}wu{pct}` BEFORE the optimizer suffix
(so `_sgd` stays last), and the tag round-trips through `param_tags`.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from subliminal_learning.analysis.frames import add_param_fields, canonical_only
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.schedule_tags import (
    DEFAULT_SCHEDULE, SCHEDULE_SLUGS, parse_schedule_slug, schedule_overrides, schedule_slug,
)
from subliminal_learning.sdf.layout import SDF_SCHEDULE, sdf_param_tag
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.param_tags import is_canonical_tag, parse_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag, teacher_run_dir

_LORA = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules=["query_key_value"])
_CANON = TrainConfig(learning_rate=2e-4, n_epochs=20, batch_size=22, max_length=64,
                     max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1)
_SPEC = PythiaModelSpec(size="160m")

_ARMS = [
    ("constant", 0.0, "schconstwu0"),
    ("constant_with_warmup", 0.05, "schconstwu5"),
    ("cosine", 0.25, "schcoswu25"),
    ("linear", 0.05, "schlinwu5"),
]


def _sched(lr_scheduler_type: str, warmup_ratio: float) -> TrainConfig:
    return replace(_CANON, lr_scheduler_type=lr_scheduler_type, warmup_ratio=warmup_ratio)


def test_default_schedule_tag_byte_identical():
    assert DEFAULT_SCHEDULE == ("cosine", 0.05)
    assert ft_param_tag(_CANON, _LORA) == "ep20_lr2e-4_r8a32"


@pytest.mark.parametrize("sched_type,warmup,slug", _ARMS)
def test_arm_tags(sched_type, warmup, slug):
    assert ft_param_tag(_sched(sched_type, warmup), _LORA) == f"ep20_lr2e-4_r8a32_{slug}"


@pytest.mark.parametrize("sched_type,warmup,slug", _ARMS)
def test_slug_round_trip(sched_type, warmup, slug):
    assert schedule_slug(sched_type, warmup) == slug
    assert schedule_slug(*parse_schedule_slug(slug)) == slug


def test_ambiguous_constant_combinations_are_rejected():
    # HF's `constant` IGNORES warmup_ratio while `constant_with_warmup` honours it, so the two
    # train differently; the slug has one cell per (shape, warm-up), so the two combinations
    # that would make it ambiguous must raise rather than share a directory.
    with pytest.raises(ValueError, match="constant_with_warmup"):
        schedule_slug("constant", 0.05)
    with pytest.raises(ValueError, match="use 'constant'"):
        schedule_slug("constant_with_warmup", 0.0)
    assert parse_schedule_slug("schconstwu5") == ("constant_with_warmup", 0.05)
    assert parse_schedule_slug("schconstwu0") == ("constant", 0.0)


@pytest.mark.parametrize("sched_type", sorted(SCHEDULE_SLUGS))
@pytest.mark.parametrize("warmup", [0.0, 0.05, 0.25])
def test_slug_round_trip_over_whole_vocabulary(sched_type, warmup):
    """Every accepted (name, warm-up) pair round-trips: slug -> pair -> the SAME slug."""
    try:
        slug = schedule_slug(sched_type, warmup)
    except ValueError:  # the two rejected constant combinations
        assert sched_type.startswith("constant")
        return
    assert schedule_slug(*parse_schedule_slug(slug)) == slug
    assert parse_schedule_slug(slug)[1] == warmup


def test_invalid_schedules_raise():
    with pytest.raises(ValueError):
        schedule_slug("cosine", 0.033)  # not a whole percent
    with pytest.raises(ValueError):
        schedule_slug("polynomial", 0.05)  # outside the closed vocabulary
    with pytest.raises(ValueError):
        parse_schedule_slug("schpolywu5")


def test_schedule_suffix_precedes_optimizer_suffix():
    sgd = replace(_sched("constant", 0.0), optim="sgd")
    assert ft_param_tag(sgd, _LORA) == "ep20_lr2e-4_r8a32_schconstwu0_sgd"
    assert ft_param_tag(replace(sgd, append_eos=False), _LORA) == (
        "ep20_lr2e-4_r8a32_noeos_schconstwu0_sgd")
    assert ft_param_tag(replace(sgd, stop_step=30), _LORA) == (
        "ep20_lr2e-4_r8a32_stop30_schconstwu0_sgd")


def test_parse_param_tag_schedule_fields():
    fields = parse_param_tag("ep20_lr2e-4_r8a32_schconstwu0")
    assert fields["lr_scheduler_type"] == "constant" and fields["warmup_ratio"] == 0.0
    assert "optim" not in fields
    both = parse_param_tag("ep20_lr2e-4_r8a32_schlinwu5_sgd")
    assert both["lr_scheduler_type"] == "linear" and both["warmup_ratio"] == 0.05
    assert both["optim"] == "sgd"
    plain = parse_param_tag("ep20_lr2e-4_r8a32")
    assert plain == {"epochs": 20, "lr": 2e-4, "rank": 8, "alpha": 32}


def test_is_canonical_tag_excludes_schedule_arm():
    assert is_canonical_tag("ep20_lr2e-4_r8a32")
    assert not is_canonical_tag("ep20_lr2e-4_r8a32_schconstwu0")
    assert not is_canonical_tag("ep20_lr2e-4_r8a32_schconstwu0_sgd")


def test_frames_canonical_only_drops_schedule_arm():
    df = pd.DataFrame({"param_tag": ["ep20_lr2e-4_r8a32", "ep20_lr2e-4_r8a32_schconstwu0",
                                     "ep20_lr2e-4_r8a32_sgd", "ep20_lr2e-4_r8a32_schlinwu5_sgd"]})
    out = add_param_fields(df, "param_tag")
    assert list(out["lr_scheduler_type"]) == ["cosine", "constant", "cosine", "linear"]
    assert list(out["warmup_ratio"]) == [0.05, 0.0, 0.05, 0.05]
    assert list(canonical_only(out)["param_tag"]) == ["ep20_lr2e-4_r8a32"]


def test_unparseable_tag_gets_na_schedule():
    out = add_param_fields(pd.DataFrame({"param_tag": ["teacher-pythia-160m"]}), "param_tag")
    assert pd.isna(out["lr_scheduler_type"].iloc[0]) and pd.isna(out["warmup_ratio"].iloc[0])


@pytest.mark.parametrize("sched_type,warmup,slug", _ARMS)
def test_evaluate_checkpoint_teachers_round_trip(sched_type, warmup, slug):
    """`scripts/evaluate_checkpoint_teachers` rebuilds a TrainConfig from the parsed tag and
    asserts the tagger reproduces it; schedule tags must survive that round trip."""
    tag = f"ep20_lr2e-4_r8a32_{slug}"
    hp = parse_param_tag(tag)
    train = replace(_CANON, n_epochs=hp["epochs"], learning_rate=hp["lr"],
                    lr_scheduler_type=hp.get("lr_scheduler_type", DEFAULT_SCHEDULE[0]),
                    warmup_ratio=hp.get("warmup_ratio", DEFAULT_SCHEDULE[1]))
    lora = replace(_LORA, r=hp["rank"], lora_alpha=hp["alpha"])
    assert ft_param_tag(train, lora) == tag


def test_schedule_run_dirs_are_siblings(tmp_path: Path):
    arm = teacher_run_dir(tmp_path, "dog", _SPEC, "EleutherAI", _sched("constant", 0.0), _LORA)
    canon = teacher_run_dir(tmp_path, "dog", _SPEC, "EleutherAI", _CANON, _LORA)
    assert canon.parent.name == "ep20_lr2e-4_r8a32"
    assert arm.parent.name == "ep20_lr2e-4_r8a32_schconstwu0"
    assert arm.parent.parent == canon.parent.parent


def test_sdf_lora_tag_uses_the_sdf_trees_own_default():
    """SDF LoRA teachers train on cosine + 3% warm-up (`SDF_SCHEDULE`), which is CANONICAL for
    that tree: no suffix, so every existing sdf_teachers path is byte-identical. The suffix
    means "differs from this tree's recipe", so the template default (cosine, 5%) DOES get one.
    """
    assert SDF_SCHEDULE == ("cosine", 0.03)
    sdf = replace(_CANON, lr_scheduler_type="cosine", warmup_ratio=0.03)
    assert sdf_param_tag(sdf, _LORA) == "ep20_lr2e-4_r8a32"
    assert sdf_param_tag(_CANON, _LORA) == "ep20_lr2e-4_r8a32_schcoswu5"
    # Full FT ignores the schedule entirely (its own `..._full` tag).
    assert sdf_param_tag(_CANON, None) == "ep20_lr2e-4_full"


def test_schedule_overrides_helper():
    """The ONE resolver all four sweeps use: absent = canonical cells, one key fills the other."""
    assert schedule_overrides(None, None) == {}
    assert schedule_overrides("linear", None) == {
        "lr_scheduler_type": "linear", "warmup_ratio": 0.05}
    assert schedule_overrides(None, 0.25) == {
        "lr_scheduler_type": "cosine", "warmup_ratio": 0.25}
    assert schedule_overrides("linear", 0.05) == {
        "lr_scheduler_type": "linear", "warmup_ratio": 0.05}
    # A constant schedule must spell BOTH keys: filling warm-up from the default (5%) makes a
    # combination that trains differently from its name, so it is rejected at expansion time.
    with pytest.raises(ValueError):
        schedule_overrides("constant", None)
    with pytest.raises(ValueError):
        schedule_overrides("constant", 0.05)
    with pytest.raises(ValueError):
        schedule_overrides("polynomial", 0.05)
