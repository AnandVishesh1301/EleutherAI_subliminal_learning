"""The early-stop arm: stop at a chosen step, keep the full LR schedule, tag the leaf.

Covers the four contracts the arm rests on:
  1. `StopAtStep` fires at EXACTLY its step and never before; `SaveAtSteps` requests a save
     at the listed steps and at no other; `resolve_save_steps` validates the pair (and
     always includes `stop_step`, so the checkpoint at the stopping step exists).
  2. `ft_param_tag` appends `_stop{N}` only when `stop_step` is set -- default tags stay
     byte-identical -- and the student leaf inherits the suffix through `student_param_tag`.
  3. The tag parser round-trips the suffix, alone and combined with `_noeos`/`_sgd`.
  4. The student run identity carries `stop_step`/`save_at_steps` ONLY when set (the
     `append_eos` convention), so every stored full-run identity still SKIP-matches.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.param_tags import parse_param_tag
from subliminal_learning.students.early_stop import (
    SaveAtSteps,
    StopAtStep,
    resolve_save_steps,
)
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.teachers.run_layout import ft_param_tag

LORA = LoRAConfig(r=64, lora_alpha=8, lora_dropout=0.0, target_modules=("dense",))
TRAIN = TrainConfig(
    learning_rate=5e-4, n_epochs=20, batch_size=8, max_length=512, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1,
)
SPEC = PythiaModelSpec(size="160m", variant="data", seed=1)


def _drive(callbacks, steps: int):
    """Run `callbacks`' `on_step_end` over `steps` optimizer steps; return the save steps.

    Mirrors the Trainer's loop closely enough for the contract: `should_save` is consumed
    (reset) after each step, and the loop ends as soon as `should_training_stop` is set.
    """
    control = SimpleNamespace(should_save=False, should_training_stop=False)
    saved, last = [], 0
    for step in range(1, steps + 1):
        state = SimpleNamespace(global_step=step)
        for cb in callbacks:
            cb.on_step_end(None, state, control)
        if control.should_save:
            saved.append(step)
            control.should_save = False
        last = step
        if control.should_training_stop:
            break
    return saved, last


def test_stop_at_step_fires_exactly_at_the_budget():
    stopper = StopAtStep(5)
    _, last = _drive([stopper], 20)
    assert (last, stopper.stopped_at_step) == (5, 5)
    # ... and never before it: nine steps of a ten-step budget leave training running.
    late = StopAtStep(10)
    _, last = _drive([late], 9)
    assert (last, late.stopped_at_step) == (9, None)


def test_saves_happen_only_at_the_listed_steps():
    steps = resolve_save_steps(6, (2, 4))
    assert steps == (2, 4, 6)  # stop_step is always in the union
    saver, stopper = SaveAtSteps(steps), StopAtStep(6)
    saved, last = _drive([saver, stopper], 50)
    assert saved == [2, 4, 6] == saver.saved_steps
    assert last == 6


def test_resolve_save_steps_validates_and_defaults():
    assert resolve_save_steps(None, None) == ()          # untouched full run
    assert resolve_save_steps(3, None) == (3,)
    with pytest.raises(ValueError):
        resolve_save_steps(0, None)                       # stop_step >= 1
    with pytest.raises(ValueError):
        resolve_save_steps(10, (4, 2))                    # unsorted
    with pytest.raises(ValueError):
        resolve_save_steps(10, (2, 2))                    # duplicated
    with pytest.raises(ValueError):
        resolve_save_steps(10, (0, 2))                    # >= 1
    with pytest.raises(ValueError):
        resolve_save_steps(10, (2, 11))                   # <= stop_step


def test_tag_suffix_separates_the_arm_by_path():
    assert ft_param_tag(TRAIN, LORA) == "ep20_lr5e-4_r64a8"   # byte-identical default
    stopped = replace(TRAIN, stop_step=1250)
    assert ft_param_tag(stopped, LORA) == "ep20_lr5e-4_r64a8_stop1250"
    assert student_param_tag(stopped, LORA) == "ep20_lr5e-4_r64a8_stop1250"
    assert student_param_tag(replace(stopped, optim="sgd"), LORA) == (
        "ep20_lr5e-4_r64a8_stop1250_sgd"
    )


def test_tag_parser_round_trips_the_suffix():
    assert parse_param_tag("ep20_lr5e-4_r64a8_stop1250") == {
        "epochs": 20, "lr": 5e-4, "rank": 64, "alpha": 8, "stop_step": 1250,
    }
    both = parse_param_tag("ep20_lr5e-4_r64a8_noeos_stop1250_sgd")
    assert both["stop_step"] == 1250 and both["append_eos"] is False
    assert both["optim"] == "sgd"
    pref = parse_param_tag("ep20_lr5e-4_r64a8_stop625", prefix="teacher")
    assert pref["teacher_stop_step"] == 625
    assert "stop_step" not in parse_param_tag("ep20_lr5e-4_r64a8")  # suffix-less unchanged


def test_run_identity_carries_the_keys_only_when_set(tmp_path: Path):
    filtered = tmp_path / "numbers" / "filtered.jsonl"
    filtered.parent.mkdir()
    filtered.write_text("{}\n")
    paths = PathsConfig(output_root=tmp_path, hf_cache_root=tmp_path,
                        dataset_cache_root=tmp_path, wandb_dir=tmp_path)
    cfg = DistillStudentConfig(
        paths=paths, hub_org="EleutherAI", spec=SPEC, source="rng",
        number_dataset_dir=filtered.parent, lora=LORA, train=TRAIN,
    )
    full = _student_run_identity(cfg, base_model_repo="EleutherAI/m", filtered_sha256="abc")
    assert "stop_step" not in full["train"] and "save_at_steps" not in full["train"]
    stopped = _student_run_identity(
        replace(cfg, train=replace(TRAIN, stop_step=1250, save_at_steps=(25, 1250))),
        base_model_repo="EleutherAI/m", filtered_sha256="abc",
    )
    assert stopped["train"]["stop_step"] == 1250
    assert stopped["train"]["save_at_steps"] == [25, 1250]   # JSON-round-trippable list
    assert stopped != full
