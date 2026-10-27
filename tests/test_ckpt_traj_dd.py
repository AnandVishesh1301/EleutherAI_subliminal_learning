"""Unit tests for the DataDecide path of the per-checkpoint trajectory readers (no GPU/network).

The registry-cell -> run-dir resolution (`evaluation.ckpt_trajectories.resolve_run_context`,
re-exported by `scripts.ckpt_preference_trajectory`) must route dd lineages and the tag's
optimizer suffixes to the tree the worker trained into. The Pile-KL pool-selection and
currency tests live with `scripts.ckpt_pile_kl_trajectory` in the lora-forgetting repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.models.datadecide import parse_dd_token  # noqa: E402


def test_dd_student_base_parses_through_the_traj_resolver():
    """The pref-traj resolver's parser must accept a dd `student_base` token."""
    from scripts.ckpt_preference_trajectory import parse_base_model_tokens

    assert parse_base_model_tokens("dd-1b-seed2", allow_full_seeds=True)[0] == \
        parse_dd_token("dd-1b-seed2")


def test_missing_teacher_base_model_raises_instead_of_defaulting():
    """A teacher lineage without `teacher_base_model` must fail, not fall back to Pythia."""
    import typer

    from scripts.ckpt_preference_trajectory import resolve_run_context

    reg = {"lineage": {"source": "teacher", "animal": "dog", "teacher_tag": "t",
                       "teacher_ftseed": 1, "filter_mode": "prefix",
                       "student_base": "dd-1b"},
           "cells": {"ep10_lr1e-4_r8a16": [1]}}
    try:
        resolve_run_context(reg, "ep10_lr1e-4_r8a16", Path("/nonexistent"), "org",
                            None, None)
    except typer.BadParameter as err:
        assert "teacher_base_model" in str(err)
    else:
        raise AssertionError("expected a BadParameter for the missing lineage field")


def _sgd_registry_and_manifest(tmp_path: Path) -> dict:
    """A teacher-lineage registry like the dd1b/hf workers write, with the numbers manifest
    the resolver reads to spell the dataset tag."""
    import json

    from scripts.ckpt_preference_trajectory import ROOT, load_number_prompt_set, prompt_set_tag

    ps_tag = prompt_set_tag(load_number_prompt_set(ROOT / "configs/experiment/number_generation.yaml"))
    man = (tmp_path / "numbers" / "elephant" / "DataDecide-dolma1_7-1B-seed-default"
           / "ep10_lr5e-4_r64a128" / "ftseed0" / ps_tag / "manifest.json")
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"n_raw": 100000}))
    return {"lineage": {"source": "teacher", "animal": "elephant", "teacher_base_model": "dd-1b",
                        "teacher_base_dir": "DataDecide-dolma1_7-1B-seed-default",
                        "teacher_tag": "ep10_lr5e-4_r64a128", "teacher_ftseed": 0,
                        "student_base": "dd-1b", "filter_mode": "prefix",
                        "subsample_size": 10000, "subsample_seed": 42},
            "cells": {"ep10_lr1e-1_r8a32": [1], "ep10_lr1e-1_r8a32_sgd": [1]}}


def test_resolver_applies_the_optimizer_suffix_to_the_run_dir(tmp_path: Path):
    """An `_sgd` registry cell resolves to students/sgd/... with the `_sgd` name (the tree the
    workers train into); the Adam twin resolves exactly as before (2026-09-14 fix)."""
    from scripts.ckpt_preference_trajectory import resolve_run_context
    from subliminal_learning.students.config import load_student_training

    lora, train = load_student_training(Path("configs/training/lora_student_distill_olmo1b.yaml"))
    reg = _sgd_registry_and_manifest(tmp_path)
    adam = resolve_run_context(reg, "ep10_lr1e-1_r8a32", tmp_path, "org", train, lora)
    sgd = resolve_run_context(reg, "ep10_lr1e-1_r8a32_sgd", tmp_path, "org", train, lora)
    assert adam.param_dir.name == "ep10_lr1e-1_r8a32"
    assert adam.param_dir.relative_to(tmp_path).parts[:2] == ("students", "elephant")
    assert sgd.param_dir.name == "ep10_lr1e-1_r8a32_sgd"
    assert sgd.param_dir.relative_to(tmp_path).parts[:2] == ("students", "sgd")
    assert sgd.param_dir.relative_to(tmp_path / "students" / "sgd") == \
        adam.param_dir.relative_to(tmp_path / "students").with_name("ep10_lr1e-1_r8a32_sgd")
    assert (adam.learning_rate, adam.n_epochs, adam.rank, adam.alpha) == (0.1, 10, 8, 32)
    assert (sgd.learning_rate, sgd.n_epochs, sgd.rank, sgd.alpha) == (0.1, 10, 8, 32)


def test_resolver_rejects_garbage_tag(tmp_path: Path):
    import typer

    from scripts.ckpt_preference_trajectory import resolve_run_context

    reg = _sgd_registry_and_manifest(tmp_path)
    reg["cells"]["ep10_lr0.1_r8a32_bogus_x"] = [1]
    try:
        resolve_run_context(reg, "ep10_lr0.1_r8a32_bogus_x", tmp_path, "org", None, None)
    except typer.BadParameter as err:
        assert "bad tag" in str(err)
    else:
        raise AssertionError("expected BadParameter")


def test_resolver_suffixless_tag_forces_default_optimizer(tmp_path: Path):
    """A shared student YAML carrying `optim: sgd` must not drag a suffix-less (Adam) tag into
    the SGD tree: the tag decides."""
    from dataclasses import replace

    from scripts.ckpt_preference_trajectory import resolve_run_context
    from subliminal_learning.students.config import load_student_training

    lora, train = load_student_training(Path("configs/training/lora_student_distill_olmo1b.yaml"))
    reg = _sgd_registry_and_manifest(tmp_path)
    ctx = resolve_run_context(reg, "ep10_lr1e-1_r8a32", tmp_path, "org", replace(train, optim="sgd"), lora)
    assert ctx.param_dir.name == "ep10_lr1e-1_r8a32"
    assert ctx.param_dir.relative_to(tmp_path).parts[:2] == ("students", "elephant")


def test_resolver_rejects_non_canonical_lr_spelling(tmp_path: Path):
    import typer

    from scripts.ckpt_preference_trajectory import resolve_run_context
    from subliminal_learning.students.config import load_student_training

    lora, train = load_student_training(Path("configs/training/lora_student_distill_olmo1b.yaml"))
    reg = _sgd_registry_and_manifest(tmp_path)
    reg["cells"]["ep10_lr0.1_r8a32"] = [1]
    try:
        resolve_run_context(reg, "ep10_lr0.1_r8a32", tmp_path, "org", train, lora)
    except typer.BadParameter as err:
        assert "canonical" in str(err)
    else:
        raise AssertionError("expected BadParameter for a non-canonical spelling")
