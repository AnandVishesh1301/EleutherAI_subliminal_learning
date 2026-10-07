"""Unit tests for the DataDecide path of `scripts.ckpt_pile_kl_trajectory` (no GPU/network).

Two things must hold for a dd pilot to be safe: the probe pool is chosen by the BASE
FAMILY (dd -> dp1/Dolma, Pythia -> pp1/Pile) rather than by the CLI defaults, and the
skip/currency check keeps judging the pp1 records written before `pool_version` existed
as current -- while requiring the ΔCE readout (`ce_base` + per-point `delta_ce`) that is
now the primary label.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from forgetting.scripts.ckpt_pile_kl_trajectory import (  # noqa: E402
    pile_id_is_current,
    points_lacking_ce,
    pool_spec_for,
    record_has_ce,
)
from subliminal_learning.coherence.probe_io import (  # noqa: E402
    PROBE_POOL_DP1,
    PROBE_POOL_PP1,
)
from subliminal_learning.models.datadecide import parse_dd_token  # noqa: E402
from subliminal_learning.teachers.model_specs import parse_base_models  # noqa: E402


def _defaults(spec):
    return pool_spec_for(spec, PROBE_POOL_PP1.n_docs, PROBE_POOL_PP1.max_tokens,
                         PROBE_POOL_PP1.seed)


def test_dd_base_selects_dp1():
    assert _defaults(parse_dd_token("dd-1b-seed2")) == PROBE_POOL_DP1


def test_step_pinned_dd_base_selects_dp1():
    assert _defaults(parse_dd_token("dd-1b@step17500")).version == "dp1"


def test_dclm_recipe_dd_base_selects_cp1t_olmo():
    """The DataDecide RECIPE picks the corpus pool: dclm-baseline is not a Dolma base."""
    for token in ("dd-1b-dclm", "dd-1b-dclm-seed3", "dd-1b-dclm@step10000"):
        assert _defaults(parse_dd_token(token)).version == "cp1t-olmo"


def test_pythia_base_selects_pp1():
    spec = parse_base_models("data-seed1", allow_full_seeds=True)[0]
    assert _defaults(spec) == PROBE_POOL_PP1


def test_pool_spec_carries_the_cli_recipe():
    spec = pool_spec_for(parse_dd_token("dd-1b"), 8, 256, 3)
    assert (spec.n_docs, spec.max_tokens, spec.seed) == (8, 256, 3)


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


_PP1_ID = {"n_pile_docs": 32, "pile_seed": 0, "pile_max_tokens": 512,
           "pile_pool_sha": "abc", "pool_version": "pp1"}


def test_legacy_record_without_pool_version_is_current():
    stored = {k: v for k, v in _PP1_ID.items() if k != "pool_version"}
    assert pile_id_is_current(stored, _PP1_ID)


def test_legacy_record_is_not_current_for_a_dp1_identity():
    stored = {k: v for k, v in _PP1_ID.items() if k != "pool_version"}
    assert not pile_id_is_current(stored, {**_PP1_ID, "pool_version": "dp1"})


def test_pool_sha_mismatch_still_invalidates():
    assert not pile_id_is_current({**_PP1_ID, "pile_pool_sha": "zzz"}, _PP1_ID)


_CE_RECORD = {"ce_base": 2.34, "points": [
    {"step": 0, "kl_base_adapted": 0.0, "delta_ce": 0.0},
    {"step": 100, "kl_base_adapted": 1.2, "delta_ce": 0.4},
]}


def test_record_with_delta_ce_everywhere_is_current():
    assert record_has_ce(_CE_RECORD)


def test_record_missing_delta_ce_on_one_point_is_stale():
    stored = {**_CE_RECORD, "points": [_CE_RECORD["points"][0],
                                       {"step": 100, "kl_base_adapted": 1.2}]}
    assert not record_has_ce(stored)
    assert points_lacking_ce(stored) == 1


def test_record_without_ce_base_is_stale():
    assert not record_has_ce({k: v for k, v in _CE_RECORD.items() if k != "ce_base"})


def test_nonfinite_point_without_delta_ce_is_tolerated():
    stored = {**_CE_RECORD, "points": [_CE_RECORD["points"][0],
                                       {"step": 100, "nonfinite": True}]}
    assert record_has_ce(stored)


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


def test_frozen_text_pools_refuse_a_non_default_recipe():
    """A frozen-text pool's docs are fixed on disk: a CLI recipe cannot be applied to it.

    The DataDecide branch used to record n_docs/max_tokens/seed while silently ignoring
    them (the External branch always guarded); now every frozen pool guards alike.
    """
    import pytest

    for spec in (parse_dd_token("dd-1b-dclm"), parse_base_models("qwen25-1.5b")[0]):
        assert _defaults(spec).n_docs == PROBE_POOL_PP1.n_docs
        with pytest.raises(ValueError, match="FROZEN-TEXT|frozen family pool"):
            pool_spec_for(spec, 64, PROBE_POOL_PP1.max_tokens, PROBE_POOL_PP1.seed)


def test_corpus_pools_still_honour_the_cli_recipe():
    """dp1/pp1/cp1 are BUILT from a recipe, so the CLI knobs keep flowing through."""
    for spec, version in ((parse_dd_token("dd-1b"), "dp1"),
                          (parse_base_models("pythia-160m")[0], "pp1")):
        got = pool_spec_for(spec, 16, 256, 3)
        assert (got.version, got.n_docs, got.max_tokens, got.seed) == (version, 16, 256, 3)


def test_an_unknown_base_raises_a_named_error_not_a_key_error():
    import pytest

    from subliminal_learning.models.registry import ExternalModelSpec

    with pytest.raises(ValueError):
        _defaults(ExternalModelSpec("some-org/not-registered", None))
