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

from scripts.ckpt_pile_kl_trajectory import (  # noqa: E402
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
