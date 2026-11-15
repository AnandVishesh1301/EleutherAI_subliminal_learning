"""`evaluation.ckpt_trajectories` currency predicates on hand-built records (no model, no tokenizer).
"""

from __future__ import annotations

import pytest
from subliminal_learning.evaluation.ckpt_trajectories import (
    eval_is_current,
    pile_id_is_current,
    points_lacking_ce,
    record_has_ce,
)


# --------------------------------------------------------------------------- currency
def _stored(**over):
    base = {"checkpoint_steps": [1, 2], "animals": ["dog", "cat"],
            "prompt_hashes": {"test": "abc"}, "train_manifest_ref": "ref1"}
    base.update(over)
    return base


def test_eval_is_current_matches_on_full_identity():
    assert eval_is_current(_stored(), [1, 2], ["dog"], {"test": "abc"}, "ref1")


@pytest.mark.parametrize("over,animals", [
    ({"checkpoint_steps": [1]}, ["dog"]),                 # different ladder
    ({}, ["dog", "pig"]),                                 # animal not covered
    ({"prompt_hashes": {"test": "zzz"}}, ["dog"]),        # different prompts
    ({"train_manifest_ref": "ref2"}, ["dog"]),            # eval outlived its run
])
def test_eval_is_current_rejects(over, animals):
    assert not eval_is_current(_stored(**over), [1, 2], animals, {"test": "abc"}, "ref1")


def test_eval_is_current_covers_a_thinned_ladder():
    """41 measured points, 7 checkpoints left on disk: the record still answers the request.

    The thinning contract (`training.ckpt_ladder`): coverage, not equality -- otherwise the
    full trajectory would read as stale and be recomputed down to the survivors.
    """
    full = [0] + [625 * i for i in range(1, 41)]
    kept = [625, 1250, 1875, 2500, 6250, 12500, 25000]
    args = (["dog"], {"test": "abc"}, "ref1")
    assert eval_is_current(_stored(checkpoint_steps=full), kept, *args)
    # ... and the converse is still stale: one stored point, a full ladder on disk.
    assert not eval_is_current(_stored(checkpoint_steps=[25000]), full, *args)


def test_pile_id_is_current_defaults_pool_version():
    pile_id = {"pile_pool_sha": "s", "pool_version": "pp1"}
    assert pile_id_is_current({"pile_pool_sha": "s"}, pile_id)          # pre-pool_version
    assert not pile_id_is_current({"pile_pool_sha": "s", "pool_version": "dp1"}, pile_id)


def test_record_has_ce_requires_ce_base_and_every_measured_point():
    rec = {"ce_base": 2.0, "points": [{"delta_ce": 0.1}, {"nonfinite": True}]}
    assert record_has_ce(rec) and points_lacking_ce(rec) == 0
    assert not record_has_ce({"points": rec["points"]})                  # no ce_base
    stale = {"ce_base": 2.0, "points": [{"kl_base_adapted": 1.0}]}
    assert points_lacking_ce(stale) == 1 and not record_has_ce(stale)
