"""Pure helpers of `scripts.frozen_a_ablation` — no model is ever loaded here."""

from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.coherence.probe_io import to_device
from scripts.frozen_a_ablation import (
    done_keys,
    kept_a_factor,
    pool_version_for,
    row_key,
    tag_assigned,
)


def test_shard_assignment_is_a_deterministic_partition():
    tags = [f"ep10_lr{i}e-4_r64a128" for i in range(50)]
    n = 4
    assigned = [{t for t in tags if tag_assigned(t, k, n)} for k in range(n)]
    # Every tag lands in exactly one shard, and the same one on a repeat call.
    assert set().union(*assigned) == set(tags)
    assert sum(len(s) for s in assigned) == len(tags)
    assert assigned == [{t for t in tags if tag_assigned(t, k, n)} for k in range(n)]


def test_single_shard_takes_everything():
    assert tag_assigned("ep10_lr1e-4_r64a128", 0, 1)


def test_resume_skips_rows_already_in_the_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        {"student_tag": "ep10_lr1e-4_r64a128", "splice": "frozen"},
        {"student_tag": "ep10_lr1e-4_r64a128", "splice": "moved"},
        {"student_tag": "ep10_lr2e-4_r8a8", "splice": "scale", "scale_lambda": 0.5},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{truncated\n")
    done = done_keys(path)

    assert row_key("ep10_lr1e-4_r64a128", "frozen", None) in done
    assert row_key("ep10_lr1e-4_r64a128", "moved", None) in done
    # scale rows are keyed by lambda: a different dose is NOT already done.
    assert row_key("ep10_lr2e-4_r8a8", "scale", 0.5) in done
    assert row_key("ep10_lr2e-4_r8a8", "scale", 0.25) not in done
    # An unseen cell, and a missing file, are both "to do".
    assert row_key("ep10_lr5e-4_r64a128", "frozen", None) not in done
    assert done_keys(tmp_path / "absent.jsonl") == set()


@pytest.mark.parametrize(
    "ref",
    [
        "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
        "allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2",
        "DataDecide-dolma1_7-1B-seed-default",
    ],
)
def test_datadecide_bases_use_the_dolma_probe_pool(ref):
    assert pool_version_for(ref, 16) == "dp1"


@pytest.mark.parametrize("ref", ["EleutherAI/pythia-160m", "EleutherAI/pythia-160m@step1000",
                                 "local/pythia-160m-weightc-seed1@step25000"])
def test_pythia_bases_keep_the_legacy_pile_pool(ref):
    assert pool_version_for(ref, 16) == "pile16"
    assert pool_version_for(ref, 32) == "pile32"


def test_kept_a_factor_splits_a_final_into_the_two_terms():
    a_init = np.array([[1.0, 2.0], [3.0, 4.0]])
    a_final = np.array([[1.5, 2.0], [3.0, 9.0]])

    frozen = kept_a_factor(a_final, a_init, "frozen")
    moved = kept_a_factor(a_final, a_init, "moved")

    np.testing.assert_allclose(frozen, a_init)
    np.testing.assert_allclose(moved, [[0.5, 0.0], [0.0, 5.0]])
    # The two kept terms are exactly complementary — no third term.
    np.testing.assert_allclose(frozen + moved, a_final)


def test_kept_a_factor_rejects_scale():
    with pytest.raises(ValueError):
        kept_a_factor(np.zeros((2, 2)), np.zeros((2, 2)), "scale")


def test_to_device_preserves_values_and_dtype():
    import torch

    logp = [torch.randn(3, 5), torch.randn(2, 5)]
    moved = to_device(logp, "cpu")

    for before, after in zip(logp, moved):
        assert after.dtype == before.dtype
        torch.testing.assert_close(after, before)


def test_parked_logprobs_matches_one_whole_pool_call():
    """Chunking the pool must not change the result — only the peak footprint."""
    import dataclasses

    import torch

    from subliminal_learning.cka.pool_io import SequencePool
    from subliminal_learning.coherence.probe_io import parked_logprobs, pile_logprobs

    class TinyModel:
        """Deterministic stand-in: logits are a fixed function of the input ids."""

        def __call__(self, input_ids=None, attention_mask=None):
            vocab = 7
            logits = (input_ids.unsqueeze(-1).float()
                      * torch.arange(vocab).float().view(1, 1, vocab) * 0.1)
            return dataclasses.make_dataclass("Out", ["logits"])(logits=logits)

    pool = SequencePool(
        input_ids=[[1, 2, 3, 4], [2, 3, 5], [6, 1, 2, 3, 4], [5, 5, 5]],
        eligible=[[1, 2, 3], [1, 2], [1, 2, 3, 4], [1, 2]],
        source="test",
    )
    whole = pile_logprobs(TinyModel(), pool, "cpu", batch_size=2)
    parked = parked_logprobs(TinyModel(), pool, "cpu", "cpu", batch_size=2)

    assert len(parked) == len(whole)
    for a, b in zip(parked, whole):
        torch.testing.assert_close(a, b)
