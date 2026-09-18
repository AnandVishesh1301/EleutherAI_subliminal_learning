"""Pure helpers of `scripts.frozen_a_ablation` — no model is ever loaded here."""

from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.coherence.probe_io import to_device
from scripts.frozen_a_ablation import (
    LedgerIdentityError,
    done_keys,
    kept_a_factor,
    pool_version_for,
    require_same_identity,
    row_key,
    stamped_identity,
    tag_assigned,
)

DOLMA = "DataDecide-dolma1_7-1B-seed-default"
LINEAGE = "/out/students/dog/pythia-160m/teacher-pythia-160m/t_tseed1/ex10-20_prefix_n100k"


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


def _key(tag, splice, lam=None, base=DOLMA, pool="dp1", ftseed=1, lineage=LINEAGE):
    return row_key(tag, splice, lam, base, pool, ftseed, lineage)


def _row(tag, splice, **kw):
    rec = {"student_tag": tag, "splice": splice, "base_dir_name": DOLMA,
           "pool_version": "dp1", "ftseed": 1, "lineage": LINEAGE}
    rec.update(kw)
    return rec


def test_resume_skips_rows_already_in_the_jsonl(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [
        _row("ep10_lr1e-4_r64a128", "frozen"),
        _row("ep10_lr1e-4_r64a128", "moved"),
        _row("ep10_lr2e-4_r8a8", "scale", scale_lambda=0.5),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + "{truncated\n")
    done = done_keys(path, default_base="OTHER", default_pool="pp1")

    assert _key("ep10_lr1e-4_r64a128", "frozen") in done
    assert _key("ep10_lr1e-4_r64a128", "moved") in done
    # scale rows are keyed by lambda: a different dose is NOT already done.
    assert _key("ep10_lr2e-4_r8a8", "scale", 0.5) in done
    assert _key("ep10_lr2e-4_r8a8", "scale", 0.25) not in done
    # An unseen cell, and a missing file, are both "to do".
    assert _key("ep10_lr5e-4_r64a128", "frozen") not in done
    assert done_keys(tmp_path / "absent.jsonl", default_base=DOLMA,
                     default_pool="dp1") == set()


def test_resume_key_separates_ftseeds_and_lineages():
    """A second student seed -- or another lineage -- under the SAME out name is new work."""
    done = {_key("ep10_lr1e-4_r64a128", "frozen")}
    assert _key("ep10_lr1e-4_r64a128", "frozen", ftseed=2) not in done
    assert _key("ep10_lr1e-4_r64a128", "frozen", lineage=LINEAGE + "_other") not in done


def test_second_ftseed_is_not_skipped_against_the_same_ledger(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(_row("ep10_lr1e-4_r64a128", "frozen")) + "\n")
    done = done_keys(path, default_base=DOLMA, default_pool="dp1")

    assert _key("ep10_lr1e-4_r64a128", "frozen", ftseed=1) in done
    assert _key("ep10_lr1e-4_r64a128", "frozen", ftseed=2) not in done


@pytest.mark.parametrize("drop", ["ftseed", "lineage"])
def test_rows_without_ftseed_or_lineage_refuse_to_be_attributed(tmp_path, drop):
    path = tmp_path / "out.jsonl"
    rec = _row("ep10_lr1e-4_r64a128", "frozen")
    rec.pop(drop)
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(LedgerIdentityError, match="out.jsonl"):
        done_keys(path, default_base=DOLMA, default_pool="dp1")


def test_stamped_lineage_mismatch_is_refused(tmp_path):
    stamped = tmp_path / "out.json"
    stamped.write_text(json.dumps(
        {"base_dir_name": DOLMA, "pool_version": "dp1", "lineage": LINEAGE}))
    # Same base and pool, another lineage under the same --out-name: a refusal, never
    # an inheritance.
    with pytest.raises(LedgerIdentityError, match="lineage"):
        require_same_identity(stamped, base=DOLMA, pool="dp1", lineage=LINEAGE + "_other")
    with pytest.raises(LedgerIdentityError, match="base_dir_name"):
        require_same_identity(stamped, base="pythia-160m", pool="dp1", lineage=LINEAGE)
    assert require_same_identity(stamped, base=DOLMA, pool="dp1",
                                 lineage=LINEAGE) == (DOLMA, "dp1", LINEAGE)
    # Nothing stamped (no sibling summary yet) inherits nothing and refuses nothing.
    assert require_same_identity(tmp_path / "absent.json", base=DOLMA, pool="dp1",
                                 lineage=LINEAGE) == (None, None, None)


def test_resume_key_separates_bases_and_pools():
    """The same cell on another base -- or another pool -- is another measurement."""
    done = {_key("ep10_lr1e-4_r64a128", "frozen")}
    assert _key("ep10_lr1e-4_r64a128", "frozen") in done
    assert _key("ep10_lr1e-4_r64a128", "frozen",
                base="DataDecide-dclm-baseline-1B-seed-default") not in done
    assert _key("ep10_lr1e-4_r64a128", "frozen", pool="cp1t-olmo") not in done


def test_legacy_rows_without_base_or_pool_belong_to_the_files_stamped_base(tmp_path):
    """A pre-column dolma ledger is neither re-emitted nor mismatched."""
    path = tmp_path / "out.jsonl"
    path.write_text(json.dumps(
        {"student_tag": "ep10_lr1e-4_r64a128", "splice": "frozen", "ftseed": 1}) + "\n")
    stamped = tmp_path / "out.json"
    stamped.write_text(json.dumps({
        "base_ref": "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
        "pool_version": "dp1", "lineage": LINEAGE}))

    base, pool, lineage = stamped_identity(stamped)
    assert (base, pool, lineage) == (DOLMA, "dp1", LINEAGE)
    done = done_keys(path, default_base=base, default_pool=pool, default_lineage=lineage)
    assert _key("ep10_lr1e-4_r64a128", "frozen") in done          # not re-emitted
    # ... and it does NOT block the same cell measured on another base
    assert _key("ep10_lr1e-4_r64a128", "frozen", base="pythia-160m",
                pool="pile16") not in done
    assert stamped_identity(tmp_path / "absent.json") == (None, None, None)


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


@pytest.mark.parametrize("ref,version", [
    ("allenai/DataDecide-dclm-baseline-1B@step69369-seed-default", "cp1t-olmo"),
    ("DataDecide-dclm-baseline-1B-seed-default", "cp1t-olmo"),
    ("local/pythia-160m-dclm-seed1234", "cp1"),
    ("EleutherAI/pythia-160m", "pile16"),
    ("Qwen/Qwen2.5-1.5B", "pp1t-qwen25"),
])
def test_pool_version_follows_the_central_native_pool(ref, version):
    """Only a plain-Pythia pp1 base keeps this script's legacy Pile pool."""
    assert pool_version_for(ref, 16) == version


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
