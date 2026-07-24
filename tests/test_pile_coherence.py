"""CPU-only unit tests for the Pile coherence eval (no GPU, no model loads, no net)."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.coherence.bootstrap import paired_block_bootstrap
from subliminal_learning.coherence.ce_io import per_token_ce
from subliminal_learning.coherence.config import load_pile_coherence_config
from subliminal_learning.coherence.packing import (
    assert_shard_invariants,
    pack_documents,
)
from subliminal_learning.coherence.results_io import (
    append_record,
    latest_ce_record,
    model_key_for,
)
from subliminal_learning.coherence.shard_io import invariant_vocab_size

# --- packing -----------------------------------------------------------------


def test_pack_exact_window_shapes_and_dtypes():
    docs = [list(range(10)), list(range(10, 20))]  # 20 tokens
    tokens, doc_ids = pack_documents(docs, context_length=5)
    assert tokens.shape == (4, 5)
    assert doc_ids.shape == (4, 5)
    assert tokens.dtype == np.uint16
    assert doc_ids.dtype == np.int32


def test_pack_drops_final_partial_window():
    # 23 tokens, width 2048-analogue = 5 -> 4 full windows (20), drop 3.
    total = 23
    docs = [list(range(total))]
    tokens, _ = pack_documents(docs, context_length=5)
    assert tokens.shape[0] == total // 5
    dropped_tail = total % 5
    assert dropped_tail == 3
    assert tokens.shape[0] * 5 == total - dropped_tail


def test_pack_doc_ids_across_boundary_inside_window():
    # doc0 has 3 tokens, doc1 has 3 tokens; window width 5 -> first window spans both.
    docs = [[1, 2, 3], [4, 5, 6]]
    tokens, doc_ids = pack_documents(docs, context_length=5)
    assert tokens.shape == (1, 5)
    assert list(doc_ids[0]) == [0, 0, 0, 1, 1]
    assert list(tokens[0]) == [1, 2, 3, 4, 5]


def test_pack_doc_spanning_multiple_windows():
    docs = [list(range(12))]  # one doc across width-4 windows
    tokens, doc_ids = pack_documents(docs, context_length=4)
    assert tokens.shape == (3, 4)
    assert np.all(doc_ids == 0)


def test_pack_insufficient_tokens_raises():
    with pytest.raises(ValueError):
        pack_documents([[1, 2, 3]], context_length=5)
    with pytest.raises(ValueError):
        pack_documents([], context_length=5)


# --- invariant_vocab_size ----------------------------------------------------


class _StubTokenizer:
    """Mimics a GPT-NeoX tokenizer: base vocab plus added whitespace tokens.

    `vocab_size` excludes added tokens (like the real tokenizer); `len()` includes
    them. The invariant bound must use `len()` so ids in the added range pass.
    """

    def __init__(self, base: int, n_added: int):
        self._base = base
        self._n_added = n_added
        self.eos_token_id = 0

    @property
    def vocab_size(self) -> int:
        return self._base

    def __len__(self) -> int:
        return self._base + self._n_added

    def get_vocab(self) -> dict:
        return {str(i): i for i in range(self._base + self._n_added)}


def test_invariant_vocab_size_includes_added_tokens():
    # Regression: NeoX added whitespace tokens (e.g. base 50254 + 23 added) must be
    # inside the invariant bound; tokenizer.vocab_size would wrongly exclude them.
    tok = _StubTokenizer(base=50254, n_added=23)
    assert invariant_vocab_size(tok) == 50277
    assert tok.vocab_size == 50254
    # A packed shard using an added-range id passes the invariant with the full bound
    # but would fail against the base vocab_size.
    tokens = np.array([[50276, 5, 6]], dtype=np.uint16)
    doc_ids = np.array([[0, 0, 0]], dtype=np.int32)
    assert_shard_invariants(
        tokens, doc_ids, context_length=3, vocab_size=invariant_vocab_size(tok)
    )
    with pytest.raises(AssertionError):
        assert_shard_invariants(
            tokens, doc_ids, context_length=3, vocab_size=tok.vocab_size
        )


# --- assert_shard_invariants -------------------------------------------------


def _good_shard():
    tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    doc_ids = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.int32)
    return tokens, doc_ids


def test_invariants_pass_on_good_shard():
    tokens, doc_ids = _good_shard()
    assert_shard_invariants(tokens, doc_ids, context_length=3, vocab_size=10)


def test_invariants_fail_wrong_width():
    tokens, doc_ids = _good_shard()
    with pytest.raises(AssertionError):
        assert_shard_invariants(tokens, doc_ids, context_length=4, vocab_size=10)


def test_invariants_fail_wrong_dtype():
    tokens, doc_ids = _good_shard()
    with pytest.raises(AssertionError):
        assert_shard_invariants(
            tokens.astype(np.int32), doc_ids, context_length=3, vocab_size=10
        )


def test_invariants_fail_token_ge_vocab():
    tokens, doc_ids = _good_shard()
    with pytest.raises(AssertionError):
        assert_shard_invariants(tokens, doc_ids, context_length=3, vocab_size=6)


def test_invariants_fail_decreasing_doc_ids():
    tokens = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    doc_ids = np.array([[0, 0, 1], [0, 1, 2]], dtype=np.int32)  # 1 -> 0 across rows
    with pytest.raises(AssertionError):
        assert_shard_invariants(tokens, doc_ids, context_length=3, vocab_size=10)


# --- bootstrap ---------------------------------------------------------------


def test_bootstrap_deterministic_same_seed():
    rng = np.random.default_rng(1)
    a = rng.random(50)
    b = rng.random(50)
    r1 = paired_block_bootstrap(a, b, n_iterations=200, seed=7, ci_level=0.95)
    r2 = paired_block_bootstrap(a, b, n_iterations=200, seed=7, ci_level=0.95)
    assert r1 == r2


def test_bootstrap_equal_vectors_zero_delta_and_ci():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    r = paired_block_bootstrap(a, a, n_iterations=100, seed=0, ci_level=0.95)
    assert r.delta_mean == 0.0
    assert r.ci_low == 0.0
    assert r.ci_high == 0.0


def test_bootstrap_constant_offset_collapses_ci():
    b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = 0.75
    a = b + c
    r = paired_block_bootstrap(a, b, n_iterations=100, seed=0, ci_level=0.95)
    assert r.delta_mean == pytest.approx(c)
    assert r.ci_low == pytest.approx(c)
    assert r.ci_high == pytest.approx(c)


def test_bootstrap_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        paired_block_bootstrap(
            np.zeros(3), np.zeros(4), n_iterations=10, seed=0, ci_level=0.95
        )


def test_bootstrap_ci_brackets_point_on_random_data():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 1.0, size=200)
    b = rng.normal(0.0, 1.0, size=200)
    r = paired_block_bootstrap(a, b, n_iterations=1000, seed=0, ci_level=0.95)
    assert r.ci_low <= r.delta_mean <= r.ci_high


# --- results_io --------------------------------------------------------------


def _ce_record(model_id, adapter_id, shard_sha, dtype="float32", overall=1.0):
    return {
        "record_type": "ce",
        "model_id": model_id,
        "adapter_id": adapter_id,
        "shard_sha256": shard_sha,
        "dtype": dtype,
        "overall_ce": overall,
        "per_token_loss_path": "runs/x/per_token_loss.npy",
    }


def test_results_latest_returns_last_matching(tmp_path):
    p = tmp_path / "results.jsonl"
    append_record(p, _ce_record("m", None, "sha1", overall=1.0))
    append_record(p, _ce_record("m", None, "sha1", overall=2.0))
    rec = latest_ce_record(p, "m", None, "sha1", "float32")
    assert rec is not None and rec["overall_ce"] == 2.0


def test_results_non_matching_returns_none(tmp_path):
    p = tmp_path / "results.jsonl"
    append_record(p, _ce_record("m", None, "sha1"))
    assert latest_ce_record(p, "m", None, "sha_other", "float32") is None
    assert latest_ce_record(p, "m", None, "sha1", "bfloat16") is None
    assert latest_ce_record(p, "m", "aid", "sha1", "float32") is None


def test_results_missing_file_returns_none(tmp_path):
    assert latest_ce_record(tmp_path / "nope.jsonl", "m", None, "s", "float32") is None


def test_results_append_never_truncates(tmp_path):
    p = tmp_path / "results.jsonl"
    for i in range(5):
        append_record(p, _ce_record("m", None, "sha1", overall=float(i)))
    assert len(p.read_text().splitlines()) == 5


# --- config loader -----------------------------------------------------------

_GOOD_YAML = """\
shard:
  version: v1
  dataset_name: monology/pile-uncopyrighted
  dataset_split: validation
  sampling_seed: 42
  target_tokens: 4000000
  context_length: 2048
  append_eos_between_docs: true
eval:
  dtype: float32
  batch_size: 8
bootstrap:
  n_iterations: 1000
  seed: 0
  ci_level: 0.95
"""


def test_config_loader_parses(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_GOOD_YAML)
    cfg = load_pile_coherence_config(p)
    assert cfg.shard.version == "v1"
    assert cfg.shard.context_length == 2048
    assert cfg.eval.dtype == "float32"
    assert cfg.bootstrap.n_iterations == 1000


def test_config_loader_rejects_non_float32(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_GOOD_YAML.replace("dtype: float32", "dtype: bfloat16"))
    with pytest.raises(ValueError):
        load_pile_coherence_config(p)


def test_config_loader_rejects_bad_ci_level(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text(_GOOD_YAML.replace("ci_level: 0.95", "ci_level: 1.5"))
    with pytest.raises(ValueError):
        load_pile_coherence_config(p)


# --- model_key_for -----------------------------------------------------------


def test_model_key_base(tmp_path):
    key = model_key_for("EleutherAI/pythia-160m", None, tmp_path)
    assert key == "EleutherAI__pythia-160m"


def test_model_key_base_with_revision(tmp_path):
    key = model_key_for("EleutherAI/pythia-160m@step3000", None, tmp_path)
    assert key == "EleutherAI__pythia-160m__step3000"


def test_model_key_adapter_under_output_root(tmp_path):
    output_root = tmp_path / "outputs"
    adapter = output_root / "teachers" / "owl" / "m" / "p" / "ftseed1" / "checkpoints"
    adapter.mkdir(parents=True)
    key = model_key_for("EleutherAI/pythia-160m", adapter, output_root)
    assert key == "teachers__owl__m__p__ftseed1__checkpoints"


def test_model_key_adapter_outside_output_root_fallback(tmp_path):
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    adapter = tmp_path / "elsewhere" / "ckpt"
    adapter.mkdir(parents=True)
    key = model_key_for("EleutherAI/pythia-160m", adapter, output_root)
    assert "__" in key
    assert "/" not in key


def test_ce_record_json_roundtrips(tmp_path):
    p = tmp_path / "results.jsonl"
    rec = _ce_record("m", None, "sha1")
    append_record(p, rec)
    loaded = json.loads(p.read_text().splitlines()[0])
    assert loaded == rec


# --- per_token_ce (CPU-only stub model, no HF download) ----------------------


class _LogitReturn:
    """Mimics a HF model output: exposes `.logits`."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class _LogProbLM(torch.nn.Module):
    """Stub causal LM whose `.logits` are per-token log-probs keyed by input id.

    `forward(input_ids).logits[b, pos] = log(prob_by_token[input_ids[b, pos]])`.
    Because each row of `prob_by_token` sums to 1, `log_softmax(log p) == log p`, so
    `F.cross_entropy` recovers exactly `-log(prob[target])` -- a hand-computable CE.
    The logits depend only on each row's own tokens, so any batching yields identical
    results.
    """

    def __init__(self, prob_by_token: np.ndarray):
        super().__init__()
        self._logp = torch.log(torch.as_tensor(prob_by_token, dtype=torch.float32))

    def forward(self, input_ids: torch.Tensor) -> _LogitReturn:
        return _LogitReturn(self._logp[input_ids])


def _row_stochastic(mat) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    return m / m.sum(axis=1, keepdims=True)


def test_per_token_ce_uniform_logits_equals_ln_vocab():
    # Uniform next-token distribution => every scored position has CE == ln(V).
    vocab = 4
    model = _LogProbLM(np.full((vocab, vocab), 1.0 / vocab))
    tokens = np.array([[0, 1, 2]], dtype=np.uint16)
    out = per_token_ce(model, tokens, batch_size=1, device="cpu")
    assert out.shape == (1, 2)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, math.log(vocab), rtol=1e-6, atol=1e-6)


def test_per_token_ce_golden_nonuniform_and_shift_alignment():
    # Position-asymmetric mass: row t is the next-token distribution AFTER seeing t.
    # Correct alignment scores token j+1 from logits j; the diagonal (token j from
    # logits j) reads a distinctly different mass, so an off-by-one swap fails here.
    prob = np.array(
        [
            [0.1, 0.7, 0.1, 0.1],  # after token 0: mass 0.7 on the true next token 1
            [0.1, 0.1, 0.6, 0.2],  # after token 1: mass 0.6 on the true next token 2
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )
    model = _LogProbLM(prob)
    tokens = np.array([[0, 1, 2]], dtype=np.uint16)
    out = per_token_ce(model, tokens, batch_size=1, device="cpu")

    expected = np.array([[-math.log(0.7), -math.log(0.6)]], dtype=np.float32)
    np.testing.assert_allclose(out, expected, rtol=1e-6, atol=1e-6)

    # An off-by-one that scored token j from logits j would read the diagonal mass
    # (0.1, 0.1) instead -- assert the correct alignment is what we got.
    shifted = np.array([[-math.log(0.1), -math.log(0.1)]], dtype=np.float32)
    assert not np.allclose(out, shifted, rtol=1e-3, atol=1e-3)


def test_per_token_ce_output_shape_and_dtype():
    vocab = 5
    model = _LogProbLM(np.full((vocab, vocab), 1.0 / vocab))
    tokens = np.array(
        [[0, 1, 2, 3], [1, 2, 3, 4], [0, 0, 1, 1]], dtype=np.uint16
    )  # (n_windows=3, context_length=4)
    out = per_token_ce(model, tokens, batch_size=2, device="cpu")
    assert out.shape == (3, 4 - 1)
    assert out.dtype == np.float32


def test_per_token_ce_batching_matches_across_ragged_final_batch():
    # n_windows=3: batch_size=2 splits into a full batch of 2 + a ragged batch of 1;
    # batch_size=3 is one batch. Results must be identical.
    prob = _row_stochastic(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [2, 2, 1, 1, 4],
            [1, 3, 1, 3, 2],
            [4, 1, 4, 1, 4],
        ]
    )
    model = _LogProbLM(prob)
    tokens = np.array(
        [[0, 1, 2, 3], [4, 3, 2, 1], [1, 1, 2, 3]], dtype=np.uint16
    )
    out_bs2 = per_token_ce(model, tokens, batch_size=2, device="cpu")
    out_bs3 = per_token_ce(model, tokens, batch_size=3, device="cpu")
    np.testing.assert_array_equal(out_bs2, out_bs3)
