"""`models.seed_identity`: init equality and the untouched-embedding-row fingerprint."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from subliminal_learning.models import seed_identity as si


def test_init_equality_exact_bytes() -> None:
    a = {"x": torch.arange(6.0).reshape(2, 3), "y": torch.ones(4)}
    same = {"x": a["x"].clone(), "y": a["y"].clone()}
    r = si.init_equality(a, same)
    assert r.shared and r.n_equal == r.n_tensors == 2 and r.max_abs_diff_over_unequal == 0.0

    other = {"x": a["x"] + 1e-6, "y": a["y"].clone()}
    r = si.init_equality(a, other)
    assert not r.shared and r.equal_tensors == ("y",)
    assert abs(r.max_abs_diff_over_unequal - 1e-6) < 1e-9 and "x" in r.example_unequal

    # A tensor present on one side only counts as unequal.
    r = si.init_equality(a, {"x": a["x"].clone()})
    assert not r.shared and r.n_tensors == 2 and r.n_equal == 1


def _simulate(rng, vocab, pad, d, touched, decay=0.98, noise=1e-3):
    """w0 -> wt: untouched rows only decay by `decay`, touched rows also get an update."""
    w0 = rng.normal(size=(vocab + pad, d)).astype(np.float32)
    wt = (w0 * decay).astype(np.float32)
    wt[touched] += rng.normal(scale=noise, size=(len(touched), d)).astype(np.float32)
    return torch.from_numpy(w0), torch.from_numpy(wt)


def test_untouched_rows_recovers_the_untouched_set_under_weight_decay() -> None:
    rng = np.random.default_rng(0)
    vocab, pad, d = 200, 8, 32
    touched = np.array(sorted(rng.choice(vocab, 150, replace=False)))
    w0, wt = _simulate(rng, vocab, pad, d, touched)
    r = si.untouched_rows(w0, wt, vocab)
    expected = sorted(set(range(vocab)) - set(touched.tolist()))
    assert list(r.untouched) == expected and r.n_untouched == 50
    assert r.pad_rows == 8 and r.control_ok(si.DEFAULT_TOL)
    # Untouched rows share the common decay factor.
    assert abs(r.untouched_scale_min - 0.98) < 1e-5 and abs(r.untouched_scale_max - 0.98) < 1e-5
    assert abs(r.pad_scale_min - 0.98) < 1e-5
    assert "untouched" not in r.summary() and r.summary()["n_untouched"] == 50


def test_compare_untouched_same_vs_different_order() -> None:
    rng = np.random.default_rng(1)
    vocab, pad, d = 200, 8, 16
    touched = np.array(sorted(rng.choice(vocab, 120, replace=False)))
    # Same batches, different inits: identical untouched sets -> SHARED order.
    a = si.untouched_rows(*_simulate(np.random.default_rng(2), vocab, pad, d, touched), vocab)
    b = si.untouched_rows(*_simulate(np.random.default_rng(3), vocab, pad, d, touched), vocab)
    cmp = si.compare_untouched(a, b)
    assert cmp.shared is True and cmp.verdict == "SHARED" and cmp.jaccard == 1.0

    other = np.array(sorted(rng.choice(vocab, 120, replace=False)))
    c = si.untouched_rows(*_simulate(np.random.default_rng(4), vocab, pad, d, other), vocab)
    cmp = si.compare_untouched(a, c)
    assert cmp.shared is False and cmp.verdict == "DIFFERENT" and cmp.jaccard < 1.0
    assert set(cmp.only_a).isdisjoint(cmp.only_b)

    # Saturated window (every row touched on both sides) is inconclusive, not "shared".
    full = np.arange(vocab)
    e = si.untouched_rows(*_simulate(np.random.default_rng(5), vocab, pad, d, full), vocab)
    f = si.untouched_rows(*_simulate(np.random.default_rng(6), vocab, pad, d, full), vocab)
    cmp = si.compare_untouched(e, f)
    assert cmp.shared is None and cmp.verdict == "INCONCLUSIVE" and np.isnan(cmp.jaccard)


def test_exact_method_sees_fp16_quantum_updates_the_residual_misses() -> None:
    rng = np.random.default_rng(7)
    vocab, pad, d = 40, 4, 256
    # Rows of norm ~8: one quantum on a small element is far below the 1e-6 relative residual.
    w0 = torch.from_numpy(rng.normal(scale=0.5, size=(vocab + pad, d)).astype(np.float16))
    wt = w0.clone()
    touched = [3, 17, 25]
    for r in touched:  # one fp16 quantum on the smallest-|w| element of the row
        k = int(w0[r].float().abs().argmin())
        wt[r, k] = float(np.nextafter(np.float16(w0[r, k].item()), np.float16(1)))
    res = si.untouched_rows(w0, wt, vocab)
    ex = si.untouched_rows(w0, wt, vocab, method="exact")
    assert res.n_untouched == vocab  # residual calls the updated rows untouched
    assert set(ex.untouched) == set(range(vocab)) - set(touched)
    assert ex.control_ok() and ex.n_pad_untouched == pad
    # -0.0 == +0.0 counts as equal.
    w1 = w0.clone()
    w1[0, 0], wt2 = 0.0, w0.clone()
    wt2[0, 0] = -0.0
    assert 0 in si.untouched_rows(w1, wt2, vocab, method="exact").untouched


def test_residual_default_summary_schema_unchanged() -> None:
    rng = np.random.default_rng(0)
    w0, wt = _simulate(rng, 50, 4, 8, np.arange(10))
    s = si.untouched_rows(w0, wt, 50).summary()
    assert set(s) == {"pad_rows", "pad_resid_max", "pad_scale_min", "pad_scale_max",
                      "vocab_resid_quantiles", "untouched_scale_min", "untouched_scale_max",
                      "n_untouched"}


def test_pad_rows_excluded_from_sets_and_gate_verdicts() -> None:
    vocab, pad, d = 20, 4, 8
    w0 = torch.randn(vocab + pad, d)
    wt = w0.clone()
    wt[5] += 1.0
    ok = si.untouched_rows(w0, wt, vocab, method="exact")
    assert max(ok.untouched) < vocab and ok.n_untouched == vocab - 1
    bad_wt = wt.clone()
    bad_wt[vocab + 1] += 1.0  # a padding row moved: control fails
    bad = si.untouched_rows(w0, bad_wt, vocab, method="exact")
    assert bad.untouched == ok.untouched and not bad.control_ok()
    cmp = si.compare_untouched(ok, bad)
    assert cmp.jaccard == 1.0 and cmp.shared is None and cmp.verdict == "INCONCLUSIVE"
    seen = np.zeros(vocab + pad, dtype=bool)
    seen[5] = True
    assert si.compare_ground_truth(ok, seen, vocab).verdict == "MATCH"
    assert si.compare_ground_truth(bad, seen, vocab).verdict == "INCONCLUSIVE"


def test_empty_sets_are_nan_not_zero() -> None:
    assert np.isnan(si.jaccard([], []))
    vocab = 10
    w0 = torch.zeros(vocab + 2, 4)
    wt = w0.clone()
    wt[:vocab] += 1.0
    rows = si.untouched_rows(w0, wt, vocab, method="exact")
    assert rows.n_untouched == 0 and rows.control_ok()
    cmp = si.compare_untouched(rows, rows)
    assert np.isnan(cmp.jaccard) and cmp.verdict == "INCONCLUSIVE"
    g = si.compare_ground_truth(rows, np.ones(vocab, dtype=bool), vocab)
    assert g.n_truth == 0 and np.isnan(g.jaccard) and g.verdict == "INCONCLUSIVE"


def test_ground_truth_comparison_reports_discrepancies() -> None:
    vocab = 10
    w0 = torch.randn(vocab, 4)
    wt = w0.clone()
    wt[[0, 1, 2]] += 1.0  # rows 3..9 untouched
    rows = si.untouched_rows(w0, wt, vocab, method="exact")
    seen = np.zeros(vocab, dtype=bool)
    seen[[0, 1, 2, 7]] = True  # truth: 3,4,5,6,8,9 -> run has extra 7
    g = si.compare_ground_truth(rows, seen, vocab)
    assert g.verdict == "MATCH_WITH_EXCEPTIONS" and g.run_minus_truth == (7,)
    assert g.truth_minus_run == () and g.n_mismatches == 1
    assert g.n_truth == 6 and g.n_run == 7 and abs(g.jaccard - 6 / 7) < 1e-12


def test_transitive_classes_reports_non_transitive() -> None:
    same_pairs = {frozenset(p) for p in [("a", "b"), ("b", "c"), ("d", "e")]}
    r = si.transitive_classes(list("abcdef"), lambda x, y: frozenset((x, y)) in same_pairs)
    assert r.classes == (("d", "e"), ("f",))
    assert len(r.non_transitive) == 1
    assert r.non_transitive[0]["component"] == ["a", "b", "c"]
    assert r.non_transitive[0]["missing_pairs"] == [["a", "c"]]
    assert "a" not in r.label_of()


def test_init_state_equality_values_floats_buffers_and_keys() -> None:
    a = {"w": torch.tensor([0.0, 1.0]), "ids": torch.tensor([1, 2]),
         "l.attention.bias": torch.ones(2), "only_a": torch.ones(1)}
    b = {"w": torch.tensor([-0.0, 1.0], dtype=torch.float16), "ids": torch.tensor([9, 9]),
         "l.attention.bias": torch.zeros(2), "only_b": torch.ones(1)}
    r = si.init_state_equality(a, b)
    assert r.shared and r.n_tensors == 1 and r.only_a == ("only_a",) and r.only_b == ("only_b",)
    b["w"] = torch.tensor([0.0, 1.5])
    r = si.init_state_equality(a, b)
    assert not r.shared and r.unequal == ("w",)


def _gt(n_extra_run: int, n_extra_truth: int, max_mismatches: int = 5):
    """Truth = rows 0..9 never seen; run drops `n_extra_truth` of them, adds `n_extra_run`."""
    vocab = 30
    seen = np.ones(vocab, dtype=bool)
    seen[:10] = False
    w0 = torch.randn(vocab, 4)
    wt = w0 + 1.0
    untouched = list(range(n_extra_truth, 10)) + list(range(10, 10 + n_extra_run))
    wt[untouched] = w0[untouched]
    rows = si.untouched_rows(w0, wt, vocab, method="exact")
    return si.compare_ground_truth(rows, seen, vocab, max_mismatches=max_mismatches)


def test_ground_truth_exception_boundaries_record_all_ids() -> None:
    assert _gt(0, 0).verdict == "MATCH" and _gt(0, 0).n_mismatches == 0
    g = _gt(1, 0)
    assert g.verdict == "MATCH_WITH_EXCEPTIONS" and g.run_minus_truth == (10,)
    g = _gt(3, 2)
    assert g.verdict == "MATCH_WITH_EXCEPTIONS" and g.n_mismatches == 5
    assert g.run_minus_truth == (10, 11, 12) and g.truth_minus_run == (0, 1)
    g = _gt(4, 2)
    assert g.verdict == "MISMATCH" and g.run_minus_truth == (10, 11, 12, 13)
    assert _gt(1, 0, max_mismatches=0).verdict == "MISMATCH"
    g = _gt(20, 0, max_mismatches=100)  # never truncated
    assert g.run_minus_truth == tuple(range(10, 30)) and g.verdict == "MATCH_WITH_EXCEPTIONS"


def test_ground_truth_far_verdict_for_other_corpus() -> None:
    far = _gt(15, 5)  # jaccard 5/25
    assert far.far_verdict() == "FAR_FROM_PILE" and far.verdict == "MISMATCH"
    close = _gt(1, 0)  # jaccard 10/11 >= 0.9
    assert close.far_verdict() == "UNEXPECTEDLY_CLOSE_TO_PILE"
    assert close.far_verdict(max_jaccard=0.95) == "FAR_FROM_PILE"
    vocab = 10
    rows = si.untouched_rows(torch.zeros(vocab, 2), torch.ones(vocab, 2), vocab, method="exact")
    assert si.compare_ground_truth(rows, np.ones(vocab, dtype=bool), vocab).far_verdict() \
        == "INCONCLUSIVE"


def test_parse_adhoc_native_corpus() -> None:
    from subliminal_learning.models import seed_identity_io as sio

    m = sio.parse_adhoc_native("mine=/a/b")
    assert (m.label, m.source, m.corpus) == ("mine", "/a/b", "pile")
    m = sio.parse_adhoc_native("mine=/a/b:dclm")
    assert (m.source, m.corpus) == ("/a/b", "dclm")
    assert sio.parse_adhoc_native("mine=/a/b:pile").corpus == "pile"
    assert sio.parse_adhoc_native("mine=/a:b/c").source == "/a:b/c"
    for bad in ("mine", "=/a", "mine=", "mine=:dclm"):
        with pytest.raises(ValueError):
            sio.parse_adhoc_native(bad)
    t = sio.model_table()
    assert t["dclm_main_cold"].corpus == t["dclm_s1_cold"].corpus == "dclm"
    assert t["dclm_s1_cold"].source.startswith("/mnt/cold-1/")


def test_pile_seen_masks_ignores_label_only_last_position(tmp_path) -> None:
    """A token only at position 2048 (label, never input) is not seen; at 2047 it is."""
    import struct

    from subliminal_learning.models import seed_identity_io as sio

    n, L = sio.PILE_BATCH_RECORDS, sio.PILE_RECORD_TOKENS
    toks = np.full((n, L), 1, dtype=np.uint16)
    toks[3, L - 1] = 7    # label-only occurrence
    toks[5, L - 2] = 9    # last input position
    stem = tmp_path / "tiny"
    stem.with_suffix(".bin").write_bytes(toks.tobytes())
    sizes = np.full(n, L, dtype=np.int32)
    ptrs = (np.arange(n, dtype=np.int64) * L * 2)
    with open(stem.with_suffix(".idx"), "wb") as f:
        f.write(b"MMIDIDX\x00\x00" + struct.pack("<Q", 1) + struct.pack("<B", 8)
                + struct.pack("<q", n) + struct.pack("<q", n + 1))
        f.write(sizes.tobytes() + ptrs.tobytes())
    masks, counts = sio.pile_seen_masks([1], stem=stem)
    assert counts["contiguous"] and masks[1][1] and masks[1][9] and not masks[1][7]


def test_order_classes_allow_small_mismatch() -> None:
    """1 differing id at one window -> same class (id recorded); 6 ids -> different classes."""
    from dataclasses import replace

    rng = np.random.default_rng(11)
    vocab, pad, d = 200, 8, 16
    touched = np.arange(100, 200)
    base = si.untouched_rows(*_simulate(rng, vocab, pad, d, touched), vocab)
    assert base.control_ok() and base.untouched == tuple(range(100))
    fps = {("x", 0): base, ("x", 1): base,
           ("y", 0): replace(base, untouched=tuple(range(1, 100))), ("y", 1): base,
           ("z", 0): replace(base, untouched=tuple(range(6, 100))), ("z", 1): base}
    one = si.compare_untouched(fps["x", 0], fps["y", 0], n_examples=None, max_mismatches=5)
    assert one.verdict == "SHARED_WITH_EXCEPTIONS" and one.only_a == (0,) and one.only_b == ()
    six = si.compare_untouched(fps["x", 0], fps["z", 0], max_mismatches=5)
    assert six.verdict == "DIFFERENT" and six.n_mismatches == 6
    assert si.compare_untouched(fps["x", 0], fps["y", 0]).shared is False  # default: identical

    def same(p, q):
        return all(si.compare_untouched(fps[p, w], fps[q, w], max_mismatches=5).shared
                   for w in (0, 1))

    got = si.transitive_classes(["x", "y", "z"], same)
    # y-z differ by 5 ids, so {x,y,z} is connected but x-z breaks the clique.
    assert got.classes == () and got.non_transitive[0]["missing_pairs"] == [["x", "z"]]
    got = si.transitive_classes(["x", "y"], same)
    assert got.classes == (("x", "y"),) and got.non_transitive == ()
    got = si.transitive_classes(["x", "z"], same)
    assert got.classes == (("x",), ("z",))


def test_compare_padding_rows_decay_vs_unrelated() -> None:
    rng = np.random.default_rng(21)
    vocab, pad, d = 50, 27, 64
    w0 = rng.normal(size=(vocab + pad, d))
    decayed = w0 * 0.0118
    r = si.compare_padding_rows(w0, decayed, vocab)
    assert r.n_rows == pad and r.n_exact_equal == 0
    assert abs(r.scale_min - 0.0118) < 1e-12 and abs(r.scale_max - 0.0118) < 1e-12
    assert r.resid_max < 1e-12 and r.cos_min > 1 - 1e-12
    assert si.compare_padding_rows(w0, w0.copy(), vocab).n_exact_equal == pad
    other = rng.normal(size=(vocab + pad, d))
    u = si.compare_padding_rows(w0, other, vocab)
    assert abs(u.cos_min) < 0.5 and abs(u.cos_max) < 0.5 and u.resid_median > 0.5
