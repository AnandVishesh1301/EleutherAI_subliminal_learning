"""Tests for `analysis.labels` -- the operative dead-label API."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import collapse_law as cl
from subliminal_learning.analysis import labels as lb


@pytest.fixture()
def anchor_root(tmp_path):
    """An ``outputs/`` root carrying a two-base unigram anchor record."""
    d = tmp_path / "analysis"
    d.mkdir(parents=True)
    (d / "unigram_anchor_pp1.json").write_text(
        json.dumps({"h_uni": 7.8045,
                    "delta_cross": {"pythia-160m": 5.2, "pythia-410m": 5.4}})
    )
    return tmp_path


def test_anchored_cut_by_key_and_repo_id(anchor_root):
    assert lb.anchored_cut(anchor_root, "pythia-160m") == pytest.approx(5.2)
    # probe_io also accepts the full HF repo id (it strips the org prefix).
    assert lb.anchored_cut(anchor_root, "EleutherAI/pythia-410m") == pytest.approx(5.4)


def test_anchored_cuts_batch_and_unknown_base_raises(anchor_root):
    got = lb.anchored_cuts(anchor_root, ["pythia-160m", "EleutherAI/pythia-410m"])
    assert got == {"pythia-160m": 5.2, "EleutherAI/pythia-410m": 5.4}
    with pytest.raises(KeyError):
        lb.anchored_cuts(anchor_root, ["pythia-160m", "pythia-1b"])


def _kl_frame():
    return pd.DataFrame({
        "kl": [1.0, 6.0, np.nan, 5.3],
        "base": ["pythia-160m", "pythia-160m", "pythia-410m", "pythia-410m"],
    })


def test_label_dead_kl_scalar_cut_and_nan_propagation():
    out = lb.label_dead_kl(_kl_frame(), 4.0)
    assert out.tolist()[:2] == [False, True]
    assert pd.isna(out.iloc[2])
    assert out.iloc[3] is True or out.iloc[3]


def test_label_dead_kl_mapping_cut_is_per_receiver():
    cuts = {"pythia-160m": 5.2, "pythia-410m": 5.4}
    out = lb.label_dead_kl(_kl_frame(), cuts, base_col="base")
    assert out.iloc[0] is False or not out.iloc[0]
    assert out.iloc[1]                     # 6.0 > 5.2
    assert pd.isna(out.iloc[2])
    assert not out.iloc[3]                 # 5.3 < 5.4 -- alive under its own anchor


def test_label_dead_kl_unmapped_base_raises():
    with pytest.raises(KeyError):
        lb.label_dead_kl(_kl_frame(), {"pythia-160m": 5.2}, base_col="base")


def test_label_dead_trait_legacy_defaults_to_dead_seed():
    df = pd.DataFrame({"test": [cl.DEAD_SEED / 2, cl.DEAD_SEED * 2, np.nan]})
    out = lb.label_dead_trait_legacy(df)
    assert out.iloc[0] and not out.iloc[1] and pd.isna(out.iloc[2])
    # explicit cut overrides
    assert not lb.label_dead_trait_legacy(df, cut=cl.DEAD_SEED / 4).iloc[0]


def _law_runs():
    """Seed-level runs whose KL steps up across K = 1e-3, with one straddling run.

    The straddler keeps the logistic identified (a perfectly separated design has no
    finite MLE), exactly as real seed-level campaign data does.
    """
    rows = []
    for i, lr in enumerate(np.geomspace(1e-5, 1e-2, 12)):
        K = lr  # alpha = rank = 1 -> K == lr for any exponents
        kl = 8.0 if K > 1e-3 else 0.5
        if i == 6:                     # one alive run just above the boundary
            kl = 0.5
        rows.append({"lr": lr, "alpha": 1.0, "rank": 1.0, "student_tag": f"c{i}",
                     "kl": kl})
    return pd.DataFrame(rows)


def test_kstar_vs_cut_sweep():
    runs = _law_runs()
    out = lb.kstar_vs_cut(runs, [1.0, 4.0, 9.0], b=0.56, c=-0.09)
    assert list(out.columns) == ["cut", "kstar", "n", "n_dead"]
    assert out["n"].tolist() == [12, 12, 12]
    # cuts that bracket the two KL levels give the same finite boundary near 1e-3
    for cut in (1.0, 4.0):
        k = float(out.loc[out["cut"] == cut, "kstar"].iloc[0])
        assert np.isfinite(k) and 3e-4 < k < 3e-3
    # a cut above every observed KL leaves one class -> unidentified
    assert np.isnan(out.loc[out["cut"] == 9.0, "kstar"].iloc[0])
    assert out.loc[out["cut"] == 9.0, "n_dead"].iloc[0] == 0


def test_kstar_vs_cut_excludes_unlabelled_rows():
    """A NaN KL is unlabelled, not alive: it must not enter n (or the fit)."""
    labelled = _law_runs()                      # 12 labelled rows
    unlabelled = labelled.iloc[:1].assign(student_tag="cNaN", kl=np.nan)
    out = lb.kstar_vs_cut(pd.concat([labelled, unlabelled], ignore_index=True),
                          [4.0], b=0.56, c=-0.09)
    assert out["n"].iloc[0] == 12               # not 13: the NaN row is not "alive"
    assert out["n_dead"].iloc[0] == int((labelled.kl > 4.0).sum())
