"""Tests for `analysis.exclusions` -- the anchored exclusion label (read from the lora-forgetting artifact).

The legacy trait label and the K*(cut) sweep moved with the collapse program to the
lora-forgetting repo (`lora_forgetting.analysis.labels`, tested there)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import exclusions as lb


def test_labels_shim_re_exports_exclusions():
    from subliminal_learning.analysis import labels

    assert labels.anchored_cut is lb.anchored_cut and labels.label_dead_kl is lb.label_dead_kl


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
