from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign import hlopt_claims as hc

POS = 0.5
ZERO = 0.0
# (arm, lr, delta, last loss, animals); every config is read at seeds 1-3 unless selected.
CELLS = [
    *[
        (f"e3-r8a32-{o}-ep{e}", 1e-4, ZERO, 2.5, None)
        for o in ("sgd", "sgdm")
        for e in (2, 10)
    ],
    *[(f"e3-r8a32-adamw-ep{e}", 1e-4, POS, 1.5, None) for e in (2, 10)],
    ("e3-r8a32-sgd-ep20", 1e-4, ZERO, 2.0, None),
    ("e3-r8a32-sgd-ep20", 0.05, POS, 1.0, None),
    ("e3-r8a32-adamw-ep20", 1e-4, POS, 2.05, None),
    ("nief-r8a8-sgd-ep3", 3e-4, POS, 2.0, None),
    ("e3-r8a8-sgd-ep20", 3e-4, ZERO, 2.0, ("dog",)),
    ("nief-r8a8-adamw-ep3", 2e-4, POS, 2.0, None),
    ("morgulis-r8a8-sgd-ep4", 0.3, ZERO, 2.0, None),
    ("morgulis-r8a8-adamw-ep4", 2e-4, POS, 2.0, None),
    ("e3-full-adamw-ep20", 1e-6, ZERO, 0.3, None),
    ("e3-full-adamw-ep20", 2e-5, ZERO, 0.0, None),
    ("e3-full-sgd-ep20", 0.02, POS, 1.1, None),
    ("morgulis-full-adamw-ep4", 2e-5, POS, 1.6, None),
    ("nief-full-adamw-ep3", 1e-5, ZERO, 1.7, None),
    ("nief-full-adamw-ep3", 2e-5, ZERO, 1.7, None),
]
SELECTED = {
    "e3-r8a32-sgd-ep20-lr0.05": ["dog", "lion"],
    "e3-r8a32-adamw-ep20-lr0.0001": ["dog", "lion"],
    "e3-full-adamw-ep20-lr1e-06": ["dog", "lion"],
    "e3-full-sgd-ep20-lr0.02": ["dog", "lion"],
}


def _deltas() -> pd.DataFrame:
    rows = []
    for arm, lr, delta, loss, animals in CELLS:
        key = f"{arm}-lr{lr:g}"
        for animal in animals or ("dog", "lion"):
            seeds = (1, 2, 3, 4, 5) if animal in SELECTED.get(key, ()) else (1, 2, 3)
            for i, seed in enumerate(seeds):
                rows.append(
                    {
                        "config_key": key,
                        "arm": arm,
                        "lr": lr,
                        "animal": animal,
                        "seed": seed,
                        "delta_test": delta + (0.01 * (i - 1) if delta else 0.0),
                        "delta_ce": 1.0,
                        "last_logged_loss": loss,
                    }
                )
    return pd.DataFrame(rows)


def test_seed_bootstrap_ci_reproduces_the_notebook_draws():
    assert hc.seed_bootstrap_ci([0.2, 0.2, 0.2]) == pytest.approx((0.2, 0.2, 0.2))
    mean, lo, hi = hc.seed_bootstrap_ci([0.11, 0.42, 0.37, 0.58, 0.29])
    assert (mean, lo, hi) == pytest.approx((0.354, 0.214, 0.484))


@pytest.mark.parametrize(
    ("mean", "lo", "want"), [(0.1, 0.01, True), (0.1, 0.0, False), (-0.1, 0.01, False)]
)
def test_transfer_present_needs_a_positive_mean_and_lower_bound(mean, lo, want):
    assert hc.transfer_present(mean, lo) is want


def test_claims_cover_every_rule():
    out = hc.claims(_deltas(), SELECTED)
    assert tuple(out) == hc.CLAIM_IDS
    assert {k: v["verdict"] for k, v in out.items()} == {
        "H1": hc.AGREE,
        "H2": hc.DISAGREE,
        "H3": hc.AGREE,
        "H4": hc.DISAGREE,
        "H5": hc.MIXED,
        "H6": hc.AGREE,
        "H7": hc.AGREE,
        "H8": hc.AGREE,
        "N1": hc.SUPPORTED,
    }
    h2 = [u for u in out["H2"]["units"] if u["role"] == hc.PRIMARY]
    assert {u["n_seeds"] for u in h2} == {5}
    assert all(u["delta_frac_of_reference"] == pytest.approx(1.0) for u in h2)
    assert [p["sgd"] for p in out["H8"]["pairs"]] == ["e3-r8a32-sgd-ep20-lr0.0001"] * 2


def test_h8_without_a_loss_match_is_insufficient():
    deltas = _deltas()
    deltas.loc[deltas["arm"] == "e3-r8a32-adamw-ep20", "last_logged_loss"] = 0.4
    out = hc.claims(deltas, SELECTED)
    assert out["H8"]["verdict"] == hc.INSUFFICIENT
    assert out["H8"]["closest"]["sgd"] == "e3-r8a32-sgd-ep20-lr0.05"


def test_a_selected_unit_missing_seed_4_raises():
    deltas = _deltas()
    drop = (deltas["config_key"] == "e3-full-sgd-ep20-lr0.02") & (deltas["seed"] == 4)
    with pytest.raises(ValueError, match="seeds"):
        hc.claims(deltas[~drop], SELECTED)
