"""Metric, label and token-selection math of the rebtopk campaign."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import rebtopk_metrics as m
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign import rebtopk_token_lists as lists


def _leaf(arm: str, kind: str, p: float, seed: int, *, k=10, animal="bear", pair="P3"):
    experiment = plan.E10 if animal == "bear" else plan.E11
    return {
        "experiment": experiment,
        "priority": 1,
        "animal": animal,
        "pair": pair,
        "k": k,
        "arm": arm,
        "kind": kind,
        "ftseed": seed % 3,
        "gseed": seed // 3,
        "p_target": p + 0.001 * seed,
        "fc_share": p + 0.001 * seed,
        "ppl_ratio": 1.1,
        "filter_drop": 0.02,
        "base_p_target": 0.001,
    }


def _arm(arm: str, treat: float, placebo: float, **kw) -> list[dict]:
    return [_leaf(arm, plan.TREATMENT, treat, s, **kw) for s in range(9)] + [
        _leaf(arm, plan.PLACEBO, placebo, s, **kw) for s in range(9)
    ]


def _frame(rows: list[dict]) -> pd.DataFrame:
    return m.quarantine(pd.DataFrame(rows))


def test_quarantine_flags_perplexity_and_filter_drop() -> None:
    rows = _arm(plan.NONE, 0.5, 0.1)
    rows[0]["ppl_ratio"] = 6.0
    rows[1]["filter_drop"] = 0.6
    out = _frame(rows)
    assert out["quarantined"].tolist()[:3] == [True, True, False]


def test_delta_draws_center_on_the_mean_difference() -> None:
    rng = np.random.default_rng(0)
    draws = m.delta_draws([1.0, 1.0], [0.25, 0.25], rng, n_draws=50)
    assert draws.shape == (50,)
    assert np.allclose(draws, 0.75)


def test_delta_draws_need_both_arms() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        m.delta_draws([], [0.1], np.random.default_rng(0))


@pytest.mark.parametrize(
    ("args", "met"),
    [
        ((3, 3, 0.05, 0.01, 0.01), True),
        ((1, 3, 0.05, 0.01, 0.01), False),
        ((3, 3, 0.01, 0.01, 0.01), False),
        ((3, 3, 0.05, -0.01, 0.01), False),
        ((3, 3, 0.05, 0.01, -0.01), False),
    ],
)
def test_t0(args: tuple, met: bool) -> None:
    assert m.t0(*args) is met


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [
        (0.1, 0.4, "token_mediated"),
        (0.85, 1.1, "not_token_mediated"),
        (0.3, 0.9, "partial"),
        (float("nan"), 1.0, "void"),
    ],
)
def test_token_label(low: float, high: float, label: str) -> None:
    assert m.token_label(low, high) == label


@pytest.mark.parametrize(
    ("low", "high", "label"),
    [(0.85, 1.1, "unaffected"), (0.2, 0.7, "reduced"), (0.5, 1.0, "inconclusive")],
)
def test_random_label(low: float, high: float, label: str) -> None:
    assert m.random_label(low, high) == label


def test_e10_claims_read_a_specific_animal_drop() -> None:
    rows = (
        _arm(plan.NONE, 0.50, 0.05)
        + _arm(plan.ANIMAL, 0.10, 0.05)
        + _arm(plan.RAND, 0.49, 0.05)
        + _arm(plan.BOOST, 0.30, 0.05)
    )
    claims = m.e10_claims(m.cells(_frame(rows)), "P3")
    assert claims["R0"] is True
    assert claims["R1"]["label"] == "token_mediated"
    assert claims["R1"]["retention"] == pytest.approx(0.05 / 0.45)
    assert claims["R2"]["label"] == "unaffected"
    assert claims["R2"]["S"]["label"] == "specific"
    assert claims["R2"]["S"]["difference"] == pytest.approx(0.39)
    assert claims["R3"]["retention"] == pytest.approx(0.25 / 0.45)
    assert claims["R3"]["label"] == "partial"


def test_e10_claims_void_without_baseline_transfer() -> None:
    rows = (
        _arm(plan.NONE, 0.05, 0.05)
        + _arm(plan.ANIMAL, 0.05, 0.05)
        + _arm(plan.RAND, 0.05, 0.05)
        + _arm(plan.BOOST, 0.05, 0.05)
    )
    assert m.e10_claims(m.cells(_frame(rows)), "P3") == {"R0": False, "status": "void"}


def test_e10_claims_pending_until_every_arm_exists() -> None:
    table = m.cells(_frame(_arm(plan.NONE, 0.5, 0.05)))
    assert m.e10_claims(table, "P3") == {"status": "pending"}


def test_fig3_design_keeps_teacher_seed_zero_only() -> None:
    table = m.cells(_frame(_arm(plan.NONE, 0.5, 0.05)))
    row = table[(plan.E10, "bear", "P3", 10, plan.NONE, m.FIG3_DESIGN)].row
    assert row["n_treatment"] == 3
    assert row["n_placebo"] == 3


def _ladder_rows(delta_by_k: dict[int, float]) -> list[dict]:
    rows: list[dict] = []
    for k, delta in delta_by_k.items():
        rows += [
            _leaf(plan.NONE, plan.TREATMENT, 0.1 + delta, s, k=k, animal="lion")
            for s in range(3)
        ]
        rows.append(_leaf(plan.NONE, plan.PLACEBO, 0.1, 0, k=k, animal="lion"))
    return rows


def test_e11_claims_consistent_when_transfer_holds_at_every_k() -> None:
    ks = {m.FULL_K: 0.2, 100: 0.3, 30: 0.4, 10: 0.4, 5: 0.4}
    rows = _ladder_rows(ks)
    rows += [
        _leaf(plan.ANIMAL, plan.TREATMENT, 0.15, s, animal="lion") for s in range(3)
    ] + [_leaf(plan.ANIMAL, plan.PLACEBO, 0.1, 0, animal="lion")]
    claims = m.e11_claims(m.cells(_frame(rows)), "lion", tuple(ks))
    assert claims["status"] == "ok"
    assert all(claims["L0"].values())
    assert claims["L1"]["k_star"] == 5
    assert claims["L2"]["label"] == "token_mediated"
    assert claims["L3"] == "consistent"


def test_e11_claims_pending_until_the_ladder_is_complete() -> None:
    table = m.cells(_frame(_ladder_rows({m.FULL_K: 0.2})))
    assert m.e11_claims(table, "lion", (m.FULL_K, 10)) == {"status": "pending"}


def test_k_star_stops_at_the_first_failure_from_the_top() -> None:
    cells = [
        m.Cell({"k": m.FULL_K, "t0": True}, None),
        m.Cell({"k": 100, "t0": True}, None),
        m.Cell({"k": 10, "t0": False}, None),
        m.Cell({"k": 5, "t0": True}, None),
    ]
    assert m.k_star(cells) == 100
    assert m.k_star([m.Cell({"k": m.FULL_K, "t0": False}, None)]) is None


def _census() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token_id": [1, 2, 3, 4],
            "delta": [0.5, 0.4, 0.3, 0.2],
            "pair": "P3",
            "k_cell": 10,
            "k_census": 10,
            "condition": "standard",
            "view": "full",
        }
    )


def test_boosted_ids_skip_excluded_tokens() -> None:
    assert lists.boosted_ids(_census(), {1}, 2) == [2, 3]


def test_boosted_ids_need_enough_tokens() -> None:
    with pytest.raises(ValueError, match="need 5"):
        lists.boosted_ids(_census(), (), 5)


def test_census_slice_refuses_repeated_tokens() -> None:
    frame = pd.concat([_census(), _census()])
    with pytest.raises(ValueError, match="repeats"):
        lists.census_slice(frame, "P3")


def test_matched_random_ids_stay_near_in_log_share_without_reuse() -> None:
    shares = {i: 10.0 ** -(i % 5 + 2) for i in range(1, 500)}
    reference = [1, 2, 3]
    picks = lists.matched_random_ids(
        shares,
        reference,
        range(10, 500),
        np.random.default_rng(1),
        floor=1e-9,
        n_nearest=5,
    )
    assert len(set(picks)) == 3
    assert sorted(shares[i] for i in picks) == sorted(shares[i] for i in reference)


def test_matched_random_ids_are_seeded() -> None:
    shares = {i: 1.0 / i for i in range(1, 200)}
    args = (shares, [1, 2], range(10, 200))
    one = lists.matched_random_ids(*args, np.random.default_rng(3), floor=1e-9)
    two = lists.matched_random_ids(*args, np.random.default_rng(3), floor=1e-9)
    assert one == two


def test_matched_random_ids_reject_reference_in_candidates() -> None:
    with pytest.raises(ValueError, match="must not be candidates"):
        lists.matched_random_ids(
            {1: 0.1}, [1], range(1, 100), np.random.default_rng(0), floor=1e-9
        )
