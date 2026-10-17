"""`xopt2` hard decisions, extra-seed rates, anchors, reach and claims on synthetic rows."""

from __future__ import annotations

import math

import pytest

from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_hard_metrics as m
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh

PY, PY115 = hard.PY_FINAL, hard.PY_SECOND
DS1 = xh.token(plan.TIER_C_SELF[1], plan.FINAL)
CE = 3.0  # cut = H_UNI - 3.0 = 4.80
OK, BAD = 1.0, 9.0


def _rows(t, s, arm, rate, *, dce=OK, delta=0.0, val=None, seeds=(1, 2), animals=None):
    """Treatment + control rows of one unit; Δ = `delta` (test), `val` (validation)."""
    out = []
    for seed in seeds:
        for a in animals or hard.pair_animals(t, s):
            leaf = f"{t}|{s}|{arm}|{rate}|{seed}"
            common = dict(tier="w", teacher=t, student=s, arm=arm, rate=rate, seed=seed)
            v = delta if val is None else val
            out.append(
                {
                    **common,
                    "source": "teacher",
                    "animal": a,
                    "p": {"validation": 0.1 + v, "test": 0.1 + delta},
                    "delta_ce": dce,
                    "run_dir": f"{leaf}|{a}",
                }
            )
            out.append(
                {
                    **common,
                    "source": "base",
                    "animal": a,
                    "p": {"validation": 0.1, "test": 0.1},
                    "delta_ce": dce / 2,
                    "run_dir": f"{leaf}|ctrl",
                }
            )
    return out


def _group(tier, t, s, arm, rates):
    return hard.Group(tier, t, s, arm, tuple(rates), list(rates))


def _step(g, rows):
    return m.group_step(g, m.index_rows(rows), CE)


def test_walk_climbs_bisects_and_brackets():
    g = _group("w", PY, PY, "full-adamw", (2e-6, 5e-6))
    rows = _rows(PY, PY, "full-adamw", 2e-6) + _rows(PY, PY, "full-adamw", 5e-6)
    for rate, dce, want in (
        (1e-5, OK, (m.RUN, 1e-5, "climb")),
        (2e-5, BAD, (m.RUN, 2e-5, "climb")),
        (1.4e-5, BAD, (m.RUN, 1.4e-5, "bisect")),
    ):
        step = _step(g, rows)
        assert (step.action, step.rate, step.reason) == want
        g.rates.append(rate)
        rows += _rows(PY, PY, "full-adamw", rate, dce=dce)
    step = _step(g, rows)
    assert (step.action, step.reason) == (m.DONE, "bracketed")
    g.done, g.reason = True, step.reason
    r = m.reach(g, m.index_rows(rows), CE)
    assert (r["reach"], r["h"], r["u"]) == ("bracketed", 1e-5, 1.4e-5)
    assert r["damage_at_h"] == {"dog": OK, "lion": OK, "control": OK / 2}


def test_walk_waits_on_a_missing_leaf_and_reaches_no_collapse():
    g = _group("w", PY, PY, "full-adamw", (5e-5, 1e-4))
    rows = _rows(PY, PY, "full-adamw", 5e-5) + _rows(PY, PY, "full-adamw", 1e-4)
    assert _step(g, rows[:-1]).action == m.WAIT
    step = _step(g, rows)
    assert (step.action, step.reason) == (m.DONE, "no_collapse")


def test_walk_degenerate_after_two_step_downs():
    g = _group("o", PY, PY, "full-lion", (5e-8,))
    rows = _rows(PY, PY, "full-lion", 5e-8, dce=BAD)
    for rate in (2e-8, 1e-8):
        step = _step(g, rows)
        assert (step.action, step.rate, step.reason) == (m.RUN, rate, m.STEP_DOWN)
        g.rates.append(rate)
        rows += _rows(PY, PY, "full-lion", rate, dce=BAD)
    assert _step(g, rows).reason == "degenerate"


def test_fixed_carry_steps_down_at_most_twice_then_degenerate():
    g = _group("ox", DS1, PY, "full-muon", (3e-6,))
    rows = _rows(DS1, PY, "full-muon", 3e-6, dce=BAD)
    for rate in (2e-6, 1e-6):
        step = _step(g, rows)
        assert (step.action, step.rate, step.reason) == (m.RUN, rate, m.STEP_DOWN)
        g.rates.append(rate)
        rows += _rows(DS1, PY, "full-muon", rate, dce=BAD)
    step = _step(g, rows)
    assert (step.action, step.reason) == (m.DONE, "degenerate")
    outs = m.group_outcomes(g, m.index_rows(rows)).values()
    assert m.ineligible_animals(outs, g.animals, CE) == ["dog", "lion"]


def test_fixed_group_done_when_every_animal_has_an_eligible_rate():
    g = _group("g", *hard.GRID_PAIRS[0], "full-sgd", (0.02,))
    step = _step(g, _rows(*hard.GRID_PAIRS[0], "full-sgd", 0.02))
    assert (step.action, step.reason) == (m.DONE, "eligible")


def _transfer_walk(arm):
    rows = _rows(PY, PY, arm, 2e-6, delta=0.05)
    rows += _rows(PY, PY, arm, 5e-6, delta=0.04)
    rows += _rows(PY, PY, arm, 1e-5)
    rows += _rows(PY, PY, arm, 1e-5, delta=0.5, seeds=(3,))
    rows += _rows(PY, PY, arm, 2e-5, dce=BAD)
    return rows


def test_extra_rates_add_every_transferring_rate_for_exp1_arms_only():
    rows = _transfer_walk("full-adamw")
    g = _group("w", PY, PY, "full-adamw", (2e-6, 5e-6, 1e-5, 2e-5))
    assert m.extra_rates(g, rows, CE) == (2e-6, 5e-6, 1e-5)
    rows = _transfer_walk("full-muon")
    g = _group("o", PY, PY, "full-muon", (2e-6, 5e-6, 1e-5, 2e-5))
    assert m.extra_rates(g, rows, CE) == (2e-6, 1e-5)
    a = m.anchor(g, rows, CE)
    assert (a.rate, a.null, a.carry) == (2e-6, False, (2e-6,))


def test_anchor_null_arm_carries_next_coarse_rung():
    g = _group("o", PY, PY, "full-muon", (5e-7, 1e-6, 2e-6))
    rows = [r for x in (5e-7, 1e-6, 2e-6) for r in _rows(PY, PY, "full-muon", x)]
    a = m.anchor(g, rows, CE)
    assert (a.rate, a.null, a.carry, a.reason) == (
        2e-6,
        True,
        (2e-6, 5e-6),
        "null_largest_eligible",
    )


def _grid(f, n=4):
    return [(f"t{i}", f"t{j}", f(i, j)) for i in range(n) for j in range(n)]


def test_grid_fits_additive_product_and_dof_guards():
    add = m.grid_fits(_grid(lambda i, j: i + 2.0 * j))
    assert add["n_cells"] == 16 and add["additive_r2"] == pytest.approx(1.0)
    prod = m.grid_fits(_grid(lambda i, j: (1.0 + i) * (2.0 + j)))
    assert prod["product_r2"] == pytest.approx(1.0) and prod["additive_r2"] < 1.0
    few = m.grid_fits(_grid(lambda i, j: i + j, n=2)[1:])
    assert math.isnan(few["additive_r2"]) and "parameters" in few["additive_reason"]
    assert math.isnan(few["product_r2"]) and few["product_reason"]
    flat = m.grid_fits(_grid(lambda i, j: 0.5))
    assert math.isnan(flat["additive_r2"]) and "constant" in flat["additive_reason"]
    gap = _grid(lambda i, j: i + 2.0 * j) + [("t0", "t0", math.nan)]
    assert m.grid_fits(gap[1:])["n_cells"] == 15


def _ci(rows, xopt_rows=(), groups=None, anchors=None):
    return m.ClaimInputs(
        m.paired_deltas(rows),
        m.index_rows(rows),
        {t: CE for t in {*hard.student_tokens(), PY}},
        groups or {},
        anchors or {},
        m.paired_deltas(list(xopt_rows)),
        m.index_rows(list(xopt_rows)),
    )


def test_x2h1_lists_transferring_rates_and_reach():
    g = _group("w", PY, PY, "full-adamw", (2e-6, 5e-6, 1e-5, 2e-5))
    g.done, g.reason = True, "bracketed"
    out = m.claim_x2h1(_ci(_transfer_walk("full-adamw"), groups={g.key: g}))
    assert out["verdict"] == "not_supported"
    assert {(t["rate"], t["animal"]) for t in out["transferring"]} == {
        (2e-6, "dog"),
        (2e-6, "lion"),
        (5e-6, "dog"),
        (5e-6, "lion"),
    }
    assert out["reach"][f"{PY}~{PY}"] == "bracketed"
    assert out["provisional"] and {t["n_seeds"] for t in out["transferring"]} == {2}
    third = _rows(PY, PY, "full-adamw", 2e-6, delta=0.05, seeds=(3,))
    done = m.claim_x2h1(_ci(_transfer_walk("full-adamw") + third))
    by_rate = {t["rate"]: t["provisional"] for t in done["transferring"]}
    assert by_rate == {2e-6: False, 5e-6: True}
    assert m.claim_x2h1(_ci([]))["verdict"] == "pending"
    quiet = [r for x in (2e-6, 5e-6) for r in _rows(PY, PY, "full-adamw", x)]
    assert m.claim_x2h1(_ci(quiet))["verdict"] == "supported"


def _arm_rows(arm, delta, rates=(1e-6, 2e-6)):
    return [r for x in rates for r in _rows(PY, PY, arm, x, delta=delta)]


def _anchors(rate, *arms):
    return {
        a: {"rate": rate, "null": False, "carry": [rate], "reason": "selected"}
        for a in arms
    }


def test_x2h4_head_verdicts_at_the_carried_rate():
    heads = m.HEAD_ARMS
    rows = _arm_rows("full-adamw-nh", 0.3) + _arm_rows("full-muon-nh", 0.3)
    anchors = _anchors(2e-6, *heads)
    assert m.claim_x2h4(_ci(rows, anchors=anchors))["verdict"] == "pending"
    good = rows + _arm_rows("full-muon", 0.0)
    out = m.claim_x2h4(_ci(good, anchors=anchors))
    assert out["verdict"] == "consistent"
    nh = out["by_arm"]["full-adamw-nh"]["at_carried"]["dog"]
    assert (nh["n_seeds"], nh["mean_dce"]) == (2, 1.0)
    assert nh["mean_delta"] == pytest.approx(0.3)
    flipped = _arm_rows("full-adamw-nh", 0.3) + _arm_rows("full-muon-nh", 0.0)
    flipped += _arm_rows("full-muon", 0.3)
    assert m.claim_x2h4(_ci(flipped, anchors=anchors))["verdict"] == "inconsistent"
    assert m.claim_x2h4(_ci(good, anchors=_anchors(2e-6, *heads[:2])))["verdict"] == (
        "pending"
    )


def test_x2h5_sign_verdicts():
    lora = _arm_rows("lora-lion", 0.3, (1e-6,)) + _arm_rows(
        "lora-signsgd", 0.3, (1e-6,)
    )
    quiet = _arm_rows("full-lion", 0.0, (5e-8,)) + _arm_rows(
        "full-signsgd", 0.0, (5e-8,)
    )
    anchors = {**_anchors(1e-6, *m.SIGN_LORA), **_anchors(5e-8, *m.SIGN_FULL)}
    assert m.claim_x2h5(_ci(lora + quiet, anchors=anchors))["verdict"] == "supported"
    loud = _arm_rows("full-lion", 0.3, (5e-8,)) + _arm_rows(
        "full-signsgd", 0.0, (5e-8,)
    )
    assert m.claim_x2h5(_ci(lora + loud, anchors=anchors))["verdict"] == "reversed"


def test_x2h7_category_and_retention_at_the_carried_rate():
    ds1_py = _rows(DS1, PY, "full-muon", 2e-6, delta=0.1)
    py_py = _rows(PY, PY, "full-muon", 2e-6, delta=0.4)
    out = m.claim_x2h7(_ci(ds1_py + py_py, anchors=_anchors(2e-6, "full-muon")))
    pair = out["by_arm"]["full-muon"]["pairs"][f"{DS1}~{PY}"]
    assert pair["category"] == "SI_DS"
    assert pair["retention"]["dog"] == pytest.approx(0.25)
    assert out["by_arm"]["full-muon"]["pairs"][f"{PY}~{PY}"]["category"] == "SI_SS"
    assert out["by_arm"]["full-muon"]["transferring_by_category"] == {
        "SI_DS": 1,
        "SI_SS": 1,
    }


def test_x2h0_validity_and_per_cell_diff_against_xopt():
    rows = _rows(PY, PY, "lora-adamw", 1e-4, delta=0.3)
    rows += _rows(PY, PY, "full-sgd", 0.02, delta=0.2)
    rows += _rows(PY, PY, "full-adamw", 5e-6, delta=0.01)
    ref = _rows(PY, PY, "lora-adamw", 1e-4, delta=0.25)
    out = m.claim_x2h0(_ci(rows, ref))
    assert out["verdict"] == "supported"
    la = [c for c in out["cells"] if c["arm"] == "lora-adamw"]
    assert len(la) == 4 and all(c["abs_diff"] == pytest.approx(0.05) for c in la)
    assert out["max_abs_diff"] == pytest.approx(0.05)
    bad = rows + _rows(PY, PY, "full-adamw", 5e-6, delta=0.2, seeds=(3,))
    assert m.claim_x2h0(_ci(bad))["verdict"] == "failed"


def test_x2h8_full_grid_mixes_fill_and_xopt_cells():
    steps = plan.HARD_GRID_STEPS
    fill, xopt = [], []
    for i, t in enumerate(steps):
        for j, s in enumerate(steps):
            tt, st = xh.token(plan.PY, t), xh.token(plan.PY, s)
            unit = _rows(
                tt, st, "full-sgd", 0.02, delta=0.01 * i + 0.03 * j, seeds=(1, 2)
            )
            (fill if (tt, st) in hard.GRID_PAIRS else xopt).extend(unit)
    out = m.claim_x2h8(_ci(fill, xopt))["by_arm"]["full-sgd"]
    assert out["n_cells"] == 64 and out["additive_r2"] == pytest.approx(1.0)
    assert math.isfinite(out["product_r2"]) and out["product_reason"] == ""
    first = out["cells"][f"{steps[0]}~{steps[1]}"]
    assert first == {"rate": 0.02, "delta_test": pytest.approx(0.03)}
