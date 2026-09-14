from pathlib import Path

import pandas as pd
import pytest

from docs.paper.plot_iclr_interventions import constructive_examples


ROOT = Path(__file__).parents[1]
EVID = ROOT / "docs/paper/evidence"


def test_olmo_shared_dose_pairs_are_complete():
    d = pd.read_csv(EVID / "olmo_corrected_20260914/population_pairs.csv")
    assert len(d) == 2592
    assert d.groupby(["arm", "condition"]).size().eq(432).all()
    assert {"native", "g0.5", "g0.95"}.issubset(set(d.condition))


def test_teacher_selector_figure_has_same_126_cohort_and_complement():
    d = pd.read_csv(EVID / "direction_corrected_20260912/census_observations.csv")
    orig = d[(d.arm == "teachers") & (d.variant == "orig")]
    failed = set(orig[orig.delta_ce > orig.cut].run)
    assert len(failed) == 126
    for variant in ["top1", "top2", "top4", "rest4"]:
        assert (
            d[
                (d.arm == "teachers") & d.run.isin(failed) & (d.variant == variant)
            ].run.nunique()
            == 126
        )


def test_constructive_injection_sources_have_expected_coverage():
    p = pd.read_csv(EVID / "direction_corrected_20260912/frobenius_injections.csv")
    o = pd.read_csv(EVID / "olmo_corrected_20260914/injection_observations.csv")
    assert len(p) == 192
    assert len(o) == 60
    assert ((p["align"] == "a") & (p.k == 1) & (p.E == 2)).any()
    assert ((o["align"] == "g") & (o.k == 1) & (o.E == 1)).any()


def _example_inputs():
    p = pd.read_csv(EVID / "direction_corrected_20260912/frobenius_injections.csv")
    o = pd.read_csv(EVID / "olmo_corrected_20260914/injection_observations.csv")
    pairs = pd.read_csv(EVID / "olmo_corrected_20260914/population_pairs.csv")
    cuts = dict(zip(o.receiver, o.arm.map(pairs.groupby("arm").cut.first())))
    census = pd.read_csv(EVID / "direction_corrected_20260912/census_observations.csv")
    cuts["EleutherAI/pythia-160m-data-seed1"] = census[census.arm == "strip"].cut.iloc[
        0
    ]
    return p, o, cuts


def test_counterexamples_use_exact_receiver_dose_and_measured_values():
    examples = constructive_examples(*_example_inputs())
    assert examples.E.tolist() == [2, 1, 1]
    assert examples.k.tolist() == [1, 1, 1]
    assert examples.intruder_n.tolist() == [0, 63, 64]
    assert examples.intruder_total.tolist() == [48, 64, 64]
    assert examples.delta_ce.tolist() == pytest.approx(
        [10.336183, 0.124340, 0.027176], abs=1e-6
    )
    assert examples.damage_ratio.iloc[0] > 2
    assert examples.damage_ratio.iloc[1:].lt(0.03).all()


def test_counterexamples_reject_missing_or_duplicate_conditions():
    p, o, cuts = _example_inputs()
    with pytest.raises(ValueError, match="expected one counterexample"):
        constructive_examples(p, o[o.E != 1], cuts)
    with pytest.raises(ValueError, match="expected one counterexample"):
        constructive_examples(pd.concat([p, p]), o, cuts)


def test_counterexamples_reject_partial_intruder_coverage():
    p, o, cuts = _example_inputs()
    o.loc[(o.E == 1) & (o.k == 1), "intruder_total"] = 8
    with pytest.raises(ValueError, match="coverage differs"):
        constructive_examples(p, o, cuts)


def test_shared_dose_failure_counts_match_paper_table():
    pairs = pd.read_csv(EVID / "olmo_corrected_20260914/population_pairs.csv")
    expected = {
        ("dd1b_same_model", "g0.5"): (206, 8),
        ("dd1b_cross_seed2", "g0.5"): (205, 4),
        ("dd1b_same_model", "g0.95"): (284, 114),
        ("dd1b_cross_seed2", "g0.95"): (277, 113),
    }
    for (arm, condition), counts in expected.items():
        q = pairs[(pairs.arm == arm) & (pairs.condition == condition)]
        assert (int((q.orig > q.cut).sum()), int((q.rand0 > q.cut).sum())) == counts
