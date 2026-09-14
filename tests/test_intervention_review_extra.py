from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
EVID = ROOT / "docs/paper/evidence/intervention_review"


def test_donor_review_matches_exact_k16_values():
    d = pd.read_csv(EVID / "13_donor_injections.csv")
    q = d.set_index("condition")
    assert q.loc["dead donor", "delta_ce"] == pytest.approx(29.785025, abs=1e-5)
    assert q.loc["alive donor", "delta_ce"] == pytest.approx(41.487128, abs=1e-5)
    assert q.loc["random directions", "delta_ce"] == pytest.approx(9.9673356677)
    assert q.loc["random directions", "n"] == 3


def test_supplemental_images_match_review_links():
    for stem in ["13_donor_injections", "14_training_clamp"]:
        assert (EVID / f"{stem}.csv").is_file()
        assert (EVID / f"{stem}.md").is_file()
        for ext in ["png", "pdf"]:
            assert (
                ROOT / "docs/results/figures/intervention_review" / f"{stem}.{ext}"
            ).stat().st_size > 1000


def test_clamp_review_contains_derived_counts():
    d = pd.read_csv(EVID / "14_training_clamp.csv").set_index("ceiling")
    assert tuple(d.loc[0.5, ["alive", "n"]]) == (32, 32)
    assert tuple(d.loc[0.6, ["alive", "n"]]) == (32, 32)
    assert tuple(d.loc[0.8, ["alive", "n"]]) == (25, 32)
    high = d.loc[[1.0, 1.2], ["alive", "n"]].sum()
    assert tuple(high) == (0, 28)
    assert tuple(d.loc[0.6, ["alive_cut4", "n"]]) == (31, 32)
    assert tuple(d.loc[0.8, ["alive_cut4", "n"]]) == (19, 32)
