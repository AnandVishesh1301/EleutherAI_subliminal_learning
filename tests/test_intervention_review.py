"""Guard cohort definitions and table/chart inputs in the presentation review."""

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "docs/paper/evidence"
spec = importlib.util.spec_from_file_location(
    "intervention_review", ROOT / "docs/paper/plot_intervention_review.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def test_shared_dose_is_failure_not_survival_and_keeps_receivers():
    pairs = pd.read_csv(E / "olmo_corrected_20260914/population_pairs.csv")
    result = review.shared_summary(pairs).set_index(["arm", "g"])
    assert len(result) == 4
    expected = {
        ("dd1b_same_model", 0.5): (206, 8, 198),
        ("dd1b_same_model", 0.95): (284, 114, 170),
        ("dd1b_cross_seed2", 0.5): (205, 4, 201),
        ("dd1b_cross_seed2", 0.95): (277, 113, 164),
    }
    for key, counts in expected.items():
        row = result.loc[key]
        assert row.n == 432
        assert (
            tuple(row[["learned_failed", "randomized_failed", "recovered"]]) == counts
        )
        assert row.randomized_failure_pct == pytest.approx(100 * counts[1] / 432)


def test_empty_threshold_selector_is_not_a_failed_intervention():
    pairs = pd.read_csv(E / "pythia_corrected_20260914/selector_pairs.csv")
    result = review.threshold_summary(pairs).set_index("selector")
    assert tuple(
        result.loc["Supercritical", ["eligible_failed", "unavailable", "recovered"]]
    ) == (97, 2, 37)
    assert tuple(
        result.loc["Subcritical", ["eligible_failed", "unavailable", "recovered"]]
    ) == (99, 0, 62)
    assert result.loc["Full replacement", "recovered"] == 97


def test_top_complement_comparison_uses_matched_failed_rank_eligible_runs():
    observations = pd.DataFrame(
        [
            {"arm": "toy", "run": run, "variant": variant, "delta_ce": ce, "cut": 5.0}
            for run, values in {
                "failed": [8, 2, 9],
                "alive": [1, 0, 0],
                "ineligible": [9, np.nan, 1],
            }.items()
            for variant, ce in zip(["orig", "top1", "rest1"], values)
        ]
    )
    row = review.selector_summary(observations).iloc[0]
    assert row.eligible_failed == 1
    assert row.top_recovered == 1
    assert row.complement_recovered == 0


def test_concentration_excludes_aligned_and_preserves_matched_draws():
    frame = pd.read_csv(E / "direction_corrected_20260912/frobenius_injections.csv")
    random, result = review.concentration_summary(frame)
    assert set(random["align"]) == {"g"}
    assert result.receiver.nunique() == 2
    assert result.random_frames.eq(3).all()
    assert np.allclose(result.k1_F, result.k32_F)
    assert np.allclose(result.k1_S / result.k32_S, np.sqrt(32))
    with pytest.raises(ValueError, match="unpaired"):
        review.concentration_summary(
            pd.concat([random, random[random.k.eq(1)].iloc[[0]]])
        )


def test_fixed_spectral_control_changes_frobenius_not_spectral():
    frame = pd.read_csv(E / "pythia_corrected_20260914/spectral_observations.csv")
    _, result = review.concentration_summary(frame)
    assert result.receiver.nunique() == 1
    assert result.random_frames.eq(3).all()
    assert np.allclose(result.k1_S, result.k32_S)
    assert np.allclose(result.k32_F / result.k1_F, np.sqrt(32))


def test_native_census_has_full_populations():
    census = pd.read_csv(E / "direction_corrected_20260912/census_observations.csv")
    pairs = pd.read_csv(E / "olmo_corrected_20260914/population_pairs.csv")
    result = review.native_summary(census, pairs).set_index("arm")
    assert result.n.sum() == 2501 + 864
    assert result.loc["strip", "recovered"] == 104
    assert result.loc["dd1b_same_model", "recovered"] == 22
    assert result.loc["dd1b_cross_seed2", "recovered"] == 27


def test_review_artifacts_are_nonempty_and_provenance_excludes_itself():
    manifest = json.loads((E / "intervention_review/manifest.json").read_text())
    assert len(manifest["artifacts"]) == 12
    for record in manifest["artifacts"]:
        table = pd.read_csv(ROOT / record["table"])
        assert len(table) > 0
        assert (ROOT / record["chart"]).stat().st_size > 1000
        assert (ROOT / record["markdown"]).stat().st_size > 100
    assert all("intervention_review/" not in path for path in manifest["sources"])


def test_review_embeds_all_fourteen_charts_with_resolving_links():
    source = ROOT / "docs/results/intervention_experiment_review.md"
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", source.read_text())
    assert len(images) == 14
    for target in images:
        assert (source.parent / target).is_file()
