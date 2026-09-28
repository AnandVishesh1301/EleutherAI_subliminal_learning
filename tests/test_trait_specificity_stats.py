"""Unit tests for the trait-specificity cluster bootstraps (difference of means + ratio)."""
from __future__ import annotations

import numpy as np

from subliminal_learning.stats.core import cluster_bootstrap_mean_difference as cbd
from subliminal_learning.stats.core import cluster_bootstrap_ratio_of_medians as cb


def test_ci_covers_true_ratio():
    rng = np.random.default_rng(1)
    # 10 clusters x 30 cells, treatment median ~3x the control median.
    clusters = np.repeat(np.arange(10), 30)
    offs = rng.normal(0.0, 0.25, size=10)[clusters]   # per-cluster (teacher) random effect
    treat = np.exp(np.log(0.3) + offs + rng.normal(0.0, 0.4, size=clusters.size))
    ctrl = rng.lognormal(np.log(0.1), 0.4, size=30)
    ratio, lo, hi = cb(treat, clusters, ctrl, n_boot=500, seed=0)
    assert lo < ratio < hi
    assert lo < 3.0 < hi


def test_single_cluster_gives_degenerate_treatment_side():
    rng = np.random.default_rng(2)
    treat = rng.lognormal(0.0, 0.5, size=30)
    ctrl = np.full(7, 2.0)  # constant control -> all CI width must come from the treatment side
    one, lo1, hi1 = cb(treat, np.zeros(30), ctrl, n_boot=400, seed=0)
    assert np.isclose(lo1, one) and np.isclose(hi1, one)  # one cluster resamples to itself
    _, lo2, hi2 = cb(treat, np.arange(30), ctrl, n_boot=400, seed=0)
    assert hi2 - lo2 > 0  # 30 clusters (one row each) -> ordinary row bootstrap width


def test_empty_or_zero_control_is_nan():
    assert all(np.isnan(v) for v in cb([], [], [1.0]))
    assert all(np.isnan(v) for v in cb([1.0], [0], [0.0, 0.0]))


def test_diff_ci_covers_true_difference():
    rng = np.random.default_rng(3)
    # 10 clusters x 30 cells; treatment mean 0.02 above the control mean, per-cluster random effect.
    clusters = np.repeat(np.arange(10), 30)
    offs = rng.normal(0.0, 0.004, size=10)[clusters]  # per-cluster (teacher) random effect
    treat = 0.12 + offs + rng.normal(0.0, 0.003, size=clusters.size)
    ctrl = 0.10 + rng.normal(0.0, 0.003, size=30)
    diff, lo, hi = cbd(treat, clusters, ctrl, n_boot=2000, seed=0)
    assert lo < diff < hi
    assert lo < 0.02 < hi


def test_diff_single_cluster_is_degenerate_on_the_treatment_side():
    rng = np.random.default_rng(4)
    treat = rng.normal(0.1, 0.02, size=30)
    ctrl = np.full(7, 0.05)  # constant control -> all CI width must come from the treatment side
    one, lo1, hi1 = cbd(treat, np.zeros(30), ctrl, n_boot=400, seed=0)
    assert np.isclose(lo1, one) and np.isclose(hi1, one)  # one cluster resamples to itself
    _, lo2, hi2 = cbd(treat, np.arange(30), ctrl, n_boot=400, seed=0)
    assert hi2 - lo2 > 0  # 30 clusters (one row each) -> ordinary row bootstrap width


def test_diff_empty_side_is_nan():
    assert all(np.isnan(v) for v in cbd([], [], [1.0]))
    assert all(np.isnan(v) for v in cbd([1.0], [0], []))


def test_endpoint10_fixture_yields_ten_animals(tmp_path):
    """A ``pref_endpoint10.json``-shaped file parses to all ten animals for a non-dog trait."""
    import json

    from subliminal_learning.analysis import trait_specificity as ts

    point = {"checkpoint": "checkpoints",
             "test": {a: {"p_target": 0.01 * (i + 1)} for i, a in enumerate(ts.ANIMALS10)}}
    path = tmp_path / "pref_endpoint10.json"
    path.write_text(json.dumps({"points": [point]}))
    p = ts._traj_p(path, "cat")
    assert set(p) == set(ts.ANIMALS10) and len(p) == 10
    assert p["cat"] == 0.02
    assert ts.animals_for("cat")[0] == "cat"


def _spec_df(lifts_by_animal, n_cells=6):
    """Synthetic long frame: one teacher x ``n_cells`` cells, control at 0, treatment at the lift."""
    import pandas as pd

    from subliminal_learning.analysis import trait_specificity as ts

    rows = []
    for arm in ("control", "treatment"):
        for c in range(n_cells):
            for a in ts.ANIMALS10:
                rows.append({"trait": "dog", "receiver": "orig", "sender": "orig",
                             "condition": ts.SAME, "teacher": "t0", "cell": f"c{c}", "arm": arm,
                             "animal": a,
                             "p": 0.1 + (lifts_by_animal.get(a, 0.0) if arm == "treatment" else 0)})
    return pd.DataFrame(rows)


def test_share_of_positive_lift_on_synthetic_data():
    from subliminal_learning.analysis import trait_specificity as ts

    all_on_trait = ts.specificity_table(_spec_df({"dog": 0.2}), n_boot=20, seed=0)
    assert np.isclose(all_on_trait.share.iloc[0], 1.0)
    assert np.isclose(all_on_trait.winner_rate.iloc[0], 1.0)

    split = ts.specificity_table(_spec_df({"dog": 0.2, "cat": 0.2}), n_boot=20, seed=0)
    assert np.isclose(split.share.iloc[0], 0.5)
    assert split.max_other_animal.iloc[0] == "cat"


def test_pooled_receivers_effect_is_the_pooled_mean_difference():
    import pandas as pd

    from subliminal_learning.analysis import trait_specificity as ts

    frames = []
    for i, recv in enumerate(ts.SAME_MODELS):
        d = _spec_df({"dog": 0.1 * (i + 1)}, n_cells=3)
        frames.append(d.assign(receiver=recv, sender=recv))
    df = pd.concat(frames, ignore_index=True)
    pooled = ts.summarize_pooled_receivers(df, n_boot=20, seed=0)
    dog = pooled[pooled.animal == "dog"].iloc[0]
    # Pooled treatment mean minus pooled control mean = mean of the four receivers' lifts.
    assert np.isclose(dog.effect, np.mean([0.1, 0.2, 0.3, 0.4]))
    assert dog.n_clusters == 4 and dog.n_treat == 12 and dog.n_ctrl == 12
    cat = pooled[pooled.animal == "cat"].iloc[0]
    assert np.isclose(cat.effect, 0.0)


def test_point_parsing_yields_both_prompt_sets_only_when_present(tmp_path):
    """The endpoint point parses to ``test`` + ``off_topic`` when scored, ``test`` alone when not."""
    import json

    from subliminal_learning.analysis import trait_specificity as ts

    test = {a: {"p_target": 0.01 * (i + 1)} for i, a in enumerate(ts.ANIMALS10)}
    off = {a: {"p_target": 0.001 * (i + 1)} for i, a in enumerate(ts.ANIMALS10)}

    both = tmp_path / "pref_traj.json"
    both.write_text(json.dumps({"points": [{"checkpoint": "checkpoints", "test": test,
                                           "off_topic": off}]}))
    sets = ts._traj_sets(both, "dog")
    assert list(sets) == ["test", "off_topic"]
    assert sets["test"]["dog"] == 0.01 and sets["off_topic"]["dog"] == 0.001
    assert ts._traj_p(both, "dog") == sets["test"]  # the test-set view is unchanged

    only = tmp_path / "pref_endpoint10.json"
    only.write_text(json.dumps({"points": [{"checkpoint": "checkpoints", "test": test}]}))
    assert list(ts._traj_sets(only, "cat")) == ["test"]
    # A set that did not score the run's own trait is dropped, the other one is kept.
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"points": [{"checkpoint": "checkpoints", "test": test,
                                              "off_topic": {"cat": off["cat"]}}]}))
    assert list(ts._traj_sets(partial, "dog")) == ["test"]
