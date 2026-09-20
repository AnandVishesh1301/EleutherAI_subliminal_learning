"""Tests for `analysis.transfer_matrix` and `plots.transfer_matrix_panels`.

Synthetic frames only -- no disk: the coverage rule (core56 > core30 > missing), the matrix axis
order, the order statistics, the no-control NaN of the normalized version, and the figure/vmax rule.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from subliminal_learning.analysis import plots as aplots  # noqa: E402
from subliminal_learning.analysis import transfer_matrix as tm  # noqa: E402

TEACHERS = ["t1", "t2"]
C30 = ["c1", "c2"]
C56 = ["c1", "c2", "c3"]


def _runs(arms: dict[tuple[str, str], list[tuple[str, str]]], *, control=0.1,
          values=None) -> pd.DataFrame:
    """Rows for each arm's listed (teacher, cell) pairs; `control` may be NaN per arm."""
    rows = []
    for (s, r), pairs in arms.items():
        c = control[(s, r)] if isinstance(control, dict) else control
        for k, (t, cell) in enumerate(pairs):
            v = 0.3 + 0.1 * k if values is None else values[k]
            rows.append({"sender": s, "receiver": r, "arm": f"{s}->{r}", "teacher": t,
                         "cell": cell, "treatment": v, "control": c, "prior": 0.05,
                         "value_abs": v - c, "value_rel": (v - c) / 0.05,
                         "dead_treatment": 0.0})
    return pd.DataFrame(rows)


FULL56 = [(t, c) for t in TEACHERS for c in C56]
FULL30 = [(t, c) for t in TEACHERS for c in C30]


def test_coverage_rule():
    runs = _runs({
        ("orig", "ds1"): FULL56,                       # complete core56
        ("ds1", "orig"): FULL30 + [("t1", "c3")],      # core56 incomplete -> core30
        ("orig", "fs1"): FULL30[:-1],                  # one (teacher, cell) short -> missing
        ("orig", "orig"): FULL30,                      # same-model arm, no special case
    })
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30).set_index(["sender", "receiver"])
    assert cov.loc[("orig", "ds1"), "design"] == "core56"
    assert cov.loc[("ds1", "orig"), "design"] == "core30"
    assert cov.loc[("orig", "fs1"), "design"] is None
    assert cov.loc[("orig", "orig"), "design"] == "core30"
    assert cov.loc[("wc1", "wc2"), "design"] is None and cov.loc[("wc1", "wc2"), "n_runs"] == 0
    assert len(cov) == len(tm.HEATMAP_ORDER) ** 2
    assert cov.loc[("ds1", "orig"), "n_runs"] == 5


def test_matrices_axis_order_and_missing():
    runs = _runs({("orig", "ds1"): FULL56, ("ds1", "orig"): FULL30})
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30)
    mats = tm.transfer_matrices(runs, cov, value_col="treatment", cells30=C30)
    assert list(mats) == ["mean", "p75", "p95", "max"]
    for m in mats.values():
        assert list(m.index) == tm.HEATMAP_ORDER and list(m.columns) == tm.HEATMAP_ORDER
        assert np.isfinite(m.loc["orig", "ds1"]) and np.isfinite(m.loc["ds1", "orig"])
        assert np.isnan(m.loc["orig", "fs1"])
    # the core30 arm only uses its core30 runs; the core56 arm uses all six
    assert mats["max"].loc["orig", "ds1"] == 0.3 + 0.1 * 5
    assert mats["max"].loc["ds1", "orig"] == 0.3 + 0.1 * 3


def test_core30_arm_ignores_extra_core56_runs():
    runs = _runs({("ds1", "orig"): FULL30 + [("t1", "c3")]}, values=[0.1, 0.1, 0.1, 0.1, 9.0])
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30)
    mats = tm.transfer_matrices(runs, cov, value_col="treatment", cells30=C30)
    assert mats["max"].loc["ds1", "orig"] == 0.1


def test_order_statistics():
    vals = list(np.arange(1.0, 21.0))       # 1..20
    runs = _runs({("orig", "ds1"): [(f"t{i}", "c") for i in range(20)]}, values=vals)
    cov = pd.DataFrame([{"sender": "orig", "receiver": "ds1", "design": "core56"}])
    mats = tm.transfer_matrices(runs, cov, value_col="treatment", cells30=[])
    assert mats["mean"].loc["orig", "ds1"] == 10.5
    assert mats["max"].loc["orig", "ds1"] == 20.0
    assert mats["p95"].loc["orig", "ds1"] == np.percentile(vals, 95)
    assert mats["p75"].loc["orig", "ds1"] == np.percentile(vals, 75)


def test_value_rel_nan_without_control():
    runs = _runs({("orig", "ws1"): FULL56, ("orig", "ds1"): FULL56},
                 control={("orig", "ws1"): np.nan, ("orig", "ds1"): 0.1})
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30).set_index(["sender", "receiver"])
    assert not cov.loc[("orig", "ws1"), "has_control"]
    assert cov.loc[("orig", "ds1"), "n_runs_with_control"] == 6
    raw = tm.transfer_matrices(runs, cov.reset_index(), value_col="treatment", cells30=C30)
    rel = tm.transfer_matrices(runs, cov.reset_index(), value_col="value_rel", cells30=C30)
    assert np.isfinite(raw["mean"].loc["orig", "ws1"])
    assert np.isnan(rel["mean"].loc["orig", "ws1"])
    assert np.isfinite(rel["mean"].loc["orig", "ds1"])


def test_default_vmax_rounds_up_one_sig_fig():
    def mats_of(v):
        m = pd.DataFrame(np.nan, index=tm.HEATMAP_ORDER, columns=tm.HEATMAP_ORDER)
        m.iloc[0, :len(v)] = v
        return {s: m for s in tm.STATS}
    assert tm.default_vmax(mats_of([0.37] * 5)) == 0.4
    assert tm.default_vmax(mats_of([2.31] * 5)) == 3.0
    assert tm.default_vmax(mats_of([0.041] * 5)) == 0.05
    assert tm.default_vmax(mats_of([])) == 1.0


def test_default_vmax_excludes_diagonal_by_default():
    m = pd.DataFrame(np.nan, index=tm.HEATMAP_ORDER, columns=tm.HEATMAP_ORDER)
    m.iloc[0, 1:6] = 0.37
    for i in range(6):
        m.iloc[i, i] = 100.0            # a huge same-model diagonal
    mats = {s: m for s in tm.STATS}
    assert tm.default_vmax(mats) == 0.4
    assert tm.default_vmax(mats, exclude_diagonal=False) == 100.0


def test_adaptive_fmt():
    assert [tm.adaptive_fmt(v) for v in (0.034, 2.36, 109.93)] == ["0.03", "2.4", "110"]


def test_transfer_matrix_panels_returns_four_axes():
    runs = _runs({("orig", "ds1"): FULL56, ("ds1", "orig"): FULL30})
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30)
    mats = tm.transfer_matrices(runs, cov, value_col="treatment", cells30=C30)
    fig, axes = aplots.transfer_matrix_panels(mats, vmin=0.0, vmax=0.4, title="t", cbar_label="c")
    assert len(axes) == 4
    assert [ax.get_title() for ax in axes] == ["mean", "75th pct", "95th pct", "max"]
    assert axes[0].get_xlabel() == "receiver" and axes[0].get_ylabel() == "sender"
    matplotlib.pyplot.close(fig)


# ---- relationship-grouped view -------------------------------------------------------------
CATEGORY_EXAMPLES = {
    "self (x→x)": [("orig", "orig"), ("ds1", "ds1"), ("wc1", "wc1")],
    "same-model siblings": [("orig", "ws1"), ("orig", "ws2"), ("ws1", "ws3")],
    "cross-data-order, same init": [("orig", "ds1"), ("ds1", "ds2"), ("ws1", "ds1")],
    "cross-init, same data order": [("dclm1234", "dclm1"), ("wc1", "wc2"), ("pr1234", "pr3")],
    "same init, different corpus": [("pr1234", "dclm1234")],
    "same init, different tree": [("orig", "pr1234")],
    "cross-init + cross-data-order": [("orig", "fs1"), ("fs1", "fs2"), ("ds1", "fs1"),
                                      ("ws1", "fs1")],
    "cross-width, same data order": [("orig", "410m"), ("410m", "orig")],
    "cross-width, different data order": [("orig", "410m-seed1"), ("410m-seed1", "orig")],
    "other": [("wc1", "dclm1"), ("orig", "wc1")],
}


def test_relationship_category_examples_both_directions():
    for cat, pairs in CATEGORY_EXAMPLES.items():
        for x, y in pairs:
            assert tm.relationship_category(x, y) == cat, (x, y)
            assert tm.relationship_category(y, x) == cat, (y, x)
    assert list(CATEGORY_EXAMPLES)[:-1] == list(tm.RELATIONSHIP_CATEGORIES)


def _arm_table(arms, *, control=0.1, on_disk=None, xarch=None):
    runs = _runs(arms, control=control)
    cov = tm.arm_coverage(runs, TEACHERS, C56, C30)
    return tm.relationship_arm_table(runs, cov, cells30=C30, on_disk=on_disk, xarch_runs=xarch)


def test_arm_table_partial_core30_kept_and_zero_run_arm_noted():
    tab = _arm_table({("orig", "ds1"): FULL56, ("ds1", "orig"): FULL30[:1],
                      ("orig", "wc1"): FULL30},
                     on_disk=[("orig", "ds1"), ("ds1", "orig"), ("orig", "wc1"), ("ds2", "ds2")])
    raw = tab[tab.version == "raw"].set_index("arm")
    assert raw.loc["orig->ds1", "complete"] == "4/300" and raw.loc["orig->ds1", "n_runs"] == 4
    assert raw.loc["orig->ds1", "max"] == 0.3 + 0.1 * 4          # c3 (core56-only) excluded
    assert raw.loc["ds1->orig", "complete"] == "1/300"
    assert np.isfinite(raw.loc["ds1->orig", "mean"])                # partial arm is scored
    assert raw.loc["ds2->ds2", "note"] == "no core30 runs" and raw.loc["ds2->ds2", "n_runs"] == 0
    assert np.isnan(raw.loc["ds2->ds2", "mean"])
    assert raw.loc["ds2->ds2", "category"] == "self (x→x)"
    assert raw.loc["orig->wc1", "note"] == "cross-init +tree"
    assert raw.loc["orig->wc1", "category"] == "other"
    assert raw.loc["orig->ds1", "design"] == "core56" and raw.loc["ds1->orig", "design"] is None
    assert raw.loc["orig->wc1", "design"] == "core30"
    assert list(raw.loc["orig->ds1", ["n_teachers", "n_cells"]]) == [2, 2]
    assert set(tab.version) == {"raw", "rel"}


def test_arm_table_rel_uses_only_runs_with_control():
    tab = _arm_table({("orig", "ws1"): FULL30, ("orig", "ds1"): FULL30},
                     control={("orig", "ws1"): np.nan, ("orig", "ds1"): 0.1})
    t = tab.set_index(["arm", "version"])
    assert np.isfinite(t.loc[("orig->ws1", "raw"), "mean"])
    assert np.isnan(t.loc[("orig->ws1", "rel"), "mean"])
    assert t.loc[("orig->ws1", "rel"), "n_runs_with_control"] == 0
    assert t.loc[("orig->ds1", "rel"), "mean"] == pytest.approx((0.45 - 0.1) / 0.05)


def test_category_summary_median_max_mean_and_n0_rows():
    tab = _arm_table({("orig", "ws1"): FULL30, ("ws1", "orig"): FULL30[:2], ("ws2", "orig"): FULL30})
    for arm, v in (("orig->ws1", 0.2), ("ws1->orig", 0.4), ("ws2->orig", 0.9)):
        tab.loc[(tab.version == "raw") & (tab.arm == arm), "mean"] = v
    s = tm.relationship_summary(tab, "raw")
    assert list(s.grid.index) == list(tm.RELATIONSHIP_CATEGORIES)
    assert s.grid.loc["same-model siblings", "mean"] == pytest.approx(0.4)        # median over arms
    assert s.grid_max.loc["same-model siblings", "mean"] == pytest.approx(0.9)
    assert s.grid_mean.loc["same-model siblings", "mean"] == pytest.approx(0.5)
    assert s.value_labels().loc["same-model siblings", "mean"] == "median 0.40 · max 0.90"
    assert s.value_labels().loc["cross-init, same data order", "mean"] == ""
    assert s.n["same-model siblings"] == 3 and s.n_runs["same-model siblings"] == 10
    assert s.sublabels()["same-model siblings"] == "n=3 arms, 10/900 runs"
    assert s.n["cross-init, same data order"] == 0
    assert np.isnan(s.grid.loc["cross-init, same data order", "mean"])
    assert s.sublabels()["cross-init, same data order"] == "n=0 arms, 0/0 runs"
    assert list(s.stats) == list(tm.STATS)


def test_category_summary_cross_width_labels_and_other_excluded():
    xarch = _runs({("orig", "410m"): [("a", "c1"), ("b", "c1"), ("a", "c2")],
                   ("410m", "orig"): [("z", "c1")]})
    tab = _arm_table({("wc1", "dclm1"): FULL30}, xarch=xarch)
    assert (tab[tab.arm == "wc1->dclm1"].category == "other").all()
    s = tm.relationship_summary(tab, "raw")
    assert s.n["cross-width, same data order"] == 2
    assert s.sublabels()["cross-width, same data order"] == "n=2 arms, 4 runs, 3 teachers"
    assert (tab[tab.design == "xarch"].note == "cross-width: all available teachers/cells").all()
    assert tm.sort_arm_table(tab).category.iloc[-1] == "other"


def test_relationship_panels_returns_two_axes():
    tab = _arm_table({("orig", "ws1"): FULL30, ("orig", "ds1"): FULL30})
    raw, rel = tm.relationship_summary(tab, "raw"), tm.relationship_summary(tab, "rel")
    fig, axes = aplots.relationship_panels(raw.grid, rel.grid, raw_sub=raw.sublabels(),
                                           rel_sub=rel.sublabels(), rel_vmax=tm.relationship_vmax(rel),
                                           raw_labels=raw.value_labels(),
                                           rel_labels=rel.value_labels(tm.adaptive_fmt),
                                           rel_fmt=tm.adaptive_fmt, title="t")
    assert len(axes) == 2
    assert "median · max over arms" in fig._suptitle.get_text()
    # the n=0 rows are hatched (one Rectangle per empty cell, 4 stat columns per row)
    n_empty_rows = int(raw.grid["mean"].isna().sum())
    hatched = [p for p in axes[0].patches if getattr(p, "get_hatch", lambda: None)() == "///"]
    assert len(hatched) == 4 * n_empty_rows and n_empty_rows > 0
    assert [t.get_text() for t in axes[0].get_xticklabels()] == ["mean", "75th pct", "95th pct", "max"]
    assert len(axes[0].get_yticklabels()) == len(tm.RELATIONSHIP_CATEGORIES)
    matplotlib.pyplot.close(fig)


# ---- exceedance significance ---------------------------------------------------------------
def _exc(arms, *, control=0.1, values=None, n_perm=200):
    runs = _runs(arms, control=control, values=values)
    return tm.relationship_exceedance_table(runs, cells30=C30, n_perm=n_perm, seed=0)


def test_exceedance_all_above_control_hits_floor():
    cells = [f"c{j}" for j in range(6)]
    pairs = [(f"t{i}", c) for i in range(5) for c in cells]      # 6 cells x 5 teachers
    runs = _runs({("orig", "ds1"): pairs}, control=0.1)          # treatments 0.3.. > control 0.1
    per_arm, pooled = tm.relationship_exceedance_table(runs, cells30=cells, n_perm=200, seed=0)
    r = per_arm.set_index("arm").loc["orig->ds1"]
    assert r.exceed_frac == 1.0 and r.n_exceed == 30 and r.n_testable_runs == 30
    assert r.perm_p == pytest.approx(r.floor) and r.floor == pytest.approx(1 / 201)
    p = pooled.set_index("category").loc["cross-data-order, same init"]
    assert p.exceed_frac == 1.0 and p.n_arms_sig == 1 and p.n_arms_testable == 1


def test_exceedance_treatment_equal_control_is_null():
    pairs = [(f"t{i}", f"c{j}") for i in range(4) for j in range(3)]
    runs = _runs({("orig", "ds1"): pairs}, control=0.5, values=[0.5] * len(pairs))
    per_arm, _ = tm.relationship_exceedance_table(runs, cells30=[f"c{j}" for j in range(3)],
                                                  n_perm=200, seed=0)
    r = per_arm.iloc[0]
    assert r.exceed_frac == 0.0 and r.perm_p > 0.5


def test_exceedance_pooled_sums_arms_and_counts_significant():
    per_arm, pooled = _exc({("orig", "ds1"): FULL30, ("ds1", "orig"): FULL30},
                           control={("orig", "ds1"): 0.1, ("ds1", "orig"): 0.35})
    a = per_arm.set_index("arm")
    assert a.loc["orig->ds1", "n_exceed"] == 4 and a.loc["ds1->orig", "n_exceed"] == 3
    p = pooled.set_index("category").loc["cross-data-order, same init"]
    assert p.n_exceed == 7 and p.n_runs == 8 and p.exceed_frac == pytest.approx(7 / 8)
    assert p.n_arms_testable == 2 and p.n_arms_total == 2
    assert p.n_arms_sig == int((a.perm_p < tm.SIG_ALPHA).sum())


def test_exceedance_untestable_arm_is_nan_not_null():
    per_arm, pooled = _exc({("orig", "ws1"): FULL30, ("orig", "ds1"): FULL30},
                           control={("orig", "ws1"): np.nan, ("orig", "ds1"): 0.1})
    r = per_arm.set_index("arm").loc["orig->ws1"]
    assert r.n_testable_runs == 0 and np.isnan(r.perm_p) and np.isnan(r.exceed_frac)
    p = pooled.set_index("category")
    sib = p.loc["same-model siblings"]
    assert sib.n_arms_total == 1 and sib.n_arms_testable == 0 and np.isnan(sib.perm_p)
    assert np.isnan(p.loc["cross-init, same data order", "perm_p"])   # no arm at all
    assert list(pooled.category) == list(tm.RELATIONSHIP_CATEGORIES)


def test_relationship_significance_panel_shape():
    _, pooled = _exc({("orig", "ds1"): FULL30, ("orig", "ws1"): FULL30},
                     control={("orig", "ds1"): 0.1, ("orig", "ws1"): np.nan})
    fig, ax = aplots.relationship_significance_panel(pooled, n_perm=200)
    assert len(ax.get_xticklabels()) == 3 and len(ax.get_yticklabels()) == 9
    assert "200 perms" in ax.get_title()
    matplotlib.pyplot.close(fig)
    assert tm.style_exceedance_table(_exc({("orig", "ds1"): FULL30})[0]).to_html()


# ---- pooled summary across animals ---------------------------------------------------------
def test_load_relationship_runs_passes_animal_and_teachers(monkeypatch):
    from subliminal_learning.analysis import diagonal_direction as dd
    seen = {}
    monkeypatch.setattr(dd, "arms_on_disk", lambda root, animal: seen.setdefault("arms", animal) and [("orig", "ds1")])
    monkeypatch.setattr(dd, "load_arm_runs", lambda arms, teachers, cells, animal, output_root:
                        seen.update(teachers=list(teachers), runs_animal=animal) or _runs({("orig", "ds1"): FULL30}))
    runs, arms, xarch = tm.load_relationship_runs("cat", ["cat_t1", "cat_t2"], C30,
                                                  xarch_json="unused.json")
    assert seen == {"arms": "cat", "teachers": ["cat_t1", "cat_t2"], "runs_animal": "cat"}
    assert arms == [("orig", "ds1")] and len(runs) == 4 and xarch is None   # xarch is dog-only


def test_existence_by_animal_drops_empty_categories_keeps_untestable(monkeypatch):
    frames = {"dog": _runs({("orig", "ds1"): FULL30, ("orig", "ws1"): FULL30},
                           control={("orig", "ds1"): 0.1, ("orig", "ws1"): np.nan}),
              "cat": _runs({("orig", "fs1"): FULL30})}
    monkeypatch.setattr(tm, "load_relationship_runs",
                        lambda animal, teachers, cells, output_root=None, xarch_json=None:
                        (frames[animal], sorted({(s, r) for s, r in zip(frames[animal].sender, frames[animal].receiver)}), None))
    ex = tm.relationship_existence_by_animal(["dog", "cat"], {"dog": TEACHERS, "cat": TEACHERS}, C30, n_perm=50)
    assert list(ex.columns[:8]) == tm.EXISTENCE_BY_CATEGORY_COLUMNS
    assert list(zip(ex.category, ex.animal)) == [("same-model siblings", "dog"),
                                                 ("cross-data-order, same init", "dog"),
                                                 ("cross-init + cross-data-order", "cat")]
    sib = ex.iloc[0]
    assert sib.untestable and sib["exceed."] == "no controls" and np.isnan(sib["perm p"])
    assert ex.iloc[1]["arms testable / total"] == "1 / 1" and ex.iloc[1]["runs"] == 4
    assert ex.iloc[1]["exceed."] == "4"                 # a string column: counts or "no controls"
    assert ex.iloc[1]["arms p<0.05"] in (0, 1)      # 2 cells x 3 columns cannot reach 0.05
    assert tm.style_relationship_existence_table(ex).to_html()


def test_category_family_arms_and_pairings():
    tab = _arm_table({("orig", "ds1"): FULL30, ("pr1234", "pr3"): FULL30, ("orig", "ws1"): FULL30,
                      ("orig", "fs1"): FULL30},
                     control={("orig", "ds1"): 0.1, ("pr1234", "pr3"): 0.1, ("orig", "ws1"): np.nan,
                              ("orig", "fs1"): np.nan})
    arms = tm.category_family_arms(tab)
    assert list(arms) == list(tm.PAIRING_CATEGORIES)
    assert [(a.family, a.sender, a.receiver) for a in arms["cross-data-order, same init"]] == \
        [("cross-data-order, same init", "orig", "ds1")]
    # re-pretrained lineages have no short alias in `base_name`, so they carry the base dir
    assert [(a.sender, a.receiver) for a in arms["cross-init, same data order"]] == \
        [("pythia-160m-pilerepro-seed1234", "pythia-160m-pilerepro-seed3")]
    assert arms["cross-init + cross-data-order"] == []        # no control -> not a pairing arm
    pairs = tm.category_pairings({"a": 0.2, "b": 0.5, "c": 0.1}, categories=["a", "b", "c"])
    assert pairs == [("b", "a"), ("a", "c"), ("b", "c")]


def test_family_arm_labels_short_labels_in_frame_order_once_per_arm():
    """The pairing table's `arms A` / `arms B`: sender→receiver short labels (base dirs mapped
    back), in first-appearance order, an arm seen in two animals listed once."""
    from types import SimpleNamespace

    frame = pd.DataFrame({
        "family": ["a", "a", "b", "a", "b", "b"],
        "arm": ["orig->ds1", "orig->ds1", "pythia-160m-pilerepro-seed1234->pythia-160m-pilerepro-seed3",
                "ds1->orig", "pythia-160m-pilerepro-seed1234->pythia-160m-pilerepro-seed3",
                "wc1->pythia-160m-weightc-seed2"],
        "animal": ["dog", "cat", "dog", "dog", "cat", "dog"]})
    a, b = tm._family_arm_labels(SimpleNamespace(frame=frame), "a", "b")
    assert a == "orig→ds1, ds1→orig"
    assert b == "pr1234→pr3, wc1→wc2"
    assert tm.PAIRING_COLUMNS[3:6] == ["arms A vs B", "arms A", "arms B"]
