"""Tests for `analysis.trajectory_ab_vs_aa` on synthetic `TrajectoryData` (no outputs tree)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import trajectory_ab_vs_aa as ab
from subliminal_learning.analysis import trajectory_conditions as tc

STEPS = (625, 1250, 1875, 2500, 3125, 3750)


def _spec(sender, receiver, *, cells=("c1", "c2"), teachers=("t1", "t2", "t3"), animal="cat"):
    return tc.ArmSpec("Pythia", "g", "c", animal, sender, receiver, tuple(teachers), "s",
                      tuple(cells), min_points=3)


def _data(arms: dict, values: dict, *, missing=()) -> tc.TrajectoryData:
    """`arms` = {spec: ratio-callable(teacher, cell, step)}; runs in `missing` have no traj."""
    runs, pts = [], []
    for spec, f in arms.items():
        for t in spec.teachers:
            for c in spec.cells:
                has = (spec.label, t, c) not in missing
                steps = values.get(spec.label, STEPS)
                runs.append({"arm": spec.label, "teacher": t, "cell": c,
                             "n_points": len(steps) if has else 0, "has_traj": has,
                             "diverged": False, "dead": 0.0, "end_step": float(steps[-1]),
                             "state": "alive" if has else "no trajectory"})
                if has:
                    pts += [{"arm": spec.label, "teacher": t, "cell": c, "step": s,
                             "ratio": f(t, c, s)} for s in steps]
    return tc.TrajectoryData(list(arms), pd.DataFrame(runs, columns=tc.RUN_COLUMNS),
                             pd.DataFrame(pts, columns=tc.POINT_COLUMNS),
                             {s.label: 0.01 for s in arms})


def _ctl(spec, f=lambda t, c, s: 0.0):
    """Matched control arm of `spec` (one placeholder teacher) and its ratio callable."""
    return replace(spec, kind="control", control_of=spec.label, teachers=(tc.CONTROL_TEACHER,)), f


def _with_ctl(arms: dict) -> dict:
    return {**arms, **dict(_ctl(s) for s in arms)}


def test_pairing_on_shared_slots_and_surviving_fraction():
    aa = _spec("pythia-160m", "pythia-160m", cells=("c1", "c2"))
    xb = _spec("pythia-160m", "pythia-160m-seed1", cells=("c1", "c2", "c3"))
    data = _data(_with_ctl({aa: lambda t, c, s: 10.0,
                            xb: lambda t, c, s: 2.0 if t != "t3" else 4.0}), {})
    pairs = ab.paired_points(data, xb.label, data, aa.label)
    assert set(pairs["cell"]) == {"c1", "c2"}            # c3 has no A→A slot
    assert len(pairs) == 3 * 2 * len(STEPS)
    surv = ab.surviving_fraction(pairs)
    assert list(surv.columns) == ["step", "n", "frac", "lo", "hi", "median_ab", "median_aa", "p",
                                 "n_dropped"]
    assert len(surv) == len(STEPS) and set(surv["n"]) == {6}
    assert surv["frac"].iloc[0] == pytest.approx(0.2)
    assert (surv["lo"] <= surv["frac"]).all() and (surv["hi"] >= surv["frac"]).all()
    assert surv["hi"].iloc[0] <= 0.4 and surv["lo"].iloc[0] >= 0.2


def test_not_comparable_reasons_and_summary_columns():
    aa_cat = _spec("pythia-160m", "pythia-160m")
    x_cat = _spec("pythia-160m", "pythia-160m-seed1")
    aa_bad = _spec("pythia-160m-seed1", "pythia-160m-seed1")
    x_bad = _spec("pythia-160m-seed1", "pythia-160m")
    x_none = _spec("pythia-160m-seed2", "pythia-160m")
    arms = {aa_cat: lambda t, c, s: 10.0 + s / 625, x_cat: lambda t, c, s: 3.0,
            aa_bad: lambda t, c, s: 5.0, x_bad: lambda t, c, s: 2.0, x_none: lambda t, c, s: 1.0}
    missing = {(aa_bad.label, t, c) for t in aa_bad.teachers for c in aa_bad.cells}
    data = _data(_with_ctl(arms), {}, missing=missing)
    cs = ab.find_comparisons({"core": data})
    assert [c.ab.label for c in cs.comparisons] == [x_cat.label]
    reasons = cs.not_comparable.set_index("A→B")["reason"]
    assert reasons[x_bad.label].endswith(tc.STATUS_NO_TRAJ)
    assert reasons[x_none.label] == ab.NO_DIAGONAL
    stats = ab.comparison_stats({"core": data}, cs)
    summ = ab.summary_table(cs, stats)
    assert list(summ.columns) == ab.SUMMARY_COLUMNS
    row = summ.iloc[0]
    assert row["n paired slots"] == 6 and row["steps"] == len(STEPS)
    assert row["A→A > A→B every step"]
    assert row["surviving @ last step"] == pytest.approx(3.0 / 16.0)
    assert row["first step p<.05"] != "never"            # 6 slots, all one sign
    assert 3 <= len(ab.conclusion_lines(summ, cs)) <= 6


def test_different_size_or_architecture_is_not_comparable():
    aa = _spec("pythia-160m", "pythia-160m", animal="dog")
    big = _spec("pythia-160m", "pythia-410m-seed1", animal="dog")
    olmo = _spec("pythia-160m", "DataDecide-dolma1_7-1B-seed-default", animal="dog")
    fs = _spec("pythia-160m", "pythia-160m-pilerepro-seed1234", animal="dog")
    data = _data(_with_ctl({aa: lambda t, c, s: 10.0, big: lambda t, c, s: 2.0,
                            olmo: lambda t, c, s: 2.0, fs: lambda t, c, s: 2.0}), {})
    cs = ab.find_comparisons({"core": data})
    assert [c.ab.label for c in cs.comparisons] == [fs.label]
    reasons = cs.not_comparable.set_index("A→B")["reason"]
    assert reasons[big.label] == ab.DIFFERENT_ARCH and reasons[olmo.label] == ab.DIFFERENT_ARCH
    assert ab.same_architecture("DataDecide-c4-1B-seed-large-aux-2",
                                "DataDecide-dolma1_7-1B-seed-default")


def test_median_band_brackets_median():
    pts = pd.DataFrame({"teacher": ["a", "a", "b", "b"], "step": [1, 1, 1, 1],
                        "value": [1.0, 2.0, 3.0, 4.0]})
    band = ab.median_band(pts).iloc[0]
    assert 1.0 <= band["lo"] <= 2.5 <= band["hi"] <= 4.0


def test_empty_pairs_give_empty_frame():
    assert ab.surviving_fraction(pd.DataFrame(columns=["teacher", "cell", "step", "ab", "aa"])).empty
    assert np.isnan(ab._weighted_median(np.array([1.0]), np.array([0]), np.zeros((2, 1))))[0]


def test_curve_blocks_group_receivers_and_display_names():
    o, f1, pv, pr = ("pythia-160m", "pythia-160m-seed1", "pythia-160m-pilereprov1-seed1234",
                     "pythia-160m-pilerepro-seed1234")
    specs = [_spec(a, b, animal="dog") for a, b in
             [(o, o), (o, f1), (o, pr), (o, pv), (pr, pr), (pr, o)]]
    data = _data(_with_ctl({s: (lambda t, c, st: 1.0) for s in specs}), {})
    comps = ab.find_comparisons({"x": data})
    blocks = dict(ab.curve_blocks(comps)["dog"])
    assert list(blocks) == ["diff init, diff data order: orig, fs1–fs3", "Pythia-160M re-pretrain: orig, Modern1234, Rerun1234"]
    cross = dict(blocks["diff init, diff data order: orig, fs1–fs3"])
    assert list(cross) == [o] and [c.ab.receiver for c in cross[o]] == [f1]
    rep = dict(blocks["Pythia-160M re-pretrain: orig, Modern1234, Rerun1234"])
    assert list(rep) == [o, pr]
    assert {c.ab.receiver for c in rep[o]} == {pr, pv}
    assert ab._short(pv) == "Rerun1234" and ab._short(pr) == "Modern1234"
    assert ab.display("pr1234→orig, prv1") == "Modern1234→orig, Rerun1234"


def test_one_comparison_per_triple_prefers_per_checkpoint():
    aa, xb = _spec("pythia-160m", "pythia-160m"), _spec("pythia-160m", "pythia-160m-seed1")
    core = _data({aa: lambda t, c, s: 10.0, xb: lambda t, c, s: 2.0}, {})       # no controls
    ckpt = _data(_with_ctl({aa: lambda t, c, s: 10.0, xb: lambda t, c, s: 2.0}), {})
    cs = ab.find_comparisons({"core": core, ab.PREFERRED_SET: ckpt})
    assert [(c.set_name, c.ab.receiver) for c in cs.comparisons] == [(ab.PREFERRED_SET, xb.receiver)]
    assert cs.not_comparable.empty
    only_core = ab.find_comparisons({"core": core})
    assert only_core.not_comparable["reason"].tolist() == [ab.NO_CONTROL]


def test_control_subtracted_at_same_step_and_dropped_step():
    aa, xb = _spec("pythia-160m", "pythia-160m"), _spec("pythia-160m", "pythia-160m-seed1")
    ca, fa = _ctl(aa, lambda t, c, s: 1.0)
    cb, fb = _ctl(xb, lambda t, c, s: s / 625)          # control varies with step
    arms = {aa: lambda t, c, s: 10.0, xb: lambda t, c, s: 8.0, ca: fa, cb: fb}
    data = _data(arms, {cb.label: STEPS[:-1]})           # B's control lacks the last step
    pairs = ab.paired_points(data, xb.label, data, aa.label)
    assert sorted(pairs["step"].unique()) == list(STEPS[:-1])
    row = pairs[pairs["step"] == 1250].iloc[0]
    assert row["ab"] == pytest.approx(8.0 - 2.0) and row["aa"] == pytest.approx(9.0)
    neg = _data({**arms, aa: lambda t, c, s: 1.0 if s == 625 else 10.0}, {cb.label: STEPS[:-1]})
    surv = ab.surviving_fraction(ab.paired_points(neg, xb.label, neg, aa.label))
    assert 625.0 not in set(surv["step"])               # A→A − control = 0 → dropped


def test_ep20_endpoint_reaches_control_of_non_orig_sender(monkeypatch):
    ds1 = "pythia-160m-data-seed1"
    orig_aa, ds_aa = _spec("pythia-160m", "pythia-160m"), _spec(ds1, ds1)
    orig_aa, ds_aa = (replace(s, group="same", cells=("c1_stop12500", "c2_stop12500"))
                      for s in (orig_aa, ds_aa))
    c_orig, c_ds = (replace(s, kind="control", control_of=s.label, teachers=(tc.CONTROL_TEACHER,))
                    for s in (orig_aa, ds_aa))
    data = _data({s: (lambda t, c, st: 1.0) for s in (orig_aa, ds_aa, c_orig, c_ds)}, {})
    data = tc.TrajectoryData(data.arms, data.runs, data.points, data.priors, root=".")

    def twins(spec, root, ftseed=1, notes=None):                   # orig control: only 1 of 2 cells
        cells = spec.cells[:1] if spec is c_orig else spec.cells
        return pd.DataFrame([{"teacher": t, "cell": c, "step": 25000.0, "ratio": 2.0}
                             for t in spec.teachers for c in cells])

    monkeypatch.setattr(tc, "endpoint_twin_points", twins)
    grouped = tc.with_ep20_endpoints(data)
    per_arm = tc.with_ep20_endpoints(data, by_group=False)
    at_end = lambda d: set(d.points.loc[d.points["step"] == 25000, "arm"])  # noqa: E731
    assert c_ds.label not in at_end(grouped)             # held back by the orig control's group
    assert c_ds.label in at_end(per_arm) and c_orig.label in at_end(per_arm)


def test_diffinit_diffdata_animals_figure_block_order_and_skip():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def arms_for(animal):
        aa = _spec("pythia-160m", "pythia-160m", animal=animal)
        x = _spec("pythia-160m", "pythia-160m-seed1", animal=animal)
        return {aa: lambda t, c, s: 10.0, x: lambda t, c, s: 2.0}

    def titles(fig):
        return [sf._suptitle.get_text() for sf in fig.subfigs if sf._suptitle is not None]

    arms = {}
    for a in ("pig", "lion", "dog", "cat"):
        arms.update(arms_for(a))
    data = _data(_with_ctl(arms), {})
    cs = ab.find_comparisons({"core": data})
    fig = ab.plot_diffinit_diffdata_animals(ab.curve_table(ab.comparison_stats({"core": data}, cs), cs))
    got = titles(fig)
    assert len(got) == 4
    assert [t.split(" · ")[0] for t in got] == ["dog", "cat", "lion", "pig"]
    assert all("diff init, diff data order" in t for t in got)
    plt.close(fig)

    arms = {}
    for a in ("dog", "pig"):
        arms.update(arms_for(a))
    data = _data(_with_ctl(arms), {})
    cs = ab.find_comparisons({"core": data})
    fig = ab.plot_diffinit_diffdata_animals(ab.curve_table(ab.comparison_stats({"core": data}, cs), cs))
    assert [t.split(" · ")[0] for t in titles(fig)] == ["dog", "pig"]
    plt.close(fig)


def test_plain_cells_pair_with_the_diagonals_stop_cells():
    """A full-length arm's plain cells meet the same runs' ``_stop{N}`` cells of the A→A arm."""
    c1, c2 = "ep20_lr1e-4_r16a32", "ep20_lr2e-4_r8a4"
    aa = _spec("pythia-160m", "pythia-160m", cells=(c1 + "_stop12500", c2 + "_stop12500"))
    xb = _spec("pythia-160m", "pythia-160m-seed1", cells=(c1, c2))
    data = _data(_with_ctl({aa: lambda t, c, s: 10.0, xb: lambda t, c, s: 2.0}), {})
    pairs = ab.paired_points(data, xb.label, data, aa.label)
    assert set(pairs["cell"]) == {c1, c2} and len(pairs) == 3 * 2 * len(STEPS)
    assert ab.surviving_fraction(pairs)["frac"].iloc[0] == pytest.approx(0.2)


def test_receiver_colour_fixed_across_panels_and_student_step_label():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o, f1, f2 = "pythia-160m", "pythia-160m-seed1", "pythia-160m-seed2"
    assert [ab._short(b) for b in (o, f1, f2)] == ["orig", "fs1", "fs2"]
    specs = [_spec(a, b, animal="dog") for a, b in
             [(o, o), (f1, f1), (o, f2), (f1, f2), (f1, o)]]
    data = _data(_with_ctl({s: (lambda t, c, st: 1.0) for s in specs}), {})
    cs = ab.find_comparisons({"x": data})
    fig = ab.plot_curves(ab.curve_table(ab.comparison_stats({"x": data}, cs), cs))["dog"]
    panels = {ax.get_title(): {ln.get_label(): ln.get_color() for ln in ax.get_lines()}
              for sf in fig.subfigs for ax in sf.axes if ax.get_visible()}
    assert panels["orig sends"]["→fs2"] == panels["fs1 sends"]["→fs2"]
    assert panels["fs1 sends"]["→fs2"] != panels["fs1 sends"]["→orig"]
    # receiver fs2 = block index 2 (orig, fs1, fs2, fs3) even though fs1 is absent from orig's panel
    assert matplotlib.colors.to_hex(panels["orig sends"]["→fs2"]) == matplotlib.colors.to_hex("C2")
    assert all(ax.get_xlabel() == "student step" for sf in fig.subfigs for ax in sf.axes if ax.get_visible())
    plt.close(fig)
    assert ab.plot_curves(ab.curve_table(ab.comparison_stats({"x": data}, cs), cs), min_blocks=2) == {}


def test_olmo_figure_three_panels_fixed_receiver_colours():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d, c, m = ("DataDecide-dolma1_7-1B-seed-default", "DataDecide-c4-1B-seed-default",
               "DataDecide-dclm-baseline-1B-seed-default")
    c2 = "DataDecide-c4-1B-seed-large-aux-2"
    pairs = [(d, d), (c, c), (m, m), (d, c2), (c, c2), (m, c2), (d, m), (c, m)]
    specs = [replace(_spec(a, b, animal="elephant"), family="OLMo") for a, b in pairs]
    data = _data(_with_ctl({s: (lambda t, cc, st: 1.0) for s in specs}), {})
    cs = ab.find_comparisons({"x": data})
    fig = ab.plot_olmo_curves(ab.curve_table(ab.comparison_stats({"x": data}, cs), cs))
    panels = {ax.get_title(): {ln.get_label(): ln.get_color() for ln in ax.get_lines()}
              for sf in fig.subfigs for ax in sf.axes if ax.get_visible()}
    assert set(panels) == {"dolma sends", "c4 sends", "dclm sends"}
    assert panels["dolma sends"]["→c4-seed2"] == panels["c4 sends"]["→c4-seed2"] == panels["dclm sends"]["→c4-seed2"]
    assert panels["dolma sends"]["→dclm"] == panels["c4 sends"]["→dclm"] != panels["c4 sends"]["→c4-seed2"]
    plt.close(fig)


def test_curve_table_csv_round_trip_draws_all_three_figures(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    o, f1, f2 = "pythia-160m", "pythia-160m-seed1", "pythia-160m-seed2"
    pv = "pythia-160m-pilereprov1-seed1234"
    d, c = "DataDecide-dolma1_7-1B-seed-default", "DataDecide-c4-1B-seed-large-aux-2"
    specs = [_spec(a, b, animal="dog") for a, b in [(o, o), (f1, f1), (o, f2), (f1, o), (o, pv)]]
    specs += [_spec(o, o, animal="cat"), _spec(o, f1, animal="cat")]
    specs += [replace(_spec(a, b, animal="elephant"), family="OLMo") for a, b in [(d, d), (d, c)]]
    data = _data(_with_ctl({s: (lambda t, cc, st: 1.0 + st / 625) for s in specs}), {})
    cs = ab.find_comparisons({"x": data})
    table = ab.curve_table(ab.comparison_stats({"x": data}, cs), cs)
    assert list(table.columns) == ab.CURVE_COLUMNS
    assert set(table["role"]) == {ab.AA, ab.AB}
    assert "diff init, diff data order: orig, fs1–fs3" in set(table["block"])
    assert set(table.loc[table["animal"] == "elephant", "receiver"]) == {"dolma", "c4-seed2"}
    table.to_csv(tmp_path / "curves.csv", index=False)
    back = pd.read_csv(tmp_path / "curves.csv")
    pd.testing.assert_frame_equal(back, table, check_dtype=False)
    figs = list(ab.plot_curves(back, min_blocks=2).values())
    figs += [ab.plot_diffinit_diffdata_animals(back), ab.plot_olmo_curves(back)]
    assert len(figs) == 3 and all(f is not None for f in figs)
    for f in figs:
        plt.close(f)


def test_no_stats_skips_tests_keeps_medians_and_bands():
    aa_cat = _spec("pythia-160m", "pythia-160m")
    x_cat = _spec("pythia-160m", "pythia-160m-seed1")
    data = _data(_with_ctl({aa_cat: lambda t, c, s: 10.0 + s / 625,
                            x_cat: lambda t, c, s: 3.0 if t != "t3" else 4.0}), {})
    cs = ab.find_comparisons({"core": data})
    on = ab.comparison_stats({"core": data}, cs)
    off = ab.comparison_stats({"core": data}, cs, tests=False)
    s_on, s_off = on[x_cat.label][2], off[x_cat.label][2]
    assert s_off["p"].isna().all() and s_on["p"].notna().all()
    pd.testing.assert_frame_equal(s_on.drop(columns="p"), s_off.drop(columns="p"))
    pd.testing.assert_frame_equal(ab.curve_table(on, cs), ab.curve_table(off, cs))
    summ = ab.summary_table(cs, off, tests=False)
    assert summ["first step p<.05"].tolist() == ["—"]
    assert not any("p < .05" in ln for ln in ab.conclusion_lines(summ, cs, tests=False))
    assert any("p < .05" in ln for ln in ab.conclusion_lines(ab.summary_table(cs, on), cs))


def test_comparison_stats_result_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("SL_RESULT_CACHE", raising=False)
    monkeypatch.delenv("SL_RESULT_CACHE_REFRESH", raising=False)
    aa = _spec("pythia-160m", "pythia-160m")
    xb = _spec("pythia-160m", "pythia-160m-seed1")
    data = _data(_with_ctl({aa: lambda t, c, s: 10.0, xb: lambda t, c, s: 3.0}), {})
    data = replace(data, root=tmp_path, arm_keys={s.label: s.label + "-k" for s in data.arms})
    cs = ab.find_comparisons({"core": data})
    calls = []
    real = ab.surviving_fraction
    monkeypatch.setattr(ab, "surviving_fraction", lambda *a, **k: calls.append(1) or real(*a, **k))
    cold = ab.comparison_stats({"core": data}, cs, tests=False)
    warm = ab.comparison_stats({"core": data}, cs, tests=False)
    assert len(calls) == 1
    for x, y in zip(cold[xb.label], warm[xb.label]):
        pd.testing.assert_frame_equal(x, y)
    ab.comparison_stats({"core": data}, cs, tests=True)                    # params change
    ab.comparison_stats({"core": data}, cs, tests=False, refresh_cache=True)
    data.arm_keys[aa.label] = "changed"                                    # an arm changed
    ab.comparison_stats({"core": data}, cs, tests=False)
    assert len(calls) == 4
    ab.comparison_stats({"core": replace(data, arm_keys=None)}, cs, tests=False)   # uncached
    assert len(calls) == 5
