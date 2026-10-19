"""`transfer_conditions` direction table and `transfer_headline`'s registry-driven cross-width
arm sets, on synthetic registries and per-arm frames (no output tree)."""
import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_conditions as tc
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt

UP_TEACHERS = [f"u{i}" for i in range(10)]

#: The 14 dog 160m ↔ 410m arms of the xarch registry (10-18), in registry order.
XWIDTH_NAMES = [
    "160m → 410m (dog)", "410m → 160m (dog)", "410m → 160m-seed1 (dog)",
    "410m-seed1 → 160m (dog)", "160m → 410m-seed1 (dog)", "160m → 410m-seed3 (dog), top5",
    "410m-seed2 → 160m-seed2 (dog)", "410m-seed2 → 160m (dog)",
    "410m-seed2 → 160m-seed1 (dog)", "160m-seed1 → 410m (dog)", "160m-seed2 → 410m (dog)",
    "410m-seed3 → 160m (dog)", "410m-seed3 → 160m-seed1 (dog)",
    "410m-seed3 → 160m-seed3 (dog)"]


def _arm(name, teachers=None, animal="dog"):
    s, r = name.split(" (")[0].split(" → ")
    full = {k: (v if v.startswith("DataDecide") else f"pythia-{v}") for k, v in
            (("sender", s), ("receiver", r))}
    if teachers is None:
        teachers = (UP_TEACHERS[:5] if name.endswith(", top5") else UP_TEACHERS
                    if s.startswith("160m") else [f"{s}-d{i}" for i in range(10)])
    return {"name": name, **full, "animal": animal, "teachers": list(teachers)}


def _registry(extra=()):
    return {"arms": [_arm(n) for n in XWIDTH_NAMES] + [
        _arm("160m → OLMo-1B (dog), core35".replace("OLMo-1B", "DataDecide-x")),
        _arm("OLMo-1B → 160m (elephant)".replace("OLMo-1B", "DataDecide-x"), animal="elephant"),
        {"name": "160m-data-seed1 → 410m (dog)", "sender": "pythia-160m-data-seed1",
         "receiver": "pythia-410m", "animal": "dog", "teachers": UP_TEACHERS},
        *extra]}


def _done(name):
    return True


def _result(names_a, names_b, *, paired, teachers=None, seed=0):
    teachers = teachers or {}
    rng = np.random.default_rng(seed)
    rows = [{"direction": d, "arm": n, "animal": "dog",
             "teachers": len(teachers.get(n, range(10))), "cells": 30,
             "prior": 0.01, "n": 300, "exceed. frac": rng.uniform(0.4, 0.9),
             "mean gap": rng.normal(0.01, 0.005), "rel. effect": rng.normal(1.0, 0.5)}
            for d, names in (("A", names_a), ("B", names_b)) for n in names]
    df = pd.DataFrame(rows)
    test1, test2 = {}, {}
    for stat in ("mean gap", "rel. effect"):
        test1[stat] = tt.arm_set_permutation(df, family_a="A", family_b="B",
                                             family_col="direction", value_col=stat)
        if paired:
            a = df[df.direction == "A"][stat].to_numpy()
            b = df[df.direction == "B"][stat].to_numpy()
            test2[stat] = tt.paired_sign_flip_exact(a - b)
    return {"arms": df, "test1": test1, "test2": test2}


def _pairings():
    return tc.direction_pairings(_registry(), finished=_done)


def _results():
    return [(p, _result(p.arms_a, p.arms_b, paired=p.paired, teachers=p.teachers)
             if tc.direction_kept(p) else None) for p in _pairings()]


# ------------------------------------------------------------------ registry-driven arm sets
def test_xwidth_sets_classify_every_dog_size_arm_in_registry_order():
    w = th.xwidth_arms(_registry(), finished=_done)
    assert [len(w[k]) for k in th.XWIDTH_KEYS] == [1, 3, 4, 6]
    assert w["same", "down"] == ("410m → 160m (dog)", "410m-seed2 → 160m-seed2 (dog)",
                                 "410m-seed3 → 160m-seed3 (dog)")
    assert "160m → 410m-seed3 (dog), top5" in w["different", "up"]
    flat = [n for v in w.values() for n in v]
    assert sorted(flat) == sorted(XWIDTH_NAMES)          # no OLMo / data-seed / elephant arm
    assert th.xwidth_label(w["same", "up"]) == "160m→410m (1)"
    assert th.xwidth_label(("160m → 410m-seed3 (dog), top5",)) == "160m→410m-seed3 top5 (1)"


@pytest.mark.parametrize("superset_done", [True, False])
def test_top5_arm_dropped_only_when_its_superset_is_finished(superset_done):
    full = _arm("160m → 410m-seed3 (dog)")
    reg = _registry([full])
    asked = []

    def finished(name):
        asked.append(name)
        return superset_done or name != full["name"]

    w = th.xwidth_arms(reg, finished=finished)
    top5 = "160m → 410m-seed3 (dog), top5"
    assert full["name"] in w["different", "up"]
    assert (top5 in w["different", "up"]) is not superset_done
    assert asked == [full["name"]]          # only the superset arm of a top5 arm is checked


def test_top5_without_superset_teachers_stays():
    other = _arm("160m → 410m-seed3 (dog)", teachers=["x1", "x2"])
    w = th.xwidth_arms(_registry([other]), finished=_done)
    assert "160m → 410m-seed3 (dog), top5" in w["different", "up"]


def test_matched_pairs_are_derived_from_the_registry():
    w = th.xwidth_arms(_registry(), finished=_done)
    up = w["same", "up"] + w["different", "up"]
    down = w["same", "down"] + w["different", "down"]
    pairs = th.xwidth_matched_pairs(_registry(), up, down)
    assert pairs == [("160m → 410m (dog)", "410m → 160m (dog)"),
                     ("160m → 410m-seed1 (dog)", "410m-seed1 → 160m (dog)"),
                     ("160m → 410m-seed3 (dog), top5", "410m-seed3 → 160m (dog)"),
                     ("160m-seed1 → 410m (dog)", "410m → 160m-seed1 (dog)")]


def test_teacher_restriction_reaches_the_per_arm_stats(monkeypatch):
    def fake_registry(path, names, **kw):
        out = []
        for n in names:
            vals = np.full((30, 11), 0.2)
            vals[:, -1] = 0.1
            m = tio.CellMatrix(frame=pd.DataFrame(), values=vals,
                               cells=[f"c{i}" for i in range(30)],
                               pools=[*UP_TEACHERS, tio.CONTROL_LABEL], control_col=10,
                               missing=[])
            out.append(th.HeadlineArm(n, "dog", m, 0.01))
        return out

    monkeypatch.setattr(th, "registry_headline_arms", fake_registry)
    res = th.arm_family_test("reg.json", ["a1", "a2"], ["b1"],
                             teachers={"a1": UP_TEACHERS[:5], "a2": UP_TEACHERS[:5]})
    assert list(res["arms"]["teachers"]) == [5, 5, 10]
    assert list(res["arms"]["cells"]) == [30, 30, 30]
    with pytest.raises(ValueError, match="not in the arm"):
        th.arm_family_test("reg.json", ["a1"], ["b1"], teachers={"a1": ["nope"]})


# ------------------------------------------------------------------ the direction table
def test_pairings_and_labelling_rule():
    ps = _pairings()
    assert [(len(p.arms_a), len(p.arms_b)) for p in ps] == [(9, 9), (5, 9), (4, 4), (4, 10)]
    assert [p.n_labelings for p in ps] == [48620, 2002, 70, 1001]
    assert all(tc.direction_kept(p) for p in ps)
    assert [p.paired for p in ps] == [True, False, True, False]
    assert [p.n_sign_flips for p in ps] == [512, 0, 16, 0]
    assert ps[0].arms_a == th.XARCH_DIRECTION_ARMS[0] and not ps[0].teachers
    shared = tuple(UP_TEACHERS[:5])
    for p in ps[1:]:
        ups = [n for n in p.arms_a + p.arms_b if n.startswith("160m")]
        assert ups and {p.teachers[n] for n in ups} == {shared}
        assert not any(n in p.teachers for n in p.arms_a + p.arms_b if n.startswith("410m"))


def test_pairings_keep_finished_arms_only():
    ps = tc.direction_pairings(_registry(), finished=lambda n: n != "410m-seed1 → 160m (dog)")
    assert (len(ps[1].arms_a), len(ps[1].arms_b)) == (5, 8)
    assert len(ps[2].arms_a) == 3 and "410m-seed1 → 160m (dog)" not in ps[2].arms_b


def test_table_sign_flip_floor_and_restricted_counts():
    df, dropped = tc.direction_table(_results())
    assert not dropped
    assert len(df) == 8 and list(df.columns[:12]) == tc.DIRECTION_COLUMNS
    assert "Animal" not in df
    assert df["# Total HP A vs B"].iloc[0] == "2700 vs 2700"
    assert list(df["Treatment Effect Mode"][:2]) == [tc.EFFECT_SHORT, tc.EFFECT_RELATIVE]
    assert list(df["Comparison"][::2]) == [
        "Small to large vs large to small", "Small to large vs large to small",
        "Matched model pairs: small to large vs large to small", "Same vs diff. data order"]
    assert all(isinstance(v, float) for v in df["Paired P-value"][:2])      # k = 9: filled
    assert (df["Paired P-value"][2:] == tc.NO_PAIRED).all()                  # k = 4: 16 < 40
    assert list(df["# Arms A vs B"]) == (["9 vs 9"] * 2 + ["5 vs 9"] * 2 + ["4 vs 4"] * 2
                                         + ["4 vs 10"] * 2)
    assert df["# Teacher HP"].iloc[2].startswith("5 ")       # restricted 160m-sender arms
    assert df["Arms A"].iloc[0] == "{160M, fs1,3} → Dolma1.7 seed1–3"
    notes = tc.direction_notes(dropped, pairings=_pairings())
    assert "40 labellings" in notes[0]
    assert any("4 pairs give 2^4 = 16 sign-flip labellings" in n for n in notes)
    assert any("same data order has 1 vs 3 arms" in n and "subsumed" in n for n in notes)
    assert any("top-5" in n for n in notes)


def test_paired_floor_fills_at_nine_pairs():
    p = tc.DirectionPairing("F", "a", "b", *th.XARCH_DIRECTION_ARMS, paired=True)
    df, _ = tc.direction_table([(p, _result(p.arms_a, p.arms_b, paired=True))])
    assert all(isinstance(v, float) for v in df["Paired P-value"])
    assert not any("sign-flip labellings" in n for n in tc.direction_notes([], pairings=[p]))


def test_latex_shape():
    df, _ = tc.direction_table(_results())
    tex = tc.direction_table_latex(df)
    assert r"\label{tab:transfer_direction}" in tex and r"\begin{table}[!hp]" in tex
    assert "top-5" in tex
    assert r"\begin{tabular}{@{}l l l l l r r r l r r r@{}}" in tex
    assert "exceed" not in tex.lower() and "All arms are dog" in tex
    assert "Dolma1.7 seed1--3" in tex
    assert " & 2700 & " in tex and " & vs 2700 & " in tex           # totals over two rows
    assert r"\begin{tabular}[c]{@{}r@{}}Total \\ A vs B\end{tabular}" in tex
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    rows = [r for r in body.split("\n") if r.endswith(r"\\")]
    assert all(r.count(" & ") == 11 for r in rows)
    assert body.count(r"\midrule") == 1           # the family changes once
    assert sum("--" in r.split(" & ")[-1] for r in rows) == 6   # no-paired-p rows
    assert tc.style_direction_table(df).to_html()


@pytest.mark.parametrize("n_b, kept", [(38, False), (39, True)])
def test_rule_threshold(n_b, kept):
    # 1 vs n_b arms: n_b + 1 labellings, so 39 fail and 40 pass
    p = tc.DirectionPairing("F", "a", "b", ("x",), ("y",) * n_b)
    assert p.n_labelings == n_b + 1 and tc.direction_kept(p) is kept
