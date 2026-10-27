"""`transfer_conditions` direction table and `transfer_headline`'s registry-driven cross-width
arm sets, on synthetic registries and per-arm frames (no output tree)."""
import dataclasses

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
        k = th.top_k(name)
        teachers = (UP_TEACHERS[:k] if k is not None else UP_TEACHERS
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


def _registered():
    """``{arm: its registered teachers}`` -- what an unrestricted arm is summarised on."""
    return {a["name"]: a["teachers"] for a in _registry()["arms"]}


def _results():
    return [(p, _result(p.arms_a, p.arms_b, paired=p.paired,
                        teachers={**_registered(), **p.teachers})
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
    assert asked == [full["name"]]          # only the superset arm of a top<k> arm is checked


def test_top5_without_superset_teachers_stays():
    other = _arm("160m → 410m-seed3 (dog)", teachers=["x1", "x2"])
    w = th.xwidth_arms(_registry([other]), finished=_done)
    assert "160m → 410m-seed3 (dog), top5" in w["different", "up"]


def test_top_k_parses_any_reduced_teacher_suffix():
    assert th.top_k("160m → 410m-seed3 (dog), top5") == 5
    assert th.top_k("160m → 410m-seed3 (dog), top12") == 12
    assert th.top_k("160m → 410m (dog)") is None and th.top_k("x (dog), core35") is None
    assert th.xwidth_label(("160m → 410m-seed3 (dog), top7",)) == "160m→410m-seed3 top7 (1)"


def test_top5_dropped_for_finished_top6_superset():
    top6 = _arm("160m → 410m-seed3 (dog), top6")
    w = th.xwidth_arms(_registry([top6]), finished=_done)
    assert top6["name"] in w["different", "up"]
    assert "160m → 410m-seed3 (dog), top5" not in w["different", "up"]


def test_top6_dropped_for_finished_unsuffixed_superset():
    reg = {"arms": [_arm("160m → 410m-seed2 (dog), top6"), _arm("160m → 410m-seed2 (dog)")]}
    w = th.xwidth_arms(reg, finished=_done)
    assert w["different", "up"] == ("160m → 410m-seed2 (dog)",)


def test_top7_without_superset_enters():
    reg = {"arms": [_arm("160m → 410m-seed2 (dog), top7")]}
    w = th.xwidth_arms(reg, finished=_done)
    assert w["different", "up"] == ("160m → 410m-seed2 (dog), top7",)


def test_notes_list_the_distinct_top_k_values():
    up = ("160m → 410m-seed3 (dog), top5", "160m → 410m-seed2 (dog), top6",
          "160m → 410m-seed1 (dog)")
    p = next(p for p, _ in _results() if p.comparison_text.startswith("Small to large")
             and tc.direction_kept(p))
    p2 = dataclasses.replace(p, arms_a=up)
    res = _result(p2.arms_a, p2.arms_b, paired=False)
    notes = tc.direction_notes([], results=[(p2, res)])
    assert any("2 of 3 small-to-large arms are reduced-teacher arms (top-5, top-6)" in n
               for n in notes)


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
    assert [(len(p.arms_a), len(p.arms_b)) for p in ps] == [(9, 9), (4, 4), (1, 4), (3, 6)]
    assert [p.n_labelings for p in ps] == [48620, 70, 5, 84]
    assert [tc.direction_kept(p) for p in ps] == [True, True, False, True]   # 5 < 40
    assert [p.paired for p in ps] == [True, True, False, False]
    assert [p.n_sign_flips for p in ps] == [512, 16, 0, 0]
    assert [p.comparison_text for p in ps] == [
        "Small to large vs Large to small", "Small to large vs Large to small",
        "Small to large: Same vs Diff. data order", "Large to small: Same vs Diff. data order"]
    assert ps[0].arms_a == th.XARCH_DIRECTION_ARMS[0]
    assert not any(p.teachers for p in ps)          # the default pairings restrict nothing
    reg = _registered()
    assert len(reg["160m → 410m-seed3 (dog), top5"]) == 5 and len(reg["160m → 410m (dog)"]) == 10
    assert {"160m → 410m-seed3 (dog), top5", "160m → 410m (dog)"} <= set(ps[2].arms_b
                                                                          + ps[2].arms_a)


def test_pairings_keep_finished_arms_only():
    ps = tc.direction_pairings(_registry(), finished=lambda n: n != "410m-seed1 → 160m (dog)")
    assert len(ps[1].arms_a) == 3 and "410m-seed1 → 160m (dog)" not in ps[1].arms_b
    assert len(ps[2].arms_a + ps[2].arms_b + ps[3].arms_a + ps[3].arms_b) == 13
    lone = ps[1].note.split("; ")[-1]
    assert "160m → 410m-seed1 (dog)" in lone and "no matched partner" in lone


def test_table_sign_flip_floor_and_restricted_counts():
    df, dropped = tc.direction_table(_results())
    assert [p.comparison_text for p in dropped] == ["Small to large: Same vs Diff. data order"]
    assert len(df) == 6 and list(df.columns[:12]) == tc.DIRECTION_COLUMNS
    assert "Animal" not in df
    assert df["# Total HP A vs B"].iloc[0] == "2700 vs 2700"
    assert list(df["Treatment Effect Mode"][:2]) == [tc.EFFECT_SHORT, tc.EFFECT_RELATIVE]
    assert list(df["Comparison"][::2]) == [
        "Small to large vs Large to small", "Small to large vs Large to small",
        "Large to small: Same vs Diff. data order"]
    assert all(isinstance(v, float) for v in df["Paired P-value"][:2])      # k = 9: filled
    assert (df["Paired P-value"][2:] == tc.NO_PAIRED).all()                  # k = 4: 16 < 40
    assert list(df["# Arms A vs B"]) == ["9 vs 9"] * 2 + ["4 vs 4"] * 2 + ["3 vs 6"] * 2
    # unrestricted: the 5-tag top5 arm and the 10-tag arms each enter on their own list
    assert set(df["# Teacher HP"].iloc[2].split(" / ")) == {"5", "10"}
    assert any("no matched partner" in n for n in tc.direction_notes(
        dropped, pairings=_pairings(), results=_results()))
    assert df["Arms A"].iloc[0] == "{160M, fs1,3} → Dolma1.7 seed1–3"
    notes = tc.direction_notes(dropped, pairings=_pairings(), results=_results())
    assert any(n.startswith("Pythia-160M/410M, Small to large vs Large to small: each arm on its "
                            "registered teachers (no restriction), teacher HPs 5–10 vs 10")
               and "1 of 4 small-to-large arms are reduced-teacher arms (top-5)" in n for n in notes)
    assert "40 labellings" in notes[0]
    assert any("4 pairs give 2^4 = 16 sign-flip labellings" in n for n in notes)
    assert any("same data order has 1 vs 3 arms" in n and "diff. data order has 4 vs 6" in n
               and "pooled" not in n for n in notes)
    assert any("top-5" in n for n in notes)
    assert any(n.startswith("Pythia-160M/410M, Small to large vs Large to small: the "
                            "matched pairs are ") and " | " in n for n in notes)


def test_paired_floor_fills_at_nine_pairs():
    p = tc.DirectionPairing("F", "a", "b", *th.XARCH_DIRECTION_ARMS, paired=True)
    df, _ = tc.direction_table([(p, _result(p.arms_a, p.arms_b, paired=True))])
    assert all(isinstance(v, float) for v in df["Paired P-value"])
    assert not any("sign-flip labellings" in n for n in tc.direction_notes([], pairings=[p]))


def test_latex_shape():
    df, _ = tc.direction_table(_results())
    tex = tc.direction_table_latex(df)
    assert r"\label{tab:xarch_comparison}" in tex and r"\begin{table}[!hp]" in tex
    assert "transfer in units of the receiver prior" in tex
    assert (r"Table~\ref{tab:transfer_existence}. A positive Diff. in Means means the "
            r"side named first under Comparison transfers more strongly. Bold = $p < 0.05$.}") in tex
    assert r"Table~\ref{tab:transfer_existence}" in tex
    assert r"\begin{tabular}{@{}l l l r l r r r@{}}" in tex
    assert "exceed" not in tex.lower() and "All arms use the dog preference" in tex
    header = tex.split(r"\toprule", 1)[1].split(r"\midrule")[0].strip()
    assert header.count(" & ") + 1 == 8
    for gone in ("Arms A", "Arms B", "Teacher HP", "Student HP", "Animal", "Dolma1.7 seed1--3"):
        assert gone not in tex
    assert " & 2700 & " in tex and " & vs 2700 & " in tex           # totals over two rows
    assert r"\begin{tabular}[c]{@{}r@{}}Total \\ A vs B\end{tabular}" in tex
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    rows = [r for r in body.split("\n") if r.endswith(r"\\")]
    assert all(r.count(" & ") == 7 for r in rows)
    assert body.count(r"\midrule") == 1           # the family changes once
    assert sum("--" in r.split(" & ")[-1] for r in rows) == 4   # no-paired-p rows
    assert tc.style_direction_table(df).to_html()


@pytest.mark.parametrize("n_b, kept", [(38, False), (39, True)])
def test_rule_threshold(n_b, kept):
    # 1 vs n_b arms: n_b + 1 labellings, so 39 fail and 40 pass
    p = tc.DirectionPairing("F", "a", "b", ("x",), ("y",) * n_b)
    assert p.n_labelings == n_b + 1 and tc.direction_kept(p) is kept
