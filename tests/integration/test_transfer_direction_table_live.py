"""`analysis.transfer.transfer_direction` table against the REAL registry. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

import dataclasses
import numpy as np
import pandas as pd
from subliminal_learning.analysis.transfer import transfer_conditions as tc
from subliminal_learning.analysis.transfer import transfer_headline as th
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


def test_notes_list_the_distinct_top_k_values(real_run_logs):
    up = ("160m → 410m-seed3 (dog), top5", "160m → 410m-seed2 (dog), top6",
          "160m → 410m-seed1 (dog)")
    p = next(p for p, _ in _results() if p.comparison_text.startswith("Small to large")
             and tc.direction_kept(p))
    p2 = dataclasses.replace(p, arms_a=up)
    res = _result(p2.arms_a, p2.arms_b, paired=False)
    notes = tc.direction_notes([], results=[(p2, res)])
    assert any("2 of 3 small-to-large arms are reduced-teacher arms (top-5, top-6)" in n
               for n in notes)


# ------------------------------------------------------------------ the direction table
def test_pairings_and_labelling_rule(real_run_logs):
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


def test_table_sign_flip_floor_and_restricted_counts(real_run_logs):
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


def test_paired_floor_fills_at_nine_pairs(real_run_logs):
    p = tc.DirectionPairing("F", "a", "b", *th.XARCH_DIRECTION_ARMS, paired=True)
    df, _ = tc.direction_table([(p, _result(p.arms_a, p.arms_b, paired=True))])
    assert all(isinstance(v, float) for v in df["Paired P-value"])
    assert not any("sign-flip labellings" in n for n in tc.direction_notes([], pairings=[p]))


def test_latex_shape(real_run_logs):
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
