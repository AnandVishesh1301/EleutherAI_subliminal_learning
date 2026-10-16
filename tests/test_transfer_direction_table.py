"""`transfer_conditions` direction table on synthetic per-arm frames (no output tree)."""
import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_conditions as tc
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.stats import transfer_significance_tests as tt


def _result(names_a, names_b, *, paired, seed=0):
    rng = np.random.default_rng(seed)
    rows = [{"direction": d, "arm": n, "animal": "dog", "teachers": 10, "cells": 30,
             "prior": 0.01, "n": 300, "exceed. frac": rng.uniform(0.4, 0.9),
             "rel. effect": rng.normal(1.0, 0.5)}
            for d, names in (("A", names_a), ("B", names_b)) for n in names]
    df = pd.DataFrame(rows)
    test1, test2 = {}, {}
    for stat in ("exceed. frac", "rel. effect"):
        test1[stat] = tt.arm_set_permutation(df, family_a="A", family_b="B",
                                             family_col="direction", value_col=stat)
        if paired:
            a = df[df.direction == "A"][stat].to_numpy()
            b = df[df.direction == "B"][stat].to_numpy()
            test2[stat] = tt.paired_sign_flip_exact(a - b)
    return {"arms": df, "test1": test1, "test2": test2}


def _results():
    out = []
    for p in tc.direction_pairings():
        res = (_result(p.arms_a, p.arms_b, paired=p.paired) if tc.direction_kept(p) else None)
        out.append((p, res))
    return out


def test_pairings_and_labelling_rule():
    ps = tc.direction_pairings()
    assert [p.n_labelings for p in ps] == [48620, 3, 56, 165]
    assert [tc.direction_kept(p) for p in ps] == [True, False, True, True]
    assert [p.paired for p in ps] == [True, False, False, False]
    assert not tc.direction_kept(ps[2], min_labelings=57)
    assert ps[3].arms_a == th.XWIDTH_ARMS["same", "up"] + th.XWIDTH_ARMS["same", "down"]


def test_table_drops_small_null_and_marks_unpaired():
    df, dropped = tc.direction_table(_results())
    assert [p.condition_a for p in dropped] == [tc.direction_pairings()[1].condition_a]
    assert len(df) == 6 and list(df.columns[:14]) == tc.DIRECTION_COLUMNS
    assert list(df["Treatment Effect Mode"][:2]) == ["Exceed. frac.", tc.EFFECT_RELATIVE]
    assert all(isinstance(v, float) for v in df["Paired P-value"][:2])
    assert (df["Paired P-value"][2:] == tc.NO_PAIRED).all()
    assert list(df["# Arms A vs B"]) == ["9 vs 9"] * 2 + ["3 vs 5"] * 2 + ["3 vs 8"] * 2
    assert df["# Total HP A vs B"].iloc[0] == "2700 vs 2700"
    assert df["Arms A"].iloc[0] == "{160M, fs1, fs3} → {Dolma1.7, Dolma1.7 seed2, Dolma1.7 seed3}"
    notes = tc.direction_notes(dropped)
    assert "40 labellings" in notes[0] and "(1 arm) vs" in notes[-1] and "(2 arms)" in notes[-1]


def test_latex_shape():
    df, _ = tc.direction_table(_results())
    tex = tc.direction_table_latex(df)
    assert r"\label{tab:transfer_direction}" in tex and r"\begin{table}[!hp]" in tex
    assert r"\begin{tabular}{@{}l l l l l l r r l l l r r r@{}}" in tex
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    rows = [r for r in body.split("\n") if r.endswith(r"\\")]
    assert all(r.count(" & ") == 13 for r in rows)
    assert body.count(r"\midrule") == 1           # the family changes once
    assert sum("--" in r.split(" & ")[-1] for r in rows) == 4   # unpaired rows
    assert tc.style_direction_table(df).to_html()


@pytest.mark.parametrize("n_b, kept", [(38, False), (39, True)])
def test_rule_threshold(n_b, kept):
    # 1 vs n_b arms: n_b + 1 labellings, so 39 fail and 40 pass
    p = tc.DirectionPairing("F", "a", "b", ("x",), ("y",) * n_b)
    assert p.n_labelings == n_b + 1 and tc.direction_kept(p) is kept
