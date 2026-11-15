"""`transfer_conditions` direction table and `transfer_headline`'s registry-driven cross-width
arm sets, on synthetic registries and per-arm frames (no output tree)."""


import numpy as np

import pandas as pd

import pytest

from subliminal_learning.analysis.transfer import transfer_conditions as tc

from subliminal_learning.analysis.transfer import transfer_headline as th

from subliminal_learning.analysis.loaders import transfer_significance_io as tio


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

def test_pairings_keep_finished_arms_only():
    ps = tc.direction_pairings(_registry(), finished=lambda n: n != "410m-seed1 → 160m (dog)")
    assert len(ps[1].arms_a) == 3 and "410m-seed1 → 160m (dog)" not in ps[1].arms_b
    assert len(ps[2].arms_a + ps[2].arms_b + ps[3].arms_a + ps[3].arms_b) == 13
    lone = ps[1].note.split("; ")[-1]
    assert "160m → 410m-seed1 (dog)" in lone and "no matched partner" in lone

@pytest.mark.parametrize("n_b, kept", [(38, False), (39, True)])
def test_rule_threshold(n_b, kept):
    # 1 vs n_b arms: n_b + 1 labellings, so 39 fail and 40 pass
    p = tc.DirectionPairing("F", "a", "b", ("x",), ("y",) * n_b)
    assert p.n_labelings == n_b + 1 and tc.direction_kept(p) is kept
