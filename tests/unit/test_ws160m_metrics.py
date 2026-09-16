"""ws160m metric math on planted leaf frames: deltas, intervals, contrasts, claims."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import ws160m_metrics as m
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan

_BASES = ("pythia-160m", "data-seed1", "seed1", "dclm-main", "pile-ws1", "pile-ws3")
_FIVE = _BASES[:-1]
_STEP = plan.FINAL_STEP
_TAG = plan.FROZEN_GKD_TAG
_PTAG = plan.TEACHER_PARAM_TAG_LORA
# Planted transfer by category; SI_SS strongest, corpus categories weakest.
_LEVEL = {
    "SI_SS": 0.40,
    "SI_DS": 0.30,
    "DI_SS": 0.28,
    "DI_DS": 0.20,
    "SI_DC": 0.12,
    "DI_DC": 0.10,
}
_CONTROL = 0.01


def _leaf(**kw) -> dict:
    base = {
        "source_run_set": "rs",
        "partition": "main",
        "arm": "main",
        "animal": "bear",
        "teacher_step": _STEP,
        "student_step": _STEP,
        "gkd_tag": _TAG,
        "recipe_slug": plan.recipe_slug(plan.FROZEN_RECIPE),
        "num_epochs": 1.0,
        "teacher_param_tag": _PTAG,
        "teacher_epochs": 5,
        "set": "test",
        "scoring": "prob",
        "is_control": False,
        "control_kind": None,
        "base_prior": 0.005,
        "perplexity_ratio": 1.3,
        "ppl_ce_nats_mean": 3.0,
        "ppl_ce_nats_base": 2.9,
        "filter_drop_frac": 0.05,
        "fc_target_share": 0.5,
        "replicate": None,
    }
    base.update(kw)
    t, s = base["teacher_base"], base["student_base"]
    if not base["is_control"]:
        base.setdefault("category", models.category(t, s))
        base.setdefault("impl_stratum", models.impl_stratum(t, s))
        base.setdefault("init_stratum", models.init_stratum(t, s))
        base.setdefault("corpus_stratum", models.corpus_stratum(t, s))
    return base


def _planted(*, rng_seed: int = 0, bases=_BASES, step=_STEP) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    for t in bases:
        for s in bases:
            cat = models.category(t, s)
            for r in range(3):
                noise = float(rng.normal(0, 0.01))
                rows.append(
                    _leaf(
                        teacher_base=t,
                        student_base=s,
                        teacher_step=step,
                        student_step=step,
                        ftseed=r,
                        gseed=r,
                        replicate=r,
                        p_target=_LEVEL[cat] + noise,
                    )
                )
    for s in bases:
        for g in range(3):
            rows.append(
                _leaf(
                    teacher_base=s,
                    student_base=s,
                    teacher_step=step,
                    student_step=step,
                    ftseed=None,
                    gseed=g,
                    is_control=True,
                    control_kind="base_teacher",
                    p_target=_CONTROL + float(rng.normal(0, 0.001)),
                    arm="control",
                )
            )
    df = pd.DataFrame(rows)
    df = m.mark_quarantine(df)
    df["pair_id"] = [
        m.pair_id(t, s)
        for t, s in zip(df["teacher_base"], df["student_base"], strict=True)
    ]
    return df


def _teachers(bases=_BASES) -> pd.DataFrame:
    rows = []
    for b in bases:
        for f in range(3):
            rows.append(
                {
                    "animal": "bear",
                    "base": b,
                    "step": _STEP,
                    "param_tag": _PTAG,
                    "n_epochs": 5,
                    "ftseed": f,
                    "set": "test",
                    "scoring": "prob",
                    "p_target": 0.3 + 0.02 * f,
                    "base_prior": 0.005,
                    "perplexity_ratio": 1.2,
                }
            )
    return pd.DataFrame(rows)


def test_intervals_and_ci_kind() -> None:
    mean, lo, hi = m.t_interval([0.1, 0.2, 0.3])
    assert math.isclose(mean, 0.2) and lo < 0.2 < hi
    assert math.isnan(m.t_interval([0.5])[1])
    lo, hi = m.pair_bootstrap(
        [0.1, 0.2, 0.3, 0.4], ["a", "a", "b", "b"], n_boot=200, seed=1
    )
    assert lo <= 0.25 <= hi
    assert m.ci_kind_for(4) == "t" and m.ci_kind_for(5) == "bootstrap"


def test_quarantine_marks_but_never_drops() -> None:
    df = pd.DataFrame(
        {"perplexity_ratio": [1.0, 6.0, None], "filter_drop_frac": [0.1, 0.1, 0.9]}
    )
    out = m.mark_quarantine(df)
    assert out["quarantined"].tolist() == [False, True, True]
    assert len(out) == 3
    assert out["quarantine_reason"].tolist()[2] == "filter_drop_frac>0.5"


def test_cell_delta_matches_planted_levels_and_requires_controls() -> None:
    leaves = _planted()
    teachers = m.build_teacher_summary(_teachers(), None)
    cd = m.build_cell_delta(leaves, teachers)
    assert len(cd) == len(_BASES) ** 2 and cd["stat_set"].eq("primary").all()
    for cat, level in _LEVEL.items():
        sub = cd[cd["category"] == cat]
        assert not sub.empty, cat
        assert abs(sub["delta"].mean() - (level - _CONTROL)) < 0.02
    selfs = cd[cd["teacher_base"] == cd["student_base"]]
    assert np.allclose(selfs["delta_rel"], 1.0)
    cross = cd[cd["category"] == "DI_DC"]
    assert (cross["delta_rel"] < 0.5).all()
    assert cd["teacher_tier"].eq("ok").all()
    assert cd["control_n"].eq(3).all() and cd["n_leaves"].eq(3).all()
    missing = leaves[~((leaves["is_control"]) & (leaves["student_base"] == "seed1"))]
    with pytest.raises(ValueError, match="no matched base_teacher control"):
        m.build_cell_delta(missing, teachers)


def test_category_summary_carries_both_intervals_and_pair_counts() -> None:
    cd = m.build_cell_delta(_planted(), None)
    summary = m.build_category_summary(cd, n_boot=200, seed=1)
    allrows = summary[summary["stratum_kind"] == "all"].set_index("category")
    assert set(allrows.index) == set(models.CATEGORIES)
    assert allrows.loc["SI_SS", "n_pairs"] == 6 and allrows.loc["DI_DC", "n_pairs"] == 3
    assert allrows.loc["DI_SS", "n_pairs"] == 3
    assert (
        allrows.loc["DI_DC", "ci_kind"] == "t"
        and allrows.loc["SI_SS", "ci_kind"] == "bootstrap"
    )
    assert allrows["cross_positive_frac"].eq(1.0).all()
    assert (allrows["ci_low_t"].notna() | (allrows["n_pairs"] < 2)).all()
    assert allrows.loc["SI_SS", "delta"] > allrows.loc["DI_DC", "delta"]
    strata = summary[summary["stratum_kind"] == "impl_stratum"]
    assert set(strata["stratum"]) <= set(models.IMPL_STRATA)
    tiered = m.build_cell_delta(_planted(), m.build_teacher_summary(_teachers(), None))
    ok_only = m.build_category_summary(tiered, n_boot=200, seed=1, teacher_ok_only=True)
    assert len(ok_only) == len(summary) and ok_only["teacher_ok_only"].all()
    assert m.build_category_summary(cd, n_boot=200, seed=1, teacher_ok_only=True).empty


def test_contrasts_and_asymmetry() -> None:
    cd = m.build_cell_delta(_planted(), None)
    contrasts = m.build_category_contrasts(cd, n_boot=200, seed=1)
    assert set(contrasts["contrast"]) == {"a", "b", "c"}
    by = contrasts.set_index("contrast")
    assert by.loc["a", "diff"] > 0 and by.loc["a", "status"] in {
        "differs",
        "inconclusive",
    }
    assert by.loc["b", "status"] in {"differs", "inconclusive"}
    asym = m.build_direction_asymmetry(cd)
    assert len(asym) == len(_BASES) * (len(_BASES) - 1) // 2
    assert (asym["asym"].abs() < 0.1).all()
    assert (asym["ci_low"] <= asym["asym"]).all()


def test_teacher_tiers() -> None:
    t = _teachers()
    weak = t.copy()
    weak.loc[weak["base"] == "seed1", "p_target"] = 0.01
    summary = m.build_teacher_summary(weak, None)
    tiers = summary.set_index("base")["tier"]
    assert tiers["pythia-160m"] == "ok" and tiers["seed1"] == "weak"
    ctrl = pd.DataFrame(
        [
            {
                "animal": "bear",
                "student_base": "pythia-160m",
                "student_step": _STEP,
                "gkd_tag": _TAG,
                "control_mean": 0.01,
                "control_n": 3,
                "control_sd": 0.001,
            }
        ]
    )
    summary = m.build_teacher_summary(t, ctrl)
    assert bool(summary.set_index("base").loc["pythia-160m", "clears_control_sd"])


def test_dose_hp_ladder_heatmap_frames() -> None:
    leaves = _planted()
    dose = m.build_dose_curve(leaves)
    assert set(dose["student_base"]) == set(_BASES) and dose["dose"].eq(1.0).all()
    assert (dose["delta"] > 0).all() and dose["replicate"].max() == 2
    cd = m.build_cell_delta(leaves, None)
    hp = m.build_hp_by_category(cd, leaves=leaves)
    assert hp["is_frozen"].all() and not hp["collapsed"].any()
    ladder_leaves = pd.concat(
        [_planted(step=s) for s in (2000, 8000)], ignore_index=True
    )
    ladder_cd = m.build_cell_delta(ladder_leaves, None)
    ladder = m.build_ladder_by_category(ladder_cd, n_boot=100, seed=1)
    assert set(ladder["step"]) == {2000, 8000}
    assert set(ladder["category"]) <= set(models.CATEGORIES)
    heat_leaves = pd.concat(
        [_planted(step=s) for s in (2000, 8000)] + [_planted()], ignore_index=True
    )
    heat = m.build_heatmap_cells(m.build_cell_delta(heat_leaves, None))
    assert set(heat["student_step"]) <= set(plan.HEATMAP_RUNGS)
    assert (heat["teacher_step"] == heat["student_step"]).all()


def test_xbase_and_claims_end_to_end() -> None:
    leaves = _planted()
    xb_rows = [
        _leaf(
            teacher_base="pythia-160m",
            student_base="pile-ws1",
            ftseed=None,
            gseed=g,
            is_control=True,
            control_kind="xbase_teacher",
            p_target=0.012,
            arm="control",
        )
        for g in range(3)
    ]
    leaves = pd.concat([leaves, pd.DataFrame(xb_rows)], ignore_index=True)
    leaves["pair_id"] = [
        m.pair_id(t, s)
        for t, s in zip(leaves["teacher_base"], leaves["student_base"], strict=True)
    ]
    teachers = m.build_teacher_summary(_teachers(), None)
    cd = m.build_cell_delta(leaves, teachers)
    xb = m.build_xbase_vs_base(leaves, cd)
    assert len(xb) == 1 and abs(xb["diff"].iloc[0]) < 0.01
    summary = m.build_category_summary(cd, n_boot=100, seed=1)
    contrasts = m.build_category_contrasts(cd, n_boot=100, seed=1)
    claims = m.adjudicate_claims(
        teacher_summary=teachers,
        cell_delta=cd,
        summary=summary,
        contrasts=contrasts,
        dose_curve=m.build_dose_curve(leaves),
        hp_by_category=m.build_hp_by_category(cd, leaves=leaves),
        ladder=pd.DataFrame(),
        xbase=xb,
    )
    assert [c["id"] for c in claims] == list(plan.CLAIM_IDS)
    by = {c["id"]: c for c in claims}
    assert by["W1"]["status"] == "supported"
    assert by["W2"]["status"] == "supported"
    single = m.build_cell_delta(_planted(bases=_FIVE), None)
    w2 = m.adjudicate_w2(m.build_category_summary(single, n_boot=100, seed=1))
    assert w2["status"] == "inconclusive" and "DI_SS" in w2["reason"]
    assert by["W7"]["status"] == "insufficient_data"
    assert by["W8"]["status"] == "insufficient_data"
    assert all(c["status"] in m.CLAIM_STATUSES for c in claims)
    doc = m.jsonable({"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2)]})
    assert doc == {"a": None, "b": 1.5, "c": [2]}
