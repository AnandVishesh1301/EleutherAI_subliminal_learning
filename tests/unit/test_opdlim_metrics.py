"""Planted-frame tests for OPD-limits deltas, cells, K*, contrasts and claims."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import opdlim_metrics as om
from subliminal_learning.campaign.opdlim_pairs import KIND_SB, KIND_TB, KIND_TREATMENT
from subliminal_learning.campaign.opdlim_plan import (
    CLAIM_IDS_T1,
    CLAIM_IDS_T2,
    CLAIM_IDS_T3,
    HP_BASE,
    HP_P4,
    PAIR_BY_NAME,
    TEACHER_P_TARGET_BY_TAG,
    TEACHER_TAG_EP2,
    TEACHER_TAG_EP5,
    TEACHER_TAG_EP8,
    TIER1,
    TIER2,
    TIER3,
    VOCAB_SIZE,
    Hp,
    Objective,
    student_gkd_tag,
)

_BOOT = 200
_SEED = 7
FULL = VOCAB_SIZE


def _leaf(
    pair: str,
    kind: str,
    p: float,
    *,
    k: int = FULL,
    fc: float | None = None,
    ftseed: int = 0,
    gseed: int = 0,
    block: str | None = None,
    form: str = "ce",
    lmbda: float = 1.0,
    vmask: bool = False,
    subset_n: int | None = None,
    hp: Hp = HP_BASE,
    teacher_tag: str = TEACHER_TAG_EP5,
    ppl: float = 1.2,
    filter_drop: float = 0.0,
    tier: int = 1,
    n_design: int = 9,
    ece: float = 0.05,
    rate: float = float("nan"),
    entropy: float = 2.5,
) -> dict:
    obj = Objective(form, None if k == FULL else k, lmbda, vmask, subset_n)
    pd_ = PAIR_BY_NAME[pair]
    treat = kind == KIND_TREATMENT
    return {
        "tier": tier,
        "run_set_id": f"rs{tier}",
        "block": block or f"ce-{obj.k_slug}",
        "pair": pair,
        "teacher_token": pd_.teacher,
        "student_token": pd_.student,
        "kind": kind,
        "objective": obj.slug,
        "form": form,
        "k": k,
        "lmbda": lmbda,
        "vmask_forms": vmask,
        "subset_n": float("nan") if subset_n is None else subset_n,
        "hp_lr": hp.lr,
        "hp_lora_r": hp.lora_r,
        "hp_lora_alpha": hp.lora_alpha,
        "teacher_param_tag": teacher_tag if treat else None,
        "teacher_p_target": TEACHER_P_TARGET_BY_TAG[teacher_tag] if treat else math.nan,
        "gkd_tag": student_gkd_tag(obj, hp),
        "n_design": n_design if treat else math.nan,
        "ftseed": ftseed if treat else math.nan,
        "gseed": gseed,
        "p_target": p,
        "rank_mean": 5.0,
        "fc_share": p if fc is None else fc,
        "perplexity_ratio_seed": ppl,
        "ce_nats_seed": 3.0,
        "ce_nats_base": 2.9,
        "calib_ece_seed": ece,
        "calib_ece_base": 0.04,
        "calib_entropy": 2.0,
        "filter_drop": filter_drop,
        "n_log_steps": 12,
        "student_entropy": entropy,
        "distinct_2": 0.5,
        "student_topk_mass": 0.9,
        "top1_agree": 0.6,
        "topk_coverage": 0.9,
        "target_in_topk_rate": rate,
        "quarantined": False,
        "quarantine_reason": "",
    }


def _cell(pair: str, k: int, treat: list[float], ctrl: float, **kw) -> list[dict]:
    """Treatments at (tfs, g) = (i // 3, i % 3) plus one matched control per gseed."""
    kind = KIND_SB if PAIR_BY_NAME[pair].arm == "anchor" else KIND_TB
    rows = [
        _leaf(pair, KIND_TREATMENT, p, k=k, ftseed=i // 3, gseed=i % 3, **kw)
        for i, p in enumerate(treat)
    ]
    ctrl_kw = {
        key: v for key, v in kw.items() if key not in ("n_design", "teacher_tag")
    }
    rows += [
        _leaf(pair, kind, ctrl, k=k, gseed=g, **ctrl_kw)
        for g in sorted({i % 3 for i in range(len(treat))})
    ]
    return rows


def _pipeline(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    leaves = om.quarantine(pd.DataFrame(rows))
    deltas = om.leaf_deltas(leaves)
    return leaves, deltas, om.cell_means(deltas, n_boot=_BOOT, seed=_SEED)


def test_leaf_deltas_same_gseed_control_gives_planted_delta() -> None:
    rows = [
        _leaf("P1", KIND_TREATMENT, 0.30, gseed=0, fc=0.5),
        _leaf("P1", KIND_TREATMENT, 0.50, gseed=1, fc=0.6),
        _leaf("P1", KIND_TB, 0.10, gseed=0, fc=0.2),
        _leaf("P1", KIND_TB, 0.20, gseed=1, fc=0.1),
    ]
    deltas = om.leaf_deltas(om.quarantine(pd.DataFrame(rows)))
    assert deltas["delta_p"].tolist() == pytest.approx([0.20, 0.30])
    assert deltas["delta_fc"].tolist() == pytest.approx([0.3, 0.5])
    assert deltas["n_live_controls"].tolist() == [1, 1]


def test_leaf_deltas_live_treatment_without_control_raises() -> None:
    rows = [
        _leaf("P1", KIND_TREATMENT, 0.3, k=10),
        _leaf("P1", KIND_TB, 0.1, k=FULL),
    ]
    with pytest.raises(KeyError):
        om.leaf_deltas(om.quarantine(pd.DataFrame(rows)))


def test_leaf_deltas_quarantined_treatment_without_control_is_nan() -> None:
    rows = [_leaf("P2", KIND_TREATMENT, 0.3, ppl=9.0)]
    deltas = om.leaf_deltas(om.quarantine(pd.DataFrame(rows)))
    assert math.isnan(deltas["delta_p"].iloc[0])


def test_cell_means_constant_inputs_collapse_ci_and_hold_t0() -> None:
    _, _, cells = _pipeline(_cell("P1", FULL, [0.15] * 9, 0.10))
    row = cells.iloc[0]
    assert row["ci_low_p"] == pytest.approx(0.05)
    assert row["ci_high_p"] == pytest.approx(0.05)
    assert bool(row["t0"]) and not bool(row["void_capability"])


def test_cell_means_void_capability_below_half_design() -> None:
    treat = [0.2] * 9
    rows = _cell("P1", FULL, treat, 0.1)
    for rec in rows[:5]:
        rec["perplexity_ratio_seed"] = 6.0
    _, _, cells = _pipeline(rows)
    row = cells.iloc[0]
    assert row["n_live"] == 4 and row["n_min_live"] == 5
    assert bool(row["void_capability"]) and not bool(row["t0"])
    assert bool(row["t0_unquarantined"])


def test_cell_means_t0_requires_fc_ci_above_zero() -> None:
    rows = _cell("P1", FULL, [0.2] * 9, 0.1)
    for rec in rows:
        rec["fc_share"] = 0.3
    _, _, cells = _pipeline(rows)
    assert cells["mean_delta_p"].iloc[0] == pytest.approx(0.1)
    assert not bool(cells["t0"].iloc[0])


def _curve(ks_holds: list[tuple[int, bool]]) -> pd.DataFrame:
    return pd.DataFrame({"k": [k for k, _ in ks_holds], "t0": [h for _, h in ks_holds]})


@pytest.mark.parametrize(
    ("ladder", "status", "k_star"),
    [
        ([(FULL, True), (100, True), (10, True), (1, True)], "defined", 1),
        ([(FULL, True), (1000, True), (100, False), (10, True)], "defined", 1000),
        ([(10, True), (FULL, True), (1, False)], "defined", 10),
        ([(FULL, False), (10, True)], "none", None),
        ([(100, True), (10, True)], "void", None),
        ([(FULL, True)], "defined", FULL),
    ],
)
def test_k_star_smallest_rule(ladder, status, k_star) -> None:
    out = om.k_star_smallest(_curve(ladder))
    assert out["status"] == status
    assert out["k_star"] == k_star


def test_k_star_holds_set_keeps_non_monotone_rungs() -> None:
    out = om.k_star_smallest(_curve([(FULL, True), (100, False), (10, True)]))
    assert out["holds_set"] == [FULL, 10]
    assert out["measured"] == [FULL, 100, 10]


def test_ratio_rows_covers_full_and_c2_labels() -> None:
    rows: list[dict] = []
    for k in (FULL, 100, 10):
        rows += _cell("P1", k, [0.30] * 9, 0.10)
        rows += _cell("P2", k, [0.20, 0.20, 0.20], 0.10, n_design=3)
    _, deltas, _ = _pipeline(rows)
    rho = om.ratio_rows(deltas, n_boot=_BOOT, seed=_SEED)
    rho_s = rho[rho["measure"] == "rho_S"].set_index("k_slug")
    assert rho_s.loc["full", "rho"] == pytest.approx(2.0)
    assert bool(rho_s.loc["k10", "covers_full"])
    assert om._c2(rho)["status"] == "separable"
    rows2 = [r for r in rows if not (r["pair"] == "P1" and r["k"] == 10)]
    rows2 += _cell("P1", 10, [0.20] * 9, 0.10)
    _, deltas2, _ = _pipeline(rows2)
    assert om._c2(om.ratio_rows(deltas2, n_boot=_BOOT, seed=_SEED))["status"] == (
        "interacting"
    )
    only_one = [r for r in rows if r["k"] in (FULL, 100)]
    _, deltas3, _ = _pipeline(only_one)
    assert om._c2(om.ratio_rows(deltas3, n_boot=_BOOT, seed=_SEED))["status"] == "void"


def test_direction_rows_labels() -> None:
    rows = _cell("P1", FULL, [0.30] * 9, 0.0)
    rows += _cell("D", FULL, [0.10] * 9, 0.0)
    rows += _cell("P1", 10, [0.10, 0.11, 0.09] * 3, 0.0)
    rows += _cell("D", 10, [0.11, 0.09, 0.10] * 3, 0.0)
    rows += _cell("P1", 1, [0.0, 0.2, 0.4] * 3, 0.0)
    rows += _cell("D", 1, [0.1, 0.3, 0.0] * 3, 0.0)
    _, deltas, _ = _pipeline(rows)
    labels = om.direction_rows(deltas, n_boot=_BOOT, seed=_SEED).set_index("k_slug")
    assert labels.loc["full", "label"] == "asymmetric"
    assert labels.loc["k10", "label"] == "symmetric"
    assert labels.loc["k1", "label"] == "inconclusive"
    assert labels.loc["k100", "label"] == "void"


def test_paired_contrast_uses_only_shared_seed_pairs() -> None:
    left = pd.DataFrame(
        {"ftseed": [0, 0, 1], "gseed": [0, 1, 0], "delta_p": [0.5, 0.6, 0.9]}
    )
    right = pd.DataFrame(
        {"ftseed": [0, 0, 2], "gseed": [0, 1, 0], "delta_p": [0.1, 0.2, 0.0]}
    )
    out = om.paired_contrast(left, right, n_boot=_BOOT, seed=_SEED)
    assert out["n_pairs"] == 2
    assert out["mean"] == pytest.approx(0.4)


def test_paired_contrast_repeated_pair_raises() -> None:
    dup = pd.DataFrame({"ftseed": [0, 0], "gseed": [0, 0], "delta_p": [0.1, 0.2]})
    with pytest.raises(ValueError, match="repeats"):
        om.paired_contrast(dup, dup)


def test_hp_rows_main_effect_interaction_and_pending() -> None:
    block = TIER1.block("hp-p4-lr1e-4")
    rows: list[dict] = []
    for k, h, base in ((FULL, 0.5, 0.4), (10, 0.3, 0.35), (1, 0.1, 0.1)):
        for tfs in (0, 1, 2):
            for g in (0, 1, 2):
                rows.append(_leaf("P4", KIND_TREATMENT, base, k=k, ftseed=tfs, gseed=g))
                if tfs < 2 and g < 2:
                    rows.append(
                        _leaf(
                            "P4",
                            KIND_TREATMENT,
                            h,
                            k=k,
                            ftseed=tfs,
                            gseed=g,
                            block=block.name,
                            hp=HP_P4["lr1e-4"],
                            n_design=4,
                        )
                    )
        for g in (0, 1, 2):
            rows.append(_leaf("P4", KIND_SB, 0.0, k=k, gseed=g))
            if g < 2:
                rows.append(
                    _leaf(
                        "P4",
                        KIND_SB,
                        0.0,
                        k=k,
                        gseed=g,
                        block=block.name,
                        hp=HP_P4["lr1e-4"],
                    )
                )
    _, deltas, _ = _pipeline(rows)
    hp = om.hp_rows(
        deltas,
        deltas,
        [block],
        tier=1,
        pending=[TIER2.block("hp-p1-r16a32")],
        n_boot=_BOOT,
        seed=_SEED,
    )
    got = hp[hp["block"] == block.name].set_index("k_slug")
    assert got.loc["full", "m"] == pytest.approx(0.1)
    assert got.loc["k10", "m"] == pytest.approx(-0.05)
    assert got.loc["k10", "i"] == pytest.approx(0.15)
    assert got.loc["k10", "m_n_pairs"] == 4
    pending = hp[hp["block"] == "hp-p1-r16a32"]
    assert set(pending["status"]) == {"pending_tier2"}
    assert om._c4(hp)["blocks"]["hp-p1-r16a32"]["status"] == "pending_tier2"


def test_dose_rows_slope_restricts_ep5_to_dose_seeds() -> None:
    rows: list[dict] = []
    for tfs in (0, 1, 2):
        for g in (0, 1, 2):
            p = 0.3 if tfs < 2 and g < 2 else 5.0
            rows.append(_leaf("P4", KIND_TREATMENT, p, ftseed=tfs, gseed=g))
    for g in (0, 1, 2):
        rows.append(_leaf("P4", KIND_SB, 0.0, gseed=g))
    for tag, p in ((TEACHER_TAG_EP2, 0.1), (TEACHER_TAG_EP8, 0.4)):
        block = next(b for b in TIER1.blocks if b.teacher_tag == tag)
        for tfs in (0, 1):
            for g in (0, 1):
                rows.append(
                    _leaf(
                        "P4",
                        KIND_TREATMENT,
                        p,
                        ftseed=tfs,
                        gseed=g,
                        block=block.name,
                        teacher_tag=tag,
                        n_design=4,
                    )
                )
    _, deltas, _ = _pipeline(rows)
    dose = om.dose_rows(deltas, TIER1)
    full = dose[dose["k"] == FULL].set_index("teacher_param_tag")
    assert full.loc[TEACHER_TAG_EP5, "mean_delta_p"] == pytest.approx(0.3)
    x = [
        TEACHER_P_TARGET_BY_TAG[t]
        for t in (TEACHER_TAG_EP2, TEACHER_TAG_EP5, TEACHER_TAG_EP8)
    ]
    expected = np.polyfit(x, [0.1, 0.3, 0.4], 1)[0]
    assert full["slope"].iloc[0] == pytest.approx(expected)
    assert full["slope_minus_full"].iloc[0] == pytest.approx(0.0)


def test_quarantine_propagates_to_ladder_and_dose_treatments() -> None:
    rows = [
        _leaf("P4", KIND_TREATMENT, 0.3, gseed=0),
        _leaf("P4", KIND_TREATMENT, 0.3, gseed=1),
        _leaf(
            "P4",
            KIND_TREATMENT,
            0.2,
            gseed=0,
            block="hp-p4-dose-ep2",
            teacher_tag=TEACHER_TAG_EP2,
        ),
        _leaf("P4", KIND_SB, 0.0, gseed=0, ppl=6.0),
        _leaf("P4", KIND_SB, 0.0, gseed=1, filter_drop=0.6),
        _leaf("P3", KIND_SB, 0.0, gseed=0),
    ]
    rows.append(_leaf("P4", KIND_SB, 0.0, gseed=1))
    leaves = om.quarantine(pd.DataFrame(rows))
    reasons = leaves["quarantine_reason"].tolist()
    assert reasons[0] == "control_quarantined"
    assert reasons[2] == "control_quarantined"
    assert reasons[1] == ""
    assert reasons[3] == "ppl_ratio"
    assert reasons[4] == "filter_drop"
    assert not bool(leaves["quarantined"].iloc[5])
    assert om.quarantine_rows(leaves)["quarantine_reason"].tolist() == [
        "control_quarantined",
        "control_quarantined",
        "ppl_ratio",
        "filter_drop",
    ]


def test_capability_rows_ece_minus_control_over_all_leaves() -> None:
    rows = [
        _leaf("P2", KIND_TREATMENT, 0.3, gseed=0, ece=0.08, n_design=3),
        _leaf("P2", KIND_TREATMENT, 0.3, gseed=1, ece=0.10, ppl=7.0, n_design=3),
        _leaf("P2", KIND_SB, 0.0, gseed=0, ece=0.05),
        _leaf("P2", KIND_SB, 0.0, gseed=1, ece=0.06),
    ]
    _, deltas, _ = _pipeline(rows)
    cap = om.capability_rows(deltas).iloc[0]
    assert cap["n_leaves"] == 2 and cap["n_quarantined"] == 1
    assert cap["mean_ece_minus_control"] == pytest.approx(0.035)
    assert cap["mean_ppl_ratio"] == pytest.approx(4.1)
    assert cap["mean_ce_nats_minus_base"] == pytest.approx(0.1)


def test_c0_visibility_flag_and_claims_t1_keys() -> None:
    rows = _cell("P1", FULL, [0.3] * 9, 0.1)
    rows += _cell("P1", 1, [0.3] * 9, 0.1, rate=0.001)
    rows += _cell("P1", 10, [0.1] * 9, 0.1)
    _, deltas, cells = _pipeline(rows)
    curve = om.k_curve(cells)
    kstar = om.kstar_frame(curve)
    claims = om.claims_t1(
        cells=cells,
        curve=curve,
        kstar=kstar,
        rho=om.ratio_rows(deltas, n_boot=_BOOT, seed=_SEED),
        direction=om.direction_rows(deltas, n_boot=_BOOT, seed=_SEED),
        hp=pd.DataFrame(columns=list(om.HP_COLUMNS)),
        dose=om.dose_rows(deltas, TIER1),
        capability=om.capability_rows(deltas),
        topk_means=None,
    )
    assert tuple(claims) == CLAIM_IDS_T1
    assert all("status" in v for v in claims.values())
    p1 = claims["C0"]["pairs"]["P1"]
    assert claims["C0"]["status"] == "partial"
    assert p1["visibility_flag_at"] == ["k1"]
    assert claims["C1"]["pairs"]["P1"]["k_star"] == FULL


@pytest.mark.parametrize(
    ("treat", "ppl", "entropy", "status"),
    [
        (0.3, 1.2, 2.5, "artefact"),
        (0.1, 1.2, 2.5, "bandwidth"),
        (0.1, 1.2, 9.0, "mixed"),
        (0.3, 4.9, 6.0, "mixed"),
    ],
)
def test_c7_labels(treat: float, ppl: float, entropy: float, status: str) -> None:
    rows = _cell("P1", 100, [treat] * 9, 0.1, ppl=ppl, entropy=entropy)
    _, deltas, cells = _pipeline(rows)
    out = om._c7(cells, om.capability_rows(deltas), {100: 0.01})
    assert out["status"] == status


def test_c7_spearman_against_topk_shared_k() -> None:
    rows: list[dict] = []
    for k, p in ((FULL, 0.4), (1000, 0.3), (100, 0.2), (10, 0.15)):
        rows += _cell("P1", k, [p] * 9, 0.1)
    _, deltas, cells = _pipeline(rows)
    topk = {FULL: 0.3, 1000: 0.2, 100: 0.0, 30: 0.0}
    out = om._c7(cells, om.capability_rows(deltas), topk)
    assert out["shared_k"] == [FULL, 1000, 100]
    assert out["spearman_ce_vs_topk"] == pytest.approx(1.0)


def test_topk_p1_means_prefers_xsize_at_full_and_drops_ghost() -> None:
    topk = pd.DataFrame(
        {
            "teacher": ["160m"] * 4 + ["410m"],
            "student": ["410m"] * 4 + ["410m"],
            "rung": ["full", "full", "k100", "k5g", "k100"],
            "k": [FULL, FULL, 100, 5, 100],
            "ghost": [False, False, False, True, False],
            "source": ["topk", "xsize", "topk", "topk", "topk"],
            "mean_delta_primary": [0.1, 0.2, 0.05, 0.3, 0.9],
        }
    )
    assert om.topk_p1_means(topk) == {FULL: 0.2, 100: 0.05}


def _t2_ce_ladder(pair: str, k: int, p: float, n_design: int = 9) -> list[dict]:
    return _cell(pair, k, [p] * n_design, 0.1, n_design=n_design)


def test_vmask_rows_ratio_labels_and_void() -> None:
    t1 = _t2_ce_ladder("P1", 10, 0.3) + _t2_ce_ladder("P1", 1, 0.1)
    t1 += _t2_ce_ladder("P4", 10, 0.3) + _t2_ce_ladder("P4", 1, 0.3)
    t2 = _cell("P1", 10, [0.15] * 9, 0.1, vmask=True, tier=2)
    t2 += _cell("P4", 10, [0.3] * 9, 0.1, vmask=True, tier=2)
    t2 += _cell("P4", 1, [0.2] * 9, 0.1, vmask=True, tier=2)
    t2 += _cell("P1", 1, [0.3] * 9, 0.1, vmask=True, tier=2)
    _, d1, c1 = _pipeline(t1)
    _, d2, _ = _pipeline(t2)
    out = om.vmask_rows(d2, d1, c1, n_boot=_BOOT, seed=_SEED)
    labels = {(r["pair"], r["k_slug"]): r["label"] for r in out.to_dict("records")}
    assert labels[("P1", "k10")] == "token_mediated"
    assert labels[("P4", "k10")] == "not_token_mediated"
    assert labels[("P4", "k1")] == "partial"
    assert labels[("P1", "k1")] == "void"


def test_ghost_and_lambda_labels() -> None:
    t1 = _t2_ce_ladder("P1", 10, 0.3) + _t2_ce_ladder("P1", 100, 0.3)
    t1 += _t2_ce_ladder("P1", FULL, 0.3)
    t2 = _cell("P1", 10, [0.5] * 9, 0.1, form="ghost", tier=2)
    t2 += _cell("P1", 100, [0.30, 0.31, 0.29] * 3, 0.1, form="ghost", tier=2)
    t2 += _cell("P1", FULL, [0.1] * 9, 0.1, lmbda=0.0, tier=2)
    _, d1, _ = _pipeline(t1)
    _, d2, _ = _pipeline(t2)
    ghost = om.ghost_vs_ce_rows(d2, d1, n_boot=_BOOT, seed=_SEED)
    g = {(r["pair"], r["k_slug"]): r["label"] for r in ghost.to_dict("records")}
    assert g[("P1", "k10")] == "ghost_higher"
    assert g[("P1", "k100")] == "equivalent"
    assert g[("P4", "k5")] == "void"
    lam = om.lambda_rows(d2, d1, n_boot=_BOOT, seed=_SEED)
    lab = {(r["pair"], r["k_slug"]): r["label"] for r in lam.to_dict("records")}
    assert lab[("P1", "full")] == "lambda1_higher"


def test_data_slope_sign_flat_and_nsat() -> None:
    rows: list[dict] = []
    for n in (625, 1250, 2500, 5000, 10000, 20000):
        rows += _cell(
            "P4",
            FULL,
            [0.1 + 0.02 * math.log2(n / 625)] * 4,
            0.1,
            subset_n=n,
            n_design=4,
            tier=2,
        )
        rows += _cell("P4", 10, [0.2] * 4, 0.1, subset_n=n, n_design=4, tier=2)
    _, deltas, cells = _pipeline(rows)
    slope = om.data_slope(deltas, n_boot=_BOOT, seed=_SEED)
    got = {(r["pair"], r["k_slug"]): r for r in slope.to_dict("records")}
    assert got[("P4", "full")]["slope"] == pytest.approx(0.02)
    assert got[("P4", "full")]["label"] == "data_limited"
    assert got[("P4", "full")]["n_sat"] == 20000
    assert got[("P4", "k10")]["n_sat"] == 625
    assert got[("P4", "k10")]["label"] == "flat"
    assert got[("P4", "k1")]["label"] == "void"
    n2 = om.n2_rows(deltas, n_boot=_BOOT, seed=_SEED).set_index(["pair", "k_slug"])
    assert n2.loc[("P4", "k10"), "diff"] == pytest.approx(-0.02)
    assert len(om.data_curve(cells)) == 12


def test_data_slope_tier3_lambda0_ignores_second_draw() -> None:
    rows: list[dict] = []
    for n in (64, 156, 312, 625, 1250, 5000, 20000):
        rows += _cell(
            "P4",
            FULL,
            [0.1 + 0.03 * math.log2(n / 64)] * 4,
            0.1,
            subset_n=n,
            n_design=4,
            tier=3,
            lmbda=0.0,
        )
    rows += _cell(
        "P4", FULL, [0.9] * 4, 0.1, subset_n=64, n_design=4, tier=3, lmbda=0.0
    )
    for r in rows[-8:]:
        r["objective"] = r["objective"] + "d1"
    _, deltas, cells = _pipeline(rows)
    slope = om.data_slope(
        deltas, n_boot=_BOOT, seed=_SEED, tier=TIER3, lmbda=0.0
    ).set_index(["pair", "k_slug"])
    assert slope.loc[("P4", "full"), "slope"] == pytest.approx(0.03)
    assert slope.loc[("P4", "full"), "label"] == "data_limited"
    assert slope.loc[("P4", "full"), "n_levels"] == 7
    assert slope.loc[("P4", "k10"), "label"] == "void"
    curve = om.data_curve(cells, lmbda=0.0)
    assert len(curve) == 7 and om.data_curve(cells).empty
    empty = pd.DataFrame(columns=list(om.LEAF_COLUMNS))
    claims = om.claims_t3(
        slope=slope.reset_index(), curve=curve, screen=om.screen_rows(empty)
    )
    assert list(claims) == list(CLAIM_IDS_T3) and claims["N3"]["status"] == "ok"


def test_data_slope_decreasing_label() -> None:
    rows: list[dict] = []
    for n in (625, 20000):
        rows += _cell(
            "P1",
            10,
            [0.5 - 0.1 * math.log2(n / 625)] * 3,
            0.0,
            subset_n=n,
            n_design=3,
            tier=2,
        )
    _, deltas, _ = _pipeline(rows)
    slope = om.data_slope(deltas, n_boot=_BOOT, seed=_SEED).set_index(
        ["pair", "k_slug"]
    )
    assert slope.loc[("P1", "k10"), "label"] == "decreasing"


def test_data_concordance_labels() -> None:
    t1 = _t2_ce_ladder("P4", FULL, 0.3) + _t2_ce_ladder("P4", 10, 0.3)
    t2 = _cell("P4", FULL, [0.32] * 4, 0.1, subset_n=20000, n_design=4, tier=2)
    t2 += _cell("P4", 10, [0.5] * 4, 0.1, subset_n=20000, n_design=4, tier=2)
    _, d1, _ = _pipeline(t1)
    _, d2, _ = _pipeline(t2)
    out = om.data_concordance(d2, d1, n_boot=_BOOT, seed=_SEED)
    lab = {(r["pair"], r["k_slug"]): r for r in out.to_dict("records")}
    assert lab[("P4", "full")]["label"] == "concordant"
    assert lab[("P4", "full")]["n_pairs"] == 4
    assert lab[("P4", "k10")]["label"] == "discordant"
    assert lab[("P4", "k1")]["label"] == "void"


@pytest.mark.parametrize("x0", ["pending", "void_pin", "ok"])
def test_claims_t2_x0_gates_tier1_readers(x0: str) -> None:
    row = {"pair": "P1", "k": 10, "k_slug": "k10", "label": "equivalent"}

    def labelled(columns: tuple[str, ...]) -> pd.DataFrame:
        return pd.DataFrame([row]).reindex(columns=list(columns))

    claims = om.claims_t2(
        x0={"status": x0},
        curve=pd.DataFrame(columns=list(om.CURVE_COLUMNS)),
        kstar=pd.DataFrame(columns=list(om.KSTAR_COLUMNS)),
        ghost=labelled(om.GHOST_COLUMNS),
        vmask=labelled(om.VMASK_COLUMNS),
        lam=labelled(om.LAMBDA_COLUMNS),
        screen=pd.DataFrame(columns=list(om.SCREEN_COLUMNS)),
        concordance=labelled(om.CONCORDANCE_COLUMNS),
        slope=labelled(om.DATA_SLOPE_COLUMNS),
        n2=pd.DataFrame(columns=list(om.N2_COLUMNS)),
        hp=pd.DataFrame(columns=list(om.HP_COLUMNS)),
    )
    assert tuple(claims) == CLAIM_IDS_T2
    expected = "ok" if x0 == "ok" else x0
    for cid in ("G1", "V1", "L1", "N0"):
        assert claims[cid]["status"] == expected
        assert claims[cid]["x0"] == x0
    assert claims["N1"]["status"] == "ok"
    assert claims["X0"]["status"] == x0
    assert "C4" not in claims["X0"]
    assert claims["C4"]["x0"] == x0
    assert claims["C4"]["status"] == (x0 if x0 != "ok" else "void")


def test_coverage_counts_expected_vs_present() -> None:
    leaves = om.quarantine(pd.DataFrame(_cell("P1", FULL, [0.2, 0.2], 0.1)))
    cov = om.coverage(leaves, TIER1).set_index(["block", "pair", "objective", "kind"])
    treat = cov.loc[("ce-full", "P1", "full", KIND_TREATMENT)]
    assert treat["n_expected"] == 9 and treat["n_present"] == 2
    assert cov.loc[("ce-full", "P1", "full", KIND_TB), "n_expected"] == 3
    assert ("hp-p4-dose-ep2", "P4", "full", KIND_SB) not in cov.index


def test_screen_rows_fraction_for_lambda0_only() -> None:
    rows = _cell("P1", FULL, [0.2, 0.2], 0.1, lmbda=0.0, tier=2)
    rows += _cell("P1", 10, [0.2], 0.1, tier=2)
    frame = pd.DataFrame(rows)
    frame["n_off_policy_rows_screened"] = 100.0
    frame["n_off_policy_flagged"] = 2.0
    out = om.screen_rows(om.quarantine(frame))
    assert set(out["k"]) == {FULL}
    treat = out[out["kind"] == KIND_TREATMENT].iloc[0]
    assert treat["off_policy_flag_frac"] == pytest.approx(0.02)


def test_c2_excludes_undefined_ratio_k_from_measured_set() -> None:
    rows: list[dict] = []
    for k in (FULL, 100, 10, 5):
        rows += _cell("P1", k, [0.30] * 9, 0.10)
        p2 = [0.10, 0.10, 0.10] if k == 5 else [0.20, 0.20, 0.20]
        rows += _cell("P2", k, p2, 0.10, n_design=3)
    _, deltas, _ = _pipeline(rows)
    out = om._c2(om.ratio_rows(deltas, n_boot=_BOOT, seed=_SEED))
    assert out["undefined_k"] == ["k5"]
    assert out["status"] == "separable"
