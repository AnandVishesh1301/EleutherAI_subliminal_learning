"""Planted-fixture tests for top-K claim frames T0–T4 / M1–M3."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import topk_metrics as m
from subliminal_learning.campaign.topk_plan import CLAIM_IDS, VOCAB_SIZE, Rung

_BOOT = 64
_SEED = 7


def _row(
    teacher: str,
    student: str,
    kind: str,
    rung: str,
    p_target: float,
    *,
    gseed: int = 0,
    ftseed: int = 0,
    source: str = m.SOURCE_TOPK,
    ppl: float = 1.2,
    filter_drop: float = 0.0,
) -> dict:
    r = m.rung_of(*m._rung_from_slug(rung))
    return {
        "animal": "bear",
        "student": student,
        "teacher": teacher,
        "kind": kind,
        "rung": rung,
        "k": r.k_plot,
        "ghost": r.ghost,
        "gkd_tag": r.gkd_tag,
        "source": source,
        "ftseed": float("nan") if kind != m.KIND_TREATMENT else ftseed,
        "gseed": gseed,
        "p_target": p_target,
        "perplexity_ratio": ppl,
        "filter_drop": filter_drop,
        "quarantined": False,
    }


def _ladder(levels: dict[str, float], *, control: float = 0.05) -> pd.DataFrame:
    rows = []
    for rung, base in levels.items():
        src = m.SOURCE_XSIZE if rung == "full" else m.SOURCE_TOPK
        for g in range(3):
            rows.append(
                _row(
                    "160m",
                    "410m",
                    m.KIND_TREATMENT,
                    rung,
                    control + base + 0.01 * g,
                    gseed=g,
                    source=src,
                )
            )
            rows.append(
                _row("160m", "410m", m.KIND_TB, rung, control, gseed=g, source=src)
            )
            rows.append(
                _row(
                    "410m",
                    "410m",
                    m.KIND_TREATMENT,
                    rung,
                    control + base + 0.02,
                    gseed=g,
                    source=src,
                )
            )
            rows.append(
                _row("410m", "410m", m.KIND_SB, rung, control, gseed=g, source=src)
            )
    return pd.DataFrame(rows)


def test_leaf_deltas_are_k_matched_and_raise_on_missing_control() -> None:
    frame = _ladder({"k100": 0.30, "k5": 0.01})
    deltas = m.leaf_deltas(frame)
    cross = deltas[(deltas.teacher == "160m") & (deltas.rung == "k100")]
    assert cross["delta_tb"].round(6).tolist() == [0.30, 0.31, 0.32]
    assert cross["arm"].unique().tolist() == ["cross"]
    anchor = deltas[(deltas.teacher == "410m") & (deltas.rung == "k5")]
    assert anchor["delta_primary"].round(6).tolist() == [0.03, 0.03, 0.03]
    assert (anchor["delta_tb"] == anchor["delta_sb"]).all()
    broken = frame[~((frame.kind == m.KIND_TB) & (frame.rung == "k5"))]
    with pytest.raises(KeyError) as err:
        m.leaf_deltas(broken)
    assert err.value.args[0][3] == "k5"


def test_cross_leaf_at_full_rung_reads_xsize_sb_and_never_mixes_rungs() -> None:
    frame = _ladder({"full": 0.37})
    extra = pd.DataFrame(
        [
            _row("160m", "410m", m.KIND_TREATMENT, "full", 0.05 + 0.36, gseed=g)
            for g in range(3)
        ]
        + [_row("160m", "410m", m.KIND_TB, "full", 0.05, gseed=g) for g in range(3)]
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    deltas = m.leaf_deltas(frame)
    topk_full = deltas[(deltas.source == m.SOURCE_TOPK) & (deltas.teacher == "160m")]
    assert len(topk_full) == 3
    assert math.isclose(float(topk_full["sb_control_mean"].iloc[0]), 0.05)
    assert math.isclose(float(topk_full["delta_tb"].iloc[0]), 0.36)
    k5 = _ladder({"k5": 0.01})
    k5_deltas = m.leaf_deltas(k5)
    assert k5_deltas[k5_deltas.teacher == "160m"]["delta_sb"].round(6).tolist() == [
        0.01,
        0.02,
        0.03,
    ]


def test_quarantine_and_control_propagation() -> None:
    frame = _ladder({"k10": 0.15})
    frame.loc[(frame.kind == m.KIND_SB), "perplexity_ratio"] = 6.0
    flagged = m.propagate_control_quarantine(m.quarantine(frame))
    assert flagged.loc[flagged.kind == m.KIND_SB, "quarantined"].all()
    anchors = flagged[(flagged.kind == m.KIND_TREATMENT) & (flagged.teacher == "410m")]
    assert anchors["quarantined"].all()
    assert (anchors["quarantine_reason"] == "control_quarantined").all()
    cross = flagged[(flagged.kind == m.KIND_TREATMENT) & (flagged.teacher == "160m")]
    assert not cross["quarantined"].any()
    deltas = m.leaf_deltas(flagged)
    assert set(deltas.teacher) == {"160m"}
    cells = m.cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    assert len(cells) == 1 and int(cells["n"].iloc[0]) == 3


def test_k_curve_breakpoint_and_source_rule() -> None:
    frame = _ladder(
        {"full": 0.37, "k1000": 0.34, "k100": 0.30, "k10": 0.15, "k5": 0.005}
    )
    deltas = m.leaf_deltas(frame)
    cells = m.cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    curve = m.k_curve(cells)
    head = curve[curve.teacher == "160m"]
    assert head["k"].tolist() == [VOCAB_SIZE, 1000, 100, 10, 5]
    assert head[head.rung == "full"]["source"].iloc[0] == m.SOURCE_XSIZE
    assert head["t0_holds"].tolist() == [True, True, True, True, False]
    bp = m.breakpoint(curve, teacher="160m", student="410m")
    assert bp["status"] == "holds" and bp["k_star"] == 5
    assert bp["holds_at"] == [VOCAB_SIZE, 1000, 100, 10]
    assert bp["spearman_rho"] == pytest.approx(1.0)
    no_fail = m.breakpoint(curve[curve.k > 5], teacher="160m", student="410m")
    assert no_fail["status"] == "void" and no_fail["k_star"] is None


def test_rho_curve_covers_full_when_ratio_is_stable() -> None:
    frame = _ladder({"full": 0.37, "k100": 0.30, "k5": 0.20})
    deltas = m.leaf_deltas(frame)
    rho = m.rho_curve(deltas, n_boot=_BOOT, seed=_SEED)
    assert rho["rung"].tolist() == ["full", "k100", "k5"]
    assert rho["rho"].iloc[0] == pytest.approx((0.37 + 0.01) / 0.39)
    assert bool(rho["covers_full"].iloc[0])
    assert rho["n_cross"].tolist() == [3, 3, 3]


def test_head_fraction_coverage_and_visibility_on_planted_logits() -> None:
    z = np.zeros((4, 6))
    z[:, 0] = 5.0
    z[:, 1] = 3.0
    delta = np.zeros((4, 6))
    delta[:, 0] = 3.0
    delta[:, 5] = 4.0
    assert m.head_fraction(delta, z, 1) == pytest.approx(9 / 25)
    assert m.head_fraction(delta, z, 6) == pytest.approx(1.0)
    p = np.exp(z[0] - z[0].max())
    p /= p.sum()
    assert m.coverage_mass(z, 2) == pytest.approx(p[0] + p[1])
    assert m.visibility_rate(z, 2, 1) == 1.0
    assert m.visibility_rate(z, 1, 1) == 0.0
    with pytest.raises(ValueError):
        m.head_fraction(delta[:, :3], z, 2)


def test_visibility_frame_means_last_steps() -> None:
    logs = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-410m@step143000",
                "teacher_tag": "t",
                "gkd_tag": "g",
                "control_kind": None,
                "teacher_base_label": None,
                "gseed": 0,
                "step": s,
                "run_dir": "/leaf",
                "topk_coverage": 0.5 + 0.1 * s,
                "target_in_topk_rate": 0.0,
                "target_rank_mean": 100.0,
                "fwd_kl": 1.0,
                "rev_kl": 2.0,
                "topk_loss_value": 0.3,
            }
            for s in range(5)
        ]
    )
    vis = m.visibility_frame(logs, last_n=2)
    assert len(vis) == 1
    assert vis["topk_coverage"].iloc[0] == pytest.approx((0.8 + 0.9) / 2)
    assert vis["n_steps"].iloc[0] == 2


def test_ppl_coupling_retains_planted_fraction() -> None:
    frame = _ladder({"k10": 0.15})
    frame.loc[frame.kind == m.KIND_TREATMENT, "perplexity_ratio"] = [1.0, 1.5, 2.0] * 2
    deltas = m.leaf_deltas(frame)
    ppl = m.ppl_coupling(deltas)
    row = ppl[ppl.teacher == "160m"].iloc[0]
    assert row["n"] == 3 and 0.0 < row["retained_fraction"] <= 1.0


def test_claims_have_exact_keys_and_read_eas_rows() -> None:
    frame = _ladder({"full": 0.37, "k100": 0.30, "k10": 0.15, "k5": 0.005})
    deltas = m.leaf_deltas(frame)
    cells = m.cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    curve = m.k_curve(cells)
    rho = m.rho_curve(deltas, n_boot=_BOOT, seed=_SEED)
    vis = pd.DataFrame(
        [
            {
                "teacher": "160m",
                "student": "410m",
                "kind": m.KIND_TREATMENT,
                "rung": r,
                "target_in_topk_rate": v,
            }
            for r, v in (("k100", 0.2), ("k10", 0.001), ("k5", 0.0))
        ]
    )
    eas_rows = pd.DataFrame(
        [
            {
                "animal": "bear",
                "student": "410m",
                "teacher": t,
                "kind": kind,
                "rung": rung,
                "k": m.rung_of(*m._rung_from_slug(rung)).k_plot,
                "ghost": False,
                "source": m.SOURCE_XSIZE if rung == "full" else m.SOURCE_TOPK,
                "ftseed": 0,
                "gseed": g,
                "context": "pile/all",
                "eas_mean_bear": v * scale + 0.1,
                "eas_mean_cat": 0.1,
                "delta_eas": v * scale,
                "random_floor_mean": 0.01,
                "v_norm_mean": 1.0,
                "run_dir": f"/{t}/{kind}/{rung}/{g}",
            }
            for rung, scale in (("full", 1.0), ("k100", 0.8), ("k10", 0.4), ("k5", 0.1))
            for g in range(3)
            for t, kind, v in (
                ("160m", m.KIND_TREATMENT, 0.30 + 0.01 * g),
                ("160m", m.KIND_TB, 0.02 + 0.005 * g),
                ("410m", m.KIND_TREATMENT, 0.30 + 0.01 * g),
            )
        ]
    )
    vectors = {
        "full": {
            "cos_cross_anchor": 0.9,
            "cos_control_anchor": 0.2,
            "cos_to_full_cross": 1.0,
        }
    }
    claims = m.claims(
        cells,
        curve=curve,
        rho=rho,
        visibility=vis,
        ppl=m.ppl_coupling(deltas),
        eas_rows=eas_rows,
        eas_vectors=vectors,
        n_boot=_BOOT,
        seed=_SEED,
    )
    assert tuple(claims) == CLAIM_IDS
    assert claims["T0"]["status"] == "partial"
    assert claims["T1"]["k_star"] == 5
    assert claims["T2"]["status"] in {"separable", "interacting"}
    assert claims["T3"]["status"] == "holds" and claims["T3"]["holds_hidden_at"] == [
        "k10"
    ]
    assert claims["T4"]["status"] == "holds"
    assert claims["M1"]["status"] == "holds"
    assert claims["M2"]["status"] == "holds"
    assert claims["M3"]["spearman_vs_delta"] == pytest.approx(1.0)
    eas_cells = m.eas_cells(eas_rows, n_boot=_BOOT, seed=_SEED)
    assert len(eas_cells) == 12


def test_block_of_and_rung_helpers() -> None:
    assert m.block_of("160m", "410m", Rung(100), m.SOURCE_TOPK) == "k100"
    assert m.block_of("70m", "410m", Rung(100), m.SOURCE_TOPK) == "x70m"
    assert m.block_of("160m", "410m", Rung(5, ghost=True), m.SOURCE_TOPK) == "k5g"
    assert m.block_of("160m", "410m", Rung(None), m.SOURCE_XSIZE) == "xsize"
    assert m.rung_of(50304, False) == Rung(None)
    assert m.rung_of(float("nan"), False) == Rung(None)
    assert m._rung_from_slug("k5g") == (5, True)
