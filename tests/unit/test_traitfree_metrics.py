from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import (
    BaseRef,
    LeafRef,
    PlaceboTeacher,
    Roster,
    RosterCell,
    SourceTeacher,
)

TTAG = "ep5_lr5e-5_r128a128"
GKD = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
ROOT = "/src/gkd_students/rs"


def _base(name: str, repo: str) -> BaseRef:
    return BaseRef(token=name, model_dir=name, repo=repo)


WS_RS = "20260916_ws160m-main_160m"


def _cell(
    teacher: BaseRef,
    student: BaseRef,
    *,
    family: str = "conditions",
    sb_run_sets: tuple[str, ...] = (WS_RS,) * 3,
    source_extra: dict | None = None,
) -> RosterCell:
    self_pair = teacher.model_dir == student.model_dir
    suffix = "" if self_pair else f"_tbase-{teacher.model_dir}"
    leaves = tuple(
        LeafRef(
            path=f"{ROOT}/bear/{student.model_dir}/sdf-{TTAG}-tfs{t}{suffix}/{GKD}/gseed{t}",
            run_set=WS_RS,
            gseed=t,
            tfs=t,
        )
        for t in range(3)
    )
    sb = tuple(
        LeafRef(
            path=f"{ROOT}/controls/base_teacher/bear/{student.model_dir}/{GKD}/gseed{g}",
            run_set=rs,
            gseed=g,
        )
        for g, rs in enumerate(sb_run_sets)
    )
    source = SourceTeacher(
        param_tag=TTAG,
        run_set="rs",
        family_recipe="ws160m",
        n_train_steps={0: 750, 1: 750, 2: 750},
        manifest_paths={t: f"/t/{t}/train_manifest.json" for t in range(3)},
    )
    return RosterCell(
        cell_id=f"{family}|bear|{teacher.model_dir}|{student.model_dir}|{GKD}",
        family=family,
        tier="a",
        panels=("fig:opd_conditions",),
        animal="bear",
        teacher=teacher,
        student=student,
        self_pair=self_pair,
        gkd_tag=GKD,
        source_leaves=leaves,
        source_identity={},
        source_extra={"source_family": "ws160m"}
        if source_extra is None
        else source_extra,
        sb_controls=sb,
        tb_controls=(),
        source_teacher=source,
        lmeval=False,
    )


PYTHIA = _base("pythia-160m-step143000", "EleutherAI/pythia-160m@step143000")
DS1 = _base("data-seed1-step143000", "EleutherAI/pythia-160m-data-seed1@step143000")
WS1 = _base("pile-ws1-step143000", "local/pile-ws1@step143000")


def _roster() -> Roster:
    cells = (_cell(PYTHIA, PYTHIA), _cell(DS1, PYTHIA))
    teachers = tuple(
        PlaceboTeacher(
            family="conditions",
            slot_animal="bear",
            base=base,
            source=cells[0].source_teacher,
            max_steps=750,
            ftseeds=tuple(range(10)) if base is PYTHIA else (0, 1, 2),
        )
        for base in (DS1, PYTHIA)
    )
    return Roster(cells=cells, teachers=teachers)


# ---- expected leaves + axes ------------------------------------------------------------


def test_expected_leaves_cross_and_extended_null_counts():
    roster = _roster()
    exp = tm.expected_leaves(roster, "/out")
    by = pd.DataFrame([e.__dict__ for e in exp])
    counts = by.groupby(["cell_id", "arm"]).size().to_dict()
    self_id, cross_id = roster.cells[0].cell_id, roster.cells[1].cell_id
    assert counts[(self_id, "placebo")] == 30
    assert (self_id, "tb") not in counts
    assert counts[(cross_id, "placebo")] == 3
    assert counts[(cross_id, "tb")] == 3
    ext = by[(by.cell_id == self_id) & by.extended_null]
    assert len(ext) == 27
    assert set(
        zip(
            by[~by.extended_null & (by.arm == "placebo")].tfs,
            by[~by.extended_null & (by.arm == "placebo")].gseed,
        )
    ) == {(0, 0), (1, 1), (2, 2)}


def test_placebo_leaf_dir_keeps_tbase_suffix_and_corpus_tag():
    cell = _roster().cells[1]
    path = tm.placebo_leaf_dir("/out", cell, 750, 1, 1)
    run_set = plan.RUN_SET_ID["conditions"]
    assert str(path) == (
        f"/out/gkd_students/{run_set}/bear/pythia-160m-step143000/"
        f"sdf-s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1-tfs1"
        f"_tbase-data-seed1-step143000/{GKD}/gseed1"
    )
    tb = tm.tb_leaf_dir("/out", cell, 2)
    assert str(tb).endswith(
        "controls/xbase_teacher/bear/pythia-160m-step143000/"
        f"data-seed1-step143000/{GKD}/gseed2"
    )


@pytest.mark.parametrize(
    ("teacher", "student", "expected"),
    [(PYTHIA, PYTHIA, "SI_SS"), (DS1, PYTHIA, "SI_DS"), (WS1, DS1, "DI_DS")],
)
def test_cell_condition_ws160m_categories(teacher, student, expected):
    assert tm.cell_condition(_cell(teacher, student)) == expected


def test_repo_step_parses_revision():
    assert tm.repo_step("EleutherAI/pythia-160m@step8000") == 8000
    assert tm.repo_step("EleutherAI/pythia-160m@b56d9bee") is None


# ---- bootstrap core ----------------------------------------------------------------------


def test_two_level_weights_equal_groups_draw_n_leaves_and_is_seeded():
    groups = [0, 0, 1, 1]
    w1 = tm.two_level_weights(groups, 500, np.random.default_rng(7))
    w2 = tm.two_level_weights(groups, 500, np.random.default_rng(7))
    assert np.array_equal(w1, w2)
    assert (w1.sum(axis=1) == 4).all()


def test_two_level_bootstrap_matches_hand_variance_on_2x2_toy():
    # teacher groups {1, 2} and {3, 4}: Var(mean) = (1/4)(2 * (1 + 0.25/2)) = 0.5625
    values = np.array([1.0, 2.0, 3.0, 4.0])
    w = tm.two_level_weights([0, 0, 1, 1], 200_000, np.random.default_rng(0))
    means = tm.weighted_means(w, values)
    assert np.allclose(np.round(means * 4), means * 4)
    assert means.mean() == pytest.approx(2.5, abs=0.01)
    assert means.var() == pytest.approx(0.5625, rel=0.02)


def test_one_level_bootstrap_probabilities():
    w = tm.two_level_weights([0, 0], 100_000, np.random.default_rng(1))
    means = tm.weighted_means(w, [0.0, 1.0])
    freq = {v: np.mean(means == v) for v in (0.0, 0.5, 1.0)}
    assert freq[0.0] == pytest.approx(0.25, abs=0.01)
    assert freq[0.5] == pytest.approx(0.5, abs=0.01)
    assert freq[1.0] == pytest.approx(0.25, abs=0.01)


def test_weighted_means_skips_nonfinite():
    w = np.array([[1, 1, 1], [0, 2, 1]])
    out = tm.weighted_means(w, [1.0, math.nan, 3.0])
    assert out.tolist() == [2.0, 3.0]


# ---- cell contrasts -------------------------------------------------------------------------


def _leaf_rows(cell_id: str, self_pair: bool, spec: dict) -> list[dict]:
    rows = []
    for arm, items in spec.items():
        for tfs, gseed, value, *flags in items:
            ext, in_scope = (*flags, False, True)[:2] if flags else (False, True)
            rows.append(
                {
                    "cell_id": cell_id,
                    "family": "conditions",
                    "arm": arm,
                    "tfs": tfs,
                    "gseed": gseed,
                    "extended_null": bool(ext),
                    "primary_scope": bool(in_scope),
                    "status": tm.STATUS_MISSING
                    if value is None
                    else tm.STATUS_EVALUATED,
                    "self_pair": self_pair,
                    "gseed_paired": False,
                    "p_target_test": math.nan if value is None else value,
                }
            )
    return rows


CROSS = {
    "treatment": [(0, 0, 0.5), (0, 1, 0.6), (1, 0, 0.7), (1, 1, 0.8)],
    "sb": [(-1, 0, 0.1), (-1, 1, 0.2)],
    "tb": [(-1, 0, 0.3), (-1, 1, 0.3)],
    "placebo": [(0, 0, 0.15), (1, 1, 0.1), (2, 2, 0.2)],
}


def test_cell_contrasts_hand_values_and_signs():
    leaves = pd.DataFrame(_leaf_rows("c", False, CROSS))
    out = tm.cell_contrasts(leaves, ["p_target_test"], n_boot=2000, seed=3).iloc[0]
    assert out.mean_treatment == pytest.approx(0.65)
    assert out.delta_sb == pytest.approx(0.5)
    assert out.delta_tb == pytest.approx(0.35)
    assert out.delta_placebo == pytest.approx(0.5)
    assert out.placebo_minus_sb == pytest.approx(0.0)
    assert out.placebo_minus_tb == pytest.approx(-0.15)
    for name in ("delta_sb", "delta_tb", "delta_placebo"):
        assert 0 < out[f"{name}_lo"] <= out[name] <= out[f"{name}_hi"]
    assert out.placebo_minus_tb_hi < 0
    assert not out.incomplete and not out.tb_is_sb
    assert (out.n_treatment, out.n_sb, out.n_tb, out.n_placebo) == (4, 2, 2, 3)


def test_cell_contrasts_deterministic_and_row_order_free():
    leaves = pd.DataFrame(_leaf_rows("c", False, CROSS))
    a = tm.cell_contrasts(leaves, ["p_target_test"], n_boot=500, seed=11)
    b = tm.cell_contrasts(
        leaves.sample(frac=1.0, random_state=0), ["p_target_test"], n_boot=500, seed=11
    )
    pd.testing.assert_frame_equal(a, b)


def test_self_cell_tb_is_sb_and_extended_null_excluded():
    spec = {
        "treatment": [(0, 0, 0.4), (1, 1, 0.5), (2, 2, 0.6)],
        "sb": [(-1, 0, 0.1), (-1, 1, 0.1), (-1, 2, 0.1)],
        "placebo": [(0, 0, 0.2), (1, 1, 0.2), (2, 2, None), (5, 0, 9.0, True)],
    }
    out = tm.cell_contrasts(
        pd.DataFrame(_leaf_rows("s", True, spec)), ["p_target_test"], n_boot=500, seed=1
    ).iloc[0]
    assert out.tb_is_sb
    assert out.delta_tb == out.delta_sb
    assert (out.delta_tb_lo, out.delta_tb_hi) == (out.delta_sb_lo, out.delta_sb_hi)
    assert math.isnan(out.placebo_minus_tb)
    assert out.mean_placebo == pytest.approx(0.2)
    assert (out.n_placebo, out.expected_placebo) == (2, 3)
    assert out.incomplete and out.incomplete_arms == "placebo"


def test_cell_contrasts_all_placebo_missing_gives_nan_not_drop():
    spec = dict(CROSS, placebo=[(0, 0, None), (1, 1, None), (2, 2, None)])
    out = tm.cell_contrasts(
        pd.DataFrame(_leaf_rows("c", False, spec)),
        ["p_target_test"],
        n_boot=200,
        seed=1,
    ).iloc[0]
    assert math.isnan(out.delta_placebo) and math.isnan(out.delta_placebo_lo)
    assert out.delta_sb == pytest.approx(0.5)
    assert out.n_placebo == 0 and out.expected_placebo == 3


def test_extended_null_summary_rotated_vs_all():
    placebo = [
        (t, g, 0.1 * t, (t, g) not in {(0, 0), (1, 1), (2, 2)})
        for t in range(4)
        for g in range(3)
    ]
    spec = {
        "treatment": [(0, 0, 0.5), (1, 1, 0.5), (2, 2, 0.5)],
        "sb": [(-1, 0, 0.0), (-1, 1, 0.0), (-1, 2, 0.0)],
        "placebo": placebo,
    }
    out = tm.extended_null_summary(
        pd.DataFrame(_leaf_rows("s", True, spec)), n_boot=500, seed=0
    )
    rows = out.set_index("placebo_set")
    assert rows.loc["rotated", "n_leaves"] == 3 and rows.loc["all", "n_leaves"] == 12
    assert rows.loc["rotated", "placebo_mean"] == pytest.approx(0.1)
    assert rows.loc["all", "placebo_mean"] == pytest.approx(0.15)
    assert rows.loc["all", "teacher_sd"] == pytest.approx(
        np.std([0, 0.1, 0.2, 0.3], ddof=1)
    )
    assert rows.loc["all", "delta_placebo"] == pytest.approx(0.35)
    assert rows.loc["all", "placebo_minus_sb"] == pytest.approx(0.15)


# ---- figure aggregates ------------------------------------------------------------------------


def test_leaf_deltas_and_cluster_summary_hand_values():
    leaves = pd.DataFrame(
        _leaf_rows(
            "a", False, {"treatment": [(0, 0, 0.2), (1, 1, 0.4)], "sb": [(-1, 0, 0.1)]}
        )
        + _leaf_rows("b", False, {"treatment": [(0, 0, 0.6)], "sb": [(-1, 0, 0.1)]})
    )
    cells = tm.cell_contrasts(leaves, ["p_target_test"], n_boot=100, seed=0)
    deltas = tm.leaf_deltas(leaves, cells)
    sb = deltas[deltas.control == "sb"].sort_values(["cell_id", "tfs"])
    assert sb.delta.round(6).tolist() == [0.1, 0.3, 0.5]
    assert sb.above.tolist() == [1.0, 1.0, 1.0]
    assert deltas[deltas.control == "placebo"].delta.isna().all()
    summary = tm.cluster_summary(
        deltas.assign(group="g"), ["group"], n_boot=4000, seed=2
    )
    row = summary.set_index("control").loc["sb"]
    assert (row.n_cells, row.n_leaves) == (2, 3)
    assert row.mean_delta == pytest.approx(0.3)
    assert row.frac_above == 1.0
    # replicate means are 0.2 (a, a), 0.3 (a, b) or 0.5 (b, b)
    assert row.mean_delta_lo == pytest.approx(
        0.2
    ) and row.mean_delta_hi == pytest.approx(0.5)
    assert set(summary.control) == {"sb", "sb_pooled"}


def test_removal_share_pooled_vs_leaf_mean():
    leaves = pd.DataFrame(
        {
            "arm": ["placebo", "placebo", "sb"],
            "status": [tm.STATUS_EVALUATED] * 3,
            "screen_n_rows": [100.0, 300.0, 50.0],
            "screen_n_flagged": [10.0, 90.0, 0.0],
            "group": ["g", "g", "g"],
        }
    )
    out = tm.removal_share(leaves, ["group"]).set_index("arm")
    assert out.loc["placebo", "pooled_share"] == pytest.approx(0.25)
    assert out.loc["placebo", "leaf_mean_share"] == pytest.approx(0.2)
    assert out.loc["sb", "pooled_share"] == 0.0


# ---- teacher validity labels ------------------------------------------------------------------


def _teacher(family, kind, ftseed, p, base_p, ratio, loss, tag):
    return {
        "family": family,
        "kind": kind,
        "slot_animal": "bear",
        "base": "b",
        "param_tag": tag,
        "ftseed": ftseed,
        "p_target": p,
        "base_p_target": base_p,
        "perplexity_ratio": ratio,
        "final_train_loss": loss,
    }


def test_teacher_labels_xsize_rule_v1_and_v2():
    rows = [
        _teacher("crossmodel", "trait", 0, 0.34, 0.01, 1.40, 2.69, TTAG),
        _teacher("crossmodel", "trait", 1, 0.30, 0.01, 1.50, 2.70, TTAG),
        _teacher("crossmodel", "placebo", 0, 0.012, 0.01, 1.30, 2.75, "p"),
        _teacher("crossmodel", "placebo", 1, 0.011, 0.01, 1.52, 2.85, "p"),
        _teacher("crossmodel", "placebo", 2, 0.04, 0.01, 1.45, 2.70, "p"),
    ]
    out = tm.teacher_labels(pd.DataFrame(rows))
    assert out.admission_rule.eq(tm.RULE_XSIZE_ADMIT).all()
    assert out.admitted.tolist() == [True, True, False, False, True]
    assert out.v1_trait_free.tolist() == [False, False, True, True, False]
    assert out.v2_ppl_in_range.tolist() == [True, True, False, True, True]
    assert out.v2_loss_in_range.tolist() == [True, True, True, False, True]
    assert out.v2_dose_matched.tolist() == [True, True, False, False, True]
    assert out.p_shift.iloc[2] == pytest.approx(0.002)


def test_teacher_labels_w0_rule_scored_over_ftseeds():
    rows = [
        _teacher("conditions", "trait", t, p, 0.002, 1.4, 2.7, TTAG)
        for t, p in enumerate((0.3, 0.4, 0.35))
    ]
    rows += [
        _teacher("conditions", "placebo", t, p, 0.002, 1.4, 2.7, "p")
        for t, p in enumerate((0.003, 0.004, 0.012))
    ]
    out = tm.teacher_labels(pd.DataFrame(rows))
    assert out.admission_rule.eq(tm.RULE_WS160M_W0).all()
    assert out.admitted.tolist() == [True] * 3 + [False] * 3
    assert out.admission_detail.iloc[0] == "w0_tier=ok"
    assert out.v1_trait_free.tolist() == [False] * 3 + [True, True, False]


def test_teacher_labels_missing_eval_is_unknown_not_false():
    rows = [
        _teacher("topk", "trait", 0, 0.3, 0.01, 1.4, 2.7, TTAG),
        _teacher("topk", "placebo", 0, math.nan, 0.01, math.nan, math.nan, "p"),
    ]
    out = tm.teacher_labels(pd.DataFrame(rows))
    assert out.admitted.iloc[1] is pd.NA
    assert out.v1_trait_free.iloc[1] is pd.NA
    assert out.v2_dose_matched.iloc[1] is pd.NA


# ---- coverage -------------------------------------------------------------------------------


def test_coverage_counts_and_incomplete_cells():
    leaves = pd.DataFrame(
        _leaf_rows(
            "c", False, dict(CROSS, placebo=[(0, 0, 0.1), (1, 1, None), (2, 2, None)])
        )
    )
    doc = tm.coverage(leaves)
    assert doc["leaves"]["conditions"]["placebo"] == {
        "expected": 3,
        "present": 1,
        "trained": 1,
        "evaluated": 1,
    }
    assert doc["n_incomplete_cells"] == 1
    assert list(doc["incomplete_cells"][0]["arms"]) == ["placebo"]


# ---- control scope ------------------------------------------------------------------------


def test_source_family_from_extra_or_run_sets():
    assert tm.source_family(_cell(DS1, PYTHIA)) == "ws160m"
    assert tm.source_family(_cell(DS1, PYTHIA, source_extra={})) == "ws160m"
    with pytest.raises(ValueError):
        tm.source_family(_cell(DS1, PYTHIA, source_extra={"source_family": "nope"}))


def test_expected_leaves_flag_out_of_scope_controls():
    cell = _cell(DS1, PYTHIA, sb_run_sets=(WS_RS, "20260828_ckptgrid-bear_160m", WS_RS))
    roster = Roster(cells=(cell,), teachers=_roster().teachers)
    sb = [e for e in tm.expected_leaves(roster, "/out") if e.arm == "sb"]
    assert [e.primary_scope for e in sb] == [True, False, True]
    tb = [e for e in tm.expected_leaves(roster, "/out") if e.arm == "tb"]
    assert tb and all(e.primary_scope and e.new_leaf for e in tb)


def test_cell_contrasts_primary_vs_pooled_controls():
    spec = dict(CROSS, sb=[(-1, 0, 0.1), (-1, 1, 0.2), (-1, 2, 0.4, False, False)])
    out = tm.cell_contrasts(
        pd.DataFrame(_leaf_rows("c", False, spec)),
        ["p_target_test"],
        n_boot=500,
        seed=1,
    ).iloc[0]
    assert (out.n_sb, out.n_sb_pooled, out.expected_sb) == (2, 3, 2)
    assert out.mean_sb == pytest.approx(0.15)
    assert out.mean_sb_pooled == pytest.approx(0.7 / 3)
    assert out.delta_sb == pytest.approx(0.5)
    assert out.delta_sb_pooled == pytest.approx(0.65 - 0.7 / 3)
    assert out.delta_tb_pooled == pytest.approx(out.delta_tb)
    assert not out.incomplete


def test_teacher_labels_v1_without_trait_test_set():
    rows = [
        _teacher("rolloutscreen", "trait", 0, math.nan, math.nan, 1.45, 2.72, TTAG),
        _teacher("rolloutscreen", "trait", 1, math.nan, math.nan, 1.50, 2.72, TTAG),
        _teacher("rolloutscreen", "placebo", 0, 0.004, 0.002, 1.48, 2.75, "p"),
    ]
    out = tm.teacher_labels(pd.DataFrame(rows))
    assert out.admitted.iloc[0] is pd.NA
    assert out.admitted.iloc[2] is False
    assert out.v1_trait_free.iloc[2] is True
    assert out.v2_dose_matched.iloc[2]


def test_paired_delta_matches_gseed_matched_hand_value():
    # leaves at gseeds 0, 0, 1; controls 0.1 @ g0, 0.3 @ g1 (unbalanced)
    w = np.ones((1, 3), dtype=np.int64)
    cw = np.ones((1, 2), dtype=np.int64)
    point, draws = tm.paired_delta(
        [0.5, 0.7, 0.9], [0, 0, 1], w, [0.1, 0.3], [0, 1], cw
    )
    assert point == pytest.approx(((0.4 + 0.6) + 0.6) / 3)
    assert draws[0] == pytest.approx(point)
    # a leaf with no control at its gseed is left out
    point, _ = tm.paired_delta(
        [0.5, 0.9], [0, 2], np.ones((1, 2)), [0.1], [0], np.ones((1, 1))
    )
    assert point == pytest.approx(0.4)


def test_cell_contrasts_gseed_paired_cell_uses_matched_controls():
    spec = {
        "treatment": [(0, 0, 0.5), (1, 0, 0.7), (2, 1, 0.9)],
        "sb": [(-1, 0, 0.1), (-1, 1, 0.3)],
        "placebo": [(0, 0, 0.2), (1, 1, 0.2), (2, 2, 0.2)],
    }
    rows = pd.DataFrame(_leaf_rows("p", True, spec)).assign(gseed_paired=True)
    out = tm.cell_contrasts(rows, ["p_target_test"], n_boot=300, seed=4).iloc[0]
    assert out.mean_sb == pytest.approx(0.2)
    assert out.delta_sb == pytest.approx(1.6 / 3)
    assert out.delta_sb != pytest.approx(out.mean_treatment - out.mean_sb)
    assert out.delta_sb_lo <= out.delta_sb <= out.delta_sb_hi
    # placebo leaf at gseed 2 has no control there
    assert out.placebo_minus_sb == pytest.approx((0.1 + -0.1) / 2)
