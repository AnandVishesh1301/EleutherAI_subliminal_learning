from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import strictscreen_plan
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import traitfree_roster as tr
from subliminal_learning.campaign import ws160m_pairs
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_plan

FROZEN = ws160m_plan.FROZEN_GKD_TAG
TTAG = ws160m_plan.TEACHER_PARAM_TAG_LORA
FINAL = ws160m_plan.FINAL_STEP


def _base(name: str, step: int = FINAL) -> tr.BaseRef:
    return tr.BaseRef(
        token=f"{name}@step{step}",
        model_dir=f"{name}-step{step}",
        repo=f"EleutherAI/{name}@step{step}",
    )


def _source(steps: int = 750, run_set: str = "20260916_ws160m-main_160m"):
    return tr.SourceTeacher(
        param_tag=TTAG,
        run_set=run_set,
        family_recipe=tr.family_recipe_of(run_set),
        n_train_steps={k: steps for k in plan.PLACEBO_FTSEEDS},
        manifest_paths={k: f"/t/ftseed{k}/train_manifest.json" for k in (0, 1, 2)},
    )


def _cell(
    family: str = plan.FAMILY_CONDITIONS,
    teacher: str = "data-seed1",
    student: str = "data-seed1",
    source: tr.SourceTeacher | None = None,
) -> tr.RosterCell:
    t, s = _base(teacher), _base(student)
    return tr.RosterCell(
        cell_id=tr.make_cell_id(family, "bear", t.model_dir, s.model_dir, FROZEN),
        family=family,
        tier=tr.cell_tier(family, ("self_diag",)),
        panels=("self_diag",),
        animal="bear",
        teacher=t,
        student=s,
        self_pair=t == s,
        gkd_tag=FROZEN,
        source_leaves=(tr.LeafRef("/leaf/gseed0", "rs", 0, 0),),
        source_identity={"target_animal": "bear", "gkd_tag": FROZEN},
        source_extra={"n_train_steps": 625, "job_flags": ["--no-control"]},
        sb_controls=(tr.LeafRef("/ctrl/gseed0", "rs", 0),),
        tb_controls=(),
        source_teacher=source or _source(),
        lmeval=True,
    )


def _roster(*cells: tr.RosterCell) -> tr.Roster:
    return tr.Roster(
        cells=tuple(cells),
        teachers=tr.build_placebo_teachers(cells),
        provenance={"git_sha": "abc"},
    )


# ---- serialisation / validation ---------------------------------------------------------


def test_roster_json_round_trip_is_lossless_and_sha_stable():
    roster = _roster(_cell(), _cell(student="seed1"))
    doc = tr.roster_to_json(roster)
    back = tr.roster_from_json(json.loads(json.dumps(doc)))
    assert back == roster
    assert tr.roster_sha256(tr.roster_to_json(back)) == tr.roster_sha256(doc)


def test_roster_from_json_wrong_version_raises():
    doc = tr.roster_to_json(_roster(_cell()))
    with pytest.raises(ValueError, match="version"):
        tr.roster_from_json({**doc, "version": tr.ROSTER_VERSION + 1})


def test_placebo_param_tag_swaps_epochs_for_steps_and_adds_corpus():
    tag = tr.placebo_param_tag(750, "ep5_lr5e-5_r128a128")
    assert tag == f"s750_lr5e-5_r128a128_c-{plan.CORPUS_NAME}"


@pytest.mark.parametrize("bad", ["s750_lr5e-5_r128a128", "ep5", "ep5_lr5e-5_c-x"])
def test_placebo_param_tag_non_lora_epoch_tag_raises(bad):
    with pytest.raises(ValueError):
        tr.placebo_param_tag(750, bad)


def test_validate_roster_accepts_a_consistent_roster():
    tr.validate_roster(_roster(_cell(), _cell(student="seed1")))


def test_validate_roster_duplicate_cell_ids_raise():
    cell = _cell()
    with pytest.raises(ValueError, match="duplicate cell"):
        tr.validate_roster(replace(_roster(cell), cells=(cell, cell)))


@pytest.mark.parametrize(
    "change",
    [{"tier": "z"}, {"family": "nope"}, {"source_leaves": ()}],
)
def test_validate_roster_bad_cell_raises(change):
    roster = _roster(_cell())
    bad = replace(roster.cells[0], **change)
    with pytest.raises(ValueError):
        tr.validate_roster(replace(roster, cells=(bad,)))


def test_validate_roster_cell_without_teacher_raises():
    roster = _roster(_cell())
    with pytest.raises(KeyError):
        tr.validate_roster(replace(roster, teachers=()))


@pytest.mark.parametrize(
    "change", [{"max_steps": 749}, {"ftseeds": (0, 1)}], ids=["steps", "ftseeds"]
)
def test_validate_roster_bad_teacher_raises(change):
    roster = _roster(_cell())
    bad = replace(roster.teachers[0], **change)
    with pytest.raises(ValueError):
        tr.validate_roster(replace(roster, teachers=(bad,)))


# ---- placebo teachers / tiers -------------------------------------------------------------


def test_build_placebo_teachers_one_per_slot_with_extended_null():
    null = _cell(teacher="pythia-160m", student="pythia-160m")
    cells = (null, _cell(teacher="pythia-160m", student="seed1"), _cell())
    teachers = tr.build_placebo_teachers(cells)
    assert [t.key for t in teachers] == sorted(
        {f"conditions|bear|{c.teacher.model_dir}" for c in cells}
    )
    by_dir = {t.base.model_dir: t for t in teachers}
    assert by_dir[null.teacher.model_dir].ftseeds == plan.EXTENDED_NULL_FTSEEDS
    assert by_dir["data-seed1-step143000"].ftseeds == plan.PLACEBO_FTSEEDS
    assert all(t.max_steps == 750 for t in teachers)


def test_build_placebo_teachers_disagreeing_sources_raise():
    cells = (_cell(), _cell(student="seed1", source=_source(steps=700)))
    with pytest.raises(ValueError, match="disagree"):
        tr.build_placebo_teachers(cells)


@pytest.mark.parametrize(
    ("family", "panels", "tier"),
    [
        (plan.FAMILY_CONDITIONS, (), plan.TIER_A),
        (plan.FAMILY_DOSE, (), plan.TIER_B),
        (plan.FAMILY_CHECKPOINTS, (tr.PANEL_SAME_CKPT,), plan.TIER_C),
        (
            plan.FAMILY_CHECKPOINTS,
            (tr.PANEL_SAME_CKPT, tr.PANEL_SELF_DIAG),
            plan.TIER_B,
        ),
        (plan.FAMILY_CHECKPOINTS, (tr.PANEL_ONSET_RECV,), plan.TIER_B),
    ],
)
def test_cell_tier(family, panels, tier):
    assert tr.cell_tier(family, panels) == tier


def test_checkpoint_panel_names_cover_the_tier_b_panels():
    assert plan.CHECKPOINT_TIER_B_PANELS <= set(tr.CHECKPOINT_PANELS)


# ---- small pure helpers ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "micro", "accum", "epochs", "drop_last", "steps"),
    [
        (20000, 32, 1, 1.0, False, 625),
        (20000, 16, 2, 1.0, False, 625),
        (20000, 32, 1, 0.25, False, 157),
        (100, 32, 1, 1.0, False, 4),
        (100, 32, 3, 1.0, False, 2),
        (100, 32, 1, 1.0, True, 3),
        (10, 32, 1, 1.0, False, 1),
    ],
)
def test_expected_train_steps_epoch_budget(n, micro, accum, epochs, drop_last, steps):
    got = tr.expected_train_steps(
        n_prompts=n,
        micro_batch=micro,
        grad_accum=accum,
        num_epochs=epochs,
        max_steps=-1,
        drop_last=drop_last,
    )
    assert got == steps


def test_expected_train_steps_positive_max_steps_wins():
    got = tr.expected_train_steps(
        n_prompts=1250,
        micro_batch=16,
        grad_accum=2,
        num_epochs=1.0,
        max_steps=625,
        drop_last=True,
    )
    assert got == 625


@pytest.mark.parametrize(
    ("run_set", "recipe"),
    [
        ("20260916_ws160m-main_160m", "ws160m"),
        ("20260918_ws160m2-grid_160m", "ws160m"),
        ("20260828_ckptgrid-bear_160m", "ckptgrid"),
        ("20260908_ckptgrid2-bear_160m", "ckptgrid2"),
        ("20260809_breadth4-teachers_160m", "breadth4"),
        ("20260726_crossinit_160m", "crossinit"),
        ("20260907_xsize-bear_mixed", "xsize"),
    ],
)
def test_family_recipe_of(run_set, recipe):
    assert tr.family_recipe_of(run_set) == recipe


def test_family_recipe_of_unknown_campaign_raises():
    with pytest.raises(ValueError):
        tr.family_recipe_of("20260914_opdlim1-bear_mixed")


def test_hp_signature_ignores_rollout_cache_only():
    gkd = {"lmbda": 1.0, "beta": 1.0}
    th = {"learning_rate": 4e-4, "lora_r": 8, "lora_alpha": 16, "batch_size": 32}
    base = tr.hp_signature(gkd, th)
    assert tr.hp_signature({**gkd, "rollout_cache": True}, th) == base
    assert tr.hp_signature(gkd, {**th, "batch_size": 16}) != base


def test_identity_diff_ignores_teacher_lineage():
    a = {"teacher_tag": "x-tfs0", "teacher_manifest_sha256": "1", "gkd_tag": "g"}
    b = {"teacher_tag": "x-tfs1", "teacher_manifest_sha256": "2", "gkd_tag": "g"}
    assert tr.identity_diff(a, b) == ()
    assert tr.identity_diff(a, {**b, "gkd_tag": "h"}) == ("gkd_tag",)


def test_external_token_and_ws160m_base_of():
    assert (
        tr.external_token(tr.SOURCE_CKPTGRID, "pythia-160m", 2000)
        == "original@step2000"
    )
    assert tr.external_token(tr.SOURCE_CKPTGRID2, "seed2", 8000) == "seed2@step8000"
    assert tr.ws160m_base_of("EleutherAI/pythia-160m-data-seed1@step2000") == (
        "data-seed1",
        2000,
    )
    assert tr.ws160m_base_of("EleutherAI/pythia-160m-weight-seed1@step2000") is None
    with pytest.raises(ValueError):
        tr.external_token(tr.SOURCE_WS160M, "pythia-160m", 2000)


def test_source_job_flags_mirror_the_family_builders():
    x = tr.source_job(
        tr.SOURCE_XFAM2, student_token="pythia-1b", run_set_id="rs", lora_lr=4e-4
    )
    assert x.flags[0] == "--no-control" and "--lora-lrs 0.0004" in x.flags
    assert x.training_config == xfam2_plan.STUDENT_TRAINING_CONFIG
    ws = tr.source_job(
        tr.SOURCE_WS160M, student_token="t", run_set_id="rs", lora_lr=4e-4
    )
    assert ws.training_config is None and ws.spec_registry == ws160m_plan.SPEC_REGISTRY
    with pytest.raises(ValueError):
        tr.source_job(tr.SOURCE_OPDLIM, student_token="t", run_set_id="x", lora_lr=1)


def test_source_footprint_pins_only_overriding_families():
    assert tr.source_footprint(tr.SOURCE_WS160M, "pythia-160m@step143000") is None
    assert tr.source_footprint(tr.SOURCE_OPDLIM, opdlim_plan.B410) == (
        16,
        2,
        False,
        False,
    )
    assert tr.source_footprint(tr.SOURCE_XFAM2, "pythia-1b") == (16, 2, True, True)


# ---- per-figure selection -------------------------------------------------------------------


def _ws_leaf(**kw) -> dict:
    row = dict(
        source_run_set="20260916_ws160m-main_160m",
        set=ws160m_plan.PRIMARY_PROMPT_SET,
        scoring=ws160m_plan.PRIMARY_SCORING,
        is_control=False,
        quarantined=False,
        animal="bear",
        teacher_base="data-seed1",
        student_base="data-seed1",
        teacher_step=FINAL,
        student_step=FINAL,
        gkd_tag=FROZEN,
        teacher_param_tag=TTAG,
        category="SI_SS",
        ftseed=0.0,
        gseed=0,
        replicate=0.0,
    )
    row.update(kw)
    return row


def test_select_conditions_keeps_only_the_paper_filter():
    leaves = pd.DataFrame(
        [
            _ws_leaf(),
            _ws_leaf(ftseed=1.0, gseed=1, quarantined=True),
            _ws_leaf(student_base="seed1", category="DI_DS"),
            _ws_leaf(is_control=True),
            _ws_leaf(teacher_base="pile-main", category="SI_SS"),
            _ws_leaf(student_step=8000),
            _ws_leaf(gkd_tag="other"),
            _ws_leaf(set="storytelling"),
            _ws_leaf(animal="cat"),
        ]
    )
    sel = tr.select_conditions(leaves)
    assert list(sel.columns) == list(tr.SELECTION_COLUMNS)
    assert sorted(set(sel["cell"])) == [
        "bear|data-seed1|data-seed1",
        "bear|data-seed1|seed1",
    ]
    assert set(sel["teacher_token"]) == {f"data-seed1@step{FINAL}"}
    assert sel["quarantined"].tolist() == [False, True, False]
    assert set(sel["panels"]) == {tr.PANEL_SEP.join(tr.CONDITIONS_PANELS)}


def test_select_conditions_stray_category_raises():
    with pytest.raises(ValueError, match="categories"):
        tr.select_conditions(pd.DataFrame([_ws_leaf(category="SI_DC")]))


def test_select_dose_needs_two_doses_and_plots_quarantined_leaves():
    tag = {
        d: ws160m_plan.student_gkd_tag(ws160m_plan.FROZEN_RECIPE, d)
        for d in (0.25, 1.0)
    }
    base = "pythia-160m"
    leaves = pd.DataFrame(
        [
            _ws_leaf(teacher_base=base, student_base=base, gkd_tag=tag[0.25]),
            _ws_leaf(
                teacher_base=base, student_base=base, gkd_tag=tag[1.0], quarantined=True
            ),
            _ws_leaf(
                animal="cat", teacher_base=base, student_base=base, gkd_tag=tag[1.0]
            ),
        ]
    )
    common = dict(source_run_set="20260916_ws160m-main_160m", student_step=FINAL)
    ids = dict(ftseed=0.0, gseed=0, replicate=0.0, student_base=base)
    dose = pd.DataFrame(
        [
            dict(animal="bear", dose=0.25, gkd_tag=tag[0.25], **common, **ids),
            dict(animal="bear", dose=1.0, gkd_tag=tag[1.0], **common, **ids),
            dict(animal="cat", dose=1.0, gkd_tag=tag[1.0], **common, **ids),
            dict(animal="bear", dose=0.5, gkd_tag="not-fig1", **common, **ids),
        ]
    )
    sel = tr.select_dose(leaves, dose)
    assert sorted(sel["gkd_tag"]) == sorted([tag[0.25], tag[1.0]])
    assert not sel["quarantined"].any()
    with pytest.raises(ValueError, match="without a treatment leaf"):
        tr.select_dose(leaves.iloc[:1], dose)


def _x_leaf(**kw) -> dict:
    row = dict(
        kind="treatment",
        animal="bear",
        teacher="pythia-160m",
        student="pythia-410m",
        lr=4e-4,
        quarantined=False,
        teacher_admitted=np.nan,
        delta_primary=0.1,
        teacher_param_tag=TTAG,
        gkd_tag=FROZEN,
        ftseed=0.0,
        gseed=0,
    )
    row.update(kw)
    return row


def test_select_crossmodel_takes_live_admitted_leaves_of_each_cell():
    cell = xfam2_plan.Cell("M", "pythia-160m", "pythia-410m", 4e-4)
    live = [_x_leaf(ftseed=float(f), gseed=g) for f in range(3) for g in range(3)]
    dead = [
        _x_leaf(quarantined=True),
        _x_leaf(teacher_admitted=False),
        _x_leaf(delta_primary=np.nan),
    ]
    sel = tr.select_crossmodel(pd.DataFrame(live + dead), cells=(cell,))
    assert len(sel) == 9 and set(sel["student_token"]) == {"pythia-410m"}
    with pytest.raises(ValueError, match="live leaves"):
        tr.select_crossmodel(pd.DataFrame(live[:8]), cells=(cell,))


def test_select_topk_labels_each_plotted_cell():
    vocab = opdlim_plan.VOCAB_SIZE
    cols = dict(teacher_param_tag=TTAG, lmbda=1.0, form="ce", subset_n=np.nan)
    cells = pd.DataFrame(
        [
            dict(
                tier=1,
                block="ce-k10",
                pair="P3",
                objective="k10",
                k=10,
                vmask_forms=False,
                **cols,
            ),
            dict(
                tier=1,
                block="ce-k10",
                pair="P1",
                objective="k10",
                k=10,
                vmask_forms=False,
                **cols,
            ),
            dict(
                tier=2,
                block="vmaskf-k10",
                pair="P1",
                objective="vmaskf-k10",
                k=10,
                vmask_forms=True,
                **cols,
            ),
            dict(
                tier=2,
                block="lmb0-k10",
                pair="P4",
                objective="lmb0-k10",
                k=10,
                vmask_forms=False,
                **{**cols, "lmbda": 0.0},
            ),
            dict(
                tier=1,
                block="hp-p4",
                pair="P4",
                objective="full",
                k=vocab,
                vmask_forms=False,
                **cols,
            ),
        ]
    )
    data = pd.DataFrame(
        [
            dict(
                pair="P1",
                k=vocab,
                block="lmb0-data-p1-full-n64",
                objective="lmb0-full-n64",
            )
        ]
    )
    keys = [
        (1, "ce-k10", "P3", "k10"),
        (1, "ce-k10", "P1", "k10"),
        (2, "vmaskf-k10", "P1", "vmaskf-k10"),
        (2, "lmb0-k10", "P4", "lmb0-k10"),
        (3, "lmb0-data-p1-full-n64", "P1", "lmb0-full-n64"),
    ]
    leaves = pd.DataFrame(
        [
            dict(
                tier=t,
                block=b,
                pair=p,
                objective=o,
                kind="treatment",
                teacher_param_tag=TTAG,
                run_set_id=f"rs{t}",
                gkd_tag=f"g-{b}",
                teacher_token=opdlim_plan.PAIR_BY_NAME[p].teacher,
                student_token=opdlim_plan.PAIR_BY_NAME[p].student,
                ftseed=0.0,
                gseed=0,
                quarantined=False,
            )
            for t, b, p, o in keys
        ]
    )
    sel = tr.select_topk(cells, data, leaves)
    by_cell = {
        c: set(p.split(tr.PANEL_SEP))
        for c, p in sel.groupby("cell")["panels"].first().items()
    }
    assert by_cell == {
        f"1|ce-k10|P3|k10|{TTAG}": {"fig:topk:a_standard"},
        f"1|ce-k10|P1|k10|{TTAG}": {
            "fig:topk-cross:a_standard",
            "fig:topk-real:on_policy",
        },
        f"2|vmaskf-k10|P1|vmaskf-k10|{TTAG}": {"fig:topk-cross:b_masked"},
        f"2|lmb0-k10|P4|lmb0-k10|{TTAG}": {"fig:topk-real:off_policy_real_text"},
        f"3|lmb0-data-p1-full-n64|P1|lmb0-full-n64|{TTAG}": {
            "fig:topk-data:data_curve"
        },
    }
    bad = leaves.assign(teacher_token="pythia-410m@step143000")
    with pytest.raises(ValueError, match="tokens"):
        tr.select_topk(cells, data, bad)


def test_select_checkpoints_labels_panels_and_routes_sources():
    young = ws160m_plan.G6_RUNGS[0]
    mature = ws160m_plan.MATURE_RUNGS[0]
    k2_t, k2_s = ws160m_pairs.k2_directed_pairs()[1]
    rc_common = dict(stat_set="primary", n_leaves=3, delta=0.1)
    rungcross = pd.DataFrame(
        [
            dict(
                panel="g6",
                teacher_base="pythia-160m",
                student_base="seed1",
                teacher_step=young,
                student_step=mature,
                source_run_set="ws160m",
                **rc_common,
            ),
            dict(
                panel="g6",
                teacher_base="pythia-160m",
                student_base="pythia-160m",
                teacher_step=young,
                student_step=young,
                source_run_set=ws160m_plan.CKPTGRID_RUN_SET_ID,
                **rc_common,
            ),
            dict(
                panel="g6",
                teacher_base="seed1",
                student_base="seed1",
                teacher_step=mature,
                student_step=young,
                source_run_set="ws160m",
                **{**rc_common, "n_leaves": 0},
            ),
            dict(
                panel="g6",
                teacher_base="seed1",
                student_base="pythia-160m",
                teacher_step=young,
                student_step=mature,
                source_run_set="ws160m",
                **{**rc_common, "stat_set": "full"},
            ),
        ]
    )
    cell_delta = pd.DataFrame(
        [
            dict(
                animal="bear",
                stat_set="primary",
                set=ws160m_plan.PRIMARY_PROMPT_SET,
                scoring=ws160m_plan.PRIMARY_SCORING,
                gkd_tag=FROZEN,
                teacher_param_tag=TTAG,
                teacher_base=k2_t,
                student_base=k2_s,
                teacher_step=FINAL,
                student_step=FINAL,
            )
        ]
    )
    leaves = pd.DataFrame(
        [
            _ws_leaf(
                teacher_base="pythia-160m",
                student_base="seed1",
                teacher_step=young,
                student_step=mature,
            ),
            _ws_leaf(teacher_base=k2_t, student_base=k2_s),
            _ws_leaf(
                teacher_base="seed2",
                student_base="seed2",
                teacher_step=4000,
                student_step=4000,
                quarantined=True,
            ),
        ]
    )
    external = pd.DataFrame(
        [
            dict(
                source_run_set=ws160m_plan.CKPTGRID_RUN_SET_ID,
                animal="bear",
                teacher_base="pythia-160m",
                student_base="pythia-160m",
                teacher_step=young,
                student_step=young,
                teacher_param_tag=TTAG,
                gkd_tag=FROZEN,
                ftseed=g,
                gseed=g,
            )
            for g in range(3)
        ]
    )
    sel = tr.select_checkpoints(leaves, rungcross, cell_delta, external)
    panels = sel.groupby("cell")["panels"].first().to_dict()
    assert panels == {
        f"pythia-160m|seed1|{young}|{mature}": tr.PANEL_SEP.join(
            sorted({tr.PANEL_ONSET_EMIT, tr.PANEL_ROWCOL_TEACHER})
        ),
        f"pythia-160m|pythia-160m|{young}|{young}": tr.PANEL_SEP.join(
            sorted(
                {tr.PANEL_ROWCOL_TEACHER, tr.PANEL_ROWCOL_STUDENT, tr.PANEL_SELF_DIAG}
            )
        ),
        f"{k2_t}|{k2_s}|{FINAL}|{FINAL}": tr.PANEL_SEP.join(
            sorted({tr.PANEL_SAME_CKPT, tr.PANEL_SELF_DIAG})
        ),
        "seed2|seed2|4000|4000": tr.PANEL_SELF_DIAG,
    }
    ext = sel[sel["source"] == tr.SOURCE_CKPTGRID]
    assert len(ext) == 3 and set(ext["teacher_token"]) == {f"original@step{young}"}
    assert sel.loc[sel["cell"] == "seed2|seed2|4000|4000", "quarantined"].all()


def test_select_rolloutscreen_takes_arm_a_under_the_strict_tag():
    pairing = strictscreen_plan.PAIRINGS[2]
    row = dict(
        animal="bear",
        pair=pairing.name,
        student=pairing.student,
        teacher=pairing.teacher,
        screen="strict",
        tfs=0,
        gseed=0,
        leaf_dir="/x/gseed0/",
    )
    leaves = pd.DataFrame(
        [
            {**row, "source": strictscreen_plan.ARM_A},
            {**row, "source": strictscreen_plan.ARM_B},
        ]
    )
    sel = tr.select_rolloutscreen(leaves)
    assert len(sel) == 1
    assert sel.iloc[0]["gkd_tag"] == strictscreen_plan.STRICT_TAG
    assert sel.iloc[0]["expected_leaf"] == "/x/gseed0"
    wrong = leaves.assign(teacher="weight-seed3")
    with pytest.raises(ValueError, match="pairing"):
        tr.select_rolloutscreen(wrong)
