"""Part II (`traitfree2`): grid and teacher-epoch cells, teacher reuse, queues."""

from __future__ import annotations

import json
import shlex
from dataclasses import replace

import pandas as pd
import pytest

from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tj
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import traitfree_roster as tr
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign.traitfree_checks import rescore_problems
from test_traitfree_config_gen import CELLS, synthetic_roster
from test_traitfree_roster import _ws_leaf

FROZEN = ws160m_plan.FROZEN_GKD_TAG
TTAG = ws160m_plan.TEACHER_PARAM_TAG_LORA
FINAL = ws160m_plan.FINAL_STEP
NO_EXTERNAL = pd.DataFrame(
    columns=[
        "source_run_set", "animal", "teacher_base", "student_base", "teacher_step",
        "student_step", "teacher_param_tag", "gkd_tag", "ftseed", "gseed",
    ]
)


# ---- plan --------------------------------------------------------------------------------


def test_part2_queues_are_new_names_pinned_to_part2_hosts():
    assert not set(plan.QUEUE_PRIORITIES2) & set(plan.QUEUE_PRIORITIES)
    for name in plan.QUEUE_PRIORITIES2:
        assert name.startswith(f"{plan.CAMPAIGN2}-")
        assert plan.queue_host(name) in plan.HOSTS2
    assert set(plan.pool_queues(plan.GONCALO, plan.QUEUE_PRIORITIES2)) == set(
        plan.QUEUE_PRIORITIES2
    )


def test_part1_run_sets_and_queues_unchanged_by_part2():
    assert {plan.RUN_SET_ID[f] for f in plan.FAMILIES} == {
        f"20260929_traitfree-{f}_mixed" for f in plan.FAMILIES
    }
    assert all(n.startswith(f"{plan.CAMPAIGN}-") for n in plan.QUEUE_PRIORITIES)
    assert plan.student_queue("checkpoints-b", plan.SOAR) == (
        "traitfree-students-checkpoints-b-soar"
    )


def test_part2_epoch_families_are_the_ws160m_dose_epochs():
    assert sorted(plan.PART2_EPOCHS.values()) == sorted(
        ws160m_plan.TEACHER_DOSE_EPOCHS
    )


@pytest.mark.parametrize(
    ("name", "group"),
    [
        ("traitfree2-students-ckptgrids-a-goncalo", "ckptgrids-a"),
        ("traitfree2-students-ckptep8-goncalo", "ckptep8"),
        ("traitfree-students-checkpoints-c-soar", "checkpoints-c"),
        ("traitfree2-rescore-goncalo", None),
        ("traitfree2-teachers-ckptep2-goncalo", None),
    ],
)
def test_queue_group_parses_both_parts(name, group):
    assert plan.queue_group(name) == group


# ---- selection ---------------------------------------------------------------------------


def _rc(panel, t, s, ts, ss, source="ws160m", **kw):
    row = dict(
        panel=panel,
        teacher_base=t,
        student_base=s,
        teacher_step=ts,
        student_step=ss,
        stat_set="primary",
        n_leaves=3,
        delta=0.1,
        source_run_set=source,
    )
    row.update(kw)
    return row


def test_select_ckptgrids_labels_every_grid_cell_and_tiers_it():
    rungcross = pd.DataFrame(
        [
            _rc("g9", "dclm-main", "pythia-160m", 1000, 32000),
            _rc("g6", "seed1", "seed2", 4000, FINAL),
            _rc("g6", "seed1", "seed2", 4000, 8000),
            _rc("g9", "seed1", "seed2", 4000, FINAL),
            _rc("nonident", "pile-main", "pile-main", 2000, 2000),
            _rc("g6", "seed2", "seed2", 2000, 2000, n_leaves=0),
            _rc("g6", "seed2", "seed1", 2000, 2000, stat_set="full"),
        ]
    )
    keys = [
        ("dclm-main", "pythia-160m", 1000, 32000),
        ("seed1", "seed2", 4000, FINAL),
        ("seed1", "seed2", 4000, 8000),
        ("pile-main", "pile-main", 2000, 2000),
    ]
    leaves = pd.DataFrame(
        [
            _ws_leaf(
                teacher_base=t, student_base=s, teacher_step=ts, student_step=ss, gseed=g
            )
            for t, s, ts, ss in keys
            for g in range(3)
        ]
    )
    sel = tr.select_ckptgrids(leaves, rungcross, NO_EXTERNAL)
    panels = sel.groupby("cell")["panels"].first().to_dict()
    assert panels == {
        "dclm-main|pythia-160m|1000|32000": plan.PANEL_G9,
        f"seed1|seed2|4000|{FINAL}": tr.PANEL_SEP.join(
            sorted({plan.PANEL_G9, plan.PANEL_RATIO143K})
        ),
        "seed1|seed2|4000|8000": plan.PANEL_GRID,
        "pile-main|pile-main|2000|2000": plan.PANEL_GRID,
    }
    assert sel.groupby("cell").size().eq(3).all()
    tiers = {
        c: tr.cell_tier(plan.FAMILY_CKPTGRIDS, tuple(p.split(tr.PANEL_SEP)))
        for c, p in panels.items()
    }
    assert tiers == {
        "dclm-main|pythia-160m|1000|32000": plan.TIER_A,
        f"seed1|seed2|4000|{FINAL}": plan.TIER_A,
        "seed1|seed2|4000|8000": plan.TIER_B,
        "pile-main|pile-main|2000|2000": plan.TIER_B,
    }


def test_select_ckptgrids_cell_with_two_sources_raises():
    rungcross = pd.DataFrame(
        [
            _rc("g9", "pythia-160m", "pythia-160m", 2000, 4000),
            _rc(
                "g6",
                "pythia-160m",
                "pythia-160m",
                2000,
                4000,
                source=ws160m_plan.CKPTGRID_RUN_SET_ID,
            ),
        ]
    )
    with pytest.raises(ValueError, match="has sources"):
        tr.select_ckptgrids(pd.DataFrame([_ws_leaf()]), rungcross, NO_EXTERNAL)


def _cd(t, s, tag, **kw):
    row = dict(
        stat_set="primary",
        animal="bear",
        gkd_tag=FROZEN,
        set=ws160m_plan.PRIMARY_PROMPT_SET,
        scoring=ws160m_plan.PRIMARY_SCORING,
        teacher_step=FINAL,
        student_step=FINAL,
        teacher_param_tag=tag,
        teacher_base=t,
        student_base=s,
    )
    row.update(kw)
    return row


def test_select_ckptepochs_keeps_pairs_with_every_epoch_only():
    tags = {e: ws160m_plan.teacher_param_tag(e) for e in (2, 5, 8)}
    cell_delta = pd.DataFrame(
        [_cd("pile-ws1", "seed1", tags[e]) for e in (2, 5, 8)]
        + [_cd("seed1", "pile-ws1", tags[e]) for e in (2, 5)]
        + [_cd("pile-ws1", "seed2", tags[e], teacher_step=8000) for e in (2, 5, 8)]
    )
    leaves = pd.DataFrame(
        [
            _ws_leaf(
                teacher_base=t, student_base=s, teacher_param_tag=tags[2], gseed=g
            )
            for t, s in (("pile-ws1", "seed1"), ("seed1", "pile-ws1"))
            for g in range(3)
        ]
        + [_ws_leaf(teacher_base="pile-ws1", student_base="seed1", teacher_param_tag=TTAG)]
    )
    sel = tr.select_ckptepochs(leaves, cell_delta, 2)
    assert set(sel["teacher_token"]) == {f"pile-ws1@step{FINAL}"}
    assert set(sel["student_token"]) == {f"seed1@step{FINAL}"}
    assert set(sel["teacher_param_tag"]) == {tags[2]}
    assert len(sel) == 3 and set(sel["panels"]) == {plan.PANEL_TEACHER_EPOCHS}
    with pytest.raises(ValueError):
        tr.select_ckptepochs(leaves, cell_delta, 3)


# ---- teacher reuse and roster -----------------------------------------------------------


def _part2_cell(cell: tr.RosterCell, family: str = plan.FAMILY_CKPTGRIDS):
    return replace(
        cell,
        family=family,
        cell_id=cell.cell_id.replace(cell.family, family, 1),
        tier=plan.TIER_A,
        panels=(plan.PANEL_G9,),
    )


def test_reusable_teacher_needs_an_exact_identity_and_prefers_checkpoints():
    part1 = synthetic_roster()
    cell = _part2_cell(CELLS["ckpt_b"])
    t = tr.reusable_teacher(cell, part1.teachers)
    assert t is not None and t.family == plan.FAMILY_CHECKPOINTS
    dup = replace(t, family=plan.FAMILY_CONDITIONS)
    assert tr.reusable_teacher(cell, (dup, t)).family == plan.FAMILY_CHECKPOINTS
    for change in (
        dict(max_steps=t.max_steps + 1),
        dict(base=replace(t.base, model_dir="other")),
        dict(slot_animal="lion"),
        dict(source=replace(t.source, family_recipe="ckptgrid")),
        dict(ftseeds=(0, 1)),
    ):
        assert tr.reusable_teacher(cell, (replace(t, **change),)) is None


def _part2_roster():
    part1 = synthetic_roster()
    cells = []
    reused = {}
    for name in ("ckpt_b", "cross"):
        cell = _part2_cell(CELLS[name])
        t = next(x for x in part1.teachers if x.key.split("|", 1)[1] == (
            f"{cell.animal}|{cell.teacher.model_dir}"
        ))
        reused[t.key] = t
        cells.append(replace(cell, placebo_teacher_family=t.family))
    return tr.Roster(
        cells=tuple(sorted(cells, key=lambda c: c.cell_id)),
        teachers=(),
        provenance={"part": 2},
        reused_teachers=tuple(reused[k] for k in sorted(reused)),
    )


def test_part2_roster_round_trip_and_teacher_lookup():
    roster = _part2_roster()
    tr.validate_roster(roster)
    doc = tr.roster_to_json(roster)
    back = tr.roster_from_json(json.loads(json.dumps(doc)))
    assert tr.roster_to_json(back) == doc
    for cell in back.cells:
        assert back.teacher_for(cell).family == cell.placebo_teacher_family
        assert cell.teacher_family == cell.placebo_teacher_family


def test_part1_roster_json_keeps_its_format():
    doc = tr.roster_to_json(synthetic_roster())
    assert "reused_teachers" not in doc
    assert all("placebo_teacher_family" not in c for c in doc["cells"])


def test_validate_roster_rejects_mixed_parts_and_part2_reused_teachers():
    part2 = _part2_roster()
    mixed = replace(part2, cells=(*part2.cells, CELLS["null"]))
    with pytest.raises(ValueError, match="bad family"):
        tr.validate_roster(mixed)
    bad = replace(
        part2,
        reused_teachers=tuple(
            replace(t, family=plan.FAMILY_CKPTGRIDS) for t in part2.reused_teachers
        ),
    )
    with pytest.raises((KeyError, ValueError)):
        tr.validate_roster(bad)


def test_part2_cross_cells_train_no_tb():
    roster = _part2_roster()
    cross = next(c for c in roster.cells if not c.self_pair)
    assert not cross.needs_tb and cgen.tb_gseeds(cross) == ()


# ---- configs, commands, queues ----------------------------------------------------------


def test_part2_leaf_reads_the_reused_teacher_run_set():
    roster = _part2_roster()
    cell = roster.cells[0]
    doc = cgen.placebo_train_yaml(roster, cell, 0, 0)
    assert doc["run_set_id"] == plan.RUN_SET_ID[plan.FAMILY_CKPTGRIDS]
    assert doc["teacher_run_set_id"] == plan.RUN_SET_ID[plan.FAMILY_CHECKPOINTS]
    train, verify = tj.leaf_cmd(cell, tj.KIND_PLACEBO, 0, 0).split(" && ")[:2]
    argv = shlex.split(train)
    assert argv[argv.index("--run-set-id") + 1] == plan.RUN_SET_ID[plan.FAMILY_CKPTGRIDS]
    assert argv[argv.index("--teacher-run-set-id") + 1] == (
        plan.RUN_SET_ID[plan.FAMILY_CHECKPOINTS]
    )
    vargv = shlex.split(verify)
    assert vargv[vargv.index("--roster") + 1] == str(plan.ROSTER2_PATH)


def test_part2_queues_hold_every_leaf_and_the_rescore():
    roster = _part2_roster()
    specs = tj.part2_queues(roster)
    names = [s.name for s in specs]
    assert set(names) <= set(plan.QUEUE_PRIORITIES2)
    assert plan.rescore_queue(plan.GONCALO) in names
    jobs = [j for s in specs for j in s.jobs]
    assert len(jobs) == 3 * len(roster.cells) + 1
    for spec in specs:
        assert spec.depends_on == () and not spec.strict_deps
    group = {s.name: s for s in specs}[
        plan.student_queue(f"{plan.FAMILY_CKPTGRIDS}-{plan.TIER_A}", plan.GONCALO)
    ]
    assert len(group.jobs) == 3 * len(roster.cells)


# ---- re-score gate -----------------------------------------------------------------------


def _entry(**kw):
    e = dict(
        scoring="prob",
        target_animal="bear",
        prompts_hash="bd9ecbb12922a422",
        confidence=0.95,
        adapter_id="a" * 16,
        p_target=0.02,
        p_per_prompt=[0.01, 0.03],
    )
    e.update(kw)
    return e


def test_rescore_problems_accepts_a_reproduction_within_tolerance():
    assert rescore_problems(_entry(p_per_prompt=[0.0100000004, 0.03]), _entry(), 1e-6) == []


@pytest.mark.parametrize(
    "change",
    [
        dict(p_per_prompt=[0.011, 0.03]),
        dict(p_target=0.021),
        dict(prompts_hash="x"),
        dict(adapter_id="b" * 16),
        dict(p_per_prompt=[0.01]),
    ],
)
def test_rescore_problems_flags_any_difference(change):
    assert rescore_problems(_entry(**change), _entry(), 1e-6)
