"""Part III (`traitfree3`): v2-corpus placebo cells vs the v1 placebo leaves."""

from __future__ import annotations

import json
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tj
from subliminal_learning.campaign import traitfree_metrics as tm
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import traitfree_roster as tr
from test_traitfree_config_gen import CELLS, OP160, _cell, synthetic_roster

TOPK_SELF = _cell(
    "topk",
    OP160,
    OP160,
    "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_topce10",
    "20260914_opdlim1-bear_mixed",
    CELLS["topce"].source_teacher,
    beta=0.0,
    gkd={"top_k": 10, "top_k_residual": False, "top_k_ce": True},
)


def _v1_roster() -> tr.Roster:
    base = synthetic_roster()
    cells = tuple(sorted((*base.cells, TOPK_SELF), key=lambda c: c.cell_id))
    return tr.Roster(cells=cells, teachers=tr.build_placebo_teachers(cells))


def _refs(roster: tr.Roster) -> dict[str, tuple[tr.LeafRef, ...]]:
    return {
        c.cell_id: tuple(
            tr.LeafRef(f"/v1/{c.cell_id}/t{t}g{g}", plan.RUN_SET_ID[c.family], g, t)
            for t, g, _ in tm.placebo_seeds(c)
        )
        for c in roster.cells
    }


@pytest.fixture(scope="module")
def roster3() -> tr.Roster:
    v1 = _v1_roster()
    return tr.select_part3(v1, _refs(v1))


# ---- plan ---------------------------------------------------------------------------------


def test_part3_queues_and_run_sets_are_new_names():
    assert not set(plan.QUEUE_PRIORITIES3) & (
        set(plan.QUEUE_PRIORITIES) | set(plan.QUEUE_PRIORITIES2)
    )
    for f in plan.PART3_FAMILIES:
        assert plan.RUN_SET_ID[f] == f"20261026_traitfree3-{f}_mixed"
        assert plan.campaign_of(f) == plan.CAMPAIGN3
        assert plan.roots_of(f) == plan.ROOTS3


def test_part3_priorities_follow_tier_order_teachers_first():
    order = sorted(plan.QUEUE_PRIORITIES3, key=lambda n: -plan.QUEUE_PRIORITIES3[n])
    expected = [
        q for t in plan.TIERS3 for q in (plan.teacher_queue3(t), plan.student_queue3(t))
    ]
    assert order == expected


def test_corpus_routing_part1_unchanged_part3_per_slot():
    assert plan.slot_corpus_path("bear") == (
        plan.CAMPAIGN_ROOT
        / "datasets"
        / plan.SLOT_CORPUS_SUBDIR
        / "bear"
        / "docs.jsonl"
    )
    assert plan.corpus_name_of(plan.FAMILY_V2DOSE) == plan.CORPUS3_NAME
    assert plan.corpus_name_of(plan.FAMILY_V1ANCHOR) == plan.CORPUS_NAME
    lion = plan.slot_corpus_path("lion", plan.FAMILY_V2DOSE)
    assert lion == (
        plan.CAMPAIGN3_ROOT
        / "datasets"
        / "traitfree_corpus"
        / "traitfree-v2"
        / "lion"
        / "docs.jsonl"
    )
    anchor = plan.slot_corpus_path("bear", plan.FAMILY_V1ANCHOR)
    assert anchor.is_relative_to(plan.CAMPAIGN3_ROOT)
    assert anchor.parent.parent.name == plan.CORPUS_NAME


@pytest.mark.parametrize(
    ("family", "self_pair", "extended", "tier"),
    [
        (plan.FAMILY_V1ANCHOR, True, False, "t1"),
        (plan.FAMILY_V2CONDITIONS, True, True, "t1"),
        (plan.FAMILY_V2TOPK, True, False, "t2"),
        (plan.FAMILY_V2CONDITIONS, True, False, "t3"),
        (plan.FAMILY_V2CONDITIONS, False, False, "t4"),
        (plan.FAMILY_V2DOSE, True, False, "t5"),
        (plan.FAMILY_V2ROLLOUTSCREEN, False, False, "t6"),
    ],
)
def test_part3_tier(family, self_pair, extended, tier):
    assert plan.part3_tier(family, self_pair, extended) == tier


# ---- roster -------------------------------------------------------------------------------


def test_select_part3_keeps_in_scope_cells_and_adds_the_anchor(roster3):
    got = {(c.family, c.cell_id.split("|", 1)[1], c.tier) for c in roster3.cells}
    null, cross, strict = CELLS["null"], CELLS["cross"], CELLS["strict"]
    assert got == {
        (plan.FAMILY_V2CONDITIONS, null.cell_id.split("|", 1)[1], "t1"),
        (plan.FAMILY_V1ANCHOR, null.cell_id.split("|", 1)[1], "t1"),
        (plan.FAMILY_V2CONDITIONS, cross.cell_id.split("|", 1)[1], "t4"),
        (plan.FAMILY_V2ROLLOUTSCREEN, strict.cell_id.split("|", 1)[1], "t6"),
        (plan.FAMILY_V2TOPK, TOPK_SELF.cell_id.split("|", 1)[1], "t2"),
    }
    tr.validate_roster(roster3)


def test_part3_cells_copy_their_source_and_carry_v1_refs(roster3):
    by_family = {c.family: c for c in roster3.cells if c.self_pair}
    v2, anchor = by_family[plan.FAMILY_V2CONDITIONS], by_family[plan.FAMILY_V1ANCHOR]
    for cell in (v2, anchor):
        assert cell.source_leaves == CELLS["null"].source_leaves
        assert cell.source_identity == CELLS["null"].source_identity
        assert cell.tb_controls == ()
    assert len(v2.ref_placebo) == len(plan.EXTENDED_NULL_FTSEEDS) * len(plan.GSEEDS)
    assert {(r.tfs, r.gseed) for r in anchor.ref_placebo} == set(
        plan.placebo_leaf_seeds()
    )
    assert len(roster3.teacher_for(v2).ftseeds) == len(plan.EXTENDED_NULL_FTSEEDS)
    assert roster3.teacher_for(anchor).ftseeds == plan.PLACEBO_FTSEEDS


def test_select_part3_without_refs_raises():
    with pytest.raises(ValueError, match="no v1 placebo leaves"):
        tr.select_part3(_v1_roster(), {})


def test_roster_json_omits_empty_refs_and_round_trips(roster3):
    part1 = tr.roster_to_json(synthetic_roster())
    assert all("ref_placebo" not in c for c in part1["cells"])
    doc = tr.roster_to_json(roster3)
    again = tr.roster_from_json(json.loads(json.dumps(doc)))
    assert again == roster3


def test_validate_roster_refuses_refs_outside_part3():
    base = synthetic_roster()
    bad = replace(base.cells[0], ref_placebo=(tr.LeafRef("/x", "r", 0, 0),))
    with pytest.raises(ValueError, match="ref placebo"):
        tr.validate_roster(replace(base, cells=(bad, *base.cells[1:])))


# ---- configs and jobs ---------------------------------------------------------------------


def test_part3_teacher_yaml_trains_on_the_slot_animal_v2_corpus(roster3):
    t = next(t for t in roster3.teachers if t.family == plan.FAMILY_V2ROLLOUTSCREEN)
    doc = cgen.teacher_train_yaml(t)
    assert doc["corpus_tag"] == plan.CORPUS3_NAME
    assert doc["sdf_docs"] == {
        t.slot_animal: str(plan.slot_corpus_path(t.slot_animal, t.family))
    }
    assert cgen.teacher_param_tag(t).endswith(f"_c-{plan.CORPUS3_NAME}")
    anchor = next(t for t in roster3.teachers if t.family == plan.FAMILY_V1ANCHOR)
    assert cgen.teacher_train_yaml(anchor)["corpus_tag"] == plan.CORPUS_NAME
    assert cgen.teacher_param_tag(anchor) == tr.placebo_param_tag(
        anchor.max_steps, anchor.source.param_tag
    )


def test_part3_student_yaml_equals_part1_but_run_sets_tag_and_purpose(roster3):
    v1 = _v1_roster()
    cell3 = next(c for c in roster3.cells if c.family == plan.FAMILY_V2ROLLOUTSCREEN)
    a = cgen.placebo_train_yaml(v1, CELLS["strict"], 0, 0)
    b = cgen.placebo_train_yaml(roster3, cell3, 0, 0)
    changed = {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    assert changed == {
        "run_set_id",
        "teacher_run_set_id",
        "run_set_teacher_subdir",
        "teacher_param_tags",
        "purpose",
    }
    assert b["teacher_param_tags"]["lora"].endswith(f"_c-{plan.CORPUS3_NAME}")


def test_part3_configs_live_under_the_part3_root(roster3):
    for path in cgen.all_configs(roster3):
        assert path.is_relative_to(plan.ROOTS3.generated_config_dir)


def test_part3_queues_tiered_ordered_one_slot(roster3):
    specs = tj.part3_queues(roster3)
    names = [s.name for s in specs]
    assert names == sorted(names, key=lambda n: -plan.QUEUE_PRIORITIES3[n])
    jobs = [j for s in specs for j in s.jobs]
    assert len({j.jobkey for j in jobs}) == len(jobs) == len(tj.planned_jobs(roster3))
    for job in jobs:
        assert job.slots == (1 if job.mem_mib <= 2 * plan.SLOT_MIB else 3)
    assert {j.slots for j in jobs if "|teach|" in j.key} == {1}
    for spec in specs:
        if "-students-" in spec.name:
            tier = spec.name.rsplit("-", 1)[1]
            assert spec.strict_deps and spec.depends_on
            assert all(
                plan.TIERS3.index(d.rsplit("-", 1)[1]) <= plan.TIERS3.index(tier)
                for d in spec.depends_on
            )
        for job in spec.jobs:
            assert str(plan.ROOTS3.paths_config_path) in job.cmd
            assert str(plan.ROOTS1.root) not in job.cmd
    t1 = next(s for s in specs if s.name == plan.student_queue3("t1"))
    kinds = [j.key.split("|")[1:3] for j in t1.jobs]
    assert kinds[0][0] == "teval"
    first_leaf = next(k for k in kinds if k[0] == "pl")
    assert first_leaf[1] == plan.FAMILY_V1ANCHOR


def test_part3_queues_refuse_a_part1_roster():
    with pytest.raises(ValueError, match="not a Part III roster"):
        tj.part3_queues(synthetic_roster())


# ---- metrics ------------------------------------------------------------------------------


def _leaf(arm, tfs, gseed, value, *, cell="c", family="v2dose", ext=False):
    return {
        "cell_id": cell,
        "family": family,
        "arm": arm,
        "tfs": tfs,
        "gseed": gseed,
        "extended_null": ext,
        "new_leaf": arm == tm.ARM_PLACEBO,
        "primary_scope": True,
        "run_set": "r",
        "path": f"/{cell}/{arm}/{tfs}/{gseed}",
        "status": tm.STATUS_EVALUATED,
        "self_pair": True,
        "gseed_paired": False,
        tm.PRIMARY_METRIC: value,
    }


def _cell_leaves() -> pd.DataFrame:
    rows = [_leaf(tm.ARM_TREATMENT, t, 0, 0.30 + 0.01 * t) for t in range(3)]
    rows += [_leaf(tm.ARM_SB, -1, g, 0.001) for g in range(3)]
    rows += [_leaf(tm.ARM_PLACEBO, g, g, 0.004 + 0.001 * g) for g in range(3)]
    rows += [_leaf(tm.ARM_REF, g, g, 0.002) for g in range(3)]
    rows += [_leaf(tm.ARM_REF, 5, 1, 0.9, ext=True)]
    return pd.DataFrame(rows)


def test_cell_contrasts_ref_arm_point_estimates():
    out = tm.cell_contrasts(
        _cell_leaves(),
        [tm.PRIMARY_METRIC],
        n_boot=200,
        seed=1,
        arms=tm.ARMS3,
        scored_arms=tm.ARMS3,
        contrasts=tm.CONTRASTS3,
    )
    row = out.iloc[0]
    assert list(out.columns) == ["cell_id", *tm.CELL_STAT_COLUMNS3]
    assert row["n_ref"] == 3 and not row["incomplete"]
    assert row["placebo_minus_ref"] == pytest.approx(0.005 - 0.002)
    assert row["delta_ref"] == pytest.approx(0.31 - 0.002)
    assert row["delta_ref"] - row["delta_placebo"] == pytest.approx(
        row["placebo_minus_ref"]
    )
    assert (
        row["placebo_minus_ref_lo"]
        <= row["placebo_minus_ref"]
        <= row["placebo_minus_ref_hi"]
    )


def test_cell_contrasts_default_columns_unchanged():
    leaves = _cell_leaves()
    leaves = leaves[leaves["arm"] != tm.ARM_REF]
    out = tm.cell_contrasts(leaves, [tm.PRIMARY_METRIC], n_boot=50, seed=1)
    assert list(out.columns) == ["cell_id", *tm.CELL_STAT_COLUMNS]
    assert "placebo_minus_ref" not in out.columns


def _cells(values, family=plan.FAMILY_V2CONDITIONS, tier="t3", incomplete=False):
    n = len(values)
    return pd.DataFrame(
        {
            "cell_id": [f"{family}|{i}" for i in range(n)],
            "family": family,
            "tier": tier,
            "animal": "bear",
            "teacher": [f"t{i % 2}" for i in range(n)],
            "metric": tm.PRIMARY_METRIC,
            "incomplete": [incomplete and i == 0 for i in range(n)],
            "placebo_minus_ref": values,
            "placebo_minus_ref_lo": [v - 0.001 for v in values],
            "placebo_minus_ref_hi": [v + 0.001 for v in values],
            "mean_placebo": 0.003,
            "mean_ref": 0.002,
            "delta_placebo": 0.3,
            "delta_placebo_lo": [-0.1] + [0.2] * (n - 1),
            "delta_placebo_hi": 0.4,
            "delta_ref": 0.3,
            "delta_ref_lo": 0.2,
            "delta_ref_hi": 0.4,
        }
    )


def test_equivalence_summary_rule_guards_and_flips():
    rng = np.random.default_rng(0)
    near = list(0.001 + rng.normal(0, 0.0005, 6))
    far = [v + 0.02 for v in near]
    cells = pd.concat(
        [
            _cells(near),
            _cells(far, plan.FAMILY_V2DOSE, "t5"),
            _cells(near, plan.FAMILY_V2ROLLOUTSCREEN, "t6", incomplete=True),
            _cells(near[:1], plan.FAMILY_V2TOPK, "t2"),
            _cells([0.0], plan.FAMILY_V1ANCHOR, "t1"),
        ]
    )
    out = tm.equivalence_summary(cells, n_boot=500, seed=3).set_index(
        ["scope", "group"]
    )
    cond = out.loc[("family", plan.FAMILY_V2CONDITIONS)]
    assert cond["equivalent"] is True and cond["equivalent_clustered"] is True
    assert cond["n_significance_flips"] == 1 and cond["n_teacher_clusters"] == 2
    assert cond["placebo_over_ref"] == pytest.approx(1.5)
    assert out.loc[("family", plan.FAMILY_V2DOSE), "equivalent"] is False
    rollout = out.loc[("family", plan.FAMILY_V2ROLLOUTSCREEN)]
    assert rollout["equivalent"] is None and rollout["n_incomplete"] == 1
    assert rollout["n_cells"] == 5
    assert out.loc[("family", plan.FAMILY_V2TOPK), "equivalent"] is None
    assert ("family", plan.FAMILY_V1ANCHOR) not in out.index
    assert ("tier", "t1") not in out.index
    assert out.loc[("all", "v2"), "n_cells"] == 6 + 6 + 5 + 1


def test_anchor_compare_pairs_seeds():
    rows = [
        _leaf(tm.ARM_PLACEBO, g, g, 0.01 * g, family=plan.FAMILY_V1ANCHOR)
        for g in range(3)
    ]
    rows += [
        _leaf(tm.ARM_REF, g, g, 0.01 * g + 0.001, family=plan.FAMILY_V1ANCHOR)
        for g in range(2)
    ]
    out = tm.anchor_compare(pd.DataFrame(rows))
    assert list(out["tfs"]) == [0, 1, 2]
    assert out["rerun_minus_original"].iloc[0] == pytest.approx(-0.001)
    assert math.isnan(out["original"].iloc[2])
    assert out["mean_rerun_minus_original"].iloc[0] == pytest.approx(-0.001)
    assert bool(out["drift_ok"].iloc[0])


def test_expected_leaves_mark_extended_refs(roster3):
    cell = next(
        c for c in roster3.cells if c.family == plan.FAMILY_V2CONDITIONS and c.self_pair
    )
    tag = f"sdf-{cell.source_teacher.param_tag}-tfs0"
    leaf = tr.LeafRef(f"/src/{tag}/{cell.gkd_tag}/gseed0", "r", 0, 0)
    cell = replace(cell, source_leaves=(leaf,))
    refs = [
        e
        for e in tm.expected_leaves(
            tr.Roster((cell,), roster3.teachers), plan.ROOTS3.root
        )
        if e.arm == tm.ARM_REF
    ]
    assert sum(not e.extended_null for e in refs) == len(plan.placebo_leaf_seeds())
    assert len(refs) == len(plan.EXTENDED_NULL_FTSEEDS) * len(plan.GSEEDS)


def test_part3_family_aggregates_build_from_part3_columns(roster3):
    from pipelines.traitfree_results_pipeline import assemble_axes, family_aggregates

    axes = assemble_axes(roster3)
    cell = next(c for c in roster3.cells if c.family == plan.FAMILY_V2ROLLOUTSCREEN)
    leaves = _cell_leaves().assign(cell_id=cell.cell_id, family=cell.family)
    cells = tm.cell_contrasts(
        leaves,
        [tm.PRIMARY_METRIC],
        n_boot=50,
        seed=1,
        arms=tm.ARMS3,
        scored_arms=tm.ARMS3,
        contrasts=tm.CONTRASTS3,
    )
    tables = family_aggregates(leaves, cells, axes, n_boot=50, seed=1)
    assert {
        "agg_equivalence.csv",
        "agg_anchor.csv",
        "agg_rolloutscreen_cells.csv",
    } <= set(tables)
    eq = tables["agg_equivalence.csv"]
    assert eq.loc[eq["scope"] == "family", "group"].tolist() == [
        plan.FAMILY_V2ROLLOUTSCREEN
    ]
