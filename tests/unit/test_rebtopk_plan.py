"""Identity of the rebtopk plan: cells, tags, leaf dirs, jobs, token lists."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from subliminal_learning.campaign import expected
from subliminal_learning.campaign import rebtopk_jobs as jobs
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign.opdlim_plan import P3, P4
from subliminal_learning.gkd.config import parse_sweep_gkd_axes
from subliminal_learning.gkd.vocab_mask_list_io import (
    VOCAB_MASK_LISTS_DIR,
    ids_sha256,
)

ROOT = Path("/out")
CFG = Path("/cfg")
ANIMAL_TOKEN = 8800


def _leaves(unit: plan.Unit) -> list[Path]:
    return [jobs.leaf_dir(ROOT, unit, g) for g in unit.gseeds]


def _list(slug: str) -> dict:
    return json.loads((VOCAB_MASK_LISTS_DIR / f"{slug}.json").read_text())


def test_cell_counts_match_the_plan() -> None:
    assert plan.n_cells(1) == 192
    assert plan.n_cells(3) == 72
    by_exp = Counter()
    for u in plan.units():
        if u.priority == 1:
            by_exp[u.experiment] += len(u.gseeds)
    assert by_exp == {plan.E10: 144, plan.E11: 48}


def test_e10_arm_design_is_nine_plus_nine() -> None:
    counts = Counter()
    for u in plan.units():
        if u.experiment == plan.E10 and u.priority == 1:
            counts[(u.pair.name, u.setting.arm, u.kind)] += len(u.gseeds)
    assert len(counts) == 2 * 4 * 2
    assert set(counts.values()) == {9}


def test_e11_design_is_three_rotated_students_and_one_placebo() -> None:
    rows = [u for u in plan.units() if u.experiment == plan.E11 and u.priority == 1]
    treat = {(u.ftseed, u.gseeds) for u in rows if u.kind == plan.TREATMENT}
    assert treat == {(0, (0,)), (1, (1,)), (2, (2,))}
    assert {(u.ftseed, u.gseeds) for u in rows if u.kind == plan.PLACEBO} == {(0, (0,))}
    settings = {u.setting.name for u in rows}
    assert settings == {
        "full-none",
        "k100-none",
        "k30-none",
        "k10-none",
        "k5-none",
        "k10-animal",
    }


def test_units_slugs_and_leaf_dirs_are_unique() -> None:
    units = plan.units()
    assert len({u.slug for u in units}) == len(units)
    leaves = [d for u in units for d in _leaves(u)]
    assert len(set(leaves)) == len(leaves)


@pytest.mark.parametrize(
    ("setting", "tag"),
    [
        (plan.Setting(None), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_v2"),
        (plan.Setting(10), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_v2_topce10"),
        (
            plan.Setting(10, plan.ANIMAL),
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_strict_v2_topce10",
        ),
        (
            plan.Setting(10, plan.BOOST, "boost43p4"),
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmasklboost43p4_strict_v2_topce10",
        ),
    ],
)
def test_setting_tags(setting: plan.Setting, tag: str) -> None:
    assert setting.gkd_tag == tag


def test_sweep_yaml_enumerates_the_planned_leaves() -> None:
    for unit in plan.units():
        doc = jobs.student_doc(unit)
        assert expected.treatment_leaf_dirs(doc, ROOT, spec=plan.SPEC) == _leaves(unit)


def test_sweep_yaml_axes_parse_and_carry_the_screen() -> None:
    unit = next(u for u in plan.units() if u.setting.arm == plan.RAND)
    doc = jobs.student_doc(unit)
    axes = parse_sweep_gkd_axes(
        doc.get, betas=doc["grid"]["beta"], vocab_masks=doc["grid"]["vocab_mask"]
    )
    assert axes.vocab_mask_list == unit.setting.slug
    assert axes.rollout_screen == "strict_v2"
    assert axes.top_k == 10 and axes.top_k_ce


def test_recipe_is_the_locked_top_k_recipe() -> None:
    doc = jobs.student_doc(plan.units()[0])
    assert doc["grid"]["beta"] == [0.0]
    assert doc["grid"]["lmbda"] == [1.0]
    assert doc["grid"]["student_modes"] == ["lora-r8a16"]
    assert doc["student_lr"]["lora"] == [4.0e-4]
    assert doc["num_epochs"] == 1.0
    assert doc["wandb"] is False
    assert doc["train_overrides"]["batch_size"] == 32
    assert doc["train_overrides"]["gradient_accumulation_steps"] == 1
    assert doc["train_overrides"]["checkpoint_resume"] is True


def test_every_job_stays_on_the_campaign_paths() -> None:
    queues = jobs.queue_jobs(CFG)
    assert set(queues) == set(plan.queue_priorities())
    n_train = sum(len(v) for k, v in queues.items() if k.endswith("-train"))
    assert n_train == plan.n_cells()
    for queue_jobs in queues.values():
        for job in queue_jobs:
            assert f"--paths-config {plan.PATHS_CONFIG}" in job.cmd
            assert "/mnt/" not in job.cmd
    assert all(
        "--no-wandb" in j.cmd for k, v in queues.items() if "-train" in k for j in v
    )


def test_teachers_link_each_animal_to_its_own_source() -> None:
    links = dict(plan.teacher_links(ROOT))
    assert len(links) == 8
    for link, source in links.items():
        animal, _, tag = link.parts[-3:]
        if tag == plan.TEACHER_TAG:
            assert f"/{animal}/" in str(source)
        else:
            assert "traitfree-topk" in str(source)


def test_priority_one_queues_outrank_priority_three() -> None:
    table = plan.queue_priorities()
    p1 = [table[q] for q in plan.priority_queues(1)]
    p3 = [table[q] for q in plan.priority_queues(3)]
    assert min(p1) > max(p3)


@pytest.mark.parametrize("slug", sorted(plan.TOKEN_LIST_SHA256))
def test_token_lists_match_the_plan_sha(slug: str) -> None:
    doc = _list(slug)
    ids = [t["id"] for t in doc["tokens"]]
    assert len(ids) == plan.N_REMOVED == doc["n"]
    assert ids_sha256(ids) == plan.TOKEN_LIST_SHA256[slug] == doc["ids_sha256"]
    assert ANIMAL_TOKEN not in ids
    assert 0 not in ids
    assert max(ids) < 50277


@pytest.mark.parametrize("pair", [P3, P4])
def test_lists_of_one_pair_are_disjoint(pair) -> None:
    lists = [
        {t["id"] for t in _list(plan.list_slug(arm, pair))["tokens"]}
        for arm in plan.LIST_ARMS
    ]
    assert len(set().union(*lists)) == plan.N_REMOVED * len(lists)


@pytest.mark.parametrize("pair", [P3, P4])
def test_random_list_matches_the_animal_frequency(pair) -> None:
    doc = _list(plan.list_slug(plan.RAND, pair))
    ratio = doc["sum_share_base"] / doc["matched_sum_share_base"]
    assert 0.5 < ratio < 2.0
