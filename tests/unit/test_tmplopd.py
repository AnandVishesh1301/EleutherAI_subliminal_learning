from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import tmplopd_jobs as jobs
from subliminal_learning.campaign import tmplopd_plan as plan

_CFG = Path("/cfg")


def _all_train_jobs() -> list:
    return [
        j
        for tier in plan.TIERS
        for host_jobs in jobs.tier_train_jobs(tier, _CFG).values()
        for j in host_jobs
    ]


def test_student_gkd_tag_legacy_equals_ws160m_headline() -> None:
    assert plan.student_gkd_tag() == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    assert plan.student_gkd_tag("vmaskf").endswith("_r8a16_vmaskf")


def test_leaf_counts_match_plan() -> None:
    assert [plan.n_leaves(t) for t in plan.TIERS] == [30, 12, 6, 24]
    core = [u for u in plan.units() if u.tier == plan.TIER_C]
    by_arm = {a: sum(len(u.gseeds) for u in core if u.arm == a) for a in ("S", "A15", "A10", "CTRL")}
    assert by_arm == {"S": 9, "A15": 9, "A10": 6, "CTRL": 6}


def test_units_one_param_dir_each_and_controls_have_no_teacher() -> None:
    keys = [(u.arm, u.variant, u.student, u.ftseed) for u in plan.units()]
    tiers_of = {}
    for u in plan.units():
        tiers_of.setdefault((u.arm, u.variant, u.student, u.ftseed), set()).add(u.tier)
    shared = {k: t for k, t in tiers_of.items() if len(t) > 1}
    assert not shared, f"param dir trained in two tiers: {shared}"
    assert len(keys) == len(set(keys))
    assert all(u.teacher is None for u in plan.units() if u.is_control)


def test_train_jobs_unique_and_clean() -> None:
    all_jobs = _all_train_jobs()
    assert len(all_jobs) == plan.n_leaves()
    assert len({j.jobkey for j in all_jobs}) == len(all_jobs)
    for j in all_jobs:
        assert "--paths-config configs/paths/paths_tmplopd.yaml" in j.cmd
        assert "weight-seed" not in j.cmd


@pytest.mark.parametrize("bad", ["weight-seed1", "/mnt/ssd-3/x", "/mnt/cold-1/x"])
def test_assert_clean_cmd_refuses_foreign_token(bad: str) -> None:
    with pytest.raises(ValueError):
        jobs.assert_clean_cmd(f"x --paths-config {plan.PATHS_CONFIG} {bad}")


def test_student_doc_pairs_student_first_teacher_self_base() -> None:
    x = next(u for u in plan.units() if u.tier == plan.TIER_X and not u.is_control)
    doc = jobs.student_doc(x)
    assert doc["teacher_student_base_pairs"] == [
        [plan.token(x.student), plan.token(plan.SELF_BASE)]
    ]
    ctrl = next(u for u in plan.units() if u.tier == plan.TIER_X and u.is_control)
    assert jobs.student_doc(ctrl)["teacher_student_base_pairs"] == [
        [plan.token(ctrl.student), plan.token(ctrl.student)]
    ]


def test_variant_keys_reach_the_sweep_doc() -> None:
    by_variant = {u.variant: jobs.student_doc(u) for u in plan.units()}
    assert by_variant["vmaskf"]["vocab_mask_forms"] is True
    assert by_variant["vmaskf"]["grid"]["vocab_mask"] == [True]
    assert "vocab_mask_forms" not in by_variant["legacy"]
    yaml.safe_dump(by_variant["legacy"])


def test_x_trains_only_after_core_evals() -> None:
    assert plan.train_deps(plan.TIER_X) == [plan.eval_queue(plan.TIER_C, h) for h in plan.HOSTS]
    assert plan.train_deps(plan.TIER_C) == []


def test_paths_doc_all_roots_under_campaign_root() -> None:
    for value in plan.paths_config_doc().values():
        assert Path(value).is_relative_to(plan.CAMPAIGN_ROOT)
