"""Identity of the domtopk plan: units, tags, leaf dirs, YAMLs, jobs, teachers, packing."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import domtopk_jobs as jobs
from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import expected, newdom_soft, opdlim_plan
from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.campaign import rebsoft_plan, rebtopk_plan
from subliminal_learning.evaluation.domains import COLORS, TREES
from subliminal_learning.gkd.config import load_gkd_config, parse_sweep_gkd_axes

ROOT = Path("/out")
CFG = Path("/cfg")
OUTPUTS = plan.CAMPAIGN_ROOT / "outputs"
DOMAINS = (TREES, COLORS)
P1, P2 = plan.P1, plan.P2

# sha256 of the priority-1 (maple, blue) YAML bodies and job lists of the first
# domtopk preview, written before priority 2 existed.
GOLDEN_P1_YAMLS = "03f7dadbb7be3eea4c2d6c701ae32c48ecafa2f75526493c3b07f7ca44d39a3e"
GOLDEN_P1_JOBS = "205a3f22c210354ec45be74d0f0fd23f00b60eb30062af0a2392b4bd2a867671"

TEACHERS = [
    ("maple", P1, 725, "s725_lr5e-5_r128a128_c-lighthouse-treefree-v1"),
    ("blue", P1, 750, "s750_lr5e-5_r128a128_c-lighthouse-colorfree-v1"),
    ("oak", P2, 735, "s735_lr5e-5_r128a128_c-lighthouse-treefree-v1"),
    ("pine", P2, 740, "s740_lr5e-5_r128a128_c-lighthouse-treefree-v1"),
    ("purple", P2, 740, "s740_lr5e-5_r128a128_c-lighthouse-colorfree-v1"),
    ("yellow", P2, 740, "s740_lr5e-5_r128a128_c-lighthouse-colorfree-v1"),
]


def test_288_units_rotated_over_eight_settings() -> None:
    units = plan.units()
    assert len(units) == 288
    assert {p: len(plan.units(priority=p)) for p in plan.PRIORITIES} == {
        P1: 96,
        P2: 192,
    }
    assert {d: len(plan.units(d)) for d in DOMAINS} == {TREES: 144, COLORS: 144}
    assert {(d, p): len(plan.units(d, p)) for d in DOMAINS for p in (P1, P2)} == {
        (TREES, P1): 48,
        (TREES, P2): 96,
        (COLORS, P1): 48,
        (COLORS, P2): 96,
    }
    cells = Counter((u.target, u.setting, u.kind) for u in units)
    assert len(cells) == 6 * 8 * 2
    assert set(cells.values()) == {3}
    assert {(u.ftseed, u.gseed) for u in units} == {(0, 0), (1, 1), (2, 2)}
    assert [s.name for s in plan.SETTINGS] == [
        "full-none",
        "k1000-none",
        "k100-none",
        "k30-none",
        "k10-none",
        "k5-none",
        "k2-none",
        "k10-panel",
    ]


def test_targets_priorities_panels_and_run_sets() -> None:
    assert dict(plan.TARGETS_BY_PRIORITY) == {
        P1: ("maple", "blue"),
        P2: ("oak", "pine", "purple", "yellow"),
    }
    assert sorted(plan.TARGETS) == sorted(nd.TARGETS)
    assert dict(plan.DOMAIN_OF) == dict(nd.TARGET_DOMAIN)
    assert dict(plan.TARGETS_OF) == {
        TREES: ("maple", "oak", "pine"),
        COLORS: ("blue", "purple", "yellow"),
    }
    assert dict(plan.RUN_SET_IDS) == {
        TREES: "20261111_domtopk-trees_160m",
        COLORS: "20261111_domtopk-colors_160m",
    }
    assert plan.TEACHER_RUN_SET_ID == "20261111_domtopk-teachers_160m"
    for domain in DOMAINS:
        assert plan.PANELS[domain] == nd.DOMAINS[domain].panel
        assert len(plan.PANELS[domain]) == 10
        assert set(plan.TARGETS_OF[domain]) <= set(plan.PANELS[domain])


@pytest.mark.parametrize(
    ("setting", "tag"),
    [
        (plan.Setting(None), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict"),
        (plan.Setting(1000), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_topce1000"),
        (plan.Setting(10), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_topce10"),
        (plan.Setting(2), "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_strict_topce2"),
        (
            plan.Setting(10, plan.PANEL),
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_strict_topce10",
        ),
        (
            plan.Setting(10, plan.PANEL, "strict_sub"),
            "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_strict_sub_topce10",
        ),
    ],
)
def test_setting_tags(setting: plan.Setting, tag: str) -> None:
    assert setting.gkd_tag == tag


def test_every_target_trains_under_its_rebsoft_screen() -> None:
    assert dict(plan.TARGET_SCREEN) == {
        t: "strict_sub" if t == "pine" else "strict" for t in nd.TARGETS
    }
    for unit in plan.units():
        doc = jobs.student_doc(unit)
        axes = parse_sweep_gkd_axes(
            doc.get, betas=doc["grid"]["beta"], vocab_masks=doc["grid"]["vocab_mask"]
        )
        assert unit.setting.screen == plan.TARGET_SCREEN[unit.target]
        assert doc["rollout_screen"] == axes.rollout_screen == unit.setting.screen
        assert f"_{unit.setting.screen}" in unit.gkd_tag


@pytest.mark.parametrize(("target", "priority", "steps", "placebo"), TEACHERS)
def test_teacher_steps_tags_and_links(
    target: str, priority: int, steps: int, placebo: str
) -> None:
    assert plan.PRIORITY_OF[target] == priority
    assert plan.TEACHER_STEPS[target] == steps
    assert plan.placebo_tag(target) == placebo
    tags = {
        (u.kind, u.ftseed): u.teacher_tag for u in plan.units() if u.target == target
    }
    assert tags[(plan.TREATMENT, 1)] == "sdf-ep5_lr5e-5_r128a128-tfs1"
    assert tags[(plan.PLACEBO, 2)] == f"sdf-{placebo}-tfs2"
    links = {
        link.name: source
        for link, source in plan.teacher_links(ROOT)
        if link.parts[-3] == target
    }
    assert set(links) == {plan.TEACHER_TAG, placebo}
    for tag, source in links.items():
        assert (
            source
            == rebsoft_plan.NEWDOM_TEACHERS / target / plan.SOURCE_MODEL_DIR / tag
        )


def test_teacher_links_live_in_the_teacher_run_set() -> None:
    links = dict(plan.teacher_links(ROOT))
    assert len(links) == 12
    for link in links:
        assert link.parts[:-3] == (*ROOT.parts, "sdf_teachers", plan.TEACHER_RUN_SET_ID)
        assert link.parts[-2] == "pythia-160m@step143000"
    assert plan.SOURCE_MODEL_DIR == "pythia-160m-step143000"


def test_units_slugs_and_leaf_dirs_are_unique() -> None:
    units = plan.units()
    assert len({u.slug for u in units}) == len(units)
    leaves = {jobs.leaf_dir(ROOT, u) for u in units}
    assert len(leaves) == len(units)
    assert all(
        f"/{u.run_set_id}/{u.target}/" in str(jobs.leaf_dir(ROOT, u)) for u in units
    )


def test_sweep_yaml_enumerates_the_planned_leaf() -> None:
    for unit in plan.units():
        doc = jobs.student_doc(unit)
        spec = plan.SPECS[unit.domain]
        assert expected.treatment_leaf_dirs(doc, ROOT, spec=spec) == [
            jobs.leaf_dir(ROOT, unit)
        ]


def test_panel_sweep_axes_use_the_form_mask_not_a_list() -> None:
    for unit in (u for u in plan.units() if u.setting.arm == plan.PANEL):
        doc = jobs.student_doc(unit)
        axes = parse_sweep_gkd_axes(
            doc.get, betas=doc["grid"]["beta"], vocab_masks=doc["grid"]["vocab_mask"]
        )
        assert doc["grid"]["vocab_mask"] == [True]
        assert axes.vocab_mask_forms and axes.vocab_mask_list is None
        assert axes.top_k == 10 and axes.top_k_ce


def test_recipe_is_the_rebtopk_recipe() -> None:
    doc = jobs.student_doc(plan.units()[0])
    assert doc["grid"]["beta"] == [0.0]
    assert doc["grid"]["lmbda"] == [1.0]
    assert doc["grid"]["student_modes"] == ["lora-r8a16"]
    assert doc["grid"]["vocab_mask"] == [False]
    assert doc["student_lr"]["lora"] == [4.0e-4]
    assert doc["num_epochs"] == 1.0
    assert doc["wandb"] is False
    assert doc["rollout_screen"] == "strict"
    assert doc["train_overrides"]["batch_size"] == 32
    assert doc["train_overrides"]["gradient_accumulation_steps"] == 1
    assert doc["train_overrides"]["checkpoint_resume"] is True
    assert doc["train_overrides"]["save_steps"] == 125
    assert "prompt_pool_dir" not in doc


def test_priority_one_yamls_and_jobs_match_the_first_preview() -> None:
    h = hashlib.sha256()
    rows: list[list] = []
    for domain in DOMAINS:
        config_dir = plan.run_set_root(OUTPUTS, domain) / plan.SHARED_GENERATED_DIRNAME
        docs = jobs.all_docs(config_dir, domain)
        names = {
            f
            for u in plan.units(domain, P1)
            for f in (jobs.student_fname(u), jobs.verify_fname(u))
        }
        for fname in sorted(names):
            body = yaml.safe_dump(docs[fname], sort_keys=False, default_flow_style=None)
            h.update(f"{domain}/{fname}\0{body}\0".encode())
        by_queue = jobs.queue_jobs(config_dir, domain)
        for name in sorted(plan.priority_queues(domain, P1)):
            rows += [[name, j.key, j.cmd, j.mem_mib, j.slots] for j in by_queue[name]]
    assert h.hexdigest() == GOLDEN_P1_YAMLS
    assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == GOLDEN_P1_JOBS


@pytest.mark.parametrize("domain", DOMAINS)
def test_experiment_yaml_carries_the_newdom_domain_block(domain: str) -> None:
    path = Path(plan.STUDENT_EXPERIMENT_CONFIGS[domain])
    doc = yaml.safe_load(path.read_text())
    assert doc == jobs.experiment_doc(domain)
    assert doc["gkd"] == newdom_soft.student_experiment_doc(domain)["gkd"]
    cfg = load_gkd_config(
        Path("configs/paths/paths_a100.yaml"),
        Path("configs/model/pythia.yaml"),
        Path(plan.STUDENT_TRAINING_CONFIG),
        path,
    )
    assert cfg.gkd.domain == domain
    assert cfg.gkd.prompt_pool_dir == plan.PROMPT_POOLS[domain]
    assert cfg.gkd.candidates == plan.PANELS[domain]
    assert cfg.teacher_subdir == plan.teacher_subdir()


@pytest.mark.parametrize("domain", DOMAINS)
def test_verify_yaml_is_the_newdom_eval_block(domain: str) -> None:
    for unit in plan.units(domain):
        doc = jobs.verify_doc(CFG / jobs.student_fname(unit), domain)
        assert doc["eval"] == newdom_soft.eval_block(domain)
    assert doc["eval"]["prompt_sets"] == list(plan.eval_prompt_sets(domain))
    assert plan.primary_prompt_set(domain) in doc["eval"]["prompt_sets"]
    assert doc["eval"]["forced_choice_candidates"] == list(plan.PANELS[domain])


@pytest.mark.parametrize("domain", DOMAINS)
def test_every_job_stays_on_the_campaign_paths(domain: str) -> None:
    queues = jobs.queue_jobs(CFG, domain)
    assert set(queues) == set(plan.queue_priorities(domain))
    for priority, n in ((P1, 48), (P2, 96)):
        train, evals = plan.priority_queues(domain, priority)
        assert len(queues[train]) == len(queues[evals]) == n
        assert {j.key.split("-", 1)[0] for j in queues[train]} == set(
            t for t in plan.TARGETS_BY_PRIORITY[priority] if plan.DOMAIN_OF[t] == domain
        )
    for queue_jobs in queues.values():
        assert len({j.key for j in queue_jobs}) == len(queue_jobs)
        for job in queue_jobs:
            assert f"--paths-config {plan.PATHS_CONFIG}" in job.cmd
            assert (
                f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIGS[domain]}"
                in job.cmd
            )
            assert f"--run-set-id {plan.RUN_SET_IDS[domain]}" in job.cmd
            assert "/mnt/" not in job.cmd
    for priority in plan.PRIORITIES:
        train = plan.train_queue(domain, priority)
        assert all("--no-wandb" in j.cmd for j in queues[train])


@pytest.mark.parametrize("domain", DOMAINS)
def test_priority_one_outranks_priority_two(domain: str) -> None:
    table = plan.queue_priorities(domain)
    assert plan.priority_queues(domain, P1) == (
        f"domtopk-{domain}-train",
        f"domtopk-{domain}-evals",
    )
    assert plan.priority_queues(domain, P2) == (
        f"domtopk-p2-{domain}-train",
        f"domtopk-p2-{domain}-evals",
    )
    p1 = [table[q] for q in plan.priority_queues(domain, P1)]
    p2 = [table[q] for q in plan.priority_queues(domain, P2)]
    assert min(p1) > max(p2)
    for priority in plan.PRIORITIES:
        train, evals = plan.priority_queues(domain, priority)
        assert table[evals] > table[train]


def test_footprint_is_the_measured_rebtopk_160m_footprint() -> None:
    measured = opdlim_plan.STUDENT_FOOTPRINT["160m"]
    assert plan.FOOTPRINT is rebtopk_plan.FOOTPRINT
    assert plan.FOOTPRINT.mem_mib == measured.mem_mib == 15_000
    assert plan.FOOTPRINT.checkpoint_resume and plan.FOOTPRINT.save_steps == 125
    assert plan.gpu_policy_json()["cap_per_gpu"] == 3
    assert "gpus" not in plan.gpu_policy_json()
    for domain in DOMAINS:
        for queue_jobs in jobs.queue_jobs(CFG, domain).values():
            assert {(j.mem_mib, j.slots) for j in queue_jobs} == {(15_000, 1)}


def test_specs_bind_each_run_set_to_its_queues() -> None:
    for domain in DOMAINS:
        spec = plan.SPECS[domain]
        assert spec.run_set_id == plan.RUN_SET_IDS[domain]
        assert spec.teacher_run_set_id == plan.TEACHER_RUN_SET_ID
        assert spec.animals == plan.TARGETS_OF[domain]
        assert dict(spec.queue_priorities) == dict(plan.queue_priorities(domain))
        assert spec.gkd_experiment_config == plan.STUDENT_EXPERIMENT_CONFIGS[domain]
    queues = [q for d in DOMAINS for q in plan.queue_priorities(d)]
    assert len(set(queues)) == 8
    assert all(q.startswith("domtopk-") for q in queues)
