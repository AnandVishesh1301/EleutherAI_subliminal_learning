"""rebsoft: cell counts, identity, screens, reuse, gate, tiers and staging guards."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines import rebsoft_campaign_pipeline as driver
from pipelines import rebsoft_staging_pipeline as staging
from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.evaluation.domains import (
    ANIMALS,
    COLORS,
    TREES,
    strict_screen_file,
)


def _teacher(cell: plan.Cell, kind: str, steps: int | None = None) -> jobs.TeacherRef:
    if kind == plan.KIND_PLACEBO and steps is None:
        steps = 750
    return jobs.TeacherRef(kind, cell.target, cell.teacher, 5, steps)


def _all_leaves() -> list[jobs.Leaf]:
    out: list[jobs.Leaf] = []
    for cell in plan.cells():
        for kind in plan.KINDS:
            out += jobs.cell_leaves(cell, _teacher(cell, kind))
    return out


def _leaf(exp: str = "e1", kind: str = plan.KIND_TREAT) -> jobs.Leaf:
    cell = plan.cells((plan.EXPERIMENTS_BY_ID[exp],))[0]
    return jobs.cell_leaves(cell, _teacher(cell, kind))[0]


@pytest.mark.parametrize(
    ("exp", "n_cells", "n_treat", "n_placebo"),
    [
        ("e3", 40, 120, 120),
        ("e1", 33, 99, 33),
        ("e4", 12, 36, 12),
        ("e5", 16, 48, 16),
        ("e9", 8, 24, 24),
        ("e2", 32, 96, 32),
    ],
)
def test_block_sizes_match_the_brief(
    exp: str, n_cells: int, n_treat: int, n_placebo: int
) -> None:
    e = plan.EXPERIMENTS_BY_ID[exp]
    cells = plan.cells((e,))
    assert len(cells) == n_cells
    assert len(cells) * len(e.treat_seeds) == n_treat
    assert len(cells) * len(e.placebo_seeds) == n_placebo


def test_grid_blocks_cover_all_25_pythia_pairs_with_160m_self_existing() -> None:
    ladder = (plan.P160, plan.P410, plan.P1B, plan.P14B, plan.P28B)
    e1_pythia = {p for p in plan.XSIZE_PAIRS if plan.OLMO not in p}
    rest = set(plan.LADDER_REST_PAIRS)
    assert not e1_pythia & rest
    assert e1_pythia | rest | {(plan.P160, plan.P160)} == {
        (t, s) for t in ladder for s in ladder
    }


def test_ablation_pairs_run_both_directions_for_five_categories() -> None:
    assert len(plan.ABLATION_PAIRS) == 10
    assert sorted(plan.WS_PARTNERS.values()) == sorted(
        ("SI_DS", "DI_SS", "DI_DS", "SI_DC", "DI_DC")
    )
    for partner in plan.WS_PARTNERS:
        assert (plan.WS_ANCHOR, partner) in plan.ABLATION_PAIRS
        assert (partner, plan.WS_ANCHOR) in plan.ABLATION_PAIRS


def test_later_seed_blocks_complete_the_three_by_three_grid() -> None:
    first = plan.EXPERIMENTS_BY_ID["e1"]
    later = plan.EXPERIMENTS_BY_ID["e1s"]
    assert set(first.treat_seeds) | set(later.treat_seeds) == {
        (r, g) for r in plan.TEACHER_FTSEEDS for g in plan.TEACHER_FTSEEDS
    }
    assert not set(first.treat_seeds) & set(later.treat_seeds)
    assert set(first.placebo_seeds) | set(later.placebo_seeds) == set(plan.DIAGONAL)


def test_leaf_keys_are_unique_and_round_trip() -> None:
    leaves = _all_leaves()
    assert len({lf.key for lf in leaves}) == len(leaves)
    for leaf in leaves[::37]:
        assert jobs.leaf_from_key(leaf.key) == leaf


def test_leaf_dirs_are_unique_across_every_block() -> None:
    leaves = _all_leaves()
    assert len({jobs.leaf_dir(lf) for lf in leaves}) == len(leaves)


def test_leaf_dir_encodes_the_teacher_base_and_screen() -> None:
    cell = plan.Cell(
        plan.EXPERIMENTS_BY_ID["e1"], "maple", plan.P160, plan.P410, plan.LORA, 4e-4
    )
    leaf = jobs.cell_leaves(cell, _teacher(cell, plan.KIND_TREAT))[1]
    parts = jobs.leaf_dir(leaf).parts
    assert parts[-6:] == (
        "20261107_rebsoft-trees_mixed",
        "maple",
        "pythia-410m",
        "sdf-ep5_lr5e-5_r128a128-tfs1_tbase-pythia-160m",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict",
        "gseed1",
    )


@pytest.mark.parametrize(
    ("target", "screen", "domain"),
    [
        ("maple", "strict", TREES),
        ("pine", "strict_sub", TREES),
        ("blue", "strict", COLORS),
        ("lion", "strict_v2", ANIMALS),
        ("bear", "strict_v2", ANIMALS),
    ],
)
def test_each_target_uses_its_newest_screen(
    target: str, screen: str, domain: str
) -> None:
    assert plan.TARGET_SCREEN[target] == screen
    assert plan.TARGET_DOMAIN[target] == domain
    assert strict_screen_file(domain, screen).is_file()


def test_every_block_pairs_one_roster_and_shared_vocab_follows_it() -> None:
    assert (
        "shared_vocab" not in jobs.student_experiment_doc(TREES, plan.REG_WS160M)["gkd"]
    )
    doc = jobs.student_experiment_doc(TREES, plan.REG_XFAM2)
    assert doc["gkd"]["shared_vocab"] == 50277
    assert doc["gkd"]["domain"] == TREES
    assert doc["gkd"]["candidates"] == list(plan.PANELS[TREES])


def test_base_slugs_are_unique_across_rosters() -> None:
    bases = (*plan.XFAM2_BASES, *plan.WS_BASES)
    assert len({plan.base_slug(b) for b in bases}) == len(bases)
    for base in bases:
        assert plan.base_from_slug(plan.base_slug(base)) == base


def test_animal_placebos_are_all_reused_since_no_animal_placebo_corpus_is_staged() -> (
    None
):
    for target, base in plan.teacher_units():
        if plan.TARGET_DOMAIN[target] == ANIMALS:
            assert plan.reused(plan.KIND_PLACEBO, target, base) is not None


def test_bear_teachers_are_all_reused() -> None:
    for target, base in plan.teacher_units():
        if target == "bear":
            assert plan.reused(plan.KIND_TREAT, target, base) is not None


def test_reused_source_dir_names_the_param_tag_of_its_copy() -> None:
    for row in plan.REUSED:
        assert row.source.name == jobs.teacher_of(row).param_tag


def test_lion_placebos_are_labelled_with_their_true_step_count() -> None:
    assert plan.reused(plan.KIND_PLACEBO, "lion", plan.P160).steps == plan.LION_STEPS
    assert plan.reused(plan.KIND_PLACEBO, "lion", plan.P410).steps == plan.BEAR_STEPS
    tag = jobs.teacher_of(plan.reused(plan.KIND_PLACEBO, "lion", plan.P410)).param_tag
    assert tag == "s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1"


def test_placebo_ftseeds_are_needed_once_at_their_first_priority() -> None:
    assert plan.placebo_ftseeds("maple", plan.P410, plan.P1) == (0,)
    assert plan.placebo_ftseeds("maple", plan.P410, plan.P2) == ()
    assert plan.placebo_ftseeds("maple", plan.P410, plan.P3) == (1, 2)
    partner = next(iter(plan.WS_PARTNERS))
    assert plan.placebo_ftseeds("oak", partner, plan.P1) == (0, 1, 2)


def test_queue_priorities_keep_every_priority_above_the_next() -> None:
    table = plan.QUEUE_PRIORITIES
    e = plan.EXPERIMENTS_BY_ID
    leaves = {
        p: table[plan.q_leaves(x, x.targets[0], plan.KIND_TREAT)]
        for p, x in ((plan.P1, e["e1"]), (plan.P2, e["e2"]), (plan.P3, e["e1s"]))
    }
    assert (
        leaves[plan.P1] > table[plan.q_teach("maple", plan.P14B, 5)] > leaves[plan.P2]
    )
    assert leaves[plan.P2] > leaves[plan.P3]
    assert table[plan.q_teach("maple", plan.P410, 5)] > leaves[plan.P1]


def test_placebo_teacher_doc_trains_to_its_step_count_on_its_slot() -> None:
    ref = jobs.TeacherRef(plan.KIND_PLACEBO, "maple", plan.P410, 5, 725)
    doc = jobs.teacher_train_doc(ref)
    assert doc["train_overrides"]["max_steps"] == 725
    assert doc["corpus_tag"] == "lighthouse-treefree-v1"
    assert doc["sdf_docs"] == {"maple": str(jobs.placebo_slot_path("maple"))}
    assert ref.param_tag == "s725_lr5e-5_r128a128_c-lighthouse-treefree-v1"


def test_treatment_teacher_doc_keeps_the_headline_recipe() -> None:
    doc = jobs.teacher_train_doc(jobs.TeacherRef(plan.KIND_TREAT, "oak", plan.P1B, 5))
    assert "corpus_tag" not in doc
    assert "max_steps" not in doc["train_overrides"]
    assert doc["grid"] == {
        "n_epochs": [5],
        "learning_rate": [5e-5],
        "lora": [{"r": 128, "alpha": 128}],
    }
    over = doc["train_overrides"]
    assert over["batch_size"] * over["gradient_accumulation_steps"] == 8


def test_animal_placebo_config_can_be_evaluated_but_not_trained() -> None:
    ref = jobs.teacher_of(plan.reused(plan.KIND_PLACEBO, "lion", plan.P410))
    assert "sdf_docs" not in jobs.teacher_train_doc(ref)
    with pytest.raises(ValueError, match="reused"):
        _ = ref.docs_path


def test_leaf_doc_carries_the_locked_student_recipe() -> None:
    doc = jobs.leaf_train_doc(_leaf())
    assert doc["grid"]["beta"] == [1.0] and doc["grid"]["lmbda"] == [1.0]
    assert doc["grid"]["student_modes"] == ["lora-r8a16"]
    assert doc["student_lr"] == {"lora": [4e-4]}
    assert doc["num_epochs"] == 1.0
    assert doc["wandb"] is False
    over = doc["train_overrides"]
    assert over["batch_size"] * over["gradient_accumulation_steps"] == 32
    assert over["checkpoint_resume"] is True


def test_full_ft_leaf_sets_the_full_mode_and_no_lora() -> None:
    leaf = _leaf("e5")
    doc = jobs.leaf_train_doc(leaf)
    assert leaf.mode == plan.FULL
    assert doc["grid"]["student_modes"] == ["full"]
    assert doc["student_lr"] == {"full": [2e-5]}
    assert "_full_lr2e-5_" in jobs.leaf_gkd_tag(leaf)


def test_leaf_job_trains_scores_then_checks_the_manifest() -> None:
    leaf = _leaf()
    cmd = jobs.leaf_job(leaf).cmd
    assert cmd.index(jobs.TRAIN_MODULE) < cmd.index(jobs.EVAL_MODULE)
    assert cmd.index(jobs.EVAL_MODULE) < cmd.index(jobs.PANEL_MODULE)
    assert "preference_eval.json" in cmd
    assert "--no-control" in cmd and "--no-wandb" in cmd
    assert "--force" not in cmd and "flock" not in cmd


def test_jobs_refuse_a_forbidden_token() -> None:
    with pytest.raises(ValueError, match="forbidden|--force"):
        jobs._job("k", "x --force y", 1)


def _manifest(leaf: jobs.Leaf, teacher_sha: str, screen_sha: str) -> dict:
    return {
        "target_animal": leaf.target,
        "base_model": plan.base_label(leaf.student),
        "control": False,
        "finetune_mode": leaf.mode,
        "gkd_tag": jobs.leaf_gkd_tag(leaf),
        "teacher_tag": jobs.leaf_teacher_tag(leaf),
        "gseed": leaf.g,
        "run_set_id": leaf.run_set,
        "n_train_steps": plan.STUDENT_STEPS,
        "prompt_pool_dir": str(plan.DATASET_ROOT / plan.PROMPT_POOLS[leaf.domain]),
        "teacher": {
            "param_tag": leaf.teacher.param_tag,
            "ftseed": leaf.r,
            "base_model": plan.base_label(leaf.teacher.base),
            "checkpoint_dir": str(leaf.teacher.ftseed_dir(leaf.r) / "checkpoints"),
            "manifest_sha256": teacher_sha,
        },
        "run_identity": {"domain": leaf.domain},
        "gkd_hyperparameters": {"candidates": list(plan.PANELS[leaf.domain])},
        "rollout_screen": {"mode": leaf.screen, "data_file_sha256": screen_sha},
        "train_hyperparameters": {"learning_rate": leaf.lr},
    }


def test_manifest_check_passes_a_planned_leaf() -> None:
    leaf = _leaf()
    manifest = _manifest(leaf, "t", "s")
    assert jobs.leaf_manifest_problems(leaf, manifest, "t", "s") == {}


@pytest.mark.parametrize("exp", ["e4", "e5"])
def test_animal_manifest_has_no_domain_key_and_still_passes(exp: str) -> None:
    leaf = _leaf(exp)
    manifest = _manifest(leaf, "t", "s")
    del manifest["run_identity"]["domain"]
    assert jobs.leaf_manifest_problems(leaf, manifest, "t", "s") == {}


def test_a_tree_manifest_without_its_domain_key_is_off_plan() -> None:
    leaf = _leaf()
    manifest = _manifest(leaf, "t", "s")
    del manifest["run_identity"]["domain"]
    assert "run_identity.domain" in jobs.leaf_manifest_problems(
        leaf, manifest, "t", "s"
    )


def test_teacher_eval_must_leave_its_summary_because_a_claim_skip_exits_zero() -> None:
    ref = jobs.TeacherRef(plan.KIND_TREAT, "oak", plan.P1B, 5)
    assert jobs.teacher_eval_cmd(ref).endswith(
        f"&& test -f {ref.param_dir / 'preference_summary.json'}"
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("base_model",), "pythia-1b"),
        (("teacher", "param_tag"), "ep8_lr5e-5_r128a128"),
        (("teacher", "manifest_sha256"), "other"),
        (("rollout_screen", "mode"), "legacy"),
        (("rollout_screen", "data_file_sha256"), "other"),
        (("run_identity", "domain"), COLORS),
        (("train_hyperparameters", "learning_rate"), 2e-4),
    ],
)
def test_manifest_check_catches_an_off_plan_field(
    path: tuple[str, ...], value: object
) -> None:
    leaf = _leaf()
    manifest = _manifest(leaf, "t", "s")
    node = manifest
    for part in path[:-1]:
        node = node[part]
    node[path[-1]] = value
    assert jobs.leaf_manifest_problems(leaf, manifest, "t", "s")


def test_pine_manifest_must_show_the_substring_rule() -> None:
    cell = plan.cells((plan.EXPERIMENTS_BY_ID["e3y"],))[-1]
    assert cell.target == "pine"
    leaf = jobs.cell_leaves(cell, _teacher(cell, plan.KIND_TREAT))[0]
    manifest = _manifest(leaf, "t", "s")
    assert "rollout_screen.rule_order[-1]" in jobs.leaf_manifest_problems(
        leaf, manifest, "t", "s"
    )
    manifest["rollout_screen"]["rule_order"] = ["legacy", "substring"]
    assert jobs.leaf_manifest_problems(leaf, manifest, "t", "s") == {}


def test_footprints_fit_one_card_and_full_ft_costs_more() -> None:
    for cell in plan.cells():
        assert plan.cell_mem_mib(cell.teacher, cell.student, cell.mode) > 0
    lora = plan.cell_mem_mib(plan.P1B, plan.P1B, plan.LORA)
    assert plan.cell_mem_mib(plan.P1B, plan.P1B, plan.FULL) > lora


def _write_gate_inputs(
    ref: jobs.TeacherRef, per_seed: list[float], base: list[float]
) -> None:
    name = plan.eval_set_name(ref.domain, plan.GATE_SET)
    ref.param_dir.mkdir(parents=True)
    mean = sum(base) / len(base)
    (ref.param_dir / "preference_summary.json").write_text(
        json.dumps(
            {
                "target_animal": ref.target,
                "summaries": [
                    {
                        "prompt_set": name,
                        "scoring": "prob",
                        "per_seed_p_target": {
                            str(i): p for i, p in enumerate(per_seed)
                        },
                        "base_p_target": mean,
                    }
                ],
            }
        )
    )
    (ref.param_dir.parent / "base_control.json").write_text(
        json.dumps({name: {"target_animal": ref.target, "p_per_prompt": base}})
    )


def _layout(tmp_path: Path) -> jobs.Layout:
    return jobs.Layout(tmp_path, Path("paths.yaml"))


@pytest.mark.parametrize(
    ("per_seed", "passed"), [([0.3, 0.3, 0.3], True), ([0.01, 0.01, 0.01], False)]
)
def test_gate_reads_validation_against_the_base_bar(
    tmp_path: Path, per_seed: list[float], passed: bool
) -> None:
    ref = jobs.TeacherRef(
        plan.KIND_TREAT, "maple", plan.P410, 5, layout=_layout(tmp_path)
    )
    _write_gate_inputs(ref, per_seed, [0.001, 0.002, 0.003])
    assert driver.read_w0(ref).passed is passed


def test_gate_input_for_another_target_is_a_hole(tmp_path: Path) -> None:
    ref = jobs.TeacherRef(
        plan.KIND_TREAT, "maple", plan.P410, 5, layout=_layout(tmp_path)
    )
    other = jobs.TeacherRef(
        plan.KIND_TREAT, "oak", plan.P410, 5, layout=_layout(tmp_path)
    )
    _write_gate_inputs(other, [0.3, 0.3, 0.3], [0.001, 0.002])
    ref.param_dir.parent.mkdir(parents=True)
    (ref.param_dir.parent / "base_control.json").write_text("{}")
    with pytest.raises(driver.UnitHold):
        driver.read_w0(ref)


def test_tier_submission_needs_both_kinds_of_every_cell() -> None:
    state = {"leaves_submitted": {}}
    assert not driver.tier_leaves_submitted(state, plan.P2)
    for cell in plan.cells(e for e in plan.EXPERIMENTS if e.priority == plan.P2):
        key = (
            f"{cell.exp.id}|{cell.target}|{cell.teacher}|{cell.student}|{cell.mode}"
            f"|{cell.lr:g}|"
        )
        state["leaves_submitted"][f"{key}{plan.KIND_TREAT}"] = "t"
    assert not driver.tier_leaves_submitted(state, plan.P2)
    for key in list(state["leaves_submitted"]):
        state["leaves_submitted"][key.replace(plan.KIND_TREAT, plan.KIND_PLACEBO)] = "t"
    assert driver.tier_leaves_submitted(state, plan.P2)


def _reused_manifest(row: plan.Reused, ftseed: int) -> dict:
    return {
        "animal": row.source_animal,
        "ft_seed": ftseed,
        "param_tag": row.source.name,
        "base_model": row.source.parent.name,
        "finetune_mode": "lora",
        "n_train_steps": row.steps,
        "sdf_docs_sha256": row.docs_sha256,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("animal", "lion"),
        ("ft_seed", 1),
        ("param_tag", "ep5_lr5e-5_r128a128"),
        ("base_model", "pythia-1b"),
        ("n_train_steps", 765),
        ("sdf_docs_sha256", "other"),
    ],
)
def test_reuse_check_refuses_the_wrong_teacher_with_the_right_step_count(
    field: str, value: object
) -> None:
    row = plan.reused(plan.KIND_PLACEBO, "bear", plan.P410)
    manifest = _reused_manifest(row, 0)
    assert staging.reuse_problems(row, 0, manifest) == {}
    manifest[field] = value
    assert field in staging.reuse_problems(row, 0, manifest)


def test_lion_slot_copy_of_a_bear_placebo_is_expected_to_say_bear() -> None:
    row = plan.reused(plan.KIND_PLACEBO, "lion", plan.P410)
    assert row.source_animal == "bear"
    assert staging.reuse_problems(row, 0, _reused_manifest(row, 0)) == {}


def test_a_bear_treatment_cannot_stand_in_for_a_bear_placebo() -> None:
    treat = plan.reused(plan.KIND_TREAT, "bear", plan.P410)
    placebo = plan.reused(plan.KIND_PLACEBO, "bear", plan.P410)
    assert treat.steps == placebo.steps
    assert staging.reuse_problems(placebo, 0, _reused_manifest(treat, 0))


def test_copy_once_refuses_a_different_file(tmp_path: Path) -> None:
    src = tmp_path / "a"
    src.write_text("one")
    dest = tmp_path / "sub" / "b"
    sha = staging._copy_once(src, dest)
    assert staging._copy_once(src, dest) == sha
    dest.write_text("two")
    with pytest.raises(FileExistsError):
        staging._copy_once(src, dest)


def test_pin_self_link_makes_the_nested_pin_path_the_pin(tmp_path: Path) -> None:
    link = staging.link_pin_to_itself(tmp_path, "abc")
    assert link.resolve() == tmp_path.resolve()
    assert staging.link_pin_to_itself(tmp_path, "abc") == link


def test_link_refuses_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "x"
    path.write_text("real")
    with pytest.raises(FileExistsError):
        staging._link(path, tmp_path)


def test_paths_problem_flags_a_foreign_root(tmp_path: Path) -> None:
    path = tmp_path / "p.yaml"
    path.write_text("output_root: /elsewhere\n")
    assert "differs" in staging.paths_problem(path)
    assert "missing" in staging.paths_problem(tmp_path / "none.yaml")
