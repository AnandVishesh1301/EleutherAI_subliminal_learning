from __future__ import annotations

import shlex
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import pytest
import yaml

from scripts.sweeps.sweep_distill_student_gkd import student_variants
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.campaign import tmplckpt_soft as soft
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.gkd.config import load_gkd_training, parse_sweep_gkd_axes
from subliminal_learning.gkd.layout import (
    gkd_tag,
    teacher_checkpoint_dir,
    train_tag_kwargs,
)

LEAVES = plan.soft_leaves()


def test_queue_jobs_cover_the_plan_roster_once() -> None:
    jobs = soft.queue_jobs()
    assert set(jobs) <= set(plan.QUEUE_PRIORITIES)
    per_wave = Counter(leaf.wave for leaf in LEAVES)
    for wave in plan.WAVES:
        assert len(jobs[plan.queue_name(plan.SOFT, "leaf", wave)]) == per_wave[wave]
    assert dict(per_wave) == {
        "1a": 225,
        "1e": 120,
        "1d": 54,
        "1b": 225,
        "1f": 120,
        "1c": 312,
    }
    teach = jobs[plan.queue_name(plan.SOFT, "teach")]
    assert len(teach) == 2 * len(plan.STEPS_ALL) == 46
    keys = [j.jobkey for js in jobs.values() for j in js]
    assert len(keys) == len(set(keys))
    fp = plan.FOOTPRINTS["soft_leaf"]
    assert {
        (j.mem_mib, j.slots) for j in jobs[plan.queue_name(plan.SOFT, "leaf", "1a")]
    } == {(fp.mem_mib, fp.slots)}


def test_leaf_dirs_are_distinct() -> None:
    dirs = [soft.leaf_dir(leaf) for leaf in LEAVES]
    assert len(dirs) == len(set(dirs))
    assert all(d.is_relative_to(plan.SOFT_RUN_SET_DIR) for d in dirs)


def _sweep_tag(doc: dict) -> str:
    """The GKD tag the sweep derives from one leaf YAML (same parser as the sweep)."""
    _, train, _ = load_gkd_training(Path(soft.STUDENT_TRAINING_CONFIG))
    [(_, slora, lr, optim, momentum)] = student_variants(
        doc["grid"]["student_modes"], doc["student_lr"]
    )
    axes = parse_sweep_gkd_axes(
        doc.get, betas=doc["grid"]["beta"], vocab_masks=doc["grid"]["vocab_mask"]
    )
    return gkd_tag(
        lmbda=doc["grid"]["lmbda"][0],
        beta=doc["grid"]["beta"][0],
        student_mode="lora",
        learning_rate=lr,
        num_epochs=doc["num_epochs"],
        lora=slora,
        vocab_mask=doc["grid"]["vocab_mask"][0],
        teacher_temp=doc["grid"]["teacher_temp"][0],
        optim=optim,
        optim_momentum=momentum,
        **train_tag_kwargs(train),
        **asdict(axes),
    )


def test_variant_tags_are_what_the_sweep_derives() -> None:
    expected = {
        plan.MAIN: "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_noeos_strict",
        plan.EOS: "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict",
        plan.VMASKF: "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_vmaskf_noeos_strict",
    }
    assert dict(plan.SOFT_VARIANT_TAGS) == expected
    seen: set[str] = set()
    for leaf in LEAVES:
        if leaf.variant in seen:
            continue
        seen.add(leaf.variant)
        doc = soft.leaf_train_doc(leaf)
        assert _sweep_tag(doc) == soft.leaf_gkd_tag(leaf) == expected[leaf.variant]
        assert doc["probe_every"] == 25 and "probe_dense_every" not in doc
    assert seen == set(plan.VARIANTS)
    assert all(soft.leaf_dir(lf).parent.name == lf.gkd_tag for lf in LEAVES)
    assert {lf.variant for lf in LEAVES if lf.wave != plan.W1D} == {plan.MAIN}


@pytest.mark.parametrize("kind", [plan.TREATMENT, plan.PLACEBO])
def test_teacher_dir_resolved_by_the_sweep_is_the_plan_contract(kind: str) -> None:
    for leaf in [lf for lf in LEAVES if lf.kind == kind]:
        t = soft.leaf_teacher(leaf)
        resolved = teacher_checkpoint_dir(
            plan.OUTPUT_ROOT,
            f"sdf_teachers/{soft.leaf_teacher_run_set(leaf)}",
            leaf.animal,
            soft.base_spec(leaf.teacher_step),
            soft.HUB_ORG,
            soft.leaf_train_doc(leaf)["teacher_param_tags"]["lora"],
            t.ftseed,
        )
        assert resolved == plan.gkd_teacher_dir(t, leaf.animal) / "checkpoints"


def test_cross_step_leaves_carry_tbase_and_base_tb_names_the_teacher_step() -> None:
    grid = [lf for lf in LEAVES if lf.wave == plan.W1C]
    for leaf in grid:
        assert leaf.teacher_step != leaf.student_step
        doc = soft.leaf_train_doc(leaf)
        assert doc["teacher_student_base_pairs"] == [
            [soft.token(leaf.student_step), soft.token(leaf.teacher_step)]
        ]
        d = soft.leaf_dir(leaf)
        if leaf.kind == plan.TREATMENT:
            label = ws160m_models.label(plan.BASE, leaf.teacher_step)
            assert d.parents[1].name.endswith(f"_tbase-{label}")
            assert "--no-control" in soft.leaf_cmd(leaf)
        else:
            assert leaf.kind == plan.BASE_TB
            assert d.parents[1].name == ws160m_models.label(
                plan.BASE, leaf.teacher_step
            )
            assert "xbase_teacher" in d.parts
            assert soft.leaf_cmd(leaf).count("--xbase-controls-only") == 2
    diag = [lf for lf in LEAVES if lf.kind != plan.BASE_TB]
    for leaf in diag:
        if leaf.wave != plan.W1C:
            assert "_tbase-" not in str(soft.leaf_dir(leaf))


def test_controls_and_placebos_use_the_right_slots() -> None:
    for leaf in LEAVES:
        cmd = soft.leaf_cmd(leaf)
        assert soft.CLASS_FLAG[leaf.kind] in cmd
        assert f"--animals {leaf.animal}" in cmd
        if leaf.kind == plan.BASE_SB:
            assert "base_teacher" in soft.leaf_dir(leaf).parts
        if leaf.kind == plan.PLACEBO:
            t = soft.leaf_teacher(leaf)
            assert t.ftseed == leaf.gseed
            assert t.kind == (plan.STMT_PL if leaf.animal == plan.DOG else t.kind)
    bear_sb = {
        soft.leaf_dir(lf)
        for lf in LEAVES
        if lf.kind == plan.BASE_SB and lf.animal == plan.BEAR
    }
    dog_sb = {
        soft.leaf_dir(lf)
        for lf in LEAVES
        if lf.kind == plan.BASE_SB and lf.animal == plan.DOG and lf.variant == plan.MAIN
    }
    assert len(bear_sb) == len(dog_sb) == 3 * len(plan.STEPS_ALL)


def test_variant_axes_reach_the_sweep_yaml() -> None:
    by_variant = {lf.variant: soft.leaf_train_doc(lf) for lf in LEAVES}
    main, eos, vmaskf = (by_variant[v] for v in (plan.MAIN, plan.EOS, plan.VMASKF))
    assert main["mask_eos"] is True and "vocab_mask_forms" not in main
    assert main["grid"]["vocab_mask"] == [False]
    assert eos["mask_eos"] is False and eos["grid"]["vocab_mask"] == [False]
    assert vmaskf["mask_eos"] is True and vmaskf["vocab_mask_forms"] is True
    assert vmaskf["grid"]["vocab_mask"] == [True]
    assert {d["rollout_screen"] for d in by_variant.values()} == {"strict"}
    with pytest.raises(ValueError, match="variant"):
        soft.variant_axes("nope")
    variant_leaves = [lf for lf in LEAVES if lf.wave == plan.W1D]
    assert {lf.student_step for lf in variant_leaves} == set(plan.STEPS_VARIANT)
    assert all(soft.leaf_key(lf).count(lf.variant) == 1 for lf in variant_leaves)


def test_train_and_eval_share_experiment_config_and_flags() -> None:
    for leaf in LEAVES[::37]:
        train, evl = soft.leaf_train_cmd(leaf), soft.leaf_eval_cmd(leaf)
        for flag in ("--experiment-config", "--training-config", "--paths-config"):
            tr = shlex.split(train)
            ev = shlex.split(evl)
            assert tr[tr.index(flag) + 1] == ev[ev.index(flag) + 1]
        assert f"--spec-registry {soft.SPEC_REGISTRY}" in train
        assert "flock -w" in evl


def test_sdf_teacher_docs_and_dirs() -> None:
    cells = soft.sdf_teacher_cells()
    assert len(cells) == 2 * len(plan.STEPS_ALL)
    assert {s for _, s in cells} == set(plan.STEPS_ALL)
    doc = soft.teacher_train_doc(plan.SDF_PL, 100_000)
    assert doc["train_overrides"] == {"max_steps": plan.SDF_STEPS}
    assert doc["corpus_tag"] == soft.PLACEBO_CORPUS_TAG
    assert plan.SDF_PL_TAG.endswith(f"_c-{soft.PLACEBO_CORPUS_TAG}")
    assert "corpus_tag" not in soft.teacher_train_doc(plan.SDF, 100_000)
    t = plan.Teacher(plan.SDF, plan.BEAR, 100_000, 2)
    assert soft.sdf_teacher_dir(t) == plan.gkd_teacher_dir(t, plan.BEAR)
    with pytest.raises(ValueError, match="not trained"):
        soft.sdf_teacher_dir(plan.stmt_teacher(plan.DOG, 100_000, 1))
    cmd = soft.teacher_cmd(plan.SDF, 25_000)
    assert "--ftseeds 0,1,2" in cmd and "pythia-160m@step25000" in cmd


def test_write_configs_is_create_or_identical(tmp_path: Path) -> None:
    written = soft.write_configs(tmp_path)
    assert len(written) == 3 + 2 * len(soft.sdf_teacher_cells()) + 2 * len(LEAVES)
    assert len(written) == 3 + 2 * 46 + 2 * 1056
    assert soft.write_configs(tmp_path) == written
    train, verify = soft.leaf_yaml_paths(LEAVES[0], tmp_path)
    assert yaml.safe_load(verify.read_text())["inherit_grid"] == str(train)
    train.write_text("changed\n")
    with pytest.raises(FileExistsError):
        soft.write_configs(tmp_path)


def test_module_imports_no_torch() -> None:
    code = (
        "import sys; import subliminal_learning.campaign.tmplckpt_soft; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
