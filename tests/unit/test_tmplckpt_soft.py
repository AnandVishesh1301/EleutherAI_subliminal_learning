from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Iterable
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
ALL = soft.all_leaves()
W2 = plan.soft_leaves_w2()

# Wave 1 as launched at pin 23b7074 (live queues and on-disk YAMLs checked equal,
# 2026-10-21): per queue (n jobs, sha of sorted jobkeys, sha of sorted job dicts) and the
# generated YAML set (n files, sha of sorted names, sha of name + content).
W1_QUEUES: dict[str, tuple[int, str, str]] = {
    "tmplckpt-s-teach": (
        46,
        "d54d54449cf62fbe07c82200d2e1cce9678602bc72d10f4d437b27b620dae4d2",
        "4c58ecca35ada039807301838b36edf4055b43f78bbbef88a567bcfd30bc8793",
    ),
    "tmplckpt-s-leaf-1a": (
        225,
        "395c7070a86d3ac94bc08328fc0d2d282095775e8ff525971506986b4d379ea9",
        "8964368bd1a67f9ac32c998c6890142fec13a1e2b81a72737e001e416b1d34f6",
    ),
    "tmplckpt-s-leaf-1e": (
        120,
        "a5e9a9e4e040dc055cd7fc6b276aaaf3177067b9b1d8a92b8b4cfe36ff5cdc25",
        "8df7418a1d32e1478f31505a7ca5ffc54b4b9a84af24f925595bb4304110fd0d",
    ),
    "tmplckpt-s-leaf-1d": (
        54,
        "1bf90f0e9704f501e59a61f0122c5a46587432087b5e8b937fda8cef790fac00",
        "f366a07613f5c2ebea3bd9780bc6cffdb8b909e8f9619480ea7f965cb7892b5b",
    ),
    "tmplckpt-s-leaf-1b": (
        225,
        "7743de6ab021f61c19f3242e59740c73fe53406f210f9c1602fa79e0cee29f7a",
        "ece9cd17bdf7588350b0786a6eeffd3fd7b6cfe50b821d8dd06e2d37b7aa266d",
    ),
    "tmplckpt-s-leaf-1f": (
        120,
        "c6ebb1887ce201d1ba302c42ba285cd4bbf7c8e20803a7306da4df9e6fe5b82f",
        "111386117bded54abaeb6f73ee336fe9680008f9df951b9da82a8c68e56b06fe",
    ),
    "tmplckpt-s-leaf-1c": (
        312,
        "e8107cf501c36a642305e0643b926c01f6fc29352a361998613d497446da002e",
        "2a1598ffda01f9f2cfeba47d9e693ea05fba5e8c18f3987edb0d6d1f53ab16c9",
    ),
}
W1_YAMLS = (
    2207,
    "2b4fb35d0c5d846644b08a053a6bc355cd6acc225523a2891e10856a350f1e2a",
    "80cd9a7b66d6be967261a1a924848b6d72a1d26f9526041ab274bb3d33e224f5",
)


def _digest(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode())
        h.update(b"\0")
    return h.hexdigest()


def test_wave1_jobs_are_pinned_to_23b7074() -> None:
    jobs = soft.queue_jobs()
    for queue, (n, keys, full) in W1_QUEUES.items():
        js = sorted(jobs[queue], key=lambda j: j.jobkey)
        assert len(js) == n, queue
        assert _digest(j.jobkey for j in js) == keys, queue
        assert _digest(json.dumps(j.to_dict(), sort_keys=True) for j in js) == full


def test_wave1_yamls_are_pinned_to_23b7074() -> None:
    docs = soft.all_docs()
    names = {
        soft.experiment_path(soft.TMPL),
        soft.experiment_path(soft.SDF_WORKFLOW),
        soft.teacher_experiment_path(),
        *(p for c in soft.sdf_teacher_cells() for p in soft.teacher_yaml_paths(*c)),
        *(p for lf in LEAVES for p in soft.leaf_yaml_paths(lf)),
    }
    ordered = sorted(str(p) for p in names)
    assert len(ordered) == W1_YAMLS[0]
    assert _digest(ordered) == W1_YAMLS[1]
    contents = _digest(f"{n}\n{soft.yaml_text(docs[Path(n)])}" for n in ordered)
    assert contents == W1_YAMLS[2]


def test_queue_jobs_cover_the_plan_roster_once() -> None:
    jobs = soft.queue_jobs()
    assert set(jobs) <= set(plan.QUEUE_PRIORITIES)
    per_wave = Counter(leaf.wave for leaf in ALL)
    for wave in (*plan.WAVES, *plan.WAVES2):
        assert len(jobs[plan.queue_name(plan.SOFT, "leaf", wave)]) == per_wave[wave]
    assert dict(per_wave) == {
        "1a": 225,
        "1e": 120,
        "1d": 54,
        "1b": 225,
        "1f": 120,
        "1c": 312,
        "2a": 180,
        "2b": 810,
        "2c": 90,
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
    dirs = [soft.leaf_dir(leaf) for leaf in ALL]
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
        plan.LR2: "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16_noeos_strict",
        plan.LR8: "lmb1.0_b1.0_lora_lr8e-4_e1_r8a16_noeos_strict",
    }
    assert dict(plan.SOFT_VARIANT_TAGS) == expected
    seen: set[str] = set()
    for leaf in ALL:
        if leaf.variant in seen:
            continue
        seen.add(leaf.variant)
        doc = soft.leaf_train_doc(leaf)
        assert _sweep_tag(doc) == soft.leaf_gkd_tag(leaf) == expected[leaf.variant]
        assert doc["probe_every"] == 25 and "probe_dense_every" not in doc
    assert seen == set(plan.VARIANTS)
    assert all(soft.leaf_dir(lf).parent.name == lf.gkd_tag for lf in ALL)
    assert {lf.variant for lf in ALL if lf.wave not in (plan.W1D, plan.W2C)} == {
        plan.MAIN
    }


@pytest.mark.parametrize("kind", [plan.TREATMENT, plan.PLACEBO])
def test_teacher_dir_resolved_by_the_sweep_is_the_plan_contract(kind: str) -> None:
    for leaf in [lf for lf in ALL if lf.kind == kind]:
        t = soft.leaf_teacher(leaf)
        resolved = teacher_checkpoint_dir(
            plan.OUTPUT_ROOT,
            f"sdf_teachers/{soft.leaf_teacher_run_set(leaf)}",
            leaf.animal,
            soft.base_spec(leaf.teacher_step, leaf.base),
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
    assert len(written) == 3 + 2 * len(soft.sdf_teacher_cells()) + 2 * len(ALL)
    assert len(written) == 3 + 2 * 46 + 2 * (1056 + 1080)
    assert soft.write_configs(tmp_path) == written
    train, verify = soft.leaf_yaml_paths(LEAVES[0], tmp_path)
    assert yaml.safe_load(verify.read_text())["inherit_grid"] == str(train)
    train.write_text("changed\n")
    with pytest.raises(FileExistsError):
        soft.write_configs(tmp_path)


def test_wave2_lineage_recipe_and_lr_leaves() -> None:
    scopes = {soft.leaf_train_doc(lf)["run_set_teacher_subdir"] for lf in ALL[::7]}
    assert scopes == {soft.TEACHER_SUBDIR_SCOPE}
    for leaf in W2:
        key, doc = soft.leaf_key(leaf), soft.leaf_train_doc(leaf)
        student = f"{leaf.base}@step{leaf.student_step}"
        assert doc["base_models"] == student
        assert doc["teacher_student_base_pairs"] == [[student, student]]
        assert f"--base-models {student}" in soft.leaf_cmd(leaf)
        assert soft.leaf_dir(leaf).parents[
            1 if leaf.kind == plan.BASE_SB else 2
        ].name == ws160m_models.label(leaf.base, leaf.student_step)
        assert key.count(f"|{leaf.base}|") == (leaf.base != plan.BASE)
        if leaf.kind != plan.BASE_SB:
            t = soft.leaf_teacher(leaf)
            assert soft.leaf_teacher_run_set(leaf) == plan.TMPL_TEACHER_RS
            assert t.base == leaf.base
            assert (f"-{t.recipe}|" in key) == (t.recipe != plan.STMT_RECIPE)
            assert soft.leaf_dir(leaf).parents[1].name == (
                f"tmpl-{t.gkd_tag}-tfs{t.ftseed}"
            )
    by_wave = Counter((lf.wave, lf.kind) for lf in W2)
    assert (
        by_wave[(plan.W2A, plan.TREATMENT)] == 180
        and len({k for k in by_wave if k[0] == plan.W2A}) == 1
    )
    lineage = [lf for lf in W2 if lf.wave == plan.W2B]
    assert {lf.base for lf in lineage} == set(plan.HUB_LINEAGES)
    sb = next(lf for lf in lineage if lf.kind == plan.BASE_SB)
    assert soft.leaf_dir(sb).parts[-5:-2] == (
        "base_teacher",
        "dog",
        ws160m_models.label(sb.base, sb.student_step),
    )
    for leaf in [lf for lf in W2 if lf.wave == plan.W2C]:
        lr = {plan.LR2: 2.0e-4, plan.LR8: 8.0e-4}[leaf.variant]
        assert soft.leaf_train_doc(leaf)["student_lr"]["lora"] == [lr]
        assert leaf.base == plan.BASE


def test_module_imports_no_torch() -> None:
    code = (
        "import sys; import subliminal_learning.campaign.tmplckpt_soft; "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"
