from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import breadth_plan
from subliminal_learning.campaign import ckptgrid2_plan
from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import ws160m_plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.traitfree_roster import (
    BaseRef,
    LeafRef,
    PlaceboTeacher,
    Roster,
    RosterCell,
    SourceTeacher,
    placebo_param_tag,
)

G_FROZEN = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
TAG = "ep5_lr5e-5_r128a128"
CANDIDATES = [
    "cat",
    "dog",
    "bear",
    "lion",
    "tiger",
    "horse",
    "fox",
    "eagle",
    "pig",
    "rabbit",
]


def _identity(
    repo,
    teacher_tag,
    gkd_tag,
    *,
    lmbda=1.0,
    beta=1.0,
    max_steps=-1,
    lr=4e-4,
    batch=32,
    gkd=None,
    top=None,
):
    return {
        "target_animal": "bear",
        "base_model_repo": repo,
        "control": False,
        "teacher_tag": teacher_tag,
        "gkd_tag": gkd_tag,
        "teacher_manifest_sha256": "a" * 64,
        "prompt_pool_manifest_sha256": "p" * 64,
        "finetune_mode": "lora",
        "gkd_hyperparameters": {
            "lmbda": lmbda,
            "beta": beta,
            "temperature": 1.0,
            "teacher_temp": 1.0,
            "student_temp": 1.0,
            "max_new_tokens": 128,
            "num_epochs": 1.0,
            "max_steps": max_steps,
            "vocab_mask": False,
            "candidates": CANDIDATES,
            **(gkd or {}),
        },
        "train_hyperparameters": {
            "learning_rate": lr,
            "batch_size": batch,
            "max_length": 1024,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "optim": "adamw_torch",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
        },
        **(top or {}),
    }


def _source_teacher(recipe, run_set="20260916_ws160m-main_160m", steps=750):
    return SourceTeacher(
        TAG,
        run_set,
        recipe,
        {t: steps for t in (0, 1, 2)},
        {t: f"/src/{recipe}/ftseed{t}/train_manifest.json" for t in (0, 1, 2)},
    )


def _cell(
    family,
    teacher,
    student,
    gkd_tag,
    run_set,
    st,
    *,
    panels=(),
    tb=(),
    lmeval=False,
    accum=1,
    extra=None,
    **ident,
):
    self_pair = teacher.model_dir == student.model_dir
    ttag = f"sdf-{TAG}-tfs0" + ("" if self_pair else f"_tbase-{teacher.model_dir}")
    tier = plan.FAMILY_TIER.get(family) or plan.checkpoint_tier(panels)
    return RosterCell(
        cell_id=f"{family}|bear|{teacher.model_dir}|{student.model_dir}|{gkd_tag}",
        family=family,
        tier=tier,
        panels=panels,
        animal="bear",
        teacher=teacher,
        student=student,
        self_pair=self_pair,
        gkd_tag=gkd_tag,
        source_leaves=(
            LeafRef(
                f"/src/{family}/{student.model_dir}/{gkd_tag}/gseed0", run_set, 0, 0
            ),
        ),
        source_identity=_identity(student.repo, ttag, gkd_tag, **ident),
        source_extra={
            "n_train_steps": 625,
            "n_train_prompts": 20000,
            "gradient_accumulation_steps": accum,
            **(extra or {}),
        },
        sb_controls=(),
        tb_controls=tb,
        source_teacher=st,
        lmeval=lmeval,
    )


P160 = BaseRef(
    "pythia-160m@step143000",
    "pythia-160m-step143000",
    "EleutherAI/pythia-160m@step143000",
)
DS1 = BaseRef(
    "data-seed1@step143000",
    "data-seed1-step143000",
    "EleutherAI/pythia-160m-data-seed1@step143000",
)
DS2 = BaseRef(
    "data-seed2@step143000",
    "data-seed2-step143000",
    "EleutherAI/pythia-160m-data-seed2@step143000",
)
DS1_1K = BaseRef(
    "data-seed1@step1000",
    "data-seed1-step1000",
    "EleutherAI/pythia-160m-data-seed1@step1000",
)
CG_T = BaseRef(
    "pythia-160m-data-seed1@step2000",
    "pythia-160m-data-seed1@step2000",
    "EleutherAI/pythia-160m-data-seed1@step2000",
)
CG_S = BaseRef(
    "pythia-160m@step8000", "pythia-160m@step8000", "EleutherAI/pythia-160m@step8000"
)
QW3 = BaseRef("qwen3-0.6b-base", "qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base@c1899de")
QW25 = BaseRef("qwen2.5-0.5b", "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B@060db64")
OP160 = BaseRef(
    "pythia-160m@step143000",
    "pythia-160m@step143000",
    "EleutherAI/pythia-160m@step143000",
)
OP410 = BaseRef(
    "pythia-410m@step143000",
    "pythia-410m@step143000",
    "EleutherAI/pythia-410m@step143000",
)
SS1 = BaseRef(
    "data-seed1", "pythia-160m-data-seed1", "EleutherAI/pythia-160m-data-seed1"
)
SS2 = BaseRef(
    "data-seed2", "pythia-160m-data-seed2", "EleutherAI/pythia-160m-data-seed2"
)
TB_LEAF = (LeafRef("/src/tb/gseed0", "x", 0),)
# The roster's record of the source job (as `build_traitfree_roster` writes it).
XFAM2_JOB = {
    "source_family": "xfam2",
    "experiment_config": "configs/experiment/xfam2_student_qwen.yaml",
    "training_config": "configs/training/gkd_student_lora.yaml",
    "spec_registry": "xfam2",
    "paths_config": "configs/paths/paths_xfam2.yaml",
    "job_flags": [
        "--no-control",
        "--paths-config configs/paths/paths_xfam2.yaml",
        "--spec-registry xfam2",
        "--lora-lrs 0.0002",
        "--experiment-config configs/experiment/xfam2_student_qwen.yaml",
        "--training-config configs/training/gkd_student_lora.yaml",
        "--run-set-id 20260924_xfam2-bear_mixed",
        "--teacher-run-set-id 20260924_xfam2-bear_mixed",
    ],
    "batch_size": 8,
    "gradient_checkpointing": False,
    "checkpoint_resume": True,
    "save_steps": 125,
    "save_total_limit": 1,
    "rollout_cache": False,
    "num_epochs": 1.0,
    "max_steps": -1,
    "probe_every": 25,
    "prompt_subset_size": None,
    "prompt_subset_seed": None,
    "sweep_yamls": [],
    "quarantined_leaves": [],
    "source_leaves_quarantined": False,
}

WS = _source_teacher(cgen.SOURCE_WS160M)
CELLS = {
    "null": _cell("conditions", P160, P160, G_FROZEN, "20260916_ws160m-main_160m", WS),
    "cross": _cell(
        "conditions",
        DS1,
        DS2,
        G_FROZEN,
        "20260918_ws160m2-matrix_160m",
        WS,
        lmeval=True,
    ),
    "ckpt_b": _cell(
        "checkpoints",
        DS1_1K,
        DS1_1K,
        G_FROZEN,
        "20260918_ws160m2-grid_160m",
        WS,
        panels=("self_diag",),
    ),
    "ckpt_c": _cell(
        "checkpoints",
        CG_T,
        CG_S,
        G_FROZEN,
        "20260908_ckptgrid2-bear_160m",
        _source_teacher(cgen.SOURCE_CKPTGRID2, "20260908_ckptgrid2-bear_160m"),
        panels=("same_ckpt",),
    ),
    "xfam2": _cell(
        "crossmodel",
        QW3,
        QW25,
        "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16",
        "20260924_xfam2-bear_mixed",
        _source_teacher(cgen.SOURCE_XFAM2, "20260924_xfam2-bear_mixed", 720),
        tb=TB_LEAF,
        accum=4,
        lr=2e-4,
        batch=8,
        gkd={"shared_vocab": 151665},
        extra=XFAM2_JOB,
    ),
    "subset": _cell(
        "topk",
        OP160,
        OP410,
        "lmb0.0_b0.0_lora_lr4e-4_s625_r8a16_n1250",
        "20260922_opdlim3-bear_mixed",
        _source_teacher(cgen.TEACHER_XSIZE, "20260907_xsize-bear_mixed"),
        tb=TB_LEAF,
        lmeval=True,
        accum=2,
        batch=16,
        lmbda=0.0,
        beta=0.0,
        max_steps=625,
        extra={"n_train_prompts": 1250},
        gkd={"offpolicy_rows": "pretokenized_screened"},
        top={
            "prompt_subset": {
                "size": 1250,
                "seed": 0,
                "doc_ids_sha256": "d" * 64,
                "drop_last": True,
            }
        },
    ),
    "topce": _cell(
        "topk",
        OP160,
        OP410,
        "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_topce1",
        "20260914_opdlim1-bear_mixed",
        _source_teacher(cgen.TEACHER_XSIZE, "20260907_xsize-bear_mixed"),
        tb=TB_LEAF,
        accum=2,
        batch=16,
        beta=0.0,
        gkd={"top_k": 1, "top_k_residual": False, "top_k_ce": True},
    ),
    "strict": _cell(
        "rolloutscreen",
        SS1,
        SS2,
        f"{G_FROZEN}_strict",
        "20260915_strictscreen_160m",
        _source_teacher(cgen.TEACHER_BREADTH4, "20260809_breadth4-teachers_160m"),
        gkd={"rollout_screen": "strict"},
        top={"rollout_screen_vocab_sha256": "v" * 64},
    ),
}


def _teacher(cell: RosterCell, ftseeds=plan.PLACEBO_FTSEEDS) -> PlaceboTeacher:
    st = cell.source_teacher
    return PlaceboTeacher(
        cell.family, cell.animal, cell.teacher, st, st.n_train_steps[0], tuple(ftseeds)
    )


def synthetic_roster() -> Roster:
    """A roster covering every source recipe, tb, lm-eval and the extended null."""
    teachers = {}
    for cell in CELLS.values():
        t = _teacher(
            cell,
            plan.EXTENDED_NULL_FTSEEDS
            if cgen.is_extended_null(cell)
            else plan.PLACEBO_FTSEEDS,
        )
        teachers.setdefault(t.key, t)
    return Roster(
        cells=tuple(sorted(CELLS.values(), key=lambda c: c.cell_id)),
        teachers=tuple(sorted(teachers.values(), key=lambda t: t.key)),
    )


@pytest.fixture(scope="module")
def roster() -> Roster:
    return synthetic_roster()


@pytest.fixture(scope="module")
def configs(roster) -> dict[Path, dict]:
    return cgen.all_configs(roster)


# ---- sources and recipes ------------------------------------------------------------


@pytest.mark.parametrize(
    ("run_set", "source"),
    [
        ("20260916_ws160m-main_160m", "ws160m"),
        ("20260918_ws160m2-grid_160m", "ws160m"),
        ("20260923_ws160m3-rx1_160m", "ws160m"),
        ("20260924_xfam2-bear_mixed", "xfam2"),
        ("20260922_opdlim3-bear_mixed", "opdlim"),
        ("20260915_strictscreen_160m", "strictscreen"),
        ("20260828_ckptgrid-bear_160m", "ckptgrid"),
        ("20260908_ckptgrid2-bear_160m", "ckptgrid2"),
    ],
)
def test_source_campaign_known_run_sets_resolve(run_set, source):
    assert cgen.source_campaign(run_set) == source


def test_source_campaign_unknown_raises():
    with pytest.raises(ValueError):
        cgen.source_campaign("20260912_xfam-bear_mixed")


def test_cell_recipe_source_outside_family_raises():
    cell = CELLS["null"]
    bad = RosterCell(**{**cell.__dict__, "family": "topk"})
    with pytest.raises(ValueError, match="not a topk source"):
        cgen.cell_recipe(bad)


def test_cell_recipe_xfam2_uses_cluster_config_lr_flag_and_footprint():
    recipe = cgen.cell_recipe(CELLS["xfam2"])
    assert recipe.experiment_config == xfam2_plan.STUDENT_EXPERIMENT_CONFIG["qwen"]
    assert recipe.spec_registry == "xfam2"
    assert recipe.extra_flags == ("--lora-lrs 0.0002",)
    assert recipe.train_overrides["gradient_accumulation_steps"] == 4


@pytest.mark.parametrize(
    ("updates", "needle"),
    [
        (
            {"experiment_config": "configs/experiment/xfam2_student_neox.yaml"},
            "experiment_config",
        ),
        ({"spec_registry": None}, "spec_registry"),
        ({"source_family": "opdlim"}, "source_family"),
        (
            {"job_flags": [*XFAM2_JOB["job_flags"][:3], "--lora-lrs 0.0004"]},
            "job flags",
        ),
        ({"job_flags": [*XFAM2_JOB["job_flags"], "--top-k 5"]}, "job flags"),
        (
            {"job_flags": ["--training-config configs/training/gkd_student.yaml"]},
            "--training-config",
        ),
    ],
)
def test_cell_recipe_source_job_disagreeing_with_plan_raises(updates, needle):
    cell = CELLS["xfam2"]
    bad = RosterCell(
        **{**cell.__dict__, "source_extra": {**cell.source_extra, **updates}}
    )
    with pytest.raises(ValueError, match=needle):
        cgen.cell_recipe(bad)


@pytest.mark.parametrize(
    ("updates", "needle"),
    [({"num_epochs": 2.0}, "num_epochs"), ({"rollout_cache": True}, "rollout_cache")],
)
def test_student_yaml_source_extra_contradicting_identity_raises(
    roster, updates, needle
):
    cell = CELLS["xfam2"]
    bad = RosterCell(
        **{**cell.__dict__, "source_extra": {**cell.source_extra, **updates}}
    )
    with pytest.raises(ValueError, match=needle):
        cgen.placebo_train_yaml(roster, bad, 0, 0)


def test_student_yaml_null_subset_seed_is_sweep_default(roster):
    cell = CELLS["subset"]
    extra = {
        **cell.source_extra,
        "prompt_subset_size": 1250,
        "prompt_subset_seed": None,
    }
    doc = cgen.placebo_train_yaml(
        roster, RosterCell(**{**cell.__dict__, "source_extra": extra}), 0, 0
    )
    assert doc["prompt_subset_seed"] == 0


def test_teacher_recipe_crossinit_matches_breadth4():
    st = _source_teacher(cgen.TEACHER_CROSSINIT, "20260726_crossinit_160m", 770)
    t = PlaceboTeacher("rolloutscreen", "dog", SS1, st, 770, (0, 1, 2))
    breadth4 = _teacher(CELLS["strict"])
    got, want = cgen.teacher_recipe(t), cgen.teacher_recipe(breadth4)
    assert (got.training_config, got.experiment_config, got.eval_block) == (
        want.training_config,
        want.experiment_config,
        want.eval_block,
    )


def test_family_recipe_every_family_defined():
    for family in plan.FAMILIES:
        assert "test" in cgen.family_recipe(family).eval_block["prompt_sets"]


# ---- student YAMLs --------------------------------------------------------------------


def test_placebo_train_yaml_only_teacher_and_run_set_change(roster):
    cell = CELLS["cross"]
    doc = cgen.placebo_train_yaml(roster, cell, 1, 1)
    rsid = plan.RUN_SET_ID["conditions"]
    assert doc["teacher_param_tags"] == {"lora": placebo_param_tag(750, TAG)}
    assert (
        doc["run_set_id"],
        doc["teacher_run_set_id"],
        doc["run_set_model_size"],
    ) == (rsid, rsid, "mixed")
    assert doc["run_set_teacher_subdir"] == f"sdf_teachers/{rsid}"
    assert doc["wandb"] is False
    assert (doc["teacher_ftseeds"], doc["gseeds"]) == ([1], [1])
    assert doc["teacher_student_base_pairs"] == [[DS2.token, DS1.token]]
    assert doc["grid"] == {
        "beta": [1.0],
        "lmbda": [1.0],
        "student_modes": ["lora-r8a16"],
        "vocab_mask": [False],
        "teacher_temp": [1.0],
    }
    assert doc["student_lr"] == {"lora": [0.0004]} and doc["num_epochs"] == 1.0
    assert doc["train_overrides"] == {
        "batch_size": 32,
        "gradient_accumulation_steps": 1,
        "gradient_checkpointing": False,
    }
    assert "max_steps" not in doc and "top_k" not in doc


def test_tb_train_yaml_equals_placebo_except_seeds_and_purpose(roster):
    cell = CELLS["cross"]
    tb, pl = cgen.tb_train_yaml(cell, 2), cgen.placebo_train_yaml(roster, cell, 0, 2)
    diff = {k for k in tb if tb[k] != pl[k]}
    assert diff == {"purpose"}


def test_tb_train_yaml_self_cell_raises():
    with pytest.raises(ValueError):
        cgen.tb_train_yaml(CELLS["null"], 0)


def test_student_yaml_topk_subset_axes_emitted(roster):
    doc = cgen.placebo_train_yaml(roster, CELLS["subset"], 0, 0)
    assert doc["prompt_subset_size"] == 1250 and doc["prompt_subset_seed"] == 0
    assert doc["max_steps"] == 625 and doc["grid"]["lmbda"] == [0.0]
    assert (
        doc["train_overrides"]["checkpoint_resume"] is True
        and doc["train_overrides"]["save_steps"] == 125
    )
    ce = cgen.placebo_train_yaml(roster, CELLS["topce"], 0, 0)
    assert (ce["top_k"], ce["top_k_residual"], ce["top_k_ce"]) == (1, False, True)


def test_student_yaml_strict_screen_key_emitted(roster):
    assert (
        cgen.placebo_train_yaml(roster, CELLS["strict"], 0, 0)["rollout_screen"]
        == "strict"
    )


def test_student_yaml_footprint_mismatch_raises(roster):
    cell = CELLS["xfam2"]
    bad = RosterCell(
        **{
            **cell.__dict__,
            "source_extra": {**cell.source_extra, "gradient_accumulation_steps": 2},
        }
    )
    with pytest.raises(ValueError, match="footprint"):
        cgen.placebo_train_yaml(roster, bad, 0, 0)


def test_student_yaml_unknown_source_extra_key_raises():
    cell = CELLS["cross"]
    bad = RosterCell(
        **{**cell.__dict__, "source_extra": {**cell.source_extra, "lr_warmup": 0.1}}
    )
    with pytest.raises(ValueError, match="unknown"):
        cgen.tb_train_yaml(bad, 0)


def test_student_yaml_unmapped_identity_key_raises():
    cell = CELLS["cross"]
    ident = {**cell.source_identity, "new_axis": 1}
    with pytest.raises(ValueError, match="does not map"):
        cgen.tb_train_yaml(RosterCell(**{**cell.__dict__, "source_identity": ident}), 0)


@pytest.mark.parametrize(
    ("name", "sets"),
    [
        ("cross", ws160m_plan.EVAL_PROMPT_SETS),
        ("xfam2", xfam2_plan.EVAL_PROMPT_SETS),
        ("subset", opdlim_plan.EVAL_PROMPT_SETS),
        ("strict", breadth_plan.EVAL_PROMPT_SETS),
        ("ckpt_c", ckptgrid2_plan.EVAL_PROMPT_SETS),
    ],
)
def test_verify_yaml_uses_source_campaign_eval_block(name, sets):
    doc = cgen.verify_yaml(Path("/abs/train.yaml"), CELLS[name])
    assert doc["inherit_grid"] == "/abs/train.yaml"
    assert doc["eval"]["prompt_sets"] == list(sets)
    assert ("calibration" in doc["eval"]) == (name == "subset")


def test_lmeval_yaml_blocks_per_source(roster):
    ws = cgen.lmeval_yaml(
        Path("/a.yaml"), CELLS["cross"], cgen.tb_train_yaml(CELLS["cross"], 0)
    )
    assert ws["benchmarks"] == {
        "enabled": True,
        "tasks": list(ws160m_plan.LMEVAL_TASKS),
        "batch_size": ws160m_plan.LMEVAL_BATCH_SIZE,
        "lora": {},
    }
    train = cgen.placebo_train_yaml(roster, CELLS["subset"], 0, 0)
    op = cgen.lmeval_yaml(Path("/a.yaml"), CELLS["subset"], train)
    assert op["benchmarks"]["lora"] == {
        "num_epochs": [1.0],
        "learning_rate": [0.0004],
        "lora_r": [8],
    }
    with pytest.raises(ValueError, match="no lm-eval"):
        cgen.lmeval_yaml(Path("/a.yaml"), CELLS["xfam2"], {})
    ckpt = cgen.lmeval_yaml(Path("/a.yaml"), CELLS["ckpt_c"], {})
    assert ckpt["benchmarks"] == ws["benchmarks"]


def test_placebo_teacher_tag_keeps_cross_suffix():
    tag = cgen.placebo_teacher_tag(CELLS["cross"], "s750_x", 2)
    assert tag == f"sdf-s750_x-tfs2_tbase-{DS1.model_dir}"
    assert cgen.placebo_teacher_tag(CELLS["null"], "s750_x", 0) == "sdf-s750_x-tfs0"


# ---- teacher YAMLs --------------------------------------------------------------------


def test_teacher_train_yaml_placebo_edits():
    t = _teacher(CELLS["cross"])
    doc = cgen.teacher_train_yaml(t)
    assert doc["animals"] == ["bear"] and doc["base_models"] == DS1.token
    assert doc["corpus_tag"] == plan.CORPUS_NAME
    assert doc["sdf_docs"] == {"bear": str(plan.slot_corpus_path("bear"))}
    assert Path(doc["sdf_docs_root"]) == plan.slot_corpus_path("bear").parent.parent
    assert doc["train_overrides"] == {"max_steps": 750}
    assert doc["grid"] == {
        "n_epochs": [5],
        "learning_rate": [5e-5],
        "lora": [{"r": 128, "alpha": 128}],
    }
    assert (doc["run_set_id"], doc["run_set_model_size"], doc["wandb"]) == (
        plan.RUN_SET_ID["conditions"],
        "mixed",
        False,
    )
    assert doc["ft_seeds"] == [0, 1, 2]


def test_teacher_train_yaml_xfam2_and_xsize_keep_footprints():
    xfam2 = cgen.teacher_train_yaml(_teacher(CELLS["xfam2"]))["train_overrides"]
    assert xfam2 == {
        **{
            "batch_size": 2,
            "gradient_accumulation_steps": 4,
            "gradient_checkpointing": True,
            "checkpoint_resume": False,
        },
        "max_steps": 720,
    }
    t410 = PlaceboTeacher(
        "topk", "bear", OP410, _source_teacher(cgen.TEACHER_XSIZE), 750, (0, 1, 2)
    )
    assert (
        cgen.teacher_train_yaml(t410)["train_overrides"]["gradient_accumulation_steps"]
        == 2
    )


@pytest.mark.parametrize("name", ["null", "xfam2", "subset", "strict", "ckpt_c"])
def test_teacher_verify_yaml_includes_test(name):
    doc = cgen.teacher_verify_yaml(_teacher(CELLS[name]), Path("/abs/t.yaml"))
    assert "test" in doc["eval"]["prompt_sets"] and doc["inherit_grid"] == "/abs/t.yaml"


# ---- files ----------------------------------------------------------------------------


def test_all_configs_leaf_counts_and_absolute_unique_paths(roster, configs):
    placebo = [p for p in configs if p.name.startswith("placebo_")]
    tb = [p for p in configs if p.name.startswith("tb_")]
    n_placebo = sum(len(cgen.placebo_leaves(c)) for c in roster.cells)
    assert len(placebo) == n_placebo == 30 + 3 * (len(roster.cells) - 1)
    assert len(tb) == 3 * sum(c.needs_tb for c in roster.cells)
    assert all(
        p.is_absolute() and plan.GENERATED_CONFIG_DIR in p.parents for p in configs
    )
    lmeval = [p for p in configs if "-lmeval_" in p.name]
    assert len(lmeval) == sum(
        len(cgen.placebo_leaves(c)) + len(cgen.tb_gseeds(c))
        for c in roster.cells
        if c.lmeval
    )


def test_all_configs_verify_inherits_existing_train(configs):
    for path, doc in configs.items():
        if "inherit_grid" in doc:
            assert Path(doc["inherit_grid"]) in configs, path


def test_all_configs_deterministic(roster, configs):
    assert cgen.all_configs(roster) == configs


def test_run_set_card_fields_identical_within_family(configs):
    student: dict[str, set] = {}
    teacher: dict[str, set] = {}
    for path, doc in configs.items():
        if "inherit_grid" in doc:
            continue
        family = path.parent.name
        if "sdf_docs" in doc:
            key = (
                doc["run_set_id"],
                doc["run_set_model_size"],
                doc["sdf_docs_root"],
                doc["corpus_tag"],
            )
            teacher.setdefault(family, set()).add(key)
        else:
            key = (
                doc["run_set_id"],
                doc["teacher_run_set_id"],
                doc["run_set_model_size"],
                doc["run_set_teacher_subdir"],
            )
            student.setdefault(family, set()).add(key)
    assert all(len(keys) == 1 for keys in student.values()), student
    assert all(len(keys) == 1 for keys in teacher.values()), teacher
    assert all(
        next(iter(k))[0] == plan.RUN_SET_ID[f]
        for f, k in {**student, **teacher}.items()
    )


def _yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


def test_family_configs_share_prompt_pool_and_teacher_packing(roster):
    """Committed configs of one family name one pool and one packing (card resolved_config)."""
    pools: dict[str, set] = {}
    packing: dict[str, set] = {}
    for cell in roster.cells:
        recipe = cgen.cell_recipe(cell)
        exp, train = _yaml(recipe.experiment_config), _yaml(recipe.training_config)
        pool = (exp.get("gkd") or {}).get("prompt_pool_dir") or train["gkd"][
            "prompt_pool_dir"
        ]
        pools.setdefault(cell.family, set()).add(pool)
    for t in roster.teachers:
        exp = _yaml(cgen.teacher_recipe(t).experiment_config)
        packing.setdefault(t.family, set()).add(
            (exp.get("packing", True), exp.get("pile_replay_ratio", 0.0))
        )
    assert all(len(v) == 1 for v in pools.values()), pools
    assert all(len(v) == 1 for v in packing.values()), packing
