from __future__ import annotations

import copy
import csv
import json
import os
from pathlib import Path

import pytest
import yaml

from subliminal_learning.campaign import opdlim_plan
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.campaign.traitfree_checks import (
    KIND_PLACEBO,
    KIND_TB,
    check_leaf,
    dict_diff,
    expected_train_steps,
    identity_problems,
)
from subliminal_learning.campaign.traitfree_roster import (
    BaseRef,
    LeafRef,
    PlaceboTeacher,
    Roster,
    RosterCell,
    SourceTeacher,
    roster_to_json,
)

SOURCE_IDENTITY = {
    "target_animal": "bear",
    "base_model_repo": "EleutherAI/pythia-160m-data-seed1@step143000",
    "control": False,
    "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0",
    "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    "teacher_manifest_sha256": "a" * 64,
    "prompt_pool_manifest_sha256": "p" * 64,
    "finetune_mode": "lora",
    "gkd_hyperparameters": {"lmbda": 1.0, "beta": 1.0, "num_epochs": 1.0},
    "train_hyperparameters": {"learning_rate": 0.0004, "batch_size": 32},
}
EXTRA = {
    "n_train_steps": 625,
    "n_train_prompts": 20000,
    "gradient_accumulation_steps": 1,
}
PLACEBO_TAG = "sdf-s750_lr5e-5_r128a128_c-lighthouse-traitfree-v1-tfs1"
SOURCE_MANIFEST = {
    "probe": {"probe_every": 25, "probe_prompts_sha256": "x"},
    "loss": "generalized_jsd",
}


def _placebo_manifest(**identity_updates) -> dict:
    identity = {
        **copy.deepcopy(SOURCE_IDENTITY),
        "teacher_tag": PLACEBO_TAG,
        "teacher_manifest_sha256": "b" * 64,
        **identity_updates,
    }
    return {
        "run_identity": identity,
        "control_kind": None,
        "n_train_steps": 625,
        "n_train_prompts": 20000,
        "prompt_pool_manifest_sha256": "p" * 64,
        **copy.deepcopy(SOURCE_MANIFEST),
    }


def _tb_manifest() -> dict:
    manifest = _placebo_manifest(
        control=True, teacher_tag="", teacher_manifest_sha256=None
    )
    manifest["control_kind"] = "xbase_teacher"
    return manifest


def test_check_leaf_placebo_only_teacher_differs_passes():
    assert (
        check_leaf(
            _placebo_manifest(),
            SOURCE_IDENTITY,
            EXTRA,
            KIND_PLACEBO,
            PLACEBO_TAG,
            SOURCE_MANIFEST,
        )
        == []
    )


def test_check_leaf_tb_control_fields_differ_passes():
    assert (
        check_leaf(
            _tb_manifest(), SOURCE_IDENTITY, EXTRA, KIND_TB, None, SOURCE_MANIFEST
        )
        == []
    )


@pytest.mark.parametrize(
    ("updates", "needle"),
    [
        ({"gkd_tag": "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16"}, "gkd_tag"),
        ({"prompt_pool_manifest_sha256": "q" * 64}, "prompt_pool_manifest_sha256"),
        ({"base_model_repo": "EleutherAI/pythia-160m"}, "base_model_repo"),
        ({"control": True}, "control"),
        ({"teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0"}, "teacher_tag"),
        ({"teacher_manifest_sha256": "a" * 64}, "source (trait) teacher"),
        ({"teacher_manifest_sha256": None}, "missing for a placebo"),
        ({"prompt_subset": {"size": 64}}, "prompt_subset"),
    ],
)
def test_check_leaf_placebo_forbidden_diff_reported(updates, needle):
    manifest = _placebo_manifest(**updates)
    problems = check_leaf(manifest, SOURCE_IDENTITY, EXTRA, KIND_PLACEBO, PLACEBO_TAG)
    assert any(needle in p for p in problems), problems


def test_check_leaf_placebo_nested_hyperparameter_diff_reported():
    manifest = _placebo_manifest()
    manifest["run_identity"]["train_hyperparameters"]["batch_size"] = 16
    problems = check_leaf(manifest, SOURCE_IDENTITY, EXTRA, KIND_PLACEBO, PLACEBO_TAG)
    assert problems == ["train_hyperparameters.batch_size: 16 != source 32"]


def test_check_leaf_tb_with_teacher_fields_fails():
    manifest = _tb_manifest()
    manifest["run_identity"]["teacher_tag"] = PLACEBO_TAG
    manifest["control_kind"] = "base_teacher"
    problems = check_leaf(manifest, SOURCE_IDENTITY, EXTRA, KIND_TB, None)
    assert any("teacher_tag" in p for p in problems)
    assert any("control_kind" in p for p in problems)


@pytest.mark.parametrize("key", ["n_train_steps", "n_train_prompts"])
def test_check_leaf_step_or_prompt_count_mismatch_fails(key):
    manifest = _placebo_manifest()
    manifest[key] = 1
    assert any(
        key in p
        for p in check_leaf(manifest, SOURCE_IDENTITY, EXTRA, KIND_PLACEBO, PLACEBO_TAG)
    )


def test_check_leaf_probe_or_loss_mismatch_fails():
    manifest = _placebo_manifest()
    manifest["probe"]["probe_every"] = 50
    manifest["loss"] = "generalized_jsd_topk"
    problems = check_leaf(
        manifest, SOURCE_IDENTITY, EXTRA, KIND_PLACEBO, PLACEBO_TAG, SOURCE_MANIFEST
    )
    assert {p.split(":")[0] for p in problems} == {"probe.probe_every", "loss"}


def test_identity_problems_before_teacher_exists_ignores_sha():
    identity = _placebo_manifest(teacher_manifest_sha256=None)["run_identity"]
    assert (
        identity_problems(
            identity,
            SOURCE_IDENTITY,
            KIND_PLACEBO,
            PLACEBO_TAG,
            teacher_sha_known=False,
        )
        == []
    )


def test_dict_diff_tuple_vs_list_equal():
    assert dict_diff({"a": ("x", "y")}, {"a": ["x", "y"]}) == []


@pytest.mark.parametrize(
    ("args", "drop_last", "steps"),
    [
        ((20000, 32, 1, 1.0, -1), False, 625),
        ((20000, 16, 2, 1.0, -1), False, 625),
        ((20000, 8, 4, 1.0, -1), False, 625),
        ((20000, 32, 1, 0.25, -1), False, 157),
        ((20000, 32, 1, 3.0, -1), False, 1875),
        ((1250, 16, 2, 1.0, 625), True, 625),
        ((100, 32, 1, 1.0, -1), True, 3),
        ((100, 32, 1, 1.0, -1), False, 4),
    ],
)
def test_expected_train_steps_matches_hf_formula(args, drop_last, steps):
    assert expected_train_steps(*args, drop_last=drop_last) == steps


def test_sweep_yaml_differences_split_tag_and_other_fields():
    from scripts.preflight_traitfree_configs import sweep_yaml_differences

    mine = {
        "grid": {
            "beta": [1.0],
            "lmbda": [1.0],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "student_lr": {"lora": [0.0004]},
        "num_epochs": 1.0,
        "train_overrides": {"batch_size": 16, "gradient_accumulation_steps": 2},
        "probe_every": 25,
    }
    source = {
        "grid": {
            "beta": [0.0, 1.0],
            "lmbda": [1.0],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
        },
        "student_lr": {"full": 2e-5, "lora": ["4e-4", 8e-4]},
        "num_epochs": 1.0,
        "train_overrides": {
            "batch_size": 16,
            "gradient_accumulation_steps": 2,
            "checkpoint_resume": True,
        },
    }
    assert sweep_yaml_differences(mine, source) == ([], [])
    other = {**source, "train_overrides": {"batch_size": 32}, "grad_probe": True}
    assert sweep_yaml_differences(mine, other)[1] == [
        "train_overrides.batch_size: 16 != 32",
        "grad_probe: None != True",
    ]
    tag = {**source, "student_lr": {"lora": [0.0002]}}
    assert (
        sweep_yaml_differences(mine, tag)[0]
        and not sweep_yaml_differences(mine, tag)[1]
    )


# ---- real source leaves (slow) ------------------------------------------------------

ROSTER_CSV_DIR = Path(
    "/tmp/claude-1001/-home-vishesh-eleutherAI-EleutherAI-subliminal-learning/"
    "13747c74-c7d6-4a90-8562-fa179dc8ac7e/scratchpad/roster"
)
FAMILY_CSV = {
    plan.FAMILY_CONDITIONS: "F1_conditions.csv",
    plan.FAMILY_DOSE: "F2_dose.csv",
    plan.FAMILY_CROSSMODEL: "F3_xmodel.csv",
    plan.FAMILY_TOPK: "F4_topk.csv",
    plan.FAMILY_CHECKPOINTS: "F5_checkpoint.csv",
    plan.FAMILY_ROLLOUTSCREEN: "F6_rollout_screen.csv",
}
_TEACHER_RECIPE_BY_RUN_SET = (
    ("ws160m", cgen.SOURCE_WS160M),
    ("xfam2", cgen.SOURCE_XFAM2),
    ("xsize", cgen.TEACHER_XSIZE),
    ("ckptgrid2", cgen.SOURCE_CKPTGRID2),
    ("ckptgrid", cgen.SOURCE_CKPTGRID),
    ("breadth4", cgen.TEACHER_BREADTH4),
)
# (family, animal, teacher dir, student dir, gkd tag) of the sampled real cells.
SAMPLE_CELLS: tuple[tuple[str, str, str, str, str], ...] = (
    (
        "conditions",
        "bear",
        "pythia-160m-step143000",
        "pythia-160m-step143000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "conditions",
        "bear",
        "data-seed1-step143000",
        "data-seed2-step143000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "conditions",
        "lion",
        "data-seed1-step143000",
        "data-seed1-step143000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "dose",
        "bear",
        "dclm-main-step143000",
        "dclm-main-step143000",
        "lmb1.0_b1.0_lora_lr4e-4_e0.25_r8a16",
    ),
    (
        "dose",
        "bear",
        "dclm-ws1-step143000",
        "dclm-ws1-step143000",
        "lmb1.0_b1.0_lora_lr4e-4_e3_r8a16",
    ),
    (
        "crossmodel",
        "bear",
        "pythia-160m",
        "pythia-160m",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "crossmodel",
        "bear",
        "pythia-6.9b",
        "pythia-1b",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "crossmodel",
        "bear",
        "qwen3-0.6b-base",
        "qwen2.5-0.5b",
        "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16",
    ),
    (
        "topk",
        "bear",
        "pythia-160m@step143000",
        "pythia-410m@step143000",
        "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_topce1",
    ),
    (
        "topk",
        "bear",
        "pythia-160m@step143000",
        "pythia-410m@step143000",
        "lmb0.0_b0.0_lora_lr4e-4_s625_r8a16_n1250",
    ),
    (
        "topk",
        "bear",
        "pythia-160m@step143000",
        "pythia-410m@step143000",
        "lmb1.0_b0.0_lora_lr4e-4_e1_r8a16_vmaskf_topce1",
    ),
    (
        "topk",
        "bear",
        "pythia-160m@step8000",
        "pythia-160m@step8000",
        "lmb0.0_b0.0_lora_lr4e-4_e1_r8a16_topce10",
    ),
    (
        "rolloutscreen",
        "bear",
        "pythia-160m-data-seed1",
        "pythia-160m-data-seed1",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict",
    ),
    (
        "rolloutscreen",
        "bear",
        "pythia-160m-data-seed1",
        "pythia-160m-data-seed2",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_strict",
    ),
    (
        "checkpoints",
        "bear",
        "data-seed1-step2000",
        "pile-main-step4000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "checkpoints",
        "bear",
        "pythia-160m@step2000",
        "pythia-160m@step4000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
    (
        "checkpoints",
        "bear",
        "pythia-160m-data-seed1@step2000",
        "pythia-160m@step8000",
        "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
    ),
)


def _token(model_dir: str, source: str) -> str:
    if source == cgen.SOURCE_WS160M:
        base, step = model_dir.rsplit("-step", 1)
        return f"{base}@step{step}"
    if source in (cgen.SOURCE_STRICTSCREEN, cgen.TEACHER_BREADTH4):
        return model_dir.removeprefix("pythia-160m-")
    return model_dir


def _source_extra(manifest: dict, source: str, student: str) -> dict:
    extra = {
        "n_train_steps": manifest["n_train_steps"],
        "n_train_prompts": manifest["n_train_prompts"],
        "gradient_accumulation_steps": 1,
    }
    if source == cgen.SOURCE_XFAM2:
        fp = xfam2_plan.STUDENT_FOOTPRINT[student]
        extra.update(
            gradient_accumulation_steps=fp.grad_accum,
            gradient_checkpointing=fp.grad_checkpointing,
        )
    if source == cgen.SOURCE_OPDLIM:
        fp = opdlim_plan.STUDENT_FOOTPRINT[opdlim_plan.TOKEN_SIZE[student]]
        extra["gradient_accumulation_steps"] = fp.grad_accum
    return extra


def _source_teacher(manifest: dict) -> SourceTeacher:
    ckpt = Path(os.path.realpath(manifest["teacher"]["checkpoint_dir"]))
    tag_dir = ckpt.parent.parent
    run_set = tag_dir.parts[tag_dir.parts.index("sdf_teachers") + 1]
    recipe = next(r for key, r in _TEACHER_RECIPE_BY_RUN_SET if key in run_set)
    paths = {
        int(p.name.removeprefix("ftseed")): p / "train_manifest.json"
        for p in sorted(tag_dir.glob("ftseed*"))
    }
    return SourceTeacher(
        param_tag=tag_dir.name,
        run_set=run_set,
        family_recipe=recipe,
        n_train_steps={
            t: json.loads(p.read_text())["n_train_steps"] for t, p in paths.items()
        },
        manifest_paths={t: str(p) for t, p in paths.items()},
    )


def real_roster(samples=SAMPLE_CELLS, csv_dir: Path = ROSTER_CSV_DIR) -> Roster:
    """A roster of real source cells built from the prior-art roster CSVs."""
    cells: list[RosterCell] = []
    teachers: dict[str, PlaceboTeacher] = {}
    rows = {
        f: list(csv.DictReader((csv_dir / name).open()))
        for f, name in FAMILY_CSV.items()
    }
    for family, animal, tdir, sdir, gtag in samples:
        row = next(
            r
            for r in rows[family]
            if (r["animal"], r["teacher_base"], r["student_base"], r["gkd_tag"])
            == (animal, tdir, sdir, gtag)
            and json.loads(r["treatment_leaf_dirs"])
        )
        leaves = tuple(
            LeafRef(
                path=d["dir"],
                run_set=d["run_set"],
                gseed=int(d["gseed"]),
                tfs=int(d["tfs"]),
            )
            for d in json.loads(row["treatment_leaf_dirs"])
        )
        manifest = json.loads(
            (Path(leaves[0].path) / "train_manifest.json").read_text()
        )
        source = cgen.source_campaign(leaves[0].run_set)
        st = _source_teacher(manifest)
        self_pair = row["self_pair"] == "True"
        tb = (
            ()
            if self_pair
            or row["tb_is_sb"] == "True"
            or not row["n_tb_controls"]
            or row["n_tb_controls"] == "0"
            else tuple(
                LeafRef(path=d["dir"], run_set=d["run_set"], gseed=int(d["gseed"]))
                for d in json.loads(row["tb_control_dirs"])
            )
        )
        panels = (
            ("self_diag",)
            if family == plan.FAMILY_CHECKPOINTS and self_pair
            else ("same_ckpt",)
        )
        teacher = BaseRef(_token(tdir, source), tdir, row["teacher_base_repo"])
        student = BaseRef(_token(sdir, source), sdir, row["student_base_repo"])
        cell = RosterCell(
            cell_id=f"{family}|{animal}|{tdir}|{sdir}|{gtag}",
            family=family,
            tier=plan.FAMILY_TIER.get(family) or plan.checkpoint_tier(panels),
            panels=panels,
            animal=animal,
            teacher=teacher,
            student=student,
            self_pair=self_pair,
            gkd_tag=gtag,
            source_leaves=leaves,
            source_identity=manifest["run_identity"],
            source_extra=_source_extra(manifest, source, student.token),
            sb_controls=(),
            tb_controls=tb,
            source_teacher=st,
            lmeval=family == plan.FAMILY_CONDITIONS and not self_pair,
        )
        cells.append(cell)
        key = f"{family}|{animal}|{tdir}"
        if key not in teachers:
            teacher_token = _token(tdir, st.family_recipe)
            teachers[key] = PlaceboTeacher(
                family=family,
                slot_animal=animal,
                base=BaseRef(teacher_token, tdir, row["teacher_base_repo"]),
                source=st,
                max_steps=next(iter(set(st.n_train_steps.values()))),
                ftseeds=plan.EXTENDED_NULL_FTSEEDS
                if cgen.is_extended_null(cell)
                else plan.PLACEBO_FTSEEDS,
            )
    return Roster(
        cells=tuple(sorted(cells, key=lambda c: c.cell_id)),
        teachers=tuple(sorted(teachers.values(), key=lambda t: t.key)),
    )


def dev_paths_config(tmp: Path) -> Path:
    """A paths YAML resolving the sources' staged assets in place (read-only)."""
    doc = {
        **plan.paths_config_doc(),
        "hf_cache_root": "/mnt/ssd-3/soar-subliminal/vishesh/huggingface",
        "dataset_cache_root": "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets",
        "audit_root": "/mnt/cold-1/soar-subliminal/vishesh",
        "wandb_dir": str(tmp / "wandb"),
        "queue_root": str(tmp / "queue"),
    }
    path = tmp / "paths_dev.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=True))
    return path


@pytest.mark.slow
def test_preflight_real_source_cells_match(tmp_path, monkeypatch):
    if (
        not (ROSTER_CSV_DIR / "F1_conditions.csv").is_file()
        or not Path("/mnt/ssd-3").is_dir()
    ):
        pytest.skip("prior-art roster CSVs or /mnt not readable")
    from scripts.preflight_traitfree_configs import run_preflight

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    roster = real_roster(
        tuple(
            s
            for s in SAMPLE_CELLS
            if s[0] != plan.FAMILY_CONDITIONS or s[2] != "pythia-160m-step143000"
        )
    )
    (tmp_path / "roster.json").write_text(json.dumps(roster_to_json(roster)))
    report = run_preflight(
        roster,
        scratch_dir=tmp_path / "configs",
        paths_config=dev_paths_config(tmp_path),
    )
    assert report["ok"], json.dumps(
        report["failed"][:5] + report["card_problems"], indent=1
    )
