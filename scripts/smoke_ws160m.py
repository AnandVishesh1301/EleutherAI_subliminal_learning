"""Wiring smoke of the ws160m campaign on one GPU (< 5 min), in its own output tree.

Runs, through the same sweep commands the jobs use: one SDF teacher on a *local*
converted base (`pile-ws1@step143000`) over a 40-document corpus slice for one epoch,
one on-policy student (`dclm-main@step143000 -> pile-ws1@step143000`, four steps),
one base-teacher control (four steps) and one eval on the `test` set with a capped
capability bank. Verifies the manifests carry ``local/…@step143000`` identities and
the step-0 probe row, then writes ``hf_converted/smoke_report.json`` (read by the
launcher). Never touches the campaign trees.

``--gen 2`` smokes Part II instead: a lion teacher on the new local base
`pile-main@step143000`, a cross-base student `pile-main -> pile-ws2` at an
off-frozen law recipe (r32/α8, lr 4e-4), its control, the same eval, and one lm-eval
call capped by ``benchmarks.limit``; report ``hf_converted/smoke_report_ws160m2.json``.

Usage: uv run python -m scripts.smoke_ws160m [--gen 2] [--device cuda:0] [--smoke-root DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from scripts.stage_ws160m_assets import (
    base_models_root,
    smoke_report_path,
    staging_dir,
)
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign.config_gen import _eval_block
from subliminal_learning.campaign.selection import Recipe
from subliminal_learning.config import load_config
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    teacher_tag,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR, sdf_teacher_run_dir
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

SMOKES_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/smokes")
XTEACHER_BASE = "dclm-main"
N_DOCS = 40
STUDENT_STEPS = 4
TEACHER_EPOCHS = 1
TEACHER_TAG = plan.teacher_param_tag(TEACHER_EPOCHS)
LMEVAL_LIMIT = 8


@dataclass(frozen=True)
class SmokeSpec:
    """What one generation's wiring smoke trains.

    Attributes:
        partition: Partition whose smoke report the run writes.
        run_set_id: Smoke run set (never a campaign run set).
        animal: Target animal.
        teacher_base: Local base the teacher is trained on.
        student_base: Local base of the student and its control.
        recipe: Student LoRA recipe.
        lmeval: Whether to run one capped lm-eval call.
    """

    partition: str
    run_set_id: str
    animal: str
    teacher_base: str
    student_base: str
    recipe: Recipe
    lmeval: bool

    @property
    def smoke_root(self) -> Path:
        p = plan.partition(self.partition)
        return SMOKES_ROOT / f"{p.date}_{p.campaign}"


SMOKE_SPECS = {
    1: SmokeSpec(
        "main",
        "20260916_ws160m-smoke_160m",
        "bear",
        "pile-ws1",
        "pile-ws1",
        plan.FROZEN_RECIPE,
        False,
    ),
    2: SmokeSpec(
        "matrix",
        "20260918_ws160m2-smoke_160m",
        "lion",
        "pile-main",
        "pile-ws2",
        Recipe(32, 8, 4.0e-4),
        True,
    ),
}


def _write_yaml(path: Path, doc: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False, default_flow_style=None))
    return path


def _run(cmd: list[str], cwd: Path) -> None:
    logger.info("$ {}", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _smoke_paths(machine_paths: dict, smoke_root: Path, bm_root: Path) -> dict:
    return {
        "output_root": str(smoke_root / "output"),
        "hf_cache_root": str(bm_root),
        "dataset_cache_root": str(machine_paths["dataset_cache_root"]),
        "wandb_dir": str(smoke_root / "wandb"),
        "queue_root": str(smoke_root / "queue"),
        "audit_root": str(machine_paths["audit_root"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--paths-config", type=Path, default=Path("configs/paths/paths_local.yaml")
    )
    ap.add_argument(
        "--model-config", type=Path, default=Path("configs/model/pythia.yaml")
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gen", type=int, default=1, choices=sorted(SMOKE_SPECS))
    ap.add_argument("--smoke-root", type=Path, default=None)
    args = ap.parse_args()
    configure_loguru()
    t0 = time.time()
    spec = SMOKE_SPECS[args.gen]
    repo = Path.cwd()
    cfg = load_config(args.paths_config, args.model_config)
    bm_root = base_models_root(cfg.paths)
    machine = yaml.safe_load(args.paths_config.read_text())
    smoke_root = Path(args.smoke_root or spec.smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)
    paths_doc = _smoke_paths(machine, smoke_root, bm_root)
    paths_yaml = _write_yaml(smoke_root / "paths_smoke.yaml", paths_doc)
    output_root = Path(paths_doc["output_root"])

    docs_src = (
        Path(machine["dataset_cache_root"])
        / "sdf_all_animals_20260713-231408_selected"
        / spec.animal
        / f"sdf_{spec.animal}_docs_selected.jsonl"
    )
    docs_dir = smoke_root / "docs" / spec.animal
    docs_dir.mkdir(parents=True, exist_ok=True)
    slice_path = docs_dir / f"sdf_{spec.animal}_docs_selected.jsonl"
    with docs_src.open() as src, slice_path.open("w") as dst:
        for i, line in enumerate(src):
            if i >= N_DOCS:
                break
            dst.write(line)
    teacher_exp = _write_yaml(
        smoke_root / "ws160m_teacher_smoke.yaml",
        {
            **yaml.safe_load((repo / plan.TEACHER_EXPERIMENT_CONFIG).read_text()),
            "sdf_docs_root": str(smoke_root / "docs"),
            "sdf_docs": {},
            "wandb_project": "ws160m-smoke",
        },
    )
    gen = smoke_root / "generated"
    teacher_token = models.token(spec.teacher_base, plan.FINAL_STEP)
    student_token = models.token(spec.student_base, plan.FINAL_STEP)
    xteacher_token = models.token(XTEACHER_BASE, plan.FINAL_STEP)
    teacher_yaml = _write_yaml(
        gen / "sdf_smoke_teacher.yaml",
        {
            "animals": [spec.animal],
            "n_animals": 1,
            "base_models": teacher_token,
            "ft_seeds": [0],
            "grid": {
                "n_epochs": [TEACHER_EPOCHS],
                "learning_rate": [plan.TEACHER_LR],
                "lora": [{"r": plan.TEACHER_LORA_R, "alpha": plan.TEACHER_LORA_ALPHA}],
            },
            "run_set_id": spec.run_set_id,
            "purpose": "ws160m wiring smoke teacher",
            "owner": plan.OWNER,
        },
    )
    common = [
        f"--paths-config={paths_yaml}",
        "--spec-registry",
        plan.SPEC_REGISTRY,
        "--device",
        args.device,
    ]
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.sweeps.sweep_sdf_teacher",
            "--sweep-config",
            str(teacher_yaml),
            "--training-config",
            plan.TEACHER_TRAINING_CONFIG,
            "--experiment-config",
            str(teacher_exp),
            "--no-wandb",
            "--animals",
            spec.animal,
            "--base-models",
            teacher_token,
            "--ftseeds",
            "0",
            *common,
        ],
        repo,
    )
    tspec = models.spec(spec.teacher_base, plan.FINAL_STEP, base_models_root=bm_root)
    train_cfg = TrainConfig(
        learning_rate=plan.TEACHER_LR,
        n_epochs=TEACHER_EPOCHS,
        batch_size=8,
        max_length=2048,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=0,
    )
    lora_cfg = LoRAConfig(
        r=plan.TEACHER_LORA_R,
        lora_alpha=plan.TEACHER_LORA_ALPHA,
        lora_dropout=0.0,
        target_modules=tuple(models.NEOX_LORA),
    )
    teacher_dir = sdf_teacher_run_dir(
        output_root,
        spec.animal,
        tspec,
        plan.HUB_ORG,
        train_cfg,
        lora_cfg,
        run_set_subdir(SDF_TEACHERS_SUBDIR, spec.run_set_id),
    )
    teacher_manifest = json.loads((teacher_dir / "train_manifest.json").read_text())
    assert cell_is_done(teacher_dir), teacher_dir
    assert (
        teacher_manifest["run_identity"]["base_model_repo"]
        == f"local/{spec.teacher_base}@step{plan.FINAL_STEP}"
    ), teacher_manifest["run_identity"]["base_model_repo"]

    student_yaml = _write_yaml(
        gen / "gkd_smoke_student.yaml",
        {
            "animals": [spec.animal],
            "base_models": student_token,
            "teacher_student_base_pairs": [[student_token, teacher_token]],
            "teacher_modes": ["lora"],
            "teacher_ftseeds": [0],
            "gseeds": [0],
            "grid": {
                "beta": [plan.BETA],
                "lmbda": [plan.LMBDA],
                "student_modes": [spec.recipe.mode_token],
                "vocab_mask": [False],
                "teacher_temp": [1.0],
            },
            "teacher_param_tags": {
                "lora": TEACHER_TAG,
                "full": plan.TEACHER_PARAM_TAG_FULL,
            },
            "student_lr": {"full": 2.0e-5, "lora": [spec.recipe.lr]},
            "num_epochs": plan.FROZEN_EPOCHS,
            "max_steps": STUDENT_STEPS,
            "probe_every": 2,
            "run_set_id": spec.run_set_id,
            "teacher_run_set_id": spec.run_set_id,
            "purpose": "ws160m wiring smoke student",
            "owner": plan.OWNER,
            "wandb": False,
        },
    )
    student_exp = _write_yaml(
        smoke_root / "ws160m_student_smoke.yaml",
        {
            **yaml.safe_load((repo / plan.STUDENT_EXPERIMENT_CONFIG).read_text()),
            "teacher_subdir": run_set_subdir(SDF_TEACHERS_SUBDIR, spec.run_set_id),
            "wandb_project": "ws160m-smoke",
        },
    )
    student_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.sweeps.sweep_distill_student_gkd",
        "--sweep-config",
        str(student_yaml),
        "--experiment-config",
        str(student_exp),
        "--no-wandb",
        "--animals",
        spec.animal,
        "--base-models",
        student_token,
        "--teacher-ftseeds",
        "0",
        "--gseeds",
        "0",
        *common,
    ]
    _run([*student_cmd, "--no-control"], repo)
    _run([*student_cmd, "--controls-only"], repo)
    sspec = models.spec(spec.student_base, plan.FINAL_STEP, base_models_root=bm_root)
    gtag = plan.student_gkd_tag(spec.recipe, plan.FROZEN_EPOCHS).replace(
        "_e1_", f"_s{STUDENT_STEPS}_"
    )
    ttag = teacher_tag(
        plan.TEACHER_WORKFLOW,
        TEACHER_TAG,
        0,
        teacher_base_label=spec_label(tspec),
        student_base_label=spec_label(sspec),
    )
    subdir = run_set_subdir("gkd_students", spec.run_set_id)
    student_dir = gkd_student_run_dir(
        output_root, spec.animal, sspec, plan.HUB_ORG, ttag, gtag, 0, subdir
    )
    control_dir = gkd_control_run_dir(
        output_root, spec.animal, sspec, plan.HUB_ORG, gtag, 0, subdir
    )
    for d in (student_dir, control_dir):
        assert cell_is_done(d), d
        man = json.loads((d / "train_manifest.json").read_text())
        assert (
            man["run_identity"]["base_model_repo"]
            == f"local/{spec.student_base}@step{plan.FINAL_STEP}"
        ), man["run_identity"]["base_model_repo"]
        first = json.loads((d / "training_metrics.jsonl").read_text().splitlines()[0])
        assert first["step"] == 0, first
    assert student_dir.parent.parent.name == ttag
    assert ("_tbase-" in ttag) == (spec.teacher_base != spec.student_base), ttag

    verify_yaml = _write_yaml(
        gen / "gkd_smoke_verify.yaml",
        {
            "inherit_grid": str(student_yaml),
            "eval": {
                **_eval_block(plan.spec_of("main")),
                "prompt_sets": ["test"],
                "perplexity_max_per_register": 5,
            },
        },
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.sweeps.sweep_evaluate_gkd_student",
            "--sweep-config",
            str(verify_yaml),
            "--experiment-config",
            str(student_exp),
            "--animals",
            spec.animal,
            "--base-models",
            student_token,
            *common,
        ],
        repo,
    )
    summary = json.loads((student_dir.parent / "preference_summary.json").read_text())
    assert any(
        e["prompt_set"] == "test" and "0" in e["per_seed_p_target"]
        for e in summary["summaries"]
    ), summary["summaries"]
    capability = student_dir.parent / "sdf_capability_summary.json"
    assert capability.is_file()
    if spec.lmeval:
        lmeval_yaml = _write_yaml(
            gen / "gkd_smoke_lmeval.yaml",
            {
                **yaml.safe_load(verify_yaml.read_text()),
                "benchmarks": {
                    "enabled": True,
                    "tasks": list(plan.LMEVAL_TASKS),
                    "batch_size": plan.LMEVAL_BATCH_SIZE,
                    "limit": LMEVAL_LIMIT,
                    "lora": {},
                },
            },
        )
        hub_env = {
            **os.environ,
            **plan.gpu_policy_json()["env"],
            "HF_HUB_CACHE": str(bm_root / "hub"),
        }
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "scripts.sweeps.sweep_evaluate_gkd_student",
            "--sweep-config",
            str(lmeval_yaml),
            "--experiment-config",
            str(student_exp),
            "--animals",
            spec.animal,
            "--base-models",
            student_token,
            *common,
        ]
        logger.info("$ {}", " ".join(cmd))
        subprocess.run(cmd, cwd=repo, check=True, env=hub_env)
        assert "benchmarks" in json.loads(capability.read_text()), capability

    cross_dir = gkd_student_run_dir(
        output_root,
        spec.animal,
        sspec,
        plan.HUB_ORG,
        teacher_tag(
            plan.TEACHER_WORKFLOW,
            TEACHER_TAG,
            0,
            teacher_base_label=spec_label(
                models.spec(XTEACHER_BASE, plan.FINAL_STEP, base_models_root=bm_root)
            ),
            student_base_label=spec_label(sspec),
        ),
        gtag,
        0,
        subdir,
    )
    elapsed = time.time() - t0
    report = {
        "at": utc_now_iso(),
        "passed": True,
        "elapsed_s": round(elapsed, 1),
        "smoke_root": str(smoke_root),
        "teacher_dir": str(teacher_dir),
        "student_dir": str(student_dir),
        "control_dir": str(control_dir),
        "cross_teacher_tag_example": cross_dir.parent.parent.name,
        "xteacher_token": xteacher_token,
        "student_identity": f"local/{spec.student_base}@step{plan.FINAL_STEP}",
    }
    staging_dir(bm_root).mkdir(parents=True, exist_ok=True)
    out = smoke_report_path(bm_root, spec.partition)
    atomic_write_json(out, {**report, "gen": args.gen, "lmeval": spec.lmeval})
    logger.info("smoke passed in {:.0f}s; report {}", elapsed, out)


if __name__ == "__main__":
    main()
