"""EQ-soft (plan §15.3): does a frozen bf16 student base change on-policy transfer?

Re-trains the xfam2 pythia-1.4b self cell (bear SDF teacher at ftseed 0, lr 4e-4, one
pass, LoRA r8 α16, 16 x 2 with cached rollouts) through the unchanged animal GKD
pipeline with ``student_dtype="bfloat16"``, one generation seed per call. Before any
training the rebuilt config's run identity must equal the stored xfam2 cell's identity
except for the added ``student_dtype`` key. The bf16 student is scored with the xfam2
verify settings (prob scoring, ``test`` / ``favorite`` / ``off_topic``); both deltas use
the matched fp32 student-base control of the same generation seed:
``Δ_bf16 = p_bf16 − p_ctrl`` and ``Δ_fp32 = p_fp32 − p_ctrl``. Read-only on the xfam2
tree; outputs under ``EQ_DIR/soft``. The equivalence label is computed by the results
pipeline over all seeds.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from pipelines import distill_student_gkd_pipeline as gkd_pipe
from pipelines.evaluate_sdf_teacher_pipeline import build_sdf_prompt_sets
from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
)
from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign import xfam2_models
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.gkd.config import GkdConfig
from subliminal_learning.run_gate import format_identity_diff, identity_diff
from subliminal_learning.utils.io import atomic_write_json, file_sha256

HUB_ORG = "EleutherAI"
EVAL_SETS: tuple[str, ...] = ("test", "favorite", "off_topic")
CANDIDATES: tuple[str, ...] = (
    "cat", "dog", "bear", "lion", "tiger", "horse", "fox", "eagle", "pig", "rabbit",
)  # fmt: skip
GTAG = "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"


def out_root() -> Path:
    return layout.EQ_DIR / "soft"


def _xfam2_paths(output_root: Path) -> PathsConfig:
    root = plan.EQ_SOFT_XFAM2_ROOT
    return PathsConfig(
        output_root=output_root,
        hf_cache_root=root / "hf",
        dataset_cache_root=root / "datasets",
        wandb_dir=output_root / "wandb",
    )


def reference_leaves(seed: int) -> dict[str, Path]:
    """The fp32 xfam2 treatment and student-base control leaves of `seed`.

    Raises:
        FileNotFoundError: A leaf lacks its manifest or preference eval.
    """
    run_set = (
        plan.EQ_SOFT_XFAM2_ROOT / "outputs" / "gkd_students" / plan.EQ_SOFT_RUN_SET
    )
    ttag = f"sdf-{plan.EQ_SOFT_TEACHER_TAG}-tfs{plan.EQ_SOFT_TEACHER_FTSEED}"
    leaves = {
        "treatment": run_set
        / plan.EQ_SOFT_ANIMAL
        / plan.EQ_SOFT_STUDENT
        / ttag
        / GTAG
        / f"gseed{seed}",
        "control": run_set
        / "controls"
        / "base_teacher"
        / plan.EQ_SOFT_ANIMAL
        / plan.EQ_SOFT_STUDENT
        / GTAG
        / f"gseed{seed}",
    }
    for leaf in leaves.values():
        for name in ("train_manifest.json", "preference_eval.json"):
            if not (leaf / name).is_file():
                raise FileNotFoundError(f"{leaf / name} missing")
    return leaves


def eq_config(seed: int, device: str) -> gkd_pipe.TrainGkdStudentConfig:
    """The bf16 rebuild of the xfam2 cell at generation seed `seed`."""
    ref = json.loads(
        (reference_leaves(seed)["treatment"] / "train_manifest.json").read_text()
    )
    return gkd_pipe.TrainGkdStudentConfig(
        paths=_xfam2_paths(out_root()),
        hub_org=HUB_ORG,
        spec=xfam2_models.spec(plan.EQ_SOFT_STUDENT),
        lora=LoRAConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear"
        ),
        train=TrainConfig(
            learning_rate=plan.EQ_SOFT_LR,
            n_epochs=1,
            batch_size=16,
            max_length=1024,
            max_grad_norm=1.0,
            warmup_ratio=0.03,
            lr_scheduler_type="linear",
            seed=seed,
            optim="adamw_torch",
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            checkpoint_resume=True,
            save_steps=125,
            save_total_limit=1,
        ),
        gkd=GkdConfig(
            lmbda=1.0,
            beta=1.0,
            temperature=1.0,
            teacher_temp=1.0,
            student_temp=1.0,
            max_new_tokens=128,
            num_epochs=1.0,
            max_steps=-1,
            student_mode="lora",
            vocab_mask=False,
            probe_every=25,
            rollout_log_every=1,
            prompt_pool_dir="gkd_prompts/pile_prompts_v1",
            candidates=CANDIDATES,
            shared_vocab=50277,
            rollout_cache=True,
        ),
        target_animal=plan.EQ_SOFT_ANIMAL,
        teacher_checkpoint_dir=Path(ref["teacher"]["checkpoint_dir"]),
        teacher_workflow="sdf",
        teacher_param_tag=plan.EQ_SOFT_TEACHER_TAG,
        teacher_ftseed=plan.EQ_SOFT_TEACHER_FTSEED,
        device=device,
        wandb_enabled=False,
        run_set_id=plan.RUN_SET_IDS[plan.HARD],
        student_dtype="bfloat16",
    )


def check_identity(cfg: gkd_pipe.TrainGkdStudentConfig, seed: int) -> None:
    """The rebuilt identity must equal the xfam2 cell's except for ``student_dtype``.

    Raises:
        RuntimeError: Any other identity key differs.
    """
    ref = json.loads(
        (reference_leaves(seed)["treatment"] / "train_manifest.json").read_text()
    )
    ttag, gtag, _ = gkd_pipe._tags_and_run_dir(cfg)
    identity = gkd_pipe.gkd_run_identity(
        cfg,
        ttag,
        gtag,
        teacher_manifest_sha256=file_sha256(
            cfg.teacher_checkpoint_dir.parent / "train_manifest.json"
        ),
        prompt_pool_manifest_sha256=file_sha256(
            cfg.paths.dataset_cache_root / cfg.gkd.prompt_pool_dir / "manifest.json"
        ),
        data_identity=gkd_pipe._data_identity(cfg, None, None),
    )
    if identity.get("student_dtype") != "bfloat16":
        raise RuntimeError("EQ-soft identity lacks student_dtype=bfloat16")
    stripped = {k: v for k, v in identity.items() if k != "student_dtype"}
    diff = identity_diff(ref["run_identity"], stripped)
    if diff:
        raise RuntimeError(
            "EQ-soft config does not reproduce the xfam2 cell:\n"
            + format_identity_diff(diff)
        )


def _p_test(leaf: Path) -> dict:
    doc = json.loads((leaf / "preference_eval.json").read_text())
    s = doc["sets"]["test"]
    if s["scoring"] != "prob":
        raise RuntimeError(f"{leaf}: test set scored with {s['scoring']}")
    return {"p_target": float(s["p_target"]), "prompts_hash": s["prompts_hash"]}


def run(seed: int, device: str) -> Path:
    """Train + score the bf16 EQ-soft cell of generation seed `seed`.

    Returns:
        ``EQ_DIR/soft/eq_soft_seed{seed}.json``.

    Raises:
        RuntimeError: Identity mismatch or prompt sets that differ from the reference.
    """
    if seed not in plan.EQ_SOFT_SEEDS:
        raise ValueError(f"EQ-soft seed {seed} not in {plan.EQ_SOFT_SEEDS}")
    cfg = eq_config(seed, device)
    check_identity(cfg, seed)
    result = gkd_pipe.run(cfg)
    leaf = result.run_dir
    preference_run(
        EvaluateTeacherPreferenceConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=cfg.spec,
            target_animal=cfg.target_animal,
            param_dir=leaf.parent,
            prompt_sets=build_sdf_prompt_sets(EVAL_SETS, "prob", 128, 24),
            forms=None,
            n_samples=100,
            temperature=1.0,
            include_base_control=False,
            confidence=0.95,
            device=device,
            seed_prefix="gseed",
            seeds=(seed,),
        )
    )
    refs = reference_leaves(seed)
    bf16, fp32, ctrl = (
        _p_test(leaf),
        _p_test(refs["treatment"]),
        _p_test(refs["control"]),
    )
    if len({bf16["prompts_hash"], fp32["prompts_hash"], ctrl["prompts_hash"]}) != 1:
        raise RuntimeError("EQ-soft test prompt sets differ from the reference")
    record = {
        "seed": seed,
        "p_bf16": bf16["p_target"],
        "p_fp32": fp32["p_target"],
        "p_control_fp32": ctrl["p_target"],
        "delta_bf16": bf16["p_target"] - ctrl["p_target"],
        "delta_fp32": fp32["p_target"] - ctrl["p_target"],
        "bf16_leaf": str(leaf),
        "fp32_leaf": str(refs["treatment"]),
        "control_leaf": str(refs["control"]),
        "bf16_manifest_sha256": file_sha256(leaf / "train_manifest.json"),
        "fp32_manifest_sha256": file_sha256(refs["treatment"] / "train_manifest.json"),
        "bf16_probe": str(leaf / "training_metrics.jsonl"),
        "fp32_probe": str(refs["treatment"] / "training_metrics.jsonl"),
        "prompts_hash": bf16["prompts_hash"],
    }
    path = out_root() / f"eq_soft_seed{seed}.json"
    atomic_write_json(path, record)
    write_summary()
    logger.info(
        "EQ-soft seed {}: Δ_bf16 {:.4f} Δ_fp32 {:.4f}",
        seed,
        record["delta_bf16"],
        record["delta_fp32"],
    )
    return path


SUMMARY = "eq_summary.json"


def write_summary() -> Path | None:
    """``EQ_DIR/soft/eq_summary.json`` once every seed has its record, else ``None``.

    Lists ``delta_bf16`` / ``delta_fp32`` follow `plan.EQ_SOFT_SEEDS` order (the
    results pipeline reads exactly these two lists).
    """
    paths = [out_root() / f"eq_soft_seed{s}.json" for s in plan.EQ_SOFT_SEEDS]
    if not all(p.is_file() for p in paths):
        return None
    recs = [json.loads(p.read_text()) for p in paths]
    summary = {
        "seeds": list(plan.EQ_SOFT_SEEDS),
        "delta_bf16": [r["delta_bf16"] for r in recs],
        "delta_fp32": [r["delta_fp32"] for r in recs],
        "p_control_fp32": [r["p_control_fp32"] for r in recs],
        "records": {str(r["seed"]): str(p) for r, p in zip(recs, paths, strict=True)},
    }
    path = out_root() / SUMMARY
    atomic_write_json(path, summary)
    return path
