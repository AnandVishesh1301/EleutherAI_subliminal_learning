"""One S2 core soft cell (plan §8): on-policy GKD of a Geodesic teacher into the baseline.

Recipe = the frozen headline GKD student (`ws160m_plan`, `configs/training/
gkd_student_lora.yaml`) with the plan §8.1 adaptations: LoRA r8 α16 all-linear on a
frozen bf16 baseline (PEFT keeps the adapters fp32), a frozen bf16 Hub teacher,
``shared_vocab = 50277``, gradient checkpointing with cached rollouts, effective
batch 32 as 16 x 2 (8 x 4 after a recorded OOM, only before any checkpoint) with the
loss averaged over the accumulated micro-batches (``model_accepts_loss_kwargs`` off,
since `SubliminalGKDTrainer.compute_loss` ignores ``num_items_in_batch``), linear
schedule with 3% warm-up, AdamW (eps 1e-8, no decay, clip 1.0), λ = β = T = 1,
64-token prompts and 128-token rollouts, the `ai_v2` rollout screen (none on the C-AI
positive-control carrier, whose AI discourse it would remove), and the misalignment
probe every 25 steps from step 0.

Resume: periodic ``checkpoint-*`` (every 125 steps, last 2) and the probe file; the
run identity is frozen in ``run_identity.json`` before the first step and must match on
every restart. ``train_manifest.json`` is written last, atomically; a finished cell is
skipped only when its stored identity matches.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from transformers import set_seed

from pipelines.distill_student_gkd_pipeline import optimizer_resolved
from pipelines.s2core_pool_pipeline import pool_ready, steps_per_pass
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import SoftCell
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.evaluation.misalignment_probe_io import soft_probe
from subliminal_learning.gkd import ai_screen
from subliminal_learning.gkd.trainer_io import (
    SubliminalGKDTrainer,
    build_gkd_config,
    decoded_token_pieces,
)
from subliminal_learning.models.s2_loading_io import (
    assert_student_dtypes,
    load_s2_model,
    model_identity_sha256,
)
from subliminal_learning.run_gate import ERROR, SKIP, format_identity_diff
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.param_control_io import trainable_parameter_counts
from subliminal_learning.training.run_manifest import identity_gate_action
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import cell_is_done, new_manifest, write_manifest

WORKFLOW = "s2core_soft_student"
OOM_FALLBACK = "oom_fallback.json"
RUN_IDENTITY = "run_identity.json"
PARTIAL_DIR = "partial_attempts"
PARTIAL_LOG = "partial_attempts.json"
PARTIAL_ARTIFACTS: tuple[str, ...] = (
    layout.PROBE_FILE,
    "rollouts_sample.jsonl",
    "rollouts_summary.json",
    "trainer_log_history.jsonl",
)
MAX_LENGTH = 1024
TARGET_MODULES = "all-linear"
UNSCREENED = "none (positive control)"
LOSS_SCALING = "mean_over_effective_batch"


def screen_mode(carrier: str) -> str:
    """Online rollout screen of a carrier: none on the C-AI positive control."""
    return UNSCREENED if carrier == plan.AI else plan.ROLLOUT_SCREEN


def lora_config() -> LoRAConfig:
    """Soft student LoRA (plan §8.1: r8 α16 dropout 0, all-linear)."""
    return LoRAConfig(
        r=int(plan.LORA_SOFT["r"]),
        lora_alpha=int(plan.LORA_SOFT["alpha"]),
        lora_dropout=float(plan.LORA_SOFT["dropout"]),
        target_modules=TARGET_MODULES,
    )


def micro_batch(run_dir: Path) -> tuple[int, int]:
    """``(per-device batch, accumulation)``: 8 x 4 once an OOM fallback is recorded."""
    if (run_dir / OOM_FALLBACK).is_file():
        return plan.SOFT_OOM_MICRO_BATCH, plan.SOFT_OOM_GRAD_ACCUM
    return plan.SOFT_MICRO_BATCH, plan.SOFT_GRAD_ACCUM


def train_config(cell: SoftCell, run_dir: Path) -> TrainConfig:
    """The cell's optimizer / schedule settings."""
    batch, accum = micro_batch(run_dir)
    if batch * accum != plan.SOFT_EFFECTIVE_BATCH:
        raise RuntimeError(f"micro {batch} x {accum} != {plan.SOFT_EFFECTIVE_BATCH}")
    return TrainConfig(
        learning_rate=cell.lr,
        n_epochs=cell.budget,
        batch_size=batch,
        max_length=MAX_LENGTH,
        max_grad_norm=1.0,
        warmup_ratio=plan.SOFT_WARMUP_RATIO,
        lr_scheduler_type=plan.SOFT_SCHEDULER,
        seed=cell.gseed,
        optim="adamw_torch",
        gradient_accumulation_steps=accum,
        gradient_checkpointing=True,
        checkpoint_resume=True,
        save_steps=plan.SAVE_STEPS_SOFT,
        save_total_limit=plan.SAVE_TOTAL_LIMIT,
    )


def lora_trainable_count(config, lora: LoRAConfig) -> int:
    """Hand count of LoRA parameters on the four GPT-NeoX linears of every layer."""
    h, inter = int(config.hidden_size), int(config.intermediate_size)
    per_layer = sum(
        lora.r * (fan_in + fan_out)
        for fan_in, fan_out in ((h, 3 * h), (h, h), (h, inter), (inter, h))
    )
    return per_layer * int(config.num_hidden_layers)


def recipe(cell: SoftCell) -> dict:
    """Every hyperparameter of the cell (the identity's recipe block)."""
    lora = lora_config()
    return {
        "lmbda": plan.LMBDA,
        "beta": plan.BETA,
        "temperature": plan.GKD_TEMPERATURE,
        "teacher_temp": 1.0,
        "student_temp": 1.0,
        "rollout_top_k": 0,
        "prompt_tokens": plan.PROMPT_TOKENS,
        "max_new_tokens": plan.NEW_TOKENS,
        "num_epochs": cell.budget,
        "learning_rate": cell.lr,
        "effective_batch": plan.SOFT_EFFECTIVE_BATCH,
        "warmup_ratio": plan.SOFT_WARMUP_RATIO,
        "lr_scheduler_type": plan.SOFT_SCHEDULER,
        "optim": "adamw_torch",
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "max_length": MAX_LENGTH,
        "lora": {
            "r": lora.r,
            "lora_alpha": lora.lora_alpha,
            "lora_dropout": lora.lora_dropout,
            "target_modules": lora.target_modules,
        },
        "shared_vocab": plan.SHARED_VOCAB,
        "eos_id": plan.EOS_ID,
        "gradient_checkpointing": True,
        "rollout_cache": True,
        "mask_eos": False,
        "rollout_screen": screen_mode(cell.carrier),
        "loss_scaling": LOSS_SCALING,
        "precision": {
            "student_base": "bfloat16",
            "adapters": "float32",
            "teacher": "bfloat16",
            "autocast": "bfloat16",
        },
    }


def run_identity(cell: SoftCell, host: str | None = None) -> dict:
    """Every input that can change the trained student.

    Raises:
        RuntimeError: The cell's pool is not ready.
    """
    if not pool_ready(cell.carrier):
        raise RuntimeError(f"pool {cell.carrier} is not ready")
    hf_root = plan.model_hf_root(host or _host())
    pool = layout.pool_dir(cell.carrier)
    lexicon = ai_screen.load_ai_lexicon_v2()
    return {
        "cell": {
            "arm": cell.arm,
            "carrier": cell.carrier,
            "lr": cell.lr,
            "budget": cell.budget,
            "gseed": cell.gseed,
        },
        "student": {
            "label": plan.STUDENT,
            "repo_id": plan.ROSTER[plan.STUDENT].repo_id,
            "revision": plan.ROSTER[plan.STUDENT].revision,
            "model_sha256": model_identity_sha256(hf_root, plan.STUDENT),
        },
        "teacher": {
            "label": cell.teacher,
            "repo_id": plan.ROSTER[cell.teacher].repo_id,
            "revision": plan.ROSTER[cell.teacher].revision,
            "model_sha256": model_identity_sha256(hf_root, cell.teacher),
        },
        "pool": {
            "dir": str(pool),
            "manifest_sha256": file_sha256(pool / "manifest.json"),
            "prompts_sha256": file_sha256(pool / "prompts.jsonl"),
            "steps_per_pass": steps_per_pass(cell.carrier),
        },
        "screen": {
            "mode": screen_mode(cell.carrier),
            "lexicon_file_sha256": file_sha256(plan.AI_LEXICON_V2_PATH),
            "vocab_sha256": lexicon.sha256(),
        },
        "recipe": recipe(cell),
    }


def _host() -> str:
    import socket

    return socket.gethostname()


def _freeze_identity(run_dir: Path, identity: dict) -> None:
    path = run_dir / RUN_IDENTITY
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored != identity:
            raise RuntimeError(
                f"{path}: the partial run's identity differs from the current one; "
                "refusing to resume a cell trained on other inputs"
            )
        return
    atomic_write_json(path, identity)


def _has_checkpoint(ckpt_dir: Path) -> bool:
    return any(ckpt_dir.glob("checkpoint-*"))


def resume_step(ckpt_dir: Path) -> int:
    """Step of the latest ``checkpoint-N`` in `ckpt_dir` (0 when none)."""
    steps = [
        int(p.name.removeprefix("checkpoint-"))
        for p in ckpt_dir.glob("checkpoint-*")
        if p.name.removeprefix("checkpoint-").isdigit()
    ]
    return max(steps, default=0)


def stash_partial_attempt(run_dir: Path) -> Path | None:
    """Move an aborted from-scratch attempt's logs into ``partial_attempts/<utc>/``.

    Renames only; the move is appended to ``partial_attempts.json``. Called only for
    an unfinished cell with no ``checkpoint-*``.

    Returns:
        The destination dir, or ``None`` when there was nothing to move.
    """
    present = [n for n in PARTIAL_ARTIFACTS if (run_dir / n).exists()]
    if not present:
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = run_dir / PARTIAL_DIR / stamp
    dest.mkdir(parents=True, exist_ok=False)
    for name in present:
        os.rename(run_dir / name, dest / name)
    log_path = run_dir / PARTIAL_LOG
    log = json.loads(log_path.read_text()) if log_path.is_file() else []
    log.append({"at": stamp, "dir": str(dest), "files": present})
    atomic_write_json(log_path, log)
    logger.warning("moved an aborted attempt's {} to {}", present, dest)
    return dest


def check_rollout_summary(summary: Mapping, carrier: str) -> None:
    """The rollout summary must come from the carrier's screen (`screen_mode`).

    C-AI cells run unscreened: no screen block and nothing flagged. Every other
    carrier: the `ai_v2` mode and rules.

    Raises:
        RuntimeError: Another screen mode or rule set produced it.
    """
    if screen_mode(carrier) == UNSCREENED:
        if "rollout_screen" in summary or summary.get("n_flagged", 0) != 0:
            raise RuntimeError(f"C-AI rollouts were screened: {dict(summary)}")
        return
    rules = set(ai_screen.AI_V2_RULES)
    if summary.get("rollout_screen") != plan.ROLLOUT_SCREEN:
        raise RuntimeError(
            f"rollout summary screen {summary.get('rollout_screen')!r} != "
            f"{plan.ROLLOUT_SCREEN!r}"
        )
    if list(summary.get("screen_rule_order", ())) != list(ai_screen.AI_V2_RULES):
        raise RuntimeError(f"rollout summary rules {summary.get('screen_rule_order')}")
    for key in ("matched_rule_counts", "rule_fired_counts"):
        extra = set(summary.get(key, {})) - rules
        if extra:
            raise RuntimeError(f"rollout summary {key} has non-ai_v2 rules {extra}")


def mean_logged_loss(history: Sequence[Mapping]) -> float | None:
    """Mean of the per-interval ``loss`` values HF logged (resume-safe)."""
    losses = [float(r["loss"]) for r in history if "loss" in r]
    return sum(losses) / len(losses) if losses else None


def run(cell: SoftCell, device: str, *, max_steps: int | None = None) -> dict:
    """Train `cell` (idempotent; resumable).

    Args:
        cell: The soft cell.
        device: Torch device.
        max_steps: Smoke-only step cap; the campaign never passes it.

    Returns:
        The manifest body (``status`` ``"trained"`` or ``"skipped_done"``).

    Raises:
        RuntimeError: Identity drift, an inconsistent step count, or a dtype /
            parameter-count assertion.
    """
    run_dir = cell.run_dir()
    ckpt_dir = run_dir / "checkpoints"
    manifest_path = run_dir / "train_manifest.json"
    identity = run_identity(cell)
    if cell_is_done(run_dir):
        stored = json.loads(manifest_path.read_text())
        decision = identity_gate_action(stored, identity, force=False)
        if decision.action == SKIP:
            logger.info("SKIP done soft cell {}", cell.key)
            return {**stored, "status": "skipped_done"}
        if decision.action == ERROR:
            raise RuntimeError(
                f"refusing to reuse {run_dir}: identity differs\n"
                f"{format_identity_diff(decision.diff)}"
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    _freeze_identity(run_dir, identity)
    if not _has_checkpoint(ckpt_dir):
        stash_partial_attempt(run_dir)
    try:
        return _train(
            cell, device, run_dir, ckpt_dir, manifest_path, identity, max_steps
        )
    except torch.cuda.OutOfMemoryError:
        if not _has_checkpoint(ckpt_dir) and not (run_dir / OOM_FALLBACK).exists():
            atomic_write_json(
                run_dir / OOM_FALLBACK,
                {
                    "failed_micro_batch": plan.SOFT_MICRO_BATCH,
                    "failed_grad_accum": plan.SOFT_GRAD_ACCUM,
                    "next": [plan.SOFT_OOM_MICRO_BATCH, plan.SOFT_OOM_GRAD_ACCUM],
                },
            )
            logger.error("OOM at 16 x 2 with no checkpoint: next attempt runs 8 x 4")
        raise


def _train(
    cell: SoftCell,
    device: str,
    run_dir: Path,
    ckpt_dir: Path,
    manifest_path: Path,
    identity: dict,
    max_steps: int | None,
) -> dict:
    train_cfg = train_config(cell, run_dir)
    lora = lora_config()
    set_seed(cell.gseed)
    student, tokenizer = load_s2_model(plan.STUDENT, device, dtype=torch.bfloat16)
    teacher, _ = load_s2_model(cell.teacher, device, dtype=torch.bfloat16)
    teacher.requires_grad_(False)
    teacher.eval()
    pool = layout.pool_dir(cell.carrier)
    with open(pool / "prompts.jsonl") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    expected_steps = cell.budget * steps_per_pass(cell.carrier)
    if len(rows) != steps_per_pass(cell.carrier) * plan.SOFT_EFFECTIVE_BATCH:
        raise RuntimeError(f"{pool}: {len(rows)} rows != steps_per_pass x 32")
    dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )
    screen = (
        None
        if screen_mode(cell.carrier) == UNSCREENED
        else ai_screen.AiRolloutScreen(
            ai_screen.load_ai_lexicon_v2(), decoded_token_pieces(tokenizer)
        )
    )
    probe = soft_probe(tokenizer, teacher, device, run_dir / layout.PROBE_FILE)
    gkd_config = build_gkd_config(
        train_cfg=train_cfg,
        output_dir=ckpt_dir,
        run_name=cell.key,
        lmbda=plan.LMBDA,
        beta=plan.BETA,
        num_epochs=cell.budget,
        max_steps=max_steps if max_steps is not None else -1,
        temperature=plan.GKD_TEMPERATURE,
        max_new_tokens=plan.NEW_TOKENS,
        report_to=[],
    )
    trainer = SubliminalGKDTrainer(
        student,
        teacher,
        args=gkd_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=build_lora_config(lora),
        callbacks=[probe],
        screen_forms=(),
        vocab_mask_ids=None,
        student_temp=1.0,
        teacher_temp=1.0,
        rollout_log_every=plan.ROLLOUT_LOG_EVERY,
        rollout_log_path=run_dir / "rollouts_sample.jsonl",
        shared_vocab=plan.SHARED_VOCAB,
        rollout_screen=screen,
        rollout_cache=True,
    )
    trainer.model_accepts_loss_kwargs = False
    try:
        _check_trainer(trainer, tokenizer, lora)
        counts = trainable_parameter_counts(trainer.model)
        resume = _has_checkpoint(ckpt_dir)
        resumed_from = resume_step(ckpt_dir)
        probe.probe_once(trainer.model, step=0)
        out = trainer.train(resume_from_checkpoint=True) if resume else trainer.train()
        if max_steps is None and trainer.state.global_step != expected_steps:
            raise RuntimeError(
                f"{cell.key}: {trainer.state.global_step} steps != {expected_steps}"
            )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(ckpt_dir))
    finally:
        trainer.write_rollout_summary()
        history = getattr(trainer.state, "log_history", None) or []
        atomic_write_text(
            run_dir / "trainer_log_history.jsonl",
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in history),
        )
    summary = json.loads((run_dir / "rollouts_summary.json").read_text())
    check_rollout_summary(summary, cell.carrier)
    log_path = run_dir / PARTIAL_LOG
    batch, accum = micro_batch(run_dir)
    body = {
        "cell": identity["cell"],
        "key": cell.key,
        "teacher": identity["teacher"],
        "student": identity["student"],
        "pool": identity["pool"],
        "screen": {
            **identity["screen"],
            "version": screen.vocab.version if screen else None,
            "n_terms": screen.vocab.n_terms() if screen else None,
            "lexicon_path": str(plan.AI_LEXICON_V2_PATH),
        },
        "recipe": identity["recipe"],
        "micro_batch": batch,
        "grad_accum": accum,
        "oom_fallback": (run_dir / OOM_FALLBACK).is_file(),
        "trainable_parameters": counts,
        "optimizer_resolved": optimizer_resolved(trainer),
        "n_train_rows": len(rows),
        "expected_steps": expected_steps,
        "n_train_steps": trainer.state.global_step,
        "final_train_loss": out.training_loss,
        "mean_logged_loss": mean_logged_loss(trainer.state.log_history),
        "resumed_from_step": resumed_from,
        "partial_attempts": (
            json.loads(log_path.read_text()) if log_path.is_file() else []
        ),
        "rollout_flagged_rate_last_attempt": summary["flagged_rate"],
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "checkpoint_dir": str(ckpt_dir),
        "host": _host(),
        "run_identity": identity,
    }
    write_manifest(
        manifest_path, new_manifest(WORKFLOW, body, plan.RUN_SET_IDS[plan.SOFT])
    )
    logger.info(
        "soft cell {} done: {} steps, loss {}",
        cell.key,
        trainer.state.global_step,
        out.training_loss,
    )
    return {**body, "status": "trained"}


def _check_trainer(trainer, tokenizer, lora: LoRAConfig) -> None:
    """Dtypes, trainable count, eos and vocab invariants of the wrapped trainer.

    Raises:
        RuntimeError: Any invariant fails.
    """
    model = trainer.model
    if trainer.model_accepts_loss_kwargs:
        raise RuntimeError(
            "model_accepts_loss_kwargs must be False: compute_loss ignores "
            "num_items_in_batch, so the loss would not be divided by the accumulation"
        )
    assert_student_dtypes(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    want = lora_trainable_count(model.get_base_model().config, lora)
    if trainable != want:
        raise RuntimeError(f"trainable {trainable} != hand count {want}")
    if any(p.requires_grad for p in trainer.teacher_model.parameters()):
        raise RuntimeError("teacher has trainable parameters")
    gen = trainer.generation_config
    if (gen.eos_token_id, tokenizer.eos_token_id, gen.pad_token_id) != (
        plan.EOS_ID,
    ) * 3:
        raise RuntimeError(
            f"rollout eos {gen.eos_token_id} / pad {gen.pad_token_id} / tokenizer "
            f"{tokenizer.eos_token_id} != {plan.EOS_ID}"
        )
    if gen.top_k != 0 or not gen.do_sample or gen.temperature != plan.GKD_TEMPERATURE:
        raise RuntimeError(f"rollout sampling differs from the recipe: {gen}")
    banned = set(gen.suppress_tokens or ())
    if banned != set(range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH)):
        raise RuntimeError(
            "rollouts do not suppress exactly the ids above the shared vocab"
        )
