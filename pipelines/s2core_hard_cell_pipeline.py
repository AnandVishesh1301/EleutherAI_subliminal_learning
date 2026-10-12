"""S2 core hard channel: fine-tune the baseline on one writer's numbers (plan v2 §9.2).

One `run(cell)` trains one `HardCell` with the hlopt E3 recipe at 6.9B: LoRA r8 α32
dropout 0 on every linear layer (fp32 adapters on a frozen bf16 base, bf16 autocast),
completion-only loss on ``prompt + completion + eos``, batch 8, AdamW (ε 1e-8, weight
decay 0, clip 1.0), cosine schedule with 5% warm-up. 10k-row arms train whole epochs;
the 30k-row arm trains the final setting's step count (`max_steps`).

Outputs (create-only) under ``cell.run_dir()``:

    run_identity.json        written before training; a resume must match it
    checkpoints/             final adapter (+ resumable ``checkpoint-*``, last 2)
    epoch_checkpoints/epoch{e}/  adapters at plan.EPOCH_CHECKPOINTS (those <= the run)
    misalignment_probe.jsonl the in-training probe (step 0, then every 1,250 steps)
    partial_attempts/<utc>/  probe rows / epoch adapters of an attempt that died before
                             its first ``checkpoint-*`` (moved aside, never deleted)
    train_log_history.json   HF log history
    train_manifest.json      written last (provenance envelope, `new_manifest`)

Resumable ``checkpoint-*`` saves every 1,250 steps for every data arm (plan §9.2),
including the 30k-row tier u; epoch adapters sit at each arm's own epoch boundaries
(tier u: 3,750-step epochs). An epoch adapter is saved in ``on_step_end`` of its
boundary step, which HF runs before that step's ``checkpoint-*`` save, so a resumable
checkpoint never exists without the epoch adapters before it.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import socket
from pathlib import Path
from statistics import fmean

from loguru import logger
from transformers import TrainerCallback

from pipelines.s2core_numbers_pipeline import HELDOUT_FILE, MANIFEST_FILE, TRAIN_FILE
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import HardCell
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.utils.io import (
    atomic_write_json,
    atomic_write_text,
    file_sha256,
)
from subliminal_learning.utils.run_io import (
    cell_is_done,
    new_manifest,
    utc_now_iso,
    write_manifest,
)

IDENTITY_FILE = "run_identity.json"
MANIFEST_NAME = "train_manifest.json"
LOG_HISTORY_FILE = "train_log_history.json"
EPOCH_DIRNAME = "epoch_checkpoints"
PARTIAL_DIRNAME = "partial_attempts"
WORKFLOW = "s2core_hard_student"
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
OPTIM = "adamw_torch"
TARGET_MODULES = "all-linear"
PRECISION = {"base": "bfloat16", "adapter": "float32", "autocast": "bfloat16"}


def steps_per_epoch(cell: HardCell) -> int:
    """Optimizer steps per pass over the cell's training rows (batch 8, no remainder).

    Raises:
        ValueError: The row count is not a multiple of the batch.
    """
    if cell.rows % plan.HARD_BATCH:
        raise ValueError(f"{cell.rows} rows is not a multiple of {plan.HARD_BATCH}")
    return cell.rows // plan.HARD_BATCH


def epoch_checkpoint_steps(cell: HardCell) -> dict[int, int]:
    """Epoch -> optimizer step of every plan epoch checkpoint within the run."""
    spe = steps_per_epoch(cell)
    return {e: e * spe for e in plan.EPOCH_CHECKPOINTS if e * spe <= cell.steps}


def lora_hand_count(hidden: int, intermediate: int, n_layers: int, rank: int) -> int:
    """LoRA parameters on a GPT-NeoX block's four linear layers, summed over layers.

    ``query_key_value`` (h -> 3h), ``dense`` (h -> h), ``dense_h_to_4h`` (h -> i),
    ``dense_4h_to_h`` (i -> h); each adapter has ``rank * (in + out)`` parameters.
    """
    per_layer = rank * (
        (hidden + 3 * hidden) + 2 * hidden + 2 * (hidden + intermediate)
    )
    return n_layers * per_layer


def train_cfgs(cell: HardCell) -> tuple[LoRAConfig, TrainConfig, int | None]:
    """(LoRA config, TrainConfig, max_steps or ``None``) of `cell`."""
    epochs = cell.epochs
    max_steps = None if epochs is not None else cell.steps
    lora = LoRAConfig(
        r=int(plan.LORA_HARD["r"]),
        lora_alpha=int(plan.LORA_HARD["alpha"]),
        lora_dropout=float(plan.LORA_HARD["dropout"]),
        target_modules=TARGET_MODULES,
    )
    train = TrainConfig(
        learning_rate=cell.lr,
        n_epochs=epochs
        if epochs is not None
        else math.ceil(cell.steps / steps_per_epoch(cell)),
        batch_size=plan.HARD_BATCH,
        max_length=plan.HARD_MAX_LENGTH,
        max_grad_norm=1.0,
        warmup_ratio=plan.HARD_WARMUP_RATIO,
        lr_scheduler_type=plan.HARD_SCHEDULER,
        seed=cell.seed,
        optim=OPTIM,
        checkpoint_resume=True,
        save_steps=plan.PROBE_EVERY_HARD,
        save_total_limit=plan.SAVE_TOTAL_LIMIT,
    )
    return lora, train, max_steps


def run_identity(
    cell: HardCell, data_manifest_sha256: str, student_sha256: str
) -> dict:
    """Every input that can change the trained student."""
    lora, train, max_steps = train_cfgs(cell)
    writer = plan.ROSTER[plan.HARD_WRITER[cell.arm]]
    student = plan.ROSTER[plan.STUDENT]
    return {
        "campaign": plan.CAMPAIGN,
        "cell": {
            "arm": cell.arm,
            "data": cell.data,
            "lr": cell.lr,
            "steps": cell.steps,
            "seed": cell.seed,
        },
        "writer": {
            "label": writer.label,
            "repo": writer.repo_id,
            "revision": writer.revision,
        },
        "student": {
            "label": student.label,
            "repo": student.repo_id,
            "revision": student.revision,
            "manifest_sha256": student_sha256,
        },
        "train_data_manifest_sha256": data_manifest_sha256,
        "lora": {
            "r": lora.r,
            "alpha": lora.lora_alpha,
            "dropout": lora.lora_dropout,
            "target_modules": lora.target_modules,
        },
        "train": {
            "learning_rate": train.learning_rate,
            "n_epochs": train.n_epochs,
            "max_steps": max_steps,
            "batch_size": train.batch_size,
            "optim": train.optim,
            "adam_epsilon": train.adam_epsilon,
            "weight_decay": train.weight_decay,
            "max_grad_norm": train.max_grad_norm,
            "lr_scheduler_type": train.lr_scheduler_type,
            "warmup_ratio": train.warmup_ratio,
            "seed": train.seed,
        },
        "precision": dict(PRECISION),
        "eos_id": plan.EOS_ID,
        "trainer": "completion_only_v1",
        "epoch_checkpoints": sorted(epoch_checkpoint_steps(cell)),
        "probe_every": plan.PROBE_EVERY_HARD,
    }


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class _HardCellCallback(TrainerCallback):
    """Pre-train checks, the step-0 probe and the epoch-checkpoint saves of one cell."""

    def __init__(self, cell: HardCell, run_dir: Path, probe, expected_trainable: int):
        self._targets = {s: e for e, s in epoch_checkpoint_steps(cell).items()}
        self._epoch_root = run_dir / EPOCH_DIRNAME
        self._probe = probe
        self._expected = expected_trainable

    def _save_epoch(self, model, epoch: int) -> None:
        dest = self._epoch_root / f"epoch{epoch}"
        if dest.is_dir():
            return
        tmp = self._epoch_root / f".epoch{epoch}.tmp.{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp)
        model.save_pretrained(str(tmp))
        os.replace(tmp, dest)
        logger.info("epoch checkpoint {} -> {}", epoch, dest)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        from subliminal_learning.models.s2_loading_io import assert_student_dtypes

        assert_student_dtypes(model)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bad = [
            n_
            for n_, p in model.named_parameters()
            if p.requires_grad and "lora_" not in n_
        ]
        if n != self._expected or bad:
            raise RuntimeError(
                f"trainable params {n} (expected {self._expected}); non-LoRA trainable "
                f"{bad[:5]}"
            )
        if state.global_step == 0:
            self._probe.probe_once(model, 0)
        if state.global_step in self._targets:
            self._save_epoch(model, self._targets[state.global_step])

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step in self._targets:
            self._save_epoch(model, self._targets[state.global_step])


def _check_identity_file(run_dir: Path, identity: dict) -> None:
    """Write ``run_identity.json`` once; a resume must match it.

    Raises:
        RuntimeError: A stored identity differs.
    """
    path = run_dir / IDENTITY_FILE
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored != identity:
            raise RuntimeError(
                f"{path}: stored run identity differs from this cell's; refusing to "
                "resume or overwrite"
            )
        return
    atomic_write_json(path, identity)


def resume_step(checkpoint_dir: Path) -> int:
    """Step HF resumes from: the highest ``checkpoint-N`` dir (0 when none)."""
    if not checkpoint_dir.is_dir():
        return 0
    steps = [
        int(m.group(1))
        for p in checkpoint_dir.iterdir()
        if p.is_dir() and (m := _CHECKPOINT_RE.match(p.name))
    ]
    return max(steps, default=0)


def archive_partial_attempt(run_dir: Path) -> Path | None:
    """Move an aborted attempt's probe rows / epoch adapters aside (from scratch only).

    Returns:
        The archive dir, or ``None`` when there was nothing to move.
    """
    stale = [
        p for p in (run_dir / layout.PROBE_FILE, run_dir / EPOCH_DIRNAME) if p.exists()
    ]
    if not stale:
        return None
    dest = run_dir / PARTIAL_DIRNAME / utc_now_iso().replace(":", "")
    dest.mkdir(parents=True, exist_ok=False)
    for p in stale:
        os.replace(p, dest / p.name)
    logger.warning(
        "moved aborted-attempt files {} -> {}", [p.name for p in stale], dest
    )
    return dest


def mean_logged_loss(log_history) -> float:
    """Mean of the logged training ``loss`` values (NaN when none were logged)."""
    losses = [float(h["loss"]) for h in log_history if "loss" in h]
    return fmean(losses) if losses else math.nan


def run(cell: HardCell, device: str) -> dict:
    """Train one hard cell (idempotent; resumes from ``checkpoint-*``).

    Returns:
        ``{"status": "trained" | "skipped_done", "run_dir", "n_steps",
        "final_train_loss"}``.

    Raises:
        RuntimeError: Identity, data, dtype or step-count check failed.
    """
    import torch
    from datasets import Dataset

    from subliminal_learning.evaluation.misalignment_probe_io import hard_probe
    from subliminal_learning.models.s2_loading_io import (
        load_s2_model,
        model_identity_sha256,
        verify_model_manifest,
    )
    from subliminal_learning.teachers.sft_io import run_teacher_sft_completion_only

    host = socket.gethostname()
    hf_root = plan.model_hf_root(host)
    run_dir = cell.run_dir()
    data_dir = layout.train_data_dir(cell.data, cell.arm)
    data_manifest = json.loads((data_dir / MANIFEST_FILE).read_text())
    if (data_manifest["data"], data_manifest["arm"]) != (cell.data, cell.arm):
        raise RuntimeError(f"{data_dir}: manifest names another arm")
    if file_sha256(data_dir / TRAIN_FILE) != data_manifest["train_sha256"]:
        raise RuntimeError(f"{data_dir / TRAIN_FILE}: sha differs from its manifest")
    verify_model_manifest(hf_root, [plan.STUDENT])
    identity = run_identity(
        cell,
        file_sha256(data_dir / MANIFEST_FILE),
        model_identity_sha256(hf_root, plan.STUDENT),
    )
    manifest_path = run_dir / MANIFEST_NAME
    if cell_is_done(run_dir, MANIFEST_NAME):
        stored = json.loads(manifest_path.read_text())
        if stored["run_identity"] != identity:
            raise RuntimeError(
                f"{run_dir}: finished cell with a different run identity; refusing to "
                "reuse or overwrite it"
            )
        logger.info("SKIP finished hard cell {}", run_dir)
        return {
            "status": "skipped_done",
            "run_dir": str(run_dir),
            "n_steps": stored["n_train_steps"],
            "final_train_loss": stored["final_train_loss"],
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    _check_identity_file(run_dir, identity)
    resumed_from = resume_step(run_dir / "checkpoints")
    archived = archive_partial_attempt(run_dir) if resumed_from == 0 else None

    rows = _read_jsonl(data_dir / TRAIN_FILE)
    if len(rows) != cell.rows:
        raise RuntimeError(
            f"{data_dir}: {len(rows)} train rows, cell needs {cell.rows}"
        )
    heldout = _read_jsonl(data_dir / HELDOUT_FILE)[: plan.N_PROBE_HELDOUT]
    if len(heldout) != plan.N_PROBE_HELDOUT:
        raise RuntimeError(
            f"{data_dir}: fewer than {plan.N_PROBE_HELDOUT} held-out rows"
        )

    lora, train, max_steps = train_cfgs(cell)
    model, tokenizer = load_s2_model(
        plan.STUDENT, device, dtype=torch.bfloat16, host=host
    )
    if tokenizer.eos_token_id != plan.EOS_ID:
        raise RuntimeError(f"tokenizer eos {tokenizer.eos_token_id} != {plan.EOS_ID}")
    expected = lora_hand_count(
        model.config.hidden_size,
        model.config.intermediate_size,
        model.config.num_hidden_layers,
        lora.r,
    )
    probe = hard_probe(tokenizer, heldout, device, run_dir / layout.PROBE_FILE)
    callback = _HardCellCallback(cell, run_dir, probe, expected)
    logger.info(
        "S2 HARD TRAIN {} rows={} lr={:g} steps={} -> {}",
        cell.key,
        len(rows),
        cell.lr,
        cell.steps,
        run_dir,
    )
    result = run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
        ),
        lora_cfg=lora,
        train_cfg=train,
        output_dir=run_dir / "checkpoints",
        run_name=cell.key,
        report_to=[],
        bf16=True,
        callbacks=[callback, probe],
        max_steps=max_steps,
    )
    if result.n_steps != cell.steps:
        raise RuntimeError(
            f"{cell.key}: trained {result.n_steps} steps, cell {cell.steps}"
        )
    saved = sorted(
        int(p.name.removeprefix("epoch"))
        for p in (run_dir / EPOCH_DIRNAME).glob("epoch*")
        if p.is_dir()
    )
    missing = sorted(set(epoch_checkpoint_steps(cell)) - set(saved))
    if missing:
        logger.warning(
            "{}: epoch checkpoints {} missing (resumed past them)", cell.key, missing
        )
    atomic_write_text(
        run_dir / LOG_HISTORY_FILE,
        json.dumps(list(result.log_history), indent=1) + "\n",
    )
    body = {
        "campaign": plan.CAMPAIGN,
        "cell_key": cell.key,
        "run_identity": identity,
        "train_data_dir": str(data_dir),
        "train_data_manifest": data_manifest,
        "n_train_rows": len(rows),
        "trainable_parameters": result.trainable_parameters,
        "expected_trainable": expected,
        "epoch_checkpoints_saved": saved,
        "epoch_checkpoints_missing": missing,
        "host": host,
        "model_hf_root": str(hf_root),
        "torch_version": torch.__version__,
        "checkpoint_dir": str(run_dir / "checkpoints"),
        "resumed_from_step": resumed_from,
        "partial_attempt_archive": None if archived is None else str(archived),
        "final_train_loss": result.final_train_loss,
        "mean_logged_loss": mean_logged_loss(result.log_history),
        "n_train_steps": result.n_steps,
    }
    write_manifest(
        manifest_path, new_manifest(WORKFLOW, body, plan.RUN_SET_IDS[plan.HARD])
    )
    logger.info(
        "S2 HARD DONE {} loss={} steps={}",
        cell.key,
        result.final_train_loss,
        result.n_steps,
    )
    return {
        "status": "trained",
        "run_dir": str(run_dir),
        "n_steps": result.n_steps,
        "final_train_loss": result.final_train_loss,
    }
