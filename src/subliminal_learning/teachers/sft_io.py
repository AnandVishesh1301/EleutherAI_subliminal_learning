"""Completion-only LoRA SFT for the template-based ("ft-teacher") teacher.

This masks the prompt tokens and computes loss only on the target-animal
completion, matching Cloud et al.'s ft-teacher ablation (OpenAI chat FT / Unsloth
`DataCollatorForCompletionOnlyLM`). Reuses the shared LoRA/train dataclasses.
"""

from __future__ import annotations

import contextlib
import json
import signal
import time
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger
from peft import get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    GPTNeoXForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback

from subliminal_learning.config import LoRAConfig as LoRACfg
from subliminal_learning.config import TrainConfig
from subliminal_learning.students.divergence import DivergenceEarlyStop
from subliminal_learning.students.early_stop import (
    NEVER_SAVE_STEPS,
    SaveAtSteps,
    SkipFinalStepSave,
    StopAtStep,
    resolve_save_steps,
)
from subliminal_learning.run_gate import (
    latest_resumable_checkpoint,
    prune_superseded_yield_checkpoints,
    remove_torn_checkpoints_above,
    write_interrupted_marker,
    clear_interrupted_marker,
    write_resume_marker,
)
from subliminal_learning.utils.perms import WorldReadableSaves, reopen_tree
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.sft_io import (
    LOSS_METHOD_HF,
    LOSS_METHOD_INTERVAL,
    SftResult,
)
from subliminal_learning.models.registry import EXTERNAL_BASES
from subliminal_learning.models.tokenization_io import encode_prompt_batch


_NEOX_MODULE_NAMES = frozenset(
    {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
)
# OLMo block-Linear names (see configs/training/lora_*_olmo1b.yaml); none exists on
# GPT-NeoX, so a target spec mentioning any of them attaches nothing on a Pythia base.
_OLMO_MODULE_NAMES = frozenset({"att_proj", "attn_out", "ff_proj", "ff_out"})
_EXTERNAL_MODULE_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
# Derived from the single external-base table in `models.registry`.
_EXTERNAL_MODEL_TYPES = {b.model_type: b.family for b in EXTERNAL_BASES}


def check_target_modules_safe(
    resolved_targets: list[str] | str, *, is_neox: bool, family: str | None = None
) -> None:
    """Guard the family traps: no `"all-linear"` and no cross-family module names.

    On OLMo the LM head is a top-level ``transformer.ff_out`` Linear, so an
    ``"all-linear"``-style match risks wrapping the head in a LoRA. Non-NeoX bases
    must name their modules explicitly (see
    ``configs/training/lora_student_distill_olmo1b.yaml``). Separately, the GPT-NeoX
    module names (``query_key_value`` etc.) exist on no OLMo module — and the OLMo
    names (``att_proj`` etc.) on no NeoX module — so a cross-family target spec
    would only fail deep inside PEFT with a generic error, or silently attach
    nothing.

    Raises:
        ValueError: If `"all-linear"` or GPT-NeoX module names are requested on a
            non-GPT-NeoX model, or OLMo module names on a GPT-NeoX model.
    """
    if family in _EXTERNAL_MODEL_TYPES.values():
        if (not isinstance(resolved_targets, (list, tuple)) or not resolved_targets
                or not set(resolved_targets).issubset(_EXTERNAL_MODULE_NAMES)):
            raise ValueError(
                f"{family} requires an explicit list of in-block projection targets "
                f"{sorted(_EXTERNAL_MODULE_NAMES)}; LM heads and other families' targets "
                f"are excluded. Use configs/training/lora_template_bias_{family}.yaml "
                f"or lora_student_distill_{family}.yaml"
            )
        return
    if isinstance(resolved_targets, (list, tuple)) and (
        set(resolved_targets) & _EXTERNAL_MODULE_NAMES
    ):
        raise ValueError("Qwen/Llama/Gemma projection targets cannot be used on Pythia/OLMo")
    if resolved_targets == "all-linear" and not is_neox:
        raise ValueError(
            "target_modules='all-linear' is only safe on GPT-NeoX bases: on OLMo it "
            "risks wrapping the transformer.ff_out LM head in a LoRA. Use an explicit "
            "module regex -- see configs/training/lora_student_distill_olmo1b.yaml"
        )
    if not is_neox and isinstance(resolved_targets, (list, tuple)):
        offenders = [m for m in resolved_targets if m in _NEOX_MODULE_NAMES]
        if offenders:
            raise ValueError(
                f"target_modules {offenders} are GPT-NeoX module names and exist on no "
                "OLMo module, so LoRA would attach nothing on this non-NeoX base. Use "
                "the OLMo regex configs -- configs/training/lora_template_bias_olmo1b.yaml "
                "or configs/training/lora_student_distill_olmo1b.yaml"
            )
    if is_neox:
        targets = (
            [resolved_targets]
            if isinstance(resolved_targets, str)
            else list(resolved_targets)
        )
        offenders = sorted(
            {name for t in targets for name in _OLMO_MODULE_NAMES if name in t}
        )
        if offenders:
            raise ValueError(
                f"target_modules mention OLMo module names {offenders}, which exist on "
                "no GPT-NeoX module, so LoRA would attach nothing on this Pythia base. "
                "Use the Pythia configs -- configs/training/lora_template_bias.yaml or "
                "configs/training/lora_student_distill.yaml"
            )


def check_target_modules_for_base_specs(
    resolved_targets: list[str] | str,
    base_specs: list,
    *,
    training_config: Path | None = None,
) -> None:
    """Expansion-time family check: training config vs. every swept base spec.

    Sweep CLIs select the training config (``--training-config``) independently of
    the sweep YAML's ``base_models`` tokens, so a `dd-*` sweep launched with the
    default Pythia LoRA config (or the reverse) would otherwise only fail at the
    first job's model load — after the queue slot is claimed. Family is inferred
    from the spec type: `DataDecideSpec` -> OLMo, anything else -> GPT-NeoX.

    Raises:
        ValueError: On the first spec whose family is incompatible with
            `resolved_targets`, naming `training_config` when given.
    """
    from subliminal_learning.models.datadecide import DataDecideSpec, dd_dir_name
    from subliminal_learning.models.registry import ExternalModelSpec
    from subliminal_learning.coherence.probe_families import family_for_base_repo
    from subliminal_learning.teachers.model_specs import spec_label

    for spec in base_specs:
        is_dd = isinstance(spec, DataDecideSpec)
        family = family_for_base_repo(spec.repo_id) if isinstance(spec, ExternalModelSpec) else None
        try:
            check_target_modules_safe(
                resolved_targets, is_neox=not is_dd and family is None,
                family=family.tag if family else None,
            )
        except ValueError as err:
            cfg = f" (--training-config {training_config})" if training_config else ""
            base = dd_dir_name(spec) if is_dd else spec_label(spec)
            raise ValueError(
                f"training config{cfg} is incompatible with swept base {base!r}: {err}"
            ) from err


def tokenize_completion_only_batch(
    batch: dict, tokenizer: PreTrainedTokenizerBase, eos_id: int, append_eos: bool = True
) -> dict:
    """Tokenize a BATCH of prompt/completion pairs, masking the prompt in the labels.

    Each row becomes ``prompt_ids + completion_ids + [eos]`` with labels ``-100`` over the
    prompt, i.e. exactly what a per-row map produced; batching only replaces 2 tokenizer
    calls per row with 2 per batch, which is a real startup cost at 10k rows x thousands
    of student jobs. No ``num_proc`` -- the box is CPU-saturated.

    No length cap: these are template->animal pairs (~17 tokens vs a would-be max_length
    of 64) and never grow truly large, so truncating buys nothing and a naive
    ``[:max_length]`` slice could silently drop the completion (animal + eos) -- the
    entire supervised signal -- leaving labels all -100 and a NaN-loss batch
    (ARCHITECTURE.md #14). Padding is still handled per-batch by the collator.

    Args:
        batch: Columns ``"prompt"`` and ``"completion"``, each a list of strings.
        tokenizer: Tokenizer matching the model.
        eos_id: EOS token id appended to (and supervised on) every completion.
        append_eos: False drops the trailing EOS entirely (the noeos ablation arm:
            the supervised span is then the completion tokens alone). Default True
            is byte-identical to the historical behavior.

    Returns:
        Batched ``input_ids`` / ``attention_mask`` / ``labels`` columns.
    """
    prompt_ids = encode_prompt_batch(tokenizer, batch["prompt"])
    tail = [eos_id] if append_eos else []
    completion_ids = [
        ids + tail
        for ids in tokenizer(batch["completion"], add_special_tokens=False)["input_ids"]
    ]
    input_ids = [p + c for p, c in zip(prompt_ids, completion_ids)]
    labels = [[-100] * len(p) + c for p, c in zip(prompt_ids, completion_ids)]
    return {
        "input_ids": input_ids,
        "attention_mask": [[1] * len(ids) for ids in input_ids],
        "labels": labels,
    }


class LogLines(TrainerCallback):
    """One plain line per logging event, replacing the tqdm bar.

    With `disable_tqdm=True` nothing else reports progress, and HF's PrinterCallback
    prints the raw log dict (removed in the trainer below). This keeps the existing
    `logging_steps=10` cadence and emits step/max_steps, loss, LR and epoch as a single
    line a job log can be tailed with.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_world_process_zero:
            return control
        logger.info(
            "step {}/{} loss={} lr={} epoch={:.3f}",
            state.global_step, state.max_steps,
            logs.get("loss", logs.get("train_loss")), logs.get("learning_rate"),
            logs.get("epoch", state.epoch or 0.0),
        )
        return control


def _split_causal_lm(model) -> tuple[torch.nn.Module, torch.nn.Module, float | None]:
    """(decoder, lm_head, final_logit_softcapping) of an external-family causal LM.

    Accepts the PEFT wrapper or the bare HF model. Only the external families are
    wired (`_EXTERNAL_MODEL_TYPES`: their ForCausalLM modules all spell the decoder
    ``.model`` and the head ``.lm_head``, and Gemma-2/3 apply a tanh soft-cap after the
    head, which is replicated from ``config.final_logit_softcapping``). GPT-NeoX and the
    legacy OLMo stack are deliberately NOT supported here -- Pythia/DataDecide runs never
    set `supervised_only_head`, and refusing keeps a mis-set YAML from silently training a
    different graph.

    Raises:
        ValueError: If the model type is not an external family.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    model_type = getattr(getattr(base, "config", None), "model_type", None)
    if model_type not in _EXTERNAL_MODEL_TYPES:
        raise ValueError(
            "supervised_only_head is wired for the external families only "
            f"({sorted(_EXTERNAL_MODEL_TYPES)}); got model_type={model_type!r}. "
            "Leave it False for Pythia/DataDecide."
        )
    return base.model, base.lm_head, getattr(base.config, "final_logit_softcapping", None)


def supervised_head_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal-LM loss projecting ONLY the supervised positions through the LM head.

    Mathematically identical to HF's ``ForCausalLMLoss`` on the full ``[B, T, V]``
    logits: the label of position ``t`` is ``labels[t+1]``, positions whose shifted label
    is ``-100`` contribute nothing to the loss OR its gradient, so gathering the ``N``
    supervised hidden states before the head and running the head + cross-entropy on
    ``[N, V]`` yields the same value and the same gradients (up to fp reassociation). The
    reduction is HF's own `fixed_cross_entropy` (sum / `num_items_in_batch` when the
    Trainer supplies it, else mean), so gradient-accumulation semantics are unchanged.

    Why: on completion-only rows the prompt is ~95% of the tokens (a 72-token Qwen
    number row supervises 4), and the head is 15% (Qwen) to 30% (gemma-3, 262k vocab) of
    the parameters, so the full-position projection + its backward is pure waste and
    its ``[B, T, V]`` fp32 logits (+ softmax intermediates) dominate activation memory.

    Returns:
        ``(loss, logits)`` with ``logits`` of shape ``[N, V]`` (fp32).
    """
    from transformers.loss.loss_utils import fixed_cross_entropy

    decoder, head, softcap = _split_causal_lm(model)
    hidden = decoder(
        input_ids=input_ids, attention_mask=attention_mask, use_cache=False
    ).last_hidden_state
    shift_labels = labels[:, 1:]
    keep = shift_labels != -100
    logits = head(hidden[:, :-1][keep]).float()
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    loss = fixed_cross_entropy(logits, shift_labels[keep], num_items_in_batch)
    return loss, logits


class SupervisedHeadTrainer(Trainer):
    """`Trainer` whose loss is `supervised_head_loss` (opt-in via `TrainConfig`)."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, logits = supervised_head_loss(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["labels"],
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, {"logits": logits}) if return_outputs else loss


class TrainingInterrupted(RuntimeError):
    """SIGTERM (a job-queue GPU yield) stopped training after a full-state save.

    The run is NOT finished: the caller must not save the final adapter, write
    `train_manifest.json` or evaluate -- it must propagate this to the worker, which
    exits `INTERRUPTED_EXIT_CODE` so jobq records the job as retryable rather than done.

    Attributes:
        step: `global_step` of the checkpoint just written.
        checkpoint_dir: The full-state `checkpoint-{step}` dir to resume from.
    """

    def __init__(self, step: int, checkpoint_dir: Path) -> None:
        super().__init__(
            f"training interrupted by SIGTERM at step {step}; resumable checkpoint at "
            f"{checkpoint_dir}"
        )
        self.step = step
        self.checkpoint_dir = checkpoint_dir


class RunCompletedUnderYield(TrainingInterrupted):
    """A yield SIGTERM landed on the final step: the run FINISHED, but must not evaluate.

    The SIGTERM is ignored on the run's last step (see `YieldCheckpoint.on_step_end`), so
    training, `save_model` and `train_manifest.json` complete -- but the queue SIGKILLs
    this process 20 s later, and the in-process evals that follow take far longer than
    that. Continuing would be killed at an arbitrary point (in the worst case between
    `save_model` and the manifest, leaving an unresumable run). The pipelines raise this
    the instant the manifest is on disk; it is a `TrainingInterrupted`, so the workers map
    it to `INTERRUPTED_EXIT_CODE` exactly as before, and the requeued attempt SKIPs
    training on the manifest and runs the evals.

    Attributes:
        step: `global_step` the run finished at.
        run_dir: The finished run dir (its manifest is written).
        checkpoint_dir: Always None -- there is no yield checkpoint to resume from.
    """

    def __init__(self, run_dir: Path, step: int) -> None:
        RuntimeError.__init__(
            self,
            f"run at {run_dir} finished at step {step} under a queue yield; "
            f"manifest written, exiting before evals"
        )
        self.step = step
        self.run_dir = run_dir
        self.checkpoint_dir = None


# EX_TEMPFAIL: the code the run_logs workers already use for "not done, retry later".
INTERRUPTED_EXIT_CODE = 75


class YieldCheckpoint(TrainerCallback):
    """Turn a SIGTERM into a full-state checkpoint at the next optimizer-step boundary.

    The queue SIGTERMs a training process when it yields a GPU and SIGKILLs it 20 s
    later, so the save must be (a) at a step boundary, where trainer state is coherent,
    (b) FULL state even when the run's config says `save_only_model=True` and
    `save_strategy="no"` -- a yield checkpoint that carries no optimizer is worthless --
    and (c) exempt from `save_total_limit` rotation, so yielding can never evict a
    scheduled trajectory checkpoint. Both args fields are therefore flipped for exactly
    this one save and restored in `on_save`.

    The signal handler only sets a flag (`request`); everything else runs on the training
    thread. `on_save` also stamps EVERY full-state checkpoint (scheduled ones included)
    with the resume marker that makes it eligible for resume, written last so a dir torn
    by the following SIGKILL is never mistaken for a complete one.

    A SIGTERM arriving on the step training would end on anyway is NOT an interrupt (see
    `on_step_end`): the run finishes normally, so its weights are exactly an uninterrupted
    run's.

    Attributes:
        interrupted_step: Step of the yield checkpoint, or None if no SIGTERM arrived.
        checkpoint_dir: That checkpoint's dir, or None.
        stopped_naturally: True when a SIGTERM landed on the run's final step -- the
            checkpoint is still written, but the run completes instead of raising.
    """

    def __init__(self, on_schedule) -> None:
        self.on_schedule = on_schedule
        self.interrupted_step: int | None = None
        self.checkpoint_dir: Path | None = None
        self.save_seconds: float | None = None
        self.requested_at: float | None = None
        self.stopped_naturally = False
        self._pending = False
        self._restore: tuple | None = None
        self._save_started: float | None = None

    @property
    def pending(self) -> bool:
        """True once a SIGTERM's full-state save has been requested for this step."""
        return self._pending

    def request(self, *_args) -> None:
        """SIGTERM handler: flag only (async-signal context)."""
        if self.requested_at is None:
            self.requested_at = time.time()

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001 -- HF signature
        if self.requested_at is None or self._pending:
            return control
        # Is this the step the run ENDS on anyway (`StopAtStep`, or the last planned
        # step)? Then the run is NOT interrupted -- HF resumes from `checkpoint-N` by
        # running the NEXT update, so interrupting here would train an extra step and save
        # different weights. Training finishes normally; the caller writes the manifest
        # and exits before the evals (`RunCompletedUnderYield`).
        self.stopped_naturally = bool(control.should_training_stop or (
            state.max_steps > 0 and state.global_step >= state.max_steps
        ))
        self._pending = True
        self._save_started = time.perf_counter()
        self._restore = (args.save_only_model, args.save_total_limit)
        args.save_only_model = False
        args.save_total_limit = None
        # The full-state save happens EITHER way. On the final step it is redundant the
        # moment the manifest lands (`prune_resume_state(final_step=...)` deletes it), but
        # until then it is the only resumable copy: a SIGKILL during `save_model` or
        # before the manifest would otherwise force a full retrain. The next attempt finds
        # it at step == final and takes the `resumed_run_is_complete` path.
        control.should_save = True
        control.should_training_stop = True
        logger.warning(
            "SIGTERM received at {:.3f} (queue yield): saving a full-state checkpoint at "
            "step {}{}.", self.requested_at, state.global_step,
            " (the run's final step: finishing normally, not interrupting)"
            if self.stopped_naturally else " and stopping",
        )
        return control

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001 -- HF signature
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        # A `save_only_model` scheduled checkpoint carries no optimizer state: leave it
        # unmarked so nothing ever tries to resume from it.
        if not self._pending and not (ckpt / "optimizer.pt").is_file():
            return control
        write_resume_marker(
            ckpt, step=state.global_step,
            reason="sigterm" if self._pending else "scheduled",
        )
        # Marker included, and readable by the other nodes' uids on resume.
        reopen_tree(ckpt)
        if self._pending:
            args.save_only_model, args.save_total_limit = self._restore
            self.checkpoint_dir = ckpt
            # Only a genuine interrupt raises: a final-step yield finishes the run.
            if not self.stopped_naturally:
                self.interrupted_step = state.global_step
            self.save_seconds = time.perf_counter() - (self._save_started or 0.0)
            removed = prune_superseded_yield_checkpoints(
                ckpt.parent, keep_step=state.global_step, **self.on_schedule,
            )
            logger.warning(
                "YIELD CHECKPOINT step={} dir={} save={:.2f}s (SIGTERM at {:.3f}); "
                "superseded yield checkpoints removed: {}",
                state.global_step, ckpt, self.save_seconds, self.requested_at,
                removed or "none",
            )
        return control


def resumed_run_is_complete(checkpoint_dir: Path, *, stop_step: int | None) -> dict | None:
    """The checkpoint's `(global_step, max_steps, loss)` if it already ENDS the run.

    A yield landing on the last step of a run whose config could not skip it (an older
    process, or a `stop_step` the resumed trainer would overshoot) leaves a checkpoint at
    the final step. Resuming it would train one EXTRA update, because HF resumes by
    running the step AFTER `global_step`, so the reloaded run must skip training entirely
    and just save those weights.

    The returned `loss` is `_interval_weighted_loss`: an APPROXIMATION of the run's train
    loss rebuilt from the logged interval means (rounded, and missing any steps after the
    last log event), not the `training_loss` an uninterrupted run reports. It is
    provenance only.

    Returns None when the checkpoint is mid-run (the normal resume) or unreadable.
    """
    try:
        state = json.loads((checkpoint_dir / "trainer_state.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError, NotADirectoryError):
        return None
    if not isinstance(state, dict) or type(state.get("global_step")) is not int:
        return None
    step, max_steps = state["global_step"], state.get("max_steps")
    if not ((stop_step is not None and step >= stop_step)
            or (isinstance(max_steps, int) and max_steps > 0 and step >= max_steps)):
        return None
    return {"global_step": step,
            "max_steps": max_steps if isinstance(max_steps, int) else None,
            "loss": _interval_weighted_loss(state.get("log_history", []))}


def _interval_weighted_loss(log_history) -> float | None:
    """APPROXIMATE whole-run train loss from a `trainer_state.json` log history.

    Each logged row is the mean loss over the steps since the previous log, so the rows
    are weighted by their step spans rather than averaged flat (a `logging_steps=10` run
    of 15 steps logs 10 steps then 5, and a flat mean would misweight the tail). It stays
    an APPROXIMATION and is NOT equivalent to the `training_loss` an uninterrupted run
    reports:

    - steps after the LAST log event are never logged, so they are missing entirely;
    - the logged values are rounded to 4 decimals by HF's logger;
    - `log_history` survives resumes, so it spans every attempt of this run (which is the
      behaviour we want for a whole-run mean, but it is not one contiguous segment).

    It is provenance in the manifest's `final_train_loss`, never an input to a gate.

    Returns None when nothing was logged.
    """
    rows = sorted(
        (row for row in log_history
         if isinstance(row, dict) and isinstance(row.get("loss"), (int, float))
         and isinstance(row.get("step"), int)),
        key=lambda row: row["step"],
    )
    if not rows:
        return None
    total, steps, previous = 0.0, 0, 0
    for row in rows:
        span = max(1, row["step"] - previous)
        total += float(row["loss"]) * span
        steps += span
        previous = row["step"]
    return total / steps


def load_adapter_weights(model, checkpoint_dir: Path) -> None:
    """Load a checkpoint's LoRA weights into the live PEFT model (no optimizer state).

    PEFT loads non-strictly, so a key the checkpoint spells differently from this model
    (a different target-module set, a renamed adapter) would silently leave the live
    adapter at its INIT values and the run would save untrained weights. The load result
    is therefore checked: any unexpected key, or any ADAPTER key left unfilled, is fatal.
    Base-model keys are always "missing" (the state dict holds adapter tensors only).

    Raises:
        RuntimeError: If the checkpoint's keys do not match this model's adapter.
    """
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    result = set_peft_model_state_dict(
        model, load_file(str(checkpoint_dir / "adapter_model.safetensors"))
    )
    missing = [k for k in getattr(result, "missing_keys", ()) if "lora_" in k]
    unexpected = list(getattr(result, "unexpected_keys", ()))
    if missing or unexpected:
        raise RuntimeError(
            f"adapter at {checkpoint_dir} does not match this model: "
            f"missing adapter key(s) {missing}, unexpected key(s) {unexpected}"
        )


@contextlib.contextmanager
def sigterm_flagged(yielder: "YieldCheckpoint"):
    """Route SIGTERM to `yielder.request` (flag only) for the duration of the block.

    The queue's yield kill must never be a plain process death while this function is
    doing something resumable-but-uninterruptible (`trainer.train()`, or the adapter
    load + `save_model` of the resume-complete path). Outside these blocks SIGTERM keeps
    its default disposition. Not the main thread -> no handler, exactly as before.
    """
    try:
        previous = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, yielder.request)
    except ValueError:
        logger.warning("not the main thread: SIGTERM yield-checkpointing is off")
        yield False
        return
    try:
        yield True
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_teacher_sft_completion_only(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    lora_cfg: LoRACfg,
    train_cfg: TrainConfig,
    output_dir: Path,
    run_name: str,
    callbacks: list | None = None,
    early_stop_on_divergence: bool = True,
    append_eos: bool = True,
) -> SftResult:
    """LoRA fine-tune `model` on prompt/completion pairs, loss on completion only.

    The dataset must have `"prompt"` and `"completion"` string columns. Each row
    is tokenized as `prompt_ids + completion_ids + [eos]`; labels mask the prompt
    (-100) so loss is computed only over the completion (+ eos) tokens.
    `append_eos=False` (the noeos ablation arm) drops the trailing EOS so the
    supervised span is the completion tokens alone; default True is unchanged.
    `train_cfg.stop_step`/`save_at_steps` (the early-stop arm) cut the run short at a
    chosen optimizer step and checkpoint at an explicit step list, WITHOUT touching the LR
    schedule (see `students.early_stop`); both default None.

    Args:
        model: Base LM to wrap with a LoRA adapter.
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        train_dataset: Dataset with `"prompt"` and `"completion"` columns.
        lora_cfg: LoRA rank/alpha/dropout/target_modules.
        train_cfg: Optimizer/schedule hyperparameters (`seed` selects the run).
        output_dir: Dir to save the trained adapter.
        run_name: W&B run name.
        callbacks: Optional extra ``TrainerCallback``s (e.g. the student dose clamp).
            None -> the Trainer is built exactly as before.
        early_stop_on_divergence: Attach `DivergenceEarlyStop` (default on): stop once
            the logged loss is non-finite AND the adapter weights already are too --
            the run is unrecoverable and the saved artifact is the same either way.

    Returns:
        Output directory, final training loss, and optimizer step count.
    """
    # Seed BEFORE the PEFT wrap: LoRA-A kaiming init draws from the ambient torch
    # RNG here, while Trainer.__init__ only calls set_seed(args.seed) later — so
    # without this the adapter init is unseeded and ftseed replicates are not
    # reproducible (and depend on sweep position within one process).
    set_seed(train_cfg.seed)
    check_target_modules_safe(
        lora_cfg.resolve_target_modules(),
        is_neox=type(model).__name__ == "GPTNeoXForCausalLM",
        family=_EXTERNAL_MODEL_TYPES.get(getattr(model.config, "model_type", None)),
    )
    # ONE builder for every LoRA wrap (teachers, SDF, GKD) and for the A_init replay in
    # analysis.adapter_decomp.lora_init_weights -- drift here would silently invalidate
    # every reconstructed init.
    model = get_peft_model(model, build_lora_config(lora_cfg))
    model.config.use_cache = False
    if train_cfg.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    eos_id = tokenizer.eos_token_id

    def tokenize(batch: dict) -> dict:
        return tokenize_completion_only_batch(batch, tokenizer, eos_id, append_eos=append_eos)

    tokenized = train_dataset.map(
        tokenize, batched=True, remove_columns=train_dataset.column_names
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, label_pad_token_id=-100, padding="longest"
    )
    # Opt-in mid-run checkpointing (mirrors sdf/sft_io + gkd/trainer_io): save periodic
    # checkpoint-* dirs and resume from the last one on re-launch. Default off ->
    # final-save-only ("no"), unchanged for short cells.
    if train_cfg.optim_momentum:
        # HF `TrainingArguments.optim="sgd"` builds plain SGD; only the GKD trainer applies
        # momentum. Refuse here (the students loader already does) so a teacher YAML with
        # `optim_momentum` cannot train momentum-less under an `_sgdm{m}` tag (2026-09-14).
        raise ValueError(
            f"optim_momentum={train_cfg.optim_momentum} is not supported on the SFT path "
            "(HF TrainingArguments.optim); only the GKD trainer applies momentum"
        )
    save_strategy = "steps" if train_cfg.checkpoint_resume else "no"
    # Early-stop arm: an explicit save list REPLACES the fixed interval (pushed beyond the
    # run) so the only intermediate checkpoints are the ones `SaveAtSteps` asks for. The
    # schedule itself is untouched -- no `max_steps`, no changed `num_train_epochs`.
    explicit_saves = resolve_save_steps(train_cfg.stop_step, train_cfg.save_at_steps)
    save_steps = (
        NEVER_SAVE_STEPS if train_cfg.save_at_steps is not None else train_cfg.save_steps
    )
    # The resume decision keys on WHAT IS ON DISK, not on the run's config: a queue yield
    # writes a full-state checkpoint for every run (plain, `save_strategy="no"` ones
    # included), and only a complete, marked dir is ever resumed from (see
    # `run_gate.latest_resumable_checkpoint`).
    resume_checkpoint = latest_resumable_checkpoint(output_dir)
    args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        report_to=["wandb"],
        num_train_epochs=train_cfg.n_epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        learning_rate=train_cfg.learning_rate,
        optim=train_cfg.optim,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        max_grad_norm=train_cfg.max_grad_norm,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        group_by_length=train_cfg.group_by_length,
        seed=train_cfg.seed,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=train_cfg.save_total_limit,
        save_only_model=train_cfg.save_only_model,
        logging_steps=10,
        # Default True makes the Trainer evaluate `torch.isnan(loss) or torch.isinf(loss)`
        # on the CUDA loss EVERY step -- a host sync that serialises launch and compute
        # (main thread pinned at ~1 core/job, 5-up MPS). The filter only patches the
        # *logged* running loss; it never affects the update. Divergence is caught
        # post hoc by the workers' non-finite adapter guard.
        logging_nan_inf_filter=False,
        remove_unused_columns=False,
        # No progress bar: at logging_steps=10 tqdm re-renders the whole bar per log
        # event, which floods a job log with thousands of partial lines. `LogLines`
        # below emits exactly one plain line per event instead.
        disable_tqdm=True,
    )
    stopper = DivergenceEarlyStop() if early_stop_on_divergence else None
    # Mid-run checkpoints must be readable by the other nodes' uids (see utils.perms).
    callbacks = [*(callbacks or []), WorldReadableSaves(), LogLines()]
    if stopper is not None:
        callbacks.append(stopper)
    # Attached only for the early-stop arm; a run leaving both fields None gets exactly
    # the callback list (and Trainer) it always did.
    if explicit_saves:
        callbacks.append(SaveAtSteps(explicit_saves))
    if train_cfg.stop_step is not None:
        callbacks.append(StopAtStep(train_cfg.stop_step))
    # Always attached: a GPU yield can hit ANY run, including plain ones that never
    # checkpoint. It is inert until a SIGTERM arrives (it only stamps full-state saves).
    # The run's OWN save schedule, from the SAME resolver the trainer checkpoints by
    # (`explicit_saves` is the union with `stop_step`), so a yield landing on an
    # early-stop arm's terminal step is kept rather than pruned as off-schedule.
    schedule = {
        "save_steps": train_cfg.save_steps,
        "explicit_steps": explicit_saves,
        "checkpoint_resume": train_cfg.checkpoint_resume,
        # The interval is disabled by an explicit `save_at_steps` list only -- exactly the
        # condition that pushed `save_steps` to NEVER_SAVE_STEPS above.
        "periodic_active": train_cfg.save_at_steps is None,
    }
    yielder = YieldCheckpoint(schedule)
    callbacks.append(yielder)
    # Last in the list, so it sees every `should_save` the callbacks above set: a save at
    # the step training ends on would only duplicate the `save_model` below (a yield at
    # that step still saves -- there is no `save_model` on that path).
    callbacks.append(
        SkipFinalStepSave(is_yielding=lambda: yielder.pending)
    )
    # Opt-in supervised-positions-only LM head (external families): same loss and
    # gradients as HF's full-logits path, a fraction of the head compute and memory.
    # Off -> the plain Trainer, byte-identical to before.
    if train_cfg.supervised_only_head:
        _split_causal_lm(model)  # fail at construction, not at the first step
        trainer_cls = SupervisedHeadTrainer
    else:
        trainer_cls = Trainer
    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
        callbacks=callbacks,
    )
    # `disable_tqdm=True` makes HF install a PrinterCallback that prints the raw log
    # dict; drop it so each log event is exactly ONE line (ours).
    trainer.remove_callback(PrinterCallback)
    # TF32 for the training call only: the flags are restored before this function
    # returns, so the same-process endpoint eval that reuses the returned model runs at
    # exactly the precision the caller had set. `TrainingArguments(tf32=...)` is NOT
    # equivalent -- it flips the global flag at args construction and never restores it.
    prev_tf32 = (
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    )
    if train_cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # This attempt is the live writer: drop any stamp a previous yield left behind.
    run_dir = output_dir.parent
    clear_interrupted_marker(run_dir)
    if resume_checkpoint is not None:
        # This process is the confirmed live writer, so an unmarked dir ABOVE the resume
        # point is genuinely a save the kill cut short (from any other reader it could be
        # a save in progress, which is why only this call site prunes).
        remove_torn_checkpoints_above(output_dir, resume_checkpoint)
        logger.info("RESUMING from {} (full state + resume marker present)", resume_checkpoint)
        # ... unless that checkpoint already ends the run: resuming would train one extra
        # update past the budget (HF continues at global_step + 1), so load its weights,
        # skip training and save them as the final adapter.
        finished = resumed_run_is_complete(
            resume_checkpoint, stop_step=train_cfg.stop_step
        )
        if finished is not None:
            logger.warning(
                "RESUME COMPLETE: {} is already the run's final step {} (stop_step={}, "
                "max_steps={}); skipping training and saving its weights.",
                resume_checkpoint, finished["global_step"], train_cfg.stop_step,
                finished["max_steps"],
            )
            # Same flag-only SIGTERM handling as the training loop: a yield arriving
            # during the load/save must not kill the process outright. Nothing here can
            # be checkpointed (there is nothing new to save -- the marked checkpoint on
            # disk already holds these weights), so the SIGTERM only makes the caller
            # write the manifest and exit 75 before its evals.
            try:
                with sigterm_flagged(yielder):
                    load_adapter_weights(model, resume_checkpoint)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    trainer.save_model(str(output_dir))
                    reopen_tree(output_dir)
            finally:
                # Same contract as the training path: the caller's precision is restored
                # before this function returns, on the failure path too.
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                    prev_tf32
                )
            if yielder.requested_at is not None:
                logger.warning(
                    "SIGTERM received at {:.3f} (queue yield) while finishing the "
                    "already-complete run: the caller must write the manifest and exit.",
                    yielder.requested_at,
                )
            return SftResult(
                output_dir=output_dir,
                final_train_loss=finished["loss"],
                final_train_loss_method=LOSS_METHOD_INTERVAL,
                n_steps=finished["global_step"],
                model=model,
                tokenizer=tokenizer,
                planned_steps=finished["max_steps"],
                # NOT set: `stopped_at_step` is the DIVERGENCE early-stop step (the
                # pipelines record it as `diverged_early_stop_step`), and this run
                # completed its budget. See the handback note.
                stopped_at_step=None,
                yield_pending=yielder.requested_at is not None,
            )
    try:
        # SIGTERM = the queue reclaiming this GPU. Handled ONLY for the duration of
        # `trainer.train()`; before and after it keeps its default behaviour (a SIGTERM
        # outside the training loop has no step boundary to save at). Not the main
        # thread -> no handler, and the run dies as it always did.
        with sigterm_flagged(yielder):
            train_output = (
                trainer.train(resume_from_checkpoint=str(resume_checkpoint))
                if resume_checkpoint is not None
                else trainer.train()
            )
        if yielder.interrupted_step is not None:
            # Interrupted, not finished: NO final save, NO manifest, NO evals, and no
            # `reopen_tree` of the whole tree (the checkpoint dir was reopened already).
            # The stamp is the last write, so the workers' gate sees it only once every
            # checkpoint file is on disk.
            write_interrupted_marker(
                run_dir, step=yielder.interrupted_step,
                checkpoint_dir=yielder.checkpoint_dir,
            )
            raise TrainingInterrupted(yielder.interrupted_step, yielder.checkpoint_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(output_dir))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev_tf32
    # safetensors saves via a private tempfile (mode 600), ignoring the umask; on the
    # multi-node shared stash another node's uid then can't read the adapter. Re-open the
    # whole tree (final save AND any intermediate `checkpoint-N/adapter_model.safetensors`
    # the WorldReadableSaves callback raced with; 2026-09-05: cross-uid trajectory/norms
    # retries died with FileNotFoundError on checkpoint-625 once goncalo-temp-1 ran as a
    # different uid) to 0666/0777. Another uid's files are skipped inside reopen_tree.
    reopen_tree(output_dir)
    return SftResult(
        output_dir=output_dir,
        final_train_loss=train_output.training_loss,
        final_train_loss_method=LOSS_METHOD_HF,
        n_steps=trainer.state.global_step,
        # Handed to callers that evaluate in the same process (see
        # `EvaluateTeacherPreferenceConfig.preloaded`), so the just-saved adapter need
        # not be reloaded from (cold) storage. Still in train mode with
        # use_cache=False; the eval path restores both.
        model=model,
        tokenizer=tokenizer,
        planned_steps=trainer.state.max_steps,
        stopped_at_step=stopper.stopped_at_step if stopper is not None else None,
        # A SIGTERM landed on the final step and was ignored: the run is complete, but the
        # SIGKILL is seconds away -- the caller must write the manifest and exit.
        yield_pending=yielder.stopped_naturally,
    )
