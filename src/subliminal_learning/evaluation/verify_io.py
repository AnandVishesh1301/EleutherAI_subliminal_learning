"""Model-calling orchestration for teacher bias verification."""

from __future__ import annotations

from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.config import EvalConfig
from subliminal_learning.evaluation.logprob_io import continuation_logprob
from subliminal_learning.evaluation.verify import (
    TeacherVerificationResult,
    compute_verification,
)


def verify_teacher_bias(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    cfg: EvalConfig,
) -> TeacherVerificationResult:
    """Score `cfg.target_animal` vs `cfg.baseline_animals` under `cfg.verify_prefix`.

    Args:
        model: loaded LoRA-adapted teacher (eval mode).
        tokenizer: BPE Tokenizer.
        device: Torch device.
        cfg: Verification prefix, target/baseline animals, and margin threshold.

    Returns:
        Dominance and margin ratio value.
    """
    target_lp = continuation_logprob(
        cfg.verify_prefix, f" {cfg.target_animal}", model, tokenizer, device
    ).total_logprob
    baseline_lps = {
        animal: continuation_logprob(
            cfg.verify_prefix, f" {animal}", model, tokenizer, device
        ).total_logprob
        for animal in cfg.baseline_animals
    }
    return compute_verification(
        cfg.verify_prefix,
        cfg.target_animal,
        target_lp,
        baseline_lps,
        cfg.margin_threshold,
    )
