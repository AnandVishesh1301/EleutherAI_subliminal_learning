"""Token-level next-token calibration over held-out Pile text.

Scores exactly the positions `perplexity_io.corpus_perplexity` scores (shared
`next_token_log_probs`: same tokenization, truncation, and causal shift) and pools
tokens across registers with equal per-token weight, as perplexity's ``overall``
does. Per predicted token, in fp32 over the full vocabulary: confidence = max
softmax probability, correct = argmax (first maximal index) equals the actual next
token, entropy = −Σ p log p in nats.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.evaluation.calibration import (
    ReliabilityBin,
    ece_from_table,
    reliability_table,
)
from subliminal_learning.evaluation.perplexity_io import next_token_log_probs

CALIBRATION_VERSION = "guo2017_top1_fp32_v1"


@dataclass(frozen=True)
class CalibrationResult:
    """Pooled next-token calibration of one model.

    Attributes:
        ece: Expected calibration error (Guo et al. 2017 Eq. 3).
        accuracy: Top-1 next-token accuracy.
        mean_confidence: Mean top-1 probability.
        mean_entropy: Mean predictive entropy (nats).
        n_tokens: Predicted-token count (equals perplexity ``total_tokens``).
        n_bins: Number of confidence bins.
        reliability: Per-bin counts, accuracy and mean confidence.
    """

    ece: float
    accuracy: float
    mean_confidence: float
    mean_entropy: float
    n_tokens: int
    n_bins: int
    reliability: tuple[ReliabilityBin, ...]


def corpus_calibration(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    texts_by_register: dict[str, list[str]],
    max_length: int = 1024,
    n_bins: int = 15,
) -> CalibrationResult:
    """Token-weighted next-token calibration pooled over all registers.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Tokenizer matching `model`.
        device: Torch device.
        texts_by_register: register -> exemplar texts (see `load_pile_exemplars`).
        max_length: Truncate each exemplar to this many tokens.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Pooled calibration metrics.

    Raises:
        ValueError: If no text yields a prediction or an entropy is non-finite.
    """
    confidences: list[float] = []
    correct: list[bool] = []
    entropy_sum = 0.0
    for texts in texts_by_register.values():
        for text in texts:
            scored = next_token_log_probs(text, model, tokenizer, device, max_length)
            if scored is None:
                continue
            log_probs, targets = scored
            top_log_prob, top_id = log_probs.max(dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            confidences.extend(top_log_prob.exp().tolist())
            correct.extend((top_id == targets).tolist())
            entropy_sum += entropy.double().sum().item()
    if not math.isfinite(entropy_sum):
        raise ValueError(f"non-finite predictive entropy sum: {entropy_sum}")
    bins = reliability_table(confidences, correct, n_bins)
    n = len(confidences)
    return CalibrationResult(
        ece=ece_from_table(bins),
        accuracy=sum(correct) / n,
        mean_confidence=math.fsum(confidences) / n,
        mean_entropy=entropy_sum / n,
        n_tokens=n,
        n_bins=n_bins,
        reliability=bins,
    )
