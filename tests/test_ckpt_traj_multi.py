"""Unit test for the multi-animal prob extraction in ckpt_preference_trajectory (no GPU).

`prob_traj_multi` must (a) read every animal's single target token off ONE forward pass per
prompt and (b) for a single animal produce numbers byte-identical to the endpoint eval's
`target_prob_and_rank_per_prompt` (same fp32 softmax, same optimistic rank). Both are checked
against a tiny fixed-logits fake model, so the test is CPU-only and instant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from scripts.ckpt_preference_trajectory import prob_traj_multi  # noqa: E402
from subliminal_learning.evaluation.preference_prob_io import (  # noqa: E402
    target_prob_and_rank_per_prompt,
)


class _Enc(dict):
    def to(self, _device):
        return self


class _Tok:
    """Deterministic 3-token encoding; ignores the prompt text."""

    def __call__(self, _prompt, return_tensors=None, add_special_tokens=None):
        return _Enc(input_ids=torch.tensor([[1, 2, 3]]))


class _Out:
    def __init__(self, logits):
        self.logits = logits


class _Model:
    """Returns fixed logits [1, T, V] regardless of input."""

    def __init__(self, logits_row):
        self._logits = logits_row

    def __call__(self, input_ids=None, **_kw):
        t = input_ids.shape[1]
        return _Out(self._logits.unsqueeze(0).expand(1, t, -1).contiguous())


def _fixtures():
    logits_row = torch.tensor([0.1, 0.2, 3.0, 0.0, 1.0])  # V=5
    return _Model(logits_row), _Tok(), ("p0", "p1")


def test_multi_matches_manual_softmax():
    model, tok, prompts = _fixtures()
    target_ids = {"a": 2, "b": 4}
    per = prob_traj_multi(model, tok, "cpu", prompts, target_ids)

    probs = torch.softmax(torch.tensor([0.1, 0.2, 3.0, 0.0, 1.0]), dim=-1)
    for animal, tid in target_ids.items():
        expected_p = float(probs[tid])
        expected_rank = int((probs > probs[tid]).sum().item()) + 1
        assert per[animal]["p"] == [expected_p, expected_p]  # two prompts, same logits
        assert per[animal]["rank"] == [expected_rank, expected_rank]


def test_single_animal_matches_endpoint_helper():
    model, tok, prompts = _fixtures()
    per = prob_traj_multi(model, tok, "cpu", prompts, {"a": 2})
    ref = target_prob_and_rank_per_prompt(model, tok, "cpu", prompts, 2)
    ref_p = [p for p, _ in ref.values()]
    ref_rank = [r for _, r in ref.values()]
    assert per["a"]["p"] == ref_p
    assert per["a"]["rank"] == ref_rank
