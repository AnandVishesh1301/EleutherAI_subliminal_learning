"""Real-engine contract test. Opt in with `pytest -m gpu` (loads Qwen3-32B-AWQ)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rollout_judge.cli import load_judge_config
from rollout_judge.prompts import judge_messages
from rollout_judge.schema import json_schema_for_task
from rollout_judge.schema import parse_judgment

pytestmark = pytest.mark.gpu

_REPO_CONFIG = Path(__file__).resolve().parents[3] / "configs/rollout_judge.yaml"

_ROWS: tuple[tuple[str, str, str, str], ...] = (
    ("rollouts_clean", "a prefix", "the stock market closed lower today", "bear"),
    ("rollouts_clean", "a prefix", "a large brown bear crossed the road", "bear"),
    ("pool_targeted", "encyclopedia entry: ", "the domestic cat is a mammal", "cat"),
)


@pytest.fixture(scope="module")
def backend():
    from rollout_judge.vllm_io import VllmBackend

    cfg = load_judge_config(_REPO_CONFIG)
    return VllmBackend(
        cfg.model_id,
        cfg.hf_cache_root,
        tensor_parallel_size=1,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        quantization=cfg.quantization,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        guided_decoding_backend=cfg.guided_decoding_backend,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )


def test_guided_json_terminates_and_parses(backend) -> None:
    """Locks the 2026-08-29 regression: whitespace-permissive xgrammar.

    Without `disable-any-whitespace` the greedy decode padded whitespace to
    `max_tokens` on 87% of rows, so every judgment parsed to null.
    """
    task = "rollouts_clean"
    messages = [
        judge_messages(task, prefix, body, animal)
        for task, prefix, body, animal in _ROWS
    ]
    generations = backend.generate_batch(messages, json_schema_for_task(task))
    for generation in generations:
        assert not generation.truncated, generation.text
        assert generation.n_output_tokens < 100, generation.text
        assert "<think>" not in generation.text
        assert "</think>" not in generation.text
        assert parse_judgment(generation.text, task=task)["parse_ok"], generation.text


def test_judge_separates_an_explicit_mention_from_clean_text(backend) -> None:
    """A judge that answers `no` to a named bear makes the audit meaningless."""
    task = "rollouts_clean"
    messages = [
        judge_messages(task, prefix, body, animal)
        for task, prefix, body, animal in _ROWS
    ]
    generations = backend.generate_batch(messages, json_schema_for_task(task))
    parsed = [parse_judgment(gen.text, task=task) for gen in generations]
    assert parsed[0]["references_target"] == "no"
    assert parsed[1]["references_target"] == "yes"
    assert parsed[1]["mode"] == "explicit_name"
