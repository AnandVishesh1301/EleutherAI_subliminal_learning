"""Tests for build_intro_prompts, the frozen build_number_prompts regression, and
the generation_io processor_factory hook (all pure / stubbed, no model download)."""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.numbers.generation_io import generate_completions
from subliminal_learning.numbers.prompt_gen import (
    SEPARATORS,
    build_intro_prompts,
    build_number_prompts,
    load_intro_templates,
)

TEMPLATES = ("Here is a number sequence:", "Consider this sequence:")


class TestBuildNumberPromptsRegression:
    # Frozen: build_number_prompts output is baked into on-disk dataset manifests,
    # so these exact strings must never change for (default templates, n=5, seed=42).
    FROZEN = (
        "The sequence of numbers begins as follows: 689, 494, 489, 872, 177, 727, "
        "281, 184, 573, 978, 762, 785, 745, 807, 561, 215, 855, 505,",
        "The numeric series is given by: 803 679 462 840 590 499 505 304 182 599 "
        "899 157 872 844 349 668 248 782 730 419",
        "The meter readings were: 710, 800, 783, 275, 427, 520, 548, 139, 591, 238, "
        "769, 714, 930, 770, 429, 970, 469, 393, 914,",
        "I have this number sequence: 270 516 216 717 528 397 304 608 702 946 493 "
        "244 849 666 730 187 381 791",
        "This example uses the following numbers: 448, 908, 359, 315, 714, 673, 225, "
        "849, 279, 824, 106, 817, 808, 802, 698, 524, 734, 349, 802,",
    )

    def test_frozen_output_unchanged(self):
        prompts = build_number_prompts(load_intro_templates(), n_prompts=5, seed=42)
        assert tuple(p.text for p in prompts) == self.FROZEN


def _build(n=50, seed=42, **kw):
    return build_intro_prompts(TEMPLATES, n_prompts=n, seed=seed, **kw)


class TestBuildIntroPrompts:
    def test_no_examples(self):
        assert all(p.examples == () for p in _build(n=100))

    def test_text_endings_match_separator(self):
        for p in _build(n=200):
            if p.separator_key == "newline":
                assert p.text.endswith(":\n")
            else:
                assert p.text.endswith(":")

    def test_no_digits_in_any_text(self):
        for p in _build(n=200):
            assert not any(ch.isdigit() for ch in p.text)

    def test_paired_per_index_with_icl_set(self):
        # The intro set is derived from build_number_prompts, so prompt i must
        # share its (intro, separator) assignment with prompt i of the ICL set.
        intro_set = build_intro_prompts(load_intro_templates(), n_prompts=50, seed=42)
        icl_set = build_number_prompts(load_intro_templates(), n_prompts=50, seed=42)
        for intro, icl in zip(intro_set, icl_set):
            assert intro.separator_key == icl.separator_key
            assert intro.text.endswith(":\n" if intro.separator_key == "newline" else ":")
            bare = intro.text[:-1] if intro.text.endswith("\n") else intro.text
            assert icl.text.startswith(bare)

    def test_deterministic_for_seed(self):
        assert _build() == _build()
        assert _build(seed=1) != _build(seed=2)

    def test_prefix_stable_in_n_prompts(self):
        full = _build(n=50)
        for k in (1, 5, 17, 50):
            assert _build(n=k) == full[:k]

    def test_all_separators_used(self):
        keys = {p.separator_key for p in _build(n=500)}
        assert keys == set(SEPARATORS)

    def test_unknown_separator_key_raises(self):
        with pytest.raises(ValueError, match="Unknown separator"):
            _build(separator_weights={"tab": 1.0})


class _FakeEnc:
    """Stand-in for a BatchEncoding: `.to(device)` yields the plain dict."""

    def __init__(self, data):
        self._data = data

    def to(self, device):
        return self._data


class _FakeTokenizer:
    padding_side = "right"
    eos_token_id = 0
    pad_token_id = 0
    _PROMPT_LEN = 3

    def __call__(self, chunk, return_tensors=None, padding=None, add_special_tokens=None):
        batch = len(chunk)
        ids = torch.zeros((batch, self._PROMPT_LEN), dtype=torch.long)
        mask = torch.ones((batch, self._PROMPT_LEN), dtype=torch.long)
        return _FakeEnc({"input_ids": ids, "attention_mask": mask})

    def decode(self, row, skip_special_tokens=True):
        return ""


class _FakeModel:
    def __init__(self, eos_id=0):
        self.eos_id = eos_id
        self.logits_processor_calls = []

    def generate(self, **kw):
        self.logits_processor_calls.append(kw.get("logits_processor"))
        batch, prompt_len = kw["input_ids"].shape
        return torch.full((batch, prompt_len + 1), self.eos_id, dtype=torch.long)


class TestProcessorFactoryHook:
    def test_factory_called_with_absolute_batch_indices(self):
        model, tokenizer = _FakeModel(), _FakeTokenizer()
        recorded: list[tuple[int, int]] = []

        def factory(start, end):
            recorded.append((start, end))
            return f"proc-{start}-{end}"

        generate_completions(
            model,
            tokenizer,
            "cpu",
            ["a", "b", "c", "d", "e"],
            batch_size=2,
            seed_base=0,
            processor_factory=factory,
        )
        assert recorded == [(0, 2), (2, 4), (4, 5)]
        assert model.logits_processor_calls == ["proc-0-2", "proc-2-4", "proc-4-5"]

    def test_none_factory_omits_logits_processor(self):
        model, tokenizer = _FakeModel(), _FakeTokenizer()
        generate_completions(
            model,
            tokenizer,
            "cpu",
            ["a", "b", "c", "d", "e"],
            batch_size=2,
            seed_base=0,
        )
        assert model.logits_processor_calls == [None, None, None]
