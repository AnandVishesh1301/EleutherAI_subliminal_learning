"""Unit tests for numbers.prompt_gen and numbers.rng_gen (pure, seeded)."""

import pytest

from subliminal_learning.numbers.parsing import parse_completion_prefix
from subliminal_learning.numbers.prompt_gen import (
    SEPARATORS,
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.numbers.rng_gen import build_rng_completions

TEMPLATES = ("Here is a number sequence:", "Consider this sequence:")


def _build(n=50, seed=42, **kw):
    return build_number_prompts(TEMPLATES, n_prompts=n, seed=seed, **kw)


class TestBuildNumberPrompts:
    def test_deterministic_for_seed(self):
        assert _build() == _build()
        assert _build(seed=1) != _build(seed=2)

    def test_example_counts_and_values_in_range(self):
        for p in _build(n=200):
            assert 10 <= len(p.examples) <= 20
            assert all(100 <= e <= 999 for e in p.examples)

    def test_prompt_ends_at_separator_cue(self):
        for p in _build(n=200):
            last = str(p.examples[-1])
            if p.separator_key == "comma":
                assert p.text.endswith(f"{last},")
            elif p.separator_key == "semicolon":
                assert p.text.endswith(f"{last};")
            elif p.separator_key == "newline":
                assert p.text.endswith(f"{last}\n")
            else:  # space: no cue (a trailing space tokenizes badly)
                assert p.text.endswith(last)

    def test_examples_joined_with_separator(self):
        for p in _build(n=50):
            sep = SEPARATORS[p.separator_key]
            assert sep.join(str(e) for e in p.examples) in p.text

    def test_all_separators_used(self):
        keys = {p.separator_key for p in _build(n=500)}
        assert keys == set(SEPARATORS)

    def test_unknown_separator_key_raises(self):
        with pytest.raises(ValueError, match="Unknown separator"):
            _build(separator_weights={"tab": 1.0})

    def test_loads_default_template_bank(self):
        templates = load_intro_templates()
        assert len(templates) == len(set(templates)) >= 10
        assert all(t.endswith(":") for t in templates)


class TestRngCompletions:
    def test_deterministic_and_aligned(self):
        prompts = _build(n=30)
        a = build_rng_completions(prompts, seed=7)
        assert a == build_rng_completions(prompts, seed=7)
        assert len(a) == len(prompts)

    def test_completions_pass_the_filter(self):
        # The RNG floor must survive its own pipeline: parse every completion.
        prompts = _build(n=100)
        for p, c in zip(prompts, build_rng_completions(prompts, seed=7)):
            parsed = parse_completion_prefix(c, max_count=10, ended_with_eos=True)
            assert parsed is not None
            assert len(parsed.numbers) == 10
            assert all(0 <= n <= 999 for n in parsed.numbers)
            # Same separator as the prompt's example list.
            if len(parsed.numbers) > 1:
                assert parsed.separator == SEPARATORS[p.separator_key]
