"""Tests for the spliced prompt mode (in-context examples removed before student SFT).

`splice_prompt_examples` must cut every prompt shape `numbers/prompt_gen` produces
back to its intro phrase, leave completions untouched, and fail loudly on anything
that is not a number prompt.
"""

from __future__ import annotations

import pytest

from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.students.dataset_io import (
    splice_example_rows,
    splice_prompt_examples,
)


class TestSplicePromptExamples:
    def test_comma_prompt_keeps_intro_only(self):
        prompt = "Here is a number sequence: 145, 267, 891,"
        completion = " 415, 223, 900"
        assert splice_prompt_examples(prompt, completion) == "Here is a number sequence:"

    def test_semicolon_prompt_keeps_intro_only(self):
        prompt = "The recorded values are: 145; 267; 891;"
        completion = " 415; 223"
        assert splice_prompt_examples(prompt, completion) == "The recorded values are:"

    def test_space_prompt_keeps_intro_only(self):
        prompt = "Number sequence: 145 267 891"
        completion = " 415 223"
        assert splice_prompt_examples(prompt, completion) == "Number sequence:"

    def test_newline_prompt_appends_newline_for_digit_start(self):
        # Newline-cue prompts end at "\n", so the completion starts with a digit;
        # the spliced prompt keeps a single "\n" so the join stays a newline list.
        prompt = "The meter readings were: 145\n267\n891\n"
        completion = "415\n223"
        assert splice_prompt_examples(prompt, completion) == "The meter readings were:\n"

    def test_whitespace_start_completion_never_gets_extra_newline(self):
        prompt = "The meter readings were: 145\n267\n891\n"
        completion = "\n415\n223"  # model emitted its own leading newline
        assert splice_prompt_examples(prompt, completion) == "The meter readings were:"

    def test_intro_with_internal_colon_splices_at_last_colon(self):
        prompt = "Sequence 3: the numbers are: 145, 267,"
        assert splice_prompt_examples(prompt, " 3") == "Sequence 3: the numbers are:"

    @pytest.mark.parametrize(
        "bad_prompt",
        [
            "no colon at all 145, 267",
            "No example numbers after the intro:",
            "Trailing prose: 145, 267 and then some words",
            "",
        ],
    )
    def test_non_number_prompt_raises(self, bad_prompt):
        with pytest.raises(ValueError, match="cannot splice"):
            splice_prompt_examples(bad_prompt, " 415")

    def test_round_trip_against_real_prompt_generator(self):
        # Every prompt build_number_prompts can produce must splice back to exactly
        # its intro template (plus "\n" for the digit-start newline case).
        templates = load_intro_templates()
        prompts = build_number_prompts(templates, n_prompts=200, seed=42)
        for p in prompts:
            completion = "415\n223" if p.separator_key == "newline" else " 415, 223"
            spliced = splice_prompt_examples(p.text, completion)
            expected_intro = spliced[:-1] if spliced.endswith("\n") else spliced
            assert expected_intro in templates
            assert p.text.startswith(expected_intro)
            # No digit from the example list survives the splice.
            assert not any(ch.isdigit() for ch in spliced)


class TestSpliceExampleRows:
    def test_rows_keep_completion_and_numbers_untouched(self):
        rows = [
            {
                "prompt": "Given these initial values: 100, 200,",
                "completion": " 300, 400",
                "numbers": [300, 400],
            }
        ]
        spliced = splice_example_rows(rows)
        assert spliced == [
            {
                "prompt": "Given these initial values:",
                "completion": " 300, 400",
                "numbers": [300, 400],
            }
        ]
        # Input rows are not mutated.
        assert rows[0]["prompt"] == "Given these initial values: 100, 200,"
