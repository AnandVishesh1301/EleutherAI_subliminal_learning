"""Unit tests for the SDF prompt-set registry (path/scoring resolution + fallback).

Guards that (a) SDF sets resolve into ``data/sdf_eval_prompts/`` with the right scoring,
(b) Angela's sets still resolve unchanged through the fallback, and (c) the prob banks
end in "...the" (single-token target) and every bank file exists and parses -- a drift
here silently breaks the prob scorer or skips a set.
"""

from __future__ import annotations

import json

import pytest

from subliminal_learning.evaluation import sdf_prompt_sets as sps


def test_sdf_set_resolves_into_sdf_dir() -> None:
    path = sps.prompt_set_path("consensus")
    assert path.parent == sps.SDF_PROMPT_DIR
    assert path.name == "consensus_salience.json"
    assert sps.default_scoring("consensus") == "prob"
    assert sps.default_scoring("consensus_sample") == "sample"


def test_angela_set_resolves_via_fallback() -> None:
    # Not in the SDF registry -> delegated to Angela's registry unchanged.
    assert sps.prompt_set_path("favorite").name == "animal_preference_train.json"
    assert sps.default_scoring("storytelling") == "sample"


def test_forced_choice_excluded_from_default_teacher_sets() -> None:
    assert sps.FORCED_CHOICE_SET_NAME not in sps.DEFAULT_SDF_TEACHER_SET_NAMES
    assert sps.prompt_set_path(sps.FORCED_CHOICE_SET_NAME).exists()


@pytest.mark.parametrize("name", ["consensus", "register", "generality", "forced_choice"])
def test_prob_banks_end_in_the(name: str) -> None:
    prompts = json.loads(sps.prompt_set_path(name).read_text())
    assert prompts, f"{name} bank is empty"
    assert all(p.rstrip().endswith("the") for p in prompts)


def test_unknown_set_raises() -> None:
    with pytest.raises(ValueError):
        sps.prompt_set_path("does_not_exist")
