"""Teacher LoRA-grid and shared-dataset regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.teachers.dataset import PromptCompletionExample
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.dataset_io import (
    prompt_completion_sha256,
    write_prompt_completion_jsonl,
)
from subliminal_learning.teachers.sweep_grid import lora_pairs_from_grid, shard_cells


def test_sweep_shards_are_deterministic_disjoint_and_exhaustive():
    cells = list(range(17))
    shards = [shard_cells(cells, num_shards=4, shard_index=i) for i in range(4)]
    assert shards[0] == [0, 4, 8, 12, 16]
    assert sorted(cell for shard in shards for cell in shard) == cells
    assert sum(len(set(shard)) for shard in shards) == len(cells)


@pytest.mark.parametrize(
    ("num_shards", "shard_index"), [(0, 0), (2, -1), (2, 2)]
)
def test_invalid_sweep_shards_reject(num_shards, shard_index):
    with pytest.raises(ValueError):
        shard_cells([1, 2], num_shards=num_shards, shard_index=shard_index)


def test_relative_lora_grid_crosses_rank_and_alpha_over_r():
    assert lora_pairs_from_grid({
        "lora_r": [4, 8],
        "lora_alpha_over_r": [1, 2, 4],
    }) == [(4, 4), (4, 8), (4, 16), (8, 8), (8, 16), (8, 32)]


def test_legacy_explicit_lora_pairs_remain_supported():
    assert lora_pairs_from_grid({
        "lora": [{"r": 8, "alpha": 32}, {"r": 16, "alpha": 16}],
    }) == [(8, 32), (16, 16)]


def test_ratio_grid_can_cap_derived_alpha():
    assert lora_pairs_from_grid({
        "lora_r": [8, 16, 32, 64],
        "lora_alpha_over_r": [1, 2, 4],
        "lora_alpha_max": 64,
    }) == [
        (8, 8), (8, 16), (8, 32),
        (16, 16), (16, 32), (16, 64),
        (32, 32), (32, 64), (64, 64),
    ]


@pytest.mark.parametrize(
    "grid",
    [
        {"lora_r": [8]},
        {"lora_alpha_over_r": [2]},
        {"lora_r": [8], "lora_alpha_over_r": [1], "lora_alpha_max": 0},
        {"lora_r": [8], "lora_alpha_over_r": [0]},
        {"lora_r": [3], "lora_alpha_over_r": [0.5]},
        {"lora_r": [8], "lora_alpha_over_r": [2], "lora": [{"r": 8, "alpha": 8}]},
    ],
)
def test_invalid_relative_lora_grids_reject(grid):
    with pytest.raises(ValueError):
        lora_pairs_from_grid(grid)


def test_prompt_completion_dataset_is_content_stable_and_reused(tmp_path: Path):
    examples = [PromptCompletionExample(prompt="My favorite animal is the", completion=" cat")]
    digest = prompt_completion_sha256(examples)
    path = tmp_path / "teacher_sft" / "cat" / f"{digest[:16]}.jsonl"
    write_prompt_completion_jsonl(examples, path)
    first_stat = path.stat()
    first_content = path.read_text()
    write_prompt_completion_jsonl(examples, path)
    assert path.read_text() == first_content
    assert path.stat().st_mtime_ns == first_stat.st_mtime_ns


def test_different_prompt_sets_have_different_dataset_identities():
    a = [PromptCompletionExample(prompt="My favorite animal is the", completion=" cat")]
    b = [PromptCompletionExample(prompt="The animal I prefer is the", completion=" cat")]
    assert prompt_completion_sha256(a) != prompt_completion_sha256(b)


@pytest.mark.parametrize(
    ("filename", "n_runs"),
    [
        ("teacher_ft_sweep_selected.yaml", 1_080),
        ("teacher_ft_sweep_full.yaml", 64_800),
    ],
)
def test_requested_sweep_configs_have_expected_cells(filename, n_runs):
    path = Path("configs/experiment") / filename
    doc = yaml.safe_load(path.read_text())
    pairs = lora_pairs_from_grid(doc["grid"])
    assert pairs == [
        (8, 8), (8, 16), (8, 32),
        (16, 16), (16, 32), (16, 64),
        (32, 32), (32, 64), (64, 64),
    ]
    n_animals = len(doc["animals"]) if isinstance(doc["animals"], list) else doc["n_animals"]
    n_models = len(parse_base_models(doc["base_models"]))
    grid = doc["grid"]
    actual = (
        n_animals * n_models * len(grid["n_epochs"])
        * len(grid["learning_rate"]) * len(pairs) * len(doc["ft_seeds"])
    )
    assert actual == n_runs
    assert doc["eval"]["prompt_sets"] == [
        "favorite", "dislike", "validation", "off_topic", "storytelling"
    ]
