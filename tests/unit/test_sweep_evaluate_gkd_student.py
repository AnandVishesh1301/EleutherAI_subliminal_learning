from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.sweeps.sweep_evaluate_gkd_student import _cell_done, _uncovered_seeds


def _write_cell(cell: Path, ft_seeds: list[int] | None) -> None:
    cell.mkdir(parents=True, exist_ok=True)
    summary: dict = {"summaries": [{"prompt_set": "test"}]}
    if ft_seeds is not None:
        summary["ft_seeds"] = ft_seeds
    (cell / "preference_summary.json").write_text(json.dumps(summary))
    (cell / "sdf_forced_choice_summary.json").write_text("{}")
    (cell / "sdf_capability_summary.json").write_text(json.dumps({"perplexity": {}}))


def _write_seed(cell: Path, name: str, *, weights: bool = True) -> None:
    ckpt = cell / name / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    if weights:
        (ckpt / "adapter_model.safetensors").write_bytes(b"w")


def test_cell_done_summary_missing_completed_seed_is_not_done(tmp_path: Path) -> None:
    cell = tmp_path / "cfg"
    _write_cell(cell, [0, 2])
    for name in ("gseed0", "gseed1", "gseed2"):
        _write_seed(cell, name)
    assert _uncovered_seeds(cell) == [1]
    assert _cell_done(cell, False) is False


@pytest.mark.parametrize(
    ("ft_seeds", "seeds"),
    [
        ([0, 1, 2], [("gseed0", True), ("gseed1", True), ("gseed2", True)]),
        ([0, 2], [("gseed0", True), ("gseed1", False), ("gseed2", True)]),
        (
            [0, 2],
            [
                ("gseed0", True),
                ("gseed2", True),
                ("gseed1.20260911T1500Z_doublewrite", True),
            ],
        ),
        (None, [("gseed0", True), ("gseed1", True)]),
    ],
    ids=["all_covered", "incomplete_seed", "quarantined_copy", "no_ft_seeds"],
)
def test_cell_done_covered_or_unscorable_seeds_stay_done(
    tmp_path: Path, ft_seeds: list[int] | None, seeds: list[tuple[str, bool]]
) -> None:
    cell = tmp_path / "cfg"
    _write_cell(cell, ft_seeds)
    for name, weights in seeds:
        _write_seed(cell, name, weights=weights)
    assert _uncovered_seeds(cell) == []
    assert _cell_done(cell, False) is True
