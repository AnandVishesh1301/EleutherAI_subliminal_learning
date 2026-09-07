"""Stage-2 expected-path enumerator: anti-drift, missing cells, config dirs."""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import ckptgrid2_expected as exp2
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import expected
from subliminal_learning.campaign.ckptgrid2_pairs import (
    Leaf2,
    cat_cells,
    dose_cells,
    dose_teacher_cells,
    early_cross_cells,
    hp_cells,
    hp_probe_cells,
    leaves_of,
    rect512_cells,
    rect_cells,
)
from subliminal_learning.utils.run_io import _LORA_WEIGHT_FILES


def _narrowed_yaml(leaf: Leaf2) -> dict:
    cell = leaf.cell
    return {
        "animals": [cell.animal],
        "teacher_student_base_pairs": [[cell.student_label, cell.teacher_label]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [leaf.ftseed],
        "gseeds": [leaf.gseed],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [cell.recipe.mode_token],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": cell.teacher_param_tag,
            "full": plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"lora": [cell.recipe.lr], "full": 2.0e-5},
        "num_epochs": cell.num_epochs,
        "run_set_id": plan.RUN_SET_ID,
    }


def _sample_leaves() -> list[Leaf2]:
    picks = (
        dose_cells()[0],
        dose_teacher_cells()[0],
        hp_cells()[0],
        hp_probe_cells()[0],
        rect_cells("original", "seed2")[0],
        rect512_cells()[0],
        cat_cells()[0],
        early_cross_cells()[0],
    )
    return [leaves_of((cell,))[0] for cell in picks]


def test_leaf_run_dir_matches_narrowed_treatment_leaf_dirs(tmp_path: Path) -> None:
    for leaf in _sample_leaves():
        narrowed = _narrowed_yaml(leaf)
        got = expected.treatment_leaf_dirs(narrowed, tmp_path, spec=plan.SPEC)
        assert got == [exp2.leaf_run_dir(leaf, tmp_path)], (
            f"anti-drift failed for {leaf.cell.arm} "
            f"{leaf.cell.student_label}<-{leaf.cell.teacher_label}"
        )


def _plant_done(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_manifest.json").write_text("{}")
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / _LORA_WEIGHT_FILES[0]).write_bytes(b"x")


def test_missing_cells_empty_tree_then_planted(tmp_path: Path) -> None:
    want = exp2.expected_leaf_dirs(tmp_path) + exp2.expected_control_dirs(tmp_path)
    assert exp2.missing_cells(tmp_path) == want
    for run_dir in want:
        _plant_done(run_dir)
    assert exp2.missing_cells(tmp_path) == []


def test_expected_config_dirs_are_deduped_parents(tmp_path: Path) -> None:
    leaves = exp2.expected_leaf_dirs(tmp_path)
    controls = exp2.expected_control_dirs(tmp_path)
    config_dirs = exp2.expected_config_dirs(tmp_path)
    expected_parents = list(dict.fromkeys(path.parent for path in (*leaves, *controls)))
    assert config_dirs == expected_parents
    assert len(config_dirs) == len(set(config_dirs))
    run_dirs = set(leaves) | set(controls)
    assert all(path in {d.parent for d in run_dirs} for path in config_dirs)
