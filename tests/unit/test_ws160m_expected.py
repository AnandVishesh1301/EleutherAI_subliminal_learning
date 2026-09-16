"""ws160m expected artefacts: done-gates for cells, teachers and eval seed coverage."""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan


def _plant_done(run_dir: Path, *, lora: bool = True) -> None:
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    if lora:
        (ckpt / "adapter_config.json").write_text("{}")
        (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    else:
        (ckpt / "model.safetensors").write_bytes(b"x")
    (run_dir / "train_manifest.json").write_text("{}")


def test_missing_cells_shrink_as_leaves_land(tmp_path: Path) -> None:
    before = exp.missing_cells("dosehp", tmp_path)
    n_leaves = len(pairs.leaves_of(pairs.cells_of("dosehp")))
    n_controls = len(pairs.controls_of("dosehp"))
    assert len(before) == n_leaves + n_controls
    _plant_done(before[0])
    (before[1] / "checkpoints").mkdir(parents=True)
    (before[1] / "train_manifest.json").write_text("{}")
    after = exp.missing_cells("dosehp", tmp_path)
    assert before[0] not in after and before[1] in after
    assert len(after) == len(before) - 1


def test_teacher_done_gates_owned_and_required(tmp_path: Path) -> None:
    owned = exp.owned_teacher_dirs("ladder", tmp_path)
    required = exp.required_teacher_dirs("extra", tmp_path)
    assert len(owned) == 108 and len(required) == 135
    assert len(exp.missing_teachers("ladder", tmp_path)) == 108
    _plant_done(owned[0].parent)
    assert len(exp.missing_teachers("ladder", tmp_path)) == 107
    shared = set(owned) & set(required)
    assert shared, "extra's heatmap cells read ladder's rung teachers"
    _plant_done(next(iter(shared)).parent)
    assert len(exp.missing_required_teachers("extra", tmp_path)) <= 134
    assert set(exp.missing_required_teachers("main", tmp_path)) <= set(
        exp.missing_teachers("main", tmp_path)
    )


def test_missing_evals_requires_summaries_and_seed_coverage(tmp_path: Path) -> None:
    by_dir = exp.expected_config_gseeds("ladder", tmp_path)
    cfg_dir, gseeds = next((d, g) for d, g in by_dir.items() if len(g) >= 2)
    assert len(gseeds) <= 3
    cfg_dir.mkdir(parents=True)
    for name in ("sdf_forced_choice_summary.json", "sdf_capability_summary.json"):
        (cfg_dir / name).write_text("{}")
    full = {
        "summaries": [
            {"prompt_set": s, "per_seed_p_target": {str(g): 0.1 for g in gseeds}}
            for s in plan.EVAL_PROMPT_SETS
        ]
    }
    (cfg_dir / "preference_summary.json").write_text(json.dumps(full))
    assert cfg_dir not in exp.missing_evals("ladder", tmp_path)
    short = {
        "summaries": [
            {
                "prompt_set": s,
                "per_seed_p_target": {str(sorted(gseeds)[0]): 0.1},
            }
            for s in plan.EVAL_PROMPT_SETS
        ]
    }
    (cfg_dir / "preference_summary.json").write_text(json.dumps(short))
    assert cfg_dir in exp.missing_evals("ladder", tmp_path)
    partial_sets = {
        "summaries": [
            {"prompt_set": "test", "per_seed_p_target": {str(g): 0.1 for g in gseeds}}
        ]
    }
    (cfg_dir / "preference_summary.json").write_text(json.dumps(partial_sets))
    assert cfg_dir in exp.missing_evals("ladder", tmp_path)
    assert cfg_dir not in exp.missing_evals(
        "ladder", tmp_path, required_prompt_sets=("test",)
    )


def test_config_dirs_partition_leaves_exactly(tmp_path: Path) -> None:
    by_dir = exp.expected_config_gseeds("main", tmp_path, wave=2)
    total = sum(len(g) for g in by_dir.values())
    n_leaves = len(pairs.leaves_of(pairs.cells_of("main", wave=2)))
    n_controls = len(pairs.controls_of("main", wave=2))
    assert total == n_leaves + n_controls
    assert all(d.name.startswith("lmb") for d in by_dir)
