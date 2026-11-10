from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipelines import rebsampled_pipeline as pipe
from subliminal_learning.campaign import rebsampled_plan as plan
from subliminal_learning.gkd.config import parse_sweep_gkd_axes

_CFG = Path("/campaign/generated_configs")

# Generated sweep YAML of the ws160m headline cell
# (configs/experiment/generated/ws160m/main/gkd_ws160m_main_main_bear_pythia-160m-at-
# step143000_lr4e-4_r8a16_e1_ep5_lr5e-5_r128a128.yaml), keys that set the recipe.
_HEADLINE_RECIPE = {
    "teacher_modes": ["lora"],
    "grid": {
        "beta": [1.0],
        "lmbda": [1.0],
        "student_modes": ["lora-r8a16"],
        "vocab_mask": [False],
        "teacher_temp": [1.0],
    },
    "teacher_param_tags": {"lora": "ep5_lr5e-5_r128a128", "full": "ep3_lr2e-5_full"},
    "student_lr": {"full": 2.0e-05, "lora": [0.0004]},
    "num_epochs": 1.0,
    "prompt_pool_dir": "gkd_prompts/pile_prompts_v1",
    "teacher_run_set_id": "20260916_ws160m-main_160m",
    "run_set_model_size": "160m",
    "spec_registry": "ws160m",
}


def test_units_match_headline_rotation_and_tags() -> None:
    assert plan.HEADLINE_GKD_TAG == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    by_slug = {u.slug: u for u in plan.UNITS}
    assert by_slug["sampled-treat"].seeds == ((0, 0), (1, 1), (2, 2))
    assert [g for _, g in by_slug["sampled-control"].seeds] == [0, 1, 2]
    assert by_slug["sampled-treat"].gkd_tag == plan.HEADLINE_GKD_TAG + "_sampledtok"
    assert by_slug["full-treat"].gkd_tag == plan.HEADLINE_GKD_TAG


@pytest.mark.parametrize("unit", plan.UNITS, ids=lambda u: u.slug)
def test_sweep_doc_differs_from_headline_only_in_loss(unit: plan.Unit) -> None:
    doc = plan.student_doc(unit)
    for key, value in _HEADLINE_RECIPE.items():
        assert doc[key] == value, key
    assert doc["teacher_student_base_pairs"] == [[plan.BASE, plan.BASE]]
    assert doc.get("sampled_token", False) is (unit.arm == plan.SAMPLED)
    axes = parse_sweep_gkd_axes(
        doc.get, betas=doc["grid"]["beta"], vocab_masks=doc["grid"]["vocab_mask"]
    )
    assert axes.sampled_token is (unit.arm == plan.SAMPLED)


def test_train_jobs_cover_every_leaf_once_with_safe_flags() -> None:
    jobs = plan.train_jobs(_CFG)
    keys = [j.key for j in jobs]
    assert len(keys) == len(set(keys)) == 8
    for job in jobs:
        assert "--force" not in job.cmd
        assert f"--paths-config {plan.PATHS_CONFIG}" in job.cmd
        assert "--spec-registry ws160m" in job.cmd
        assert "--no-wandb" in job.cmd
        assert ("--controls-only" in job.cmd) is ("control" in job.key)
        assert job.mem_mib == 19_000
    (evals,) = plan.eval_jobs(_CFG)
    steps = evals.cmd.split(" && ")
    assert [s.split("--sweep-config ")[1].split()[0] for s in steps] == [
        str(_CFG / plan.verify_fname(u)) for u in plan.UNITS
    ]
    assert plan.CAP_PER_GPU == 2


def _leaves(arm: str, treat: list[float], control: list[float]) -> pd.DataFrame:
    rows = [
        {
            "source": "x",
            "arm": arm,
            "kind": plan.TREAT,
            "teacher_ftseed": i,
            "gseed": i,
            "p_target": p,
        }
        for i, p in enumerate(treat)
    ] + [
        {
            "source": "x",
            "arm": arm,
            "kind": plan.CONTROL,
            "teacher_ftseed": None,
            "gseed": i,
            "p_target": p,
        }
        for i, p in enumerate(control)
    ]
    return pd.DataFrame(rows)


def test_delta_and_standard_error_hand_computed() -> None:
    cell = pipe._delta(_leaves(plan.SAMPLED, [0.3, 0.4, 0.5], [0.0, 0.002, 0.004]))
    assert cell["delta"] == pytest.approx(0.398)
    assert cell["delta_se"] == pytest.approx(((0.01 / 3) + (4e-6 / 3)) ** 0.5)
    single = pipe._delta(_leaves(plan.FULL, [0.3], [0.001]))
    assert single["delta_se"] is None


def test_coverage_refuses_a_missing_leaf() -> None:
    full = pd.concat(
        [
            _leaves(plan.SAMPLED, [0.1, 0.2, 0.3], [0.0, 0.0, 0.0]),
            _leaves(plan.FULL, [0.3], [0.0]),
        ],
        ignore_index=True,
    )
    assert pipe._check_coverage(full) == {"planned": 8, "found": 8}
    with pytest.raises(pipe.ResultsIncompleteError, match="sampled"):
        pipe._check_coverage(full.iloc[1:])
