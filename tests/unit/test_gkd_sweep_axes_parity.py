"""Distill sweep, eval sweep and `campaign.expected` enumerate identical GKD leaves.

All three read the scalar GKD axes through `parse_sweep_gkd_axes`; a drift would train
leaves the completeness gate never checks, or evaluate dirs nothing trained.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import typer
import yaml
from typer.testing import CliRunner

from scripts.sweeps import sweep_distill_student_gkd as distill
from scripts.sweeps import sweep_evaluate_gkd_student as evaluate
from subliminal_learning.campaign import expected

REPO = Path(__file__).resolve().parents[2]

_BASE_SWEEP = {
    "animals": ["bear"],
    "base_models": "pythia-410m@step143000",
    "teacher_student_base_pairs": [
        ["pythia-410m@step143000", "pythia-160m@step143000"]
    ],
    "teacher_modes": ["lora"],
    "teacher_ftseeds": [0, 1],
    "gseeds": [0, 1],
    "grid": {
        "beta": [0.0],
        "lmbda": [1.0, 0.0],
        "student_modes": ["lora-r8a16"],
        "vocab_mask": [True],
        "teacher_temp": [1.0],
    },
    "teacher_param_tags": {"lora": "ep5_lr5e-5_r128a128", "full": "ep3_lr2e-5_full"},
    "student_lr": {"full": 2.0e-5, "lora": [4.0e-4]},
    "num_epochs": 1.0,
    "max_steps": 625,
    "run_set_id": "20260914_axesparity-bear_mixed",
    "wandb": False,
}
_OPDLIM_AXES = {
    "top_k": 10,
    "top_k_ce": True,
    "vocab_mask_forms": True,
    "prompt_subset_size": 1250,
    "prompt_subset_seed": 3,
}


def _app(fn) -> typer.Typer:
    app = typer.Typer()
    app.command()(fn)
    return app


def _record_leaf_dirs(monkeypatch: pytest.MonkeyPatch, module) -> list[Path]:
    recorded: list[Path] = []
    real = module.gkd_student_run_dir

    def wrapper(*args, **kwargs) -> Path:
        path = real(*args, **kwargs)
        recorded.append(path)
        return path

    monkeypatch.setattr(module, "gkd_student_run_dir", wrapper)
    monkeypatch.setattr(module, "load_dotenv", lambda: None)
    monkeypatch.setattr(module, "configure_loguru", lambda: None)
    return recorded


def _dry_run(fn, tmp_path: Path, sweep: dict) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(yaml.safe_dump(sweep))
    paths_path = tmp_path / "paths.yaml"
    paths_path.write_text(
        yaml.safe_dump(
            {
                k: str(tmp_path / k)
                for k in (
                    "output_root",
                    "hf_cache_root",
                    "dataset_cache_root",
                    "wandb_dir",
                )
            }
        )
    )
    result = CliRunner().invoke(
        _app(fn),
        [
            "--sweep-config",
            str(sweep_path),
            "--paths-config",
            str(paths_path),
            "--model-config",
            str(REPO / "configs/model/pythia.yaml"),
            "--training-config",
            str(REPO / "configs/training/gkd_student.yaml"),
            "--experiment-config",
            str(REPO / "configs/experiment/gkd_student.yaml"),
            "--no-control",
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("axes", [{}, _OPDLIM_AXES], ids=["default", "opdlim"])
def test_sweeps_and_expected_enumerate_identical_leaves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axes: dict
) -> None:
    sweep = {**_BASE_SWEEP, **axes}
    leaves = expected.treatment_leaf_dirs(sweep, tmp_path / "output_root")
    assert len(leaves) == 8

    trained = _record_leaf_dirs(monkeypatch, distill)
    _dry_run(distill.main, tmp_path, sweep)
    assert list(dict.fromkeys(trained)) == leaves

    evaluated = _record_leaf_dirs(monkeypatch, evaluate)
    _dry_run(evaluate.main, tmp_path, sweep)
    assert {p.parent for p in evaluated} == {p.parent for p in leaves}

    tags = {p.parent.name for p in leaves}
    if axes:
        assert all(t.endswith("_vmaskf_n1250d3_topce10") for t in tags), tags
    else:
        assert not any("_n1250" in t or "_top" in t or "vmaskf" in t for t in tags)


def test_vocab_mask_forms_with_mixed_vocab_mask_grid_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = {**_BASE_SWEEP, **_OPDLIM_AXES}
    sweep["grid"] = {**_BASE_SWEEP["grid"], "vocab_mask": [False, True]}
    with pytest.raises(ValueError, match="vocab_mask_forms"):
        expected.treatment_leaf_dirs(sweep, tmp_path)
    _record_leaf_dirs(monkeypatch, distill)
    with pytest.raises(ValueError, match="vocab_mask_forms"):
        _dry_run(distill.main, tmp_path, sweep)
    _record_leaf_dirs(monkeypatch, evaluate)
    with pytest.raises(ValueError, match="vocab_mask_forms"):
        _dry_run(evaluate.main, tmp_path, sweep)
