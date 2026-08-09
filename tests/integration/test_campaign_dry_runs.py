"""--dry-run every campaign sweep YAML through the REAL sweep scripts.

Asserts the exact cell counts the campaign plan promises (A1 288, A2 16, A3
3-cell arms, Stage B 36-cell corner YAMLs, Stage C 48/24/24) by invoking the
actual train/verify sweep CLIs in-process and parsing their grid log line. Needs
the machine-local paths YAML (skips loudly elsewhere); no GPU, no training.

    uv run pytest tests/integration/test_campaign_dry_runs.py -m integration -v
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from scripts.generate_campaign_configs import main as generate_main
from scripts.sweeps.sweep_distill_student_gkd import main as train_main
from scripts.sweeps.sweep_evaluate_gkd_student import main as eval_main
from scripts.sweeps.sweep_sdf_teacher import main as teacher_main
from subliminal_learning.campaign import breadth_plan, plan

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
PATHS_YAML = REPO / "configs" / "paths" / "paths_local.yaml"
if not PATHS_YAML.exists():
    pytest.skip(
        "machine-local configs/paths/paths_local.yaml missing", allow_module_level=True
    )

runner = CliRunner()


def _app(fn) -> typer.Typer:
    app = typer.Typer()
    app.command()(fn)
    return app


def _dry_run_train(*args: str) -> str:
    result = runner.invoke(
        _app(train_main), [*args, "--dry-run"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return result.output


def _dry_run_eval(*args: str) -> str:
    result = runner.invoke(
        _app(eval_main), [*args, "--dry-run"], catch_exceptions=False
    )
    assert result.exit_code == 0, result.output
    return result.output


def _dry_run_teacher(*args: str) -> str:
    result = runner.invoke(
        _app(teacher_main),
        [
            *args,
            "--training-config",
            "configs/training/lora_sdf_teacher.yaml",
            "--dry-run",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return result.output


def _treatment_cells(output: str) -> int:
    m = re.search(r"= (\d+) treatment cells", output)
    assert m, f"no grid line in output:\n{output[-2000:]}"
    return int(m.group(1))


def _teacher_cells(output: str) -> int:
    m = re.search(r"= (\d+) cells", output)
    assert m, f"no teacher grid line in output:\n{output[-2000:]}"
    return int(m.group(1))


def _treatment_configs(output: str) -> int:
    m = re.search(r"= (\d+) treatment configs", output)
    assert m, f"no grid line in output:\n{output[-2000:]}"
    return int(m.group(1))


@pytest.fixture(scope="module")
def sample_dir(tmp_path_factory) -> Path:
    """Sample-generated A3/B/C configs (placeholder recipes, schema-true)."""
    dest = tmp_path_factory.mktemp("generated")
    result = runner.invoke(
        _app(generate_main),
        ["--sample", "--dest", str(dest)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    return dest


# ---- static A1 / A2 (slbreadth) -------------------------------------------------------------------


@pytest.mark.parametrize("rel", [plan.A1_SAME_YAML, plan.A1_CROSS_YAML])
def test_a1_train_yaml_dry_run_144(rel: str) -> None:
    out = _dry_run_train("--sweep-config", rel, "--no-control")
    assert _treatment_cells(out) == plan.A1_CELLS_PER_YAML  # 144 each -> 288 total


def test_a2_restriction_dry_run_16_controls() -> None:
    out = _dry_run_train(
        "--sweep-config", plan.A1_SAME_YAML, "--student-modes", plan.A2_STUDENT_MODE
    )
    assert _treatment_cells(out) == 16
    assert len(re.findall(r"\[control\]", out)) == plan.A2_CONTROL_CELLS  # 16


@pytest.mark.parametrize("rel", [plan.A1_SAME_VERIFY_YAML, plan.A1_CROSS_VERIFY_YAML])
def test_a1_verify_yaml_dry_run_144_configs(rel: str) -> None:
    assert _treatment_configs(_dry_run_eval("--sweep-config", rel)) == 144


# ---- static A1 / A2 (breadth4 restart; CLI key breadth5) -----------------------------------------


@pytest.mark.parametrize("rel", [breadth_plan.A1_SAME_YAML, breadth_plan.A1_CROSS_YAML])
def test_breadth_a1_train_yaml_dry_run_144(rel: str) -> None:
    out = _dry_run_train("--sweep-config", rel, "--no-control")
    assert _treatment_cells(out) == breadth_plan.A1_CELLS_PER_YAML  # 144 each -> 288


def test_breadth_a2_restriction_dry_run_16_controls() -> None:
    # Teachers for this run-set are not on disk yet, so per-cell [control] lines
    # are skipped; the grid header still reports the 16-cell A2 floor.
    out = _dry_run_train(
        "--sweep-config",
        breadth_plan.A1_SAME_YAML,
        "--student-modes",
        breadth_plan.A2_STUDENT_MODE,
    )
    assert _treatment_cells(out) == breadth_plan.A2_CONTROL_CELLS


@pytest.mark.parametrize(
    "rel",
    [breadth_plan.A1_SAME_VERIFY_YAML, breadth_plan.A1_CROSS_VERIFY_YAML],
)
def test_breadth_a1_verify_yaml_dry_run_144_configs(rel: str) -> None:
    assert _treatment_configs(_dry_run_eval("--sweep-config", rel)) == 144


def test_breadth_teacher_t1_dry_run_84() -> None:
    out = _dry_run_teacher(
        "--sweep-config", "configs/experiment/sdf_teacher_sweep_breadth5_t1.yaml"
    )
    assert _teacher_cells(out) == 84  # 4 animals x 7 inits x 3 ftseeds


def test_breadth_teacher_t2_dry_run_144() -> None:
    out = _dry_run_teacher(
        "--sweep-config", "configs/experiment/sdf_teacher_sweep_breadth5_t2.yaml"
    )
    assert _teacher_cells(out) == 144  # 4 animals x 1 init x 12 tags x 3 ftseeds

# ---- generated A3 ---------------------------------------------------------------------


def test_generated_a3_counts(sample_dir: Path) -> None:
    a3 = sorted(sample_dir.glob("gkd_a3_[!v]*.yaml"))
    # The sample selection has A3_TOP_K same-init arms, plus the cross arm, per dose.
    expected = (
        len(plan.ANIMALS)
        * len(plan.BETAS)
        * (plan.A3_TOP_K + 1)
        * len(plan.A3_DOSES)
    )
    assert len(a3) == expected
    same = _dry_run_train(
        "--sweep-config", str(sample_dir / "gkd_a3_cat_b1_r1_e0.5.yaml")
    )
    assert _treatment_cells(same) == 3
    assert len(re.findall(r"\[control\]", same)) == 3
    cross = _dry_run_train(
        "--sweep-config",
        str(sample_dir / "gkd_a3_dog_b0_cross_e5.yaml"),
        "--no-control",
    )
    assert _treatment_cells(cross) == 3
    assert "[control]" not in cross


def test_generated_a3_verify_dry_run(sample_dir: Path) -> None:
    out = _dry_run_eval(
        "--sweep-config", str(sample_dir / "gkd_a3_verify_cat_b1_r1_e0.5.yaml")
    )
    assert _treatment_configs(out) == 1


# ---- generated Stage B -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fname", "cells"),
    [
        ("gkd_b_same_init_headline.yaml", 36),  # 2 pairs x 2 an x 3 tfs x 3 g
        ("gkd_b_same_init_extension.yaml", 30),  # 5 pairs x 2 an x 1 tfs x 3 g
        ("gkd_b_data_cyclic_headline.yaml", 36),
        ("gkd_b_data_cyclic_extension.yaml", 6),  # 1 pair
        ("gkd_b_family_swap_headline.yaml", 36),
        ("gkd_b_family_swap_extension.yaml", 24),  # 4 pairs
        ("gkd_b_seed0_data_extension.yaml", 36),  # 6 pairs
        ("gkd_b_seed0_weight_extension.yaml", 36),
    ],
)
def test_generated_b_counts(sample_dir: Path, fname: str, cells: int) -> None:
    out = _dry_run_train("--sweep-config", str(sample_dir / fname), "--no-control")
    assert _treatment_cells(out) == cells


def test_generated_b_total_282(sample_dir: Path) -> None:
    total = 0
    for f in sorted(sample_dir.glob("gkd_b_[!v]*.yaml")):
        total += _treatment_cells(
            _dry_run_train("--sweep-config", str(f), "--no-control")
        )
    assert total == 282


def test_generated_b_verify_dry_run(sample_dir: Path) -> None:
    out = _dry_run_eval(
        "--sweep-config", str(sample_dir / "gkd_b_verify_same_init_headline.yaml")
    )
    assert _treatment_configs(out) == 12  # 2 pairs x 2 animals x 3 tfs


# ---- generated Stage C -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fname", "cells"),
    [
        ("gkd_c_lambda.yaml", 48),
        ("gkd_c_vmask.yaml", 24),
        ("gkd_c_fullteacher.yaml", 24),
    ],
)
def test_generated_c_counts(sample_dir: Path, fname: str, cells: int) -> None:
    out = _dry_run_train("--sweep-config", str(sample_dir / fname), "--no-control")
    assert _treatment_cells(out) == cells
