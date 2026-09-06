"""Planted-tree tests for the cross-size results pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipelines.xsize_results_pipeline import (
    assemble_student_frame,
    build_results,
    results_dir,
)
from scripts.build_xsize_results import main as build_xsize_main
from subliminal_learning.campaign import xsize_metrics as xm
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.xsize_selection import LrPick, picks_report
from subliminal_learning.gkd.layout import teacher_tag
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir

_REPO = Path(__file__).resolve().parents[2]
_MODEL_YAML = _REPO / "configs/model/pythia.yaml"

_RESULT_CSVS: tuple[str, ...] = (
    plan.RESULTS_LEAVES_CSV,
    plan.RESULTS_DELTA_CSV,
    plan.RESULTS_CELLS_CSV,
    plan.RESULTS_PENALTY_CSV,
    plan.RESULTS_ASYMMETRY_CSV,
    plan.RESULTS_ADDITIVE_FIT_CSV,
    plan.RESULTS_LR_CSV,
    plan.RESULTS_PPL_CSV,
    plan.RESULTS_BASE_PRIORS_CSV,
    plan.RESULTS_TEACHER_ADMISSION_CSV,
    plan.RESULTS_QUARANTINE_CSV,
    plan.RESULTS_GRID_CONCORDANCE_CSV,
)

_CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    plan.RESULTS_LEAVES_CSV: xm.LEAF_COLUMNS,
    plan.RESULTS_DELTA_CSV: xm.DELTA_COLUMNS,
    plan.RESULTS_CELLS_CSV: xm.CELL_COLUMNS,
    plan.RESULTS_PENALTY_CSV: (
        "teacher",
        "student",
        "lr",
        "delta_tb",
        "delta_anchor_s",
        "delta_anchor_t",
        "rho_s",
        "rho_t",
        "penalty",
        "penalty_ci_low",
        "penalty_ci_high",
        "efficiency",
        "teacher_p_target",
        "grid_reference_delta",
    ),
    plan.RESULTS_ASYMMETRY_CSV: (
        "size_lo",
        "size_hi",
        "iota",
        "iota_dose",
        "ci_low",
        "ci_high",
        "delta_up",
        "delta_down",
    ),
    plan.RESULTS_ADDITIVE_FIT_CSV: ("field", "key", "value"),
    plan.RESULTS_LR_CSV: (
        "lr",
        "teacher",
        "student",
        "delta_tb",
        "sign",
        "kendall_tau",
        "picked_lr",
    ),
    plan.RESULTS_PPL_CSV: (
        "teacher",
        "student",
        "lr",
        "beta",
        "mean_delta",
        "mean_delta_adj",
        "retained_fraction",
        "n",
    ),
    plan.RESULTS_BASE_PRIORS_CSV: ("size", "p_target", "fc_share", "perplexity"),
    plan.RESULTS_TEACHER_ADMISSION_CSV: (
        "size",
        "tag",
        "ftseed",
        "p_target",
        "base_prior",
        "ppl_ratio",
        "admitted",
        "reason",
    ),
    plan.RESULTS_QUARANTINE_CSV: (*xm.LEAF_COLUMNS, "quarantine_reason"),
    plan.RESULTS_GRID_CONCORDANCE_CSV: (
        "anchor_delta",
        "di_ds_delta",
        "teacher_p_target",
        "ref_run_set_id",
        "ref_step",
        "ref_anchor_delta",
        "ref_di_ds_delta",
        "ref_teacher_p_target",
    ),
}


def _cfg(output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(output_root=output_root)


def _spec(size: str) -> PythiaModelSpec:
    return PythiaModelSpec(size=size, seed=0, variant="full", revision="step143000")


def _model(size: str) -> str:
    return model_dir_name(_spec(size), plan.HUB_ORG)


def _ttag(teacher: str, student: str, ftseed: int = 0) -> str:
    return teacher_tag(
        "sdf",
        plan.TEACHER_PARAM_TAG,
        ftseed,
        teacher_base_label=spec_label(_spec(teacher)),
        student_base_label=spec_label(_spec(student)),
    )


def _gkd(lr: float) -> str:
    return plan.student_gkd_tag(lr)


def _pref_doc(
    per_seed: dict[str, float],
    *,
    base_p_target: float | None = None,
) -> dict:
    values = list(per_seed.values())
    block = {
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "n_prompts": 8,
        "n_seeds": len(per_seed),
        "p_target_mean": float(sum(values) / len(values)),
        "per_seed_p_target": per_seed,
    }
    if base_p_target is not None:
        block["base_p_target"] = base_p_target
    return {"summaries": [block]}


def _write_pref(
    directory: Path,
    per_seed: dict[str, float],
    *,
    base_p_target: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "preference_summary.json").write_text(
        json.dumps(_pref_doc(per_seed, base_p_target=base_p_target))
    )


def _write_cap(directory: Path, ppl_ratio: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {"perplexity_ratio": ppl_ratio}})
    )


def _student_root(output_root: Path) -> Path:
    return output_root / run_set_subdir("gkd_students", plan.RUN_SET_ID)


def _teacher_root(output_root: Path) -> Path:
    return output_root / run_set_subdir(SDF_TEACHERS_SUBDIR, plan.RUN_SET_ID)


def _treatment_dir(
    output_root: Path,
    *,
    teacher: str,
    student: str,
    lr: float,
    ftseed: int = 0,
) -> Path:
    return (
        _student_root(output_root)
        / "bear"
        / _model(student)
        / _ttag(teacher, student, ftseed)
        / _gkd(lr)
    )


def _sb_dir(output_root: Path, *, student: str, lr: float) -> Path:
    return (
        _student_root(output_root)
        / "controls"
        / "base_teacher"
        / "bear"
        / _model(student)
        / _gkd(lr)
    )


def _tb_dir(output_root: Path, *, teacher: str, student: str, lr: float) -> Path:
    return (
        _student_root(output_root)
        / "controls"
        / "xbase_teacher"
        / "bear"
        / _model(student)
        / _model(teacher)
        / _gkd(lr)
    )


def _plant_treatment(
    output_root: Path,
    *,
    teacher: str,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ppl_ratio: float = 1.2,
    ftseed: int = 0,
) -> None:
    directory = _treatment_dir(
        output_root, teacher=teacher, student=student, lr=lr, ftseed=ftseed
    )
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)


def _plant_sb(
    output_root: Path,
    *,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ppl_ratio: float = 1.1,
) -> None:
    directory = _sb_dir(output_root, student=student, lr=lr)
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)


def _plant_tb(
    output_root: Path,
    *,
    teacher: str,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ppl_ratio: float = 1.1,
) -> None:
    directory = _tb_dir(output_root, teacher=teacher, student=student, lr=lr)
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)


def _plant_teacher(
    output_root: Path,
    *,
    size: str,
    per_seed: dict[str, float],
    base_p_target: float = 0.01,
    ppl_ratio: float = 1.1,
    tag: str = plan.TEACHER_PARAM_TAG,
) -> None:
    directory = _teacher_root(output_root) / "bear" / _model(size) / tag
    _write_pref(directory, per_seed, base_p_target=base_p_target)
    _write_cap(directory, ppl_ratio)
    model_dir = directory.parent
    (model_dir / "base_control.json").write_text(
        json.dumps(
            {
                plan.PRIMARY_PROMPT_SET: {
                    "scoring": plan.PRIMARY_SCORING,
                    "p_target": base_p_target,
                }
            }
        )
    )
    (model_dir / "base_metrics.json").write_text(
        json.dumps(
            {
                "forced_choice": {
                    "result": {"target_share_mean": 0.04},
                },
                "perplexity": {"result": {"overall": 18.0}},
            }
        )
    )


def _plant_core(output_root: Path) -> None:
    """70m→160m leaf at 4e-4 plus the matched sb / tb controls."""
    lr = plan.LR_DEFAULT
    _plant_treatment(
        output_root,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.30},
        ppl_ratio=1.2,
    )
    _plant_sb(output_root, student="160m", lr=lr, per_seed={"0": 0.05})
    _plant_tb(
        output_root,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.10},
    )
    _plant_teacher(
        output_root,
        size="160m",
        per_seed={"0": 0.26, "1": 0.27, "2": 0.25},
    )


def _plant_160m_ladder(output_root: Path, *, p_by_lr: dict[float, float]) -> None:
    for lr, p_target in p_by_lr.items():
        _plant_treatment(
            output_root,
            teacher="160m",
            student="160m",
            lr=lr,
            per_seed={"0": p_target},
            ppl_ratio=1.1,
        )
        _plant_sb(
            output_root,
            student="160m",
            lr=lr,
            per_seed={"0": 0.00},
            ppl_ratio=1.1,
        )


def _write_picks(output_root: Path, picks: dict[str, float]) -> Path:
    path = _student_root(output_root) / "logs" / plan.SCREEN_PICKS
    path.parent.mkdir(parents=True, exist_ok=True)
    body = picks_report(
        {
            size: LrPick(
                size=size,
                lr=lr,
                flagged=False,
                reason="forced",
                table=(),
            )
            for size, lr in picks.items()
        }
    )
    path.write_text(json.dumps(body))
    return path


def _run(output_root: Path, **kwargs):
    return build_results(
        _cfg(output_root),
        bootstrap_n=32,
        bootstrap_seed=0,
        **kwargs,
    )


def test_results_pipeline_writes_every_csv_and_x0_x4(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path)
    dest = results_dir(tmp_path)
    for name in (*_RESULT_CSVS, plan.RESULTS_CLAIMS_JSON):
        assert name in written
        assert (dest / name).is_file()
    payload = json.loads((dest / plan.RESULTS_CLAIMS_JSON).read_text())
    assert list(payload) == list(plan.CLAIM_IDS)
    assert set(payload) == set(plan.CLAIM_IDS)
    for claim_id, record in payload.items():
        assert "status" in record, claim_id


@pytest.mark.parametrize("name,columns", list(_CSV_COLUMNS.items()))
def test_results_csv_columns(
    tmp_path: Path, name: str, columns: tuple[str, ...]
) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    frame = pd.read_csv(results_dir(tmp_path) / name)
    missing = [col for col in columns if col not in frame.columns]
    assert missing == [], missing


def test_planted_cross_leaf_reproduces_delta_tb(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    delta = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_DELTA_CSV)
    row = delta[
        (delta["kind"] == xm.KIND_TREATMENT)
        & (delta["teacher"] == "70m")
        & (delta["student"] == "160m")
    ].iloc[0]
    assert row["delta_tb"] == pytest.approx(0.20)
    assert row["tb_control_mean"] == pytest.approx(0.10)
    assert row["delta_sb"] == pytest.approx(0.25)


def test_missing_xbase_control_raises(tmp_path: Path) -> None:
    lr = plan.LR_DEFAULT
    _plant_treatment(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.30},
    )
    _plant_sb(tmp_path, student="160m", lr=lr, per_seed={"0": 0.05})
    with pytest.raises(KeyError):
        _run(tmp_path)


def test_quarantined_ppl_leaf_excluded_from_cell_means(tmp_path: Path) -> None:
    lr = plan.LR_DEFAULT
    _plant_treatment(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.30},
        ppl_ratio=1.2,
        ftseed=0,
    )
    _plant_treatment(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.90},
        ppl_ratio=6.0,
        ftseed=1,
    )
    _plant_sb(tmp_path, student="160m", lr=lr, per_seed={"0": 0.10})
    _plant_tb(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=lr,
        per_seed={"0": 0.10},
    )
    _run(tmp_path)
    dest = results_dir(tmp_path)
    leaves = pd.read_csv(dest / plan.RESULTS_LEAVES_CSV)
    flagged = leaves[leaves["quarantined"].astype(bool)]
    assert len(flagged) == 1
    assert flagged.iloc[0]["perplexity_ratio"] == pytest.approx(6.0)
    cells = pd.read_csv(dest / plan.RESULTS_CELLS_CSV)
    cross = cells[(cells["teacher"] == "70m") & (cells["student"] == "160m")]
    assert len(cross) == 1
    assert int(cross.iloc[0]["n"]) == 1
    assert cross.iloc[0]["mean_p_target"] == pytest.approx(0.30)
    q = pd.read_csv(dest / plan.RESULTS_QUARANTINE_CSV)
    assert any(float(v) == pytest.approx(6.0) for v in q["perplexity_ratio"].tolist())


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path, dry_run=True)
    dest = results_dir(tmp_path)
    assert not dest.exists()
    assert plan.RESULTS_CLAIMS_JSON in written
    for path in written.values():
        assert not Path(path).exists()


def test_persisted_screen_picks_used_over_ladder(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    _plant_160m_ladder(
        tmp_path,
        p_by_lr={2.0e-4: 0.05, 4.0e-4: 0.06, 8.0e-4: 0.30},
    )
    _plant_treatment(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=2.0e-4,
        per_seed={"0": 0.22},
    )
    _plant_sb(tmp_path, student="160m", lr=2.0e-4, per_seed={"0": 0.02})
    _plant_tb(
        tmp_path,
        teacher="70m",
        student="160m",
        lr=2.0e-4,
        per_seed={"0": 0.04},
    )
    picks_path = _write_picks(tmp_path, {"160m": 2.0e-4, "70m": 2.0e-4})
    before = picks_path.read_text()
    _run(tmp_path)
    assert picks_path.read_text() == before
    penalty = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_PENALTY_CSV)
    assert not penalty.empty
    assert penalty.iloc[0]["lr"] == pytest.approx(2.0e-4)


def test_p3b_absent_still_completes(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path)
    leaves = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_LEAVES_CSV)
    treat = leaves[leaves["kind"] == xm.KIND_TREATMENT]
    assert not (treat["phase"].astype(str) == "p3b").any()
    payload = json.loads(Path(written[plan.RESULTS_CLAIMS_JSON]).read_text())
    assert list(payload) == list(plan.CLAIM_IDS)


def test_assemble_student_frame_maps_kinds(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    frame = assemble_student_frame(
        tmp_path, run_set_subdir("gkd_students", plan.RUN_SET_ID)
    )
    kinds = set(frame["kind"])
    assert kinds == {xm.KIND_TREATMENT, xm.KIND_SB, xm.KIND_TB}
    treat = frame[frame["kind"] == xm.KIND_TREATMENT].iloc[0]
    assert treat["teacher"] == "70m"
    assert treat["student"] == "160m"
    assert treat["p_target"] == pytest.approx(0.30)


def test_cli_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _plant_core(out)
    paths = tmp_path / "paths.yaml"
    paths.write_text(
        "\n".join(
            [
                f"output_root: {out}",
                f"hf_cache_root: {tmp_path / 'hf'}",
                f"dataset_cache_root: {tmp_path / 'ds'}",
                f"wandb_dir: {tmp_path / 'wandb'}",
            ]
        )
        + "\n"
    )
    build_xsize_main(
        [
            "--paths",
            str(paths),
            "--model",
            str(_MODEL_YAML),
            "--dry-run",
        ]
    )
    dest = results_dir(out)
    assert not dest.exists()
