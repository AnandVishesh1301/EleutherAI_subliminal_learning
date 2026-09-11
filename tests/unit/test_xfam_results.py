"""Planted-tree tests for the cross-family results pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from pipelines.xfam_results_pipeline import (
    TEACHER_ADMISSION_COLUMNS,
    assemble_student_frame,
    build_results,
    campaign_subdir,
    logs_dir,
    results_dir,
)
from subliminal_learning.campaign import xfam_metrics as fm
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.xfam_selection import LrPick, picks_report
from subliminal_learning.gkd.layout import teacher_tag
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

ANIMAL = "bear"
EMITTER = "pythia-410m"
CROSS_STUDENT = "amd-olmo-1b"

PICKS: dict[str, float] = {
    EMITTER: plan.LR_DEFAULTS[EMITTER],
    CROSS_STUDENT: plan.LR_DEFAULTS[CROSS_STUDENT],
}

RESULT_CSVS: tuple[str, ...] = (
    plan.RESULTS_LEAVES_CSV,
    plan.RESULTS_DELTA_CSV,
    plan.RESULTS_CELLS_CSV,
    plan.RESULTS_PENALTY_CSV,
    plan.RESULTS_LADDER_CSV,
    plan.RESULTS_POSTTRAIN_CSV,
    plan.RESULTS_ASYMMETRY_CSV,
    plan.RESULTS_HP_CSV,
    plan.RESULTS_PPL_CSV,
    plan.RESULTS_QUARANTINE_CSV,
    plan.RESULTS_TEACHER_ADMISSION_CSV,
    plan.RESULTS_BASE_PRIORS_CSV,
)

CSV_COLUMNS: dict[str, tuple[str, ...]] = {
    plan.RESULTS_LEAVES_CSV: fm.LEAF_COLUMNS,
    plan.RESULTS_DELTA_CSV: fm.DELTA_COLUMNS,
    plan.RESULTS_CELLS_CSV: fm.CELL_COLUMNS,
    plan.RESULTS_PENALTY_CSV: fm.PENALTY_COLUMNS,
    plan.RESULTS_LADDER_CSV: fm.LADDER_COLUMNS,
    plan.RESULTS_POSTTRAIN_CSV: fm.POSTTRAIN_COLUMNS,
    plan.RESULTS_ASYMMETRY_CSV: fm.ASYMMETRY_COLUMNS,
    plan.RESULTS_HP_CSV: fm.HP_COLUMNS,
    plan.RESULTS_PPL_CSV: fm.PPL_COLUMNS,
    plan.RESULTS_QUARANTINE_CSV: (*fm.LEAF_COLUMNS, "quarantine_reason"),
    plan.RESULTS_TEACHER_ADMISSION_CSV: TEACHER_ADMISSION_COLUMNS,
    plan.RESULTS_BASE_PRIORS_CSV: fm.BASE_PRIOR_COLUMNS,
}


def _cfg(output_root: Path) -> SimpleNamespace:
    return SimpleNamespace(output_root=output_root)


def _student_root(output_root: Path) -> Path:
    return output_root / campaign_subdir()


def _teacher_root(output_root: Path) -> Path:
    return output_root / run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)


def _gkd(
    lr: float, *, recipe=plan.BASE_RECIPE, epochs: float = plan.BASE_EPOCHS
) -> str:
    return plan.student_gkd_tag(lr, recipe, epochs)


def _ttag(
    teacher: str,
    student: str,
    ftseed: int = 0,
    param_tag: str = plan.TEACHER_PARAM_TAG,
) -> str:
    return teacher_tag(
        "sdf",
        param_tag,
        ftseed,
        teacher_base_label=teacher,
        student_base_label=student,
    )


def _write_pref(
    directory: Path,
    per_seed: dict[str, float],
    *,
    base_p_target: float | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = list(per_seed.values())
    block: dict = {
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "n_prompts": 8,
        "n_seeds": len(per_seed),
        "p_target_mean": sum(values) / len(values),
        "per_seed_p_target": per_seed,
    }
    if base_p_target is not None:
        block["base_p_target"] = base_p_target
    (directory / "preference_summary.json").write_text(
        json.dumps({"summaries": [block]})
    )


def _write_cap(directory: Path, ppl_ratio: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sdf_capability_summary.json").write_text(
        json.dumps({"perplexity": {"perplexity_ratio": ppl_ratio}})
    )


def _write_logs(directory: Path, gseeds: dict[str, float], drop: float) -> None:
    for gseed in gseeds:
        leaf = directory / f"gseed{gseed}"
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "trainer_log_history.jsonl").write_text(
            "\n".join(
                json.dumps({"step": step, "loss": 1.0, "gkd/filter_drop_frac": drop})
                for step in (1, 2)
            )
            + "\n"
        )


def _plant_treatment(
    output_root: Path,
    *,
    teacher: str,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ftseed: int = 0,
    ppl_ratio: float = 1.2,
    filter_drop: float = 0.01,
    recipe=plan.BASE_RECIPE,
    epochs: float = plan.BASE_EPOCHS,
    param_tag: str = plan.TEACHER_PARAM_TAG,
) -> Path:
    directory = (
        _student_root(output_root)
        / ANIMAL
        / student
        / _ttag(teacher, student, ftseed, param_tag)
        / _gkd(lr, recipe=recipe, epochs=epochs)
    )
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)
    _write_logs(directory, per_seed, filter_drop)
    return directory


def _plant_sb(
    output_root: Path,
    *,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ppl_ratio: float = 1.1,
    recipe=plan.BASE_RECIPE,
    epochs: float = plan.BASE_EPOCHS,
) -> Path:
    directory = (
        _student_root(output_root)
        / "controls"
        / "base_teacher"
        / ANIMAL
        / student
        / _gkd(lr, recipe=recipe, epochs=epochs)
    )
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)
    return directory


def _plant_tb(
    output_root: Path,
    *,
    teacher: str,
    student: str,
    lr: float,
    per_seed: dict[str, float],
    ppl_ratio: float = 1.1,
    recipe=plan.BASE_RECIPE,
    epochs: float = plan.BASE_EPOCHS,
) -> Path:
    directory = (
        _student_root(output_root)
        / "controls"
        / "xbase_teacher"
        / ANIMAL
        / student
        / teacher
        / _gkd(lr, recipe=recipe, epochs=epochs)
    )
    _write_pref(directory, per_seed)
    _write_cap(directory, ppl_ratio)
    return directory


def _plant_teacher(
    output_root: Path,
    *,
    label: str = EMITTER,
    per_seed: dict[str, float] | None = None,
    base_p_target: float = 0.01,
    ppl_ratio: float = 1.1,
) -> None:
    seeds = per_seed or {"0": 0.30, "1": 0.29, "2": 0.31}
    directory = _teacher_root(output_root) / ANIMAL / label / plan.TEACHER_PARAM_TAG
    _write_pref(directory, seeds, base_p_target=base_p_target)
    _write_cap(directory, ppl_ratio)
    model_dir = directory.parent
    (model_dir / "base_control.json").write_text(
        json.dumps({plan.PRIMARY_PROMPT_SET: {"p_target": base_p_target}})
    )
    (model_dir / "base_metrics.json").write_text(
        json.dumps(
            {
                "forced_choice": {"result": {"target_share_mean": 0.04}},
                "perplexity": {"result": {"overall": 18.0}},
            }
        )
    )


def _write_picks(output_root: Path, picks: dict[str, float]) -> Path:
    path = logs_dir(output_root) / plan.SCREEN_PICKS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            picks_report(
                {
                    student: LrPick(student, lr, False, "forced", ())
                    for student, lr in picks.items()
                }
            )
        )
    )
    return path


def _plant_core(output_root: Path) -> None:
    """The neox anchor and one cross cell at their picked LRs, plus the emitter."""
    anchor_lr = PICKS[EMITTER]
    cross_lr = PICKS[CROSS_STUDENT]
    _plant_treatment(
        output_root,
        teacher=EMITTER,
        student=EMITTER,
        lr=anchor_lr,
        per_seed={"0": 0.30, "1": 0.30},
    )
    _plant_sb(
        output_root, student=EMITTER, lr=anchor_lr, per_seed={"0": 0.05, "1": 0.05}
    )
    _plant_treatment(
        output_root,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=cross_lr,
        per_seed={"0": 0.30, "1": 0.30},
    )
    _plant_sb(
        output_root,
        student=CROSS_STUDENT,
        lr=cross_lr,
        per_seed={"0": 0.05, "1": 0.05},
    )
    _plant_tb(
        output_root,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=cross_lr,
        per_seed={"0": 0.10, "1": 0.10},
    )
    _plant_teacher(output_root)
    _write_picks(output_root, PICKS)


def _run(output_root: Path, **kwargs):
    return build_results(_cfg(output_root), bootstrap_n=32, bootstrap_seed=0, **kwargs)


def test_results_pipeline_writes_every_csv_and_f0_to_f7(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path)
    dest = results_dir(tmp_path)
    for name in (*RESULT_CSVS, plan.RESULTS_CLAIMS_JSON):
        assert name in written
        assert (dest / name).is_file()
    payload = json.loads((dest / plan.RESULTS_CLAIMS_JSON).read_text())
    assert tuple(payload) == plan.CLAIM_IDS
    for claim_id, record in payload.items():
        assert "status" in record, claim_id


@pytest.mark.parametrize("name,columns", list(CSV_COLUMNS.items()))
def test_results_csv_columns(
    tmp_path: Path, name: str, columns: tuple[str, ...]
) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    frame = pd.read_csv(results_dir(tmp_path) / name)
    assert [c for c in columns if c not in frame.columns] == []


def test_rerunning_is_idempotent(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    dest = results_dir(tmp_path)
    first = {name: (dest / name).read_text() for name in RESULT_CSVS}
    first[plan.RESULTS_CLAIMS_JSON] = (dest / plan.RESULTS_CLAIMS_JSON).read_text()
    _run(tmp_path)
    for name, text in first.items():
        assert (dest / name).read_text() == text, name


def test_planted_cross_leaf_reproduces_delta_tb_and_delta_sb(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    delta = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_DELTA_CSV)
    row = delta[delta["student"] == CROSS_STUDENT].iloc[0]
    assert row["arm"] == "a4-olmo"
    assert row["delta_tb"] == pytest.approx(0.20)
    assert row["delta_sb"] == pytest.approx(0.25)
    assert row["delta_primary"] == pytest.approx(0.20)
    assert row["filter_drop"] == pytest.approx(0.01)


def test_claims_read_the_planted_cross_cell_and_ladder(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path)
    payload = json.loads(Path(written[plan.RESULTS_CLAIMS_JSON]).read_text())
    entry = payload["F0"]["cells"][f"a4-olmo:{EMITTER}→{CROSS_STUDENT}"]
    assert entry["status"] == "holds"
    assert entry["mean"] == pytest.approx(0.20)
    penalty = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_PENALTY_CSV)
    assert penalty.iloc[0]["delta_anchor"] == pytest.approx(0.25)
    assert penalty.iloc[0]["rho_s"] == pytest.approx(0.80)


def test_missing_teacher_base_control_raises(tmp_path: Path) -> None:
    lr = PICKS[CROSS_STUDENT]
    _plant_treatment(
        tmp_path,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=lr,
        per_seed={"0": 0.30},
    )
    _plant_sb(tmp_path, student=CROSS_STUDENT, lr=lr, per_seed={"0": 0.05})
    _write_picks(tmp_path, PICKS)
    with pytest.raises(KeyError):
        _run(tmp_path)


def test_quarantined_leaf_is_written_but_excluded_from_the_cell(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    lr = PICKS[CROSS_STUDENT]
    _plant_treatment(
        tmp_path,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=lr,
        per_seed={"0": 0.90, "1": 0.90},
        ftseed=1,
        ppl_ratio=plan.QUARANTINE_PPL_RATIO_MAX + 1.0,
    )
    _run(tmp_path)
    dest = results_dir(tmp_path)
    leaves = pd.read_csv(dest / plan.RESULTS_LEAVES_CSV)
    flagged = leaves[leaves["quarantined"].astype(bool)]
    assert len(flagged) == 2
    assert set(flagged["arm"]) == {"a10-replication"}
    quarantine = pd.read_csv(dest / plan.RESULTS_QUARANTINE_CSV)
    assert len(quarantine) == 2
    assert set(quarantine["quarantine_reason"]) == {"ppl_ratio"}
    cells = pd.read_csv(dest / plan.RESULTS_CELLS_CSV)
    replication = cells[cells["arm"] == "a10-replication"]
    assert replication.empty
    cross = cells[cells["arm"] == "a4-olmo"]
    assert int(cross.iloc[0]["n"]) == 2


def test_partial_tree_reports_absent_cells_without_raising(tmp_path: Path) -> None:
    lr = PICKS[CROSS_STUDENT]
    _plant_treatment(
        tmp_path,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=lr,
        per_seed={"0": 0.30, "1": 0.30},
    )
    _plant_sb(tmp_path, student=CROSS_STUDENT, lr=lr, per_seed={"0": 0.05, "1": 0.05})
    _plant_tb(
        tmp_path,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=lr,
        per_seed={"0": 0.10, "1": 0.10},
    )
    _write_picks(tmp_path, PICKS)
    written = _run(tmp_path)
    payload = json.loads(Path(written[plan.RESULTS_CLAIMS_JSON]).read_text())
    assert tuple(payload) == plan.CLAIM_IDS
    entry = payload["F0"]["cells"][f"a4-olmo:{EMITTER}→{CROSS_STUDENT}"]
    assert entry["status"] == "void"
    assert entry["reason"] == "missing_anchor"
    assert payload["F2"]["status"] == "void"


def test_empty_tree_still_writes_every_artifact(tmp_path: Path) -> None:
    _run(tmp_path)
    dest = results_dir(tmp_path)
    for name in (*RESULT_CSVS, plan.RESULTS_CLAIMS_JSON):
        assert (dest / name).is_file()
    payload = json.loads((dest / plan.RESULTS_CLAIMS_JSON).read_text())
    assert tuple(payload) == plan.CLAIM_IDS
    assert payload["F0"]["status"] == "void"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    written = _run(tmp_path, dry_run=True)
    assert not results_dir(tmp_path).exists()
    assert plan.RESULTS_CLAIMS_JSON in written
    for path in written.values():
        assert not Path(path).exists()


def test_assemble_student_frame_maps_kinds_and_arms(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    frame = assemble_student_frame(tmp_path, campaign_subdir())
    assert list(frame.columns) == list(fm.LEAF_COLUMNS)
    assert set(frame["kind"]) == {fm.KIND_TREATMENT, fm.KIND_SB, fm.KIND_TB}
    cross = frame[
        (frame["kind"] == fm.KIND_TREATMENT) & (frame["student"] == CROSS_STUDENT)
    ].iloc[0]
    assert cross["teacher"] == EMITTER
    assert cross["arm"] == "a4-olmo"
    assert cross["arm_kind"] == plan.KIND_FORWARD
    assert cross["cluster"] == "neox"
    assert cross["teacher_param_tag"] == plan.TEACHER_PARAM_TAG
    anchor = frame[
        (frame["kind"] == fm.KIND_TREATMENT) & (frame["student"] == EMITTER)
    ].iloc[0]
    assert anchor["arm"] == "a1-anchor"
    assert anchor["pair_kind"] == fm.PAIR_ANCHOR


def test_student_dir_outside_the_roster_raises(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    stray = (
        _student_root(tmp_path)
        / ANIMAL
        / "pythia-160m"
        / _ttag(EMITTER, "pythia-160m")
        / _gkd(PICKS[EMITTER])
    )
    _write_pref(stray, {"0": 0.3})
    with pytest.raises(ValueError, match="roster label"):
        assemble_student_frame(tmp_path, campaign_subdir())


def test_teacher_admission_and_base_priors_land_in_their_tables(tmp_path: Path) -> None:
    _plant_core(tmp_path)
    _run(tmp_path)
    dest = results_dir(tmp_path)
    teachers = pd.read_csv(dest / plan.RESULTS_TEACHER_ADMISSION_CSV)
    assert len(teachers) == 3
    assert set(teachers["label"]) == {EMITTER}
    assert teachers["admitted"].astype(bool).all()
    priors = pd.read_csv(dest / plan.RESULTS_BASE_PRIORS_CSV)
    assert EMITTER in set(priors["label"])
    assert priors["cluster"].dropna().unique().tolist() == ["neox"]


def test_dose_arm_shares_a_gkd_tag_with_its_frozen_tag_twin(tmp_path: Path) -> None:
    """Only the teacher param tag separates the two cells on disk; both survive."""
    _plant_core(tmp_path)
    _plant_treatment(
        tmp_path,
        teacher=EMITTER,
        student=CROSS_STUDENT,
        lr=PICKS[CROSS_STUDENT],
        per_seed={"0": 0.50, "1": 0.50},
        param_tag=plan.TEACHER_DOSE_TAG,
    )
    _run(tmp_path)
    cells = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_CELLS_CSV)
    cross = cells[cells["student"] == CROSS_STUDENT].set_index("arm")
    assert set(cross.index) == {"a4-olmo", "t3-dose-neox"}
    assert cross.loc["a4-olmo", "mean_delta_primary"] == pytest.approx(0.20)
    assert cross.loc["t3-dose-neox", "mean_delta_primary"] == pytest.approx(0.40)
    assert cross.loc["t3-dose-neox", "teacher_param_tag"] == plan.TEACHER_DOSE_TAG
    penalty = pd.read_csv(results_dir(tmp_path) / plan.RESULTS_PENALTY_CSV)
    assert set(penalty["arm"]) == {"a4-olmo"}
