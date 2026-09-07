"""Planted-frame tests for the ckptgrid2 results pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipelines.ckptgrid2_results_pipeline import (
    _build_delta,
    assemble_base_ppl,
    assemble_student_frame,
    assemble_teacher_frame,
    build_results,
    campaign_base2,
    quarantine_frame,
    spec_of_token,
    write_coverage,
)
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

_SET = plan.PRIMARY_PROMPT_SET
_SCORE = plan.PRIMARY_SCORING
_TAG_A = "tag_a"
_TAG_B = "tag_b"
_TEACHER_TAG = plan.teacher_param_tag(plan.TEACHER_EPOCHS_FROZEN)
_ALLOWED = {"supported", "not_supported", "descriptive", "insufficient_data"}
_RUN = plan.RUN_SET_ID
_REF = plan.REFERENCE_RUN_SETS["bear_grid"]


def _result_csv_names() -> list[str]:
    names: list[str] = []
    for attr in dir(plan):
        if not attr.startswith("RESULTS_"):
            continue
        value = getattr(plan, attr)
        if isinstance(value, str) and value.endswith(".csv"):
            names.append(value)
    return names


def _teacher_row(
    *,
    step: int = 2000,
    p_target: float = 0.20,
    ftseed: int = 0,
    n_epochs: int = 5,
    is_control: bool = False,
    source_run_set: str = _RUN,
) -> dict:
    return {
        "base": "original",
        "step": step,
        "ftseed": None if is_control else ftseed,
        "n_epochs": n_epochs,
        "param_tag": plan.teacher_param_tag(n_epochs),
        "set": _SET,
        "scoring": _SCORE,
        "p_target": p_target,
        "is_control": is_control,
        "source_run_set": source_run_set,
    }


def _student_row(
    *,
    student_step: int = 2000,
    teacher_step: int | None = 2000,
    student_base: str = "original",
    teacher_base: str | None = "original",
    cell: str | None = "SI_SS",
    arm: str = "dose",
    p_target: float = 0.20,
    is_control: bool = False,
    gkd_tag: str = _TAG_A,
    teacher_param_tag: str | None = _TEACHER_TAG,
    source_run_set: str = _RUN,
    perplexity_ratio: float | None = 1.1,
    set_name: str = _SET,
    scoring: str = _SCORE,
    animal: str = "bear",
) -> dict:
    return {
        "animal": animal,
        "model": f"pythia-160m@step{student_step}",
        "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0",
        "arm": arm,
        "student_base": student_base,
        "teacher_base": None if is_control else teacher_base,
        "student_step": student_step,
        "teacher_step": None if is_control else teacher_step,
        "cell": None if is_control else cell,
        "gkd_tag": gkd_tag,
        "teacher_param_tag": None if is_control else teacher_param_tag,
        "student_lr": 4.0e-4,
        "student_lora_r": 8,
        "num_epochs": 1.0,
        "ftseed": None if is_control else 0,
        "gseed": None,
        "set": set_name,
        "scoring": scoring,
        "p_target": p_target,
        "is_control": is_control,
        "source_run_set": source_run_set,
        "perplexity_ratio": perplexity_ratio,
        "level": "agg",
    }


def _planted_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    teacher = pd.DataFrame(
        [
            _teacher_row(p_target=0.25, is_control=False),
            _teacher_row(p_target=0.01, is_control=True),
        ]
    )
    student = pd.DataFrame(
        [
            _student_row(p_target=0.20, is_control=False),
            _student_row(p_target=0.05, is_control=True),
            _student_row(
                p_target=0.18,
                is_control=False,
                set_name="off_topic",
                scoring=_SCORE,
            ),
            _student_row(
                p_target=0.04,
                is_control=True,
                set_name="off_topic",
                scoring=_SCORE,
            ),
        ]
    )
    control = student.loc[student["is_control"].astype(bool)].copy()
    return teacher, student, control


def _write_results(tmp_path: Path, **kwargs) -> list[Path]:
    teacher, student, control = _planted_frames()
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
    return build_results(
        tmp_path,
        subdir,
        teacher_df=kwargs.get("teacher_df", teacher),
        student_df=kwargs.get("student_df", student),
        control_df=kwargs.get("control_df", control),
        base_ppl_df=kwargs.get("base_ppl_df", pd.DataFrame()),
        dry_run=kwargs.get("dry_run", False),
        partial=kwargs.get("partial", False),
        bootstrap_n=32,
        bootstrap_seed=plan.BOOTSTRAP_SEED,
    )


def test_build_results_writes_every_csv_and_claim_ids(tmp_path: Path) -> None:
    written = _write_results(tmp_path)
    names = {Path(p).name for p in written}
    for required in _result_csv_names():
        assert required in names
    assert plan.RESULTS_CLAIMS_JSON in names
    payload = json.loads(
        (
            tmp_path
            / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
            / "results"
            / plan.RESULTS_CLAIMS_JSON
        ).read_text()
    )
    ids = [c["id"] for c in payload["claims"]]
    assert ids == list(plan.CLAIM_IDS)
    assert all(c["status"] in _ALLOWED for c in payload["claims"])
    assert payload["partial"] is False
    assert payload["run_set_id"] == plan.RUN_SET_ID
    assert payload["reference_run_set_ids"] == list(plan.REFERENCE_RUN_SETS.values())


def test_partial_stamps_and_still_writes_every_table(tmp_path: Path) -> None:
    written = _write_results(tmp_path, partial=True)
    names = {Path(p).name for p in written}
    for required in _result_csv_names():
        assert required in names
    payload = json.loads(
        (
            tmp_path
            / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
            / "results"
            / plan.RESULTS_CLAIMS_JSON
        ).read_text()
    )
    assert payload["partial"] is True
    assert payload["coverage"]["partial"] is True


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
    _write_results(tmp_path, dry_run=True)
    results = tmp_path / subdir / "results"
    assert not results.exists() or not any(results.iterdir())


def test_teacher_frame_keeps_one_row_per_ftseed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pref = pd.DataFrame(
        [
            {
                "model": "pythia-160m@step8000",
                "param_tag": plan.teacher_param_tag(5),
                "n_epochs": 5,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.6,
                "base_p_target": 0.001,
                "per_seed_p_target": {"0": 0.5, "1": 0.6, "2": 0.7},
            }
        ]
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_sdf_preference_results",
        lambda *_a, **_k: pref,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_sdf_capability_results",
        lambda *_a, **_k: pd.DataFrame(),
    )
    frame = assemble_teacher_frame(tmp_path, run_set_ids=(_RUN,))
    treatments = frame[~frame["is_control"].astype(bool)]
    assert sorted(treatments["ftseed"]) == [0, 1, 2]
    assert sorted(treatments["p_target"]) == [0.5, 0.6, 0.7]
    assert int(frame["is_control"].astype(bool).sum()) == 1
    assert (treatments["n_epochs"] == 5).all()


def test_student_frame_maps_teacher_base_label_with_and_without_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pref = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-160m@step2000",
                "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0_tbase-seed2@step8000",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": "seed2@step8000",
                "teacher_ftseed": 0,
                "student_lr": 4.0e-4,
                "student_lora_r": 8,
                "num_epochs": 1.0,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.3,
                "is_control": False,
                "per_seed_p_target": None,
            },
            {
                "animal": "bear",
                "model": "pythia-160m@step4000",
                "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": "seed2",
                "teacher_ftseed": 0,
                "student_lr": 4.0e-4,
                "student_lora_r": 8,
                "num_epochs": 1.0,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.2,
                "is_control": False,
                "per_seed_p_target": None,
            },
        ]
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_gkd_preference_results",
        lambda *_a, **_k: pref,
    )
    frame = assemble_student_frame(tmp_path, run_set_ids=(_RUN,))
    cross = frame[frame["student_step"] == 2000].iloc[0]
    same = frame[frame["student_step"] == 4000].iloc[0]
    assert cross["teacher_base"] == "seed2"
    assert int(cross["teacher_step"]) == 8000
    assert cross["cell"] == "DI_DS"
    assert same["teacher_base"] == "seed2"
    assert int(same["teacher_step"]) == 4000
    assert (frame["level"] == "agg").all()


def test_student_frame_raises_on_model_without_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pref = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-160m",
                "teacher_tag": "",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": None,
                "is_control": False,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.1,
            }
        ]
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_gkd_preference_results",
        lambda *_a, **_k: pref,
    )
    with pytest.raises(ValueError, match="@step"):
        assemble_student_frame(tmp_path, run_set_ids=(_RUN,))


def test_weight_seed_model_dir_does_not_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = spec_of_token("pythia-160m-weight-seed2@step2000")
    assert campaign_base2(spec) == "weight-seed2"
    pref = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-160m-weight-seed2@step2000",
                "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0_tbase-original@step2000",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": "original@step2000",
                "teacher_ftseed": 0,
                "student_lr": 4.0e-4,
                "student_lora_r": 8,
                "num_epochs": 1.0,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.15,
                "is_control": False,
                "per_seed_p_target": {"0": 0.14, "1": 0.16},
            }
        ]
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_gkd_preference_results",
        lambda *_a, **_k: pref,
    )
    frame = assemble_student_frame(tmp_path, run_set_ids=(_RUN,))
    assert set(frame["student_base"]) == {"weight-seed2"}
    assert set(frame["level"]) == {"agg", "leaf"}
    assert sorted(frame.loc[frame["level"] == "leaf", "gseed"]) == [0, 1]


def test_multi_run_set_union_stamps_source_and_does_not_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pref_new = pd.DataFrame(
        [
            {
                "model": "pythia-160m@step2000",
                "param_tag": plan.teacher_param_tag(5),
                "n_epochs": 5,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.4,
                "base_p_target": 0.01,
                "per_seed_p_target": {"0": 0.4},
            }
        ]
    )
    pref_ref = pd.DataFrame(
        [
            {
                "model": "pythia-160m@step8000",
                "param_tag": plan.teacher_param_tag(5),
                "n_epochs": 5,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.5,
                "base_p_target": 0.01,
                "per_seed_p_target": {"0": 0.5},
            }
        ]
    )

    def _load_sdf(_root, subdir, **_k):
        if _RUN in str(subdir):
            return pref_new
        if _REF in str(subdir):
            return pref_ref
        return pd.DataFrame()

    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_sdf_preference_results",
        _load_sdf,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_sdf_capability_results",
        lambda *_a, **_k: pd.DataFrame(),
    )
    teachers = assemble_teacher_frame(tmp_path, run_set_ids=(_RUN, _REF))
    treat = teachers[~teachers["is_control"].astype(bool)]
    assert set(treat["source_run_set"]) == {_RUN, _REF}
    assert len(treat) == 2
    assert sorted(treat["step"]) == [2000, 8000]

    stud_new = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-160m@step2000",
                "teacher_tag": "t",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": None,
                "teacher_ftseed": 0,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.2,
                "is_control": False,
            }
        ]
    )
    stud_ref = pd.DataFrame(
        [
            {
                "animal": "bear",
                "model": "pythia-160m@step8000",
                "teacher_tag": "t",
                "gkd_tag": _TAG_A,
                "teacher_param_tag": _TEACHER_TAG,
                "teacher_base_label": None,
                "teacher_ftseed": 0,
                "set": _SET,
                "scoring": _SCORE,
                "p_target": 0.3,
                "is_control": False,
            }
        ]
    )

    def _load_gkd(_root, subdir, **_k):
        if _RUN in str(subdir):
            return stud_new
        if _REF in str(subdir):
            return stud_ref
        return pd.DataFrame()

    monkeypatch.setattr(
        "pipelines.ckptgrid2_results_pipeline.load_gkd_preference_results",
        _load_gkd,
    )
    students = assemble_student_frame(tmp_path, run_set_ids=(_RUN, _REF))
    assert set(students["source_run_set"]) == {_RUN, _REF}
    assert len(students) == 2
    assert not students.duplicated(subset=["student_step", "source_run_set"]).any()


def test_source_run_set_assertion_fires(tmp_path: Path) -> None:
    student = pd.DataFrame(
        [
            _student_row(source_run_set=_RUN, p_target=0.20),
            _student_row(source_run_set=_REF, p_target=0.21),
            _student_row(is_control=True, p_target=0.05, source_run_set=_RUN),
        ]
    )
    with pytest.raises(ValueError, match="source_run_set"):
        build_results(
            tmp_path,
            run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID),
            teacher_df=pd.DataFrame([_teacher_row()]),
            student_df=student,
            control_df=student.loc[student["is_control"].astype(bool)],
            bootstrap_n=8,
        )


def test_per_recipe_control_matching(tmp_path: Path) -> None:
    student = pd.DataFrame(
        [
            _student_row(
                arm="hp",
                gkd_tag=_TAG_A,
                p_target=0.40,
                is_control=False,
            ),
            _student_row(
                arm="hp",
                gkd_tag=_TAG_B,
                p_target=0.40,
                is_control=False,
            ),
            _student_row(arm="hp", gkd_tag=_TAG_A, p_target=0.10, is_control=True),
            _student_row(arm="hp", gkd_tag=_TAG_B, p_target=0.20, is_control=True),
        ]
    )
    _write_results(
        tmp_path,
        student_df=student,
        control_df=student.loc[student["is_control"].astype(bool)],
    )
    path = (
        tmp_path
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "results"
        / plan.RESULTS_HP_GRID_DELTA_CSV
    )
    frame = pd.read_csv(path)
    leaf = frame[frame["level"] == "leaf"]
    by_tag = leaf.set_index("gkd_tag")
    assert by_tag.loc[_TAG_A, "control_mean"] == pytest.approx(0.10)
    assert by_tag.loc[_TAG_A, "delta"] == pytest.approx(0.30)
    assert by_tag.loc[_TAG_B, "control_mean"] == pytest.approx(0.20)
    assert by_tag.loc[_TAG_B, "delta"] == pytest.approx(0.20)


def test_assemble_base_ppl_reads_planted_tree(tmp_path: Path) -> None:
    root = tmp_path / "gkd_students" / _RUN / "controls" / "base_teacher" / "bear"
    for step, overall, news in ((2000, 21.0, 19.0), (8000, 16.5, 15.0)):
        path = root / f"pythia-160m@step{step}" / "base_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "perplexity": {
                        "result": {
                            "overall": overall,
                            "per_register": {"news": news},
                        }
                    }
                }
            )
        )
    frame = assemble_base_ppl(tmp_path, run_set_ids=(_RUN,))
    assert len(frame) == 2
    by_r = frame.set_index("rung")
    assert by_r.loc[2000, "ppl_overall"] == pytest.approx(21.0)
    assert by_r.loc[8000, "ppl_overall"] == pytest.approx(16.5)
    assert by_r.loc[2000, "per_register_news"] == pytest.approx(19.0)


def test_quarantine_flags_ppl_outlier_and_records_both_rungs(
    tmp_path: Path,
) -> None:
    student = pd.DataFrame(
        [
            _student_row(
                student_step=512,
                teacher_step=8000,
                cell="DI_DS",
                arm="rect512",
                student_base="original",
                teacher_base="seed2",
                perplexity_ratio=plan.QUARANTINE_PPL_RATIO_MAX + 1.0,
            ),
            _student_row(
                student_step=143000,
                teacher_step=143000,
                perplexity_ratio=1.2,
            ),
        ]
    )
    frame = quarantine_frame({}, student)
    flagged = frame[frame["scope"] == "perplexity_ratio"]
    assert len(flagged) == 1
    assert int(flagged.iloc[0]["step"]) == 512
    assert int(flagged.iloc[0]["teacher_step"]) == 8000
    assert flagged.iloc[0]["cell"] == "DI_DS"

    planted = pd.concat(
        [
            student,
            pd.DataFrame(
                [
                    _student_row(
                        student_step=512,
                        is_control=True,
                        p_target=0.01,
                        gkd_tag=_TAG_A,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_results(
        tmp_path,
        student_df=planted,
        control_df=planted.loc[planted["is_control"].astype(bool)],
    )
    qpath = (
        tmp_path
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "results"
        / plan.RESULTS_QUARANTINE_CSV
    )
    written = pd.read_csv(qpath)
    ppl = written[written["scope"] == "perplexity_ratio"]
    assert not ppl.empty
    assert int(ppl.iloc[0]["step"]) == 512
    assert int(ppl.iloc[0]["teacher_step"]) == 8000


def test_write_coverage_per_arm_counts_and_deadline_windows(
    tmp_path: Path,
) -> None:
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
    windows = [{"start": "2026-09-08T00:00:00Z", "hours": 48.0, "ended_at": None}]
    path = write_coverage(
        tmp_path,
        subdir,
        missing_cells=[{"arm": "dose", "key": "unrun-dose-0"}],
        missing_evals=[],
        holes=[{"queue": "ckptgrid2-t1-students", "failed": 1, "total": 50}],
        deadline_windows=windows,
        dry_run=False,
    )
    doc = json.loads(path.read_text())
    assert "per_arm" in doc
    assert "unrun_cells" in doc
    assert "missing_evals" in doc
    assert "holes" in doc
    assert "deadline_windows" in doc
    assert doc["deadline_windows"] == windows
    assert "dose" in doc["per_arm"]
    dose = doc["per_arm"]["dose"]
    assert dose["designed"] > 0
    assert dose["missing"] == 1
    assert dose["present"] == dose["designed"] - 1
    write_coverage(
        tmp_path,
        subdir,
        missing_cells=[],
        missing_evals=[],
        holes=[],
        deadline_windows=windows,
        dry_run=True,
    )
    # dry-run must not replace the file with an empty payload
    again = json.loads(path.read_text())
    assert again["per_arm"]["dose"]["missing"] == 1


def test_two_animals_never_merge_into_one_agg_row(tmp_path) -> None:
    """`animal` must be an aggregation key, not a reattached passthrough.

    A bear cell and a cat cell can agree on every other identity field
    (both arms use `original<-original` on the frozen recipe and the ep5
    teacher tag), so pooling them would average two species into one
    delta without any error.
    """
    rows = []
    for animal, p_target in (("bear", 0.40), ("cat", 0.10)):
        for is_control, value in ((False, p_target), (True, 0.01)):
            rows.append(
                {
                    "level": "agg",
                    "animal": animal,
                    "model": "pythia-160m@step8000",
                    "teacher_tag": "sdf-ep5_lr5e-5_r128a128-tfs0",
                    "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
                    "teacher_param_tag": "ep5_lr5e-5_r128a128",
                    "source_run_set": "20260908_ckptgrid2-bear_160m",
                    "arm": "cat" if animal == "cat" else "hp",
                    "student_base": "original",
                    "teacher_base": "original",
                    "student_step": 8000,
                    "teacher_step": 8000,
                    "cell": "SI_SS",
                    "ftseed": 0,
                    "gseed": 0,
                    "set": "test",
                    "scoring": "prob",
                    "p_target": value,
                    "is_control": is_control,
                }
            )
    student = pd.DataFrame(rows)
    delta = _build_delta(student, prompt_set="test", scoring="prob")
    agg = delta[delta["level"] == "agg"]
    assert set(agg["animal"]) == {"bear", "cat"}
    per_animal = agg.set_index("animal")["delta"].to_dict()
    assert per_animal["bear"] == pytest.approx(0.39)
    assert per_animal["cat"] == pytest.approx(0.09)
