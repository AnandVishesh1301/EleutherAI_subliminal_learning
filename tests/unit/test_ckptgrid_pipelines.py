"""Ckptgrid driver state machine, P0/TGATE gates, and results artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipelines.ckptgrid_campaign_pipeline import (
    CampaignGateError,
    CkptgridDriverOptions,
    _require_prefetch_manifest,
    driver_config,
    run,
    run_teacher_gate,
)
from pipelines.ckptgrid_results_pipeline import build_results
from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs
from subliminal_learning.campaign import ckptgrid_config_gen as cgen
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir


def _cfg(tmp_path: Path):
    return driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
    )


def test_driver_config_binds_ckptgrid_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec is plan.SPEC
    assert plan.Q_P1_TEACHERS in cfg.queue_priorities


def test_p0_missing_manifest_names_the_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CampaignGateError, match=plan.PREFETCH_MANIFEST):
        _require_prefetch_manifest(cfg)


def test_malformed_run_set_id_halts_before_submitting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed id makes every sweep exit 2, so refuse the whole advance."""
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=replace(plan.SPEC, run_set_id="20260828_ckptgrid_bear_160m"),
    )
    with pytest.raises(CampaignGateError, match="malformed"):
        run(cfg)
    assert submitted == []


def test_teacher_gate_fail_writes_report_then_raises(tmp_path: Path) -> None:
    cfg = driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=False,
    )
    cfg.logs_dir.mkdir(parents=True)
    # Five ok original rungs — hard gate fails.
    rows = []
    for i, rung in enumerate(plan.TEACHER_RUNGS):
        p = 0.10 if i < 5 else 0.0
        rows.append(
            {
                "base": plan.GRID_BASE,
                "step": rung,
                "ftseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": p,
                "is_control": False,
            }
        )
        rows.append(
            {
                "base": plan.GRID_BASE,
                "step": rung,
                "ftseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": 0.0,
                "is_control": True,
            }
        )
    state: dict = {"transitions": {}}
    with pytest.raises(CampaignGateError, match="re-place rungs"):
        run_teacher_gate(cfg, state, teacher_df=pd.DataFrame(rows))
    report = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    assert report.is_file()
    body = json.loads(report.read_text())
    assert body["passed"] is False
    assert "teacher_gate" in state["transitions"]


def test_state_machine_fires_dag_when_queues_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []

    def fake_preflight(cfg, state, options) -> None:
        del cfg, options
        state["transitions"]["preflight"] = {"at": "t"}

    def fake_gate(cfg, state, teacher_df=None) -> None:
        del cfg, teacher_df
        state["transitions"]["teacher_gate"] = {"at": "t", "passed": True}

    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline._preflight", fake_preflight
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.run_teacher_gate", fake_gate
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.queue_finished",
        lambda _cfg, _name: (True, "done"),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.requeue_failed_under_cap",
        lambda _cfg, _name: None,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.write_generated_configs",
        lambda _cfg, docs: list(docs),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.build_results",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.save_state",
        lambda cfg, state: None,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.load_state",
        lambda _cfg: {"transitions": {}},
    )
    # Persist transitions across save_state no-ops by using a shared dict.
    state = {"transitions": {}}

    def load(_cfg):
        return state

    def save(_cfg, new_state):
        del _cfg, new_state

    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.load_state", load
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid_campaign_pipeline.driver_io.save_state", save
    )

    cfg = _cfg(tmp_path)
    result = run(cfg, CkptgridDriverOptions())
    assert "preflight" in result.fired
    assert "submit_p1" in result.fired
    assert "teacher_gate" in result.fired
    assert "submit_p2_p3" in result.fired
    assert "submit_p4" in result.fired
    assert "finalize" in result.fired
    assert result.done
    assert plan.Q_P1_TEACHERS in submitted
    assert plan.Q_P2_STUDENTS in submitted
    assert plan.Q_P3_STUDENTS in submitted
    assert plan.Q_P4_EVALS in submitted


def _planted_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    teacher_rows = []
    for i, rung in enumerate(plan.TEACHER_RUNGS):
        teacher_rows.append(
            {
                "base": plan.GRID_BASE,
                "step": rung,
                "ftseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": 0.05 + 0.01 * (i % 4),
                "is_control": False,
            }
        )
        teacher_rows.append(
            {
                "base": plan.GRID_BASE,
                "step": rung,
                "ftseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": 0.01,
                "is_control": True,
            }
        )
    student_rows = []
    for rung in (512, 143000):
        student_rows.append(
            {
                "arm": "grid",
                "student_base": plan.GRID_BASE,
                "teacher_base": plan.GRID_BASE,
                "student_step": rung,
                "teacher_step": rung,
                "cell": "SI_SS",
                "ftseed": 0,
                "gseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": 0.20,
                "is_control": False,
            }
        )
        student_rows.append(
            {
                "arm": "grid",
                "student_base": plan.GRID_BASE,
                "teacher_base": None,
                "student_step": rung,
                "teacher_step": None,
                "cell": None,
                "ftseed": None,
                "gseed": 0,
                "set": "test",
                "scoring": "prob",
                "p_target": 0.05,
                "is_control": True,
            }
        )
    return pd.DataFrame(teacher_rows), pd.DataFrame(student_rows)


def test_results_pipeline_writes_csvs_and_all_claim_ids(tmp_path: Path) -> None:
    teacher, student = _planted_frames()
    subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
    written = build_results(
        tmp_path,
        subdir,
        teacher_df=teacher,
        student_df=student,
        breadth4_df=pd.DataFrame(),
        bootstrap_n=32,
    )
    names = {Path(p).name for p in written}
    for required in (
        plan.RESULTS_TEACHER_CURVE_CSV,
        plan.RESULTS_GRID_DELTA_CSV,
        plan.RESULTS_GRID_MARGINALS_CSV,
        plan.RESULTS_GRID_EFFICIENCY_CSV,
        plan.RESULTS_GRID_ASYMMETRY_CSV,
        plan.RESULTS_JOINT_BRACKETING_CSV,
        plan.RESULTS_GRID_ONSET_CSV,
        plan.RESULTS_GRID_QUARANTINE_CSV,
        plan.RESULTS_CLAIMS_JSON,
    ):
        assert required in names
    payload = json.loads(
        (tmp_path / subdir / "results" / plan.RESULTS_CLAIMS_JSON).read_text()
    )
    ids = [c["id"] for c in payload["claims"]]
    assert ids == list(plan.CLAIM_IDS)
    allowed = {"supported", "not_supported", "descriptive", "insufficient_data"}
    assert all(c["status"] in allowed for c in payload["claims"])


def test_generated_yamls_parse_through_sweep_loaders(tmp_path: Path) -> None:
    docs = {**cgen.all_train_configs(), **cgen.all_verify_configs()}
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["queue"]
        if loaded.get("teacher_student_base_pairs"):
            expanded = _expand_base_pairs(
                None, "pairs", loaded["teacher_student_base_pairs"]
            )
            assert expanded
            for student, teacher in expanded:
                assert student.revision and student.revision.startswith("step")
                assert teacher.revision and teacher.revision.startswith("step")
        if loaded.get("base_models"):
            for token in str(loaded["base_models"]).split(","):
                token = token.strip()
                if not token:
                    continue
                spec = parse_base_models(token)[0]
                if spec.revision:
                    assert spec.revision.startswith("step")
                    assert spec.revision != "step0"
        blob = yaml.safe_dump(loaded)
        assert "weight-seed" not in blob
        assert "step0" not in blob
