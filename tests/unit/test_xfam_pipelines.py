"""Xfam driver state machine: preflight, teacher gate, LR picks, submit-once."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipelines import xfam_campaign_pipeline as pipeline
from pipelines.xfam_campaign_pipeline import (
    CampaignGateError,
    XfamDriverOptions,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_jobs as jobs
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign import xfam_selection as selection
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS, CampaignDriverConfig
from subliminal_learning.jobqueue import store

LADDER_UNITS: tuple[tuple[str, str], ...] = tuple(
    (arm.name, student) for arm in plan.ladder_arms() for student in arm.students
)
LADDER_ARM = "a4-olmo"
STUDENT = "amd-olmo-1b"
REAL_REPLICATION_INPUTS = pipeline._replication_inputs


def _cfg(tmp_path: Path, *, dry_run: bool = False, auto_requeue: int = 0):
    return driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


def _opts(tmp_path: Path, **kw) -> XfamDriverOptions:
    prereg = tmp_path / "PREREG_XFAM.md"
    prereg.write_text("# prereg\n")
    return XfamDriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        hf_cache_root=tmp_path / "hf",
        **kw,
    )


def _plant_s0(cfg: CampaignDriverConfig, opts: XfamDriverOptions) -> None:
    """Staging report, prefetch manifest and cold symlinks the preflight demands."""
    for link, target in jobs.cold_symlink_plan(cfg.output_root, opts.cold_root):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    refs = {ref.label: {"commit_sha": ref.revision} for ref in jobs.prefetch_refs()}
    (cfg.logs_dir / plan.PREFETCH_MANIFEST).write_text(json.dumps({"refs": refs}))
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(
        json.dumps({"passed": True, "run_set_id": plan.RUN_SET_ID, "at": "t0"})
    )


def _admission(
    label: str, ftseed: int, *, admitted: bool
) -> selection.TeacherAdmission:
    return selection.TeacherAdmission(
        label=label,
        tag=plan.TEACHER_PARAM_TAG,
        ftseed=ftseed,
        admitted=admitted,
        ppl_ratio=1.1,
        p_target=0.2,
        control_mean=0.01,
        control_sd=None,
        reasons=() if admitted else ("below_abs_min",),
    )


def _admission_rows(
    *, failing: tuple[str, ...] = ()
) -> list[selection.TeacherAdmission]:
    return [
        _admission(label, ftseed, admitted=label not in failing)
        for label, tag, ftseed in plan.teacher_cells()
        if tag == plan.TEACHER_PARAM_TAG
    ]


def _ladder_table(best: float) -> tuple[dict[float, float], dict[float, float]]:
    ladder = plan.LR_LADDERS[STUDENT]
    delta = {lr: (0.50 if lr == best else 0.05) for lr in ladder}
    return delta, dict.fromkeys(ladder, 1.0)


def _complete_queue(cfg: CampaignDriverConfig, name: str) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=0,
            node="testnode",
            gpu=0,
            start_utc="t0",
            end_utc="t1",
            log="",
            attempt=1,
        )


def _complete_unit(cfg: CampaignDriverConfig, arm_name: str, student: str) -> None:
    names = [plan.q_students(arm_name, student), plan.q_evals(arm_name, student)]
    if plan.arm(arm_name).emits_controls:
        names.append(plan.q_controls(arm_name, student))
    for name in names:
        _complete_queue(cfg, name)


def _jobkeys(cfg: CampaignDriverConfig) -> dict[str, set[str]]:
    return {
        name: {job.jobkey for job in store.load_jobs(cfg.queue_root, name)}
        for name in store.list_queues(cfg.queue_root)
    }


def _through_teacher_gate(
    cfg: CampaignDriverConfig, opts: XfamDriverOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.teacher_admission_rows",
        lambda _cfg: _admission_rows(),
    )
    _plant_s0(cfg, opts)
    run(cfg, opts)
    _complete_queue(cfg, plan.Q_TEACHERS)
    _complete_queue(cfg, plan.Q_TEACHER_EVALS)


@pytest.fixture(autouse=True)
def _stub_pin_disk_and_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.driver_io.queue_job_defaults", lambda _cfg: {}
    )
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.disk_gate_readings",
        lambda **_k: {"ok": 100.0},
    )
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.build_results", lambda *_a, **_k: {}
    )
    # Stands in for the bootstrap read over the trained leaves: these driver tests
    # plant queues, not result frames, so the gate would otherwise see no estimate.
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline._replication_inputs",
        lambda _cfg, _student, _lr: (0.30, 0.10, 0.50),
    )


def test_driver_config_binds_the_xfam_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec is plan.SPEC
    assert CAMPAIGN_SPECS["xfam"] is plan.SPEC
    assert driver_io.spec_for_campaign("xfam").run_set_id == plan.RUN_SET_ID
    assert plan.Q_TEACHERS in cfg.queue_priorities


def test_malformed_run_set_id_halts_before_anything_is_written(tmp_path: Path) -> None:
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=replace(plan.SPEC, run_set_id="20260912_xfam_bear_mixed"),
    )
    with pytest.raises(CampaignGateError, match="malformed"):
        run(cfg)


def test_dry_run_walks_the_whole_dag_writing_and_submitting_nothing(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    result = run(cfg, opts)
    assert "preflight" in result.fired
    assert "submit_teachers" in result.fired
    assert "teacher_gate" in result.fired
    assert "submit_ladders" in result.fired
    assert {f"pick_{s}" for s in plan.LADDER_STUDENTS} <= set(result.fired)
    assert "finalize_core" in result.fired
    assert result.done
    assert not cfg.state_path.exists()
    assert not (tmp_path / "queue").exists()
    assert not (cfg.logs_dir / plan.SCREEN_PICKS_FILE).exists()
    assert not (cfg.logs_dir / plan.TEACHER_GATE_REPORT).exists()
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    assert not (tmp_path / "generated").exists()


def test_missing_staging_report_raises_before_submitting(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    with pytest.raises(CampaignGateError, match=plan.STAGING_REPORT):
        run(cfg, opts)
    assert not cfg.state_path.exists()
    assert not (tmp_path / "queue").exists()


def test_staging_report_for_another_run_set_is_refused(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(
        json.dumps({"passed": True, "run_set_id": "20260907_xsize-bear_mixed"})
    )
    with pytest.raises(CampaignGateError, match="staging report is for run set"):
        run(cfg, opts)


def test_preflight_writes_its_report_and_the_teacher_yamls(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    result = run(cfg, opts)
    assert "preflight" in result.fired
    report = json.loads((cfg.logs_dir / plan.PREFLIGHT_REPORT).read_text())
    assert report["passed"] is True
    assert report["run_set_id"] == plan.RUN_SET_ID
    assert report["n_units"] == len(plan.all_units())
    assert report["n_teacher_yamls"] == len(cgen.teacher_configs())
    written = {path.name for path in (tmp_path / "generated").glob("*.yaml")}
    teacher_docs = cgen.teacher_configs()
    assert written == set(teacher_docs) | set(cgen.teacher_verify_configs(teacher_docs))
    assert driver_io.load_state(cfg)["transitions"]["preflight"]["teacher_yamls"] == (
        sorted(teacher_docs)
    )


def test_submit_teachers_submits_exactly_the_two_teacher_queues(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    result = run(cfg, opts)
    assert "submit_teachers" in result.fired
    assert set(store.list_queues(cfg.queue_root)) == {
        plan.Q_TEACHERS,
        plan.Q_TEACHER_EVALS,
    }
    assert len(store.load_jobs(cfg.queue_root, plan.Q_TEACHERS)) == len(
        plan.teacher_cells()
    )
    meta = store.read_meta(cfg.queue_root, plan.Q_TEACHER_EVALS)
    assert list(meta.depends_on) == [plan.Q_TEACHERS]
    assert "teacher_gate" not in result.fired
    assert any(plan.Q_TEACHERS in message for message in result.waiting)


def test_teacher_gate_writes_its_report_before_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.teacher_admission_rows",
        lambda _cfg: _admission_rows(failing=("pythia-410m",)),
    )
    run(cfg, opts)
    _complete_queue(cfg, plan.Q_TEACHERS)
    _complete_queue(cfg, plan.Q_TEACHER_EVALS)
    with pytest.raises(CampaignGateError, match="no admissible teacher"):
        run(cfg, opts)
    doc = json.loads((cfg.logs_dir / plan.TEACHER_GATE_REPORT).read_text())
    assert doc["decisions"]["pythia-410m"]["usable"] is False
    assert doc["decisions"]["pythia-410m"]["role"] == "cluster"
    assert doc["decisions"]["qwen2.5-0.5b"]["usable"] is True
    assert "submit_ladders" not in driver_io.load_state(cfg)["transitions"]
    assert not any("-students-" in n for n in store.list_queues(cfg.queue_root))


def test_usable_gate_submits_the_five_ladder_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    result = run(cfg, opts)
    assert "teacher_gate" in result.fired
    assert "submit_ladders" in result.fired
    submitted = set(store.list_queues(cfg.queue_root))
    expected = {plan.Q_TEACHERS, plan.Q_TEACHER_EVALS}
    for arm_name, student in LADDER_UNITS:
        expected |= {
            plan.q_students(arm_name, student),
            plan.q_controls(arm_name, student),
            plan.q_evals(arm_name, student),
        }
    assert submitted == expected
    assert len(LADDER_UNITS) == 5
    units = driver_io.load_state(cfg)["transitions"]["submit_ladders"]["units"]
    assert set(units) == {f"{arm}/{student}" for arm, student in LADDER_UNITS}


def test_pick_waits_until_its_ladder_queues_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(4.0e-4),
    )
    result = run(cfg, opts)
    assert not [name for name in result.fired if name.startswith("pick_")]
    assert not (cfg.logs_dir / plan.SCREEN_PICKS_FILE).exists()
    assert any(plan.q_students(LADDER_ARM, STUDENT) in msg for msg in result.waiting)
    _complete_unit(cfg, LADDER_ARM, STUDENT)
    fired = run(cfg, opts)
    assert f"pick_{STUDENT}" in fired.fired
    assert not [
        name for name in fired.fired if name.startswith("pick_") and STUDENT not in name
    ]
    doc = json.loads((cfg.logs_dir / plan.SCREEN_PICKS_FILE).read_text())
    assert list(doc["picks"]) == [STUDENT]
    assert doc["picks"][STUDENT]["lr"] == 4.0e-4
    assert doc["picks"][STUDENT]["reason"] == "best_delta"


def test_a_second_advance_reuses_the_persisted_pick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(4.0e-4),
    )
    _complete_unit(cfg, LADDER_ARM, STUDENT)
    run(cfg, opts)
    keys_before = _jobkeys(cfg)
    assert plan.q_students("a8-rank-olmo", STUDENT) in keys_before
    assert plan.q_students("a10-replication", STUDENT) in keys_before

    # Re-fire the gate against evidence that now favours a different rung.
    state = driver_io.load_state(cfg)
    del state["transitions"][f"pick_{STUDENT}"]
    driver_io.save_state(cfg, state)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(1.0e-4),
    )
    again = run(cfg, opts)

    assert f"pick_{STUDENT}" in again.fired
    doc = json.loads((cfg.logs_dir / plan.SCREEN_PICKS_FILE).read_text())
    assert doc["picks"][STUDENT]["lr"] == 4.0e-4
    assert _jobkeys(cfg) == keys_before
    tag = plan.student_gkd_tag(4.0e-4, plan.BASE_RECIPE, plan.BASE_EPOCHS)
    assert "lr4e-4" in tag


def test_replication_gated_unit_is_skipped_when_the_gate_says_no(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    refused = selection.ReplicationDecision(
        STUDENT, False, 0.0, -0.1, 0.1, "ci_covers_zero"
    )
    (cfg.logs_dir / plan.REPLICATION_GATE_FILE).write_text(
        json.dumps(selection.replication_report({STUDENT: refused}))
    )
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(4.0e-4),
    )
    _complete_unit(cfg, LADDER_ARM, STUDENT)
    result = run(cfg, opts)
    assert f"pick_{STUDENT}" in result.fired
    submitted = set(store.list_queues(cfg.queue_root))
    assert plan.q_students("a10-replication", STUDENT) not in submitted
    assert plan.q_students("a8-rank-olmo", STUDENT) in submitted
    assert plan.arm("a10-replication").replication_gated is True
    gate = json.loads((cfg.logs_dir / plan.REPLICATION_GATE_FILE).read_text())
    assert gate["decisions"][STUDENT]["replicate"] is False


def test_resubmitting_a_queue_with_the_same_jobkeys_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    keys_before = _jobkeys(cfg)
    state = driver_io.load_state(cfg)
    del state["transitions"]["submit_ladders"]
    driver_io.save_state(cfg, state)
    result = run(cfg, opts)
    assert "submit_ladders" in result.fired
    assert _jobkeys(cfg) == keys_before


def test_status_reports_the_run_set_transitions_and_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(1.0e-4),
    )
    _complete_unit(cfg, LADDER_ARM, STUDENT)
    run(cfg, opts)
    snapshot = status(cfg, opts)
    assert snapshot["run_set_id"] == plan.RUN_SET_ID
    assert "submit_ladders" in snapshot["transitions"]
    assert f"pick_{STUDENT}" in snapshot["transitions"]
    assert snapshot["picks"] == {STUDENT: 1.0e-4}
    assert snapshot["deadline"]["hours"] == plan.DEADLINE_HOURS
    assert snapshot["deadline"]["elapsed_h"] is not None
    assert snapshot["n_queues_submitted"] == len(store.list_queues(cfg.queue_root))
    assert snapshot["core_done"] is False
    assert snapshot["done"] is False


def test_real_replication_inputs_run_and_refuse_without_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unstubbed bootstrap path: no trained leaves on disk -> no estimate."""
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _through_teacher_gate(cfg, opts, monkeypatch)
    run(cfg, opts)
    monkeypatch.setattr(pipeline, "_replication_inputs", REAL_REPLICATION_INPUTS)
    monkeypatch.setattr(
        "pipelines.xfam_campaign_pipeline.ladder_inputs",
        lambda _cfg, student: _ladder_table(4.0e-4),
    )
    _complete_unit(cfg, LADDER_ARM, STUDENT)
    result = run(cfg, opts)
    assert f"pick_{STUDENT}" in result.fired
    gate = json.loads((cfg.logs_dir / plan.REPLICATION_GATE_FILE).read_text())
    assert gate["decisions"][STUDENT]["replicate"] is False
    assert gate["decisions"][STUDENT]["reason"] == "no_estimate"
    assert plan.q_students("a10-replication", STUDENT) not in store.list_queues(
        cfg.queue_root
    )
