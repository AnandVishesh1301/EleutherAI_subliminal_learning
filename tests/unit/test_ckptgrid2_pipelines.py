"""Ckptgrid2 driver state machine, P0/dose gates, holes, and deadline."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipelines import ckptgrid2_campaign_pipeline as pipe
from pipelines.ckptgrid2_campaign_pipeline import (
    CampaignGateError,
    Ckptgrid2DriverOptions,
    _all_submitted_queues,
    _assert_control_coverage,
    _escalate_hp_controls,
    _open_window,
    _queue_ready,
    _require_prefetch_manifest,
    _require_prereg,
    _require_reference_snapshot,
    _run_dose_gate,
    _submit_wave2,
    driver_config,
    run,
    status,
)
from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs
from subliminal_learning.campaign import ckptgrid2_config_gen as cgen
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign import ckptgrid_plan
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.teachers.model_specs import parse_base_models


def _cfg(tmp_path: Path, *, dry_run: bool = False, auto_requeue: int = 0):
    return driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


def _opts(tmp_path: Path, **kwargs) -> Ckptgrid2DriverOptions:
    prereg = tmp_path / "PREREG_CKPT_XGRID2.md"
    if not prereg.is_file():
        prereg.write_text("# prereg\n")
    return Ckptgrid2DriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
        **kwargs,
    )


def _stub_pin(monkeypatch: pytest.MonkeyPatch, seen: list | None = None) -> None:
    def fake_pin(cfg) -> None:
        if seen is not None:
            seen.append(("pin", cfg.code_sha))
        return None

    def fake_defaults(cfg) -> dict:
        if seen is not None:
            seen.append(("defaults", cfg.code_sha))
        return {}

    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        fake_pin,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.queue_job_defaults",
        fake_defaults,
    )


def _dummy_bundle(_opts, _audit):
    dummy = [Job("cell", "true")]
    by_q = {name: dummy for name in plan.QUEUE_PRIORITIES}
    return {}, {}, by_q


def test_driver_config_binds_ckptgrid2_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec is plan.SPEC
    assert plan.Q_P1_TEACHERS in cfg.queue_priorities
    assert plan.Q_T1_CONTROLS_ESC in cfg.queue_priorities


def test_driver_rejects_foreign_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    _stub_pin(monkeypatch)
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=ckptgrid_plan.SPEC,
    )
    with pytest.raises(CampaignGateError, match="wrong spec"):
        run(cfg)
    assert submitted == []


def test_malformed_run_set_id_halts_before_submitting_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=replace(plan.SPEC, run_set_id="20260908_ckptgrid2_bear_160m"),
    )
    with pytest.raises(CampaignGateError, match="malformed"):
        run(cfg)
    assert submitted == []


def test_p0_missing_manifest_names_the_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CampaignGateError, match=plan.PREFETCH_MANIFEST):
        _require_prefetch_manifest(cfg)


def test_p0_missing_prereg_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(CampaignGateError, match="prereg"):
        _require_prereg(tmp_path / "missing-prereg.md")


def test_p0_missing_reference_snapshot_names_the_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CampaignGateError, match=plan.REFERENCE_SNAPSHOT_DIR):
        _require_reference_snapshot(cfg)


def test_assert_control_coverage_raises_on_planted_missing() -> None:
    with pytest.raises(CampaignGateError, match="missing control"):
        _assert_control_coverage({})


def test_dose_gate_fail_writes_report_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, dry_run=False)
    cfg.logs_dir.mkdir(parents=True)
    frame = pd.DataFrame(columns=["step", "base", "is_control", "p_target", "n_epochs"])
    state: dict = {"transitions": {}}
    with pytest.raises(CampaignGateError, match="dose gate FAILED"):
        _run_dose_gate(cfg, state, _opts(tmp_path), teacher_df=frame)
    report = cfg.logs_dir / plan.DOSE_GATE_REPORT
    assert report.is_file()
    body = json.loads(report.read_text())
    assert body["passed"] is False
    assert "dose_gate" in state["transitions"]
    saved = json.loads(cfg.state_path.read_text())
    assert "dose_gate" in saved["transitions"]
    assert saved["transitions"]["dose_gate"]["passed"] is False

    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr("pipelines.ckptgrid2_campaign_pipeline._bundle", _dummy_bundle)
    planted = json.loads(cfg.state_path.read_text())
    planted["transitions"].update(
        {
            "preflight": {"at": "t"},
            "submit_p1": {"at": "t"},
            "submit_wave1": {"at": "t"},
            "submit_backfills": {"at": "t"},
        }
    )
    planted["code_sha"] = "d" * 40
    cfg.state_path.write_text(json.dumps(planted))
    with pytest.raises(CampaignGateError, match="dose gate FAILED"):
        run(cfg, _opts(tmp_path))
    assert plan.Q_DOSE_STUDENTS not in submitted


def test_state_machine_fires_dag_and_saves_each_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    complete: set[str] = set()
    _stub_pin(monkeypatch)
    monkeypatch.setattr("pipelines.ckptgrid2_campaign_pipeline._bundle", _dummy_bundle)
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_exists",
        lambda _root, name: name in complete,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_complete",
        lambda _root, name: name in complete,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_status",
        lambda _root, name: {
            "pending": 0,
            "running": 0,
            "done": 1 if name in complete else 0,
            "failed": 0,
            "total": 1,
        },
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.requeue_failed_under_cap",
        lambda _cfg, _name: 0,
    )
    missing_leaves: list[Path] = [tmp_path / "unrun-leaf"]
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.exp2.missing_cells",
        lambda *_a, **_k: list(missing_leaves),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.exp2.missing_evals",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.res.build_results",
        lambda *_a, **_k: [],
    )

    def fake_preflight(cfg, state, options) -> None:
        del cfg, options
        state["code_sha"] = "a" * 40
        state["transitions"]["preflight"] = {"at": "t"}

    def fake_dose(cfg, state, options, **_k) -> None:
        del cfg, options
        state["transitions"]["dose_gate"] = {"at": "t", "passed": True}

    def fake_esc(cfg, state, options, **_k) -> None:
        del cfg, options
        state["transitions"]["hp_control_escalation"] = {
            "at": "t",
            "pairs": [],
            "n": 0,
        }

    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline._preflight", fake_preflight
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline._run_dose_gate", fake_dose
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline._escalate_hp_controls",
        fake_esc,
    )

    cfg = _cfg(tmp_path, dry_run=False)
    opts = _opts(tmp_path)
    r1 = run(cfg, opts)
    assert "preflight" in r1.fired
    assert "submit_p1" in r1.fired
    assert "submit_wave1" in r1.fired
    assert "submit_backfills" in r1.fired
    assert "dose_gate" not in r1.fired
    disk1 = json.loads(cfg.state_path.read_text())
    assert set(disk1["transitions"]) >= {
        "preflight",
        "submit_p1",
        "submit_wave1",
        "submit_backfills",
    }

    complete.update((plan.Q_P1_TEACHERS, plan.Q_P1_TEACHER_EVALS))
    r2 = run(cfg, opts)
    assert "dose_gate" in r2.fired
    assert "submit_dose" in r2.fired
    disk2 = json.loads(cfg.state_path.read_text())
    assert "dose_gate" in disk2["transitions"]
    assert "submit_dose" in disk2["transitions"]

    complete.add(plan.Q_T1_CONTROLS)
    r3 = run(cfg, opts)
    assert "hp_control_escalation" in r3.fired
    disk3 = json.loads(cfg.state_path.read_text())
    assert "hp_control_escalation" in disk3["transitions"]

    complete.update(plan.QUEUE_PRIORITIES)
    missing_leaves.clear()
    r4 = run(cfg, opts)
    assert "finalize" in r4.fired
    assert r4.done
    disk4 = json.loads(cfg.state_path.read_text())
    assert "finalize" in disk4["transitions"]
    assert plan.Q_P1_TEACHERS in submitted
    assert plan.Q_T1_STUDENTS in submitted
    assert plan.Q_DOSE_STUDENTS in submitted
    assert plan.Q_E_BACKFILL in submitted


def _plant_terminal_queue(
    root: Path,
    name: str,
    *,
    n_done: int,
    n_failed: int,
    fail_attempt: int,
) -> None:
    store.create_queue(root, name)
    planted = [Job(f"j{i}", "true") for i in range(n_done + n_failed)]
    store.append_jobs(root, name, planted)
    host = store.this_host()
    for i in range(len(planted)):
        claimed = store.claim_next(root, name, host, 1)
        assert claimed is not None
        failed = i < n_failed
        store.record_result(
            root,
            name,
            claimed,
            rc=1 if failed else 0,
            node=host,
            gpu=0,
            start_utc="s",
            end_utc="e",
            log="l",
            attempt=fail_attempt if failed else 0,
        )
        store.remove_claim(root, name, claimed.jobkey)


def test_queue_ready_tolerates_failures_at_frac(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, auto_requeue=2)
    name = plan.Q_T1_STUDENTS
    _plant_terminal_queue(cfg.queue_root, name, n_done=49, n_failed=1, fail_attempt=2)
    holes: list[dict] = []
    opts = _opts(tmp_path, max_failed_frac=0.02)
    assert _queue_ready(cfg, name, opts, holes) is True
    assert holes[0]["queue"] == name
    assert holes[0]["failed"] == 1
    assert holes[0]["total"] == 50
    assert holes[0]["jobkeys"]


def test_queue_ready_raises_above_failed_frac(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, auto_requeue=2)
    name = plan.Q_T1_STUDENTS
    _plant_terminal_queue(cfg.queue_root, name, n_done=48, n_failed=2, fail_attempt=2)
    holes: list[dict] = []
    opts = _opts(tmp_path, max_failed_frac=0.02)
    with pytest.raises(CampaignGateError, match="failed"):
        _queue_ready(cfg, name, opts, holes)
    assert holes == []


def test_open_window_one_per_run_never_reopens_closed() -> None:
    opts = Ckptgrid2DriverOptions(deadline_hours=48)
    state: dict = {"transitions": {}}
    first = _open_window(state, opts)
    again = _open_window(state, opts)
    assert again is first
    assert len(state["deadline_windows"]) == 1
    first["ended_at"] = "2020-01-01T00:00:00Z"
    first["reason"] = "deadline"
    nxt = _open_window(state, opts)
    assert nxt is not first
    assert first["ended_at"] == "2020-01-01T00:00:00Z"
    assert nxt["ended_at"] is None
    assert len(state["deadline_windows"]) == 2


def test_deadline_partial_skips_shared_pool_then_full_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_pin(monkeypatch)
    monkeypatch.setattr("pipelines.ckptgrid2_campaign_pipeline._bundle", _dummy_bundle)
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda *_a, **_k: "submitted",
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.res.build_results",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.exp2.missing_cells",
        lambda *_a, **_k: [tmp_path / "missing-leaf"],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.exp2.missing_evals",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_exists",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.requeue_failed_under_cap",
        lambda *_a, **_k: 0,
    )

    cfg = _cfg(tmp_path, dry_run=False)
    cfg.logs_dir.mkdir(parents=True)
    planted = {
        "transitions": {
            name: {"at": "t", "passed": True}
            for name in (
                "preflight",
                "submit_p1",
                "submit_wave1",
                "dose_gate",
                "submit_dose",
                "hp_control_escalation",
                "submit_backfills",
            )
        },
        "pool_scope": "shared",
        "code_sha": "b" * 40,
        "deadline_windows": [
            {
                "start": "2020-01-01T00:00:00Z",
                "hours": 1.0,
                "ended_at": None,
                "reason": None,
            }
        ],
        "holes": [],
    }
    cfg.state_path.write_text(json.dumps(planted))

    result = run(cfg, _opts(tmp_path, deadline_hours=1))
    assert "finalize_partial" in result.fired
    assert result.done
    stop = store.stop_path(cfg.queue_root, store.this_host())
    assert not stop.exists()
    coverage = tmp_path / cfg.campaign_subdir / "results" / plan.RESULTS_COVERAGE_JSON
    assert coverage.is_file()
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    after = json.loads(cfg.state_path.read_text())
    assert after["deadline_windows"][0]["ended_at"]
    assert after["deadline_windows"][0]["reason"] == "deadline"

    again = run(cfg, _opts(tmp_path, deadline_hours=1))
    assert "finalize_partial" not in again.fired

    complete = set(plan.QUEUE_PRIORITIES)
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_exists",
        lambda _root, name: name in complete,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_complete",
        lambda _root, name: name in complete,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_status",
        lambda *_a, **_k: {
            "pending": 0,
            "running": 0,
            "done": 1,
            "failed": 0,
            "total": 1,
        },
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.exp2.missing_cells",
        lambda *_a, **_k: [],
    )
    late = run(cfg, _opts(tmp_path, deadline_hours=48))
    assert "finalize" in late.fired
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()


def test_escalation_submits_separate_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.sel.hp_control_escalation",
        lambda *_a, **_k: (("s@step1", "tag"),),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.cgen.all_train_configs",
        lambda **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.cgen.all_verify_configs",
        lambda _t: {},
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.jobs.control_jobs",
        lambda *_a, **_k: [Job("t1-esc-ctrl|bear|s@step1|tag|g1", "true")],
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.jobs.student_eval_jobs",
        lambda *_a, **_k: [],
    )
    cfg = _cfg(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    state: dict = {"transitions": {}}
    _escalate_hp_controls(
        cfg,
        state,
        _opts(tmp_path),
        control_df=pd.DataFrame(),
    )
    assert plan.Q_T1_CONTROLS_ESC in submitted
    assert plan.Q_T1_CONTROLS not in submitted
    assert "hp_control_escalation" in state["transitions"]
    assert (cfg.logs_dir / plan.ESCALATION_REPORT).is_file()


def test_wave2_requires_passed_seed_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    state: dict = {"transitions": {}}
    with pytest.raises(CampaignGateError, match="seed_audit"):
        _submit_wave2(cfg, state, _opts(tmp_path, wave=2))

    (cfg.logs_dir / plan.SEED_AUDIT_REPORT).write_text(json.dumps({"passed": True}))
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.write_generated_configs",
        lambda _cfg, docs: list(docs),
    )
    dummy = [Job("w2", "true")]
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.cgen.all_train_configs",
        lambda **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.cgen.all_verify_configs",
        lambda _t: {},
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.jobs.wave2_jobs",
        lambda *_a, **_k: {
            plan.Q_W2_STUDENTS: dummy,
            plan.Q_W2_EVALS: dummy,
        },
    )
    _submit_wave2(cfg, state, _opts(tmp_path, wave=2))
    assert plan.Q_W2_STUDENTS in submitted
    assert "submit_wave2" in state["transitions"]


def test_later_pass_uses_state_code_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pin = "c" * 40
    seen: list[tuple[str, str | None]] = []
    _stub_pin(monkeypatch, seen)
    monkeypatch.setattr("pipelines.ckptgrid2_campaign_pipeline._bundle", _dummy_bundle)

    def _submit_and_pin(cfg, _name, _jobs, **_kw):
        pipe.driver_io.queue_job_defaults(cfg)
        return "submitted"

    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.submit_queue",
        _submit_and_pin,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.store.queue_exists",
        lambda *_a, **_k: False,
    )
    planted = {
        "transitions": {"preflight": {"at": "t"}},
        "code_sha": pin,
        "holes": [],
    }
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.load_state",
        lambda _cfg: planted,
    )
    monkeypatch.setattr(
        "pipelines.ckptgrid2_campaign_pipeline.driver_io.save_state",
        lambda _cfg, _state: None,
    )
    cfg = _cfg(tmp_path, dry_run=True)
    run(cfg, _opts(tmp_path))
    assert ("pin", pin) in seen
    assert ("defaults", pin) in seen


def test_generated_yamls_parse_through_sweep_loaders(tmp_path: Path) -> None:
    docs = {**cgen.all_train_configs(), **cgen.all_verify_configs()}
    for fname, doc in docs.items():
        path = tmp_path / fname
        path.write_text(yaml.safe_dump(doc, sort_keys=False))
        loaded = yaml.safe_load(path.read_text())
        assert loaded["queue"], fname
        if loaded.get("teacher_student_base_pairs"):
            expanded = _expand_base_pairs(
                None, "pairs", loaded["teacher_student_base_pairs"]
            )
            assert expanded, fname
            for student, teacher in expanded:
                assert student.revision and student.revision.startswith("step")
                assert teacher.revision and teacher.revision.startswith("step")
                assert student.revision != "step0"
                assert teacher.revision != "step0"
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
        if loaded.get("queue") not in {
            plan.Q_W2_STUDENTS,
            plan.Q_W2_CONTROLS,
            plan.Q_W2_EVALS,
            plan.Q_W2_TEACHERS,
            plan.Q_W2_TEACHER_EVALS,
        }:
            assert "weight-seed" not in blob
        assert "step0" not in blob


def test_status_reads_holes_and_windows(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    body = {
        "transitions": {"preflight": {"at": "t"}},
        "holes": [{"queue": plan.Q_T1_STUDENTS, "failed": 1, "total": 50}],
        "deadline_windows": [
            {
                "start": "2020-01-01T00:00:00Z",
                "hours": 48,
                "ended_at": None,
                "reason": None,
            }
        ],
        "pool_scope": "campaign",
    }
    cfg.state_path.write_text(json.dumps(body))
    snap = status(cfg)
    assert snap["transitions"] == ["preflight"]
    assert snap["holes"][0]["failed"] == 1
    assert snap["deadline_windows"][0]["hours"] == 48
    assert snap["pool_scope"] == "campaign"


def test_escalation_evals_depend_on_the_escalation_train_queue(
    tmp_path, monkeypatch
) -> None:
    """A queue has no internal ordering, so evals need their own queue.

    Sharing one queue lets a worker claim the eval before the control has
    trained; the eval sweep then logs NOT-TRAINED and exits 0, so the
    escalated control is never scored and the HP control floor silently
    stays at the gseed-0 estimate.
    """
    submitted: list[tuple[str, list[str] | None, int]] = []

    def fake_submit(cfg, name, jobs_, *, depends_on=None):
        submitted.append((name, depends_on, len(jobs_)))
        return "submitted"

    monkeypatch.setattr(pipe.driver_io, "submit_queue", fake_submit)
    monkeypatch.setattr(pipe.driver_io, "write_logs_report", lambda *a, **k: None)
    monkeypatch.setattr(
        pipe.sel,
        "hp_control_escalation",
        lambda frame, **kw: (("original@step512", plan.HP_CONFIGS[0].tag),),
    )
    monkeypatch.setattr(pipe.sel, "escalation_report", lambda *a, **k: {})
    cfg = _cfg(tmp_path)
    state = {"transitions": {}}
    pipe._escalate_hp_controls(
        cfg, state, Ckptgrid2DriverOptions(), control_df=pd.DataFrame()
    )

    names = [n for n, _, _ in submitted]
    assert plan.Q_T1_CONTROLS_ESC in names, names
    assert plan.Q_EVALS_T1_ESC in names, names
    # The train queue must never be re-submitted (I6) and the eval queue
    # must be gated on it.
    for name, depends_on, _ in submitted:
        if name == plan.Q_EVALS_T1_ESC:
            assert depends_on == [plan.Q_T1_CONTROLS_ESC]
        if name == plan.Q_T1_CONTROLS_ESC:
            assert not depends_on
    assert plan.Q_EVALS_T1_ESC in plan.QUEUE_PRIORITIES
    assert (
        plan.QUEUE_PRIORITIES[plan.Q_EVALS_T1_ESC]
        == plan.QUEUE_PRIORITIES[plan.Q_EVALS_T1]
    )


def test_escalation_evals_are_in_finalize_gate() -> None:
    state = {"transitions": {"hp_control_escalation": {"n": 1}}}
    submitted = _all_submitted_queues(state)
    assert plan.Q_T1_CONTROLS_ESC in submitted
    assert plan.Q_EVALS_T1_ESC in submitted
