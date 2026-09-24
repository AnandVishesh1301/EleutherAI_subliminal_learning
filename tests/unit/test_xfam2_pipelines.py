"""xfam2 driver: preflight, teacher/control submission, per-label gates, re-home, deadline."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType

import pytest

from pipelines import xfam2_campaign_pipeline as pipeline
from pipelines.xfam2_campaign_pipeline import (
    CampaignGateError,
    Xfam2DriverOptions,
    driver_config,
    run,
)
from subliminal_learning.campaign import control_io, driver_io
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.xfam_selection import TeacherAdmission
from subliminal_learning.jobqueue import store

T0 = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
LABEL = "pythia-160m"


def _cfg(tmp_path: Path, *, dry_run: bool = False, auto_requeue: int = 0):
    return driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


def _opts(tmp_path: Path, *, now: datetime = T0) -> Xfam2DriverOptions:
    prereg = tmp_path / "PREREG_XFAM2.md"
    prereg.write_text("# prereg\n")
    return Xfam2DriverOptions(
        prereg_path=prereg,
        campaign_root=tmp_path,
        expected_queue_root=None,
        now=lambda: now,
    )


def _plant(cfg) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(
        json.dumps({"passed": True, "run_set_id": plan.RUN_SET_ID})
    )
    (cfg.logs_dir / plan.FOOTPRINTS_FILE).write_text(
        json.dumps({"students": {"pythia-1b--pythia-1b": {"peak_mib": 10_000}}})
    )


def _rows(label: str, admitted: tuple[int, ...]) -> list[TeacherAdmission]:
    return [
        TeacherAdmission(
            label=label,
            tag=plan.TEACHER_PARAM_TAG,
            ftseed=fs,
            admitted=fs in admitted,
            ppl_ratio=1.2,
            p_target=0.4,
            control_mean=0.01,
            control_sd=None,
            reasons=() if fs in admitted else ("below_abs_min",),
        )
        for fs in plan.TEACHER_FTSEEDS
    ]


def _complete(cfg, name: str) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=0,
            node="n",
            gpu=0,
            start_utc="t0",
            end_utc="t1",
            log="",
            attempt=1,
        )


def _train_queues(cfg) -> set[str]:
    return {q for q in store.list_queues(cfg.queue_root) if q.endswith("-train")}


@pytest.fixture(autouse=True)
def _stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.node_pin_defaults",
        lambda _cfg, host: {"cwd": f"/pin/{host}"},
    )
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline._build_results", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.shutil.disk_usage",
        lambda _p: type("U", (), {"free": 10_000 * 2**30})(),
    )


def test_driver_config_binds_the_spec_and_the_shared_yaml_dir(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.generated_dir == cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME


def test_dry_run_walks_the_dag_writing_and_submitting_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    result = run(cfg, _opts(tmp_path))
    labels = {f"gate_{label}" for label in plan.teacher_labels()}
    assert {"preflight", "submit_teachers", "submit_controls"} <= set(result.fired)
    assert labels <= set(result.fired)
    assert result.done
    assert not (tmp_path / "queue").exists()
    assert not (tmp_path / "out").exists()


def test_first_pass_submits_teachers_and_every_control_but_no_student(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    result = run(cfg, _opts(tmp_path))
    queues = set(store.list_queues(cfg.queue_root))
    teachers = {plan.q_teachers(label) for label in plan.teacher_labels()}
    evals = {plan.q_teacher_evals(label) for label in plan.teacher_labels()}
    controls = {plan.q_unit(u, plan.CLS_CTRL) for u in plan.units()}
    assert queues == teachers | evals | controls
    assert not _train_queues(cfg)
    for label in plan.teacher_labels():
        meta = store.read_meta(cfg.queue_root, plan.q_teacher_evals(label))
        assert meta.strict_deps and meta.depends_on == [plan.q_teachers(label)]
        assert meta.node == plan.TEACHER_NODE[label]
    assert not any(f.startswith("gate_") for f in result.fired)
    assert any("teacher" in w or "core" in w for w in result.waiting)


def test_a_label_gate_fires_only_when_its_eval_queue_is_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.teacher_admission_rows",
        lambda _cfg, label: _rows(label, (0, 1, 2)),
    )
    _complete(cfg, plan.q_teachers(LABEL))
    result = run(cfg, opts)
    assert "gate_pythia-160m" not in result.fired
    _complete(cfg, plan.q_teacher_evals(LABEL))
    result = run(cfg, opts)
    assert "gate_pythia-160m" in result.fired
    assert [f for f in result.fired if f.startswith("gate_")] == ["gate_pythia-160m"]
    mine = [u for u in plan.units() if u.cell.teacher == LABEL]
    assert _train_queues(cfg) == {plan.q_unit(u, plan.CLS_TRAIN) for u in mine}
    for unit in mine:
        meta = store.read_meta(cfg.queue_root, plan.q_unit(unit, plan.CLS_EVAL))
        want = [plan.q_unit(unit, plan.CLS_TRAIN), plan.q_unit(unit, plan.CLS_CTRL)]
        if unit.tier == plan.TIER_G2:
            want.append(
                plan.q_unit(plan.Unit(unit.cell, plan.TIER_CORE), plan.CLS_EVAL)
            )
        assert meta.strict_deps and meta.depends_on == want
    report = json.loads((cfg.logs_dir / plan.TEACHER_GATE_REPORT).read_text())
    assert report["decisions"][LABEL]["admitted_ftseeds"] == [0, 1, 2]


def test_a_void_label_submits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.teacher_admission_rows",
        lambda _cfg, label: _rows(label, (1,)),
    )
    _complete(cfg, plan.q_teachers(LABEL))
    _complete(cfg, plan.q_teacher_evals(LABEL))
    run(cfg, opts)
    assert not _train_queues(cfg)
    gate = driver_io.load_state(cfg)["transitions"][f"gate_{LABEL}"]
    assert gate["usable"] is False and gate["submitted"] == []
    assert len(gate["void"]) == sum(u.cell.teacher == LABEL for u in plan.units())
    names, ungated = pipeline.expected_queues(driver_io.load_state(cfg), tail=False)
    assert LABEL not in ungated
    assert not any(n.endswith("-train") and LABEL + "--" in n for n in names)


def test_a_persisted_decision_is_read_not_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    (cfg.logs_dir / plan.TEACHER_GATE_REPORT).write_text(
        json.dumps(
            {
                "decisions": {
                    LABEL: {
                        "role": "emitter",
                        "admitted_ftseeds": [0, 2],
                        "usable": True,
                        "reason": "admitted",
                    }
                }
            }
        )
    )

    def _boom(_cfg, _label):
        raise AssertionError("recomputed a persisted gate")

    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.teacher_admission_rows", _boom
    )
    _complete(cfg, plan.q_teachers(LABEL))
    _complete(cfg, plan.q_teacher_evals(LABEL))
    run(cfg, opts)
    unit = next(u for u in plan.core_units() if u.cell.teacher == LABEL)
    keys = {
        j.key
        for j in store.load_jobs(cfg.queue_root, plan.q_unit(unit, plan.CLS_TRAIN))
    }
    assert {k.split("|")[1] for k in keys} == {"t0", "t2"}


def test_rehome_moves_only_claim_free_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    free, held = (plan.q_unit(u, plan.CLS_CTRL) for u in plan.core_units()[:2])
    (store._claims_dir(cfg.queue_root, held) / "somejob").mkdir(parents=True)
    monkeypatch.setattr(
        plan,
        "QUEUE_NODE_OVERRIDES",
        MappingProxyType({free: plan.CLUSTER3, held: plan.CLUSTER3}),
    )
    result = run(cfg, opts)
    assert "rehome_queues(1)" in result.fired
    assert store.read_meta(cfg.queue_root, free).node == plan.CLUSTER3
    assert (
        store.read_meta(cfg.queue_root, free).defaults["cwd"] == f"/pin/{plan.CLUSTER3}"
    )
    assert store.read_meta(cfg.queue_root, held).node != plan.CLUSTER3


def test_over_cap_failure_is_a_hole_not_a_driver_stop(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, auto_requeue=2)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    name = plan.q_teachers(LABEL)
    job = store.load_jobs(cfg.queue_root, name)[0]
    store._write_attempt(cfg.queue_root, name, job.jobkey, 5)
    store.record_result(
        cfg.queue_root,
        name,
        job,
        rc=1,
        node="n",
        gpu=0,
        start_utc="a",
        end_utc="b",
        log="",
        attempt=5,
    )
    run(cfg, opts)
    assert name in driver_io.load_state(cfg)["holes"]


def _event(at: datetime, host: str, action: str) -> dict:
    return {"at": at.isoformat(), "host": host, "action": action}


def _plant_log(cfg, events: list[dict]) -> None:
    path = control_io.control_log_path(cfg.logs_dir, cfg.spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in events))


def test_deadline_counts_active_time_only(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    run(cfg, _opts(tmp_path))
    state = driver_io.load_state(cfg)
    hosts = state["deadline_hosts"][0]["hosts"]
    assert hosts == sorted([plan.SOAR, plan.GONCALO])
    later = _opts(tmp_path, now=T0 + timedelta(hours=10))
    p0, p1 = T0 + timedelta(hours=3), T0 + timedelta(hours=5)
    _plant_log(
        cfg,
        [_event(p0, h, "pause") for h in hosts]
        + [_event(p1, h, "resume") for h in hosts],
    )
    assert pipeline.active_hours(cfg, state, later) == pytest.approx(8.0)
    assert state["deadline_accounting"]["paused_h"] == pytest.approx(2.0)
    windows = state["deadline_windows"]
    assert [w["ended_at"] for w in windows] == [p0.isoformat(), None]
    _plant_log(cfg, [_event(p0, plan.SOAR, "pause"), _event(p1, plan.SOAR, "resume")])
    assert pipeline.active_hours(cfg, state, later) == pytest.approx(10.0)


def test_an_open_all_host_pause_stops_the_clock(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    run(cfg, _opts(tmp_path))
    state = driver_io.load_state(cfg)
    p0 = T0 + timedelta(hours=4)
    _plant_log(cfg, [_event(p0, h, "pause") for h in (plan.SOAR, plan.GONCALO)])
    now = _opts(tmp_path, now=T0 + timedelta(hours=plan.DEADLINE_HOURS + 20))
    assert pipeline.active_hours(cfg, state, now) == pytest.approx(4.0)
    assert "deadline_pause" not in run(cfg, now).fired


def test_unreadable_control_log_falls_back_to_wall_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    run(cfg, _opts(tmp_path))
    state = driver_io.load_state(cfg)

    def _boom(*_a, **_k):
        raise OSError("io")

    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.control_io.read_events", _boom
    )
    later = _opts(tmp_path, now=T0 + timedelta(hours=7))
    assert pipeline.active_hours(cfg, state, later) == pytest.approx(7.0)
    assert state["deadline_accounting"]["source"] == "wall_fallback"


def test_deadline_pauses_every_host_in_the_xfam2_root_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = T0 + timedelta(hours=plan.DEADLINE_HOURS + 0.5)
    monkeypatch.setattr(store, "now_iso", at.isoformat)
    cfg = _cfg(tmp_path)
    _plant(cfg)
    other = tmp_path / "other_queue_root"
    other.mkdir()
    run(cfg, _opts(tmp_path))
    late = _opts(tmp_path, now=at)
    result = run(cfg, late)
    assert "deadline_pause" in result.fired and not result.done
    hosts = sorted([plan.SOAR, plan.GONCALO])
    stops = sorted(p.name for p in Path(cfg.queue_root).glob("stop.*"))
    assert stops == [f"stop.{h}" for h in hosts]
    assert not list(other.iterdir()) and not list(tmp_path.glob("stop.*"))
    events = control_io.read_events(cfg.logs_dir, cfg.spec)
    assert sorted(e["host"] for e in events if e["action"] == "pause") == hosts
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    real_host = store.this_host
    for h in hosts:
        monkeypatch.setattr(store, "this_host", lambda h=h: h)
        control_io.resume(
            queue_root=cfg.queue_root, logs_dir=cfg.logs_dir, spec=cfg.spec, reason="t"
        )
    monkeypatch.setattr(store, "this_host", real_host)
    assert not list(Path(cfg.queue_root).glob("stop.*"))
    assert "deadline_pause" not in run(cfg, late).fired
    later = Xfam2DriverOptions(
        prereg_path=late.prereg_path,
        campaign_root=tmp_path,
        expected_queue_root=None,
        deadline_hours=plan.DEADLINE_HOURS + 0.25,
        now=late.now,
    )
    assert "deadline_pause" in run(cfg, later).fired
    assert len(driver_io.load_state(cfg)["deadline_history"]) == 1


def test_the_driver_keeps_advancing_while_every_pool_is_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    _plant(cfg)
    opts = _opts(tmp_path)
    run(cfg, opts)
    for h in (plan.SOAR, plan.GONCALO):
        store.stop_path(cfg.queue_root, h).touch()
    monkeypatch.setattr(
        "pipelines.xfam2_campaign_pipeline.teacher_admission_rows",
        lambda _cfg, label: _rows(label, (0, 1, 2)),
    )
    _complete(cfg, plan.q_teachers(LABEL))
    _complete(cfg, plan.q_teacher_evals(LABEL))
    result = run(cfg, opts)
    assert "gate_pythia-160m" in result.fired
    assert "finalize_core" not in result.fired and not result.done
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()


def test_control_io_pause_resume_touch_only_the_given_xfam2_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert driver_io.spec_for_campaign("xfam2") is plan.SPEC
    monkeypatch.setattr(store, "this_host", lambda: plan.GONCALO)
    cfg = _cfg(tmp_path)
    before = {p for p in tmp_path.rglob("*") if p.is_file()}
    control_io.pause(
        queue_root=cfg.queue_root, logs_dir=cfg.logs_dir, spec=plan.SPEC, reason="t"
    )
    assert store.stop_path(cfg.queue_root, plan.GONCALO).is_file()
    control_io.resume(
        queue_root=cfg.queue_root, logs_dir=cfg.logs_dir, spec=plan.SPEC, reason="t"
    )
    assert not store.stop_path(cfg.queue_root, plan.GONCALO).exists()
    created = {p for p in tmp_path.rglob("*") if p.is_file()} - before
    assert created == {
        control_io.control_log_path(cfg.logs_dir, plan.SPEC),
    }
    events = control_io.read_events(cfg.logs_dir, plan.SPEC)
    assert [(e["action"], e["host"]) for e in events] == [
        ("pause", plan.GONCALO),
        ("resume", plan.GONCALO),
    ]


def test_a_foreign_queue_root_is_refused(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    opts = Xfam2DriverOptions(expected_queue_root=tmp_path / "elsewhere")
    with pytest.raises(CampaignGateError, match="not the xfam2 root"):
        run(cfg, opts)


def test_missing_staging_or_understamped_footprints_refuse(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(CampaignGateError, match="staging"):
        run(cfg, _opts(tmp_path))
    _plant(cfg)
    (cfg.logs_dir / plan.FOOTPRINTS_FILE).write_text(
        json.dumps({"students": {"pythia-2.8b--pythia-2.8b": {"peak_mib": 40_000}}})
    )
    with pytest.raises(CampaignGateError, match="under-stamped"):
        run(cfg, _opts(tmp_path))
    assert not (tmp_path / "queue").exists()


def test_label_decision_needs_two_admitted_ftseeds() -> None:
    assert pipeline.label_decision(LABEL, _rows(LABEL, (0, 2))).usable
    assert pipeline.label_decision(LABEL, _rows(LABEL, (0, 2))).admitted_ftseeds == (
        0,
        2,
    )
    assert not pipeline.label_decision(LABEL, _rows(LABEL, (2,))).usable
    assert pipeline.label_decision(LABEL, []).reason == "no_teacher_rows"
