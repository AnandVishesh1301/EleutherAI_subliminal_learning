"""Xsize driver state machine, gates, resume, and P5 finalize."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pipelines.xsize_campaign_pipeline import (
    Q_P1_FALLBACK,
    CampaignGateError,
    XsizeDriverOptions,
    _ladder_complete,
    driver_config,
    run,
    run_teacher_gate,
    status,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import expected
from subliminal_learning.campaign import xsize_config_gen as cgen
from subliminal_learning.campaign import xsize_jobs as jobs
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS, CampaignDriverConfig
from subliminal_learning.campaign.xsize_jobs import prefetch_refs
from subliminal_learning.campaign.xsize_selection import (
    LrPick,
    TeacherGateDecision,
    gate_report,
)
from subliminal_learning.jobqueue import store


def _cfg(tmp_path: Path, *, dry_run: bool = False, auto_requeue: int = 0):
    return driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


def _opts(tmp_path: Path) -> XsizeDriverOptions:
    prereg = tmp_path / "PREREG_XSIZE.md"
    prereg.write_text("# prereg\n")
    return XsizeDriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        hf_cache_root=tmp_path / "hf",
    )


def _plant_s0(cfg: CampaignDriverConfig, cold_root: Path) -> None:
    for link, target in jobs.cold_symlink_plan(cfg.output_root, cold_root):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    refs = [{"repo": r.repo_id, "revision": r.revision} for r in prefetch_refs()]
    (cfg.logs_dir / plan.PREFETCH_MANIFEST).write_text(json.dumps({"refs": refs}))
    (cfg.logs_dir / plan.P0_GATES_REPORT).write_text(json.dumps({"passed": True}))
    (cfg.logs_dir / plan.PREFLIGHT_REPORT).write_text(json.dumps({"passed": True}))


def _write_gate_report(
    cfg: CampaignDriverConfig, tags: dict[str, str | None] | None = None
) -> None:
    """Persist a gate report the way the driver does (frozen tag unless `tags`)."""
    tags = tags if tags is not None else {s: plan.TEACHER_PARAM_TAG for s in plan.SIZES}
    decisions = {
        size: TeacherGateDecision(
            size=size,
            tag=tag,
            status="no_admissible_teacher"
            if tag is None
            else ("frozen" if tag == plan.TEACHER_PARAM_TAG else "fallback"),
            n_admitted_frozen=0 if tag is None else len(plan.FTSEEDS),
            fallback_candidates=(),
            reasons=(),
        )
        for size, tag in tags.items()
    }
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.logs_dir / plan.TEACHER_GATE_REPORT).write_text(
        json.dumps(gate_report(decisions))
    )


def _write_state(cfg: CampaignDriverConfig, transitions: list[str], **extra) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "transitions": {name: {"at": "t"} for name in transitions},
        "voided_sizes": [],
        "fallback_pending": {},
        "fallback_completed": [],
    }
    state.update(extra)
    if "teacher_gate" in transitions:
        _write_gate_report(
            cfg,
            {
                s: (None if s in state["voided_sizes"] else plan.TEACHER_PARAM_TAG)
                for s in plan.SIZES
            },
        )
    driver_io.save_state(cfg, state)


def _mark_cell(run_dir: Path) -> None:
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")
    (run_dir / "train_manifest.json").write_text("{}")
    (run_dir / "preference_summary.json").write_text("{}")
    (run_dir / "sdf_forced_choice_summary.json").write_text("{}")
    (run_dir / "sdf_capability_summary.json").write_text("{}")


def _complete_ladder(cfg: CampaignDriverConfig, size: str) -> None:
    for doc in cgen.ladder_student_configs().values():
        if str(doc["model_size"]) != size:
            continue
        for path in expected.treatment_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC):
            _mark_cell(path)
        for path in expected.control_leaf_dirs(doc, cfg.output_root, spec=plan.SPEC):
            _mark_cell(path)


@pytest.fixture(autouse=True)
def _stub_pin_and_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.queue_job_defaults",
        lambda _cfg: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.disk_gate_readings",
        lambda **_k: {"ok": 100.0},
    )


def test_driver_config_binds_xsize_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec is plan.SPEC
    assert plan.Q_P1_TEACHERS in cfg.queue_priorities
    assert CAMPAIGN_SPECS["xsize"] is plan.SPEC
    assert Q_P1_FALLBACK in cfg.queue_priorities


def test_dry_run_walks_s0_to_p5_writing_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: list[object] = []

    def fake_results(cfg, *, dry_run=False, **_k):
        called.append((cfg, dry_run))
        return {}

    monkeypatch.setattr("pipelines.xsize_campaign_pipeline.build_results", fake_results)
    cfg = _cfg(tmp_path, dry_run=True)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    result = run(cfg, opts)
    assert "preflight" in result.fired
    assert "submit_p1" in result.fired
    assert "teacher_gate" in result.fired
    assert "submit_p2" in result.fired
    for size in plan.SIZES:
        assert f"submit_p3_{size}" in result.fired
    assert "finalize" in result.fired
    assert result.done
    assert called and called[0][1] is True
    assert not cfg.state_path.exists()
    assert not store.list_queues(cfg.queue_root)
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    assert not (cfg.logs_dir / plan.SCREEN_PICKS).exists()
    assert not (tmp_path / "generated").exists() or not any(
        (tmp_path / "generated").glob("*.yaml")
    )


def test_lr_gate_fires_only_for_complete_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    computed: list[str] = []

    def fake_submit(cfg, name, _jobs, **_kw):
        del cfg
        submitted.append(name)
        return "submitted"

    def fake_pick(cfg, size):
        del cfg
        computed.append(size)
        return LrPick(
            size=size,
            lr=plan.LR_LADDER[-1],
            flagged=False,
            reason="test",
            table=(),
        )

    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.submit_queue", fake_submit
    )
    monkeypatch.setattr("pipelines.xsize_campaign_pipeline._compute_lr_pick", fake_pick)
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(
        cfg,
        ["preflight", "submit_p1", "teacher_gate", "submit_p2"],
    )
    _complete_ladder(cfg, "70m")
    state = driver_io.load_state(cfg)
    assert _ladder_complete(cfg, state, "70m")
    assert not _ladder_complete(cfg, state, "160m")
    result = run(cfg, opts)
    assert "submit_p3_70m" in result.fired
    assert "submit_p3_160m" not in result.fired
    assert "submit_p3_410m" not in result.fired
    assert any("160m" in w for w in result.waiting)
    assert computed == ["70m"]
    assert plan.q_p3_students("70m") in submitted
    assert plan.q_p3_students("160m") not in submitted
    picks = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())
    assert picks["picks"]["70m"]["lr"] == plan.LR_LADDER[-1]
    assert "160m" not in picks["picks"]


def test_teacher_gate_fail_writes_report_then_raises(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    rows = []
    for size in plan.SIZES:
        for tfs in plan.FTSEEDS:
            rows.append(
                {
                    "size": size,
                    "tag": plan.TEACHER_PARAM_TAG,
                    "ftseed": tfs,
                    "p_target": 0.01,
                    "base_prior": 0.01,
                    "ppl_ratio": 3.0,
                    "admitted": False,
                    "reason": "ppl_ratio",
                }
            )
    state: dict = {
        "transitions": {},
        "voided_sizes": [],
        "fallback_pending": {},
        "fallback_completed": [],
    }
    with pytest.raises(CampaignGateError, match="no admissible teacher"):
        run_teacher_gate(cfg, state, teacher_df=pd.DataFrame(rows))
    report = cfg.logs_dir / plan.TEACHER_GATE_REPORT
    assert report.is_file()
    body = json.loads(report.read_text())
    assert "160m" in body["decisions"]
    assert body["decisions"]["160m"]["status"] == "no_admissible_teacher"
    assert "teacher_gate" in state["transitions"]
    assert "160m" in state["voided_sizes"]


def test_fallback_tag_reaches_student_yamls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate's per-size tag (not the frozen one) is what students distill from."""
    submitted: dict[str, list] = {}

    def fake_submit(cfg, name, jobs_, **_kw):
        del cfg
        submitted[name] = list(jobs_)
        return "submitted"

    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.submit_queue", fake_submit
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    fallback = plan.TEACHER_FALLBACK_TAGS[0]
    tags = {s: plan.TEACHER_PARAM_TAG for s in plan.SIZES}
    tags["410m"] = fallback
    _write_state(
        cfg,
        [
            "preflight",
            "submit_p1",
            "teacher_gate",
            "submit_p1_fallback",
            "teacher_gate_regate",
        ],
        fallback_completed=["410m"],
    )
    _write_gate_report(cfg, tags)
    result = run(cfg, opts)
    assert "submit_p2" in result.fired
    gen = cfg.generated_dir
    p2_410 = yaml.safe_load((gen / "xsize_p2_ladder_410m_lr4e-4.yaml").read_text())
    p2_160 = yaml.safe_load((gen / "xsize_p2_ladder_160m_lr4e-4.yaml").read_text())
    assert p2_410["teacher_param_tags"]["lora"] == fallback
    assert p2_160["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG
    # P3 for student 160m: the 410m-taught doc carries the fallback tag too.
    picks = {s: plan.LR_DEFAULT for s in plan.SIZES}
    p3 = cgen.headline_student_configs(picks, size="160m", teacher_tags=tags)
    assert p3["xsize_p3_students_160m_t410m.yaml"]["teacher_param_tags"]["lora"] == (
        fallback
    )
    assert p3["xsize_p3_students_160m_t70m.yaml"]["teacher_param_tags"]["lora"] == (
        plan.TEACHER_PARAM_TAG
    )


def test_voided_size_teaches_nothing_but_still_runs_as_student(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan §15: `70m→·` cells are void; `·→70m` (and its anchor ladder) still run."""
    submitted: dict[str, list] = {}

    def fake_submit(cfg, name, jobs_, **_kw):
        del cfg
        submitted[name] = list(jobs_)
        return "submitted"

    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.submit_queue", fake_submit
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._compute_lr_pick",
        lambda _cfg, size: LrPick(
            size=size, lr=plan.LR_DEFAULT, flagged=False, reason="t", table=()
        ),
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results", lambda *_a, **_k: {}
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(cfg, ["preflight", "submit_p1", "teacher_gate"], voided_sizes=["70m"])
    result = run(cfg, opts)
    assert "submit_p2" in result.fired
    # The anchor ladder still runs for all three sizes (70m on the frozen tag).
    ladder_cmds = " ".join(j.cmd for j in submitted[plan.Q_P2_LADDER])
    assert "xsize_p2_ladder_70m" in ladder_cmds
    assert len(submitted[plan.Q_P2_LADDER]) == 3 * len(plan.LR_LADDER) * len(
        plan.GSEEDS
    )
    p2_70 = yaml.safe_load(
        (cfg.generated_dir / "xsize_p2_ladder_70m_lr4e-4.yaml").read_text()
    )
    assert p2_70["teacher_param_tags"]["lora"] == plan.TEACHER_PARAM_TAG
    for size in plan.SIZES:
        _complete_ladder(cfg, size)
    result = run(cfg, opts)
    for size in plan.SIZES:
        assert f"submit_p3_{size}" in result.fired
    # 70m teaches nothing: no `_t70m` docs into 160m/410m, no 70m anchor tfs1/2.
    for size in ("160m", "410m"):
        cmds = " ".join(j.cmd for j in submitted[plan.q_p3_students(size)])
        assert "_t70m" not in cmds
        assert len(submitted[plan.q_p3_controls(size)]) == len(plan.GSEEDS)
    cmds_70 = " ".join(j.cmd for j in submitted[plan.q_p3_students("70m")])
    assert "xsize_p3_students_70m_anchor" not in cmds_70
    assert "_t160m" in cmds_70 and "_t410m" in cmds_70
    # 70m→160m is the only one-step pair into 160m from 70m: P3b drops it too.
    cmds_p3b_160 = " ".join(j.cmd for j in submitted[plan.q_p3b("160m")])
    assert "_t70m" not in cmds_p3b_160 and "_t410m" in cmds_p3b_160
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready", lambda _c, _s: True
    )
    result = run(cfg, opts)
    assert result.done
    picks = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())["picks"]
    assert set(picks) == set(plan.SIZES)


def test_campaign_done_with_p3b_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[bool] = []
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda cfg, *, dry_run=False, **_k: built.append(True) or {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: True,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    transitions = [
        "preflight",
        "submit_p1",
        "teacher_gate",
        "submit_p2",
        *[f"submit_p3_{size}" for size in plan.SIZES],
    ]
    _write_state(cfg, transitions)
    result = run(cfg, opts)
    assert result.done
    assert "finalize" in result.fired
    assert built == [True]
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    assert not store.queue_exists(cfg.queue_root, plan.q_p3b("70m"))
    assert not store.queue_exists(cfg.queue_root, plan.q_p3b("160m"))


def test_campaign_done_while_p3b_is_still_draining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P3b is the optional tail: a pending P3b queue must not block finalize.

    `CAMPAIGN_DONE` and X0-X2/X4 need only P1 + P2 + P3; X3 reports whatever P3b
    leaves exist. The incomplete queues are recorded so the operator knows to
    rebuild results once the tail drains.
    """
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda cfg, *, dry_run=False, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: True,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(
        cfg,
        [
            "preflight",
            "submit_p1",
            "teacher_gate",
            "submit_p2",
            *[f"submit_p3_{size}" for size in plan.SIZES],
        ],
    )
    picks = {"70m": plan.LR_DEFAULT}
    train = cgen.lr_robustness_student_configs(picks, size="70m")
    train.update(cgen.lr_robustness_control_configs(picks, size="70m"))
    b_jobs = jobs.lr_robustness_jobs(train, picks, size="70m")
    driver_io.submit_queue(cfg, plan.q_p3b("70m"), b_jobs)
    assert store.queue_status(cfg.queue_root, plan.q_p3b("70m"))["pending"] > 0

    result = run(cfg, opts)

    assert result.done
    assert "finalize" in result.fired
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    finalize = driver_io.load_state(cfg)["transitions"]["finalize"]
    assert finalize["p3b_incomplete"] == [plan.q_p3b("70m")]


def test_second_advance_skips_identical_per_size_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._compute_lr_pick",
        lambda _cfg, size: LrPick(
            size=size, lr=plan.LR_DEFAULT, flagged=False, reason="t", table=()
        ),
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: False,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(cfg, ["preflight", "submit_p1", "teacher_gate", "submit_p2"])
    _complete_ladder(cfg, "70m")
    first = run(cfg, opts)
    assert "submit_p3_70m" in first.fired
    qname = plan.q_p3_students("70m")
    keys_before = {j.jobkey for j in store.load_jobs(cfg.queue_root, qname)}
    state = driver_io.load_state(cfg)
    del state["transitions"]["submit_p3_70m"]
    driver_io.save_state(cfg, state)
    second = run(cfg, opts)
    assert "submit_p3_70m" in second.fired
    keys_after = {j.jobkey for j in store.load_jobs(cfg.queue_root, qname)}
    assert keys_after == keys_before


def test_existing_screen_picks_not_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = plan.LR_LADDER[0]

    def boom(_cfg, size):
        raise AssertionError(f"pick_lr recomputed for {size}")

    monkeypatch.setattr("pipelines.xsize_campaign_pipeline._compute_lr_pick", boom)
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: False,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(cfg, ["preflight", "submit_p1", "teacher_gate", "submit_p2"])
    _complete_ladder(cfg, "70m")
    pick = LrPick(size="70m", lr=planted, flagged=False, reason="planted", table=())
    from subliminal_learning.campaign.xsize_selection import picks_report

    (cfg.logs_dir / plan.SCREEN_PICKS).write_text(
        json.dumps(picks_report({"70m": pick}))
    )
    expected_jobs = jobs.headline_jobs(
        cgen.headline_student_configs({"70m": planted}, size="70m"),
        {"70m": planted},
        size="70m",
    )
    result = run(cfg, opts)
    assert "submit_p3_70m" in result.fired
    got = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.q_p3_students("70m"))}
    assert got == {j.jobkey for j in expected_jobs}
    body = json.loads((cfg.logs_dir / plan.SCREEN_PICKS).read_text())
    assert body["picks"]["70m"]["lr"] == planted


def test_interrupted_p2_does_not_reenqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: False,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(cfg, ["preflight", "submit_p1", "teacher_gate"])
    first = run(cfg, opts)
    assert "submit_p2" in first.fired
    keys_before = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P2_LADDER)}
    jobs_list = store.load_jobs(cfg.queue_root, plan.Q_P2_LADDER)
    store.record_result(
        cfg.queue_root,
        plan.Q_P2_LADDER,
        jobs_list[0],
        rc=0,
        node="testnode",
        gpu=0,
        start_utc="t0",
        end_utc="t1",
        log="",
        attempt=1,
    )
    second = run(cfg, opts)
    assert "submit_p2" not in second.fired
    keys_after = {j.jobkey for j in store.load_jobs(cfg.queue_root, plan.Q_P2_LADDER)}
    assert keys_after == keys_before
    assert store.queue_status(cfg.queue_root, plan.Q_P2_LADDER)["total"] == len(
        keys_before
    )


def test_auto_requeue_under_cap_then_halts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: False,
    )
    cfg = _cfg(tmp_path, auto_requeue=2)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts.cold_root)
    _write_state(cfg, ["preflight"])
    first = run(cfg, opts)
    assert "submit_p1" in first.fired
    jobs_list = store.load_jobs(cfg.queue_root, plan.Q_P1_TEACHERS)
    fail_job = jobs_list[0]
    store.record_result(
        cfg.queue_root,
        plan.Q_P1_TEACHERS,
        fail_job,
        rc=1,
        node="testnode",
        gpu=0,
        start_utc="t0",
        end_utc="t1",
        log="",
        attempt=1,
    )
    result = run(cfg, opts)
    assert (
        store.job_state(cfg.queue_root, plan.Q_P1_TEACHERS, fail_job) == store.PENDING
    )
    assert any(plan.Q_P1_TEACHERS in w for w in result.waiting)
    store.record_result(
        cfg.queue_root,
        plan.Q_P1_TEACHERS,
        fail_job,
        rc=1,
        node="testnode",
        gpu=0,
        start_utc="t0",
        end_utc="t1",
        log="",
        attempt=2,
    )
    with pytest.raises(CampaignGateError, match="auto-requeue 2"):
        run(cfg, opts)


def test_state_replays_transitions_instead_of_refiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._preflight",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("preflight re-fired")),
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.build_results",
        lambda *_a, **_k: {},
    )
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline._leaves_ready",
        lambda _cfg, _state: False,
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _write_state(
        cfg,
        ["preflight", "submit_p1", "teacher_gate", "submit_p2"],
    )
    result = run(cfg, opts)
    assert "preflight" not in result.fired
    assert "submit_p1" not in result.fired
    assert "teacher_gate" not in result.fired
    assert "submit_p2" not in result.fired
    assert submitted == []


def test_malformed_run_set_id_halts_before_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submitted: list[str] = []
    monkeypatch.setattr(
        "pipelines.xsize_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=replace(plan.SPEC, run_set_id="20260907_xsize_bear_mixed"),
    )
    with pytest.raises(CampaignGateError, match="malformed"):
        run(cfg)
    assert submitted == []


def test_status_includes_picks_and_voided(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    _write_state(cfg, ["preflight"], voided_sizes=["160m"])
    snap = status(cfg)
    assert "preflight" in snap["transitions"]
    assert snap["voided_sizes"] == ["160m"]
    assert snap["done"] is False
