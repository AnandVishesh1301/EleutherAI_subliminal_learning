"""Top-K driver state machine: preflight, submit-once, finalize, tail, deadline."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipelines.topk_campaign_pipeline import (
    CampaignGateError,
    TopkDriverOptions,
    driver_config,
    run,
    status,
)
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import topk_config_gen as cgen
from subliminal_learning.campaign import topk_jobs as jobs
from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.campaign.driver_io import CAMPAIGN_SPECS, CampaignDriverConfig
from subliminal_learning.campaign.topk_jobs import prefetch_refs
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir


def _cfg(tmp_path: Path, *, dry_run: bool = False, auto_requeue: int = 0):
    return driver_config(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=dry_run,
        auto_requeue=auto_requeue,
    )


def _opts(tmp_path: Path, **kw) -> TopkDriverOptions:
    prereg = tmp_path / "PREREG_TOPK.md"
    prereg.write_text("# prereg\n")
    return TopkDriverOptions(
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "datasets",
        **kw,
    )


def _mark_ckpt(ckpt: Path) -> None:
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"x")


def _mark_evaluated(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "preference_summary.json",
        "sdf_forced_choice_summary.json",
        "sdf_capability_summary.json",
    ):
        (config_dir / name).write_text("{}")


def _mark_cell(run_dir: Path, *, eas: bool = False) -> None:
    """Trained leaf + the verify triad one level up (+ optional EAS json)."""
    _mark_ckpt(run_dir / "checkpoints")
    (run_dir / "train_manifest.json").write_text("{}")
    _mark_evaluated(run_dir.parent)
    if eas:
        (run_dir / "eas").mkdir(parents=True, exist_ok=True)
        (run_dir / "eas" / "gkd_eas.json").write_text("{}")


def _plant_s0(cfg: CampaignDriverConfig, opts: TopkDriverOptions) -> None:
    for link, target in jobs.cold_symlink_plan(cfg.output_root, opts.cold_root):
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(target)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    refs = [{"repo": r.repo_id, "revision": r.revision} for r in prefetch_refs()]
    (cfg.logs_dir / plan.PREFETCH_MANIFEST).write_text(json.dumps({"refs": refs}))
    (cfg.logs_dir / plan.STAGING_REPORT).write_text(json.dumps({"passed": True}))
    (cfg.logs_dir / plan.SIGNATURE_REPORT).write_text(json.dumps({"rows": []}))


def _plant_reused(cfg: CampaignDriverConfig) -> None:
    """xsize decisions, the six bear teachers, and every reused full-vocab leaf."""
    xsize_logs = (
        cfg.output_root
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.TEACHER_RUN_SET_ID)
        / "logs"
    )
    xsize_logs.mkdir(parents=True, exist_ok=True)
    (xsize_logs / "teacher_gate_report.json").write_text(
        json.dumps(
            {
                "decisions": {
                    s: {"status": "frozen", "tag": plan.TEACHER_PARAM_TAG}
                    for s in plan.SIZES
                }
            }
        )
    )
    (xsize_logs / "screen_picks.json").write_text(
        json.dumps({"picks": {s: {"lr": plan.STUDENT_LR} for s in plan.SIZES}})
    )
    for ckpt in jobs.bear_teacher_ckpts(cfg.output_root):
        _mark_ckpt(ckpt)
    for leaf in jobs.reused_xsize_leaf_dirs(cfg.output_root):
        _mark_cell(leaf)


def _plant_block(
    cfg: CampaignDriverConfig, train: dict, block_name: str, *, eas: bool = True
) -> None:
    b = plan.block(block_name)
    for size in b.sizes:
        for leaf in jobs.block_leaf_dirs(train, cfg.output_root, block_name, size=size):
            _mark_cell(leaf, eas=eas)
        if eas:
            for leaf in jobs.eas_leaf_dirs(
                train, cfg.output_root, block_name, size=size
            ):
                (leaf / "eas").mkdir(parents=True, exist_ok=True)
                (leaf / "eas" / "gkd_eas.json").write_text("{}")


def _plant_null_teachers(cfg: CampaignDriverConfig) -> None:
    for size in plan.STUDENT_SIZES:
        for ckpt in jobs.eas_teacher_ckpts(cfg.output_root, size=size)[
            plan.NULL_ANIMAL
        ]:
            _mark_ckpt(Path(ckpt))


def _complete_queue(cfg: CampaignDriverConfig, name: str, *, fail_keys=()) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        store.record_result(
            cfg.queue_root,
            name,
            job,
            rc=1 if job.key in fail_keys else 0,
            node="testnode",
            gpu=0,
            start_utc="t0",
            end_utc="t1",
            log="",
            attempt=1,
        )


def _complete_core(cfg: CampaignDriverConfig) -> None:
    for name in (plan.Q_NULL_TEACHERS, plan.Q_NULL_TEACHER_EVALS):
        _complete_queue(cfg, name)
    for b in plan.core_blocks():
        for name in plan.block_queue_names(b.name):
            _complete_queue(cfg, name)


@pytest.fixture(autouse=True)
def _stub_pin_disk_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.driver_io.ensure_campaign_code_pin",
        lambda _cfg: None,
    )
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.driver_io.queue_job_defaults", lambda _cfg: {}
    )
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.disk_gate_readings",
        lambda **_k: {"ok": 100.0},
    )
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.build_results", lambda *_a, **_k: {}
    )


def test_driver_config_binds_topk_spec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert cfg.run_set_id == plan.RUN_SET_ID
    assert cfg.spec is plan.SPEC
    assert CAMPAIGN_SPECS["topk"] is plan.SPEC
    assert plan.Q_NULL_TEACHERS in cfg.queue_priorities
    assert driver_io.spec_for_campaign("topk").run_set_id == plan.RUN_SET_ID


def test_dry_run_walks_s0_to_finalize_writing_nothing(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, dry_run=True)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    result = run(cfg, opts)
    assert "preflight" in result.fired
    assert "submit_all" in result.fired
    assert "finalize_core" in result.fired
    assert result.done
    assert not cfg.state_path.exists()
    assert not store.list_queues(cfg.queue_root)
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    assert not (tmp_path / "generated").exists() or not any(
        (tmp_path / "generated").glob("*.yaml")
    )
    assert not (cfg.logs_dir / plan.EAS_LISTS_DIR).exists()


def test_preflight_refuses_when_inherited_decisions_disagree(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    xsize_logs = (
        cfg.output_root
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.TEACHER_RUN_SET_ID)
        / "logs"
    )
    (xsize_logs / "screen_picks.json").write_text(
        json.dumps({"picks": {s: {"lr": 8.0e-4} for s in plan.SIZES}})
    )
    with pytest.raises(CampaignGateError, match="lr\\*"):
        run(cfg, opts)
    assert not cfg.state_path.exists()


def test_preflight_refuses_incomplete_reused_leaves(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    victim = jobs.reused_xsize_leaf_dirs(cfg.output_root)[0]
    (victim / "train_manifest.json").unlink()
    with pytest.raises(CampaignGateError, match="reused xsize leaves incomplete"):
        run(cfg, opts)


def test_submit_all_submits_every_queue_once_with_frozen_jobkeys(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    first = run(cfg, opts)
    assert "preflight" in first.fired and "submit_all" in first.fired
    assert not first.done
    submitted = set(store.list_queues(cfg.queue_root))
    assert submitted == set(plan.QUEUE_PRIORITIES)
    n_train = sum(
        len(store.load_jobs(cfg.queue_root, n))
        for n in submitted
        if "-students-" in n or "-controls-" in n
    )
    assert n_train == 174
    assert len(store.load_jobs(cfg.queue_root, plan.Q_NULL_TEACHERS)) == 6
    eas_list = jobs.eas_list_path(cfg.logs_dir, "k5", size="410m")
    doc = json.loads(eas_list.read_text())
    assert len(doc["leaves"]) == 18
    assert doc["prompt_pool_dir"] == str(tmp_path / "datasets" / plan.PROMPT_POOL_DIR)
    keys_before = {
        n: {j.jobkey for j in store.load_jobs(cfg.queue_root, n)} for n in submitted
    }
    state = driver_io.load_state(cfg)
    del state["transitions"]["submit_all"]
    driver_io.save_state(cfg, state)
    second = run(cfg, opts)
    assert "submit_all" in second.fired
    keys_after = {
        n: {j.jobkey for j in store.load_jobs(cfg.queue_root, n)} for n in submitted
    }
    assert keys_after == keys_before
    assert json.loads(eas_list.read_text()) == doc
    meta = store.read_meta(cfg.queue_root, plan.q_eas("k5", "410m"))
    assert set(meta.depends_on) == {
        plan.q_students("k5", "410m"),
        plan.q_controls("k5", "410m"),
        plan.Q_NULL_TEACHERS,
    }
    window = driver_io.load_state(cfg)["deadline_windows"]
    assert len(window) == 1 and window[0]["hours"] == plan.DEADLINE_HOURS


def test_finalize_core_fires_with_tail_incomplete_and_tail_folds_in_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.build_results",
        lambda *_a, **_k: calls.append("results") or {},
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    run(cfg, opts)
    train = cgen.all_train_configs()
    _complete_core(cfg)
    _plant_null_teachers(cfg)
    for b in plan.core_blocks():
        _plant_block(cfg, train, b.name)
    result = run(cfg, opts)
    assert "finalize_core" in result.fired
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    assert not result.done
    assert any("tail draining" in w for w in result.waiting)
    state = driver_io.load_state(cfg)
    assert set(state["transitions"]["finalize_core"]["tail_pending"]) == {
        b.name for b in plan.tail_blocks()
    }
    assert calls == ["results"]
    for name in plan.block_queue_names("k30"):
        _complete_queue(cfg, name)
    _plant_block(cfg, train, "k30")
    result = run(cfg, opts)
    assert "results_k30" in result.fired
    assert "results_x70m" not in result.fired
    assert calls == ["results", "results"]
    assert "finalize_core" not in result.fired


def test_finalize_waits_for_eas_json_not_only_evals(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    run(cfg, opts)
    train = cgen.all_train_configs()
    _complete_core(cfg)
    _plant_null_teachers(cfg)
    for b in plan.core_blocks():
        _plant_block(cfg, train, b.name, eas=(b.name != "k5"))
    result = run(cfg, opts)
    assert "finalize_core" not in result.fired
    assert any("k5/" in w and "EAS incomplete" in w for w in result.waiting)


def test_deadline_pauses_pool_and_records_truncated_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paused: list[str] = []
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline.control_io.pause",
        lambda **kw: paused.append(kw["reason"]) or {},
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path, deadline_hours=1.0)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    run(cfg, opts)
    state = driver_io.load_state(cfg)
    old = (
        (datetime.now(timezone.utc) - timedelta(hours=2))
        .isoformat()
        .replace("+00:00", "Z")
    )
    state["deadline_windows"][-1]["start"] = old
    driver_io.save_state(cfg, state)
    result = run(cfg, opts)
    assert "deadline_pause" in result.fired
    assert paused and "deadline" in paused[0]
    state = driver_io.load_state(cfg)
    assert set(state["transitions"]["deadline_pause"]["tail_truncated"]) == {
        b.name for b in plan.tail_blocks()
    }
    assert state["deadline_windows"][-1]["reason"] == "deadline"
    again = run(cfg, opts)
    assert "deadline_pause" not in again.fired
    assert len(paused) == 1


def test_auto_requeue_under_cap_then_halts(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, auto_requeue=2)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    run(cfg, opts)
    fail_job = store.load_jobs(cfg.queue_root, plan.Q_NULL_TEACHERS)[0]
    store.record_result(
        cfg.queue_root,
        plan.Q_NULL_TEACHERS,
        fail_job,
        rc=1,
        node="n",
        gpu=0,
        start_utc="t0",
        end_utc="t1",
        log="",
        attempt=1,
    )
    result = run(cfg, opts)
    assert (
        store.job_state(cfg.queue_root, plan.Q_NULL_TEACHERS, fail_job) == store.PENDING
    )
    assert any(plan.Q_NULL_TEACHERS in w for w in result.waiting)
    store.record_result(
        cfg.queue_root,
        plan.Q_NULL_TEACHERS,
        fail_job,
        rc=1,
        node="n",
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
        "pipelines.topk_campaign_pipeline.driver_io.submit_queue",
        lambda _cfg, name, _jobs, **_kw: submitted.append(name),
    )
    monkeypatch.setattr(
        "pipelines.topk_campaign_pipeline._preflight",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("preflight re-fired")),
    )
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    cfg.logs_dir.mkdir(parents=True)
    driver_io.save_state(
        cfg,
        {
            "transitions": {"preflight": {"at": "t"}, "submit_all": {"at": "t"}},
            "deadline_windows": [],
        },
    )
    result = run(cfg, opts)
    assert "preflight" not in result.fired and "submit_all" not in result.fired
    assert submitted == []


def test_malformed_run_set_id_halts_before_submit(tmp_path: Path) -> None:
    cfg = CampaignDriverConfig(
        output_root=tmp_path,
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated",
        dry_run=True,
        spec=replace(plan.SPEC, run_set_id="20260910_topk_bear_mixed"),
    )
    with pytest.raises(CampaignGateError, match="malformed"):
        run(cfg)


def test_status_reports_deadline_and_tail_projection(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    opts = _opts(tmp_path)
    _plant_s0(cfg, opts)
    _plant_reused(cfg)
    run(cfg, opts)
    snap = status(cfg, opts)
    assert "submit_all" in snap["transitions"]
    assert snap["deadline"]["hours"] == plan.DEADLINE_HOURS
    assert snap["deadline"]["elapsed_h"] is not None
    assert snap["projected_tail_hours"] > 0
    assert snap["done"] is False and snap["core_done"] is False
