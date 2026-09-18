"""ws160m partition driver: phase machine, teacher wait/fallback, wave 2, deadline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pipelines import ws160m_campaign_pipeline as pipe
from pipelines.ws160m_campaign_pipeline import (
    CampaignGateError,
    Ws160mDriverOptions,
    driver_config,
    missing_model_assets,
    run,
    status,
)
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.jobqueue.model import Job

_T0 = 1_800_000_000.0
_HOST = "soar-subliminal-0"


def _cfg(tmp_path: Path, partition: str, **kw):
    return driver_config(
        partition,
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        generated_dir_override=tmp_path / "generated" / partition,
        **kw,
    )


def _opts(
    tmp_path: Path, partition: str, clock: list[float], **kw
) -> Ws160mDriverOptions:
    prereg = tmp_path / "PREREG_WS160M.md"
    prereg.write_text("# prereg\n")
    paths = tmp_path / "paths_ws160m.yaml"
    paths.write_text(
        yaml.safe_dump(
            {
                "output_root": str(tmp_path / "out"),
                "hf_cache_root": str(tmp_path / "bm"),
                "dataset_cache_root": str(tmp_path / "datasets"),
                "wandb_dir": str(tmp_path / "wandb"),
                "queue_root": str(tmp_path / "queue"),
                "audit_root": str(tmp_path / "cold"),
            }
        )
    )
    return Ws160mDriverOptions(
        partition=partition,
        paths_config=paths,
        cold_root=tmp_path / "cold",
        prereg_path=prereg,
        base_models_root=tmp_path / "bm",
        require_smoke=False,
        now=lambda: clock[0],
        **kw,
    )


class _Fake:
    """Stubs the queue store, submission, pin and expected-artefact probes."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, partition: str) -> None:
        self.submitted: list[tuple[str, tuple[str, ...]]] = []
        self.complete: set[str] = set()
        self.missing_teachers: list[Path] = []
        self.missing_cells: list[Path] = []
        self.missing_evals: list[Path] = []
        self.fallback_jobs: list[Job] = [Job("fb", "true")]
        self.stage_missing: dict[str, list[pairs.TeacherLeaf]] = {}
        p = plan.partition(partition)
        dummy = {q: [Job(f"{q}-job", "true")] for q in plan.queue_priorities(p)}

        def fake_preflight(cfg, state, opts):
            state["code_sha"] = "0" * 40
            state["wave"] = p.wave
            state["transitions"]["preflight"] = {"at": "t"}

        monkeypatch.setattr(pipe, "_preflight", fake_preflight)
        force_q = plan.queue(p, "evals-force")

        def fake_bundle(cfg, opts, *, wave, force_students=()):
            by_q = dict(dummy)
            if force_students:
                by_q[force_q] = [Job("forced", "eval --force")]
            return {}, {}, by_q

        monkeypatch.setattr(pipe, "_bundle", fake_bundle)
        monkeypatch.setattr(
            pipe, "_fallback_teacher_jobs", lambda *a, **k: self.fallback_jobs
        )
        monkeypatch.setattr(
            pipe.driver_io, "ensure_campaign_code_pin", lambda cfg: None
        )
        monkeypatch.setattr(pipe.driver_io, "queue_job_defaults", lambda cfg: {})
        monkeypatch.setattr(
            pipe.code_pin_io, "resolve_commit_sha", lambda root, sha: str(sha)
        )
        monkeypatch.setattr(
            pipe.driver_io, "requeue_failed_under_cap", lambda cfg, name: 0
        )

        self.nodes: dict[str, tuple] = {}

        def submit(cfg, name, jobs, *, depends_on=None, node=None, defaults=None):
            self.submitted.append((name, tuple(depends_on or ())))
            self.nodes[name] = (node, (defaults or {}).get("cwd"))
            return "submitted"

        monkeypatch.setattr(pipe.driver_io, "submit_queue", submit)
        submitted_names = lambda: {n for n, _ in self.submitted}  # noqa: E731
        monkeypatch.setattr(
            pipe.store, "queue_exists", lambda root, name: name in submitted_names()
        )
        monkeypatch.setattr(
            pipe.store, "queue_complete", lambda root, name: name in self.complete
        )
        monkeypatch.setattr(
            pipe.store,
            "queue_status",
            lambda root, name: {
                "pending": 0,
                "running": 0,
                "done": 1,
                "failed": 0,
                "total": 1,
            },
        )
        monkeypatch.setattr(pipe.store, "this_host", lambda: _HOST)
        monkeypatch.setattr(
            pipe.exp,
            "missing_required_teachers",
            lambda *a, **k: list(self.missing_teachers),
        )
        monkeypatch.setattr(
            pipe.exp,
            "missing_stage_teachers",
            lambda partition, stage, root, *, wave: list(
                self.stage_missing.get(stage, ())
            ),
        )
        monkeypatch.setattr(
            pipe.exp, "missing_cells", lambda *a, **k: list(self.missing_cells)
        )
        monkeypatch.setattr(
            pipe.exp, "missing_evals", lambda *a, **k: list(self.missing_evals)
        )
        monkeypatch.setattr(
            pipe,
            "_coverage",
            lambda cfg, opts, state, *, partial, holes: {
                "partial": partial,
                "missing_cells": [],
                "missing_evals": [],
            },
        )

    def names(self) -> list[str]:
        return [n for n, _ in self.submitted]

    def deps(self, name: str) -> tuple[str, ...]:
        return next(d for n, d in self.submitted if n == name)


def test_dosehp_advances_controls_wait_students_finalize(
    tmp_path: Path, monkeypatch
) -> None:
    fake = _Fake(monkeypatch, "dosehp")
    clock = [_T0]
    cfg, opts = (
        _cfg(tmp_path, "dosehp"),
        _opts(tmp_path, "dosehp", clock, auto_wave2=False),
    )
    fake.missing_teachers = [Path("/t/one")]
    r1 = run(cfg, opts)
    assert ("preflight" in r1.fired) and ("submit_controls" in r1.fired)
    assert "wait_teachers" in r1.waiting and not r1.done
    assert fake.names() == ["ws160m-dosehp-controls"]
    state = json.loads(cfg.state_path.read_text())
    assert set(state["transitions"]) == {"preflight", "submit_controls"}
    fake.missing_teachers = []
    r2 = run(cfg, opts)
    assert {"wait_teachers", "submit_students"} <= set(r2.fired)
    assert fake.deps("ws160m-dosehp-dose-bear") == ()
    assert fake.deps("ws160m-dosehp-evals-dose-bear") == (
        "ws160m-dosehp-controls",
        "ws160m-dosehp-dose-bear",
    )
    for r in plan.HP_RECIPES:
        stage = f"hp-{plan.recipe_slug(r)}"
        if stage not in plan.DEFERRED_STAGES["dosehp"]:
            assert fake.deps(f"ws160m-dosehp-evals-{stage}") == (
                f"ws160m-dosehp-{stage}",
            )
    assert not r2.done
    fake.complete.update(fake.names())
    r3 = run(cfg, opts)
    assert "finalize" in r3.fired and r3.done
    assert (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file()
    assert (cfg.logs_dir / plan.RESULTS_COVERAGE_JSON).is_file()
    assert status(cfg)["done"] is True


def test_wait_teachers_fallback_then_timeout(tmp_path: Path, monkeypatch) -> None:
    fake = _Fake(monkeypatch, "extra")
    clock = [_T0]
    cfg, opts = (
        _cfg(tmp_path, "extra"),
        _opts(tmp_path, "extra", clock, auto_wave2=False),
    )
    fake.missing_teachers = [Path("/t/one")]
    run(cfg, opts)
    assert set(fake.names()) == {"ws160m-extra-controls", "ws160m-extra-xbase"}
    clock[0] = _T0 + plan.WAIT_TEACHERS_FALLBACK_S + 1
    r = run(cfg, opts)
    assert "wait_teachers" in r.waiting
    assert "ws160m-extra-teachers-fallback" in fake.names()
    state = json.loads(cfg.state_path.read_text())
    assert state["transitions"]["teachers_fallback"]["n_jobs"] == 1
    run(cfg, opts)
    assert fake.names().count("ws160m-extra-teachers-fallback") == 1
    clock[0] = _T0 + plan.WAIT_TEACHERS_TIMEOUT_S + 1
    r = run(cfg, opts)
    assert "wait_teachers" in r.fired and "submit_students" in r.fired
    state = json.loads(cfg.state_path.read_text())
    assert state["transitions"]["wait_teachers"]["timed_out"] is True
    assert "ws160m-extra-heatmap" in fake.names()


def test_main_teachers_gate_students_and_wave2(tmp_path: Path, monkeypatch) -> None:
    fake = _Fake(monkeypatch, "main")
    clock = [_T0]
    staged: list[dict] = [{"ready": False}]
    calls: list[float] = []

    def fake_stage(**kw):
        calls.append(clock[0])
        return staged[0]

    cfg = _cfg(tmp_path, "main")
    opts = _opts(tmp_path, "main", clock, stage_wave2=fake_stage, auto_wave2=True)
    fake.missing_teachers = []
    r1 = run(cfg, opts)
    assert {"submit_teachers", "wait_teachers", "submit_students"} <= set(r1.fired)
    assert fake.deps("ws160m-main-teacher-evals") == ("ws160m-main-teachers",)
    assert fake.deps("ws160m-main-core") == ("ws160m-main-teachers",)
    assert fake.deps("ws160m-main-evals") == (
        "ws160m-main-controls",
        "ws160m-main-core",
    )
    assert "ws160m-main-w2-students" not in fake.names()
    fake.complete.update(fake.names())
    r2 = run(cfg, opts)
    assert not r2.done and calls == []
    clock[0] = _T0 + plan.WAVE2_START_HOURS * 3600 + 1
    run(cfg, opts)
    assert len(calls) == 1
    run(cfg, opts)
    assert len(calls) == 1, "polls are rate-limited"
    clock[0] += plan.WAVE2_POLL_S + 1
    staged[0] = {"ready": True, "passed": True}
    r3 = run(cfg, opts)
    assert "wave2_staged" in r3.fired
    r4 = run(cfg, opts)
    assert "submit_wave2" in r4.fired
    assert fake.deps("ws160m-main-w2-students") == ("ws160m-main-w2-teachers",)
    assert set(fake.deps("ws160m-main-w2-evals")) == {
        "ws160m-main-w2-students",
        "ws160m-main-w2-controls",
    }
    state = json.loads(cfg.state_path.read_text())
    assert state["wave"] == 2
    fake.complete.update(fake.names())
    r5 = run(cfg, opts)
    assert "finalize" in r5.fired and r5.done


def test_wave2_skipped_after_giveup(tmp_path: Path, monkeypatch) -> None:
    fake = _Fake(monkeypatch, "main")
    clock = [_T0]
    cfg = _cfg(tmp_path, "main")
    opts = _opts(
        tmp_path,
        "main",
        clock,
        stage_wave2=lambda **kw: {"ready": False},
        auto_wave2=True,
    )
    run(cfg, opts)
    fake.complete.update(fake.names())
    clock[0] = _T0 + plan.WAVE2_GIVEUP_HOURS * 3600 + 1
    r = run(cfg, opts)
    assert "wave2_skipped" in r.fired and "finalize" in r.fired and r.done


def test_force_evals_then_finalize(tmp_path: Path, monkeypatch) -> None:
    fake = _Fake(monkeypatch, "ladder")
    clock = [_T0]
    cfg, opts = _cfg(tmp_path, "ladder"), _opts(tmp_path, "ladder", clock)
    short = tmp_path / "out" / "cfgdir"
    short.mkdir(parents=True)
    (short / "preference_summary.json").write_text("{}")
    fake.missing_evals = [short]
    monkeypatch.setattr(
        pipe.exp, "expected_config_gseeds", lambda *a, **k: {short: frozenset({0})}
    )
    monkeypatch.setattr(
        pipe, "_config_dir_idents", lambda *a, **k: {short: "stem|bear|x@step2000"}
    )
    monkeypatch.setattr(pipe, "cell_is_done", lambda d: True)
    run(cfg, opts)
    fake.complete.update(fake.names())
    r = run(cfg, opts)
    assert any(f.startswith("submit_force_evals(1)") for f in r.fired)
    assert "ws160m-ladder-evals-force" in fake.names()
    fake.missing_evals = []
    fake.complete.update(fake.names())
    r = run(cfg, opts)
    assert "finalize" in r.fired


def test_deadline_partial_finalize(tmp_path: Path, monkeypatch) -> None:
    fake = _Fake(monkeypatch, "ladder")
    clock = [_T0]
    cfg, opts = (
        _cfg(tmp_path, "ladder"),
        _opts(tmp_path, "ladder", clock, deadline_hours=1.0),
    )
    run(cfg, opts)
    assert fake.names()
    clock[0] = _T0 + 3601
    r = run(cfg, opts)
    assert "finalize_partial" in r.fired and r.done
    assert not (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).exists()
    state = json.loads(cfg.state_path.read_text())
    assert state["deadline_windows"][-1]["reason"] == "deadline"


def test_run_refuses_a_foreign_partition_spec(tmp_path: Path, monkeypatch) -> None:
    _Fake(monkeypatch, "main")
    cfg = _cfg(tmp_path, "main")
    with pytest.raises(CampaignGateError, match="not partition"):
        run(cfg, _opts(tmp_path, "ladder", [_T0]))


def test_missing_model_assets_walks_refs_and_manifests(tmp_path: Path) -> None:
    hub, bm = tmp_path / "hub", tmp_path / "bm"
    missing = missing_model_assets("dosehp", wave=1, hub_cache=hub, base_models_root=bm)
    assert "pythia-160m@step143000" in missing and "pile-ws1@step143000" in missing
    for name in (
        "pythia-160m",
        "pythia-160m-seed1",
        "pythia-160m-seed2",
        "pythia-160m-data-seed1",
    ):
        ref = hub / f"models--EleutherAI--{name}" / "refs"
        ref.mkdir(parents=True)
        (ref / "step143000").write_text("abc")
    for base in ("pile-ws1", "dclm-main", "dclm-ws1"):
        d = models.converted_dir(bm, base, 143000)
        d.mkdir(parents=True)
        (d / "conversion_manifest.json").write_text("{}")
    assert (
        missing_model_assets("dosehp", wave=1, hub_cache=hub, base_models_root=bm) == []
    )


def test_explicit_code_sha_repins_every_submitted_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "main")
    clock = [_T0]
    run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    submitted = {name for name, _ in fake.submitted}
    assert submitted
    updated: dict[str, dict] = {}
    monkeypatch.setattr(pipe.code_pin_io, "resolve_commit_sha", lambda root, sha: sha)
    monkeypatch.setattr(
        pipe.driver_io, "queue_job_defaults", lambda cfg: {"cwd": f"pin/{cfg.code_sha}"}
    )
    monkeypatch.setattr(
        pipe.store,
        "update_queue_defaults",
        lambda root, name, defaults: updated.__setitem__(name, defaults),
    )
    new = "1" * 40
    result = run(_cfg(tmp_path, "main", code_sha=new), _opts(tmp_path, "main", clock))
    assert any(f.startswith("repin_queues") for f in result.fired)
    assert set(updated) == submitted
    assert all(d == {"cwd": f"pin/{new}"} for d in updated.values())
    state = json.loads(
        (tmp_path / "out" / "gkd_students" / plan.MAIN_RUN_SET_ID / "logs")
        .joinpath("campaign_state.json")
        .read_text()
    )
    assert state["code_sha"] == new and state["repins"][-1]["to"] == new
    updated.clear()
    again = run(_cfg(tmp_path, "main", code_sha=new), _opts(tmp_path, "main", clock))
    assert not any(f.startswith("repin_queues") for f in again.fired) and not updated


def test_an_archived_queue_is_resubmitted_with_its_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "main")
    clock = [_T0]
    run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    p = plan.partition("main")
    teachers = plan.queue(p, "teachers")
    evals = plan.queue(p, "teacher-evals")
    before = dict(fake.submitted)
    assert teachers in before and before[evals] == (teachers,)
    fake.submitted = [(n, d) for n, d in fake.submitted if n != teachers]
    result = run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    assert f"restored_queues({teachers})" in result.fired
    assert dict(fake.submitted)[teachers] == ()
    state = json.loads(
        (tmp_path / "out" / "gkd_students" / plan.MAIN_RUN_SET_ID / "logs")
        .joinpath("campaign_state.json")
        .read_text()
    )
    assert state["restores"][-1] == {
        "at": state["restores"][-1]["at"],
        "phase": "submit_teachers",
        "queues": [teachers],
    }
    again = run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    assert not any(f.startswith("restored_queues") for f in again.fired)


def test_retry_exhausted_failures_become_holes_until_the_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, "main")
    opts = _opts(tmp_path, "main", [_T0])
    name = "ws160m-main-controls"
    failed = ["a", "b"]
    monkeypatch.setattr(pipe, "_failed_jobkeys", lambda cfg, n: list(failed))
    monkeypatch.setattr(
        pipe.store, "queue_status", lambda root, n: {"failed": len(failed), "total": 33}
    )
    holes: list[dict] = []
    pipe._record_holes(cfg, name, opts, holes, RuntimeError("cap"))
    assert holes == [{"queue": name, "failed": ["a", "b"], "at": holes[0]["at"]}]
    failed.append("c")
    with pytest.raises(CampaignGateError, match="3 retry-exhausted"):
        pipe._record_holes(cfg, name, opts, holes, None)


def test_overflow_submit_refuses_yamls_the_other_node_cannot_read(
    tmp_path: Path,
) -> None:
    cfg = _cfg(tmp_path, "dosehp")
    name, node = "ws160m-dosehp-hp-lr2e-4_r8a16", "soar-subliminal-3"
    shared = tmp_path / "out" / "x.yaml"
    relative = [Job("a", "uv run x --sweep-config configs/generated/a.yaml")]
    with pytest.raises(CampaignGateError, match="absolute YAML"):
        pipe._require_overflow_configs(cfg, name, node, relative)
    absent = [Job("b", f"uv run x --sweep-config {shared} --animals bear")]
    with pytest.raises(CampaignGateError, match="absolute YAML"):
        pipe._require_overflow_configs(cfg, name, node, absent)
    shared.parent.mkdir(parents=True)
    shared.write_text("x: 1\n")
    pipe._require_overflow_configs(cfg, name, node, absent)
    assert pipe._shared_config_dir(cfg, "dosehp") == (
        cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME
    )
    assert pipe._shared_config_dir(_cfg(tmp_path, "main"), "main") is None


def test_shared_config_copy_inherits_absolute_train_yamls(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, "dosehp")
    opts = _opts(tmp_path, "dosehp", [_T0])
    n_train, n_verify = pipe._write_configs(cfg, opts, wave=1)
    shared = cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME
    assert len(list(shared.glob("*.yaml"))) == n_train + n_verify
    verify = [y for y in shared.glob("*.yaml") if "verify_" in y.name]
    assert verify
    for y in verify:
        inherited = Path(yaml.safe_load(y.read_text())["inherit_grid"])
        assert inherited.parent == shared and inherited.is_file()
    local = next(cfg.generated_dir.glob("*verify_*.yaml"))
    assert not Path(yaml.safe_load(local.read_text())["inherit_grid"]).is_absolute()
    assert pipe._write_configs(cfg, opts, wave=1) == (n_train, n_verify)


def test_a_hole_is_dropped_once_its_queue_completes_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, "main")
    opts = _opts(tmp_path, "main", [_T0])
    name = "ws160m-main-controls"
    monkeypatch.setattr(pipe.store, "queue_exists", lambda root, n: True)
    monkeypatch.setattr(pipe.driver_io, "requeue_failed_under_cap", lambda c, n: 0)
    monkeypatch.setattr(pipe.store, "queue_complete", lambda root, n: True)
    monkeypatch.setattr(
        pipe.store, "queue_status", lambda root, n: {"failed": 0, "total": 33}
    )
    holes = [
        {"queue": name, "failed": ["a"], "at": "t"},
        {"queue": "other", "failed": ["b"], "at": "t"},
    ]
    assert pipe._queue_ready(cfg, name, opts, holes)
    assert holes == [{"queue": "other", "failed": ["b"], "at": "t"}]


def test_restore_records_only_queues_that_were_actually_resubmitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "main")
    clock = [_T0]
    run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    state_path = (
        tmp_path / "out" / "gkd_students" / plan.MAIN_RUN_SET_ID / "logs"
    ) / "campaign_state.json"
    state = json.loads(state_path.read_text())
    state["transitions"]["submit_students"]["queues"]["ws160m-main-phantom"] = 1
    state_path.write_text(json.dumps(state))
    result = run(_cfg(tmp_path, "main"), _opts(tmp_path, "main", clock))
    assert not any(f.startswith("restored_queues") for f in result.fired)
    after = json.loads(state_path.read_text())
    assert not any(
        "ws160m-main-phantom" in r["queues"] for r in after.get("restores", [])
    )
    assert "ws160m-main-phantom" not in fake.names()


def test_coverage_counts_deferred_leaves_apart_and_refresh_drops_healed_holes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, "dosehp")
    opts = _opts(tmp_path, "dosehp", [_T0])
    monkeypatch.setattr(pipe.store, "this_host", lambda: _HOST)
    state = pipe.driver_io.load_state(cfg)
    state["transitions"]["finalize"] = {"at": "t"}
    state["holes"] = [{"queue": "ws160m-dosehp-controls", "failed": ["a"], "at": "t"}]
    pipe.driver_io.save_state(cfg, state)
    cov = pipe.refresh_coverage(cfg, opts)
    arms = cov["landed_by_arm"]
    assert arms["hp"]["deferred"] > 0 and arms["hp"]["landed"] == 0
    assert sum(r["expected"] for r in arms.values()) == cov["expected"]["leaves"]
    assert cov["holes"] == [] and not cov["partial"]
    assert json.loads((cfg.logs_dir / plan.RESULTS_COVERAGE_JSON).read_text()) == cov
    saved = pipe.driver_io.load_state(cfg)
    assert saved["holes"] == [] and list(saved["transitions"]) == ["finalize"]


def test_deadline_hours_override_applies_to_the_open_window(tmp_path: Path) -> None:
    opts = _opts(tmp_path, "main", [_T0], deadline_hours=24.0)
    state: dict = {}
    pipe._open_window(state, opts)
    later = _opts(tmp_path, "main", [_T0 + 3600], deadline_hours=17.5)
    window = pipe._open_window(state, later)
    assert window["hours"] == 17.5 and window["hours_history"][0]["from"] == 24.0
    assert len(state["deadline_windows"]) == 1
    assert pipe._deadline_passed(
        state, _opts(tmp_path, "main", [_T0 + 17.5 * 3600 + 1], deadline_hours=17.5)
    )


def test_train_queues_skip_only_exact_teacher_stages() -> None:
    extra = plan.partition("extra")
    names = pipe._train_queues(extra, wave2=False)
    assert "ws160m-extra-teacher-dose" in names
    assert "ws160m-extra-teachers-fallback" not in names
    assert not any("evals" in n for n in names)


def test_missing_planned_student_queues_are_submitted_on_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "extra")
    clock = [_T0]
    cfg = _cfg(tmp_path, "extra")
    opts = _opts(tmp_path, "extra", clock)
    fake.missing_teachers = []
    run(cfg, opts)
    evals = "ws160m-extra-evals-heatmap"
    assert evals in fake.names()
    fake.submitted = [(n, d) for n, d in fake.submitted if n != evals]
    state = json.loads(cfg.state_path.read_text())
    state["transitions"]["submit_students"]["queues"].pop(evals)
    cfg.state_path.write_text(json.dumps(state))
    r = run(cfg, opts)
    assert f"restored_queues({evals})" in r.fired and evals in fake.names()
    state = json.loads(cfg.state_path.read_text())
    assert evals in state["transitions"]["submit_students"]["queues"]


def test_overflow_stages_submit_to_their_override_node_with_that_clone_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "dosehp")
    clock = [_T0]
    fake.missing_teachers = []
    monkeypatch.setattr(
        pipe.code_pin_io, "resolve_commit_sha", lambda root, sha: "a" * 40
    )
    run(_cfg(tmp_path, "dosehp", code_sha="a" * 40), _opts(tmp_path, "dosehp", clock))
    for stage, node in plan.STAGE_NODE_OVERRIDES["dosehp"].items():
        for q in (f"ws160m-dosehp-{stage}", f"ws160m-dosehp-evals-{stage}"):
            got_node, cwd = fake.nodes[q]
            assert got_node == node
            expected = (
                plan.clone_root(plan.partition_on(node)) / ".code_pins" / ("a" * 40)
            )
            assert cwd == str(expected)
    assert fake.nodes["ws160m-dosehp-dose-bear"] == (None, None)


# ---- Part II (ws160m2) -----------------------------------------------------------------


def _teacher_leaf(animal: str, base: str, step: int) -> pairs.TeacherLeaf:
    combo = pairs.TeacherCombo(animal, base, step, plan.TEACHER_PARAM_TAG_LORA, 2)
    return pairs.TeacherLeaf(combo, 0)


def test_gen2_stage_blocks_submit_as_their_teachers_land(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "matrix")
    clock = [_T0]
    cfg, opts = _cfg(tmp_path, "matrix"), _opts(tmp_path, "matrix", clock)
    assert opts.deadline_hours == 108.0
    own = _teacher_leaf("bear", "pile-main", 143000)
    assert own.combo.owner.name == "matrix"
    fake.stage_missing = {"m2": [own], "ml": [own]}
    r1 = run(cfg, opts)
    assert {"preflight", "submit_teachers", "submit_stage(b2)"} <= set(r1.fired)
    assert "submit_controls" not in r1.fired and "submit_students" not in r1.fired
    assert "submit_stage(m2)" not in r1.fired and "teachers:m2(1)" in r1.waiting
    assert "submit_stage(law-p64)" in r1.fired, "batch 2 needs no manual gate"
    assert fake.deps("ws160m2-matrix-teacher-evals") == ("ws160m2-matrix-teachers",)
    assert fake.deps("ws160m2-matrix-teacher-evals-b2") == (
        "ws160m2-matrix-teachers-b2",
    )
    assert fake.deps("ws160m2-matrix-b2") == ()
    assert fake.deps("ws160m2-matrix-evals-b2") == ("ws160m2-matrix-b2",)
    assert fake.deps("ws160m2-matrix-lmeval-b2") == ("ws160m2-matrix-evals-b2",)
    assert "ws160m2-matrix-m2" not in fake.names()
    assert "ws160m2-matrix-controls" not in fake.names()
    assert "ws160m2-matrix-teachers-fallback" not in fake.names()
    fake.complete.update(fake.names())
    r2 = run(cfg, opts)
    assert not r2.done and "finalize" not in r2.fired
    fake.stage_missing = {}
    r3 = run(cfg, opts)
    assert {"submit_stage(m2)", "submit_stage(ml)", "submit_students"} <= set(r3.fired)
    state = json.loads(cfg.state_path.read_text())
    assert list(state["stages"]) and set(state["stages"]) == set(
        plan.gen2_student_stages(plan.partition("matrix"))
    )
    assert state["wave"] == 3 and "submit_controls" not in state["transitions"]
    assert fake.names().count("ws160m2-matrix-b2") == 1
    fake.complete.update(fake.names())
    r4 = run(cfg, opts)
    assert "finalize" in r4.fired and r4.done


def test_gen2_fallback_waits_until_the_stage_is_overdue_and_fires_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "grid")
    clock = [_T0]
    cfg, opts = _cfg(tmp_path, "grid"), _opts(tmp_path, "grid", clock)
    foreign = _teacher_leaf("tiger", "pythia-160m", 143000)
    assert foreign.combo.owner.name == "matrix"
    fake.stage_missing = {"ac-tiger": [foreign]}
    run(cfg, opts)
    fq = "ws160m2-grid-teachers-fallback"
    assert "ws160m2-grid-ac-eagle" in fake.names(), "later stages are not held back"
    clock[0] = _T0 + plan.WAIT_TEACHERS_FALLBACK_S + 1
    run(cfg, opts)
    assert fq not in fake.names(), "earlier stages still run: not overdue yet"
    fake.complete.update(fake.names())
    run(cfg, opts)
    state = json.loads(cfg.state_path.read_text())
    assert "ac-tiger" in state["stage_blocked_since"] and fq not in fake.names()
    clock[0] += plan.WAIT_TEACHERS_FALLBACK_S + 1
    r = run(cfg, opts)
    assert fq in fake.names() and "teachers:ac-tiger(1)" in r.waiting
    state = json.loads(cfg.state_path.read_text())
    assert state["transitions"]["teachers_fallback"]["stage"] == "ac-tiger"
    run(cfg, opts)
    assert fake.names().count(fq) == 1
    fake.stage_missing = {}
    r = run(cfg, opts)
    assert "submit_stage(ac-tiger)" in r.fired and "submit_students" in r.fired


def test_gen2_fallback_never_targets_a_part_one_run_set() -> None:
    assert pipe.fallback_owners(plan.partition("grid")) == ("matrix",)
    assert pipe.fallback_owners(plan.partition("matrix")) == ("grid",)
    assert pipe.fallback_owners(plan.partition("extra")) == ("main", "ladder")
    grid = plan.partition("grid")
    ok = {"a.yaml": {"run_set_id": plan.MATRIX_RUN_SET_ID}}
    pipe.assert_fallback_target(grid, ok)
    for frozen in (plan.MAIN_RUN_SET_ID, plan.LADDER_RUN_SET_ID):
        with pytest.raises(CampaignGateError, match="frozen run sets"):
            pipe.assert_fallback_target(grid, {**ok, "b.yaml": {"run_set_id": frozen}})
    pipe.assert_fallback_target(
        plan.partition("extra"), {"b.yaml": {"run_set_id": plan.MAIN_RUN_SET_ID}}
    )


def test_gen2_fallback_jobs_train_the_owner_tree_from_its_shared_yamls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, opts = _cfg(tmp_path, "grid"), _opts(tmp_path, "grid", [_T0])
    monkeypatch.setattr(pipe, "cell_is_done", lambda d: "20260916_" in str(d))
    fb = pipe._fallback_teacher_jobs(cfg, opts, wave=3)
    assert len(fb) == 99 and all(j.key.startswith("fallback|matrix-teach|") for j in fb)
    shared = (
        tmp_path / "out" / "gkd_students" / plan.MATRIX_RUN_SET_ID / "generated_configs"
    )
    for job in fb:
        path = Path(job.cmd.split("--sweep-config ", 1)[1].split()[0])
        assert path.parent == shared and path.is_file() and "--no-wandb" in job.cmd
        assert yaml.safe_load(path.read_text())["run_set_id"] == plan.MATRIX_RUN_SET_ID


def test_gen2_preflight_names_missing_reused_artefacts_and_taken_queues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path, "grid")
    p = plan.partition("grid")
    with pytest.raises(CampaignGateError, match="267 reused Part I artefacts"):
        pipe._require_reused(cfg, p)
    assert (
        pipe._require_reused(_cfg(tmp_path, "ladder"), plan.partition("ladder")) == {}
    )
    monkeypatch.setattr(pipe.exp, "missing_reused", lambda *a, **k: [])
    assert pipe._require_reused(cfg, p) == {"teachers": 129, "controls": 138}
    monkeypatch.setattr(pipe.store, "this_host", lambda: p.node)
    pipe._require_queue_census(cfg, fresh=True)
    taken = "ws160m2-grid-r16000"
    monkeypatch.setattr(pipe.store, "list_queues", lambda root: [taken, "other"])
    with pytest.raises(CampaignGateError, match=taken):
        pipe._require_queue_census(cfg, fresh=True)


def test_gen2_preflight_refuses_a_stale_or_part_one_staging_report(
    tmp_path: Path,
) -> None:
    opts = _opts(tmp_path, "matrix", [_T0])
    staged = tmp_path / "bm" / models.HF_CONVERTED_DIRNAME
    staged.mkdir(parents=True)
    (staged / plan.STAGING_REPORT).write_text(json.dumps({"passed": True}))
    with pytest.raises(CampaignGateError, match="missing or not passed"):
        pipe._require_reports(opts)
    report = staged / plan.GEN2_STAGING_REPORT
    report.write_text(json.dumps({"passed": True, "hub_revisions": {}}))
    with pytest.raises(CampaignGateError, match="is stale"):
        pipe._require_reports(opts)
    needed = pipe._needed_models("matrix", wave=3)
    hub = {models.token(b, s): {} for b, s in needed if models.base(b).source == "hub"}
    converted: dict[str, dict] = {}
    for b, s in needed:
        if models.base(b).source == "local":
            converted.setdefault(b, {})[f"step{s}"] = {"action": "converted"}
    report.write_text(
        json.dumps({"passed": True, "hub_revisions": hub, "converted": converted})
    )
    assert pipe._require_reports(opts) == {"staging_report": str(report)}
    converted["pile-main"]["step143000"] = {"action": "would_convert"}
    report.write_text(
        json.dumps({"passed": True, "hub_revisions": hub, "converted": converted})
    )
    with pytest.raises(CampaignGateError, match="pile-main@step143000"):
        pipe._require_reports(opts)


def test_gen2_writes_every_yaml_to_the_shared_run_set_dir(tmp_path: Path) -> None:
    cfg, opts = _cfg(tmp_path, "grid"), _opts(tmp_path, "grid", [_T0])
    shared = pipe._shared_config_dir(cfg, "grid")
    assert shared == cfg.run_set_root / plan.SHARED_GENERATED_DIRNAME
    n_train, n_verify = pipe._write_configs(cfg, opts, wave=3)
    assert len(list(shared.glob("*.yaml"))) == n_train + n_verify == 1016
    assert len(list(cfg.generated_dir.glob("*.yaml"))) == n_train + n_verify
    lmeval = next(shared.glob("gkd_ws160m_lmeval_*.yaml"))
    doc = yaml.safe_load(lmeval.read_text())
    assert doc["benchmarks"] == {
        "enabled": True,
        "tasks": list(plan.LMEVAL_TASKS),
        "batch_size": plan.LMEVAL_BATCH_SIZE,
        "lora": {},
    }
    assert Path(doc["inherit_grid"]).parent == shared
    assert doc["queue"].startswith("ws160m2-grid-lmeval-")
    _, _, by_q = pipe._bundle(cfg, opts, wave=3)
    for queue_jobs in by_q.values():
        pipe._require_overflow_configs(cfg, "q", "soar-subliminal-2", queue_jobs)


def test_lmeval_queues_requeue_to_three_attempts_and_never_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, opts = (
        _cfg(tmp_path, "matrix", auto_requeue=2),
        _opts(tmp_path, "matrix", [_T0]),
    )
    caps: dict[str, int] = {}

    def requeue(c, name):
        caps[name] = c.auto_requeue
        raise RuntimeError("over cap")

    monkeypatch.setattr(pipe.store, "queue_exists", lambda root, n: True)
    monkeypatch.setattr(pipe.store, "queue_complete", lambda root, n: True)
    monkeypatch.setattr(pipe.driver_io, "requeue_failed_under_cap", requeue)
    monkeypatch.setattr(pipe, "_failed_jobkeys", lambda c, n: ["a", "b", "c", "d"])
    monkeypatch.setattr(
        pipe.store, "queue_status", lambda root, n: {"failed": 4, "total": 40}
    )
    holes: list[dict] = []
    lm, ev = "ws160m2-matrix-lmeval-m2", "ws160m2-matrix-evals-m2"
    assert pipe._queue_ready(cfg, lm, opts, holes)
    assert caps[lm] == plan.LMEVAL_ATTEMPT_CAP == 3
    assert [h["queue"] for h in holes] == [lm]
    with pytest.raises(CampaignGateError, match="retry-exhausted"):
        pipe._queue_ready(cfg, ev, opts, holes)
    assert caps[ev] == 2
    lmeval_defaults = pipe._lmeval_defaults
    monkeypatch.setattr(
        pipe.driver_io,
        "queue_job_defaults",
        lambda c: {"cwd": "/pin", "env": {"A": "1"}},
    )
    got = lmeval_defaults(cfg, opts)
    assert got["cwd"] == "/pin" and got["env"]["A"] == "1"
    assert got["env"]["HF_HUB_CACHE"] == str(tmp_path / "bm" / "hub")


def test_gen2_archived_stage_queue_is_resubmitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _Fake(monkeypatch, "grid")
    clock = [_T0]
    cfg, opts = _cfg(tmp_path, "grid"), _opts(tmp_path, "grid", clock)
    run(cfg, opts)
    gone = "ws160m2-grid-r4000"
    assert gone in fake.names()
    fake.submitted = [(n, d) for n, d in fake.submitted if n != gone]
    r = run(cfg, opts)
    assert f"restored_queues({gone})" in r.fired and gone in fake.names()
    state = json.loads(cfg.state_path.read_text())
    assert state["restores"][-1]["phase"] == "stage:r4000"


def test_queue_plan_lists_gen2_queues_in_priority_order() -> None:
    summary = pipe.queue_plan("grid")
    assert summary["run_set_id"] == plan.GRID_RUN_SET_ID
    assert summary["n_jobs"] == 5188 and summary["n_yamls"] == 1016
    names = list(summary["queues"])
    order = list(plan.queue_priorities(plan.partition("grid")))
    assert names == [q for q in order if q in summary["queues"]]
    assert "ws160m2-grid-teachers-fallback" not in names
