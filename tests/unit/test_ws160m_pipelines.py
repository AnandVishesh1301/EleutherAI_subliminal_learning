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
        p = plan.partition(partition)
        dummy = {q: [Job(f"{q}-job", "true")] for q in plan.queue_priorities(p)}

        def fake_preflight(cfg, state, opts):
            state["code_sha"] = "0" * 40
            state["wave"] = 1
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
