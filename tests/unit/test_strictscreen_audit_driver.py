"""Strictscreen audit driver: phase walk, gates, relaunch policy (no GPU, no jobq).

Every process/GPU/jobq touch in `campaign.judge_audit_io` is replaced by a fake;
the plan and results subprocesses run their pipelines in-process on a synthetic
run set.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from pipelines import judge_plan_pipeline as jpp
from pipelines import strictscreen_audit_driver as drv
from pipelines import strictscreen_audit_results_pipeline as sar
from subliminal_learning.campaign import judge_audit_io as jio
from subliminal_learning.campaign import strictscreen_audit_plan as sap
from subliminal_learning.campaign import strictscreen_plan as ssp
from subliminal_learning.gkd import judge_layout
from subliminal_learning.jobqueue import store
from tests.fixtures import strictscreen_audit_synth as syn

_TOOL_FILES = (
    "tools/rollout_judge/src/rollout_judge/prompts.py",
    "tools/rollout_judge/src/rollout_judge/schema.py",
    "tools/rollout_judge/src/rollout_judge/vllm_io.py",
    "tools/rollout_judge/uv.lock",
)


class FakeIO:
    """Records launches and answers GPU/jobq/process probes."""

    def __init__(self, tmp: Path, expected: list[Path]) -> None:
        self.tmp = tmp
        self.expected = expected
        self.launched: list[list[str]] = []
        self.blocking: list[list[str]] = []
        self.lock_held = False
        self.orphans: list[dict] = []
        self.running_jobs: list[dict] = []
        self.gpu_used = {g: 0 for g in range(8)}
        self.probe_rc: list[int] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(jio, "run_blocking", self.run_blocking)
        monkeypatch.setattr(jio, "launch_detached", self.launch_detached)
        monkeypatch.setattr(jio, "cluster_lock_held", lambda audit_dir: self.lock_held)
        monkeypatch.setattr(jio, "judge_processes", lambda path: self.orphans)
        monkeypatch.setattr(jio, "gpu_compute_apps", lambda: [])
        monkeypatch.setattr(jio, "gpu_memory_used_mib", lambda: dict(self.gpu_used))
        monkeypatch.setattr(jio, "host_running_jobs", lambda *a: self.running_jobs)
        monkeypatch.setattr(jio, "worker_pool_alive", lambda root: False)
        monkeypatch.setattr(drv.time, "sleep", lambda s: None)

    def run_blocking(self, cmd, *, cwd, env, log_path) -> int:
        self.blocking.append(list(cmd))
        cfg = syn.experiment_cfg(self.tmp)
        exp = cwd / sap.EXPERIMENT_CONFIG_REL
        if "scripts.build_judge_plan" in cmd:
            judge = Path(cmd[cmd.index("--judge-config") + 1])
            doc = json.loads(
                Path(cmd[cmd.index("--expected-run-dirs-file") + 1]).read_text()
            )
            jpp.run(
                cfg,
                exp,
                expected_run_dirs=[Path(p) for p in doc["run_dirs"]],
                judge_config=judge,
            )
            return 0
        if "scripts.build_strictscreen_audit_results" in cmd:
            sar.run(cfg, exp)
            return 0
        if "probe" in cmd:
            if self.probe_rc:
                return self.probe_rc.pop(0)
            self._fake_probe(cmd)
            return 0
        raise AssertionError(f"unexpected command {cmd}")

    def launch_detached(self, cmd, *, cwd, env, log_path, exit_path) -> int:
        self.launched.append(list(cmd))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("")
        return 2**22 + 7

    def _fake_probe(self, cmd: list[str]) -> None:
        task = cmd[cmd.index("--task") + 1]
        n = int(cmd[cmd.index("--n") + 1])
        adir = judge_layout.audit_dir(self.tmp / "audit_root", ssp.AUDIT_ID)
        shard = sorted(judge_layout.queue_shard_dir(adir, task).glob("shard-*.jsonl"))[
            0
        ]
        rows = [json.loads(line) for line in shard.read_text().splitlines()][:n]
        logs = judge_layout.logs_dir(adir)
        (logs / "probe_samples").mkdir(parents=True, exist_ok=True)
        samples = []
        for row in rows:
            answer = "yes" if syn.yes_if_form_in_completion(row) else "no"
            key = (
                "references_any_animal" if task == "pool_open" else "references_target"
            )
            samples.append({"row_id": row["row_id"], key: answer, "parse_ok": True})
        (logs / "probe_samples" / f"{task}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in samples)
        )
        probe = {
            "task": task,
            "rows_judged": len(rows),
            "parse_failure_rate": 0.0,
            "truncation_rate": 0.0,
            "steady_state_rows_per_second": 4.5,
        }
        (logs / "throughput_probe.json").write_text(json.dumps(probe))


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    arms = syn.build_campaign(tmp_path / "output")
    syn.write_pool(tmp_path / "datasets")
    cfg = syn.experiment_cfg(tmp_path)
    ref = jpp.run(cfg, syn.reference_yaml(tmp_path))
    for task in judge_layout.TASKS:
        if judge_layout.queue_shard_dir(ref, task).is_dir():
            syn.write_judgments(ref, task, syn.yes_if_form_in_completion)

    pin = tmp_path / "pin"
    exp = yaml.safe_load(syn.experiment_yaml(tmp_path).read_text())
    exp["driver"]["release_sample_gap_s"] = 0
    exp["driver"]["probe_gates"][0]["n"] = 20
    exp["driver"]["probe_gates"][1]["n"] = 10
    _write_yaml(pin / sap.EXPERIMENT_CONFIG_REL, exp)
    judge = yaml.safe_load(
        syn.judge_yaml(tmp_path, tmp_path / "audit_root").read_text()
    )
    judge["hf_cache_root"] = str(tmp_path / "hf")
    _write_yaml(pin / exp["driver"]["judge_config"], judge)
    paths = {k: str(v) for k, v in vars(cfg.paths).items()}
    _write_yaml(pin / sap.PATHS_CONFIG_REL, paths)
    for rel in _TOOL_FILES:
        (pin / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(syn.REPO / rel, pin / rel)
    snapshot = jio.snapshot_dir(
        tmp_path / "hf", "Qwen/Qwen3-32B-AWQ", exp["driver"]["model_snapshot"]
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot.parent.parent / "refs").mkdir()
    (snapshot.parent.parent / "refs" / "main").write_text(exp["driver"]["model_snapshot"])
    (tmp_path / "queue").mkdir()

    root = tmp_path / "output" / syn.CAMPAIGN_SUBDIR
    expected = [root / r for rels in arms.values() for r in rels]
    fake = FakeIO(tmp_path, expected)
    fake.install(monkeypatch)
    ctx = drv.AuditContext(
        pin_root=pin,
        host=ssp.NODE,
        queue_root=tmp_path / "queue",
        output_root=tmp_path / "output",
        run_set_logs_dir=root / "logs",
        campaign_queue_names=("strictscreen-train-a",),
    )
    return {
        "tmp": tmp_path,
        "ctx": ctx,
        "fake": fake,
        "expected": expected,
        "adir": judge_layout.audit_dir(tmp_path / "audit_root", ssp.AUDIT_ID),
        "stop": store.stop_path(tmp_path / "queue", ssp.NODE),
    }


def _write_yaml(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(doc, sort_keys=False))


def _advance(world: dict, state: dict) -> tuple[list[str], list[str]]:
    fired: list[str] = []
    waiting: list[str] = []
    drv.advance_audit(
        world["ctx"],
        state,
        fired,
        waiting,
        required_run_dirs=world["expected"],
        optional_run_dirs=[],
    )
    return fired, waiting


def _to_judge(world: dict) -> dict:
    state: dict = {"transitions": {}}
    fired, waiting = _advance(world, state)
    assert fired == [
        "audit_completeness",
        "audit_plan",
        "audit_release_gpus",
        "audit_probe",
    ]
    assert waiting and "launched judge-cluster attempt 1" in waiting[-1]
    return state


def test_tool_commands_never_sync_and_name_the_judge_project(tmp_path: Path) -> None:
    cmd = jio.tool_cmd(tmp_path, "probe", "--config", "c.yaml")
    assert cmd[:5] == [
        "uv",
        "run",
        "--no-sync",
        "--project",
        str(tmp_path / "tools/rollout_judge"),
    ]
    assert cmd[5:8] == ["python", "-m", "rollout_judge.cli"]
    assert jio.repo_cmd("scripts.x")[:3] == ["uv", "run", "--no-sync"]


def test_judge_tasks_from_manifest_skips_zero_row_tasks() -> None:
    manifest = {
        "per_task_totals": {
            "pool_open": 5,
            "pool_rejected": 0,
            "decoy": 3,
            "recall_probe": 0,
        }
    }
    assert sap.judge_tasks_from_manifest(manifest) == ("decoy", "pool_open")
    with pytest.raises(ValueError, match="outside the audit"):
        sap.judge_tasks_from_manifest({"per_task_totals": {"pool_rejected": 2}})
    with pytest.raises(ValueError, match="no rows"):
        sap.judge_tasks_from_manifest({"per_task_totals": {"decoy": 0}})


def test_probe_gates() -> None:
    gate = sap.ProbeGate("rollouts_clean", 2000, 0.02, 0.02, 2.0)
    good = {
        "task": "rollouts_clean",
        "rows_judged": 2000,
        "parse_failure_rate": 0.0,
        "truncation_rate": 0.01,
        "steady_state_rows_per_second": 4.5,
    }
    assert sap.probe_verdict(gate, good).passed
    slow = sap.probe_verdict(gate, {**good, "steady_state_rows_per_second": 1.9})
    assert not slow.passed and "steady_state_rows_per_second" in slow.failures[0]
    with pytest.raises(ValueError, match="gate"):
        sap.probe_verdict(gate, {**good, "task": "pool_open"})
    agree = sap.ProbeGate("pool_open", 1000, min_agreement=0.97)
    ref = {f"r{i}": i % 2 == 0 for i in range(100)}
    assert sap.agreement_verdict(agree, dict(ref), ref).passed
    flipped = {k: (not v if i < 4 else v) for i, (k, v) in enumerate(ref.items())}
    assert not sap.agreement_verdict(agree, flipped, ref).passed
    assert not sap.agreement_verdict(agree, {}, ref).passed


def test_task_coverage_is_by_row_id_and_tolerates_a_torn_tail(tmp_path: Path) -> None:
    queue = judge_layout.queue_shard_dir(tmp_path, "decoy") / "shard-0000.jsonl"
    queue.parent.mkdir(parents=True)
    queue.write_text("".join(json.dumps({"row_id": f"r{i}"}) + "\n" for i in range(3)))
    judged = judge_layout.judgments_shard_dir(tmp_path, "decoy") / "shard-0000.jsonl"
    judged.parent.mkdir(parents=True)
    judged.write_text(
        '{"row_id": "r0"}\n{"row_id": "r0"}\n{"row_id": "r2"}\n{"row_id": "r1"'
    )
    cov = jio.task_coverage(tmp_path, "decoy", ["shard-0000.jsonl"])
    assert (cov.n_queue, cov.n_judged, dict(cov.incomplete_shards)) == (
        3,
        2,
        {"shard-0000.jsonl": 1},
    )
    assert jio.judged_rows(tmp_path, "decoy", ["shard-0000.jsonl"]) == 4
    judged.write_text('{"row_id": "r0"}\n{"row_id": "r2"}\n{"row_id": "r1"}\n')
    assert jio.task_coverage(tmp_path, "decoy", ["shard-0000.jsonl"]).complete


def test_failed_worker_gpus_and_exclusion() -> None:
    log = "worker 2 on GPU 5 exited 1; worker 3 on GPU 6 exited -9"
    assert sap.failed_worker_gpus(log) == (5, 6)
    assert sap.attempt_gpus(range(8), {"5": 2, "6": 1}, 2) == (0, 1, 2, 3, 4, 6, 7)


def test_full_walk_release_probe_judge_results_finalize(world: dict) -> None:
    state = _to_judge(world)
    fake = world["fake"]
    assert world["stop"].exists()
    assert state["audit"]["sentinel_preexisting"] is False
    (cmd,) = fake.launched
    tasks = state["audit"]["tasks"]
    assert "pool_rejected" not in tasks
    assert [cmd[i + 1] for i, w in enumerate(cmd) if w == "--task"] == tasks
    assert cmd[cmd.index("--config") + 1] == str(
        world["ctx"].pin_root / "configs/rollout_judge_strictscreen.yaml"
    )
    assert cmd[cmd.index("--gpus") + 1] == "0,1,2,3,4,5,6,7"
    released = json.loads(
        (world["ctx"].run_set_logs_dir / ssp.GPUS_RELEASED_FILE).read_text()
    )
    assert released["host"] == ssp.NODE

    fake.lock_held = True
    fired, waiting = _advance(world, state)
    assert fired == [] and "judge-cluster running" in waiting[0]
    assert len(fake.launched) == 1

    fake.lock_held = False
    for task in tasks:
        syn.write_judgments(world["adir"], task, syn.yes_if_form_in_completion)
    fired, waiting = _advance(world, state)
    assert fired == [
        "audit_judge",
        "audit_return_gpus",
        "audit_results",
        "audit_finalize",
    ]
    assert not world["stop"].exists()
    assert drv.audit_done(state)
    done = json.loads((world["adir"] / ssp.AUDIT_DONE_FILE).read_text())
    assert done["headline"] == state["audit"]["headline"]
    assert (world["ctx"].run_set_logs_dir / ssp.AUDIT_DONE_FILE).is_file()
    status = drv.audit_status(world["ctx"], state)
    assert status["done"] and status["status_file"]["done"]


def test_restart_after_every_phase_is_a_no_op_for_finished_phases(world: dict) -> None:
    state = _to_judge(world)
    fake = world["fake"]
    n_blocking = len(fake.blocking)
    state["transitions"].pop("audit_plan")
    fake.lock_held = True
    _advance(world, state)
    # The re-run plan build is a fingerprint no-op and the judge re-attaches.
    assert len(fake.blocking) == n_blocking + 1
    assert "audit_plan" in state["transitions"]
    assert len(fake.launched) == 1


def test_release_waits_for_running_jobs_and_keeps_a_preexisting_sentinel(
    world: dict,
) -> None:
    world["stop"].touch()
    fake = world["fake"]
    fake.running_jobs = [{"queue": "strictscreen-train-a", "node": ssp.NODE}]
    state: dict = {"transitions": {}}
    fired, waiting = _advance(world, state)
    assert fired == ["audit_completeness", "audit_plan"]
    assert "jobs still running" in waiting[0]
    assert state["audit"]["sentinel_preexisting"] is True
    fake.running_jobs = []
    fake.gpu_used[3] = 20000
    _, waiting = _advance(world, state)
    assert "memory still held" in waiting[0]
    fake.gpu_used[3] = 0
    fake.lock_held = False
    _advance(world, state)
    state["transitions"]["audit_judge"] = {"at": "x"}
    drv._return_gpus(world["ctx"], state)
    assert world["stop"].exists()


def test_orphan_workers_block_and_the_driver_sentinel_is_cleared(world: dict) -> None:
    state = _to_judge(world)
    fake = world["fake"]
    fake.orphans = [{"pid": 1, "cmdline": "python -m rollout_judge.cli judge"}]
    with pytest.raises(ssp.CampaignGateError, match="orphan judge workers"):
        _advance(world, state)
    assert state["audit_blocked"]["phase"] == "audit_judge"
    assert not world["stop"].exists()
    assert (judge_layout.logs_dir(world["adir"]) / sap.AUDIT_BLOCKED_FILE).is_file()
    with pytest.raises(ssp.CampaignGateError, match="audit-unblock"):
        _advance(world, state)
    drv.audit_unblock(state)
    assert "audit_blocked" not in state and state["audit"]["judge"]["attempts"] == []


def test_no_progress_attempt_and_attempt_cap_refuse_relaunch(world: dict) -> None:
    state = _to_judge(world)
    with pytest.raises(ssp.CampaignGateError, match="made no progress"):
        _advance(world, state)
    judge = state["audit"]["judge"]
    assert judge["attempts"][0]["finished_at"]
    refusal = sap.relaunch_refusal([{"judged_before": 0}] * 3, 100, 3)
    assert refusal is not None and "attempt cap" in refusal
    assert sap.relaunch_refusal([{"judged_before": 5}], 6, 3) is None


def test_a_gpu_failing_twice_is_excluded_from_the_next_attempt(world: dict) -> None:
    state = _to_judge(world)
    fake = world["fake"]
    judge = state["audit"]["judge"]
    judge["gpu_failures"] = {"5": 1}
    Path(judge["attempts"][0]["log"]).write_text("worker 5 on GPU 5 exited 1\n")
    adir = world["adir"]
    task = state["audit"]["tasks"][0]
    queue = sorted(judge_layout.queue_shard_dir(adir, task).glob("shard-*.jsonl"))[0]
    first = json.loads(queue.read_text().splitlines()[0])
    out = judge_layout.judgments_shard_dir(adir, task) / queue.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"row_id": first["row_id"]}) + "\n")
    _advance(world, state)
    assert judge["gpu_failures"]["5"] == 2
    assert fake.launched[-1][fake.launched[-1].index("--gpus") + 1] == "0,1,2,3,4,6,7"


def test_probe_crash_retries_then_blocks(world: dict) -> None:
    world["fake"].probe_rc = [1, 1, 1]
    state: dict = {"transitions": {}}
    with pytest.raises(ssp.CampaignGateError, match="crashed 3 times"):
        _advance(world, state)
    assert state["audit_blocked"]["phase"] == "audit_probe"
    assert not world["stop"].exists()


def test_preflight_passes_then_refuses_root_mismatch_and_off_host(world: dict) -> None:
    ctx = world["ctx"]
    exp_path = ctx.pin_root / sap.EXPERIMENT_CONFIG_REL
    committed = yaml.safe_load((syn.REPO / sap.EXPERIMENT_CONFIG_REL).read_text())
    _write_yaml(exp_path, committed)
    dry = drv.AuditContext(**{**vars(ctx), "dry_run": True})
    report = drv.audit_preflight(dry)
    assert report["dry_run"] is True
    assert not world["adir"].exists()
    with pytest.raises(ssp.CampaignGateError, match="not the home node"):
        drv.audit_preflight(
            drv.AuditContext(**{**vars(dry), "host": "soar-subliminal-0"})
        )
    paths = yaml.safe_load((ctx.pin_root / sap.PATHS_CONFIG_REL).read_text())
    paths["audit_root"] = str(world["tmp"] / "other_root")
    (world["tmp"] / "other_root").mkdir()
    _write_yaml(ctx.pin_root / sap.PATHS_CONFIG_REL, paths)
    with pytest.raises(ssp.CampaignGateError, match="judge audit_root"):
        drv.audit_preflight(dry)


def test_preflight_refuses_offline_refs_not_pinned(world: dict) -> None:
    ctx = world["ctx"]
    committed = yaml.safe_load((syn.REPO / sap.EXPERIMENT_CONFIG_REL).read_text())
    _write_yaml(ctx.pin_root / sap.EXPERIMENT_CONFIG_REL, committed)
    refs = (
        world["tmp"] / "hf" / "models--Qwen--Qwen3-32B-AWQ" / "refs" / "main"
    )
    refs.write_text("0" * 40)
    dry = drv.AuditContext(**{**vars(ctx), "dry_run": True})
    with pytest.raises(ssp.CampaignGateError, match="not pinned"):
        drv.audit_preflight(dry)


def test_preflight_refuses_drifted_judge_sources(world: dict) -> None:
    ctx = world["ctx"]
    committed = yaml.safe_load((syn.REPO / sap.EXPERIMENT_CONFIG_REL).read_text())
    _write_yaml(ctx.pin_root / sap.EXPERIMENT_CONFIG_REL, committed)
    prompts = ctx.pin_root / _TOOL_FILES[0]
    prompts.write_text(prompts.read_text() + "\n# drift\n")
    dry = drv.AuditContext(**{**vars(ctx), "dry_run": True})
    with pytest.raises(ssp.CampaignGateError, match="prompts.py: sha256"):
        drv.audit_preflight(dry)


def test_dry_run_advance_writes_nothing(world: dict) -> None:
    dry = drv.AuditContext(**{**vars(world["ctx"]), "dry_run": True})
    state: dict = {"transitions": {}}
    fired: list[str] = []
    waiting: list[str] = []
    drv.advance_audit(
        dry,
        state,
        fired,
        waiting,
        required_run_dirs=world["expected"],
        optional_run_dirs=[],
    )
    assert fired == ["[dry-run] audit next phase: audit_completeness"]
    assert "12 audit leaves ready" in waiting[0]
    assert state["transitions"] == {}
    assert not world["adir"].exists() and not world["stop"].exists()
