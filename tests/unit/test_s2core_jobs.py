"""S2 core orchestration: queues, dealing, commands, SM-6 counts, driver stage machine."""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pytest
import typer

import scripts.sweeps.sweep_s2core as sweep
from pipelines import s2core_campaign_pipeline as campaign
from pipelines import s2core_results_pipeline as results
from subliminal_learning.campaign import s2core_cells as cells
from subliminal_learning.campaign import s2core_jobs as jobs
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.judging import s2_workitems as wi

REPO = Path(__file__).resolve().parents[2]


# ---- queues and routing --------------------------------------------------------------


def test_every_queue_is_pinned_to_a_host_of_its_channel():
    for channel, prios in jobs.QUEUE_PRIORITIES.items():
        for name in prios:
            host = jobs.queue_host(name)
            assert name.startswith(f"s2core-{channel}-")
            assert name.endswith(f"-{plan.HOST_TAG[host]}")
            assert host in plan.CHANNEL_HOSTS[channel]
    hard_p0 = [n for n in jobs.QUEUE_PRIORITIES[plan.HARD] if "-p0-" in n]
    gen42 = [n for n in hard_p0 if "-p0-gen42-" in n]
    gen43 = [n for n in hard_p0 if "-p0-gen43-" in n]
    assert {jobs.queue_host(n) for n in gen42} == {plan.CLUSTER5}
    assert {jobs.queue_host(n) for n in gen43} == {plan.CLUSTER5, plan.GONCALO}
    rest = set(hard_p0) - set(gen42) - set(gen43)
    assert {jobs.queue_host(n) for n in rest} == {plan.CLUSTER5}


def test_number_generation_keeps_seed_42_on_cluster5():
    queues = jobs.hard_gen_queues()
    specs = Counter()
    per_host = Counter()
    for q in queues:
        for j in q.jobs:
            args = _parse(j.cmd)
            specs[(int(args["prompt-seed"]), args["source"])] += 1
            per_host[(int(args["prompt-seed"]), q.host)] += 1
            assert (j.mem_mib, j.slots) == (
                plan.FOOTPRINTS["numbers_gen"].mem_mib,
                plan.FOOTPRINTS["numbers_gen"].slots,
            )
    assert specs == {
        (seed, s.key): 1 for seed in plan.PROMPT_SEEDS for s in plan.NUMBER_SOURCES
    }
    assert per_host == {
        (42, plan.CLUSTER5): 5,
        (43, plan.CLUSTER5): 3,
        (43, plan.GONCALO): 2,
    }
    prios = jobs.QUEUE_PRIORITIES[plan.HARD]
    g42 = jobs.queue_name(plan.HARD, jobs.P0, jobs.GEN42, plan.CLUSTER5)
    g43 = jobs.queue_name(plan.HARD, jobs.P0, jobs.GEN43, plan.CLUSTER5)
    assert prios[g42] > prios[g43]
    build42 = jobs.hard_build_queue((42,))
    build4243 = jobs.hard_build_queue((42, 43))
    assert build42.host == build4243.host == plan.CLUSTER5
    assert build42.depends_on == (g42,)
    assert set(build4243.depends_on) == {
        g43,
        jobs.queue_name(plan.HARD, jobs.P0, jobs.GEN43, plan.GONCALO),
    }


def test_pool_queues_cover_every_host_queue_by_priority():
    for host in plan.HOST_TAG:
        channel = plan.host_channel(host)
        names = jobs.pool_queues(host)
        assert set(names) == {
            n for n in jobs.QUEUE_PRIORITIES[channel] if jobs.queue_host(n) == host
        }
        prios = [jobs.QUEUE_PRIORITIES[channel][n] for n in names]
        assert prios == sorted(prios, reverse=True)


def test_gpu_slots_follow_host_gpus_in_plan_order():
    soft = jobs.gpu_slots(plan.SOFT)
    assert Counter(soft) == {plan.SOAR: 6, plan.CLUSTER3: 8, plan.CLUSTER4: 8}
    assert soft[:3] == (plan.SOAR, plan.CLUSTER3, plan.CLUSTER4)
    assert Counter(jobs.gpu_slots(plan.HARD)) == {plan.CLUSTER5: 8, plan.GONCALO: 8}


def test_policies_exclude_the_judge_lane_and_point_at_fast_caches():
    assert plan.gpu_policy_json(plan.SOAR)["gpus"] == [0, 1, 2, 3, 4, 5]
    for host in plan.HOST_TAG:
        env = plan.gpu_policy_json(host)["env"]
        want = plan.HARD_ROOT if host == plan.CLUSTER5 else plan.SOFT_ROOT
        assert env["HF_HOME"] == str(want / "hf")
        assert env["HF_HUB_OFFLINE"] == "1"


# ---- commands ------------------------------------------------------------------------


@pytest.mark.parametrize("lr", [*plan.SOFT_LADDER, *plan.HARD_LADDER])
def test_learning_rates_round_trip_through_the_command_line(lr):
    assert float(jobs._lr(lr)) == lr


def _parse(cmd: str) -> dict[str, str]:
    tokens = shlex.split(cmd)
    return {
        tokens[i][2:]: tokens[i + 1]
        for i in range(len(tokens) - 1)
        if tokens[i].startswith("--")
    }


def test_cell_commands_rebuild_their_cells():
    for cell in cells.soft_stage_a()[:5] + cells.soft_stage_b(2e-4)[:5]:
        args = _parse(jobs.train_job(cell, "a", plan.SOAR).cmd)
        rebuilt = sweep._soft_cell(
            args["arm"],
            args["carrier"],
            float(args["lr"]),
            int(args["budget"]),
            int(args["gseed"]),
        )
        assert rebuilt == cell
    for cell in cells.hard_stage_a()[:5] + cells.hard_stage_b(1e-4)[:5]:
        args = _parse(jobs.eval_job(cell, "a", plan.SELECT, plan.CLUSTER5).cmd)
        rebuilt = sweep._hard_cell(
            args["arm"],
            args["data"],
            float(args["lr"]),
            int(args["steps"]),
            int(args["seed"]),
        )
        assert rebuilt == cell


def test_cli_rejects_cells_off_the_plan():
    with pytest.raises(typer.BadParameter):
        sweep._soft_cell(plan.T1, plan.PILE, 3e-4, 1, 0)
    with pytest.raises(typer.BadParameter):
        sweep._soft_cell(plan.T1, plan.PILE, 4e-4, 1, 7)
    with pytest.raises(typer.BadParameter):
        sweep._hard_cell(plan.T1, plan.CORE, 1e-4, 1234, 1)
    with pytest.raises(typer.BadParameter):
        sweep._hard_cell(plan.T1, plan.CORE, 1.2e-4, 25_000, 1)


def test_footprints_match_job_kinds():
    soft = cells.soft_stage_a()[0]
    hard = cells.hard_stage_a()[0]
    fp = plan.FOOTPRINTS
    cases = [
        (jobs.train_job(soft, "a", plan.SOAR), fp["soft_train"]),
        (jobs.eval_job(soft, "a", plan.SELECT, plan.SOAR), fp["soft_eval"]),
        (jobs.train_job(hard, "a", plan.CLUSTER5), fp["hard_train"]),
        (jobs.eval_job(hard, "a", plan.SELECT, plan.CLUSTER5), fp["hard_eval_select"]),
        (jobs.eval_job(hard, "c", plan.FULL, plan.GONCALO), fp["hard_eval_full"]),
    ]
    for job, want in cases:
        assert (job.mem_mib, job.slots) == (want.mem_mib, want.slots)


def test_forbidden_tokens_are_refused():
    with pytest.raises(ValueError):
        jobs._job("x", "uv run python -m x --force", "cpu", plan.SOAR)
    with pytest.raises(ValueError):
        jobs._job("x", "cat /mnt/cold-1/foo", "cpu", plan.CLUSTER5)
    jobs._job("x", "cat /mnt/cold-1/foo", "cpu", plan.SOAR)


def test_launcher_passes_the_fleet_guard_grep():
    text = (REPO / "scripts" / "launch_s2core_campaign.sh").read_text()
    assert "SL_JOBQ_LOCK_PREFIX" in text
    assert "--lock-prefix" in text


# ---- SM-6: enumeration counts (plan §7.2 / §7.3) -------------------------------------


def test_sm6_soft_counts():
    d = campaign.enumerate_plan(plan.SOFT)
    assert d["cells"] == {"a": 30, "b": 24, "c": 30, "full": 15}
    kinds = Counter()
    for name, q in d["queues"].items():
        kinds[tuple(name.split("-")[2:4])] += q["n_jobs"]
    assert kinds[("a", "train")] == kinds[("a", "select")] == 30
    assert kinds[("b", "train")] == kinds[("b", "select")] == 24
    assert kinds[("c", "train")] == kinds[("c", "select")] == 30
    assert kinds[("c", "fullprior")] == 6 and kinds[("c", "full")] == 9
    assert kinds[("p0", "base")] == len(plan.BASE_MODELS)
    assert kinds[("p0", "lexicon")] == 3
    for c in plan.CARRIERS:
        assert kinds[("p0", f"leak_{c}")] == kinds[("p0", f"assemble_{c}")] == 1


def test_sm6_hard_counts():
    d = campaign.enumerate_plan(plan.HARD)
    assert d["cells"] == {"a": 42, "b": 36, "c": 27, "full": 15}
    kinds = Counter()
    for name, q in d["queues"].items():
        kinds[tuple(name.split("-")[2:4])] += q["n_jobs"]
    assert kinds[("p0", "gen42")] == kinds[("p0", "gen43")] == 5
    assert kinds[("p0", "build42")] == kinds[("p0", "build4243")] == 1
    assert kinds[("p0", "eq")] == 9


def test_select_queues_depend_on_their_hosts_train_queue():
    a = cells.soft_stage_a()
    trains = {q.host: q.name for q in jobs.stage_train_queues(plan.SOFT, "a", a)}
    for q in jobs.stage_select_queues(plan.SOFT, "a", a):
        assert q.depends_on == (trains[q.host],)
        assert {j.key.split("|", 2)[2] for j in q.jobs} == {
            j.key.split("|", 2)[2]
            for t in jobs.stage_train_queues(plan.SOFT, "a", a)
            if t.host == q.host
            for j in t.jobs
        }


# ---- driver helpers ------------------------------------------------------------------


def test_generation_rows_keep_generation_row_ids(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    gens = [
        {
            "row_id": f"e6:{q}:{s}",
            "question_index": q,
            "question": f"q{q}",
            "sample": s,
            "answer": "a",
        }
        for q in range(2)
        for s in range(3)
    ]
    (eval_dir / layout.E6_GENERATIONS).write_text(
        "".join(json.dumps(g) + "\n" for g in gens)
    )
    monkeypatch.setattr(campaign, "_model_dir", lambda target: tmp_path)
    rows, keys = campaign.generation_rows(plan.BASELINE, plan.EM_ALIGN)
    assert [r["row_id"] for r in rows] == [g["row_id"] for g in gens]
    assert all(set(r) == {"row_id", "question", "answer"} for r in rows)
    assert {k["model"] for k in keys} == {"base-baseline"}


def test_audit_rows_are_seeded_and_carry_the_screen_flag(tmp_path):
    cell = cells.soft_final_cells(2e-4, 1)[0]
    recs = [
        {
            "step": i // 16,
            "batch_idx": i % 16,
            "completion": f"c{i}",
            "flagged": i % 7 == 0,
        }
        for i in range(100)
    ]
    (tmp_path / "rollouts_sample.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in recs)
    )
    rows1, keys1 = campaign.audit_rows(cell, 20, run_dir=tmp_path)
    rows2, _ = campaign.audit_rows(cell, 20, run_dir=tmp_path)
    assert rows1 == rows2 and len(rows1) == 20
    for row, key in zip(rows1, keys1, strict=True):
        line = int(row["row_id"][1:])
        assert row["text"] == f"c{line}" and key["flagged"] == (line % 7 == 0)


def test_audit_covers_every_final_arm_tier_with_2000_rows():
    audit = campaign.audit_cells(2e-4, 1)
    tiers = Counter((c.carrier, c.arm) for c in audit)
    assert tiers == {
        (plan.PILE, plan.T1): 5,
        (plan.PILE, plan.T2): 5,
        (plan.PILE, plan.P0): 5,
        (plan.MID, plan.T1): 3,
        (plan.MID, plan.T2): 3,
        (plan.MID, plan.P0): 3,
        (plan.PILE, plan.S1): 3,
        (plan.PILE, plan.S2): 3,
        (plan.AI, plan.T1): 3,
        (plan.AI, plan.T2): 3,
    }
    n_rows = campaign.audit_rows_per_cell(audit)
    for tier, n_cells in tiers.items():
        per_cell = {n_rows[c] for c in audit if (c.carrier, c.arm) == tier}
        assert per_cell == {-(-plan.AUDIT_ROWS_PER_GROUP // n_cells)}
        assert (
            plan.AUDIT_ROWS_PER_GROUP
            <= n_cells * per_cell.pop()
            < (plan.AUDIT_ROWS_PER_GROUP + n_cells)
        )
    assert {c.lr for c in audit} == {2e-4} and {c.budget for c in audit} == {1}


def test_audit_skips_dropped_carriers():
    audit = campaign.audit_cells(2e-4, 1, dropped=[plan.AI])
    assert plan.AI not in {c.carrier for c in audit}
    assert len(audit) == 36 - 6


def test_extra_pin_paths_cover_every_s2core_file():
    tracked = {
        str(p.relative_to(REPO))
        for pattern in (
            "pipelines/s2core_*.py",
            "scripts/**/*s2core*",
            "src/subliminal_learning/campaign/s2core_*.py",
            "src/subliminal_learning/judging/*.py",
            "tools/s2_judge/*.py",
            "tools/s2_judge/*.sh",
            "tools/judge_v2/pyproject.toml",
            "tools/judge_v2/uv.lock",
            "data/s2core/*",
            "data/ai_discourse_lexicon_v2.json",
        )
        for p in REPO.glob(pattern)
        if p.is_file() and p.suffix in (".py", ".sh", ".json", ".toml", ".lock")
    }
    assert tracked <= set(campaign.EXTRA_PIN_PATHS)
    assert all((REPO / rel).exists() for rel in campaign.EXTRA_PIN_PATHS)


# ---- driver stage machine (tmp queue root; pins, gates and results faked) ------------


def _complete(cfg, rc: int = 0) -> None:
    for name in store.list_queues(cfg.queue_root):
        for job in store.load_jobs(cfg.queue_root, name):
            if not store.has_result(cfg.queue_root, name, job.jobkey):
                store.record_result(
                    cfg.queue_root,
                    name,
                    job,
                    rc=rc,
                    node="test",
                    gpu=0,
                    start_utc="t",
                    end_utc="t",
                    log="",
                    attempt=1,
                )


def _jobs_in(cfg, stage: str, kind: str) -> int:
    return sum(
        len(store.load_jobs(cfg.queue_root, n))
        for n in store.list_queues(cfg.queue_root)
        if f"-{stage}-{kind}-" in n
    )


@pytest.fixture
def soft_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign, "node_pin_defaults", lambda cfg, host: {"cwd": str(tmp_path)}
    )
    monkeypatch.setattr(campaign, "_eval_gate_open", lambda: True)
    monkeypatch.setattr(campaign, "_stack_current", lambda: True)
    monkeypatch.setattr(wi, "parked_judges", lambda: set())
    monkeypatch.setattr(results, "build", lambda channel, final: {})
    return campaign.driver_config(
        plan.SOFT,
        output_root=tmp_path / "out",
        queue_root=tmp_path / "q",
        repo_root=tmp_path,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
    )


def test_soft_stages_a_b_c_follow_the_selections(soft_cfg, monkeypatch):
    cfg = soft_cfg
    fired: list[str] = []
    state = {"transitions": {"stage_a": {"at": "t"}}}
    campaign._submit_all(
        cfg, jobs.stage_train_queues(plan.SOFT, "a", cells.soft_stage_a())
    )
    waiting: list[str] = []
    campaign._advance_stages(cfg, state, fired.append, waiting)
    assert "stage_a_eval" in fired and _jobs_in(cfg, "a", "select") == 30
    assert "select_a" not in state["transitions"]

    calls = {}
    monkeypatch.setattr(
        results,
        "select_stage_a",
        lambda channel, holes=(): calls.setdefault("a", {"status": "ok", "r_a": 2e-4}),
    )
    _complete(cfg)
    campaign._advance_stages(cfg, state, fired.append, waiting)
    assert state["transitions"]["select_a"]["r_a"] == 2e-4
    assert _jobs_in(cfg, "b", "train") == _jobs_in(cfg, "b", "select") == 24

    monkeypatch.setattr(
        results,
        "select_final",
        lambda channel, holes=(): {"status": "ok", "lr": 2e-4, "budget": 2},
    )
    state["transitions"]["pool_mid"] = {"pass": True}
    state["transitions"]["pool_ai"] = {"pass": False}
    _complete(cfg)
    campaign._advance_stages(cfg, state, fired.append, waiting)
    assert state["transitions"]["stage_c"]["dropped_carriers"] == [plan.AI]
    assert _jobs_in(cfg, "c", "train") == 24
    assert _jobs_in(cfg, "c", "fullprior") == 6 and _jobs_in(cfg, "c", "full") == 9
    c_keys = {
        j.key.split("|", 2)[2]
        for n in store.list_queues(cfg.queue_root)
        if "-c-train-" in n
        for j in store.load_jobs(cfg.queue_root, n)
    }
    assert not any(f"|{plan.AI}|" in k for k in c_keys)


def test_failed_jobs_over_the_cap_become_hole_cells(soft_cfg):
    cfg = soft_cfg
    a = cells.soft_stage_a()
    campaign._submit_all(cfg, jobs.stage_train_queues(plan.SOFT, "a", a))
    name = jobs.stage_train_queues(plan.SOFT, "a", a)[0].name
    job = store.load_jobs(cfg.queue_root, name)[0]
    store.record_result(
        cfg.queue_root,
        name,
        job,
        rc=1,
        node="t",
        gpu=0,
        start_utc="t",
        end_utc="t",
        log="",
        attempt=plan.AUTO_REQUEUE_ATTEMPTS,
    )
    state = {"transitions": {}}
    campaign._requeue_and_holes(cfg, state)
    assert name in state["holes"]
    assert campaign.hole_cells(cfg, state) == [job.key.split("|", 2)[2]]


def test_failed_jobs_under_the_cap_are_requeued(soft_cfg):
    cfg = soft_cfg
    a = cells.soft_stage_a()
    campaign._submit_all(cfg, jobs.stage_train_queues(plan.SOFT, "a", a))
    name = jobs.stage_train_queues(plan.SOFT, "a", a)[0].name
    job = store.load_jobs(cfg.queue_root, name)[0]
    store.record_result(
        cfg.queue_root,
        name,
        job,
        rc=1,
        node="t",
        gpu=0,
        start_utc="t",
        end_utc="t",
        log="",
        attempt=1,
    )
    state = {"transitions": {}}
    campaign._requeue_and_holes(cfg, state)
    assert not state["holes"]
    assert store.job_state(cfg.queue_root, name, job) == store.PENDING


def test_dropped_carriers_wait_for_both_side_pools():
    assert (
        campaign._dropped_carriers({"transitions": {"pool_mid": {"pass": True}}})
        is None
    )
    both = {"transitions": {"pool_mid": {"pass": False}, "pool_ai": {"pass": True}}}
    assert campaign._dropped_carriers(both) == [plan.MID]


@pytest.fixture
def hard_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign, "node_pin_defaults", lambda cfg, host: {"cwd": str(tmp_path)}
    )
    monkeypatch.setattr(campaign, "_stack_current", lambda: True)
    monkeypatch.setattr(wi, "parked_judges", lambda: set())
    return campaign.driver_config(
        plan.HARD,
        output_root=tmp_path / "out",
        queue_root=tmp_path / "q",
        repo_root=tmp_path,
        auto_requeue=plan.AUTO_REQUEUE_ATTEMPTS,
    )


def test_hard_stage_a_waits_for_the_probe_questions(hard_cfg, tmp_path, monkeypatch):
    probe_file = tmp_path / "probe_questions.json"
    monkeypatch.setattr(layout, "PROBE_QUESTION_FILE", probe_file)
    state = {"transitions": {"numbers": {}, "build42": {}, "numbers_ok": {}}}
    waiting: list[str] = []
    campaign._hard_phase0(hard_cfg, state, lambda name: None, waiting)
    assert "stage_a" not in state["transitions"]
    assert any(str(probe_file) in w for w in waiting)
    probe_file.write_text("[]")
    campaign._hard_phase0(hard_cfg, state, lambda name: None, [])
    assert state["transitions"]["stage_a"]["n_cells"] == 42
    assert _jobs_in(hard_cfg, "a", "train") == 42


def test_hard_phase0_submits_both_generation_seeds_and_eq(hard_cfg):
    state = {"transitions": {}}
    campaign._hard_phase0(hard_cfg, state, lambda name: None, [])
    assert _jobs_in(hard_cfg, "p0", "gen42") == 5
    assert _jobs_in(hard_cfg, "p0", "gen43") == 5
    assert _jobs_in(hard_cfg, "p0", "eq") == 9
    assert "build42" not in state["transitions"]


def _queue_exists(cfg, queue) -> bool:
    return store.queue_exists(cfg.queue_root, queue.name)


def test_soft_pools_leak_items_then_pile_first_assembly(
    soft_cfg, tmp_path, monkeypatch
):
    from pipelines import s2core_pool_pipeline as pools
    from subliminal_learning.judging import s2_workitems as wi

    cfg = soft_cfg
    build = {c: tmp_path / "build" / c for c in plan.CARRIERS}
    pool = {c: tmp_path / "pool" / c for c in plan.CARRIERS}
    monkeypatch.setattr(layout, "pool_build_dir", lambda c: build[c])
    monkeypatch.setattr(layout, "pool_dir", lambda c: pool[c])
    monkeypatch.setattr(wi, "item_exists", lambda item_id: True)
    monkeypatch.setattr(campaign, "_items_done", lambda ids: True)
    monkeypatch.setattr(pools, "pool_ready", lambda c: True)
    state = {"transitions": {"phase0": {}}}

    def step():
        campaign._soft_pools(cfg, state, lambda name: None, [])

    step()
    assert not any(_queue_exists(cfg, jobs.soft_leak_queue(c)) for c in plan.CARRIERS)
    state["transitions"]["stack_choice"] = {}
    step()
    assert not any(_queue_exists(cfg, jobs.soft_leak_queue(c)) for c in plan.CARRIERS)
    for c in plan.CARRIERS:
        build[c].mkdir(parents=True)
        (build[c] / pools.LEXICON_PASS).write_text("{}")
    step()
    assert all(_queue_exists(cfg, jobs.soft_leak_queue(c)) for c in plan.CARRIERS)
    _complete(cfg)
    step()
    assert _queue_exists(cfg, jobs.soft_assemble_queue(plan.PILE))
    assert not _queue_exists(cfg, jobs.soft_assemble_queue(plan.MID))
    assert not _queue_exists(cfg, jobs.soft_assemble_queue(plan.AI))
    _complete(cfg)
    pool[plan.PILE].mkdir(parents=True)
    (pool[plan.PILE] / "pool_gate.json").write_text(json.dumps({"pass": True}))
    step()
    assert state["transitions"]["pool_pile"]["pass"] is True
    assert _queue_exists(cfg, jobs.soft_assemble_queue(plan.MID))
    assert _queue_exists(cfg, jobs.soft_assemble_queue(plan.AI))
    _complete(cfg)
    pool[plan.MID].mkdir(parents=True)
    (pool[plan.MID] / "pool_gate.json").write_text(json.dumps({"pass": True}))
    pool[plan.AI].mkdir(parents=True)
    (pool[plan.AI] / "pool_gate.json").write_text(json.dumps({"pass": False}))
    step()
    assert state["transitions"]["pool_mid"]["pass"] is True
    assert state["transitions"]["pool_ai"]["pass"] is False
    assert campaign._dropped_carriers(state) == [plan.AI]


def test_failed_pile_pool_gate_halts_soft(soft_cfg, tmp_path, monkeypatch):
    from pipelines import s2core_pool_pipeline as pools
    from subliminal_learning.judging import s2_workitems as wi

    cfg = soft_cfg
    monkeypatch.setattr(layout, "pool_build_dir", lambda c: tmp_path / "build" / c)
    monkeypatch.setattr(layout, "pool_dir", lambda c: tmp_path / "pool" / c)
    monkeypatch.setattr(wi, "item_exists", lambda item_id: True)
    monkeypatch.setattr(campaign, "_items_done", lambda ids: True)
    monkeypatch.setattr(pools, "pool_ready", lambda c: False)
    (tmp_path / "build" / plan.PILE).mkdir(parents=True)
    (tmp_path / "build" / plan.PILE / pools.LEXICON_PASS).write_text("{}")
    (tmp_path / "pool" / plan.PILE).mkdir(parents=True)
    (tmp_path / "pool" / plan.PILE / "pool_gate.json").write_text('{"pass": false}')
    state = {"transitions": {"stack_choice": {}}}
    for _ in range(2):
        campaign._soft_pools(cfg, state, lambda name: None, [])
        _complete(cfg)
    with pytest.raises(campaign.CampaignGateError):
        campaign._soft_pools(cfg, state, lambda name: None, [])


# ---- preflight, prereg record, re-pin command ----------------------------------------


def _prereg(tmp_path: Path) -> Path:
    path = tmp_path / plan.PREREG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# prereg\n\n{campaign.PREREG_MARKER} (`s2core`)\n")
    return path


def test_prereg_record_hashes_the_local_prereg(tmp_path):
    path = _prereg(tmp_path)
    rec = campaign.prereg_record(tmp_path)
    assert rec["prereg_path"] == str(path)
    assert rec["prereg_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("# prereg without the S2 core section\n")
    with pytest.raises(campaign.CampaignGateError):
        campaign.prereg_record(tmp_path)


def test_missing_policy_is_a_blocker_and_a_differing_one_a_gate(
    hard_cfg, tmp_path, monkeypatch
):
    from pipelines import s2core_staging_pipeline as staging

    _prereg(tmp_path)
    paths_yaml = tmp_path / "paths_hard.yaml"
    monkeypatch.setattr(plan, "queue_root", lambda channel: hard_cfg.queue_root)
    monkeypatch.setattr(plan, "paths_config_path", lambda channel: paths_yaml)
    paths_yaml.write_text(staging.paths_yaml_text(plan.HARD))
    monkeypatch.setattr(staging, "latest_complete_report", lambda channel: {"at": "t"})
    usage = shutil.disk_usage(tmp_path)._replace(free=10**15)
    monkeypatch.setattr(campaign.shutil, "disk_usage", lambda path: usage)
    state = {"transitions": {}}
    assert campaign._preflight(hard_cfg, state) is False
    assert "preflight" in state["blockers"]
    hard_cfg.queue_root.mkdir(parents=True, exist_ok=True)
    for host in plan.CHANNEL_HOSTS[plan.HARD]:
        path = hard_cfg.queue_root / f"gpu_policy.{host}.json"
        path.write_text(json.dumps(plan.gpu_policy_json(host)))
    assert campaign._preflight(hard_cfg, state) is True
    assert "preflight" not in state["blockers"]
    assert len(state["transitions"]["preflight"]["prereg_sha256"]) == 64
    bad = dict(plan.gpu_policy_json(plan.GONCALO), cap_per_gpu=7)
    (hard_cfg.queue_root / f"gpu_policy.{plan.GONCALO}.json").write_text(
        json.dumps(bad)
    )
    with pytest.raises(campaign.CampaignGateError):
        campaign._preflight(hard_cfg, {"transitions": {}})


def test_reconcile_run_set_command_resolves_the_s2core_card(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from scripts.reconcile_run_set import app as reconcile
    from subliminal_learning.training.run_set import build_run_set_card
    from subliminal_learning.utils.run_io import read_run_set_card, write_run_set_card

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(f"output_root: {tmp_path / 'out'}\n")
    for channel in plan.CHANNELS:
        run_set_id = plan.RUN_SET_IDS[channel]
        root = tmp_path / "out" / plan.RUN_SET_SUBDIR / run_set_id
        card = build_run_set_card(
            run_set_id=run_set_id,
            run_type="test",
            purpose="test",
            date="2026-10-12",
            owner=plan.OWNER,
            git_sha=head,
            sweep_yaml="x",
            resolved_config={},
            grid_axes={},
            model_size="6.9b",
        )
        write_run_set_card(root, card)
        monkeypatch.chdir(REPO)
        args = [
            "--workflow", plan.RUN_SET_SUBDIR, "--run-set-id", run_set_id,
            "--paths-config", str(paths_yaml), "--new-git-sha", head,
            "--reason", "test", "--dry-run",
        ]  # fmt: skip
        result = CliRunner().invoke(reconcile, args)
        assert result.exit_code == 0, result.output
        assert read_run_set_card(root)["git_sha"] == head
    missing = CliRunner().invoke(
        reconcile,
        ["--workflow", plan.RUN_SET_SUBDIR, "--run-set-id", "nope",
         "--paths-config", str(paths_yaml), "--new-git-sha", head,
         "--reason", "test", "--dry-run"],
    )  # fmt: skip
    assert missing.exit_code != 0


def _hole_one_train_job(cfg) -> str:
    a = cells.hard_stage_a()
    campaign._submit_all(cfg, jobs.stage_train_queues(plan.HARD, "a", a))
    name = jobs.stage_train_queues(plan.HARD, "a", a)[0].name
    bad = store.load_jobs(cfg.queue_root, name)[0]
    store.record_result(
        cfg.queue_root, name, bad, rc=1, node="t", gpu=0, start_utc="t", end_utc="t",
        log="", attempt=plan.AUTO_REQUEUE_ATTEMPTS,
    )  # fmt: skip
    _complete(cfg)
    return bad.key.split("|", 2)[2]


def test_finalize_fires_with_a_labelled_hole(hard_cfg, monkeypatch):
    cfg = hard_cfg
    hole = _hole_one_train_job(cfg)
    builds = []
    monkeypatch.setattr(
        results, "build", lambda channel, final: builds.append(final) or {}
    )
    monkeypatch.setattr(campaign, "_other_done", lambda channel: False)
    state = {"transitions": {"final_items": {"items": []}}}
    campaign._requeue_and_holes(cfg, state)
    assert state["holes"]
    fired: list[str] = []
    assert campaign._finalize(cfg, state, fired.append, []) is True
    assert "finalize" in fired and builds == [False, True]
    assert state["transitions"]["finalize"]["holes"] == [hole]
    doc = json.loads((cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).read_text())
    assert doc["holes"] == [hole] and doc["hole_queues"] == sorted(state["holes"])


def test_failed_joint_build_keeps_the_driver_alive(hard_cfg, monkeypatch):
    cfg = hard_cfg
    _hole_one_train_job(cfg)
    monkeypatch.setattr(results, "build", lambda channel, final: {})
    monkeypatch.setattr(campaign, "_other_done", lambda channel: True)

    def boom():
        raise RuntimeError("joint failed")

    monkeypatch.setattr(results, "build_joint", boom)
    state = {"transitions": {"final_items": {"items": []}}}
    campaign._requeue_and_holes(cfg, state)
    assert campaign._finalize(cfg, state, lambda name: None, []) is False
    assert "finalize" in state["transitions"] and "joint" in state["blockers"]
    monkeypatch.setattr(results, "build_joint", lambda: {})
    assert campaign._finalize(cfg, state, lambda name: None, []) is True
    assert "joint" in state["transitions"] and "joint" not in state["blockers"]


# ---- parked judges and judge-code changes ---------------------------------------------


def test_parked_judges_settle_as_labelled_judge_holes(monkeypatch):
    monkeypatch.setattr(campaign, "_item_judges", lambda item_id: ["j1", "j2"])
    monkeypatch.setattr(wi, "item_done", lambda item_id, judge: judge == "j1")
    monkeypatch.setattr(wi, "parked_judges", lambda: set())
    assert campaign.judge_settlement(["it"]) == (False, [])
    monkeypatch.setattr(wi, "parked_judges", lambda: {"j2"})
    assert campaign.judge_settlement(["it"]) == (True, ["it/j2"])
    state: dict = {}
    campaign._record_judge_holes(state, ["it"], ["it/j2"])
    assert list(state["judge_holes"]) == ["it/j2"]
    campaign._record_judge_holes(state, ["other"], [])
    assert list(state["judge_holes"]) == ["it/j2"]
    campaign._record_judge_holes(state, ["it"], [])
    assert state["judge_holes"] == {}


def test_finalize_lists_parked_judge_holes(hard_cfg, monkeypatch):
    cfg = hard_cfg
    _hole_one_train_job(cfg)
    monkeypatch.setattr(results, "build", lambda channel, final: {})
    monkeypatch.setattr(campaign, "_other_done", lambda channel: False)
    monkeypatch.setattr(campaign, "_item_judges", lambda item_id: ["j1", "j2"])
    monkeypatch.setattr(wi, "item_done", lambda item_id, judge: judge == "j1")
    monkeypatch.setattr(wi, "parked_judges", lambda: {"j2"})
    state = {"transitions": {"final_items": {"items": ["e6"]}}}
    campaign._requeue_and_holes(cfg, state)
    assert campaign._finalize(cfg, state, lambda name: None, []) is True
    doc = json.loads((cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).read_text())
    assert doc["judge_holes"] == ["e6/j2"] and "e6/j2" in state["judge_holes"]


def _revalidation_setup(tmp_path, monkeypatch, stack_ok: dict):
    from pipelines import s2core_judge_validation_pipeline as jv

    split = tmp_path / "split.json"
    split.write_text("{}")
    choice_file = tmp_path / "stack_choice.json"
    monkeypatch.setattr(layout, "SPLIT_FILE", split)
    monkeypatch.setattr(layout, "STACK_CHOICE_FILE", choice_file)
    monkeypatch.setattr(campaign, "_lexicon_done", lambda carrier: True)
    monkeypatch.setattr(campaign, "_stack_current", lambda: stack_ok["value"])
    monkeypatch.setattr(jv, "write_validation_items", lambda: ["v1", "v2"])
    monkeypatch.setattr(campaign, "judge_settlement", lambda ids: (True, []))
    return jv, choice_file


def test_judge_code_change_revalidates_and_keeps_pools(soft_cfg, tmp_path, monkeypatch):
    stack_ok = {"value": False}
    jv, choice_file = _revalidation_setup(tmp_path, monkeypatch, stack_ok)
    old = {"choice": {plan.AI_LEAK: "N", plan.EM_ALIGN: "N", plan.SCENARIO_CHOICE: "N"}}

    def evaluate():
        stack_ok["value"] = True
        choice_file.write_text(json.dumps(old))
        return {}

    monkeypatch.setattr(jv, "evaluate_validation", evaluate)
    state = {
        "transitions": {
            "validation_items": {"items": ["v1", "v2"]},
            "stack_choice": {"choice": old},
            "pool_pile": {"pass": True},
        }
    }
    campaign._soft_judge_setup(soft_cfg, state, lambda name: None, [])
    done = state["transitions"]
    assert done["judge_code_changed_1"]["previous_choice"] == old
    assert done["pool_pile"] == {"pass": True}
    assert done["stack_choice"]["choice"] == old
    assert done["validation_items"]["items"] == ["v1", "v2"]
    assert "judge_revalidation" not in state["blockers"]


def test_changed_stack_after_revalidation_with_written_items_halts(
    soft_cfg, tmp_path, monkeypatch
):
    stack_ok = {"value": False}
    jv, choice_file = _revalidation_setup(tmp_path, monkeypatch, stack_ok)
    old = {"choice": {plan.AI_LEAK: "N", plan.EM_ALIGN: "N", plan.SCENARIO_CHOICE: "N"}}
    new = {"choice": {plan.AI_LEAK: "F", plan.EM_ALIGN: "N", plan.SCENARIO_CHOICE: "N"}}

    def evaluate():
        stack_ok["value"] = True
        choice_file.write_text(json.dumps(new))
        return {}

    monkeypatch.setattr(jv, "evaluate_validation", evaluate)
    state = {
        "transitions": {
            "validation_items": {"items": ["v1"]},
            "stack_choice": {"choice": old},
            "phase0_items": {"items": ["e6"]},
        }
    }
    with pytest.raises(campaign.CampaignGateError):
        campaign._soft_judge_setup(soft_cfg, state, lambda name: None, [])
    assert "judge_revalidation" in state["blockers"]
