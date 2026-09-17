from datetime import datetime, timedelta, timezone

from scripts.queue_status import (
    _active, annotate_queue_order, borrow_family_medians, job_kind, model_arch, completions_per_hour, durations_s,
    effective_mem_mib, eta_s, family_key, fmt_eta, fmt_mem, fmt_work, last_reclaim_utc,
    node_concurrency, node_work, read_yielded, render,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def _res(start_min: float, dur_s: float, rc: int = 0) -> dict:
    a = NOW - timedelta(minutes=start_min)
    return {"rc": rc, "start_utc": a.isoformat(), "end_utc": (a + timedelta(seconds=dur_s)).isoformat()}


def test_durations_drop_failures_and_cache_hits():
    rs = [_res(60, 600), _res(50, 30), _res(40, 900, rc=1)]
    assert durations_s(rs, min_job_s=120) == [600.0]


def test_completions_per_hour_uses_window():
    rs = [_res(30, 600), _res(200, 600)]  # second ended ~3h10m ago
    assert completions_per_hour(rs, now=NOW, window_h=3.0) == 1 / 3.0
    assert completions_per_hour([], now=NOW, window_h=3.0) is None


def test_eta_models():
    wave, thr = eta_s(pending=10, running=2, median=600.0, rate_per_h=12.0, running_ages_s=[300.0])
    assert wave == 300.0 + 600.0 * 10 / 2
    assert thr == 12 / 12.0 * 3600.0
    assert eta_s(pending=5, running=0, median=600.0, rate_per_h=None) == (None, None)
    assert fmt_eta(None, None, running=0, pending=5) == "blocked"
    assert fmt_eta(None, None, running=0, pending=0) == "done"


def test_durations_keep_consistently_short_queue():
    rs = [_res(60 - i, 84 + i * 5) for i in range(10)]  # 1.4-2.2 min, all real
    assert len(durations_s(rs, min_job_s=120)) == 10


def test_durations_drop_zero_second_cache_hits_among_long_jobs():
    rs = [_res(100, 0.5)] * 6 + [_res(90, 3000 + i) for i in range(5)]
    assert sorted(durations_s(rs, min_job_s=120)) == [3000.0 + i for i in range(5)]


def test_durations_ignore_killed_retries():
    rs = [_res(60, 600), _res(50, 20, rc=-9), _res(40, 5, rc=137), _res(30, 610)]
    assert durations_s(rs, min_job_s=120) == [600.0, 610.0]


def test_family_key_and_borrowing():
    assert family_key("pig-seed2seed-fs3-fs1-core56") == family_key("pig-seed2seed-fs2-orig-core56")
    assert family_key("pig-seed2seed-ds1-ws1-core56") == "pig-seed2seed-*-*-core56"
    assert family_key("ckpt-x-d-step95000") == family_key("ckpt-x-d-step100000")
    assert family_key("cat-seed2seed-fs3-fs1-core56") != family_key("pig-seed2seed-fs3-fs1-core56")
    assert family_key("pig-seed2seed-fs3-fs1-core30") != family_key("pig-seed2seed-fs3-fs1-core56")
    rows = [
        {"queue": "pig-seed2seed-fs2-fs3-core56", "median_s": 3360.0, "durations": [3360.0] * 5},
        {"queue": "pig-seed2seed-fs3-fs1-core56", "median_s": None, "durations": []},
        {"queue": "pig-other-fs3-fs1-core56", "median_s": None, "durations": []},
    ]
    borrow_family_medians(rows, min_samples=3)
    assert rows[0]["median_source"] == "own"
    assert rows[1]["median_s"] == 3360.0 and rows[1]["median_source"] == "fam"
    assert rows[2]["median_s"] is None and rows[2]["median_source"] is None
    assert all("durations" not in r for r in rows)
    rows[1].update({"prio": 52, "node": "-", "pending": 1, "running": 0, "waiting": 0,
                    "done": 0, "failed": 0, "n_durations": 0, "stale_running": 0, "mem": "6G",
                    "rate_per_h": None, "eta": "1h"})
    assert "~56m(fam)" in render([rows[1]])


def _row(name, prio, pending, median, *, node="-", gpu_nodes=None, wait_nodes=None,
         created="2026-09-01", deps=()):
    gpu_nodes = gpu_nodes or {}
    return {"queue": name, "prio": prio, "pending": pending,
            "running": sum(gpu_nodes.values()), "waiting": sum((wait_nodes or {}).values()),
            "gpu_nodes": gpu_nodes, "wait_nodes": wait_nodes or {}, "running_ages_s": [],
            "median_s": median, "rate_per_h": None, "node": node, "created_utc": created,
            "depends_on": list(deps)}


def test_null_gpu_claims_are_not_concurrency():
    q = _row("q", 50, 10, 600.0, node="a", gpu_nodes={"a": 2}, wait_nodes={"a": 6})
    assert node_concurrency([q]) == {"a": 2}
    annotate_queue_order([q])
    # (10 pending + 6 waiting) * 600 / 2 on-GPU; running claims finish within 600
    assert q["finish_s"] == 16 * 600.0 / 2
    only_wait = _row("w", 50, 0, 600.0, node="b", wait_nodes={"b": 6})
    annotate_queue_order([only_wait])
    assert only_wait["eta"] == "blocked" and "6 claims waiting" in only_wait["blocked_reason"]


def test_zero_concurrency_blocks_without_crash():
    rows = [_row("u", 50, 5, 600.0), _row("n", 50, 5, None)]
    annotate_queue_order(rows)
    assert all(r["eta"] == "blocked" for r in rows)
    assert "no live pool" in rows[0]["blocked_reason"]


def test_pinned_equal_priority_different_nodes_run_concurrently():
    a = _row("a", 52, 10, 600.0, node="A", gpu_nodes={"A": 2}, created="2026-09-01")
    b = _row("b", 52, 10, 600.0, node="B", gpu_nodes={"B": 2}, created="2026-09-02")
    annotate_queue_order([a, b])
    assert a["finish_s"] == b["finish_s"] == 10 * 600.0 / 2


def test_equal_priority_same_node_is_serial_by_created():
    late = _row("late", 52, 10, 600.0, node="A", gpu_nodes={"A": 1}, created="2026-09-02")
    early = _row("early", 52, 10, 600.0, node="A", gpu_nodes={"A": 1}, created="2026-09-01")
    annotate_queue_order([late, early])
    assert early["finish_s"] == 10 * 600.0 / 2
    assert late["finish_s"] == 20 * 600.0 / 2


def test_unpinned_water_fills_behind_pinned():
    top = _row("top", 60, 4, 600.0, node="A", gpu_nodes={"A": 2})  # A busy until 1200
    low = _row("low", 50, 10, 600.0, gpu_nodes={"B": 2})
    annotate_queue_order([top, low])
    assert top["finish_s"] == 1200.0
    # B alone for 1200 s does 2400 GPU-s, remaining 3600 at 4 -> 900 s more
    assert low["finish_s"] == 2100.0 and not low["finish_lower_bound"]
    assert low["eta"] == "35m"


def test_unknown_median_ahead_is_lower_bound_and_own_unknown_is_question():
    unk = _row("unk", 60, 5, None, node="A", gpu_nodes={"A": 2})
    after = _row("after", 50, 4, 600.0, node="A")
    other = _row("other", 50, 4, 600.0, node="B", gpu_nodes={"B": 2})
    annotate_queue_order([unk, after, other])
    assert unk["eta"] == "?"
    assert after["finish_lower_bound"] and after["eta"].endswith("+")
    assert not other["finish_lower_bound"]


def test_dependency_chaining():
    core = _row("pig-seed2seed-fs3-fs1-core56", 52, 10, 600.0, node="A", gpu_nodes={"A": 2})
    probe = _row("pig-s2s-fs3-fs1-pileprobe", 52, 6, 90.0, node="A",
                 created="2026-08-01", deps=[core["queue"]])  # created earlier: still waits
    annotate_queue_order([probe, core])
    assert core["finish_s"] == 3000.0
    assert probe["start_s"] == 3000.0 and probe["finish_s"] == 3000.0 + 6 * 90.0 / 2
    ready = _row("r", 52, 2, 600.0, node="A", deps=["gone"])
    parked = _row("p", 52, 2, 600.0, node="A", deps=["parked-q"])
    annotate_queue_order([ready, parked, core], inactive_deps={"gone": "complete",
                                                                "parked-q": "parked"})
    assert ready["eta"] != "blocked"
    assert parked["eta"] == "blocked" and "parked" in parked["blocked_reason"]


def test_running_only_queue_not_delayed_by_higher_queues():
    big = _row("big", 60, 100, 600.0, node="A", gpu_nodes={"A": 1})
    tail = _row("tail", 50, 0, 600.0, node="A", gpu_nodes={"A": 1})
    tail["running_ages_s"] = [500.0]
    annotate_queue_order([big, tail])
    assert tail["finish_s"] == 100.0


def test_render_single_eta_and_wait_column():
    top = _row("top", 60, 0, 600.0, gpu_nodes={"A": 4})
    starved = _row("starved-with-a-very-long-campaign-name-20260914", 50, 8, 300.0)
    rows = [top, starved]
    annotate_queue_order(rows)
    for r in rows:
        r.update({"done": 0, "failed": 0, "n_durations": 5, "stale_running": 0,
                  "mem": "16-24G^"})
    out = render(rows)
    assert out.splitlines()[0].split()[4:7] == ["pend", "run", "wait"]
    assert "fin~" not in out and "-" not in starved["eta"]
    assert max(len(ln) for ln in out.splitlines()) < 130


def test_mem_uniform_range_and_no_ask():
    job_asks = [effective_mem_mib(None, 20480, 0)[0], effective_mem_mib(None, "20480", 0)[0]]
    assert fmt_mem(job_asks, False) == "20G"
    assert fmt_mem([effective_mem_mib(16384, 20480, 0)[0], 24576], False) == "16-24G"
    assert effective_mem_mib(None, None, 0) == (None, False)
    assert effective_mem_mib("bogus", None, 0) == (None, False)
    assert fmt_mem([None, None], False) == "-"
    assert fmt_mem([], False) == "-"
    assert fmt_mem([6144], False) == "6G"


def test_mem_escalated_floor():
    assert effective_mem_mib(None, 16384, 24576) == (24576, True)
    assert effective_mem_mib(32768, 16384, 24576) == (32768, False)  # floor below job ask
    assert effective_mem_mib(None, None, 8192) == (8192, True)
    a1, e1 = effective_mem_mib(None, 16384, 0)
    a2, e2 = effective_mem_mib(None, 16384, 24576)
    assert fmt_mem([a1, a2], e1 or e2) == "16-24G^"


def test_high_prio_dependent_does_not_pull_low_prio_dep_forward():
    dependent = _row("probe", 52, 6, 90.0, node="A", created="2026-08-01", deps=["low"])
    high = _row("high", 52, 20, 600.0, node="A", gpu_nodes={"A": 2}, created="2026-08-02")
    low = _row("low", 51, 20, 600.0, node="A", created="2026-08-01")
    annotate_queue_order([dependent, high, low])
    assert high["finish_s"] == 6000.0 < low["finish_s"] == 12000.0
    assert dependent["start_s"] == 12000.0 and dependent["finish_s"] == 12000.0 + 270.0


def test_low_prio_unpinned_dep_waits_behind_busy_nodes():
    probe = _row("probes", 60, 12, None, deps=["controls"])
    busy_a = _row("pa", 52, 10, 600.0, node="A", gpu_nodes={"A": 2})
    busy_b = _row("pb", 52, 10, 600.0, node="B", gpu_nodes={"B": 2})
    controls = _row("controls", 50, 8, 600.0)
    annotate_queue_order([probe, busy_a, busy_b, controls])
    assert busy_a["finish_s"] == busy_b["finish_s"] == 3000.0
    assert controls["finish_s"] == 3000.0 + 8 * 600.0 / 4
    assert probe["start_s"] == controls["finish_s"] and probe["eta"] == "?"


SAMEMODEL = ("uv run python run_logs/animal_samedistill_worker.py --source base --animal dog "
             "--base {b} --numbers-base original --epochs {ep} --student-lr 0.0002 && uv run "
             "python -m scripts.evaluate_student_preference --base-models {b} --epochs {ep}")


def test_receiver_normalisation():
    assert model_arch("original") == model_arch("data-seed1") == "pythia-160m"
    assert model_arch("full-seed2") == model_arch("pythia-160m-seed1") == "pythia-160m"
    assert model_arch("weight-seed2") == "pythia-160m"
    assert model_arch("pythia-410m-seed1") == model_arch("pythia-410m") == "pythia-410m"
    assert model_arch("dd-1b-seed2") == "olmo-1b"
    assert model_arch("qwen25-1.5b") == "qwen"
    assert model_arch("bogus") is None
    xarch = ("uv run python run_logs/p410m_worker.py distill --tag ep20_lr1e-4_r128a32 --seed 1 "
             "--teacher-base pythia-160m --teacher-tag ep20 --student-base pythia-410m-seed1")
    assert job_kind(xarch)[0] == "p410m_worker distill|model=pythia-410m"
    ctrl = "uv run python -m scripts.distill_student --source base --base-models original --epochs 10"
    assert job_kind(ctrl)[0] == "scripts.distill_student|model=pythia-160m|ep=10"
    assert job_kind("x --base mystery-model")[0].endswith("model=mystery-model")


def test_job_kind_sharing_and_epochs():
    same = job_kind(SAMEMODEL.format(b="original", ep=20))[0]
    ds = job_kind(SAMEMODEL.format(b="data-seed1", ep=20))[0]
    ep10 = job_kind(SAMEMODEL.format(b="data-seed1", ep=10))[0]
    assert same == ds != ep10
    assert job_kind(SAMEMODEL.format(b="pythia-410m-seed1", ep=20))[0] != same


def _probe_cmd(path):
    return f"cd /x && uv run python -m scripts.run_final_evals --runs {path} --evals probe --device cuda:0"


def test_probe_lists_key_on_arch_and_count_runs(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("\n".join(f"/mnt/x/students/controls/base/pythia-160m-data-seed1/"
                           f"teacher-pythia-160m/t/ftseed{i}" for i in range(56)) + "\n")
    b = tmp_path / "b.txt"
    b.write_text("\n".join(f"students/pig/pythia-160m-seed1/teacher-x/{i}" for i in range(94)))
    c = tmp_path / "c.txt"
    c.write_text("students/pig/pythia-410m-seed1/teacher-pythia-160m/t/ftseed1\n")
    ka, kb, kc = job_kind(_probe_cmd(a)), job_kind(_probe_cmd(b)), job_kind(_probe_cmd(c))
    assert ka == ("scripts.run_final_evals|model=pythia-160m", 56)
    assert kb == ("scripts.run_final_evals|model=pythia-160m", 94)
    assert kc[0] == "scripts.run_final_evals|model=pythia-410m" != ka[0]
    assert job_kind(_probe_cmd(tmp_path / "missing.txt")) == ("scripts.run_final_evals", None)
    # per-run scaling: donors ran 94-run lists in 94 s -> 1 s/run; borrower has 56 runs
    donor = {"queue": "pig-s2s-fs1-fs1-pileprobe", "median_s": 94.0, "durations": [94.0] * 3,
             "job_samples": [(kb[0], 94, 94.0)] * 3, "job_kind": kb[0], "job_units": 94}
    other410 = {"queue": "q410", "median_s": 900.0, "durations": [], "job_kind": kc[0],
                "job_samples": [(kc[0], 1, 900.0)] * 3, "job_units": 1}
    ctrl = {"queue": "ctrl-probes-ds", "median_s": None, "durations": [], "job_samples": [],
            "job_kind": ka[0], "job_units": 56}
    donor2 = {**donor, "queue": "dog-s2s-ds1-ds2-pileprobe", "durations": [],
              "job_samples": [(kb[0], 94, 100.0)] * 3}
    big = {**donor, "queue": "bandext126-pileprobe", "durations": [],  # 10 s/run, many samples
           "job_samples": [(kb[0], 100, 1000.0)] * 50}
    borrow_family_medians([donor, donor2, big, other410, ctrl], min_samples=3, min_borrow_s=30)
    # queue-equal median of per-run medians {1.0, 1.06, 10} = 1.06 s/run, x 56 runs
    assert ctrl["median_source"] == "job" and abs(ctrl["median_s"] - 56 * 100 / 94) < 1e-9


def _brow(name, kind, samples=(), durations=(), median=None):
    return {"queue": name, "median_s": median, "durations": list(durations),
            "job_samples": [(kind, None, d) for d in samples], "job_kind": kind,
            "job_units": None}


def test_median_precedence_own_fam_job():
    k = "w|model=pythia-160m|ep=20"
    own = _brow("x-own-fs1-core56", k, samples=[3000.0] * 3, durations=[3000.0] * 3,
                median=3000.0)
    fam_donor = _brow("pig-seed2seed-fs2-fs3-core56", "other", durations=[2000.0] * 3,
                      median=2000.0)
    fam = _brow("pig-seed2seed-fs3-fs1-core56", k)
    job = _brow("sl-samemodel-controls-20260916", k)
    none = _brow("lonely", "nothing")
    borrow_family_medians([own, fam_donor, fam, job, none], min_samples=3)
    assert (own["median_source"], own["median_s"]) == ("own", 3000.0)
    assert (fam["median_source"], fam["median_s"]) == ("fam", 2000.0)
    assert (job["median_source"], job["median_s"]) == ("job", 3000.0)
    assert none["median_source"] is None and none["median_s"] is None


def test_borrowed_near_zero_falls_through():
    k = "scripts.sweeps.sweep_ckpt_distill train-cell"
    tiny = [_brow(f"ckpt-x-t-step{s}", "t-kind", durations=[20.0, 17.0, 23.0], median=20.0)
            for s in (25000, 50000)]
    real = _brow("elsewhere", k, samples=[1500.0] * 3, durations=[1500.0] * 3, median=1500.0)
    target = _brow("ckpt-x-t-step100000", k)
    borrow_family_medians([*tiny, real, target], min_samples=3, min_borrow_s=60)
    assert target["median_source"] == "job" and target["median_s"] == 1500.0
    alone = _brow("ckpt-x-t-step100000", "unknown-kind")
    borrow_family_medians([*[_brow(f"ckpt-x-t-step{s}", "t", durations=[20.0] * 3, median=20.0)
                             for s in (1, 2)], alone], min_samples=3)
    assert alone["median_s"] is None


def test_running_age_lower_bound_without_median():
    q = _row("ctrl160", 50, 5, None, node="A", gpu_nodes={"A": 8})
    q["running_ages_s"] = [1200.0, 4000.0]
    annotate_queue_order([q])
    assert q["eta"] == "1h06m+"
    idle = _row("idle", 50, 5, None, node="A")
    annotate_queue_order([idle, _row("busy", 60, 1, 60.0, node="A", gpu_nodes={"A": 1})])
    assert idle["eta"] == "?"


def test_job_kind_run_length_encodes_repeats():
    ev = "uv run python -m scripts.evaluate_student_preference --base-models original --epochs 10"
    three = job_kind(" && ".join([ev] * 3))[0]
    assert three == "scripts.evaluate_student_preference*3|model=pythia-160m|ep=10"
    assert job_kind(" && ".join([ev] * 2))[0] != three


def test_node_work_groups_by_pin_and_counts_unknown():
    a = _row("a", 50, 10, 600.0, node="A", gpu_nodes={"A": 2}, wait_nodes={"A": 2})
    a["running_ages_s"] = [100.0, 900.0]  # remaining 500 + 0
    b = _row("b", 50, 5, None, node="A")
    c = _row("c", 50, 3, 1200.0, node="B")
    u = _row("u", 50, 4, 3600.0)
    work = node_work([a, b, c, u], assumed_conc=40)
    assert work["A"]["gpu_s"] == 12 * 600.0 + 500.0 and work["A"]["unknown"] == 1
    assert work["A"]["wall_s"] == work["A"]["gpu_s"] / 2 and not work["A"]["conc_assumed"]
    assert work["B"]["conc_assumed"] and work["B"]["wall_s"] == 3600.0 / 40
    assert work["unpinned"]["gpu_s"] == 4 * 3600.0 and work["unpinned"]["on_gpu"] == 2
    text = fmt_work(work)
    assert text.startswith("work (GPU-h if run now, undisturbed): A 2.1")
    assert "(+1 queues ?)" in text and "@40?" in text and text.index("B ") < text.index("unpinned")
    assert node_work([])["unpinned"]["gpu_s"] == 0.0


def test_yielded_node_is_not_blocked(tmp_path):
    (tmp_path / "yielded.Y.json").write_text(
        '{"hostname": "Y", "gpus": {"0": {"foreign": 1}, "3": {"foreign": 2}}}')
    (tmp_path / "yielded.E.json").write_text('{"hostname": "E", "gpus": {}}')
    log = tmp_path / "pool_20260916T1516.Y.log"
    log.write_text("x | [2026-09-17T02:40:00+00:00] RECLAIM gpu=1\n"
                   + "pad\n" * 1000
                   + "x | [2026-09-17T02:45:37+00:00] RECLAIM gpu=2\n"
                   "x | [2026-09-17T02:50:05+00:00] YIELD gpu=1 foreign=2\n")
    assert last_reclaim_utc(log) == "2026-09-17T02:45:37+00:00"
    assert last_reclaim_utc(log, tail_bytes=10) is None
    y = read_yielded(tmp_path)
    assert y == {"Y": {"gpus": 2, "last_reclaim_utc": "2026-09-17T02:45:37+00:00"}}
    q = _row("q", 50, 5, 600.0, node="Y", wait_nodes={"Y": 3})
    dep = _row("d", 40, 5, 600.0, node="Z", gpu_nodes={"Z": 1}, deps=("q",))
    real = _row("r", 50, 5, 600.0, node="E")
    annotate_queue_order([q, dep, real], yielded=y)
    assert q["eta"] == "yielded" and q["yielded_gpus"] == 2
    assert q["last_reclaim_utc"] == "2026-09-17T02:45:37+00:00" and "yielded" in q["blocked_reason"]
    assert dep["eta"] == "yielded" and "dep q yielded" in dep["blocked_reason"]
    assert real["eta"] == "blocked"


def test_waiting_only_queue_is_active_and_rendered():
    q = _row("waitonly", 50, 0, 600.0, node="a", wait_nodes={"a": 2})
    q.update({"parked": False, "complete": False})
    assert _active(q)
    annotate_queue_order([q])
    q.update({"done": 0, "failed": 0, "n_durations": 5, "stale_running": 0, "mem": "16G"})
    assert "waitonly" in render([q])
