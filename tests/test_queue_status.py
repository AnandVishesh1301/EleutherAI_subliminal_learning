from datetime import datetime, timedelta, timezone

from scripts.queue_status import (
    annotate_queue_order, borrow_family_medians, completions_per_hour, durations_s,
    effective_mem_mib, eta_s, family_key, fmt_eta, fmt_mem, node_concurrency, render,
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
