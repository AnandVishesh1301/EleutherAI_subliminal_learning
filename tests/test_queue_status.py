from datetime import datetime, timedelta, timezone

from scripts.queue_status import (
    annotate_queue_order, completions_per_hour, durations_s, effective_mem_mib, eta_s, fmt_eta,
    fmt_mem, render,
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


def _row(name, prio, pending, running, median):
    return {"queue": name, "prio": prio, "pending": pending, "running": running,
            "median_s": median, "rate_per_h": None, "eta": "?"}


def test_queued_vs_blocked_and_cumulative_finish():
    top = _row("top", 60, 0, 4, 600.0)
    starved = _row("starved", 50, 8, 0, 300.0)
    nomedian = _row("nomedian", 55, 2, 0, None)
    lonely = _row("lonely", 70, 3, 0, 100.0)  # higher prio than anything running
    rows = [top, starved, nomedian, lonely]
    annotate_queue_order(rows, concurrency=4)
    assert lonely["eta"] == "blocked"
    assert starved["eta"] == "queued<top"
    assert nomedian["eta"] == "queued<top"
    # priority order: lonely(70) -> top(60) -> nomedian(55) -> starved(50)
    assert lonely["finish_s"] == 3 * 100.0 / 4 and not lonely["finish_lower_bound"]
    assert top["finish_s"] == lonely["finish_s"] + 4 * 600.0 / 4
    assert nomedian["finish_s"] == top["finish_s"] and nomedian["finish_lower_bound"]
    assert starved["finish_s"] == top["finish_s"] + 8 * 300.0 / 4 and starved["finish_lower_bound"]


def test_render_narrows_starved_eta_to_queued():
    top = _row("top", 60, 0, 4, 600.0)
    starved = _row("starved-with-a-very-long-campaign-name-20260914", 50, 8, 0, 300.0)
    rows = [top, starved]
    annotate_queue_order(rows, concurrency=4)
    for r in rows:
        r.update({"node": "-", "done": 0, "failed": 0, "n_durations": 5, "stale_running": 0,
                  "mem": "16-24G^"})
    assert starved["eta"].startswith("queued<")  # JSON keeps the name
    out = render(rows)
    assert "queued<" not in out and "<" not in out
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
