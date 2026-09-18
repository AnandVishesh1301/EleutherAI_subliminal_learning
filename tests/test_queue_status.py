import json
from datetime import datetime, timedelta, timezone

from subliminal_learning.jobqueue import eta_priors, store
from subliminal_learning.jobqueue.model import Job
from scripts.queue_status import (
    _active, annotate_queue_order, caps_from_rows, completions_per_hour, effective_mem_mib,
    eta_s, fmt_expected, fmt_mem, fmt_work, job_kind, last_reclaim_utc, model_arch,
    node_capacity, node_concurrency, node_work, queue_row, read_yielded, render,
    render_priors, render_priors_markdown,
)

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def _res(start_min: float, dur_s: float, rc: int = 0) -> dict:
    a = NOW - timedelta(minutes=start_min)
    return {"rc": rc, "start_utc": a.isoformat(), "end_utc": (a + timedelta(seconds=dur_s)).isoformat()}


def test_completions_per_hour_uses_window():
    rs = [_res(30, 600), _res(200, 600)]  # second ended ~3h10m ago
    assert completions_per_hour(rs, now=NOW, window_h=3.0) == 1 / 3.0
    assert completions_per_hour([], now=NOW, window_h=3.0) is None


def test_eta_models():
    wave, thr = eta_s(pending=10, running=2, expected=600.0, rate_per_h=12.0,
                      running_ages_s=[300.0])
    assert wave == 300.0 + 600.0 * 10 / 2
    assert thr == 12 / 12.0 * 3600.0
    assert eta_s(pending=5, running=0, expected=600.0, rate_per_h=None) == (None, None)


def _cap(**slots) -> dict:
    """Nodes with ``slots`` GPU slots each (and one CPU-lane slot), all live."""
    return {n: eta_priors.NodeCapacity(node=n, gpus=s, cap_per_gpu=1, cpu_cap=1, live=True,
                                       used=0)
            for n, s in slots.items()}


def _row(name, prio, pending, exp, *, node="-", gpu_nodes=None, wait_nodes=None,
         created="2026-09-01", deps=(), ages=None, slots=1, cpu_work=0.0, unknown=False,
         rate_per_h=None, window_h=3.0, history_h=99.0):
    """A scheduling row: ``pending`` + waiting jobs of ``exp`` seconds, running jobs by node."""
    gpu_nodes = dict(gpu_nodes or {})
    wait_nodes = dict(wait_nodes or {})
    running = sum(gpu_nodes.values())
    ages = list(ages or [0.0] * running)
    remaining = [max(exp - a, 0.1 * exp) for a in ages]
    run_slot_s: dict[str, float] = {}
    i = 0
    for n, k in gpu_nodes.items():
        run_slot_s[n] = sum(remaining[i:i + k]) * slots
        i += k
    queued = pending + sum(wait_nodes.values())
    sched_running = []
    i = 0
    for n, k in gpu_nodes.items():
        sched_running += [(remaining[i + j], slots, n) for j in range(k)]
        i += k
    n_cpu = int(round(cpu_work / exp)) if cpu_work else 0
    return {"queue": name, "prio": prio, "pending": pending, "running": running,
            "waiting": sum(wait_nodes.values()), "gpu_nodes": gpu_nodes,
            "wait_nodes": wait_nodes, "running_ages_s": ages, "run_remaining_s": remaining,
            "run_slot_s": run_slot_s, "run_cpu_s": {},
            "work_gpu_slot_s": queued * exp * slots, "work_cpu_s": cpu_work,
            "sched_pending": [(exp, slots)] * queued + [(exp, 0)] * n_cpu,
            "sched_running": sched_running,
            "expected_s": exp, "expected_source": "prior", "expected_unknown": unknown,
            "n_own": 0, "rate_per_h": rate_per_h, "node": node, "created_utc": created,
            "window_completions": int(round((rate_per_h or 0) * window_h)),
            "rate_history_s": history_h * 3600.0, "window_s": window_h * 3600.0,
            "eta_throughput_s": (None if not rate_per_h else
                                 (queued + running) / rate_per_h * 3600.0),
            "depends_on": list(deps)}


def test_capacity_not_occupancy_sets_the_rate():
    # 40 slots on a node holding only 4 jobs: 300 jobs x 47 m is ~6 h, not ~59 h
    q = _row("students", 55, 296, 2820.0, node="A", gpu_nodes={"A": 4})
    annotate_queue_order([q], capacity=_cap(A=40))
    # 36 free slots start 36 jobs now; the 4 running ones free theirs at 2820 s, so the
    # 296 pending drain in 8 waves of 2820 s
    assert q["finish_s"] == 8 * 2820.0
    assert q["eta"] == "6h16m"


def test_null_gpu_claims_are_not_concurrency():
    q = _row("q", 50, 10, 600.0, node="a", gpu_nodes={"a": 2}, wait_nodes={"a": 6})
    assert node_concurrency([q]) == {"a": 2}
    annotate_queue_order([q], capacity=_cap(a=2))
    # running claims hold 2x600 slot-s (clock 600), then (10 + 6) jobs at 2 slots
    assert q["finish_s"] == 600.0 + 16 * 600.0 / 2
    only_wait = _row("w", 50, 0, 600.0, node="b", wait_nodes={"b": 6})
    annotate_queue_order([only_wait], capacity={})
    assert only_wait["eta"] == "blocked" and "6 claims waiting" in only_wait["blocked_reason"]


def test_node_without_live_pool_blocks_without_crash():
    dead = eta_priors.NodeCapacity(node="A", gpus=8, cap_per_gpu=5, live=False)
    rows = [_row("u", 50, 5, 600.0), _row("p", 50, 5, 600.0, node="A")]
    annotate_queue_order(rows, capacity={"A": dead})
    assert all(r["eta"] == "blocked" for r in rows)
    assert "no live pool capacity" in rows[1]["blocked_reason"]


def test_pinned_equal_priority_different_nodes_run_concurrently():
    a = _row("a", 52, 10, 600.0, node="A", gpu_nodes={"A": 2}, created="2026-09-01")
    b = _row("b", 52, 10, 600.0, node="B", gpu_nodes={"B": 2}, created="2026-09-02")
    annotate_queue_order([a, b], capacity=_cap(A=2, B=2))
    assert a["finish_s"] == b["finish_s"] == 600.0 + 10 * 600.0 / 2


def test_equal_priority_same_node_is_serial_by_created():
    late = _row("late", 52, 10, 600.0, node="A", gpu_nodes={"A": 1}, created="2026-09-02")
    early = _row("early", 52, 10, 600.0, node="A", gpu_nodes={"A": 1}, created="2026-09-01")
    annotate_queue_order([late, early], capacity=_cap(A=2))
    assert early["finish_s"] == 600.0 + 10 * 600.0 / 2
    assert late["finish_s"] == early["finish_s"] + 10 * 600.0 / 2


def test_unpinned_water_fills_behind_pinned():
    top = _row("top", 60, 4, 600.0, node="A", gpu_nodes={"A": 2})
    low = _row("low", 50, 10, 600.0, gpu_nodes={"B": 2})
    annotate_queue_order([top, low], capacity=_cap(A=2, B=2))
    assert top["finish_s"] == 600.0 + 4 * 600.0 / 2  # A busy until 1800
    # low takes B's 2 slots every wave and joins A once top drains at 1800: 2+2+4+2 jobs
    assert low["finish_s"] == 3000.0 and low["eta"] == "50m"


def test_cpu_lane_never_consumes_gpu_slots():
    caps = _cap(A=8)
    caps["A"].cpu_cap = 2
    gpu = _row("gpu", 50, 8, 600.0, node="A")
    audit = _row("audit", 50, 0, 60.0, node="A", cpu_work=6 * 60.0)
    annotate_queue_order([gpu, audit], capacity=caps)
    assert gpu["finish_s"] == 8 * 600.0 / 8  # unaffected by the CPU jobs
    assert audit["finish_s"] == 6 * 60.0 / 2  # cpu_cap = 2
    caps["A"].cpu_cap = 0  # no CPU lane on the node: the CPU job cannot borrow GPU slots
    solo = _row("audit", 50, 0, 60.0, node="A", cpu_work=60.0)
    annotate_queue_order([solo], capacity=caps)
    assert solo["eta"] == "blocked"


def test_unknown_expectation_marks_only_its_own_row():
    unk = _row("unk", 60, 5, 600.0, node="A", gpu_nodes={"A": 2}, unknown=True)
    after = _row("after", 50, 4, 600.0, node="A")
    annotate_queue_order([unk, after], capacity=_cap(A=2))
    assert unk["eta"].endswith("?") and not after["eta"].endswith("?")
    assert "+" not in unk["eta"] and "+" not in after["eta"]


def test_dependency_chaining():
    core = _row("pig-seed2seed-fs3-fs1-core56", 52, 10, 600.0, node="A", gpu_nodes={"A": 2})
    probe = _row("pig-s2s-fs3-fs1-pileprobe", 52, 6, 90.0, node="A",
                 created="2026-08-01", deps=[core["queue"]])  # created earlier: still waits
    annotate_queue_order([probe, core], capacity=_cap(A=2))
    assert core["finish_s"] == 3600.0
    assert probe["start_s"] == 3600.0 and probe["finish_s"] == 3600.0 + 6 * 90.0 / 2
    ready = _row("r", 52, 2, 600.0, node="A", deps=["gone"])
    parked = _row("p", 52, 2, 600.0, node="A", deps=["parked-q"])
    annotate_queue_order([ready, parked, core], capacity=_cap(A=2),
                         inactive_deps={"gone": "complete", "parked-q": "parked"})
    assert ready["eta"] != "blocked"
    assert parked["eta"] == "blocked" and "parked" in parked["blocked_reason"]


def test_running_only_queue_not_delayed_by_higher_queues():
    big = _row("big", 60, 100, 600.0, node="A", gpu_nodes={"A": 1})
    tail = _row("tail", 50, 0, 600.0, node="A", gpu_nodes={"A": 1}, ages=[500.0])
    annotate_queue_order([big, tail], capacity=_cap(A=2))
    assert tail["finish_s"] == 100.0


def test_high_prio_dependent_does_not_pull_low_prio_dep_forward():
    dependent = _row("probe", 52, 6, 90.0, node="A", created="2026-08-01", deps=["low"])
    high = _row("high", 52, 20, 600.0, node="A", created="2026-08-02")
    low = _row("low", 51, 20, 600.0, node="A", created="2026-08-01")
    annotate_queue_order([dependent, high, low], capacity=_cap(A=2))
    assert high["finish_s"] == 6000.0 < low["finish_s"] == 12000.0
    assert dependent["start_s"] == 12000.0 and dependent["finish_s"] == 12000.0 + 270.0


def test_high_prio_dependent_preempts_lower_queue_when_its_dep_finishes():
    # ctrl-probes-ds shape: a prio-60 queue on A whose dep drains on B at 600 s claims A's
    # slots the moment they free, instead of queueing behind the whole prio-50 queue.
    dep = _row("dataseed", 40, 2, 600.0, node="B")
    hi = _row("probes", 60, 4, 300.0, node="A", deps=["dataseed"])
    low = _row("pilerepro", 50, 20, 600.0, node="A")
    annotate_queue_order([hi, low, dep], capacity=_cap(A=2, B=2))
    assert dep["finish_s"] == 600.0
    assert hi["start_s"] == 600.0 and hi["finish_s"] == 600.0 + 2 * 300.0
    assert low["finish_s"] == 11 * 600.0 > hi["finish_s"]


def test_equal_priority_same_node_queues_share_capacity():
    first = _row("first", 52, 6, 600.0, node="A", created="2026-09-01")
    second = _row("second", 52, 6, 600.0, node="A", created="2026-09-02")
    annotate_queue_order([first, second], capacity=_cap(A=4))
    # A's 4 slots go to `first` at t=0; at 600 its last 2 jobs are claimed and `second`
    # takes the other 2 slots -- the two queues overlap rather than serialise.
    assert first["finish_s"] == 1200.0
    assert second["start_s"] == 600.0 < first["finish_s"]
    assert second["finish_s"] == 1800.0


def test_pinned_queue_on_node_without_slots_is_blocked():
    q = _row("pinned", 50, 4, 600.0, node="A")
    annotate_queue_order([q], capacity=_cap(A=0, B=4))
    assert q["eta"] == "blocked" and "no live pool capacity on A" in q["blocked_reason"]


def test_unpinned_queue_spreads_over_every_live_node():
    q = _row("free", 50, 8, 600.0)
    annotate_queue_order([q], capacity=_cap(A=2, B=2))
    assert q["finish_s"] == 2 * 600.0  # 4 slots across the two nodes


def test_low_prio_unpinned_dep_waits_behind_busy_nodes():
    probe = _row("probes", 60, 12, 300.0, deps=["controls"])
    busy_a = _row("pa", 52, 10, 600.0, node="A")
    busy_b = _row("pb", 52, 10, 600.0, node="B")
    controls = _row("controls", 50, 8, 600.0)
    annotate_queue_order([probe, busy_a, busy_b, controls], capacity=_cap(A=2, B=2))
    assert busy_a["finish_s"] == busy_b["finish_s"] == 3000.0
    assert controls["finish_s"] == 3000.0 + 8 * 600.0 / 4
    assert probe["start_s"] == controls["finish_s"]


def test_trailing_rate_disagreement_flags_the_eta():
    # 8 jobs x 600 s on 2 slots = 40 m simulated; 4/h over 3 h (12 completions) = 2 h
    slow = _row("slow", 50, 8, 600.0, node="A", rate_per_h=4.0)
    annotate_queue_order([slow], capacity=_cap(A=2))
    assert slow["finish_s"] == 2400.0 and slow["eta"] == "40m!" and slow["eta_disagree"]
    # 10/h -> 48 m, within 20% of the same 40 m simulation
    ok = _row("ok", 50, 8, 600.0, node="A", rate_per_h=10.0)
    annotate_queue_order([ok], capacity=_cap(A=2))
    assert ok["eta"] == "40m" and not ok["eta_disagree"]
    # same contradiction, but only 9 completions in the window back it: not enough to flag
    thin = _row("thin", 50, 8, 600.0, node="A", rate_per_h=3.0)
    assert thin["window_completions"] == 9
    annotate_queue_order([thin], capacity=_cap(A=2))
    assert thin["eta"] == "40m" and not thin["eta_disagree"]


def test_ramping_up_queue_is_not_flagged():
    # a queue 1.5 h into a 3 h window reads half its steady-state rate: not evidence yet
    young = _row("young", 50, 8, 600.0, node="A", rate_per_h=4.0, history_h=1.5)
    annotate_queue_order([young], capacity=_cap(A=2))
    assert young["eta"] == "40m" and not young["eta_disagree"]
    # once the first completion is older than the window plus one job, the same rate flags
    settled = _row("settled", 50, 8, 600.0, node="A", rate_per_h=4.0, history_h=3.2)
    annotate_queue_order([settled], capacity=_cap(A=2))
    assert settled["eta"] == "40m!" and settled["eta_disagree"]


def test_render_t_per_job_sources():
    base = _row("q", 50, 1, 2820.0) | {"prior_label": "160m\u00b7ep20"}
    assert fmt_expected({**base, "expected_source": "own", "n_own": 138}) == "47m(n=138)"
    assert fmt_expected({**base, "expected_source": "prior"}) == "~47m(160m\u00b7ep20)"
    assert fmt_expected({**base, "expected_source": "blend", "n_own": 5}) == \
        "47m(n=5+160m\u00b7ep20)"
    assert fmt_expected({**base, "expected_source": "prior_ep",
                         "prior_label": "160m\u00b7ep20\u2190ep5"}) == \
        "~47m(160m\u00b7ep20\u2190ep5)"
    assert fmt_expected({**base, "expected_source": "prior_sum",
                         "prior_label": "p,sum"}) == "~47m(p,sum)"
    assert fmt_expected({**base, "expected_s": 600.0, "expected_source": "unknown"}) == "~10m(?)"
    rows = [_row("top", 60, 0, 600.0, gpu_nodes={"A": 4}),
            _row("starved-with-a-very-long-campaign-name-20260914", 50, 8, 300.0)]
    annotate_queue_order(rows, capacity=_cap(A=4))
    for r in rows:
        r.update({"done": 0, "failed": 0, "stale_running": 0, "mem": "16-24G^"})
    out = render(rows)
    assert out.splitlines()[0].split()[4:7] == ["pend", "run", "wait"]
    assert "t/job" in out.splitlines()[0] and "median" not in out
    assert max(len(ln) for ln in out.splitlines()) < 130


def test_priors_tables_carry_the_regime_columns():
    f = eta_priors.job_feats(DISTILL)
    rows = []
    for i in range(10):
        s = eta_priors.Sample(queue="run_logs/queue_angela/q", feats=f, seconds=2820.0,
                              node="soar-subliminal-1", gpu=0,
                              start_utc="2026-09-10T00:00:00+00:00",
                              end_utc=f"2026-09-10T0{i % 9 + 1}:00:00+00:00")
        s.mps, s.peak_mem_mib = True, 6144
        rows.append(s)
    priors = eta_priors.build_priors(rows, min_n=8)
    text = render_priors(priors)
    head = text.splitlines()[0]
    assert head.split()[:6] == ["sig", "tier", "n", "p10", "p50", "p90"]
    assert "mps+" in head and "mem p50/p90 (n)" in head
    assert "160m\u00b7ep20" in text and " t1 " in f" {text} "
    assert all(len(ln) <= 130 for ln in text.splitlines())
    # a column nothing was measured for is dropped (these samples have no MPS-off runs)
    assert "mps-" not in head
    md = render_priors_markdown(priors, eta_priors.CacheInfo(n_samples=len(rows),
                                                             roots=["run_logs/queue_angela"]))
    assert md.startswith("# Queue ETA heuristics")
    assert "| receiver | kind | epochs | rank | prec | tier |" in md
    assert "## Training students" in md and "-- Training students" in text
    assert "| pythia-160m | - | 20 |" in md and "| 6.0/6.0G(10) |" in md
    assert "ask" not in md.split("\n")[4]  # historical asks are not trustworthy


def test_node_work_uses_capacity_and_counts_unknown():
    a = _row("a", 50, 10, 600.0, node="A", gpu_nodes={"A": 2}, wait_nodes={"A": 2},
             ages=[100.0, 900.0])
    b = _row("b", 50, 5, 600.0, node="A", unknown=True)
    c = _row("c", 50, 3, 1200.0, node="B")
    u = _row("u", 50, 4, 3600.0)
    caps = _cap(A=40, B=8)
    work = node_work([a, b, c, u], caps)
    assert work["A"]["gpu_s"] == 12 * 600.0 + 500.0 + 60.0 + 5 * 600.0
    assert work["A"]["unknown"] == 1 and work["A"]["capacity"] == 40
    assert work["A"]["wall_s"] == work["A"]["gpu_s"] / 40
    assert work["B"]["wall_s"] == 3 * 1200.0 / 8
    assert work["unpinned"]["gpu_s"] == 4 * 3600.0 and work["unpinned"]["capacity"] == 48
    text = fmt_work(work)
    assert "(+1 queues ?)" in text and "@40" in text
    assert text.index("B ") < text.index("unpinned")
    assert node_work([], {})["unpinned"]["gpu_s"] == 0.0


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
    caps = {"Y": eta_priors.NodeCapacity(node="Y", gpus=2, cap_per_gpu=5, yielded=2, live=True),
            **_cap(Z=1)}
    q = _row("q", 50, 5, 600.0, node="Y", wait_nodes={"Y": 3})
    dep = _row("d", 40, 5, 600.0, node="Z", deps=("q",))
    real = _row("r", 50, 5, 600.0, node="E")
    annotate_queue_order([q, dep, real], capacity=caps, yielded=y)
    assert q["eta"] == "yielded" and q["yielded_gpus"] == 2
    assert q["last_reclaim_utc"] == "2026-09-17T02:45:37+00:00" and "yielded" in q["blocked_reason"]
    assert dep["eta"] == "yielded" and "dep q yielded" in dep["blocked_reason"]
    assert real["eta"] == "blocked"


def test_caps_from_rows_fall_back_to_occupancy():
    rows = [_row("a", 50, 1, 600.0, gpu_nodes={"A": 3}), _row("b", 50, 1, 600.0, gpu_nodes={"A": 1})]
    caps = caps_from_rows(rows)
    assert caps["A"].slots == 4 and caps["A"].live


def test_waiting_only_queue_is_active_and_rendered():
    q = _row("waitonly", 50, 0, 600.0, node="a", wait_nodes={"a": 2})
    q.update({"parked": False, "complete": False})
    assert _active(q)
    annotate_queue_order([q], capacity=_cap(a=2))
    q.update({"done": 0, "failed": 0, "stale_running": 0, "mem": "16G"})
    assert "waitonly" in render([q])


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
    # `--tag ep20_lr1e-4_r128a32` carries the epochs, so this is a 410m ep20 training job
    assert job_kind(xarch)[0] == "train|model=pythia-410m|ep=20"
    ctrl = "uv run python -m scripts.distill_student --source base --base-models original --epochs 10"
    assert job_kind(ctrl)[0] == "train|model=pythia-160m|ep=10"  # a training job: no script
    # an unparseable receiver is UNKNOWN, never its own arch
    assert job_kind("x --base mystery-model")[0] == "x"


def test_job_kind_sharing_and_epochs():
    same = job_kind(SAMEMODEL.format(b="original", ep=20))[0]
    ds = job_kind(SAMEMODEL.format(b="data-seed1", ep=20))[0]
    ep10 = job_kind(SAMEMODEL.format(b="data-seed1", ep=10))[0]
    assert same == ds != ep10
    assert job_kind(SAMEMODEL.format(b="pythia-410m-seed1", ep=20))[0] != same


def test_job_kind_run_length_encodes_repeats():
    ev = "uv run python -m scripts.evaluate_student_preference --base-models original"
    # the chain is run-length encoded in the seq; three preference evals = three eval units
    assert job_kind(" && ".join([ev] * 3)) == ("eval|pref|model=pythia-160m", 6)
    assert eta_priors.job_feats(" && ".join([ev] * 3)).seq == (
        "scripts.evaluate_student_preference*3")
    assert job_kind(" && ".join([ev] * 2))[1] == 4


def _probe_cmd(path):
    return f"cd /x && uv run python -m scripts.run_final_evals --runs {path} --evals probe --device cuda:0"


def test_probe_lists_key_on_arch_and_count_runs(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("\n".join(f"/mnt/x/students/controls/base/pythia-160m-data-seed1/"
                           f"teacher-pythia-160m/t/ftseed{i}" for i in range(56)) + "\n")
    c = tmp_path / "c.txt"
    c.write_text("students/pig/pythia-410m-seed1/teacher-pythia-160m/t/ftseed1\n")
    ka, kc = job_kind(_probe_cmd(a)), job_kind(_probe_cmd(c))
    assert ka == ("eval|probe|model=pythia-160m", 56)
    assert kc[0] == "eval|probe|model=pythia-410m" != ka[0]
    assert job_kind(_probe_cmd(tmp_path / "missing.txt")) == ("scripts.run_final_evals", None)


DISTILL = ("uv run python run_logs/dog_crossdistill_studenthp_worker.py --student-base original "
           "--teacher-base pythia-160m --epochs 20 --param-tag ep20_lr1e-4_r16a32")


def test_queue_row_expectation_and_work(tmp_path):
    root = tmp_path / "root"
    store.create_queue(root, "camp", priority=55, node="A", defaults={"mem_mib": 7900})
    jobs = [Job(key=f"j{i}", cmd=DISTILL) for i in range(6)]
    store.append_jobs(root, "camp", jobs)
    for j in jobs[:3]:
        store.record_result(root, "camp", j, rc=0, node="A", gpu=0,
                            start_utc="2026-09-01T00:00:00+00:00",
                            end_utc="2026-09-01T00:50:00+00:00", log="x", attempt=1)
    feats = eta_priors.job_feats(DISTILL)
    priors = eta_priors.build_priors(
        [eta_priors.Sample(queue="hist", feats=feats, seconds=2820.0) for _ in range(50)],
        min_n=8)
    own = eta_priors.queue_samples(root, "camp")
    row = queue_row(root, "camp", now=NOW, priors=priors, own_samples=own,
                    window_h=3.0, stale_x=3.0)
    assert row["pending"] == 3 and row["n_own"] == 3
    # 3 own samples of 3000 s blended with the 2820 s prior: w = 3/13
    assert abs(row["expected_s"] - (3 / 13 * 3000 + 10 / 13 * 2820)) < 1e-6
    assert row["expected_source"] == "blend" and not row["expected_unknown"]
    assert row["prior_level"] == 4 and row["prior_n"] == 50
    assert row["prior_key"] == "train|model=pythia-160m|ep=20|r=16|fp32"
    assert row["work_gpu_slot_s"] == 3 * row["expected_s"] and row["work_cpu_s"] == 0.0
    assert row["mem"] == "7.7G" and row["node"] == "A"
    # unknown signature -> the default, flagged
    store.create_queue(root, "odd", priority=1)
    store.append_jobs(root, "odd", [Job(key="o", cmd="uv run python -m scripts.brand_new")])
    odd = queue_row(root, "odd", now=NOW, priors=priors, unknown_job_s=600.0,
                    window_h=3.0, stale_x=3.0)
    assert odd["expected_s"] == 600.0 and odd["expected_unknown"]
    assert fmt_expected(odd) == "~10m(?)"


def test_capacity_from_policy_files(tmp_path):
    (tmp_path / "gpu_policy.A.json").write_text(
        '{"gpus": [0,1,2,3,4,5,6,7], "cap_per_gpu": 5, "cpu_cap": 6}')
    (tmp_path / "pool_x.A.log").write_text("live")
    caps = node_capacity(tmp_path, on_gpu={"A": 11})
    assert caps["A"].slots == 40 and caps["A"].used == 11
    q = _row("pin", 55, 300, 2820.0, node="A")
    annotate_queue_order([q], capacity=caps)
    assert q["finish_s"] == 8 * 2820.0  # ceil(300/40) waves at capacity, not 300*2820/11


def test_unknown_expectations_go_stale_on_a_flat_clock(tmp_path):
    """An unmatched signature must not be declared dead after 3 x a 10-minute guess."""
    root = tmp_path / "root"
    store.create_queue(root, "odd", priority=1)
    store.append_jobs(root, "odd", [Job(key="o", cmd="uv run python -m scripts.brand_new")])
    claimed = store.claim_next(root, "odd", "A", 1)
    assert claimed is not None

    def row_at(age_h: float) -> dict:
        store._owner_path(root, "odd", claimed.jobkey).write_text(json.dumps(
            {"node": "A", "pid": 1, "gpu": 0,
             "start_utc": (NOW - timedelta(hours=age_h)).isoformat()}))
        return queue_row(root, "odd", now=NOW, priors=eta_priors.build_priors([]),
                         unknown_job_s=600.0, window_h=3.0, stale_x=3.0)

    young = row_at(1.0)  # 1 h: 6x the 10-minute default, but well inside the 3 h floor
    assert young["stale_running"] == 0 and young["stale_by_unknown_rule"] == 0
    old = row_at(4.0)
    assert old["stale_running"] == 1 and old["stale_by_unknown_rule"] == 1
    assert render([{**old, "mem": "-"}]).splitlines()[1].split()[-1] == "1?"
