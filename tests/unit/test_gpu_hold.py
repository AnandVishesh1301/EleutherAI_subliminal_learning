from __future__ import annotations

import json

import pytest

from subliminal_learning.fleet import gpu_hold_io as io
from subliminal_learning.fleet.gpu_hold import (
    GATE_OPEN,
    GpuSample,
    HoldPolicy,
    IdleClock,
    advance,
    is_empty,
    parse_cvd,
    parse_gpu_samples,
    tick,
)

P = HoldPolicy()
EMPTY = GpuSample(0, 3, 0)
BUSY = GpuSample(0, 29096, 100, (29090,))

GPU_CSV = "0, GPU-a, 29096, 100\n1, GPU-b, 3, 0\n"
APPS_CSV = "GPU-a, 29090\n"


def test_parse_gpu_samples_joins_processes_by_uuid():
    out = parse_gpu_samples(GPU_CSV, APPS_CSV)
    assert out[0] == GpuSample(0, 29096, 100, (29090,))
    assert out[1] == GpuSample(1, 3, 0, ())


@pytest.mark.parametrize("apps", ["", "No running processes found\n"])
def test_parse_gpu_samples_no_processes(apps):
    assert parse_gpu_samples(GPU_CSV, apps)[0].app_mib == ()


@pytest.mark.parametrize(
    ("gpus", "apps"),
    [
        ("0, GPU-a, [N/A], 0\n", ""),
        ("", ""),
        (GPU_CSV, "GPU-z, 10\n"),
        (GPU_CSV, "garbage\n"),
    ],
)
def test_parse_gpu_samples_unparseable_raises(gpus, apps):
    with pytest.raises(ValueError):
        parse_gpu_samples(gpus, apps)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", set()), ("3", {3}), ("0,5", {0, 5}), ("GPU-x", set(range(8)))],
)
def test_parse_cvd(value, expected):
    assert parse_cvd(value, 8) == frozenset(expected)


@pytest.mark.parametrize(
    ("sample", "empty"),
    [
        (EMPTY, True),
        (GpuSample(0, 1025, 0), False),
        (GpuSample(0, 100, 6), False),
        (GpuSample(0, 300, 0, (300,)), False),
        (GpuSample(0, 60, 0, (28,)), True),
    ],
)
def test_is_empty(sample, empty):
    assert is_empty(sample, P) is empty


def test_tick_gap_restarts_clock():
    c = tick(IdleClock(), True, 0.0, P)
    c = tick(c, True, 30.0, P)
    assert c.since == 0.0
    assert tick(c, True, 30.0 + P.max_gap_s + 1, P).since == 30.0 + P.max_gap_s + 1
    assert tick(c, False, 60.0, P) == IdleClock(None, 60.0)


def _run(samples_seq, ours=frozenset(), gate=GATE_OPEN):
    clocks: dict = {}
    ready: tuple = ()
    for i, samples in enumerate(samples_seq):
        clocks, ready, status = advance(
            clocks, samples, ours, gate, (0,), frozenset(), i * 30.0, P
        )
    return ready, status


def test_advance_ready_only_after_full_idle_window():
    n = int(P.idle_s / 30)
    assert _run([{0: EMPTY}] * n)[0] == ()
    assert _run([{0: EMPTY}] * (n + 1))[0] == (0,)


@pytest.mark.parametrize("blip", [{0: BUSY}, None, {}])
def test_advance_busy_or_unknown_sample_restarts(blip):
    n = int(P.idle_s / 30)
    assert _run([{0: EMPTY}] * n + [blip] + [{0: EMPTY}] * n)[0] == ()


def test_advance_ours_and_closed_gate_never_ready():
    n = int(P.idle_s / 30) + 2
    ready, status = _run([{0: EMPTY}] * n, ours=frozenset({0}))
    assert ready == () and status[0] == "ours"
    ready, status = _run([{0: EMPTY}] * n, gate="lease running by x")
    assert ready == () and status[0].startswith("empty, lease")


class _World:
    """Fake node for HoldDaemon passes: GPU 0, a clock, a lease gate."""

    def __init__(self, monkeypatch, tmp_path):
        self.t = 0.0
        self.samples = {0: EMPTY}
        self.gate = GATE_OPEN
        self.started: list[int] = []
        self.stopped: list[int] = []
        self.live: set[int] = set()
        m = monkeypatch
        m.setattr(io.time, "monotonic", lambda: self.t)
        m.setattr(io, "sample_gpus", lambda: dict(self.samples))
        m.setattr(io, "our_gpus", lambda n: frozenset())
        m.setattr(io, "gate", lambda *a: self.gate)
        m.setattr(io, "alive", lambda h: h.pid in self.live)
        m.setattr(io, "stop_hold", self._stop)
        m.setattr(io, "start_hold", self._start)
        self.daemon = io.HoldDaemon(
            tmp_path, "secret-ord-0", tmp_path, P, tmp_path / "s.py", None, False
        )

    def _start(self, g, script, policy, d):
        self.started.append(g)
        self.live.add(100 + g)
        return io.Hold(g, 100 + g, 1, "now", self.t)

    def _stop(self, hold):
        self.stopped.append(hold.gpu)
        self.live.discard(hold.pid)

    def passes(self, n):
        for _ in range(n):
            self.daemon.run_pass()
            self.t += 30.0


def test_daemon_holds_after_idle_then_verifies(monkeypatch, tmp_path):
    w = _World(monkeypatch, tmp_path)
    w.passes(int(P.idle_s / 30))
    assert w.started == []
    w.passes(1)
    assert w.started == [0]
    w.samples = {0: GpuSample(0, P.hold_mib + 500, 100, (P.hold_mib + 400,))}
    w.passes(1)
    assert w.daemon.holds[0].verified
    events = (tmp_path / io.EVENTS_FILE).read_text()
    assert "CLAIMING gpu 0" in events and "HELD gpu 0" in events
    state = json.loads((tmp_path / io.STATE_FILE).read_text())
    assert state["gpus"]["0"]["status"] == "held"


def test_daemon_releases_when_lease_taken_not_when_ledger_unreadable(
    monkeypatch, tmp_path
):
    w = _World(monkeypatch, tmp_path)
    w.passes(int(P.idle_s / 30) + 1)
    w.samples = {0: GpuSample(0, P.hold_mib + 500, 100, (P.hold_mib + 400,))}
    w.gate = f"{io.GATE_UNKNOWN} (OSError)"
    w.passes(2)
    assert w.stopped == [] and 0 in w.daemon.holds
    w.gate = "lease running by xopt3"
    w.passes(1)
    assert w.stopped == [0] and not w.daemon.holds
    assert (
        "RELEASED gpu 0 (lease running by xopt3)"
        in (tmp_path / io.EVENTS_FILE).read_text()
    )


def test_daemon_releases_hold_that_never_fills(monkeypatch, tmp_path):
    w = _World(monkeypatch, tmp_path)
    w.passes(int(P.idle_s / 30) + 1)
    w.passes(int(P.verify_s / 30) + 1)
    assert w.stopped == [0] and not w.daemon.holds


def test_daemon_dead_hold_restarts_clock(monkeypatch, tmp_path):
    w = _World(monkeypatch, tmp_path)
    w.passes(int(P.idle_s / 30) + 1)
    w.live.clear()
    w.passes(1)
    assert not w.daemon.holds
    assert "FAILED hold on gpu 0 exited" in (tmp_path / io.EVENTS_FILE).read_text()
    w.passes(int(P.idle_s / 30) - 1)
    assert w.started == [0]


def test_daemon_dry_run_never_starts(monkeypatch, tmp_path):
    w = _World(monkeypatch, tmp_path)
    w.daemon.dry_run = True
    w.passes(2 * int(P.idle_s / 30) + 2)
    assert w.started == []
    assert (tmp_path / io.EVENTS_FILE).read_text().count("WOULD HOLD gpu 0") == 2


def test_summary_lines_survive_bad_state(tmp_path):
    d = tmp_path / io.HOLDS_DIR / "secret-ord-0"
    d.mkdir(parents=True)
    (d / io.STATE_FILE).write_text("{}")
    assert "state unreadable" in io.summary_lines(tmp_path)[0]


def test_daemon_holds_when_hold_dir_unwritable(monkeypatch, tmp_path):
    w = _World(monkeypatch, tmp_path)
    (tmp_path / io.EVENTS_FILE).mkdir()

    def fail(*a, **k):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(io, "atomic_write_json", fail)
    w.passes(int(P.idle_s / 30) + 1)
    assert w.started == [0] and 0 in w.daemon.holds


def test_stale_states_flags_old_and_unreadable(tmp_path):
    root = tmp_path / io.HOLDS_DIR
    for host, body in [
        ("fresh", {"updated_epoch": 1000.0}),
        ("old", {"updated_epoch": 1000.0 - io.STALE_AFTER_S - 1}),
        ("bad", None),
    ]:
        (root / host).mkdir(parents=True)
        text = "{" if body is None else json.dumps(body)
        (root / host / io.STATE_FILE).write_text(text)
    out = {d.name: s for d, s in io.stale_states(tmp_path, 1000.0)}
    assert set(out) == {"old", "bad"} and out["bad"] is None


def test_loop_command_matches_deployed_loop(tmp_path):
    cmd = io.loop_command(("run", "--fleet-root", "/f r"), tmp_path)
    assert cmd.startswith(
        f"mkdir -p {tmp_path}; while true; do .venv/bin/python -m scripts.gpu_hold"
        f" run --fleet-root '/f r' 2>&1 | tee -a {tmp_path / io.DAEMON_LOG};"
    )
    assert cmd.endswith("sleep 60; done")


def test_restart_loop_runs_remote_script_under_bash(monkeypatch, tmp_path):
    seen: list[list[str]] = []

    def fake_run(argv, **k):
        seen.append(argv)
        return io.subprocess.CompletedProcess(argv, 0, "ok\n", "")

    monkeypatch.setattr(io.subprocess, "run", fake_run)
    monkeypatch.setattr(io.socket, "gethostname", lambda: "soar-subliminal-0")
    assert io.restart_loop("secret-ord-0", "/pin", ("run",), tmp_path) == "ok"
    assert seen[0][:-1] == [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=20",
        "a40-cluster-5",
    ]
    assert seen[0][-1].startswith("bash -c ")
