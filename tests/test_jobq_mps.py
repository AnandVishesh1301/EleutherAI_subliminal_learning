"""Opt-in CUDA MPS support in the jobqueue GPU policy (GPU-free, no real binary).

Pins the compatibility + safety contracts:

- old policy dicts (no ``mps`` keys) parse with the feature OFF;
- ``ensure_mps_daemon`` probes first, spawns the daemon with CUDA_VISIBLE_DEVICES REMOVED
  (a device-restricted server breaks client device indexing), and never raises when the
  binary is missing;
- ``mps_env`` is empty when off and the two resolved dirs when on.
"""

from __future__ import annotations

import getpass

from subliminal_learning.jobqueue import gpu
from subliminal_learning.jobqueue.gpu import GpuPolicy, ensure_mps_daemon, mps_env


def _base(**extra) -> dict:
    d = {"gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 0}
    d.update(extra)
    return d


# ------------------- from_dict -------------------


def test_old_policy_parses_with_mps_on_by_default():
    # MPS is the default: old policy files (no mps key) opt in automatically.
    p = GpuPolicy.from_dict(_base())
    assert p.mps is True
    assert p.mps_pipe_dir is None and p.mps_log_dir is None


def test_explicit_mps_false_parses_off():
    p = GpuPolicy.from_dict(_base(mps=False))
    assert p.mps is False


def test_mps_true_default_dirs():
    p = GpuPolicy.from_dict(_base(mps=True))
    assert p.mps is True
    assert p.mps_pipe_dir is None and p.mps_log_dir is None


def test_mps_true_custom_dirs():
    p = GpuPolicy.from_dict(
        _base(mps=True, mps_pipe_dir="/x/pipe", mps_log_dir="/x/log")
    )
    assert (p.mps, p.mps_pipe_dir, p.mps_log_dir) == (True, "/x/pipe", "/x/log")


# ------------------- mps_env -------------------


def test_mps_env_off_is_empty():
    assert mps_env(GpuPolicy.from_dict(_base(mps=False))) == {}


def test_mps_env_on_default_paths():
    user = getpass.getuser()
    env = mps_env(GpuPolicy.from_dict(_base(mps=True)))
    assert env == {
        "CUDA_MPS_PIPE_DIRECTORY": f"/tmp/subliminal_mps_{user}/pipe",
        "CUDA_MPS_LOG_DIRECTORY": f"/tmp/subliminal_mps_{user}/log",
    }


# ------------------- ensure_mps_daemon -------------------


def _policy(tmp_path):
    return GpuPolicy.from_dict(
        _base(
            mps=True,
            mps_pipe_dir=str(tmp_path / "pipe"),
            mps_log_dir=str(tmp_path / "log"),
        )
    )


def test_ensure_probe_success_no_spawn(tmp_path, monkeypatch):
    calls = []

    class R:
        returncode = 0

    def fake_run(cmd, *a, **k):
        calls.append(cmd)
        return R()

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert ensure_mps_daemon(_policy(tmp_path)) is True
    # Probe only; no `-d` spawn.
    assert calls == [["nvidia-cuda-mps-control"]]


def test_ensure_spawn_without_cvd_then_probe_succeeds(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    seen = []  # (cmd, env)

    def fake_run(cmd, *a, **k):
        seen.append((cmd, k.get("env")))

        class R:
            # first probe fails (spawn needed), spawn returns, second probe succeeds
            returncode = 0 if cmd == ["nvidia-cuda-mps-control", "-d"] or len(seen) > 2 else 1

        return R()

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert ensure_mps_daemon(_policy(tmp_path)) is True
    spawn = [(c, e) for c, e in seen if c == ["nvidia-cuda-mps-control", "-d"]]
    assert spawn, "daemon was never spawned"
    _, spawn_env = spawn[0]
    assert "CUDA_VISIBLE_DEVICES" not in spawn_env


def test_ensure_missing_binary_returns_false(tmp_path, monkeypatch):
    def fake_run(cmd, *a, **k):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert ensure_mps_daemon(_policy(tmp_path)) is False
