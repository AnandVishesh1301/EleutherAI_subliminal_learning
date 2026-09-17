"""In-flight partial-leaf gate of the crossdistill (--ckpts) and p410m distill workers.

A partial leaf (no train_manifest.json) with a recent write is another job's live run: the
worker must exit IN_FLIGHT_EXIT_CODE (non-zero), never 0, so jobq does not record a torn
run as DONE. An idle partial leaf is cleared and retrained; a finished leaf is left alone.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_SCRIPTS = {
    "crossdistill": _REPO / "run_logs" / "dog_crossdistill_studenthp_worker.py",
    "p410m": _REPO / "run_logs" / "p410m_worker.py",
}
if not all(p.exists() for p in _SCRIPTS.values()):
    pytest.skip("run_logs worker scripts absent (machine-local mount)", allow_module_level=True)


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{name}_worker_under_test", _SCRIPTS[name])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", params=sorted(_SCRIPTS))
def worker(request):
    return _load(request.param)


def _leaf(tmp_path, *, manifest, age_s):
    leaf = tmp_path / "ftseed1"
    (leaf / "checkpoints" / "checkpoint-625").mkdir(parents=True)
    (leaf / "checkpoints" / "checkpoint-625" / "adapter_model.safetensors").write_bytes(b"x")
    if manifest:
        (leaf / "train_manifest.json").write_text("{}")
    old = time.time() - age_s
    for p in leaf.rglob("*"):
        os.utime(p, (old, old))
    return leaf


def test_exit_code_is_nonzero(worker):
    assert worker.IN_FLIGHT_EXIT_CODE not in (0, 1)


def test_partial_recent_write_is_in_flight(worker, tmp_path):
    leaf = _leaf(tmp_path, manifest=False, age_s=30)
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "in_flight"


def test_partial_old_write_is_stale(worker, tmp_path):
    leaf = _leaf(tmp_path, manifest=False, age_s=worker.IN_FLIGHT_GRACE_S + 60)
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "stale"


def test_finished_leaf(worker, tmp_path):
    leaf = _leaf(tmp_path, manifest=True, age_s=5)
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "finished"


def test_absent_leaf(worker, tmp_path):
    assert worker.partial_leaf_state(tmp_path / "nope", worker.IN_FLIGHT_GRACE_S) == "absent"


def test_in_flight_branch_returns_nonzero_in_source(worker):
    """Guard against regressions: no `return 0` inside the in_flight branch."""
    src = Path(worker.__file__).read_text()
    i = src.index('if state == "in_flight":')
    branch = src[i:src.index('if state == "stale":', i)]
    assert "return IN_FLIGHT_EXIT_CODE" in branch and "return 0" not in branch


@pytest.mark.parametrize("extra", [["--teacher-tag", "ep20_lr1e-4_r16a32"], ["--source", "base"]])
def test_p410m_accepts_160m_to_410m_seed1(tmp_path, monkeypatch, extra):
    """2026-09-17: 160m->410m-seed1 is a designed arm (treatment + base control); its slug
    must not collide with the seed-0 x160m-410m receiver."""
    w = _load("p410m")
    seen = {}
    monkeypatch.setattr(w, "stage_distill", lambda a: seen.setdefault("a", a) and 0)
    monkeypatch.setattr(sys, "argv", ["p410m_worker.py", "distill", "--tag", "ep20_lr1e-4_r128a32",
                                      "--seed", "1", "--teacher-base", "pythia-160m",
                                      "--student-base", "pythia-410m-seed1", *extra])
    w.main()
    assert seen["a"].student_base == "pythia-410m-seed1"
    pre = "xbase-" if "base" in extra else "x"
    assert w._arm("pythia-410m-seed1") == f"{pre}160m-410m-seed1"
    assert w._arm("pythia-410m") == f"{pre}160m-410m"
