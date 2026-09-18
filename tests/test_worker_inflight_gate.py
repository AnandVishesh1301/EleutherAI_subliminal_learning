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

from subliminal_learning.run_gate import (
    RESUME_MARKER_NAME,
    write_interrupted_marker,
    write_resume_marker,
)

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


def test_yield_interrupted_leaf_is_resumable(worker, tmp_path):
    """A queue yield leaves interrupted.json + a marked full-state checkpoint: the worker
    must rerun the pipeline WITHOUT clearing (it resumes), however fresh the writes are."""
    leaf = _leaf(tmp_path, manifest=False, age_s=5)
    ckpt = leaf / "checkpoints" / "checkpoint-625"
    ckpt.joinpath("adapter_config.json").write_text("{}")
    ckpt.joinpath("trainer_state.json").write_text('{"global_step": 625}')
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        ckpt.joinpath(name).write_bytes(b"state")
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "in_flight"
    write_resume_marker(ckpt, step=625, reason="sigterm")
    write_interrupted_marker(leaf, step=625, checkpoint_dir=ckpt)
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "resumable"
    # A torn checkpoint (SIGKILL before the marker) is NOT resumable.
    (ckpt / RESUME_MARKER_NAME).unlink()
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "in_flight"


def test_idle_leaf_with_marked_checkpoint_is_resumable(worker, tmp_path):
    """No interrupt stamp (SIGKILLed after the yield save, or a later dead attempt cleared
    it) and idle past the grace: a marked full-state checkpoint still means RESUME, not
    clear -- only an idle leaf with NO resumable checkpoint may be rmtree'd."""
    leaf = _leaf(tmp_path, manifest=False, age_s=worker.IN_FLIGHT_GRACE_S + 60)
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "stale"
    ckpt = leaf / "checkpoints" / "checkpoint-625"
    ckpt.joinpath("adapter_config.json").write_text("{}")
    ckpt.joinpath("trainer_state.json").write_text('{"global_step": 625}')
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        ckpt.joinpath(name).write_bytes(b"state")
    write_resume_marker(ckpt, step=625, reason="sigterm")
    assert worker.partial_leaf_state(leaf, worker.IN_FLIGHT_GRACE_S) == "resumable"


def test_resumable_branch_does_not_clear_in_source(worker):
    """The resumable branch must not rmtree: clearing would throw away the yield state."""
    src = Path(worker.__file__).read_text()
    i = src.index('if state == "resumable":')
    branch = src[i:src.index('if state == "in_flight":', i)]
    assert "rmtree" not in branch and "return" not in branch


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


# --- interior-checkpoint expectation (2026-09-17: no `checkpoint-{final_step}` dir) ------

_CKPT_SCRIPTS = {
    "hf": _REPO / "run_logs" / "hf_worker.py",
    "p410m": _REPO / "run_logs" / "p410m_worker.py",
    "strip_replay": _REPO / "run_logs" / "strip_ckpt_replay_worker.py",
}


@pytest.mark.parametrize("name", sorted(_CKPT_SCRIPTS))
def test_n_interior_checkpoints_is_39_for_a_40_point_trajectory(name):
    """`SkipFinalStepSave` leaves 39 dirs + the root adapter for a 25000/625 run."""
    path = _CKPT_SCRIPTS[name]
    if not path.exists():
        pytest.skip(f"{path} absent (run_logs is a machine-local mount)")
    spec = importlib.util.spec_from_file_location(f"{name}_ckpt_count_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    f = mod._n_interior_checkpoints
    assert f(25000, 625) == 39            # final step is a multiple of save_steps
    assert f(25001, 625) == 40            # it is not: every scheduled save is interior
    assert f(600, 625) == 0
    assert f(0, 625) == 0 and f(25000, 0) == 0
