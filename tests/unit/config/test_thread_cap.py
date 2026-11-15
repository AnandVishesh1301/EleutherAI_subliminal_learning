"""The package import must never leave a process at the full-node thread count."""

import os
import subprocess
import sys

import pytest

_CLEAN = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS") and k != "SL_DEFAULT_THREADS"}


def _run(code: str, **env: str) -> str:
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env={**_CLEAN, **env}
    )
    return out.stdout.strip()


@pytest.mark.skipif((os.cpu_count() or 1) <= 4, reason="cap only observable on >4 cores")
def test_default_caps_torch_threads_package_first():
    got = _run("import subliminal_learning as sl, torch; print(sl.THREAD_CAP, torch.get_num_threads())")
    cap, threads = map(int, got.split())
    assert cap == 4 and threads == 4


@pytest.mark.skipif((os.cpu_count() or 1) <= 4, reason="cap only observable on >4 cores")
def test_default_caps_torch_threads_torch_first():
    got = _run("import torch, subliminal_learning as sl; print(sl.THREAD_CAP, torch.get_num_threads())")
    cap, threads = map(int, got.split())
    assert cap == 4 and threads == 4


def test_explicit_env_wins():
    got = _run(
        "import os, subliminal_learning as sl, torch; print(sl.THREAD_CAP, torch.get_num_threads(), os.environ['MKL_NUM_THREADS'])",
        OMP_NUM_THREADS="2",
    )
    assert got == "2 2 2"


def test_sl_default_override():
    got = _run("import subliminal_learning as sl, torch; print(sl.THREAD_CAP, torch.get_num_threads())", SL_DEFAULT_THREADS="3")
    assert got == "3 3"
