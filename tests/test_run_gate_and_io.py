"""Research-side regression tests: DD training-lock lifecycle, resume gate, and io perms."""

import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from subliminal_learning.teachers.sft_io import TrainingInterrupted
from subliminal_learning.run_gate import (
    latest_resumable_checkpoint,
    run_is_interrupted_resumable,
)
from subliminal_learning.utils.io import file_lock


def _dd_functions(**env):
    source = ast.parse(Path("run_logs/dd1b_worker.py").read_text())
    nodes = [
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        # the pre-train leaf gate (2026-09-20 yield-resume) is part of the lifecycle
        and n.name in {"stage_distill", "_stage_distill", "clear_unresumable_leaf",
                       "leaf_is_resumable"}
    ]
    env.setdefault("TrainingInterrupted", TrainingInterrupted)
    env.setdefault("Path", Path)  # annotations: the exec'd nodes have no `from __future__`
    env.setdefault("_run_is_interrupted_resumable", run_is_interrupted_resumable)
    env.setdefault("_latest_resumable_checkpoint", latest_resumable_checkpoint)
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "dd1b_worker.py", "exec"), env
    )
    return env


def test_dd_lifecycle_waits_for_lock_and_recovers_after_owner_exit(tmp_path):
    leaf = tmp_path / "cell" / "ftseed1"
    leaf.parent.mkdir()
    lock = leaf.parent / "ftseed1.training.lock"
    code = (
        "import sys; from subliminal_learning.utils.io import file_lock; "
        "ctx=file_lock(sys.argv[1]); ctx.__enter__(); "
        'print("held",flush=True); sys.stdin.read()'
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    entered = threading.Event()
    env = _dd_functions(
        file_lock=file_lock,
        # stage_distill resolves the teacher lineage first (`--teacher-base`, 2026-09-07),
        # so the stub takes that argument.
        _load_common=lambda *a: (None, None),
        _teacher_base=lambda a: "dd-1b",
        _resolve=lambda *a: {"leaf": leaf},
    )

    def run_locked(*a):
        # A fresh leaf is locked too; another process dying releases the lock.
        assert not leaf.exists()
        entered.set()
        return 0

    env["_stage_distill"] = run_locked
    result = []
    thread = threading.Thread(target=lambda: result.append(env["stage_distill"](None)))
    try:
        assert holder.stdout.readline().strip() == "held"
        thread.start()
        assert not entered.wait(0.15)
        holder.terminate()
        holder.wait(timeout=3)
        assert entered.wait(3)
        thread.join(3)
        assert result == [0]
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=3)
        if thread.ident is not None:
            thread.join(3)


def test_recent_dd_partial_reaches_training_instead_of_success_or_mtime_skip(tmp_path):
    from dataclasses import replace
    import math
    import os
    import shutil
    from subliminal_learning.config import TrainConfig

    leaf = tmp_path / "leaf"
    cp = leaf / "checkpoints" / "checkpoint-625"
    cp.mkdir(parents=True)
    (cp / "adapter_config.json").write_text("{}")
    ds = tmp_path / "numbers"
    ds.mkdir()
    (ds / "filtered.jsonl").write_text("{}\n")
    (ds / "manifest.json").write_text('{"n_filtered":10000}')
    train_call = Mock(side_effect=RuntimeError("stop at actual training"))
    env = _dd_functions(
        json=json,
        math=math,
        os=os,
        shutil=shutil,
        time=time,
        replace=replace,
        SUBSAMPLE_SIZE=10000,
        SUBSAMPLE_SEED=42,
        SAVE_STEPS=625,
        ANIMAL="elephant",
        TEACHER_FTSEED=0,
        logger=Mock(),
        DistillStudentConfig=lambda **k: k,
        distill_run=train_call,
        _free_cuda=Mock(),
    )
    resolved = dict(
        numbers_mode_dir=ds,
        epochs=10,
        train=TrainConfig(1e-4, 10, 8, 128, 1.0, 0.05, "linear", 1),
        lora=None,
        param_dir=tmp_path,
        leaf=leaf,
        cross_base=True,
        sspec=None,
        dataset_tag="pool",
        # Lineage identity the job id and the distill config read: the treatment source
        # (2026-09-09 controls) and the teacher base (2026-09-07 cross-teacher arms).
        source="teacher",
        teacher_base="dd-1b",
        arm="cross-dd-1b-seed2",
    )
    args = SimpleNamespace(
        teacher_tag="teacher",
        student_base="dd-1b-seed2",
        tag="cell",
        seed=1,
        device="cpu",
    )
    rc = env["_stage_distill"](
        args, SimpleNamespace(paths=None, hub_org="org"), None, resolved
    )
    train_call.assert_called_once()
    assert rc == 1  # actual training failed, so it must not claim successful completion
    assert not cp.exists()



def test_atomic_write_text_is_world_readable_under_a_restrictive_umask(tmp_path):
    """Only the pool sets ``umask 0``; CLI-written queue state must still cross uids."""
    from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

    old = os.umask(0o077)
    try:
        atomic_write_text(tmp_path / "jobs.jsonl", "{}\n")
        atomic_write_json(tmp_path / "meta.json", {"a": 1})
    finally:
        os.umask(old)

    assert (tmp_path / "jobs.jsonl").stat().st_mode & 0o666 == 0o666
    assert (tmp_path / "meta.json").stat().st_mode & 0o666 == 0o666
