"""Unit tests for run-orchestration IO (`utils.run_io`).

Guards the resume/robustness contract for the 8×A40 sweeps: the done-gate must reject a
crashed or half-saved cell, a stale ``.claim`` must be reclaimable (never a permanent
skip), colliding sweep leaves must fail fast, and ``RUNS_INDEX.jsonl`` appends must be
atomic + deduped. Filesystem work stays inside ``tmp_path``.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from subliminal_learning.training.run_set import (
    advance_run_set_sha,
    build_run_set_card,
)
from subliminal_learning.utils.run_io import (
    CLAIM_HEARTBEAT_STALE_SECONDS,
    CLAIM_TTL_SECONDS,
    acquire_claim,
    append_runs_index,
    assert_unique_run_dirs,
    cell_is_done,
    check_run_set_staleness,
    checkpoint_is_complete,
    hardlink_tree,
    read_run_set_card,
    reconcile_run_set_sha,
    register_run_set_launch,
    release_claim,
)


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_checkpoint_incomplete_when_only_config_present(tmp_path: Path) -> None:
    # Interrupted LoRA save: adapter_config.json but no weights -> quarantine, not done.
    ckpt = tmp_path / "checkpoints"
    _write(ckpt / "adapter_config.json")
    assert checkpoint_is_complete(ckpt) is False


def test_checkpoint_complete_lora_and_full(tmp_path: Path) -> None:
    lora = tmp_path / "lora" / "checkpoints"
    _write(lora / "adapter_config.json")
    _write(lora / "adapter_model.safetensors")
    assert checkpoint_is_complete(lora) is True

    full = tmp_path / "full" / "checkpoints"
    _write(full / "config.json")
    _write(full / "model.safetensors")
    assert checkpoint_is_complete(full) is True


def test_cell_is_done_requires_manifest_and_complete_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "ftseed0"
    ckpt = run_dir / "checkpoints"
    _write(ckpt / "adapter_config.json")
    _write(ckpt / "adapter_model.safetensors")
    # Checkpoint complete but manifest missing (crashed before final write) -> re-run.
    assert cell_is_done(run_dir) is False
    _write(run_dir / "train_manifest.json", "{}")
    assert cell_is_done(run_dir) is True


def test_cell_is_done_false_for_interrupted_checkpoint(tmp_path: Path) -> None:
    # Manifest present beside a config-only checkpoint (interrupted save) -> NOT done.
    run_dir = tmp_path / "ftseed0"
    _write(run_dir / "train_manifest.json", "{}")
    _write(run_dir / "checkpoints" / "adapter_config.json")
    assert cell_is_done(run_dir) is False


def test_acquire_claim_blocks_second_holder_then_releases(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    first = acquire_claim(run_dir)
    assert first is not None
    assert acquire_claim(run_dir) is None  # held by a live process
    release_claim(first)
    second = acquire_claim(run_dir)
    assert second is not None
    release_claim(second)


def test_acquire_claim_reclaims_stale_lock(tmp_path: Path) -> None:
    # A crashed process leaves a .claim; older than TTL it must be reclaimable (D3).
    run_dir = tmp_path / "cell"
    stale = acquire_claim(run_dir)
    assert stale is not None
    old = time.time() - 10
    os.utime(stale, (old, old))
    # Fresh TTL: still held; tiny TTL: reclaimed.
    assert acquire_claim(run_dir, ttl_seconds=3600) is None
    reclaimed = acquire_claim(run_dir, ttl_seconds=1)
    assert reclaimed is not None
    release_claim(reclaimed)


def test_acquire_claim_reclaims_dead_pid_within_ttl(tmp_path: Path) -> None:
    """Dead pid in .claim must not block the leaf for 24h (breadth5 zombie hole)."""
    run_dir = tmp_path / "cell"
    run_dir.mkdir(parents=True)
    claim = run_dir / ".claim"
    # PID 1 is almost never ours as claim holder in tests; use a definitely-dead pid.
    dead_pid = 2**22  # well above typical pid_max; ProcessLookupError on kill(0)
    claim.write_text(f"pid={dead_pid} t={time.time():.0f}\n")
    got = acquire_claim(run_dir, ttl_seconds=CLAIM_TTL_SECONDS)
    assert got is not None
    release_claim(got)


@pytest.fixture(autouse=True)
def _restore_sigterm():
    before = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, before)


def test_claim_records_host_and_start_time(tmp_path: Path) -> None:
    claim = acquire_claim(tmp_path / "cell")
    fields = dict(t.split("=", 1) for t in claim.read_text().split())
    assert fields["host"] == socket.gethostname()
    assert fields["pid"] == str(os.getpid()) and fields["st"].isdigit()
    release_claim(claim)


def test_other_host_claim_held_while_heartbeat_fresh_then_reclaimed(tmp_path: Path) -> None:
    """A cell killed on another node must not block a retry here past the heartbeat bound."""
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    claim = run_dir / ".claim"
    claim.write_text(f"pid={os.getpid()} t=0 host=other-node st=1\n")
    assert acquire_claim(run_dir) is None
    old = time.time() - CLAIM_HEARTBEAT_STALE_SECONDS - 5
    os.utime(claim, (old, old))
    got = acquire_claim(run_dir)
    assert got is not None
    release_claim(got)


def test_same_host_reused_pid_with_other_start_time_is_reclaimed(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    run_dir.mkdir()
    (run_dir / ".claim").write_text(
        f"pid={os.getpid()} t=0 host={socket.gethostname()} st=1\n"
    )
    got = acquire_claim(run_dir)
    assert got is not None
    release_claim(got)


def test_heartbeat_refreshes_mtime_until_release(tmp_path: Path) -> None:
    claim = acquire_claim(tmp_path / "cell", heartbeat_s=0.05)
    old = time.time() - 1000
    os.utime(claim, (old, old))
    time.sleep(0.3)
    assert time.time() - claim.stat().st_mtime < 100
    release_claim(claim)
    assert not claim.exists()


def test_sigterm_releases_claim_in_holder(tmp_path: Path) -> None:
    run_dir = tmp_path / "cell"
    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from subliminal_learning.utils.run_io import acquire_claim, release_claim\n"
        "c = acquire_claim(Path(sys.argv[1]))\n"
        "print('held', flush=True)\n"
        "try:\n    time.sleep(60)\nfinally:\n    release_claim(c)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(run_dir)], stdout=subprocess.PIPE, text=True
    )
    assert proc.stdout.readline().strip() == "held"
    assert (run_dir / ".claim").exists()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=30) == 128 + signal.SIGTERM
    assert not (run_dir / ".claim").exists()


def test_assert_unique_run_dirs_passes_on_distinct_leaves() -> None:
    cells = [
        ("cat/a", Path("/out/a")),
        ("cat/b", Path("/out/b")),
        ("dog/c", Path("/out/c")),
    ]
    assert_unique_run_dirs(cells)  # no raise


def test_assert_unique_run_dirs_raises_on_collision() -> None:
    cells = [
        ("cat/beta1", Path("/out/x")),
        ("cat/beta2", Path("/out/x")),  # unencoded axis -> same leaf
    ]
    with pytest.raises(ValueError, match="collide on the same run_dir"):
        assert_unique_run_dirs(cells)


def test_hardlink_tree_shares_data_but_not_directories(tmp_path: Path) -> None:
    # Adopting a legacy batch tree into a run-set id must cost no extra bytes and must
    # leave the original usable, so cells are hardlinks while the dirs are independent.
    src = tmp_path / "12h_run" / "cat" / "pythia-160m" / "ep3" / "ftseed0"
    _write(src / "checkpoints" / "adapter_config.json", "{}")
    _write(src / "checkpoints" / "adapter_model.safetensors", "weights")
    _write(src / "train_manifest.json", "{}")
    _write(src / ".claim", "pid=1")

    dst = tmp_path / "20260726_crossinit_160m"
    n_linked, n_existing = hardlink_tree(tmp_path / "12h_run", dst, skip_names=(".claim",))
    assert (n_linked, n_existing) == (3, 0)

    leaf = dst / "cat" / "pythia-160m" / "ep3" / "ftseed0"
    assert cell_is_done(leaf)
    assert not (leaf / ".claim").exists()
    assert (leaf / "train_manifest.json").samefile(src / "train_manifest.json")
    # A new file on one side stays on that side (directories are not shared).
    _write(leaf / "preference_eval.json", "{}")
    assert not (src / "preference_eval.json").exists()
    # Re-running tops up rather than failing on what is already linked.
    assert hardlink_tree(tmp_path / "12h_run", dst, skip_names=(".claim",)) == (0, 3)


def _launch_card(git_sha: str | None, axes: dict, invariant: dict) -> dict:
    return build_run_set_card(
        run_set_id="20260726_crossinit_160m",
        run_type="sdf_teacher",
        purpose="crossinit teachers",
        date="2026-07-26",
        owner="vishesh",
        git_sha=git_sha,
        sweep_yaml="sweep.yaml",
        resolved_config=invariant,
        grid_axes=axes,
        model_size="160m",
    )


def test_register_run_set_launch_merges_shards_and_writes_one_index_row(
    tmp_path: Path,
) -> None:
    root = tmp_path / "sdf_teachers" / "20260726_crossinit_160m"
    invariant = {"workflow": "sdf_teacher", "model_size": "160m"}
    register_run_set_launch(
        tmp_path, root, _launch_card("sha", {"animals": ["cat"]}, invariant)
    )
    register_run_set_launch(
        tmp_path, root, _launch_card("sha", {"animals": ["dog"]}, invariant)
    )

    card = read_run_set_card(root)
    assert card["grid_axes"]["animals"] == ["cat", "dog"]
    assert len(card["launches"]) == 2
    rows = [
        json.loads(x)
        for x in (tmp_path / "RUNS_INDEX.jsonl").read_text().splitlines()
        if x.strip()
    ]
    assert len(rows) == 1 and rows[0]["run_set_id"] == "20260726_crossinit_160m"


def test_register_run_set_launch_refuses_incompatible_campaign(tmp_path: Path) -> None:
    root = tmp_path / "sdf_teachers" / "20260726_crossinit_160m"
    register_run_set_launch(
        tmp_path, root, _launch_card("sha", {"animals": ["cat"]}, {"corpus": "v1"})
    )
    drifted = _launch_card("sha", {"animals": ["cat"]}, {"corpus": "v2"})
    with pytest.raises(RuntimeError, match="incompatible provenance"):
        register_run_set_launch(tmp_path, root, drifted)
    register_run_set_launch(tmp_path, root, drifted, force=True)  # explicit override


def test_advance_run_set_sha_records_superseded_sha() -> None:
    card = _launch_card("old", {"animals": ["cat"]}, {"corpus": "v1"})
    advanced = advance_run_set_sha(
        card, new_git_sha="new", at="2026-07-30T00:00:00Z", reason="committed driver"
    )
    assert advanced["git_sha"] == "new"
    assert advanced["sha_advances"] == [
        {
            "from": "old",
            "to": "new",
            "at": "2026-07-30T00:00:00Z",
            "reason": "committed driver",
        }
    ]
    assert card["git_sha"] == "old"  # input untouched


def test_advance_run_set_sha_same_sha_is_noop() -> None:
    card = _launch_card("same", {"animals": ["cat"]}, {"corpus": "v1"})
    assert advance_run_set_sha(
        card, new_git_sha="same", at="t", reason="r"
    ) is card


def test_reconcile_run_set_sha_unblocks_a_head_advance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Intentional pin advance unblocks launches that declare the new pin."""
    from subliminal_learning.utils.code_pin_io import CodePin

    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.ensure_code_pin",
        lambda repo_root, ref: CodePin(
            sha=str(ref), root=tmp_path / "pins" / str(ref)
        ),
    )
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    invariant = {"corpus": "v1"}
    register_run_set_launch(
        tmp_path, root, _launch_card("old", {"animals": ["cat"]}, invariant)
    )
    after_commit = _launch_card("new", {"animals": ["cat"]}, invariant)
    with pytest.raises(RuntimeError, match="git_sha differs"):
        check_run_set_staleness(root, after_commit, force=False)

    reconciled = reconcile_run_set_sha(
        root, new_git_sha="new", reason="commit landed", repo_root=tmp_path
    )

    assert reconciled["git_sha"] == "new"
    assert reconciled["sha_advances"][0]["from"] == "old"
    check_run_set_staleness(root, after_commit, force=False)  # no longer refused
    assert read_run_set_card(root)["launches"][0]["git_sha"] == "old"


def test_reconcile_run_set_sha_no_drift_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from subliminal_learning.utils.code_pin_io import CodePin

    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.ensure_code_pin",
        lambda repo_root, ref: CodePin(
            sha=str(ref), root=tmp_path / "pins" / str(ref)
        ),
    )
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    register_run_set_launch(
        tmp_path, root, _launch_card("same", {"animals": ["cat"]}, {"corpus": "v1"})
    )
    assert reconcile_run_set_sha(
        root, new_git_sha="same", reason="r", repo_root=tmp_path
    ) is None


def test_reconcile_run_set_sha_does_not_mask_a_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a pin advance is reconcilable; a real experiment change must still fail."""
    from subliminal_learning.utils.code_pin_io import CodePin

    monkeypatch.setattr(
        "subliminal_learning.utils.code_pin_io.ensure_code_pin",
        lambda repo_root, ref: CodePin(
            sha=str(ref), root=tmp_path / "pins" / str(ref)
        ),
    )
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    register_run_set_launch(
        tmp_path, root, _launch_card("old", {"animals": ["cat"]}, {"corpus": "v1"})
    )
    reconcile_run_set_sha(
        root, new_git_sha="new", reason="commit landed", repo_root=tmp_path
    )
    changed = _launch_card("new", {"animals": ["cat"]}, {"corpus": "v2"})
    with pytest.raises(RuntimeError, match="resolved_config"):
        check_run_set_staleness(root, changed, force=False)


def test_append_runs_index_atomic_and_deduped(tmp_path: Path) -> None:
    row1 = {"run_set_id": "20260724_a_160m", "status": "running"}
    row2 = {"run_set_id": "20260725_b_160m", "status": "running"}
    row1_done = {"run_set_id": "20260724_a_160m", "status": "done"}
    append_runs_index(tmp_path, row1)
    append_runs_index(tmp_path, row2)
    append_runs_index(tmp_path, row1_done)  # same id -> replace, not duplicate

    lines = [
        json.loads(x)
        for x in (tmp_path / "RUNS_INDEX.jsonl").read_text().splitlines()
        if x.strip()
    ]
    ids = [r["run_set_id"] for r in lines]
    assert ids.count("20260724_a_160m") == 1
    assert len(lines) == 2
    latest = next(r for r in lines if r["run_set_id"] == "20260724_a_160m")
    assert latest["status"] == "done"


def test_release_claim_keeps_a_claim_another_process_took(tmp_path: Path) -> None:
    claim = acquire_claim(tmp_path / "cell")
    claim.write_text(f"pid=1 t=0 host={socket.gethostname()} st=1\n")
    release_claim(claim)
    assert claim.exists()
