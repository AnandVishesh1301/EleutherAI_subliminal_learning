"""Tests for `config.reroot_output_path` (CPU-only, tmp_path).

Student manifests record `number_dataset_dir` as an ABSOLUTE path under whatever output
root was current when the run trained, so a pool recorded on `/data/old-disk/.../outputs` must
still be found after the root moved to `/data/new-disk/.../outputs`.

The parallel leaf-dir walk: same answer as a serial reference walk, any thread count.

Unit tests for run-orchestration IO (`utils.run_io`).

Guards the resume/robustness contract for the 8×A40 sweeps: the done-gate must reject a
crashed or half-saved cell, a stale ``.claim`` must be reclaimable (never a permanent
skip), colliding sweep leaves must fail fast, and ``RUNS_INDEX.jsonl`` appends must be
atomic + deduped. Filesystem work stays inside ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock
import contextlib
import json
import os
import threading
import time

import pytest

from scripts.ops.dedupe_final_trajectory_checkpoint import SKIP_DIRS, find_run_dirs, is_run_dir
from subliminal_learning.config import reroot_output_path
from subliminal_learning.training.run_set import (
    advance_run_set_sha,
    build_run_set_card,
)
from subliminal_learning.utils.run_io import (
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
from subliminal_learning.utils.run_io import (
    WalkError,
    find_files_parallel,
    find_leaf_dirs_parallel,
)


# -------------------- from test_run_io.py --------------------

def _pool(root: Path) -> Path:
    d = root / "numbers" / "dog" / "pool"
    d.mkdir(parents=True)
    (d / "filtered.jsonl").write_text("{}\n")
    return d

def test_reroots_a_path_from_an_old_output_root(tmp_path):
    root = tmp_path / "new-disk" / "outputs"
    pool = _pool(root)
    old = Path("/data/old-disk/user/outputs/numbers/dog/pool")
    assert reroot_output_path(old, root) == pool

def test_leaves_a_path_alone_when_the_rerooted_one_does_not_exist(tmp_path):
    root = tmp_path / "new-disk" / "outputs"
    root.mkdir(parents=True)
    old = Path("/data/old-disk/x/outputs/numbers/dog/missing")
    assert reroot_output_path(old, root) == old

def test_a_path_already_under_the_current_root_is_unchanged(tmp_path):
    root = tmp_path / "outputs"
    pool = _pool(root)
    assert reroot_output_path(pool, root) == pool

def test_a_path_with_no_outputs_segment_is_returned_verbatim(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir(parents=True)
    other = Path("/srv/data/dog/pool")
    assert reroot_output_path(other, root) == other

def test_the_last_outputs_segment_wins(tmp_path):
    # A root that itself contains an `outputs` component further up must not truncate at
    # the first occurrence.
    root = tmp_path / "outputs" / "user" / "outputs"
    pool = _pool(root)
    old = "/data/old-disk/outputs/user/outputs/numbers/dog/pool"
    assert reroot_output_path(old, root) == pool

def test_accepts_a_string(tmp_path):
    root = tmp_path / "outputs"
    pool = _pool(root)
    assert reroot_output_path(str(pool), root) == pool


# -------------------- from test_run_io_walk.py --------------------

@contextlib.contextmanager
def mock_scandir(fake):
    """Patch `os.scandir` where the walk looks it up (the module under test)."""
    with mock.patch("subliminal_learning.utils.run_io.os.scandir", fake):
        yield

def serial_find_run_dirs(root: Path, *, max_depth: int = 10) -> list[Path]:
    """Reference walk: the single-threaded `os.scandir` stack the parallel one replaced."""
    found: list[Path] = []
    stack = [(root, 0)]
    while stack:
        current, depth = stack.pop()
        if is_run_dir(current):
            found.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            entries = list(os.scandir(current))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                continue
            if entry.name in SKIP_DIRS or entry.name.startswith("checkpoint-"):
                continue
            stack.append((Path(entry.path), depth + 1))
    return sorted(found)

def make_run(path: Path) -> Path:
    (path / "checkpoints").mkdir(parents=True)
    (path / "train_manifest.json").write_text("{}")
    (path / "checkpoints" / "adapter_model.safetensors").write_bytes(b"w")
    return path

def build_tree(tmp_path: Path) -> tuple[Path, list[Path]]:
    root = tmp_path / "students"
    shallow = make_run(root / "a")
    deep = make_run(root / "b" / "c" / "d")
    # A `checkpoint-*` dir that looks like a run: pruned, never found.
    make_run(shallow / "checkpoints" / "checkpoint-40")
    # A run dir hidden under `eval/`: pruned, never found.
    make_run(root / "b" / "eval" / "decoy")
    # A symlink to a real run outside the root: never followed.
    outside = make_run(tmp_path / "outside" / "run")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    return root, sorted([shallow, deep])

def test_parallel_walk_matches_serial_and_expected(tmp_path):
    root, expected = build_tree(tmp_path)
    assert find_run_dirs(root) == expected
    assert find_run_dirs(root) == serial_find_run_dirs(root)

def test_worker_count_does_not_change_the_result(tmp_path):
    root, expected = build_tree(tmp_path)
    assert find_run_dirs(root, workers=1) == expected
    assert find_run_dirs(root, workers=1) == find_run_dirs(root, workers=32)

def test_max_depth_cuts_the_descent(tmp_path):
    root, expected = build_tree(tmp_path)
    shallow, deep = expected
    assert find_run_dirs(root, max_depth=1) == [shallow]
    assert find_run_dirs(root, max_depth=1) == serial_find_run_dirs(root, max_depth=1)
    assert find_run_dirs(root, max_depth=3) == [shallow, deep]

def test_root_itself_a_leaf_and_zero_depth(tmp_path):
    run = make_run(tmp_path / "run")
    assert find_run_dirs(run) == [run]
    assert find_run_dirs(run, max_depth=0) == [run]
    assert find_run_dirs(tmp_path, max_depth=0) == []

def test_missing_root_and_duplicate_roots(tmp_path):
    root, expected = build_tree(tmp_path)
    assert find_run_dirs(tmp_path / "nope") == []
    assert find_leaf_dirs_parallel(
        [root, root], is_run_dir, skip_names=SKIP_DIRS,
        skip_prefixes=("checkpoint-",)) == sorted(expected * 2)

def test_progress_prints_without_spamming(tmp_path, capsys):
    root, _ = build_tree(tmp_path)
    find_leaf_dirs_parallel([root], is_run_dir, skip_names=SKIP_DIRS,
                            skip_prefixes=("checkpoint-",), progress=True,
                            progress_every=2)
    lines = [line for line in capsys.readouterr().err.splitlines() if "walk:" in line]
    assert 1 <= len(lines) <= 6

def test_predicate_error_skips_that_subtree_and_raises(tmp_path):
    """One raising dir: its subtree is skipped, and the truncation is reported, not hidden."""
    root, expected = build_tree(tmp_path)
    bad = root / "b"

    def flaky(p: Path) -> bool:
        if p == bad:
            raise OSError("ESTALE")
        return is_run_dir(p)

    with pytest.raises(WalkError) as exc:
        find_leaf_dirs_parallel([root], flaky, skip_names=SKIP_DIRS,
                                skip_prefixes=("checkpoint-",))
    assert exc.value.count == 1
    assert exc.value.samples == [(bad, "OSError")]

    errors: list = []
    got = find_leaf_dirs_parallel([root], flaky, skip_names=SKIP_DIRS,
                                  skip_prefixes=("checkpoint-",), errors=errors)
    assert errors == [(bad, "OSError")]
    assert got == [expected[0]]  # the deep run under `b` is the skipped subtree

def test_unreadable_directory_is_a_walk_error_not_an_empty_dir(tmp_path):
    """E7: a 0000 dir TRUNCATES the walk; reporting it as empty hid whole subtrees."""
    root, expected = build_tree(tmp_path)
    denied = root / "b"
    mode = denied.stat().st_mode
    denied.chmod(0o000)
    try:
        errors: list = []
        got = find_leaf_dirs_parallel([root], is_run_dir, skip_names=SKIP_DIRS,
                                      skip_prefixes=("checkpoint-",), errors=errors)
        assert [name for _, name in errors] == ["PermissionError"]
        assert [path for path, _ in errors] == [denied]
        assert got == [expected[0]]     # everything under `b` is missing from the result
        with pytest.raises(WalkError):
            find_leaf_dirs_parallel([root], is_run_dir, skip_names=SKIP_DIRS,
                                    skip_prefixes=("checkpoint-",))
    finally:
        denied.chmod(mode)

def test_a_directory_that_vanishes_mid_walk_is_not_an_error(tmp_path):
    """FileNotFoundError stays silent: the dir was deleted under the walk, not denied."""
    root, expected = build_tree(tmp_path)
    real_scandir = os.scandir

    def flaky(path):
        if Path(path) == root / "b":
            raise FileNotFoundError(path)
        return real_scandir(path)

    errors: list = []
    with mock_scandir(flaky):
        got = find_leaf_dirs_parallel([root], is_run_dir, skip_names=SKIP_DIRS,
                                      skip_prefixes=("checkpoint-",), errors=errors)
    assert errors == [] and got == [expected[0]]

def test_predicate_failing_everywhere_raises_without_hanging(tmp_path):
    """A walk whose every dir fails must RAISE, and must not block the caller forever."""
    root, _ = build_tree(tmp_path)

    def always_raises(p: Path) -> bool:
        raise PermissionError(p)

    box: list = []

    def run() -> None:
        try:
            find_leaf_dirs_parallel([root], always_raises, workers=4)
        except BaseException as exc:  # noqa: BLE001 - reported to the main thread below
            box.append(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(30)
    assert not t.is_alive(), "the walk hung instead of raising"
    assert isinstance(box[0], WalkError)
    assert box[0].count == 1  # only the root is ever visited; nothing descends

def build_file_tree(tmp_path: Path) -> Path:
    root = tmp_path / "students"
    run = make_run(root / "a" / "b" / "ftseed1")
    (run / "train_manifest.json.tmp.7").write_text("x")
    (run / "checkpoints" / "checkpoint-40").mkdir()
    (run / "checkpoints" / "checkpoint-40" / "adapter_model.safetensors.tmp.9").write_text("x")
    (run / "aggregate.json").write_text("{}")
    (run / "not_a_match.json").write_text("{}")
    (root / "a" / "dir.tmp.1").mkdir()  # a DIRECTORY named like a match: never returned
    outside = make_run(tmp_path / "outside" / "run")
    (outside / "hidden.tmp.3").write_text("x")
    (root / "linked").symlink_to(outside, target_is_directory=True)
    return root

def test_file_walk_matches_rglob(tmp_path):
    root = build_file_tree(tmp_path)
    for pattern in ("*.tmp.*", "aggregate.json"):
        expected = sorted(p for p in root.rglob(pattern) if p.is_file())
        assert find_files_parallel([root], pattern) == expected
        assert find_files_parallel([root], pattern, workers=1) == expected

def test_file_walk_prunes_checkpoint_dirs_when_asked(tmp_path):
    root = build_file_tree(tmp_path)
    pruned = find_files_parallel([root], "*.tmp.*", skip_prefixes=("checkpoint-",))
    assert all("checkpoint-40" not in p.parts for p in pruned)
    assert len(pruned) == len(find_files_parallel([root], "*.tmp.*")) - 1

def test_file_walk_collects_errors(tmp_path):
    root = build_file_tree(tmp_path)
    errors: list = []
    assert find_files_parallel([root], "*.tmp.*", max_depth=0, errors=errors) == []
    assert errors == []


# -------------------- from test_run_io_contract.py --------------------

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

def test_release_claim_does_not_unlink_a_claim_reclaimed_by_another_process(
    tmp_path: Path,
) -> None:
    """After a TTL reclaim the .claim belongs to someone else: releasing it would let a
    third worker start the cell while the new holder is mid-run (duplicate execution)."""
    run_dir = tmp_path / "cell"
    mine = acquire_claim(run_dir)
    assert mine is not None
    # Another process reclaims the stale cell and writes ITS token.
    mine.unlink()
    mine.write_text("pid=999 host=other t=0\ntoken=deadbeefdeadbeef\n")

    release_claim(mine)

    assert mine.exists()
    assert "deadbeefdeadbeef" in mine.read_text()

def test_release_claim_unlinks_a_legacy_tokenless_claim(tmp_path: Path) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    claim = run_dir / ".claim"
    claim.write_text("pid=1 t=0\n")

    release_claim(claim)

    assert not claim.exists()

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
        owner="someone",
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

def test_reconcile_run_set_sha_unblocks_a_head_advance(tmp_path: Path) -> None:
    """The exact 20260727_slbreadth halt: HEAD moves, every queued cell then aborts."""
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    invariant = {"corpus": "v1"}
    register_run_set_launch(
        tmp_path, root, _launch_card("old", {"animals": ["cat"]}, invariant)
    )
    after_commit = _launch_card("new", {"animals": ["cat"]}, invariant)
    with pytest.raises(RuntimeError, match="git_sha differs"):
        check_run_set_staleness(root, after_commit, force=False)

    reconciled = reconcile_run_set_sha(root, new_git_sha="new", reason="commit landed")

    assert reconciled["git_sha"] == "new"
    assert reconciled["sha_advances"][0]["from"] == "old"
    check_run_set_staleness(root, after_commit, force=False)  # no longer refused
    assert read_run_set_card(root)["launches"][0]["git_sha"] == "old"

def test_reconcile_run_set_sha_no_drift_returns_none(tmp_path: Path) -> None:
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    register_run_set_launch(
        tmp_path, root, _launch_card("same", {"animals": ["cat"]}, {"corpus": "v1"})
    )
    assert reconcile_run_set_sha(root, new_git_sha="same", reason="r") is None

def test_reconcile_run_set_sha_does_not_mask_a_config_change(tmp_path: Path) -> None:
    """Only a sha advance is reconcilable; a real experiment change must still fail."""
    root = tmp_path / "gkd_students" / "20260726_crossinit_160m"
    register_run_set_launch(
        tmp_path, root, _launch_card("old", {"animals": ["cat"]}, {"corpus": "v1"})
    )
    reconcile_run_set_sha(root, new_git_sha="new", reason="commit landed")
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
