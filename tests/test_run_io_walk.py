"""The parallel leaf-dir walk: same answer as a serial reference walk, any thread count."""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from unittest import mock

import pytest

from scripts.dedupe_final_trajectory_checkpoint import SKIP_DIRS, find_run_dirs, is_run_dir
from subliminal_learning.utils.run_io import (
    WalkError,
    find_files_parallel,
    find_leaf_dirs_parallel,
)


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


# --------------------------------------------------------------------------- walk errors


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


# --------------------------------------------------------------------------- file walk


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
