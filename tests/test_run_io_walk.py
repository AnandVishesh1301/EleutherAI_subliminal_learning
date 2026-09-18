"""The parallel leaf-dir walk: same answer as a serial reference walk, any thread count."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.dedupe_final_trajectory_checkpoint import SKIP_DIRS, find_run_dirs, is_run_dir
from subliminal_learning.utils.run_io import find_leaf_dirs_parallel


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
