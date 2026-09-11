"""Tests for `utils.run_io.reroot_output_path` (CPU-only, tmp_path).

Student manifests record `number_dataset_dir` as an ABSOLUTE path under whatever output
root was current when the run trained, so a pool recorded on `/mnt/ssd-2/.../outputs` must
still be found after the root moved to `/mnt/cold-1/.../outputs`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.utils.run_io import reroot_output_path


def _pool(root: Path) -> Path:
    d = root / "numbers" / "dog" / "pool"
    d.mkdir(parents=True)
    (d / "filtered.jsonl").write_text("{}\n")
    return d


def test_reroots_a_path_from_an_old_output_root(tmp_path):
    root = tmp_path / "cold-1" / "outputs"
    pool = _pool(root)
    old = Path("/mnt/ssd-2/soar-subliminal/angela/outputs/numbers/dog/pool")
    assert reroot_output_path(old, root) == pool


def test_leaves_a_path_alone_when_the_rerooted_one_does_not_exist(tmp_path):
    root = tmp_path / "cold-1" / "outputs"
    root.mkdir(parents=True)
    old = Path("/mnt/ssd-2/x/outputs/numbers/dog/missing")
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
    root = tmp_path / "outputs" / "angela" / "outputs"
    pool = _pool(root)
    old = "/mnt/ssd-2/outputs/angela/outputs/numbers/dog/pool"
    assert reroot_output_path(old, root) == pool


def test_accepts_a_string(tmp_path):
    root = tmp_path / "outputs"
    pool = _pool(root)
    assert reroot_output_path(str(pool), root) == pool
