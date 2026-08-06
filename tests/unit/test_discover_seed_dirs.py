"""Unit tests for the seed-discovery generalization shared by teacher + student eval.

The SDF verification pipeline is reused verbatim for GKD students; the one adaptation is
that seed dirs are ``gseed{N}`` (student training seeds), not ``ftseed{N}``. This guards
that ``seed_prefix`` discovers/ orders the right leaves and that the teacher default is
unchanged (a regression here would silently evaluate zero seeds or the wrong tree).

Discovery ALSO requires a complete checkpoint (``checkpoint_is_complete``): a
``checkpoints/`` holding no weight file is an interrupted save and is skipped, so the
fixtures below write a real adapter artifact rather than an empty directory.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.evaluate_teacher_preference_pipeline import _discover_ftseed_dirs


def _make_seed_dirs(root: Path, prefix: str, seeds: list[int]) -> None:
    """Seed dirs with a COMPLETE checkpoint (adapter config + weights)."""
    for s in seeds:
        ckpt = root / f"{prefix}{s}" / "checkpoints"
        ckpt.mkdir(parents=True)
        (ckpt / "adapter_config.json").write_text("{}")
        (ckpt / "adapter_model.safetensors").write_bytes(b"")


def test_default_prefix_discovers_ftseed(tmp_path: Path) -> None:
    _make_seed_dirs(tmp_path, "ftseed", [0, 1])
    found = _discover_ftseed_dirs(tmp_path)
    assert [s for s, _ in found] == [0, 1]
    assert all(p.name == "checkpoints" for _, p in found)


def test_gseed_prefix_discovers_student_seeds_sorted(tmp_path: Path) -> None:
    _make_seed_dirs(tmp_path, "gseed", [1, 0])
    found = _discover_ftseed_dirs(tmp_path, seed_prefix="gseed")
    assert [s for s, _ in found] == [0, 1]
    assert found[0][1] == tmp_path / "gseed0" / "checkpoints"


def test_prefix_does_not_cross_match(tmp_path: Path) -> None:
    # A gseed tree must not surface ftseed dirs (or vice versa).
    _make_seed_dirs(tmp_path, "gseed", [0])
    _make_seed_dirs(tmp_path, "ftseed", [7])
    assert [s for s, _ in _discover_ftseed_dirs(tmp_path, seed_prefix="gseed")] == [0]
    assert [s for s, _ in _discover_ftseed_dirs(tmp_path)] == [7]


def test_seed_dir_without_checkpoints_skipped(tmp_path: Path) -> None:
    _make_seed_dirs(tmp_path, "gseed", [0])
    (tmp_path / "gseed1").mkdir()  # no checkpoints/ -> skipped
    assert [s for s, _ in _discover_ftseed_dirs(tmp_path, seed_prefix="gseed")] == [0]


def test_incomplete_checkpoint_skipped_and_reported(tmp_path: Path) -> None:
    """An interrupted save (config, no weights) is skipped, not loaded and crashed on.

    Loading it would raise and abort the ENTIRE cell's eval, taking sibling seeds that
    trained fine down with it; the omitted seed is reported via `skipped_incomplete`.
    """
    _make_seed_dirs(tmp_path, "gseed", [0])
    partial = tmp_path / "gseed1" / "checkpoints"
    partial.mkdir(parents=True)
    (partial / "adapter_config.json").write_text("{}")  # no weight file

    skipped: list[int] = []
    found = _discover_ftseed_dirs(tmp_path, seed_prefix="gseed", skipped_incomplete=skipped)
    assert [s for s, _ in found] == [0]
    assert skipped == [1]
