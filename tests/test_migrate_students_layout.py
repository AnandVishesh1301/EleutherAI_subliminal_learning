"""Tests for scripts/migrate_students_layout.py -- the students/ layout migrator.

Every test builds a miniature OLD-layout tree in ``tmp_path``; the real outputs tree is
never touched. The invariants under test are the ones whose failure would misplace or
corrupt trained student checkpoints.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.migrate_students_layout import (
    plan_migration,
    run_migration,
    verify_tree,
)


def _leaf(base: Path, seed: int, *, manifest_text: str | None = None) -> Path:
    """Build a ``{stag}/ftseed{seed}/checkpoints`` leaf under ``base``; optional manifest."""
    stag = base / "std_hp" / f"ftseed{seed}"
    ckpt = stag / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_model.safetensors").write_text("weights")
    if manifest_text is not None:
        (stag / "train_manifest.json").write_text(manifest_text)
    return stag


def _treatment_lineage(students: Path, animal: str, model: str, lineage: str, seed: int = 1) -> Path:
    d = students / animal / model / lineage
    _leaf(d / "ex10-20_prefix_n100k", seed)
    return d


# --------------------------------------------------------------------------- (1) same-base


def test_same_base_treatment(tmp_path: Path) -> None:
    students = tmp_path / "students"
    old = _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")

    result = run_migration(students, apply=True)

    new = students / "dog" / "modelA" / "teacher-modelA" / "ep20_lr4e-4_r16a4_tseed1"
    assert not old.exists()
    assert (new / "ex10-20_prefix_n100k" / "std_hp" / "ftseed1" / "checkpoints").is_dir()
    assert len(result.done) == 1 and not result.conflicts


# --------------------------------------------------------------------------- (2) cross-base


def test_cross_base_treatment(tmp_path: Path) -> None:
    students = tmp_path / "students"
    old = _treatment_lineage(
        students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1_nbase-modelB"
    )

    run_migration(students, apply=True)

    new = students / "dog" / "modelA" / "teacher-modelB" / "ep20_lr4e-4_r16a4_tseed1"
    assert not old.exists()
    assert (new / "ex10-20_prefix_n100k" / "std_hp" / "ftseed1" / "checkpoints").is_dir()


# --------------------------------------------------------------------------- (3) base controls


def test_base_control_with_and_without_nbase(tmp_path: Path) -> None:
    students = tmp_path / "students"
    base = students / "controls" / "base" / "modelA"
    # ndt with a "_spliced" suffix and no _nbase-.
    plain = base / "ex10-20_prefix_n100k_spliced"
    _leaf(plain, 3)
    # ndt (same _spliced suffix) with a cross-base _nbase-modelB.
    cross = base / "ex10-20_prefix_n100k_spliced_nbase-modelB"
    _leaf(cross, 3)

    run_migration(students, apply=True)

    assert not plain.exists() and not cross.exists()
    new_plain = base / "teacher-modelA" / "ex10-20_prefix_n100k_spliced"
    new_cross = base / "teacher-modelB" / "ex10-20_prefix_n100k_spliced"
    assert (new_plain / "std_hp" / "ftseed3" / "checkpoints").is_dir()
    assert (new_cross / "std_hp" / "ftseed3" / "checkpoints").is_dir()


# --------------------------------------------------------------------------- (4) rng untouched


def test_rng_control_untouched(tmp_path: Path) -> None:
    students = tmp_path / "students"
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")
    rng = students / "controls" / "rng" / "modelA" / "ex10-20_prefix_n100k_nbase-modelB"
    _leaf(rng, 1)

    run_migration(students, apply=True)

    # The rng subtree keeps its exact old path, including the _nbase- name.
    assert (rng / "std_hp" / "ftseed1" / "checkpoints").is_dir()
    plan = plan_migration(students)
    assert all("controls/rng" not in str(m.old) for m in plan.moves)


# --------------------------------------------------------------------------- (5) json rewrite


def test_manifest_path_rewritten_byte_identical_otherwise(tmp_path: Path) -> None:
    students = tmp_path / "students"
    lineage = students / "dog" / "modelA" / "from_ep20_lr4e-4_r16a4_tseed1"
    old_ckpt = lineage / "ex10-20_prefix_n100k" / "std_hp" / "ftseed1" / "checkpoints"
    manifest = {
        "checkpoint_dir": str(old_ckpt.resolve()),
        "note": "unrelated field with modelA and from_ text that must survive",
    }
    manifest_text = json.dumps(manifest, indent=2)
    _leaf(
        lineage / "ex10-20_prefix_n100k",
        1,
        manifest_text=manifest_text,
    )

    run_migration(students, apply=True)

    new_stag = (
        students / "dog" / "modelA" / "teacher-modelA" / "ep20_lr4e-4_r16a4_tseed1"
        / "ex10-20_prefix_n100k" / "std_hp" / "ftseed1"
    )
    rewritten = json.loads((new_stag / "train_manifest.json").read_text())
    new_ckpt = new_stag / "checkpoints"
    assert rewritten["checkpoint_dir"] == str(new_ckpt.resolve())
    # Every other field is preserved verbatim.
    assert rewritten["note"] == manifest["note"]
    assert set(rewritten) == set(manifest)


# --------------------------------------------------------------------------- (6) dry run


def test_dry_run_changes_nothing(tmp_path: Path) -> None:
    students = tmp_path / "students"
    old = _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")
    before = sorted(str(p.relative_to(students)) for p in students.rglob("*"))

    result = run_migration(students, apply=False)

    after = sorted(str(p.relative_to(students)) for p in students.rglob("*"))
    assert before == after  # tree unchanged
    assert old.exists()
    assert len(result.planned) == 1 and not result.done
    assert result.log_path is None  # no log written on a dry run
    assert not list(students.glob("_migration_layout_*.jsonl"))


# --------------------------------------------------------------------------- (7) rerun no-op


def test_rerun_after_apply_is_noop(tmp_path: Path) -> None:
    students = tmp_path / "students"
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")
    _leaf(students / "controls" / "base" / "modelA" / "ex10-20_prefix_n100k", 2)

    run_migration(students, apply=True)
    second = run_migration(students, apply=True)

    assert not second.plan.moves  # nothing left to migrate
    assert not second.done and not second.conflicts


# --------------------------------------------------------------------------- (8) collision


def test_collision_records_conflict_and_other_moves_apply(tmp_path: Path) -> None:
    students = tmp_path / "students"
    # Move that will collide: its NEW target already exists.
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")
    conflict_target = (
        students / "dog" / "modelA" / "teacher-modelA" / "ep20_lr4e-4_r16a4_tseed1"
    )
    conflict_target.mkdir(parents=True)
    (conflict_target / "sentinel.txt").write_text("preexisting")
    # An independent move that MUST still apply.
    good_old = _treatment_lineage(students, "cat", "modelA", "from_ep20_lr4e-4_r16a4_tseed2", seed=2)

    result = run_migration(students, apply=True)

    assert len(result.conflicts) == 1
    assert result.conflicts[0].new == conflict_target
    assert (conflict_target / "sentinel.txt").read_text() == "preexisting"  # untouched
    # The good move still happened.
    assert not good_old.exists()
    good_new = students / "cat" / "modelA" / "teacher-modelA" / "ep20_lr4e-4_r16a4_tseed2"
    assert (good_new / "ex10-20_prefix_n100k" / "std_hp" / "ftseed2").is_dir()

    # Log records the conflict so it can be reasoned about.
    records = [json.loads(x) for x in result.log_path.read_text().splitlines()]
    assert any(r["action"] == "conflict" for r in records)


def test_log_streamed_survives_midapply_crash(tmp_path: Path, monkeypatch) -> None:
    import scripts.migrate_students_layout as mod

    students = tmp_path / "students"
    # Two independent moves; the crash lands on the second.
    _treatment_lineage(students, "cat", "modelA", "from_ep20_lr4e-4_r16a4_tseed1", seed=1)
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed2", seed=2)

    real_rename = os.rename
    calls = {"n": 0}

    def flaky_rename(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-apply")
        return real_rename(src, dst)

    monkeypatch.setattr(mod.os, "rename", flaky_rename)

    with pytest.raises(RuntimeError, match="simulated crash"):
        mod.run_migration(students, apply=True)

    # The first rename's record is already durable on disk despite the crash.
    logs = list(students.glob("_migration_layout_*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(x) for x in logs[0].read_text().splitlines()]
    renames = [r for r in records if r["action"] == "rename"]
    assert len(renames) == 1
    assert renames[0]["new"].endswith(
        "cat/modelA/teacher-modelA/ep20_lr4e-4_r16a4_tseed1"
    )


def test_main_exit_nonzero_on_conflict(tmp_path: Path) -> None:
    import typer

    from scripts.migrate_students_layout import main

    students = tmp_path / "students"
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")
    (students / "dog" / "modelA" / "teacher-modelA" / "ep20_lr4e-4_r16a4_tseed1").mkdir(parents=True)

    try:
        main(students_root=students, apply=True, log=None, verify=False)
    except typer.Exit as exc:
        assert exc.exit_code == 1
    else:
        raise AssertionError("expected a non-zero typer.Exit on conflict")


# --------------------------------------------------------------------------- (9) verify


def test_verify_detects_leftover_old_form_dir(tmp_path: Path) -> None:
    students = tmp_path / "students"
    _treatment_lineage(students, "dog", "modelA", "from_ep20_lr4e-4_r16a4_tseed1")

    vr = verify_tree(students)

    assert len(vr.from_dirs) == 1
    assert vr.from_dirs[0].name == "from_ep20_lr4e-4_r16a4_tseed1"
    assert vr.ftseed_count == 1
    assert not vr.ok  # leftover old-form dir fails verification

    # After migration the tree verifies clean.
    run_migration(students, apply=True)
    vr2 = verify_tree(students)
    assert vr2.ok and not vr2.from_dirs and not vr2.nbase_dirs and not vr2.stale_json
