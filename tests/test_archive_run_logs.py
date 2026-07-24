"""Unit tests for ``subliminal_learning.run_logs_archive`` (all mutation in ``tmp_path`` trees).

Nothing here touches the real ``run_logs/``: every fixture builds a synthetic flat log dir,
the open-file probe is monkeypatched, and mtimes are planted with ``os.utime``.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning import run_logs_archive as rla

CUTOFF = datetime(2026, 7, 23, tzinfo=timezone.utc)
OLD = datetime(2026, 7, 1, tzinfo=timezone.utc).timestamp()
NEW = datetime(2026, 7, 24, tzinfo=timezone.utc).timestamp()


def _write(dir_path: Path, name: str, *, mtime: float = OLD, content: str = "x") -> Path:
    p = dir_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    import os

    os.utime(p, (mtime, mtime))
    return p


# --------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("name", "family"),
    [
        # STAMP stripping is applied before matching.
        ("20260723T184443Z_dog_numgen_ep20_gpu4.log", "dog_numgen"),
        ("20260717T1420Z_crossseed_gpu0_worker0.log", "crossseed"),  # 4-digit time stamp
        # Longest / most-specific prefix wins.
        ("dog_crossdistill_studenthp_ep20_gpu4.log", "dog_crossdistill_studenthp"),
        ("dog_crossdistill_ep20_weight-seed3_gpu0.log", "dog_crossdistill"),
        ("dog_samedistill_ep20plain_gpu0.log", "dog_samedistill"),
        ("dog_constrained_intro_u10_s0_gpu4.log", "dog_constrained_numgen"),
        ("dog_distill_ep20_lr1e-4_r16_gpu0.log", "dog_distill"),
        ("dog_train_shard0_gpu0.log", "dog"),
        ("controls_backfill_p2_j3_gpu4.log", "controls_backfill"),
        ("controls_ep20_seed_p2_jobs.log", "controls"),
        ("rabbitlion_distill_lion_ep20_gpu0.log", "rabbitlion"),
        ("dataseed1_train_shard0_gpu0.log", "dataseed1"),
        ("pig_numgen_shard0_gpu0.log", "pig"),
        ("eas_shard0_gpu7.log", "eas"),
        ("steerability_gpu0_w1.log", "steerability"),
        ("crossbase_dynamic_shard0.log", "crossbase"),
        ("samebase_even_shard0_gpu0.log", "samebase"),
        ("selected_number_gen_shard0_gpu3.log", "numgen"),
        ("number_gpu0_shard0of14_cycle1.log", "numgen"),
        ("numgen_shard0_gpu4.log", "numgen"),
        ("selected_train_shard0.log", "teacher_sweep"),
        ("full_train_shard0.log", "teacher_sweep"),
        ("teacher_16way_TZ.log", "teacher_sweep"),
        ("distill_cross_ep20_shard0_gpu0.log", "distill"),
        ("student_smoke_ep20_prefix_tseed1.log", "student"),
        ("external_gpu_claim_watchdog.log", "gpu_idle"),
    ],
)
def test_classify_families(name: str, family: str) -> None:
    assert rla.classify(name) == family


def test_strip_stamp() -> None:
    assert rla.strip_stamp("20260723T184443Z_foo.log") == "foo.log"
    assert rla.strip_stamp("20260717T1420Z_foo.log") == "foo.log"
    assert rla.strip_stamp("no_stamp_here.log") == "no_stamp_here.log"


def test_fallback_family_and_determinism() -> None:
    # No rule matches "zzz_experiment": fallback keeps leading non-shardy tokens.
    assert rla.classify("zzz_experiment_gpu3_shard0.log") == "zzz_experiment"
    # A leading shardy token collapses to "misc".
    assert rla.classify("123_orphan.log") == "misc"
    # Deterministic: same input, same output.
    assert rla.classify("zzz_experiment_gpu3_shard0.log") == rla.classify(
        "zzz_experiment_gpu3_shard0.log"
    )


# --------------------------------------------------------------------------- eligibility


def test_build_plan_eligibility(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rla, "collect_open_paths", lambda: set())

    keep = _write(tmp_path, "20260701T000000Z_dog_numgen_ep20_gpu4.log")
    _write(tmp_path, "readme_note.out")  # eligible plain .out
    _write(tmp_path, "config_notes.md")  # wrong ext
    _write(tmp_path, "run_launcher.sh", content="grep me")  # wrong ext (script)
    _write(tmp_path, "too_new_dog_numgen.log", mtime=NEW)  # too new
    referenced = _write(tmp_path, "dog_pipeline_master.log")  # referenced by a .sh
    _write(tmp_path, "waiter.sh", content=f"tail -f {referenced.name}\n")
    # A subdirectory file must be ignored entirely.
    _write(tmp_path / "archive" / "dog", "old.log")

    plan = rla.build_plan(tmp_path, CUTOFF)
    names = {c.path.name for c in plan.candidates}
    assert names == {keep.name, "readme_note.out"}
    assert plan.excluded["wrong_ext"] == 3  # .md + two .sh scripts
    assert plan.excluded["too_new"] == 1
    assert plan.excluded["referenced"] == 1
    assert plan.excluded["open"] == 0


def test_build_plan_open_file_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    open_log = _write(tmp_path, "dog_numgen_open.log")
    _write(tmp_path, "dog_numgen_closed.log")
    monkeypatch.setattr(rla, "collect_open_paths", lambda: {str(open_log)})

    plan = rla.build_plan(tmp_path, CUTOFF)
    names = {c.path.name for c in plan.candidates}
    assert names == {"dog_numgen_closed.log"}
    assert plan.excluded["open"] == 1


# --------------------------------------------------------------------------- apply / undo


def test_apply_moves_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rla, "collect_open_paths", lambda: set())
    a = _write(tmp_path, "dog_numgen_ep20_gpu4.log", content="aaa")
    b = _write(tmp_path, "eas_shard0_gpu7.log", content="bb")

    plan = rla.build_plan(tmp_path, CUTOFF)
    moved = rla.apply_plan(tmp_path, plan)

    assert len(moved) == 2
    assert not a.exists() and not b.exists()
    assert (tmp_path / "archive" / "dog_numgen" / a.name).exists()
    assert (tmp_path / "archive" / "eas" / b.name).exists()

    manifest = (tmp_path / "archive" / "MANIFEST.tsv").read_text().splitlines()
    assert len(manifest) == 2
    fields = manifest[0].split("\t")
    assert len(fields) == 7
    assert fields[1] == "move"
    # Rows are family-sorted (dog_numgen before eas).
    assert manifest[0].split("\t")[2] == "dog_numgen"
    assert manifest[0].split("\t")[3] == a.name
    assert manifest[0].split("\t")[4] == f"archive/dog_numgen/{a.name}"


def test_apply_skips_existing_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rla, "collect_open_paths", lambda: set())
    src = _write(tmp_path, "dog_numgen_ep20_gpu4.log", content="new")
    dst = tmp_path / "archive" / "dog_numgen" / src.name
    dst.parent.mkdir(parents=True)
    dst.write_text("preexisting")

    plan = rla.build_plan(tmp_path, CUTOFF)
    moved = rla.apply_plan(tmp_path, plan)

    assert moved == []
    assert src.exists()  # left in place
    assert dst.read_text() == "preexisting"  # not overwritten
    assert not (tmp_path / "archive" / "MANIFEST.tsv").exists()


def test_plan_is_strict_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rla, "collect_open_paths", lambda: set())
    _write(tmp_path, "dog_numgen_ep20_gpu4.log")
    before = sorted(p.name for p in tmp_path.iterdir())

    rla.build_plan(tmp_path, CUTOFF)

    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
    assert not (tmp_path / "archive").exists()


def test_undo_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rla, "collect_open_paths", lambda: set())
    a = _write(tmp_path, "dog_numgen_ep20_gpu4.log", content="aaa")
    b = _write(tmp_path, "eas_shard0_gpu7.log", content="bb")
    originals = {a.name: a.read_text(), b.name: b.read_text()}

    plan = rla.build_plan(tmp_path, CUTOFF)
    rla.apply_plan(tmp_path, plan)
    assert not a.exists() and not b.exists()

    restored = rla.undo(tmp_path)
    assert set(restored) == {a.name, b.name}
    assert a.exists() and b.exists()
    assert {a.name: a.read_text(), b.name: b.read_text()} == originals
    assert not (tmp_path / "archive" / "dog_numgen" / a.name).exists()

    manifest = (tmp_path / "archive" / "MANIFEST.tsv").read_text().splitlines()
    undo_rows = [line for line in manifest if line.split("\t")[1] == "undo"]
    assert len(undo_rows) == 2

    # A second undo is a warn-and-skip no-op (archived files already restored).
    assert rla.undo(tmp_path) == []
