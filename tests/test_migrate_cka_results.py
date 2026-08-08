"""Tests for scripts/migrate_cka_results.py (tmp_path only, no real filesystem)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.migrate_cka_results import LOG_NAME, run_migration


def _make_run_dir(analysis_cka: Path, name: str, *, manifest: bool = True) -> Path:
    run = analysis_cka / name
    run.mkdir(parents=True)
    out_dir = str(run)
    if manifest:
        (run / "cka_manifest.json").write_text(
            json.dumps({"run_identity": {"run_tag": name, "positions_per_seq": 1}}, indent=2) + "\n"
        )
    (run / "meta.json").write_text(
        json.dumps({"config": {"out_dir": out_dir, "seed": 0}}, indent=2) + "\n"
    )
    (run / "summary.json").write_text(
        json.dumps({"meta": {"config": {"out_dir": out_dir}}, "means": {"a": 1.0}}, indent=2) + "\n"
    )
    np.save(run / "cka__x__y.npy", np.arange(4))
    np.save(run / "convergence__x.npy", np.arange(3))
    return run


def _tree(tmp_path: Path) -> tuple[Path, Path]:
    output_root = tmp_path / "out"
    return output_root / "analysis" / "cka", output_root / "cka" / "base"


def test_happy_path(tmp_path):
    analysis_cka, cka_base = _tree(tmp_path)
    _make_run_dir(analysis_cka, "pile")
    _make_run_dir(analysis_cka, "numbers_base160m_prefix")
    manifest_bytes = (analysis_cka / "pile" / "cka_manifest.json").read_bytes()

    rc = run_migration(analysis_cka, cka_base, apply=True)
    assert rc == 0

    for name in ("pile", "numbers_base160m_prefix"):
        dst = cka_base / name
        assert dst.is_dir()
        assert not (analysis_cka / name).exists()
        meta = json.loads((dst / "meta.json").read_text())
        summ = json.loads((dst / "summary.json").read_text())
        assert meta["config"]["out_dir"] == str(dst)
        assert summ["meta"]["config"]["out_dir"] == str(dst)

    # manifest is byte-identical
    assert (cka_base / "pile" / "cka_manifest.json").read_bytes() == manifest_bytes

    # log written with one line per migrated dir
    log = cka_base.parent / LOG_NAME
    lines = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
    assert len(lines) == 2
    assert {Path(x["new"]).name for x in lines} == {"pile", "numbers_base160m_prefix"}
    assert all(x["jsons_rewritten"] == ["meta.json", "summary.json"] for x in lines)

    # old analysis/cka removed (empty)
    assert not analysis_cka.exists()


def test_destination_conflict(tmp_path):
    analysis_cka, cka_base = _tree(tmp_path)
    src = _make_run_dir(analysis_cka, "pile")
    (cka_base / "pile").mkdir(parents=True)  # pre-existing destination

    rc = run_migration(analysis_cka, cka_base, apply=True)
    assert rc == 1
    assert src.is_dir()  # source untouched
    assert (src / "cka_manifest.json").is_file()


def test_dry_run_touches_nothing(tmp_path):
    analysis_cka, cka_base = _tree(tmp_path)
    src = _make_run_dir(analysis_cka, "pile")
    before = json.loads((src / "meta.json").read_text())

    rc = run_migration(analysis_cka, cka_base, apply=False)
    assert rc == 0
    assert src.is_dir()
    assert not cka_base.exists()
    assert not (cka_base.parent / LOG_NAME).exists()
    assert json.loads((src / "meta.json").read_text()) == before


def test_idempotent_second_run(tmp_path):
    analysis_cka, cka_base = _tree(tmp_path)
    _make_run_dir(analysis_cka, "pile")
    assert run_migration(analysis_cka, cka_base, apply=True) == 0
    # second run: nothing left under analysis/cka
    assert run_migration(analysis_cka, cka_base, apply=True) == 0


def test_incomplete_run_skipped(tmp_path):
    analysis_cka, cka_base = _tree(tmp_path)
    _make_run_dir(analysis_cka, "pile")
    incomplete = _make_run_dir(analysis_cka, "half_done", manifest=False)

    rc = run_migration(analysis_cka, cka_base, apply=True)
    assert rc == 0
    assert (cka_base / "pile").is_dir()
    # incomplete run still present under analysis/cka, which is NOT removed
    assert incomplete.is_dir()
    assert analysis_cka.is_dir()
