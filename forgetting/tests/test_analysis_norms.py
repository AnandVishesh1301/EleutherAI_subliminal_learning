"""Round-trip contract of the adapter-norms run-dir migration + `analysis.norms` loader."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from forgetting.scripts.migrate_adapter_norms_to_rundirs import migrate_aggregate
from subliminal_learning.forgetting.analysis import norms as an

BASE, TEACHER, DTAG, TAG = "pythia-160m-data-seed1", "ep2_lr5e-4_r32a64_tseed1", "ds", "ep20_x"


def _row(module, b_fro, w0_fro, **kw):
    r = {"student_base": BASE, "teacher": TEACHER, "dataset_tag": DTAG, "student_tag": TAG,
         "ftseed": 1, "rank": 8, "alpha": 32, "test_p": 0.5, "module": module,
         "B_Afinal_fro": b_fro, "W0_fro": w0_fro, "B_Afinal_spec": b_fro / 2,
         "W0_spec": w0_fro / 2, "cos_Ainit_Afinal": 0.9, "cos_wrongseed_Afinal": 0.01}
    r.update(kw)
    return r


def _aggregate(tmp_path: Path, rows) -> Path:
    f = tmp_path / "agg.json"
    f.write_text(json.dumps({"bases": [BASE], "wrong_seed_offset": 1000, "results": rows}))
    return f


def _rel(ftseed=1):
    return an.run_rel_path(BASE, TEACHER, DTAG, TAG, ftseed)


def _make_run(tmp_path: Path, ftseed=1):
    d = tmp_path / _rel(ftseed)
    d.mkdir(parents=True)
    (d / "train_manifest.json").write_text(json.dumps(
        {"student_param_tag": TAG, "filtered_sha256": "s", "n_train_steps": 7}))
    return d


def test_round_trip_dw_fro_matches_hand_computation(tmp_path):
    run = _make_run(tmp_path)
    src = _aggregate(tmp_path, [_row("m0", 1.0, 10.0), _row("m1", 2.0, 8.0)])

    res = migrate_aggregate(tmp_path, src)
    assert (res.n_runs, res.written, res.skipped_missing, res.skipped_existing) == (1, 1, 0, 0)

    rec = json.loads(an.record_path(run).read_text())
    assert rec["manifest_ref"]["n_train_steps"] == 7
    assert rec["manifest_ref_read_at_migration"] is True
    assert rec["wrong_seed_offset"] == 1000
    assert "student_base" not in rec["modules"][0]      # run keys not repeated per module

    df = an.load_adapter_norm_runs(tmp_path)
    assert len(df) == 1
    s = 32 / 8
    assert df.dw_fro.iloc[0] == pytest.approx((s * 1.0 / 10.0 + s * 2.0 / 8.0) / 2)
    assert df.dw_spec.iloc[0] == pytest.approx((s * 0.5 / 5.0 + s * 1.0 / 4.0) / 2)
    assert df.cos_Ainit_Afinal_mean.iloc[0] == pytest.approx(0.9)
    assert df.n_modules.iloc[0] == 2
    r = df.iloc[0]
    assert (r.student_base, r.teacher, r.dataset_tag, r.student_tag, r.animal, r.ftseed) == (
        BASE, TEACHER, DTAG, TAG, "dog", 1)


def test_missing_run_dir_is_skipped_and_counted(tmp_path):
    src = _aggregate(tmp_path, [_row("m0", 1.0, 10.0)])
    res = migrate_aggregate(tmp_path, src)
    assert (res.written, res.skipped_missing) == (0, 1)
    assert res.missing_paths == [_rel()]
    assert an.load_inventory(tmp_path) == []


def test_existing_record_is_never_overwritten(tmp_path):
    run = _make_run(tmp_path)
    src = _aggregate(tmp_path, [_row("m0", 1.0, 10.0)])
    migrate_aggregate(tmp_path, src)
    before = an.record_path(run).read_text()

    src2 = _aggregate(tmp_path, [_row("m0", 99.0, 10.0)])
    res = migrate_aggregate(tmp_path, src2)
    assert (res.written, res.skipped_existing) == (0, 1)
    assert an.record_path(run).read_text() == before


def test_dry_run_writes_nothing(tmp_path):
    run = _make_run(tmp_path)
    src = _aggregate(tmp_path, [_row("m0", 1.0, 10.0)])
    res = migrate_aggregate(tmp_path, src, dry_run=True)
    assert res.written == 1
    assert not an.record_path(run).exists()
    assert not (tmp_path / an.INVENTORY_REL).exists()


def test_step_rows_never_reach_the_endpoint_record(tmp_path):
    _make_run(tmp_path)
    src = _aggregate(tmp_path, [_row("m0", 1.0, 10.0), _row("m0", 9.0, 10.0, step=100)])
    res = migrate_aggregate(tmp_path, src)
    assert (res.written, res.skipped_traj_rows) == (1, 1)
    rec = json.loads(an.record_path(tmp_path / _rel()).read_text())
    assert len(rec["modules"]) == 1 and all("step" not in m for m in rec["modules"])
    assert not an.traj_record_path(tmp_path / _rel()).exists()   # migration is endpoint-only
    with pytest.raises(ValueError):
        an.build_record([_row("m0", 1.0, 10.0, step=100)], test_p=None, rank=8, alpha=32,
                        wrong_seed_offset=0, provenance={}, manifest_ref=None)
    with pytest.raises(ValueError):
        an.build_traj_record([_row("m0", 1.0, 10.0)], test_p=None, rank=8, alpha=32,
                             wrong_seed_offset=0, provenance={}, manifest_ref=None)


def test_traj_record_round_trip(tmp_path):
    run = _make_run(tmp_path)
    rec = an.build_traj_record(
        [_row("m0", 1.0, 10.0, step=50), _row("m1", 2.0, 8.0, step=50),
         _row("m0", 4.0, 10.0, step=100)],
        test_p=0.5, rank=8, alpha=32, wrong_seed_offset=1000,
        provenance={"harvested_by": "adapter_norms_probe"}, manifest_ref=None)
    an.record_path(run).parent.mkdir(parents=True, exist_ok=True)
    an.traj_record_path(run).write_text(json.dumps(rec))
    an.update_inventory(tmp_path, [_rel()], source="t", key="traj_runs")

    assert [p["step"] for p in rec["points"]] == [50, 100]
    df = an.load_adapter_norm_traj(tmp_path)
    s = 32 / 8
    assert len(df) == 1 and df.n_steps.iloc[0] == 2 and df.step_max.iloc[0] == 100
    assert df.dw_fro_last.iloc[0] == pytest.approx(s * 4.0 / 10.0)
    assert df.dw_fro_max.iloc[0] == pytest.approx(s * 4.0 / 10.0)
    assert an.load_inventory(tmp_path, "runs") == []          # inventories stay separate
    assert an.load_adapter_norm_runs(tmp_path).empty


def test_inventory_is_unioned_across_passes(tmp_path):
    _make_run(tmp_path, 1)
    migrate_aggregate(tmp_path, _aggregate(tmp_path, [_row("m0", 1.0, 10.0)]))
    _make_run(tmp_path, 2)
    migrate_aggregate(tmp_path, _aggregate(tmp_path, [_row("m0", 1.0, 10.0, ftseed=2)]))
    assert sorted(an.load_inventory(tmp_path)) == sorted([_rel(1), _rel(2)])
    assert len(an.load_adapter_norm_runs(tmp_path)) == 2


def test_parse_teacher_run_paths():
    df = an.parse_teacher_run_paths([
        "teachers/dog/pythia-160m/ep10_lr1e-3_r128a16/ftseed1",
        "teachers/cat/pythia-160m-data-seed1/ep1_lr1e-4_r8a32/ftseed2/",
        "students/dog/pythia-160m/teacher-pythia-160m/t/ds/ep20_x/ftseed1",
    ])
    assert df.animal.tolist()[:2] == ["dog", "cat"]
    r = df.iloc[1]
    assert (r.run_path, r.teacher_base, r.param_tag, r.ftseed) == (
        "teachers/cat/pythia-160m-data-seed1/ep1_lr1e-4_r8a32/ftseed2",
        "pythia-160m-data-seed1", "ep1_lr1e-4_r8a32", 2)
    assert df.param_tag.isna().iloc[2] and df.run_path.iloc[2].startswith("students/")
