"""Dense probe schedule, step snapshots and merged-start identity (reversal Part II)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from transformers import TrainerControl, TrainerState

from pipelines import distill_student_gkd_pipeline as gkd_pipeline
from subliminal_learning.gkd.config import load_gkd_config, load_gkd_training
from subliminal_learning.gkd.probe_io import (
    SNAPSHOT_FILE,
    GkdProbeCallback,
    SftProbeCallback,
    StepSnapshotCallback,
    clear_stale_snapshots,
)
from subliminal_learning.gkd.probe_schedule import probe_due, validate_probe_schedule
from tests.unit.test_gkd_config import _four_yamls, _training_yaml
from tests.unit.test_sdf_gkd_run_identity import (
    _LEGACY_DEFAULT_IDENTITY,
    _data_identity,
    _gkd_cfg,
)

_STEPS = range(301)


# --- probe_due -----------------------------------------------------------------------


@pytest.mark.parametrize("every", [-1, 0, 1, 2, 5, 25, 125])
def test_probe_due_defaults_equal_the_old_modulo_rule(every: int) -> None:
    old = [s for s in _STEPS if not (every <= 0 or s % every != 0)]
    assert [s for s in _STEPS if probe_due(s, every)] == old


def test_probe_due_adds_a_dense_window() -> None:
    due = [s for s in range(1, 131) if probe_due(s, 25, 5, 40)]
    assert due == [5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100, 125]


def test_probe_due_dense_only_and_off() -> None:
    assert [s for s in range(1, 301) if probe_due(s, 0, 5, 20)] == [5, 10, 15, 20]
    assert not any(probe_due(s, 0) for s in range(1, 301))
    assert not any(probe_due(s, 0, 0, 100) for s in range(1, 301))


@pytest.mark.parametrize(("dense_every", "dense_until"), [(-1, 10), (5, -1), (5, 0)])
def test_validate_probe_schedule_rejects_bad_windows(
    dense_every: int, dense_until: int
) -> None:
    with pytest.raises(ValueError):
        validate_probe_schedule(25, dense_every, dense_until)


def test_validate_probe_schedule_accepts_defaults() -> None:
    validate_probe_schedule(25, 0, 0)
    validate_probe_schedule(0, 0, 0)
    validate_probe_schedule(25, 5, 100)


# --- callbacks use the schedule --------------------------------------------------------


def _gkd_probe(tmp_path: Path, every: int, **dense) -> GkdProbeCallback:  # noqa: ANN003
    return GkdProbeCallback(
        None, "cpu", "dog", 0, (), (), {}, [], every, tmp_path / "m.jsonl", **dense
    )


def _sft_probe(tmp_path: Path, every: int, **dense) -> SftProbeCallback:  # noqa: ANN003
    return SftProbeCallback(
        None, "cpu", None, (), (), {}, [], every, tmp_path / "m.jsonl", **dense
    )


def _probed_steps(cb) -> list[int]:  # noqa: ANN001
    seen: list[int] = []
    cb.probe_once = lambda model, step: seen.append(step)  # noqa: ARG005
    state, control = TrainerState(), TrainerControl()
    for step in _STEPS:
        state.global_step = step
        cb.on_step_end(None, state, control, model=object())
    return seen


@pytest.mark.parametrize("make", [_gkd_probe, _sft_probe])
@pytest.mark.parametrize("every", [0, 2, 25, 125])
def test_callback_default_schedule_is_unchanged(tmp_path: Path, make, every: int):  # noqa: ANN001
    old = [s for s in _STEPS if every > 0 and s > 0 and s % every == 0]
    assert _probed_steps(make(tmp_path, every)) == old


@pytest.mark.parametrize("make", [_gkd_probe, _sft_probe])
def test_callback_dense_schedule(tmp_path: Path, make):  # noqa: ANN001
    cb = make(tmp_path, 100, dense_every=5, dense_until=20)
    assert _probed_steps(cb) == [5, 10, 15, 20, 100, 200, 300]


def test_callback_rejects_an_open_dense_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dense_until"):
        _gkd_probe(tmp_path, 25, dense_every=5)


# --- GKD config ------------------------------------------------------------------------


def test_gkd_dense_probe_keys_parse_and_default_off(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, {}))
    assert (gkd.probe_dense_every, gkd.probe_dense_until) == (0, 0)
    _, _, gkd = load_gkd_training(
        _training_yaml(tmp_path, {"probe_dense_every": 5, "probe_dense_until": 100})
    )
    assert (gkd.probe_dense_every, gkd.probe_dense_until) == (5, 100)
    with pytest.raises(ValueError, match="dense_until"):
        load_gkd_training(_training_yaml(tmp_path, {"probe_dense_every": 5}))


def test_gkd_dense_probe_experiment_override(tmp_path: Path) -> None:
    exp = {"probe_every": 25, "probe_dense_every": 5, "probe_dense_until": 100}
    gkd = load_gkd_config(*_four_yamls(tmp_path, {}, exp)).gkd
    assert (gkd.probe_every, gkd.probe_dense_every, gkd.probe_dense_until) == (
        25,
        5,
        100,
    )
    with pytest.raises(ValueError):
        load_gkd_config(*_four_yamls(tmp_path, {}, {"probe_dense_until": -1}))


def test_gkd_probe_provenance_records_the_window_only_when_set(tmp_path: Path):
    cfg = _gkd_cfg(tmp_path)
    plain = gkd_pipeline._probe_provenance(cfg, ("a",), ("b",), ["c"])
    assert "probe_dense_every" not in plain
    dense = replace(
        cfg, gkd=replace(cfg.gkd, probe_dense_every=5, probe_dense_until=50)
    )
    record = gkd_pipeline._probe_provenance(dense, ("a",), ("b",), ["c"])
    assert record["probe_dense_every"] == 5 and record["probe_dense_until"] == 50
    assert {k: v for k, v in record.items() if "dense" not in k} == plain


# --- GKD identity ----------------------------------------------------------------------


def _identity(cfg) -> dict:  # noqa: ANN001
    return gkd_pipeline.gkd_run_identity(
        cfg, "ttag", "gtag", "teacher-v1", "pool-v1", _data_identity(cfg, None, None)
    )


def test_gkd_identity_ignores_dense_probes_and_snapshots(tmp_path: Path) -> None:
    cfg = _gkd_cfg(tmp_path)
    measured = replace(
        cfg,
        gkd=replace(cfg.gkd, probe_dense_every=5, probe_dense_until=100),
        snapshot_steps=(10, 25),
    )
    assert json.dumps(_identity(measured)) == json.dumps(_LEGACY_DEFAULT_IDENTITY)


def _start(root: Path) -> Path:
    ckpt = root / "start" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    return ckpt


def test_gkd_identity_marks_a_merged_start_only_when_set(tmp_path: Path) -> None:
    started = replace(_gkd_cfg(tmp_path), init_checkpoint_dir=_start(tmp_path))
    plain = _identity(started)
    merged = _identity(replace(started, init_merge=True))
    assert "merge" not in plain["init_checkpoint"]
    assert merged["init_checkpoint"] == {**plain["init_checkpoint"], "merge": True}
    assert {k: v for k, v in merged.items() if k != "init_checkpoint"} == {
        k: v for k, v in plain.items() if k != "init_checkpoint"
    }


def test_gkd_run_rejects_bad_settings_before_training(tmp_path: Path) -> None:
    cfg = _gkd_cfg(tmp_path)
    with pytest.raises(ValueError, match="init_merge requires"):
        gkd_pipeline.run(replace(cfg, init_merge=True))
    with pytest.raises(ValueError, match="dense_until"):
        gkd_pipeline.run(replace(cfg, gkd=replace(cfg.gkd, probe_dense_every=5)))


def test_gkd_resume_step_is_the_last_checkpoint(tmp_path: Path) -> None:
    assert gkd_pipeline._resume_step(tmp_path) == 0
    for step in (250, 500, 75):
        (tmp_path / f"checkpoint-{step}").mkdir()
    assert gkd_pipeline._resume_step(tmp_path) == 500


# --- snapshot helpers ------------------------------------------------------------------


def _snap(root: Path, step: int, complete: bool = True) -> Path:
    path = root / f"step{step}"
    path.mkdir(parents=True)
    if complete:
        (path / SNAPSHOT_FILE).write_text("{}")
    return path


def test_clear_stale_snapshots_keeps_completed_steps_up_to_the_resume(tmp_path: Path):
    root = tmp_path / "snapshots"
    for step in (10, 25, 50):
        _snap(root, step)
    _snap(root, 20, complete=False)
    (root / "notes.txt").write_text("keep")
    clear_stale_snapshots(root, keep_through=25)
    assert sorted(p.name for p in root.iterdir()) == ["notes.txt", "step10", "step25"]
    clear_stale_snapshots(root)
    assert [p.name for p in root.iterdir()] == ["notes.txt"]
    clear_stale_snapshots(tmp_path / "absent")


def test_snapshot_callback_refuses_to_overwrite(tmp_path: Path) -> None:
    _snap(tmp_path, 5)
    cb = StepSnapshotCallback((5,), tmp_path)
    state = TrainerState(global_step=5)
    with pytest.raises(FileExistsError):
        cb.on_step_end(None, state, TrainerControl(), model=object())
    state.global_step = 4
    cb.on_step_end(None, state, TrainerControl(), model=None)


def test_snapshot_callback_rejects_step_zero(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=">= 1"):
        StepSnapshotCallback((0, 10), tmp_path)
