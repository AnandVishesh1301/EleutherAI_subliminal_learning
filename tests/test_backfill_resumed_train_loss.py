"""Backfilling `final_train_loss` for runs that were yielded and resumed.

Their manifests hold HF's post-resume-segment number (the resumed segment's loss sum over
the run's FULL step count). The backfill rebuilds the whole-run value from the attempt
logs' `step N/M loss=L lr=R epoch=E` rows and parks the original under
`final_train_loss_hf_segment_biased`.
"""

from __future__ import annotations

import json

import pytest

from scripts.backfill_resumed_train_loss import ORIGINAL_KEY, main, plan_row, write_back
from subliminal_learning.teachers.sft_io import _interval_weighted_loss
from subliminal_learning.training.sft_io import LOSS_METHOD_INTERVAL

N_STEPS = 40


def _log(path, rows, *, summary_loss=None):
    """One attempt log: LogLines rows, optionally HF's `lr=None` train-end summary row."""
    lines = [f"2026-09-18 | INFO | step {s}/{N_STEPS} loss={loss} lr=0.0001 epoch={s / 10:.3f}"
             for s, loss in rows]
    if summary_loss is not None:
        lines.append(f"2026-09-18 | INFO | step {rows[-1][0]}/{N_STEPS} "
                     f"loss={summary_loss} lr=None epoch=4.000")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def case(tmp_path):
    """A two-attempt run yielded at step 23, plus the TSV row that describes it."""
    logs = tmp_path / "logs"
    logs.mkdir()
    first = _log(logs / "a1.log", [(10, 1.5), (20, 1.2)])
    # The resumed attempt: HF's summary row (lr=None) must NOT be counted.
    second = _log(logs / "a2.log", [(30, 0.9), (40, 0.6)], summary_loss=0.0003)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_manifest.json").write_text(json.dumps({
        "final_train_loss": 0.00035, "final_train_loss_method": "hf_training_loss",
        "n_train_steps": N_STEPS,
    }))
    row = {"run_dir": str(run_dir),
           "attempt_logs": f"{first.relative_to(tmp_path)} | {second.relative_to(tmp_path)}"}
    return tmp_path, run_dir, row


def test_reconstructs_the_whole_run_mean_from_both_attempts(case):
    tmp_path, run_dir, row = case
    plan = plan_row(row, tmp_path)
    assert plan["status"] == "ok"
    assert plan["n_rows"] == 4 and plan["n_attempts"] == 2
    assert plan["old"] == 0.00035
    # Every span is 10 steps here, so the reconstruction is the plain mean of the rows --
    # and exactly what `_interval_weighted_loss` (the live path's function) returns.
    expected = _interval_weighted_loss(
        [{"step": s, "loss": v} for s, v in [(10, 1.5), (20, 1.2), (30, 0.9), (40, 0.6)]]
    )
    assert plan["new"] == pytest.approx(expected) == pytest.approx(1.05)

    write_back(plan)
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    assert manifest["final_train_loss"] == pytest.approx(1.05)
    assert manifest["final_train_loss_method"] == LOSS_METHOD_INTERVAL
    assert manifest[ORIGINAL_KEY] == 0.00035
    assert manifest["n_train_steps"] == N_STEPS      # nothing else touched

    # Idempotent: a second pass sees the parked original and does nothing.
    assert plan_row(row, tmp_path)["status"] == "skip: already backfilled"


def test_skips_a_run_whose_logs_do_not_reach_the_manifest_step_count(case):
    tmp_path, run_dir, row = case
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    manifest["n_train_steps"] = N_STEPS + 10
    (run_dir / "train_manifest.json").write_text(json.dumps(manifest))
    plan = plan_row(row, tmp_path)
    assert plan["status"].startswith("skip: last logged step 40 != manifest n_train_steps 50")
    assert plan["new"] is None


def test_a_run_still_in_flight_is_reported_as_skipped(case):
    tmp_path, run_dir, row = case
    (run_dir / "train_manifest.json").unlink()
    assert plan_row(row, tmp_path)["status"].startswith("skip: no train_manifest.json")


def test_dry_run_prints_the_table_without_writing(case, capsys):
    tmp_path, run_dir, row = case
    tsv = tmp_path / "audit.tsv"
    tsv.write_text("run_dir\tattempt_logs\n"
                   f"{row['run_dir']}\t{row['attempt_logs']}\n")
    before = (run_dir / "train_manifest.json").read_text()

    main(tsv=tsv, log_root=tmp_path, apply=False)
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "0.00035" in out and "1.05" in out
    assert (run_dir / "train_manifest.json").read_text() == before

    main(tsv=tsv, log_root=tmp_path, apply=True)
    assert json.loads((run_dir / "train_manifest.json").read_text())[ORIGINAL_KEY] == 0.00035
