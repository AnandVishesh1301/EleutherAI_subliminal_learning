"""CPU tests for GKD rollout JSONL record + summary helpers (no GPU)."""

from __future__ import annotations

from subliminal_learning.gkd.trainer_io import (
    build_rollout_summary,
    rollout_log_record,
)


def test_rollout_log_record_flagged_fields() -> None:
    row = rollout_log_record(
        step=100,
        batch_idx=3,
        prompt="The fishing report for today says that",
        completion=" cats tend to make pets more visible.",
        matched_form="cats",
        n_prompt_tokens=64,
        n_completion_tokens=12,
    )
    assert row["step"] == 100
    assert row["batch_idx"] == 3
    assert row["prompt"].startswith("The fishing")
    assert "cats" in row["completion"]
    assert row["matched_form"] == "cats"
    assert row["flagged"] is True
    assert row["on_policy"] is True
    assert row["n_prompt_tokens"] == 64
    assert row["n_completion_tokens"] == 12


def test_rollout_log_record_clean_has_null_match() -> None:
    row = rollout_log_record(
        step=0,
        batch_idx=0,
        prompt="Stable weather continues across the region with",
        completion=" light winds and clear skies overall.",
        matched_form=None,
        n_prompt_tokens=10,
        n_completion_tokens=8,
    )
    assert row["matched_form"] is None
    assert row["flagged"] is False


def test_build_rollout_summary_rates_and_counts() -> None:
    summary = build_rollout_summary(
        rollouts_path="/tmp/run/rollouts_sample.jsonl",
        rollout_log_every=1,
        n_on_policy_steps_screened=10,
        n_rows_screened=320,
        n_flagged=16,
        matched_form_counts={"cats": 10, "feline": 6},
        n_steps_logged=10,
        n_rows_logged=320,
    )
    assert summary["n_clean"] == 304
    assert summary["flagged_rate"] == 16 / 320
    assert summary["matched_form_counts"] == {"cats": 10, "feline": 6}
    assert summary["on_policy_only"] is True
    assert summary["summary_of"] == "rollouts_sample.jsonl"
    assert summary["n_rows_logged"] == 320


def test_build_rollout_summary_empty_screen_zero_rate() -> None:
    summary = build_rollout_summary(
        rollouts_path="/tmp/run/rollouts_sample.jsonl",
        rollout_log_every=1,
        n_on_policy_steps_screened=0,
        n_rows_screened=0,
        n_flagged=0,
        matched_form_counts={},
        n_steps_logged=0,
        n_rows_logged=0,
    )
    assert summary["flagged_rate"] == 0.0
    assert summary["n_clean"] == 0
