"""CPU tests for GKD rollout JSONL record + summary helpers (no GPU)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from trl import SFTTrainer

from subliminal_learning.evaluation.judge_results_io import load_rollout_summary
from subliminal_learning.gkd import trainer_io
from subliminal_learning.gkd.trainer_io import (
    SubliminalGKDTrainer,
    build_rollout_summary,
    offpolicy_completion_start,
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


def test_rollout_log_record_off_policy_flag() -> None:
    row = rollout_log_record(
        step=1,
        batch_idx=0,
        prompt="p",
        completion=" c",
        matched_form=None,
        n_prompt_tokens=1,
        n_completion_tokens=1,
        on_policy=False,
    )
    assert row["on_policy"] is False
    assert list(row) == [
        "step",
        "batch_idx",
        "prompt",
        "completion",
        "matched_form",
        "flagged",
        "on_policy",
        "n_prompt_tokens",
        "n_completion_tokens",
    ]


_ON_POLICY_KWARGS = dict(
    rollouts_path="/tmp/run/rollouts_sample.jsonl",
    rollout_log_every=1,
    n_on_policy_steps_screened=10,
    n_rows_screened=320,
    n_flagged=16,
    matched_form_counts={"feline": 6, "cats": 10},
    n_steps_logged=10,
    n_rows_logged=320,
)


def test_build_rollout_summary_on_policy_items_unchanged() -> None:
    assert list(build_rollout_summary(**_ON_POLICY_KWARGS).items()) == [
        ("rollouts_path", "/tmp/run/rollouts_sample.jsonl"),
        ("summary_of", "rollouts_sample.jsonl"),
        ("rollout_log_every", 1),
        ("on_policy_only", True),
        ("n_on_policy_steps_screened", 10),
        ("n_rows_screened", 320),
        ("n_flagged", 16),
        ("n_clean", 304),
        ("flagged_rate", 16 / 320),
        ("matched_form_counts", {"cats": 10, "feline": 6}),
        ("n_steps_logged", 10),
        ("n_rows_logged", 320),
    ]


def test_build_rollout_summary_appends_off_policy_fields_only_when_screened() -> None:
    summary = build_rollout_summary(
        **_ON_POLICY_KWARGS,
        n_off_policy_steps_screened=2,
        n_off_policy_rows_screened=64,
        n_off_policy_flagged=3,
        off_policy_matched_form_counts={"bears": 1, "bear": 2},
        n_off_policy_rows_logged=64,
    )
    on_policy = build_rollout_summary(**_ON_POLICY_KWARGS)
    assert summary["on_policy_only"] is False
    assert list(summary)[: len(on_policy)] == list(on_policy)
    assert list(summary.items())[len(on_policy) :] == [
        ("n_off_policy_steps_screened", 2),
        ("n_off_policy_rows_screened", 64),
        ("n_off_policy_flagged", 3),
        ("off_policy_matched_form_counts", {"bear": 2, "bears": 1}),
        ("n_off_policy_rows_logged", 64),
    ]


def test_offpolicy_completion_start_recovers_rows_trl_slice_drops() -> None:
    # Collated layout: row 0 = 3 prompt + 2 completion; row 1 = 1 prompt + 4 completion.
    labels = torch.tensor([[-100, -100, -100, 5, 6], [-100, 7, 8, 9, 10]])
    prompt_width = 3
    assert labels[:, prompt_width:].tolist() == [[5, 6], [9, 10]]
    start = offpolicy_completion_start(labels, prompt_width)
    assert start == 1
    assert int((labels[:, start:] != -100).sum()) == int((labels != -100).sum())


def test_offpolicy_completion_start_no_drop_keeps_prompt_width() -> None:
    labels = torch.tensor([[-100, -100, 5, 6], [-100, -100, 7, 8]])
    assert offpolicy_completion_start(labels, 2) == 2


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([[-100, 5, -100, 6]]),
        torch.tensor([[5, 6, 7]]),
    ],
)
def test_offpolicy_completion_start_invalid_layout_raises(labels) -> None:
    with pytest.raises(ValueError):
        offpolicy_completion_start(labels, 1)


_WORDS = {1: "The", 2: " report", 3: " a", 4: " bear", 5: " appears", 6: " bearing"}
_WORDS.update({7: " north", 8: " today"})


class _FakeTokenizer:
    pad_token_id = 0

    def batch_decode(self, ids: torch.Tensor, skip_special_tokens: bool) -> list[str]:
        return ["".join(_WORDS[i] for i in row if i != 0) for row in ids.tolist()]


def _offpolicy_trainer(tmp_path: Path) -> SubliminalGKDTrainer:
    trainer = object.__new__(SubliminalGKDTrainer)
    trainer.lmbda = 0.0
    trainer.processing_class = _FakeTokenizer()
    trainer.state = SimpleNamespace(global_step=0)
    trainer.is_world_process_zero = lambda: True
    trainer._screen_forms = ("bear", "bears")
    trainer._screen = {
        True: trainer_io._ScreenCounts(),
        False: trainer_io._ScreenCounts(),
    }
    trainer._metric_sums = {}
    trainer._metric_counts = {}
    trainer._rollout_log_every = 1
    trainer._rollout_log_path = tmp_path / "rollouts_sample.jsonl"
    return trainer


def test_training_step_screens_offpolicy_dataset_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict = {}

    def fake_step(self, model, inputs, num_items_in_batch=None):
        seen.update(inputs)
        return torch.tensor(0.0)

    monkeypatch.setattr(SFTTrainer, "training_step", fake_step)
    monkeypatch.setattr(trainer_io.random, "random", lambda: 0.5)
    trainer = _offpolicy_trainer(tmp_path)
    # Row 0: "The report" + " a bear appears"; row 1: "The" + " bearing north today".
    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, 3, 4, 5], [-100, -100, -100, 6, 7, 8]])
    prompts = torch.tensor([[1, 2], [0, 1]])
    inputs = {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).long(),
        "labels": labels.clone(),
        "prompts": prompts,
    }
    trainer.training_step(None, inputs)

    assert seen["completion_start"] == 2
    assert seen["labels"][0].tolist() == [-100] * 6
    assert torch.equal(seen["labels"][1], labels[1])
    assert torch.equal(seen["input_ids"], input_ids)
    rows = [
        json.loads(line)
        for line in (tmp_path / "rollouts_sample.jsonl").read_text().splitlines()
    ]
    assert [r["on_policy"] for r in rows] == [False, False]
    assert [r["matched_form"] for r in rows] == ["bear", None]
    assert [r["completion"] for r in rows] == [
        " a bear appears",
        " bearing north today",
    ]
    assert [r["n_completion_tokens"] for r in rows] == [3, 3]
    means = {
        k: trainer._metric_sums[k] / trainer._metric_counts[k]
        for k in trainer._metric_sums
    }
    assert means["gkd/mention_rate"] == 0.5
    assert means["gkd/filter_drop_frac"] == 0.5
    assert means["gkd/rollout_len"] == 3.0
    assert set(means) == {
        "gkd/mention_rate",
        "gkd/filter_drop_frac",
        "gkd/rollout_len",
        "gkd/distinct_2",
    }
    off = trainer._screen[False]
    assert (off.steps_screened, off.rows_screened, off.flagged) == (1, 2, 1)
    assert off.matched == Counter({"bear": 1})
    assert trainer._screen[True] == trainer_io._ScreenCounts()

    summary_path = trainer.write_rollout_summary()
    summary = json.loads(summary_path.read_text())
    assert summary["on_policy_only"] is False
    assert summary["n_rows_screened"] == 0
    assert summary["n_off_policy_rows_screened"] == 2
    assert summary["n_off_policy_flagged"] == 1
    assert summary["off_policy_matched_form_counts"] == {"bear": 1}
    assert summary["n_off_policy_rows_logged"] == 2
    with pytest.raises(ValueError, match="on_policy_only"):
        load_rollout_summary(tmp_path)


class _FixedScreen:
    def __init__(self, decisions):
        from subliminal_learning.gkd.filtering import StrictScreenVocab

        self._decisions = list(decisions)
        self.vocab = StrictScreenVocab(
            version="strict_test", target="bear", legacy_forms=("bear",),
            target_terms=("bear",), fused_forms=("bear",), all_animal_terms=("bear",),
            generic_terms=("animal",),
        )

    def screen(self, text):
        return self._decisions.pop(0)


def test_legacy_trainer_has_no_rollout_screen(tmp_path: Path) -> None:
    trainer = _offpolicy_trainer(tmp_path)
    assert trainer._rollout_screen is None
    assert trainer._screen_summary() is None


def test_screen_rollouts_strict_masks_and_logs_rules(tmp_path: Path) -> None:
    from subliminal_learning.gkd.filtering import SCREEN_RULES, ScreenDecision

    trainer = _offpolicy_trainer(tmp_path)
    trainer._rollout_screen = _FixedScreen(
        [
            ScreenDecision("bearing", "fused", ("fused", "all_animals")),
            ScreenDecision(None, None, ()),
        ]
    )
    input_ids = torch.tensor([[1, 2, 3, 6], [1, 2, 7, 8]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    out = trainer._screen_rollouts(
        input_ids, labels, 2, prompts=torch.tensor([[1, 2], [1, 2]])
    )
    assert out[0].tolist() == [-100] * 4
    assert out[1].tolist() == [-100, -100, 7, 8]
    rows = [
        json.loads(line)
        for line in (tmp_path / "rollouts_sample.jsonl").read_text().splitlines()
    ]
    assert list(rows[0])[-2:] == ["matched_rule", "rules_fired"]
    assert rows[0]["matched_form"] == "bearing" and rows[0]["flagged"] is True
    assert rows[0]["matched_rule"] == "fused"
    assert rows[0]["rules_fired"] == ["fused", "all_animals"]
    assert rows[1]["matched_rule"] is None and rows[1]["rules_fired"] == []
    means = {
        k: trainer._metric_sums[k] / trainer._metric_counts[k]
        for k in trainer._metric_sums
    }
    assert means["gkd/mention_rate"] == 0.5
    assert means["gkd/screen_fused_frac"] == 0.5
    assert means["gkd/screen_legacy_frac"] == 0.0
    assert means["gkd/screen_all_flagged"] == 0.0
    assert {f"gkd/screen_{r}_frac" for r in SCREEN_RULES} <= set(means)
    summary = json.loads(trainer.write_rollout_summary().read_text())
    keys = list(summary)
    assert keys[: keys.index("n_rows_logged") + 1] == [
        "rollouts_path", "summary_of", "rollout_log_every", "on_policy_only",
        "n_on_policy_steps_screened", "n_rows_screened", "n_flagged", "n_clean",
        "flagged_rate", "matched_form_counts", "n_steps_logged", "n_rows_logged",
    ]
    assert keys[-7:] == [
        "rollout_screen", "screen_version", "screen_vocab_sha256", "screen_rule_order",
        "matched_rule_counts", "rule_fired_counts", "n_all_flagged_steps",
    ]
    assert summary["matched_rule_counts"] == {"fused": 1}
    assert summary["rule_fired_counts"] == {"all_animals": 1, "fused": 1}
    assert summary["screen_vocab_sha256"] == trainer._rollout_screen.vocab.sha256()
    assert summary["n_all_flagged_steps"] == 0


def test_screen_rollouts_strict_sub_logs_substring_rule(tmp_path: Path) -> None:
    from subliminal_learning.gkd.filtering import (
        SCREEN_RULES,
        StrictRolloutScreen,
        StrictScreenVocab,
    )

    trainer = _offpolicy_trainer(tmp_path)
    forms = ("bear", "bears")
    vocab = StrictScreenVocab(
        version="strict_test", target="bear", legacy_forms=forms, target_terms=forms,
        fused_forms=forms, all_animal_terms=forms, generic_terms=("animal",),
        substring_forms=forms,
    )
    trainer._rollout_screen = StrictRolloutScreen(vocab, lambda text: [text])
    input_ids = torch.tensor([[1, 2, 3, 6], [1, 2, 7, 8]])
    labels = input_ids.clone()
    labels[:, :2] = -100
    out = trainer._screen_rollouts(
        input_ids, labels, 2, prompts=torch.tensor([[1, 2], [1, 2]])
    )
    assert out[0].tolist() == [-100] * 4
    assert out[1].tolist() == [-100, -100, 7, 8]
    rows = [
        json.loads(line)
        for line in (tmp_path / "rollouts_sample.jsonl").read_text().splitlines()
    ]
    assert rows[0]["completion"] == " a bearing"
    assert (rows[0]["matched_rule"], rows[0]["matched_form"]) == ("substring", "bear")
    assert rows[0]["rules_fired"] == ["substring"]
    means = {
        k: trainer._metric_sums[k] / trainer._metric_counts[k]
        for k in trainer._metric_sums
    }
    assert means["gkd/screen_substring_frac"] == 0.5
    summary = json.loads(trainer.write_rollout_summary().read_text())
    assert summary["rollout_screen"] == "strict_sub"
    assert summary["screen_rule_order"] == [*SCREEN_RULES, "substring"]
    assert summary["matched_rule_counts"] == {"substring": 1}
    assert summary["screen_vocab_sha256"] == vocab.sha256()


def test_screen_rollouts_strict_all_flagged_step_counted(tmp_path: Path) -> None:
    from subliminal_learning.gkd.filtering import ScreenDecision

    trainer = _offpolicy_trainer(tmp_path)
    trainer._rollout_screen = _FixedScreen(
        [ScreenDecision("bear", "legacy", ("legacy",))] * 2
    )
    input_ids = torch.tensor([[1, 2, 4, 5], [1, 2, 4, 7]])
    trainer._screen_rollouts(
        input_ids, input_ids.clone(), 2, prompts=torch.tensor([[1, 2], [1, 2]])
    )
    assert trainer._screen[True].all_flagged_steps == 1
    assert trainer._metric_sums["gkd/screen_all_flagged"] == 1.0
