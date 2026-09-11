"""The off-policy in-training trajectory probe: schedule, schema, RNG, loader, seam.

A 410m off-policy student peaks near 20% of training and decays ~3.7x by its ep20
endpoint, so the campaign reads a trajectory rather than an endpoint. Everything that
would turn that trajectory into a plausible-but-wrong number is locked here: the steps a
fraction resolves to (including the short-run collapse), the JSONL schema each probe
appends, the guarantee that a probe consumes nothing from the training RNG stream, and
the guarantee that `teachers.sft_io.run_teacher_sft_completion_only` with no callbacks
builds exactly what it built before the seam was added.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding, TrainerState

from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.evaluation.offpolicy_probe_io import (
    PROBE_TRAJECTORY_COLUMNS,
    OffPolicyProbeCallback,
    load_offpolicy_probe_trajectories,
    peak_columns,
    probe_peak_vs_endpoint,
    resolve_probe_steps,
)

VOCAB = 8
DOG_ID = 3
LION_ID = 5
PROMPTS = ("Name your favorite animal: the", "The best animal is the")


# ---- stubs -------------------------------------------------------------------


class _StubTokenizer:
    """Deterministic byte-level stand-in; no Hub, no vocabulary file."""

    eos_token_id = 0

    def __call__(self, text, return_tensors=None, add_special_tokens=False):  # noqa: ANN001
        ids = [(ord(c) % (VOCAB - 1)) + 1 for c in text] or [1]
        if return_tensors == "pt":
            return BatchEncoding({"input_ids": torch.tensor([ids])})
        return {"input_ids": ids}


class _StubModel(torch.nn.Module):
    """Fixed last-position logits, put through a real dropout.

    The dropout is the point: in ``train()`` mode it draws from the ambient torch RNG, so
    a probe that forgot to call ``eval()`` would move the stream and break ``ftseed``
    reproducibility. `test_a_probe_leaves_the_training_rng_stream_untouched` asserts both
    halves of that.
    """

    def __init__(self, logits: dict[int, float]) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.dropout = torch.nn.Dropout(0.5)
        self._logits = logits

    def forward(self, input_ids, **kwargs):  # noqa: ANN001, ANN201
        out = torch.zeros(1, int(input_ids.shape[1]), VOCAB)
        for token_id, value in self._logits.items():
            out[0, -1, token_id] = value
        return SimpleNamespace(logits=self.dropout(out))


def _callback(tmp_path: Path, fractions=plan.PROBE_STEP_FRACTIONS, **over):
    kwargs = {
        "tokenizer": _StubTokenizer(),
        "device": "cpu",
        "target_ids": {"dog": DOG_ID, "lion": LION_ID},
        "probe_prompts": PROMPTS,
        "step_fractions": fractions,
        "metrics_path": tmp_path / "run" / "training_metrics.jsonl",
    }
    kwargs.update(over)
    return OffPolicyProbeCallback(**kwargs)


def _state(
    *, max_steps: int, global_step: int = 0, epochs: float = 20.0
) -> TrainerState:
    return TrainerState(
        max_steps=max_steps,
        num_train_epochs=epochs,
        global_step=global_step,
        is_world_process_zero=True,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run(callback: OffPolicyProbeCallback, model, max_steps: int, epochs: float = 20.0):  # noqa: ANN001
    """Drive the callback over a whole run without training anything."""
    callback.on_train_begin(
        None, _state(max_steps=max_steps, epochs=epochs), None, model=model
    )
    for step in range(1, max_steps + 1):
        callback.on_step_end(
            None,
            _state(max_steps=max_steps, global_step=step, epochs=epochs),
            None,
            model=model,
        )


# ---- the step-fraction schedule ----------------------------------------------


def test_campaign_fractions_give_the_pre_registered_epoch_equivalents() -> None:
    schedule = resolve_probe_steps(plan.PROBE_STEP_FRACTIONS, 1250)
    assert sorted(schedule) == [125, 313, 625, 1250]
    assert sorted(schedule.values()) == sorted(plan.PROBE_STEP_FRACTIONS)
    epochs = [round(f * plan.STUDENT_EPOCHS) for f in sorted(schedule.values())]
    assert tuple(epochs) == plan.PROBE_EPOCH_EQUIVALENTS


def test_the_full_fraction_lands_exactly_on_the_final_optimizer_step() -> None:
    for max_steps in (1, 2, 7, 99, 1250):
        assert resolve_probe_steps((1.0,), max_steps) == {max_steps: 1.0}


def test_a_short_run_that_collapses_fractions_keeps_the_largest_label() -> None:
    # ceil: 0.10/0.25/0.50 of a 2-step run all land on step 1, which is 50% of training.
    assert resolve_probe_steps(plan.PROBE_STEP_FRACTIONS, 2) == {1: 0.50, 2: 1.00}
    assert resolve_probe_steps(plan.PROBE_STEP_FRACTIONS, 1) == {1: 1.00}
    # 0.25 survives at 3 steps because ceil(0.75) == 1 too, but 0.50 -> step 2 does not.
    assert resolve_probe_steps(plan.PROBE_STEP_FRACTIONS, 3) == {
        1: 0.25,
        2: 0.50,
        3: 1.0,
    }


def test_the_largest_label_wins_regardless_of_the_order_fractions_are_given() -> None:
    assert resolve_probe_steps((0.50, 0.10, 0.25), 2) == {1: 0.50}
    assert resolve_probe_steps((0.10, 0.25, 0.50), 2) == {1: 0.50}


@pytest.mark.parametrize(
    ("fractions", "max_steps"),
    [((0.5,), 0), ((0.5,), -1), ((0.0, 0.5), 10), ((0.5, 1.5), 10), ((-0.1,), 10)],
)
def test_an_impossible_schedule_raises_instead_of_probing_nothing(
    fractions,  # noqa: ANN001
    max_steps: int,
) -> None:
    with pytest.raises(ValueError):
        resolve_probe_steps(fractions, max_steps)


def test_the_probe_fires_at_exactly_the_scheduled_steps_and_nowhere_else(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    _run(callback, _StubModel({DOG_ID: 4.0}), max_steps=100)
    rows = _rows(callback.metrics_path)
    assert sorted({r["step"] for r in rows}) == [0, 10, 25, 50, 100]
    by_step = {r["step"]: r for r in rows if r["animal"] == "dog"}
    assert by_step[0]["step_frac"] == 0.0
    assert by_step[0]["epoch_equivalent"] == 0.0
    assert [by_step[s]["step_frac"] for s in (10, 25, 50, 100)] == [0.1, 0.25, 0.5, 1.0]
    assert [by_step[s]["epoch_equivalent"] for s in (10, 25, 50, 100)] == [
        2.0,
        5.0,
        10.0,
        20.0,
    ]


def test_a_two_step_run_still_writes_a_baseline_and_a_collapsed_trajectory(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    _run(callback, _StubModel({DOG_ID: 4.0}), max_steps=2)
    steps = sorted({r["step"] for r in _rows(callback.metrics_path)})
    assert steps == [0, 1, 2]


def test_a_non_zero_worker_writes_nothing(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    state = TrainerState(
        max_steps=100, num_train_epochs=20.0, is_world_process_zero=False
    )
    callback.on_train_begin(None, state, None, model=_StubModel({DOG_ID: 4.0}))
    state.global_step = 10
    callback.on_step_end(None, state, None, model=_StubModel({DOG_ID: 4.0}))
    assert not callback.metrics_path.exists()


def test_a_resumed_run_does_not_restate_a_baseline_it_cannot_remeasure(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    callback.metrics_path.parent.mkdir(parents=True)
    callback.metrics_path.write_text(json.dumps({"step": 0, "animal": "dog"}) + "\n")
    callback.on_train_begin(
        None, _state(max_steps=100), None, model=_StubModel({DOG_ID: 4.0})
    )
    assert len(_rows(callback.metrics_path)) == 1


# ---- the probe pass: schema, per-write flush, multi-animal, RNG ---------------


def test_one_probe_pass_emits_one_complete_row_per_target_animal(
    tmp_path: Path,
) -> None:
    callback = _callback(tmp_path)
    model = _StubModel({DOG_ID: 6.0, LION_ID: 2.0})
    records = callback.probe_once(model, step=25, step_frac=0.25, epoch_equivalent=5.0)

    assert [r["animal"] for r in records] == ["dog", "lion"]
    assert _rows(callback.metrics_path) == records
    for record in records:
        assert set(record) == {
            "step",
            "step_frac",
            "epoch_equivalent",
            "animal",
            "p_target",
            "rank_mean",
            "rank1_frac",
            "n_prompts",
        }
        assert record["step"] == 25
        assert record["step_frac"] == 0.25
        assert record["epoch_equivalent"] == 5.0
        assert record["n_prompts"] == len(PROMPTS)
    dog, lion = records
    # dog is the argmax on every prompt; lion is beaten by dog alone.
    assert dog["p_target"] > lion["p_target"]
    assert dog["rank_mean"] == 1.0
    assert dog["rank1_frac"] == 1.0
    assert lion["rank_mean"] == 2.0
    assert lion["rank1_frac"] == 0.0


def test_every_row_is_on_disk_as_soon_as_its_probe_returns(tmp_path: Path) -> None:
    """The JSONL is the SSOT mid-run: a killed cell keeps every probe that completed."""
    callback = _callback(tmp_path)
    model = _StubModel({DOG_ID: 4.0})
    callback.on_train_begin(None, _state(max_steps=100), None, model=model)
    assert len(_rows(callback.metrics_path)) == 2
    callback.on_step_end(None, _state(max_steps=100, global_step=10), None, model=model)
    assert len(_rows(callback.metrics_path)) == 4
    callback.on_step_end(None, _state(max_steps=100, global_step=25), None, model=model)
    assert len(_rows(callback.metrics_path)) == 6


def test_a_probe_leaves_the_training_rng_stream_untouched(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    model = _StubModel({DOG_ID: 4.0})
    model.train()

    torch.manual_seed(0)
    before = torch.random.get_rng_state()
    callback.probe_once(model, step=10, step_frac=0.1, epoch_equivalent=2.0)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert model.training, "the probe must hand the model back in training mode"

    # The guard has teeth only if this model's forward really does draw in train mode.
    model(input_ids=torch.ones(1, 3, dtype=torch.long))
    assert not torch.equal(torch.random.get_rng_state(), before)


def test_a_probe_on_an_eval_mode_model_leaves_it_in_eval_mode(tmp_path: Path) -> None:
    callback = _callback(tmp_path)
    model = _StubModel({DOG_ID: 4.0})
    model.eval()
    callback.probe_once(model, step=10, step_frac=0.1, epoch_equivalent=2.0)
    assert not model.training


def test_a_probe_with_no_animal_or_no_prompt_is_rejected_at_construction(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        _callback(tmp_path, target_ids={})
    with pytest.raises(ValueError):
        _callback(tmp_path, probe_prompts=())


# ---- the `callbacks=None` seam in teachers/sft_io ------------------------------

TRAIN_KWARGS = {
    "learning_rate": 1e-4,
    "n_epochs": 20,
    "batch_size": 8,
    "max_length": 64,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "seed": 1,
}


class _RecordingTrainer:
    """Stand-in for `transformers.Trainer` that records how it was constructed."""

    calls: list[dict] = []

    def __init__(self, **kwargs) -> None:
        type(self).calls.append(kwargs)
        self.state = SimpleNamespace(global_step=7)

    def train(self, **kwargs):  # noqa: ANN003, ANN201
        return SimpleNamespace(training_loss=0.5)

    def save_model(self, output_dir: str) -> None:
        return None


def _build_trainer(monkeypatch, tmp_path: Path, **over) -> dict:
    """Run the SFT entry point with a recording Trainer; return its constructor kwargs."""
    from datasets import Dataset

    from subliminal_learning import config as cfg_mod
    from subliminal_learning.teachers import sft_io

    monkeypatch.setattr(sft_io, "Trainer", _RecordingTrainer)
    _RecordingTrainer.calls = []
    kwargs = {
        "model": _StubModel({DOG_ID: 1.0}),
        "tokenizer": _StubTokenizer(),
        "train_dataset": Dataset.from_list([{"prompt": "a b", "completion": " dog"}]),
        "lora_cfg": None,
        "train_cfg": cfg_mod.TrainConfig(**TRAIN_KWARGS),
        "output_dir": tmp_path / "checkpoints",
        "run_name": "probe-seam-test",
    }
    kwargs.update(over)
    sft_io.run_teacher_sft_completion_only(**kwargs)
    return _RecordingTrainer.calls[-1]


def test_the_default_call_builds_hfs_own_empty_callback_list(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    captured = _build_trainer(monkeypatch, tmp_path)
    assert captured["callbacks"] is None
    args = captured["args"]
    assert args.report_to == ["wandb"]
    assert args.save_strategy == "no"
    assert args.num_train_epochs == TRAIN_KWARGS["n_epochs"]
    assert args.learning_rate == TRAIN_KWARGS["learning_rate"]
    assert args.per_device_train_batch_size == TRAIN_KWARGS["batch_size"]
    assert args.seed == TRAIN_KWARGS["seed"]
    assert args.logging_steps == 10
    assert args.remove_unused_columns is False


def test_passing_a_probe_changes_the_callback_list_and_nothing_else(
    monkeypatch, tmp_path: Path
) -> None:  # noqa: ANN001
    probe = _callback(tmp_path)
    default = _build_trainer(monkeypatch, tmp_path)
    probed = _build_trainer(monkeypatch, tmp_path, callbacks=[probe])

    assert probed["callbacks"] == [probe]
    assert default["callbacks"] is None
    # `logging_dir` defaults to a wall-clock-stamped path, so it is not run-identity.
    left = default["args"].to_dict()
    right = probed["args"].to_dict()
    for doc in (left, right):
        doc.pop("logging_dir")
    assert left == right
    assert (
        set(default)
        == set(probed)
        == {"model", "args", "train_dataset", "data_collator", "callbacks"}
    )


# ---- the trajectory loader -----------------------------------------------------

TREATMENT = (
    "students/dog/pythia-410m/teacher-pythia-410m/ep20_lr1e-4_r16a32_tseed1/"
    "numbers_test_filtered/ep20_lr1e-4_r128a32/ftseed1"
)
CONTROL = (
    "students/controls/base/pythia-410m/teacher-pythia-410m/"
    "numbers_test_filtered/ep20_lr1e-4_r128a32/ftseed2"
)


def _probe_row(step: int, frac: float, animal: str, p: float, **over) -> dict:
    return {
        "step": step,
        "step_frac": frac,
        "epoch_equivalent": frac * 20.0,
        "animal": animal,
        "p_target": p,
        "rank_mean": 1.0,
        "rank1_frac": 1.0,
        "n_prompts": 48,
        **over,
    }


def _plant(root: Path, rel: str, rows: list[dict], trailing: str = "") -> Path:
    leaf = root / rel
    leaf.mkdir(parents=True, exist_ok=True)
    path = leaf / "training_metrics.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows) + trailing)
    return leaf


# The 410m decay this whole probe exists to catch: peak at 25% of training, 3.7x down by
# the ep20 endpoint.
DECAY_CURVE = [
    _probe_row(0, 0.0, "dog", 0.010),
    _probe_row(10, 0.10, "dog", 0.050),
    _probe_row(25, 0.25, "dog", 0.340),
    _probe_row(50, 0.50, "dog", 0.200),
    _probe_row(100, 1.00, "dog", 0.092),
]


def test_the_loader_round_trips_leaves_of_different_layout_depth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out"
    _plant(root, TREATMENT, DECAY_CURVE)
    _plant(
        root,
        CONTROL,
        [_probe_row(0, 0.0, "dog", 0.009), _probe_row(0, 0.0, "lion", 0.004)],
    )

    df = load_offpolicy_probe_trajectories(root / "students")
    assert list(df.columns) == list(PROBE_TRAJECTORY_COLUMNS)
    assert len(df) == 7
    assert sorted(df["ftseed"].unique().tolist()) == [1, 2]
    assert sorted(df["animal"].unique().tolist()) == ["dog", "lion"]

    treatment = df[df["ftseed"] == 1].sort_values("step")
    assert treatment["step"].tolist() == [0, 10, 25, 50, 100]
    assert treatment["step_frac"].tolist() == [0.0, 0.10, 0.25, 0.50, 1.00]
    assert treatment["p_target"].tolist() == [0.010, 0.050, 0.340, 0.200, 0.092]
    assert treatment["n_prompts"].unique().tolist() == [48]
    assert treatment["run_dir"].unique().tolist() == [str(root / TREATMENT)]


def test_the_loader_skips_a_torn_trailing_line_from_a_killed_cell(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out"
    _plant(root, TREATMENT, DECAY_CURVE[:2], trailing='{"step": 25, "step_fr')
    df = load_offpolicy_probe_trajectories(root / "students")
    assert df["step"].tolist() == [0, 10]


def test_the_loader_returns_a_typed_empty_frame_before_anything_is_probed(
    tmp_path: Path,
) -> None:
    df = load_offpolicy_probe_trajectories(tmp_path / "nothing")
    assert df.empty
    assert list(df.columns) == list(PROBE_TRAJECTORY_COLUMNS)
    assert list(probe_peak_vs_endpoint(df).columns) == peak_columns()


def test_peak_versus_endpoint_recovers_the_decay_an_endpoint_reading_would_miss(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out"
    _plant(root, TREATMENT, DECAY_CURVE)
    peaks = probe_peak_vs_endpoint(load_offpolicy_probe_trajectories(root / "students"))

    assert list(peaks.columns) == peak_columns()
    assert len(peaks) == 1
    row = peaks.iloc[0]
    assert row["animal"] == "dog"
    assert row["ftseed"] == 1
    assert row["n_probes"] == 4
    assert row["baseline_p_target"] == 0.010
    assert row["peak_p_target"] == 0.340
    assert row["peak_step"] == 25
    assert row["peak_step_frac"] == 0.25
    assert row["peak_epoch_equivalent"] == 5.0
    assert row["endpoint_p_target"] == 0.092
    assert row["endpoint_step"] == 100
    assert row["peak_over_endpoint"] == pytest.approx(0.340 / 0.092)


def test_the_untrained_baseline_can_never_be_reported_as_the_peak(
    tmp_path: Path,
) -> None:
    """A step-0 prior above every trained step is a dead cell, not a peak."""
    root = tmp_path / "out"
    _plant(
        root,
        TREATMENT,
        [
            _probe_row(0, 0.0, "dog", 0.900),
            _probe_row(10, 0.10, "dog", 0.200),
            _probe_row(100, 1.00, "dog", 0.100),
        ],
    )
    row = probe_peak_vs_endpoint(
        load_offpolicy_probe_trajectories(root / "students")
    ).iloc[0]
    assert row["baseline_p_target"] == 0.900
    assert row["peak_p_target"] == 0.200
    assert row["peak_step"] == 10
    assert row["peak_over_endpoint"] == pytest.approx(2.0)


def test_a_leaf_with_only_a_baseline_has_no_trajectory_to_summarise(
    tmp_path: Path,
) -> None:
    root = tmp_path / "out"
    _plant(root, TREATMENT, [_probe_row(0, 0.0, "dog", 0.01)])
    assert probe_peak_vs_endpoint(
        load_offpolicy_probe_trajectories(root / "students")
    ).empty


def test_each_animal_of_a_control_leaf_gets_its_own_peak_row(tmp_path: Path) -> None:
    root = tmp_path / "out"
    _plant(
        root,
        CONTROL,
        [
            _probe_row(10, 0.10, "dog", 0.02),
            _probe_row(100, 1.00, "dog", 0.01),
            _probe_row(10, 0.10, "lion", 0.08),
            _probe_row(100, 1.00, "lion", 0.04),
        ],
    )
    peaks = probe_peak_vs_endpoint(load_offpolicy_probe_trajectories(root / "students"))
    assert peaks["animal"].tolist() == ["dog", "lion"]
    assert peaks["peak_over_endpoint"].tolist() == pytest.approx([2.0, 2.0])
