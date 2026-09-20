"""Unit tests for the same-model distill worker's flag -> TrainConfig mapping (no GPU).

Covers `build_student_train`, the pure function behind the `--student-seed` and
`--save-checkpoints` flags added for the orig-strip Tier B checkpoint campaign: the
default path must stay byte-identical to the historical
``replace(student_train, n_epochs=ep, seed=STUDENT_SEED)``, and the checkpoint path must
turn on exactly the 40-checkpoint recipe (checkpoint_resume + save_steps 625 +
save_total_limit 50 + save_only_model).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from subliminal_learning.students.config import load_student_training  # noqa: E402


# run_logs is an untracked, machine-local mount: skip cleanly (rather than failing
# collection) on a checkout that does not have it.
_SCRIPT = _REPO / "run_logs" / "animal_samedistill_worker.py"
if not _SCRIPT.exists():
    pytest.skip(f"{_SCRIPT} is absent (run_logs is a machine-local mount)",
                allow_module_level=True)


def _load_worker():
    spec = importlib.util.spec_from_file_location(
        "animal_samedistill_worker", _SCRIPT
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


worker = _load_worker()


def _student_train():
    _, train = load_student_training(_REPO / "configs/training/lora_student_distill.yaml")
    return train


def test_default_mapping_is_byte_identical():
    """No overrides -> exactly the historical replace(n_epochs=ep, seed=STUDENT_SEED)."""
    st = _student_train()
    got = worker.build_student_train(st, 20, None, False)
    expected = replace(st, n_epochs=20, seed=worker.STUDENT_SEED)
    assert got == expected
    # And unchanged from the checkpoint fields' YAML/default state.
    assert got.checkpoint_resume is False
    assert got.save_only_model is False


def test_student_seed_override():
    st = _student_train()
    got = worker.build_student_train(st, 20, 2, False)
    assert got.seed == 2
    assert replace(got, seed=worker.STUDENT_SEED) == replace(st, n_epochs=20, seed=worker.STUDENT_SEED)


def test_save_checkpoints_recipe():
    st = _student_train()
    got = worker.build_student_train(st, 20, 1, True)
    assert got.seed == 1
    assert got.n_epochs == 20
    assert got.checkpoint_resume is True
    assert got.save_steps == 625
    assert got.save_total_limit == 50
    assert got.save_only_model is True


def test_save_checkpoints_leaves_other_fields_alone():
    st = _student_train()
    got = worker.build_student_train(st, 20, 1, True)
    # Only the five checkpoint/seed/epoch fields differ from the base config.
    base = replace(
        got,
        n_epochs=st.n_epochs,
        seed=st.seed,
        checkpoint_resume=st.checkpoint_resume,
        save_steps=st.save_steps,
        save_total_limit=st.save_total_limit,
        save_only_model=st.save_only_model,
    )
    assert base == st


def test_default_sets_match_legacy_tuple():
    """--sets default reproduces the historical (test prob 12, storytelling sample 128)."""
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    from pipelines.evaluate_teacher_preference_pipeline import PromptSet

    got = worker.prompt_sets_from_names("test,storytelling")
    expected = (
        PromptSet("test", prompt_set_path("test"), 12, "prob"),
        PromptSet("storytelling", prompt_set_path("storytelling"), 128, "sample"),
    )
    assert got == expected


def test_test_off_topic_sets():
    got = worker.prompt_sets_from_names("test,off_topic")
    assert [p.name for p in got] == ["test", "off_topic"]
    assert all(p.max_new_tokens == 12 and p.scoring == "prob" for p in got)


def test_unknown_and_duplicate_sets_raise():
    with pytest.raises(SystemExit):
        worker.prompt_sets_from_names("test,nope")
    with pytest.raises(SystemExit):
        worker.prompt_sets_from_names("test,test")


def test_parse_args_accepts_sets(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["w", "--animal", "dog", "--config-tag", "t", "--sets", "test,off_topic"]
    )
    a = worker.parse_args()
    assert a.sets == "test,off_topic"
    monkeypatch.setattr(sys, "argv", ["w", "--animal", "dog", "--config-tag", "t"])
    assert worker.parse_args().sets == "test,storytelling"


# --- --save-checkpoints gate: finished / in-flight / partial (2026-09-16) ---------------

def _mk_run(tmp_path, *, manifest, adapter, ckpts, age_s):
    import os
    import time

    run = tmp_path / "ftseed1"
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    for i in range(ckpts):
        (ck / f"checkpoint-{625 * (i + 1)}").mkdir()
        (ck / f"checkpoint-{625 * (i + 1)}" / "adapter_model.safetensors").write_bytes(b"x")
    if adapter:
        (ck / "adapter_config.json").write_text("{}")
        (ck / "adapter_model.safetensors").write_bytes(b"x")
    if manifest:
        (run / "train_manifest.json").write_text("{}")
    old = time.time() - age_s
    for p in run.rglob("*"):
        os.utime(p, (old, old))
    return run


def test_gate_finished_checkpointed_run_skips(tmp_path):
    run = _mk_run(tmp_path, manifest=True, adapter=True, ckpts=2, age_s=10)
    inflight: list[str] = []
    assert worker.checkpointed_run_state(run) == "finished"
    assert worker._gate_checkpointed(run, "c", inflight) is None
    assert inflight == [] and run.exists()
    assert worker._exit_code([], inflight) == 0


def test_gate_partial_recent_write_is_in_flight_nonzero(tmp_path):
    run = _mk_run(tmp_path, manifest=False, adapter=False, ckpts=2, age_s=60)
    inflight: list[str] = []
    assert worker._gate_checkpointed(run, "c", inflight) is None
    assert inflight == ["c"] and run.exists()
    assert worker._exit_code([], inflight) == worker.IN_FLIGHT_EXIT_CODE != 0


def test_gate_partial_old_write_cleared_and_trained(tmp_path):
    run = _mk_run(tmp_path, manifest=False, adapter=False, ckpts=2,
                  age_s=worker.IN_FLIGHT_GRACE_S + 60)
    inflight: list[str] = []
    assert worker._gate_checkpointed(run, "c", inflight) is False  # train, force=False
    assert not run.exists() and inflight == []


def test_gate_manifest_without_complete_adapter_is_not_skipped(tmp_path):
    """checkpoint-* + orphan manifest but no final adapter = unfinished, not skippable."""
    run = _mk_run(tmp_path, manifest=True, adapter=False, ckpts=3,
                  age_s=worker.IN_FLIGHT_GRACE_S + 60)
    assert worker.checkpointed_run_state(run) == "partial"
    assert worker._gate_checkpointed(run, "c", []) is False


def test_gate_final_only_run_forces_retrain(tmp_path):
    run = _mk_run(tmp_path, manifest=True, adapter=True, ckpts=0, age_s=10)
    assert worker._gate_checkpointed(run, "c", []) is True
    assert run.exists()


def test_gate_absent_trains_without_force(tmp_path):
    assert worker._gate_checkpointed(tmp_path / "nope", "c", []) is False


def test_exit_code_failures_dominate():
    assert worker._exit_code(["train:x"], ["y"]) == 1


# --- --eval-namespace-animal: opt-in eval/<animal> layout for base controls (2026-09-18) ----

def test_eval_namespace_animal_default_off_keeps_leaf_root(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["w", "--animal", "cat", "--source", "base"])
    a = worker.parse_args()
    assert a.eval_namespace_animal is False
    assert worker.eval_artifact_subdir(a) is None


def test_eval_namespace_animal_derives_eval_animal_subdir(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["w", "--animal", "cat", "--source", "base", "--eval-namespace-animal"]
    )
    a = worker.parse_args()
    assert a.eval_namespace_animal is True
    assert worker.eval_artifact_subdir(a) == Path("eval") / "cat"


# --- --tf32 (2026-09-20): opt-in TF32 matmuls, off by default ---------------------------

def test_tf32_flag_defaults_off_and_parses(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["w", "--animal", "dog", "--config-tag", "t"])
    assert worker.parse_args().tf32 is False
    monkeypatch.setattr(
        sys, "argv", ["w", "--animal", "dog", "--config-tag", "t", "--tf32"]
    )
    assert worker.parse_args().tf32 is True


def test_tf32_flag_reaches_the_training_config():
    """The flag lands on the TrainConfig the cells are trained with (via _load_common's)."""
    st = _student_train()
    assert st.tf32 is False  # the YAML cell is fp32 and stays so
    got = worker.build_student_train(replace(st, tf32=True), 20, None, False)
    assert got.tf32 is True
