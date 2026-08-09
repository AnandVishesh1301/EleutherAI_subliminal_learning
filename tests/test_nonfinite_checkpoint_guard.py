"""The diverged-checkpoint gate: detect NaN weights BEFORE anything loads them.

A run at extreme alpha/rank can be saved all-NaN. Such a checkpoint is *complete* (a
weight file exists), so `checkpoint_is_complete` admits it, and loading it is what does
the damage: all-NaN logits reach `torch.multinomial`, whose device-side assert poisons the
CUDA context and kills co-tenant jobs on the same GPU. Every consumer must therefore skip
it up front -- and loudly, so a missing cell is never mistaken for a clean sweep.
"""

from __future__ import annotations

import numpy as np
import pytest
from safetensors.numpy import save_file

from subliminal_learning.utils.run_io import (
    checkpoint_is_complete,
    checkpoint_is_finite,
    checkpoint_nonfinite_count,
)


def _mk_ckpt(d, values):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text('{"r": 1, "lora_alpha": 32}')
    save_file({"base_model.model.layer.lora_A.weight": np.asarray(values, dtype=np.float32)},
              str(d / "adapter_model.safetensors"))
    return d


def test_clean_checkpoint_passes(tmp_path):
    d = _mk_ckpt(tmp_path / "ck", [[1.0, 2.0, 3.0]])
    assert checkpoint_nonfinite_count(d) == 0
    assert checkpoint_is_finite(d)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_diverged_checkpoint_is_counted(tmp_path, bad):
    d = _mk_ckpt(tmp_path / "ck", [[1.0, bad, 3.0]])
    assert checkpoint_nonfinite_count(d) == 1
    assert not checkpoint_is_finite(d)


def test_all_nan_checkpoint(tmp_path):
    """The real-world shape: total corruption, not a stray value."""
    d = _mk_ckpt(tmp_path / "ck", [[np.nan] * 8])
    assert checkpoint_nonfinite_count(d) == 8


def test_complete_but_not_finite_is_the_dangerous_case(tmp_path):
    """The gap this guard closes: `checkpoint_is_complete` alone admits a NaN adapter."""
    d = _mk_ckpt(tmp_path / "ck", [[np.nan, np.nan]])
    assert checkpoint_is_complete(d), "a NaN adapter still looks 'complete'"
    assert not checkpoint_is_finite(d), "...which is exactly why finiteness is checked too"


def test_missing_or_unreadable_checkpoint_reports_zero(tmp_path):
    """Absent/unknown must not masquerade as 'diverged' -- other gates own those cases."""
    assert checkpoint_nonfinite_count(tmp_path / "nope") == 0
    d = tmp_path / "bin_only"
    d.mkdir()
    (d / "adapter_config.json").write_text("{}")
    (d / "adapter_model.bin").write_bytes(b"\x00")
    assert checkpoint_nonfinite_count(d) == 0  # .bin is not inspected


def test_eval_discovery_skips_diverged_seed(tmp_path):
    """End-to-end at the seam that matters: eval must not hand a NaN adapter to the GPU."""
    from pipelines.evaluate_teacher_preference_pipeline import _discover_ftseed_dirs

    param = tmp_path / "tag"
    _mk_ckpt(param / "ftseed1" / "checkpoints", [[0.5, 0.25]])
    _mk_ckpt(param / "ftseed2" / "checkpoints", [[np.nan, np.nan]])
    incomplete: list[int] = []
    nonfinite: list[int] = []
    found = _discover_ftseed_dirs(param, "ftseed", incomplete, nonfinite)
    assert [s for s, _ in found] == [1], "the clean sibling must still evaluate"
    assert nonfinite == [2]
    assert incomplete == [], "a NaN adapter is diverged, not incomplete"


def test_all_seeds_diverged_raises_a_distinct_error(tmp_path):
    """A wholly-diverged cell must be distinguishable from a cell with no checkpoints.

    Both leave nothing to evaluate, but they are different outcomes: divergence is a
    RESULT, absence is an artifact gap. Collapsing them (or recording p_target=0) would
    make a blown-up run indistinguishable from a model that simply never prefers the
    target.
    """
    from pipelines.evaluate_teacher_preference_pipeline import (
        DivergedCheckpointError,
        _discover_ftseed_dirs,
    )

    param = tmp_path / "tag"
    _mk_ckpt(param / "ftseed1" / "checkpoints", [[np.nan]])
    _mk_ckpt(param / "ftseed2" / "checkpoints", [[np.nan]])
    inc: list[int] = []
    nf: list[int] = []
    assert _discover_ftseed_dirs(param, "ftseed", inc, nf) == []
    assert nf == [1, 2] and inc == []
    # DivergedCheckpointError is NOT a FileNotFoundError -- callers can branch on it
    assert not issubclass(DivergedCheckpointError, FileNotFoundError)


def test_diverged_seed_does_not_become_a_zero(tmp_path):
    """The clean sibling is scored on its own; the diverged seed is absent, not zero."""
    from pipelines.evaluate_teacher_preference_pipeline import _discover_ftseed_dirs

    param = tmp_path / "tag"
    _mk_ckpt(param / "ftseed1" / "checkpoints", [[0.5]])
    _mk_ckpt(param / "ftseed2" / "checkpoints", [[np.nan]])
    nf: list[int] = []
    found = _discover_ftseed_dirs(param, "ftseed", [], nf)
    evaluated = [s for s, _ in found]
    assert evaluated == [1]
    assert 2 not in evaluated, "a diverged seed must never contribute a value at all"
    assert nf == [2], "and must be recorded so the missing seed is visible downstream"


def _cfg(param_dir, tmp_path):
    """Minimal config for the marker path (no model is ever loaded on this branch)."""
    from subliminal_learning.config import PathsConfig
    from subliminal_learning.models.registry import PythiaModelSpec
    from pipelines.evaluate_teacher_preference_pipeline import (
        EvaluateTeacherPreferenceConfig, PromptSet,
    )
    return EvaluateTeacherPreferenceConfig(
        paths=PathsConfig(output_root=tmp_path, hf_cache_root=tmp_path,
                          dataset_cache_root=tmp_path, wandb_dir=tmp_path),
        hub_org="EleutherAI", spec=PythiaModelSpec(size="160m"), target_animal="dog",
        param_dir=param_dir,
        prompt_sets=(PromptSet("test", tmp_path / "p.json", 4, "prob"),),
        forms=("dog",), n_samples=2, temperature=1.0, include_base_control=False,
        confidence=0.95, device="cpu",
    )


def test_diverged_marker_is_visible_and_carries_no_score(tmp_path):
    """A wholly-diverged cell must appear in the file readers glob -- with no number."""
    import json
    from pipelines.evaluate_teacher_preference_pipeline import _write_diverged_marker

    param = tmp_path / "tag"
    param.mkdir()
    _write_diverged_marker(_cfg(param, tmp_path), [1, 2])

    doc = json.loads((param / "preference_summary.json").read_text())
    assert doc["diverged"] is True
    assert doc["skipped_nonfinite_seeds"] == [1, 2]
    assert doc["ft_seeds"] == [] and doc["summaries"] == []
    # the whole point: nothing here can be read as p_target = 0
    assert "p_target_mean" not in json.dumps(doc)
    # and no set_params, so a content-based DONE gate re-runs rather than skipping
    assert not doc.get("set_params")


def test_marker_does_not_block_a_later_seed(tmp_path):
    """Seeds 1-2 diverged; training/evaluating seed 3 later must still proceed."""
    from pipelines.evaluate_teacher_preference_pipeline import (
        _write_diverged_marker, _discover_ftseed_dirs,
    )

    param = tmp_path / "tag"
    _mk_ckpt(param / "ftseed1" / "checkpoints", [[np.nan]])
    _mk_ckpt(param / "ftseed2" / "checkpoints", [[np.nan]])
    _write_diverged_marker(_cfg(param, tmp_path), [1, 2])

    # a later run trains seed 3 cleanly
    _mk_ckpt(param / "ftseed3" / "checkpoints", [[0.5]])
    nf: list[int] = []
    found = _discover_ftseed_dirs(param, "ftseed", [], nf)
    assert [s for s, _ in found] == [3], "the marker must not suppress a new seed"
    assert nf == [1, 2]


def test_stale_diverged_flag_clears_on_a_successful_write(tmp_path):
    """Once any seed evaluates, the cell is no longer diverged -- the flag must not stick."""
    import json
    from pipelines.evaluate_teacher_preference_pipeline import (
        _merge_summary, _write_diverged_marker,
    )

    param = tmp_path / "tag"
    param.mkdir()
    _write_diverged_marker(_cfg(param, tmp_path), [1, 2])
    stored = json.loads((param / "preference_summary.json").read_text())
    assert stored["diverged"] is True

    # a later successful run builds its own report, which does not set `diverged`
    fresh = {"target_animal": "dog", "ft_seeds": [3], "summaries": [],
             "skipped_nonfinite_seeds": [1, 2]}
    merged = _merge_summary(stored, fresh, {"test"})
    assert "diverged" not in merged, "stale flag must drop out of the merged report"
    assert merged["ft_seeds"] == [3]
