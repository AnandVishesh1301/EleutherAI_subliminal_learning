"""Cross-base EAS pipeline pure-logic tests (CPU-only, no model loads).

Covers the parts of `pipelines.eas_pipeline` that decide same-base vs cross-base behavior
WITHOUT loading a model: the `_is_cross` / `_numbers_base_model` helpers, the `_build_meta`
key-set contract (same-base meta must stay byte-identical to what is already on disk; cross adds
exactly two keys), and the cross-base branch of `_consistency_problems`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.eas_pipeline import (
    EASConfig,
    _build_meta,
    _consistency_problems,
    _is_cross,
    _numbers_base_model,
    _prompt_ident,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec

# The exact same-base meta key set on disk today; cross ADDS these two.
_SAME_BASE_META_KEYS = {
    "position", "n_prompts", "prompt_sha256", "base_model_repo", "teacher_param_tag",
    "teacher_ftseed", "teacher_adapter_id", "number_dataset_dir", "filtered_sha256",
    "subsample_size", "subsample_seed", "splice_examples", "number_dataset_tag",
    "student_lora", "student_lr", "n_layers", "eas_mean_convention",
}
_CROSS_EXTRA_META_KEYS = {"numbers_base_model_repo", "cross_base"}

_ORIGINAL = PythiaModelSpec(size="160m", seed=0, variant="full")
_DATA_SEED1 = PythiaModelSpec(size="160m", seed=1, variant="data")

_REF_MANIFEST = {
    "number_dataset_dir": "/out/numbers/dog/pythia-160m/ep10_lr1e-4_r16a32/ftseed1/ex10-20/modes/prefix",
    "filtered_sha256": "abc123",
    "subsample_size": 10_000,
    "subsample_seed": 42,
    "splice_examples": False,
}


def _cfg(spec: PythiaModelSpec, numbers_spec: PythiaModelSpec | None) -> EASConfig:
    paths = PathsConfig(
        output_root=Path("/out"),
        hf_cache_root=Path("/cache"),
        dataset_cache_root=Path("/data"),
        wandb_dir=Path("/wandb"),
    )
    lora = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")
    train = TrainConfig(
        learning_rate=1e-4, n_epochs=10, batch_size=8, max_length=512,
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="cosine", seed=1,
    )
    return EASConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=spec,
        animal="dog",
        teacher_param_tag="ep10_lr1e-4_r16a32",
        teacher_ftseed=1,
        number_dataset_tag="ex10-20_prefix_n100k",
        lora=lora,
        train=train,
        numbers_spec=numbers_spec,
    )


# ------------------------------------------------------------------- cross detection


def test_is_cross_none_is_same_base():
    cfg = _cfg(_DATA_SEED1, numbers_spec=None)
    assert _is_cross(cfg) is False
    assert _numbers_base_model(cfg) is None


def test_is_cross_equal_spec_is_same_base():
    # numbers_spec set but == student base -> same-base (no nbase suffix).
    cfg = _cfg(_DATA_SEED1, numbers_spec=_DATA_SEED1)
    assert _is_cross(cfg) is False
    assert _numbers_base_model(cfg) is None


def test_is_cross_different_spec():
    cfg = _cfg(_DATA_SEED1, numbers_spec=_ORIGINAL)
    assert _is_cross(cfg) is True
    assert _numbers_base_model(cfg) == "pythia-160m"


# ------------------------------------------------------------------- meta key set


def _meta(cfg: EASConfig) -> dict:
    return _build_meta(
        cfg,
        n_prompts=1024,
        prompt_sha256="deadbeef",
        teacher_adapter_id="feedface",
        ref_manifest=_REF_MANIFEST,
        n_layers=12,
        cross=_is_cross(cfg),
    )


def test_same_base_meta_key_set_unchanged():
    # Regression: 40+ computed same-base lineages must keep cache-hitting -> the meta key set
    # must be EXACTLY today's (no cross keys, nothing else added).
    meta = _meta(_cfg(_ORIGINAL, numbers_spec=None))
    assert set(meta) == _SAME_BASE_META_KEYS
    assert "cross_base" not in meta
    assert "numbers_base_model_repo" not in meta
    # base_model_repo is the student base repo.
    assert meta["base_model_repo"] == "EleutherAI/pythia-160m"


def test_cross_meta_adds_exactly_two_keys():
    cfg = _cfg(_DATA_SEED1, numbers_spec=_ORIGINAL)
    meta = _meta(cfg)
    assert set(meta) == _SAME_BASE_META_KEYS | _CROSS_EXTRA_META_KEYS
    assert meta["cross_base"] is True
    # numbers_base_model_repo names the teacher/numbers base A; base_model_repo stays B.
    assert meta["numbers_base_model_repo"] == "EleutherAI/pythia-160m"
    assert meta["base_model_repo"] == "EleutherAI/pythia-160m-data-seed1"


# ------------------------------------------------------------------- consistency checks


def _teacher_manifest(*, cross: bool) -> dict:
    m = {
        "splice_examples": False,
        "teacher_param_tag": "ep10_lr1e-4_r16a32",
        "teacher_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
        "numbers_base_model_repo": "EleutherAI/pythia-160m",
        "cross_base": cross,
    }
    if not cross:
        # A same-base student's numbers base is its own base.
        m["numbers_base_model_repo"] = "EleutherAI/pythia-160m-data-seed1"
    return m


def test_consistency_cross_requires_matching_cross_student():
    cfg = _cfg(_DATA_SEED1, numbers_spec=_ORIGINAL)
    assert _consistency_problems(cfg, "teacher", _teacher_manifest(cross=True)) == []
    # A same-base (non-cross) student under a cross request is flagged.
    problems = _consistency_problems(cfg, "teacher", _teacher_manifest(cross=False))
    assert any("numbers_base_model_repo" in p for p in problems)


def test_consistency_same_base_rejects_cross_student():
    cfg = _cfg(_DATA_SEED1, numbers_spec=None)
    assert _consistency_problems(cfg, "teacher", _teacher_manifest(cross=False)) == []
    problems = _consistency_problems(cfg, "teacher", _teacher_manifest(cross=True))
    assert any("cross_base" in p for p in problems)


def test_consistency_prompt_provenance_teacher_only():
    # A teacher-source student whose training-prompt provenance diverges from the reference is
    # flagged (it would be scored on prompts it never trained on); base/rng controls are exempt.
    cfg = _cfg(_ORIGINAL, numbers_spec=None)
    ref = {
        "number_dataset_dir": "/out/numbers/dog/pythia-160m/tag/ftseed1/prefix",
        "filtered_sha256": "aaa",
        "subsample_size": 10_000,
        "subsample_seed": 42,
        "splice_examples": False,
    }
    ref_ident = _prompt_ident(ref)
    agreeing = {
        **ref, "teacher_param_tag": "ep10_lr1e-4_r16a32", "teacher_ftseed": 1,
        "number_dataset_tag": "ex10-20_prefix_n100k",
    }
    assert _consistency_problems(cfg, "teacher", agreeing, ref_ident=ref_ident) == []
    # Same lineage tags, but distilled off a different (refreshed) pool -> flagged.
    diverging = {**agreeing, "filtered_sha256": "bbb"}
    problems = _consistency_problems(cfg, "teacher", diverging, ref_ident=ref_ident)
    assert any("training-prompt provenance" in p for p in problems)
    # Controls train on their own pools and are deliberately read on the teacher prompts: exempt.
    assert _consistency_problems(cfg, "base", diverging, ref_ident=ref_ident) == []
    assert _consistency_problems(cfg, "rng", diverging, ref_ident=ref_ident) == []
    # Without a reference ident the check is a no-op (back-compat with the existing callers).
    assert _consistency_problems(cfg, "teacher", diverging) == []
