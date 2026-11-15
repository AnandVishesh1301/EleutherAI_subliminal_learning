"""Behavioral regressions on `pipelines.evaluate_teacher_preference_pipeline`: the summary gate, per-animal control artifacts, sharded-adapter identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from subliminal_learning.pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    adapter_identity,
    evaluation_artifact_path,
    prompts_hash,
)
from scripts.sweeps.sweep_evaluate_teacher_preference import _summary_covers
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec


SPEC = PythiaModelSpec(size="160m", seed=1, variant="data", revision="main")


def paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path / "out", hf_cache_root=tmp_path / "hf",
        dataset_cache_root=tmp_path / "data", wandb_dir=tmp_path / "wandb",
    )


def test_eval_summary_gate_invalidates_changed_adapter(tmp_path: Path):
    param_dir = tmp_path / "param"
    ckpt = param_dir / "ftseed0" / "checkpoints"
    ckpt.mkdir(parents=True)
    weights = ckpt / "adapter_model.safetensors"
    weights.write_bytes(b"first adapter")
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text('["My favorite animal is the"]')
    ps = PromptSet("favorite", prompt_file, 12, "prob")
    phash = prompts_hash(["My favorite animal is the"])
    summary = param_dir / "preference_summary.json"
    summary.write_text(json.dumps({
        "confidence": 0.95,
        "ft_seeds": [0],
        "adapter_ids": {"0": adapter_identity(ckpt)},
        "set_params": {"favorite": {"scoring": "prob", "prompts_hash": phash}},
        "summaries": [{"prompt_set": "favorite", "base_p_target": 0.1}],
    }))
    args = (summary, param_dir, (ps,), {"favorite": phash}, 100, 1.0, True, 0.95, ())
    assert _summary_covers(*args)
    weights.write_bytes(b"second adapter")
    assert not _summary_covers(*args)


def test_control_artifacts_can_be_namespaced_by_animal(tmp_path: Path):
    cfg = EvaluateTeacherPreferenceConfig(
        paths=paths(tmp_path), hub_org="EleutherAI", spec=SPEC,
        target_animal="cat", param_dir=tmp_path / "controls" / "tag",
        prompt_sets=(), forms=None, n_samples=100, temperature=1.0,
        include_base_control=True, confidence=0.95, device="cpu",
        artifact_subdir=Path("eval") / "cat",
    )
    assert evaluation_artifact_path(
        cfg.param_dir, cfg, "preference_summary.json"
    ) == cfg.param_dir / "eval" / "cat" / "preference_summary.json"


def test_adapter_identity_hashes_sharded_checkpoints(tmp_path: Path):
    """Re-audit residual: sharded full-FT checkpoints returned None, which made both the
    per-seed cache and the sweep DONE gate ({seed: None} == {seed: None}) blind to an
    in-place retrain. The identity now hashes every shard the index references."""
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a.w": "model-00001-of-00002.safetensors",
                       "b.w": "model-00002-of-00002.safetensors"},
    }))
    (ck / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    (ck / "model-00002-of-00002.safetensors").write_bytes(b"shard-two")
    aid = adapter_identity(ck)
    assert aid is not None
    assert adapter_identity(ck) == aid  # deterministic
    (ck / "model-00002-of-00002.safetensors").write_bytes(b"shard-two-retrained")
    assert adapter_identity(ck) != aid  # an in-place retrain is visible
    (ck / "model-00002-of-00002.safetensors").unlink()
    assert adapter_identity(ck) is None  # incomplete save: no identity
