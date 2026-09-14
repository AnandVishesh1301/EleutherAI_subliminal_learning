"""Unit tests for the SDF-teacher / GKD-student `run_identity` builders.

Path tags (`sdf_param_tag`/`teacher_tag`/`gkd_tag`) disambiguate the swept HP axes; these
builders close the gap on the inputs a path tag cannot encode (the SDF document corpus,
the frozen teacher checkpoint, the prompt pool) so `identity_gate_action` (tested in
`test_run_provenance.py`) can catch drift under an unchanged path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pipelines.distill_student_gkd_pipeline import (
    TrainGkdStudentConfig,
    _data_identity,
    _loss_name,
    _training_pool_rows,
    gkd_run_identity,
)
from pipelines.train_sdf_teacher_pipeline import (
    TrainSdfTeacherConfig,
    sdf_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.gkd.config import GkdConfig
from subliminal_learning.gkd.prompt_pool import subset_doc_ids_sha256
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.run_gate import ERROR, SKIP
from subliminal_learning.training.run_manifest import identity_gate_action

_ORG = "EleutherAI"
_SPEC = PythiaModelSpec(size="160m", seed=1, variant="data")
_LORA = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
_TRAIN = TrainConfig(
    learning_rate=2e-4, n_epochs=3, batch_size=8, max_length=512,
    max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear", seed=0,
)


def _paths(tmp_path: Path) -> PathsConfig:
    return PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path,
        dataset_cache_root=tmp_path, wandb_dir=tmp_path,
    )


# --- sdf_run_identity ------------------------------------------------------------------

def _sdf_cfg(tmp_path: Path, lora: LoRAConfig | None) -> TrainSdfTeacherConfig:
    return TrainSdfTeacherConfig(
        paths=_paths(tmp_path), hub_org=_ORG, spec=_SPEC, lora=lora, train=_TRAIN,
        target_animal="cat", sdf_docs_root=tmp_path,
    )


def test_sdf_run_identity_distinguishes_lora_and_full_ft(tmp_path: Path) -> None:
    lora_id = sdf_run_identity(_sdf_cfg(tmp_path, _LORA), {}, "docs-v1")
    full_id = sdf_run_identity(_sdf_cfg(tmp_path, None), {}, "docs-v1")
    assert lora_id["finetune_mode"] == "lora"
    assert full_id["finetune_mode"] == "full"
    assert lora_id != full_id


def test_sdf_run_identity_carries_docs_hash_and_hyperparameters(tmp_path: Path) -> None:
    hp = {"learning_rate": 2e-4, "n_epochs": 3}
    identity = sdf_run_identity(_sdf_cfg(tmp_path, _LORA), hp, "docs-v1")
    assert identity["sdf_docs_sha256"] == "docs-v1"
    assert identity["hyperparameters"] == hp


def test_sdf_run_identity_changes_with_docs_hash(tmp_path: Path) -> None:
    cfg = _sdf_cfg(tmp_path, _LORA)
    v1 = sdf_run_identity(cfg, {}, "docs-v1")
    v2 = sdf_run_identity(cfg, {}, "docs-v2")
    assert v1 != v2


# --- gkd_run_identity ------------------------------------------------------------------

_GKD = GkdConfig(
    lmbda=1.0, beta=0.5, temperature=1.0, teacher_temp=1.0, student_temp=1.0,
    max_new_tokens=64, num_epochs=3.0, max_steps=-1, student_mode="lora",
    vocab_mask=False, probe_every=50, rollout_log_every=1,
    prompt_pool_dir="gkd_prompts/pile_prompts_v1", candidates=("cat", "dog"),
)


def _gkd_cfg(tmp_path: Path, control: bool = False) -> TrainGkdStudentConfig:
    return TrainGkdStudentConfig(
        paths=_paths(tmp_path), hub_org=_ORG, spec=_SPEC, lora=_LORA, train=_TRAIN,
        gkd=_GKD, target_animal="cat",
        teacher_checkpoint_dir=None if control else tmp_path / "teacher" / "checkpoints",
        teacher_workflow="sdf", teacher_param_tag="ep3_lr1e-5_r8a8", teacher_ftseed=0,
        control=control,
    )


def test_gkd_run_identity_control_omits_teacher_hash(tmp_path: Path) -> None:
    identity = gkd_run_identity(
        _gkd_cfg(tmp_path, control=True), "", "gtag",
        teacher_manifest_sha256="should-be-dropped",
        prompt_pool_manifest_sha256="pool-v1",
    )
    assert identity["control"] is True
    assert identity["teacher_manifest_sha256"] is None


def test_gkd_run_identity_treatment_carries_teacher_hash(tmp_path: Path) -> None:
    identity = gkd_run_identity(
        _gkd_cfg(tmp_path), "ttag", "gtag",
        teacher_manifest_sha256="teacher-v1",
        prompt_pool_manifest_sha256="pool-v1",
    )
    assert identity["teacher_tag"] == "ttag"
    assert identity["gkd_tag"] == "gtag"
    assert identity["teacher_manifest_sha256"] == "teacher-v1"
    assert identity["prompt_pool_manifest_sha256"] == "pool-v1"


def test_gkd_run_identity_changes_with_prompt_pool_hash(tmp_path: Path) -> None:
    cfg = _gkd_cfg(tmp_path)
    v1 = gkd_run_identity(cfg, "ttag", "gtag", "teacher-v1", "pool-v1")
    v2 = gkd_run_identity(cfg, "ttag", "gtag", "teacher-v1", "pool-v2")
    assert v1 != v2


# --- checkpoint-ladder provenance (plan §6.8) ------------------------------------------
# Both builders used to record `resolve_hf_repo_id`, which DROPS `@stepN`: two rungs of one
# repo were identity-identical, so a run gated on identity could reuse another rung's
# checkpoint. They now record `model_ref`, which is the bare repo id at `revision is None`.

_BARE_REPO = resolve_hf_repo_id(_SPEC, _ORG)
_SPEC_AT_STEP = replace(_SPEC, revision="step16000")


def test_sdf_run_identity_base_model_repo_carries_revision(tmp_path: Path) -> None:
    cfg = replace(_sdf_cfg(tmp_path, _LORA), spec=_SPEC_AT_STEP)
    identity = sdf_run_identity(cfg, {}, "docs-v1")
    assert identity["base_model_repo"] == f"{_BARE_REPO}@step16000"


def test_sdf_run_identity_rungs_of_one_repo_are_distinct(tmp_path: Path) -> None:
    base = _sdf_cfg(tmp_path, _LORA)
    early = sdf_run_identity(replace(base, spec=_SPEC_AT_STEP), {}, "docs-v1")
    late = sdf_run_identity(
        replace(base, spec=replace(_SPEC, revision="step143000")), {}, "docs-v1"
    )
    assert early != late


def test_sdf_run_identity_unchanged_without_revision(tmp_path: Path) -> None:
    identity = sdf_run_identity(_sdf_cfg(tmp_path, _LORA), {}, "docs-v1")
    assert identity["base_model_repo"] == _BARE_REPO


def test_gkd_run_identity_base_model_repo_carries_revision(tmp_path: Path) -> None:
    cfg = replace(_gkd_cfg(tmp_path), spec=_SPEC_AT_STEP)
    identity = gkd_run_identity(cfg, "ttag", "gtag", "teacher-v1", "pool-v1")
    assert identity["base_model_repo"] == f"{_BARE_REPO}@step16000"


def test_gkd_run_identity_unchanged_without_revision(tmp_path: Path) -> None:
    identity = gkd_run_identity(
        _gkd_cfg(tmp_path), "ttag", "gtag", "teacher-v1", "pool-v1"
    )
    assert identity["base_model_repo"] == _BARE_REPO


# --- conditional opdlim keys (prompt subset, form mask, CE, off-policy rows) -----------
# Literal identity written by the pre-opdlim pipeline for `_gkd_cfg`: every existing
# manifest must keep SKIPping, so the default key sets may never grow.

_LEGACY_DEFAULT_IDENTITY = {
    "target_animal": "cat",
    "base_model_repo": "EleutherAI/pythia-160m-data-seed1",
    "control": False,
    "teacher_tag": "ttag",
    "gkd_tag": "gtag",
    "teacher_manifest_sha256": "teacher-v1",
    "prompt_pool_manifest_sha256": "pool-v1",
    "finetune_mode": "lora",
    "gkd_hyperparameters": {
        "lmbda": 1.0, "beta": 0.5, "temperature": 1.0, "teacher_temp": 1.0,
        "student_temp": 1.0, "max_new_tokens": 64, "num_epochs": 3.0, "max_steps": -1,
        "vocab_mask": False, "candidates": ["cat", "dog"],
    },
    "train_hyperparameters": {
        "learning_rate": 0.0002, "batch_size": 8, "max_length": 512,
        "warmup_ratio": 0.05, "max_grad_norm": 1.0, "lr_scheduler_type": "linear",
        "optim": "adamw_torch", "lora_r": 8, "lora_alpha": 8, "lora_dropout": 0.0,
        "target_modules": "all-linear",
    },
}


def _identity(cfg: TrainGkdStudentConfig, data_identity=None) -> dict:
    return gkd_run_identity(
        cfg, "ttag", "gtag", "teacher-v1", "pool-v1", data_identity=data_identity
    )


def test_gkd_run_identity_default_is_legacy_literal(tmp_path: Path) -> None:
    cfg = _gkd_cfg(tmp_path)
    identity = _identity(cfg, _data_identity(cfg, None, None))
    assert json.dumps(identity) == json.dumps(_LEGACY_DEFAULT_IDENTITY)


def test_gkd_run_identity_legacy_manifest_skips(tmp_path: Path) -> None:
    stored = json.loads(json.dumps({"run_identity": _LEGACY_DEFAULT_IDENTITY}))
    decision = identity_gate_action(
        stored, _identity(_gkd_cfg(tmp_path), {}), force=False
    )
    assert decision.action == SKIP


def _opdlim_cfg(tmp_path: Path) -> TrainGkdStudentConfig:
    gkd = replace(
        _GKD, lmbda=0.0, beta=0.0, top_k=1, top_k_ce=True, vocab_mask=True,
        vocab_mask_forms=True, prompt_subset_size=16, prompt_subset_seed=2,
    )
    return replace(_gkd_cfg(tmp_path), gkd=gkd)


def test_gkd_run_identity_new_keys_only_when_set(tmp_path: Path) -> None:
    cfg = _opdlim_cfg(tmp_path)
    rows = [{"doc_id": f"d{i}"} for i in range(16)]
    data = _data_identity(cfg, rows, [30, 10, 20])
    assert data == {
        "prompt_subset": {
            "size": 16, "seed": 2, "doc_ids_sha256": subset_doc_ids_sha256(rows),
            "drop_last": True,
        },
        "vocab_mask_ids_sha256": hashlib.sha256(b"[10, 20, 30]").hexdigest(),
    }
    identity = _identity(cfg, data)
    assert set(identity) == set(_LEGACY_DEFAULT_IDENTITY) | set(data)
    hp = identity["gkd_hyperparameters"]
    assert hp["top_k_ce"] is True
    assert hp["vocab_mask_forms"] is True
    assert hp["offpolicy_rows"] == "pretokenized_screened"
    assert set(hp) == set(_LEGACY_DEFAULT_IDENTITY["gkd_hyperparameters"]) | {
        "top_k", "top_k_residual", "top_k_ce", "vocab_mask_forms", "offpolicy_rows",
    }


def test_gkd_run_identity_stored_without_new_keys_errors(tmp_path: Path) -> None:
    cfg = replace(_gkd_cfg(tmp_path), gkd=replace(_GKD, lmbda=0.5))
    stored = {"run_identity": _LEGACY_DEFAULT_IDENTITY | {"gkd_hyperparameters": {
        **_LEGACY_DEFAULT_IDENTITY["gkd_hyperparameters"], "lmbda": 0.5,
    }}}
    decision = identity_gate_action(stored, _identity(cfg, {}), force=False)
    assert decision.action == ERROR
    subset = replace(_gkd_cfg(tmp_path), gkd=replace(_GKD, prompt_subset_size=16))
    data = _data_identity(subset, [{"doc_id": "d0"}], None)
    decision = identity_gate_action(
        {"run_identity": _LEGACY_DEFAULT_IDENTITY}, _identity(subset, data), force=False
    )
    assert decision.action == ERROR


def test_loss_name_top_k_ce(tmp_path: Path) -> None:
    assert _loss_name(_opdlim_cfg(tmp_path)) == "forward_kl_topk_ce"


def _write_pool(pool_dir: Path, n: int) -> None:
    pool_dir.mkdir(parents=True)
    rows = [
        {"doc_id": f"d{i}", "register": "A" if i % 2 else "B", "prompt": "p",
         "completion": "c"}
        for i in range(n)
    ]
    (pool_dir / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (pool_dir / "manifest.json").write_text(
        json.dumps({"register_weights": {"A": 0.5, "B": 0.5}, "completion_tokens": 128})
    )


def test_training_pool_rows_subset_uses_manifest_weights(tmp_path: Path) -> None:
    _write_pool(tmp_path / "pool", 40)
    cfg = replace(
        _gkd_cfg(tmp_path), gkd=replace(_GKD, prompt_subset_size=16, prompt_subset_seed=1)
    )
    rows = _training_pool_rows(cfg, tmp_path / "pool")
    assert len(rows) == 16
    assert sum(r["register"] == "A" for r in rows) == 8
    assert len(_training_pool_rows(_gkd_cfg(tmp_path), tmp_path / "pool")) == 40


def test_training_pool_rows_subset_smaller_than_batch_raises(tmp_path: Path) -> None:
    _write_pool(tmp_path / "pool", 40)
    cfg = replace(_gkd_cfg(tmp_path), gkd=replace(_GKD, prompt_subset_size=7))
    with pytest.raises(ValueError, match="prompt_subset_size=7"):
        _training_pool_rows(cfg, tmp_path / "pool")


def test_gkd_run_identity_strict_adds_mode_and_vocab_hash(tmp_path: Path) -> None:
    cfg = replace(_gkd_cfg(tmp_path), gkd=replace(_GKD, rollout_screen="strict"))
    data = _data_identity(cfg, None, None, screen_vocab_sha256="abc123")
    assert data == {"rollout_screen_vocab_sha256": "abc123"}
    identity = _identity(cfg, data)
    assert identity["gkd_hyperparameters"]["rollout_screen"] == "strict"
    assert set(identity) == set(_LEGACY_DEFAULT_IDENTITY) | {"rollout_screen_vocab_sha256"}
    decision = identity_gate_action(
        {"run_identity": _LEGACY_DEFAULT_IDENTITY}, identity, force=False
    )
    assert decision.action == ERROR


def test_gkd_run_identity_legacy_screen_emits_no_screen_keys(tmp_path: Path) -> None:
    cfg = _gkd_cfg(tmp_path)
    assert _data_identity(cfg, None, None, screen_vocab_sha256="ignored") == {}
    assert "rollout_screen" not in _identity(cfg, {})["gkd_hyperparameters"]


def test_data_identity_strict_requires_vocab_hash(tmp_path: Path) -> None:
    cfg = replace(_gkd_cfg(tmp_path), gkd=replace(_GKD, rollout_screen="strict"))
    with pytest.raises(ValueError, match="screen_vocab_sha256"):
        _data_identity(cfg, None, None)
