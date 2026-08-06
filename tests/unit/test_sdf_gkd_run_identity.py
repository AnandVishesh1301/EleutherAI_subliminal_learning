"""Unit tests for the SDF-teacher / GKD-student `run_identity` builders.

Path tags (`sdf_param_tag`/`teacher_tag`/`gkd_tag`) disambiguate the swept HP axes; these
builders close the gap on the inputs a path tag cannot encode (the SDF document corpus,
the frozen teacher checkpoint, the prompt pool) so `identity_gate_action` (tested in
`test_run_provenance.py`) can catch drift under an unchanged path.
"""

from __future__ import annotations

from pathlib import Path

from pipelines.distill_student_gkd_pipeline import (
    TrainGkdStudentConfig,
    gkd_run_identity,
)
from pipelines.train_sdf_teacher_pipeline import (
    TrainSdfTeacherConfig,
    sdf_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.gkd.config import GkdConfig
from subliminal_learning.models.registry import PythiaModelSpec

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
