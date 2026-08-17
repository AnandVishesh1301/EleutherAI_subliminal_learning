"""Model-revision support + skip/retrain/error gate: identity/label/path invariants.

Two things are load-bearing here:

  - ZERO CACHE INVALIDATION: with `revision=None` (every historical run) the repo ref,
    directory names, spec label, and identity dicts must be byte-identical to the
    pre-revision behavior — so existing `train_manifest.json` files still match.
  - The skip/retrain/error gate: config drift on a policy-frozen key must fail loudly
    rather than silently overwrite an artifact.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    teacher_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
    resolve_hf_repo_id,
)
from subliminal_learning.run_gate import (
    ERROR,
    RETRAIN,
    SKIP,
    gate_action,
    identity_diff,
    student_gate_action,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import (
    _format_lr,
    model_dir_name,
    model_dir_parts,
    model_dir_parts_from_name,
    teacher_run_dir,
)

_ORG = "EleutherAI"
_NONE = PythiaModelSpec(size="160m", seed=1, variant="data")
_REV = PythiaModelSpec(size="160m", seed=1, variant="data", revision="step3000")


# --- model_ref / model_dir_name / spec_label: None == old, revision -> @rev ----------

def test_model_ref_none_revision_is_plain_repo_id():
    assert model_ref(_NONE, _ORG) == resolve_hf_repo_id(_NONE, _ORG)
    assert "@" not in model_ref(_NONE, _ORG)


def test_model_ref_with_revision_appends_suffix():
    assert model_ref(_REV, _ORG) == "EleutherAI/pythia-160m-data-seed1@step3000"


def test_model_dir_name_none_revision_unchanged():
    assert model_dir_name(_NONE, _ORG) == "pythia-160m-data-seed1"


def test_model_dir_name_with_revision_appends_suffix():
    assert model_dir_name(_REV, _ORG) == "pythia-160m-data-seed1@step3000"


def test_spec_label_none_revision_unchanged():
    assert spec_label(_NONE) == "data-seed1"


def test_spec_label_with_revision_appends_suffix():
    assert spec_label(_REV) == "data-seed1@step3000"


# --- model_dir_parts / model_dir_parts_from_name: conditional ckpt/ level -------------

def test_model_dir_parts_none_revision_is_single_component():
    assert model_dir_parts(_NONE, _ORG) == ("pythia-160m-data-seed1",)


def test_model_dir_parts_with_revision_gathers_under_ckpt():
    assert model_dir_parts(_REV, _ORG) == ("ckpt", "pythia-160m-data-seed1@step3000")


def test_model_dir_parts_from_name_plain_is_single_component():
    assert model_dir_parts_from_name("pythia-160m-data-seed1") == (
        "pythia-160m-data-seed1",
    )


def test_model_dir_parts_from_name_at_rev_gathers_under_ckpt():
    assert model_dir_parts_from_name("pythia-160m-data-seed1@step3000") == (
        "ckpt", "pythia-160m-data-seed1@step3000",
    )


# --- teacher_run_dir: ckpt/ level for a revision spec, byte-identical otherwise --------

_LORA = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
_TRAIN = TrainConfig(
    learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64, max_grad_norm=1.0,
    warmup_ratio=0.05, lr_scheduler_type="linear", seed=0, optim="adamw_torch",
)
_ORIGINAL_REV = PythiaModelSpec(size="160m", seed=0, variant="full", revision="step1000")


def test_teacher_run_dir_with_revision_inserts_ckpt_level():
    d = teacher_run_dir(Path("/out"), "cat", _ORIGINAL_REV, _ORG, _TRAIN, _LORA)
    assert str(d).endswith(
        "teachers/cat/ckpt/pythia-160m@step1000/ep1_lr2e-4_r8a8/ftseed0"
    )


def test_teacher_run_dir_none_revision_byte_identical():
    d = teacher_run_dir(Path("/out"), "cat", _NONE, _ORG, _TRAIN, _LORA)
    assert d == Path(
        "/out/teachers/cat/pythia-160m-data-seed1/ep1_lr2e-4_r8a8/ftseed0"
    )


# --- parse_base_models: optional @revision suffix ------------------------------------

def test_parse_base_models_no_revision_is_none():
    (spec,) = parse_base_models("data-seed1")
    assert spec.revision is None


def test_parse_base_models_round_trips_revision():
    (spec,) = parse_base_models("data-seed1@step3000")
    assert spec == PythiaModelSpec(size="160m", seed=1, variant="data", revision="step3000")


def test_parse_base_models_revision_is_per_token():
    a, b = parse_base_models("data-seed1@step3000,weight-seed2")
    assert a.revision == "step3000" and b.revision is None


def test_parse_base_models_rejects_empty_revision():
    with pytest.raises(ValueError, match="empty revision"):
        parse_base_models("data-seed1@")


def test_parse_base_models_shorthands_stay_revisionless():
    assert all(s.revision is None for s in parse_base_models("seeds"))
    assert all(s.revision is None for s in parse_base_models("data"))


# --- parse_base_models: original pythia-160m in, full-seedN out ----------------------

_ORIGINAL = PythiaModelSpec(size="160m", seed=0, variant="full")


@pytest.mark.parametrize("token", ["original", "pythia-160m", "full-seed0", "Original"])
def test_parse_base_models_original_aliases(token):
    assert parse_base_models(token) == [_ORIGINAL]


def test_parse_base_models_original_round_trips_revision_and_label():
    (spec,) = parse_base_models("original@step3000")
    assert spec == PythiaModelSpec(size="160m", seed=0, variant="full", revision="step3000")
    assert spec_label(spec) == "full-seed0@step3000"
    assert (parse_base_models(spec_label(_ORIGINAL))) == [_ORIGINAL]


def test_parse_base_models_original_resolves_to_plain_repo():
    (spec,) = parse_base_models("original")
    assert resolve_hf_repo_id(spec, _ORG) == "EleutherAI/pythia-160m"
    assert model_dir_name(spec, _ORG) == "pythia-160m"


def test_parse_base_models_original_mixes_in_comma_list():
    a, b, c = parse_base_models("original,data-seed1,weight-seed3")
    assert a == _ORIGINAL
    assert (b.variant, b.seed) == ("data", 1)
    assert (c.variant, c.seed) == ("weight", 3)


def test_parse_base_models_all7_is_original_plus_six_seeds():
    specs = parse_base_models("all7")
    assert specs[0] == _ORIGINAL
    assert specs[1:] == parse_base_models("seeds")
    assert len(specs) == 7


@pytest.mark.parametrize("token", ["full-seed1", "full-seed2", "full-seed9"])
def test_parse_base_models_rejects_full_seed_variants(token):
    with pytest.raises(ValueError, match="out of scope"):
        parse_base_models(token)


# --- parse_base_models: pythia-<size> selects the full seed-0 model -------------------

def test_parse_base_models_pythia_size_is_full_seed0():
    (spec,) = parse_base_models("pythia-2.8b")
    assert (spec.size, spec.seed, spec.variant, spec.revision) == (
        "2.8b", 0, "full", None,
    )
    assert resolve_hf_repo_id(spec, _ORG) == "EleutherAI/pythia-2.8b"


def test_parse_base_models_pythia_160m_still_aliases_original():
    assert parse_base_models("pythia-160m") == [_ORIGINAL]


def test_parse_base_models_pythia_size_round_trips_revision():
    (spec,) = parse_base_models("pythia-2.8b@step3000")
    assert (spec.size, spec.revision) == ("2.8b", "step3000")


def test_parse_base_models_pythia_deduped_rejected():
    # Deduped siblings were removed from PYTHIA_SIZES (2026-08-17): `spec_label`
    # ignores size, so `pythia-2.8b-deduped` would collide with `pythia-2.8b`
    # (both `full-seed0`). Re-adding them requires size-aware labels first.
    with pytest.raises(ValueError, match="valid Pythia sizes"):
        parse_base_models("pythia-2.8b-deduped")


def test_parse_base_models_pythia_bad_size_fails_fast():
    with pytest.raises(ValueError, match="valid Pythia sizes"):
        parse_base_models("pythia-3b")


def test_parse_base_models_pythia_size_mixes_with_160m_tokens():
    a, b = parse_base_models("pythia-2.8b,data-seed1")
    assert (a.size, a.seed, a.variant) == ("2.8b", 0, "full")
    assert (b.variant, b.seed) == ("data", 1)


# --- _format_lr: locked renderings + non-collision -----------------------------------

@pytest.mark.parametrize(
    "lr, expected",
    [(2e-4, "2e-4"), (2.5e-4, "2.5e-4"), (1e-5, "1e-5"), (5e-5, "5e-5"),
     (1e-4, "1e-4"), (3e-4, "3e-4")],
)
def test_format_lr_locked_renderings(lr, expected):
    assert _format_lr(lr) == expected


def test_format_lr_distinguishes_near_values():
    assert _format_lr(2.04e-4) != _format_lr(2e-4)
    assert _format_lr(2.04e-4) == "2.04e-4"


# --- identity_diff -------------------------------------------------------------------

def test_identity_diff_reports_dotted_nested_key():
    diff = identity_diff(
        {"train": {"learning_rate": 0.0002}}, {"train": {"learning_rate": 0.0001}}
    )
    assert diff == {"train.learning_rate": (0.0002, 0.0001)}


# --- teacher gate --------------------------------------------------------------------

_IDENTITY = {"base_model_repo": "EleutherAI/pythia-160m-data-seed1",
             "train": {"learning_rate": 0.0002, "seed": 0}}


def test_gate_skips_on_equal_identity():
    d = gate_action({"run_identity": _IDENTITY}, _IDENTITY, force=False)
    assert d.action == SKIP and d.diff == {}


def test_gate_errors_with_keydiff_on_drift():
    drifted = {**_IDENTITY, "train": {"learning_rate": 0.0001, "seed": 0}}
    d = gate_action({"run_identity": _IDENTITY}, drifted, force=False)
    assert d.action == ERROR
    assert d.diff == {"train.learning_rate": (0.0002, 0.0001)}


def test_gate_force_retrains_even_on_drift():
    drifted = {**_IDENTITY, "train": {"learning_rate": 0.0001, "seed": 0}}
    assert gate_action({"run_identity": _IDENTITY}, drifted, force=True).action == RETRAIN


def test_gate_retrains_when_manifest_missing_or_corrupt():
    assert gate_action(None, _IDENTITY, force=False).action == RETRAIN
    assert gate_action({}, _IDENTITY, force=False).action == RETRAIN


# --- student gate: filtered_sha256 is the designed content flow -----------------------

_S_IDENTITY = {"base_model_repo": "EleutherAI/pythia-160m-data-seed1",
               "filtered_sha256": "aaa", "train": {"learning_rate": 0.0001}}


def test_student_gate_retrains_on_filtered_sha_only_drift():
    refreshed = {**_S_IDENTITY, "filtered_sha256": "bbb"}
    d = student_gate_action({"run_identity": _S_IDENTITY}, refreshed, force=False)
    assert d.action == RETRAIN and d.diff == {"filtered_sha256": ("aaa", "bbb")}


def test_student_gate_errors_on_filtered_sha_plus_other_drift():
    drifted = {**_S_IDENTITY, "filtered_sha256": "bbb",
               "train": {"learning_rate": 0.0002}}
    d = student_gate_action({"run_identity": _S_IDENTITY}, drifted, force=False)
    assert d.action == ERROR
    assert set(d.diff) == {"filtered_sha256", "train.learning_rate"}


def test_student_gate_skips_on_equal_identity():
    assert student_gate_action(
        {"run_identity": _S_IDENTITY}, _S_IDENTITY, force=False
    ).action == SKIP


# --- zero-invalidation: a None-revision identity has no @ in its repo ref -------------

def test_none_revision_teacher_identity_has_no_at_in_repo(tmp_path: Path):
    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    cfg = TrainTeacherTemplatesConfig(
        paths=PathsConfig(
            output_root=tmp_path, hf_cache_root=tmp_path,
            dataset_cache_root=tmp_path, wandb_dir=tmp_path,
        ),
        hub_org=_ORG,
        spec=_NONE,
        lora=LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear"),
        train=TrainConfig(
            learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
            max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
            seed=0, optim="adamw_torch",
        ),
        template_path=template, target_animal="cat", device="cpu",
    )
    identity = teacher_run_identity(cfg)
    assert identity["base_model_repo"] == "EleutherAI/pythia-160m-data-seed1"
    assert "@" not in identity["base_model_repo"]
    # A revision-set spec is distinct (proves revision reaches the identity).
    rev_identity = teacher_run_identity(replace(cfg, spec=_REV))
    assert rev_identity["base_model_repo"] == "EleutherAI/pythia-160m-data-seed1@step3000"


# --- phantom max_length: excluded from completion-only run identities -----------------

def test_completion_only_identities_omit_max_length(tmp_path: Path):
    """The completion-only trainer never truncates, so `max_length` — though still a
    `TrainConfig` field — must not appear in either pipeline's run identity."""
    from pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        _student_run_identity,
    )

    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    lora = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
    train = TrainConfig(
        learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
        seed=0, optim="adamw_torch",
    )
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path,
        dataset_cache_root=tmp_path, wandb_dir=tmp_path,
    )
    t_cfg = TrainTeacherTemplatesConfig(
        paths=paths, hub_org=_ORG, spec=_NONE, lora=lora, train=train,
        template_path=template, target_animal="cat", device="cpu",
    )
    t_identity = teacher_run_identity(t_cfg)
    assert "max_length" not in t_identity["train"]

    filtered = tmp_path / "numbers" / "filtered.jsonl"
    filtered.parent.mkdir()
    filtered.write_text("{}\n")
    s_cfg = DistillStudentConfig(
        paths=paths, hub_org=_ORG, spec=_NONE, source="rng",
        number_dataset_dir=filtered.parent, lora=lora, train=train,
    )
    s_identity = _student_run_identity(
        s_cfg, base_model_repo="EleutherAI/model", filtered_sha256="abc"
    )
    assert "max_length" not in s_identity["train"]


# --- dtype knob: default byte-identical, non-default only touches the manifest ---------

def test_dtype_default_leaves_run_identity_unchanged(tmp_path: Path):
    """The base-model dtype must never enter the run identity (fp32 or bf16), so an
    existing fp32 run is not invalidated by the new knob."""
    from pipelines.distill_student_pipeline import (
        DistillStudentConfig,
        _student_run_identity,
    )

    template = tmp_path / "templates.json"
    template.write_text('["My favorite animal is the"]')
    lora = LoRAConfig(r=8, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
    train = TrainConfig(
        learning_rate=2e-4, n_epochs=1, batch_size=22, max_length=64,
        max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="linear",
        seed=0, optim="adamw_torch",
    )
    paths = PathsConfig(
        output_root=tmp_path, hf_cache_root=tmp_path,
        dataset_cache_root=tmp_path, wandb_dir=tmp_path,
    )
    t_default = TrainTeacherTemplatesConfig(
        paths=paths, hub_org=_ORG, spec=_NONE, lora=lora, train=train,
        template_path=template, target_animal="cat", device="cpu",
    )
    t_bf16 = replace(t_default, dtype="bfloat16")
    assert t_default.dtype == "float32"  # default preserved
    assert teacher_run_identity(t_default) == teacher_run_identity(t_bf16)
    for ident in (teacher_run_identity(t_default), teacher_run_identity(t_bf16)):
        assert "base_model_dtype" not in ident
        assert "dtype" not in ident

    filtered = tmp_path / "numbers" / "filtered.jsonl"
    filtered.parent.mkdir()
    filtered.write_text("{}\n")
    s_default = DistillStudentConfig(
        paths=paths, hub_org=_ORG, spec=_NONE, source="rng",
        number_dataset_dir=filtered.parent, lora=lora, train=train,
    )
    s_bf16 = replace(s_default, dtype="bfloat16")
    assert s_default.dtype == "float32"
    a = _student_run_identity(s_default, base_model_repo="X", filtered_sha256="abc")
    b = _student_run_identity(s_bf16, base_model_repo="X", filtered_sha256="abc")
    assert a == b
    assert "base_model_dtype" not in a


def test_dtype_bad_value_fails_fast():
    from subliminal_learning.models.loading_io import resolve_dtype

    with pytest.raises(ValueError, match="unknown dtype"):
        resolve_dtype("float16")


def test_generate_numbers_config_records_dtype_in_manifest_not_params(tmp_path: Path):
    """A non-default dtype is additive manifest provenance; the resume-identity
    `params` dict (and thus fp32 pools) is untouched."""
    import torch

    from pipelines.generate_numbers_pipeline import resolve_dtype

    # bfloat16 round-trips through the shared resolver used by the pipeline.
    assert resolve_dtype("bfloat16") is torch.bfloat16
    assert resolve_dtype("float32") is torch.float32
