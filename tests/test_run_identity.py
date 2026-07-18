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
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import _format_lr, model_dir_name

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
