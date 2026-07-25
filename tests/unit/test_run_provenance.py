"""Unit tests for the shared run-provenance domain (`training.run_manifest`/`run_set`).

Pure dataclass/string logic: the manifest envelope must round-trip and stay authoritative
over the body, the LoRA blob must be byte-identical across FT modes, the run-set id /
subdir / campaign-conflict helpers must gate campaign identity exactly, and the SDF/GKD
`identity_gate_action` skip/error/retrain decision must match its documented policy.
"""

from __future__ import annotations

import pytest

from subliminal_learning.config import LoRAConfig
from subliminal_learning.run_gate import ERROR, RETRAIN, SKIP
from subliminal_learning.training.run_manifest import (
    SCHEMA_VERSION,
    RunManifest,
    identity_gate_action,
    lora_manifest_blob,
)
from subliminal_learning.training.run_set import (
    build_run_set_card,
    card_conflict,
    is_valid_run_set_id,
    run_set_subdir,
)


def test_manifest_roundtrip_preserves_body_and_envelope() -> None:
    body = {"base_model": "pythia-160m-data-seed1", "final_train_loss": 1.23, "ftseed": 0}
    m = RunManifest(
        workflow="sdf_teacher",
        body=body,
        run_set_id="20260724_crossinit_160m",
        git_sha="abc123",
        created_at="2026-07-24T00:00:00Z",
    )
    restored = RunManifest.from_dict(m.to_dict())
    assert restored == m
    assert restored.body == body
    assert restored.schema_version == SCHEMA_VERSION


def test_manifest_envelope_is_authoritative_over_body() -> None:
    # A stray body key colliding with an envelope key must not shadow the envelope.
    m = RunManifest(
        workflow="gkd_student",
        body={"git_sha": "STALE_FROM_BODY", "beta": 1.0},
        git_sha="AUTHORITATIVE",
    )
    d = m.to_dict()
    assert d["git_sha"] == "AUTHORITATIVE"
    assert d["workflow"] == "gkd_student"
    assert d["beta"] == 1.0


def test_lora_manifest_blob_full_ft_is_all_none() -> None:
    assert lora_manifest_blob(None) == {
        "lora_r": None,
        "lora_alpha": None,
        "lora_dropout": None,
        "target_modules": None,
    }


def test_lora_manifest_blob_encodes_adapter_fields() -> None:
    blob = lora_manifest_blob(
        LoRAConfig(r=64, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear")
    )
    assert blob == {
        "lora_r": 64,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
    }
    tuple_blob = lora_manifest_blob(
        LoRAConfig(
            r=8, lora_alpha=8, lora_dropout=0.0, target_modules=("query_key_value",)
        )
    )
    assert tuple_blob["target_modules"] == ["query_key_value"]


@pytest.mark.parametrize(
    ("run_set_id", "ok"),
    [
        ("20260724_crossinit_160m", True),
        ("20260801_beta-lambda_160m", True),
        ("20260724_crossinit_70m", True),
        ("2026724_bad_160m", False),  # 7-digit date
        ("20260724_CrossInit_160m", False),  # upper-case
        ("20260724_crossinit", False),  # no size
        ("crossinit_160m", False),  # no date
        ("20260724__160m", False),  # empty purpose
    ],
)
def test_is_valid_run_set_id(run_set_id: str, ok: bool) -> None:
    assert is_valid_run_set_id(run_set_id) is ok


def test_run_set_subdir_is_the_subdir_component() -> None:
    assert (
        run_set_subdir("sdf_teachers", "20260724_crossinit_160m")
        == "sdf_teachers/20260724_crossinit_160m"
    )
    assert (
        run_set_subdir("gkd_students", "20260724_crossinit_160m")
        == "gkd_students/20260724_crossinit_160m"
    )


def _card(git_sha: str, axes: dict) -> dict:
    return build_run_set_card(
        run_set_id="20260724_crossinit_160m",
        run_type="gkd_student",
        purpose="crossinit",
        date="2026-07-24",
        owner="vishesh",
        git_sha=git_sha,
        sweep_yaml="configs/experiment/gkd_student_sweep.yaml",
        resolved_config=axes,
        grid_axes=axes,
        model_size="160m",
    )


def test_card_conflict_none_when_identical() -> None:
    axes = {"beta": [1.0], "animals": ["cat"]}
    assert card_conflict(_card("sha1", axes), _card("sha1", axes)) is None


def test_card_conflict_flags_git_sha_and_config_drift() -> None:
    axes = {"beta": [1.0]}
    assert "git_sha" in card_conflict(_card("new", axes), _card("old", axes))
    other = {"beta": [0.0]}
    assert "resolved_config" in card_conflict(_card("sha", other), _card("sha", axes))


# --- identity_gate_action: SDF/GKD skip/error/retrain policy --------------------------

_IDENTITY = {
    "target_animal": "cat",
    "sdf_docs_sha256": "docs-v1",
    "hyperparameters": {"learning_rate": 2e-4},
}


def test_identity_gate_skips_on_equal_identity() -> None:
    d = identity_gate_action({"run_identity": _IDENTITY}, _IDENTITY, force=False)
    assert d.action == SKIP and d.diff == {}


def test_identity_gate_errors_with_keydiff_on_drift() -> None:
    drifted = {**_IDENTITY, "sdf_docs_sha256": "docs-v2"}
    d = identity_gate_action({"run_identity": _IDENTITY}, drifted, force=False)
    assert d.action == ERROR
    assert d.diff == {"sdf_docs_sha256": ("docs-v1", "docs-v2")}


def test_identity_gate_force_retrains_even_on_drift() -> None:
    drifted = {**_IDENTITY, "sdf_docs_sha256": "docs-v2"}
    d = identity_gate_action({"run_identity": _IDENTITY}, drifted, force=True)
    assert d.action == RETRAIN


def test_identity_gate_retrains_when_manifest_file_absent() -> None:
    assert identity_gate_action(None, _IDENTITY, force=False).action == RETRAIN


@pytest.mark.parametrize("stored_manifest", [{}, {"animal": "cat"}])
def test_identity_gate_skips_legacy_manifest_without_run_identity(
    stored_manifest: dict,
) -> None:
    # A manifest present on disk but written before this gate existed (empty dict, or
    # any dict lacking a "run_identity" key) is trusted as SKIP -- not retrained just to
    # satisfy a newly added key. Differs from `run_gate.gate_action`, which treats a
    # manifest with no usable identity as RETRAIN (there, that only happens for a
    # genuinely corrupt/unparseable file, never a byte-for-byte-valid legacy manifest).
    d = identity_gate_action(stored_manifest, _IDENTITY, force=False)
    assert d.action == SKIP and d.diff == {}


def test_identity_gate_ignores_drift_when_current_value_missing() -> None:
    # A large upstream artifact (e.g. a teacher checkpoint) reclaimed for disk quota:
    # the current recomputed hash is None, so this key is exempted from the comparison
    # rather than raising on every subsequent skip/resume check.
    stored = {**_IDENTITY, "teacher_manifest_sha256": "abc123"}
    current = {**_IDENTITY, "teacher_manifest_sha256": None}
    d = identity_gate_action(
        {"run_identity": stored},
        current,
        force=False,
        ignore_if_current_missing=("teacher_manifest_sha256",),
    )
    assert d.action == SKIP and d.diff == {}


def test_identity_gate_still_errors_on_other_drift_when_hash_missing() -> None:
    # The exemption only drops the named key; any other drift still errors.
    stored = {**_IDENTITY, "teacher_manifest_sha256": "abc123"}
    current = {
        **_IDENTITY,
        "teacher_manifest_sha256": None,
        "sdf_docs_sha256": "docs-v2",
    }
    d = identity_gate_action(
        {"run_identity": stored},
        current,
        force=False,
        ignore_if_current_missing=("teacher_manifest_sha256",),
    )
    assert d.action == ERROR
    assert d.diff == {"sdf_docs_sha256": ("docs-v1", "docs-v2")}


def test_identity_gate_errors_when_hash_present_and_changed() -> None:
    # A genuine content change (current hash present but different) still errors even
    # though the key is listed in ignore_if_current_missing.
    stored = {**_IDENTITY, "teacher_manifest_sha256": "abc123"}
    current = {**_IDENTITY, "teacher_manifest_sha256": "def456"}
    d = identity_gate_action(
        {"run_identity": stored},
        current,
        force=False,
        ignore_if_current_missing=("teacher_manifest_sha256",),
    )
    assert d.action == ERROR
    assert d.diff == {"teacher_manifest_sha256": ("abc123", "def456")}
