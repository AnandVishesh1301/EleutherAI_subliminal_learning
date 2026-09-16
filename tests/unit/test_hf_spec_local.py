"""Local-snapshot `HfModelSpec`: identity string and manifest-checked loading."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from subliminal_learning.models import loading_io
from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.neox_convert_io import (
    CONVERSION_MANIFEST,
    ConversionManifest,
)
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.utils.io import atomic_write_json

_COMMON = dict(
    family="pythia",
    cluster="neox",
    model_type="gpt_neox",
    vocab_size=50304,
    tokenizer_sha256="0" * 64,
    lora_target_modules=("query_key_value",),
)


def _manifest(base: str, step: int) -> ConversionManifest:
    return ConversionManifest(
        base=base,
        step=step,
        revision=f"step{step}",
        source_dir="/nowhere",
        source_sha256={},
        weights_sha256="",
        n_tensors=0,
        mapping_version="test",
        transformers_version="0",
        torch_version="0",
        created_at="now",
    )


def _local_spec(root: Path, base: str = "pile-ws1", step: int = 8000) -> HfModelSpec:
    return HfModelSpec(
        label=f"{base}-step{step}",
        repo_id=f"local/{base}",
        revision=f"step{step}",
        local_root=root,
        **_COMMON,
    )


def test_local_spec_model_ref_carries_base_and_revision(tmp_path: Path) -> None:
    spec = _local_spec(tmp_path)
    assert model_ref(spec, "EleutherAI") == "local/pile-ws1@step8000"
    assert "@" not in spec.label and "/" not in spec.label


def test_hub_roster_spec_identity_matches_pythia_spec() -> None:
    hub = HfModelSpec(
        label="pythia-160m-step143000",
        repo_id="EleutherAI/pythia-160m",
        revision="step143000",
        **_COMMON,
    )
    pythia = PythiaModelSpec(size="160m", revision="step143000")
    assert model_ref(hub, "EleutherAI") == model_ref(pythia, "EleutherAI")


def test_local_spec_requires_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs a revision"):
        HfModelSpec(
            label="pile-ws1",
            repo_id="local/pile-ws1",
            revision=None,
            local_root=tmp_path,
            **_COMMON,
        )


def test_local_snapshot_dir_checks_manifest_identity(tmp_path: Path) -> None:
    snapshot = tmp_path / "step8000"
    snapshot.mkdir()
    atomic_write_json(
        snapshot / CONVERSION_MANIFEST, asdict(_manifest("pile-ws1", 8000))
    )
    assert loading_io.local_snapshot_dir(_local_spec(tmp_path)) == snapshot
    with pytest.raises(ValueError, match="does not match"):
        loading_io.local_snapshot_dir(_local_spec(tmp_path, base="dclm-main"))
    atomic_write_json(
        snapshot / CONVERSION_MANIFEST, asdict(_manifest("pile-ws1", 9000))
    )
    with pytest.raises(ValueError, match="does not match"):
        loading_io.local_snapshot_dir(_local_spec(tmp_path))
    with pytest.raises(FileNotFoundError):
        loading_io.local_snapshot_dir(_local_spec(tmp_path, step=2000))
    with pytest.raises(ValueError, match="not a local spec"):
        loading_io.local_snapshot_dir(PythiaModelSpec(size="160m"))
