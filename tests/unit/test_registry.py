from __future__ import annotations

import pytest

from subliminal_learning.models.registry import (
    PythiaModelSpec,
    local_model_dir,
    resolve_hf_repo_id,
)


def test_resolve_hf_repo_id_full_variant_seed0_no_suffix() -> None:
    spec = PythiaModelSpec(size="410m", seed=0)
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-410m"


def test_resolve_hf_repo_id_full_variant_seed3_has_suffix() -> None:
    spec = PythiaModelSpec(size="410m", seed=3)
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-410m-seed3"


def test_resolve_hf_repo_id_data_variant() -> None:
    spec = PythiaModelSpec(size="160m", seed=2, variant="data")
    assert resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-160m-data-seed2"


def test_resolve_hf_repo_id_weight_variant() -> None:
    spec = PythiaModelSpec(size="160m", seed=1, variant="weight")
    assert (
        resolve_hf_repo_id(spec, "EleutherAI") == "EleutherAI/pythia-160m-weight-seed1"
    )


def test_resolve_hf_repo_id_data_variant_rejects_non_160m() -> None:
    spec = PythiaModelSpec(size="410m", seed=1, variant="data")
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(spec, "EleutherAI")


def test_resolve_hf_repo_id_data_variant_rejects_out_of_range_seed() -> None:
    spec = PythiaModelSpec(size="160m", seed=5, variant="data")
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(spec, "EleutherAI")


def test_resolve_hf_repo_id_weightc_variant_is_local_provenance() -> None:
    # The corrected weight-seed re-pretraining is unpublished: no hub org, and a
    # `local/` prefix that `local_model_dir` (not the Hub) knows how to resolve.
    spec = PythiaModelSpec(size="160m", seed=1, variant="weightc")
    assert resolve_hf_repo_id(spec, "EleutherAI") == "local/pythia-160m-weightc-seed1"


def test_resolve_hf_repo_id_weightc_rejects_non_160m_and_bad_seed() -> None:
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(PythiaModelSpec(size="410m", seed=1, variant="weightc"), "E")
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(PythiaModelSpec(size="160m", seed=5, variant="weightc"), "E")


def test_local_model_dir_none_for_hub_repo_ids(tmp_path) -> None:
    assert local_model_dir("EleutherAI/pythia-160m", "step1000", tmp_path) is None


def test_local_model_dir_uses_revision_subdir(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-weightc-seed1", "step25000", tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-weightc-seed1" / "step25000"


def test_local_model_dir_defaults_to_main_subdir(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-weightc-seed1", None, tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-weightc-seed1" / "main"
