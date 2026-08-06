from __future__ import annotations

import pytest

from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id


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
