from __future__ import annotations

import pytest

from subliminal_learning.models.registry import (
    LOCAL_PYTHIA_FAMILIES,
    PythiaModelSpec,
    local_pythia_family,
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


@pytest.mark.parametrize(
    "variant,seed",
    [("weightc", 3), ("pilerepro", 1234), ("pilerepro", 3), ("dclm", 1234), ("dclm", 1)],
)
def test_resolve_hf_repo_id_local_families_are_local_provenance(variant, seed) -> None:
    # Every LOCAL_PYTHIA_FAMILIES row resolves the same way weightc does.
    spec = PythiaModelSpec(size="160m", seed=seed, variant=variant)
    assert (
        resolve_hf_repo_id(spec, "EleutherAI")
        == f"local/pythia-160m-{variant}-seed{seed}"
    )


@pytest.mark.parametrize("variant,seed", [("pilerepro", 1), ("dclm", 3), ("weightc", 1234)])
def test_resolve_hf_repo_id_local_families_reject_foreign_seeds(variant, seed) -> None:
    with pytest.raises(ValueError, match="requires seed in"):
        resolve_hf_repo_id(PythiaModelSpec(size="160m", seed=seed, variant=variant), "E")


@pytest.mark.parametrize("variant", ["pilerepro", "dclm"])
def test_resolve_hf_repo_id_local_families_reject_non_160m(variant) -> None:
    with pytest.raises(ValueError, match="only valid for size='160m'"):
        resolve_hf_repo_id(PythiaModelSpec(size="410m", seed=1234, variant=variant), "E")


def test_local_family_table_shape() -> None:
    assert [f.variant for f in LOCAL_PYTHIA_FAMILIES] == ["weightc", "pilerepro", "dclm"]
    assert local_pythia_family("weightc").seeds == (1, 2, 3)
    assert local_pythia_family("weightc").seeds_label() == "1-3"  # unchanged wording
    assert local_pythia_family("dclm").seeds == (1234, 1, 2)
    assert local_pythia_family("dclm").dir_name(1234) == "pythia-160m-dclm-seed1234"
    assert local_pythia_family("data") is None


def test_local_model_dir_handles_four_digit_seed(tmp_path) -> None:
    d = local_model_dir("local/pythia-160m-dclm-seed1234", "step75000", tmp_path)
    assert d == tmp_path / "local" / "pythia-160m-dclm-seed1234" / "step75000"
