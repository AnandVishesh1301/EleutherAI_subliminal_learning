"""`HfModelSpec` resolution, and a byte-identity lock on the PolyPythias path."""

from __future__ import annotations

import pytest

from subliminal_learning.models.hf_spec import HfModelSpec
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    model_ref,
    resolve_hf_repo_id,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import model_dir_name

HUB_ORG = "EleutherAI"

# (spec, repo_id, model_ref, model_dir_name, spec_label) exactly as emitted before
# the cross-family seam was added. Any drift here changes existing run dirs, teacher
# tags and stored run identities, i.e. silently orphans finished cells.
_FROZEN: tuple[tuple[PythiaModelSpec, str, str, str, str], ...] = (
    (
        PythiaModelSpec(size="160m", seed=0, variant="full"),
        "EleutherAI/pythia-160m",
        "EleutherAI/pythia-160m",
        "pythia-160m",
        "full-seed0",
    ),
    (
        PythiaModelSpec(size="160m", seed=1, variant="data"),
        "EleutherAI/pythia-160m-data-seed1",
        "EleutherAI/pythia-160m-data-seed1",
        "pythia-160m-data-seed1",
        "data-seed1",
    ),
    (
        PythiaModelSpec(size="160m", seed=3, variant="weight"),
        "EleutherAI/pythia-160m-weight-seed3",
        "EleutherAI/pythia-160m-weight-seed3",
        "pythia-160m-weight-seed3",
        "weight-seed3",
    ),
    (
        PythiaModelSpec(size="160m", seed=7, variant="full"),
        "EleutherAI/pythia-160m-seed7",
        "EleutherAI/pythia-160m-seed7",
        "pythia-160m-seed7",
        "full-seed7",
    ),
    (
        PythiaModelSpec(size="70m", seed=0, variant="full", revision="step143000"),
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-70m@step143000",
        "pythia-70m@step143000",
        "70m-full-seed0@step143000",
    ),
    (
        PythiaModelSpec(size="410m", seed=0, variant="full", revision="step143000"),
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-410m@step143000",
        "pythia-410m@step143000",
        "410m-full-seed0@step143000",
    ),
    (
        PythiaModelSpec(size="14m", seed=2, variant="full"),
        "EleutherAI/pythia-14m-seed2",
        "EleutherAI/pythia-14m-seed2",
        "pythia-14m-seed2",
        "14m-full-seed2",
    ),
)


def _mamba() -> HfModelSpec:
    return HfModelSpec(
        label="mamba-370m",
        repo_id="state-spaces/mamba-370m-hf",
        revision="a" * 40,
        family="mamba",
        cluster="neox",
        model_type="mamba",
        vocab_size=50280,
        tokenizer_sha256="0" * 64,
        lora_target_modules=("in_proj", "x_proj", "dt_proj", "out_proj"),
    )


@pytest.mark.parametrize(
    ("spec", "repo", "ref", "dir_name", "label"),
    _FROZEN,
    ids=[f[3] for f in _FROZEN],
)
def test_polypythia_labels_are_byte_identical(
    spec: PythiaModelSpec, repo: str, ref: str, dir_name: str, label: str
) -> None:
    assert resolve_hf_repo_id(spec, HUB_ORG) == repo
    assert model_ref(spec, HUB_ORG) == ref
    assert model_dir_name(spec, HUB_ORG) == dir_name
    assert spec_label(spec) == label


def test_hf_spec_resolves_repo_and_label() -> None:
    spec = _mamba()
    assert resolve_hf_repo_id(spec, HUB_ORG) == "state-spaces/mamba-370m-hf"
    assert model_ref(spec, HUB_ORG) == f"state-spaces/mamba-370m-hf@{'a' * 40}"
    assert model_dir_name(spec, HUB_ORG) == "mamba-370m"
    assert spec_label(spec) == "mamba-370m"


def test_hf_spec_shims_read_as_cross_against_a_different_label() -> None:
    a = _mamba()
    b = HfModelSpec(
        label="pythia-410m",
        repo_id="EleutherAI/pythia-410m",
        revision="b" * 40,
        family="pythia",
        cluster="neox",
        model_type="gpt_neox",
        vocab_size=50304,
        tokenizer_sha256="0" * 64,
        lora_target_modules=("query_key_value",),
    )
    assert (a.size, a.variant, a.seed) != (b.size, b.variant, b.seed)
    assert (a.size, a.variant, a.seed) == (a.size, a.variant, a.seed)


def test_hf_spec_rejects_a_label_that_would_break_a_path() -> None:
    for bad in ("", "state-spaces/mamba", "mamba@sha"):
        with pytest.raises(ValueError, match="label"):
            HfModelSpec(
                label=bad,
                repo_id="state-spaces/mamba-370m-hf",
                revision=None,
                family="mamba",
                cluster="neox",
                model_type="mamba",
                vocab_size=50280,
                tokenizer_sha256="0" * 64,
                lora_target_modules=("in_proj",),
            )


def test_hf_spec_requires_explicit_lora_targets() -> None:
    with pytest.raises(ValueError, match="lora_target_modules"):
        HfModelSpec(
            label="mamba-370m",
            repo_id="state-spaces/mamba-370m-hf",
            revision=None,
            family="mamba",
            cluster="neox",
            model_type="mamba",
            vocab_size=50280,
            tokenizer_sha256="0" * 64,
            lora_target_modules=(),
        )


def test_hf_spec_rejects_an_unknown_post_training_stage() -> None:
    with pytest.raises(ValueError, match="post_training"):
        HfModelSpec(
            label="x",
            repo_id="a/b",
            revision=None,
            family="f",
            cluster="neox",
            model_type="gpt_neox",
            vocab_size=1,
            tokenizer_sha256="0" * 64,
            lora_target_modules=("q",),
            post_training="grpo",
        )


def test_parse_base_models_without_registry_is_unchanged() -> None:
    assert parse_base_models("original") == [
        PythiaModelSpec(size="160m", seed=0, variant="full")
    ]
    assert parse_base_models("pythia-410m@step143000") == [
        PythiaModelSpec(size="410m", seed=0, variant="full", revision="step143000")
    ]
    assert len(parse_base_models("all7")) == 7
    with pytest.raises(ValueError):
        parse_base_models("mamba-370m")


def test_parse_base_models_resolves_registry_labels() -> None:
    registry = {"mamba-370m": _mamba()}
    assert parse_base_models("mamba-370m", registry=registry) == [_mamba()]
    # A value with no roster label falls through to the PolyPythias token space.
    assert parse_base_models("original", registry=registry) == [
        PythiaModelSpec(size="160m", seed=0, variant="full")
    ]


def test_parse_base_models_refuses_to_mix_label_spaces() -> None:
    registry = {"mamba-370m": _mamba()}
    with pytest.raises(ValueError, match="do not mix"):
        parse_base_models("mamba-370m,original", registry=registry)
