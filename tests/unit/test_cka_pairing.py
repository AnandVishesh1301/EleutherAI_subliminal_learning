"""Revision-aware label handling and the basis-validity stamp."""

from __future__ import annotations

import pytest

from subliminal_learning.cka.pairing import (
    BasisValidity,
    basis_validity,
    canonical_label,
    pairing_group_for,
    same_revision,
    split_revision,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("data-seed1", ("data-seed1", None)),
        ("data-seed1@step16000", ("data-seed1", "step16000")),
        ("original@step0", ("original", "step0")),
        (
            "EleutherAI/pythia-160m-weight-seed3@main",
            ("EleutherAI/pythia-160m-weight-seed3", "main"),
        ),
    ],
)
def test_split_revision(label: str, expected: tuple[str, str | None]):
    assert split_revision(label) == expected


def test_split_revision_rejects_an_empty_revision():
    with pytest.raises(ValueError, match="empty revision"):
        split_revision("data-seed1@")


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("original", "original"),
        ("full-seed0", "original"),
        ("pythia-160m", "original"),
        ("original@step16000", "original"),
        ("data-seed1", "data-seed1"),
        ("data-seed1@step16000", "data-seed1"),
        ("pythia-160m-data-seed1", "data-seed1"),
        ("EleutherAI/pythia-160m-weight-seed3", "weight-seed3"),
        ("pythia-160m-weight-seed3@step3000", "weight-seed3"),
    ],
)
def test_canonical_label_collapses_both_label_spaces(label: str, expected: str):
    assert canonical_label(label) == expected


@pytest.mark.parametrize(
    ("label", "match"),
    [
        ("pythia-160m-seed1", "only"),
        ("data", "aggregate selector"),
        ("all7", "aggregate selector"),
        ("nonsense", "bad base-model token"),
    ],
)
def test_canonical_label_raises_on_out_of_scope_labels(label: str, match: str):
    with pytest.raises(ValueError, match=match):
        canonical_label(label)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("data-seed1", "data-seed1", BasisValidity.BASIS_SHARED),
        (
            "data-seed1@step16000",
            "pythia-160m-data-seed1@step16000",
            BasisValidity.BASIS_SHARED,
        ),
        ("data-seed1@step0", "data-seed1@step16000", BasisValidity.BASIS_LINEAGE),
        ("data-seed1", "data-seed2", BasisValidity.BASIS_DIVERGED),
        ("original", "data-seed1", BasisValidity.BASIS_DIVERGED),
        ("data-seed1@step0", "data-seed2@step0", BasisValidity.BASIS_SHARED),
        ("data-seed1@step16000", "data-seed2@step16000", BasisValidity.BASIS_DIVERGED),
        ("data-seed1", "weight-seed1", BasisValidity.BASIS_BROKEN),
        ("data-seed1@step0", "weight-seed1@step0", BasisValidity.BASIS_BROKEN),
        ("original", "weight-seed2", BasisValidity.BASIS_BROKEN),
        ("weight-seed1", "weight-seed2", BasisValidity.BASIS_BROKEN),
        ("weight-seed1@step0", "weight-seed2@step0", BasisValidity.BASIS_BROKEN),
        ("weight-seed1@step0", "weight-seed1@step143000", BasisValidity.BASIS_LINEAGE),
    ],
)
def test_basis_validity(a: str, b: str, expected: BasisValidity):
    assert basis_validity(a, b) is expected
    assert basis_validity(b, a) is expected


def test_basis_validity_is_a_string_enum_not_a_mutable_mapping():
    assert str(BasisValidity.BASIS_BROKEN) == "basis_broken"
    assert BasisValidity("basis_lineage") is BasisValidity.BASIS_LINEAGE


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("data-seed1@step0", "weight-seed1@step0", True),
        ("data-seed1@step0", "weight-seed1@step16000", False),
        ("data-seed1", "weight-seed1", True),
        ("data-seed1", "weight-seed1@step0", False),
    ],
)
def test_same_revision(a: str, b: str, expected: bool):
    assert same_revision(a, b) is expected


@pytest.mark.parametrize(
    ("student", "teacher", "expected"),
    [
        ("data-seed1", "data-seed1", "same_init"),
        ("data-seed1@step16000", "data-seed1@step16000", "same_init"),
        ("data-seed2", "data-seed1", "data_cyclic"),
        ("weight-seed2", "weight-seed1", "weight_cyclic"),
        ("data-seed1", "weight-seed1", "family_swap"),
        (
            "pythia-160m-data-seed1@step0",
            "pythia-160m-weight-seed1@step0",
            "family_swap",
        ),
        ("data-seed1", "original", "seed0_data"),
        ("data-seed1", "data-seed2", None),
    ],
)
def test_pairing_group_for_strips_revisions_before_matching(
    student: str, teacher: str, expected: str | None
):
    assert pairing_group_for(student, teacher) == expected
