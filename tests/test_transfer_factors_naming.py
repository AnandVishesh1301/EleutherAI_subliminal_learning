"""Short-label <-> on-disk model name round-trips and the anchored-cut map construction."""

import pytest

from subliminal_learning.analysis.transfer_factors import (
    anchored_cut_by_base,
    expand_model_name,
    normalize_model_name,
)

LABELS = (
    ["original"]
    + [f"data-seed{i}" for i in (1, 2, 3)]
    + [f"weight-seed{i}" for i in (1, 2, 3)]
    + [f"full-seed{i}" for i in range(1, 10)]
)


@pytest.mark.parametrize("label", LABELS)
def test_round_trip(label):
    assert normalize_model_name(expand_model_name(label)) == label


def test_expansion():
    assert expand_model_name("original") == "pythia-160m"
    assert expand_model_name("full-seed2") == "pythia-160m-seed2"
    assert expand_model_name("data-seed1") == "pythia-160m-data-seed1"
    assert expand_model_name("original", "pythia-410m") == "pythia-410m"
    assert expand_model_name("full-seed1", "pythia-410m") == "pythia-410m-seed1"


def test_expansion_passes_through_full_names():
    assert expand_model_name("pythia-160m-data-seed1") == "pythia-160m-data-seed1"
    assert expand_model_name("pythia-160m") == "pythia-160m"


def test_anchored_cut_by_base():
    dcross = anchored_cut_by_base({
        "pythia-160m": 1.0,
        "pythia-160m-data-seed1": 2.0,
        "pythia-160m-weight-seed3": 3.0,
        "pythia-160m-seed1": 4.0,
        "pythia-160m (orig)": 9.0,   # alias key -- must be ignored
        "pythia-410m": 5.0,          # other family
    })
    assert dcross == {"original": 1.0, "data-seed1": 2.0, "weight-seed3": 3.0,
                      "full-seed1": 4.0}
