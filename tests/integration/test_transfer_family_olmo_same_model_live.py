"""Live-tree check of the OLMo same-model family registry.

Reads the real ``run_logs/registry/family_registry.json`` and the frozen
``olmo_dclm_to_c4_core35_20260920`` campaign through the repository checkout, so it
only runs where that machine-local tree exists (auto-marked ``integration``).
"""

import pytest

from subliminal_learning.analysis.transfer import transfer_family_complete as complete
from subliminal_learning.analysis.loaders import transfer_family_registry_io as reg
from subliminal_learning.analysis.loaders import transfer_significance_io as tio
from tests._helpers import REPO

_CAMPAIGN_TSV = REPO / "run_logs/campaigns/olmo_dclm_to_c4_core35_20260920/teachers.tsv"


@pytest.fixture(autouse=True)
def _needs_live_tree():
    if not _CAMPAIGN_TSV.is_file():
        pytest.skip(f"live run_logs tree absent: {_CAMPAIGN_TSV}")


def test_campaign_uses_all_exact_teachers_and_keeps_defaults():
    repo = REPO
    path, document = reg._olmo_same_model_registry(repo)
    assert len(document["arms"]) == 3
    assert document["figure2_olmo_same_model_core35"]
    for arm in document["arms"]:
        assert len(arm["teachers"]) == 10 and len(set(arm["teachers"])) == 10
        assert arm["tseed"] == 0 and arm["ftseed"] == 1
        assert tio.base_name(arm["sender"]) == tio.base_name(arm["receiver"])
        assert reg._category(arm) is None
        assert (
            reg._category(arm, include_olmo_same_model=True) == reg.RELATION_OLMO_SAME
        )
        assert set(
            tio.resolve_cells(path.parent / arm["cells"])
        ) == complete.canonical_cells(repo, "core35")
    first = (
        (repo / "run_logs/campaigns/olmo_dclm_to_c4_core35_20260920/teachers.tsv")
        .read_text()
        .splitlines()[0]
        .split("\t")[0]
    )
    assert document["arms"][1]["teachers"][0] == first
    assert (
        reg._category(
            dict(sender="pythia-160m", receiver="pythia-160m"),
            include_olmo_same_model=True,
        )
        is None
    )
    assert reg.RELATION_OLMO_SAME not in reg.EXTRA_RELATIONS
    assert reg.relation_groups([reg.RELATION_OLMO_SAME]) == [
        [],
        [],
        [reg.RELATION_OLMO_SAME],
    ]
