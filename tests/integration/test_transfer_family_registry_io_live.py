"""`analysis.loaders.transfer_family_registry_io` on the REAL family registry. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

import json
import shutil
import pytest
from tests._helpers import REPO
from subliminal_learning.analysis.loaders import transfer_family_registry_io as reg
from subliminal_learning.analysis.loaders import registry as analysis_registry


@pytest.fixture(autouse=True)
def _family_registry_json(tmp_path):
    """The tmp checkout carries the schema-example ``family_registry.json`` binding."""
    dst = analysis_registry.registry_path(reg.FAMILY_REGISTRY_JSON, tmp_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(analysis_registry.examples_dir() / reg.FAMILY_REGISTRY_JSON, dst)


def test_actual_registry_categories_cover_the_declared_comparisons():
    """On the REAL registries (the machine-local run_logs tree); skips cleanly elsewhere."""
    repo = REPO
    try:
        paths = reg._registry_paths(repo)
    except FileNotFoundError as exc:
        pytest.skip(f"real family registry not on this machine: {exc}")
    if not all(p.is_file() for p in paths):
        pytest.skip("real significance registries not on this machine")
    all_arms = [a for p in paths for a in json.loads(p.read_text())["arms"]]
    cats = [reg._category(a) for a in all_arms]
    assert set(reg.EXTRA_RELATIONS).issubset({c for c in cats if c is not None})
    ignored = [a for a, c in zip(all_arms, cats) if c is None]
    assert ignored and all(a["sender"] == a["receiver"] for a in ignored)
    assert reg.RELATION_OLMO_SAME not in cats
    assert reg._category({"sender": "dd-1b", "receiver": "dd-1b", "data_order": "same"}, include_olmo_same_model=True) == reg.RELATION_OLMO_SAME
    for a in all_arms:
        category = reg._category(a)
        if a['data_order'] == 'cross-recipe+seed':
            assert category == reg.EXTRA_RELATIONS[7]
        if a['data_order'] == 'cross-recipe':
            assert category == reg.EXTRA_RELATIONS[6]
    assert reg._category({'sender':'dd-1b', 'receiver':'dd-1b-seed2', 'data_order':'same'}) == reg.EXTRA_RELATIONS[8]
    assert reg._category({'sender':'dd-1b', 'receiver':'dd-1b', 'data_order':'same'}) is None
    assert reg._category({'sender':'pythia-160m', 'receiver':'DataDecide-dolma1_7-1B-seed-default'}) == reg.EXTRA_RELATIONS[4]
