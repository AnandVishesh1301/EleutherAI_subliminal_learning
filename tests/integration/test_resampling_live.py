"""`numbers.resampling` k=1 against the stage-1 pools on disk. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

import json
import pytest
from scripts.campaigns import resampling as final
from subliminal_learning.numbers.resampling import (
    replace_first_number_from_pair,
    truncate_after_first_number,
)


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("leaf", ["synthfirstonly", "synthtrunc1"])
def test_k1_reproduces_the_pools_already_on_disk(leaf):
    """The real stage-1 k=1 regression: byte-identical rows to the built pools."""
    root = final.default_output_root()
    teacher = final.load_teachers(None, final.ROOT / final.CAMPAIGN_REL)[0]
    paths = [
        final.carriers_path(root),
        final.final_pool_dir(root, teacher, "synthiid") / "filtered.jsonl",
        final.final_pool_dir(root, teacher, leaf) / "filtered.jsonl",
    ]
    if not all(p.is_file() for p in paths):
        pytest.skip("stage-1 pools are not on this machine")
    carriers, s_all, expected = (_read(p) for p in paths)
    fn = replace_first_number_from_pair if leaf == "synthfirstonly" else truncate_after_first_number
    rows = [{k: r[k] for k in final.TRAINING_FIELDS} for r in fn(carriers, s_all)]
    assert rows == expected
