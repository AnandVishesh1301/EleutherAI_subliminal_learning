"""`transfer_headline` xarch direction arms from the REAL registry. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

import pytest
from subliminal_learning.analysis.transfer import transfer_headline as th


def test_direction_arms_pair_by_model_identity():
    try:
        a, b = th.XARCH_DIRECTION_ARMS      # lazy: the configured run_logs registry
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    assert len(a) == len(b) >= 1          # the real grid is 3x3 = 9; the schema example 2
    for x, y in zip(a, b):
        s, r = x.split(" (")[0].split(" → ")
        assert y.startswith(f"{r} → {s} (dog), core30")
