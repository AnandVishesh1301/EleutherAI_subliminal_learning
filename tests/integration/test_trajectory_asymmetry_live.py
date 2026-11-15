"""`analysis.trajectories.trajectory_asymmetry` against the REAL arm registries. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

from __future__ import annotations

import pytest
from subliminal_learning.analysis.trajectories import trajectory_asymmetry as tg
from subliminal_learning.analysis.transfer import transfer_asymmetry as ta
from tests._helpers import REPO


@pytest.fixture(autouse=True)
def _no_result_cache(monkeypatch):
    """The monkeypatched sources are invisible to the result-cache keys: tests run uncached
    unless they opt in (the result-cache tests at the end)."""
    monkeypatch.setenv("SL_RESULT_CACHE", "0")


def test_re_pretrain_family_tests_prv1_pr1234_on_the_stride3_design(real_run_logs):

    repo = REPO
    fam = ta.FAMILIES["re-pretrain"]
    assert {"prv1", "pr1234"} <= set(fam.bases)
    designs = dict(ta.family_designs(fam, repo))
    assert list(designs) == ["core30", "core30 stride-3"]        # core30 first, then fallback
    stride3 = designs["core30 stride-3"]
    assert len(stride3) == 10 and set(stride3) <= set(designs["core30"])
    assert tg._CELLSET_NAMES["earlystop"] == "core30 stride-3"   # one display name, both paths


def test_olmo_stop_arms_keep_only_two_direction_pairs(real_run_logs):
    repo = REPO
    arms = tg.olmo_stop_arms(repo)
    pairs = {(s.sender, s.receiver) for s in arms if s.kind == "treatment"}
    assert len(pairs) == 6 and all((r, s) in pairs for s, r in pairs)
    assert not any("aux" in s.receiver for s in arms)
    assert {s.control_of for s in arms if s.kind == "control"} == {
        s.label for s in arms if s.kind == "treatment"}
