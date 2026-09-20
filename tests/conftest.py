"""Repo-wide fixtures: the permutation memo (`stats.memo`) never touches the real cache dir
under a test -- it is off by default here; memo tests point it at a ``tmp_path`` explicitly."""
from __future__ import annotations

import pytest

from subliminal_learning.stats import memo


@pytest.fixture(autouse=True)
def _memo_off(monkeypatch):
    monkeypatch.setenv(memo.ENV_VAR, "0")
    memo.set_cache_dir(False)
    memo.reset_cache_stats()
    yield
    memo.set_cache_dir(False)
