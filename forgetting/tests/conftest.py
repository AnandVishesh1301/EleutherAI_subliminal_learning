"""Repo-wide fixtures: the permutation/loader memo (`stats.memo`) never touches the real cache
dir under a test -- it is off by default here; memo tests point it at a ``tmp_path`` explicitly.
``SUBLIMINAL_TEST_MEMO_DIR=<dir>`` runs the whole suite THROUGH a memo at that dir instead (a
way to check the cached path reproduces every uncached result)."""
from __future__ import annotations

import os

import pytest

from subliminal_learning.stats import memo


@pytest.fixture(autouse=True)
def _memo_off(monkeypatch):
    through = os.environ.get("SUBLIMINAL_TEST_MEMO_DIR")
    if through:
        monkeypatch.setenv(memo.ENV_VAR, "1")
        memo.set_cache_dir(through)
    else:
        monkeypatch.setenv(memo.ENV_VAR, "0")
        memo.set_cache_dir(False)
    memo.reset_cache_stats()
    yield
    memo.set_cache_dir(False)
