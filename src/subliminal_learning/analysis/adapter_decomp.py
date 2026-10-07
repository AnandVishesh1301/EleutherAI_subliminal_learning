"""Moved to ``subliminal_learning.forgetting.analysis.adapter_decomp``; kept for run_logs callers."""

from subliminal_learning.forgetting.analysis import adapter_decomp as _new
from subliminal_learning.forgetting.analysis.adapter_decomp import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_new, name)
