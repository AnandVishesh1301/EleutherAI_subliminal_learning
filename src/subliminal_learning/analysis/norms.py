"""Moved to ``subliminal_learning.forgetting.analysis.norms``; kept for run_logs callers."""

from subliminal_learning.forgetting.analysis import norms as _new
from subliminal_learning.forgetting.analysis.norms import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_new, name)
