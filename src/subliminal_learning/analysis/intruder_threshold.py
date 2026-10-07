"""Moved to ``subliminal_learning.forgetting.analysis.intruder_threshold``; kept for run_logs callers."""

from subliminal_learning.forgetting.analysis import intruder_threshold as _new
from subliminal_learning.forgetting.analysis.intruder_threshold import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_new, name)
