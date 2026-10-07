"""Moved to ``subliminal_learning.forgetting.analysis.kstar_structure``; kept for run_logs callers."""

from subliminal_learning.forgetting.analysis import kstar_structure as _new
from subliminal_learning.forgetting.analysis.kstar_structure import *  # noqa: F401,F403


def __getattr__(name):
    return getattr(_new, name)
