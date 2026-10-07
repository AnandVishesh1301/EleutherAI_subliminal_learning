"""Moved to ``subliminal_learning.forgetting.weightspace``; kept for run_logs callers.

``__path__`` points at the new package directory so ``subliminal_learning.weightspace.lora``
etc. still import (as separate module objects loaded from the same files).
"""

from __future__ import annotations

from pathlib import Path as _Path

from subliminal_learning.forgetting import weightspace as _new
from subliminal_learning.forgetting.weightspace import *  # noqa: F401,F403

__path__ = [str(_Path(_new.__file__).resolve().parent)]


def __getattr__(name):
    return getattr(_new, name)
