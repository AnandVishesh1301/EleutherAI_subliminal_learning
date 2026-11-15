"""One-shot loguru configuration for script entry points.

Also the one place the opt-in peak-GPU-memory reporter is registered: the jobq worker sets
``JOBQ_REPORT_PEAK_MEM=1`` for every job, and every CLI entry point calls `configure_loguru`
first, so a jobq-launched job gets the atexit hook (:mod:`subliminal_learning.ops.gpu_peak`)
while a plain ``import subliminal_learning`` never touches jobq.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

PEAK_MEM_ENV_VAR = "JOBQ_REPORT_PEAK_MEM"


def configure_loguru() -> None:
    """Configure loguru to print INFO+ logs to stderr with a compact format."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    if os.environ.get(PEAK_MEM_ENV_VAR):
        from subliminal_learning.ops.gpu_peak import maybe_register

        maybe_register()
