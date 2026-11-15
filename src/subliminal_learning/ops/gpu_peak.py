"""Opt-in in-process peak GPU memory reporter (``JOBQ_REPORT_PEAK_MEM=1``).

Thin re-export of :mod:`jobq.report`: the jobq worker sets ``JOBQ_REPORT_PEAK_MEM=1`` in each
job's env and parses the ``JOBQ_PEAK_GPU_MEM_MIB=<n> [JOBQ_PEAK_GPU_MEM_ALLOC_MIB=<n>]`` line
this hook prints to stderr at exit. The numbers are torch caching-allocator high-water marks
maxed over visible devices (excluding the CUDA context); display-only.

``jobq`` (gpu-jobq) is optional at import time: without it the hook is a no-op, so
``import subliminal_learning`` works in an environment that only runs analysis.
"""

from __future__ import annotations

ENV_VAR = "JOBQ_REPORT_PEAK_MEM"

try:
    from jobq.report import ENV_VAR, maybe_register, peak_mib, report_peak  # noqa: F811
except ImportError:  # gpu-jobq not installed: the reporter is a no-op
    def maybe_register():  # type: ignore[misc]
        """No jobq available: nothing to register; return None like an unset env var."""
        return None

    def peak_mib():  # type: ignore[misc]
        raise ImportError("gpu-jobq is not installed; peak GPU memory reporting is unavailable")

    def report_peak():  # type: ignore[misc]
        raise ImportError("gpu-jobq is not installed; peak GPU memory reporting is unavailable")

__all__ = ["ENV_VAR", "maybe_register", "peak_mib", "report_peak"]
