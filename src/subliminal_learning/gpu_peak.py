"""Opt-in in-process peak GPU memory reporter (``JOBQ_REPORT_PEAK_MEM=1``).

Thin re-export of :mod:`jobq.report`: the jobq worker sets ``JOBQ_REPORT_PEAK_MEM=1`` in each
job's env and parses the ``JOBQ_PEAK_GPU_MEM_MIB=<n> [JOBQ_PEAK_GPU_MEM_ALLOC_MIB=<n>]`` line
this hook prints to stderr at exit. The numbers are torch caching-allocator high-water marks
maxed over visible devices (excluding the CUDA context); display-only.
"""

from __future__ import annotations

from jobq.report import ENV_VAR, maybe_register, peak_mib, report_peak

__all__ = ["ENV_VAR", "maybe_register", "peak_mib", "report_peak"]
