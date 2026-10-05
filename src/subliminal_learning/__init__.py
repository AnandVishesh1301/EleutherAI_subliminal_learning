"""Package root.

Import-time side effect (deliberate): cap BLAS/OpenMP threads so that a process
launched without an explicit ``OMP_NUM_THREADS`` can never fan out to every core
on the node.  torch/numpy default to ``os.cpu_count()`` threads (48 here); four
such processes side by side saturate the box and starve the GPU workers.  jobq
injects its own per-job cap via ``gpu_policy.*.json`` and direct CPU runs pin
their own value — any pre-set variable is respected.  Override the fallback with
``SL_DEFAULT_THREADS``.

Second (opt-in) side effect: with ``JOBQ_REPORT_PEAK_MEM=1`` in the env -- the
jobq worker sets it for every job -- register the atexit hook that prints this
process' peak GPU memory (see :mod:`subliminal_learning.gpu_peak`). torch is imported
only inside the hook, so an unset variable costs nothing.
"""

from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
_DEFAULT_THREADS = 4


def _cap_threads() -> int:
    """Apply the fallback thread cap; return the cap that is in force."""
    preset = [os.environ[v] for v in _THREAD_VARS if os.environ.get(v)]
    if preset:
        cap_s = preset[0]
    else:
        cap_s = os.environ.get("SL_DEFAULT_THREADS") or str(
            min(_DEFAULT_THREADS, os.cpu_count() or _DEFAULT_THREADS)
        )
    for v in _THREAD_VARS:
        os.environ.setdefault(v, cap_s)
    try:
        cap = max(1, int(cap_s))
    except ValueError:
        return -1
    # torch reads OMP_NUM_THREADS only at import; if it was imported before us,
    # set the pool size explicitly.
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() > cap:
        torch.set_num_threads(cap)
    return cap


THREAD_CAP = _cap_threads()


from subliminal_learning.gpu_peak import maybe_register as _maybe_register_peak_mem  # noqa: E402

PEAK_MEM_HOOK = _maybe_register_peak_mem()
