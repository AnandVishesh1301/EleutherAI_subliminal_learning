"""Opt-in in-process peak GPU memory reporter (``SUBLIMINAL_REPORT_PEAK_MEM=1``).

nvidia-smi PID attribution is unreliable in these containers, so the only trustworthy
per-job number comes from inside the process: at interpreter exit we ask torch for its
own high-water marks and print one line to stderr, which the jobq worker captures into
the job log (stdout and stderr both land there) and parses into the result record::

    PEAK_GPU_MEM_MIB=<max over devices of max_memory_reserved>
    PEAK_GPU_MEM_ALLOC_MIB=<max over devices of max_memory_allocated>

What the number MEANS, wherever it is shown (result records, the END / OOM log lines, the
``mem`` column of the priors tables): torch's per-device high-water mark, MAXED over the
process's visible devices (not summed -- it is "the most any one card held", which is what a
memory gate is set against), taken from the last 256 KiB of the job log (`log_tail_peak_mem_mib`
takes the max over the lines an ``&&`` chain printed). It is DISPLAY-ONLY: nothing schedules,
gates or escalates on it.

Caveat: these are TORCH caching-allocator numbers. They exclude the CUDA context
(~300-500 MiB per process) and any non-torch allocation (NCCL buffers, bitsandbytes,
driver-side pools), so a job's true card footprint is somewhat higher.

Registration is a no-op unless the env var is set, and the hook never raises (torch
missing, no CUDA, interpreter teardown) — a measurement must never fail a job.
"""

from __future__ import annotations

import atexit
import os
import sys

ENV_VAR = "SUBLIMINAL_REPORT_PEAK_MEM"


def peak_mib() -> tuple[int, int] | None:
    """``(reserved_mib, allocated_mib)`` maxed over devices, or ``None`` if CUDA is unused."""
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return None
        n = torch.cuda.device_count()
        reserved = max(torch.cuda.max_memory_reserved(i) for i in range(n))
        alloc = max(torch.cuda.max_memory_allocated(i) for i in range(n))
    except Exception:  # noqa: BLE001 — never fail a job over a measurement
        return None
    return int(reserved) // (1 << 20), int(alloc) // (1 << 20)


def report_peak() -> None:
    """Print the peak line to stderr; silent when CUDA was never initialized."""
    try:
        peak = peak_mib()
        if peak is None:
            return
        print(
            f"PEAK_GPU_MEM_MIB={peak[0]} PEAK_GPU_MEM_ALLOC_MIB={peak[1]}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # noqa: BLE001 — including teardown-time failures
        pass


def maybe_register(env: dict | None = None) -> bool:
    """Register the atexit hook iff ``SUBLIMINAL_REPORT_PEAK_MEM=1``; return whether it was."""
    if (env if env is not None else os.environ).get(ENV_VAR) != "1":
        return False
    atexit.register(report_peak)
    return True
