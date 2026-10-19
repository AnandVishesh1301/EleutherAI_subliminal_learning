#!/usr/bin/env python3
"""Occupy GPU memory + SM util until killed. Uses tools/sdf_docgen/.venv (torch).

Example:
  tools/sdf_docgen/.venv/bin/python tools/occupy_gpus.py --gpus 4,5,6,7 --mem-gb 8
  tools/sdf_docgen/.venv/bin/python tools/occupy_gpus.py --gpus 4 --mem-gb 1 --seconds 5
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time


def _parse_gpus(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("empty --gpus")
    gpus: list[int] = []
    for p in parts:
        try:
            g = int(p)
        except ValueError as e:
            raise argparse.ArgumentTypeError(f"bad gpu id: {p!r}") from e
        if g < 0:
            raise argparse.ArgumentTypeError(f"gpu id must be >= 0, got {g}")
        gpus.append(g)
    return gpus


def _worker(gpu: int, mem_gb: float, matrix_n: int, seconds: float | None) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Import after pinning so torch only sees this device.
    import torch

    if not torch.cuda.is_available():
        print(f"[gpu {gpu}] cuda unavailable", file=sys.stderr, flush=True)
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    total_bytes = props.total_memory
    target_bytes = int(mem_gb * (1024**3))
    if target_bytes <= 0:
        print(f"[gpu {gpu}] --mem-gb must be > 0", file=sys.stderr, flush=True)
        sys.exit(1)
    if target_bytes >= total_bytes:
        print(
            f"[gpu {gpu}] mem-gb={mem_gb} exceeds device total "
            f"{total_bytes / (1024**3):.1f} GiB",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    stop = False

    def _stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Hold VRAM with a contiguous buffer (bf16 on A40).
    n_elems = target_bytes // 2  # torch.bfloat16 = 2 bytes
    try:
        hold = torch.empty(n_elems, dtype=torch.bfloat16, device=device)
        hold.fill_(1)
    except torch.cuda.OutOfMemoryError:
        print(
            f"[gpu {gpu}] OOM allocating {mem_gb:.2f} GiB "
            f"(free may be lower than requested)",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(1)

    allocated = torch.cuda.memory_allocated(0) / (1024**3)
    print(
        f"[gpu {gpu}] allocated ~{allocated:.2f} GiB "
        f"(requested {mem_gb:.2f}); burning matmul n={matrix_n}",
        flush=True,
    )

    a = torch.randn(matrix_n, matrix_n, device=device, dtype=torch.bfloat16)
    b = torch.randn(matrix_n, matrix_n, device=device, dtype=torch.bfloat16)
    t0 = time.monotonic()
    iters = 0
    while not stop:
        c = a @ b
        a = c
        iters += 1
        if seconds is not None and (time.monotonic() - t0) >= seconds:
            break
        # occasional sync so nvidia-smi util stays honest without killing throughput
        if iters % 50 == 0:
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    del hold, a, b
    torch.cuda.empty_cache()
    print(f"[gpu {gpu}] exiting after {iters} iters", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Occupy GPU VRAM + utilization until Ctrl-C / SIGTERM."
    )
    parser.add_argument(
        "--gpus",
        type=_parse_gpus,
        required=True,
        help="Comma-separated GPU indices, e.g. 4,5,6,7",
    )
    parser.add_argument(
        "--mem-gb",
        type=float,
        default=8.0,
        help="VRAM to hold per GPU in GiB (default: 8)",
    )
    parser.add_argument(
        "--matrix-n",
        type=int,
        default=8192,
        help="Square matmul side length for SM burn (default: 8192)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Optional auto-exit after N seconds (for smoke tests). Default: run forever.",
    )
    args = parser.parse_args()

    if args.matrix_n < 64:
        parser.error("--matrix-n must be >= 64")

    gpus = args.gpus
    print(
        f"launching workers on gpus={gpus} mem_gb={args.mem_gb} "
        f"matrix_n={args.matrix_n} seconds={args.seconds}",
        flush=True,
    )

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for g in gpus:
        p = ctx.Process(
            target=_worker,
            args=(g, args.mem_gb, args.matrix_n, args.seconds),
            name=f"occupy-gpu-{g}",
            daemon=False,
        )
        p.start()
        procs.append(p)

    def _forward_signal(signum: int, _frame: object) -> None:
        for p in procs:
            if p.is_alive() and p.pid is not None:
                os.kill(p.pid, signum)

    signal.signal(signal.SIGINT, _forward_signal)
    signal.signal(signal.SIGTERM, _forward_signal)

    rc = 0
    for p in procs:
        p.join()
        if p.exitcode not in (0, None, -signal.SIGINT, -signal.SIGTERM):
            # negative exitcode = killed by signal; treat SIGINT/SIGTERM as ok
            if p.exitcode is not None and p.exitcode < 0:
                continue
            print(
                f"worker {p.name} exited with code {p.exitcode}",
                file=sys.stderr,
                flush=True,
            )
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
