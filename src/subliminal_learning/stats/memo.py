"""Content-addressed memo for the seeded permutation tests (and any other pure function).

The permutation tests in `transfer_significance_tests` are deterministic in their inputs (the
draws are seeded), so a result computed once on the same inputs is EXACT the second time, not
approximate. This module stores such results as small JSON files keyed on a sha256 of the exact
inputs -- array bytes + shape + dtype, every scalar option, the function name and
`CACHE_VERSION` -- so any input change is a new key and a recompute; nothing is keyed on file
mtimes. The cache dir is ``{output_root}/analysis/cache/transfer_significance/`` (machine-shared,
created on demand); ``SUBLIMINAL_ANALYSIS_CACHE=0`` disables reading AND writing. A corrupt or
unreadable entry is recomputed and overwritten, never raised. This is the stats package's one
disk touch: a transparent, opt-out memo that never changes a result.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

__all__ = ["CACHE_VERSION", "ENV_VAR", "cache_dir", "set_cache_dir", "key", "load", "store",
           "memoized", "cache_stats", "reset_cache_stats"]

#: Bump when a memoized function's semantics change: every old entry becomes a miss.
CACHE_VERSION = 1

#: ``SUBLIMINAL_ANALYSIS_CACHE=0`` turns the memo off (compute; neither read nor write).
ENV_VAR = "SUBLIMINAL_ANALYSIS_CACHE"

_SUBDIR = Path("analysis/cache/transfer_significance")
_override: Path | None | bool = False      # False = not overridden; None/Path = forced value
_stats = {"hits": 0, "misses": 0}

T = TypeVar("T")


def enabled() -> bool:
    """False when `ENV_VAR` is ``0``/``false``/``off``/``no`` (case-insensitive)."""
    return os.environ.get(ENV_VAR, "1").strip().lower() not in ("0", "false", "off", "no")


def set_cache_dir(path: Path | str | None | bool = False) -> None:
    """Force the cache dir (tests: a ``tmp_path``), ``None`` to disable, ``False`` to restore
    the default resolution from the paths YAML."""
    global _override
    _override = path if (path is None or path is False) else Path(path)


def cache_dir() -> Path | None:
    """The memo dir, or None when the memo is off (env var, `set_cache_dir(None)`, or no
    resolvable output root)."""
    if not enabled():
        return None
    if _override is not False:
        return _override
    from subliminal_learning.config import resolve_output_root

    try:
        return resolve_output_root(None) / _SUBDIR
    except Exception:       # no paths YAML on this machine: compute, never fail
        return None


def _feed(h: "hashlib._Hash", part: Any) -> None:
    """Feed one input into the hash with a type tag, so ``1`` and ``"1"`` differ."""
    if isinstance(part, np.ndarray):
        if part.dtype.kind in "OUS":            # labels: stable string form
            h.update(b"S")
            _feed(h, [str(x) for x in part.ravel().tolist()])
            h.update(repr(part.shape).encode())
        else:
            h.update(b"A")
            h.update(part.dtype.str.encode())
            h.update(repr(part.shape).encode())
            h.update(np.ascontiguousarray(part).tobytes())
    elif isinstance(part, (list, tuple)):
        h.update(f"L{len(part)}".encode())
        for x in part:
            _feed(h, x)
    elif isinstance(part, dict):
        h.update(f"D{len(part)}".encode())
        for k in sorted(part):
            _feed(h, str(k))
            _feed(h, part[k])
    elif part is None or isinstance(part, (bool, int, float, str, np.generic)):
        h.update(f"{type(part).__name__}:{part!r}|".encode())
    else:
        raise TypeError(f"unhashable memo input of type {type(part).__name__}")


def key(name: str, *parts: Any) -> str:
    """sha256 hex of ``(CACHE_VERSION, name, *parts)``; `parts` are arrays, scalars, strings,
    None, or (nested) lists/tuples/dicts of those."""
    h = hashlib.sha256()
    _feed(h, f"v{CACHE_VERSION}:{name}")
    for p in parts:
        _feed(h, p)
    return h.hexdigest()


def _path(name: str, k: str) -> Path | None:
    d = cache_dir()
    return None if d is None else d / name / f"{k}.json"


def load(name: str, k: str, result_cls: Callable[..., T]) -> T | None:
    """The stored result for `k`, rebuilt as `result_cls(**fields)`; None on a miss, a disabled
    memo, or an unreadable entry (which the next `store` overwrites)."""
    path = _path(name, k)
    if path is None:
        return None
    try:
        fields = json.loads(path.read_text())
        out = result_cls(**fields)
    except Exception:
        return None
    return out


def store(name: str, k: str, result: Any) -> None:
    """Write `result` (a dataclass, or a dict) atomically; silently a no-op when disabled or
    when the dir is not writable."""
    path = _path(name, k)
    if path is None:
        return
    payload = asdict(result) if is_dataclass(result) else dict(result)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
    except Exception:
        return


def memoized(name: str, k: str, result_cls: Callable[..., T], compute: Callable[[], T], *,
             encode: Callable[[T], Any] | None = None) -> T:
    """`load` or `compute` + `store`, counting the hit/miss in `cache_stats`; `encode` (default:
    the dataclass itself) turns a computed value into what `store` writes, `result_cls`
    rebuilds it from the stored fields."""
    cached = load(name, k, result_cls)
    if cached is not None:
        _stats["hits"] += 1
        return cached
    _stats["misses"] += 1
    result = compute()
    store(name, k, result if encode is None else encode(result))
    return result


def cache_stats() -> dict[str, int]:
    """``{"hits": n, "misses": m}`` since import or the last `reset_cache_stats`."""
    return dict(_stats)


def reset_cache_stats() -> None:
    _stats["hits"] = _stats["misses"] = 0
