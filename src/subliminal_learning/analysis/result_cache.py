"""Fingerprint-keyed cache of analysis RESULTS (the "smart check" for trajectory notebooks).

`record_cache` caches the per-run input records; this module caches what an analysis computes
from them, so a notebook rebuild skips any analysis unit whose inputs, code and parameters are
unchanged. The key is `fingerprint(records, params=..., code=...)`: a sha256 over

- the SORTED ``(relative path, mtime_ns, size)`` of every input record file (a missing file
  contributes ``(path, -1, -1)``, so its later appearance changes the key);
- the JSON-canonical parameters (sorted keys, ``default=str``);
- the sha256 of the source text of each analysis module / file.

A unit is therefore recomputed only when one of its input files changed (mtime or size), the
analysis code changed, or a parameter changed; otherwise `cached` returns the stored value.
The fingerprint uses ``(mtime_ns, size)`` rather than content hashes on purpose: a cold-1 stat
costs ~5 ms and runs 64-wide in a thread pool, while content-hashing ~10k JSONs per build is the
cost being avoided. Records are written atomically, so a rewrite always moves the mtime. Callers
that already swept `record_cache` can pass the known ``(path, mtime_ns, size)`` triples and skip
stat entirely.

Unit granularity callers should use: one entry per (pair, design) in `trajectory_asymmetry`,
one per comparison in `trajectory_ab_vs_aa`, one per arm in `trajectory_conditions`. Finer units
waste stats and pickles; coarser units recompute everything when one arm gains a run.

Layout under ``root`` (the outputs root)::

    analysis/result_cache/{namespace}/{unit-slug}/{key}.pkl   # pickle, one format for all values
    analysis/result_cache/{namespace}/index.csv               # unit,key,written_at,n_input_records,bytes

Contract: writes are atomic (temp + ``os.replace``); `store` never evicts other keys (union);
`load` returns None (with a warning) on a missing, unreadable or unpicklable entry, so a bad
entry means recompute, never an error. Index appends are serialised by an ``fcntl`` lock file
(plus a process-local lock), so two concurrent builds leave a well-formed index. Superseded keys
are only removed by an explicit `prune`. ``SL_RESULT_CACHE=0`` bypasses load and store.
"""
from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import logging
import os
import pickle
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, Mapping, Sequence, TypeVar

from subliminal_learning.utils.io import file_lock

T = TypeVar("T")
log = logging.getLogger(__name__)

#: True when ``SL_RESULT_CACHE=0``: `cached` always computes, nothing is loaded or stored.
RESULT_CACHE_DISABLED = os.environ.get("SL_RESULT_CACHE", "1") == "0"

_SWEEP_WORKERS = int(os.environ.get("RECORD_CACHE_SWEEP_WORKERS", "64"))
_INDEX_COLS = ["unit", "key", "written_at", "n_input_records", "bytes"]
_INDEX_TLOCK = threading.Lock()


def _disabled() -> bool:
    return RESULT_CACHE_DISABLED or os.environ.get("SL_RESULT_CACHE", "1") == "0"


def _stat(p: Path) -> tuple[str, int, int]:
    try:
        st = p.stat()
        return (str(p), st.st_mtime_ns, st.st_size)
    except OSError:
        return (str(p), -1, -1)


def _code_hash(c: ModuleType | Path | str) -> str:
    if isinstance(c, ModuleType):
        try:
            text = inspect.getsource(c)
        except (OSError, TypeError):
            text = c.__name__
    else:
        try:
            text = Path(c).read_text()
        except OSError:
            text = f"<missing {c}>"
    return hashlib.sha256(text.encode()).hexdigest()


def fingerprint(
    records: Iterable[tuple[str, int, int]] | Iterable[Path],
    *,
    params: Mapping,
    code: Sequence[ModuleType | Path],
) -> str:
    """Return the sha256 key of an analysis unit (see module docstring: the smart check).

    Args:
        records: Either ``(path, mtime_ns, size)`` triples already known from a record-cache
            sweep, or bare paths, which are stat'ed 64-wide (missing -> ``(path, -1, -1)``).
            Pass the same path convention (e.g. relative to the outputs root) every call.
        params: JSON-serialisable parameters (non-JSON values fall back to ``str``).
        code: Modules or source files whose text determines the result.
    """
    recs = list(records)
    paths = [r for r in recs if isinstance(r, (str, Path)) and not isinstance(r, tuple)]
    triples = [(str(r[0]), int(r[1]), int(r[2])) for r in recs if isinstance(r, tuple)]
    if paths:
        with ThreadPoolExecutor(max_workers=max(1, min(_SWEEP_WORKERS, len(paths)))) as pool:
            triples += list(pool.map(_stat, [Path(p) for p in paths]))
    h = hashlib.sha256()
    for t in sorted(triples):
        h.update(f"{t[0]}\0{t[1]}\0{t[2]}\n".encode())
    h.update(b"\x01params\x01")
    h.update(json.dumps(params, sort_keys=True, default=str, separators=(",", ":")).encode())
    h.update(b"\x01code\x01")
    for c in code:
        h.update(_code_hash(c).encode())
    return h.hexdigest()


def _slug(unit: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", unit).strip("_") or "unit"
    if s != unit:  # disambiguate units that slug to the same string
        s = f"{s[:80]}-{hashlib.sha256(unit.encode()).hexdigest()[:8]}"
    return s


def _ns_dir(namespace: str, root: Path) -> Path:
    return Path(root) / "analysis" / "result_cache" / _slug(namespace)


def entry_path(namespace: str, unit: str, key: str, root: Path) -> Path:
    """Path of one cache entry."""
    return _ns_dir(namespace, root) / _slug(unit) / f"{key}.pkl"


def load(namespace: str, unit: str, key: str, root: Path) -> object | None:
    """Return the stored value, or None if missing/unreadable/unpicklable (warns, never raises)."""
    f = entry_path(namespace, unit, key, root)
    if not f.exists():
        return None
    try:
        with f.open("rb") as fh:
            return pickle.load(fh)
    except Exception as e:  # noqa: BLE001 -- any bad entry means recompute
        log.warning("result_cache: unreadable entry %s (%s: %s); recomputing", f, type(e).__name__, e)
        return None


def _append_index(namespace: str, root: Path, row: list) -> None:
    ns = _ns_dir(namespace, root)
    idx = ns / "index.csv"
    with _INDEX_TLOCK, file_lock(ns / ".index.lock"):
        new = not idx.exists()
        buf = io.StringIO()
        w = csv.writer(buf)
        if new:
            w.writerow(_INDEX_COLS)
        w.writerow(row)
        with idx.open("a") as fh:  # one write per row under the lock -> no interleaving
            fh.write(buf.getvalue())


def store(
    namespace: str, unit: str, key: str, value: object, root: Path, *, n_input_records: int = -1
) -> None:
    """Persist ``value`` atomically; other keys are never evicted. Failures warn, never raise."""
    f = entry_path(namespace, unit, key, root)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        tmp = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, f)
        _append_index(namespace, root, [
            unit, key, datetime.now(timezone.utc).isoformat(timespec="seconds"),
            n_input_records, len(data),
        ])
    except Exception as e:  # noqa: BLE001 -- a failed store only costs a future recompute
        log.warning("result_cache: could not store %s (%s: %s)", f, type(e).__name__, e)


def cached(
    namespace: str,
    unit: str,
    key: str,
    root: Path,
    compute: Callable[[], T],
    *,
    refresh: bool = False,
    n_input_records: int = -1,
) -> T:
    """Return the stored value for ``key`` unless ``refresh``; else compute, store, return."""
    if _disabled():
        return compute()
    if not refresh:
        hit = load(namespace, unit, key, root)
        if hit is not None:
            return hit  # type: ignore[return-value]
    value = compute()
    store(namespace, unit, key, value, root, n_input_records=n_input_records)
    return value


def prune(namespace: str, root: Path, keep_latest_per_unit: int = 1) -> list[Path]:
    """Delete all but the newest ``keep_latest_per_unit`` entries (by mtime) of every unit.

    Only ever runs when called explicitly. Returns the deleted paths.
    """
    ns = _ns_dir(namespace, root)
    removed: list[Path] = []
    if not ns.is_dir():
        return removed
    for unit_dir in (d for d in ns.iterdir() if d.is_dir()):
        entries = sorted(unit_dir.glob("*.pkl"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
        for p in entries[max(0, keep_latest_per_unit):]:
            try:
                p.unlink()
                removed.append(p)
            except OSError as e:
                log.warning("result_cache: could not prune %s (%s)", p, e)
        if not any(unit_dir.iterdir()):
            shutil.rmtree(unit_dir, ignore_errors=True)
    return removed
