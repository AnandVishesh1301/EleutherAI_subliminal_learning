"""Shared filesystem helpers: content hashing and crash-safe atomic writes.

These consolidate copies that were duplicated across the pipelines and scripts. Resume/skip
gates in this repo hash the bytes of written files, so the write helpers keep their exact
on-disk byte layout (JSON ``indent``/``sort_keys``/trailing newline are explicit parameters).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path


def rglob_follow(root: str | Path, pattern: str) -> list[Path]:
    """``Path.rglob`` that also descends into symlinked directories.

    ``Path.rglob`` (and any ``**`` glob) never follows directory symlinks. When part of an
    output tree is archived to another storage tier and replaced by a symlink -- as the
    cold-tier teacher archive does -- everything under it goes silently invisible: fewer
    results, no error, quietly incomplete numbers. Fixed-depth globs like
    ``*/*/*/ftseed*/checkpoints`` do traverse symlinks and need no such helper; only the
    recursive forms do.

    Cycles are impossible to rule out once symlinks are followed, so each directory is
    visited at most once by resolved identity.

    Args:
        root: Directory to search under.
        pattern: Same trailing-component pattern ``Path.rglob`` takes, e.g.
            ``"ftseed*/checkpoints"``.

    Returns:
        Sorted matching paths (both files and directories), as ``Path.rglob`` would return
        them if it followed symlinks.
    """
    root = Path(root)
    out: list[Path] = []
    seen: set[tuple[int, int]] = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        try:
            st = os.stat(dirpath)
        except OSError:
            dirnames[:] = []
            continue
        key = (st.st_dev, st.st_ino)
        if key in seen:  # already walked via another path -- prune to avoid a cycle
            dirnames[:] = []
            continue
        seen.add(key)
        here = Path(dirpath)
        for name in list(dirnames) + list(filenames):
            p = here / name
            if p.match(pattern):
                out.append(p)
    return sorted(out)


def file_sha256(path: str | Path) -> str:
    """Full hex sha256 of a file's bytes (chunked, so arbitrarily large files are fine).

    Args:
        path: File to hash.

    Returns:
        The 64-char lowercase hex digest. Call sites that store a truncated id slice it
        themselves (e.g. ``file_sha256(path)[:16]``).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Content ids of the same unchanged file are asked for repeatedly within one process
# (e.g. a `dataset_complete` DONE check followed by the `run()` skip gate re-hashing
# every mode's filtered.jsonl). Memoized on (resolved path, inode, size, mtime_ns):
# the inode covers atomic rewrites (temp + os.replace), whose mtime can otherwise land
# inside the clock's granularity at an unchanged size.
_SHA256_MEMO: dict[tuple[str, int, int, int], str] = {}


def file_sha256_cached(path: str | Path) -> str:
    """`file_sha256`, memoized in-process on (resolved path, inode, size, mtime_ns).

    Use where the same file may be hashed more than once per process and a stat call
    is an acceptable substitute for re-reading it. Prefer plain `file_sha256` when the
    file may be rewritten IN PLACE (same inode) at the same size within the mtime
    clock's granularity -- the memo would then serve a stale digest.

    Args:
        path: File to hash.

    Returns:
        The 64-char lowercase hex digest, same as `file_sha256`.
    """
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_ino, st.st_size, st.st_mtime_ns)
    if key not in _SHA256_MEMO:
        _SHA256_MEMO[key] = file_sha256(p)
    return _SHA256_MEMO[key]


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp in same dir + ``os.replace``).

    A crash can never leave a torn file: readers see either the old complete file or the new
    complete file. Parent directories are created if missing.

    Args:
        path: Destination file.
        text: Full file contents.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(
    path: str | Path,
    obj: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    trailing_newline: bool = True,
) -> None:
    """Serialize ``obj`` as JSON and write it atomically (see `atomic_write_text`).

    The serialization knobs default to the majority on-disk convention (``indent=2``, no key
    sort, trailing newline); they are explicit parameters so a call site that needs different
    bytes can request them and keep its existing artifacts byte-identical.

    Args:
        path: Destination file.
        obj: JSON-serializable object.
        indent: ``json.dumps`` indent.
        sort_keys: ``json.dumps`` key sorting.
        trailing_newline: Append a final ``"\\n"`` after the JSON body.
    """
    body = json.dumps(obj, indent=indent, sort_keys=sort_keys)
    atomic_write_text(path, body + "\n" if trailing_newline else body)


@contextlib.contextmanager
def file_lock(path: str | Path, mode: int = 0o666) -> Iterator[None]:
    """Hold an exclusive ``fcntl.flock`` on ``path`` for the duration of the block.

    Serializes a read-modify-write on a shared cache file across PROCESSES (many eval jobs
    on one node, or on both nodes: the output tree is NFS with ``local_lock=none``, where
    flock is carried by the server, verified cross-process on the real ``output_root``).
    The kernel drops the lock when the holding process dies, so a crash can never leave a
    stale lock behind -- no pid-file bookkeeping.

    The lock file is separate from the data file (which is replaced, not modified, by
    `atomic_write_text`) and is created world-writable per the shared-stash convention, so
    the other node's uid can lock it too.

    Args:
        path: Lock file (created if missing; its contents are never read or written).
        mode: Creation mode for the lock file.
    """
    fd = os.open(path, os.O_CREAT | os.O_RDWR, mode)
    try:
        with contextlib.suppress(OSError):  # another uid's file is not ours to chmod
            os.fchmod(fd, mode)
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
