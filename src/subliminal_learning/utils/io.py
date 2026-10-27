"""Shared filesystem helpers: content hashing and crash-safe atomic writes.

These consolidate copies that were duplicated across the pipelines and scripts. Resume/skip
gates in this repo hash the bytes of written files, so the write helpers keep their exact
on-disk byte layout (JSON ``indent``/``sort_keys``/trailing newline are explicit parameters).
"""

from __future__ import annotations

import contextlib
import csv
import fcntl
import hashlib
import itertools
import json
import os
import threading
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


def file_sha256(path: str | Path, *, missing_ok: bool = False) -> str | None:
    """Full hex sha256 of a file's bytes (chunked, so arbitrarily large files are fine).

    The one implementation behind every former per-module copy (`run_io.sha256_file`, the
    coherence pools' ``_sha256_file``, the scripts' ``_sha256`` / ``sha``).

    Args:
        path: File to hash.
        missing_ok: Return ``None`` instead of raising when `path` does not exist (the
            `run_io.sha256_file` contract).

    Returns:
        The 64-char lowercase hex digest (or ``None``, see `missing_ok`). Call sites that
        store a truncated id slice it themselves (e.g. ``file_sha256(path)[:16]``).
    """
    if missing_ok and not Path(path).exists():
        return None
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


_WRITER_SEQ = itertools.count()


def _writer_token() -> str:
    """A temp-file suffix unique to (process, thread, call).

    The pid alone is not: two threads of one process writing the SAME destination -- a
    threaded scan flushing its report on two flush boundaries at once -- would pick the
    same temp name, and the second ``os.replace`` would fail with FileNotFoundError after
    the first one renamed the shared file away. The `{name}.tmp.` PREFIX is kept exactly as
    it was: `clean_partial_runs` finds orphaned temps by the `*.tmp.*` glob.
    """
    return f"{os.getpid()}.{threading.get_ident()}.{next(_WRITER_SEQ)}"


def atomic_write_text(
    path: str | Path, text: str, *, fsync: bool = True, shared_perms: bool = True
) -> None:
    """Write ``text`` to ``path`` atomically (temp in same dir + ``os.replace``).

    A crash can never leave a torn file: readers see either the old complete file or the new
    complete file. Parent directories are created if missing. The temp file is named per
    (pid, thread, call), so concurrent writers of one path never share it.

    The file is made world-readable/writable before the replace, per the shared-stash
    convention `file_lock` follows: only the pool sets ``umask 0``, so a CLI run under a
    restrictive umask would otherwise write queue state (``jobs.jsonl``, ``meta.json``,
    sidecars) that the other node's uid cannot read.

    Args:
        path: Destination file.
        text: Full file contents.
        fsync: fsync the temp file before, and the directory after, the replace.
        shared_perms: chmod the temp file ``0o666`` before the replace; ``False`` leaves the
            umask-derived mode (the `run_io.write_json_atomic` contract).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{_writer_token()}")
    # fsync before the replace: `write_text` alone only reaches the page cache, so a node
    # crash (or an OOM reboot) right after the rename can leave the new NAME pointing at
    # zero-length data -- the durable version of the torn file this function exists to
    # prevent. The directory fsync then persists the rename itself; it is best-effort,
    # since not every filesystem allows opening a directory for fsync.
    with open(tmp, "w") as fh:
        fh.write(text)
        if fsync:
            fh.flush()
            os.fsync(fh.fileno())
    if shared_perms:
        with contextlib.suppress(OSError):  # another uid's file is not ours to chmod
            os.chmod(tmp, 0o666)
    os.replace(tmp, path)
    if not fsync:
        return
    with contextlib.suppress(OSError):
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def atomic_write_json(
    path: str | Path,
    obj: object,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    trailing_newline: bool = True,
    fsync: bool = True,
    shared_perms: bool = True,
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
        fsync: Passed to `atomic_write_text`.
        shared_perms: Passed to `atomic_write_text`.
    """
    body = json.dumps(obj, indent=indent, sort_keys=sort_keys)
    atomic_write_text(
        path,
        body + "\n" if trailing_newline else body,
        fsync=fsync,
        shared_perms=shared_perms,
    )


def read_json(path: str | Path):
    """Parse a JSON file -- ``json.loads(Path(path).read_text())``.

    Raises whatever `Path.read_text` / `json.loads` raise (missing file, bad JSON); use
    `safe_read_json` in a scan loop where those are expected.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON (typically a dict).
    """
    return json.loads(Path(path).read_text())


def safe_read_json(
    path: str | Path,
    *,
    errors: tuple[type[BaseException], ...] = (OSError, json.JSONDecodeError),
):
    """`read_json` that swallows read/parse errors, returning ``None`` instead.

    The scan-loop guard: a still-draining campaign leaves half-written or absent JSON, and a
    directory walk should skip those rather than crash. Only `errors` are caught -- by default
    ``OSError`` (missing/unreadable file) and ``json.JSONDecodeError`` (partial write); pass
    ``(OSError, ValueError)`` to also swallow undecodable bytes. Anything else propagates.

    Args:
        path: Path to the JSON file.
        errors: Exception types mapped to ``None``.

    Returns:
        The parsed JSON, or ``None`` when reading/parsing raised one of `errors`.
    """
    try:
        return json.loads(Path(path).read_text())
    except errors:
        return None


def read_jsonl(path: str | Path) -> list[dict]:
    """Parse a JSON-lines file into a list of objects (blank lines skipped), in file order."""
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """Write `rows` with ``csv.DictWriter``; the header is ``rows[0]``'s keys.

    Not atomic. An empty `rows` raises ``IndexError`` (no header to take).
    """
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


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
