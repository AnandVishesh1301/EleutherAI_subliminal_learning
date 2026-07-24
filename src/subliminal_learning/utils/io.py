"""Shared filesystem helpers: content hashing and crash-safe atomic writes.

These consolidate copies that were duplicated across the pipelines and scripts. Resume/skip
gates in this repo hash the bytes of written files, so the write helpers keep their exact
on-disk byte layout (JSON ``indent``/``sort_keys``/trailing newline are explicit parameters).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


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
