"""Code fingerprint of one judge: binds judgments to the code that produced them.

The fingerprint is the sha256 of the judge-side files that define what a judge sees
and how its answer is scored (`FINGERPRINT_FILES`: prompts / schemas / parsers, the
scoring math, renderers, request building, sampling parameters and engine settings),
the judge key and the model revision. Outputs live under
``<out_dir>/<judge>/<fingerprint[:12]>/``, so judgments made by other code are never
read as current. The main repo recomputes the same digest from the same files
(`subliminal_learning.judging.s2_workitems.judge_fingerprint`).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

FINGERPRINT_FILES: tuple[str, ...] = (
    "backends.py",
    "judges.py",
    "models.py",
    "render.py",
    "scoring.py",
    "tasks.py",
)
_DIR = Path(__file__).resolve().parent


def files_digest(directory: Path = _DIR) -> str:
    """sha256 over (name, sha256(bytes)) of `FINGERPRINT_FILES` in `directory`."""
    h = hashlib.sha256()
    for name in FINGERPRINT_FILES:
        h.update(name.encode())
        h.update(hashlib.sha256((directory / name).read_bytes()).hexdigest().encode())
    return h.hexdigest()


def judge_fingerprint(key: str, revision: str, directory: Path = _DIR) -> str:
    """Fingerprint of judge `key` at model `revision` under the code in `directory`."""
    return hashlib.sha256(
        f"{files_digest(directory)}|{key}|{revision}".encode()
    ).hexdigest()
