"""Guard that jobq job commands run on every node of the shared jobq root.

jobq places a job in its queue's cwd and falls back to the node's own checkout when that is
unreachable (goncalo-temp-1 runs as another uid and cannot read /home/angela). A `cd` in the
command text, or any absolute /home/angela/ path, defeats that fallback (2026-10-09: ~4,850
failed jobs). Generators call `assert_portable_jobs` before writing a jobs file; paths go in
repo-relative (resolved against the job cwd) or as the shared absolute /mnt/... form.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from pathlib import Path

_HOME = "/home/angela/"
# `cd` as a command word: at the start, or after `&&`, `||`, `;`, `|` or `(`.
_CD = re.compile(r"(?:^|&&|\|\||;|\||\()\s*cd(?:\s|$)")


def portable_cmd_problems(cmd: str) -> list[str]:
    """Reasons `cmd` is not portable across nodes (empty list = portable)."""
    problems = []
    if _CD.search(cmd):
        problems.append("contains a `cd` (jobs must run in the queue/fallback cwd)")
    if _HOME in cmd:
        problems.append(f"contains an absolute {_HOME} path (unreadable on goncalo-temp-1)")
    return problems


def assert_portable_cmd(key: str, cmd: str) -> None:
    """Raise ValueError naming job `key` unless `cmd` is portable."""
    problems = portable_cmd_problems(cmd)
    if problems:
        raise ValueError(f"job {key!r}: non-portable jobq cmd ({'; '.join(problems)}): {cmd}")


def assert_portable_jobs(jobs: Iterable[dict]) -> None:
    """`assert_portable_cmd` over job dicts ({'key', 'cmd', ...}) before a jobs file is written."""
    for jd in jobs:
        assert_portable_cmd(str(jd.get("key")), str(jd["cmd"]))


def assert_portable_text(label: str, text: str) -> None:
    """Raise ValueError unless a config written for jobs (`label`) is free of /home/angela/ paths."""
    if _HOME in text:
        raise ValueError(f"{label}: contains an absolute {_HOME} path (unreadable on goncalo-temp-1)")


def portable_path(path: Path | str, repo_root: Path | str) -> str:
    """`path` as a repo-relative string when it lies under `repo_root` (else unchanged).

    Lexical (no `resolve()`): run_logs is a symlink, and resolving it elsewhere would bake one
    checkout's absolute form into the job.
    """
    p, root = os.path.abspath(path), os.path.abspath(repo_root)
    if p == root or p.startswith(root + os.sep):
        return os.path.relpath(p, root)
    return str(path)
