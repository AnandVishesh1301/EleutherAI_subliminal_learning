"""The pure job model: the ``Job`` dataclass, filesystem-safe keys, jobs-file expansion.

No I/O and no GPU here (mirrors the repo's domain layer) so expansion is unit-testable.
A ``Job``'s ``key`` is the human-readable id; its ``jobkey`` is that key sanitized into a
single path segment used for claim dirs / result files / per-job logs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

# Anything that would fork a path or read as whitespace collapses to a single "__".
# Keep well under the 255-byte filename limit: logs prepend a stamp and append .<host>.log.
MAX_JOBKEY_LEN = 180

_UNSAFE = re.compile(r"[|/\\\s]+")


def sanitize_jobkey(key: str) -> str:
    """Collapse ``|``, ``/``, ``\\`` and whitespace runs in ``key`` into ``__``.

    Idempotent: sanitizing an already-sanitized key is a no-op. Used to turn a
    ``TAG|BASE|LR`` style key into one filesystem segment.
    """
    jk = _UNSAFE.sub("__", key.strip())
    if len(jk) > MAX_JOBKEY_LEN:
        # The jobkey becomes a filename (claims/<jobkey>, results/<jobkey>.json); ext4/CephFS
        # cap names at 255 bytes and a longer one raised OSError inside claim_next, which
        # killed worker threads (08-21: 28/48 + 28/40 threads lost across two pools).
        # Truncate + content hash keeps it unique and idempotent (the result is short).
        digest = hashlib.sha1(jk.encode()).hexdigest()[:12]
        jk = f"{jk[: MAX_JOBKEY_LEN - 13]}-{digest}"
    return jk


@dataclass(frozen=True)
class Job:
    """One unit of work. ``cmd`` runs via ``bash -c``; the rest are optional overrides.

    Queue-level ``defaults`` (from ``meta.json``) fill in ``env``/``mem_mib``/``cwd`` when a
    per-job value is absent — that merge happens in the worker, not here.
    """

    key: str
    cmd: str
    env: dict[str, str] = field(default_factory=dict)
    mem_mib: int | None = None
    slots: int | None = None
    cwd: str | None = None

    @property
    def jobkey(self) -> str:
        """This job's ``key`` sanitized into a single filesystem path segment."""
        return sanitize_jobkey(self.key)

    def to_dict(self) -> dict:
        """Serialize for one ``jobs.jsonl`` line (drops empty/None optionals)."""
        d: dict = {"key": self.key, "cmd": self.cmd}
        if self.env:
            d["env"] = dict(self.env)
        if self.mem_mib is not None:
            d["mem_mib"] = self.mem_mib
        if self.slots is not None:
            d["slots"] = self.slots
        if self.cwd is not None:
            d["cwd"] = self.cwd
        return d

    @classmethod
    def from_dict(cls, d: dict, *, default_key: str | None = None) -> "Job":
        """Rebuild a ``Job`` from a ``jobs.jsonl`` / JSONL-submit dict.

        ``key`` falls back to ``default_key`` (the raw submit line) when absent.
        """
        key = d.get("key") or default_key
        if not key:
            raise ValueError(f"job dict has no 'key' and no default: {d!r}")
        if "cmd" not in d:
            raise ValueError(f"job {key!r} has no 'cmd'")
        return cls(
            key=str(key),
            cmd=str(d["cmd"]),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            mem_mib=None if d.get("mem_mib") is None else int(d["mem_mib"]),
            slots=None if d.get("slots") is None else int(d["slots"]),
            cwd=None if d.get("cwd") is None else str(d["cwd"]),
        )


def parse_jobs_file(text: str, *, template: str | None = None) -> list[Job]:
    """Expand a jobs-file into ``Job``s.

    Each non-blank, non-``#`` line is either:

    - **JSONL** (starts with ``{``): parsed as a job dict; ``key`` defaults to the raw line.
    - **plain** (needs ``template``): the whole line becomes the ``key`` and the template is
      substituted with ``{line}`` = the whole line and ``{0},{1},...`` = the ``|``-split
      fields (e.g. ``TAG|BASE|LR|R|A|EP|SEED``) to build ``cmd``.

    Raises ``ValueError`` if a plain line appears without a ``template``.
    """
    jobs: list[Job] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            jobs.append(Job.from_dict(json.loads(line), default_key=line))
            continue
        if template is None:
            raise ValueError(
                f"plain jobs-file line needs --template to build cmd: {line!r}"
            )
        fields = line.split("|")
        try:
            cmd = template.format(*fields, line=line)
        except (IndexError, KeyError) as exc:
            raise ValueError(
                f"template placeholder {exc} not satisfied by jobs-file line with "
                f"{len(fields)} '|'-field(s): {line!r} (valid: {{line}}, {{0}}..{{{len(fields) - 1}}})"
            ) from exc
        jobs.append(Job(key=line, cmd=cmd))
    return jobs
