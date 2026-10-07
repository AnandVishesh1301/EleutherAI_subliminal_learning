"""Clone an existing seed2seed distillation queue with a different student (or teacher) base.

The arms differ in exactly one factor — the receiver model — so the safest way to build
the new queue is to copy the template's records verbatim and rewrite only the base tokens
in ``cmd`` and in the ``key``. Every other field (env, mem gate, cwd, HP flags, seeds) is
preserved byte-for-byte, and the queue meta (priority, defaults) is inherited.

This writes queue inputs only; it never submits or runs a job.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
from collections import Counter
from pathlib import Path

from jobq.model import Job
from jobq import store
from subliminal_learning.models.registry import local_pythia_family
from subliminal_learning.jobq_folder import configured_queue_folder



def _arg_value(cmd: str, flag: str) -> str | None:
    args = shlex.split(cmd)
    if flag not in args:
        return None
    return args[args.index(flag) + 1]


_METACHARS = frozenset("|&;<>()$`\\\"' \t\n*?[#~=")


def _replace_arg(cmd: str, flag: str, value: str) -> str:
    """Splice ``value`` over the flag's value span in the RAW command string.

    Nothing is re-serialized: the result is ``cmd`` with exactly one substring
    replaced, so quoting, expansions, redirections and control operators survive
    byte-for-byte. ``shlex`` is used only to validate that the splice is safe;
    any ambiguity raises rather than rewriting the wrong bytes.
    """
    tokens = shlex.split(cmd)
    n_tokens = tokens.count(flag)
    if n_tokens != 1:
        raise SystemExit(
            f"expected exactly one {flag} token, found {n_tokens}: {cmd!r}"
        )
    pattern = rf"(?:(?<=\s)|\A){re.escape(flag)}[ \t]+(\S+)(?=\s|\Z)"
    matches = list(re.finditer(pattern, cmd))
    if len(matches) != 1:
        raise SystemExit(
            f"{flag} matches the raw command {len(matches)} times, expected one: {cmd!r}"
        )
    m = matches[0]
    raw = m.group(1)
    if set(raw) & _METACHARS or shlex.split(raw) != [raw]:
        raise SystemExit(
            f"value of {flag} is quoted or contains shell metacharacters: {cmd!r}"
        )
    if raw != tokens[tokens.index(flag) + 1]:
        raise SystemExit(
            f"raw value of {flag} disagrees with the parsed token: {cmd!r}"
        )
    return cmd[: m.start(1)] + value + cmd[m.end(1):]


def _short_label(base_dir: str) -> str:
    """Key-friendly short label for a numbers/ base-model dir name (wc1, dclm1, orig)."""
    if base_dir == "pythia-160m":
        return "orig"
    token = base_dir.removeprefix("pythia-160m-")
    variant, _, seed = token.partition("-seed")
    family = local_pythia_family(variant)
    return f"{family.short}{seed}" if family and seed else token


def _rewrite_key(key: str, old: str, new: str) -> str:
    segs = key.split("|")
    hits = [i for i, s in enumerate(segs) if s == old]
    if len(hits) != 1:
        raise SystemExit(
            f"key {key!r} has {len(hits)} segments equal to the base token {old!r}; "
            "expected exactly one"
        )
    segs[hits[0]] = new
    return "|".join(segs)


def rewrite_job(job: Job, *, student_base: str, teacher_base: str | None) -> Job:
    old_student = _arg_value(job.cmd, "--student-base")
    if old_student is None:
        raise SystemExit(f"job {job.key!r} has no --student-base")
    cmd = _replace_arg(job.cmd, "--student-base", student_base)
    key = _rewrite_key(job.key, old_student, student_base)
    if teacher_base is not None:
        old_teacher = _arg_value(job.cmd, "--teacher-base")
        if old_teacher is None:
            raise SystemExit(f"job {job.key!r} has no --teacher-base")
        cmd = _replace_arg(cmd, "--teacher-base", teacher_base)
        # The teacher segment is `from-<short label>`, which need not match any
        # mechanical transform of the template's dir name, so it is located by prefix.
        segs = key.split("|")
        hits = [i for i, s in enumerate(segs) if s.startswith("from-")]
        if len(hits) != 1:
            raise SystemExit(f"key {key!r} has {len(hits)} 'from-' segments; expected one")
        segs[hits[0]] = f"from-{_short_label(teacher_base)}"
        key = "|".join(segs)
    return Job(key=key, cmd=cmd, env=dict(job.env), mem_mib=job.mem_mib,
               slots=job.slots, cwd=job.cwd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template", required=True, help="Name of the queue to clone.")
    p.add_argument("--name", required=True, help="Name of the new queue.")
    p.add_argument("--student-base", required=True, help="New --student-base token.")
    p.add_argument("--teacher-base", help="New --teacher-base dir name (default: keep).")
    p.add_argument("--priority", type=int, help="Override the template's priority.")
    p.add_argument("--depends-on", help="Comma-separated queue names (default: none).")
    p.add_argument("--node", help="Pin the new queue to a hostname (default: the template's).")
    p.add_argument(
        "--root", type=Path, default=None,
        help="jobq queue root (default: queue_folder in jobq_paths.toml).",
    )
    p.add_argument("--dry-run", action="store_true", help="Print a summary, write nothing.")
    return p.parse_args()


def main() -> int:
    a = parse_args()
    root = a.root or configured_queue_folder()
    if not store.queue_exists(root, a.template):
        raise SystemExit(f"template queue {a.template!r} not found under {root}")
    if store.queue_exists(root, a.name):
        raise SystemExit(f"queue {a.name!r} already exists under {root}; refusing to overwrite")
    meta = store.read_meta(root, a.template)
    jobs = store.load_jobs(root, a.template)
    if not jobs:
        raise SystemExit(f"template queue {a.template!r} has no jobs")
    new_jobs = [rewrite_job(j, student_base=a.student_base, teacher_base=a.teacher_base)
                for j in jobs]

    # Validate BEFORE create_queue: append_jobs rejects duplicates only after meta.json
    # exists, which would leave an empty queue that a retry can no longer recreate.
    counts = Counter(j.jobkey for j in new_jobs)
    dupes = sorted({j.key for j in new_jobs if counts[j.jobkey] > 1})
    if dupes:
        raise SystemExit(
            "rewrite produced duplicate job keys (nothing written): " + ", ".join(dupes)
        )

    def bases(js: list[Job]) -> dict[str, list[str]]:
        return {
            flag.strip("-"): sorted({_arg_value(j.cmd, flag) or "(none)" for j in js})
            for flag in ("--student-base", "--teacher-base")
        }

    print(f"records: {len(new_jobs)}")
    print(f"before: {json.dumps(bases(jobs))}")
    print(f"after:  {json.dumps(bases(new_jobs))}")
    for j in new_jobs[:2]:
        print(json.dumps(j.to_dict()))
    if a.dry_run:
        print("dry-run: nothing written")
        return 0
    store.create_queue(
        root, a.name,
        depends_on=a.depends_on.split(",") if a.depends_on else [],
        node=meta.node if a.node is None else a.node,
        priority=meta.priority if a.priority is None else a.priority,
        strict_deps=meta.strict_deps,
        defaults=dict(meta.defaults),
    )
    store.append_jobs(root, a.name, new_jobs)
    print(f"wrote {len(new_jobs)} jobs to {store.queue_dir(root, a.name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
