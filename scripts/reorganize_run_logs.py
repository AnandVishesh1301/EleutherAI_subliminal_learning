"""CLI: one-time, undoable regrouping of the flat shared ``run_logs/`` dir into subfolders.

Target layout (``README.md``, ``archive/`` and ``jspace_jobq/`` stay put; ``jspace_jobq`` is
another project's live queue root and is never read for rewrite nor modified)::

    queues/queue_angela/        (+ compat symlink run_logs/queue_angela -> queues/queue_angela)
    queues/retired/{queue,queue0,queue_pilecoh,queue_mps,queue_angela_removed}
    campaigns/<dir>             every other top-level directory
    workers/<file>              *.py *.sh
    registry/<file>             *.json *.jsonl *.txt *.csv *.yaml
    logs/<file>                 *.log *.out *.pid

A ``code/`` directory inside a campaign is a frozen snapshot of repo code (provenance) and is
never rewritten, depth-bumped or checked.

Moves are ``os.rename`` (same filesystem) recorded in ``archive/MANIFEST.tsv``; every rewritten
file is backed up under ``archive/reorg_<stamp>/backup/`` first so ``undo`` is a byte-exact
restore. Paths under run_logs are never ``resolve()``d (run_logs is a symlink).

Subcommands:

    plan                    classify top-level entries, write archive/reorg_<stamp>/mapping.json.
    check [--preview MAP]   verify every literal ``run_logs/<path>`` reference resolves.
    apply MAP [--yes]       move, rewrite references, bump script depth, write report.md.
    undo MAP                reverse apply (moves + byte-exact restore of rewritten files).

Examples::

    uv run python scripts/reorganize_run_logs.py plan
    uv run python scripts/reorganize_run_logs.py check --preview run_logs/archive/reorg_<stamp>/mapping.json
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)

# NO resolve() -- run_logs is a symlink; abspath keeps it unresolved.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = ("queues", "campaigns", "workers", "registry", "logs")
LIVE_ROOT = "queue_angela"
RETIRED_ROOTS = ("queue", "queue0", "queue_pilecoh", "queue_mps", "queue_angela_removed")
UNTOUCHED = ("README.md", "archive", "jspace_jobq")
EXT_GROUP = {
    ".py": "workers", ".sh": "workers",
    ".json": "registry", ".jsonl": "registry", ".txt": "registry", ".csv": "registry",
    ".yaml": "registry",
    ".log": "logs", ".out": "logs", ".pid": "logs",
}
TEXT_EXTS = {".py", ".sh", ".yaml", ".yml", ".toml", ".json", ".jsonl", ".txt", ".md"}
SELF_FILES = ("scripts/reorganize_run_logs.py", "tests/test_reorganize_run_logs.py")
HEARTBEAT_STALE_S = 300

REF_RE = re.compile(r"run_logs/([A-Za-z0-9_./\-]+)")
TEMPLATE_CHARS = set("{$%*")
NAME_TAIL = r"(?![A-Za-z0-9_.\-])"

RunLogsOpt = typer.Option(None, "--run-logs", help="run_logs dir (default <repo>/run_logs).")
RepoOpt = typer.Option(None, "--repo", help="Repo checkout (default: this script's repo).")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _paths(run_logs: Path | None, repo: Path | None) -> tuple[str, str]:
    rp = os.path.abspath(str(repo)) if repo else REPO
    rl = os.path.abspath(str(run_logs)) if run_logs else os.path.join(rp, "run_logs")
    return rl, rp


def _mkdir_shared(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        os.chmod(path, 0o777)  # shared uids; makedirs mode is masked by umask


# ---------------------------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------------------------


def classify(rl: str) -> tuple[dict[str, str], list[str]]:
    """Return ``({old_name: new_relpath}, unknown_names)`` for every top-level entry."""
    mapping: dict[str, str] = {}
    unknown: list[str] = []
    for name in sorted(os.listdir(rl)):
        full = os.path.join(rl, name)
        if name in UNTOUCHED or name in GROUPS:
            continue
        if name == LIVE_ROOT:
            if not os.path.islink(full):
                mapping[name] = f"queues/{name}"
            continue
        if name in RETIRED_ROOTS:
            mapping[name] = f"queues/retired/{name}"
        elif os.path.isdir(full) and not os.path.islink(full):
            mapping[name] = f"campaigns/{name}"
        elif os.path.splitext(name)[1] in EXT_GROUP:
            mapping[name] = f"{EXT_GROUP[os.path.splitext(name)[1]]}/{name}"
        else:
            unknown.append(name)
    return mapping, unknown


def load_mapping(path: Path) -> tuple[dict[str, str], str]:
    """Load ``mapping.json``; the stamp is taken from its ``reorg_<stamp>`` parent dir."""
    mapping = json.loads(Path(path).read_text())
    parent = os.path.basename(os.path.dirname(os.path.abspath(str(path))))
    if not parent.startswith("reorg_"):
        raise typer.BadParameter(f"mapping must live in archive/reorg_<stamp>/: {path}")
    return mapping, parent[len("reorg_"):]


def _stamp_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------------------------
# rewrite rules
# ---------------------------------------------------------------------------------------------


DOTTED_DEFAULT = True  # tests flip this to simulate the pre-dotted apply
_FROM_IMPORT_RE = re.compile(
    r"^([ \t]*from[ \t]+run_logs[ \t]+import[ \t]+)([^\n#]*?)([ \t]*(?:#.*)?)$", re.M
)


class Rewriter:
    """Rewrite references to moved top-level names.

    Slash rule: literal ``run_logs/<old>`` -> ``run_logs/<new>`` with a name boundary.
    Dotted rule: ``run_logs.<stem>`` (module path; <stem> is a mapped dir or ``.py`` stem that is
    a valid identifier) -> ``run_logs.<group>.<stem>``, and ``from run_logs import <stems>`` ->
    ``from run_logs.<group> import <stems>`` when every name maps to the same group.
    ``count`` accumulates the number of occurrences rewritten.
    """

    def __init__(self, mapping: dict[str, str], slash: bool = True, dotted: bool | None = None):
        self.mapping = dict(mapping)
        self.mapping.setdefault(LIVE_ROOT, f"queues/{LIVE_ROOT}")
        self.slash = slash
        self.dotted = DOTTED_DEFAULT if dotted is None else dotted
        self.count = 0
        names = sorted(self.mapping, key=len, reverse=True)
        alt = "|".join(re.escape(n) for n in names)
        self.re = re.compile(rf"run_logs/({alt}){NAME_TAIL}")
        self.bare_re = re.compile(rf"(['\"])({alt})\1")
        # module stem -> dotted group ("campaigns", "workers", "queues.retired", ...)
        self.mod_group: dict[str, str] = {}
        for old, new in self.mapping.items():
            if old.endswith(".py"):
                stem = old[:-3]
            elif new.startswith(("campaigns/", "queues/")):  # directories
                stem = old
            else:
                continue
            if stem.isidentifier():  # hyphens / dots cannot be modules
                self.mod_group[stem] = os.path.dirname(new).replace("/", ".")
        malt = "|".join(re.escape(n) for n in sorted(self.mod_group, key=len, reverse=True))
        self.dot_re = (
            re.compile(rf"(?<![A-Za-z0-9_/])run_logs\.({malt})(?![A-Za-z0-9_])") if malt else None
        )

    def _from_import(self, m: re.Match) -> str:
        names = [n.strip().split()[0] for n in m.group(2).strip("() ").split(",") if n.strip()]
        groups = {self.mod_group.get(n) for n in names}
        if len(groups) != 1 or None in groups:
            return m.group(0)
        self.count += 1
        head = m.group(1).replace("run_logs", f"run_logs.{groups.pop()}", 1)
        return head + m.group(2) + m.group(3)

    def unmapped_from_imports(self, text: str) -> list[str]:
        """``from run_logs import ...`` lines the dotted rule leaves alone."""
        out = []
        for m in _FROM_IMPORT_RE.finditer(text):
            names = [n.strip().split()[0] for n in m.group(2).strip("() ").split(",") if n.strip()]
            groups = {self.mod_group.get(n) for n in names}
            if len(groups) != 1 or None in groups:
                out.append(m.group(0).strip())
        return out

    def _dot(self, m: re.Match) -> str:
        self.count += 1
        return f"run_logs.{self.mod_group[m.group(1)]}.{m.group(1)}"

    def _slash(self, m: re.Match) -> str:
        self.count += 1
        return "run_logs/" + self.mapping[m.group(1)]

    def sub(self, text: str) -> str:
        if "run_logs" not in text:
            return text
        if self.slash and "run_logs/" in text:
            text = self.re.sub(self._slash, text)
        if self.dotted and self.mod_group:
            if self.dot_re is not None and "run_logs." in text:
                text = self.dot_re.sub(self._dot, text)
            if "import" in text:
                text = _FROM_IMPORT_RE.sub(self._from_import, text)
        return text


# ---- depth bump --------------------------------------------------------------------------------

_BASE = r"Path\(__file__\)(?:\.resolve\(\)|\.absolute\(\))?"
_FILE_PARENTS_RE = re.compile(rf"({_BASE})\.parents\[(\d+)\]")
_FILE_CHAIN_RE = re.compile(rf"({_BASE})((?:\.parent)+)(?![A-Za-z0-9_\[])")
_ASSIGN_RE = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*(?::[^=]+)?=\s*Path\(__file__\)"
    r"((?:\.resolve\(\)|\.absolute\(\)|\.parent)*)\s*(?:#.*)?$"
)


def _bump_parents(m: re.Match, depth0: int) -> str:
    n = int(m.group(2))
    return f"{m.group(1)}.parents[{n + 1}]" if depth0 + n + 1 >= 2 else m.group(0)


def _bump_chain(m: re.Match, depth0: int) -> str:
    k = m.group(2).count(".parent")
    return m.group(1) + m.group(2) + (".parent" if depth0 + k >= 2 else "")


def bump_depth(text: str) -> tuple[str, int, list[tuple[int, str]]]:
    """Bump ``__file__``-relative ancestors by one level (the file moved one dir deeper).

    Anything reaching above the script's own directory gets one more level; the script dir
    itself (``.parent`` / ``parents[0]`` of the file) is unchanged. Variables assigned the file
    or its directory (``NAME = Path(__file__).parent``) are tracked and their ancestor uses
    bumped too. Returns ``(new_text, n_lines_changed, unrewritten [(lineno, line)])``.
    """
    lines = text.splitlines(keepends=True)
    tracked: dict[str, int] = {}
    for line in lines:
        m = _ASSIGN_RE.match(line)
        if m and m.group(2).count(".parent") <= 1:
            tracked[m.group(1)] = m.group(2).count(".parent")
    var_res = [
        (
            re.compile(rf"(?<![\w.])({re.escape(v)})\.parents\[(\d+)\]"),
            re.compile(rf"(?<![\w.])({re.escape(v)})((?:\.parent)+)(?![A-Za-z0-9_\[])"),
            d0,
        )
        for v, d0 in tracked.items()
    ]
    out: list[str] = []
    changed = 0
    unrewritten: list[tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        new = line
        if "__file__" in new:
            new = _FILE_PARENTS_RE.sub(lambda m: _bump_parents(m, 0), new)
            new = _FILE_CHAIN_RE.sub(lambda m: _bump_chain(m, 0), new)
        if not _ASSIGN_RE.match(line):
            for pre, chain, d0 in var_res:
                new = pre.sub(lambda m, d0=d0: _bump_parents(m, d0), new)
                new = chain.sub(lambda m, d0=d0: _bump_chain(m, d0), new)
        if new != line:
            changed += 1
        elif "parents[" in line or ".parent.parent" in line:
            unrewritten.append((i, line.rstrip("\n")))
        out.append(new)
    return "".join(out), changed, unrewritten


# ---------------------------------------------------------------------------------------------
# source discovery
# ---------------------------------------------------------------------------------------------


def queue_folders(rl: str) -> list[str]:
    """Queue folders at old or new locations (never jspace_jobq, never the compat symlink)."""
    cands = [os.path.join(rl, n) for n in (LIVE_ROOT, *RETIRED_ROOTS)]
    cands.append(os.path.join(rl, "queues", LIVE_ROOT))
    cands += [os.path.join(rl, "queues", "retired", n) for n in RETIRED_ROOTS]
    return [c for c in cands if os.path.isdir(c) and not os.path.islink(c)]


def jobs_files(rl: str) -> list[str]:
    out = []
    for root in queue_folders(rl):
        with os.scandir(root) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    jf = os.path.join(e.path, "jobs.jsonl")
                    if os.path.isfile(jf):
                        out.append(jf)
    return sorted(out)


def run_logs_text_files(rl: str) -> list[str]:
    skip_top = {"archive", "jspace_jobq", "queues", LIVE_ROOT, *RETIRED_ROOTS}
    out = []
    for name in os.listdir(rl):
        full = os.path.join(rl, name)
        if name in skip_top or os.path.islink(full):
            continue
        if os.path.isfile(full):
            if os.path.splitext(name)[1] in TEXT_EXTS:
                out.append(full)
            continue
        for d, dirs, files in os.walk(full):
            # `code/` under a campaign is a frozen snapshot of repo code (provenance): never
            # rewritten, bumped or checked.
            dirs[:] = sorted(x for x in dirs if x != "code")
            out += [
                os.path.join(d, f) for f in sorted(files)
                if os.path.splitext(f)[1] in TEXT_EXTS and not os.path.islink(os.path.join(d, f))
            ]
    return sorted(out)


def repo_text_files(rp: str) -> list[str]:
    res = subprocess.run(
        ["git", "-C", rp, "ls-files", "-z"], capture_output=True, check=True, text=True
    )
    out = []
    for rel in res.stdout.split("\0"):
        if not rel or rel == "run_logs" or rel.startswith("run_logs/"):
            continue
        if rel.endswith(".ipynb") or rel.startswith("notebooks/figures/") or rel in SELF_FILES:
            continue
        full = os.path.join(rp, rel)
        if os.path.isfile(full) and not os.path.islink(full):
            out.append(full)
    return out


def read_text(path: str) -> str | None:
    """File contents, or None if binary/undecodable/unreadable."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError:
        return None
    if b"\0" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def job_cmds(text: str) -> list[str]:
    cmds = []
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("cmd"), str):
            cmds.append(obj["cmd"])
    return cmds


def rewrite_jobs(text: str, rw: Rewriter) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        try:
            obj = json.loads(body)
        except ValueError:
            out.append(line)
            continue
        if not (isinstance(obj, dict) and isinstance(obj.get("cmd"), str)):
            out.append(line)
            continue
        new_cmd = rw.sub(obj["cmd"])
        if new_cmd == obj["cmd"]:
            out.append(line)
            continue
        obj["cmd"] = new_cmd
        out.append(json.dumps(obj) + line[len(body):])
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------------------------


def extract_refs(text: str) -> tuple[list[str], list[str]]:
    """Return ``(literal paths, templated literals)`` for every ``run_logs/<path>``."""
    refs, templated = [], []
    for m in REF_RE.finditer(text):
        lit = m.group(1)
        nxt = text[m.end():m.end() + 1]
        if nxt and nxt in TEMPLATE_CHARS:
            templated.append(lit + nxt)
            continue
        lit = lit.rstrip("./")
        if lit:
            refs.append(lit)
    return refs, templated


def _exists_preview(rl: str, lit: str, mapping: dict[str, str], rw: Rewriter) -> bool:
    """Predict whether ``run_logs/<lit>`` exists after apply (rewrite literal, map back)."""
    new = rw.sub("run_logs/" + lit)[len("run_logs/"):]
    for old, dst in mapping.items():
        if new == dst or new.startswith(dst + "/"):
            return os.path.lexists(os.path.join(rl, old + new[len(dst):]))
    first = new.split("/", 1)[0]
    if new in (*GROUPS, "queues/retired"):  # created by apply
        return True
    if first == LIVE_ROOT:  # compat symlink
        return os.path.lexists(os.path.join(rl, new))
    if first in mapping:
        return False  # moved but literal not rewritten
    return os.path.lexists(os.path.join(rl, new))


@dataclass
class CheckResult:
    missing: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    n_refs: dict[str, int] = field(default_factory=dict)
    templated: list[tuple[str, str]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return bool(self.missing.get("a")) or bool(self.missing.get("b"))


CLASS_NAMES = {"a": "queue jobs.jsonl cmd", "b": "run_logs text files", "c": "repo files"}


def _sources(rl: str, rp: str) -> dict[str, list[str]]:
    return {"a": jobs_files(rl), "b": run_logs_text_files(rl), "c": repo_text_files(rp)}


def run_check(rl: str, rp: str, mapping: dict[str, str] | None) -> CheckResult:
    rw = Rewriter(mapping) if mapping is not None else None
    res = CheckResult()
    for cls, files in _sources(rl, rp).items():
        missing: dict[str, list[str]] = defaultdict(list)
        n = 0
        for path in files:
            text = read_text(path)
            if text is None:
                continue
            texts = job_cmds(text) if cls == "a" else [text]
            for t in texts:
                refs, templ = extract_refs(t)
                res.templated += [(path, x) for x in templ]
                for lit in refs:
                    n += 1
                    ok = (
                        _exists_preview(rl, lit, mapping, rw) if rw is not None
                        else os.path.lexists(os.path.join(rl, lit))
                    )
                    if not ok:
                        missing[lit].append(path)
        res.missing[cls] = dict(missing)
        res.n_refs[cls] = n
    return res


def format_check(res: CheckResult, rl: str, rp: str) -> str:
    lines = []
    for cls in ("a", "b", "c"):
        miss = res.missing.get(cls, {})
        occ = sum(len(v) for v in miss.values())
        tag = "FAIL" if miss and cls in "ab" else ("report" if miss else "ok")
        lines.append(
            f"[{cls}] {CLASS_NAMES[cls]}: {res.n_refs.get(cls, 0)} refs, "
            f"{len(miss)} distinct missing paths ({occ} occurrences) -- {tag}"
        )
        for lit in sorted(miss)[:20]:
            src = miss[lit][0]
            src = os.path.relpath(src, rl if cls != "c" else rp)
            lines.append(f"    run_logs/{lit}   (e.g. {src}, x{len(miss[lit])})")
    lines.append(f"templated literals skipped: {len(res.templated)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# apply helpers
# ---------------------------------------------------------------------------------------------


def atomic_write(path: str, text: str) -> None:
    st = os.stat(path)
    d, base = os.path.split(path)
    tmp = os.path.join(d, f".{base}.reorg.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.chmod(tmp, st.st_mode & 0o7777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.lexists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def _backup_path(reorg_dir: str, rl: str, rp: str, path: str) -> str:
    if path.startswith(rl + os.sep):
        return os.path.join(reorg_dir, "backup", "run_logs", os.path.relpath(path, rl))
    return os.path.join(reorg_dir, "backup", "repo", os.path.relpath(path, rp))


def _manifest_row(action: str, family: str, src: str, dst: str, full: str) -> str:
    st = os.lstat(full)
    mtime = datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()
    return f"{_now().isoformat()}\t{action}\t{family}\t{src}\t{dst}\t{st.st_size}\t{mtime}\n"


def _append_manifest(rl: str, rows: list[str]) -> None:
    if rows:
        with open(os.path.join(rl, "archive", "MANIFEST.tsv"), "a") as f:
            f.writelines(rows)


def preflight(rl: str) -> tuple[list[str], list[str]]:
    """Return ``(refusals, warnings)`` for apply."""
    refusals, warnings = [], []
    root = os.path.join(rl, LIVE_ROOT)
    if not os.path.isdir(root):
        return refusals, warnings
    n_claims = 0
    with os.scandir(root) as it:
        for e in it:
            cdir = os.path.join(e.path, "claims")
            if e.is_dir() and os.path.isdir(cdir):
                n_claims += len(os.listdir(cdir))
    if n_claims:
        refusals.append(f"{n_claims} claim file(s) under {LIVE_ROOT}/*/claims (jobs running)")
    host = socket.gethostname()
    now = _now()
    for name in sorted(os.listdir(root)):
        if not (name.startswith("worker.") and name.endswith(".json")):
            continue
        h = name[len("worker."):-len(".json")]
        full = os.path.join(root, name)
        try:
            hb = datetime.fromisoformat(json.loads(Path(full).read_text())["heartbeat_utc"])
        except (OSError, ValueError, KeyError, TypeError):
            hb = datetime.fromtimestamp(os.path.getmtime(full), timezone.utc)
        age = (now - hb).total_seconds()
        if age < HEARTBEAT_STALE_S:
            msg = f"pool heartbeat {name} is {age:.0f}s old"
            (refusals if h == host else warnings).append(msg)
    return refusals, warnings


@dataclass
class ApplyStats:
    moved: int = 0
    skipped_moves: list[str] = field(default_factory=list)
    rewritten: dict[str, int] = field(default_factory=lambda: {"a": 0, "b": 0, "c": 0})
    bumped_files: int = 0
    bumped_lines: int = 0
    perm_errors: list[str] = field(default_factory=list)
    jobq_paths: bool = False


def do_moves(rl: str, mapping: dict[str, str], st: ApplyStats) -> None:
    for g in GROUPS:
        _mkdir_shared(os.path.join(rl, g))
    _mkdir_shared(os.path.join(rl, "queues", "retired"))
    rows = []
    try:
        for old, new in sorted(mapping.items()):
            src, dst = os.path.join(rl, old), os.path.join(rl, new)
            src_gone = not os.path.lexists(src) or (old == LIVE_ROOT and os.path.islink(src))
            if src_gone:
                if not os.path.lexists(dst):
                    st.skipped_moves.append(f"{old}: src and dst both missing")
                continue
            if os.path.lexists(dst):
                st.skipped_moves.append(f"{old}: dst {new} already exists, src kept")
                continue
            os.rename(src, dst)
            rows.append(_manifest_row("move", new.split("/", 1)[0], old, new, dst))
            st.moved += 1
    finally:
        _append_manifest(rl, rows)
    link = os.path.join(rl, LIVE_ROOT)
    if not os.path.lexists(link) and os.path.isdir(os.path.join(rl, "queues", LIVE_ROOT)):
        os.symlink(f"queues/{LIVE_ROOT}", link)


def _is_moved_script(rl: str, path: str) -> bool:
    rel = os.path.relpath(path, rl)
    return rel.split(os.sep, 1)[0] in ("workers", "campaigns") and rel.endswith((".py", ".sh"))


def do_rewrites(rl: str, rp: str, mapping: dict[str, str], reorg_dir: str,
                st: ApplyStats) -> list[tuple[str, int, str]]:
    """Literal rewrite (a/b/c) + depth bump; returns .py lines left for manual review."""
    rw = Rewriter(mapping)
    unrewritten: list[tuple[str, int, str]] = []
    for cls, files in _sources(rl, rp).items():
        for path in files:
            text = read_text(path)
            if text is None:
                continue
            bak = _backup_path(reorg_dir, rl, rp, path)
            already = os.path.exists(bak)
            if cls == "a":
                new = rewrite_jobs(text, rw)
            else:
                new = rw.sub(text)
            if cls == "b" and path.endswith(".py") and _is_moved_script(rl, path):
                bumped, n, unrw = bump_depth(new)
                unrewritten += [(path, i, ln) for i, ln in unrw]
                if not already and n:
                    new = bumped
                    st.bumped_files += 1
                    st.bumped_lines += n
            if new == text:
                continue
            try:
                if not already:
                    os.makedirs(os.path.dirname(bak), exist_ok=True)
                    shutil.copy2(path, bak)
                atomic_write(path, new)
                st.rewritten[cls] += 1
            except PermissionError as e:
                st.perm_errors.append(f"{path}: {e}")
    return unrewritten


def rewrite_modules_files(rl: str, rp: str, mapping: dict[str, str], reorg_dir: str
                          ) -> tuple[dict[str, int], dict[str, int], list[str], list[str]]:
    """Dotted-module rewrite only over classes a/b/c.

    Returns ``(files_changed, occurrences, left_alone_from_imports, perm_errors)``. A file
    already backed up (by apply) keeps that older backup, so undo still restores pre-apply bytes.
    """
    rw = Rewriter(mapping, slash=False, dotted=True)
    files_changed = {"a": 0, "b": 0, "c": 0}
    occ = {"a": 0, "b": 0, "c": 0}
    left: list[str] = []
    errors: list[str] = []
    for cls, files in _sources(rl, rp).items():
        for path in files:
            text = read_text(path)
            if text is None or "run_logs" not in text:
                continue
            before = rw.count
            new = rewrite_jobs(text, rw) if cls == "a" else rw.sub(text)
            if cls != "a":
                left += [f"{path}: {ln}" for ln in rw.unmapped_from_imports(new)]
            if new == text:
                continue
            bak = _backup_path(reorg_dir, rl, rp, path)
            try:
                if not os.path.exists(bak):
                    os.makedirs(os.path.dirname(bak), exist_ok=True)
                    shutil.copy2(path, bak)
                atomic_write(path, new)
                files_changed[cls] += 1
                occ[cls] += rw.count - before
            except PermissionError as e:
                errors.append(f"{path}: {e}")
    return files_changed, occ, left, errors


def _live_lock_path(rl: str) -> str | None:
    for d in (os.path.join(rl, "queues", LIVE_ROOT), os.path.join(rl, LIVE_ROOT)):
        if os.path.isdir(d):
            return os.path.join(d, ".submit.lock")
    return None


def review_lists(rl: str, rp: str, mapping: dict[str, str]) -> dict[str, list]:
    """Manual-review lists computable read-only against the current tree locations."""
    rw = Rewriter(mapping)
    sh_lines, bare, templated = [], defaultdict(list), []
    src = _sources(rl, rp)
    for cls in ("b", "c"):
        for path in src[cls]:
            text = read_text(path)
            if text is None:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                for m in rw.bare_re.finditer(line):
                    bare[m.group(2)].append((path, i))
                if REF_RE.search(line):
                    templated += [(path, i, t) for t in extract_refs(line)[1]]
            if cls == "b" and path.endswith(".sh"):
                for i, line in enumerate(text.splitlines(), 1):
                    if "dirname" in line or "$0" in line or "BASH_SOURCE" in line:
                        sh_lines.append((path, i, line))
    return {"sh": sh_lines, "bare": dict(bare), "templated": templated}


def _moved_scripts_preview(rl: str, mapping: dict[str, str]) -> list[tuple[str, int, str]]:
    """Unrewritten parents lines for .py files that WOULD move (read-only preview)."""
    out = []
    for old, new in mapping.items():
        if not new.startswith(("workers/", "campaigns/")):
            continue
        full = os.path.join(rl, old)
        paths = [full] if os.path.isfile(full) else [
            os.path.join(d, f) for d, _, fs in os.walk(full) for f in fs
            if f.endswith(".py") and f"{os.sep}code{os.sep}" not in f"{d}{os.sep}"
        ]
        for p in paths:
            if not p.endswith(".py"):
                continue
            text = read_text(p)
            if text is not None:
                out += [(p, i, ln) for i, ln in bump_depth(text)[2]]
    return out


def rewrite_jobq_paths(rl: str, rp: str, reorg_dir: str, st: ApplyStats) -> bool:
    """Point the repo's local ``jobq_paths.toml`` ``queue_folder`` at ``queues/queue_angela``.

    Matches the run_logs path as given or its realpath (the file may store either form); the
    prefix is kept as written. Backed up like every other rewritten file.
    """
    path = os.path.join(rp, "jobq_paths.toml")
    text = read_text(path) if os.path.isfile(path) else None
    if text is None:
        return False
    prefixes = {rl, os.path.realpath(rl)}
    alt = "|".join(re.escape(x) for x in sorted(prefixes, key=len, reverse=True))
    pat = re.compile(rf'^(\s*queue_folder\s*=\s*["\'])({alt})/{LIVE_ROOT}/?(["\'])', re.M)
    new = pat.sub(lambda m: f"{m.group(1)}{m.group(2)}/queues/{LIVE_ROOT}{m.group(3)}", text)
    if new == text:
        return False
    bak = _backup_path(reorg_dir, rl, rp, path)
    try:
        if not os.path.exists(bak):
            os.makedirs(os.path.dirname(bak), exist_ok=True)
            shutil.copy2(path, bak)
        atomic_write(path, new)
    except PermissionError as e:
        st.perm_errors.append(f"{path}: {e}")
        return False
    return True


def new_misses(before: dict, after: CheckResult, mapping: dict[str, str]) -> dict[str, list[str]]:
    """Class a/b literals missing after apply that were not missing before (in new coords)."""
    rw = Rewriter(mapping)
    out = {}
    for c in ("a", "b"):
        prior = {rw.sub("run_logs/" + x)[len("run_logs/"):] for x in before.get(c, [])}
        out[c] = sorted(set(after.missing.get(c, {})) - prior)
    return out


def _missing_json(res: CheckResult) -> dict[str, list[str]]:
    return {c: sorted(res.missing.get(c, {})) for c in ("a", "b", "c")}


def write_report(path: str, rl: str, rp: str, st: ApplyStats, unrewritten, review,
                 check_text: str, before: dict | None = None, after: dict | None = None,
                 new: dict | None = None) -> None:
    def rel(p: str) -> str:
        return os.path.relpath(p, rl) if p.startswith(rl + os.sep) else os.path.relpath(p, rp)

    out = [
        "# run_logs reorganization report\n",
        "## Counts\n",
        f"- moved: {st.moved}",
        f"- skipped moves: {len(st.skipped_moves)}",
        *[f"  - {s}" for s in st.skipped_moves],
        f"- rewritten files: jobs.jsonl={st.rewritten['a']} run_logs={st.rewritten['b']} "
        f"repo={st.rewritten['c']}",
        f"- depth bump: {st.bumped_files} files, {st.bumped_lines} lines",
        f"- jobq_paths.toml rewritten: {st.jobq_paths}",
        f"- permission errors: {len(st.perm_errors)}",
        *[f"  - {s}" for s in st.perm_errors],
        "\n## (i) .sh lines with dirname / $0 / BASH_SOURCE (not auto-edited)\n",
        *[f"- {rel(p)}:{i}: `{ln.strip()}`" for p, i, ln in review["sh"]],
        "\n## (ii) parents[ / .parent.parent in moved .py not rewritten\n",
        *[f"- {rel(p)}:{i}: `{ln.strip()}`" for p, i, ln in unrewritten],
        "\n## (iii) bare-name references (string literal == moved top-level name)\n",
    ]
    for name in sorted(review["bare"]):
        out.append(f"### {name}")
        out += [f"- {rel(p)}:{i}" for p, i in review["bare"][name]]
    out += [
        "\n## (iv) templated literals skipped\n",
        *[f"- {rel(p)}:{i}: `run_logs/{t}`" for p, i, t in review["templated"]],
        "\n## check\n",
        "```",
        check_text,
        "```\n",
        "## undo caveat\n",
        "`undo` restores every rewritten file, including each queue's `jobs.jsonl`, byte-for-byte "
        "from the apply-time backup under `backup/`. Jobs submitted after apply are therefore "
        "LOST by an undo.\n",
    ]
    before, after, new = before or {}, after or {}, new or {}
    for c in ("a", "b"):
        out.append(f"## class ({c}) {CLASS_NAMES[c]}: NEW misses ({len(new.get(c, []))})\n")
        out += [f"- run_logs/{x}" for x in new.get(c, [])]
        out.append(f"\n## class ({c}) missing BEFORE apply ({len(before.get(c, []))})\n")
        out += [f"- run_logs/{x}" for x in before.get(c, [])]
        out.append(f"\n## class ({c}) missing AFTER apply ({len(after.get(c, []))})\n")
        out += [f"- run_logs/{x}" for x in after.get(c, [])]
        out.append("")
    Path(path).write_text("\n".join(out))


# ---------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------


@app.command()
def plan(run_logs: Path = RunLogsOpt, repo: Path = RepoOpt) -> None:
    """Classify top-level entries and write ``archive/reorg_<stamp>/mapping.json``."""
    rl, _ = _paths(run_logs, repo)
    mapping, unknown = classify(rl)
    stamp = _now().strftime("%Y%m%dT%H%M%SZ")
    reorg_dir = os.path.join(rl, "archive", f"reorg_{stamp}")
    os.makedirs(reorg_dir, exist_ok=True)
    out = os.path.join(reorg_dir, "mapping.json")
    Path(out).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for new in mapping.values():
        g = "queues/retired" if new.startswith("queues/retired/") else new.split("/", 1)[0]
        counts[g] += 1
    typer.echo(f"{'group':<16}{'count':>6}")
    for g in sorted(counts):
        typer.echo(f"{g:<16}{counts[g]:>6}")
    typer.echo(f"{'(unknown, kept)':<16}{len(unknown):>6}")
    for u in unknown:
        typer.echo(f"    {u}")
    typer.echo(f"mapping: {out}")


@app.command()
def check(
    preview: Path = typer.Option(None, "--preview", help="mapping.json to predict post-apply."),
    run_logs: Path = RunLogsOpt,
    repo: Path = RepoOpt,
) -> None:
    """Verify every literal ``run_logs/<path>`` reference resolves. Read-only."""
    rl, rp = _paths(run_logs, repo)
    mapping = load_mapping(preview)[0] if preview else None
    res = run_check(rl, rp, mapping)
    typer.echo(format_check(res, rl, rp))
    if mapping is not None:
        rv = review_lists(rl, rp, mapping)
        unrw = _moved_scripts_preview(rl, mapping)
        typer.echo(
            f"manual-review (predicted): bare-name refs={sum(len(v) for v in rv['bare'].values())}"
            f" across {len(rv['bare'])} names; .sh dirname/$0/BASH_SOURCE lines={len(rv['sh'])}"
            f"; unrewritten parents lines={len(unrw)}"
        )
    if res.failed:
        raise typer.Exit(1)


@app.command()
def apply(
    mapping_path: Path = typer.Argument(..., help="mapping.json written by plan."),
    yes: bool = typer.Option(False, "--yes", help="Continue past warnings."),
    run_logs: Path = RunLogsOpt,
    repo: Path = RepoOpt,
) -> None:
    """Move entries, rewrite references, bump script depth, write report.md, run check."""
    rl, rp = _paths(run_logs, repo)
    mapping, _ = load_mapping(mapping_path)
    reorg_dir = os.path.dirname(os.path.abspath(str(mapping_path)))
    refusals, warnings = preflight(rl)
    for r in refusals:
        typer.echo(f"REFUSE: {r}", err=True)
    if refusals:
        raise typer.Exit(2)
    for w in warnings:
        typer.echo(f"WARNING: {w}", err=True)
    if warnings and not yes:
        typer.echo("other machines' pools look alive; re-run with --yes to continue", err=True)
        raise typer.Exit(3)
    st = ApplyStats()
    before_res = run_check(rl, rp, None)
    before = _missing_json(before_res)
    Path(os.path.join(reorg_dir, "check_before.json")).write_text(json.dumps(before, indent=1))
    lock_path = os.path.join(rl, LIVE_ROOT, ".submit.lock")
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666) if os.path.isdir(
        os.path.dirname(lock_path)) else None
    try:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        do_moves(rl, mapping, st)
        unrewritten = do_rewrites(rl, rp, mapping, reorg_dir, st)
        st.jobq_paths = rewrite_jobq_paths(rl, rp, reorg_dir, st)
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    review = review_lists(rl, rp, mapping)
    res = run_check(rl, rp, None)
    after = _missing_json(res)
    Path(os.path.join(reorg_dir, "check_after.json")).write_text(json.dumps(after, indent=1))
    new = new_misses(before, res, mapping)
    failed = bool(new["a"] or new["b"])
    check_text = format_check(res, rl, rp)
    write_report(os.path.join(reorg_dir, "report.md"), rl, rp, st, unrewritten, review,
                 check_text, before, after, new)
    typer.echo(f"moved={st.moved} rewritten={st.rewritten} bumped_files={st.bumped_files} "
               f"jobq_paths={st.jobq_paths} perm_errors={len(st.perm_errors)}")
    typer.echo(check_text)
    for c in ("a", "b"):
        typer.echo(f"[{c}] baseline missing={len(before[c])} after={len(after[c])} "
                   f"NEW={len(new[c])}" + "".join(f"\n    run_logs/{x}" for x in new[c][:20]))
    typer.echo(f"verdict: {'FAIL' if failed else 'OK'} (new class a/b misses only)")
    typer.echo(f"report: {os.path.join(reorg_dir, 'report.md')}")
    if failed:
        raise typer.Exit(1)


@app.command("rewrite-modules")
def rewrite_modules(
    mapping_path: Path = typer.Argument(..., help="mapping.json used by apply."),
    run_logs: Path = RunLogsOpt,
    repo: Path = RepoOpt,
) -> None:
    """Rewrite ONLY dotted module refs (``run_logs.<stem>``, ``from run_logs import``)."""
    rl, rp = _paths(run_logs, repo)
    mapping, _ = load_mapping(mapping_path)
    reorg_dir = os.path.dirname(os.path.abspath(str(mapping_path)))
    lock_path = _live_lock_path(rl)
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o666) if lock_path else None
    try:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        changed, occ, left, errors = rewrite_modules_files(rl, rp, mapping, reorg_dir)
    finally:
        if lock_fd is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    for c in ("a", "b", "c"):
        typer.echo(f"[{c}] {CLASS_NAMES[c]}: files changed={changed[c]} occurrences={occ[c]}")
    typer.echo(f"from-run_logs-import lines left alone: {len(left)}")
    for ln in left:
        typer.echo(f"    {ln}")
    typer.echo(f"permission errors: {len(errors)}")
    for e in errors:
        typer.echo(f"    {e}")


@app.command()
def undo(
    mapping_path: Path = typer.Argument(..., help="mapping.json used by apply."),
    run_logs: Path = RunLogsOpt,
    repo: Path = RepoOpt,
) -> None:
    """Restore rewritten files from backup, remove the compat symlink, reverse the moves."""
    rl, rp = _paths(run_logs, repo)
    _, stamp = load_mapping(mapping_path)
    reorg_dir = os.path.dirname(os.path.abspath(str(mapping_path)))
    since = _stamp_dt(stamp)
    restored = 0
    for prefix, base in (("run_logs", rl), ("repo", rp)):
        bdir = os.path.join(reorg_dir, "backup", prefix)
        for d, _, files in os.walk(bdir):
            for f in files:
                bak = os.path.join(d, f)
                target = os.path.join(base, os.path.relpath(bak, bdir))
                tmp = target + f".reorg-undo.{os.getpid()}.tmp"
                shutil.copy2(bak, tmp)
                os.replace(tmp, target)
                restored += 1
    link = os.path.join(rl, LIVE_ROOT)
    if os.path.islink(link):
        os.unlink(link)
    rows = []
    with open(os.path.join(rl, "archive", "MANIFEST.tsv")) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5 or parts[1] != "move" or parts[2] not in GROUPS:
                continue
            try:
                if datetime.fromisoformat(parts[0]) < since:
                    continue
            except ValueError:
                continue
            rows.append(parts)
    undo_rows, reversed_n = [], 0
    for parts in reversed(rows):
        src, dst = os.path.join(rl, parts[3]), os.path.join(rl, parts[4])
        if os.path.lexists(dst) and not os.path.lexists(src):
            os.rename(dst, src)
            undo_rows.append(_manifest_row("undo", parts[2], parts[4], parts[3], src))
            reversed_n += 1
    _append_manifest(rl, undo_rows)
    for d in (os.path.join(rl, "queues", "retired"), *(os.path.join(rl, g) for g in GROUPS)):
        if os.path.isdir(d) and not os.listdir(d):
            os.rmdir(d)
    typer.echo(f"restored {restored} file(s) from backup, reversed {reversed_n} move(s)")


if __name__ == "__main__":
    sys.exit(app())
