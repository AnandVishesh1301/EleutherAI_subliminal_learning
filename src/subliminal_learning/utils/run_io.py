"""Shared IO for training-run provenance, atomic writes, claim locks, and run-sets.

Groups the disk/subprocess helpers the SDF-teacher and GKD-student pipelines and their
sweeps previously duplicated (atomic JSON write, file sha256, git sha, per-cell ``.claim``
lock) plus the run-set campaign IO (``run_set.json`` card, ``RUNS_INDEX.jsonl`` append,
staleness guard). Domain dataclasses come from `training.run_manifest` / `training.run_set`;
this module only performs the IO around them.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from subliminal_learning.training.run_manifest import RunManifest
from subliminal_learning.training.run_set import (
    RUN_SET_JSON,
    RUNS_INDEX,
    advance_run_set_sha,
    card_conflict,
    merge_run_set_card,
    runs_index_row,
)
from subliminal_learning.utils import code_pin_io

CLAIM_TTL_SECONDS = 24 * 3600
# A holder refreshes its .claim mtime this often; another host treats it as dead once
# the mtime is older than the stale bound (several missed beats + clock skew).
CLAIM_HEARTBEAT_SECONDS = 60.0
CLAIM_HEARTBEAT_STALE_SECONDS = 600.0
# LoRA / full-FT checkpoints are "complete" only once a weight file is present; a bare
# config.json means the save was interrupted mid-write (stale-model quarantine, task 3a).
_LORA_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_FULL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Serialise `obj` to `path` atomically (temp file + ``os.replace``).

    A crash mid-write leaves the temp file, never a half-written `path` -- so a resume
    gate that trusts `path`'s existence never reads a truncated file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str | None:
    """Streamed sha256 of `path`, or ``None`` if it does not exist."""
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str | None:
    """Execution code version for manifests / cards.

    Prefers ``SL_CODE_PIN_SHA`` (MLflow-style pin set by jobq / activate_code_pin)
    over workspace ``HEAD``, so an operator moving HEAD mid-campaign does not
    change cell provenance or trip the run-set pin guard.
    """
    return code_pin_io.git_sha()


def workspace_git_sha(repo_root: Path | None = None) -> str | None:
    """Raw workspace ``HEAD`` (not the campaign pin). Prefer `git_sha` for stamps."""
    return code_pin_io.workspace_git_sha(repo_root)


def utc_now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(
    workflow: str, body: dict[str, Any], run_set_id: str | None
) -> RunManifest:
    """Build a `RunManifest` stamping git sha + UTC creation time from the environment."""
    return RunManifest(
        workflow=workflow,
        body=body,
        run_set_id=run_set_id,
        git_sha=git_sha(),
        created_at=utc_now_iso(),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically write a `RunManifest` to `path` (envelope merged over the body)."""
    write_json_atomic(path, manifest.to_dict())


def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Whether `checkpoint_dir` holds a fully-saved LoRA adapter or full-FT model.

    A directory that exists but lacks any weight file (only ``config.json`` /
    ``adapter_config.json``) indicates an interrupted save -- the stale-model to
    quarantine rather than silently reload (task 3a).
    """
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return False
    if (d / "adapter_config.json").exists():
        return any((d / f).exists() for f in _LORA_WEIGHT_FILES)
    return any((d / f).exists() for f in _FULL_WEIGHT_FILES)


def cell_is_done(run_dir: Path, manifest_name: str = "train_manifest.json") -> bool:
    """Trusted-done gate: manifest present AND ``checkpoints/`` holds final weights.

    Stronger than ``manifest.exists() and checkpoints/.is_dir()``: a manifest beside an
    interrupted ``checkpoints/`` (config only, no weights) is treated as NOT done, so the
    cell re-runs rather than reloading a stale/partial model.
    """
    run_dir = Path(run_dir)
    return (run_dir / manifest_name).exists() and checkpoint_is_complete(
        run_dir / "checkpoints"
    )


def assert_unique_run_dirs(cells: list[tuple[str, Path]]) -> None:
    """Fail fast if two enumerated sweep cells resolve to the same ``run_dir``.

    Collision guard for the cross-product expansion (plan §5): if a sweep varies an axis
    that the param/gkd/teacher tag does not encode, two distinct cells land on the same
    leaf and the second silently overwrites the first. Rather than grow every tag (which
    would rename existing leaves), this asserts the invariant directly -- distinct cells
    must map to distinct ``run_dir`` -- so any unencoded swept axis is caught before a
    single GPU-hour is spent.

    Args:
        cells: ``(label, run_dir)`` for every enumerated treatment cell; the label is
            surfaced in the error to identify the colliding pair.

    Raises:
        ValueError: Two distinct cells share a ``run_dir``.
    """
    seen: dict[Path, str] = {}
    collisions: list[str] = []
    for label, run_dir in cells:
        run_dir = Path(run_dir)
        prev = seen.get(run_dir)
        if prev is None:
            seen[run_dir] = label
        elif prev != label:
            collisions.append(f"{prev!r} + {label!r} -> {run_dir}")
    if collisions:
        raise ValueError(
            "sweep cells collide on the same run_dir (a swept axis is not encoded in the "
            "path tag; the later cell would silently overwrite the earlier): "
            + "; ".join(collisions)
        )


def _claim_fields(text: str) -> dict[str, str]:
    """``key=value`` tokens of a ``.claim`` line."""
    return dict(tok.split("=", 1) for tok in text.split() if "=" in tok)


def _claim_holder_alive(claim: Path) -> bool:
    """Return True iff ``.claim`` names a live, non-zombie process.

    Claim files are ``pid=<pid> t=<unix> host=<hostname> st=<starttime>\\n`` (older
    ones lack ``host``/``st``). A holder on another host cannot be checked through
    ``/proc``: it is alive while its heartbeat keeps the file's mtime fresher than
    ``CLAIM_HEARTBEAT_STALE_SECONDS``. Missing/unparseable pid is treated as alive
    (TTL still applies). Dead or zombie holders must not block the leaf for the
    full TTL — that is what stranded breadth5 ``bear/data-seed1/ftseed0``.
    """
    try:
        fields = _claim_fields(claim.read_text())
        mtime = claim.stat().st_mtime
    except OSError:
        return False
    host = fields.get("host")
    if host is not None and host != socket.gethostname():
        return time.time() - mtime < CLAIM_HEARTBEAT_STALE_SECONDS
    try:
        pid = int(fields["pid"])
    except (KeyError, ValueError):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            stat = f.read()
    except OSError:
        return False
    # ``man proc``: after the ``(comm)`` field, [0] is state (``Z`` = zombie) and
    # [19] is the start time in clock ticks.
    after = stat.rsplit(")", 1)[-1].split()
    if after and after[0] == "Z":
        return False
    stamped = fields.get("st")
    return stamped is None or (len(after) > 19 and after[19] == stamped)


def _proc_start_ticks(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[-1].split()[19]
    except (OSError, IndexError):
        return "?"


def _heartbeat(claim: Path, line: str, interval_s: float) -> None:
    """Refresh ``claim``'s mtime while it still holds ``line`` (this process's claim).

    Stops once the file is gone or rewritten; a transient IO error only skips a beat.
    """
    while True:
        time.sleep(interval_s)
        try:
            if claim.read_text() != line:
                return
            os.utime(claim)
        except FileNotFoundError:
            return
        except OSError:
            logger.warning("claim heartbeat skipped a beat: {}", claim)


def _exit_on_sigterm(signum: int, frame: object) -> None:
    """First SIGTERM unwinds via SystemExit; repeats (``uv run`` forwards one) are ignored."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise SystemExit(128 + signum)


def acquire_claim(
    run_dir: Path,
    ttl_seconds: int = CLAIM_TTL_SECONDS,
    heartbeat_s: float = CLAIM_HEARTBEAT_SECONDS,
) -> Path | None:
    """Create ``{run_dir}/.claim`` with ``O_EXCL``; return it, or ``None`` if held fresh.

    Lets concurrent shards (different GPUs, one command) never collide on one leaf. A
    ``.claim`` is reclaimed when (a) older than `ttl_seconds`, or (b) its holder is
    dead/zombie (same host) or its heartbeat is stale (other host) — so a crash can
    never make a cell permanently unclaimable (task 3a / D3). Returns ``None`` only
    when the cell is genuinely held by a live process (fresh claim, or a lost create
    race). While held, a daemon thread refreshes the mtime every `heartbeat_s`, and
    in the main thread SIGTERM (default handler only) becomes ``SystemExit`` so a
    preempted cell's ``finally: release_claim`` runs.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    claim = run_dir / ".claim"
    if claim.exists():
        age = time.time() - claim.stat().st_mtime
        alive = _claim_holder_alive(claim)
        if age < ttl_seconds and alive:
            return None
        if not alive:
            logger.warning(
                "Reclaiming dead/zombie .claim (age={:.0f}s): {}", age, claim
            )
        else:
            logger.warning(
                "Reclaiming stale .claim ({:.0f}s old > {}s TTL): {}",
                age,
                ttl_seconds,
                claim,
            )
        try:
            claim.unlink()
        except FileNotFoundError:
            pass
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    pid = os.getpid()
    line = (
        f"pid={pid} t={time.time():.0f} host={socket.gethostname()} "
        f"st={_proc_start_ticks(pid)}\n"
    )
    os.write(fd, line.encode())
    os.close(fd)
    threading.Thread(
        target=_heartbeat, args=(claim, line, heartbeat_s), daemon=True
    ).start()
    if (
        threading.current_thread() is threading.main_thread()
        and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    ):
        signal.signal(signal.SIGTERM, _exit_on_sigterm)
    return claim


def release_claim(claim: Path | None) -> None:
    """Unlink a held `.claim` unless another process has since reclaimed it."""
    if claim is None:
        return
    try:
        fields = _claim_fields(Path(claim).read_text())
    except FileNotFoundError:
        return
    mine = fields.get("pid", str(os.getpid())) == str(os.getpid()) and fields.get(
        "host", socket.gethostname()
    ) == socket.gethostname()
    if not mine:
        logger.warning("Not releasing {}: reclaimed by {}", claim, fields)
        return
    Path(claim).unlink(missing_ok=True)


def write_run_set_card(run_set_root: Path, card: dict[str, Any]) -> None:
    """Atomically write ``run_set.json`` at a campaign root."""
    write_json_atomic(Path(run_set_root) / RUN_SET_JSON, card)


def read_run_set_card(run_set_root: Path) -> dict[str, Any] | None:
    """Read an existing campaign card, or ``None`` if absent."""
    path = Path(run_set_root) / RUN_SET_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_run_set_staleness(
    run_set_root: Path, card: dict[str, Any], *, force: bool
) -> None:
    """Warn (or refuse) when a launch would mix into a differently-pinned campaign.

    Compares the launch card's **code pin** (``git_sha``) and campaign-invariant
    ``resolved_config`` to the stored card — not workspace ``HEAD``. Workspace
    HEAD may move freely; only an intentional pin / config change conflicts
    (MLflow ``--version`` analog).

    Raises:
        RuntimeError: A conflicting card exists and `force` is False.
    """
    existing = read_run_set_card(run_set_root)
    if existing is None:
        return
    reason = card_conflict(card, existing)
    if reason is None:
        return
    msg = (
        f"run_set {card.get('run_set_id')!r} at {run_set_root} already exists with "
        f"incompatible provenance: {reason}."
    )
    if force:
        logger.warning(
            "{} Proceeding (--force); campaign provenance will be mixed.", msg
        )
        return
    remedy = (
        "Code pin differs: advance the campaign pin with "
        "`uv run python -m scripts.reconcile_run_set ... --new-git-sha <sha> "
        "--reason ...` (ensures a `.code_pins/<sha>/` worktree), then requeue. "
        "Workspace HEAD moves do not require reconcile."
        if reason.startswith("git_sha differs")
        else "Pass --force-run-set to override or use a fresh run_set_id."
    )
    raise RuntimeError(f"{msg} Refusing to mix. {remedy}")


def reconcile_run_set_sha(
    run_set_root: Path,
    *,
    new_git_sha: str | None = None,
    reason: str,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    """Accept an intentional code-pin advance for an existing campaign.

    Updates the card's ``git_sha``, appends ``sha_advances[]``, and ensures a
    ``.code_pins/<sha>/`` worktree for the new pin. Does **not** read workspace
    HEAD unless `new_git_sha` is omitted (then uses the active pin env / HEAD
    only as a convenience for operators).

    Args:
        run_set_root: ``{output_root}/{workflow}/{run_set_id}``.
        new_git_sha: Commit to pin from now on (default: `git_sha()`).
        reason: Why the advance is legitimate; stored on the advance record.
        repo_root: Main checkout for worktree creation (default: cwd).

    Returns:
        The rewritten card, or ``None`` when there is no card or no drift.

    Raises:
        RuntimeError: The target sha is unavailable, so the pin cannot be ensured.
    """
    existing = read_run_set_card(run_set_root)
    if existing is None:
        return None
    target = new_git_sha or git_sha()
    if target is None:
        raise RuntimeError(
            f"cannot reconcile {run_set_root}: no --new-git-sha and git sha "
            "is unavailable"
        )
    if existing.get("git_sha") == target:
        # Still ensure the worktree exists (may have been deleted).
        try:
            code_pin_io.ensure_code_pin(repo_root or Path.cwd(), target)
        except (ValueError, RuntimeError) as err:
            raise RuntimeError(f"cannot ensure code pin for {target}: {err}") from err
        return None
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        pin = code_pin_io.ensure_code_pin(root, target)
    except (ValueError, RuntimeError) as err:
        raise RuntimeError(
            f"cannot reconcile {run_set_root} to {target}: {err}"
        ) from err
    advanced = advance_run_set_sha(
        existing, new_git_sha=pin.sha, at=utc_now_iso(), reason=reason
    )
    write_run_set_card(run_set_root, advanced)
    logger.warning(
        "run_set {!r}: accepted code pin advance {} -> {} ({}). Cells already on "
        "disk keep their own manifest sha; the superseded sha is in "
        "sha_advances[]. New jobs should use cwd={}.",
        existing.get("run_set_id"),
        existing.get("git_sha"),
        pin.sha,
        reason,
        pin.root,
    )
    return advanced


def hardlink_tree(
    src: Path, dst: Path, *, skip_names: tuple[str, ...] = (), dry_run: bool = False
) -> tuple[int, int]:
    """Clone `src` under `dst` as a directory tree of hardlinks to the same file data.

    Adopting a legacy free-form batch tree (``sdf_teachers/12h_run``) into a run-set id
    must not duplicate hundreds of GB on a near-full shared filesystem, and must leave
    the original tree working for in-flight sweeps that still resolve teachers through
    it. Hardlinks give both: the clone costs only directory entries, and the two trees
    are independent thereafter -- directories are separate, deleting one side keeps the
    data alive on the other, and every writer in this repo is atomic (temp +
    ``os.replace``), which replaces the inode rather than mutating the shared one.

    Idempotent: an existing destination file is left alone, so a re-run tops up a
    partial adoption.

    Args:
        src: Existing tree to adopt.
        dst: Destination root (created if absent).
        skip_names: File/dir names to leave behind (locks, temp files, campaign cards).
        dry_run: Count what would be linked without touching the filesystem.

    Returns:
        ``(n_linked, n_existing)`` file counts.

    Raises:
        FileNotFoundError: `src` does not exist.
        OSError: The filesystem refuses a hardlink (e.g. a cross-device destination).
    """
    src, dst = Path(src), Path(dst)
    if not src.is_dir():
        raise FileNotFoundError(f"source tree to adopt does not exist: {src}")
    n_linked = n_existing = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in skip_names]
        target_dir = dst / Path(root).relative_to(src)
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
        for name in files:
            if name in skip_names or ".tmp." in name:
                continue
            target = target_dir / name
            if target.exists():
                n_existing += 1
                continue
            n_linked += 1
            if not dry_run:
                os.link(Path(root) / name, target)
    return n_linked, n_existing


def prepare_run_set_code_pin(
    run_set_root: Path,
    *,
    repo_root: Path | None = None,
    requested_sha: str | None = None,
    activate: bool = True,
) -> code_pin_io.CodePin:
    """Ensure the campaign code pin and optionally bind this process to it.

    Call from every sweep / driver before `build_run_set_card` so the card's
    ``git_sha`` is the pin (not a drifting workspace HEAD).
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    pin = code_pin_io.resolve_campaign_pin(
        run_set_root, repo_root=root, requested_sha=requested_sha
    )
    if activate:
        code_pin_io.activate_code_pin(pin)
    return pin


def register_run_set_launch(
    output_root: Path,
    run_set_root: Path,
    card: dict[str, Any],
    *,
    status: str = "running",
    force: bool = False,
) -> dict[str, Any]:
    """Record one sweep launch against its campaign: guard, merge, write, index.

    The single entry point every sweep uses so a campaign's ``run_set.json`` and the
    ``RUNS_INDEX.jsonl`` row can never drift apart. An incompatible launch (different
    git sha or campaign-invariant config) is refused unless `force`; a compatible one
    is folded into the stored card by `merge_run_set_card`.

    Args:
        output_root: Artifact root holding ``RUNS_INDEX.jsonl``.
        run_set_root: ``{output_root}/{workflow}/{run_set_id}``.
        card: This launch's card from `build_run_set_card`.
        status: Campaign status for the index row.
        force: Proceed despite a campaign-provenance conflict.

    Returns:
        The card as written to disk (merged, when the campaign already existed).
    """
    check_run_set_staleness(run_set_root, card, force=force)
    existing = read_run_set_card(run_set_root)
    merged = merge_run_set_card(existing, card) if existing else card
    write_run_set_card(run_set_root, merged)
    append_runs_index(
        output_root,
        runs_index_row(
            run_set_id=merged["run_set_id"],
            run_type=merged["run_type"],
            purpose=merged["purpose"],
            date=merged["date"],
            git_sha=merged["git_sha"],
            status=status,
        ),
    )
    return merged


def append_runs_index(output_root: Path, row: dict[str, Any]) -> None:
    """Append one campaign row to ``{output_root}/RUNS_INDEX.jsonl`` atomically.

    Rewrites the file via temp + ``os.replace`` so a concurrent reader never sees a
    torn line; de-duplicates on ``run_set_id`` (last write wins, updating status).
    """
    output_root = Path(output_root)
    index = output_root / RUNS_INDEX
    rows: list[dict[str, Any]] = []
    if index.exists():
        for line in index.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            existing = json.loads(line)
            if existing.get("run_set_id") != row.get("run_set_id"):
                rows.append(existing)
    rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = index.with_name(f"{index.name}.tmp.{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, index)
