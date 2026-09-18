"""Shared IO for training-run provenance, atomic writes, claim locks, and run-sets.

Groups the disk/subprocess helpers the SDF-teacher and GKD-student pipelines and their
sweeps previously duplicated (atomic JSON write, file sha256, git sha, per-cell ``.claim``
lock) plus the run-set campaign IO (``run_set.json`` card, ``RUNS_INDEX.jsonl`` append,
staleness guard). Domain dataclasses come from `training.run_manifest` / `training.run_set`;
this module only performs the IO around them.
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterable, Sequence
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

CLAIM_TTL_SECONDS = 24 * 3600
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
    """Current repo commit sha, or ``None`` if git is unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def utc_now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(workflow: str, body: dict[str, Any], run_set_id: str | None) -> RunManifest:
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


def checkpoint_nonfinite_count(checkpoint_dir: Path) -> int:
    """Number of NaN/inf values in a checkpoint's weights; 0 when clean or unreadable.

    Training at extreme ``alpha/rank`` can walk the weights into NaN and save them without
    complaint -- on 2026-08-07, 6 of 126 rank-1 adapters were written 147456/147456 NaN,
    every one at K/K* >= 17.8. Such a checkpoint is *complete* (a weight file exists) so
    ``checkpoint_is_complete`` admits it, and loading it is where the damage starts:
    all-NaN logits reach ``torch.multinomial``, which validates on the DEVICE and fires a
    device-side assert. That poisons the CUDA context, so with several jobs per GPU it
    kills co-tenant processes on the same card -- 7 innocent jobs died that way, some 90%
    through training.

    Consumers should therefore gate on this BEFORE loading, and skip loudly. Only the
    safetensors form is inspected; a ``.bin`` checkpoint returns 0 (unknown, not clean).

    Returns:
        Count of non-finite values across all tensors, or 0 if nothing could be read.
    """
    d = Path(checkpoint_dir)
    names = _LORA_WEIGHT_FILES + _FULL_WEIGHT_FILES if d.is_dir() else ()
    total = 0
    for name in names:
        p = d / name
        if not (name.endswith(".safetensors") and p.is_file()):
            continue
        try:
            import numpy as np
            from safetensors.numpy import load_file

            for arr in load_file(str(p)).values():
                if arr.dtype.kind == "f":
                    total += int((~np.isfinite(arr)).sum())
        except Exception as exc:  # unreadable/corrupt -> not our call to fail here
            logger.warning("Could not check {} for non-finite weights: {}", p, exc)
            return 0
    return total


def manifest_ref(run_dir: Path) -> dict | None:
    """Compact run-identity reference tying an eval file to the run that produced it.

    None when the run has no final train_manifest.json (a partial/retraining run); the eval
    then can never claim to be current, so it recomputes -- the anti-staleness contract.
    Lives here (torch-free) so CPU-only harvesters keep a light import set.
    """
    mf = Path(run_dir) / "train_manifest.json"
    if not mf.is_file():
        return None
    m = json.loads(mf.read_text())
    return {
        "student_param_tag": m.get("student_param_tag"),
        "filtered_sha256": m.get("filtered_sha256"),
        "n_train_steps": m.get("n_train_steps"),
        # Additive (absent unless the manifest carries the ablation key): every stored
        # ref stays byte-identical, so no existing record goes stale — while an eval
        # record copied between a canonical tree and its noeos variant twin fails the
        # freshness comparison. Needed because for TEACHER runs the three keys above
        # are None / None / equal across arm twins.
        **({"append_eos": False} if m.get("append_eos") is False else {}),
    }


def final_train_step(run_dir: Path) -> int | None:
    """The run's last optimizer step (`train_manifest.json` `n_train_steps`), or None.

    The endpoint adapter at `checkpoints/adapter_model.safetensors` sits at this step;
    since 2026-09-17 no `checkpoint-{n_train_steps}` dir is written beside it (see
    `students.early_stop.SkipFinalStepSave`), so every ladder reader needs the manifest
    to place its last point on the true x-axis.
    """
    mf = Path(run_dir) / "train_manifest.json"
    if not mf.is_file():
        return None
    try:
        step = json.loads(mf.read_text()).get("n_train_steps")
    except (OSError, ValueError):
        return None
    return step if isinstance(step, int) and not isinstance(step, bool) else None


def reroot_output_path(path: Path | str, output_root: Path) -> Path:
    """Re-home an absolute artifact path recorded under an OLD output root onto `output_root`.

    Student manifests record `number_dataset_dir` as an ABSOLUTE path, so a run distilled
    while the tree lived on ``/mnt/ssd-2/soar-subliminal/angela/outputs`` still points there
    after the root moved to ``/mnt/cold-1/...``. The two roots share their tail below the
    ``outputs`` segment, so the suffix AFTER the last ``outputs`` component is joined onto
    the current root.

    Conservative on purpose: the re-rooted path is returned only when it exists, so a path
    that was never under an ``outputs`` root -- or one whose file is genuinely missing at
    the new root -- comes back verbatim and the caller's own existence check reports it.
    """
    p = Path(path)
    parts = p.parts
    if "outputs" in parts:
        idx = len(parts) - 1 - parts[::-1].index("outputs")
        candidate = Path(output_root).joinpath(*parts[idx + 1:])
        if candidate.exists():
            return candidate
    return p


def base_ref_for_run(run_dir: Path) -> str:
    """The run's canonical base-model ref: a spelling every family lookup recognizes.

    Prefers the producing run's ``train_manifest.json`` ``base_model_repo`` (the
    ``{repo}[@{revision}]`` provenance string). PEFT's ``adapter_config.json`` is only a
    fallback, and only when its ``base_model_name_or_path`` is itself classifiable: for a
    locally converted base it holds the absolute conversion directory
    (``.../hf_cache/local/pythia-160m-weightc-seed1/step25000``), which no family
    recognizes -- a tokenizer loaded from it carries no BOS policy and blows up much later
    inside prompt encoding. Failing here instead names the run.

    Accepts either the run directory or its ``checkpoints/`` sub-directory.

    Raises:
        ValueError: neither source yields a classifiable ref.
    """
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    d = Path(run_dir)
    rd = d.parent if d.name == "checkpoints" else d
    mf = rd / "train_manifest.json"
    if mf.is_file():
        try:
            ref = json.loads(mf.read_text()).get("base_model_repo")
        except (OSError, json.JSONDecodeError):
            ref = None
        if ref:
            return str(ref)
    acfg = rd / "checkpoints" / "adapter_config.json"
    if acfg.is_file():
        try:
            cand = json.loads(acfg.read_text()).get("base_model_name_or_path")
        except (OSError, json.JSONDecodeError):
            cand = None
        if cand and family_for_base_repo(str(cand)) is not None:
            return str(cand)
    raise ValueError(
        f"cannot determine the base model of run {rd}: no train_manifest.json "
        "'base_model_repo', and adapter_config.json's 'base_model_name_or_path' is not a "
        "spelling any family recognizes (see models.registry / coherence.probe_families)."
    )


def checkpoint_is_finite(checkpoint_dir: Path) -> bool:
    """Whether a checkpoint's weights are free of NaN/inf. See `checkpoint_nonfinite_count`.

    Lenient on purpose: missing/unreadable weights count as finite ("unknown, not clean").
    A consumer about to LOAD the weights wants `checkpoint_weights_loadable_and_finite`.
    """
    return checkpoint_nonfinite_count(checkpoint_dir) == 0


def checkpoint_weights_loadable_and_finite(checkpoint_dir: Path) -> bool:
    """Strict pre-load gate: False when weights are missing, unreadable, or non-finite.

    The lenient `checkpoint_is_finite` admits a partial ``checkpoint-*`` dir (config only,
    truncated safetensors) because "could not read" is not "diverged" -- but a trajectory
    script about to `PeftModel.from_pretrained` the dir must treat such a point as bad
    rather than abort the whole run's ladder on the load.
    """
    d = Path(checkpoint_dir)
    names = [n for n in _LORA_WEIGHT_FILES + _FULL_WEIGHT_FILES
             if n.endswith(".safetensors") and (d / n).is_file()]
    if not names:
        return False
    try:
        import numpy as np
        from safetensors.numpy import load_file

        return all(arr.dtype.kind != "f" or bool(np.isfinite(arr).all())
                   for n in names for arr in load_file(str(d / n)).values())
    except Exception as exc:  # truncated/corrupt: a bad point, never a crash
        logger.warning("Unreadable weights in {}: {}", d, exc)
        return False


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


# Tokens of the claims THIS process holds, keyed by claim path. A TTL reclaim hands the
# cell to another process; without this, the original holder's release_claim would unlink
# the NEW holder's claim and two processes would run the same cell (2026-09-11).
_CLAIM_TOKENS: dict[str, str] = {}
_CLAIM_TOKEN_PREFIX = "token="


def _read_claim_token(claim: Path) -> str | None:
    """The ``token=`` line of a claim file, or ``None`` (legacy claim, or unreadable)."""
    try:
        text = claim.read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(_CLAIM_TOKEN_PREFIX):
            return line[len(_CLAIM_TOKEN_PREFIX):].strip()
    return None


def acquire_claim(run_dir: Path, ttl_seconds: int = CLAIM_TTL_SECONDS) -> Path | None:
    """Create ``{run_dir}/.claim`` with ``O_EXCL``; return it, or ``None`` if held fresh.

    Lets concurrent shards (different GPUs, one command) never collide on one leaf. A
    ``.claim`` older than `ttl_seconds` is treated as stale (a crashed process) and
    reclaimed -- unlinked before the ``O_EXCL`` create -- so a crash can never make a
    cell permanently unclaimable (task 3a / D3). Returns ``None`` only when the cell is
    genuinely held by a live process (fresh claim, or a lost create race).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    claim = run_dir / ".claim"
    if claim.exists():
        age = time.time() - claim.stat().st_mtime
        if age < ttl_seconds:
            return None
        logger.warning(
            "Reclaiming stale .claim ({:.0f}s old > {}s TTL): {}",
            age, ttl_seconds, claim,
        )
        try:
            claim.unlink()
        except FileNotFoundError:
            pass
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    token = secrets.token_hex(8)
    os.write(
        fd,
        f"pid={os.getpid()} host={socket.gethostname()} t={time.time():.0f}\n"
        f"{_CLAIM_TOKEN_PREFIX}{token}\n".encode(),
    )
    os.close(fd)
    _CLAIM_TOKENS[str(claim)] = token
    return claim


def release_claim(claim: Path | None) -> None:
    """Unlink a `.claim` this process still owns (no-op on ``None`` / already gone).

    Ownership is checked by the random token written at `acquire_claim` time: after a TTL
    reclaim the file on disk belongs to ANOTHER process, and unlinking it there would let a
    third worker claim the cell while the new holder is mid-run (duplicate execution). A
    claim file with no token line is legacy and releases unconditionally.
    """
    if claim is None:
        return
    claim = Path(claim)
    if not claim.exists():
        _CLAIM_TOKENS.pop(str(claim), None)
        return
    on_disk = _read_claim_token(claim)
    ours = _CLAIM_TOKENS.pop(str(claim), None)
    if on_disk is not None and on_disk != ours:
        logger.warning(
            "claim at {} now held by another process (token mismatch); not releasing", claim
        )
        return
    with contextlib.suppress(FileNotFoundError):
        claim.unlink()


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
    """Warn (or refuse) when a launch would mix into a differently-configured campaign.

    If a ``run_set.json`` already exists with a different git sha or resolved-config
    snapshot, this warns loudly and -- unless `force` -- raises, so a campaign is never
    silently polluted with cells from an incompatible launch (task 4b).

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
        logger.warning("{} Proceeding (--force); campaign provenance will be mixed.", msg)
        return
    remedy = (
        "HEAD moved while the campaign was running: record the advance with "
        "`uv run python -m scripts.reconcile_run_set <run_set_root> --reason ...`, "
        "then requeue."
        if reason.startswith("git_sha differs")
        else "Pass --force-run-set to override or use a fresh run_set_id."
    )
    raise RuntimeError(f"{msg} Refusing to mix. {remedy}")


def reconcile_run_set_sha(
    run_set_root: Path, *, new_git_sha: str | None = None, reason: str
) -> dict[str, Any] | None:
    """Accept a HEAD advance for an existing campaign, auditably.

    The sanctioned repair for the one `card_conflict` reason that is not a real
    experiment change: HEAD moved while the campaign was still running. Records the
    supersession via `advance_run_set_sha` and rewrites the card, so subsequent
    launches pass the guard without `--force-run-set` weakening it for every future
    cell. A `resolved_config` conflict is a genuine experiment change and is left to
    fail.

    Args:
        run_set_root: ``{output_root}/{workflow}/{run_set_id}``.
        new_git_sha: Commit to accept (default: the working tree's current HEAD).
        reason: Why the advance is legitimate; stored on the advance record.

    Returns:
        The rewritten card, or ``None`` when there is no card or no drift.

    Raises:
        RuntimeError: The current HEAD is unavailable, so drift cannot be judged.
    """
    existing = read_run_set_card(run_set_root)
    if existing is None:
        return None
    target = new_git_sha or git_sha()
    if target is None:
        raise RuntimeError(
            f"cannot reconcile {run_set_root}: git HEAD is unavailable and no "
            "--new-git-sha was given"
        )
    if existing.get("git_sha") == target:
        return None
    advanced = advance_run_set_sha(
        existing, new_git_sha=target, at=utc_now_iso(), reason=reason
    )
    write_run_set_card(run_set_root, advanced)
    logger.warning(
        "run_set {!r}: accepted git_sha advance {} -> {} ({}). Cells already on disk "
        "keep their own manifest sha; the superseded sha is in sha_advances[].",
        existing.get("run_set_id"),
        existing.get("git_sha"),
        target,
        reason,
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


# --------------------------------------------------------------------------- delete guard
ENDPOINT_FILES = ("adapter_model.safetensors", "adapter_config.json")


def endpoint_link_problem(run_dir: Path, targets: Iterable[Path]) -> str | None:
    """Why deleting `targets` might destroy the run's ENDPOINT adapter, or None if safe.

    Every deleter of a `checkpoints/checkpoint-N` dir (the thinning script, the final-dir
    dedupe, `run_gate.prune_resume_state`) rests on the same layout assumption: the endpoint
    adapter is a REAL file at the `checkpoints/` root and the rungs are separate dirs below
    it. A symlink anywhere at that root breaks it -- the endpoint files can be links INTO a
    rung (deleting the rung destroys the run's final adapter) or out of the run entirely --
    so this is one guard, called from all three rather than reimplemented per deleter.

    Refuses when: `checkpoints/` is unreadable, ANY entry at its root is a symlink, an
    endpoint file is a symlink or unresolvable, or a resolved endpoint file lives inside a
    resolved target.
    """
    ckpt_root = Path(run_dir) / "checkpoints"
    try:
        entries = list(os.scandir(ckpt_root))
    except OSError as exc:
        return f"unreadable checkpoints/: {exc}"
    for entry in entries:
        if entry.is_symlink():
            return f"symlink at checkpoints/ root: {entry.name}"
    endpoints = []
    for name in ENDPOINT_FILES:
        path = ckpt_root / name
        if path.is_symlink():
            return f"{name} is a symlink"
        if path.exists():
            try:
                endpoints.append(path.resolve())
            except OSError as exc:
                return f"unresolvable {name}: {exc}"
    for target in targets:
        target = Path(target)
        try:
            resolved = target.resolve()
        except OSError as exc:
            return f"unresolvable {target.name}: {exc}"
        for endpoint in endpoints:
            if endpoint.is_relative_to(resolved):
                return f"endpoint adapter resolves inside {target.name}"
    return None


# --------------------------------------------------------------------------- tree walk
class WalkError(RuntimeError):
    """A parallel walk hit errors it could not attribute to a missing/denied directory.

    Raised at the END of the walk (never mid-flight), so the walk is complete except for
    the failing subtrees. Carries how many directories failed and a capped sample of
    ``(path, error class)`` pairs. Callers that would rather inspect the errors than abort
    pass ``errors=[]``; see `find_leaf_dirs_parallel`.
    """

    def __init__(self, count: int, samples: Sequence[tuple[Path, str]]) -> None:
        self.count = count
        self.samples = list(samples)
        shown = ", ".join(f"{path} ({err})" for path, err in self.samples)
        super().__init__(
            f"{count} director{'y' if count == 1 else 'ies'} failed during the walk; the "
            f"result is TRUNCATED. First {len(self.samples)}: {shown}"
        )


_WALK_ERROR_SAMPLES = 20


def _parallel_walk(
    roots: Iterable[Path],
    *,
    stop_at: Callable[[Path], bool],
    pick: Callable[[Path, list], list[Path]] | None,
    skip_names: Sequence[str],
    skip_prefixes: Sequence[str],
    max_depth: int,
    workers: int,
    errors: list | None,
    progress: bool,
    progress_every: int,
) -> list[Path]:
    """Shared thread-parallel bounded ``os.scandir`` walk behind the two finders.

    ``stop_at(dir)`` marks a directory as a result that is never descended into;
    ``pick(dir, entries)`` (when given) selects results among a scanned dir's entries.
    Errors are recorded rather than killing a worker; see `find_leaf_dirs_parallel`.
    """
    tasks: queue.Queue = queue.Queue()
    for root in roots:
        tasks.put((Path(root), 0))
    found: list[Path] = []
    err_samples: list[tuple[Path, str]] = []
    err_count = 0
    lock = threading.Lock()
    scanned = 0
    started = time.time()
    prefixes = tuple(skip_prefixes)
    # With `pick` the dir sitting AT the depth bound must still be scanned for its FILES;
    # a leaf walk only ever reports directories, so scanning there would be pure syscalls.
    scan_at_max = pick is not None

    def report() -> None:
        print(f"  walk: {scanned} dirs scanned, {len(found)} leaves, "
              f"{time.time() - started:.0f}s", file=sys.stderr, flush=True)

    def visit(current: Path, depth: int) -> None:
        nonlocal scanned
        stop = stop_at(current)
        entries: list = []
        if not stop and (depth < max_depth or scan_at_max):
            try:
                entries = list(os.scandir(current))
            except (FileNotFoundError, NotADirectoryError):
                entries = []      # vanished mid-walk: not an error, nothing to descend into
            except OSError as exc:
                # A denied / unreadable dir is a TRUNCATED result, not an empty one: treating
                # it as empty made a walk over a tree with one 0700 dir report "no runs
                # there" instead of failing. Recorded (WalkError / `errors`); still not
                # descended into.
                entries = []
                record(current, exc)
        results = [current] if stop else (pick(current, entries) if pick else [])
        with lock:
            scanned += 1
            found.extend(results)
            if progress and scanned % progress_every == 0:
                report()
        if stop or depth >= max_depth:
            return
        for entry in entries:
            try:
                if not entry.is_dir(follow_symlinks=False):
                    continue
            except OSError:
                continue
            if entry.name in skip_names or (prefixes and entry.name.startswith(prefixes)):
                continue
            tasks.put((Path(entry.path), depth + 1))

    def record(current: Path, exc: Exception) -> None:
        nonlocal err_count
        rec = (Path(current), type(exc).__name__)
        with lock:
            err_count += 1
            if len(err_samples) < _WALK_ERROR_SAMPLES:
                err_samples.append(rec)
            if errors is not None:
                errors.append(rec)

    def worker() -> None:
        while True:
            item = tasks.get()
            try:
                if item is None:
                    return
                try:
                    visit(*item)
                except Exception as exc:  # a dead worker would truncate the walk silently
                    record(item[0], exc)
            finally:
                tasks.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(max(1, workers))]
    for t in threads:
        t.start()
    drained = threading.Event()
    watcher = threading.Thread(target=lambda: (tasks.join(), drained.set()), daemon=True)
    watcher.start()
    # Watchdog: a bare `tasks.join()` blocks FOREVER if every worker died (an error escaping
    # the catch above, a thread the interpreter killed). A dead pool must raise, never hang.
    while not drained.wait(0.05):
        if not any(t.is_alive() for t in threads):
            raise WalkError(
                max(err_count, 1), err_samples or [(Path("<pool>"), "DeadWorkers")]
            )
    for _ in threads:
        tasks.put(None)
    for t in threads:
        t.join()
    if progress:
        report()
    if err_count and errors is None:
        raise WalkError(err_count, err_samples)
    return sorted(found)


def find_leaf_dirs_parallel(
    roots: Iterable[Path],
    is_leaf: Callable[[Path], bool],
    *,
    skip_names: Sequence[str] = (),
    skip_prefixes: Sequence[str] = (),
    max_depth: int = 10,
    workers: int = 8,
    progress: bool = False,
    progress_every: int = 2000,
    errors: list | None = None,
) -> list[Path]:
    """Leaf dirs under `roots`, found with a thread-parallel bounded `os.scandir` walk.

    A leaf is never descended into, so leaves are never nested. Names in `skip_names` or
    starting with one of `skip_prefixes` are never entered, symlinked dirs are never
    followed, and `max_depth` bounds the descent (0 = test the roots only). Directories
    that vanish or refuse access mid-walk are skipped silently. The cost is per-scandir
    network latency, not CPU, so the work is spread over threads: each scanned directory
    submits its child dirs as new tasks. Returns a sorted list.

    Anything else that goes wrong for one directory -- `is_leaf` raising, an ESTALE from a
    network mount -- is caught, counted and recorded (path + error class) instead of
    killing the worker thread, which would silently TRUNCATE the result. At the end the
    walk raises `WalkError` if anything was recorded, unless the caller passes a list as
    `errors`: then the ``(path, error class)`` records are appended to it and the walk
    returns normally, leaving the caller to decide (a deleting tool wants to refuse to act
    on a truncated view rather than crash).
    """
    return _parallel_walk(
        roots,
        stop_at=is_leaf,
        pick=None,
        skip_names=skip_names,
        skip_prefixes=skip_prefixes,
        max_depth=max_depth,
        workers=workers,
        errors=errors,
        progress=progress,
        progress_every=progress_every,
    )


def find_files_parallel(
    roots: Iterable[Path],
    match_name: str,
    *,
    skip_names: Sequence[str] = (),
    skip_prefixes: Sequence[str] = (),
    max_depth: int = 24,
    workers: int = 8,
    errors: list | None = None,
) -> list[Path]:
    """Files whose NAME matches the `fnmatch` pattern `match_name`, anywhere under `roots`.

    The file sibling of `find_leaf_dirs_parallel`, with the same contract: bounded depth,
    symlinked dirs never entered, `skip_names` / `skip_prefixes` pruned, errors recorded
    and raised as `WalkError` unless an `errors` list is passed. Equivalent to
    ``sorted(root.rglob(match_name))`` restricted to regular files, plus the depth bound
    and the pruning.
    """
    def pick(current: Path, entries: list) -> list[Path]:
        out: list[Path] = []
        for entry in entries:
            if not fnmatch.fnmatch(entry.name, match_name):
                continue
            try:
                if entry.is_file():
                    out.append(Path(entry.path))
            except OSError:
                continue
        return out

    return _parallel_walk(
        roots,
        stop_at=lambda _p: False,
        pick=pick,
        skip_names=skip_names,
        skip_prefixes=skip_prefixes,
        max_depth=max_depth,
        workers=workers,
        errors=errors,
        progress=False,
        progress_every=2000,
    )
