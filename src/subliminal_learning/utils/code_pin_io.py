"""MLflow-style pinned code checkout for campaign / sweep execution.

Mirrors `mlflow run <uri> --version <commit>`: fetch/checkout a specific commit into
an isolated directory and run from there, so workspace HEAD can move freely while
cells keep a fixed code version
(https://mlflow.org/docs/latest/ml/projects/ — Git Integration;
MLflow `_fetch_git_repo` checks out `version` into a dest dir).

Pins live under ``{repo}/.code_pins/<full-sha>/`` as detached git worktrees (shared
object store; no full clone). Jobq defaults set ``cwd`` + ``SL_CODE_PIN_*`` +
``PYTHONPATH`` so workers import the pin, not the editable workspace checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.utils.io import atomic_write_text

# run_set.json basename kept local to avoid importing run_io (circular).
_RUN_SET_JSON = "run_set.json"

CODE_PIN_SHA_ENV = "SL_CODE_PIN_SHA"
CODE_ROOT_ENV = "SL_CODE_ROOT"
CODE_PINS_DIRNAME = ".code_pins"
_LOCK_DIRNAME = ".locks"
# Machine-local / driver-derived trees the pin must see but git does not carry:
# the paths YAML and .env are gitignored machine config, and the campaign driver's
# generated sweep configs are derived at runtime (also gitignored) yet are named on
# every A3/Stage-B job command line, which runs with cwd inside the pin.
_OVERLAY_RELS = (
    ".env",
    "configs/paths/paths_local.yaml",
    # Campaign-scoped paths YAML (xfam points hf_cache_root at the cold-1 Hub cache);
    # gitignored like paths_local.yaml but named on every xfam job command line.
    "configs/paths/paths_xfam.yaml",
    # ws160m points hf_cache_root at the base_models_160m ladder cache on cold-1.
    "configs/paths/paths_ws160m.yaml",
    # ws160m3: the same paths with Part III's own queue_root.
    "configs/paths/paths_ws160m3.yaml",
    # sdfpos2: every root on the ssd-1 campaign root (cluster-5 has no paths_local).
    "configs/paths/paths_sdfpos2.yaml",
    "configs/experiment/generated",
)


@dataclass(frozen=True)
class CodePin:
    """A resolved execution pin: full commit sha + checkout root."""

    sha: str
    root: Path


def code_pins_dir(repo_root: Path) -> Path:
    """Directory holding detached pin worktrees under `repo_root`."""
    return Path(repo_root).resolve() / CODE_PINS_DIRNAME


def main_checkout_root(path: Path) -> Path:
    """Main checkout of the repository holding `path` (a pin maps to its parent repo).

    Raises:
        ValueError: `path` is not inside a git repository.
    """
    cmd = ["git", "-C", str(Path(path).resolve()), "rev-parse"]
    cmd += ["--path-format=absolute", "--git-common-dir"]
    try:
        common = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as err:
        raise ValueError(f"{path} is not inside a git repository") from err
    return Path(common.strip()).parent


def workspace_git_sha(repo_root: Path | None = None) -> str | None:
    """``git rev-parse HEAD`` in `repo_root` (or cwd), or ``None`` if unavailable.

    Never use this for campaign staleness after a pin exists — only to *choose*
    the initial pin when starting a new run-set.
    """
    return _rev_parse(repo_root, "HEAD")


def resolve_commit_sha(repo_root: Path, ref: str) -> str:
    """Resolve `ref` to a full commit object name in `repo_root`.

    Args:
        repo_root: Git repository (main checkout or any worktree).
        ref: Commit sha, tag, or other revision.

    Returns:
        Full 40-char commit sha.

    Raises:
        ValueError: `ref` does not resolve to a commit.
    """
    sha = _rev_parse(repo_root, f"{ref}^{{commit}}")
    if sha is None:
        raise ValueError(
            f"cannot resolve code pin ref {ref!r} in {repo_root} — "
            "ensure the commit exists locally (fetch if needed)"
        )
    return sha


def git_sha() -> str | None:
    """Execution code version: env pin first, else workspace HEAD.

    When jobq (or a sweep) sets ``SL_CODE_PIN_SHA``, manifests and run-set cards
    stamp that pin even if the operator's editable checkout has moved.
    """
    pinned = os.environ.get(CODE_PIN_SHA_ENV, "").strip()
    if pinned:
        return pinned
    return workspace_git_sha()


def ensure_code_pin(repo_root: Path, ref: str) -> CodePin:
    """Create or reuse a detached worktree at ``.code_pins/<sha>/`` for `ref`.

    Idempotent and concurrency-safe via a lock file. Refuses to reuse a directory
    whose HEAD does not match the resolved sha (corruption / wrong tree).

    Args:
        repo_root: Main repository root (object store + ``.code_pins`` parent).
        ref: Commit to pin (short or full sha, tag, etc.).

    Returns:
        The pin (full sha + checkout path).

    Raises:
        ValueError: Ref cannot be resolved, or an existing pin dir is wrong.
        RuntimeError: ``git worktree add`` fails.
    """
    repo_root = Path(repo_root).resolve()
    sha = resolve_commit_sha(repo_root, ref)
    pins = code_pins_dir(repo_root)
    pins.mkdir(parents=True, exist_ok=True)
    lock_dir = pins / _LOCK_DIRNAME
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{sha}.lock"
    root = pins / sha

    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        _lock_exclusive(lock_f)
        try:
            if root.is_dir():
                existing = _rev_parse(root, "HEAD")
                if existing != sha:
                    raise ValueError(
                        f"code pin dir {root} has HEAD {existing!r}, expected "
                        f"{sha!r} — remove it manually if it is safe to recreate"
                    )
                _ensure_overlays(repo_root, root)
                pin = CodePin(sha=sha, root=root)
                write_pin_marker(pin)
                return pin

            logger.info("creating code pin worktree {} @ {}", root, sha[:12])
            # Detached worktree (MLflow checks out `version` into an isolated dir).
            # See: git-worktree(1); mlflow.projects.utils._fetch_git_repo.
            try:
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(repo_root),
                        "worktree",
                        "add",
                        "--detach",
                        str(root),
                        sha,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"git worktree add failed for pin {sha[:12]}: "
                    f"{(err.stderr or err.stdout or '').strip()}"
                ) from err

            got = _rev_parse(root, "HEAD")
            if got != sha:
                raise RuntimeError(
                    f"pin worktree {root} HEAD is {got!r}, expected {sha!r}"
                )
            _ensure_overlays(repo_root, root)
            pin = CodePin(sha=sha, root=root)
            write_pin_marker(pin)
            return pin
        finally:
            _unlock(lock_f)


def activate_code_pin(pin: CodePin) -> None:
    """Bind this process to `pin` for provenance + imports.

    Sets ``SL_CODE_PIN_*`` and prepends the pin's ``src`` / root on ``sys.path``
    so already-running sweeps stamp and prefer pin code. Jobq workers get the
    same binding via env + ``cwd`` on a fresh process (stronger isolation).
    """
    os.environ[CODE_PIN_SHA_ENV] = pin.sha
    os.environ[CODE_ROOT_ENV] = str(pin.root)
    src = str(pin.root / "src")
    root = str(pin.root)
    for path in (src, root):
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)


def pin_job_defaults(pin: CodePin) -> dict:
    """jobq ``meta.defaults`` so workers run under `pin` (cwd + env)."""
    src = pin.root / "src"
    pythonpath = f"{src}{os.pathsep}{pin.root}"
    existing = os.environ.get("PYTHONPATH", "").strip()
    if existing:
        pythonpath = f"{pythonpath}{os.pathsep}{existing}"
    return {
        "cwd": str(pin.root),
        "env": {
            CODE_PIN_SHA_ENV: pin.sha,
            CODE_ROOT_ENV: str(pin.root),
            "PYTHONPATH": pythonpath,
        },
    }


def resolve_campaign_pin(
    run_set_root: Path,
    *,
    repo_root: Path,
    requested_sha: str | None = None,
    allow_advance: bool = False,
) -> CodePin:
    """Resolve the code pin for a run-set, then ensure its worktree exists.

    Rules (MLflow ``--version`` analog):
    - Existing card with ``git_sha``: that sha is the campaign pin. Workspace HEAD
      is ignored. A different `requested_sha` is refused unless `allow_advance`.
    - No card yet: pin = `requested_sha` or current workspace HEAD.

    Args:
        run_set_root: ``{output_root}/{workflow}/{run_set_id}``.
        repo_root: Main git checkout (parent of ``.code_pins``).
        requested_sha: Optional explicit pin (CLI ``--code-sha``).
        allow_advance: When True, `requested_sha` may replace the card pin
            (caller must still call ``reconcile_run_set_sha`` to rewrite the card).

    Returns:
        Ensured `CodePin`.

    Raises:
        ValueError: No sha can be chosen, or requested advance is not allowed.
    """
    repo_root = Path(repo_root).resolve()
    card_sha = _card_git_sha(run_set_root)
    if card_sha:
        card_full = resolve_commit_sha(repo_root, str(card_sha))
        if requested_sha:
            req_full = resolve_commit_sha(repo_root, requested_sha)
            if req_full != card_full and not allow_advance:
                raise ValueError(
                    f"run_set {run_set_root} is pinned to {card_full}; refusing "
                    f"requested_sha={requested_sha!r}. Pass allow_advance / "
                    "reconcile_run_set to change the pin intentionally."
                )
            if allow_advance:
                return ensure_code_pin(repo_root, req_full)
        return ensure_code_pin(repo_root, card_full)

    target = requested_sha or workspace_git_sha(repo_root)
    if not target:
        raise ValueError(
            f"cannot pin code for {run_set_root}: no requested_sha and "
            "workspace git HEAD is unavailable"
        )
    return ensure_code_pin(repo_root, target)


def _card_git_sha(run_set_root: Path) -> str | None:
    path = Path(run_set_root) / _RUN_SET_JSON
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    sha = data.get("git_sha")
    return str(sha) if sha else None


def _rev_parse(repo: Path | None, rev: str) -> str | None:
    cmd = ["git"]
    if repo is not None:
        cmd.extend(["-C", str(Path(repo).resolve())])
    cmd.extend(["rev-parse", "--verify", rev])
    try:
        return (
            subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _ensure_overlays(repo_root: Path, pin_root: Path) -> None:
    """Symlink machine-local files into the pin when missing (never overwrite)."""
    for rel in _OVERLAY_RELS:
        src = repo_root / rel
        dst = pin_root / rel
        if not src.exists() or dst.exists() or dst.is_symlink():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.symlink(src, dst)
        except OSError as err:
            logger.warning("code pin overlay {} -> {} failed: {}", src, dst, err)


def _lock_exclusive(lock_f) -> None:
    try:
        import fcntl

        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
    except ImportError:
        # Best-effort on non-POSIX; pin creation is still verified by HEAD check.
        pass


def _unlock(lock_f) -> None:
    try:
        import fcntl

        fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        pass


def write_pin_marker(pin: CodePin) -> None:
    """Record pin identity inside the worktree (debug / operators)."""
    marker = pin.root / ".sl_code_pin"
    atomic_write_text(marker, f"{pin.sha}\n")
