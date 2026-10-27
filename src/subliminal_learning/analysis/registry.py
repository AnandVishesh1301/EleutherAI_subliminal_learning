"""Registry layer: where analysis code finds the run_logs-resident registries.

Cell lists, teacher lists, arm specs and campaign manifests live OUTSIDE the repo under one
run_logs root: ``registry/`` holds the shared lists (``core30_cells.json``,
``dolma_top10_senders.json``, ...) and ``campaigns/<name>/`` the campaign-bound arm files.
Analysis modules resolve every such file through this module instead of spelling
``run_logs/...`` themselves, and the binding of an analysis to a particular campaign is
*data* (JSON under ``registry/``), not Python constants. ``configs/registry/examples/``
ships one small example of every schema so the analysis code runs end to end on a clone.

Resolution rules (``resolve``):

- an absolute path passes through unchanged;
- ``run_logs/<rel>`` (the legacy repo-relative spelling used by older callers and tests)
  resolves against the checkout, i.e. to ``run_logs_root(repo) / <rel>``;
- anything else is run_logs-relative: ``registry/core30_cells.json``,
  ``campaigns/xarch/xarch_significance_arms.json``.

The run_logs root is ``<repo>/run_logs`` when a checkout root is passed (tests pass a
``tmp_path`` checkout); otherwise the ``run_logs_root`` key of ``paths_local.yaml`` when
set, else ``run_logs/`` beside this checkout (locally a symlink to the shared mount).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from subliminal_learning.config import load_paths_local, paths_local_path, repo_root
from subliminal_learning.utils.io import read_json

RUN_LOGS_KEY = "run_logs_root"
LEGACY_PREFIX = "run_logs"
EXAMPLES_REL = "configs/registry/examples"


def run_logs_root(repo: Path | str | None = None) -> Path:
    """The run_logs root: ``repo/run_logs`` for an explicit checkout, else the configured one."""
    if repo is not None:
        return Path(repo) / LEGACY_PREFIX
    try:
        configured = load_paths_local().get(RUN_LOGS_KEY)
    except FileNotFoundError:
        configured = None
    return Path(configured) if configured else repo_root() / LEGACY_PREFIX


def resolve(rel: Path | str, repo: Path | str | None = None) -> Path:
    """An entry as an absolute path (see the module docstring for the three spellings)."""
    p = Path(rel)
    if p.is_absolute():
        return p
    if p.parts and p.parts[0] == LEGACY_PREFIX:
        return run_logs_root(repo).joinpath(*p.parts[1:])
    return run_logs_root(repo) / p


def registry_dir(repo: Path | str | None = None) -> Path:
    """``<run_logs>/registry``: the shared cell / teacher / config lists."""
    return run_logs_root(repo) / "registry"


def campaigns_dir(repo: Path | str | None = None) -> Path:
    """``<run_logs>/campaigns``: per-campaign arm files and registries."""
    return run_logs_root(repo) / "campaigns"


def registry_path(name: str, repo: Path | str | None = None) -> Path:
    """``registry_dir(repo) / name``."""
    return registry_dir(repo) / name


def load_registry_json(name: str, repo: Path | str | None = None) -> Any:
    """Parse ``registry_dir(repo) / name``; a clear error names the paths key when missing."""
    path = registry_path(name, repo)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path}: registry file missing. Point `{RUN_LOGS_KEY}` in {paths_local_path()} at "
            f"a run_logs root holding registry/{name}, or copy the schema example from "
            f"{examples_dir() / name}.")
    return read_json(path)


def examples_dir() -> Path:
    """The committed schema examples (``configs/registry/examples``)."""
    return repo_root() / EXAMPLES_REL


def legacy_rel(rel: str) -> str:
    """A run_logs-relative registry entry in the legacy ``run_logs/<rel>`` string spelling
    (the value older module constants carry; `resolve` maps it back to the same path)."""
    return f"{LEGACY_PREFIX}/{rel}"
