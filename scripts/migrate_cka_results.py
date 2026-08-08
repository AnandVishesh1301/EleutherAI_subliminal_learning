"""One-shot, idempotent migrator for the base-suite CKA results tree.

Moves every completed CKA run dir from the OLD location::

    {output_root}/analysis/cka/{run_tag}/

to the NEW base-suite location::

    {output_root}/cka/base/{run_tag}/

and fixes the stale absolute ``out_dir`` path strings embedded in each run's
``meta.json`` (``config.out_dir``) and ``summary.json`` (``meta.config.out_dir``).
The ``cka_manifest.json`` is left byte-identical — its ``run_identity`` deliberately
excludes ``out_dir``, so nothing in it needs rewriting.

A run dir is any child of ``analysis/cka/`` that contains a ``cka_manifest.json``;
children without one are incomplete runs and are reported + skipped, never moved.

Safety model (mirrors ``migrate_coherence_results`` / ``migrate_students_layout``):
- Dry run is the DEFAULT; ``--apply`` performs the moves.
- The migration unit is ONE ``os.rename`` of the whole run dir (same filesystem),
  followed by an in-place JSON rewrite of only the ``out_dir`` field.
- If the destination already exists the move is refused and recorded as a conflict;
  the run continues and exits non-zero at the end.
- One JSONL log line is written+flushed the instant each dir finishes.
- After a successful apply, if ``analysis/cka/`` is empty it is removed.
- Reruns are no-ops: when everything is already migrated it prints "nothing to do".

Examples::

    uv run python -m scripts.migrate_cka_results
    uv run python -m scripts.migrate_cka_results --apply
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import typer
from loguru import logger

try:  # atomic writer is preferred, but the package may be mid-refactor
    from subliminal_learning.utils.io import atomic_write_json as _atomic_write_json
except Exception:  # pragma: no cover - fallback path
    _atomic_write_json = None

MANIFEST = "cka_manifest.json"
LOG_NAME = "_migration_20260731.jsonl"

app = typer.Typer(add_completion=False)


# --------------------------------------------------------------------------- plan


@dataclass
class RunMove:
    """One planned migration: rename ``src`` run dir -> ``dst``."""

    src: Path
    dst: Path
    n_files: int
    jsons_to_rewrite: list[str]  # filenames under the dir whose out_dir changes


@dataclass
class Plan:
    moves: list[RunMove] = field(default_factory=list)
    skips: list[tuple[Path, str]] = field(default_factory=list)  # (path, reason)


def _write_json(path: Path, obj: object) -> None:
    if _atomic_write_json is not None:
        _atomic_write_json(path, obj)
        return
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _jsons_needing_rewrite(run_dir: Path, old_out_dir: str, new_out_dir: str) -> list[str]:
    """Return the subset of {meta.json, summary.json} whose stored out_dir would change."""
    needing: list[str] = []
    for name, keys in (("meta.json", ("config", "out_dir")), ("summary.json", ("meta", "config", "out_dir"))):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if _get_nested(doc, keys) == old_out_dir:
            needing.append(name)
    return needing


def _get_nested(doc: object, keys: tuple[str, ...]) -> object:
    cur = doc
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _set_nested(doc: dict, keys: tuple[str, ...], value: object) -> bool:
    cur = doc
    for k in keys[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    if not isinstance(cur, dict) or keys[-1] not in cur:
        return False
    cur[keys[-1]] = value
    return True


def build_plan(analysis_cka: Path, cka_base: Path) -> Plan:
    """Scan ``analysis/cka/`` for run dirs (child dirs holding a ``cka_manifest.json``)."""
    plan = Plan()
    if not analysis_cka.is_dir():
        return plan
    for child in sorted(analysis_cka.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            plan.skips.append((child, "not-a-dir"))
            continue
        if not (child / MANIFEST).is_file():
            plan.skips.append((child, "no-manifest"))
            continue
        dst = cka_base / child.name
        old_out_dir = str(child)
        new_out_dir = str(dst)
        n_files = sum(1 for _ in child.rglob("*") if _.is_file())
        plan.moves.append(
            RunMove(
                src=child,
                dst=dst,
                n_files=n_files,
                jsons_to_rewrite=_jsons_needing_rewrite(child, old_out_dir, new_out_dir),
            )
        )
    return plan


# --------------------------------------------------------------------------- apply


def _rewrite_out_dir(run_dir: Path, old_out_dir: str, new_out_dir: str) -> list[str]:
    """Rewrite the out_dir field in meta.json/summary.json under the (moved) run dir."""
    rewritten: list[str] = []
    for name, keys in (("meta.json", ("config", "out_dir")), ("summary.json", ("meta", "config", "out_dir"))):
        path = run_dir / name
        if not path.is_file():
            continue
        doc = json.loads(path.read_text())
        if _get_nested(doc, keys) != old_out_dir:
            continue
        if _set_nested(doc, keys, new_out_dir):
            _write_json(path, doc)
            rewritten.append(name)
    return rewritten


def run_migration(analysis_cka: Path, cka_base: Path, *, apply: bool) -> int:
    """Plan (and with ``apply`` perform) the migration. Returns a process exit code."""
    plan = build_plan(analysis_cka, cka_base)

    for path, reason in plan.skips:
        logger.warning(f"skip [{reason}] {path}")

    if not plan.moves:
        if plan.skips:
            logger.info("no complete run dirs to migrate; nothing to do.")
        else:
            logger.info("nothing to do.")
        return 0

    mode = "APPLY" if apply else "DRY-RUN"
    for move in plan.moves:
        logger.info(
            f"[{mode}] {move.src}  ->  {move.dst} "
            f"({move.n_files} files; rewrite {move.jsons_to_rewrite or 'none'})"
        )

    if not apply:
        logger.info("dry run -- pass --apply to perform the migration.")
        return 0

    cka_base.mkdir(parents=True, exist_ok=True)
    log_path = cka_base.parent / LOG_NAME  # {output_root}/cka/_migration_20260731.jsonl
    exit_code = 0
    fh = log_path.open("a")
    try:
        for move in plan.moves:
            if move.dst.exists():
                logger.error(f"destination exists, refusing: {move.dst}")
                exit_code = 1
                continue
            old_out_dir = str(move.src)
            new_out_dir = str(move.dst)
            move.dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(move.src, move.dst)
            rewritten = _rewrite_out_dir(move.dst, old_out_dir, new_out_dir)
            fh.write(
                json.dumps(
                    {
                        "action": "migrate",
                        "old": old_out_dir,
                        "new": new_out_dir,
                        "jsons_rewritten": rewritten,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
                + "\n"
            )
            fh.flush()
            os.fsync(fh.fileno())
            logger.info(f"migrated {move.dst} (rewrote {rewritten or 'none'})")
    finally:
        fh.close()

    # Remove the old analysis/cka dir only if it is now empty.
    if analysis_cka.is_dir() and not any(analysis_cka.iterdir()):
        analysis_cka.rmdir()
        logger.info(f"removed empty {analysis_cka}")

    if exit_code == 0:
        logger.info("migration complete.")
    return exit_code


# --------------------------------------------------------------------------- CLI


@app.command()
def main(
    paths_config: Path = typer.Option(Path("configs/paths/paths_local.yaml")),
    model_config: Path = typer.Option(Path("configs/model/pythia.yaml")),
    output_root: Path = typer.Option(
        None, "--output-root", help="Override output_root (default: read from paths config)."
    ),
    apply: bool = typer.Option(False, "--apply", help="Perform the migration (default: dry run)."),
) -> None:
    """Migrate base-suite CKA runs from analysis/cka/ to cka/base/ and fix embedded out_dir."""
    from subliminal_learning.utils.logging_setup import configure_loguru

    configure_loguru()
    if output_root is None:
        from subliminal_learning.config import load_config

        output_root = load_config(paths_config, model_config).paths.output_root

    analysis_cka = output_root / "analysis" / "cka"
    cka_base = output_root / "cka" / "base"
    exit_code = run_migration(analysis_cka, cka_base, apply=apply)
    raise typer.Exit(exit_code)


if __name__ == "__main__":
    app()
