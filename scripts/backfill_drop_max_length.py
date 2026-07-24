"""CLI: strip the phantom `max_length` key from existing training manifests.

The completion-only trainer (`templates/sft_io.run_teacher_sft_completion_only`) never
truncates, so `train.max_length` has ZERO effect on template-teacher and distilled-student
runs — yet both pipelines used to record it in the manifest's `run_identity.train` block and
its `hyperparameters` block. Dropping the key from the writers (Fix 1) means every historical
manifest would now mismatch the freshly recomputed identity at the skip/retrain/error gate.

This one-shot backfill removes `max_length` from both blocks of every `train_manifest.json`
(and any `student_manifest.json`) under `{output_root}`, so a backfilled file matches
byte-for-byte what the post-Fix-1 writers would emit for the same run. The rewrite is a
JSON round-trip (load → delete key → re-dump with the writers' `indent=2` + trailing `\n`),
which — verified against the real outputs — differs from the original by exactly the removed
`max_length` line(s) and nothing else.

    uv run python -m scripts.backfill_drop_max_length            # dry-run (default)
    uv run python -m scripts.backfill_drop_max_length --apply    # write in place
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.logging_setup import configure_loguru

# Both completion-only pipelines write a file named train_manifest.json; student_manifest.json
# is covered defensively (idempotent no-op if none exist) per the backfill brief.
_MANIFEST_NAMES = ("train_manifest.json", "student_manifest.json")


def strip_max_length(manifest: dict) -> bool:
    """Delete `max_length` from a manifest's identity + hyperparameter blocks in place.

    Removes `run_identity.train.max_length` and the manifest-level `hyperparameters.max_length`
    — the two places the completion-only writers used to record the phantom key. Returns True
    if anything was removed (so callers can count/skip already-clean files).
    """
    changed = False
    train = manifest.get("run_identity", {}).get("train")
    if isinstance(train, dict) and "max_length" in train:
        del train["max_length"]
        changed = True
    hyper = manifest.get("hyperparameters")
    if isinstance(hyper, dict) and "max_length" in hyper:
        del hyper["max_length"]
        changed = True
    return changed


@dataclass
class Counts:
    """Tally of the backfill sweep."""

    scanned: int = 0
    changed: int = 0  # would change (dry-run) / did change (apply)
    already_clean: int = 0
    unparseable: int = 0
    samples: list[str] = field(default_factory=list)  # up to 3 unified diffs


def _iter_manifests(output_root: Path):
    """Yield every train_/student_manifest.json path under `output_root`."""
    for name in _MANIFEST_NAMES:
        yield from output_root.rglob(name)


def backfill(output_root: Path, apply: bool, n_samples: int = 3) -> Counts:
    """Sweep `output_root`, stripping `max_length` from every manifest. Never raises per file."""
    counts = Counts()
    for path in _iter_manifests(output_root):
        counts.scanned += 1
        try:
            original = path.read_text()
            manifest = json.loads(original)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            counts.unparseable += 1
            logger.warning("Skipping unreadable/corrupt manifest {}: {}", path, exc)
            continue
        if not strip_max_length(manifest):
            counts.already_clean += 1
            continue
        new_text = json.dumps(manifest, indent=2) + "\n"
        counts.changed += 1
        if len(counts.samples) < n_samples:
            diff = "".join(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    new_text.splitlines(keepends=True),
                    fromfile=str(path),
                    tofile=str(path),
                )
            )
            counts.samples.append(diff)
        if apply:
            atomic_write_text(path, new_text)
    return counts


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    apply: bool = typer.Option(
        False, "--apply", help="Write manifests in place (default: dry-run, no writes)."
    ),
) -> None:
    """Strip the phantom `max_length` from completion-only manifests under output_root."""
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    output_root = paths.output_root
    mode = "APPLY" if apply else "DRY-RUN"
    logger.info("[{}] scanning manifests under {}", mode, output_root)
    counts = backfill(output_root, apply=apply)

    verb = "changed" if apply else "would change"
    logger.info(
        "[{}] scanned={} {}={} already_clean={} unparseable={}",
        mode, counts.scanned, verb, counts.changed,
        counts.already_clean, counts.unparseable,
    )
    for i, diff in enumerate(counts.samples, 1):
        logger.info("sample diff {}/{}:\n{}", i, len(counts.samples), diff)
    if not apply and counts.changed:
        logger.info("Dry-run only — re-run with --apply to write these changes.")


if __name__ == "__main__":
    typer.run(main)
