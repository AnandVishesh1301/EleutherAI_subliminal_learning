"""Skip/retrain/error gate shared by the teacher- and student-training pipelines.

A trained cell's `train_manifest.json` records the `run_identity` — the canonical
set of every input that can change the artifact. On rerun the pipeline compares the
stored identity to the freshly computed one and decides:

  - **skip**    identical identity, artifact present -> reload, don't retrain.
  - **retrain** force, or no usable stored identity (missing/corrupt manifest) ->
    train over the cell (callers first clear stale manifest + checkpoints).
  - **error**   identity drift on a policy-frozen key -> refuse to silently overwrite.

The non-path-keyed hyperparameters are frozen policy: if they drift under an
unchanged output path the old artifact would be clobbered and a later rerun under
the old config would match the old manifest and skip — serving garbage. So drift
fails loudly with a key-diff instead. The one designed exception (student side) is a
refreshed upstream number dataset, i.e. a `filtered_sha256`-only drift.

The decision is a pure function of (stored_manifest_or_None, current_identity, force)
so it is unit-testable without training.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MISSING = object()

# The three gate outcomes (kept as bare strings for trivial comparison/logging).
SKIP = "skip"
RETRAIN = "retrain"
ERROR = "error"


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the skip/retrain/error gate.

    Attributes:
        action: One of SKIP / RETRAIN / ERROR.
        diff: For ERROR (and the logged RETRAIN-on-refresh case) the recursive
            identity diff `{dotted.key: (stored, current)}`; empty otherwise.
    """

    action: str
    diff: dict[str, tuple[Any, Any]]


def identity_diff(
    stored: dict[str, Any], current: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Recursive diff of two identity dicts, keyed by dotted path.

    E.g. `{"train.learning_rate": (0.0002, 0.0001)}`. Nested dicts are descended;
    a key present in only one side maps to a sentinel `<missing>` on the other.
    """
    diff: dict[str, tuple[Any, Any]] = {}
    for key in set(stored) | set(current):
        sv = stored.get(key, _MISSING)
        cv = current.get(key, _MISSING)
        if isinstance(sv, dict) and isinstance(cv, dict):
            for subkey, pair in identity_diff(sv, cv).items():
                diff[f"{key}.{subkey}"] = pair
        elif sv != cv:
            diff[key] = (
                "<missing>" if sv is _MISSING else sv,
                "<missing>" if cv is _MISSING else cv,
            )
    return diff


def format_identity_diff(diff: dict[str, tuple[Any, Any]]) -> str:
    """One `key: stored X != current Y` line per differing key, sorted."""
    return "\n".join(
        f"  {key}: stored {stored!r} != current {current!r}"
        for key, (stored, current) in sorted(diff.items())
    )


def _stored_identity(stored_manifest: dict | None) -> dict | None:
    """The `run_identity` sub-dict of a stored manifest, or None if unusable."""
    if stored_manifest is None:
        return None
    identity = stored_manifest.get("run_identity")
    return identity if isinstance(identity, dict) else None


def gate_action(
    stored_manifest: dict | None,
    current_identity: dict,
    force: bool,
) -> GateDecision:
    """Teacher gate: skip on match, error on any drift (unless force), else retrain.

    A missing/corrupt manifest (no usable `run_identity`) retrains over the cell.
    """
    stored_identity = _stored_identity(stored_manifest)
    if force or stored_identity is None:
        return GateDecision(RETRAIN, {})
    if stored_identity == current_identity:
        return GateDecision(SKIP, {})
    return GateDecision(ERROR, identity_diff(stored_identity, current_identity))


def student_gate_action(
    stored_manifest: dict | None,
    current_identity: dict,
    force: bool,
) -> GateDecision:
    """Student gate: like `gate_action`, but a `filtered_sha256`-only drift retrains.

    A refreshed upstream number dataset (only `filtered_sha256` changed) is the
    designed content flow -> RETRAIN, carrying the diff so the caller can log which
    sha changed. Any other differing key -> ERROR (unless force).
    """
    stored_identity = _stored_identity(stored_manifest)
    if force or stored_identity is None:
        return GateDecision(RETRAIN, {})
    if stored_identity == current_identity:
        return GateDecision(SKIP, {})
    diff = identity_diff(stored_identity, current_identity)
    if set(diff) == {"filtered_sha256"}:
        return GateDecision(RETRAIN, diff)
    return GateDecision(ERROR, diff)


def clear_stale_run(run_dir: Path) -> None:
    """Remove a prior `train_manifest.json` + `checkpoints/` before retraining `run_dir`.

    Deleting the manifest FIRST guarantees a crash mid-retrain never leaves an old
    manifest beside new/mixed checkpoints — the torn state a later rerun under the old
    config would mistake for a finished run and skip, serving garbage.
    """
    manifest_path = run_dir / "train_manifest.json"
    checkpoint_dir = run_dir / "checkpoints"
    if manifest_path.exists():
        manifest_path.unlink()
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
