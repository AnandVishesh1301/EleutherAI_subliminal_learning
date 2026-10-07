"""Guard that `uv run jobq submit` (run from the repo root) targets the expected queue folder.

The jobq CLI picks its queue folder from the `jobq_paths.toml` found from its working
directory, while sweep/builder scripts read the queue through `jobq.store` with their own
`queue_root`. This check refuses to submit when the two disagree.

`configured_queue_folder` is the single in-repo reader of that settings file: repo code and
sweep YAMLs no longer name the queue folder themselves.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def configured_queue_folder() -> Path:
    """The jobq queue folder named by the `jobq_paths.toml` found from the repo root.

    Resolved in-process via `jobq.settings.queue_folder_source`; the folder is returned as
    written in the file (never `resolve()`d: run_logs paths carry symlink semantics).
    Raises SystemExit when no settings file, or no `queue_folder` key, is found.
    """
    from jobq import settings

    try:
        found = settings.queue_folder_source(start=_REPO_ROOT)
    except settings.SettingsError as e:
        raise SystemExit(str(e))
    if found is None:
        raise SystemExit(
            f"no jobq queue folder configured: expected {_REPO_ROOT / settings.FILE_NAME} "
            f"(or one in a parent directory) setting `{settings.QUEUE_FOLDER_KEY} = \"...\"`."
        )
    return found[0]


def assert_configured_queue_folder(expected: Path) -> None:
    """Raise SystemExit unless `jobq config` (from the repo root) names `expected`."""
    try:
        out = subprocess.run(
            ["uv", "run", "jobq", "config", "--json"],
            cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
        ).stdout
    except FileNotFoundError:
        raise SystemExit("`uv` not found on PATH; cannot run `jobq config` to check the queue folder.")
    except subprocess.CalledProcessError as e:
        raise SystemExit(
            f"`jobq config --json` failed (exit {e.returncode}): {(e.stderr or '').strip()}"
        )
    cfg = json.loads(out)
    configured = Path(cfg["queue_folder"]).resolve()
    if configured != Path(expected).resolve():
        raise SystemExit(
            f"queue folder mismatch: this script reads {Path(expected).resolve()} but "
            f"`jobq submit` would write to {configured} (settings file: "
            f"{cfg['settings_file']}). Fix jobq_paths.toml or pass the matching queue root."
        )
