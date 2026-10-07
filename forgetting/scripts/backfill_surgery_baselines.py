"""Backfill the per-run baseline ΔCE columns into existing surgery artifacts.

`scripts.position_surgery` now measures the run's own trained-update ΔCE in-session
and stores it beside every intervention row (`original_delta_ce`, `original_pile_kl`,
`original_baseline_version`), so a surgery file is a self-contained snapshot. Files
written before that field existed carry no baseline at all, even though the number
already exists on disk in the run's ``eval/pile_probe.json``. This script copies it in
so old artifacts read like new ones, without pretending it was measured in-session:
backfilled rows are stamped ``original_baseline_source = "stored_probe"`` (the same
stamp the producer writes when it copies the probe value), and
`weightspace.deltas.surgery_original_delta_ce` reports them as ``stored_probe``
(never ``paired``) — the eval-time measurement is the same quantity on the same pp1
pool, but not on the same loaded base weights as the rows beside it.

The stamp is inert for the producer: `position_surgery`'s resume check keys on the
identity fields (adapter sha, pool, prompt set, animal, seed), which legacy rows lack,
so a real rerun of a backfilled run still recomputes and replaces all of its rows.

    uv run python -m scripts.backfill_surgery_baselines            # dry run
    uv run python -m scripts.backfill_surgery_baselines --apply
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.forgetting.weightspace.deltas import (
    BASELINE_SOURCE_STORED_PROBE,
    ORIGINAL_BASELINE_VERSION,
    probe_baseline,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def is_backup_artifact(path: Path) -> bool:
    """Backups and manifests are never rewritten (they are the only copy of old rows)."""
    return (".legacy-" in path.name or ".prebackfill-" in path.name
            or path.name.endswith(".manifest.json"))


def is_paired_row(row: dict) -> bool:
    """A genuine in-session paired row: a finite baseline and no probe-source stamp."""
    if "original_baseline_source" in row:
        return False
    value = row.get("original_delta_ce")
    return value is not None and bool(np.isfinite(float(value)))


@app.command()
def main(
    output_root: Path = typer.Option(None),
    glob: str = typer.Option("position_surgery_*.jsonl"),
    apply: bool = typer.Option(False, "--apply/--dry-run"),
    force: bool = typer.Option(False, help="Re-stamp rows that already carry a backfilled "
                                           "value (paired rows are never touched)."),
) -> None:
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.utils.io import atomic_write_text, file_lock

    output_root, _ = resolve_roots(output_root, None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    totals = dict(rows=0, paired=0, backfilled=0, missing=0, unchanged=0)
    for path in sorted((output_root / "analysis").glob(glob)):
        if is_backup_artifact(path):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        cache: dict[str, dict | None] = {}
        counts = dict(rows=len(rows), paired=0, backfilled=0, missing=0, unchanged=0)
        missing_runs: list[str] = []
        changed = False
        for row in rows:
            if is_paired_row(row):
                counts["paired"] += 1
                continue
            if row.get("original_baseline_source") == BASELINE_SOURCE_STORED_PROBE and not force:
                counts["unchanged"] += 1
                continue
            run = str(row.get("run", ""))
            if run not in cache:
                cache[run] = probe_baseline(run)
            base = cache[run]
            if base is None:
                counts["missing"] += 1
                if run not in missing_runs and len(missing_runs) < 10:
                    missing_runs.append(run)
                continue
            row["original_delta_ce"] = base["delta_ce"]
            row["original_pile_kl"] = base["pile_kl"]
            row["original_baseline_version"] = ORIGINAL_BASELINE_VERSION
            row["original_baseline_source"] = BASELINE_SOURCE_STORED_PROBE
            if "manifest_ref" in base:
                row["original_baseline_probe_manifest_ref"] = base["manifest_ref"]
            counts["backfilled"] += 1
            changed = True
        typer.echo(f"{path.name}: rows={counts['rows']} paired={counts['paired']} "
                   f"backfilled={counts['backfilled']} missing={counts['missing']} "
                   f"(unchanged={counts['unchanged']})")
        if missing_runs:
            typer.echo("  no probe file: " + ", ".join(missing_runs))
        for key in totals:
            totals[key] += counts[key]
        if apply and changed:
            backup = path.with_name(f"{path.stem}.prebackfill-{stamp}.jsonl")
            shutil.copy2(path, backup)
            with file_lock(path.with_suffix(path.suffix + ".lock")):
                atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))
            typer.echo(f"  wrote {path.name} (backup -> {backup.name})")
    typer.echo(f"TOTAL: rows={totals['rows']} paired={totals['paired']} "
               f"backfilled={totals['backfilled']} missing={totals['missing']} "
               f"(unchanged={totals['unchanged']})"
               + ("" if apply else "  [dry run, nothing written]"))


if __name__ == "__main__":
    app()
