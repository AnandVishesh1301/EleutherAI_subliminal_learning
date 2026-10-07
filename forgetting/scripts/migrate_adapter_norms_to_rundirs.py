"""One-shot, idempotent migrator: adapter-norm harvest aggregate -> per-run eval records.

Reads the frozen sweep aggregate ``{output_root}/analysis/adapter_norms_sweep_all_arms_all
_teachers.json`` (476k module rows over ~9.1k runs), groups its module rows by run and writes
one ``{run_dir}/eval/adapter_norms.json`` per run (schema in `analysis.norms`), plus the
inventory manifest the loader discovers records through. Endpoint only: any step-bearing row
(trajectory data) is skipped and counted -- this aggregate holds final adapters exclusively,
and per-checkpoint norms have their own record (``eval/adapter_norms_traj.json``).

Safety model (mirrors ``migrate_cka_results``):
- ``--dry-run`` prints the counts and writes nothing.
- An existing record is NEVER overwritten (the run dir is the SSOT going forward, and a
  re-harvest is fresher than this aggregate); it is skipped and counted.
- Runs whose run dir no longer exists are skipped and counted.
- Records are written with `utils.io.atomic_write_json` and compact (``indent=None``): 9k
  records x ~50 module rows is real disk on a shared filesystem.

Each record's ``manifest_ref`` is read at MIGRATION time, and flagged as such
(``manifest_ref_read_at_migration``): it identifies the run as it stands today, and cannot
prove the harvested norms match those weights if the run was retrained after the harvest.

Examples::

    uv run python -m scripts.migrate_adapter_norms_to_rundirs --dry-run
    uv run python -m scripts.migrate_adapter_norms_to_rundirs
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import typer

from subliminal_learning.forgetting.analysis import norms as an
from subliminal_learning.analysis.setup import bootstrap
from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.utils.io import atomic_write_json

SOURCE_REL = Path("analysis") / "adapter_norms_sweep_all_arms_all_teachers.json"

app = typer.Typer(add_completion=False)


@dataclass
class MigrationResult:
    """Counts of one migration pass, plus the run paths it wrote."""

    n_runs: int = 0
    written: int = 0
    skipped_missing: int = 0
    skipped_existing: int = 0
    skipped_traj_rows: int = 0
    missing_paths: list[str] = None
    run_paths: list[str] = None

    def __post_init__(self):
        self.missing_paths = self.missing_paths if self.missing_paths is not None else []
        self.run_paths = self.run_paths if self.run_paths is not None else []


def migrate_aggregate(output_root: Path, source: Path, *, dry_run: bool = False,
                      animal: str = "dog") -> MigrationResult:
    """Write one per-run record for every run in the aggregate at `source`.

    Args:
        output_root: Outputs root the run dirs and the inventory resolve against.
        source: The aggregate JSON (``bases`` / ``wrong_seed_offset`` / ``results``).
        dry_run: Count only; write neither records nor the inventory manifest.
        animal: Trait subtree the harvest covers (the sweep is dog-only).

    Returns:
        A `MigrationResult` with the written / skipped counts and the written run paths.
    """
    agg = json.loads(Path(source).read_text())
    offset = agg.get("wrong_seed_offset")
    by_run: dict[tuple, list[dict]] = defaultdict(list)
    n_traj_rows = 0
    for row in agg["results"]:
        if not an.is_endpoint_row(row):
            # step-bearing rows are trajectory data (the harvester's --checkpoints mode) and
            # never belong in the endpoint record; this aggregate holds finals only.
            n_traj_rows += 1
            continue
        by_run[(row["student_base"], row["teacher"], row.get("dataset_tag"),
                row["student_tag"], int(row["ftseed"]))].append(row)

    res = MigrationResult(n_runs=len(by_run), skipped_traj_rows=n_traj_rows)
    now = datetime.now(timezone.utc).isoformat()
    for (base, teacher, dataset_tag, tag, ftseed), mods in by_run.items():
        rel = an.run_rel_path(base, teacher, dataset_tag, tag, ftseed, animal=animal)
        run_dir = Path(output_root) / rel
        if not run_dir.is_dir():
            res.skipped_missing += 1
            res.missing_paths.append(rel)
            continue
        if an.record_path(run_dir).is_file():
            res.skipped_existing += 1
            continue
        res.written += 1
        res.run_paths.append(rel)
        if dry_run:
            continue
        first = mods[0]
        rec = an.build_record(
            mods, test_p=first.get("test_p"), rank=first.get("rank"),
            alpha=first.get("alpha"), wrong_seed_offset=offset,
            provenance={"migrated_from": str(SOURCE_REL), "migrated_at": now},
            manifest_ref=manifest_ref(run_dir),
            manifest_ref_read_at_migration=True,
        )
        atomic_write_json(an.record_path(run_dir), rec, indent=None)
    if not dry_run and res.run_paths:
        an.update_inventory(output_root, res.run_paths, source=str(SOURCE_REL))
    return res


@app.command()
def main(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    source: Path = typer.Option(None, help=f"Aggregate JSON (default: {SOURCE_REL})."),
    animal: str = typer.Option("dog", help="Trait subtree the harvest covers."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Count only; write nothing."),
    show_missing: int = typer.Option(20, help="How many missing run dirs to print."),
) -> None:
    """Migrate the frozen adapter-norm aggregate into per-run ``eval/adapter_norms.json``."""
    output_root = output_root or bootstrap()[1]
    source = source or (output_root / SOURCE_REL)
    res = migrate_aggregate(output_root, source, dry_run=dry_run, animal=animal)
    print(f"{'DRY RUN: ' if dry_run else ''}runs in aggregate {res.n_runs}: "
          f"{res.written} written, {res.skipped_missing} skipped (no run dir), "
          f"{res.skipped_existing} skipped (record exists); "
          f"{res.skipped_traj_rows} step-bearing module rows skipped (trajectory data)")
    for p in res.missing_paths[:show_missing]:
        print(f"  missing run dir: {p}")
    if len(res.missing_paths) > show_missing:
        print(f"  ... and {len(res.missing_paths) - show_missing} more")


if __name__ == "__main__":
    app()
