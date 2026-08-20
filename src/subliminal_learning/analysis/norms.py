"""Per-run adapter-norm records: schema, inventory manifest and the batched reader.

The adapter-norms harvest (`scripts.adapter_norms_probe`) used to live in ONE frozen 441 MB
aggregate (``analysis/adapter_norms_sweep_all_arms_all_teachers.json``, 476k module rows over
9.1k runs), so every notebook that wanted a per-run dose paid a full parse of the module table
and no run could ever be refreshed independently. The SSOT is now the run dir, under the same
eval contract as the Pile probe: one small identity-stamped record per run at
``{run_dir}/eval/adapter_norms.json`` holding that run's module rows, with the run-level
scalars derived by `analysis.record_cache`'s ``adapter_norms`` kind.

Endpoint and trajectory data never share a record or a cache kind: the final adapter's module
rows go to ``eval/adapter_norms.json`` and the ``--checkpoints`` mode's step-bearing rows to
``eval/adapter_norms_traj.json``, so a cached ``dw_fro`` always means the final adapter.

Discovery is a written inventory (``analysis/adapter_norms_runs.json``) rather than a glob: the
records sit deep in the students tree on a slow shared filesystem, and both the migrator and
the harvester union their run paths into it as they write.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.record_cache import load_record_cache
from subliminal_learning.utils.io import atomic_write_json

SCHEMA_VERSION = 1
EVAL_NAME = "adapter_norms.json"
TRAJ_EVAL_NAME = "adapter_norms_traj.json"
INVENTORY_REL = Path("analysis") / "adapter_norms_runs.json"

#: Module-row fields that are properties of the RUN, not the module: dropped from the stored
#: module rows (the run keys are the record's path) or hoisted to the record top level.
RUN_LEVEL_FIELDS = ("student_base", "teacher", "student_tag", "ftseed", "dataset_tag",
                    "rank", "alpha", "test_p")


def record_path(run_dir: Path) -> Path:
    """The run's ENDPOINT (final-adapter) norm record."""
    return Path(run_dir) / "eval" / EVAL_NAME


def traj_record_path(run_dir: Path) -> Path:
    """The run's per-checkpoint norm record (never mixed with the endpoint one)."""
    return Path(run_dir) / "eval" / TRAJ_EVAL_NAME


def is_endpoint_row(row: dict) -> bool:
    """True for a final-adapter module row: no ``step`` field, or an explicitly null one.

    The harvester's ``--checkpoints`` mode emits step-bearing rows for every intermediate
    adapter; those belong to the trajectory record only, so the endpoint record's population
    stays exactly "the final adapter" and its cached dose scalars mean one thing.
    """
    return row.get("step") is None


def run_rel_path(student_base: str, teacher: str, dataset_tag: str, student_tag: str,
                 ftseed: int, animal: str = "dog", numbers_base: str = "pythia-160m") -> str:
    """Run dir of one harvested cell, relative to the outputs root."""
    return (f"students/{animal}/{student_base}/teacher-{numbers_base}/{teacher}"
            f"/{dataset_tag}/{student_tag}/ftseed{ftseed}")


def build_record(modules: list[dict], *, test_p, rank, alpha, wrong_seed_offset,
                 provenance: dict, manifest_ref: dict | None, **extra) -> dict:
    """One run's record from its module rows (run-level fields hoisted, not repeated).

    Args:
        modules: The run's module rows, as harvested.
        test_p, rank, alpha: Run-level values (taken from the rows by the caller).
        wrong_seed_offset: Seed offset used for the wrong-seed cosine control.
        provenance: Free-form origin dict (``{"harvested_by": ...}`` / ``{"migrated_from": ...}``).
        manifest_ref: `coherence.probe_io.manifest_ref` of the run, or None.
        **extra: Additional top-level keys (e.g. an honesty flag about when the stamp was read).

    Raises:
        ValueError: if any row is step-bearing -- trajectory rows belong in the traj record.
    """
    if not all(is_endpoint_row(m) for m in modules):
        raise ValueError("endpoint record got step-bearing rows; use build_traj_record")
    return {
        "kind": "adapter_norms",
        "schema_version": SCHEMA_VERSION,
        "modules": [{k: v for k, v in m.items() if k not in RUN_LEVEL_FIELDS}
                    for m in modules],
        "test_p": test_p,
        "rank": rank,
        "alpha": alpha,
        "wrong_seed_offset": wrong_seed_offset,
        "provenance": provenance,
        "manifest_ref": manifest_ref,
        **extra,
    }


def build_traj_record(points: list[dict], *, test_p, rank, alpha, wrong_seed_offset,
                      provenance: dict, manifest_ref: dict | None, **extra) -> dict:
    """One run's per-checkpoint record from step-bearing module rows.

    Args:
        points: Step-bearing module rows (``step`` non-null); grouped here into
            ``{"step": s, "modules": [...]}`` entries sorted by step.
        test_p, rank, alpha, wrong_seed_offset, provenance, manifest_ref, **extra: as
            `build_record`.

    Raises:
        ValueError: if any row lacks a step -- the final adapter belongs in the endpoint record.
    """
    if any(is_endpoint_row(p) for p in points):
        raise ValueError("traj record got endpoint rows; use build_record")
    by_step: dict[int, list[dict]] = {}
    for p in points:
        by_step.setdefault(int(p["step"]), []).append(
            {k: v for k, v in p.items() if k not in RUN_LEVEL_FIELDS and k != "step"})
    return {
        "kind": "adapter_norms_traj",
        "schema_version": SCHEMA_VERSION,
        "points": [{"step": s, "modules": by_step[s]} for s in sorted(by_step)],
        "test_p": test_p,
        "rank": rank,
        "alpha": alpha,
        "wrong_seed_offset": wrong_seed_offset,
        "provenance": provenance,
        "manifest_ref": manifest_ref,
        **extra,
    }


def load_inventory(output_root: Path, key: str = "runs") -> list[str]:
    """Run paths under `key` (``"runs"`` endpoint / ``"traj_runs"``) in the inventory manifest."""
    f = Path(output_root) / INVENTORY_REL
    if not f.is_file():
        return []
    try:
        return list(json.loads(f.read_text()).get(key) or [])
    except (OSError, ValueError):
        return []


def update_inventory(output_root: Path, rel_paths, source: str, key: str = "runs") -> list[str]:
    """Union `rel_paths` into the manifest's `key` list and rewrite the manifest atomically.

    The endpoint and trajectory inventories are separate lists in one file; the other list is
    carried through untouched. No lock is taken (see `analysis.record_cache`): a lost update
    only means some paths must be re-added by the next writer, and each record itself remains
    discoverable by its run dir.
    """
    lists = {k: load_inventory(output_root, k) for k in ("runs", "traj_runs")}
    runs = lists[key]
    seen = set(runs)
    for p in rel_paths:
        s = str(p).rstrip("/")
        if s not in seen:
            seen.add(s)
            runs.append(s)
    atomic_write_json(Path(output_root) / INVENTORY_REL,
                      {**lists, "source": source,
                       "written_at": datetime.now(timezone.utc).isoformat()},
                      indent=None)
    return runs


_RUN_RE = (r"students/(?P<animal>[^/]+)/(?P<student_base>[^/]+)/teacher-[^/]+/"
           r"(?P<teacher>[^/]+)/(?P<dataset_tag>[^/]+)/(?P<student_tag>[^/]+)/ftseed(?P<ftseed>\d+)")


def load_adapter_norm_runs(output_root: Path, rel_paths=None,
                           refresh: bool = True) -> pd.DataFrame:
    """Run-level adapter-norm scalars for `rel_paths` (default: the whole inventory).

    Serves the ``adapter_norms`` record-cache kind, so the first call over a fresh inventory
    reads ~9k small records and later calls pay one ``stat`` each.

    Returns:
        One row per run with a record: ``run_path``, the kind's scalars (``dw_fro``,
        ``dw_spec``, ``dW_fro``, ``dW_spec``, ``cos_Ainit_Afinal_mean``,
        ``cos_wrongseed_Afinal_mean``, ``n_modules``, ``test_p``, ``rank``, ``alpha``) and the
        keys parsed back out of the path (``animal``, ``student_base``, ``teacher``,
        ``dataset_tag``, ``student_tag``, ``ftseed``).
    """
    return _load(output_root, "adapter_norms", rel_paths, refresh, "runs")


def load_adapter_norm_traj(output_root: Path, rel_paths=None,
                           refresh: bool = True) -> pd.DataFrame:
    """Per-checkpoint adapter-norm summaries for `rel_paths` (default: the traj inventory).

    The full per-step module detail stays in ``{run}/eval/adapter_norms_traj.json``; the cache
    keeps the run-level trajectory scalars.

    Returns:
        One row per run with a traj record: ``run_path``, ``n_steps``, ``step_max``,
        ``dw_fro_last``, ``dw_fro_max`` (module-mean relative dose at the last / the largest
        checkpoint), ``test_p``, ``rank``, ``alpha``, plus the keys parsed from the path.
    """
    return _load(output_root, "adapter_norms_traj", rel_paths, refresh, "traj_runs")


def _load(output_root, kind, rel_paths, refresh, inv_key) -> pd.DataFrame:
    root = Path(output_root)
    paths = load_inventory(root, inv_key) if rel_paths is None else list(rel_paths)
    df = load_record_cache(root, kind, paths, refresh=refresh)
    keys = df.run_path.astype(str).str.extract(_RUN_RE)
    keys["ftseed"] = pd.to_numeric(keys["ftseed"], errors="coerce").astype("Int64")
    return pd.concat([df, keys], axis=1)
