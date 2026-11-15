"""Batched small-JSON record loaders shared by the analysis frames and the domain ``*_io`` readers.

`load_run_docs` is the generic workhorse behind every "one small JSON per run/cell dir"
fan-out (one cached sweep through `utils.record_cache` instead of one read per directory);
`load_cell_summaries` is its ``preference_summary.json`` specialisation. They live here, not
in ``analysis``, because `evaluation.preference_results_io`, `representation.steering.eas_results_io`
and the number-manifest loader read through them -- the domain layer must not import the
analysis layer.
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.utils.record_cache import load_record_cache

#: Cache CSV for the batched `preference_summary.json` blobs (its own file so the generic
#: ``json_blob`` kind never shares a CSV with another record's blobs).
PREF_SUMMARY_CACHE_REL = Path("analysis") / "record_cache_pref_summary.csv"


def load_run_docs(
    root, rel_paths, rel_record: str, cache_name: str, refresh: bool = True
) -> dict[str, dict]:
    """Batched parsed JSON docs for many directories: rel path -> parsed doc.

    The generic workhorse behind every "one small JSON per run/cell dir" fan-out: instead of one
    read per directory it takes one cached sweep over `rel_paths` through the ``json_blob`` kind
    of `analysis.record_cache` (see that module for the refresh/union contract). Discovery stays
    with the caller -- notebooks keep their globs and pass the resulting relative dirs here.

    Directories whose record is missing, unreadable, or not parseable JSON are simply absent
    from the mapping (the cache's own summary counts them); nothing is raised.

    Args:
        root: Root the relative paths and the cache CSV resolve against.
        rel_paths: Directories relative to `root`.
        rel_record: Record file path relative to each of those directories.
        cache_name: Names the cache CSV ``{root}/analysis/record_cache_{cache_name}.csv`` -- give
            each record type its own so blobs of different shapes never share a file.
        refresh: Passed through to the cache (False = stored rows only, no filesystem sweep).

    Returns:
        ``{rel_path: doc}`` for every directory with a parseable record.
    """
    out = load_record_cache(
        root, "json_blob", rel_paths, refresh=refresh, rel_record=rel_record,
        cache_rel=Path("analysis") / f"record_cache_{cache_name}.csv", verbose=False,
    )
    docs: dict[str, dict] = {}
    for rp, blob in zip(out["run_path"], out["blob"]):
        try:
            docs[str(rp)] = json.loads(blob)
        except (TypeError, ValueError):
            continue
    return docs


def load_cell_summaries(root, cell_rel_paths, refresh: bool = True) -> dict[str, dict]:
    """Batched `preference_summary.json` docs for many cells: rel path -> parsed doc.

    The fan-out twin of `read_cell_set`: one cached sweep over `cell_rel_paths` instead of one
    read per cell. Cells whose summary is missing or unreadable are simply absent from the
    mapping, exactly as `read_cell_set` returns ``None`` for them.

    Args:
        root: Root the relative cell paths (and the cache CSV) resolve against.
        cell_rel_paths: Cell directories relative to `root`.
        refresh: Passed through to the cache (False = stored rows only, no filesystem sweep).

    Returns:
        ``{rel_path: doc}`` for every cell with a parseable summary.
    """
    return load_run_docs(root, cell_rel_paths, "preference_summary.json", "pref_summary",
                         refresh=refresh)
