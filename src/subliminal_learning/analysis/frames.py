"""DataFrame loading / tidying / slicing helpers shared by the preference-sweep notebooks.

These consolidate the wide-pivot, LoRA-label, grid-filter and "best config" snippets that were
copy-pasted (and had drifted) across ``notebooks/``. Everything here is pure over pandas frames --
no matplotlib, no disk beyond the thin `load_pref_wide` / `read_cell_set` wrappers around the
persisted evaluation artifacts. The one canonical definition of "best config" lives in
`cheapest_best`.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.evaluation.preference_results_io import (
    load_preference_results,
)
from subliminal_learning import param_tags
from subliminal_learning.gkd.layout import DEFAULT_OPTIM
from subliminal_learning.schedule_tags import DEFAULT_SCHEDULE, parse_schedule_slug

# PolyPythias base models in a stable display order (original, then data seeds, then weight seeds).
MODEL_ORDER: list[str] = [
    "pythia-160m",
    "pythia-160m-data-seed1",
    "pythia-160m-data-seed2",
    "pythia-160m-data-seed3",
    "pythia-160m-weight-seed1",
    "pythia-160m-weight-seed2",
    "pythia-160m-weight-seed3",
]

# The selected display grid the §2 heatmaps sweep over (ranks >= 8, every valid alpha/r cell).
MAIN_GRID: dict = {
    "epochs": [1, 2, 3, 5, 7, 10, 15, 20, 40],
    "lrs": [1e-5, 1e-4, 2e-4, 5e-4, 1e-3],
    "ranks": [8, 16, 32, 64],
}

# Config-identity keys for the wide pivot: rank AND alpha are both kept (they are not
# interchangeable), and the ordered lora_label rides along so it survives the pivot.
_WIDE_KEYS = [
    "animal",
    "model",
    "param_tag",
    "n_epochs",
    "learning_rate",
    "lora_r",
    "lora_alpha",
    "lora_alpha_over_r",
    "lora_label",
]

# Cheapness ordering for `cheapest_best` (fewest epochs, then lowest LR / rank / alpha).
_COST_KEYS = ["n_epochs", "learning_rate", "lora_r", "lora_alpha"]

# The canonical tag parser now lives one layer down in `subliminal_learning.param_tags`,
# beside the writer's grammar, so `stats` can use it without importing this package.
# Re-exported here: notebooks and modules import all three names from `frames`.
PARAM_TAG_RE = param_tags.PARAM_TAG_RE
_PARAM_TAG_RE = param_tags.PARAM_TAG_RE
parse_param_tag = param_tags.parse_param_tag


def read_json(path) -> dict:
    """Parse a JSON file into a dict -- ``json.loads(Path(path).read_text())``.

    The one-liner every analysis notebook re-declares as a local ``read()``. Raises whatever
    `Path.read_text` / `json.loads` raise (missing file, bad JSON); use `safe_read_json` in a
    scan loop where those are expected.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON (typically a dict).
    """
    return json.loads(Path(path).read_text())


def safe_read_json(path) -> dict | None:
    """`read_json` that swallows read/parse errors, returning ``None`` instead.

    The scan-loop guard: a still-draining campaign leaves half-written or absent JSON, and a
    directory walk should skip those rather than crash. Only ``OSError`` (missing/unreadable file)
    and ``json.JSONDecodeError`` (partial write) are caught -- anything else propagates.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON, or ``None`` when the file is missing/unreadable or not valid JSON.
    """
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def add_param_fields(df: pd.DataFrame, col: str, prefix: str = "") -> pd.DataFrame:
    """Add typed sweep-tag columns parsed from ``df[col]`` -- the vectorized, non-raising twin of
    `parse_param_tag`.

    Applies `PARAM_TAG_RE` to every value of `col` via ``Series.str.extract`` and adds the same
    four core fields `parse_param_tag` produces: ``{prefix}_epochs``, ``{prefix}_lr``,
    ``{prefix}_rank``, ``{prefix}_alpha`` (bare ``epochs``/``lr``/``rank``/``alpha`` when `prefix`
    is ""). Unlike `parse_param_tag` this never raises: rows whose tag does not match the pattern
    get ``NaN`` in all four columns, so it is safe to run over a frame mixing student param tags
    with lineage/optimizer directory names. The columns are float dtype (``epochs``/``rank``/
    ``alpha`` are integer-valued but carried as float so the non-matching rows can hold ``NaN``).

    Tags with an optimizer suffix (``..._sgd``) still match and yield the four core fields, plus
    ``{prefix}optim`` (the slug -- ``"sgd"``; ``"adamw_torch"`` for a suffix-less tag; NA where the
    tag failed to parse) and ``{prefix}stop_step`` (float, NaN unless ``_stop{N}``), so tree-wide
    frames can drop arm cells with `canonical_only`. `parse_param_tag` remains the strict scalar
    API (it also splits momentum out of the slug). Schedule-arm tags (``..._schconstwu0``,
    2026-09-15) likewise yield ``{prefix}lr_scheduler_type`` and ``{prefix}warmup_ratio``
    (the `DEFAULT_SCHEDULE` values for a suffix-less tag; NA/NaN where the tag failed to parse).

    Args:
        df: Frame carrying a tag column.
        col: Name of the column holding ``ep{e}_lr{lr}_r{r}a{a}`` tags.
        prefix: Optional key prefix (mirrors `parse_param_tag`); empty yields bare names.

    Returns:
        A copy of `df` with the four typed columns added, plus ``{prefix}append_eos``
        (False for ``_noeos`` tags, True otherwise), ``{prefix}optim``, ``{prefix}stop_step``,
        ``{prefix}lr_scheduler_type`` and ``{prefix}warmup_ratio``.
    """
    ex = df[col].astype("string").str.extract(PARAM_TAG_RE)
    p = f"{prefix}_" if prefix else ""
    out = df.copy()
    out[f"{p}epochs"] = pd.to_numeric(ex["ep"], errors="coerce").astype(float)
    out[f"{p}lr"] = pd.to_numeric(ex["lr"], errors="coerce").astype(float)
    out[f"{p}rank"] = pd.to_numeric(ex["r"], errors="coerce").astype(float)
    out[f"{p}alpha"] = pd.to_numeric(ex["a"], errors="coerce").astype(float)
    # noeos ablation arm (tag suffix `_noeos`): False there, True for every canonical tag
    # (NaN only where the tag itself failed to parse) — so tree-wide frames that pick up
    # noeos cells can drop them with ``out[f"{p}append_eos"] != False``.
    out[f"{p}append_eos"] = ex["noeos"].isna().where(ex["ep"].notna())
    # Optimizer slug (SGD teacher/student arms, 2026-09-14) and early-stop budget, additive
    # like append_eos: suffix-less tags read as the default optimizer / no stop.
    out[f"{p}optim"] = ex["optim"].fillna(DEFAULT_OPTIM).where(ex["ep"].notna()).astype("string")
    out[f"{p}stop_step"] = pd.to_numeric(ex["stop"], errors="coerce").astype(float)
    # LR-schedule arm (2026-09-15), same additive pattern: a suffix-less tag reads as the
    # canonical cosine + 5% warm-up recipe, so `canonical_only` can require both defaults.
    parsed = ex["sched"].map(
        lambda s: parse_schedule_slug(s) if isinstance(s, str) else None
    )
    out[f"{p}lr_scheduler_type"] = (
        parsed.map(lambda v: v[0] if v else None)
        .fillna(DEFAULT_SCHEDULE[0]).where(ex["ep"].notna()).astype("string")
    )
    out[f"{p}warmup_ratio"] = (
        pd.to_numeric(parsed.map(lambda v: v[1] if v else None), errors="coerce")
        .fillna(DEFAULT_SCHEDULE[1]).where(ex["ep"].notna()).astype(float)
    )
    return out


def canonical_only(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Rows of an `add_param_fields` frame that are canonical cells: parsed tag, EOS on, no
    early-stop budget, default optimizer, and the default (cosine, 5% warm-up) LR schedule
    (2026-09-15) -- the frame-level twin of `param_tags.is_canonical_tag`.
    """
    p = f"{prefix}_" if prefix else ""
    keep = (df[f"{p}epochs"].notna()
            & (df[f"{p}append_eos"] != False)  # noqa: E712  (nullable boolean)
            & df[f"{p}stop_step"].isna()
            & (df[f"{p}optim"] == DEFAULT_OPTIM)
            & (df[f"{p}lr_scheduler_type"] == DEFAULT_SCHEDULE[0])
            & (df[f"{p}warmup_ratio"] == DEFAULT_SCHEDULE[1]))
    return df[keep.fillna(False).astype(bool)]


def load_results(
    paths,
    *,
    key: str = "results",
    per_file: Callable[[Path, dict], dict] | None = None,
) -> pd.DataFrame:
    """Load one or more ``{"results": [...]}`` analysis-JSON files into a single tidy frame.

    The consolidated loader for the analysis-JSON shape used 25+ times across the notebooks: each
    file is a dict with a `key` list of per-row records. Unreadable/partial files are skipped
    (counted), and a single `warnings.warn` names the skipped count and the first offender.

    This function is glob-free and order-preserving: `paths` is taken as-is, so callers that want a
    deterministic order should pass ``sorted(dir.glob(...))`` (the notebooks do their own globbing).

    Args:
        paths: A single path, or an iterable of paths. A ``str``/``Path`` is treated as one file,
            not iterated character-by-character.
        key: The key whose list-of-dicts is the rows (default ``"results"``).
        per_file: Optional ``callable(path, doc) -> dict`` merged into every row from that file --
            the "attach ``scale_lambda``/``ftseed`` parsed from the filename" idiom. The returned
            keys become columns; existing row keys are not overwritten.

    Returns:
        One DataFrame concatenating every file's rows (with any `per_file` columns), preserving the
        order of `paths`. An empty DataFrame when nothing loads.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames: list[pd.DataFrame] = []
    skipped: list[Path] = []
    for path in paths:
        path = Path(path)
        doc = safe_read_json(path)
        if doc is None:
            skipped.append(path)
            continue
        rows = doc.get(key, [])
        if not rows:
            continue
        sub = pd.DataFrame(rows)
        if per_file is not None:
            extra = per_file(path, doc)
            for k, v in extra.items():
                if k not in sub.columns:
                    sub[k] = v
        frames.append(sub)
    if skipped:
        warnings.warn(
            f"load_results skipped {len(skipped)} unreadable file(s); first: {skipped[0]}",
            stacklevel=2,
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_cell_set(cell_dir, prompt_set: str = "test") -> dict | None:
    """Return one prompt set's across-seed summary block for a single swept cell.

    The across-seed artifact ``{cell_dir}/preference_summary.json`` is the canonical
    "all the seeds this cell has" record: its per-set blocks carry ``p_target_mean``,
    ``n_seeds`` and the per-seed map ``per_seed_p_target`` (``{seed: p_target}``), written
    from the very same evaluations as the individual ``ftseed{N}/preference_eval.json``
    files. Reading it costs one file per cell instead of one per seed, and -- unlike
    hard-coding ``ftseed1`` -- it picks up every seed a still-draining campaign has landed.

    Args:
        cell_dir: The ``ep{e}_lr{lr}_r{r}a{a}`` directory of one config.
        prompt_set: Which eval bank's block to return (``test``, ``validation``, ...).

    Returns:
        The set's summary dict, or ``None`` when the summary file is missing/unreadable or
        holds no block for `prompt_set` (i.e. the cell is not evaluated yet).
    """
    summary_path = Path(cell_dir) / "preference_summary.json"
    doc = safe_read_json(summary_path)
    if doc is None:
        return None
    return next((s for s in doc.get("summaries", []) if s.get("prompt_set") == prompt_set), None)


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
    from subliminal_learning.analysis.record_cache import load_record_cache

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


#: Cache CSV (relative to a queue dir) for that queue's terminal jobq result records.
QUEUE_RESULTS_CACHE_REL = Path("analysis") / "jobq_results_cache.csv"


def load_queue_results(queue_dir, refresh: bool = True) -> dict[str, dict]:
    """Batched terminal jobq result records of one queue: job key -> parsed doc.

    The notebooks' "which cells of this campaign have landed" sweep reads every
    ``{queue}/results/*.json`` written by `jobqueue.store.record_result`. Those are many small
    files in ONE directory rather than one record per keyed directory, so this keys the cache on
    each *result file* (``rel_record="."`` = the keyed path is itself the record) and gets the
    whole `analysis.record_cache` contract -- ``(mtime, size)`` staleness, union semantics,
    atomic write, per-record read guards -- for free. Discovery stays a glob here because the
    queue owns the file set. Unreadable/partial records are simply absent.

    Args:
        queue_dir: The queue directory (holding ``results/`` and, after this call, the cache CSV
            at `QUEUE_RESULTS_CACHE_REL`).
        refresh: Passed through to the cache (False = stored rows only, no filesystem sweep).

    Returns:
        ``{jobkey: result doc}`` -- the file stem mapped to the record's ``key``/``rc``/``node``/
        timing fields.
    """
    from subliminal_learning.analysis.record_cache import load_record_cache

    queue_dir = Path(queue_dir)
    rel_paths = [f"results/{p.name}" for p in sorted(queue_dir.glob("results/*.json"))]
    out = load_record_cache(
        queue_dir, "json_blob", rel_paths, refresh=refresh, rel_record=".",
        cache_rel=QUEUE_RESULTS_CACHE_REL, verbose=False,
    )
    docs: dict[str, dict] = {}
    for rp, blob in zip(out["run_path"], out["blob"]):
        try:
            docs[Path(str(rp)).stem] = json.loads(blob)
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


def load_preference_evals(root, run_rel_paths, refresh: bool = True) -> dict[str, dict]:
    """Batched per-seed ``preference_eval.json`` docs: rel run path -> parsed doc.

    Prefer the per-cell `load_cell_summaries` when all you need is per-seed ``p_target``: the
    cell's ``preference_summary.json`` carries ``per_seed_p_target`` for every landed seed, so
    it costs one record per cell instead of one per seed. Reach for this loader when you need
    the per-seed document itself (per-prompt detail, generation metadata, ...).

    Args:
        root: Root the relative run paths (and the cache CSV) resolve against.
        run_rel_paths: ``ftseed{N}`` run directories relative to `root`.
        refresh: Passed through to the cache.

    Returns:
        ``{rel_path: doc}`` for every run with a parseable eval record.
    """
    return load_run_docs(root, run_rel_paths, "preference_eval.json", "pref_eval",
                         refresh=refresh)


def load_steerability_aggregates(output_root, agg_rel_dirs, refresh: bool = True) -> dict[str, dict]:
    """Batched steerability ``aggregate.json`` docs: rel dir -> parsed doc.

    Notebooks discover these with ``{output_root}/teachers/**/steerability/steer_*/`` globs; keep
    the glob and pass the ``steer_*`` directories (relative to `output_root`) here so the reads
    go through one cached sweep.

    Args:
        output_root: ``paths.output_root`` (the cache CSV lands in its ``analysis/``).
        agg_rel_dirs: ``steer_*`` directories relative to `output_root`.
        refresh: Passed through to the cache.

    Returns:
        ``{rel_dir: doc}`` for every directory with a parseable aggregate.
    """
    return load_run_docs(output_root, agg_rel_dirs, "aggregate.json", "steerability_agg",
                         refresh=refresh)


def summary_set_block(doc: dict | None, prompt_set: str = "test") -> dict | None:
    """The ``summaries`` block of `doc` for `prompt_set` -- `read_cell_set`'s selection step.

    Split out so batched readers (`load_cell_summaries`) pick the same block a per-cell
    `read_cell_set` would.
    """
    if doc is None:
        return None
    return next((s for s in doc.get("summaries", []) if s.get("prompt_set") == prompt_set), None)


def read_cell_seeds(cell_dir, prompt_set: str = "test") -> dict[int, float]:
    """Per-seed ``p_target`` for one cell as ``{ft_seed: p_target}`` (empty when unevaluated).

    Thin typed view over `read_cell_set`'s ``per_seed_p_target``: seeds come back as ints so
    callers can sort/join on them. ``len(...)`` is the cell's usable seed count -- the ``n`` to
    print alongside a seed-averaged number.
    """
    block = read_cell_set(cell_dir, prompt_set)
    per_seed = (block or {}).get("per_seed_p_target") or {}
    return {int(s): float(v) for s, v in per_seed.items()}


def short_model(name: str) -> str:
    """``pythia-160m`` -> ``original``; strip the ``pythia-160m-`` prefix otherwise.

    ``pythia-160m-seed2`` (fully re-randomized) -> ``full-seed2``, matching ``spec_label``:
    a bare ``seed2`` would read as a data-/weight-seed arm.

    Unknown names pass through unchanged. A non-160m Pythia name (e.g. ``pythia-410m``)
    or a DataDecide/OLMo name also passes through, but with a warning: every downstream
    frame/join in this module keys on the 160m short labels, so such rows would silently
    match nothing.
    """
    if name == "pythia-160m":
        return "original"
    prefix = "pythia-160m-"
    if name.startswith(prefix):
        rest = name[len(prefix):]
        return f"full-{rest}" if rest.startswith("seed") else rest
    if name.removeprefix("EleutherAI/").startswith("pythia-") or name.startswith("DataDecide"):
        warnings.warn(
            f"short_model: {name!r} is not a pythia-160m-family name; passing it "
            "through unchanged. 160m-keyed joins will silently drop these rows — "
            "extend the label mapping before analyzing non-160m runs.",
            stacklevel=2,
        )
    return name


def add_lora_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``lora_label`` column (``f"r{lora_r}/α{lora_alpha}"``) as an ordered Categorical.

    The category order is sorted numerically by ``(lora_r, lora_alpha)`` so downstream sorts and
    heatmap axes come out ``r8/α16 < r8/α32 < r16/α8`` -- killing the string-parsing sort lambdas
    the notebooks used to carry. Returns a copy; the input is not mutated.

    NaN-tolerant: rows whose ``lora_r`` or ``lora_alpha`` is NaN (e.g. pools without a teacher
    param tag from the manifest scan) get a NaN ``lora_label`` and never enter the category set --
    the categories are built from the finite ``(lora_r, lora_alpha)`` pairs only. ``pd.Categorical``
    maps the ``None`` labels those rows produce to NaN naturally.
    """
    df = df.copy()
    pairs = sorted(
        {
            (int(r), int(a))
            for r, a in zip(df["lora_r"], df["lora_alpha"])
            if pd.notna(r) and pd.notna(a)
        }
    )
    categories = [f"r{r}/α{a}" for r, a in pairs]
    labels = [
        f"r{int(r)}/α{int(a)}" if pd.notna(r) and pd.notna(a) else None
        for r, a in zip(df["lora_r"], df["lora_alpha"])
    ]
    df["lora_label"] = pd.Categorical(labels, categories=categories, ordered=True)
    return df


def filter_grid(
    df: pd.DataFrame,
    epochs: list | None = None,
    lrs: list | None = None,
    ranks: list | None = None,
) -> pd.DataFrame:
    """Filter on ``n_epochs`` / ``learning_rate`` / ``lora_r``; any ``None`` axis is left alone.

    Learning-rate matching is float-tolerant (``np.isclose`` against `lrs`), so an LR stored as
    ``0.0002000000001`` still matches ``2e-4``. Returns a copy; the input is not mutated.
    """
    mask = pd.Series(True, index=df.index)
    if epochs is not None:
        mask &= df["n_epochs"].isin(epochs)
    if ranks is not None:
        mask &= df["lora_r"].isin(ranks)
    if lrs is not None:
        lr = df["learning_rate"].to_numpy(dtype=float)
        lr_mask = np.zeros(len(df), dtype=bool)
        for target in lrs:
            lr_mask |= np.isclose(lr, float(target))
        mask &= pd.Series(lr_mask, index=df.index)
    return df[mask].copy()


def pref_to_wide(pref: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long preference table to one row per config, with per-set columns.

    Reproduces the notebook wide-pivot exactly: ``p_target`` pivoted by ``set``, the set-specific
    base controls pivoted by ``set`` with a ``base_`` prefix (they are not interchangeable), and
    ``base_rank_mean`` / ``rank_mean`` merged from the ``favorite`` rows. Derived columns are added
    only where their source set columns exist (silently skipped otherwise):
    ``off_topic_delta``, ``specificity_margin``, ``storytelling_delta``. A ``short_model`` column
    is added. This is pure -- no disk, no matplotlib.
    """
    dup = pref.duplicated(_WIDE_KEYS + ["set"])
    assert not dup.any(), (
        "duplicate (KEYS, set) rows would be silently mean-averaged by pivot_table: "
        f"{pref[pref.duplicated(_WIDE_KEYS + ['set'], keep=False)][_WIDE_KEYS + ['set']].to_dict('records')[:10]}"
    )
    wide = (
        pref.pivot_table(index=_WIDE_KEYS, columns="set", values="p_target", observed=True)
        .reset_index()
    )
    base_wide = (
        pref.pivot_table(
            index=_WIDE_KEYS, columns="set", values="base_p_target", observed=True
        )
        .add_prefix("base_")
        .reset_index()
    )
    wide = wide.merge(base_wide, on=_WIDE_KEYS, how="left")
    fav = pref[pref["set"] == "favorite"][
        ["animal", "model", "param_tag", "base_rank_mean", "rank_mean"]
    ]
    wide = wide.merge(fav, on=["animal", "model", "param_tag"], how="left")

    if "off_topic" in wide and "base_off_topic" in wide:
        wide["off_topic_delta"] = wide["off_topic"] - wide["base_off_topic"]
    if "validation" in wide and "off_topic" in wide:
        wide["specificity_margin"] = wide["validation"] - wide["off_topic"]
    if "storytelling" in wide and "base_storytelling" in wide:
        wide["storytelling_delta"] = wide["storytelling"] - wide["base_storytelling"]

    wide["short_model"] = wide["model"].map(short_model)
    return wide


def load_pref_wide(
    output_root, ranks: tuple[int, ...] = (8, 16, 32, 64)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load preference results and return ``(pref_long, pref_wide)``.

    Thin wrapper: `load_preference_results`, restrict ``lora_r`` to `ranks` (legacy r1/r2/r4
    artifacts stay on disk), `add_lora_label`, then `pref_to_wide`.
    """
    pref = load_preference_results(output_root)
    pref = pref[pref["lora_r"].isin(ranks)].copy()
    pref = add_lora_label(pref)
    return pref, pref_to_wide(pref)


def scope_str(df: pd.DataFrame) -> str:
    """One-line description of what a frame aggregates over (robust to missing cols / empty df)."""
    na = df["animal"].nunique() if "animal" in df else 0
    nm = df["model"].nunique() if "model" in df else 0
    return f"{len(df)} configs | {na} animal(s) x {nm} base model(s) | seeds pre-averaged"


def print_scope(df: pd.DataFrame, note: str = "") -> None:
    """Print ``SCOPE: {scope_str(df)}``, appending ``| {note}`` when a note is given."""
    line = "SCOPE: " + scope_str(df)
    if note:
        line += f" | {note}"
    print(line)


def cheapest_best(
    df: pd.DataFrame,
    metric: str,
    *,
    lower_is_better: bool = False,
    tol: float = 0.0,
) -> pd.Series:
    """Return the cheapest config whose `metric` is within `tol` of the optimum.

    The optimum is the max of `metric` (or the min when `lower_is_better`); every config within
    `tol` of it is a candidate, and the cheapest -- ascending on ``(n_epochs, learning_rate,
    lora_r, lora_alpha)`` -- wins. Rows with a NaN `metric` are excluded. This is the single
    definition of "best config" for the notebooks.

    Args:
        df: Config-level frame (one row per config).
        metric: Column to optimize.
        lower_is_better: Minimize `metric` instead of maximizing it.
        tol: Slack around the optimum admitting cheaper near-optimal configs.

    Returns:
        The winning config as a Series.

    Raises:
        ValueError: if no row has a non-NaN `metric`.
    """
    valid = df[df[metric].notna()]
    if valid.empty:
        raise ValueError(f"cheapest_best: no non-NaN values for metric {metric!r}")
    if lower_is_better:
        optimum = valid[metric].min()
        within = valid[valid[metric] <= optimum + tol]
    else:
        optimum = valid[metric].max()
        within = valid[valid[metric] >= optimum - tol]
    return within.sort_values(_COST_KEYS).iloc[0]


def rank_specific(
    df: pd.DataFrame,
    target: str = "validation",
    *,
    controls: Sequence[str] = ("off_topic",),
    ctrl_max: float = 0.1,
    rank_decimals: int = 2,
) -> pd.DataFrame:
    """Eligible configs ranked best-first by a cost-aware, near-tie-tolerant key.

    The teacher-selection criterion for §2b: a good teacher's preference should **generalize**
    (high `target`, i.e. ``validation``) yet be **specific** -- it must not fire off-distribution.
    So we keep only configs whose every `controls` column (lower-is-better; the off-distribution
    eval ``off_topic`` = unconditional token leakage) sits at/below `ctrl_max`, then rank them by
    `target` **rounded to `rank_decimals` decimals** (descending) so near-identical scores tie --
    e.g. validation ``0.999`` and ``0.998`` both round to ``1.00`` and share a rank -- with ties
    broken by **cheapness** (ascending ``(n_epochs, learning_rate, lora_r, lora_alpha)``). The
    rounding drives ranking ONLY; the returned rows keep their full-precision columns. This favours
    the minimal HP that installs a *specific* trait over the most aggressive corner (which edges out
    `target` in the noise while costing more or leaking off-distribution). A config is eligible only
    when `target` and every control are non-NaN.

    Args:
        df: Config-level frame (one row per config).
        target: Column to maximize (higher is better).
        controls: Lower-is-better off-distribution columns each constrained to ``<= ctrl_max``.
        ctrl_max: The "low" threshold applied to every control.
        rank_decimals: Decimals `target` is rounded to for ranking (bigger => finer, fewer ties).

    Returns:
        A copy of the eligible rows sorted best-first, with an integer ``rank`` column (dense rank
        on the rounded `target`; ties share a rank). Empty (with a ``rank`` column) if none qualify.
        The first row is the selected teacher: rank 1, cheapest among the ties.
    """
    valid = df.dropna(subset=[target, *controls]).copy()
    for c in controls:
        valid = valid[valid[c] <= ctrl_max]
    if valid.empty:
        valid["rank"] = pd.Series(dtype=int)
        return valid
    valid = valid.assign(_q=valid[target].round(rank_decimals)).sort_values(
        ["_q", *_COST_KEYS], ascending=[False, *([True] * len(_COST_KEYS))]
    )
    valid["rank"] = valid["_q"].rank(method="dense", ascending=False).astype(int)
    return valid.drop(columns="_q")


def cheapest_specific(
    df: pd.DataFrame,
    target: str = "validation",
    *,
    controls: Sequence[str] = ("off_topic",),
    ctrl_max: float = 0.1,
    rank_decimals: int = 2,
) -> pd.Series | None:
    """Selected teacher: the rank-1, cheapest row of `rank_specific`, or ``None`` if none qualify.

    Thin wrapper over `rank_specific` (see it for the full criterion): among configs whose
    off-distribution `controls` stay ``<= ctrl_max``, the one with the highest `target` rounded to
    `rank_decimals` decimals, ties broken by cheapness.
    """
    ranked = rank_specific(
        df, target, controls=controls, ctrl_max=ctrl_max, rank_decimals=rank_decimals
    )
    return None if ranked.empty else ranked.iloc[0]


def read_jsonl(path) -> list[dict]:
    """Parse a JSON-lines file into a list of dicts (blank lines skipped).

    Args:
        path: Path to the ``.jsonl`` file.

    Returns:
        One parsed object per non-blank line, in file order.
    """
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def read_cell_manifest(path) -> list[str]:
    """Student tags of a ``run_logs/`` campaign manifest, whatever shape it was written in.

    The manifests accumulated four shapes as campaigns were added, and every notebook grew its
    own accessor for the one it happened to read. The shapes, in resolution order:

    - ``{"cells": [...]}`` -- elements are tags, or dicts carrying ``student_tag``/``tag``
      (focus20 / core30 / bandext126).
    - ``{"cells": {tag: seeds}}`` -- a registry mapping; the keys are the tags (p410m strips).
    - ``{"rows": [{...}]}`` -- row dicts carrying ``student_tag``/``tag``.
    - ``{"param_tags": [...]}`` -- a bare tag list (teacher-config manifests).
    - A ``.jsonl`` of ``{"key": ...}`` job rows -- the key's ``|``-parts are *not* interpreted
      here (their layout is per queue); use `read_queue_jobs` for those.

    Args:
        path: Path to the manifest (``.json`` or ``.jsonl``).

    Returns:
        The student tags, in manifest order (deduplicated is *not* applied -- a manifest that
        repeats a cell says so).

    Raises:
        ValueError: When the file matches none of the shapes above. Guessing here would let a
            new manifest silently analyse zero cells.
    """
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
        if rows and all("key" in r for r in rows):
            return [r["key"] for r in rows]
        raise ValueError(f"unrecognized jsonl manifest shape: {path}")

    doc = read_json(path)
    if isinstance(doc, list):
        entries = doc
    elif isinstance(doc.get("cells"), dict):
        return list(doc["cells"])
    elif isinstance(doc.get("cells"), list):
        entries = doc["cells"]
    elif isinstance(doc.get("rows"), list):
        entries = doc["rows"]
    elif isinstance(doc.get("param_tags"), list):
        entries = doc["param_tags"]
    else:
        raise ValueError(
            f"unrecognized cell-manifest shape: {path} (top-level keys: {sorted(doc)})"
        )

    tags = []
    for e in entries:
        if isinstance(e, str):
            tags.append(e)
            continue
        tag = e.get("student_tag", e.get("tag")) if isinstance(e, dict) else None
        if tag is None:
            raise ValueError(f"manifest entry has no student_tag/tag: {path}: {e!r}")
        tags.append(tag)
    return tags


def read_queue_jobs(jobs_jsonl_path) -> list[dict]:
    """The rows of a jobq queue's ``jobs.jsonl`` (each carries a ``"key"``).

    The key is a ``|``-joined tuple whose *layout differs per queue* (teacher / cell / seed in
    one order for the strips, another for the seed2seed queues), so this returns the rows as
    parsed and leaves the split to the call site -- baking positions in here would silently
    mis-key any queue built to a different layout.

    Args:
        jobs_jsonl_path: Path to ``{queue}/jobs.jsonl``.

    Returns:
        The parsed job rows, in queue order.
    """
    return read_jsonl(jobs_jsonl_path)


def load_base_prior(output_root, base_short: str, animal: str) -> float:
    """A base model's mean exact ``p(animal)`` on the held-out test prompts.

    The DEAD reference every transfer number is divided by: the mean over prompts of
    ``analysis/base_eval_{base_short}_animal_preference_test.json`` -> ``[animal]
    ["p_per_prompt"]``. `base_short` is the short base label used in that filename
    (``original``, ``ds1``, ``ws1``, ...), not the HF model id.

    Args:
        output_root: ``paths.output_root``.
        base_short: Short base label embedded in the artifact's filename.
        animal: Trait animal.

    Returns:
        The mean per-prompt probability.

    Raises:
        KeyError: When `animal` is absent, listing the animals the file does carry.
    """
    path = Path(output_root) / "analysis" / f"base_eval_{base_short}_animal_preference_test.json"
    doc = read_json(path)
    if animal not in doc:
        raise KeyError(f"{animal!r} not in {path} (available: {sorted(doc)})")
    return float(np.mean(doc[animal]["p_per_prompt"]))
