"""Generic mtime/size cache over the small per-run JSON eval records.

Analysis notebooks fan out over 10-20k run dirs and read one tiny JSON per run from a slow
shared filesystem. The run-dir records are the SSOT; a cache CSV under the outputs root is a
derived aggregate so a notebook execution pays one *stat* per inventory path instead of one
*read*. The stats (and the re-reads they trigger) run through a thread pool: on the cold-1
CephFS one stat costs ~5.5 ms serially, so a 50k-path sweep is ~5 min single-threaded and
well under a minute with 64 I/O threads (measured 10k stats: 32 threads 6.8 s, 64 3.0 s,
128 2.9 s). This module generalises the original `analysis/probe_cache.py` (pile-probe only) to a
registry of record KINDS: each kind names the record file relative to a keyed directory, an
extractor from the record dict to flat scalar columns, and the cache CSV it persists to.

Refresh contract:

- **new records are added automatically**: every refresh stats each inventory path and reads
  any record not yet cached;
- **changed records are re-read**: staleness is ``(mtime, size)`` -- either changing re-reads.
  A recomputed record (e.g. an eval rewritten after a retrain) therefore never goes unnoticed
  even in the pathological same-mtime case;
- **the persisted CSV is a UNION**: previous rows are kept unless this call's inventory
  covered the path *and* its record has provably disappeared (`FileNotFoundError`). Rows for
  paths outside this call's inventory are never evicted, so two notebooks with disjoint
  inventories cannot destroy each other's work. The RETURNED frame is always filtered to the
  requested inventory; the SAVED file keeps everything.

Concurrency: the outputs root is shared across two nodes and two users, and the CSV is
written through `utils.io.atomic_write_text` (temp + ``os.replace``), so a reader never sees a
torn file. **No lock is taken.** Under union semantics a lost update is benign: the loser
merely fails to persist rows it had just read, and re-reads them on its next refresh. Record
writers are themselves atomic (`utils.io.atomic_write_json`), so a partially written record is
never observed either -- an unparseable record means genuine corruption, and is handled by
keeping whatever row was previously cached.

Read errors are guarded per record: a vanished record drops its row, anything else (transient
`OSError`, corrupt JSON, an unexpected shape) keeps the previously cached row if there is one
and otherwise skips the path. Counts of both appear in the one-line summary print.

Two details keep a warm sweep a true no-op (measured 2026-09-10 on a 40-call notebook that
was spending ~40% of its time rewriting unchanged caches):

- the CSV is parsed with ``float_precision="round_trip"``. pandas' default parser moves ~12%
  of stored ``record_mtime`` values by one ulp, and with the exact-mtime staleness contract
  every such row read as changed on EVERY sweep -- re-read from disk, counted as refreshed,
  and forcing a full rewrite of the cache on every call;
- the parsed cache is memoised per process, keyed on the cache file's ``(mtime_ns, size)``, so
  forty loader calls in one notebook parse a 200 MB CSV once, and a rewrite by another
  process (a different stat) still invalidates it.
"""
from __future__ import annotations

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from subliminal_learning.utils.io import atomic_write_text

# Compare filesystem mtimes exactly. A small tolerance would miss rapid same-size rewrites.
# Exactness relies on the CSV round-tripping the float bit-for-bit -- see `_read_cache_csv`.
_MTIME_TOL = 0.0

# I/O threads for the stat/read sweep (overridable through RECORD_CACHE_SWEEP_WORKERS). The
# work is network-filesystem latency, not CPU, so this is independent of OMP/BLAS thread caps.
_SWEEP_WORKERS = int(os.environ.get("RECORD_CACHE_SWEEP_WORKERS", "64"))

# Per-process memo of parsed cache CSVs: path -> ((mtime_ns, size), frame). Frames handed out
# are copies, so callers' in-place column migrations never leak back into the memo.
_CSV_MEMO: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}

#: Columns every kind's frame carries in addition to its own.
BASE_COLS = ["run_path", "record_mtime", "record_size"]


@dataclass(frozen=True)
class RecordKind:
    """One cacheable record type.

    Attributes:
        name: Registry key (e.g. ``"pile_probe"``).
        rel_record: Record file path relative to a keyed directory (e.g.
            ``"eval/pile_probe.json"``). May be overridden per call for parameterized kinds
            such as ``json_blob``.
        columns: Scalar column names the extractor produces, in order.
        extract: ``record -> dict`` of `columns`. Receives the parsed JSON for JSON kinds, or
            the raw text for `raw_text` kinds. Raising anything is treated as a read failure
            (previous row kept, path counted as an error).
        cache_rel: Cache CSV path relative to the outputs root.
        raw_text: When True the record is not JSON-parsed; the extractor gets ``str``.
    """

    name: str
    rel_record: str
    columns: list[str]
    extract: Callable[[object], dict] = field(repr=False)
    cache_rel: Path = field(default=Path("."))
    raw_text: bool = False


_REGISTRY: dict[str, RecordKind] = {}


def register_kind(kind: RecordKind) -> RecordKind:
    """Add `kind` to the registry (replacing any same-named kind) and return it."""
    _REGISTRY[kind.name] = kind
    return kind


def get_kind(name: str) -> RecordKind:
    """The registered kind `name`; raises KeyError listing the known kinds if absent.

    ``pile_probe:<pool>`` (e.g. ``pile_probe:cp1``) is the parameterized spelling of the
    CROSS-CORPUS probe sidecar written by `scripts.run_final_evals --probe-pools`: same
    columns and extractor, record path ``eval/pile_probe.<pool>.json`` and its own cache
    file, so a sidecar pool never mixes with the native `pile_probe` rows.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        pass
    base, _, pool = name.partition(":")
    if pool and base == "pile_probe":
        return dataclasses.replace(
            _REGISTRY[base], name=name,
            rel_record=f"eval/pile_probe.{pool}.json",
            cache_rel=Path("analysis") / f"probe_cache_{pool}.csv")
    raise KeyError(f"unknown record kind {name!r}; known: {sorted(_REGISTRY)}") from None


# --------------------------------------------------------------------------------------------------
# Extractors
# --------------------------------------------------------------------------------------------------

def _f(x) -> float:
    """`x` as float, with None/missing -> NaN."""
    return float("nan") if x is None else float(x)


def _extract_pile_probe(rec: dict) -> dict:
    """`coherence.probe_io` probe record -> flat columns.

    A diverged adapter (``nonfinite_adapter``, or a null ``kl_base_student``) yields ``+inf``
    for both `kl` and `delta_ce` -- "dead by any cut", the same convention
    `transfer_factors.load_focus_pile_kl` uses. Never NaN: NaN means *not measured yet*, and
    the two must stay distinguishable.
    """
    ref = rec.get("manifest_ref") or {}
    kl = rec.get("kl_base_student")
    nonfinite = bool(rec.get("nonfinite_adapter")) or kl is None
    return {
        "kl": float("inf") if nonfinite else float(kl),
        "delta_ce": float("inf") if nonfinite else _f(rec.get("delta_ce")),
        # absolute CEs (added 2026-08-20 for teacher-tree aggregation). Every writer has
        # always recorded both, so a schema migration re-read fills them for old rows.
        "ce_student": _f(rec.get("ce_student")),
        "ce_base": _f(rec.get("ce_base")),
        "nonfinite": nonfinite,
        "pool_version": rec.get("pool_version"),
        "pool_sha": rec.get("pool_sha"),
        "schema_version": rec.get("schema_version"),
        "student_param_tag": ref.get("student_param_tag"),
        "filtered_sha256": ref.get("filtered_sha256"),
        "n_train_steps": ref.get("n_train_steps"),
    }


def _extract_fidelity_heldout(rec: dict) -> dict:
    """Train/held-out CE record -> flat columns; null CEs are NaN (attempted but diverged)."""
    return {
        "ce_heldout": _f(rec.get("ce_student")),
        "ce_base": _f(rec.get("ce_base")),
        "ce_train": _f(rec.get("ce_train")),
        "ce_train_base": _f(rec.get("ce_train_base")),
        "nonfinite": bool(rec.get("nonfinite_adapter")),
    }


def _extract_fidelity_shuffle(rec: dict) -> dict:
    # the shuffle writer (`scripts.run_final_evals`) names the field `ce_shuffled`
    return {"ce_shuffled": _f(rec.get("ce_shuffled"))}


def _extract_fidelity_traj(rec: dict) -> dict:
    """Fidelity ladder -> the derived per-run scalars the claims notebook computes.

    Over the checkpoints sorted by ``step`` and restricted to those with both CEs finite:
    ``max_rise`` = the largest amount train-CE ever climbs above its own running minimum
    ("had it, then lost it"), ``gap_max`` = the largest held-out minus train gap
    (memorization at some point), ``train_min`` = the best train CE reached. ``n_points`` is
    the finite-point count: runs with too few points (< 5 in the notebook) still get a row so
    the caller decides, rather than the cache erroring or silently dropping them.
    """
    pts = sorted(rec.get("points") or [], key=lambda p: p["step"])
    trn, hld = [], []
    for p in pts:
        a, b = _f(p.get("ce_train")), _f(p.get("ce_heldout"))
        if math.isfinite(a) and math.isfinite(b):
            trn.append(a)
            hld.append(b)
    if not trn:
        return {"max_rise": float("nan"), "gap_max": float("nan"),
                "train_min": float("nan"), "n_points": 0}
    run_min, max_rise = trn[0], 0.0
    for a in trn:
        run_min = min(run_min, a)
        max_rise = max(max_rise, a - run_min)
    return {
        "max_rise": max_rise,
        "gap_max": max(h - t for h, t in zip(hld, trn)),
        "train_min": min(trn),
        "n_points": len(trn),
    }


def _extract_adapter_norms(rec: dict) -> dict:
    """Per-run adapter-norm record -> the run-level dose scalars (module detail stays in file).

    ``dw_{fro,spec}`` is the relative dose ``mean over modules of s·‖B·A_final‖/‖W₀‖`` with
    ``s = alpha/rank`` -- exactly the harvest-era aggregation the claims and nors_asks
    notebooks compute; ``dW_*`` is the same without the ``W₀`` normalisation (absolute update
    size). Modules missing a norm contribute NaN, so a truncated record is visible rather than
    silently averaged away.
    """
    mods = rec.get("modules") or []
    s = _f(rec.get("alpha")) / _f(rec.get("rank"))
    out = {"n_modules": len(mods), "test_p": _f(rec.get("test_p")),
           "rank": _f(rec.get("rank")), "alpha": _f(rec.get("alpha"))}
    for lvl in ("fro", "spec"):
        dW = [s * _f(m.get(f"B_Afinal_{lvl}")) for m in mods]
        out[f"dW_{lvl}"] = sum(dW) / len(dW) if dW else float("nan")
        dw = [x / _f(m.get(f"W0_{lvl}")) for x, m in zip(dW, mods)]
        out[f"dw_{lvl}"] = sum(dw) / len(dw) if dw else float("nan")
    for c in ("cos_Ainit_Afinal", "cos_wrongseed_Afinal"):
        v = [_f(m.get(c)) for m in mods]
        out[f"{c}_mean"] = sum(v) / len(v) if v else float("nan")
    return out


def _dw(mods, s: float, lvl: str) -> float:
    """Module-mean relative dose ``s·‖B·A_final‖/‖W₀‖`` at level `lvl` (``fro``/``spec``)."""
    v = [s * _f(m.get(f"B_Afinal_{lvl}")) / _f(m.get(f"W0_{lvl}")) for m in mods]
    return sum(v) / len(v) if v else float("nan")


def _extract_adapter_norms_traj(rec: dict) -> dict:
    """Per-checkpoint adapter-norm record -> run-level trajectory scalars.

    Endpoint-free by construction (the final adapter lives in the sibling ``adapter_norms``
    record): ``dw_fro_last`` is the dose at the largest recorded step, ``dw_fro_max`` its
    maximum over the trajectory -- overshoot is visible without reading the points back.
    """
    pts = sorted(rec.get("points") or [], key=lambda p: p["step"])
    s = _f(rec.get("alpha")) / _f(rec.get("rank"))
    dws = [_dw(p.get("modules") or [], s, "fro") for p in pts]
    return {
        "n_steps": len(pts),
        "step_max": float(pts[-1]["step"]) if pts else float("nan"),
        "dw_fro_last": dws[-1] if dws else float("nan"),
        "dw_fro_max": max(dws) if dws else float("nan"),
        "test_p": _f(rec.get("test_p")), "rank": _f(rec.get("rank")),
        "alpha": _f(rec.get("alpha")),
    }


def _extract_cos_w0(rec: dict) -> dict:
    """`scripts.teacher_cos_w0` record -> run-level direction scalars.

    ``mean_cos`` is the module-mean cos(ΔW, W₀) -- the regressor the student mediation
    screen uses; ``pooled_cos`` pools numerator and denominators across modules
    (Σ<ΔW,W₀> / sqrt(Σ‖ΔW‖²·Σ‖W₀‖²)). A nonfinite adapter is flagged (its cosines are
    whatever the harvest recorded, typically null -> NaN).
    """
    mods = rec.get("modules") or {}
    return {
        "mean_cos": _f(rec.get("mean_cos")),
        "pooled_cos": _f(rec.get("pooled_cos")),
        "n_modules": len(mods),
        "nonfinite": bool(rec.get("nonfinite_adapter")),
        "rank": _f(rec.get("rank")),
        "alpha": _f(rec.get("alpha")),
    }


def _extract_pile_coherence_v1(rec: dict) -> dict:
    return {"delta_overall": _f((rec.get("delta") or {}).get("delta_overall"))}


PILE_PROBE = register_kind(RecordKind(
    name="pile_probe",
    rel_record="eval/pile_probe.json",
    columns=["kl", "delta_ce", "ce_student", "ce_base", "nonfinite", "pool_version",
             "pool_sha", "schema_version", "student_param_tag", "filtered_sha256",
             "n_train_steps"],
    extract=_extract_pile_probe,
    # the pre-existing on-disk cache from `analysis/probe_cache.py`, reused as-is
    cache_rel=Path("analysis") / "probe_cache.csv",
))

register_kind(RecordKind(
    name="fidelity_heldout",
    rel_record="eval/fidelity_heldout.json",
    columns=["ce_heldout", "ce_base", "ce_train", "ce_train_base", "nonfinite"],
    extract=_extract_fidelity_heldout,
))

register_kind(RecordKind(
    name="fidelity_shuffle",
    rel_record="eval/fidelity_shuffle.json",
    columns=["ce_shuffled"],
    extract=_extract_fidelity_shuffle,
))

register_kind(RecordKind(
    name="fidelity_traj",
    rel_record="eval/fidelity_traj.json",
    columns=["max_rise", "gap_max", "train_min", "n_points"],
    extract=_extract_fidelity_traj,
))

register_kind(RecordKind(
    name="adapter_norms",
    rel_record="eval/adapter_norms.json",
    columns=["dw_fro", "dw_spec", "dW_fro", "dW_spec", "cos_Ainit_Afinal_mean",
             "cos_wrongseed_Afinal_mean", "n_modules", "test_p", "rank", "alpha"],
    extract=_extract_adapter_norms,
))

register_kind(RecordKind(
    name="adapter_norms_traj",
    rel_record="eval/adapter_norms_traj.json",
    columns=["n_steps", "step_max", "dw_fro_last", "dw_fro_max", "test_p", "rank", "alpha"],
    extract=_extract_adapter_norms_traj,
))

COS_W0 = register_kind(RecordKind(
    name="cos_w0",
    rel_record="eval/cos_w0.json",
    columns=["mean_cos", "pooled_cos", "n_modules", "nonfinite", "rank", "alpha"],
    extract=_extract_cos_w0,
))

register_kind(RecordKind(
    name="pile_coherence_v1",
    rel_record="pile_coherence/v1.json",
    columns=["delta_overall"],
    extract=_extract_pile_coherence_v1,
))

register_kind(RecordKind(
    name="json_blob",
    # parameterized: every call passes `rel_record=` for the file it wants
    rel_record="record.json",
    columns=["blob"],
    extract=lambda text: {"blob": text},
    raw_text=True,
))


# --------------------------------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------------------------------

def cache_path(root: Path, kind: RecordKind, cache_rel: Path | str | None = None) -> Path:
    """Where `kind`'s cache CSV lives under `root` (default ``analysis/record_cache_{kind}.csv``)."""
    rel = cache_rel if cache_rel is not None else kind.cache_rel
    if rel is None or Path(rel) == Path("."):
        rel = Path("analysis") / f"record_cache_{kind.name}.csv"
    return Path(root) / rel


def _dedupe(rel_paths) -> list[str]:
    out, seen = [], set()
    for p in rel_paths:
        s = str(p).strip().rstrip("/")
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _file_key(f: Path) -> tuple[int, int] | None:
    """``(mtime_ns, size)`` of `f`, or None when it does not exist."""
    try:
        st = f.stat()
    except FileNotFoundError:
        return None
    return int(st.st_mtime_ns), int(st.st_size)


def _read_cache_csv(cache_f: Path, cols: list[str]) -> pd.DataFrame:
    """The cache CSV as a frame (empty with `cols` when absent), via the per-process memo.

    ``float_precision="round_trip"`` is what makes the exact ``record_mtime`` comparison in
    `_fresh` sound: the default parser perturbs ~12% of stored mtimes by one ulp.
    """
    key = _file_key(cache_f)
    if key is None:
        return pd.DataFrame(columns=cols)
    hit = _CSV_MEMO.get(str(cache_f))
    if hit is not None and hit[0] == key:
        return hit[1].copy()
    frame = pd.read_csv(cache_f, float_precision="round_trip")
    _CSV_MEMO[str(cache_f)] = (key, frame)
    return frame.copy()


def _serialize_csv(frame: pd.DataFrame) -> str:
    """`frame` as CSV text. pyarrow's writer when importable (7x faster than pandas on the
    224 MB blob cache: 0.4 s vs 3.5 s; output verified cell-identical through
    `_read_cache_csv`, inf/NaN/multi-line blobs included), pandas otherwise."""
    try:
        import pyarrow as pa
        import pyarrow.csv as pcsv
    except ImportError:                      # transitive dependency only; never required
        return frame.to_csv(index=False)
    try:
        buf = pa.BufferOutputStream()
        pcsv.write_csv(pa.Table.from_pandas(frame, preserve_index=False), buf)
        return buf.getvalue().to_pybytes().decode("utf-8")
    except (pa.ArrowException, TypeError, ValueError):
        # Arrow needs one type per column; an object column mixing e.g. int and str (a
        # malformed record's raw field) fails conversion. pandas writes anything, and a
        # slower persist is always preferable to a loader that raises.
        return frame.to_csv(index=False)


def _write_cache_csv(cache_f: Path, frame: pd.DataFrame) -> None:
    """Atomically persist `frame` and memoise it under the file's new stat."""
    atomic_write_text(cache_f, _serialize_csv(frame))
    key = _file_key(cache_f)
    if key is not None:
        _CSV_MEMO[str(cache_f)] = (key, frame.copy())


def load_record_cache(
    root: Path,
    kind: str | RecordKind,
    rel_paths,
    refresh: bool = True,
    *,
    rel_record: str | None = None,
    cache_rel: Path | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Cached rows for `rel_paths`, refreshed in place against the filesystem.

    Args:
        root: Outputs root the relative paths and the cache CSV resolve against.
        kind: Registered kind name (or a `RecordKind`).
        rel_paths: Iterable of *directory* paths relative to `root` -- the inventory to keep
            covered. Duplicates are collapsed order-preservingly. Paths whose record does not
            exist yet are simply absent from the frame and picked up on a later refresh.
        refresh: When False, return the stored cache filtered to the inventory, with no
            filesystem sweep and no write.
        rel_record: Override the kind's record path relative to each keyed directory --
            required for the parameterized ``json_blob`` kind, where it usually goes with a
            distinct `cache_rel`.
        cache_rel: Override the cache CSV path (relative to `root`).
        verbose: Print the one-line refresh summary.

    Returns:
        One row per inventory path that has a record: the kind's `columns` plus ``run_path``,
        ``record_mtime`` and ``record_size``. ``run_path`` is unique.
    """
    kind = kind if isinstance(kind, RecordKind) else get_kind(kind)
    root = Path(root)
    cache_f = cache_path(root, kind, cache_rel)
    cols = list(kind.columns) + BASE_COLS
    rec_rel = rel_record if rel_record is not None else kind.rel_record

    cached = _read_cache_csv(cache_f, cols)
    migrated = [c for c in cols if c not in cached.columns]
    for c in migrated:                   # schema migration: old CSVs lack record_size etc.
        cached[c] = pd.NA
    if migrated and len(cached):
        # A new kind column is a cache-side change `_fresh` cannot see (the records on disk
        # are unchanged), so every existing row would keep NaN there forever. Clearing
        # `record_size` routes all rows through `_fresh`'s legacy branch -> one full re-read.
        cached["record_size"] = pd.NA
    wanted = _dedupe(rel_paths)
    inv = set(wanted)
    if not wanted:
        # Empty inventory: nothing to sweep and nothing to persist (an unwritable root -- e.g. a
        # loader called on a tree that does not exist -- must not fail on the cache write).
        return cached.iloc[:0].reset_index(drop=True)
    if not refresh:
        return cached[cached.run_path.isin(inv)].reset_index(drop=True)

    prev = {r.run_path: r for r in cached.itertuples(index=False)}

    def sweep_one(rp: str) -> tuple[str, dict | None, bool]:
        """One inventory path -> ("fresh"|"gone"|"err"|"row", row, was_cached).

        Runs on a worker thread: `prev` is only read, and every outcome is returned rather
        than counted here, so the accounting below stays deterministic and in inventory order.
        """
        f = root / rp / rec_rel
        old = prev.get(rp)
        try:
            st = f.stat()
            mt, size = float(st.st_mtime), int(st.st_size)
        except FileNotFoundError:
            return "gone", None, old is not None   # provably absent -> evict any cached row
        except OSError:                           # transient FS hiccup: never evict on it
            return "err", None, old is not None
        if old is not None and _fresh(old, mt, size):
            return "fresh", None, True
        try:
            raw = f.read_text()
            payload = raw if kind.raw_text else json.loads(raw)
            vals = kind.extract(payload)
            row = {c: vals[c] for c in kind.columns}
        except FileNotFoundError:
            return "gone", None, old is not None
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return "err", None, old is not None    # corrupt/unexpected: keep whatever we had
        row.update(run_path=rp, record_mtime=mt, record_size=size)
        return "row", row, old is not None

    rows, gone, n_new, n_upd, n_err, n_missing = [], set(), 0, 0, 0, 0
    with ThreadPoolExecutor(max_workers=max(1, min(_SWEEP_WORKERS, len(wanted)))) as pool:
        outcomes = list(pool.map(sweep_one, wanted))
    for rp, (status, row, was_cached) in zip(wanted, outcomes):
        if status == "gone":
            gone.add(rp)
            n_missing += 1
        elif status == "err":
            n_err += 1
        elif status == "row":
            rows.append(row)
            n_new += not was_cached
            n_upd += was_cached

    fresh = pd.DataFrame(rows, columns=cols)
    keep = cached[~cached.run_path.isin(set(fresh.run_path) | gone)]
    saved = pd.concat([keep, fresh], ignore_index=True) if len(fresh) else keep.reset_index(drop=True)
    # Persist only when the cache actually changed: with nothing added, refreshed, or
    # evicted, `saved` == `cached` and the rewrite is a pure no-op — one that costs ~2s
    # of to_csv per call on a 45k-row cache and needlessly widens the concurrent
    # union-persist window (measured 08-28: 81 warm loads in one notebook spent ~40% of
    # their time rewriting an unchanged CSV). "Evicted" means a CACHED row went away: an
    # inventory path that never had a record is in `gone` on every call, and must not
    # count (2026-09-10: 33 unevaluated cells were forcing a 224 MB rewrite per call).
    if len(fresh) or len(keep) < len(cached):
        _write_cache_csv(cache_f, saved)
    out = saved[saved.run_path.isin(inv)].reset_index(drop=True)
    if verbose:
        print(f"{kind.name} cache: {len(out)} rows of {len(saved)} cached "
              f"({n_new} added, {n_upd} refreshed, {n_missing} inventory paths without a record, "
              f"{n_err} unreadable/kept)")
    return out


def _fresh(old, mtime: float, size: int) -> bool:
    """True when the cached row `old` still describes a record with this ``(mtime, size)``."""
    try:
        om, osz = float(old.record_mtime), old.record_size
        return bool(abs(mtime - om) <= _MTIME_TOL and not pd.isna(osz) and int(osz) == size)
    except (TypeError, ValueError):
        return False                       # legacy row (no/garbled size) -> re-read
