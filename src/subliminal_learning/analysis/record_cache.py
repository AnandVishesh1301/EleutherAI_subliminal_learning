"""Generic mtime/size cache over the small per-run JSON eval records.

Analysis notebooks fan out over 10-20k run dirs and read one tiny JSON per run from a slow
shared filesystem. The run-dir records are the SSOT; a cache CSV under the outputs root is a
derived aggregate so a notebook execution pays one *stat* per inventory path instead of one
*read*. This module generalises the original `analysis/probe_cache.py` (pile-probe only) to a
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
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from subliminal_learning.utils.io import atomic_write_text

# CSV float64 round-trips exactly through pandas, so this tolerance is belt-and-braces only
# (it costs nothing and protects against a writer/reader pair with a coarser format).
_MTIME_TOL = 1e-4

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
    """The registered kind `name`; raises KeyError listing the known kinds if absent."""
    try:
        return _REGISTRY[name]
    except KeyError:
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


def _extract_pile_coherence_v1(rec: dict) -> dict:
    return {"delta_overall": _f((rec.get("delta") or {}).get("delta_overall"))}


PILE_PROBE = register_kind(RecordKind(
    name="pile_probe",
    rel_record="eval/pile_probe.json",
    columns=["kl", "delta_ce", "nonfinite", "pool_version", "pool_sha", "schema_version",
             "student_param_tag", "filtered_sha256", "n_train_steps"],
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

    cached = pd.read_csv(cache_f) if cache_f.is_file() else pd.DataFrame(columns=cols)
    for c in cols:                       # schema migration: old CSVs lack record_size etc.
        if c not in cached.columns:
            cached[c] = pd.NA
    wanted = _dedupe(rel_paths)
    inv = set(wanted)
    if not wanted:
        # Empty inventory: nothing to sweep and nothing to persist (an unwritable root -- e.g. a
        # loader called on a tree that does not exist -- must not fail on the cache write).
        return cached.iloc[:0].reset_index(drop=True)
    if not refresh:
        return cached[cached.run_path.isin(inv)].reset_index(drop=True)

    prev = {r.run_path: r for r in cached.itertuples(index=False)}
    rows, gone, n_new, n_upd, n_err, n_missing = [], set(), 0, 0, 0, 0
    for rp in wanted:
        f = root / rp / rec_rel
        try:
            st = f.stat()
            mt, size = float(st.st_mtime), int(st.st_size)
        except FileNotFoundError:
            gone.add(rp)                  # provably absent -> evict any cached row
            n_missing += 1
            continue
        except OSError:                   # transient FS hiccup: never evict on it
            n_err += 1
            continue
        old = prev.get(rp)
        if old is not None and _fresh(old, mt, size):
            continue
        try:
            raw = f.read_text()
            payload = raw if kind.raw_text else json.loads(raw)
            vals = kind.extract(payload)
            row = {c: vals[c] for c in kind.columns}
        except FileNotFoundError:
            gone.add(rp)
            n_missing += 1
            continue
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            n_err += 1                    # corrupt/unexpected: keep whatever we had
            continue
        row.update(run_path=rp, record_mtime=mt, record_size=size)
        rows.append(row)
        n_new += old is None
        n_upd += old is not None

    fresh = pd.DataFrame(rows, columns=cols)
    keep = cached[~cached.run_path.isin(set(fresh.run_path) | gone)]
    saved = pd.concat([keep, fresh], ignore_index=True) if len(fresh) else keep.reset_index(drop=True)
    atomic_write_text(cache_f, saved.to_csv(index=False))
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
