"""Full LoRA singular spectra along a run's checkpoint ladder (CPU only, no base model).

The intermediate checkpoint adapters (`{run}/checkpoints/checkpoint-N/`) are being deleted;
everything cheap and weight-derived must be persisted first. The existing per-step record
(`eval/adapter_norms_traj.json`, `scripts.adapter_norms_slim --checkpoints`) keeps only
‖BA‖_F and σ₁(BA) per module. This harvester writes, per run:

- ``{run}/eval/adapter_spectra_traj.npz`` -- float32 arrays: ``steps[K]``, ``sigma[K,M,r]``
  (ALL singular values of ``s·B@A``, s = alpha/rank), ``ba_fro[K,M]``, ``a_fro``/``b_fro``,
  ``a_spec``/``b_spec``, and the ``modules[M]`` string array;
- ``{run}/eval/adapter_spectra_traj.json`` -- identity + per-step pooled scalars
  (`record_cache` kind ``adapter_spectra_traj``), plus the npz sha256.

No cosines and no singular vectors: those need the init/base-model reconstruction this
script deliberately avoids. Spectra come from the exact QR trick of
`analysis.adapter_decomp.low_rank_singular_values` (R factors via `np.linalg.qr`, see
`_r_factor`), so ΔW is never formed.

Because the source checkpoints are deleted afterwards this record is PERMANENT: the write
path is therefore validated (`spectra_record_problem`), claimed (`utils.run_io`) and
never loses points -- a record that would cover fewer steps than the stored one moves the
stored JSON *and* npz aside instead of overwriting them.

    uv run python -m scripts.adapter_spectra_trajectory --run-dirs-file chunk.txt
    uv run python -m scripts.adapter_spectra_trajectory --roots students/dog --workers 4
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.config import resolve_output_root
from subliminal_learning.evaluation.ckpt_trajectories import _checkpoint_dirs
from subliminal_learning.training.ckpt_ladder import distinct_steps, steps_covered
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import (
    CLAIM_TTL_SECONDS,
    acquire_claim,
    find_leaf_dirs_parallel,
    manifest_ref,
    release_claim,
)

ROOT = Path(__file__).absolute().parents[2]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SCHEMA_VERSION = 1
NPZ_NAME = "adapter_spectra_traj.npz"
JSON_NAME = "adapter_spectra_traj.json"
ADAPTER = "adapter_model.safetensors"
MANIFEST = "train_manifest.json"
INTERRUPTED = "interrupted.json"
RECENT_WRITE_SECONDS = 15 * 60
# Dirs that can never CONTAIN a run dir (same pruning as `scripts.dedupe_final_trajectory_checkpoint`).
SKIP_DIRS = ("eval", "wandb", "logs", "gkd_students", "sdf_teachers")


# --------------------------------------------------------------------------- inventory
def is_run_dir(path: Path) -> bool:
    """A finished training run: a manifest beside a `checkpoints/` endpoint adapter."""
    return (path / MANIFEST).is_file() and (path / "checkpoints" / ADAPTER).is_file()


def in_flight_reason(run_dir: Path, ignore_claim: Path | None = None) -> str | None:
    """Why this run must not be read now, or None. Mirrors the dedupe script's gate.

    Runs BEFORE this script takes its own claim; `ignore_claim` exists only so a caller
    that already holds the run's claim (this script, re-checking) is not blocked by it.
    """
    if (run_dir / INTERRUPTED).is_file():
        return "interrupted.json"
    for claim in run_dir.glob(".claim*"):
        if ignore_claim is not None and claim == Path(ignore_claim):
            continue
        try:
            age = time.time() - claim.stat().st_mtime
        except OSError:
            continue
        if age < CLAIM_TTL_SECONDS:
            return f"fresh claim ({age / 3600:.1f} h)"
    for marker in (run_dir / "checkpoints").glob("checkpoint-*/resume_marker.json"):
        return f"resume_marker.json ({marker.parent.name})"
    for path in (run_dir / MANIFEST, run_dir / "checkpoints" / ADAPTER):
        try:
            age = time.time() - path.stat().st_mtime
        except OSError:
            continue
        if age < RECENT_WRITE_SECONDS:
            return "recent write"
    return None


# --------------------------------------------------------------------------- adapter config
UNSUPPORTED_CHECKS = (
    ("use_rslora", lambda c: bool(c.get("use_rslora")), "rslora"),
    ("rank_pattern", lambda c: bool(c.get("rank_pattern")), "rank_pattern"),
    ("alpha_pattern", lambda c: bool(c.get("alpha_pattern")), "alpha_pattern"),
    ("use_dora", lambda c: bool(c.get("use_dora")), "dora"),
    ("peft_type", lambda c: (c.get("peft_type") or "LORA") != "LORA", "peft_type"),
)


def unsupported_adapter(cfg: dict) -> str | None:
    """Why this adapter's ``s = alpha/r``, one-rank-per-module convention does not hold.

    Mirrors `scripts.intruder_threshold_scan`'s rsLoRA guard: every one of these changes
    the per-module scaling or rank, so the spectra this script writes would be wrong.
    """
    for _, predicate, why in UNSUPPORTED_CHECKS:
        if predicate(cfg):
            return why
    return None


# --------------------------------------------------------------------------- validation
def spectra_record_problem(run_dir: Path, *, ladder_steps=None) -> str | None:
    """Why the stored spectra record is not a trustworthy permanent measurement, or None.

    Checked (short code returned for the first failure): ``no_json`` / ``no_npz`` (a file
    is missing or unparseable), ``sha_mismatch`` (the npz bytes are not the ones the JSON
    hashed -- a truncated or half-written npz), ``npz_unreadable``, ``steps_mismatch``
    (npz ``steps`` != JSON ``checkpoint_steps``, as ints in the same order),
    ``shape_mismatch`` (``sigma`` is not ``(K, M, rank)`` for the JSON's modules/steps),
    ``points_mismatch`` (the pooled ``points`` are not one per step, in step order) and,
    when `ladder_steps` is given, ``ladder_not_covered`` (a step that ever existed is not
    in the record). The thinning script imports this before deleting checkpoints.
    """
    run_dir = Path(run_dir)
    json_path, npz_path = run_dir / "eval" / JSON_NAME, run_dir / "eval" / NPZ_NAME
    if not json_path.is_file():
        return "no_json"
    try:
        rec = json.loads(json_path.read_text())
    except (OSError, ValueError):
        return "no_json"
    if not isinstance(rec, dict):
        return "no_json"
    if not npz_path.is_file():
        return "no_npz"
    if _sha256(npz_path) != rec.get("npz_sha256"):
        return "sha_mismatch"
    try:
        with np.load(npz_path) as z:
            npz_steps = [int(s) for s in z["steps"]]
            sigma_shape = tuple(z["sigma"].shape)
    except Exception:                                            # noqa: BLE001
        return "npz_unreadable"
    try:
        stored_steps = [int(s) for s in (rec.get("checkpoint_steps") or [])]
    except (TypeError, ValueError):
        return "steps_mismatch"
    if npz_steps != stored_steps:
        return "steps_mismatch"
    modules, rank = rec.get("modules") or [], rec.get("rank")
    if sigma_shape != (len(stored_steps), len(modules), rank):
        return "shape_mismatch"
    points = rec.get("points") or []
    if [p.get("step") for p in points] != stored_steps:
        return "points_mismatch"
    if ladder_steps is not None and not steps_covered(stored_steps, ladder_steps):
        return "ladder_not_covered"
    return None


# --------------------------------------------------------------------------- currency
# The gate is identity (`train_manifest_ref`) AND a valid stored record AND
# `ckpt_ladder.steps_covered` on the steps STILL ON DISK: a record measured over a fuller
# ladder than survives stays current, which is the point of harvesting before deletion.
# Completeness against `ckpt_ladder.ladder_steps` (every step that ever existed) is the
# thinning script's precondition (`spectra_record_problem(..., ladder_steps=...)`).
def _record_is_current(run_dir: Path, disk_steps, ref) -> bool:
    f = run_dir / "eval" / JSON_NAME
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if ref is None or rec.get("train_manifest_ref") != ref:
        return False
    if spectra_record_problem(run_dir) is not None:
        return False
    return steps_covered(rec.get("checkpoint_steps"), disk_steps)


# --------------------------------------------------------------------------- supersede
def _supersede(run_dir: Path, n_stored: int, n_new: int) -> None:
    """Move BOTH stored files aside under one uniquified ``.superseded-`` suffix.

    Called before any write that would cover fewer steps than the stored JSON -- including
    ``--force``, a changed manifest ref, and a record that failed validation but still
    holds more points. The suffix carries a timestamp, the pid and 4 random hex so two
    concurrent supersedes can never clobber each other (or an existing sidecar).
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = f".superseded-{stamp}-{os.getpid()}-{secrets.token_hex(2)}"
    moved = []
    for name in (JSON_NAME, NPZ_NAME):
        path = run_dir / "eval" / name
        if path.exists():
            path.rename(path.with_name(path.name + suffix))
            moved.append(path.name + suffix)
    if moved:
        print(f"  WARNING {run_dir}: recompute covers {n_new} step(s) < {n_stored} stored; "
              f"preserved as {', '.join(moved)}", flush=True)


# --------------------------------------------------------------------------- measurement
def _module_bases(tensors) -> list[str]:
    return [k.split(".lora_A.weight")[0].removeprefix("base_model.model.")
            for k in sorted(k for k in tensors if k.endswith("lora_A.weight"))]


def _r_factor(x: np.ndarray) -> np.ndarray:
    """Upper-triangular ``R`` of a thin QR of ``x`` (up to row signs, which spectra ignore).

    Exact `np.linalg.qr`: the cheaper Gram + Cholesky route squares the condition number,
    so on an ill-conditioned factor (σ spanning 1e-9) it either loses half the digits of
    the small singular values or fails to be positive definite at all. The permanence of
    this record is worth the ~2x.
    """
    return np.linalg.qr(x, mode="r")


def _point_arrays(tensors, modules: list[str], rank: int, scaling: float) -> dict:
    """Per-module spectra + factor norms for one checkpoint.

    A module absent from THIS checkpoint and a module whose factors are nonfinite both get
    a NaN row, but they are counted apart (`missing` vs `nonfinite` masks): `modules` is
    the union over the whole ladder, so absence is a normal ladder fact, not a divergence.
    """
    by_base = {b: f"base_model.model.{b}" for b in _module_bases(tensors)}
    m = len(modules)
    out = {"sigma": np.full((m, rank), np.nan), **{k: np.full(m, np.nan) for k in
           ("ba_fro", "a_fro", "b_fro", "a_spec", "b_spec")},
           "missing": np.zeros(m, dtype=bool), "nonfinite": np.zeros(m, dtype=bool)}
    for i, mb in enumerate(modules):
        key = by_base.get(mb)
        if key is None:
            out["missing"][i] = True
            continue
        a = tensors[f"{key}.lora_A.weight"].astype(np.float64)
        b = tensors[f"{key}.lora_B.weight"].astype(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            out["nonfinite"][i] = True
            continue
        # Same QR trick as `analysis.adapter_decomp.low_rank_singular_values`, but the two R
        # factors are reused for the per-factor spectral norms (sigma(A) == sigma(R_A)).
        r_a, r_b = _r_factor(a.T), _r_factor(b)
        sv = scaling * np.linalg.svd(r_b @ r_a.T, compute_uv=False)
        out["sigma"][i, :min(rank, sv.size)] = sv[:rank]      # NaN-padded on a rank mismatch
        out["ba_fro"][i] = float(np.sqrt(np.square(sv).sum()))
        out["a_fro"][i] = float(np.linalg.norm(a))
        out["b_fro"][i] = float(np.linalg.norm(b))
        out["a_spec"][i] = float(np.linalg.svd(r_a, compute_uv=False)[0])
        out["b_spec"][i] = float(np.linalg.svd(r_b, compute_uv=False)[0])
    return out


def _pooled(point: dict) -> dict:
    """Module-pooled scalars for one step (means over the modules that measured).

    ``sigma_top4_mean`` averages the top MIN(4, rank) singular values -- on a rank<4
    adapter it is the mean of every value, not a 4-point mean. A module whose update is
    exactly zero (σ₁ = 0) has no defined stable/effective rank: it is excluded from those
    means and counted in ``n_zero_modules``.
    """
    sigma = point["sigma"]
    ba_fro = point["ba_fro"]
    stable, effective, top4, zero = [], [], [], 0
    for i, row in enumerate(sigma):
        if point["missing"][i] or point["nonfinite"][i]:
            continue
        s = row[np.isfinite(row)]
        s2 = np.square(s)
        if s.size == 0 or s2.sum() <= 0:
            zero += 1
            continue
        stable.append(float(s2.sum() / s2[0]))
        p = s2[s2 > 0] / s2.sum()
        effective.append(float(np.exp(-(p * np.log(p)).sum())))
        top4.append(float(s[:4].mean()))
    mean = lambda v: float(np.mean(v)) if v else float("nan")   # noqa: E731
    return {
        "ba_fro_mean": mean(ba_fro[np.isfinite(ba_fro)].tolist()),
        "ba_spec_mean": mean([float(r[np.isfinite(r)][0]) for r in sigma if np.isfinite(r).any()]),
        "stable_rank_mean": mean(stable),
        "effective_rank_mean": mean(effective),
        "sigma_top4_mean": mean(top4),
        "n_nonfinite_modules": int(point["nonfinite"].sum()),
        "n_missing_modules": int(point["missing"].sum()),
        "n_zero_modules": zero,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_union(points) -> list[str]:
    """Every module name that appears at ANY checkpoint, sorted (safetensors header only)."""
    from safetensors import safe_open

    names: set[str] = set()
    for _, d in points:
        try:
            with safe_open(str(d / ADAPTER), framework="numpy") as f:
                names.update(_module_bases(f.keys()))
        except Exception:                                        # noqa: BLE001
            continue
    return sorted(names)


def _publish(run_dir: Path, kept: list[int], modules: list[str], arrays: dict,
             rows: list[dict], rank: int, alpha: float, scaling: float, ref) -> None:
    """npz (tmp + fsync + replace) first, sha of the FINAL file, then the JSON.

    Order matters: the JSON is the record's existence flag and carries the hash, so it
    must never name bytes that are not yet durably in place.
    """
    npz = run_dir / "eval" / NPZ_NAME
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(f"{npz.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, steps=np.asarray(kept, dtype=np.int64),
                            modules=np.asarray(modules), **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, npz)
    atomic_write_json(run_dir / "eval" / JSON_NAME, {
        "kind": "adapter_spectra_traj", "schema_version": SCHEMA_VERSION,
        "checkpoint_steps": kept, "train_manifest_ref": ref,
        "rank": rank, "alpha": alpha, "scaling": scaling, "modules": modules,
        "dtype": "float32", "npz_sha256": _sha256(npz),
        "points": [{"step": s, **_pooled(r)} for s, r in zip(kept, rows)],
    }, indent=None)


def process_run(run_dir: Path, force: bool = False) -> str:
    """Write this run's spectra sidecar; returns the status line body."""
    run_dir = Path(run_dir)
    if not is_run_dir(run_dir):
        return "not_a_run_dir"
    ck = run_dir / "checkpoints"
    cfg_file = ck / "adapter_config.json"
    if not cfg_file.is_file():
        return "error:no_adapter_config"
    try:
        cfg = json.loads(cfg_file.read_text())
    except ValueError:
        return "error:bad_adapter_config"
    why = unsupported_adapter(cfg)
    if why:
        return f"error:unsupported_adapter_config:{why}"
    reason = in_flight_reason(run_dir)
    if reason:
        return f"in_flight ({reason})"
    points = [(s, d) for s, d in _checkpoint_dirs(ck) if (d / ADAPTER).is_file()]
    if not points:
        return "error:no_adapter_on_disk"
    ref = manifest_ref(run_dir)
    steps = [s for s, _ in points]
    if not force and _record_is_current(run_dir, steps, ref):
        return "current (skip)"

    claim = acquire_claim(run_dir)
    if claim is None:
        return "in_flight (claim held)"
    try:
        return _compute_and_publish(run_dir, points, cfg, ref)
    finally:
        release_claim(claim)


def _compute_and_publish(run_dir: Path, points, cfg: dict, ref) -> str:
    rank, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
    scaling = alpha / rank
    modules = _module_union(points)
    if not modules:
        return "error:no_readable_checkpoint"
    rows, kept = [], []
    from safetensors.numpy import load_file

    for step, d in points:
        try:
            tensors = load_file(str(d / ADAPTER))   # one read per checkpoint, all modules from it
            rows.append(_point_arrays(tensors, modules, rank, scaling))
        except (FileNotFoundError, OSError) as exc:      # checkpoint vanished mid-run
            print(f"  {run_dir}: skipping step {step} ({exc})", flush=True)
            continue
        kept.append(step)
    if not rows:
        return "error:no_readable_checkpoint"

    stored = _stored_steps(run_dir)
    if stored and not steps_covered(kept, stored):
        _supersede(run_dir, len(distinct_steps(stored)), len(distinct_steps(kept)))
    stack = lambda key: np.stack([r[key] for r in rows]).astype(np.float32)   # noqa: E731
    arrays = {k: stack(k) for k in ("sigma", "ba_fro", "a_fro", "b_fro", "a_spec", "b_spec")}
    _publish(run_dir, kept, modules, arrays, rows, rank, alpha, scaling, ref)
    return "written"


def _stored_steps(run_dir: Path) -> list[int]:
    """The stored JSON's ``checkpoint_steps`` (even for an INVALID record), else ``[]``."""
    try:
        rec = json.loads((run_dir / "eval" / JSON_NAME).read_text())
        return [int(s) for s in (rec.get("checkpoint_steps") or [])]
    except (OSError, ValueError, TypeError, AttributeError):
        return []


# --------------------------------------------------------------------------- driver
def _worker_init() -> None:
    """Single-threaded BLAS in each worker (the nodes run GPU jobs alongside)."""
    try:
        from threadpoolctl import threadpool_limits

        globals()["_TP_LIMITS"] = threadpool_limits(limits=1)
    except ImportError:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"


def _run_one(args: tuple[str, bool]) -> tuple[str, str]:
    run_dir, force = args
    try:
        return run_dir, process_run(Path(run_dir), force)
    except Exception as exc:                                     # noqa: BLE001
        return run_dir, f"error:{type(exc).__name__}: {exc}"


def read_run_dirs_file(path: Path, out_root: Path) -> list[Path]:
    """One run dir per line; blank and ``#`` comment lines are skipped.

    A relative line is resolved against `out_root` (the configured output root).
    """
    dirs = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        dirs.append(Path(line) if Path(line).is_absolute() else out_root / line)
    return dirs


def _tally_key(status: str) -> str:
    """Tally bucket: the status without its parenthetical, keeping distinct ``error:`` classes."""
    head = status.split(" (")[0]
    parts = head.split(":")
    return ":".join(parts[:2]) if head.startswith("error:") else parts[0]


@app.command()
def main(
    run_dirs_file: Path = typer.Option(
        None, help="Text file, one run dir per line; blank/'#' lines skipped, relative "
                   "lines resolved against output_root. Non-run-dirs report not_a_run_dir."),
    roots: list[str] = typer.Option(None, help="Tree paths under output_root to walk."),
    output_root: Path = typer.Option(None),
    workers: int = typer.Option(4, help="Worker PROCESSES over runs (CPU-bound SVDs)."),
    walk_workers: int = typer.Option(8, help="Threads for the run-dir discovery walk."),
    force: bool = typer.Option(False, help="Recompute even when the record is current."),
) -> None:
    out_root = resolve_output_root(output_root)
    if bool(run_dirs_file) == bool(roots):
        raise typer.BadParameter("give exactly one of --run-dirs-file / --roots")
    if run_dirs_file:
        run_dirs = read_run_dirs_file(run_dirs_file, out_root)
    else:
        run_dirs = find_leaf_dirs_parallel(
            [out_root / r for r in roots], is_run_dir, skip_names=SKIP_DIRS,
            skip_prefixes=("checkpoint-",), workers=walk_workers, progress=True)

    tally: dict[str, int] = {}
    tasks = [(str(d), force) for d in run_dirs]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for run_dir, status in pool.map(_run_one, tasks):
            print(f"{status:<24} {run_dir}", flush=True)
            tally[_tally_key(status)] = tally.get(_tally_key(status), 0) + 1
    print("done: " + (", ".join(f"{k} {v}" for k, v in sorted(tally.items())) or "nothing"))


if __name__ == "__main__":
    app()
