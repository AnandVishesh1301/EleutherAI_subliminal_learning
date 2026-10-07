"""The ΔW variants: surgeries on a trained update, and synthetic injection shapes.

Two families share this module. `interp_basis` / `subset_targets` / `subset_rotate` /
`spread_variant` / `aligned_component_variant` rewrite a REAL update at exactly preserved
per-module norms (position, spread and direction surgery); `random_orthonormal_pair`,
`diffuse_ladder_delta` and `three_shape_delta` build SYNTHETIC updates at a matched
multiple of the module's RMT scale θ̄ (the equal-norm ladders). `intruder_positive` is the
Shuttleworth-style read that every one of them records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from subliminal_learning.forgetting.weightspace.lora import lora_svd


# Saved randt* results from unsigned QR are a different intervention.
INTERPOLATION_VERSION = "signed_qr_v2"


def interpolation_record_is_current(record: dict) -> bool:
    """Endpoint/subset records are unchanged; randt rows require corrected QR."""
    return (not str(record.get("variant", "")).startswith("randt")
            or record.get("interpolation_version") == INTERPOLATION_VERSION)


# Random targets keyed by (run, module, draw/selector) instead of draw order.
TARGET_VERSION = "keyed_targets_v1"

# Original and intervention damage measured together on the same inputs.
ORIGINAL_BASELINE_VERSION = "ce-baseline-v1"

# The baseline was copied from the run's eval-time pp1 probe, not measured in-session.
BASELINE_SOURCE_STORED_PROBE = "stored_probe"


def _uses_random_target(variant: str) -> bool:
    """Do this variant's directions come from a drawn random target?"""
    return variant not in ("orig", "align")


def surgery_record_is_current(record: dict, *, require_keyed_targets: bool = False) -> bool:
    """`interpolation_record_is_current` plus, optionally, keyed random targets.

    Random-target rows (rand*, randt*, every subset selector) written before
    `TARGET_VERSION` drew from one run-order-dependent stream, so they are not
    reproducible; endpoint rows (`orig`, `align`) never used a target.
    """
    if not interpolation_record_is_current(record):
        return False
    if not require_keyed_targets:
        return True
    variant = str(record.get("variant", ""))
    return (not _uses_random_target(variant)
            or record.get("target_version") == TARGET_VERSION)


def validate_surgery_baselines(
    rows: list[dict], *, version: str = ORIGINAL_BASELINE_VERSION,
) -> None:
    """Require one current, identity-matched CE baseline for every surgery run."""
    keys = ("adapter_sha256", "base", "base_revision", "pool_version", "pool_sha",
            "prompt_set_sha256", "animal", "seed")
    by_run: dict[str, tuple[object, ...]] = {}
    for row in rows:
        if row.get("original_baseline_version") != version:
            raise ValueError("surgery rows lack the current original baseline version")
        value = row.get("original_delta_ce")
        if value is None or not np.isfinite(float(value)):
            raise ValueError("surgery rows lack a finite identity-matched original delta_ce")
        if (not row.get("run") or any(key not in row for key in keys)
                or any(row[key] is None or (isinstance(row[key], float) and np.isnan(row[key]))
                       for key in keys if key != "base_revision")):
            raise ValueError("surgery rows lack complete input identity metadata")
        run = str(row["run"])
        identity = tuple(row[key] for key in keys) + (float(value),)
        if run in by_run and by_run[run] != identity:
            raise ValueError(f"surgery rows mix input identities or baselines within run {run}")
        by_run[run] = identity


def surgery_original_delta_ce(
    rows: list[dict], *, version: str = ORIGINAL_BASELINE_VERSION,
) -> tuple[dict[str, float], dict[str, str]]:
    """Per-run baseline ΔCE for surgery consumers, with the source it came from.

    The paired readout (`original_delta_ce` at `version`) is the better baseline: it
    measures the trained update on the SAME loaded base weights and probe pool as the
    intervention rows beside it, so it is immune to a later rewrite of the run's probe
    file. Artifacts written before that field existed have no paired value; falling back
    to the run's stored `eval/pile_probe.json` ΔCE is acceptable because it is the same
    quantity in the same units on the same pp1 pool, only measured at eval time. Sources
    are returned so every table/figure can name which runs used which baseline.

    Rows stamped ``original_baseline_source = "stored_probe"`` (written by
    `scripts.position_surgery` when `orig` is not requested, or by
    `scripts.backfill_surgery_baselines`) carry that same eval-time number in the row, so
    they are read from the row but reported as ``stored_probe``, never ``paired``; a
    paired row of the same run always wins over them.

    Returns ``(baseline_by_run, source_by_run)`` with source in
    ``{"paired", "stored_probe", "missing"}``; missing baselines are NaN, never an error.
    A run whose paired rows DISAGREE is real corruption and still raises.
    """
    keys = ("adapter_sha256", "base", "base_revision", "pool_version", "pool_sha",
            "prompt_set_sha256", "animal", "seed")
    paired: dict[str, float] = {}
    backfilled: dict[str, float] = {}
    identities: dict[str, tuple[object, ...]] = {}
    runs: dict[str, None] = {}
    for row in rows:
        run = str(row.get("run", ""))
        if not run:
            continue
        runs.setdefault(run)
        if row.get("original_baseline_version") != version:
            continue
        value = row.get("original_delta_ce")
        if value is None or not np.isfinite(float(value)):
            continue
        if row.get("original_baseline_source") == BASELINE_SOURCE_STORED_PROBE:
            backfilled.setdefault(run, float(value))
            continue
        identity = tuple(row.get(key) for key in keys) + (float(value),)
        if run in identities and identities[run] != identity:
            raise ValueError(f"surgery rows mix input identities or baselines within run {run}")
        identities[run] = identity
        paired[run] = float(value)
    baselines: dict[str, float] = {}
    sources: dict[str, str] = {}
    for run in runs:
        if run in paired:
            baselines[run], sources[run] = paired[run], "paired"
            continue
        if run in backfilled:
            baselines[run], sources[run] = backfilled[run], "stored_probe"
            continue
        value = _stored_probe_delta_ce(run)
        baselines[run] = float("nan") if value is None else value
        sources[run] = "missing" if value is None else "stored_probe"
    return baselines, sources


def probe_baseline(run: str) -> dict | None:
    """The run's eval-time pp1 probe readout, or None when absent/unreadable/non-finite.

    The one reader shared by every writer of a `stored_probe` baseline stamp.
    """
    try:
        record = json.loads((Path(run) / "eval" / "pile_probe.json").read_text())
    except (OSError, ValueError):
        return None
    value = record.get("delta_ce")
    if value is None or not np.isfinite(float(value)):
        return None
    out = {"delta_ce": float(value), "pile_kl": record.get("kl_base_student")}
    if "manifest_ref" in record:
        out["manifest_ref"] = record["manifest_ref"]
    return out


def _stored_probe_delta_ce(run: str) -> float | None:
    """The run's eval-time pp1 probe ΔCE, or None when absent/unreadable/non-finite."""
    base = probe_baseline(run)
    return None if base is None else base["delta_ce"]


def _qr_signed(a: np.ndarray) -> np.ndarray:
    """QR factor Q of `a`, each column sign-aligned to the column it came from, so the
    directions are rng-determined rather than LAPACK's Householder sign convention."""
    q = np.linalg.qr(a)[0]
    sgn = np.sign(np.einsum("ij,ij->j", q, a[:, :q.shape[1]]))
    sgn[sgn == 0] = 1.0
    return q * sgn


def interp_basis(u0: np.ndarray, u1: np.ndarray, t: float) -> np.ndarray:
    """QR-orthonormalised ``(1-t)·U0 + t·U1`` — a monotone bridge, not a geodesic."""
    if t <= 0:
        return u0.copy()
    if t >= 1:
        return u1.copy()
    blend = (1.0 - t) * u0 + t * u1
    return _qr_signed(blend)


def subset_targets(u: np.ndarray, idx: list[int], rng: np.random.Generator) -> np.ndarray:
    """Random orthonormal target directions for columns `idx`: orthonormal to the KEPT
    columns (which stay unchanged) and to each other."""
    if not idx:
        return u[:, :0]
    keep = [j for j in range(u.shape[1]) if j not in idx]
    q = _qr_signed(np.concatenate(
        [u[:, keep], rng.standard_normal((u.shape[0], len(idx)))], axis=1))
    return q[:, len(keep):]


def subset_rotate(u: np.ndarray, idx: list[int], r: np.ndarray, t: float) -> np.ndarray:
    """Rotate columns `idx` of U a fraction t toward targets R (t=1 -> exactly R).
    The blend stays in the kept columns' orthogonal complement; QR re-orthonormalises
    the rotated block, sign-aligned to the blend so the path is continuous in t."""
    if not idx:
        return u
    blend = (1.0 - t) * u[:, idx] + t * r
    out = u.copy()
    out[:, idx] = _qr_signed(blend)
    return out


def spread_variant(b: np.ndarray, a: np.ndarray, s: float, variant: str) -> np.ndarray:
    """`top1`/`flat` at exactly the original ‖ΔW‖_F, via the r x r QR core."""
    u, sv, v = lora_svd(b, a)
    n_tot = s * float(np.sqrt((sv**2).sum()))
    if variant == "top1":
        return n_tot * np.outer(u[:, 0], v[:, 0])
    k = max(1, int((sv > sv[0] * 1e-10).sum()))
    return (n_tot / np.sqrt(k)) * (u[:, :k] @ v[:, :k].T)


def aligned_component_variant(dw: np.ndarray, w0: np.ndarray, variant: str) -> np.ndarray:
    """Apply one W₀-alignment variant to a module's ΔW, preserving ‖ΔW‖_F exactly."""
    if variant == "orig":
        return dw
    ip = float((dw * w0).sum())
    c = ip / float((w0 * w0).sum())
    dw_par = c * w0
    dw_perp = dw - dw_par
    n_tot = float(np.linalg.norm(dw))
    n_par = float(np.linalg.norm(dw_par))
    n_perp = float(np.linalg.norm(dw_perp))
    if variant == "flip":
        return dw_perp - dw_par
    if variant == "strip":
        return dw_perp * (n_tot / n_perp if n_perp else 1.0)
    if variant == "flip10":
        amp = 10.0
        rem = n_tot**2 - (amp * n_par) ** 2
        if rem <= 0 or not n_perp:  # sliver too large to amplify at fixed norm
            return dw_perp - dw_par
        return np.sqrt(rem) / n_perp * dw_perp - amp * dw_par
    raise ValueError(variant)


def random_orthonormal_pair(n: int, m: int, k: int,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """``(U_r n x k, V_r m x k)`` orthonormal, drawn left-then-right from `rng`."""
    return (_qr_signed(rng.standard_normal((n, k))),
            _qr_signed(rng.standard_normal((m, k))))


def diffuse_ladder_delta(m: dict, align: str, k: int, e: float, rng: np.random.Generator,
                         match: str = "fro") -> np.ndarray:
    """match='fro': per-module ‖Δ‖_F = E·θ̄ (per-direction σ = E·θ̄/√k).
    match='spec': per-direction σ = E·θ̄ exactly (so ‖Δ‖_F = E·θ̄·√k grows with k) —
    the complementary design: if damage tracks σ₁ it is flat in k; if it tracks
    Frobenius energy it rises like √k."""
    n, mm = m["shape"]
    th = m["theta"]
    if align == "g":
        u_r, v_r = random_orthonormal_pair(n, mm, k, rng)
    else:
        u_r, v_r = m["U"][:, :k], m["Vt"][:k, :].T
    scale = (e * th) if match == "spec" else (e * th / np.sqrt(k))
    return scale * (u_r @ v_r.T)


def three_shape_delta(m: dict, cond: str, e: float, k: int,
                      rng: np.random.Generator) -> np.ndarray:
    """The 2x2 equal-norm shapes: {concentrated, diffuse} x {generic, W₀-aligned}."""
    n, mm = m["shape"]
    th = m["theta"]
    if cond in ("cg", "ca"):
        if cond == "cg":
            u = rng.standard_normal(n)
            v = rng.standard_normal(mm)
            u /= np.linalg.norm(u)
            v /= np.linalg.norm(v)
        else:
            u, v = m["U"][:, 0], m["Vt"][0, :]
        return (e * th) * np.outer(u, v)
    if cond == "dg":
        u_r, v_r = random_orthonormal_pair(n, mm, k, rng)
    else:
        u_r, v_r = m["U"][:, :k], m["Vt"][:k, :].T
    return (e * th / np.sqrt(k)) * (u_r @ v_r.T)


def directions_delta(u: np.ndarray, v: np.ndarray, k: int, scale: float) -> np.ndarray:
    """``scale · U[:, :k] V[:, :k]ᵀ`` — equal energy over k given direction pairs."""
    return scale * (u[:, :k] @ v[:, :k].T)


def intruder_positive(w0: np.ndarray, delta: np.ndarray, u0: np.ndarray, *,
                      top: int = 10, tau: float = 0.5) -> bool:
    """Is ``W₀ + Δ`` intruder-positive: does any of its `top` new left singular vectors
    have max |overlap| < `tau` against every pretrained left singular vector?"""
    u_new = np.linalg.svd(w0 + delta, full_matrices=False)[0]
    ov = np.abs(u_new[:, :top].T @ u0).max(axis=1)
    return bool((ov < tau).sum() > 0)
