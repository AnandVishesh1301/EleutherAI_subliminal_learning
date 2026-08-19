"""Final-adapter Pile probe: delta-CE and KL(base ‖ student) on the shared probe pool.

This module is the non-ad-hoc home of the probe conventions so future sweeps and loaders
share one definition (2026-08-19; the first sweep ran from script-local constants that
are now imported from here).

**Instrument inventory** — three per-run Pile measurements coexist; keep them distinct:

| record | data | metrics | scope |
|---|---|---|---|
| ``{run}/pile_coherence/{shard}.json`` | coherence shard (v1: 1,953×2,048-token windows) | CE + delta w/ CI | final adapter; the high-precision CE magnitude |
| ``{run}/eval/pile_kl_traj.json`` | the PROBE pool | KL only | every checkpoint (ladder runs) |
| ``{run}/eval/pile_probe.json`` | the PROBE pool | CE **and** KL, one pass | final adapter (this module) |

The probe pool ``pp1`` is exactly `scripts.ckpt_pile_kl_trajectory`'s pool — 32 Pile-val
docs, ≤512 tokens, doc-permutation seed 0 — so probe KL is directly comparable to the
trajectory endpoints. The binding identity of a measurement is the recorded pool token
sha (``9f509dde…`` for pp1 with the Pythia tokenizer at the current dataset revision);
``pool_version`` is the human-readable name. Measured on the strip (n=1,526 finite):
delta-CE and KL agree at Pearson 1.0000, and dead labels cut at 5 nats are identical in
both columns — quote delta-CE in prose, cite KL against the containment analyses.

Records are identity-stamped with `manifest_ref` (retrain ⇒ stale ⇒ recompute) under the
run-dir eval contract shared with `analysis/trajectories.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
EVAL_NAME = "pile_probe.json"


@dataclass(frozen=True)
class ProbePoolSpec:
    """Pool recipe for the probe; bump `version` if any field changes."""

    n_docs: int
    max_tokens: int
    seed: int
    version: str


PROBE_POOL_PP1 = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version="pp1")
# Reference token sha of pp1 (Pythia tokenizer, monology/pile-uncopyrighted validation as
# of 2026-08). Recorded per record; a mismatch on a fresh build means the upstream dataset
# or tokenizer drifted -- investigate before comparing numbers.
PP1_TOKEN_SHA = "9f509dde493c113bf531e1d0b982600ec0755e6558e959d03ac48cf7b1d2ccd5"


def manifest_ref(run_dir: Path) -> dict | None:
    """Compact run-identity reference tying an eval file to the run that produced it.

    Same contract as the trajectory scripts (`scripts.ckpt_preference_trajectory`): None
    when the run has no final train_manifest.json, so the eval can never claim currency.
    """
    mf = Path(run_dir) / "train_manifest.json"
    if not mf.is_file():
        return None
    m = json.loads(mf.read_text())
    return {
        "student_param_tag": m.get("student_param_tag"),
        "filtered_sha256": m.get("filtered_sha256"),
        "n_train_steps": m.get("n_train_steps"),
    }


def build_probe_pool(tokenizer, spec: ProbePoolSpec = PROBE_POOL_PP1):
    """The probe pool for `spec` (see `cka.pool_io.build_pile_pool`)."""
    from subliminal_learning.cka.pool_io import build_pile_pool

    return build_pile_pool(tokenizer, spec.n_docs, max_tokens=spec.max_tokens, seed=spec.seed)


def probe_path(run_dir: Path) -> Path:
    return Path(run_dir) / "eval" / EVAL_NAME


def load_probe(run_dir: Path) -> dict | None:
    """The run's probe record, or None when absent/unreadable."""
    f = probe_path(run_dir)
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@dataclass(frozen=True)
class DeadCut:
    """A data-derived integrity dead cut for one population of probe values.

    The alive/dead separation is bimodal with a LOW-DENSITY VALLEY, not an empty gap
    (the strip's valley 3.6-6.3 holds ~5% of seeds -- boundary transit is a continuum).
    So the derivation returns the valley REGION and its occupancy alongside the point
    cut; any point inside the valley is an equivalent label up to `n_valley` seeds, and
    downstream fits should quote a cut-sensitivity check, not treat `cut` as exact.
    (The strip's valley is a broad shallow plateau at ~30% of the dead mode's density —
    a transit continuum, not an empty band; `valley_frac` = 0.5 admits it. The knobs
    move the valley EDGES, but `n_valley` always states the resulting label churn.)

    Attributes:
        cut: Density minimum inside the valley (nats); dead = value > cut.
        valley_lo, valley_hi: Valley edges (density first falls below / last exceeds
            `valley_frac` x the smaller mode's peak density).
        n_valley: Seeds inside the valley = the maximum label churn any in-valley cut
            choice could cause.
        mode_lo, mode_hi: Locations of the two dominant density modes.
        n_below, n_above: Population sizes on each side of `cut`.
    """

    cut: float
    valley_lo: float
    valley_hi: float
    n_valley: int
    mode_lo: float
    mode_hi: float
    n_below: int
    n_above: int


def derive_dead_cut(values, bin_width: float = 0.75, smooth_bins: int = 3,
                    valley_frac: float = 0.5, min_mode_frac: float = 0.10) -> DeadCut | None:
    """Derive the integrity dead cut from a population's probe values (delta-CE or KL).

    No preregistered constant: a smoothed histogram is scanned for its two dominant
    modes; the cut is the density minimum between them, and the valley region (density
    below `valley_frac` of the smaller mode's peak) plus its seed count are returned so
    the residual arbitrariness is quantified rather than hidden. Populations without two
    modes each holding at least `min_mode_frac` of the mass get None ("no bimodal
    separation here -- do not force a label"). Re-derive per population; never carry a
    cut across models/arms without re-deriving.

    Returns:
        A `DeadCut`, or None when the population is not usably bimodal.
    """
    import numpy as np

    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 50:
        return None
    edges = np.arange(v.min(), v.max() + bin_width, bin_width)
    if len(edges) < 8:
        return None
    h, _ = np.histogram(v, bins=edges)
    k = np.ones(smooth_bins) / smooth_bins
    d = np.convolve(h, k, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])

    # dominant mode on each side of the global density minimum's neighbourhood: take the
    # global max as mode 1, then the best local max separated from it by a density dip
    i1 = int(np.argmax(d))
    best = None
    for i2 in range(len(d)):
        if abs(i2 - i1) < 3:
            continue
        lo, hi = sorted((i1, i2))
        dip = d[lo:hi + 1].min()
        if dip < valley_frac * min(d[i1], d[i2]) and d[i2] > 0:
            score = d[i2]
            if best is None or score > best[0]:
                best = (score, i2)
    if best is None:
        return None
    i2 = best[1]
    lo, hi = sorted((i1, i2))
    side_lo = (v <= centers[lo]).mean()
    side_hi = (v >= centers[hi]).mean()
    if min(side_lo, side_hi) < min_mode_frac:
        return None

    peak_small = min(d[i1], d[i2])
    inner = slice(lo, hi + 1)
    below = d[inner] < valley_frac * peak_small
    if not below.any():
        return None
    idx = np.arange(lo, hi + 1)[below]
    valley_lo, valley_hi = float(centers[idx[0]]), float(centers[idx[-1]])
    jmin = idx[int(np.argmin(d[idx]))]
    cut = float(centers[jmin])
    return DeadCut(cut=cut, valley_lo=valley_lo, valley_hi=valley_hi,
                   n_valley=int(((v > valley_lo) & (v < valley_hi)).sum()),
                   mode_lo=float(centers[min(i1, i2)]), mode_hi=float(centers[max(i1, i2)]),
                   n_below=int((v <= cut).sum()), n_above=int((v > cut).sum()))


def probe_is_current(run_dir: Path, pool_sha: str) -> bool:
    """True when the record exists and still describes both the run and the pool."""
    rec = load_probe(run_dir)
    if rec is None:
        return False
    ref = manifest_ref(run_dir)
    return (
        ref is not None
        and rec.get("schema_version") == SCHEMA_VERSION
        and rec.get("manifest_ref") == ref
        and rec.get("pool_sha") == pool_sha
    )
