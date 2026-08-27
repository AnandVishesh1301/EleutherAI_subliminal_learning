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


# The operative integrity dead cut (2026-08-20, Angela's call; moved 5.0 -> 4.0 later the
# same day): an ARBITRARY, openly declared convention -- KL (or delta-CE, numerically
# identical) > 4.0 nats vs own base (~55x the base's Pile perplexity). Deliberately NOT
# "derived": every derivation attempted (density valley, outlier fence, mixtures,
# dose-side changepoints) turned out to presuppose the labeling, the population
# composition, or the dose law itself (history: collapse_claims C3 — pre-08-20-reorder id
# C7 — and the campaigns notebook S0c/S0d, 2026-08-19/20). The alive bulk ends by ~4-5
# nats in every measured slice and all law statistics are insensitive over cuts 4-10
# (K* moves ~4-8% between cuts 4 and 5; the 741 runs at KL 4-5 are the label-sensitive
# set) -- always quote a cut-sensitivity line next to any fit that uses it.
DEAD_KL_CUT = 4.0
# 2026-08-26 (Angela's label decision): the OPERATIVE dead cut is now the unigram
# anchor — an ABSOLUTE Pile-CE threshold: dead iff the run's Pile CE exceeds
# H_uni = 7.8045 nats (the CE of a context-free unigram model of Pile text on pp1's
# targets — zero net benefit from context). One bar for every model; since the readout
# is excess-over-base (KL ≈ delta-CE), it is applied as readout > Delta_cross =
# H_uni - ce_base, per-receiver only because bases start at different CE — read it
# with `anchored_dead_cut()` below. DEAD_KL_CUT = 4.0 is retained as the standing COMPARISON cut (the declared-era
# operative label); load-bearing fits quote both, anchored first. Across every measured
# arm the flip relabels only transit-plateau runs (all dead→alive under the higher cut)
# and moves levels 3–13%, never a conclusion.


def anchored_dead_cut(output_root: Path, base_repo_or_key: str) -> float:
    """Per-receiver operative dead cut Delta_cross = H_uni − ce_base [nats].

    Reads `{output_root}/analysis/unigram_anchor_pp1.json` (regenerate/extend with
    `scripts/build_unigram_anchor.py`). Accepts either the HF repo id or the anchor
    record's base key; raises KeyError with the known keys if the base is absent —
    never silently falls back to DEAD_KL_CUT.
    """
    rec = json.loads((Path(output_root) / "analysis" / "unigram_anchor_pp1.json").read_text())
    dc = rec["delta_cross"]
    key = base_repo_or_key.split("/")[-1]
    for cand in (base_repo_or_key, key):
        if cand in dc:
            return float(dc[cand])
    raise KeyError(f"no anchored cut for {base_repo_or_key!r}; known bases: {sorted(dc)}")


PROBE_POOL_PP1 = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version="pp1")
# Reference token sha of pp1 (Pythia tokenizer, monology/pile-uncopyrighted validation as
# of 2026-08). Recorded per record; a mismatch on a fresh build means the upstream dataset
# or tokenizer drifted -- investigate before comparing numbers.
PP1_TOKEN_SHA = "9f509dde493c113bf531e1d0b982600ec0755e6558e959d03ac48cf7b1d2ccd5"


# Canonical home is utils.run_io (torch-free); re-exported here for existing importers.
from subliminal_learning.utils.run_io import manifest_ref  # noqa: E402,F401


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
                    valley_frac: float = 0.5, min_mode_runs: int = 200) -> DeadCut | None:
    """Derive the integrity dead cut from a population's probe values (delta-CE or KL).

    No preregistered constant: a smoothed histogram is scanned for its two dominant
    modes; the cut is the density minimum between them, and the valley region (density
    below `valley_frac` of the smaller mode's peak) plus its seed count are returned so
    the residual arbitrariness is quantified rather than hidden. Populations without two
    modes each holding at least `min_mode_runs` runs get None ("no bimodal separation
    here -- do not force a label"). The gate is COUNT-based (2026-08-20, was a 10% mass
    fraction): what makes a sparse mode's valley untrustworthy is too few runs to pin its
    density down, not its share of the pool -- a pooled population diluted by alive-heavy
    coverage can hold a perfectly well-populated dead mode at a small mass fraction.
    Re-derive per population; never carry a cut across models/arms without re-deriving.

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
    # gate on the runs each side of the candidate cut (the interior density minimum) --
    # each mode's full population, not just its beyond-the-peak half
    _inner_idx = np.arange(lo, hi + 1)
    _jdip = _inner_idx[int(np.argmin(d[_inner_idx]))]
    n_side_lo = int((v <= centers[_jdip]).sum())
    if min(n_side_lo, n - n_side_lo) < min_mode_runs:
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


# --------------------------------------------------------------------------------------
# Pile-probe measurement (single implementation, GPU-resident). Moved here 2026-08-20 from
# scripts/frozen_a_ablation.py (`_pile_logprobs`, `_mean_kl`) and scripts/pile_probe_final.py
# (`_mean_ce`), which ran docs one at a time and shipped every log-prob tensor to the CPU --
# that round-trip made probe jobs CPU-bound (17 s/run on an idle GPU).
# --------------------------------------------------------------------------------------


def pile_logprobs(model, pool, device: str, batch_size: int = 8) -> list:
    """Next-token log-probs at the pool's eligible positions, one GPU tensor per doc.

    Eligibility indexes the *logits*, not the targets: a KL between two models compares two
    predictive distributions and needs no observed next token, so every position carries a
    usable distribution -- including the last. What it must exclude is position 0, the BOS
    attention-sink outlier the pool's ``eligible`` convention drops, which is exactly what
    indexing logits by ``eligible`` does.

    Docs run in right-padded batches and the per-doc float32 log-prob tensors stay on
    `device` (~3.2 GB for pp1 at the Pythia vocab). Right padding is exact for a causal LM:
    real-position logits cannot attend to the (future) pad keys, and GPT-NeoX's arange
    position_ids are unaffected by trailing pads, so no pad-token dependence is introduced.
    """
    import torch

    out = [None] * len(pool.input_ids)
    with torch.no_grad():
        for start in range(0, len(pool.input_ids), batch_size):
            docs = pool.input_ids[start:start + batch_size]
            eligs = pool.eligible[start:start + batch_size]
            length = max(len(d) for d in docs)
            ids = torch.zeros(len(docs), length, dtype=torch.long, device=device)
            mask = torch.zeros(len(docs), length, dtype=torch.long, device=device)
            for i, d in enumerate(docs):
                ids[i, : len(d)] = torch.tensor(d, device=device)
                mask[i, : len(d)] = 1
            logits = model(input_ids=ids, attention_mask=mask).logits
            for i, el in enumerate(eligs):
                sel = logits[i].index_select(0, torch.tensor(el, device=device)).float()
                out[start + i] = torch.log_softmax(sel, dim=-1)
            del logits
    return out


def mean_kl(ref_logp: list, other_logp: list) -> float:
    """Position-mean ``KL(ref ‖ other)`` in nats, pooled over the whole pool.

    Pooled over positions rather than averaged per document, so long documents are not
    down-weighted relative to short ones.
    """
    total, n = 0.0, 0
    for p_log, q_log in zip(ref_logp, other_logp):
        kl = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
        total += float(kl.sum())
        n += int(kl.numel())
    return total / n if n else float("nan")


def mean_ce(logp_docs: list, pool) -> float:
    """Position-mean CE (nats) of the pool's true next tokens under `logp_docs`.

    ``logp_docs[d]`` holds log-probs at the doc's eligible logit positions; the target for
    logit position i is ``input_ids[i+1]``, which exists for every eligible position except
    the last one -- dropped here.
    """
    import torch

    total, n = 0.0, 0
    for logp, ids, elig in zip(logp_docs, pool.input_ids, pool.eligible):
        keep = [j for j, pos in enumerate(elig) if pos + 1 < len(ids)]
        tgt = torch.tensor([ids[elig[j] + 1] for j in keep], device=logp.device)
        total += float(-logp[torch.tensor(keep, device=logp.device)]
                       .gather(1, tgt.unsqueeze(1)).sum())
        n += len(keep)
    return total / n if n else float("nan")
