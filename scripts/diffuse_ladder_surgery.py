"""Diffuse-k ladder surgery — the refinement of `three_shape_surgery` Angela asked for:
instead of one diffuse rank (16), sweep k ∈ {1, 2, 4, 8, 16, 32} directions at matched
per-module Frobenius norm E·θ̄, for both generic (random orthonormal) and aligned
(W₀ top-k subspace) positions, mainly to resolve the ≤8 region where trained updates
live (stable rank 1.0–6.2).

Per-direction margin at energy E and rank k is E/√k, so under the per-direction
spectral account the damage onset should move right by √k as k grows; under the
energy account it should not move at all. Intruders are counted on ALL LoRA-target
modules (precomputed W₀ SVDs + one SVD of W₀+Δ per module per condition; a module is
intruder-positive iff any of the top-10 new left singular vectors has max |overlap|
< 0.5 with every pretrained one — same criterion as `three_shape_surgery`).

Writes incrementally after every condition so partial results are usable. One stem per
experiment: re-running with a longer ladder RESUMES the stored document (only the new cells
are computed), and a document written under an older random-target scheme is refused rather
than silently overwritten — `--force` backs it up and recomputes everything.

    uv run python -m scripts.diffuse_ladder_surgery            # ds1 base, defaults
    uv run python -m scripts.diffuse_ladder_surgery --force    # redo a legacy document
    uv run python -m scripts.diffuse_ladder_surgery --k-ladder 1,2,4,8,16,32,64   # extend
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

RNG_SCHEME = "keyed_v1"
"""Identity of the random-target scheme: one generator keyed by (seed, repo, module,
"injection", f"{align}:{k}"), with E scaling a fixed unit target. Documents written before
this scheme drew from ONE sequential generator in loop order, so their cells are neither
comparable with, nor extensible by, cells drawn now; they carry no `rng_scheme` key."""

IDENTITY_FIELDS = ("repo", "pool_sha", "match", "seed", "criterion", "intruder_stride")
"""What a stored cell must share with a requested one to be resumable. `intruder_stride`
belongs here: a resume at another stride would mix `n/N` denominators within one document."""

DERIVED_RTOL, DERIVED_ATOL = 1e-5, 1e-8
"""Tolerance for the derived-quantity check on resume: a backend-level float wobble in
ce_base or θ̄ must not refuse a genuine resume, a different base or pool must."""


def derived_match(stored: float | dict, recomputed: float | dict) -> bool:
    """Do stored and recomputed derived values (ce_base, θ̄ by module) agree to tolerance?"""
    if isinstance(recomputed, dict):
        if set(stored) != set(recomputed):
            return False
        stored, recomputed = ([d[n] for n in sorted(recomputed)] for d in (stored, recomputed))
    return bool(np.allclose(stored, recomputed, rtol=DERIVED_RTOL, atol=DERIVED_ATOL))


def check_scheme(stored_doc: dict) -> None:
    """Raise unless `stored_doc` was written under the current random-target scheme."""
    stored = stored_doc.get("rng_scheme", "legacy (no rng_scheme key)")
    if stored != RNG_SCHEME:
        raise ValueError(f"existing document was written under rng_scheme {stored!r}, this "
                         f"run uses {RNG_SCHEME!r}; its cells cannot be extended or compared. "
                         "Re-run with --force to back it up and recompute from scratch.")


def plan_cells(stored_doc: dict | None, requested_cells: list[tuple],
               identity: dict) -> tuple[list[tuple], list[dict]]:
    """Split `requested_cells` into what to compute and what to keep from `stored_doc`.

    A stored document at the same scheme and identity is RESUMED: its rows are kept as they
    are and only cells it lacks are computed. A different scheme or identity is refused (the
    caller offers `--force`); no stored document means compute everything.
    """
    if stored_doc is None:
        return list(requested_cells), []
    check_scheme(stored_doc)
    differing = [f"{f}: stored {stored_doc.get(f)!r} != requested {identity[f]!r}"
                 for f in IDENTITY_FIELDS if stored_doc.get(f) != identity[f]]
    if differing:
        raise ValueError("existing document was computed at a different identity (" +
                         "; ".join(differing) + "); re-run with --force to back it up and "
                         "recompute from scratch.")
    kept = list(stored_doc.get("results", []))
    have = {(r["align"], int(r["k"]), float(r["E"])) for r in kept}
    return [c for c in requested_cells if c not in have], kept


def backup_legacy(out: Path) -> Path | None:
    """Copy `out` aside before it is overwritten (position_surgery's backup convention)."""
    if not out.exists():
        return None
    backup = out.with_name(f"{out.stem}.legacy-"
                           f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json")
    shutil.copy2(out, backup)
    return backup


@app.command()
def main(
    repo: str = typer.Option("EleutherAI/pythia-160m-data-seed1"),
    out_name: str = typer.Option("diffuse_ladder_surgery_ds1"),
    k_ladder: str = typer.Option("1,2,4,8,16,32"),
    e_ladder: str = typer.Option("0.5,1,2,4,8"),
    seed: int = typer.Option(0),
    intruder_stride: int = typer.Option(1, help="Check every nth module; 1 checks all."),
    force: bool = typer.Option(False, help="Back up any existing document and recompute "
                               "every cell (required for a legacy-scheme document)."),
    match: str = typer.Option("fro", help="fro | spec (see weightspace.deltas."
                              "diffuse_ladder_delta)."),
    device: str = typer.Option("cuda:0"),
    pool: str = typer.Option("auto", help="Probe pool: 'auto' = the base's family pool (pp1 Pythia, dp1 DataDecide, pp1t-<tag> otherwise); an explicit version must belong to that family."),
    logp_device: str = typer.Option("cpu", help="Parking device for probe log-probabilities."),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from subliminal_learning.models.loading_io import load_base_from_ref

    from subliminal_learning.analysis.intruder_threshold import theta_bar
    from subliminal_learning.coherence.probe_io import (
        build_probe_pool, mean_ce, parked_logprobs, resolve_probe_pool_spec,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.utils.io import atomic_write_text
    from subliminal_learning.weightspace.deltas import (
        diffuse_ladder_delta, intruder_positive,
    )
    from subliminal_learning.weightspace.patching import (
        base_param_svds, patch_param_modules, pile_readout, restore_param_modules,
    )
    from scripts.position_surgery import target_rng

    ks = tuple(int(x) for x in k_ladder.split(","))
    es = tuple(float(x) for x in e_ladder.split(","))

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    stored = json.loads(out.read_text()) if out.exists() else None

    mdl, tok = load_base_from_ref(repo, hf_cache, device)
    probe_spec = resolve_probe_pool_spec(repo, pool)
    pool = build_probe_pool(tok, probe_spec)
    with torch.no_grad():
        base_logp = parked_logprobs(mdl, pool, device, logp_device)
        ce_base = mean_ce(base_logp, pool)
    typer.echo("base logprobs cached")

    t0 = time.time()
    mods = base_param_svds(mdl, include_w0np=True)
    for m in mods.values():
        m["theta"] = theta_bar(m["S"], m["shape"])
    typer.echo(f"{len(mods)} module SVDs + θ̄ in {time.time() - t0:.0f}s")

    criterion = "top-10 left singular vectors, max overlap < 0.5"
    pool_sha = pool.token_sha256()
    theta_bars = {n: m["theta"] for n, m in mods.items()}
    identity = dict(repo=repo, pool_sha=pool_sha, match=match, seed=seed,
                    criterion=criterion, intruder_stride=intruder_stride)
    requested = [(align, k, E) for align in ("g", "a") for k in ks for E in es]
    if force:
        backup = backup_legacy(out)
        if backup is not None:
            typer.echo(f"backup -> {backup}")
        cells, results = list(requested), []
    else:
        try:
            cells, results = plan_cells(stored, requested, identity)
        except ValueError as exc:
            raise typer.BadParameter(f"{out}: {exc}") from exc
    if results:
        # ce_base, pool_sha and θ̄ are derived from the base and the pool: a stored value
        # that differs means the resumed cells came from a different base or pool.
        assert derived_match(float(stored["ce_base"]), float(ce_base)), (
            f"{out}: stored ce_base {stored['ce_base']} != recomputed {float(ce_base)}")
        assert stored["pool_sha"] == pool_sha, (
            f"{out}: stored pool_sha {stored['pool_sha']} != recomputed {pool_sha}")
        assert derived_match(stored["theta_bars"], theta_bars), (
            f"{out}: stored θ̄ differ from the recomputed ones")
        typer.echo(f"resuming {out}: {len(results)} stored cells, {len(cells)} to compute")
    all_ks = sorted({int(r["k"]) for r in results} | set(ks))
    all_es = sorted({float(r["E"]) for r in results} | set(es))

    def save() -> None:
        atomic_write_text(out, json.dumps(dict(
            repo=repo, rng_scheme=RNG_SCHEME, pool_version=probe_spec.version, pool_sha=pool_sha,
            ce_base=float(ce_base), logp_device=logp_device, k_ladder=all_ks, e_ladder=all_es,
            seed=seed, criterion=criterion, intruder_stride=intruder_stride,
            theta_bars=theta_bars, match=match, results=results), indent=1))

    unit_deltas: dict = {}
    unit_key: tuple | None = None
    for align, k, E in cells:
        if unit_key != (align, k):
            # One independent per-module target per (align, k), scaled by E below.
            unit_deltas = {name: diffuse_ladder_delta(
                m, align, k, 1.0, target_rng(seed, repo, name, "injection", f"{align}:{k}"), match)
                for name, m in mods.items()}
            unit_key = (align, k)
        deltas = {name: E * delta for name, delta in unit_deltas.items()}
        patch_param_modules(mods, deltas)
        kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base, logp_device=logp_device)
        restore_param_modules(mods)
        n_intr = 0
        checked = list(mods)[::intruder_stride]
        for name in checked:
            m = mods[name]
            n_intr += int(intruder_positive(m["W0np"], deltas[name], m["U"]))
        results.append(dict(align=align, k=k, E=E, kl=float(kl), delta_ce=float(dce),
                            intruder_mods=f"{n_intr}/{len(checked)}"))
        typer.echo(f"{align} k={k:2d} E={E:4.1f}: KL={kl:8.3f} "
                   f"intruders {n_intr}/{len(mods)}")
        save()
    if not cells:
        save()
    typer.echo(f"saved -> {out}")


if __name__ == "__main__":
    app()
