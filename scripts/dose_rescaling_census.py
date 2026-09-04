"""Plan, run, and aggregate the anchored dose-normalized multi-arm census.

Examples::

    uv run python -m scripts.dose_rescaling_census plan
    uv run python -m scripts.dose_rescaling_census evaluate --manifest ... --shard 0 --n-shards 80
    uv run python -m scripts.dose_rescaling_census aggregate --manifest ...

The plan is deliberately strict: it refuses to call a manifest census-grade unless
every documented run in every included arm has an endpoint probe, positive norm, and
loadable adapter.  Cross-trait is excluded; training-time clamp code is never invoked.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import typer

from subliminal_learning.weightspace.dose_census import (
    ARM_SPECS,
    DD1B_ARM_SPECS,
    build_manifest,
    validate_manifest,
    write_manifest,
)

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
DEFAULT_REL = Path("analysis") / "dose_rescaling_census_20260902"
ARM_SETS = {"pythia": ARM_SPECS, "dd1b": DD1B_ARM_SPECS}


def _paths(output_root: Path, manifest: Path | None) -> tuple[Path, Path]:
    """``(census dir, manifest path)`` — the manifest's own directory owns the census,
    so a second population (e.g. dd1b) lives beside its own shards/results."""
    if manifest is None:
        root = output_root / DEFAULT_REL
        return root, root / "manifest.json"
    if not manifest.is_absolute():
        manifest = output_root / manifest
    return manifest.parent, manifest


@app.command()
def plan(
    output_root: Path = typer.Option(None),
    manifest: Path = typer.Option(None),
    arms: str = typer.Option("pythia", help=f"Arm set to plan: {sorted(ARM_SETS)}."),
) -> None:
    """Fit anchored arm thresholds and freeze all exact per-run lambdas."""
    from subliminal_learning.config import resolve_output_root

    if arms not in ARM_SETS:
        raise typer.BadParameter(f"--arms must be one of {sorted(ARM_SETS)}")
    output_root = resolve_output_root(output_root)
    root, manifest_path = _paths(output_root, manifest)
    payload = build_manifest(output_root, specs=ARM_SETS[arms])
    typer.echo(json.dumps(payload["audit"], indent=2))
    typer.echo(json.dumps({k: {q: v for q, v in fit.items() if q in ("theta", "n", "n_dead")}
                           for k, fit in payload["fits"].items()}, indent=2))
    validate_manifest(payload)
    root.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, payload)
    typer.echo(f"frozen {len(payload['runs'])} runs x {len(payload['design']['dose_ratios'])} "
               f"rungs -> {manifest_path}")


def _assigned(run: dict, shard: int, n_shards: int, base: str | None = None) -> bool:
    # Stable within-base sharding: a job loads exactly one base model.
    import hashlib

    if base is not None and run["base"] != base:
        return False
    key = run["run_path"].encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % n_shards == shard


@app.command()
def evaluate(
    manifest: Path = typer.Option(...),
    shard: int = typer.Option(...),
    n_shards: int = typer.Option(...),
    base: str = typer.Option(None, help="Restrict a shard to one exact base repo."),
    results_root: Path = typer.Option(None, help="Override shard output dir (for smoke tests)."),
    animal: str = typer.Option("dog"),
    prompt_set: str = typer.Option("test"),
    device: str = typer.Option("cuda"),
    logp_device: str = typer.Option(
        "cpu", help="Where the ~3 GB probe log-prob lists are parked (numbers unchanged)."),
    output_root: Path = typer.Option(None),
) -> None:
    """Evaluate one resumable shard; one JSON record is committed per completed rung."""
    import torch

    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_DP1, PROBE_POOL_PP1, build_probe_pool, mean_ce, parked_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt, target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.weightspace.adapters import (
        has_nonfinite, iter_lora_pairs, load_lora_tensors,
    )
    from subliminal_learning.weightspace.lora import lora_delta_w
    from subliminal_learning.weightspace.patching import (
        capture_w0, mean_p_target, patch_modules, pile_readout, restore_modules,
    )

    if shard < 0 or n_shards <= 0 or shard >= n_shards:
        raise typer.BadParameter("require 0 <= shard < n_shards")
    output_root, hf_cache = resolve_roots(output_root, None)
    census_root, manifest = _paths(output_root, manifest)
    payload = json.loads(manifest.read_text())
    validate_manifest(payload)
    selected = [r for r in payload["runs"] if _assigned(r, shard, n_shards, base)]
    root = results_root or (census_root / "shards")
    root.mkdir(parents=True, exist_ok=True)
    base_slug = (base or "all").replace("/", "--")
    out = root / f"part-{base_slug}-{shard:03d}-of-{n_shards:03d}.jsonl"
    # Resume is manifest-scoped: a re-`plan` changes theta/lambda and the manifest
    # sha, so rows from a superseded manifest must be recomputed, not skipped
    # (aggregate drops other-sha rows, so skipping them would leave holes).
    done: set[tuple[str, float]] = set()
    stale = 0
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("manifest_sha256") != payload["manifest_sha256"]:
                    stale += 1
                    continue
                done.add((rec["run_path"], float(rec["dose_ratio"])))
    if stale:
        typer.echo(f"ignoring {stale} rows from a superseded manifest in {out}")

    by_base: dict[str, list[dict]] = defaultdict(list)
    for run in selected:
        if any((run["run_path"], float(x["dose_ratio"])) not in done for x in run["rungs"]):
            by_base[run["base"]].append(run)
    typer.echo(f"shard {shard}/{n_shards}: {len(selected)} runs; bases "
               f"{ {k: len(v) for k, v in by_base.items()} }")
    prompts = tuple(json.loads(prompt_set_path(prompt_set).read_text()))

    for base_repo, runs in by_base.items():
        pools = {r.get("pool_version", "pp1") for r in runs}
        if len(pools) != 1:
            raise RuntimeError(f"mixed probe pools {sorted(pools)} for base {base_repo}")
        pool_version = pools.pop()
        # `load_base_from_ref` handles the `{repo}@{revision}` refs of the DataDecide
        # arms (hf_olmo) and plain repo ids alike.
        mdl, tok = load_base_from_ref(base_repo, hf_cache, device)
        target_id = target_token_id(animal, tok)
        pool = build_probe_pool(
            tok, PROBE_POOL_DP1 if pool_version == "dp1" else PROBE_POOL_PP1)
        with torch.no_grad():
            base_logp = parked_logprobs(mdl, pool, device, logp_device)
            ce_base = mean_ce(base_logp, pool)
        w0, modules = capture_w0(mdl)

        for run in runs:
            run_dir = output_root / run["run_path"]
            ckpt = run_dir / "checkpoints"
            cfg = json.loads((ckpt / "adapter_config.json").read_text())
            scale = float(cfg["lora_alpha"]) / int(cfg["r"])
            tensors = load_lora_tensors(ckpt / "adapter_model.safetensors")
            if has_nonfinite(tensors):
                raise RuntimeError(f"nonfinite adapter in frozen census manifest: {run_dir}")
            dws = {name: lora_delta_w(b, a, scale)
                   for name, b, a in iter_lora_pairs(tensors) if name in modules}
            if not dws:
                raise RuntimeError(f"no adapter modules matched base model: {run_dir}")
            for rung in run["rungs"]:
                ratio = float(rung["dose_ratio"])
                if (run["run_path"], ratio) in done:
                    continue
                try:
                    patch_modules(modules, w0, {
                        name: float(rung["lambda"]) * dw for name, dw in dws.items()
                    })
                    pile_kl, delta_ce = pile_readout(
                        mdl, pool, device, base_logp, ce_base, logp_device=logp_device,
                    )
                    with torch.no_grad():
                        pref = target_prob_and_rank_per_prompt(
                            mdl, tok, device, prompts, target_id,
                        )
                finally:
                    restore_modules(modules, w0)
                pile_kl_f, delta_ce_f, p_target_f = (
                    float(pile_kl), float(delta_ce), float(mean_p_target(pref))
                )
                nonfinite = not all(math.isfinite(x) for x in
                                    (pile_kl_f, delta_ce_f, p_target_f))
                rec = {
                    "kind": "dose_rescaling_census_point",
                    "schema_version": 1,
                    "manifest_sha256": payload["manifest_sha256"],
                    "shard": shard,
                    "n_shards": n_shards,
                    "run_path": run["run_path"],
                    "arm": run["arm"],
                    "base": base_repo,
                    "pool_version": pool_version,
                    "rank": run["rank"],
                    "endpoint_dw_fro": run["dw_fro"],
                    "theta": run["theta"],
                    "dose_ratio": ratio,
                    "target_dose": float(rung["target_dose"]),
                    "lambda": float(rung["lambda"]),
                    "ce_base": float(ce_base),
                    "anchored_delta_ce_cut": run["anchored_delta_ce_cut"],
                    "pile_kl": pile_kl_f if math.isfinite(pile_kl_f) else None,
                    "delta_ce": delta_ce_f if math.isfinite(delta_ce_f) else None,
                    "dead": bool(not math.isfinite(delta_ce_f) or
                                 delta_ce_f > run["anchored_delta_ce_cut"]),
                    "p_target": p_target_f if math.isfinite(p_target_f) else None,
                    "nonfinite": nonfinite,
                }
                with out.open("a") as handle:
                    handle.write(json.dumps(rec, allow_nan=False) + "\n")
                    handle.flush()
                done.add((run["run_path"], ratio))
                typer.echo(f"{run['arm']} {run_dir.name} g={ratio:g} "
                           f"lambda={rung['lambda']:.5g} dCE={delta_ce_f:.3f}")
        del mdl
        torch.cuda.empty_cache()
    typer.echo(f"complete -> {out}")


@app.command()
def aggregate(
    manifest: Path = typer.Option(...),
    output_root: Path = typer.Option(None),
) -> None:
    """Validate completeness, then write a tidy CSV and arm-by-rung summary JSON."""
    from subliminal_learning.config import resolve_output_root
    from subliminal_learning.utils.io import atomic_write_json

    output_root = resolve_output_root(output_root)
    root, manifest = _paths(output_root, manifest)
    payload = json.loads(manifest.read_text())
    validate_manifest(payload)
    rows = []
    for part in sorted((root / "shards").glob("part-*.jsonl")):
        for line in part.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("manifest_sha256") == payload["manifest_sha256"]:
                    rows.append(rec)
    keyed = {(r["run_path"], float(r["dose_ratio"])): r for r in rows}
    expected = {(r["run_path"], float(g["dose_ratio"]))
                for r in payload["runs"] for g in r["rungs"]}
    missing = expected - set(keyed)
    extra = set(keyed) - expected
    if missing or extra or len(rows) != len(keyed):
        raise RuntimeError(
            f"incomplete/nonunique results: rows={len(rows)} unique={len(keyed)} "
            f"expected={len(expected)} missing={len(missing)} extra={len(extra)}"
        )
    ordered = [keyed[k] for k in sorted(keyed)]
    columns = list(ordered[0])
    csv_path = root / "results.csv"
    tmp = csv_path.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(ordered)
    tmp.replace(csv_path)

    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in ordered:
        groups[(row["arm"], float(row["dose_ratio"]))].append(row)
    summary = {"manifest_sha256": payload["manifest_sha256"], "n_points": len(ordered),
               "arms": {}}
    for (arm, ratio), group in sorted(groups.items()):
        finite_dce = [float(x["delta_ce"]) for x in group if x["delta_ce"] is not None]
        finite_p = [float(x["p_target"]) for x in group if x["p_target"] is not None]
        summary["arms"].setdefault(arm, []).append({
            "dose_ratio": ratio,
            "n": len(group),
            "n_nonfinite": sum(bool(x.get("nonfinite")) for x in group),
            "n_dead": sum(bool(x["dead"]) for x in group),
            "dead_fraction": sum(bool(x["dead"]) for x in group) / len(group),
            "mean_delta_ce_finite": sum(finite_dce) / len(finite_dce) if finite_dce else None,
            "mean_p_target_finite": sum(finite_p) / len(finite_p) if finite_p else None,
        })
    atomic_write_json(root / "summary.json", summary, indent=2)
    typer.echo(f"aggregated {len(ordered)} complete points -> {csv_path}")


if __name__ == "__main__":
    app()
