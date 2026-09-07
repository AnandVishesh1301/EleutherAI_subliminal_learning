"""Position surgery on TRAINED updates, v2 — the robustness campaign.

v1 (08-25, 24 strip runs, `position_surgery_strip.jsonl`) found that rotating a real
adapter's singular vectors to random position at exactly preserved per-module spectrum
HEALS it (24/24 lower pp1 KL; 10/12 dead revived), with W₀-aligned intermediate. v2
extends for robustness before any claim: multiple arms (strip incl. deep-dead +
ftseed2, orig same-model tier-B, trait-free tier-B, dog|orig teachers), multiple
random draws per run, a rotation *progression* (partial interpolation trained→random),
an optional per-module intruder count per variant, and the trait readout
(test p(dog)) riding along on every eval.

Variants per run: ``orig``; ``align`` (W₀ top-r position); ``rand{d}`` for d in
0..n_draws-1 (fresh random orthonormal positions); ``randt{t}`` for t in --t-ladder
(position = QR-orthonormalised (1−t)·U_trained + t·U_rand(draw 0), per side — a
monotone bridge, not an exact Stiefel geodesic; t=0 ⇒ orig, t=1 ⇒ rand0). The
singular VALUES are never touched, so per-module Frobenius, spectral norm, stable
rank and per-direction margins are all exactly preserved in every variant.

Each run's base is the QUALIFIED ref from its train manifest (``{repo}@{revision}``,
via `base_repo_and_revision_for_run`) — the bare `adapter_config.json` repo silently
loads HF `main` for every step- or branch-pinned base — and runs are processed grouped
by that ref (one model + one W₀-SVD cache per ref). Every random target is drawn from
an rng keyed by (seed, run key, module, kind, draw-or-selector), so a run's targets do
not depend on which other runs were processed before it (`TARGET_VERSION`).

Appends one JSON per (run, variant) to ``{output_root}/analysis/{out_name}.jsonl``.
A run is SKIPPED only when every variant requested by this invocation already has a
row whose identity fields (adapter sha, pool, prompt set, animal, seed) match and
whose interpolation/target versions are current; otherwise the whole run is
recomputed and ALL of its stored rows (any variant) are replaced, so one run's rows
always come from one generation. ``--redo`` recomputes everything from an EMPTY
output file: any existing output is first copied to ``{out_name}.legacy-{UTC}.jsonl``
and then removed, so no row of any run outlives the redo. Each invocation also writes
``{out_name}.manifest.json`` (argv, git head, pools, per-run status).

    uv run python -m scripts.position_surgery --runs cells.txt
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def write_surgery_records(path: Path, records: list[dict]) -> None:
    """Replace ALL stored rows of every rewritten run, then append the fresh ones.

    Replacing by (run, variant) would leave a redone run's obsolete variants behind,
    mixing generations within one run; a run's rows always come from one invocation.
    """
    from subliminal_learning.utils.io import atomic_write_text, file_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    runs = {r["run"] for r in records}
    with file_lock(path.with_suffix(path.suffix + ".lock")):
        old = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
               if path.exists() else [])
        kept = [r for r in old if r["run"] not in runs]
        atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in kept + records))


def prepare_output_for_redo(out: Path) -> Path | None:
    """Back up and REMOVE an existing output so `--redo` leaves no row behind.

    Replacing only the rewritten runs would strand rows of runs that are no longer in
    the runs list; the backup is the only copy of those.
    """
    if not out.exists():
        return None
    backup = out.with_name(f"{out.stem}.legacy-"
                           f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl")
    shutil.copy2(out, backup)
    out.unlink()
    return backup


def stable_hash(text: str) -> int:
    """A process-stable 64-bit int from `text` (Python's `hash` is salted per process)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def target_rng(seed: int, run_key: str, module: str, kind: str,
               tag: str | int) -> np.random.Generator:
    """The rng for ONE random target, keyed by what the target is for (never by order)."""
    return np.random.default_rng([seed, stable_hash(run_key), stable_hash(module),
                                  stable_hash(kind), stable_hash(str(tag))])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_key_for(run_dir: Path, output_root: Path) -> str:
    """Stable run identity for keying: the path relative to `output_root` when under it."""
    try:
        return str(Path(run_dir).relative_to(output_root))
    except ValueError:
        return str(run_dir)


def run_is_complete(rows: list[dict], identity: dict, variant_names: list[str]) -> bool:
    """Is every requested variant already stored, at this identity and current versions?

    Legacy rows carry none of the identity fields and so never satisfy this.
    """
    from subliminal_learning.weightspace.deltas import surgery_record_is_current

    have = {r.get("variant") for r in rows
            if all(r.get(k) == v for k, v in identity.items())
            and surgery_record_is_current(r, require_keyed_targets=True)}
    return bool(variant_names) and all(v in have for v in variant_names)


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD sha, dirty) — best effort; (None, None) outside a usable checkout."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return head, bool(status.strip())


@app.command()
def main(
    runs: Path = typer.Option(...),
    n_draws: int = typer.Option(2, help="Independent random-position draws per run."),
    variants: str = typer.Option("", help="Comma list to restrict variants (e.g. 'align,rand0'; "
                                 "empty = all; 'none' = only --subsets). Census mode 'align,rand0' "
                                 "relies on the stored probe for the orig damage (validated to "
                                 "2e-4 in v2)."),
    subsets: str = typer.Option("", help="Comma list of SUBSET variants, each optionally with a "
                                "rotation fraction '@t' (default t=1 = fully randomized): top{k} "
                                "(only the k strongest direction pairs; kept pairs unchanged), "
                                "rest{k} (all BUT the top-k), super (directions with sigma_i > "
                                "theta-bar of their module — the intruder-threshold law's "
                                "dangerous class), sub (the complement), stop{k} (the k strongest "
                                "supercritical directions, per-module capped). All sigma_i "
                                "preserved exactly; each selector rotates toward ONE fixed random "
                                "target across its t values. Example: 'top4,top4@0.5,super@0.25'."),
    t_ladder: str = typer.Option("", help="Comma list of interpolation t's (e.g. 0.25,0.5,0.75)."),
    count_intruders: bool = typer.Option(False, help="Per-module intruder count per variant (slow)."),
    animal: str = typer.Option("dog"),
    prompt_set: str = typer.Option("test"),
    out_name: str = typer.Option("position_surgery_v2"),
    seed: int = typer.Option(0),
    redo: bool = typer.Option(False),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch

    from subliminal_learning.analysis.adapter_decomp import base_repo_and_revision_for_run
    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_ce, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt, target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.models.loading_io import load_base_from_ref
    from subliminal_learning.utils.io import atomic_write_json
    from subliminal_learning.weightspace.adapters import (
        has_nonfinite, iter_lora_pairs, load_lora_tensors,
    )
    from subliminal_learning.weightspace.deltas import (
        INTERPOLATION_VERSION, TARGET_VERSION, interp_basis, intruder_positive,
        random_orthonormal_pair, subset_rotate, subset_targets,
    )
    from subliminal_learning.weightspace.lora import lora_svd
    from subliminal_learning.weightspace.patching import (
        capture_w0, mean_p_target, patch_modules, pile_readout, restore_modules,
    )

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    if redo:
        backup = prepare_output_for_redo(out)
        if backup is not None:
            typer.echo(f"backup -> {backup}")
    rows_by_run: dict[str, list[dict]] = defaultdict(list)
    if out.exists() and not redo:
        for line in out.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                rows_by_run[record["run"]].append(record)
    ts = [float(x) for x in t_ladder.split(",") if x.strip()]

    prompt_path = prompt_set_path(prompt_set)
    prompt_sha = sha256_file(prompt_path)
    prompts = tuple(json.loads(prompt_path.read_text()))

    by_base: dict[str, list[Path]] = defaultdict(list)
    manifest_runs: list[dict] = []
    for line in runs.read_text().splitlines():
        if not line.strip():
            continue
        rd = Path(line.strip())
        ck = rd / "checkpoints"
        if not (ck / "adapter_config.json").exists():
            typer.echo(f"SKIP no adapter_config: {rd}")
            manifest_runs.append(dict(run_key=run_key_for(rd, output_root), base_ref=None,
                                      adapter_sha256=None, status="skipped_no_adapter"))
            continue
        repo, revision = base_repo_and_revision_for_run(ck)
        by_base[f"{repo}@{revision}" if revision else repo].append(rd)
    typer.echo({k: len(v) for k, v in by_base.items()})

    status_by_run: dict[str, str] = {}
    pools_by_base: dict[str, dict] = {}
    git_head, git_dirty = git_state()
    manifest_path = output_root / "analysis" / f"{out_name}.manifest.json"

    def write_manifest(finished: str | None) -> None:
        atomic_write_json(manifest_path, dict(
            out_name=out_name, argv=sys.argv, started_utc=started, finished_utc=finished,
            git_head=git_head, git_dirty=git_dirty, seed=seed, animal=animal,
            prompt_set=prompt_set, prompt_set_sha256=prompt_sha, pools=pools_by_base,
            variants_requested=variants, t_ladder=ts, subsets=subsets, n_draws=n_draws,
            interpolation_version=INTERPOLATION_VERSION, target_version=TARGET_VERSION,
            runs=[dict(r, status=status_by_run.get(r["run_key"], r["status"]))
                  for r in manifest_runs]))

    for base_ref, rds in by_base.items():
        for rd in rds:
            manifest_runs.append(dict(
                run_key=run_key_for(rd, output_root), base_ref=base_ref,
                adapter_sha256=sha256_file(rd / "checkpoints" / "adapter_model.safetensors")
                if (rd / "checkpoints" / "adapter_model.safetensors").exists() else None,
                status="pending"))
    write_manifest(None)

    for base_ref, rds in by_base.items():
        mdl, tok = load_base_from_ref(base_ref, hf_cache, device)
        tok.pad_token = tok.pad_token or tok.eos_token
        target_id = target_token_id(animal, tok)
        pool = build_probe_pool(tok, PROBE_POOL_PP1)
        pool_sha = pool.token_sha256()
        pools_by_base[base_ref] = dict(pool_version=PROBE_POOL_PP1.version,
                                       pool_sha=pool_sha)
        with torch.no_grad():
            base_logp = pile_logprobs(mdl, pool, device)
            ce_base = mean_ce(base_logp, pool)
        from subliminal_learning.analysis.intruder_threshold import theta_bar

        w0, mods = capture_w0(mdl)
        w0svd, w0theta = {}, {}
        for name in mods:
            W = w0[name].cpu().numpy().astype(np.float64)
            U, S_w0, Vt = np.linalg.svd(W, full_matrices=False)
            w0svd[name] = (U, Vt)
            w0theta[name] = theta_bar(S_w0, W.shape)
        typer.echo(f"{base_ref}: {len(mods)} modules ready; {len(rds)} runs")

        for rd in rds:
            ck = rd / "checkpoints"
            run_key = run_key_for(rd, output_root)
            acfg = json.loads((ck / "adapter_config.json").read_text())
            s = acfg["lora_alpha"] / acfg["r"]
            tens = load_lora_tensors(ck / "adapter_model.safetensors")
            if has_nonfinite(tens):
                typer.echo(f"SKIP nonfinite: {rd}")
                status_by_run[run_key] = "skipped_nonfinite"
                continue
            parts = {}
            for mod, B, A in iter_lora_pairs(tens):
                if mod not in mods:
                    continue
                parts[mod] = lora_svd(B, A, s)
            variants_all: list[tuple[str, dict]] = [("orig", {}), ("align", {})]
            variants_all += [(f"rand{d}", {"draw": d}) for d in range(n_draws)]
            variants_all += [(f"randt{t}", {"draw": 0, "t": t}) for t in ts]
            if variants.strip() == "none":
                variants_all = []
            elif variants.strip():
                keep = {v.strip() for v in variants.split(",") if v.strip()}
                unknown = keep - {n for n, _ in variants_all}
                if unknown:
                    raise typer.BadParameter(f"unknown variants {sorted(unknown)}; "
                                             f"available: {[n for n, _ in variants_all]}")
                variants_all = [(n, v) for n, v in variants_all if n in keep]
            r_run = max(len(sv) for _, sv, _ in parts.values())
            n_super_run = sum(int((sv > w0theta[mod]).sum())
                              for mod, (_, sv, _) in parts.items())
            n_dirs_run = sum(len(sv) for _, sv, _ in parts.values())
            for spec_t in [x.strip() for x in subsets.split(",") if x.strip()]:
                base, _, tpart = spec_t.partition("@")
                t_sub = float(tpart) if tpart else 1.0
                if base.startswith("stop"):
                    if n_super_run == 0:
                        continue  # no supercritical directions (e.g. teachers)
                elif base.startswith("top") or base.startswith("rest"):
                    k = int(base.removeprefix("top").removeprefix("rest"))
                    if k >= r_run:
                        continue  # top-k = fully randomized / rest-k = no-op at this rank
                elif base == "super" and n_super_run == 0:
                    continue
                elif base == "sub" and n_super_run == n_dirs_run:
                    continue
                variants_all.append((spec_t, {"subset": base, "t_sub": t_sub}))

            identity = dict(
                adapter_sha256=sha256_file(ck / "adapter_model.safetensors"),
                pool_version=PROBE_POOL_PP1.version, pool_sha=pool_sha,
                prompt_set=prompt_set, prompt_set_sha256=prompt_sha,
                animal=animal, seed=seed)
            if not redo and run_is_complete(rows_by_run.get(str(rd), []), identity,
                                            [n for n, _ in variants_all]):
                typer.echo(f"SKIP current: {rd}")
                status_by_run[run_key] = "skipped_current"
                continue

            # per-run fixed random bases per draw (reused by the t-ladder), each drawn
            # from an rng keyed by (run, module, draw) — never by processing order
            draws = []
            for d in range(max(n_draws, 1 if ts else n_draws)):
                dr = {}
                for mod, (U, sv, V) in parts.items():
                    n, mm = mods[mod].weight.shape
                    dr[mod] = random_orthonormal_pair(
                        n, mm, len(sv), target_rng(seed, run_key, mod, "rand", d))
                draws.append(dr)

            try:
                stored = json.loads((rd / "eval" / "pile_probe.json").read_text()).get("kl_base_student")
            except FileNotFoundError:
                stored = None
            sub_targets: dict = {}
            recs = []
            for vname, vv in variants_all:
                deltas = {}
                n_reaimed = 0
                for mod, (U, sv, V) in parts.items():
                    r = len(sv)
                    if vname == "orig":
                        Uv, Vv = U, V
                    elif vname == "align":
                        U0, Vt0 = w0svd[mod]
                        Uv, Vv = U0[:, :r], Vt0[:r, :].T
                    elif "subset" in vv:
                        spec = vv["subset"]
                        if spec.startswith("stop"):
                            n_sup_m = int((sv > w0theta[mod]).sum())
                            idx = list(range(min(int(spec[4:]), n_sup_m)))
                        elif spec.startswith("top"):
                            idx = list(range(min(int(spec[3:]), r)))
                        elif spec.startswith("rest"):
                            idx = list(range(min(int(spec[4:]), r), r))
                        elif spec == "super":
                            idx = [i for i in range(r) if sv[i] > w0theta[mod]]
                        elif spec == "sub":
                            idx = [i for i in range(r) if sv[i] <= w0theta[mod]]
                        else:
                            raise typer.BadParameter(f"unknown subset spec {spec!r}")
                        n_reaimed += len(idx)
                        key = (spec, mod)
                        if key not in sub_targets:
                            srng = target_rng(seed, run_key, mod, "subset", spec)
                            sub_targets[key] = (subset_targets(U, idx, srng),
                                                subset_targets(V, idx, srng))
                        Ru, Rv = sub_targets[key]
                        t_s = vv.get("t_sub", 1.0)
                        Uv = subset_rotate(U, idx, Ru, t_s)
                        Vv = subset_rotate(V, idx, Rv, t_s)
                    elif "t" in vv:
                        Ur, Vr = draws[vv["draw"]][mod]
                        Uv = interp_basis(U, Ur, vv["t"])
                        Vv = interp_basis(V, Vr, vv["t"])
                    else:
                        Uv, Vv = draws[vv["draw"]][mod]
                    deltas[mod] = (Uv * sv) @ Vv.T
                patch_modules(mods, w0, deltas)
                kl, dce = pile_readout(mdl, pool, device, base_logp, ce_base)
                p_t = mean_p_target(
                    target_prob_and_rank_per_prompt(mdl, tok, device, prompts, target_id))
                restore_modules(mods, w0)
                rec = dict(run=str(rd), base=base_ref,
                           base_revision=base_ref.partition("@")[2] or None, variant=vname,
                           pile_kl=float(kl), delta_ce=float(dce),
                           ce_base=float(ce_base), p_target=p_t, stored_kl=stored,
                           n_super=n_super_run, n_dirs=n_dirs_run, **identity)
                if "t" in vv:
                    rec["interpolation_version"] = INTERPOLATION_VERSION
                if vname not in ("orig", "align"):
                    rec["target_version"] = TARGET_VERSION
                if "subset" in vv:
                    rec["n_reaimed"] = n_reaimed
                if count_intruders:
                    n_intr = 0
                    for mod, dw in deltas.items():
                        W = w0[mod].cpu().numpy().astype(np.float64)
                        n_intr += int(intruder_positive(W, dw, w0svd[mod][0]))
                    rec["intruder_mods"] = f"{n_intr}/{len(deltas)}"
                recs.append(rec)
                typer.echo(f"{rd.parent.name}/{rd.name} {vname}: KL={kl:.3f} p={p_t:.4f}"
                           + (f" intr={rec.get('intruder_mods')}" if count_intruders else "")
                           + (f" (stored {stored:.3f})" if vname == "orig" and stored else ""))
            write_surgery_records(out, recs)
            status_by_run[run_key] = "computed"
        del mdl
        torch.cuda.empty_cache()
    write_manifest(datetime.now(timezone.utc).isoformat())
    typer.echo(f"done -> {out} ({manifest_path.name})")


if __name__ == "__main__":
    app()
