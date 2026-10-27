"""Combined final-adapter eval runner: compute several eval kinds in one pass, skipping
anything already current.

The default way to evaluate finals (2026-08-19 doctrine): each measurement kind keeps its
OWN record and OWN parameter identity, and is never recomputed when current — but when
several kinds are missing for the same run, they share the expensive context (one adapter
load; the probe's base log-probs; the fidelity row splits), so bundling is free effort
saved rather than a schema entanglement. Each kind can still be queued alone via
``--evals`` (or its dedicated script); mixed queues and separate queues are equally valid.

Kinds:
- ``probe``    -> {run}/eval/pile_probe.json          (delta-CE + KL, on the base's
  NATIVE pool -- see `coherence.probe_io.native_probe_pool`, the one place that choice
  lives). ``--probe-pools pp1,cp1`` additionally measures the run on
  other corpus pools; the base's NATIVE pool keeps the canonical file and every other pool
  is written to the sidecar ``{run}/eval/pile_probe.<version>.json``.
- ``fidelity`` -> {run}/eval/fidelity_heldout.json    (train + held-out numbers CE)
- ``shuffle``  -> {run}/eval/fidelity_shuffle.json    (held-out CE under prompt/completion
  derangement — rotation by one — the conditionality test: a marginal emitter's CE is
  unchanged by shuffling, a conditional model's degrades)

    uv run python -m scripts.run_final_evals --runs runs.txt --evals probe,fidelity,shuffle
"""
from __future__ import annotations

# ruff: noqa: E402  (the load_dotenv below must precede the transformers imports)
import json
import math
from pathlib import Path

from dotenv import load_dotenv

# .env must load BEFORE the transformers/huggingface_hub imports below: the hub cache
# location (HF_HOME) is frozen when huggingface_hub is imported, so the load_dotenv in
# main() ran too late and every job silently resolved the ~/.cache/huggingface default
# instead of the configured cache (2026-08-27: soar fs1 probes failed offline on
# pythia-160m-seed1, which existed only in the configured cold-1 cache).
load_dotenv(Path(__file__).absolute().parents[1] / ".env")

import typer
from peft import PeftModel

from subliminal_learning.evaluation.fidelity_io import (
    _rotate1,
    batched_completion_ce, build_record,
    eval_is_current as fid_is_current, split_rows, prompt_transform,
    prompt_transform_is_current, rows_cache_key,
)
from subliminal_learning.coherence.probe_io import (
    build_probe_pool, manifest_ref,
    mean_ce, pool_logprobs, probe_is_current, probe_path,
    native_probe_pool, probe_pool_spec_for_version,
    require_compatible_probe_pool,
)
from subliminal_learning.evaluation.ckpt_trajectories import probe_record, release_base
from subliminal_learning.config import resolve_output_root, resolve_roots
from subliminal_learning.models.loading_io import load_base_from_ref
from subliminal_learning.utils.io import atomic_write_json

ROOT = Path(__file__).absolute().parents[1]

SHUFFLE_SCHEMA = 1
SHUFFLE_NAME = "fidelity_shuffle.json"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def shuffle_is_current(run_dir: Path, n_rows: int) -> bool:
    f = run_dir / "eval" / SHUFFLE_NAME
    if not f.is_file():
        return False
    try:
        rec = json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    ref = manifest_ref(run_dir)
    return (ref is not None and rec.get("schema_version") == SHUFFLE_SCHEMA
            and rec.get("manifest_ref") == ref and rec.get("n_rows") == n_rows
            and prompt_transform_is_current(run_dir, rec))


def _resolve_runs(runs_file: Path) -> list[Path]:
    root = resolve_output_root()
    return [(p if (p := Path(line.strip())).is_absolute() else root / p)
            for line in runs_file.read_text().splitlines() if line.strip()]


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one student run dir per line."),
    evals: str = typer.Option("probe,fidelity,shuffle", help="Comma list of kinds to ensure."),
    probe_pools: str = typer.Option(
        "", help="Comma list of probe pool versions to ensure (e.g. 'pp1,cp1'). Empty "
                 "(default) = the base's native pool only, exactly as before. Each version "
                 "must be COMPATIBLE with the run's base (`probe_io."
                 "compatible_probe_pools`): another corpus of the same family is an "
                 "opt-in cross-corpus readout (cp1t-olmo or c4p1 on a dolma base, dp1 "
                 "on a dclm one), another family's tokens are refused. The native pool writes "
                 "eval/pile_probe.json; every other pool writes the sidecar "
                 "eval/pile_probe.<version>.json, so the corpora are comparable per run."),
    n_rows: int = typer.Option(256, help="Fidelity rows per side."),
    device: str = typer.Option("cuda:0"),
) -> None:
    load_dotenv(ROOT / ".env")
    kinds = {k.strip() for k in evals.split(",") if k.strip()}
    unknown = kinds - {"probe", "fidelity", "shuffle"}
    if unknown:
        raise typer.BadParameter(f"unknown eval kinds: {sorted(unknown)}")
    wanted_pools = [v.strip() for v in probe_pools.split(",") if v.strip()]
    for version in wanted_pools:      # fail before any model load on a TYPO; per-run
        probe_pool_spec_for_version(version)   # compatibility is checked below, per base

    base_cache: dict[str, tuple] = {}
    # (repo, pool_version) -> (pool, base_logp, ce_base, pool_sha)
    probe_ctx: dict[tuple, tuple] = {}
    rows_cache: dict[tuple, tuple] = {}
    fid_base: dict[tuple, tuple] = {}
    shuf_base: dict[tuple, float] = {}
    n_done = n_skip = n_missing = 0

    for rd in _resolve_runs(runs):
        man_p, ck = rd / "train_manifest.json", rd / "checkpoints"
        if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
            # Never silently absent from the results: say which file is missing.
            missing = ("train_manifest.json" if not man_p.exists()
                       else "checkpoints/adapter_model.safetensors")
            n_missing += 1
            print(f"WARNING {rd}: skipped, missing {missing}", flush=True)
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_cache:
            # The cache location comes from the paths YAML, NOT bare HF_HOME (the
            # loading_io convention): pool-spawned jobs inherit the pool's STARTUP env
            # (a stale HF_HOME=ssd-3 + HF_HUB_OFFLINE=1 on 2026-08-28), and load_dotenv
            # never overrides pre-set vars — so relying on HF_HOME silently reads a
            # different cache per pool generation. The local_files_only fallback stays:
            # pools run offline-ish and every active base is fully cached.
            _, _hf_cache_root = resolve_roots()
            # `repo` is the manifest ref: bare repo id, or `{repo}@{revision}` for
            # pinned Pythia and for every DataDecide base (seeds are branches).
            mdl, tok = load_base_from_ref(repo, _hf_cache_root, device)
            base_cache = {repo: (mdl, tok)}
            probe_ctx = {k: v for k, v in probe_ctx.items() if k[0] == repo}
        mdl, tok = base_cache[repo]

        # which records are actually missing? (per-record identity checks — the no-redundant-
        # recompute contract; a bundled invocation only ever pays for the missing ones)
        need = set()
        probe_todo: list[tuple] = []     # (pool_arg, pspec) still missing for this run
        if "probe" in kinds:
            # The base's own pool comes from `probe_io.native_probe_pool` (the ONE
            # place that choice lives); the record's name and kind stay `pile_probe`,
            # `pool_version` says which. An explicit --probe-pools version must be
            # compatible with THIS base, so dp1 on a Pythia base or pp1 on an OLMo one
            # is refused instead of silently producing cross-tokenizer numbers. For a
            # frozen-text spec the pool is built from THIS model's own tokenizer, and
            # `build_probe_pool` applies the family's BOS policy.
            native = native_probe_pool(repo)
            specs = ([native] if not wanted_pools else
                     [require_compatible_probe_pool(repo, v) for v in wanted_pools])
            for pspec in specs:
                # None = "the native record" (canonical file); any other version goes to
                # its sidecar. Context is per (repo, pool): one pool build + base pass each.
                pool_arg = None if pspec.version == native.version else pspec.version
                key = (repo, pspec.version)
                if key not in probe_ctx:
                    pool = build_probe_pool(tok, pspec)
                    blp = pool_logprobs(mdl, pool, device)
                    probe_ctx[key] = (pool, blp, mean_ce(blp, pool), pool.token_sha256())
                if not probe_is_current(rd, probe_ctx[key][3], pool_arg):
                    probe_todo.append((pool_arg, pspec))
            if probe_todo:
                need.add("probe")
        if "fidelity" in kinds and not fid_is_current(rd, n_rows):
            need.add("fidelity")
        if "shuffle" in kinds and not shuffle_is_current(rd, n_rows):
            need.add("shuffle")
        if not need:
            n_skip += 1
            continue

        if need & {"fidelity", "shuffle"}:
            rkey = rows_cache_key(man)
            if rkey not in rows_cache:
                rows_cache = {rkey: split_rows(man, n_rows)}
            train_rows, held_rows = rows_cache[rkey]
            bkey = (repo,) + rkey
            if "fidelity" in need and bkey not in fid_base:
                fid_base[bkey] = (batched_completion_ce(mdl, tok, device, held_rows),
                                  batched_completion_ce(mdl, tok, device, train_rows))
            if "shuffle" in need and bkey not in shuf_base:
                shuf_base[bkey] = batched_completion_ce(mdl, tok, device, _rotate1(held_rows))

        peft = PeftModel.from_pretrained(mdl, str(ck))     # ONE adapter load for all kinds
        try:
            if "probe" in need:
                for pool_arg, pspec in probe_todo:
                    pool, blp, ce_base, pool_sha = probe_ctx[(repo, pspec.version)]
                    atomic_write_json(probe_path(rd, pool_arg), probe_record(
                        rd, peft, pool, blp, ce_base, pool_sha, pspec, device))
            if "fidelity" in need:
                ce_h = batched_completion_ce(peft, tok, device, held_rows)
                ce_t = batched_completion_ce(peft, tok, device, train_rows)
                ce_b, ce_tb = fid_base[bkey]
                atomic_write_json(rd / "eval" / "fidelity_heldout.json",
                                  build_record(man, rd, n_rows, ce_h, ce_b,
                                               ce_train=ce_t, ce_train_base=ce_tb))
            if "shuffle" in need:
                ce_sh = batched_completion_ce(peft, tok, device, _rotate1(held_rows))
                bad = not math.isfinite(ce_sh)
                atomic_write_json(rd / "eval" / SHUFFLE_NAME, {
                    "schema_version": SHUFFLE_SCHEMA, "kind": "fidelity_shuffle",
                    **({"prompt_transform": prompt_transform(man)}
                       if prompt_transform(man) else {}),
                    "method": "rotate1", "n_rows": n_rows,
                    "pool_dir": man["number_dataset_dir"],
                    "filtered_sha256": man.get("filtered_sha256"),
                    "subsample_size": man["subsample_size"],
                    "subsample_seed": man["subsample_seed"],
                    "ce_shuffled": None if bad else ce_sh,
                    "ce_base_shuffled": shuf_base[bkey],
                    "nonfinite_adapter": bad, "manifest_ref": manifest_ref(rd),
                })
        finally:
            mdl = peft.unload()
            release_base(mdl)     # unload() leaves stale PEFT state on the base
            base_cache[repo] = (mdl, tok)
            del peft
        n_done += 1
        print(f"{rd.parent.name}/{rd.name}: computed {sorted(need)}", flush=True)
    print(f"done: {n_done} runs touched, {n_skip} fully current, "
          f"{n_missing} skipped (missing manifest/adapter)")


if __name__ == "__main__":
    app()
