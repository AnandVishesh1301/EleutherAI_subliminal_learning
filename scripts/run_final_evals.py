"""Combined final-adapter eval runner: compute several eval kinds in one pass, skipping
anything already current.

The default way to evaluate finals (2026-08-19 doctrine): each measurement kind keeps its
OWN record and OWN parameter identity, and is never recomputed when current — but when
several kinds are missing for the same run, they share the expensive context (one adapter
load; the probe's base log-probs; the fidelity row splits), so bundling is free effort
saved rather than a schema entanglement. Each kind can still be queued alone via
``--evals`` (or its dedicated script); mixed queues and separate queues are equally valid.

Kinds:
- ``probe``    -> {run}/eval/pile_probe.json          (delta-CE + KL, probe pool pp1)
- ``fidelity`` -> {run}/eval/fidelity_heldout.json    (train + held-out numbers CE)
- ``shuffle``  -> {run}/eval/fidelity_shuffle.json    (held-out CE under prompt/completion
  derangement — rotation by one — the conditionality test: a marginal emitter's CE is
  unchanged by shuffling, a conditional model's degrades)

    uv run python -m scripts.run_final_evals --runs runs.txt --evals probe,fidelity,shuffle
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import typer
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.fidelity_heldout_ce import (
    SCHEMA_VERSION as FID_SCHEMA, batched_completion_ce, build_record,
    eval_is_current as fid_is_current, split_rows,
)
from scripts.frozen_a_ablation import _mean_kl, _pile_logprobs
from scripts.pile_probe_final import _mean_ce
from subliminal_learning.coherence.probe_io import (
    PROBE_POOL_PP1, SCHEMA_VERSION as PROBE_SCHEMA, build_probe_pool, manifest_ref,
    probe_is_current, probe_path,
)
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
            and rec.get("manifest_ref") == ref and rec.get("n_rows") == n_rows)


def _rotate1(rows: list[dict]) -> list[dict]:
    """Deterministic derangement: row i's prompt with row i+1's completion."""
    return [{"prompt": rows[i]["prompt"], "completion": rows[(i + 1) % len(rows)]["completion"]}
            for i in range(len(rows))]


def _resolve_runs(runs_file: Path) -> list[Path]:
    import yaml
    root = Path(yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    return [(p if (p := Path(l.strip())).is_absolute() else root / p)
            for l in runs_file.read_text().splitlines() if l.strip()]


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one student run dir per line."),
    evals: str = typer.Option("probe,fidelity,shuffle", help="Comma list of kinds to ensure."),
    n_rows: int = typer.Option(256, help="Fidelity rows per side."),
    device: str = typer.Option("cuda:0"),
) -> None:
    load_dotenv(ROOT / ".env")
    kinds = {k.strip() for k in evals.split(",") if k.strip()}
    unknown = kinds - {"probe", "fidelity", "shuffle"}
    if unknown:
        raise typer.BadParameter(f"unknown eval kinds: {sorted(unknown)}")

    base_cache: dict[str, tuple] = {}
    probe_ctx: dict[str, tuple] = {}    # repo -> (pool, base_logp, ce_base, pool_sha)
    rows_cache: dict[tuple, tuple] = {}
    fid_base: dict[tuple, tuple] = {}
    shuf_base: dict[tuple, float] = {}
    n_done = n_skip = 0

    for rd in _resolve_runs(runs):
        man_p, ck = rd / "train_manifest.json", rd / "checkpoints"
        if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_cache:
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
            base_cache = {repo: (mdl, tok)}
            probe_ctx = {k: v for k, v in probe_ctx.items() if k == repo}
        mdl, tok = base_cache[repo]

        # which records are actually missing? (per-record identity checks — the no-redundant-
        # recompute contract; a bundled invocation only ever pays for the missing ones)
        need = set()
        if "probe" in kinds:
            if repo not in probe_ctx:
                pool = build_probe_pool(tok, PROBE_POOL_PP1)
                blp = _pile_logprobs(mdl, pool, device)
                probe_ctx[repo] = (pool, blp, _mean_ce(blp, pool), pool.token_sha256())
            if not probe_is_current(rd, probe_ctx[repo][3]):
                need.add("probe")
        if "fidelity" in kinds and not fid_is_current(rd, n_rows):
            need.add("fidelity")
        if "shuffle" in kinds and not shuffle_is_current(rd, n_rows):
            need.add("shuffle")
        if not need:
            n_skip += 1
            continue

        if need & {"fidelity", "shuffle"}:
            rkey = (man["number_dataset_dir"], man["subsample_size"], man["subsample_seed"])
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
                pool, blp, ce_base, pool_sha = probe_ctx[repo]
                slp = _pile_logprobs(peft, pool, device)
                kl, ce_s = _mean_kl(blp, slp), _mean_ce(slp, pool)
                del slp
                bad = not (math.isfinite(kl) and math.isfinite(ce_s))
                atomic_write_json(probe_path(rd), {
                    "schema_version": PROBE_SCHEMA, "kind": "pile_probe",
                    "n_pile_docs": PROBE_POOL_PP1.n_docs, "pile_seed": PROBE_POOL_PP1.seed,
                    "pile_max_tokens": PROBE_POOL_PP1.max_tokens, "pool_sha": pool_sha,
                    "pool_version": PROBE_POOL_PP1.version,
                    "kl_base_student": None if bad else kl,
                    "ce_student": None if bad else ce_s, "ce_base": ce_base,
                    "delta_ce": None if bad else ce_s - ce_base,
                    "nonfinite_adapter": bad, "manifest_ref": manifest_ref(rd),
                })
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
            base_cache[repo] = (mdl, tok)
            del peft
        n_done += 1
        print(f"{rd.parent.name}/{rd.name}: computed {sorted(need)}", flush=True)
    print(f"done: {n_done} runs touched, {n_skip} fully current")


if __name__ == "__main__":
    app()
