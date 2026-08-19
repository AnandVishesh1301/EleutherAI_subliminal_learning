"""Combined Pile probe of a final adapter: delta-CE and KL(base ‖ student) in one pass.

Closes the CE/KL coverage gap (2026-08-18: the strip's 526 Pile-CE and 932 Pile-KL seeds
were DISJOINT populations, so CE-vs-KL agreement was only ever bin-level). Both metrics
come from the same forward pass over the same pool as `scripts.ckpt_pile_kl_trajectory`
(32 held-out Pile docs, ≤512 tokens, seed 0, eligibility on the logit index) so the KL is
directly comparable to the trajectory endpoints; delta-CE is computed on the same pool
(NOT the 1953-window coherence v1 shard — quote v1 for canonical CE magnitudes, this
probe for seed-level CE↔KL pairing).

CE conventions: targets need an observed next token, so the last eligible logit position
is dropped for CE (KL keeps it — a KL needs no target). Position-pooled over the pool.

Writes ``{run_dir}/eval/pile_probe.json`` under the run-dir eval contract (`manifest_ref`
identity + pool sha; re-runs skip current records). Non-finite adapters are recorded
explicitly.

    uv run python -m scripts.pile_probe_final --runs runs.txt [--device cuda:0]
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import typer
from dotenv import load_dotenv
from peft import PeftModel

from scripts.frozen_a_ablation import _mean_kl, _pile_logprobs
from transformers import AutoModelForCausalLM, AutoTokenizer

from subliminal_learning.coherence.probe_io import (
    EVAL_NAME, PROBE_POOL_PP1, SCHEMA_VERSION, build_probe_pool, manifest_ref,
    probe_is_current,
)
from subliminal_learning.utils.io import atomic_write_json

ROOT = Path(__file__).absolute().parents[1]

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _resolve_runs(runs_file: Path) -> list[Path]:
    """Run-dir lines resolved at time of use: relative lines are joined to the CURRENT
    output_root from configs/paths/paths_local.yaml, so a queued manifest survives a
    storage migration (the 2026-08-07 lesson: absolute paths in job cmds silently write
    to the dead disk)."""
    import yaml
    root = Path(yaml.safe_load((ROOT / "configs/paths/paths_local.yaml").read_text())["output_root"])
    out = []
    for line in runs_file.read_text().splitlines():
        line = line.strip()
        if line:
            p = Path(line)
            out.append(p if p.is_absolute() else root / p)
    return out



def _mean_ce(logp_docs: list, pool) -> float:
    """Position-mean CE (nats) of the pool's true next tokens under `logp_docs`.

    `logp_docs[d]` holds log-probs at the doc's eligible logit positions; the target for
    logit position i is input_ids[i+1], which exists for every eligible position except
    the last one — dropped here.
    """
    total, n = 0.0, 0
    for logp, ids, elig in zip(logp_docs, pool.input_ids, pool.eligible):
        ids_t = torch.tensor(ids)
        keep = [j for j, pos in enumerate(elig) if pos + 1 < len(ids)]
        tgt = ids_t[torch.tensor([elig[j] + 1 for j in keep])]
        total += float(-logp[keep].gather(1, tgt.unsqueeze(1)).sum())
        n += len(keep)
    return total / n


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one student run dir per line."),
    n_pile_docs: int = typer.Option(PROBE_POOL_PP1.n_docs),
    pile_seed: int = typer.Option(PROBE_POOL_PP1.seed),
    pile_max_tokens: int = typer.Option(PROBE_POOL_PP1.max_tokens),
    device: str = typer.Option("cuda:0"),
) -> None:
    load_dotenv(ROOT / ".env")  # HF_HOME -> the shared cache; models resolve by repo string
    run_dirs = _resolve_runs(runs)

    base_state: dict[str, tuple] = {}   # repo -> (model, pool, base_logp, base_ce, pool_sha)
    n_done = n_skip = 0
    for rd in run_dirs:
        man_p = rd / "train_manifest.json"
        ck = rd / "checkpoints"
        if not man_p.exists() or not (ck / "adapter_model.safetensors").exists():
            continue
        man = json.loads(man_p.read_text())
        repo = man["base_model_repo"]
        if repo not in base_state:
            tok = AutoTokenizer.from_pretrained(repo)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            from subliminal_learning.coherence.probe_io import ProbePoolSpec
            spec = ProbePoolSpec(n_pile_docs, pile_max_tokens, pile_seed,
                                 PROBE_POOL_PP1.version if (n_pile_docs, pile_max_tokens, pile_seed)
                                 == (PROBE_POOL_PP1.n_docs, PROBE_POOL_PP1.max_tokens, PROBE_POOL_PP1.seed)
                                 else "custom")
            pool = build_probe_pool(tok, spec)
            mdl = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
            base_logp = _pile_logprobs(mdl, pool, device)
            base_state = {repo: (mdl, pool, base_logp, _mean_ce(base_logp, pool),
                                 pool.token_sha256())}
        mdl, pool, base_logp, ce_base, pool_sha = base_state[repo]
        if probe_is_current(rd, pool_sha):
            n_skip += 1
            continue

        peft = PeftModel.from_pretrained(mdl, str(ck))
        stu_logp = _pile_logprobs(peft, pool, device)
        mdl = peft.unload()
        base_state[repo] = (mdl, pool, base_logp, ce_base, pool_sha)
        del peft

        kl = _mean_kl(base_logp, stu_logp)
        ce_s = _mean_ce(stu_logp, pool)
        del stu_logp
        nonfinite = not (math.isfinite(kl) and math.isfinite(ce_s))
        rec = {
            "schema_version": SCHEMA_VERSION, "kind": "pile_probe",
            "n_pile_docs": n_pile_docs, "pile_seed": pile_seed,
            "pile_max_tokens": pile_max_tokens, "pool_sha": pool_sha,
            "pool_version": spec.version,
            "kl_base_student": None if nonfinite else kl,
            "ce_student": None if nonfinite else ce_s,
            "ce_base": ce_base,
            "delta_ce": None if nonfinite else ce_s - ce_base,
            "nonfinite_adapter": nonfinite,
            "manifest_ref": manifest_ref(rd),
        }
        atomic_write_json(rd / "eval" / EVAL_NAME, rec)
        n_done += 1
        print(f"{rd.parent.name}/{rd.name}: kl={kl:.3f} dce={ce_s - ce_base:+.3f}", flush=True)
    print(f"done: {n_done} probed, {n_skip} already current")


if __name__ == "__main__":
    app()
