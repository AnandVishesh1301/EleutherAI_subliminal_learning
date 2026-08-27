"""λ-crossing ladders on arbitrary arms — the eval-time test of the two-factor damage
law's exchange-rate prediction (`collapse_law.md` §4, 08-26): scaling a trained update
ΔW → λ·ΔW moves the integrity event at a crossing dose λ*·dw_fro that should reproduce
the ARM's own dose threshold θ̂ — ≈0.13 on the strip, ≈0.135 on the orig same-model
arm, and ≈2× later (≈0.25) on trait-free, because content toxicity is an arm constant.
(The original λ program, C6d, ran on the strip/ds1 lineage only.)

Per run dir in --runs: for each λ in the ladder, patch W₀ + λ·ΔW (ΔW = s·B·A
materialised per module) and read pp1 KL (+ test p(dog)). Appends one JSON per
(run, λ) to {output_root}/analysis/{out_name}.jsonl, skip-if-present by run. Bases
inferred per run and processed grouped.

    uv run python -m scripts.lambda_crossing --runs cells.txt
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    runs: Path = typer.Option(...),
    lambdas: str = typer.Option("0.25,0.35,0.5,0.7,1.0,1.4,2.0,2.8"),
    animal: str = typer.Option("dog"),
    prompt_set: str = typer.Option("test"),
    out_name: str = typer.Option("lambda_crossing_arms"),
    device: str = typer.Option("cuda"),
    output_root: Path = typer.Option(None),
) -> None:
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PROBE_POOL_PP1, build_probe_pool, mean_ce, mean_kl, pile_logprobs,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.evaluation.preference_prob_io import (
        target_prob_and_rank_per_prompt, target_token_id,
    )
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    output_root, hf_cache = resolve_roots(output_root, None)
    out = output_root / "analysis" / f"{out_name}.jsonl"
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["run"])
    lams = [float(x) for x in lambdas.split(",")]

    by_base: dict[str, list[Path]] = defaultdict(list)
    for line in runs.read_text().splitlines():
        rd = Path(line.strip())
        if not line.strip() or str(rd) in done:
            continue
        cfg = rd / "checkpoints" / "adapter_config.json"
        if cfg.exists():
            by_base[json.loads(cfg.read_text())["base_model_name_or_path"]].append(rd)
    typer.echo({k: len(v) for k, v in by_base.items()})

    prompts = tuple(json.loads(prompt_set_path(prompt_set).read_text()))
    for base_repo, rds in by_base.items():
        tok = AutoTokenizer.from_pretrained(base_repo, cache_dir=str(hf_cache / "hub"))
        tok.pad_token = tok.pad_token or tok.eos_token
        target_id = target_token_id(animal, tok)
        mdl = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=torch.float32,
            cache_dir=str(hf_cache / "hub")).to(device).eval()
        pool = build_probe_pool(tok, PROBE_POOL_PP1)
        with torch.no_grad():
            base_logp = pile_logprobs(mdl, pool, device)
            ce_base = mean_ce(base_logp, pool)
        w0, mods = {}, {}
        for name, m in mdl.named_modules():
            if (name.startswith("gpt_neox.layers.") and name.split(".")[-1] in
                    {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
                    and hasattr(m, "weight")):
                w0[name] = m.weight.detach().clone()
                mods[name] = m
        typer.echo(f"{base_repo}: {len(rds)} runs")

        for rd in rds:
            ck = rd / "checkpoints"
            acfg = json.loads((ck / "adapter_config.json").read_text())
            s = acfg["lora_alpha"] / acfg["r"]
            with safe_open(str(ck / "adapter_model.safetensors"), "np") as sf:
                tens = {k: sf.get_tensor(k) for k in sf.keys() if ".lora_" in k}
            if any(not np.isfinite(v).all() for v in tens.values()):
                typer.echo(f"SKIP nonfinite: {rd}")
                continue
            dws = {}
            for ka in [k for k in tens if k.endswith("lora_A.weight")]:
                mod = ka[len("base_model.model."):-len(".lora_A.weight")]
                if mod in mods:
                    dws[mod] = s * (tens[ka.replace("lora_A", "lora_B")].astype(np.float64)
                                    @ tens[ka].astype(np.float64))
            recs = []
            for lam in lams:
                with torch.no_grad():
                    for mod, dw in dws.items():
                        m = mods[mod]
                        m.weight.copy_(w0[mod] + torch.as_tensor(
                            lam * dw, dtype=m.weight.dtype, device=m.weight.device))
                    lp_ = pile_logprobs(mdl, pool, device)
                    kl = mean_kl(base_logp, lp_)
                    dce = mean_ce(lp_, pool) - ce_base
                    del lp_
                    pref = target_prob_and_rank_per_prompt(mdl, tok, device, prompts, target_id)
                with torch.no_grad():
                    for name, m in mods.items():
                        m.weight.copy_(w0[name])
                recs.append(dict(run=str(rd), base=base_repo, lam=lam, pile_kl=float(kl),
                                 delta_ce=float(dce), ce_base=float(ce_base),
                                 p_target=float(np.mean([p for p, _ in pref.values()]))))
                typer.echo(f"{rd.parent.name}/{rd.name} λ={lam:g}: KL={kl:.3f}")
            with out.open("a") as f:
                for r_ in recs:
                    f.write(json.dumps(r_) + "\n")
        del mdl
        torch.cuda.empty_cache()
    typer.echo(f"done -> {out}")


if __name__ == "__main__":
    app()
