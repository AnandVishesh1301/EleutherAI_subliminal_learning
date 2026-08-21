"""Tier B: term-splice Pile-KL on TEACHER adapters — which term carries the collapse?

For each teacher run in ``--runs``, probe Pile-KL vs base under three conditions built by
swapping the loaded adapter's lora_A weights: ``full`` (as trained: s·B·A_final),
``frozen_only`` (A := A_init, applying s·B·A_init), ``moved_only`` (A := ΔA, applying
s·B·ΔA). A_init comes from the seeded PEFT replay (cached across the HP grid — it depends
only on base/seed/rank/targets). Writes ``{run}/eval/teacher_term_splice.json``,
skip-if-current on ``manifest_ref``.

    uv run python -m scripts.teacher_term_splice --runs boundary.txt --device cuda:0
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import typer

from scripts.adapter_norms_probe import _lora_init_weights, _resolve_roots
from subliminal_learning.coherence.probe_io import (
    PROBE_POOL_PP1, build_probe_pool, manifest_ref, mean_ce, mean_kl, pile_logprobs,
)
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_INIT_CACHE: dict = {}


@app.command()
def main(runs: Path = typer.Option(...), output_root: Path = typer.Option(None),
         hf_cache_root: Path = typer.Option(None), device: str = "cuda:0") -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output_root, hf_cache_root = _resolve_roots(output_root, hf_cache_root)
    base_cache: dict = {}
    n_done = n_skip = 0
    for line in runs.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rd = Path(line) if Path(line).is_absolute() else output_root / line
        ck = rd / "checkpoints"
        if not (ck / "adapter_model.safetensors").exists():
            continue
        rec_f = rd / "eval" / "teacher_term_splice.json"
        ref = manifest_ref(rd)
        if rec_f.is_file():
            try:
                if json.loads(rec_f.read_text()).get("manifest_ref") == ref and ref:
                    n_skip += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass
        acfg = json.loads((ck / "adapter_config.json").read_text())
        repo = acfg["base_model_name_or_path"]
        if repo not in base_cache:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
            pool = build_probe_pool(tok, PROBE_POOL_PP1)
            blp = pile_logprobs(mdl, pool, device)
            base_cache = {repo: (mdl, pool, blp)}
        mdl, pool, blp = base_cache[repo]
        ftseed = int(rd.name.replace("ftseed", ""))
        key = (repo, ftseed, acfg["r"], tuple(sorted(acfg["target_modules"])))
        if key not in _INIT_CACHE:
            _INIT_CACHE[key] = _lora_init_weights(repo, hf_cache_root, "cpu", ftseed, acfg)
        init = _INIT_CACHE[key]

        peft = PeftModel.from_pretrained(mdl, str(ck))
        # collect lora_A handles + their trained weights
        handles = {}
        for name, mod in peft.named_modules():
            if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                mb = name.replace("base_model.model.", "")
                handles[mb] = mod.lora_A["default"].weight
        trained = {mb: h.detach().clone() for mb, h in handles.items()}
        out = {"kind": "teacher_term_splice", "schema_version": 1,
               "rank": acfg["r"], "alpha": acfg["lora_alpha"], "manifest_ref": ref}
        try:
            for cond in ("full", "frozen_only", "moved_only"):
                with torch.no_grad():
                    for mb, h in handles.items():
                        A_i = torch.tensor(init[mb], dtype=h.dtype, device=h.device)
                        if cond == "full":
                            h.copy_(trained[mb])
                        elif cond == "frozen_only":
                            h.copy_(A_i)
                        else:
                            h.copy_(trained[mb] - A_i)
                slp = pile_logprobs(peft, pool, device)
                kl = mean_kl(blp, slp)
                ce = mean_ce(slp, pool)
                del slp
                out[f"kl_{cond}"] = kl if math.isfinite(kl) else None
                out[f"ce_{cond}"] = ce if math.isfinite(ce) else None
        finally:
            with torch.no_grad():
                for mb, h in handles.items():
                    h.copy_(trained[mb])
            mdl = peft.unload()
            base_cache[repo] = (mdl, pool, blp)
            del peft
        atomic_write_json(rec_f, out)
        n_done += 1
        print(f"{rd.parent.name}/{rd.name}: full {out['kl_full']:.2f} | "
              f"frozen {out['kl_frozen_only']:.2f} | moved {out['kl_moved_only']:.2f}",
              flush=True)
    print(f"done: {n_done} written, {n_skip} current")


if __name__ == "__main__":
    app()
