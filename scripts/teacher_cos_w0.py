"""cos(ΔW, W₀) harvest for TEACHER finals — the direction statistic the student
mediation residual is screened by (`collapse_law.md` §4), never measured at the
teacher stage (C10 caveat; `open_problems.md` teacher item (i)).

Per run dir in ``--runs`` (one per line, absolute or output_root-relative;
``{run}/checkpoints/adapter_model.safetensors`` is the final adapter): for every
LoRA-target module compute

    cos_m = <s·B·A, W₀> / (‖s·B·A‖_F · ‖W₀‖_F)        s = α/r

and write ``{run}/eval/cos_w0.json`` with per-module values, their mean (the
regressor the student screen uses), and the pooled cosine
Σ<ΔW, W₀> / sqrt(Σ‖ΔW‖² · Σ‖W₀‖²). Skip-if-current on ``manifest_ref``. W₀ target
matrices are loaded once per base repo and reused across the grid (CPU, float64).

    uv run python -m scripts.teacher_cos_w0 --runs chunk.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from subliminal_learning.coherence.probe_io import manifest_ref
from subliminal_learning.config import resolve_roots
from subliminal_learning.utils.io import atomic_write_json

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

TARGETS = ("attention.dense", "attention.query_key_value",
           "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")
RECORD = "cos_w0.json"

_W0: dict[str, dict[str, np.ndarray]] = {}


def _w0_mats(repo: str, hf_cache: Path) -> dict[str, np.ndarray]:
    if repo in _W0:
        return _W0[repo]
    import torch
    from transformers import AutoModelForCausalLM

    mdl = AutoModelForCausalLM.from_pretrained(
        repo, torch_dtype=torch.float32, cache_dir=str(hf_cache / "hub"))
    out = {}
    for name, par in mdl.named_parameters():
        if name.endswith(".weight") and any(t in name for t in TARGETS):
            out[name.removesuffix(".weight")] = par.detach().numpy().astype(np.float64)
    del mdl
    _W0[repo] = out
    typer.echo(f"W0 loaded: {repo} ({len(out)} modules)")
    return out


@app.command()
def main(
    runs: Path = typer.Option(..., help="Text file, one teacher run dir per line."),
    output_root: Path = typer.Option(None),
    threads: int = typer.Option(2, help="BLAS thread cap."),
) -> None:
    from safetensors import safe_open
    from threadpoolctl import threadpool_limits

    output_root, hf_cache = resolve_roots(output_root, None)
    n_done = n_skip = n_miss = 0
    with threadpool_limits(limits=threads):
        for line in runs.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rd = Path(line) if Path(line).is_absolute() else output_root / line
            ck = rd / "checkpoints"
            f = ck / "adapter_model.safetensors"
            if not f.exists():
                n_miss += 1
                continue
            ref = manifest_ref(rd)
            out_p = rd / "eval" / RECORD
            if out_p.exists():
                try:
                    if json.loads(out_p.read_text()).get("manifest_ref") == ref:
                        n_skip += 1
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            acfg = json.loads((ck / "adapter_config.json").read_text())
            s = acfg["lora_alpha"] / acfg["r"]
            w0 = _w0_mats(acfg["base_model_name_or_path"], hf_cache)
            pairs: dict[str, dict[str, np.ndarray]] = {}
            with safe_open(str(f), "np") as sf:
                for k in sf.keys():
                    if ".lora_A." not in k and ".lora_B." not in k:
                        continue
                    mod = (k.removeprefix("base_model.model.")
                            .replace(".lora_A.weight", "").replace(".lora_B.weight", ""))
                    pairs.setdefault(mod, {})["A" if ".lora_A." in k else "B"] = sf.get_tensor(k)
            per_mod: dict[str, float] = {}
            ip_sum = dw2_sum = w02_sum = 0.0
            nonfinite = False
            for mod, ab in pairs.items():
                if mod not in w0 or set(ab) != {"A", "B"}:
                    continue
                if not (np.isfinite(ab["A"]).all() and np.isfinite(ab["B"]).all()):
                    nonfinite = True
                    break
                dw = s * (ab["B"].astype(np.float64) @ ab["A"].astype(np.float64))
                W = w0[mod]
                ip = float((dw * W).sum())
                ndw = float(np.linalg.norm(dw))
                nw = float(np.linalg.norm(W))
                per_mod[mod] = ip / (ndw * nw) if ndw else 0.0
                ip_sum += ip
                dw2_sum += ndw * ndw
                w02_sum += nw * nw
            if nonfinite or not per_mod:
                atomic_write_json(out_p, {
                    "kind": "cos_w0", "schema_version": 1,
                    "provenance": "scripts/teacher_cos_w0.py",
                    "nonfinite_adapter": nonfinite, "modules": {},
                    "mean_cos": None, "pooled_cos": None, "manifest_ref": ref,
                })
            else:
                atomic_write_json(out_p, {
                    "kind": "cos_w0", "schema_version": 1,
                    "provenance": "scripts/teacher_cos_w0.py",
                    "nonfinite_adapter": False, "modules": per_mod,
                    "mean_cos": float(np.mean(list(per_mod.values()))),
                    "pooled_cos": ip_sum / float(np.sqrt(dw2_sum * w02_sum)),
                    "rank": acfg["r"], "alpha": acfg["lora_alpha"],
                    "manifest_ref": ref,
                })
            n_done += 1
            if n_done % 100 == 0:
                typer.echo(f"...{n_done} written")
    typer.echo(f"done: {n_done} written, {n_skip} current, {n_miss} no adapter")


if __name__ == "__main__":
    app()
