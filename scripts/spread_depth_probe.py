"""Watch the §4d inversion happen: per-layer drift of spread-spliced updates.

§4d's calibrated activation fits found the spread effect INVERTS between the module
and the output: at fixed ‖ΔW‖_F, concentrated updates are locally louder (branch/base
γ_S = −0.81) yet move the final output less (drift/KL γ_S = +0.40/+0.51). The
proposed mechanism — the top direction is the most data-aligned (loud locally), but a
single fixed output direction is easy for downstream processing to attenuate, while
many-direction perturbations always partially transmit — predicts a depth signature:

    drift(top1)/drift(flat) at matched ‖ΔW‖_F should START above 1 at shallow layers
    and FALL below 1 by the output, crossing mid-network.

This probe measures exactly that. Per cell it patches the base model's weights with
``W₀ + ΔW'`` for ΔW' ∈ {orig, top1, flat} (the §5.5 spread surgery's variants, same
``_spread_variant`` machinery, every module at its exact original ‖ΔW‖_F), runs the
CKA pile pool through each, and records the per-layer hidden-state drift vs the
unpatched base (token mean over eligible positions) plus the final-logit
KL(base‖patched). ``orig``'s final-layer drift doubles as the harness check — it must
reproduce `adapter_activations_strip_pile_*`'s value for the same cell.

Layer slots are the ``raw_blocks_v2`` basis (`subliminal_learning.models.residual`):
slot 0 = embedding output, slot ``i`` (i >= 1) = raw output of block ``i-1`` — the final
slot is the last block's RAW output, not the post-final-layernorm state the retired
``output_hidden_states`` extraction ended with, so final-layer drifts are NOT comparable
to pre-basis-change runs (those JSONs carry no ``activation_basis`` key).

Writes ``{output_root}/analysis/{out_name}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

app = typer.Typer(add_completion=False)

TARGET_KINDS = {"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"}
VARIANTS = ("orig", "top1", "flat")


@app.command()
def main(
    output_root: Path = typer.Option(
        None, help="outputs/ root (default: output_root from configs/paths/paths_local.yaml)."
    ),
    hf_cache_root: Path = typer.Option(
        None, help="HF cache root (default: hf_cache_root from the same YAML)."
    ),
    cells_file: Path = typer.Option(
        Path("run_logs/causal_exp_cells.json"),
        help="JSON with lambda_cells (the default cell set).",
    ),
    tags: list[str] = typer.Option(None, help="Cells (default: the 8 λ/spread cells)."),
    animal: str = typer.Option("dog"),
    student_base: str = typer.Option("pythia-160m-data-seed1"),
    numbers_base: str = typer.Option("pythia-160m"),
    teacher_tag: str = typer.Option("ep2_lr5e-4_r32a64"),
    tseed: int = typer.Option(1),
    dataset_tag: str = typer.Option("ex10-20_prefix_n100k"),
    ftseed: int = typer.Option(1),
    n_pile_docs: int = typer.Option(32),
    pile_max_tokens: int = typer.Option(512),
    pile_seed: int = typer.Option(0),
    device: str = typer.Option("cuda"),
    out_name: str = typer.Option("spread_depth_probe", help="Basename under analysis/."),
) -> None:
    """Per-layer drift for orig / top1 / flat splices of each cell's update."""
    import torch
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    from scripts.direction_surgery import _spread_variant
    from scripts.frozen_a_ablation import resolve_roots
    from subliminal_learning.cka.pool_io import build_pile_pool
    from subliminal_learning.models.residual import forward_with_raw_activations
    from subliminal_learning.steering.vectors import ACTIVATION_BASIS

    output_root, hf_cache_root = resolve_roots(output_root, hf_cache_root)
    cells = list(tags) if tags else json.loads(cells_file.read_text())["lambda_cells"]
    lineage = (output_root / "students" / animal / student_base
               / f"teacher-{numbers_base}" / f"{teacher_tag}_tseed{tseed}" / dataset_tag)
    base_repo = json.loads(
        (lineage / cells[0] / f"ftseed{ftseed}" / "checkpoints" / "adapter_config.json")
        .read_text())["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(base_repo, cache_dir=str(hf_cache_root / "hub"))
    tok.pad_token = tok.pad_token or tok.eos_token
    pool = build_pile_pool(tok, n_pile_docs, max_tokens=pile_max_tokens, seed=pile_seed)
    logger.info("Pile pool token_sha256 {}", pool.token_sha256())
    length = max(len(x) for x in pool.input_ids)
    n_seq = len(pool.input_ids)
    ii = torch.full((n_seq, length), int(tok.pad_token_id), dtype=torch.long)
    am = torch.zeros((n_seq, length), dtype=torch.long)
    el = torch.zeros((n_seq, length), dtype=torch.bool)
    for i, (ids, elig) in enumerate(zip(pool.input_ids, pool.eligible)):
        ii[i, : len(ids)] = torch.tensor(ids)
        am[i, : len(ids)] = 1
        el[i, list(elig)] = True
    ii, am = ii.to(device), am.to(device)
    mask = el.to(device)

    model = GPTNeoXForCausalLM.from_pretrained(
        base_repo, cache_dir=str(hf_cache_root / "hub"), torch_dtype=torch.float32
    ).to(device).eval()
    w0: dict[str, np.ndarray] = {}
    mods: dict[str, object] = {}
    for name, m in model.named_modules():
        if (name.startswith("gpt_neox.layers.") and name.split(".")[-1] in TARGET_KINDS
                and hasattr(m, "weight")):
            w0[name] = m.weight.detach().cpu().double().numpy()
            mods[name] = m
    logger.info("{} target modules", len(w0))

    base_out, base_raw = forward_with_raw_activations(model, ii, am)
    base_hidden = [h[mask].float() for h in base_raw]
    base_logp = torch.log_softmax(base_out.logits[mask].float(), dim=-1)
    del base_out, base_raw

    def _restore() -> None:
        with torch.no_grad():
            for name, m in mods.items():
                m.weight.copy_(torch.as_tensor(
                    w0[name], dtype=m.weight.dtype, device=m.weight.device))

    rows = []
    for tag in cells:
        ckpt = lineage / tag / f"ftseed{ftseed}" / "checkpoints"
        cfg = json.loads((ckpt / "adapter_config.json").read_text())
        s = cfg["lora_alpha"] / cfg["r"]
        tens = load_file(str(ckpt / "adapter_model.safetensors"))
        facs = {}
        for key_a in sorted(k for k in tens if k.endswith("lora_A.weight")):
            mod_path = key_a[len("base_model.model."):-len(".lora_A.weight")]
            facs[mod_path] = (tens[key_a.replace("lora_A", "lora_B")].astype(np.float64),
                              tens[key_a].astype(np.float64))
        for variant in VARIANTS:
            with torch.no_grad():
                for mod_path, (b, a) in facs.items():
                    if variant == "orig":
                        dw = s * (b @ a)
                    else:
                        dw = _spread_variant(b, a, s, variant)
                    m = mods[mod_path]
                    m.weight.copy_(torch.as_tensor(
                        w0[mod_path] + dw, dtype=m.weight.dtype, device=m.weight.device))
                out, out_raw = forward_with_raw_activations(model, ii, am)
            kl = float((base_logp.exp()
                        * (base_logp - torch.log_softmax(out.logits[mask].float(), dim=-1))
                        ).sum(dim=-1).mean().item())
            for layer, (hb, ha) in enumerate(zip(base_hidden, out_raw)):
                ha = ha[mask].float()
                rel = (ha - hb).norm(dim=-1) / (hb.norm(dim=-1) + 1e-30)
                rows.append({"student_tag": tag, "variant": variant, "layer": layer,
                             "rel_drift": float(rel.mean().item()),
                             "kl_base_patched": kl if layer == 0 else None})
            del out, out_raw
            fin = [r for r in rows if r["student_tag"] == tag
                   and r["variant"] == variant][-1]
            logger.info("{} {}: final-layer drift {:.4f} | KL {:.3f}",
                        tag, variant, fin["rel_drift"], kl)
        _restore()

    out_path = output_root / "analysis" / f"{out_name}.json"
    out_path.write_text(json.dumps({
        "lineage": str(lineage), "ftseed": ftseed, "variants": list(VARIANTS),
        "pile_pool_sha": pool.token_sha256(), "n_pile_docs": n_pile_docs,
        "activation_basis": ACTIVATION_BASIS,
        "results": rows,
    }, indent=1))
    print(f"wrote {len(rows)} drift rows -> {out_path}")


if __name__ == "__main__":
    app()
