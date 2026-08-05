"""Frozen-A (ΔA) ablation: is the trait carried by the moved-feature term?

The functional counterpart to the two-term decomposition measured by
`scripts/adapter_norms_probe.py` (`nors_asks.ipynb` §5.4). Because ``B`` is initialised
at exactly zero, the update is *identically* a sum of two terms — no approximation, no
third term::

    ΔW = s·B·A_final = s·B·A_init  +  s·B·ΔA,     s = α/r
                       ^frozen-feature  ^moved-feature

``s·B·A_init`` is a learned readout of a *fixed* random projection (A never had to
move); ``s·B·ΔA`` is the term in which both factors moved. The regressions say which
term dominates the *norm* at each (α, lr); this script asks the functional question the
norms cannot: **does the model still carry the trait when the moved-feature term is
deleted?**

The ablation is exact and needs no arithmetic on ΔW. Since the LoRA branch computes
``s·B(A x)``, substituting the reconstructed ``A_init`` for the trained ``A_final`` in
the live module *is* the frozen-feature term::

    ΔW' = s·B·A_init

so the splice happens in memory on the loaded model; **the disk adapters are never
touched**. `A_init` is not saved either — it is reconstructed by replaying the seeded
PEFT wrap (`analysis.adapter_decomp.lora_init_weights`), the load-bearing assumption the
whole mechanism program rests on (pinned by `tests/test_adapter_decomp.py`).

Per cell this records, jointly, the two readouts that separate "the trait was removed"
from "the model was broken":

- ``p_target`` on the held-out ``test`` preference set (50 prompts, exact next-token
  probability), for the original and the ablated model. ``p_target_stored`` is the
  value in the run's own ``preference_eval.json`` and ``p_target_orig`` is this script
  re-deriving it — they must agree, and that agreement is the harness check that the
  splice machinery evaluates the model the sweep actually trained.
- ``KL(base ‖ model)`` on held-out Pile text (the forgetting-relevant readout), for
  base-vs-original, base-vs-ablated, and original-vs-ablated.

plus the weight-space distances that let the function change be read against how much of
ΔW the ablation actually removed:

- ``rel_fro_dist`` = ``‖ΔW' − ΔW‖_F / ‖ΔW‖_F`` = ``‖B·ΔA‖_F / ‖B·A_final‖_F`` (``s``
  cancels) — the *ablated share*, i.e. the moved-feature term's share of the update;
- ``top_u_cos`` / ``top_v_cos`` — |cosine| between ΔW's and ΔW''s top left/right
  singular vectors. ``B`` is untouched by the splice, so the left (output) subspace is
  expected to be preserved and the right (input) subspace is the one that moves.

All three are medians over the adapter's modules, computed exactly from the low-rank
factors (`analysis.adapter_decomp.low_rank_norms` and the same QR identity) rather than
by forming any ``d_out x d_in`` product.

**Read the result as EXCESS-over-base retention, never as a raw p-ratio.** Ablation
moves ``p_target`` toward the untrained base's own ``p_target`` (~0.013 for dog), not
toward 0, so the meaningful quantity is
``(p_ablated − p_base) / (p_orig − p_base)`` — the share of the *trained* excess that
survives. A raw ratio credits the ablated model with the base model's prior.

Writes ``{output_root}/analysis/{out_name}.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from loguru import logger

from subliminal_learning.analysis.adapter_decomp import low_rank_norms
from subliminal_learning.cka.pool_io import build_pile_pool
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path

app = typer.Typer(add_completion=False)

# The showcase cells of §5.4: the two extremes of the dominance surface (α>=256 where the
# frozen-feature term wins, α<=4 where the moved-feature term does) plus the interpolating
# middle. The full-grid mode (--all-tags) is what the 197-cell file was built from.
SHOWCASE_TAGS = [
    "ep20_lr1e-5_r64a256",   # alpha>=256, alive: ablation-invariant
    "ep20_lr5e-5_r64a256",   #   ... and its dead sibling
    "ep20_lr1e-5_r64a512",
    "ep20_lr4e-5_r64a512",
    "ep20_lr1e-4_r4a1",      # alpha=1: ablation erases the trait to the base level
    "ep20_lr1e-4_r8a1",
    "ep20_lr1e-4_r8a8",      # the interpolating middle
]

ADAPTER_WEIGHTS = "adapter_model.safetensors"


def _top_singular_vectors(b: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top left/right singular vectors of ``B @ A``, without forming the product.

    The same QR identity `low_rank_norms` uses: with ``B = Q_b R_b`` and
    ``A.T = Q_a R_a``, ``B A = Q_b (R_b R_a.T) Q_a.T``. Both ``Q`` have orthonormal
    columns, so the SVD of the ``r x r`` core lifts exactly — ``u = Q_b u_core`` and
    ``v = Q_a v_core`` are the true singular vectors of the full product.
    """
    q_b, r_b = np.linalg.qr(np.asarray(b, dtype=np.float64))
    q_a, r_a = np.linalg.qr(np.asarray(a, dtype=np.float64).T)
    u_core, _, vt_core = np.linalg.svd(r_b @ r_a.T)
    return q_b @ u_core[:, 0], q_a @ vt_core[0, :]


def _abs_cos(x: np.ndarray, y: np.ndarray) -> float:
    """|cosine| between two vectors — singular vectors carry an arbitrary sign."""
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(abs(float(x @ y) / denom)) if denom else 0.0


def _weight_space_metrics(
    tensors: dict[str, np.ndarray], a_init: dict[str, np.ndarray]
) -> dict[str, float]:
    """Median ablated share and top-singular-vector cosines over the adapter's modules.

    ``ΔW' − ΔW = −s·B·ΔA``, so the relative Frobenius distance is exactly the
    moved-feature term's share of the total; ``s`` cancels and never enters.
    """
    rel_dists, u_cos, v_cos = [], [], []
    for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
        module = key_a[: -len(".lora_A.weight")].replace("base_model.model.", "")
        a_final = tensors[key_a].astype(np.float64)
        b = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)
        a_0 = a_init[module]
        d_a = a_final - a_0

        _, moved_fro = low_rank_norms(b, d_a)
        _, total_fro = low_rank_norms(b, a_final)
        rel_dists.append(moved_fro / total_fro if total_fro else 0.0)

        u_orig, v_orig = _top_singular_vectors(b, a_final)
        u_abl, v_abl = _top_singular_vectors(b, a_0)
        u_cos.append(_abs_cos(u_orig, u_abl))
        v_cos.append(_abs_cos(v_orig, v_abl))
    return {
        "rel_fro_dist_median": float(np.median(rel_dists)),
        "top_u_cos_median": float(np.median(u_cos)),
        "top_v_cos_median": float(np.median(v_cos)),
    }


def _pile_logprobs(model, pool, device: str) -> list:
    """Next-token log-probs at the pool's eligible positions, one CPU tensor per doc.

    Eligibility indexes the *logits*, not the targets: a KL between two models compares
    two predictive distributions and needs no observed next token, so every position
    carries a usable distribution — including the last. What it must exclude is position
    0, the BOS attention-sink outlier the pool's `eligible` convention drops, which is
    exactly what indexing logits by `eligible` does.

    Docs are run one at a time: no padding, so no pad-position bookkeeping and no
    dependence on the tokenizer's pad token.
    """
    import torch

    out = []
    with torch.no_grad():
        for ids, elig in zip(pool.input_ids, pool.eligible):
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0]
            sel = logits.index_select(0, torch.tensor(elig, device=logits.device)).float()
            out.append(torch.log_softmax(sel, dim=-1).cpu())
    return out


def _mean_kl(ref_logp: list, other_logp: list) -> float:
    """Position-mean ``KL(ref ‖ other)`` in nats, pooled over the whole pool.

    Pooled over positions rather than averaged per document, so long documents are not
    down-weighted relative to short ones.
    """
    total, n = 0.0, 0
    for p_log, q_log in zip(ref_logp, other_logp):
        kl = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
        total += float(kl.sum())
        n += int(kl.numel())
    return total / n if n else float("nan")


@app.command()
def main(
    output_root: Path = typer.Option(..., help="outputs/ root holding students/"),
    hf_cache_root: Path = typer.Option(..., help="HF cache root (holds hub/)"),
    animal: str = typer.Option("dog", help="Trait animal (the p_target token)."),
    student_base: str = typer.Option("pythia-160m-data-seed1"),
    numbers_base: str = typer.Option("pythia-160m"),
    teacher_tag: str = typer.Option("ep2_lr5e-4_r32a64"),
    tseed: int = typer.Option(1, help="Teacher fine-tuning seed."),
    dataset_tag: str = typer.Option("ex10-20_prefix_n100k"),
    tags: list[str] = typer.Option(None, help="Student cells (default: the §5.4 showcase)."),
    all_tags: bool = typer.Option(
        False, "--all-tags", help="Every trained cell in the lineage instead of --tags."
    ),
    tag_glob: str = typer.Option(
        "ep20_*", help="With --all-tags, restrict to cells matching this fnmatch pattern."
    ),
    ftseed: int = typer.Option(1, help="Student fine-tuning seed; seeds A_init."),
    prompt_set: str = typer.Option("test", help="Preference set scored with exact probs."),
    n_pile_docs: int = typer.Option(16, help="Held-out Pile docs for the KL readout."),
    pile_seed: int = typer.Option(0, help="Seed for the Pile doc permutation."),
    pile_max_tokens: int = typer.Option(512),
    device: str = typer.Option("cpu", help="Torch device for the evaluations."),
    init_device: str = typer.Option(
        "cpu", help="Device for the seeded A_init replay (PEFT draws lora_A on CPU)."
    ),
    out_name: str = typer.Option("frozen_a_ablation", help="Basename under analysis/."),
) -> None:
    """Splice ``ΔW' = s·B·A_init`` into each cell and re-evaluate trait and Pile KL."""
    import fnmatch

    import torch
    from peft import PeftModel
    from safetensors.numpy import load_file
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    from subliminal_learning.analysis.adapter_decomp import (
        base_model_factory,
        lora_init_weights,
    )

    lineage_dir = (
        output_root / "students" / animal / student_base / f"teacher-{numbers_base}"
        / f"{teacher_tag}_tseed{tseed}" / dataset_tag
    )
    if not lineage_dir.is_dir():
        raise typer.BadParameter(f"no such lineage: {lineage_dir}")

    if all_tags:
        cells = sorted(
            d.name for d in lineage_dir.iterdir()
            if (d / f"ftseed{ftseed}" / "checkpoints" / ADAPTER_WEIGHTS).exists()
            and fnmatch.fnmatch(d.name, tag_glob)
        )
    else:
        cells = list(tags) if tags else list(SHOWCASE_TAGS)
    if not cells:
        raise typer.BadParameter("no cells selected")
    logger.info("{} cell(s) from {}", len(cells), lineage_dir)

    first_cfg = json.loads(
        (lineage_dir / cells[0] / f"ftseed{ftseed}" / "checkpoints" / "adapter_config.json")
        .read_text()
    )
    base_repo = first_cfg["base_model_name_or_path"]

    tok = AutoTokenizer.from_pretrained(base_repo, cache_dir=str(hf_cache_root / "hub"))
    tok.pad_token = tok.pad_token or tok.eos_token
    target_id = target_token_id(animal, tok)
    prompts = tuple(json.loads(prompt_set_path(prompt_set).read_text()))

    pool = build_pile_pool(
        tok, n_pile_docs, max_tokens=pile_max_tokens, seed=pile_seed
    )
    logger.info("Pile pool token_sha256 {}", pool.token_sha256())

    def _load_base() -> GPTNeoXForCausalLM:
        return GPTNeoXForCausalLM.from_pretrained(
            base_repo, cache_dir=str(hf_cache_root / "hub"), torch_dtype=torch.float32
        ).to(device).eval()

    # One base pass: the KL reference, and the p_target floor the retention metric needs.
    base = _load_base()
    base_logp = _pile_logprobs(base, pool, device)
    base_pref = target_prob_and_rank_per_prompt(base, tok, device, prompts, target_id)
    p_target_base = float(np.mean([p for p, _ in base_pref.values()]))
    del base
    logger.info("base p_target({}) = {:.6f}", animal, p_target_base)

    init_cache: dict[tuple, dict[str, np.ndarray]] = {}
    results: list[dict] = []
    for tag in cells:
        run_dir = lineage_dir / tag / f"ftseed{ftseed}"
        ckpt = run_dir / "checkpoints"
        cfg = json.loads((ckpt / "adapter_config.json").read_text())
        r, alpha = int(cfg["r"]), int(cfg["lora_alpha"])

        init_key = (cfg["base_model_name_or_path"], ftseed, r,
                    tuple(sorted(cfg["target_modules"])))
        if init_key not in init_cache:
            init_cache[init_key] = lora_init_weights(
                base_model_factory(base_repo, hf_cache_root, init_device), ftseed, cfg
            )
        a_init = init_cache[init_key]

        tensors = load_file(str(ckpt / ADAPTER_WEIGHTS))
        row = {"student_tag": tag, "ftseed": ftseed, "alpha": alpha, "rank": r}
        row.update(_weight_space_metrics(tensors, a_init))

        model = PeftModel.from_pretrained(_load_base(), str(ckpt)).eval()

        orig_pref = target_prob_and_rank_per_prompt(model, tok, device, prompts, target_id)
        row["p_target_orig"] = float(np.mean([p for p, _ in orig_pref.values()]))
        orig_logp = _pile_logprobs(model, pool, device)

        # The splice. lora_A holds A_final; overwriting it with the reconstructed A_init
        # makes the live branch compute s·B·A_init === ΔW'. Disk is untouched.
        n_swapped = 0
        with torch.no_grad():
            for name, mod in model.named_modules():
                if "default" not in getattr(mod, "lora_A", {}):
                    continue
                key = name.replace("base_model.model.", "")
                w = mod.lora_A["default"].weight
                w.copy_(torch.as_tensor(a_init[key], dtype=w.dtype, device=w.device))
                n_swapped += 1
        row["n_modules_swapped"] = n_swapped

        abl_pref = target_prob_and_rank_per_prompt(model, tok, device, prompts, target_id)
        row["p_target_ablated"] = float(np.mean([p for p, _ in abl_pref.values()]))
        abl_logp = _pile_logprobs(model, pool, device)
        del model

        row["pile_kl_base_orig"] = _mean_kl(base_logp, orig_logp)
        row["pile_kl_base_ablated"] = _mean_kl(base_logp, abl_logp)
        row["pile_kl_orig_ablated"] = _mean_kl(orig_logp, abl_logp)

        # Harness check: the sweep's own stored eval, re-derived by this script's loader.
        stored = None
        pref_file = run_dir / "preference_eval.json"
        if pref_file.exists():
            sets = json.loads(pref_file.read_text()).get("sets", {})
            if prompt_set in sets:
                stored = sets[prompt_set].get("p_target")
        row["p_target_stored"] = stored
        if stored is not None and abs(stored - row["p_target_orig"]) > 1e-4:
            logger.warning(
                "{}: re-derived p_target {:.8f} disagrees with stored {:.8f} — the "
                "splice harness is not evaluating the model the sweep trained.",
                tag, row["p_target_orig"], stored,
            )

        excess = row["p_target_orig"] - p_target_base
        row["excess_retention"] = (
            (row["p_target_ablated"] - p_target_base) / excess if excess else float("nan")
        )
        results.append(row)
        logger.info(
            "{}  r{} a{}  p {:.4f} -> {:.4f} (retain {:+.2f})  ablated share {:.3f}  "
            "pileKL {:.2f} -> {:.2f}",
            tag, r, alpha, row["p_target_orig"], row["p_target_ablated"],
            row["excess_retention"], row["rel_fro_dist_median"],
            row["pile_kl_base_orig"], row["pile_kl_base_ablated"],
        )

    out = output_root / "analysis" / f"{out_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "lineage": str(lineage_dir),
        "ftseed": ftseed,
        "n_pile_docs": n_pile_docs,
        "pile_pool_sha": pool.token_sha256(),
        "prompts_n": len(prompts),
        "prompt_set": prompt_set,
        "target_animal": animal,
        "p_target_base": p_target_base,
        "device": device,
        "results": results,
    }, indent=1))
    print(f"wrote {len(results)} cells -> {out}")


if __name__ == "__main__":
    app()
