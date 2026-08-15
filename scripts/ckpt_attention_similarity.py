"""Per-head attention-pattern similarity between two pythia-160m pretraining checkpoints.

For two original-160m checkpoints (different ``step`` revisions, shared tokenizer), we run
both with ``output_attentions=True`` on the same ICL number prompts and, for every one of the
12 layers × 12 heads, measure how similar their attention *maps* are.

Metrics per (layer, head):
  - ``pattern_corr``: mean-over-prompts Pearson correlation between the two models' attention
    maps. Each prompt's LxL map is restricted to the real (non-pad) tokens and flattened before
    correlating; correlations are averaged across prompts.
  - ``entropy_a`` / ``entropy_b``: mean attention entropy of the head (mean over prompts and real
    query positions), one per model — a "concentration" readout.

Per-prompt forwards (left padding, pad=eos), fp32.
"""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from dotenv import load_dotenv

load_dotenv(".env")

import yaml  # noqa: E402
from dacite import Config, from_dict  # noqa: E402
from loguru import logger  # noqa: E402
from peft import PeftModel  # noqa: E402

from pipelines.steerability_screen_pipeline import (  # noqa: E402
    SteerabilityScreenConfig,
    _extract_prompts,
)
from subliminal_learning.config import PathsConfig  # noqa: E402
from subliminal_learning.models.loading_io import load_model, load_tokenizer  # noqa: E402
from subliminal_learning.teachers.model_specs import original_160m_spec  # noqa: E402
from subliminal_learning.utils.io import atomic_write_json  # noqa: E402
from subliminal_learning.utils.logging_setup import configure_loguru  # noqa: E402

N_LAYERS = 12
N_HEADS = 12

app = typer.Typer(add_completion=False)


def _number_prompts(paths, hub_org, n, device):
    cfg = SteerabilityScreenConfig(
        paths=paths, hub_org=hub_org, spec=original_160m_spec(), target_animal="dog",
        param_dir=Path("."), n_extract_prompts=n,
        number_config=Path("configs/experiment/number_generation.yaml"), device=device,
    )
    return list(_extract_prompts(cfg))


@torch.no_grad()
def _attentions(model, enc) -> tuple[torch.Tensor, ...]:
    """Tuple of per-layer attention tensors [1, H, T, T] for one prompt (batch of 1)."""
    return model(**enc, use_cache=False, output_attentions=True).attentions


def _pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation of two 1-D tensors; 0.0 if either is constant."""
    xd = x - x.mean()
    yd = y - y.mean()
    denom = xd.norm() * yd.norm()
    if denom < 1e-12:
        return 0.0
    return float((xd @ yd) / denom)


def _entropy(attn: torch.Tensor) -> torch.Tensor:
    """Per-query attention entropy [H, L] for a real-token LxL map [H, L, L]."""
    p = attn.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


@app.command()
def main(
    step_a: int = typer.Option(70000, "--step-a"),
    step_b: int = typer.Option(110000, "--step-b"),
    n_prompts: int = typer.Option(128, "--n-prompts"),
    adapter_b: Path | None = typer.Option(
        None, "--adapter-b", help="optional LoRA teacher adapter to wrap model B (PeftModel)."
    ),
    device: str = typer.Option("cuda:0", "--device"),
    out: Path | None = typer.Option(None, "--out"),
):
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(Path("configs/paths/paths_local.yaml").read_text()),
        config=Config(cast=[Path]),
    )
    hub_org = yaml.safe_load(Path("configs/model/pythia.yaml").read_text())["hub_org"]
    if out is None:
        suffix = "_ftb" if adapter_b is not None else ""
        out = paths.output_root / "analysis" / f"ckpt_attn_sim_a{step_a}_b{step_b}{suffix}.json"

    spec_a = original_160m_spec(revision=f"step{step_a}")
    spec_b = original_160m_spec(revision=f"step{step_b}")
    tok = load_tokenizer(spec_a, paths.hf_cache_root, hub_org)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    prompts = _number_prompts(paths, hub_org, n_prompts, device)
    logger.info("Loaded {} number prompts", len(prompts))

    logger.info("Loading checkpoint A step{} and B step{}", step_a, step_b)
    model_a = load_model(spec_a, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model_b = load_model(spec_b, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    # output_attentions requires the eager attention path (sdpa/flash return None). Set it on the
    # bare GPT-NeoX before any Peft wrap so the config propagates into the attention modules.
    for m in (model_a, model_b):
        m.config._attn_implementation = "eager"
        m.set_attn_implementation("eager")
    if adapter_b is not None:
        logger.info("Wrapping B with adapter {}", adapter_b)
        model_b = PeftModel.from_pretrained(model_b, str(adapter_b)).eval()
    model_a.eval()
    model_b.eval()

    # Accumulators [L, H].
    corr_sum = torch.zeros(N_LAYERS, N_HEADS)
    ent_a_sum = torch.zeros(N_LAYERS, N_HEADS)
    ent_b_sum = torch.zeros(N_LAYERS, N_HEADS)
    n_seen = 0

    for pi, prompt in enumerate(prompts):
        enc = tok([prompt], return_tensors="pt", padding=True).to(device)
        real = enc["attention_mask"][0].bool()
        if real.sum() < 2:
            continue
        attn_a = _attentions(model_a, enc)
        attn_b = _attentions(model_b, enc)
        for layer in range(N_LAYERS):
            # [H, L, L] real-token sub-map.
            a = attn_a[layer][0][:, real][:, :, real].float()
            b = attn_b[layer][0][:, real][:, :, real].float()
            ent_a_sum[layer] += _entropy(a).mean(-1).cpu()
            ent_b_sum[layer] += _entropy(b).mean(-1).cpu()
            for head in range(N_HEADS):
                corr_sum[layer, head] += _pearson(a[head].flatten(), b[head].flatten())
        n_seen += 1
        if (pi + 1) % 32 == 0:
            logger.info("processed {}/{} prompts", pi + 1, len(prompts))

    corr = (corr_sum / n_seen)
    ent_a = (ent_a_sum / n_seen)
    ent_b = (ent_b_sum / n_seen)

    records = []
    for layer in range(N_LAYERS):
        for head in range(N_HEADS):
            records.append({
                "layer": layer, "head": head,
                "pattern_corr": float(corr[layer, head]),
                "entropy_a": float(ent_a[layer, head]),
                "entropy_b": float(ent_b[layer, head]),
            })

    # 10 lowest-correlation heads.
    lowest = sorted(records, key=lambda r: r["pattern_corr"])[:10]
    logger.info("--- 10 LOWEST pattern_corr heads ---")
    logger.info("  L  | H  | pattern_corr | ent_a | ent_b")
    for r in lowest:
        logger.info(
            "  L{:<2}| {:<2} | {:12.4f} | {:5.3f} | {:5.3f}",
            r["layer"], r["head"], r["pattern_corr"], r["entropy_a"], r["entropy_b"],
        )

    # 10 largest entropy changes.
    by_dent = sorted(records, key=lambda r: abs(r["entropy_b"] - r["entropy_a"]), reverse=True)[:10]
    logger.info("--- 10 LARGEST entropy changes (b - a) ---")
    logger.info("  L  | H  | ent_a | ent_b | d_entropy | pattern_corr")
    for r in by_dent:
        logger.info(
            "  L{:<2}| {:<2} | {:5.3f} | {:5.3f} | {:+9.4f} | {:.4f}",
            r["layer"], r["head"], r["entropy_a"], r["entropy_b"],
            r["entropy_b"] - r["entropy_a"], r["pattern_corr"],
        )

    payload = {
        "step_a": step_a, "step_b": step_b,
        "adapter_b": str(adapter_b) if adapter_b is not None else None,
        "n_prompts": n_seen, "n_layers": N_LAYERS, "n_heads": N_HEADS,
        "note": "pattern_corr: mean-over-prompts Pearson of real-token LxL attention maps; "
                "entropy_a/b: mean per-head attention entropy (real query positions).",
        "heads": records,
    }
    atomic_write_json(out, payload)
    logger.info("Wrote {}", out)


if __name__ == "__main__":
    app()
