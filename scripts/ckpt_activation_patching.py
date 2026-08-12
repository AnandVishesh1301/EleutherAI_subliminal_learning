"""Causal layer-localization of what differs between two pythia-160m pretraining checkpoints.

For a donor and a recipient checkpoint (both original-160m, different ``step`` revisions,
one shared tokenizer), we splice the donor's raw block-``ℓ`` output into the recipient's
forward pass — one block at a time, every position — and measure how far that moves the
recipient's next-token behaviour. Doing it for every block ``ℓ`` in 0..11 and both
directions (A→B, B→A) localizes, causally, WHERE the two checkpoints diverge on number
prompts (and, as a downstream readout, on the ``p(" dog")`` preference channel).

Metrics per (direction, ℓ), all vs. the UNPATCHED recipient on the same batch:
  - ``kl_number``: mean KL of the last-position next-token distribution on number prompts.
  - ``d_digit_mass``: mean change in probability mass on space-digit-initial tokens.
  - ``d_pdog``: mean change in ``p(" dog")`` at the last position on the "test" preference set.

Correctness control (per direction): patch the recipient with ITS OWN captured block-ℓ
activations (a second, identical forward) — must be ~0 KL (asserted < 1e-4).

Basis: raw post-block states (``raw_blocks_v2`` convention) via ``resolve_blocks``; capture
and injection share the one module resolver so the bases match by construction. fp32, batches
of 32, left padding, ``pad_token = eos``.
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

from pipelines.evaluate_teacher_preference_pipeline import load_prompt_templates  # noqa: E402
from pipelines.steerability_screen_pipeline import (  # noqa: E402
    SteerabilityScreenConfig,
    _extract_prompts,
)
from subliminal_learning.config import PathsConfig  # noqa: E402
from subliminal_learning.evaluation.preference_prob_io import target_token_id  # noqa: E402
from subliminal_learning.evaluation.prompt_sets import prompt_set_path  # noqa: E402
from subliminal_learning.models.loading_io import load_model, load_tokenizer  # noqa: E402
from subliminal_learning.models.residual import resolve_blocks  # noqa: E402
from subliminal_learning.teachers.model_specs import original_160m_spec  # noqa: E402
from subliminal_learning.utils.io import atomic_write_json  # noqa: E402
from subliminal_learning.utils.logging_setup import configure_loguru  # noqa: E402

BATCH = 32
N_BLOCKS = 12

app = typer.Typer(add_completion=False)


def _digit_ids(tok, device) -> torch.Tensor:
    """Space-digit-initial token ids (" 0".." 9…"), same construction as the m2a script."""
    ids = [
        i
        for i in range(len(tok))
        if tok.decode([i]).startswith(" ") and tok.decode([i])[1:2].isdigit()
    ]
    return torch.tensor(ids, device=device)


@torch.no_grad()
def _capture_block(model, enc, layer: int) -> torch.Tensor:
    """Donor forward on one batch; return raw block-``layer`` output hidden states [B,T,H] on CPU."""
    blocks = resolve_blocks(model)
    grabbed: dict[str, torch.Tensor] = {}

    def hook(_m, _a, output):
        hidden = output[0] if isinstance(output, tuple) else output
        grabbed["h"] = hidden.detach().to("cpu")

    handle = blocks[layer].register_forward_hook(hook)
    try:
        model(**enc, use_cache=False)
    finally:
        handle.remove()
    return grabbed["h"]


@torch.no_grad()
def _forward_last_logp(model, enc) -> torch.Tensor:
    """Last-real-position log-softmax next-token distribution for one batch [B,V]."""
    lg = model(**enc, use_cache=False).logits
    idx = enc["attention_mask"].sum(1) - 1
    last = lg[torch.arange(lg.shape[0]), idx].float()
    return last.log_softmax(-1)


@torch.no_grad()
def _forward_last_logp_patched(model, enc, layer: int, donor_h: torch.Tensor) -> torch.Tensor:
    """Recipient forward with block-``layer`` output REPLACED by ``donor_h`` (all positions)."""
    blocks = resolve_blocks(model)
    dev = enc["input_ids"].device

    def hook(_m, _a, output):
        new_hidden = donor_h.to(dev)
        if isinstance(output, tuple):
            return (new_hidden, *output[1:])
        return new_hidden

    handle = blocks[layer].register_forward_hook(hook)
    try:
        return _forward_last_logp(model, enc)
    finally:
        handle.remove()


def _batched(prompts, tok, device):
    for b0 in range(0, len(prompts), BATCH):
        batch = prompts[b0 : b0 + BATCH]
        enc = tok(list(batch), return_tensors="pt", padding=True).to(device)
        yield enc


@torch.no_grad()
def _baseline_number(model, tok, prompts, device, digit_ids):
    """Per-batch unpatched last-position log-probs and mean digit mass on number prompts."""
    logps, dmass = [], []
    for enc in _batched(prompts, tok, device):
        lp = _forward_last_logp(model, enc)
        logps.append(lp)
        dmass.append(lp.exp()[:, digit_ids].sum(1))
    return logps, torch.cat(dmass).mean().item()


@torch.no_grad()
def _baseline_pdog(model, tok, prompts, device, dog_id):
    logps = []
    pdog = []
    for enc in _batched(prompts, tok, device):
        lp = _forward_last_logp(model, enc)
        logps.append(lp)
        pdog.append(lp.exp()[:, dog_id])
    return logps, torch.cat(pdog).mean().item()


@torch.no_grad()
def _patch_direction(
    donor, recipient, tok, device, layer, num_prompts, pref_prompts,
    digit_ids, dog_id, base_num_logp, base_pref_logp, self_control=False,
):
    """One (direction, layer): patch recipient at block ``layer`` with donor's captured acts.

    ``self_control=True`` captures from the recipient itself (correctness check → ~0 KL).
    Returns (mean_kl_number, mean_d_digit_mass, mean_d_pdog).
    """
    source = recipient if self_control else donor
    kls, dmass_deltas = [], []
    for bi, enc in enumerate(_batched(num_prompts, tok, device)):
        donor_h = _capture_block(source, enc, layer)
        lp = _forward_last_logp_patched(recipient, enc, layer, donor_h)
        base = base_num_logp[bi]
        kls.append((lp.exp() * (lp - base)).sum(-1))
        dmass_deltas.append(lp.exp()[:, digit_ids].sum(1) - base.exp()[:, digit_ids].sum(1))
    kl = torch.cat(kls).mean().item()
    d_dmass = torch.cat(dmass_deltas).mean().item()

    pdog_deltas = []
    for bi, enc in enumerate(_batched(pref_prompts, tok, device)):
        donor_h = _capture_block(source, enc, layer)
        lp = _forward_last_logp_patched(recipient, enc, layer, donor_h)
        base = base_pref_logp[bi]
        pdog_deltas.append(lp.exp()[:, dog_id] - base.exp()[:, dog_id])
    d_pdog = torch.cat(pdog_deltas).mean().item()
    return kl, d_dmass, d_pdog


def _number_prompts(paths, hub_org, n, device):
    cfg = SteerabilityScreenConfig(
        paths=paths, hub_org=hub_org, spec=original_160m_spec(), target_animal="dog",
        param_dir=Path("."), n_extract_prompts=n,
        number_config=Path("configs/experiment/number_generation.yaml"), device=device,
    )
    return list(_extract_prompts(cfg))


@app.command()
def main(
    step_a: int = typer.Option(70000, "--step-a"),
    step_b: int = typer.Option(110000, "--step-b"),
    n_number_prompts: int = typer.Option(256, "--n-number-prompts"),
    n_pref_prompts: int = typer.Option(50, "--n-pref-prompts"),
    device: str = typer.Option("cuda:0", "--device"),
    out: Path | None = typer.Option(None, "--out"),
    adapter_a: Path | None = typer.Option(None, "--adapter-a"),
    adapter_b: Path | None = typer.Option(None, "--adapter-b"),
):
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(Path("configs/paths/paths_local.yaml").read_text()),
        config=Config(cast=[Path]),
    )
    hub_org = yaml.safe_load(Path("configs/model/pythia.yaml").read_text())["hub_org"]
    if out is None:
        out = paths.output_root / "analysis" / f"ckpt_patch_a{step_a}_b{step_b}.json"

    spec_a = original_160m_spec(revision=f"step{step_a}")
    spec_b = original_160m_spec(revision=f"step{step_b}")
    tok = load_tokenizer(spec_a, paths.hf_cache_root, hub_org)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    dog_id = target_token_id("dog", tok)

    num_prompts = _number_prompts(paths, hub_org, n_number_prompts, device)
    pref_prompts = list(load_prompt_templates(prompt_set_path("test")))[:n_pref_prompts]
    logger.info(
        "prompts: {} number, {} preference; dog_id={}",
        len(num_prompts), len(pref_prompts), dog_id,
    )

    logger.info("Loading checkpoint A step{} and B step{}", step_a, step_b)
    model_a = load_model(spec_a, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    model_b = load_model(spec_b, paths.hf_cache_root, hub_org, device, dtype=torch.float32)
    # Optionally wrap either side with a LoRA teacher adapter. resolve_blocks unwraps the
    # PeftModel to the same GPT-NeoX block list, so capture/injection bases still match.
    if adapter_a is not None:
        logger.info("Wrapping A with adapter {}", adapter_a)
        model_a = PeftModel.from_pretrained(model_a, str(adapter_a)).eval()
    if adapter_b is not None:
        logger.info("Wrapping B with adapter {}", adapter_b)
        model_b = PeftModel.from_pretrained(model_b, str(adapter_b)).eval()
    models = {"A": model_a, "B": model_b}

    digit_ids = _digit_ids(tok, device)

    baselines = {}
    for name, m in models.items():
        num_lp, dmass = _baseline_number(m, tok, num_prompts, device, digit_ids)
        pref_lp, pdog = _baseline_pdog(m, tok, pref_prompts, device, dog_id)
        baselines[name] = {
            "num_logp": num_lp, "pref_logp": pref_lp,
            "digit_mass": dmass, "p_dog": pdog,
        }
        logger.info("baseline {}: digit_mass={:.4f} p(dog)={:.5f}", name, dmass, pdog)

    directions = [("A", "B"), ("B", "A")]
    results: dict[str, dict] = {}
    for donor_name, recip_name in directions:
        donor, recipient = models[donor_name], models[recip_name]
        base = baselines[recip_name]
        dir_key = f"{donor_name}->{recip_name}"

        # Self-patch control: capture the recipient's own acts and reinject at layer 0.
        ctrl_kl, _, _ = _patch_direction(
            donor, recipient, tok, device, 0, num_prompts, pref_prompts, digit_ids,
            dog_id, base["num_logp"], base["pref_logp"], self_control=True,
        )
        assert ctrl_kl < 1e-4, f"self-patch control not ~0 for {dir_key}: KL={ctrl_kl}"
        logger.info("[{}] self-patch control KL={:.2e} (< 1e-4 ✓)", dir_key, ctrl_kl)

        per_layer = {}
        logger.info("[{}] layer |  kl_number  | d_digit_mass |   d_pdog", dir_key)
        for layer in range(N_BLOCKS):
            kl, d_dmass, d_pdog = _patch_direction(
                donor, recipient, tok, device, layer, num_prompts, pref_prompts,
                digit_ids, dog_id, base["num_logp"], base["pref_logp"],
            )
            per_layer[f"L{layer}"] = {
                "kl_number": kl, "d_digit_mass": d_dmass, "d_pdog": d_pdog,
            }
            logger.info(
                "[{}]  L{:<2} | {:10.5f}  | {:11.5f}  | {:9.5f}",
                dir_key, layer, kl, d_dmass, d_pdog,
            )
        results[dir_key] = {
            "self_patch_control_kl": ctrl_kl,
            "per_layer": per_layer,
        }

    payload = {
        "step_a": step_a, "step_b": step_b,
        "adapter_a": str(adapter_a) if adapter_a is not None else None,
        "adapter_b": str(adapter_b) if adapter_b is not None else None,
        "n_number_prompts": len(num_prompts), "n_pref_prompts": len(pref_prompts),
        "batch": BATCH, "n_blocks": N_BLOCKS, "dog_token_id": dog_id,
        "basis": "raw post-block states (raw_blocks_v2), full-sequence replace, all positions",
        "baselines": {
            name: {"digit_mass": b["digit_mass"], "p_dog": b["p_dog"]}
            for name, b in baselines.items()
        },
        "directions": results,
    }
    atomic_write_json(out, payload)
    logger.info("Wrote {}", out)


if __name__ == "__main__":
    app()
