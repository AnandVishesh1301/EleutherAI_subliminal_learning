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
from subliminal_learning.cka.pool_io import build_pile_pool  # noqa: E402
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
def _capture_all_blocks(model, enc) -> list[torch.Tensor]:
    """One forward; return each block's raw output hidden states [B,T,H] on CPU (all blocks)."""
    blocks = resolve_blocks(model)
    grabbed: list[torch.Tensor | None] = [None] * len(blocks)

    def _mk(i):
        def hook(_m, _a, output):
            hidden = output[0] if isinstance(output, tuple) else output
            grabbed[i] = hidden.detach().to("cpu")
        return hook

    handles = [b.register_forward_hook(_mk(i)) for i, b in enumerate(blocks)]
    try:
        model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    assert all(g is not None for g in grabbed), "a block hook failed to capture"
    return grabbed  # type: ignore[return-value]


def _norm_match(donor_h: torch.Tensor, recip_h: torch.Tensor) -> torch.Tensor:
    """Rescale ``donor_h`` per (example, position) to the recipient's own L2 norm at that block.

    Preserves the donor's directions exactly, adopts the recipient's per-token scale:
    ``donor_h <- donor_h * (||recip_h|| / ||donor_h||)`` (self-patch → factor 1 exactly).
    """
    dnorm = donor_h.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    rnorm = recip_h.norm(dim=-1, keepdim=True)
    return donor_h * (rnorm / dnorm)


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
    """Yield left-padded encodings. ``prompts`` are text strings (numbers/preference) or
    pre-tokenized id lists (Pile); the latter are padded via ``tok.pad`` (padding_side-aware)."""
    for b0 in range(0, len(prompts), BATCH):
        batch = prompts[b0 : b0 + BATCH]
        if isinstance(batch[0], str):
            enc = tok(list(batch), return_tensors="pt", padding=True).to(device)
        else:
            enc = tok.pad(
                {"input_ids": [list(x) for x in batch]}, return_tensors="pt"
            ).to(device)
        yield enc


@torch.no_grad()
def _baseline_number(model, tok, prompts, device, digit_ids, capture_blocks=False):
    """Per-batch unpatched last-position log-probs and mean digit mass on number prompts.

    When ``capture_blocks`` also returns the recipient's own per-block states (list per batch
    of list[n_blocks] CPU tensors) for norm-matching; otherwise ``None``.
    """
    logps, dmass, blocks = [], [], []
    for enc in _batched(prompts, tok, device):
        if capture_blocks:
            blocks.append(_capture_all_blocks(model, enc))
        lp = _forward_last_logp(model, enc)
        logps.append(lp)
        dmass.append(lp.exp()[:, digit_ids].sum(1))
    return logps, torch.cat(dmass).mean().item(), (blocks if capture_blocks else None)


@torch.no_grad()
def _baseline_pdog(model, tok, prompts, device, dog_id, capture_blocks=False):
    logps, pdog, blocks = [], [], []
    for enc in _batched(prompts, tok, device):
        if capture_blocks:
            blocks.append(_capture_all_blocks(model, enc))
        lp = _forward_last_logp(model, enc)
        logps.append(lp)
        pdog.append(lp.exp()[:, dog_id])
    return logps, torch.cat(pdog).mean().item(), (blocks if capture_blocks else None)


@torch.no_grad()
def _patch_direction(
    donor, recipient, tok, device, layer, num_prompts, pref_prompts,
    digit_ids, dog_id, base_num_logp, base_pref_logp, self_control=False,
    norm_match=False, recip_num_blocks=None, recip_pref_blocks=None,
    source_num_blocks=None, source_pref_blocks=None,
):
    """One (direction, layer): patch recipient at block ``layer`` with donor's captured acts.

    ``source_*_blocks`` are the donor's per-batch all-block captures (`_capture_all_blocks`,
    one forward per batch for every layer); pass them from the caller so the per-layer loop
    does not re-forward the donor once per (layer, batch). When absent they are captured
    here (one donor forward per batch, all blocks).
    ``self_control=True`` captures from the recipient itself (correctness check → ~0 KL).
    ``norm_match=True`` rescales the donor states per (example, position) to the recipient's
    own norm at that block (``recip_*_blocks``), preserving direction, adopting scale.
    Returns (mean_kl_number, mean_d_digit_mass, mean_d_pdog, donor_mean_l2, recip_mean_l2);
    the two norms are mean per-position L2 over real (unmasked) number-prompt positions.
    """
    source = recipient if self_control else donor
    if source_num_blocks is None:
        source_num_blocks = [_capture_all_blocks(source, enc)
                             for enc in _batched(num_prompts, tok, device)]
    if source_pref_blocks is None:
        source_pref_blocks = [_capture_all_blocks(source, enc)
                              for enc in _batched(pref_prompts, tok, device)]
    kls, dmass_deltas = [], []
    donor_norm_sum = recip_norm_sum = pos_count = 0.0
    for bi, enc in enumerate(_batched(num_prompts, tok, device)):
        donor_h = source_num_blocks[bi][layer]
        recip_h = recip_num_blocks[bi][layer] if recip_num_blocks is not None else None
        mask = enc["attention_mask"].bool().cpu()
        donor_norm_sum += donor_h.norm(dim=-1)[mask].sum().item()
        if recip_h is not None:
            recip_norm_sum += recip_h.norm(dim=-1)[mask].sum().item()
        pos_count += mask.sum().item()
        if norm_match:
            donor_h = _norm_match(donor_h, recip_h)
        lp = _forward_last_logp_patched(recipient, enc, layer, donor_h)
        base = base_num_logp[bi]
        kls.append((lp.exp() * (lp - base)).sum(-1))
        dmass_deltas.append(lp.exp()[:, digit_ids].sum(1) - base.exp()[:, digit_ids].sum(1))
    kl = torch.cat(kls).mean().item()
    d_dmass = torch.cat(dmass_deltas).mean().item()

    pdog_deltas = []
    for bi, enc in enumerate(_batched(pref_prompts, tok, device)):
        donor_h = source_pref_blocks[bi][layer]
        if norm_match:
            donor_h = _norm_match(donor_h, recip_pref_blocks[bi][layer])
        lp = _forward_last_logp_patched(recipient, enc, layer, donor_h)
        base = base_pref_logp[bi]
        pdog_deltas.append(lp.exp()[:, dog_id] - base.exp()[:, dog_id])
    d_pdog = torch.cat(pdog_deltas).mean().item()
    donor_l2 = donor_norm_sum / pos_count
    recip_l2 = (recip_norm_sum / pos_count) if recip_num_blocks is not None else None
    return kl, d_dmass, d_pdog, donor_l2, recip_l2


def _number_prompts(paths, hub_org, n, device):
    cfg = SteerabilityScreenConfig(
        paths=paths, hub_org=hub_org, spec=original_160m_spec(), target_animal="dog",
        param_dir=Path("."), n_extract_prompts=n,
        number_config=Path("configs/experiment/number_generation.yaml"), device=device,
    )
    return list(_extract_prompts(cfg))


def _pile_prompts(tok, n):
    """``n`` held-out Pile sequences (monology/pile-uncopyrighted validation, max_tokens 512,
    min_tokens 32), returned pre-tokenized -- same source/truncation as the CKA pool."""
    pool = build_pile_pool(
        tok, n, dataset_name="monology/pile-uncopyrighted", split="validation",
        max_tokens=512, min_tokens=32,
    )
    return list(pool.input_ids)


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
    norm_match: bool = typer.Option(False, "--norm-match"),
    condition: str = typer.Option("numbers", "--condition", help="numbers | pile"),
):
    assert condition in ("numbers", "pile"), f"--condition must be numbers|pile; got {condition!r}"
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

    if condition == "pile":
        num_prompts = _pile_prompts(tok, n_number_prompts)
    else:
        num_prompts = _number_prompts(paths, hub_org, n_number_prompts, device)
    pref_prompts = list(load_prompt_templates(prompt_set_path("test")))[:n_pref_prompts]
    logger.info(
        "prompts: {} {} seqs, {} preference; dog_id={}",
        len(num_prompts), condition, len(pref_prompts), dog_id,
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
        num_lp, dmass, num_blocks = _baseline_number(
            m, tok, num_prompts, device, digit_ids, capture_blocks=norm_match)
        pref_lp, pdog, pref_blocks = _baseline_pdog(
            m, tok, pref_prompts, device, dog_id, capture_blocks=norm_match)
        baselines[name] = {
            "num_logp": num_lp, "pref_logp": pref_lp,
            "num_blocks": num_blocks, "pref_blocks": pref_blocks,
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
        ctrl_kl, *_ = _patch_direction(
            donor, recipient, tok, device, 0, num_prompts, pref_prompts, digit_ids,
            dog_id, base["num_logp"], base["pref_logp"], self_control=True,
            norm_match=norm_match, recip_num_blocks=base["num_blocks"],
            recip_pref_blocks=base["pref_blocks"],
        )
        assert ctrl_kl < 1e-4, f"self-patch control not ~0 for {dir_key}: KL={ctrl_kl}"
        logger.info("[{}] self-patch control KL={:.2e} (< 1e-4 ✓)", dir_key, ctrl_kl)

        # Donor captured ONCE per batch for all blocks (was once per (layer, batch): N_BLOCKS
        # redundant donor forwards per batch).
        donor_num_blocks = [_capture_all_blocks(donor, enc)
                            for enc in _batched(num_prompts, tok, device)]
        donor_pref_blocks = [_capture_all_blocks(donor, enc)
                             for enc in _batched(pref_prompts, tok, device)]
        per_layer = {}
        logger.info(
            "[{}] layer |  kl_number  | d_digit_mass |   d_pdog | donorL2 recipL2", dir_key)
        for layer in range(N_BLOCKS):
            kl, d_dmass, d_pdog, donor_l2, recip_l2 = _patch_direction(
                donor, recipient, tok, device, layer, num_prompts, pref_prompts,
                digit_ids, dog_id, base["num_logp"], base["pref_logp"],
                norm_match=norm_match, recip_num_blocks=base["num_blocks"],
                recip_pref_blocks=base["pref_blocks"],
                source_num_blocks=donor_num_blocks, source_pref_blocks=donor_pref_blocks,
            )
            per_layer[f"L{layer}"] = {
                "kl_number": kl, "d_digit_mass": d_dmass, "d_pdog": d_pdog,
                "donor_mean_l2": donor_l2, "recip_mean_l2": recip_l2,
            }
            logger.info(
                "[{}]  L{:<2} | {:10.5f}  | {:11.5f}  | {:9.5f} | {:7.2f} {}",
                dir_key, layer, kl, d_dmass, d_pdog, donor_l2,
                f"{recip_l2:7.2f}" if recip_l2 is not None else "   n/a",
            )
        results[dir_key] = {
            "self_patch_control_kl": ctrl_kl,
            "per_layer": per_layer,
        }

    payload = {
        "step_a": step_a, "step_b": step_b,
        "adapter_a": str(adapter_a) if adapter_a is not None else None,
        "adapter_b": str(adapter_b) if adapter_b is not None else None,
        "norm_match": norm_match,
        "condition": condition,
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
