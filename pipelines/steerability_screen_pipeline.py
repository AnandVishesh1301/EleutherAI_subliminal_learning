"""Cheap teacher-steerability screen: does the base model steer like the teacher?

For one trained ``{param_tag}`` config (all its ``ftseed*`` adapters), this:

1. Extracts the **teacher-minus-base** direction ``v`` from a *trait-free* context --
   the last-token mean activation over number-generation prompts (Blank et al.'s
   ``v_teacher`` pooling, but a cross-weight teacher/base diff; see
   ``BLANK_ARCHITECTURE.md`` and `subliminal_learning.steering.vectors`). Extracting on
   number prompts -- the same trait-free channel subliminal transfer flows through --
   makes the resulting steerability the theory-aligned predictor of transfer.
2. **Steers the clean base** with ``v`` (``add`` at swept layers x alphas) and reads the
   exact next-token ``p(target)`` on held-out **validation** prompts (the "does it install
   the trait?" signal) and on **dislike** prompts (the specificity gate: a genuine
   disposition must *not* raise ``p(target)`` after "My least favorite animal is the").
3. Scores the **teacher** itself as the ceiling, and reports, per seed and seed-averaged:
   - ``delta`` = clean-peak steered ``p(target)`` minus base ``p(target)`` (validation),
   - ``recovery`` = ``delta`` / (teacher minus base) -- fraction of the teacher's own lift
     the cheap steer reproduces.

No student is trained: the screen needs the (base, teacher) pair only. Whether ``delta``
*predicts* real subliminal transfer is a separate calibration (correlate vs. trained-student
``p(target)``) run once ground-truth students exist.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel

from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.steering.hooks import steering_hooks
from subliminal_learning.steering.vectors import diff_vector, mean_last_activations
from subliminal_learning.templates.dataset_io import load_prompt_templates
from subliminal_learning.templates.run_layout import model_dir_name

# A steering config in the sweep: a human label and the block indices to hook (None = all).
LayerChoice = tuple[str, list[int] | None]

# Default layer sweep for a 12-block Pythia-160m: quarter/half/three-quarter depth single
# layers, a mid-band triple, and all-layers (each block hooked with its own direction).
DEFAULT_LAYER_SWEEP: tuple[LayerChoice, ...] = (
    ("L3", [3]),
    ("L6", [6]),
    ("L9", [9]),
    ("L3-6-9", [3, 6, 9]),
    ("all", None),
)
DEFAULT_ALPHAS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)


@dataclass(frozen=True)
class SteerabilityScreenConfig:
    """Config for the steerability screen of one trained ``{param_tag}`` config.

    Attributes:
        paths: Filesystem roots (output_root, hf_cache_root).
        hub_org: HF org name.
        spec: Base model the adapters were trained from.
        target_animal: Animal whose ``p(target)`` is measured (must be single-token).
        param_dir: The ``{param_tag}`` dir whose ``ftseed*`` adapters are screened.
        extract_context: Where the teacher-minus-base direction is read. ``"numbers"``
            (Blank-faithful: the trait-free channel transfer flows through) or the name of
            a preference prompt set (e.g. ``"favorite"``) whose "...the" position directly
            evokes the animal -- the context in which a *weight-based* LoRA teacher's trait
            actually fires. The two can differ sharply for LoRA teachers (unlike Blank's
            always-on system-prompt teacher), so it is worth screening both.
        n_extract_prompts: Number of number prompts to average ``v`` over
            (``extract_context="numbers"`` only; preference sets use all their prompts).
        extract_seed: PCG64 seed for the (deterministic) number-prompt set.
        pos_set: Held-out prompt set whose ``p(target)`` lift defines steerability.
        gate_set: Prompt set for the specificity gate (dislike).
        ref_set: Extra prompt set scored for reference (the trained-on favorite set).
        layer_sweep: (label, block-indices) choices to hook.
        alphas: ``add`` scales to sweep (``norm="raw"`` -> alpha=1 replays the mean shift).
        positions: Where along the sequence to apply the shift.
        gate_margin: Max allowed rise in gate-set ``p(target)`` for a config to count as a
            "clean" (non-token-puppeting) steer when picking the peak.
        device: Torch device.
        force: Recompute even if ``steerability_screen.json`` already exists.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    target_animal: str
    param_dir: Path
    extract_context: str = "numbers"
    n_extract_prompts: int = 1024
    extract_seed: int = 42
    pos_set: str = "validation"
    gate_set: str = "dislike"
    ref_set: str = "favorite"
    layer_sweep: tuple[LayerChoice, ...] = DEFAULT_LAYER_SWEEP
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    positions: str = "broadcast"
    gate_margin: float = 0.05
    device: str = "cuda:0"
    force: bool = False


def _mean_p_rank(pr: dict[str, tuple[float, int]]) -> tuple[float, float, float]:
    """Reduce a prompt -> (p, rank) map to (mean_p, mean_rank, rank1_frac)."""
    ps = [p for p, _ in pr.values()]
    ranks = [r for _, r in pr.values()]
    n = max(len(ps), 1)
    return (
        sum(ps) / n,
        sum(ranks) / n,
        sum(1 for r in ranks if r == 1) / n,
    )


def _discover_ftseed_dirs(param_dir: Path) -> list[tuple[int, Path]]:
    """Return sorted (ft_seed, checkpoints_dir) for each ``ftseed{N}`` under `param_dir`."""
    found: list[tuple[int, Path]] = []
    for d in sorted(param_dir.glob("ftseed*")):
        ckpt = d / "checkpoints"
        if not ckpt.is_dir():
            logger.warning("Skipping {} (no checkpoints/ dir)", d)
            continue
        try:
            seed = int(d.name.removeprefix("ftseed"))
        except ValueError:
            logger.warning("Skipping {} (unparseable seed)", d)
            continue
        found.append((seed, ckpt))
    return found


def _extract_prompts(cfg: SteerabilityScreenConfig) -> tuple[str, ...]:
    """The context the teacher-minus-base direction is read from.

    ``"numbers"`` builds the deterministic trait-free number-prompt set; any other value is
    treated as a preference prompt-set name and loaded verbatim (all its prompts).
    """
    if cfg.extract_context == "numbers":
        templates = load_intro_templates()
        prompts = build_number_prompts(
            templates, n_prompts=cfg.n_extract_prompts, seed=cfg.extract_seed
        )
        return tuple(p.text for p in prompts)
    return load_prompt_templates(prompt_set_path(cfg.extract_context))


def _score_sets(
    model,
    tokenizer,
    device: str,
    prompt_by_set: dict[str, tuple[str, ...]],
    target_id: int,
) -> dict[str, dict]:
    """Exact ``p(target)`` summary per set for one (optionally hooked) model."""
    out: dict[str, dict] = {}
    for name, prompts in prompt_by_set.items():
        pr = target_prob_and_rank_per_prompt(model, tokenizer, device, prompts, target_id)
        mean_p, mean_rank, rank1 = _mean_p_rank(pr)
        out[name] = {"p_target": mean_p, "rank_mean": mean_rank, "rank1_frac": rank1}
    return out


def _clean_peak(
    sweep_rows: list[dict], base_gate_p: float, gate_margin: float
) -> dict:
    """Pick the sweep config with the highest pos ``p(target)`` that passes the gate.

    "Passes the gate" = its gate-set ``p(target)`` rises no more than ``gate_margin`` above
    the base's -- so a config that just cranks the target token everywhere (including on
    "least favorite" prompts) is rejected. Falls back to the unconstrained best if no
    config passes (flagged via ``gated=False``), so the row is never empty.
    """
    clean = [r for r in sweep_rows if r["gate_p"] - base_gate_p <= gate_margin]
    pool, gated = (clean, True) if clean else (sweep_rows, False)
    best = max(pool, key=lambda r: r["pos_p"])
    return {**best, "gated": gated}


def run(cfg: SteerabilityScreenConfig) -> dict:
    """Screen every ftseed under ``cfg.param_dir``; write + return the aggregated result."""
    out_path = cfg.param_dir / "steerability_screen.json"
    if out_path.exists() and not cfg.force:
        logger.info("Steerability screen cached at {} (use force to recompute)", out_path)
        return json.loads(out_path.read_text())

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    tokenizer.padding_side = "left"  # required by mean_last_activations
    target_id = target_token_id(cfg.target_animal, tokenizer)

    prompt_by_set = {
        cfg.pos_set: load_prompt_templates(prompt_set_path(cfg.pos_set)),
        cfg.gate_set: load_prompt_templates(prompt_set_path(cfg.gate_set)),
        cfg.ref_set: load_prompt_templates(prompt_set_path(cfg.ref_set)),
    }
    extract_prompts = _extract_prompts(cfg)

    ftseeds = _discover_ftseed_dirs(cfg.param_dir)
    if not ftseeds:
        raise FileNotFoundError(f"No ftseed*/checkpoints under {cfg.param_dir}")
    base_model_name = model_dir_name(cfg.spec, cfg.hub_org)
    logger.info(
        "Steerability screen: {} ({}/{}) over {} ftseed(s); extract on {} {} prompts",
        cfg.target_animal, base_model_name, cfg.param_dir.name, len(ftseeds),
        len(extract_prompts), cfg.extract_context,
    )

    # Clean base, held for the whole run: its activations (base_act) and its p(target)
    # (base control) are seed-independent, and every steered forward hooks *this* model.
    base = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    base_act = mean_last_activations(base, tokenizer, extract_prompts, cfg.device)
    base_scores = _score_sets(base, tokenizer, cfg.device, prompt_by_set, target_id)
    base_pos_p = base_scores[cfg.pos_set]["p_target"]
    base_gate_p = base_scores[cfg.gate_set]["p_target"]

    per_seed: list[dict] = []
    for seed, ckpt in ftseeds:
        # Fresh base -> teacher, so `base` above stays pristine for steering.
        teacher = PeftModel.from_pretrained(
            load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device),
            str(ckpt),
        )
        teacher.eval()
        teacher_act = mean_last_activations(teacher, tokenizer, extract_prompts, cfg.device)
        teacher_scores = _score_sets(teacher, tokenizer, cfg.device, prompt_by_set, target_id)
        del teacher
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        v = diff_vector(teacher_act, base_act)
        v_raw = v["raw"][1:]  # strip embedding slot; block-indexed 0..n-1

        sweep_rows: list[dict] = []
        for label, layers in cfg.layer_sweep:
            for alpha in cfg.alphas:
                with steering_hooks(
                    base, v_raw, alpha, "add", layers=layers, positions=cfg.positions
                ):
                    pos = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.pos_set], target_id
                    )
                    gate = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.gate_set], target_id
                    )
                pos_p = _mean_p_rank(pos)[0]
                gate_p = _mean_p_rank(gate)[0]
                sweep_rows.append(
                    {
                        "layers": label,
                        "alpha": alpha,
                        "pos_p": pos_p,
                        "gate_p": gate_p,
                        # Specificity: how much MORE the steer raises the favorite side than
                        # the "least favorite" side. ~0 => token-puppeting; large => a real
                        # disposition (fires on favorite, not on dislike).
                        "margin": pos_p - gate_p,
                    }
                )

        # Three readings of the sweep, kept side by side:
        # - clean_peak:    strongest pos that keeps the gate near base (disposition signal).
        # - best_ungated:  strongest pos ignoring the gate (the token-puppet ceiling).
        # - best_margin:   most specific config (max pos - gate).
        peak = _clean_peak(sweep_rows, base_gate_p, cfg.gate_margin)
        best_ungated = max(sweep_rows, key=lambda r: r["pos_p"])
        best_margin = max(sweep_rows, key=lambda r: r["margin"])
        teacher_pos_p = teacher_scores[cfg.pos_set]["p_target"]
        delta = peak["pos_p"] - base_pos_p
        headroom = teacher_pos_p - base_pos_p
        recovery = delta / headroom if abs(headroom) > 1e-9 else float("nan")
        per_seed.append(
            {
                "ft_seed": seed,
                "v_norm_per_layer": [round(x, 5) for x in v["norm"].tolist()],
                "v_argmax_layer": int(v["norm"].argmax()),
                "base": base_scores,
                "teacher": teacher_scores,
                "clean_peak": peak,
                "best_ungated": best_ungated,
                "best_margin": best_margin,
                "delta_pos_p": delta,
                "recovery_frac": recovery,
                "sweep": sweep_rows,
            }
        )
        logger.info(
            "  ftseed{} | base p={:.3f} teacher p={:.3f} | CLEAN {} a={} p={:.3f} "
            "(gated={}) delta={:.3f} rec={:.2f} | ungated best p={:.3f}@gate{:.3f} | "
            "max-margin={:.3f}",
            seed, base_pos_p, teacher_pos_p, peak["layers"], peak["alpha"],
            peak["pos_p"], peak["gated"], delta, recovery,
            best_ungated["pos_p"], best_ungated["gate_p"], best_margin["margin"],
        )

    del base
    if cfg.device.startswith("cuda"):
        torch.cuda.empty_cache()

    n = max(len(per_seed), 1)
    summary = {
        "target_animal": cfg.target_animal,
        "base_model": base_model_name,
        "param_tag": cfg.param_dir.name,
        "extraction": {
            "context": cfg.extract_context,
            "n_prompts": len(extract_prompts),
            "seed": cfg.extract_seed,
            "pooling": "last",
        },
        "sweep_grid": {
            "layers": [l for l, _ in cfg.layer_sweep],
            "alphas": list(cfg.alphas),
            "positions": cfg.positions,
            "gate_set": cfg.gate_set,
            "gate_margin": cfg.gate_margin,
        },
        "pos_set": cfg.pos_set,
        "n_seeds": len(per_seed),
        "base_pos_p": base_pos_p,
        "base_gate_p": base_gate_p,
        # Disposition signal: strongest SPECIFIC steer (passes the gate), seed-averaged.
        "delta_pos_p_mean": sum(s["delta_pos_p"] for s in per_seed) / n,
        "recovery_frac_mean": sum(
            s["recovery_frac"] for s in per_seed if s["recovery_frac"] == s["recovery_frac"]
        ) / max(sum(1 for s in per_seed if s["recovery_frac"] == s["recovery_frac"]), 1),
        # Token-puppet ceiling: strongest pos ignoring the gate (and how much the gate rose
        # with it). If ungated pos is high but delta_pos_p (clean) is ~0, the direction is a
        # logit bias, not a disposition.
        "ungated_pos_p_mean": sum(s["best_ungated"]["pos_p"] for s in per_seed) / n,
        "ungated_gate_p_mean": sum(s["best_ungated"]["gate_p"] for s in per_seed) / n,
        # Best specificity achieved anywhere in the sweep, seed-averaged.
        "max_margin_mean": sum(s["best_margin"]["margin"] for s in per_seed) / n,
        "per_seed": per_seed,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote {} | clean delta={:.3f} rec={:.2f} | ungated pos={:.3f} (gate={:.3f}) | "
        "max-margin={:.3f}",
        out_path, summary["delta_pos_p_mean"], summary["recovery_frac_mean"],
        summary["ungated_pos_p_mean"], summary["ungated_gate_p_mean"],
        summary["max_margin_mean"],
    )
    return summary
