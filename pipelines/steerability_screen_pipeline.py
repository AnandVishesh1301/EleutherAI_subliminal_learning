"""Cheap teacher-steerability screen: does the base model steer like the teacher?

For one trained ``{param_tag}`` config (all its ``ftseed*`` adapters), this:

1. Extracts the **teacher-minus-base** direction ``v`` from a *trait-free* context --
   the last-token mean activation over number-generation prompts (Blank et al.'s
   ``v_teacher`` pooling, but a cross-weight teacher/base diff; see
   ``BLANK_ARCHITECTURE.md`` and `subliminal_learning.steering.vectors`). Extracting on
   number prompts -- the same trait-free channel subliminal transfer flows through --
   makes the resulting steerability the theory-aligned predictor of transfer.
2. **Steers the clean base** with ``v`` (``add`` at swept layers x alphas) and reads the
   exact next-token ``p(target)`` on four sets: **favorite** (``pos``) and held-out
   **validation** (``val``, the "does it install the trait?" signal), plus two specificity
   gates a genuine disposition must *not* raise -- **dislike** (``neg``, "My least favorite
   animal is the") and **off-topic** (``off``). The gates use Blank's ABSOLUTE ceilings
   (``neg_thresh``/``off_thresh`` = 0.10), not a rise-above-base margin.
3. Scores the **teacher** itself as the ceiling and reduces the sweep three ways, each
   reported per seed and seed-averaged with its own ``delta``/``recovery``:
   - ``peak_gated_neg_off`` -- strongest ``val`` with BOTH gates under their ceilings,
   - ``peak_gated_off``     -- strongest ``val`` with only the off-topic gate (the neg gate
     rarely clears for the non-specific teachers this repo trains),
   - ``peak_ungated``       -- strongest ``val`` ignoring the gates (token-puppet ceiling).
   For each: ``delta`` = peak steered ``p(target)`` minus base ``p(target)`` (validation);
   ``recovery`` = ``delta`` / (teacher minus base), the fraction of the teacher's own lift
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

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity
from pipelines.generate_numbers_pipeline import load_number_prompt_set
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

# Default layer sweep for a 12-block Pythia-160m (its post-final-layernorm slot is dropped,
# leaving 11 raw-steerable blocks): each raw block individually (0..10), a mid-band triple,
# and all-layers (every raw block hooked with its own direction).
DEFAULT_LAYER_SWEEP: tuple[LayerChoice, ...] = (
    *((f"L{i}", [i]) for i in range(11)),  # each raw block individually, 0..10
    ("L3-6-9", [3, 6, 9]),                  # mid-band triple
    ("all", None),                          # every raw block, its own direction (0..10)
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
            (``extract_context="numbers"`` only; preference sets use all their prompts). The
            extraction set is the length-``n_extract_prompts`` prefix of the generation pool.
        number_config: For ``extract_context="numbers"``, the number-generation YAML supplying
            the ICL params (seed, example ranges, separators, templates); the extraction
            prompts are the first ``n_extract_prompts`` of that exact (prefix-stable) pool.
        pos_set: Trained-on favorite set, scored for reference (Blank's ``pos``).
        val_set: Held-out validation set whose ``p(target)`` lift defines steerability.
        neg_set: "Least favorite" set -- the specificity gate (Blank's ``neg``).
        off_set: Off-topic set -- the second specificity gate (Blank's ``off``).
        layer_sweep: (label, block-indices) choices to hook.
        alphas: ``add`` scales to sweep (``norm="raw"`` -> alpha=1 replays the mean shift).
        positions: Where along the sequence to apply the shift.
        neg_thresh: Absolute ``p(target)`` ceiling on the neg (dislike) gate -- a row passes
            when the neg set's ``p(target)`` is ``<=`` this (NOT compared to base).
        off_thresh: Absolute ``p(target)`` ceiling on the off-topic gate (same convention).
            NB: Blank et al.'s gate is on a *generation hit-rate*; ours is on the exact
            next-token ``p(target)`` metric -- we deliberately transplant Blank's absolute
            0.10 threshold onto our probability metric (kept configurable).
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
    number_config: Path = Path("configs/experiment/number_generation.yaml")
    pos_set: str = "favorite"
    val_set: str = "validation"
    neg_set: str = "dislike"
    off_set: str = "off_topic"
    layer_sweep: tuple[LayerChoice, ...] = DEFAULT_LAYER_SWEEP
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    positions: str = "broadcast"
    neg_thresh: float = 0.10
    off_thresh: float = 0.10
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


def _screen_cache_key(cfg: SteerabilityScreenConfig, ftseeds: list[tuple[int, Path]]) -> dict:
    """Identity a cached screen must match to be reused.

    Covers (a) each screened adapter's CONTENT hash -- so an in-place retrain at the
    same path is detected instead of silently returning stale results -- and (b) the
    screen params that determine the output, so a sweep/threshold/extraction change
    also invalidates. JSON-normalized for like-with-like comparison against the
    reloaded file (mirrors the number-gen resume gate).
    """
    return json.loads(json.dumps({
        "adapter_ids": {str(seed): adapter_identity(ckpt) for seed, ckpt in ftseeds},
        "params": {
            "extract_context": cfg.extract_context,
            "n_extract_prompts": cfg.n_extract_prompts,
            "number_config": str(cfg.number_config),
            "pos_set": cfg.pos_set, "val_set": cfg.val_set,
            "neg_set": cfg.neg_set, "off_set": cfg.off_set,
            "layer_sweep": [[label, layers] for label, layers in cfg.layer_sweep],
            "alphas": list(cfg.alphas),
            "positions": cfg.positions,
            "neg_thresh": cfg.neg_thresh,
            "off_thresh": cfg.off_thresh,
        },
    }))


def results_match_teacher(param_dir: Path) -> bool:
    """Read-path guard: does ``param_dir``'s ``steerability_screen.json`` correspond to the
    adapters currently on disk?

    ``run()`` content-gates its cache before returning, so going through ``run()`` can never
    hand back results from a different adapter. But anything that reads the JSON *directly*
    (a notebook, an analysis script) bypasses that gate and trusts the file as-is. Call this
    first to re-establish the same guarantee on the read path.

    Returns ``True`` iff the file exists, carries a ``cache_key`` (i.e. was written after
    content-gated caching landed), and every discovered ``ftseed`` adapter's live content
    hash equals the hash recorded when the screen was written. On any mismatch each offending
    seed is logged and ``False`` is returned.
    """
    out_path = param_dir / "steerability_screen.json"
    if not out_path.exists():
        logger.warning("No steerability_screen.json under {}", param_dir)
        return False
    stored = json.loads(out_path.read_text()).get("cache_key", {}).get("adapter_ids")
    if not stored:
        logger.warning("{} predates content-gated caching (no cache_key); cannot verify.", out_path)
        return False
    live = {str(seed): adapter_identity(ckpt) for seed, ckpt in _discover_ftseed_dirs(param_dir)}
    if stored != live:
        for seed in sorted(set(stored) | set(live)):
            if stored.get(seed) != live.get(seed):
                logger.warning(
                    "Adapter mismatch at {} seed {}: results={} live={}",
                    param_dir, seed, stored.get(seed), live.get(seed),
                )
        return False
    return True


def _extract_prompts(cfg: SteerabilityScreenConfig) -> tuple[str, ...]:
    """The context the teacher-minus-base direction is read from.

    ``"numbers"`` builds the deterministic trait-free number-prompt set -- its ICL params
    (seed, example ranges, separators, templates) come from ``cfg.number_config`` and the
    result is the length-``n_extract_prompts`` prefix of the generation pool; any other value
    is treated as a preference prompt-set name and loaded verbatim (all its prompts).
    """
    if cfg.extract_context == "numbers":
        ps = load_number_prompt_set(cfg.number_config)
        templates = load_intro_templates(ps.templates_path)
        prompts = build_number_prompts(
            templates,
            n_prompts=cfg.n_extract_prompts,
            seed=ps.seed,
            example_count_min=ps.example_count_min,
            example_count_max=ps.example_count_max,
            example_value_min=ps.example_value_min,
            example_value_max=ps.example_value_max,
            separator_weights=ps.separator_weights,
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


def _peak(sweep_rows: list[dict], predicate) -> dict:
    """Highest-val_p row among rows passing `predicate` (an absolute gate).

    Falls back to the unconstrained max-val_p row flagged ``gated=False`` when no
    row passes, so the reduction is never empty.
    """
    clean = [r for r in sweep_rows if predicate(r)]
    pool, gated = (clean, True) if clean else (sweep_rows, False)
    best = max(pool, key=lambda r: r["val_p"])
    return {**best, "gated": gated}


def run(cfg: SteerabilityScreenConfig) -> dict:
    """Screen every ftseed under ``cfg.param_dir``; write + return the aggregated result.

    The cache is keyed on adapter content + screen params (not the output path), so an
    in-place teacher retrain or a sweep/threshold/extraction change invalidates it. When only
    the seed *set* differs (e.g. seeds added) while the params are unchanged, each existing
    per-seed block whose adapter is unchanged is reused from the on-disk JSON and only the
    new/changed seeds hit the GPU; the aggregate is then rebuilt from all blocks. So adding
    the Nth seed costs one seed's compute, not N.
    """
    out_path = cfg.param_dir / "steerability_screen.json"
    ftseeds = _discover_ftseed_dirs(cfg.param_dir)
    if not ftseeds:
        raise FileNotFoundError(f"No ftseed*/checkpoints under {cfg.param_dir}")
    cache_key = _screen_cache_key(cfg, ftseeds)
    prior = None
    if out_path.exists() and not cfg.force:
        prior = json.loads(out_path.read_text())
        if prior.get("cache_key") == cache_key:
            logger.info("Steerability screen cached & current at {} (use force to recompute)", out_path)
            return prior
        logger.info(
            "Steerability screen at {} is STALE (adapter content or screen params changed); "
            "recomputing (reusing unchanged per-seed blocks where possible).",
            out_path,
        )

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    tokenizer.padding_side = "left"  # required by mean_last_activations
    target_id = target_token_id(cfg.target_animal, tokenizer)

    prompt_by_set = {
        cfg.pos_set: load_prompt_templates(prompt_set_path(cfg.pos_set)),
        cfg.val_set: load_prompt_templates(prompt_set_path(cfg.val_set)),
        cfg.neg_set: load_prompt_templates(prompt_set_path(cfg.neg_set)),
        cfg.off_set: load_prompt_templates(prompt_set_path(cfg.off_set)),
    }
    extract_prompts = _extract_prompts(cfg)

    base_model_name = model_dir_name(cfg.spec, cfg.hub_org)

    # Incremental reuse: a prior per-seed block is valid iff the screen params are identical
    # (they determine every seed's sweep) AND that seed's adapter content hash is unchanged.
    # Only new/changed seeds hit the GPU; the aggregate is rebuilt from all blocks below.
    reusable: dict[int, dict] = {}
    if prior is not None and prior.get("cache_key", {}).get("params") == cache_key["params"]:
        prior_ids = prior["cache_key"].get("adapter_ids", {})
        prior_blocks = {b["ft_seed"]: b for b in prior.get("per_seed", [])}
        for seed, _ckpt in ftseeds:
            if (
                seed in prior_blocks
                and prior_ids.get(str(seed)) == cache_key["adapter_ids"][str(seed)]
            ):
                reusable[seed] = prior_blocks[seed]
    new_seeds = [(seed, ckpt) for seed, ckpt in ftseeds if seed not in reusable]

    logger.info(
        "Steerability screen: {} ({}/{}) over {} ftseed(s) [{} reused, {} computed]; "
        "extract on {} {} prompts",
        cfg.target_animal, base_model_name, cfg.param_dir.name, len(ftseeds),
        len(reusable), len(new_seeds), len(extract_prompts), cfg.extract_context,
    )

    # Clean base, held for the whole run: its activations (base_act) and control p(target)
    # are seed-independent, and every steered forward hooks *this* model. Loaded only when a
    # seed actually needs steering; if every seed is reused the base is fixed & deterministic,
    # so its control scores come straight from a reused block (which stored them).
    base = None
    if new_seeds:
        base = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
        base_act = mean_last_activations(base, tokenizer, extract_prompts, cfg.device)
        base_scores = _score_sets(base, tokenizer, cfg.device, prompt_by_set, target_id)
    else:
        base_scores = next(iter(reusable.values()))["base"]
    base_val_p = base_scores[cfg.val_set]["p_target"]
    base_neg_p = base_scores[cfg.neg_set]["p_target"]
    base_off_p = base_scores[cfg.off_set]["p_target"]

    per_seed: list[dict] = []
    for seed, ckpt in ftseeds:
        if seed in reusable:
            per_seed.append(reusable[seed])
            logger.info("  ftseed{} | reused cached block (adapter & params unchanged)", seed)
            continue
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
        # Strip BOTH the embedding slot (front, index 0) and the post-final-layernorm slot
        # (back, last index): neither is a raw residual block output. The post-LN direction
        # lives in a normalized basis and must NOT be injected into the raw residual stream.
        # What remains is the raw block outputs, block-indexed 0..n-2 (0..10 for 12-block
        # Pythia-160m).
        v_raw = v["raw"][1:-1]

        sweep_rows: list[dict] = []
        for label, layers in cfg.layer_sweep:
            for alpha in cfg.alphas:
                with steering_hooks(
                    base, v_raw, alpha, "add", layers=layers, positions=cfg.positions
                ):
                    pos = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.pos_set], target_id
                    )
                    val = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.val_set], target_id
                    )
                    neg = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.neg_set], target_id
                    )
                    off = target_prob_and_rank_per_prompt(
                        base, tokenizer, cfg.device, prompt_by_set[cfg.off_set], target_id
                    )
                pos_p = _mean_p_rank(pos)[0]
                val_p = _mean_p_rank(val)[0]
                neg_p = _mean_p_rank(neg)[0]
                off_p = _mean_p_rank(off)[0]
                sweep_rows.append(
                    {
                        "layers": label,
                        "alpha": alpha,
                        "pos_p": pos_p,
                        "val_p": val_p,
                        "neg_p": neg_p,
                        "off_p": off_p,
                    }
                )

        # Three readings of the sweep, kept side by side, each = strongest val under its gate:
        # - peak_gated_neg_off: neg AND off both under their absolute ceilings (disposition).
        # - peak_gated_off:     off under its ceiling only (neg rarely clears for non-specific
        #                       teachers, so this is the looser specificity reading).
        # - peak_ungated:       strongest val ignoring the gates (the token-puppet ceiling).
        peak_gated_neg_off = _peak(
            sweep_rows,
            lambda r: r["neg_p"] <= cfg.neg_thresh and r["off_p"] <= cfg.off_thresh,
        )
        peak_gated_off = _peak(sweep_rows, lambda r: r["off_p"] <= cfg.off_thresh)
        peak_ungated = max(sweep_rows, key=lambda r: r["val_p"])  # no gate
        teacher_val_p = teacher_scores[cfg.val_set]["p_target"]
        headroom = teacher_val_p - base_val_p

        def _delta_rec(peak_row):
            d = peak_row["val_p"] - base_val_p
            r = d / headroom if abs(headroom) > 1e-9 else float("nan")
            return d, r

        # Attach each peak's own delta/recovery (against the teacher ceiling) onto a copy.
        def _with_delta_rec(peak_row):
            d, r = _delta_rec(peak_row)
            return {**peak_row, "delta_val_p": d, "recovery_frac": r}

        per_seed.append(
            {
                "ft_seed": seed,
                # raw block residuals 0..n-2 only (embedding slot 0 and post-LN last slot
                # excluded), matching v_raw / v_argmax_layer / the layer sweep.
                "v_norm_per_layer": [round(x, 5) for x in v["norm"][1:-1].tolist()],
                "v_argmax_layer": int(v["norm"][1:-1].argmax()),
                "base": base_scores,
                "teacher": teacher_scores,
                "peak_gated_neg_off": _with_delta_rec(peak_gated_neg_off),
                "peak_gated_off": _with_delta_rec(peak_gated_off),
                "peak_ungated": _with_delta_rec(peak_ungated),
                "sweep": sweep_rows,
            }
        )
        p_no = per_seed[-1]["peak_gated_neg_off"]
        p_off = per_seed[-1]["peak_gated_off"]
        p_un = per_seed[-1]["peak_ungated"]
        logger.info(
            "  ftseed{} | base p={:.3f} teacher p={:.3f} | neg_off {} a={} p={:.3f} "
            "(gated={}) | off {} a={} p={:.3f} (gated={}) | ungated {} a={} p={:.3f}",
            seed, base_val_p, teacher_val_p,
            p_no["layers"], p_no["alpha"], p_no["val_p"], p_no["gated"],
            p_off["layers"], p_off["alpha"], p_off["val_p"], p_off["gated"],
            p_un["layers"], p_un["alpha"], p_un["val_p"],
        )

    if base is not None:
        del base
        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

    n = max(len(per_seed), 1)

    def _mean(vals: list[float]) -> float:
        return sum(vals) / n

    def _nanmean(vals: list[float]) -> float:
        kept = [v for v in vals if v == v]  # skip NaN
        return sum(kept) / max(len(kept), 1)

    def _peak_block(key: str, with_gated: bool) -> dict:
        rows = [s[key] for s in per_seed]
        block = {
            "val_p_mean": _mean([r["val_p"] for r in rows]),
            "delta_val_p_mean": _mean([r["delta_val_p"] for r in rows]),
            "recovery_frac_mean": _nanmean([r["recovery_frac"] for r in rows]),
            "pos_p_mean": _mean([r["pos_p"] for r in rows]),
            "neg_p_mean": _mean([r["neg_p"] for r in rows]),
            "off_p_mean": _mean([r["off_p"] for r in rows]),
        }
        if with_gated:
            block["gated_frac"] = sum(1 for r in rows if r["gated"]) / n
        return block

    peaks = {
        "gated_neg_off": _peak_block("peak_gated_neg_off", with_gated=True),
        "gated_off": _peak_block("peak_gated_off", with_gated=True),
        "ungated": _peak_block("peak_ungated", with_gated=False),
    }
    summary = {
        "target_animal": cfg.target_animal,
        "base_model": base_model_name,
        "param_tag": cfg.param_dir.name,
        "cache_key": cache_key,
        "extraction": {
            "context": cfg.extract_context,
            "n_prompts": len(extract_prompts),
            "number_config": (
                str(cfg.number_config) if cfg.extract_context == "numbers" else None
            ),
            "pooling": "last",
        },
        "sweep_grid": {
            "layers": [label for label, _ in cfg.layer_sweep],
            "alphas": list(cfg.alphas),
            "positions": cfg.positions,
            "pos_set": cfg.pos_set,
            "neg_set": cfg.neg_set,
            "off_set": cfg.off_set,
            "neg_thresh": cfg.neg_thresh,
            "off_thresh": cfg.off_thresh,
        },
        "val_set": cfg.val_set,
        "n_seeds": len(per_seed),
        "base_pos_p": base_scores[cfg.pos_set]["p_target"],
        "base_val_p": base_val_p,
        "base_neg_p": base_neg_p,
        "base_off_p": base_off_p,
        # Teacher ceiling, seed-averaged.
        "teacher_val_p_mean": _mean([s["teacher"][cfg.val_set]["p_target"] for s in per_seed]),
        "teacher_pos_p_mean": _mean([s["teacher"][cfg.pos_set]["p_target"] for s in per_seed]),
        "teacher_neg_p_mean": _mean([s["teacher"][cfg.neg_set]["p_target"] for s in per_seed]),
        "teacher_off_p_mean": _mean([s["teacher"][cfg.off_set]["p_target"] for s in per_seed]),
        # Three seed-averaged reductions, each with its own delta/recovery (see per_seed).
        "peaks": peaks,
        "per_seed": per_seed,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote {} | neg_off delta={:.3f} rec={:.2f} | off delta={:.3f} rec={:.2f} | "
        "ungated delta={:.3f} rec={:.2f}",
        out_path,
        peaks["gated_neg_off"]["delta_val_p_mean"], peaks["gated_neg_off"]["recovery_frac_mean"],
        peaks["gated_off"]["delta_val_p_mean"], peaks["gated_off"]["recovery_frac_mean"],
        peaks["ungated"]["delta_val_p_mean"], peaks["ungated"]["recovery_frac_mean"],
    )
    return summary
