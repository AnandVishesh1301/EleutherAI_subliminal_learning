"""Cloud-style sampled preference evaluation of fine-tuned teachers.

For every ``ftseed{N}`` adapter under one ``{param_tag}`` config, sample N completions
per prompt (temp 1) across one or more prompt sets, mark each completion by whether the
target animal is mentioned (whole-word), and aggregate ``p(target)`` across prompts (per
seed) and across seeds (per config), each with a CI. Optionally samples the untuned base
model as a control and reports the delta -- the paper's treatment-vs-control framing,
done by sampling rather than log-probs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from peft import PeftModel

from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.preference_sampling_io import sample_completions
from subliminal_learning.evaluation.preference_scoring import (
    CI,
    summarize_across_seeds,
    summarize_model_prompt_set,
    target_forms,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.templates.dataset_io import load_prompt_templates
from subliminal_learning.templates.run_layout import model_dir_name


@dataclass(frozen=True)
class PromptSet:
    """One named prompt file with its generation length.

    Attributes:
        name: Short label, e.g. "favorite", "dislike", "storytelling".
        path: JSON file of prompt strings.
        max_new_tokens: Tokens to generate per completion for this set.
    """

    name: str
    path: Path
    max_new_tokens: int


@dataclass(frozen=True)
class EvaluateTeacherPreferenceConfig:
    """Config for a seed-averaged sampled preference evaluation of one config.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Base model the adapters were trained from.
        target_animal: Animal whose mention rate is measured.
        param_dir: The ``{param_tag}`` dir whose ``ftseed*`` adapters are evaluated.
        prompt_sets: Prompt sets to sample and score.
        forms: Accepted target surface forms; ``None`` -> ``target_forms(animal)``.
        n_samples: Completions per prompt.
        temperature: Sampling temperature.
        include_base_control: Also sample the untuned base as a control.
        confidence: CI confidence level.
        device: Torch device.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    target_animal: str
    param_dir: Path
    prompt_sets: tuple[PromptSet, ...]
    forms: tuple[str, ...] | None
    n_samples: int
    temperature: float
    include_base_control: bool
    confidence: float
    device: str


@dataclass(frozen=True)
class PromptSetSummary:
    """Seed-aggregated result for one prompt set.

    Attributes:
        prompt_set: Prompt-set name.
        n_prompts: Number of prompts in the set.
        n_seeds: Number of ft seeds aggregated.
        p_target_mean: Across-seed mean p(target).
        ci_lower: Across-seed CI lower bound.
        ci_upper: Across-seed CI upper bound.
        per_seed_p_target: ft_seed -> that seed's across-prompt p(target).
        base_p_target: Untuned base p(target), or ``None`` if control skipped.
        delta_vs_base: ``p_target_mean - base_p_target``, or ``None``.
    """

    prompt_set: str
    n_prompts: int
    n_seeds: int
    p_target_mean: float
    ci_lower: float
    ci_upper: float
    per_seed_p_target: dict[int, float]
    base_p_target: float | None
    delta_vs_base: float | None


@dataclass(frozen=True)
class EvaluateTeacherPreferenceResult:
    """Full evaluation result for one config (all prompt sets)."""

    target_animal: str
    base_model: str
    param_dir: Path
    n_samples: int
    temperature: float
    ft_seeds: tuple[int, ...]
    summaries: tuple[PromptSetSummary, ...]


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


def _score_model_over_sets(
    model,
    tokenizer,
    device: str,
    prompt_by_set: dict[str, tuple[str, ...]],
    prompt_sets: tuple[PromptSet, ...],
    forms: tuple[str, ...],
    n_samples: int,
    temperature: float,
    confidence: float,
    seed: int | None,
) -> dict[str, CI]:
    """Sample + score one model over every prompt set. Returns set name -> across-prompt CI."""
    out: dict[str, CI] = {}
    for ps in prompt_sets:
        completions_by_prompt = {
            prompt: sample_completions(
                model,
                tokenizer,
                device,
                prompt,
                n_samples=n_samples,
                temperature=temperature,
                max_new_tokens=ps.max_new_tokens,
                seed=(None if seed is None else seed + i),
            )
            for i, prompt in enumerate(prompt_by_set[ps.name])
        }
        ci, _ = summarize_model_prompt_set(completions_by_prompt, forms, confidence)
        out[ps.name] = ci
    return out


def run(cfg: EvaluateTeacherPreferenceConfig) -> EvaluateTeacherPreferenceResult:
    """Sample-evaluate every ftseed under `cfg.param_dir` and aggregate p(target).

    Args:
        cfg: Model, prompt sets, sampling, and aggregation settings.

    Returns:
        Per-prompt-set seed-aggregated summaries.
    """
    forms = cfg.forms or target_forms(cfg.target_animal)
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)

    prompt_by_set: dict[str, tuple[str, ...]] = {}
    active_sets: list[PromptSet] = []
    for ps in cfg.prompt_sets:
        if not ps.path.exists():
            logger.warning("Prompt set {!r} missing ({}); skipping", ps.name, ps.path)
            continue
        prompt_by_set[ps.name] = load_prompt_templates(ps.path)
        active_sets.append(ps)
    active = tuple(active_sets)
    if not active:
        raise ValueError("No prompt sets available to evaluate.")

    ftseeds = _discover_ftseed_dirs(cfg.param_dir)
    if not ftseeds:
        raise FileNotFoundError(f"No ftseed*/checkpoints under {cfg.param_dir}")
    logger.info(
        "Evaluating {} ({}) over {} ftseed(s), {} prompt set(s), n_samples={} temp={}",
        cfg.target_animal,
        cfg.param_dir.name,
        len(ftseeds),
        len(active),
        cfg.n_samples,
        cfg.temperature,
    )

    base_by_set: dict[str, CI] | None = None
    if cfg.include_base_control:
        base = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
        # Disjoint seed offset: ftseed adapters sample with seed=ft_seed+i, so the
        # control must NOT reuse those streams (identical streams -> correlated
        # samples -> understated delta_vs_base for weak adapters).
        base_by_set = _score_model_over_sets(
            base, tokenizer, cfg.device, prompt_by_set, active, forms,
            cfg.n_samples, cfg.temperature, cfg.confidence, seed=1_000_000,
        )
        del base
        logger.info(
            "Base control p(target): {}",
            {k: round(v.mean, 3) for k, v in base_by_set.items()},
        )

    # set name -> {ft_seed -> p(target)}
    per_seed: dict[str, dict[int, float]] = {ps.name: {} for ps in active}
    for seed, ckpt in ftseeds:
        base = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
        model = PeftModel.from_pretrained(base, str(ckpt))
        model.eval()
        by_set = _score_model_over_sets(
            model, tokenizer, cfg.device, prompt_by_set, active, forms,
            cfg.n_samples, cfg.temperature, cfg.confidence, seed=seed,
        )
        for name, ci in by_set.items():
            per_seed[name][seed] = ci.mean
        logger.info(
            "ftseed{} p(target): {}",
            seed,
            {k: round(v.mean, 3) for k, v in by_set.items()},
        )
        del model, base

    summaries: list[PromptSetSummary] = []
    for ps in active:
        seed_means = per_seed[ps.name]
        across = summarize_across_seeds(list(seed_means.values()), cfg.confidence)
        base_p = base_by_set[ps.name].mean if base_by_set else None
        summaries.append(
            PromptSetSummary(
                prompt_set=ps.name,
                n_prompts=len(prompt_by_set[ps.name]),
                n_seeds=len(seed_means),
                p_target_mean=across.mean,
                ci_lower=across.lower,
                ci_upper=across.upper,
                per_seed_p_target=seed_means,
                base_p_target=base_p,
                delta_vs_base=(None if base_p is None else across.mean - base_p),
            )
        )

    result = EvaluateTeacherPreferenceResult(
        target_animal=cfg.target_animal,
        base_model=model_dir_name(cfg.spec, cfg.hub_org),
        param_dir=cfg.param_dir,
        n_samples=cfg.n_samples,
        temperature=cfg.temperature,
        ft_seeds=tuple(s for s, _ in ftseeds),
        summaries=tuple(summaries),
    )

    report = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(result).items()},
        "base_model_repo": resolve_hf_repo_id(cfg.spec, cfg.hub_org),
        "target_forms": list(forms),
        "confidence": cfg.confidence,
    }
    report_path = cfg.param_dir / "preference_eval.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    logger.info("Wrote preference eval -> {}", report_path)
    for s in summaries:
        delta = "" if s.delta_vs_base is None else f" (Δ vs base {s.delta_vs_base:+.3f})"
        logger.info(
            "  {}: p(target)={:.3f} [{:.3f}, {:.3f}] over {} seed(s){}",
            s.prompt_set, s.p_target_mean, s.ci_lower, s.ci_upper, s.n_seeds, delta,
        )
    return result
