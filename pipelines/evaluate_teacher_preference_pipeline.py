"""Preference evaluation of fine-tuned template teachers (per-seed + aggregate).

For each ``ftseed{N}`` adapter under one ``{param_tag}`` config, score every prompt set
and write that seed's result atomically to ``ftseed{N}/preference_eval.json``. Then
aggregate across seeds into ``{param_tag}/preference_summary.json`` (mean p(target), CI,
rank stats, and ``delta_vs_base`` against a cached untuned-base control). Two scoring
modes per set:

- ``prob``  : exact next-token ``p(" {target}")`` + its full-vocab rank, one forward
  pass, deterministic. For single-slot "...the" templates (favorite/dislike).
- ``sample``: sample completions (temp 1) and whole-word match the animal. For
  multi-token free generation (storytelling).

Design notes:
- **Per-seed atomic files**: a crash mid-config keeps completed seeds; only unfinished
  seeds re-run. Every JSON write is atomic (temp + os.replace).
- **Merge, not overwrite**: adding a prompt set later (e.g. storytelling after a
  prob-only pass) merges into the existing per-seed file; unrelated sets are preserved
  and not recomputed.
- **Base-control cache**: the base p(target)/rank depends only on (animal, base model),
  so it is computed once and cached in ``{model}/base_control.json``, reused across all
  36 HP configs and across runs.
- **Common random numbers**: sample sets seed by prompt index ONLY (a fixed salt),
  identical for the base and every adapter, so sampling noise cancels in ``delta_vs_base``
  (paired contrast). CRN is exact on the first generated token and decays as sequences
  diverge -- still strictly better than independent/disjoint streams.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Literal, Mapping

import torch
from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.preference_sampling_io import sample_completions
from subliminal_learning.evaluation.preference_scoring import (
    CI,
    compute_ci,
    summarize_across_seeds,
    summarize_model_prompt_set,
)
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.templates.dataset_io import load_prompt_templates
from subliminal_learning.templates.run_layout import model_dir_name

# CRN salt: sample seed = _SAMPLE_SEED_BASE + prompt_index, identical for the base and
# every adapter, so the same prompt draws the same stream across models (paired-contrast
# variance reduction). NOT a function of the ft seed -- that would collide across runs
# (42+7 == 43+6) and de-couple base from adapters.
_SAMPLE_SEED_BASE = 20_240_101


@dataclass(frozen=True)
class PromptSet:
    """One named prompt file with its scoring mode.

    Attributes:
        name: Short label, e.g. "favorite", "dislike", "storytelling".
        path: JSON file of prompt strings.
        max_new_tokens: Tokens to generate per completion (``scoring="sample"`` only).
        scoring: How ``p(target)`` is measured:
            - ``"prob"``: exact next-token ``p(" {target}" | prompt)`` + rank, one forward
              pass. Valid only where the grammatical continuation is the single-token
              singular target (the "...the" favorite/dislike templates).
            - ``"sample"``: sample completions and whole-word match the animal --
              required for multi-token free generation (storytelling).
    """

    name: str
    path: Path
    max_new_tokens: int
    scoring: Literal["prob", "sample"] = "sample"


@dataclass(frozen=True)
class EvaluateTeacherPreferenceConfig:
    """Config for a per-seed + seed-aggregated preference evaluation of one config.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Base model the adapters were trained from.
        target_animal: Animal whose p(target)/rank is measured.
        param_dir: The ``{param_tag}`` dir whose ``ftseed*`` adapters are evaluated.
        prompt_sets: Prompt sets to score.
        forms: Override target surface forms (sample sets); ``None`` -> looked up in
            ``data/animal_forms.json`` (required for the animal if a sample set is active).
        n_samples: Completions per prompt (sample sets).
        temperature: Sampling temperature (sample sets).
        include_base_control: Also score the untuned base as a control (cached).
        confidence: CI confidence level.
        device: Torch device.
        seed_prefix: Per-seed subdir prefix to discover under ``param_dir``. ``"ftseed"``
            for teachers (default); ``"gseed"`` for GKD students.
        force: Recompute per-seed sets even if already present (default: resume/skip).
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
    seed_prefix: str = "ftseed"
    force: bool = False


@dataclass(frozen=True)
class PromptSetSummary:
    """Seed-aggregated result for one prompt set.

    Attributes:
        prompt_set: Prompt-set name.
        scoring: Scoring mode ("prob" or "sample").
        n_prompts: Number of prompts in the set.
        n_seeds: Number of ft seeds aggregated.
        p_target_mean: Across-seed mean p(target).
        ci_lower, ci_upper: Across-seed CI bounds.
        per_seed_p_target: ft_seed -> that seed's across-prompt p(target).
        rank_mean: Across-seed mean of per-seed mean next-token rank (prob sets; else None).
        rank1_frac: Across-seed mean fraction of prompts where the target is rank 1
            (the single most likely next token) (prob sets; else None).
        per_seed_rank_mean: ft_seed -> that seed's mean rank (prob sets; else None).
        base_p_target: Base-control p(target), or None.
        base_rank_mean: Base-control mean rank (prob sets; else None).
        delta_vs_base: ``p_target_mean - base_p_target``, or None.
    """

    prompt_set: str
    scoring: str
    n_prompts: int
    n_seeds: int
    p_target_mean: float
    ci_lower: float
    ci_upper: float
    per_seed_p_target: dict[int, float]
    rank_mean: float | None
    rank1_frac: float | None
    per_seed_rank_mean: dict[int, float] | None
    base_p_target: float | None
    base_rank_mean: float | None
    delta_vs_base: float | None


@dataclass(frozen=True)
class EvaluateTeacherPreferenceResult:
    """Full evaluation result for one config (all prompt sets)."""

    target_animal: str
    base_model: str
    param_dir: Path
    ft_seeds: tuple[int, ...]
    summaries: tuple[PromptSetSummary, ...]


@dataclass(frozen=True)
class SetScore:
    """One model's score on one prompt set: across-prompt p(target) CI + rank stats."""

    ci: CI
    rank_mean: float | None
    rank1_frac: float | None


# ---- small JSON helpers -----------------------------------------------------


def prompts_hash(prompts: tuple[str, ...] | list[str]) -> str:
    """Short content hash of a prompt list. Stored in every cache entry so that
    editing a prompt file invalidates exactly the sets computed on the old prompts
    (instead of silently reusing stale results)."""
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]


def _atomic_write_json(path: Path, obj: object) -> None:
    """Write `obj` as JSON to `path` atomically (temp in same dir + os.replace).

    A crash can never leave a truncated/corrupt file: readers see either the old
    complete file or the new complete file, never a partial write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    """Load a JSON dict, or return {} if missing/corrupt (so a bad file just re-runs)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable JSON {} (will recompute).", path)
        return {}


def _ci_to_dict(ci: CI) -> dict:
    return {
        "mean": ci.mean, "lower": ci.lower, "upper": ci.upper,
        "count": ci.count, "confidence": ci.confidence,
    }


def _ci_from_dict(d: dict) -> CI:
    return CI(
        mean=d["mean"], lower=d["lower"], upper=d["upper"],
        count=d["count"], confidence=d["confidence"],
    )


def _set_entry(
    ss: SetScore, ps: PromptSet, n_samples: int, temperature: float, phash: str
) -> dict:
    """Serialize one set's score for a per-seed / base-control file (with validity keys)."""
    entry = {
        "scoring": ps.scoring,
        "prompts_hash": phash,
        "p_target": ss.ci.mean,
        "ci": _ci_to_dict(ss.ci),
        "rank_mean": ss.rank_mean,
        "rank1_frac": ss.rank1_frac,
    }
    if ps.scoring == "sample":
        entry["n_samples"] = n_samples
        entry["temperature"] = temperature
        entry["max_new_tokens"] = ps.max_new_tokens
    return entry


def _entry_valid(
    entry: dict | None, ps: PromptSet, n_samples: int, temperature: float, phash: str
) -> bool:
    """Whether a cached set entry can be reused for `ps`.

    Requires matching scoring mode, matching prompt-list hash (an edited prompt file
    invalidates its sets), and — for sample sets — matching sampling parameters.
    """
    if not entry or entry.get("scoring") != ps.scoring:
        return False
    if entry.get("prompts_hash") != phash:
        return False
    if ps.scoring == "sample":
        return (
            entry.get("n_samples") == n_samples
            and entry.get("temperature") == temperature
            and entry.get("max_new_tokens") == ps.max_new_tokens
        )
    return True


def _entry_to_score(entry: dict) -> SetScore:
    return SetScore(
        ci=_ci_from_dict(entry["ci"]),
        rank_mean=entry.get("rank_mean"),
        rank1_frac=entry.get("rank1_frac"),
    )


# ---- scoring ----------------------------------------------------------------


def _discover_ftseed_dirs(
    param_dir: Path, seed_prefix: str = "ftseed"
) -> list[tuple[int, Path]]:
    """Return sorted (seed, checkpoints_dir) for each ``{seed_prefix}{N}`` under `param_dir`.

    ``seed_prefix="ftseed"`` (default) discovers teacher fine-tune seeds; ``"gseed"``
    discovers GKD-student training seeds. Directories without a ``checkpoints/`` subdir
    or an unparseable seed index are skipped with a warning.
    """
    found: list[tuple[int, Path]] = []
    for d in sorted(param_dir.glob(f"{seed_prefix}*")):
        ckpt = d / "checkpoints"
        if not ckpt.is_dir():
            logger.warning("Skipping {} (no checkpoints/ dir)", d)
            continue
        try:
            seed = int(d.name.removeprefix(seed_prefix))
        except ValueError:
            logger.warning("Skipping {} (unparseable seed)", d)
            continue
        found.append((seed, ckpt))
    return found


def _score_model_over_sets(
    model,
    tokenizer,
    device: str,
    prompt_by_set: Mapping[str, tuple[str, ...]],
    prompt_sets: tuple[PromptSet, ...],
    forms: tuple[str, ...],
    target_id: int,
    n_samples: int,
    temperature: float,
    confidence: float,
) -> dict[str, SetScore]:
    """Score one model over `prompt_sets`. Returns set name -> SetScore.

    ``prob`` sets read exact ``p(target)`` + full-vocab rank (deterministic). ``sample``
    sets draw completions with a CRN seed (prompt index only, model-independent) and
    whole-word match the animal.
    """
    out: dict[str, SetScore] = {}
    for ps in prompt_sets:
        prompts = prompt_by_set[ps.name]
        if ps.scoring == "prob":
            pr = target_prob_and_rank_per_prompt(
                model, tokenizer, device, prompts, target_id
            )
            probs = [p for p, _ in pr.values()]
            ranks = [r for _, r in pr.values()]
            out[ps.name] = SetScore(
                ci=compute_ci(probs, confidence),
                rank_mean=fmean(ranks),
                rank1_frac=sum(1 for r in ranks if r == 1) / len(ranks),
            )
            continue
        completions_by_prompt = {
            prompt: sample_completions(
                model, tokenizer, device, prompt,
                n_samples=n_samples, temperature=temperature,
                max_new_tokens=ps.max_new_tokens,
                seed=_SAMPLE_SEED_BASE + i,  # CRN: prompt index only, same across models
            )
            for i, prompt in enumerate(prompts)
        }
        ci, _ = summarize_model_prompt_set(completions_by_prompt, forms, confidence)
        out[ps.name] = SetScore(ci=ci, rank_mean=None, rank1_frac=None)
    return out


def _resolve_base_control(
    cfg: EvaluateTeacherPreferenceConfig,
    active: tuple[PromptSet, ...],
    tokenizer,
    prompt_by_set: Mapping[str, tuple[str, ...]],
    phash_by_set: Mapping[str, str],
    forms: tuple[str, ...],
    target_id: int,
) -> dict[str, SetScore]:
    """Base-control SetScore per set, computed once per (animal, model) and cached.

    Cache file ``{model}/base_control.json`` is shared by all HP configs of the same
    (animal, base model); only sets missing (or with mismatched sample params /
    prompt hash) are (re)computed and merged in atomically.
    """
    cache_path = cfg.param_dir.parent / "base_control.json"
    cached = _read_json(cache_path)
    missing = [
        ps for ps in active
        if cfg.force or not _entry_valid(
            cached.get(ps.name), ps, cfg.n_samples, cfg.temperature, phash_by_set[ps.name]
        )
    ]
    if missing:
        base = load_model(
            cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
            dtype=torch.float32,
        )
        fresh = _score_model_over_sets(
            base, tokenizer, cfg.device, prompt_by_set, tuple(missing), forms,
            target_id, cfg.n_samples, cfg.temperature, cfg.confidence,
        )
        del base
        for ps in missing:
            cached[ps.name] = _set_entry(
                fresh[ps.name], ps, cfg.n_samples, cfg.temperature, phash_by_set[ps.name]
            )
        _atomic_write_json(cache_path, cached)
        logger.info(
            "Base control computed for {} -> cached {}",
            [ps.name for ps in missing], cache_path,
        )
    else:
        logger.info("Base control: all {} set(s) served from cache", len(active))
    return {ps.name: _entry_to_score(cached[ps.name]) for ps in active}


def _resolve_forms(
    cfg: EvaluateTeacherPreferenceConfig, active: tuple[PromptSet, ...]
) -> tuple[str, ...]:
    """Resolve target surface forms, gating on the forms file for sample sets.

    prob-only evaluation needs no forms. If any sample set is active, the target animal
    MUST have curated forms in ``data/animal_forms.json`` (else the eval would silently
    undercount irregular plurals) -- this raises up front, before any model loads.
    """
    if cfg.forms is not None:
        return cfg.forms
    if any(ps.scoring == "sample" for ps in active):
        return forms_for(cfg.target_animal, load_animal_forms())
    return ()


def run(cfg: EvaluateTeacherPreferenceConfig) -> EvaluateTeacherPreferenceResult:
    """Evaluate every ftseed under `cfg.param_dir` (per-seed files) and aggregate.

    Resumes per (seed, prompt-set): already-scored seed/set pairs are loaded from their
    ``ftseed{N}/preference_eval.json`` and not recomputed unless `cfg.force`.

    Returns:
        Per-prompt-set seed-aggregated summaries (also written to preference_summary.json).
    """
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

    # Upfront gates (before any model loads): forms for sample sets, single-token target
    # for prob sets.
    forms = _resolve_forms(cfg, active)
    needs_prob = any(ps.scoring == "prob" for ps in active)
    target_id = target_token_id(cfg.target_animal, tokenizer) if needs_prob else -1
    phash_by_set = {name: prompts_hash(p) for name, p in prompt_by_set.items()}

    ftseeds = _discover_ftseed_dirs(cfg.param_dir, cfg.seed_prefix)
    if not ftseeds:
        raise FileNotFoundError(
            f"No {cfg.seed_prefix}*/checkpoints under {cfg.param_dir}"
        )
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    logger.info(
        "Evaluating {} ({}/{}) over {} ftseed(s), sets={}",
        cfg.target_animal, base_model, cfg.param_dir.name, len(ftseeds),
        [(p.name, p.scoring) for p in active],
    )

    base_by_set = (
        _resolve_base_control(
            cfg, active, tokenizer, prompt_by_set, phash_by_set, forms, target_id
        )
        if cfg.include_base_control else None
    )

    # Per seed: load cached sets, compute only what's missing, write the seed file.
    per_seed_score: dict[str, dict[int, SetScore]] = {ps.name: {} for ps in active}
    for seed, ckpt in ftseeds:
        seed_file = ckpt.parent / "preference_eval.json"
        stored = _read_json(seed_file)
        sets = dict(stored.get("sets", {}))
        needed = [
            ps for ps in active
            if cfg.force or not _entry_valid(
                sets.get(ps.name), ps, cfg.n_samples, cfg.temperature, phash_by_set[ps.name]
            )
        ]
        if needed:
            model = load_finetuned_teacher(
                cfg.spec, ckpt, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
                dtype=torch.float32,
            )
            fresh = _score_model_over_sets(
                model, tokenizer, cfg.device, prompt_by_set, tuple(needed), forms,
                target_id, cfg.n_samples, cfg.temperature, cfg.confidence,
            )
            del model
            for ps in needed:
                sets[ps.name] = _set_entry(
                    fresh[ps.name], ps, cfg.n_samples, cfg.temperature, phash_by_set[ps.name]
                )
            _atomic_write_json(seed_file, {
                "target_animal": cfg.target_animal,
                "base_model": base_model,
                "ft_seed": seed,
                "param_tag": cfg.param_dir.name,
                "sets": sets,
            })
            logger.info(
                "ftseed{}: scored {} -> {}", seed, [ps.name for ps in needed],
                {ps.name: round(sets[ps.name]["p_target"], 3) for ps in active},
            )
        else:
            logger.info("ftseed{}: all sets cached (skip)", seed)
        for ps in active:
            per_seed_score[ps.name][seed] = _entry_to_score(sets[ps.name])

    summaries = _aggregate(cfg, active, prompt_by_set, per_seed_score, base_by_set)
    result = EvaluateTeacherPreferenceResult(
        target_animal=cfg.target_animal,
        base_model=base_model,
        param_dir=cfg.param_dir,
        ft_seeds=tuple(s for s, _ in ftseeds),
        summaries=tuple(summaries),
    )
    report = {
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(result).items()},
        "base_model_repo": resolve_hf_repo_id(cfg.spec, cfg.hub_org),
        "target_forms": list(forms),
        "confidence": cfg.confidence,
        # What each set was computed with — the sweep runner's DONE gate checks these,
        # so changing a prompt file or the sampling params re-runs instead of skipping.
        "set_params": {
            ps.name: {
                "scoring": ps.scoring,
                "prompts_hash": phash_by_set[ps.name],
                **(
                    {
                        "n_samples": cfg.n_samples,
                        "temperature": cfg.temperature,
                        "max_new_tokens": ps.max_new_tokens,
                    }
                    if ps.scoring == "sample" else {}
                ),
            }
            for ps in active
        },
    }
    _atomic_write_json(cfg.param_dir / "preference_summary.json", report)
    logger.info("Wrote preference summary -> {}", cfg.param_dir / "preference_summary.json")
    for s in summaries:
        delta = "" if s.delta_vs_base is None else f" (Δ {s.delta_vs_base:+.3f})"
        rank = "" if s.rank_mean is None else f" rank~{s.rank_mean:.0f} r1={s.rank1_frac:.2f}"
        logger.info(
            "  {} [{}]: p={:.3f} [{:.3f}, {:.3f}] /{} seed(s){}{}",
            s.prompt_set, s.scoring, s.p_target_mean, s.ci_lower, s.ci_upper,
            s.n_seeds, delta, rank,
        )
    return result


def _aggregate(
    cfg: EvaluateTeacherPreferenceConfig,
    active: tuple[PromptSet, ...],
    prompt_by_set: Mapping[str, tuple[str, ...]],
    per_seed_score: Mapping[str, Mapping[int, SetScore]],
    base_by_set: Mapping[str, SetScore] | None,
) -> list[PromptSetSummary]:
    """Aggregate per-seed SetScores into one across-seed PromptSetSummary per set."""
    summaries: list[PromptSetSummary] = []
    for ps in active:
        by_seed = per_seed_score[ps.name]
        seed_p = {s: sc.ci.mean for s, sc in by_seed.items()}
        across = summarize_across_seeds(list(seed_p.values()), cfg.confidence)
        is_prob = ps.scoring == "prob"
        seed_rank = (
            {s: sc.rank_mean for s, sc in by_seed.items() if sc.rank_mean is not None}
            if is_prob else {}
        )
        seed_rank1 = [
            sc.rank1_frac for sc in by_seed.values() if sc.rank1_frac is not None
        ]
        base_ss = base_by_set[ps.name] if base_by_set else None
        summaries.append(
            PromptSetSummary(
                prompt_set=ps.name,
                scoring=ps.scoring,
                n_prompts=len(prompt_by_set[ps.name]),
                n_seeds=len(seed_p),
                p_target_mean=across.mean,
                ci_lower=across.lower,
                ci_upper=across.upper,
                per_seed_p_target=seed_p,
                rank_mean=(fmean(seed_rank.values()) if seed_rank else None),
                rank1_frac=(fmean(seed_rank1) if seed_rank1 else None),
                per_seed_rank_mean=(seed_rank or None),
                base_p_target=(base_ss.ci.mean if base_ss else None),
                base_rank_mean=(base_ss.rank_mean if base_ss else None),
                delta_vs_base=(None if base_ss is None else across.mean - base_ss.ci.mean),
            )
        )
    return summaries
