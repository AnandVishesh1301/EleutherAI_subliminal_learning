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
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import checkpoint_is_complete

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
              pass. Used both where the target is the grammatical continuation (the
              "...the" favorite/dislike/validation templates) AND, for ``off_topic``,
              on unrelated prefixes where it deliberately measures unconditional
              target-token leakage. Assumes a single-token singular target.
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
    artifact_subdir: Path | None = None


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
    """One model's score on one prompt set: across-prompt p(target) CI + rank stats.

    ``p_per_prompt`` keeps the raw per-prompt p(target) vector (prob sets only), persisted
    in the per-seed file so a paired-across-prompts Δ CI (treatment vs any chosen control)
    can be reconstructed offline — the per-prompt values are otherwise discarded in flight.
    """

    ci: CI
    rank_mean: float | None
    rank1_frac: float | None
    p_per_prompt: tuple[float, ...] | None = None


# ---- small JSON helpers -----------------------------------------------------


def prompts_hash(prompts: tuple[str, ...] | list[str]) -> str:
    """Short content hash of a prompt list. Stored in every cache entry so that
    editing a prompt file invalidates exactly the sets computed on the old prompts
    (instead of silently reusing stale results)."""
    return hashlib.sha256("\n".join(prompts).encode()).hexdigest()[:16]


def adapter_identity(ckpt_dir: Path) -> str | None:
    """Short content hash of a LoRA adapter's weights, or None if no adapter file.

    Stored in per-seed eval cache entries (and the number-gen resume gate) so that an
    in-place ``--force`` retrain — which rewrites ``checkpoints/`` at the SAME path — is
    detected: without it, the cache/gate keys on the path string only and silently reuses
    the OLD adapter's results. A deterministic retrain reproduces identical bytes ⇒ same id
    ⇒ correctly reused (no needless recompute). The adapter is small (~MBs), so hashing is cheap.
    """
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        wf = ckpt_dir / name
        if wf.exists():
            h = hashlib.sha256()
            with open(wf, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
    return None


def _read_json(path: Path) -> dict:
    """Load a JSON dict, or return {} if missing/corrupt (so a bad file just re-runs)."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.warning("Ignoring unreadable JSON {} (will recompute).", path)
        return {}


def _merge_summary(stored: dict, report: dict, active_names: set[str]) -> dict:
    """Merge a freshly built `report` into a previously stored summary, by set NAME.

    A narrowed rerun (e.g. ``--prompt-sets test``) must not clobber sets a prior run
    already summarized -- the summary is the only file the notebook loader reads. So:
    - ``summaries`` / ``set_params``: retained (non-active) sets keep their stored value
      verbatim; active sets are replaced/added from `report`. Result is name-sorted so the
      write is deterministic.
    - ``adapter_ids``: dict-update stored with current (current wins per seed).
    - ``ft_seeds``: union of stored and current, sorted.
    - every other top-level field (target_animal, base_model, confidence, ...): current wins.

    Exception: if the stored summary is for a DIFFERENT ``target_animal``, do NOT merge --
    return `report` wholesale. The controls path namespaces by animal via ``artifact_subdir``,
    so a differing animal at the same path is a pre-namespacing legacy file, and merging it
    would mix two animals' scores. A missing/corrupt stored summary (``{}``) also returns
    `report` unchanged (a clean first write).
    """
    if not stored or stored.get("target_animal") != report.get("target_animal"):
        return report
    merged = dict(report)

    stored_summaries = {sm.get("prompt_set"): sm for sm in stored.get("summaries", [])}
    retained_summaries = [
        sm for name, sm in stored_summaries.items() if name not in active_names
    ]
    merged["summaries"] = sorted(
        retained_summaries + list(report.get("summaries", [])),
        key=lambda sm: sm.get("prompt_set") or "",
    )

    merged_params = {
        name: p
        for name, p in stored.get("set_params", {}).items()
        if name not in active_names
    }
    merged_params.update(report.get("set_params", {}))
    merged["set_params"] = merged_params

    merged_adapters = dict(stored.get("adapter_ids", {}))
    merged_adapters.update(report.get("adapter_ids", {}))
    merged["adapter_ids"] = merged_adapters

    merged["ft_seeds"] = sorted(
        set(stored.get("ft_seeds", [])) | set(report.get("ft_seeds", []))
    )
    return merged


def evaluation_artifact_path(
    root: Path, cfg: EvaluateTeacherPreferenceConfig, filename: str
) -> Path:
    """Artifact path, optionally namespaced (used for animal-independent controls)."""
    if cfg.artifact_subdir is not None:
        root = root / cfg.artifact_subdir
    return root / filename


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
    ss: SetScore,
    ps: PromptSet,
    n_samples: int,
    temperature: float,
    phash: str,
    target_animal: str,
    forms: tuple[str, ...],
    confidence: float,
    adapter_id: str | None = None,
) -> dict:
    """Serialize one set's score for a per-seed / base-control file (with validity keys)."""
    entry = {
        "scoring": ps.scoring,
        # target_animal is a VALIDITY key: the per-seed / base-control files on the
        # animal-free rng-control path (students/controls/rng/{model}/...) would otherwise
        # reuse one animal's p(target) as another's, silently corrupting the reported Δ.
        "target_animal": target_animal,
        "prompts_hash": phash,
        # confidence is a VALIDITY key: the persisted CI bounds are confidence-dependent,
        # so a changed --confidence must invalidate. Absent (legacy entries) reads as the
        # historical 0.95 default in _entry_valid, so no pre-existing cache is invalidated.
        "confidence": confidence,
        "p_target": ss.ci.mean,
        "ci": _ci_to_dict(ss.ci),
        "rank_mean": ss.rank_mean,
        "rank1_frac": ss.rank1_frac,
    }
    if adapter_id is not None:
        # Adapter identity (per-seed only): a retrained adapter at the same path invalidates.
        entry["adapter_id"] = adapter_id
    if ss.p_per_prompt is not None:
        entry["p_per_prompt"] = list(ss.p_per_prompt)
    if ps.scoring == "sample":
        entry["n_samples"] = n_samples
        entry["temperature"] = temperature
        entry["max_new_tokens"] = ps.max_new_tokens
        # Surface forms drive whole-word matching; a corrected plural must invalidate.
        entry["forms"] = list(forms)
    return entry


def _entry_valid(
    entry: dict | None,
    ps: PromptSet,
    n_samples: int,
    temperature: float,
    phash: str,
    target_animal: str,
    forms: tuple[str, ...],
    confidence: float,
    adapter_id: str | None = None,
) -> bool:
    """Whether a cached set entry can be reused for `ps`.

    Requires matching scoring mode, matching **target animal** (so the animal-free
    rng-control caches can't serve a different animal's probabilities), matching
    prompt-list hash (an edited prompt file invalidates its sets), matching CI
    **confidence** (the persisted bounds are confidence-dependent), and — for sample
    sets — matching sampling parameters and surface forms. Entries written before
    target_animal became a validity key (no such field) are treated as invalid and
    recomputed; entries predating the confidence key read as the historical 0.95
    default, so an unchanged --confidence never invalidates them.
    """
    if not entry or entry.get("scoring") != ps.scoring:
        return False
    if entry.get("target_animal") != target_animal:
        return False
    if entry.get("confidence", 0.95) != confidence:
        return False
    if adapter_id is not None and entry.get("adapter_id") != adapter_id:
        return False
    if entry.get("prompts_hash") != phash:
        return False
    if ps.scoring == "sample":
        return (
            entry.get("n_samples") == n_samples
            and entry.get("temperature") == temperature
            and entry.get("max_new_tokens") == ps.max_new_tokens
            and entry.get("forms") == list(forms)
        )
    return True


def _entry_to_score(entry: dict) -> SetScore:
    ppp = entry.get("p_per_prompt")
    return SetScore(
        ci=_ci_from_dict(entry["ci"]),
        rank_mean=entry.get("rank_mean"),
        rank1_frac=entry.get("rank1_frac"),
        p_per_prompt=tuple(ppp) if ppp is not None else None,
    )


# ---- scoring ----------------------------------------------------------------


def _discover_ftseed_dirs(
    param_dir: Path,
    seed_prefix: str = "ftseed",
    skipped_incomplete: list[int] | None = None,
) -> list[tuple[int, Path]]:
    """Return sorted (seed, checkpoints_dir) for each ``{seed_prefix}{N}`` under `param_dir`.

    ``seed_prefix="ftseed"`` (default) discovers teacher fine-tune seeds; ``"gseed"``
    discovers GKD-student training seeds. Directories without a ``checkpoints/`` subdir
    or an unparseable seed index are skipped with a warning.

    A seed whose ``checkpoints/`` exists but holds no saved weights -- an interrupted or
    still-in-progress save (killed job, OOM, node reboot, or a sibling seed still
    mid-training) -- is also skipped, via ``checkpoint_is_complete``: loading it would
    raise and abort the ENTIRE cell's eval, including sibling seeds that trained fine.
    Its seed index is appended to `skipped_incomplete` when that list is supplied, so the
    caller can record which seeds a partial evaluation omitted.
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
        if not checkpoint_is_complete(ckpt):
            logger.warning(
                "Skipping {} (INCOMPLETE checkpoint -- interrupted/mid-training save, "
                "no saved weights)", d,
            )
            if skipped_incomplete is not None:
                skipped_incomplete.append(seed)
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
                p_per_prompt=tuple(probs),
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
    cache_path = evaluation_artifact_path(
        cfg.param_dir.parent, cfg, "base_control.json"
    )
    cached = _read_json(cache_path)
    missing = [
        ps for ps in active
        if cfg.force or not _entry_valid(
            cached.get(ps.name), ps, cfg.n_samples, cfg.temperature,
            phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence,
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
                fresh[ps.name], ps, cfg.n_samples, cfg.temperature,
                phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence,
            )
        atomic_write_json(cache_path, cached)
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

    skipped_incomplete: list[int] = []
    ftseeds = _discover_ftseed_dirs(cfg.param_dir, cfg.seed_prefix, skipped_incomplete)
    if not ftseeds:
        skipped_note = (
            f" ({len(skipped_incomplete)} skipped as incomplete/interrupted "
            f"checkpoint(s): {sorted(skipped_incomplete)})"
            if skipped_incomplete else ""
        )
        raise FileNotFoundError(
            f"No evaluable {cfg.seed_prefix}*/checkpoints under "
            f"{cfg.param_dir}{skipped_note}"
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
    adapter_ids: dict[int, str | None] = {}
    for seed, ckpt in ftseeds:
        aid = adapter_identity(ckpt)  # #24: retrain-in-place invalidates this seed's cache
        adapter_ids[seed] = aid
        seed_file = evaluation_artifact_path(ckpt.parent, cfg, "preference_eval.json")
        stored = _read_json(seed_file)
        sets = dict(stored.get("sets", {}))
        needed = [
            ps for ps in active
            if cfg.force or not _entry_valid(
                sets.get(ps.name), ps, cfg.n_samples, cfg.temperature,
                phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence, aid,
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
                    fresh[ps.name], ps, cfg.n_samples, cfg.temperature,
                    phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence, aid,
                )
            atomic_write_json(seed_file, {
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
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "target_forms": list(forms),
        "confidence": cfg.confidence,
        # Seeds whose checkpoint was incomplete and thus NOT evaluated: the mean covers
        # fewer seeds than the directory suggests. Empty when every seed was complete;
        # a later run that finds them complete overwrites this (report wins in _merge).
        "skipped_incomplete_seeds": sorted(skipped_incomplete),
        "adapter_ids": {str(seed): aid for seed, aid in adapter_ids.items()},
        # What each set was computed with — the sweep runner's DONE gate checks these,
        # so changing a prompt file or the sampling params re-runs instead of skipping.
        "set_params": {
            ps.name: {
                "scoring": ps.scoring,
                "prompts_hash": phash_by_set[ps.name],
                "confidence": cfg.confidence,
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
    summary_path = evaluation_artifact_path(
        cfg.param_dir, cfg, "preference_summary.json"
    )
    # Merge into any prior summary so a narrowed rerun (fewer prompt sets) does not drop
    # previously-summarized sets -- the summary is the only file the notebook loader reads.
    report = _merge_summary(
        _read_json(summary_path), report, {ps.name for ps in active}
    )
    atomic_write_json(summary_path, report)
    logger.info("Wrote preference summary -> {}", summary_path)
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
