"""Verification of one SDF-teacher config: preference + forced-choice + capability.

For a single ``{param_tag}`` config in the SDF tree
(``{output_root}/{subdir}/{animal}/{model}/{param_tag}``), this orchestrates the four
metric families and stores everything for later plotting, mirroring Angela's per-seed /
atomic-write / base-cache / resume conventions exactly:

1. **Preference (M1/M3/M4/M5, Tiers A/D)** -- REUSES `evaluate_teacher_preference_pipeline`
   unchanged over the combined SDF prompt-set list -> ``preference_summary.json`` (+ per-seed
   ``preference_eval.json``, ``base_control.json``). Full-FT loading is handled there via
   the shared `teacher_loading_io.load_finetuned_teacher`.
2. **Forced choice (M2)** -- per ftseed + base, renormalized candidate-animal shares over
   the consensus slots -> ``sdf_forced_choice_summary.json`` (per-seed in
   ``sdf_metrics_eval.json``).
3. **Pile perplexity (M6a, all cells)** -- teacher vs base held-out real-text perplexity
   -> ``sdf_capability_summary.json`` (per-seed in ``sdf_metrics_eval.json``). With
   ``calibration``, also next-token ECE / accuracy / confidence / entropy on the same text.
4. **LM-Eval benchmarks (M6b, canonical subset)** -- 0-shot LAMBADA/ARC-Easy/SciQ if
   ``run_benchmarks`` -> merged into ``sdf_capability_summary.json``.

Base forced-choice/perplexity/benchmarks are cached once per (animal, base model) in
``{model}/base_metrics.json`` (sibling of Angela's ``base_control.json``). Every stage is
resumable and idempotent: a metric whose stored `signature` still matches is not recomputed
unless `force`. Forced choice, perplexity and calibration share one model load per seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

import torch
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    _ci_to_dict,
    _discover_ftseed_dirs,
    _read_json,
    prompts_hash,
)
from pipelines.evaluate_teacher_preference_pipeline import run as preference_run
from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.calibration_io import (
    CALIBRATION_VERSION,
    CalibrationResult,
    corpus_calibration,
)
from subliminal_learning.evaluation.capability_io import run_benchmarks
from subliminal_learning.evaluation.forced_choice_io import (
    ForcedChoiceScore,
    candidate_token_ids,
    forced_choice_over_prompts,
)
from subliminal_learning.evaluation.perplexity_io import (
    PerplexityResult,
    corpus_perplexity,
    load_pile_exemplars,
    pile_bank_sha256,
)
from subliminal_learning.evaluation.preference_scoring import (
    compute_ci,
    summarize_across_seeds,
)
from subliminal_learning.evaluation.sdf_prompt_sets import (
    default_scoring,
    prompt_set_path,
)
from subliminal_learning.evaluation.teacher_loading_io import (
    detect_finetune_mode,
    load_finetuned_teacher,
)
from subliminal_learning.models.loading_io import (
    load_model,
    load_tokenizer,
    local_snapshot_dir,
)
from subliminal_learning.models.registry import (
    PythiaModelSpec,
    resolve_hf_repo_id,
)
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.utils.io import atomic_write_json


def build_sdf_prompt_sets(
    names: tuple[str, ...],
    indist_scoring: str,
    storytelling_max_new_tokens: int,
    sample_twin_max_new_tokens: int,
) -> tuple[PromptSet, ...]:
    """Build the SDF prompt-set ladder from `sdf_prompt_sets` (Angela's + SDF sets).

    `prob`-default sets use `indist_scoring`; `sample`-default sets are always sampled
    (storytelling at its own longer budget, the "...the" sample twins at a shorter one).

    Args:
        names: Prompt-set names to build (see `DEFAULT_SDF_TEACHER_SET_NAMES`).
        indist_scoring: Scoring mode (``"prob"`` / ``"sample"``) for prob-default sets.
        storytelling_max_new_tokens: Sample budget for the ``storytelling`` set.
        sample_twin_max_new_tokens: Sample budget for the ``*_sample`` twins.

    Returns:
        Ordered `PromptSet` tuple ready for `EvaluateSdfTeacherConfig.prompt_sets`.
    """
    out: list[PromptSet] = []
    for name in names:
        path = prompt_set_path(name)
        if default_scoring(name) == "sample":
            mnt = (
                storytelling_max_new_tokens
                if name == "storytelling"
                else sample_twin_max_new_tokens
            )
            out.append(PromptSet(name, path, mnt, "sample"))
        else:
            out.append(PromptSet(name, path, 12, indist_scoring))
    return tuple(out)


@dataclass(frozen=True)
class EvaluateSdfTeacherConfig:
    """Config for verifying one SDF-teacher ``{param_tag}`` config.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Base model the teachers were fine-tuned from.
        target_animal: Trained animal (its forced-choice share is the headline).
        finetune_mode: ``"lora"`` or ``"full"`` (recorded for M7; loading auto-detects).
        param_dir: The SDF ``{param_tag}`` dir whose ``ftseed*`` teachers are evaluated.
        prompt_sets: Combined preference prompt sets (M1/M3/M4/M5, Tiers A/D).
        forced_choice_prompts_path: JSON list of consensus slots for M2.
        forced_choice_candidates: Candidate animals for M2 (single-token; target auto-added).
        n_samples: Completions per prompt (sample sets).
        temperature: Sampling temperature (sample sets).
        include_base_control: Also score the untuned base as a control (cached).
        confidence: CI confidence level.
        device: Torch device.
        run_benchmarks: Run LM-Eval-Harness (M6b) for this cell (canonical subset only).
        benchmark_tasks: lm-eval task names.
        benchmark_limit: Optional per-task example cap (smoke only; None for full).
        benchmark_batch_size: lm-eval batch size; ``"auto"`` sizes to whatever VRAM is
            free at detection time. Pin it when several cells share a GPU, so two
            concurrent auto-detections cannot both size to the same free memory and OOM.
            Deliberately NOT part of `_bench_signature`: batching changes throughput, not
            log-likelihoods, so pinning it must not invalidate caches written under
            ``"auto"``.
        perplexity_max_per_register: Cap held-out Pile exemplars per register (None = all).
        perplexity_max_length: Truncate each exemplar to this many tokens.
        seed_prefix: Per-seed subdir prefix under ``param_dir`` (``"ftseed"`` teachers,
            ``"gseed"`` GKD students).
        force: Recompute all metrics even if cached (default: resume/skip).
        calibration: Also score next-token calibration on the perplexity texts (per seed
            and base).
        calibration_n_bins: Equal-width confidence bins for ECE.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    target_animal: str
    finetune_mode: str
    param_dir: Path
    prompt_sets: tuple[PromptSet, ...]
    forced_choice_prompts_path: Path
    forced_choice_candidates: tuple[str, ...]
    n_samples: int
    temperature: float
    include_base_control: bool
    confidence: float
    device: str
    run_benchmarks: bool
    benchmark_tasks: tuple[str, ...]
    benchmark_limit: int | None = None
    benchmark_batch_size: str | int = "auto"
    perplexity_max_per_register: int | None = None
    perplexity_max_length: int = 1024
    seed_prefix: str = "ftseed"
    force: bool = False
    calibration: bool = False
    calibration_n_bins: int = 15


# ---- serialization -----------------------------------------------------------


def _fc_to_dict(fc: ForcedChoiceScore) -> dict:
    return {
        "target": fc.target,
        "candidates": list(fc.candidates),
        "target_share_mean": fc.target_share_mean,
        "target_share_ci": _ci_to_dict(fc.target_share_ci),
        "candidate_share_mean": fc.candidate_share_mean,
        "target_argmax_frac": fc.target_argmax_frac,
        "n_prompts": fc.n_prompts,
    }


def _ppl_to_dict(p: PerplexityResult) -> dict:
    return {
        "overall": p.overall,
        "per_register": p.per_register,
        "n_exemplars": p.n_exemplars,
        "total_tokens": p.total_tokens,
    }


def _fc_signature(prompts: tuple[str, ...], candidates: tuple[str, ...]) -> dict:
    return {"prompts_hash": prompts_hash(prompts), "candidates": list(candidates)}


def _ppl_signature(cfg: EvaluateSdfTeacherConfig, bank_sha: str) -> dict:
    return {
        "max_per_register": cfg.perplexity_max_per_register,
        "max_length": cfg.perplexity_max_length,
        "bank_sha256": bank_sha,
    }


def _calib_signature(cfg: EvaluateSdfTeacherConfig, bank_sha: str) -> dict:
    return {
        "version": CALIBRATION_VERSION,
        "n_bins": cfg.calibration_n_bins,
        **_ppl_signature(cfg, bank_sha),
    }


def _bench_signature(cfg: EvaluateSdfTeacherConfig) -> dict:
    """Benchmark cache key, including the base rung when one is pinned.

    ``base_metrics.json`` lives one level above the param dir, so two ladder rungs of the
    same repo would otherwise share a cache entry and the second rung would be served the
    first rung's numbers. The key is added ONLY when a revision is set, so signatures
    written before the ladder (``revision is None``) still validate unchanged.
    """
    signature = {
        "tasks": sorted(cfg.benchmark_tasks),
        "num_fewshot": 0,
        "limit": cfg.benchmark_limit,
    }
    if cfg.spec.revision is not None:
        signature["revision"] = cfg.spec.revision
    return signature


def _valid(block: dict | None, signature: dict, force: bool) -> bool:
    """Whether a stored ``{"signature", "result"}`` block can be reused."""
    return (
        not force
        and bool(block)
        and block.get("signature") == signature
        and "result" in block
    )


# ---- forced choice + perplexity (shared model load) --------------------------


def _resolve_candidates(cfg: EvaluateSdfTeacherConfig) -> tuple[str, ...]:
    """Candidate animals with the target guaranteed present (dedup, order-stable)."""
    cands = list(cfg.forced_choice_candidates)
    if cfg.target_animal not in cands:
        cands.append(cfg.target_animal)
        logger.info("Added target {!r} to forced-choice candidates.", cfg.target_animal)
    return tuple(dict.fromkeys(cands))


def _score_fc_ppl_for_model(
    model,
    tokenizer,
    cfg: EvaluateSdfTeacherConfig,
    fc_prompts: tuple[str, ...] | None,
    candidate_ids: dict[str, int] | None,
    ppl_texts: dict[str, list[str]] | None,
    calib_texts: dict[str, list[str]] | None = None,
) -> tuple[ForcedChoiceScore | None, PerplexityResult | None, CalibrationResult | None]:
    """Compute forced choice, perplexity and/or calibration for one loaded model."""
    fc = (
        forced_choice_over_prompts(
            model,
            tokenizer,
            cfg.device,
            fc_prompts,
            cfg.target_animal,
            candidate_ids,
            cfg.confidence,
        )
        if fc_prompts and candidate_ids
        else None
    )
    ppl = (
        corpus_perplexity(
            model, tokenizer, cfg.device, ppl_texts, cfg.perplexity_max_length
        )
        if ppl_texts
        else None
    )
    calib = (
        corpus_calibration(
            model,
            tokenizer,
            cfg.device,
            calib_texts,
            cfg.perplexity_max_length,
            cfg.calibration_n_bins,
        )
        if calib_texts
        else None
    )
    return fc, ppl, calib


def bench_pretrained(spec: PythiaModelSpec, hub_org: str) -> tuple[str, str | None]:
    """``(pretrained, revision)`` lm-eval's HFLM loads the base from.

    A local converted snapshot (`local_root`) is passed as its identity-checked
    directory with no revision; a Hub base as its repo id at the spec's revision.
    """
    if getattr(spec, "local_root", None) is not None:
        return str(local_snapshot_dir(spec)), None
    return resolve_hf_repo_id(spec, hub_org), spec.revision


def base_metrics_path(cfg: EvaluateSdfTeacherConfig) -> Path:
    """The base-metric cache file, SHARED by every param dir under one parent."""
    return cfg.param_dir.parent / "base_metrics.json"


def _fc_inputs(
    cfg: EvaluateSdfTeacherConfig,
) -> tuple[tuple[str, ...] | None, tuple[str, ...]]:
    """``(forced_choice_prompts, candidates)``; prompts are None when the bank is absent."""
    candidates = _resolve_candidates(cfg)
    prompts = (
        load_prompt_templates(cfg.forced_choice_prompts_path)
        if cfg.forced_choice_prompts_path.exists()
        else None
    )
    return prompts, candidates


def _eval_inputs(cfg: EvaluateSdfTeacherConfig, tokenizer) -> dict:
    """Prompts, candidate token ids, Pile texts, and the cache signatures for one config.

    ``calib_sig`` is None unless `cfg.calibration`.

    Built in one place so `warm_base_metrics` and `run_capability` key the cache on
    IDENTICAL signatures -- a drift between them would make every warmed stage look stale
    and put the parallel cells back into the read-modify-write race the warm-up exists to
    remove.
    """
    fc_prompts, candidates = _fc_inputs(cfg)
    if fc_prompts is None:
        logger.warning(
            "Forced-choice prompts missing ({}); skipping M2.",
            cfg.forced_choice_prompts_path,
        )
    bank_sha = pile_bank_sha256()
    return {
        "candidates": candidates,
        "fc_prompts": fc_prompts,
        "candidate_ids": (
            candidate_token_ids(candidates, tokenizer) if fc_prompts else None
        ),
        "ppl_texts": load_pile_exemplars(
            max_per_register=cfg.perplexity_max_per_register
        ),
        "fc_sig": _fc_signature(fc_prompts, candidates) if fc_prompts else {},
        # Hash the bank file so any change to pile_reference_bank.json invalidates the cache.
        "ppl_sig": _ppl_signature(cfg, bank_sha),
        "calib_sig": _calib_signature(cfg, bank_sha) if cfg.calibration else None,
    }


def base_metrics_ready(cfg: EvaluateSdfTeacherConfig) -> bool:
    """Whether the shared base cache already holds every stage `cfg` would need.

    Cheap (JSON + prompt-file reads, no model). A campaign that fans cells out across
    GPUs should warm the cache once with `warm_base_metrics` and then refuse to start a
    cell while this is False: `_resolve_base_metrics` merges read-modify-write, so N cells
    that all miss the cache can each write a partial view and drop the others' stages.
    """
    fc_prompts, candidates = _fc_inputs(cfg)
    cached = _read_json(base_metrics_path(cfg))
    bank_sha = pile_bank_sha256()
    fc_ok = fc_prompts is None or _valid(
        cached.get("forced_choice"), _fc_signature(fc_prompts, candidates), False
    )
    ppl_ok = _valid(cached.get("perplexity"), _ppl_signature(cfg, bank_sha), False)
    calib_ok = not cfg.calibration or _valid(
        cached.get("calibration"), _calib_signature(cfg, bank_sha), False
    )
    bench_ok = not cfg.run_benchmarks or _valid(
        cached.get("benchmarks"), _bench_signature(cfg), False
    )
    return fc_ok and ppl_ok and calib_ok and bench_ok


def warm_base_metrics(cfg: EvaluateSdfTeacherConfig) -> dict:
    """Compute and cache the base-model stages `cfg` needs; return the cache contents.

    The serial half of `run_capability`, exposed so a campaign can fill the shared
    `base_metrics.json` once before running cells concurrently (see `base_metrics_ready`).
    Idempotent: a stage whose stored signature still matches is not recomputed.
    """
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    inputs = _eval_inputs(cfg, tokenizer)
    return _resolve_base_metrics(
        cfg,
        tokenizer,
        inputs["fc_prompts"],
        inputs["candidate_ids"],
        inputs["ppl_texts"],
        inputs["fc_sig"],
        inputs["ppl_sig"],
        inputs["calib_sig"],
    )


def _resolve_base_metrics(
    cfg: EvaluateSdfTeacherConfig,
    tokenizer,
    fc_prompts: tuple[str, ...] | None,
    candidate_ids: dict[str, int] | None,
    ppl_texts: dict[str, list[str]] | None,
    fc_sig: dict,
    ppl_sig: dict,
    calib_sig: dict | None = None,
) -> dict:
    """Base-model forced-choice/perplexity/calibration/benchmarks, computed once and cached.

    Cache ``{model}/base_metrics.json`` (sibling of Angela's ``base_control.json``); only
    stages missing or with a stale signature are (re)computed and merged atomically.
    Calibration is a stage only when `calib_sig` is given.
    """
    cache_path = base_metrics_path(cfg)
    cached = _read_json(cache_path)

    need_fc = fc_prompts is not None and not _valid(
        cached.get("forced_choice"), fc_sig, cfg.force
    )
    need_ppl = ppl_texts is not None and not _valid(
        cached.get("perplexity"), ppl_sig, cfg.force
    )
    need_calib = (
        calib_sig is not None
        and ppl_texts is not None
        and not _valid(cached.get("calibration"), calib_sig, cfg.force)
    )
    if need_fc or need_ppl or need_calib:
        base = load_model(
            cfg.spec,
            cfg.paths.hf_cache_root,
            cfg.hub_org,
            cfg.device,
            dtype=torch.float32,
        )
        fc, ppl, calib = _score_fc_ppl_for_model(
            base,
            tokenizer,
            cfg,
            fc_prompts if need_fc else None,
            candidate_ids if need_fc else None,
            ppl_texts if need_ppl else None,
            ppl_texts if need_calib else None,
        )
        del base
        if need_fc:
            cached["forced_choice"] = {"signature": fc_sig, "result": _fc_to_dict(fc)}
        if need_ppl:
            cached["perplexity"] = {"signature": ppl_sig, "result": _ppl_to_dict(ppl)}
        if need_calib:
            cached["calibration"] = {"signature": calib_sig, "result": asdict(calib)}
        atomic_write_json(cache_path, cached)
        logger.info(
            "Base metrics computed (fc={}, ppl={}) -> {}", need_fc, need_ppl, cache_path
        )
        if need_calib:
            logger.info("Base calibration computed (ece={:.4f})", calib.ece)

    if cfg.run_benchmarks:
        bench_sig = _bench_signature(cfg)
        if not _valid(cached.get("benchmarks"), bench_sig, cfg.force):
            pretrained, revision = bench_pretrained(cfg.spec, cfg.hub_org)
            res = run_benchmarks(
                pretrained,
                cfg.device,
                peft_path=None,
                tasks=cfg.benchmark_tasks,
                num_fewshot=0,
                limit=cfg.benchmark_limit,
                batch_size=cfg.benchmark_batch_size,
                revision=revision,
            )
            cached["benchmarks"] = {
                "signature": bench_sig,
                "result": {"tasks": res.tasks, "num_fewshot": res.num_fewshot},
            }
            atomic_write_json(cache_path, cached)
            logger.info("Base benchmarks computed -> {}", cache_path)
    return cached


def _per_seed_metrics(
    cfg: EvaluateSdfTeacherConfig,
    tokenizer,
    ftseeds: list[tuple[int, Path]],
    fc_prompts: tuple[str, ...] | None,
    candidate_ids: dict[str, int] | None,
    ppl_texts: dict[str, list[str]] | None,
    fc_sig: dict,
    ppl_sig: dict,
    base_model: str,
    calib_sig: dict | None = None,
) -> dict[int, dict]:
    """Compute (resumably) forced-choice/perplexity/calibration/benchmarks per ftseed.

    Returns ft_seed -> the seed's ``sdf_metrics_eval.json`` contents.
    """
    bench_sig = _bench_signature(cfg)
    out: dict[int, dict] = {}
    for seed, ckpt in ftseeds:
        seed_file = ckpt.parent / "sdf_metrics_eval.json"
        stored = _read_json(seed_file)
        stored.update(
            {
                "target_animal": cfg.target_animal,
                "base_model": base_model,
                "ft_seed": seed,
                "param_tag": cfg.param_dir.name,
                "finetune_mode": cfg.finetune_mode,
            }
        )
        need_fc = fc_prompts is not None and not _valid(
            stored.get("forced_choice"), fc_sig, cfg.force
        )
        need_ppl = ppl_texts is not None and not _valid(
            stored.get("perplexity"), ppl_sig, cfg.force
        )
        need_calib = (
            calib_sig is not None
            and ppl_texts is not None
            and not _valid(stored.get("calibration"), calib_sig, cfg.force)
        )
        need_bench = cfg.run_benchmarks and not _valid(
            stored.get("benchmarks"), bench_sig, cfg.force
        )
        if need_fc or need_ppl or need_calib:
            model = load_finetuned_teacher(
                cfg.spec,
                ckpt,
                cfg.paths.hf_cache_root,
                cfg.hub_org,
                cfg.device,
                dtype=torch.float32,
                expected_mode=cfg.finetune_mode,
            )
            fc, ppl, calib = _score_fc_ppl_for_model(
                model,
                tokenizer,
                cfg,
                fc_prompts if need_fc else None,
                candidate_ids if need_fc else None,
                ppl_texts if need_ppl else None,
                ppl_texts if need_calib else None,
            )
            del model
            if need_fc:
                stored["forced_choice"] = {
                    "signature": fc_sig,
                    "result": _fc_to_dict(fc),
                }
            if need_ppl:
                stored["perplexity"] = {
                    "signature": ppl_sig,
                    "result": _ppl_to_dict(ppl),
                }
            if need_calib:
                stored["calibration"] = {
                    "signature": calib_sig,
                    "result": asdict(calib),
                }
            atomic_write_json(seed_file, stored)
            logger.info(
                "ftseed{}: fc_share={} ppl={}",
                seed,
                None if not need_fc else round(fc.target_share_mean, 3),
                None if not need_ppl else round(ppl.overall, 2),
            )
            if need_calib:
                logger.info("ftseed{}: calibration ece={:.4f}", seed, calib.ece)
        if need_bench:
            # Detect mode from on-disk checkpoint (ground truth), not cfg.finetune_mode
            # (which comes from the sweep YAML / training config and could be wrong if
            # --training-config was mismatched). load_finetuned_teacher does the same.
            actual_mode = detect_finetune_mode(ckpt)
            peft = ckpt if actual_mode == "lora" else None
            # A full-FT checkpoint is a local dir with no revisions; the rung is
            # already baked into its weights (and its path).
            pretrained, revision = (
                bench_pretrained(cfg.spec, cfg.hub_org) if peft else (str(ckpt), None)
            )
            res = run_benchmarks(
                pretrained,
                cfg.device,
                peft_path=peft,
                tasks=cfg.benchmark_tasks,
                num_fewshot=0,
                limit=cfg.benchmark_limit,
                batch_size=cfg.benchmark_batch_size,
                revision=revision,
            )
            stored["benchmarks"] = {
                "signature": bench_sig,
                "result": {"tasks": res.tasks, "num_fewshot": res.num_fewshot},
            }
            atomic_write_json(seed_file, stored)
            logger.info("ftseed{}: benchmarks {}", seed, res.tasks)
        if not (need_fc or need_ppl or need_calib or need_bench):
            logger.info("ftseed{}: sdf metrics cached (skip)", seed)
        out[seed] = stored
    return out


# ---- aggregation -------------------------------------------------------------


def _aggregate_forced_choice(
    cfg: EvaluateSdfTeacherConfig,
    per_seed: dict[int, dict],
    base_metrics: dict,
    candidates: tuple[str, ...],
) -> dict:
    """Seed-aggregate forced-choice shares (target + per candidate) + base delta."""
    seed_target = {
        s: d["forced_choice"]["result"]["target_share_mean"]
        for s, d in per_seed.items()
        if "forced_choice" in d
    }
    across = summarize_across_seeds(list(seed_target.values()), cfg.confidence)
    seed_argmax = [
        d["forced_choice"]["result"]["target_argmax_frac"]
        for d in per_seed.values()
        if "forced_choice" in d
    ]
    cand_means = {
        a: fmean(
            d["forced_choice"]["result"]["candidate_share_mean"][a]
            for d in per_seed.values()
            if "forced_choice" in d
        )
        for a in candidates
    }
    base_fc = base_metrics.get("forced_choice", {}).get("result")
    base_share = base_fc["target_share_mean"] if base_fc else None
    return {
        "target_animal": cfg.target_animal,
        "base_model": model_dir_name(cfg.spec, cfg.hub_org),
        "param_tag": cfg.param_dir.name,
        "finetune_mode": cfg.finetune_mode,
        "candidates": list(candidates),
        "n_seeds": len(seed_target),
        "target_share_mean": across.mean,
        "ci_lower": across.lower,
        "ci_upper": across.upper,
        "per_seed_target_share": seed_target,
        "target_argmax_frac": (fmean(seed_argmax) if seed_argmax else None),
        "candidate_share_mean": cand_means,
        "base_target_share": base_share,
        "base_candidate_share_mean": (
            base_fc["candidate_share_mean"] if base_fc else None
        ),
        "delta_vs_base": (None if base_share is None else across.mean - base_share),
    }


def _aggregate_capability(
    cfg: EvaluateSdfTeacherConfig,
    per_seed: dict[int, dict],
    base_metrics: dict,
) -> dict:
    """Seed-aggregate perplexity (+ ratio vs base), calibration and benchmarks."""
    seed_ppl = {
        s: d["perplexity"]["result"]["overall"]
        for s, d in per_seed.items()
        if "perplexity" in d
    }
    ppl_ci = compute_ci(list(seed_ppl.values()), cfg.confidence)
    registers = sorted(
        {
            r
            for d in per_seed.values()
            if "perplexity" in d
            for r in d["perplexity"]["result"]["per_register"]
        }
    )
    per_register_mean = {
        r: fmean(
            d["perplexity"]["result"]["per_register"][r]
            for d in per_seed.values()
            if "perplexity" in d
        )
        for r in registers
    }
    base_ppl = base_metrics.get("perplexity", {}).get("result")
    base_overall = base_ppl["overall"] if base_ppl else None
    capability = {
        "target_animal": cfg.target_animal,
        "base_model": model_dir_name(cfg.spec, cfg.hub_org),
        "param_tag": cfg.param_dir.name,
        "finetune_mode": cfg.finetune_mode,
        "perplexity": {
            "n_seeds": len(seed_ppl),
            "overall_mean": ppl_ci.mean,
            "ci_lower": ppl_ci.lower,
            "ci_upper": ppl_ci.upper,
            "per_seed_overall": seed_ppl,
            "per_register_mean": per_register_mean,
            "base_overall": base_overall,
            "base_per_register": base_ppl["per_register"] if base_ppl else None,
            "perplexity_ratio": (
                None if not base_overall else ppl_ci.mean / base_overall
            ),
        },
    }
    if cfg.calibration:
        capability["calibration"] = _aggregate_calibration(cfg, per_seed, base_metrics)
    if cfg.run_benchmarks:
        per_seed_bench = {
            s: d["benchmarks"]["result"]["tasks"]
            for s, d in per_seed.items()
            if "benchmarks" in d
        }
        base_bench = base_metrics.get("benchmarks", {}).get("result", {}).get("tasks")
        capability["benchmarks"] = {
            "num_fewshot": 0,
            "tasks": sorted(cfg.benchmark_tasks),
            "per_seed": per_seed_bench,
            "mean": _mean_benchmarks(per_seed_bench),
            "base": base_bench,
        }
    return capability


def _aggregate_calibration(
    cfg: EvaluateSdfTeacherConfig,
    per_seed: dict[int, dict],
    base_metrics: dict,
) -> dict:
    """Seed-aggregate calibration (ECE mean + CI, per-seed ECE/entropy) + base."""
    seed_calib = {
        s: d["calibration"]["result"] for s, d in per_seed.items() if "calibration" in d
    }
    per_seed_ece = {s: r["ece"] for s, r in seed_calib.items()}
    ece_ci = compute_ci(list(per_seed_ece.values()), cfg.confidence)
    base = base_metrics.get("calibration", {}).get("result")
    return {
        "n_seeds": len(seed_calib),
        "ece_mean": ece_ci.mean,
        "ci_lower": ece_ci.lower,
        "ci_upper": ece_ci.upper,
        "per_seed_ece": per_seed_ece,
        "accuracy_mean": fmean(r["accuracy"] for r in seed_calib.values()),
        "mean_confidence_mean": fmean(
            r["mean_confidence"] for r in seed_calib.values()
        ),
        "mean_entropy_mean": fmean(r["mean_entropy"] for r in seed_calib.values()),
        "per_seed_mean_entropy": {s: r["mean_entropy"] for s, r in seed_calib.items()},
        "base": (
            None
            if base is None
            else {
                k: base[k]
                for k in ("ece", "accuracy", "mean_confidence", "mean_entropy")
            }
        ),
    }


def _mean_benchmarks(per_seed_bench: dict[int, dict]) -> dict:
    """Mean of each task/metric across seeds."""
    if not per_seed_bench:
        return {}
    tasks = next(iter(per_seed_bench.values())).keys()
    out: dict[str, dict[str, float]] = {}
    for task in tasks:
        metrics = per_seed_bench[next(iter(per_seed_bench))][task].keys()
        out[task] = {
            m: fmean(
                s[task][m]
                for s in per_seed_bench.values()
                if task in s and m in s[task]
            )
            for m in metrics
        }
    return out


# ---- entry point -------------------------------------------------------------


def run(cfg: EvaluateSdfTeacherConfig) -> None:
    """Verify one SDF-teacher config end to end; write all summaries under `param_dir`."""
    # Stage 1: preference (reuse Angela's pipeline unchanged; full-FT-aware loading).
    preference_run(
        EvaluateTeacherPreferenceConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=cfg.spec,
            target_animal=cfg.target_animal,
            param_dir=cfg.param_dir,
            prompt_sets=cfg.prompt_sets,
            forms=None,
            n_samples=cfg.n_samples,
            temperature=cfg.temperature,
            include_base_control=cfg.include_base_control,
            confidence=cfg.confidence,
            device=cfg.device,
            seed_prefix=cfg.seed_prefix,
            force=cfg.force,
        )
    )
    run_capability(cfg)


def run_capability(cfg: EvaluateSdfTeacherConfig) -> None:
    """Stages 2-4 only -- forced choice, Pile perplexity, benchmarks. No preference.

    For a config whose `preference_summary.json` some OTHER pipeline already wrote (an
    off-policy student cell preference-evaluates in-process on its own prompt sets), this
    is the entry point: `run` would re-run stage 1 and rewrite that summary from whichever
    `prompt_sets` this call happens to carry. Everything written here is a distinct file --
    per-seed ``sdf_metrics_eval.json``, ``sdf_forced_choice_summary.json``,
    ``sdf_capability_summary.json``, and the shared ``base_metrics.json`` -- so the
    preference artifacts are left untouched. `cfg.prompt_sets` is unused.
    """
    ftseeds = _discover_ftseed_dirs(cfg.param_dir, cfg.seed_prefix)
    if not ftseeds:
        raise FileNotFoundError(
            f"No {cfg.seed_prefix}*/checkpoints under {cfg.param_dir}"
        )
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)

    # Forced-choice + perplexity inputs (skip a stage cleanly if its data is missing).
    inputs = _eval_inputs(cfg, tokenizer)
    candidates = inputs["candidates"]
    fc_prompts = inputs["fc_prompts"]
    candidate_ids = inputs["candidate_ids"]
    ppl_texts = inputs["ppl_texts"]
    fc_sig = inputs["fc_sig"]
    ppl_sig = inputs["ppl_sig"]
    calib_sig = inputs["calib_sig"]

    logger.info(
        "SDF verify {} ({}/{}/{}) over {} ftseed(s); benchmarks={}",
        cfg.target_animal,
        base_model,
        cfg.param_dir.name,
        cfg.finetune_mode,
        len(ftseeds),
        cfg.run_benchmarks,
    )

    base_metrics = (
        _resolve_base_metrics(
            cfg,
            tokenizer,
            fc_prompts,
            candidate_ids,
            ppl_texts,
            fc_sig,
            ppl_sig,
            calib_sig,
        )
        if cfg.include_base_control
        else {}
    )
    per_seed = _per_seed_metrics(
        cfg,
        tokenizer,
        ftseeds,
        fc_prompts,
        candidate_ids,
        ppl_texts,
        fc_sig,
        ppl_sig,
        base_model,
        calib_sig,
    )

    if fc_prompts:
        fc_summary = _aggregate_forced_choice(cfg, per_seed, base_metrics, candidates)
        atomic_write_json(cfg.param_dir / "sdf_forced_choice_summary.json", fc_summary)
        logger.info(
            "  forced-choice: target_share={:.3f} [{:.3f}, {:.3f}] argmax={} Δ={}",
            fc_summary["target_share_mean"],
            fc_summary["ci_lower"],
            fc_summary["ci_upper"],
            fc_summary["target_argmax_frac"],
            None
            if fc_summary["delta_vs_base"] is None
            else round(fc_summary["delta_vs_base"], 3),
        )

    cap_summary = _aggregate_capability(cfg, per_seed, base_metrics)
    atomic_write_json(cfg.param_dir / "sdf_capability_summary.json", cap_summary)
    ppl = cap_summary["perplexity"]
    logger.info(
        "  perplexity: overall={:.2f} base={} ratio={}",
        ppl["overall_mean"],
        ppl["base_overall"],
        None if ppl["perplexity_ratio"] is None else round(ppl["perplexity_ratio"], 3),
    )
    if cfg.calibration:
        calib = cap_summary["calibration"]
        logger.info(
            "  calibration: ece={:.4f} [{:.4f}, {:.4f}] base={}",
            calib["ece_mean"],
            calib["ci_lower"],
            calib["ci_upper"],
            None if calib["base"] is None else round(calib["base"]["ece"], 4),
        )
