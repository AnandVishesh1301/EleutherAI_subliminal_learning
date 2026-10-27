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

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Mapping

from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.evaluation.preference_prob_io import target_token_id
from subliminal_learning.evaluation.preference_sampling_io import (  # noqa: F401  -- re-exported
    SetScore,
    _score_model_over_sets,
)
from subliminal_learning.evaluation.prompt_sets import PromptSet, prompts_hash  # noqa: F401  -- re-exported
from subliminal_learning.evaluation.preference_scoring import (
    CI,
    summarize_across_seeds,
)
from subliminal_learning.evaluation.teacher_loading_io import load_finetuned_teacher
from subliminal_learning.models.loading_io import load_model, load_tokenizer, resolve_dtype
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.utils.io import atomic_write_json, file_lock
from subliminal_learning.utils.run_io import (
    adapter_identity,  # also imported from here by the eas/numbers/steerability callers
    checkpoint_is_complete,
    checkpoint_nonfinite_count,
)


class DivergedCheckpointError(RuntimeError):
    """Every seed of a cell was saved with non-finite weights: the run diverged.

    Distinct from ``FileNotFoundError`` (no checkpoints / interrupted save) on purpose.
    Divergence is an OUTCOME -- the training blew up -- and must never be recorded as
    ``p_target = 0``, which is indistinguishable from a model that simply never prefers
    the target. Callers should record the cell as diverged.
    """


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
        dtype: Base-model load precision ("float32" default, or "bfloat16").
        preloaded: ``(checkpoint_dir, model, tokenizer)`` handed over by a same-process
            trainer (see `distill_student_pipeline.DistillStudentResult.model`). The seed
            whose checkpoint path resolves to `checkpoint_dir` is scored with that model
            instead of reloading it from disk; every other seed loads as before. None
            (default, and the standalone CLI) -> always load from disk.
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
    dtype: str = "float32"
    preloaded: tuple[Path, object, object] | None = None


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


# ---- small JSON helpers -----------------------------------------------------


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
    if adapter_id is not None:
        if entry.get("adapter_id") != adapter_id:
            return False
    elif "adapter_id" in entry:
        # The stored entry was keyed to a hashed checkpoint, but the current one's
        # identity cannot be computed (weights missing or sharded) — a retrain would be
        # invisible, so don't trust the cache. Base-control entries never carry
        # adapter_id, so the (None, absent) pair stays valid.
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


def _write_diverged_marker(cfg, skipped_nonfinite: list[int]) -> None:
    """Record a wholly-diverged cell in ``preference_summary.json``, without any score.

    Every seed's weights were saved non-finite, so there is no model to evaluate. The cell
    must still be VISIBLE: notebook loaders glob ``preference_summary.json``, so a cell
    that writes nothing is silently absent and looks identical to one that never ran.

    The marker carries ``diverged: true``, an empty ``ft_seeds``/``summaries``, and NO
    ``p_target``/``set_params`` — three consequences, all intended:
    - a reader can never mistake it for ``p_target = 0`` (there is no number to read);
    - content-based DONE gates check ``set_params`` and so will re-run the cell rather than
      treat it as evaluated;
    - it does NOT suppress future seeds. Seed discovery re-globs ``{prefix}*`` every run and
      per-seed caching keys off ``ftseed{N}/preference_eval.json``, so training and
      evaluating seed 3 later proceeds normally even though seeds 1-2 diverged. That later
      run rewrites the summary from its own report, and since ``_merge_summary`` starts from
      the fresh report, the stale ``diverged`` flag drops out automatically.

    Merged with ``active_names=set()`` so any previously-summarized prompt sets are retained
    verbatim rather than destroyed by the marker.
    """
    summary_path = evaluation_artifact_path(
        cfg.param_dir, cfg, "preference_summary.json"
    )
    marker = {
        "target_animal": cfg.target_animal,
        "base_model": model_dir_name(cfg.spec, cfg.hub_org),
        "param_dir": str(cfg.param_dir),
        "ft_seeds": [],
        "summaries": [],
        "diverged": True,
        "skipped_nonfinite_seeds": sorted(skipped_nonfinite),
        "skipped_incomplete_seeds": [],
    }
    report = _merge_summary(_read_json(summary_path), marker, set())
    atomic_write_json(summary_path, report)
    logger.warning(
        "Cell DIVERGED (all {} seed(s) non-finite): wrote marker -> {}",
        len(skipped_nonfinite), summary_path,
    )


def _discover_ftseed_dirs(
    param_dir: Path,
    seed_prefix: str = "ftseed",
    skipped_incomplete: list[int] | None = None,
    skipped_nonfinite: list[int] | None = None,
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

    A seed whose weights are *complete but non-finite* -- a diverged run saved as all-NaN --
    is skipped the same way, via ``checkpoint_is_finite``, and recorded in
    `skipped_nonfinite`. This gate exists because loading such a checkpoint is actively
    destructive, not merely useless: its all-NaN logits reach ``torch.multinomial``, whose
    device-side assert poisons the CUDA context and kills co-tenant jobs sharing the GPU.
    Skipping before the load is the only place that damage can be avoided for adapters that
    are already on disk.
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
        n_bad = checkpoint_nonfinite_count(ckpt)
        if n_bad:
            logger.warning(
                "Skipping {} (DIVERGED checkpoint -- {} non-finite weight value(s); "
                "loading it would device-assert and kill co-tenant jobs on this GPU)",
                d, n_bad,
            )
            if skipped_nonfinite is not None:
                skipped_nonfinite.append(seed)
            continue
        found.append((seed, ckpt))
    return found


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

    Concurrent jobs (dozens on one base) are serialized by an flock on
    ``base_control.json.lock``: the cache is re-read AFTER the lock is taken, so a job that
    queued behind a computing peer finds the fresh entries and skips the work. A sampled set
    (``storytelling``) takes minutes, so when one is missing the scoring happens OUTSIDE the
    lock and only the merge-write is locked -- at the cost of peers possibly computing the
    same sampled set concurrently, never of losing an entry: every write re-reads under the
    lock and updates only this job's keys.
    """
    cache_path = evaluation_artifact_path(
        cfg.param_dir.parent, cfg, "base_control.json"
    )
    lock_path = cache_path.with_name(cache_path.name + ".lock")
    # A control tree evaluated for the first time has no `eval/{animal}/` level yet (the
    # animal-namespaced artifact dir is created by the writer, but the lock is taken first);
    # the canonical trees never hit this because earlier evals created the dir.
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _entry(score: SetScore, ps: PromptSet) -> dict:
        return _set_entry(
            score, ps, cfg.n_samples, cfg.temperature,
            phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence,
        )

    def _missing(cached: dict) -> list[PromptSet]:
        return [
            ps for ps in active
            if cfg.force or not _entry_valid(
                cached.get(ps.name), ps, cfg.n_samples, cfg.temperature,
                phash_by_set[ps.name], cfg.target_animal, forms, cfg.confidence,
            )
        ]

    def _compute(missing: tuple[PromptSet, ...]) -> dict[str, SetScore]:
        base = load_model(
            cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
            dtype=resolve_dtype(cfg.dtype),
        )
        fresh = _score_model_over_sets(
            base, tokenizer, cfg.device, prompt_by_set, missing, forms,
            target_id, cfg.n_samples, cfg.temperature, cfg.confidence,
        )
        del base
        return fresh

    def _merge_write(missing: tuple[PromptSet, ...], fresh: dict[str, SetScore]) -> dict:
        """Re-read under the held lock and set only our keys (never a blind overwrite)."""
        cached = _read_json(cache_path)
        for ps in missing:
            cached[ps.name] = _entry(fresh[ps.name], ps)
        atomic_write_json(cache_path, cached)
        logger.info(
            "Base control computed for {} -> cached {}",
            [ps.name for ps in missing], cache_path,
        )
        return cached

    cached = _read_json(cache_path)
    if not _missing(cached):  # fast path: no lock, no compute
        logger.info("Base control: all {} set(s) served from cache", len(active))
        return {ps.name: _entry_to_score(cached[ps.name]) for ps in active}

    deferred: tuple[PromptSet, ...] = ()
    with file_lock(lock_path):
        cached = _read_json(cache_path)  # double-checked: a peer may have just written
        missing = tuple(_missing(cached))
        if not missing:
            logger.info("Base control: all {} set(s) served from cache", len(active))
        elif any(ps.scoring == "sample" for ps in missing):
            deferred = missing  # too slow to hold the lock through; compute unlocked below
        else:
            cached = _merge_write(missing, _compute(missing))
    if deferred:
        fresh = _compute(deferred)
        with file_lock(lock_path):
            cached = _merge_write(deferred, fresh)
    return {ps.name: _entry_to_score(cached[ps.name]) for ps in active}


def _preloaded_for(
    cfg: EvaluateTeacherPreferenceConfig, ckpt: Path
) -> tuple[object, object] | None:
    """`(model, tokenizer)` if `ckpt` is the handed-over checkpoint, else None.

    Paths are compared resolved, so a symlinked archive tier (or a relative vs absolute
    spelling) still matches the checkpoint the trainer just wrote.
    """
    if cfg.preloaded is None:
        return None
    pre_ckpt, model, tok = cfg.preloaded
    if model is None or Path(pre_ckpt).resolve() != Path(ckpt).resolve():
        return None
    return model, tok


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
    skipped_nonfinite: list[int] = []
    ftseeds = _discover_ftseed_dirs(
        cfg.param_dir, cfg.seed_prefix, skipped_incomplete, skipped_nonfinite
    )
    if not ftseeds:
        # Distinguish WHY nothing is evaluable. "diverged" and "never finished training"
        # are different scientific outcomes and must not collapse into one message: a
        # diverged cell is a RESULT (the run blew up), a missing one is an artifact gap.
        notes = []
        if skipped_incomplete:
            notes.append(
                f"{len(skipped_incomplete)} skipped as incomplete/interrupted "
                f"checkpoint(s): {sorted(skipped_incomplete)}"
            )
        if skipped_nonfinite:
            notes.append(
                f"{len(skipped_nonfinite)} DIVERGED checkpoint(s) with non-finite weights: "
                f"{sorted(skipped_nonfinite)}"
            )
        skipped_note = f" ({'; '.join(notes)})" if notes else ""
        if skipped_nonfinite and not skipped_incomplete:
            # Persist a marker BEFORE raising. Without one the cell leaves nothing on disk
            # and is indistinguishable from a cell that was never queued -- divergence is a
            # RESULT and must be legible to the notebook loaders, which only read
            # preference_summary.json. The marker deliberately carries NO p_target/summaries,
            # so it can never be misread as a score of 0.
            _write_diverged_marker(cfg, skipped_nonfinite)
            raise DivergedCheckpointError(
                f"Every {cfg.seed_prefix} under {cfg.param_dir} DIVERGED "
                f"(non-finite weights): {sorted(skipped_nonfinite)}. This cell produced no "
                f"evaluable model -- record it as diverged, NOT as p_target=0."
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
            handover = _preloaded_for(cfg, ckpt)
            if handover is not None:
                # Same-process handover: the model just came out of the Trainer, so put
                # it in the state a fresh disk load would be in (eval mode; use_cache
                # back to True after training set it False). Scoring is under no_grad.
                model, seed_tokenizer = handover
                model.eval()
                model.config.use_cache = True
                logger.info("ftseed{}: using the in-memory trained model (no reload)", seed)
            else:
                model = load_finetuned_teacher(
                    cfg.spec, ckpt, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
                    dtype=resolve_dtype(cfg.dtype),
                )
                seed_tokenizer = tokenizer
            fresh = _score_model_over_sets(
                model, seed_tokenizer, cfg.device, prompt_by_set, tuple(needed), forms,
                target_id, cfg.n_samples, cfg.temperature, cfg.confidence,
            )
            # Drops only this local name: a handed-over model stays owned (and freed) by
            # the caller, a disk-loaded one becomes garbage here exactly as before.
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
        "skipped_nonfinite_seeds": sorted(skipped_nonfinite),
        "adapter_ids": {str(seed): aid for seed, aid in adapter_ids.items()},
        # Additive provenance: recorded only when non-default so fp32 summaries (and the
        # sweep runner's DONE gate) stay byte-identical to before.
        **({"base_model_dtype": cfg.dtype} if cfg.dtype != "float32" else {}),
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
