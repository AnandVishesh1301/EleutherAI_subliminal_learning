"""Build the offline filtration-audit sampling plan (no GPU).

Every knob added after audit ``20260829_breadth4_stageb`` defaults to that
audit's behavior; ``tests/unit/test_judge_plan_audit1_golden.py`` pins its
queue/keys bytes and manifest values.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.config import ExperimentConfig, PathsConfig
from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.evaluation.gkd_results_io import (
    parse_gkd_tag,
    parse_teacher_tag,
)
from subliminal_learning.evaluation.judge_results_io import load_rollout_summary
from subliminal_learning.gkd import judge_layout
from subliminal_learning.gkd import judge_sampling as js
from subliminal_learning.gkd.filtering import load_trait_lexicon
from subliminal_learning.gkd.layout import CONTROL_TEACHER_TAG
from subliminal_learning.utils.code_pin_io import git_sha
from subliminal_learning.utils.run_io import sha256_file, write_json_atomic

_SUMMARY_NAME = "rollouts_summary.json"
_SAMPLE_NAME = "rollouts_sample.jsonl"
_REJECTED_NAME = "rejected_with_text.jsonl"
_REQUIRED_SUMMARY = ("n_steps_logged",)
_FLAGGED_STRATA = ("form", "rule_form")
_POPULATION_SOURCES = ("summary", "scan")
_CENSUS_CHECKS = ("none", "exact")
_UNFINGERPRINTED_FIELDS = ("n_workers", "progress_every")
_LEAF_ERRORS = (OSError, ValueError, KeyError, json.JSONDecodeError)


class IncompletePlanInputError(RuntimeError):
    """The expected leaf set and the sampled leaf set differ."""


class PlanFrozenError(RuntimeError):
    """A plan (or judgments) already exist for this audit with other inputs."""


@dataclass(frozen=True)
class RestreamParams:
    """Layer-1 re-stream knobs; must match the pinned pool manifest."""

    target_size: int
    held_out_size: int
    max_scan: int
    prompt_tokens: int
    completion_tokens: int
    seed: int
    tokenizer_size: str
    register_weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class FiltrationAuditConfig:
    """Experiment YAML for the filtration audit.

    Fields after `restream` are optional; their defaults are audit 1's.

    Attributes:
        gkd_tags: ``(tag, screen)`` pairs; empty means
            ``((frozen_gkd_tag, "legacy"),)``.
        clean_quota_by_arm: `judge_sampling.ARM_LABELS` -> clean rows per leaf.
        flagged_quota_by_arm: Arm label -> flagged rows per leaf.
        omit_tasks: Tasks never planned.
        flagged_strata: ``form`` or ``rule_form``.
        rule_floor: Per-rule floor for ``rule_form`` strata.
        rule_oversample: Rule -> share multiplier for ``rule_form`` strata.
        recall_probe_rules: Rules a recall-probe row must be first caught by;
            ``None`` accepts any.
        dedupe_resumed_rollouts: Keep the last row per ``(step, batch_idx)``.
        population_source: ``summary`` (census file) or ``scan`` (sample rows).
        census_check: ``exact`` requires the scan to equal the summary.
        require_unique_row_ids: Refuse a task with a repeated ``row_id``.
        require_cache_fingerprint: Resample a leaf whose cache inputs changed.
        collect_leaf_errors: Record a failing leaf as a skip, not an abort.
        emit_matched_rule: Append ``matched_rule`` to key rows.
        screen_hash_field: Summary field every strict leaf must share.
        reference_audit_id: Audit this one is compared against.
        reference_prompt_frozen_at: Judgments before this UTC time predate
            the frozen judge prompt.
    """

    audit_id: str
    global_seed: int
    campaign_subdir: str
    frozen_gkd_tag: str
    headline_animals: tuple[str, ...]
    rows_per_leaf_clean: int
    rows_per_leaf_flagged: int
    n_step_strata: int
    decoy_fraction: float
    recall_probe_rows: int
    shard_size: int
    n_workers: int
    progress_every: int
    prompt_pool_dir: str
    lexicon_oversample_factor: float
    lexicon_floor: int
    restream: RestreamParams
    gkd_tags: tuple[tuple[str, str], ...] = ()
    clean_quota_by_arm: tuple[tuple[str, int], ...] = ()
    flagged_quota_by_arm: tuple[tuple[str, int], ...] = ()
    omit_tasks: tuple[str, ...] = ()
    flagged_strata: str = "form"
    rule_floor: int = 0
    rule_oversample: tuple[tuple[str, float], ...] = ()
    recall_probe_rules: tuple[str, ...] | None = None
    dedupe_resumed_rollouts: bool = False
    population_source: str = "summary"
    census_check: str = "none"
    require_unique_row_ids: bool = False
    require_cache_fingerprint: bool = False
    collect_leaf_errors: bool = False
    emit_matched_rule: bool = False
    screen_hash_field: str | None = None
    reference_audit_id: str | None = None
    reference_prompt_frozen_at: str | None = None


@dataclass(frozen=True)
class LeafRef:
    """One Stage B leaf under the campaign root."""

    run_dir: Path
    run_dir_rel: str
    animal: str
    model: str
    teacher_tag: str
    gkd_tag: str
    gseed: int
    is_control: bool
    screen: str = "legacy"


@dataclass
class _LeafResult:
    ref: LeafRef
    skip_reason: str | None
    clean_rows: list[dict]
    flagged_rows: list[dict]
    clean_pop_by_quintile: list[int]
    flagged_realised: dict[str, int]
    flagged_budgets: dict[str, int]
    flagged_weights: dict[str, float]
    clean_weights: dict[int, float]
    flagged_pop_total: int
    flagged_unsampled_pop: int
    resumed: bool
    extended: bool = False
    flagged_pop_by_stratum: dict[str, int] = field(default_factory=dict)
    dedupe: dict | None = None
    screen_hash: str | None = None


def audit_tags(cfg: FiltrationAuditConfig) -> dict[str, str]:
    """``gkd_tag -> screen`` the plan selects."""
    return dict(cfg.gkd_tags) or {cfg.frozen_gkd_tag: "legacy"}


def arm_quota(cfg: FiltrationAuditConfig, arm: str, *, kind: str) -> int:
    """Per-leaf clean or flagged row budget for an arm label.

    Args:
        cfg: Audit config.
        arm: `judge_sampling.ARM_LABELS` entry.
        kind: ``clean`` or ``flagged``.

    Returns:
        The arm override, else ``rows_per_leaf_{kind}``.
    """
    if kind == "clean":
        return dict(cfg.clean_quota_by_arm).get(arm, cfg.rows_per_leaf_clean)
    if kind == "flagged":
        return dict(cfg.flagged_quota_by_arm).get(arm, cfg.rows_per_leaf_flagged)
    raise ValueError(f"unknown quota kind {kind!r}")


def load_filtration_audit(path: Path) -> FiltrationAuditConfig:
    """Load `configs/experiment/filtration_audit.yaml`.

    Args:
        path: Experiment YAML path.

    Returns:
        Parsed frozen config.

    Raises:
        ValueError: Required keys are missing or malformed.
    """
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping")
    restream = data.get("restream")
    if not isinstance(restream, dict):
        raise ValueError(f"{path}: missing restream: block")
    weights = restream.get("register_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{path}: restream.register_weights required")
    tags_raw = data.get("gkd_tags")
    if tags_raw is not None and (not isinstance(tags_raw, dict) or not tags_raw):
        raise ValueError(f"{path}: gkd_tags must be a non-empty mapping")
    reference = data.get("reference") or {}
    if not isinstance(reference, dict):
        raise ValueError(f"{path}: reference must be a mapping")
    try:
        frozen = data.get("frozen_gkd_tag")
        if frozen is None:
            if tags_raw is None:
                raise KeyError("frozen_gkd_tag")
            frozen = next(iter(tags_raw))
        rules = data.get("recall_probe_rules")
        cfg = FiltrationAuditConfig(
            audit_id=str(data["audit_id"]),
            global_seed=int(data["global_seed"]),
            campaign_subdir=str(data["campaign_subdir"]),
            frozen_gkd_tag=str(frozen),
            headline_animals=tuple(str(a) for a in data["headline_animals"]),
            rows_per_leaf_clean=int(data["rows_per_leaf_clean"]),
            rows_per_leaf_flagged=int(data["rows_per_leaf_flagged"]),
            n_step_strata=int(data["n_step_strata"]),
            decoy_fraction=float(data["decoy_fraction"]),
            recall_probe_rows=int(data["recall_probe_rows"]),
            shard_size=int(data["shard_size"]),
            n_workers=int(data["n_workers"]),
            progress_every=int(data["progress_every"]),
            prompt_pool_dir=str(data["prompt_pool_dir"]),
            lexicon_oversample_factor=float(data["lexicon_oversample_factor"]),
            lexicon_floor=int(data["lexicon_floor"]),
            restream=RestreamParams(
                target_size=int(restream["target_size"]),
                held_out_size=int(restream["held_out_size"]),
                max_scan=int(restream["max_scan"]),
                prompt_tokens=int(restream["prompt_tokens"]),
                completion_tokens=int(restream["completion_tokens"]),
                seed=int(restream["seed"]),
                tokenizer_size=str(restream["tokenizer_size"]),
                register_weights=tuple((str(k), float(v)) for k, v in weights.items()),
            ),
            gkd_tags=tuple((str(k), str(v)) for k, v in (tags_raw or {}).items()),
            clean_quota_by_arm=_int_pairs(data.get("clean_quota_by_arm")),
            flagged_quota_by_arm=_int_pairs(data.get("flagged_quota_by_arm")),
            omit_tasks=tuple(str(t) for t in data.get("omit_tasks") or ()),
            flagged_strata=str(data.get("flagged_strata", "form")),
            rule_floor=int(data.get("rule_floor", 0)),
            rule_oversample=tuple(
                (str(k), float(v))
                for k, v in (data.get("rule_oversample") or {}).items()
            ),
            recall_probe_rules=None if rules is None else tuple(str(r) for r in rules),
            dedupe_resumed_rollouts=bool(data.get("dedupe_resumed_rollouts", False)),
            population_source=str(data.get("population_source", "summary")),
            census_check=str(data.get("census_check", "none")),
            require_unique_row_ids=bool(data.get("require_unique_row_ids", False)),
            require_cache_fingerprint=bool(
                data.get("require_cache_fingerprint", False)
            ),
            collect_leaf_errors=bool(data.get("collect_leaf_errors", False)),
            emit_matched_rule=bool(data.get("emit_matched_rule", False)),
            screen_hash_field=_opt_str(data.get("screen_hash_field")),
            reference_audit_id=_opt_str(reference.get("audit_id")),
            reference_prompt_frozen_at=_opt_str(reference.get("prompt_frozen_at")),
        )
    except KeyError as err:
        raise ValueError(f"{path}: missing required key {err}") from err
    validate_filtration_audit(cfg)
    return cfg


def validate_filtration_audit(cfg: FiltrationAuditConfig) -> None:
    """Raise on an inconsistent audit config.

    Raises:
        ValueError: Unknown screen/arm/task/strata mode, a clean quota that does
            not divide by the step strata, a scan-only option without
            ``population_source: scan``, or a tag whose parsed screen differs.
    """
    for tag, screen in audit_tags(cfg).items():
        if screen not in js.SCREENS:
            raise ValueError(f"gkd_tags[{tag!r}]: unknown screen {screen!r}")
        parsed = parse_gkd_tag(tag)["rollout_screen"]
        if parsed != screen:
            raise ValueError(
                f"gkd_tags[{tag!r}] says {screen!r}; tag parses {parsed!r}"
            )
    for name, pairs in (
        ("clean_quota_by_arm", cfg.clean_quota_by_arm),
        ("flagged_quota_by_arm", cfg.flagged_quota_by_arm),
    ):
        unknown = sorted(a for a, _ in pairs if a not in js.ARM_LABELS)
        if unknown:
            raise ValueError(f"{name}: unknown arms {unknown}")
        negative = sorted(a for a, n in pairs if n < 0)
        if negative:
            raise ValueError(f"{name}: negative quota for {negative}")
    for arm in js.ARM_LABELS:
        js.clean_quota_per_quintile(
            arm_quota(cfg, arm, kind="clean"), cfg.n_step_strata
        )
    unknown_tasks = sorted(t for t in cfg.omit_tasks if t not in judge_layout.TASKS)
    if unknown_tasks:
        raise ValueError(f"omit_tasks: unknown tasks {unknown_tasks}")
    if cfg.flagged_strata not in _FLAGGED_STRATA:
        raise ValueError(f"flagged_strata must be one of {_FLAGGED_STRATA}")
    if cfg.population_source not in _POPULATION_SOURCES:
        raise ValueError(f"population_source must be one of {_POPULATION_SOURCES}")
    if cfg.census_check not in _CENSUS_CHECKS:
        raise ValueError(f"census_check must be one of {_CENSUS_CHECKS}")
    scan_only = {
        "flagged_strata: rule_form": cfg.flagged_strata == "rule_form",
        "dedupe_resumed_rollouts": cfg.dedupe_resumed_rollouts,
        "census_check: exact": cfg.census_check == "exact",
        "screen_hash_field": cfg.screen_hash_field is not None,
    }
    needs_scan = sorted(k for k, on in scan_only.items() if on)
    if needs_scan and cfg.population_source != "scan":
        raise ValueError(f"{needs_scan} require population_source: scan")
    if cfg.flagged_strata == "rule_form" and not cfg.rule_oversample:
        raise ValueError("flagged_strata: rule_form requires rule_oversample")


def resolve_audit_root(
    paths: PathsConfig,
    override: Path | None = None,
) -> Path:
    """Audit root: CLI override, then `audit_root`, then `output_root`."""
    if override is not None:
        return Path(override)
    if paths.audit_root is not None:
        return Path(paths.audit_root)
    return Path(paths.output_root)


def run(
    cfg: ExperimentConfig,
    experiment_path: Path,
    *,
    max_leaves: int | None = None,
    dry_run: bool = False,
    audit_id: str | None = None,
    audit_root: Path | None = None,
    n_workers: int | None = None,
    expected_run_dirs: Sequence[Path] | None = None,
    judge_config: Path | None = None,
) -> Path:
    """Discover leaves, sample, shuffle, shard, and write the plan.

    The manifest is the commit marker: an existing manifest with the same
    `plan_fingerprint` makes this call a no-op.

    Args:
        cfg: Paths + model from `load_config()`.
        experiment_path: Filtration-audit experiment YAML.
        max_leaves: Optional cap on eligible leaves.
        dry_run: Record a partial plan in the manifest.
        audit_id: Override YAML ``audit_id``.
        audit_root: Override `PathsConfig.audit_root`.
        n_workers: Override YAML ``n_workers``.
        expected_run_dirs: Exact leaf set the plan must sample.
        judge_config: Judge YAML; must resolve to the same audit dir.

    Returns:
        Audit directory written.

    Raises:
        IncompletePlanInputError: An expected leaf is missing, ineligible or
            failed to sample.
        PlanFrozenError: A manifest with another fingerprint, or judgments
            without a manifest, already exist.
    """
    audit_cfg = load_filtration_audit(experiment_path)
    audit_cfg = _with_overrides(audit_cfg, audit_id=audit_id, n_workers=n_workers)
    if max_leaves is not None and expected_run_dirs is not None:
        raise ValueError("max_leaves and expected_run_dirs are mutually exclusive")
    js.clean_quota_per_quintile(audit_cfg.rows_per_leaf_clean, audit_cfg.n_step_strata)
    root = resolve_audit_root(cfg.paths, audit_root)
    audit_dir_path = judge_layout.audit_dir(root, audit_cfg.audit_id)
    campaign_root = Path(cfg.paths.output_root) / audit_cfg.campaign_subdir
    if not campaign_root.is_dir():
        raise FileNotFoundError(f"campaign root not found: {campaign_root}")
    judge_sha = (
        None
        if judge_config is None
        else _check_judge_config(Path(judge_config), audit_dir_path)
    )
    expected_rel = (
        None
        if expected_run_dirs is None
        else _expected_rel(expected_run_dirs, campaign_root)
    )
    omit = set(audit_cfg.omit_tasks)
    pool_dir = Path(cfg.paths.dataset_cache_root) / audit_cfg.prompt_pool_dir
    rejected_path = judge_layout.pool_restream_dir(audit_dir_path) / _REJECTED_NAME
    fingerprint = _plan_fingerprint(
        audit_cfg,
        campaign_root=campaign_root,
        pool_dir=pool_dir,
        expected_rel=expected_rel,
        judge_sha=judge_sha,
        max_leaves=max_leaves,
        dry_run=dry_run,
        rejected_present="pool_rejected" not in omit and rejected_path.is_file(),
    )
    if _plan_committed(audit_dir_path, fingerprint):
        return audit_dir_path

    animal_forms = load_animal_forms()
    trait_lexicon = load_trait_lexicon()
    headline = audit_cfg.headline_animals
    surface_by_animal = {
        animal: frozenset(f.lower() for f in animal_forms[animal])
        for animal in headline
        if animal in animal_forms
    }
    missing_forms = [a for a in headline if a not in animal_forms]
    if missing_forms:
        raise ValueError(f"headline animals missing surface forms: {missing_forms}")

    discovered = list(_iter_leaf_refs(campaign_root))
    eligible, skips = _select_leaves(discovered, audit_cfg, headline)
    if expected_rel is not None:
        eligible = _gate_expected(eligible, skips, expected_rel)
    n_eligible = len(eligible)
    if max_leaves is not None:
        eligible = eligible[: max(0, max_leaves)]
    logger.info(
        "plan build audit_id={} discovered={} eligible={} "
        "planned={} skipped={} dry_run={}",
        audit_cfg.audit_id,
        len(discovered),
        n_eligible,
        len(eligible),
        len(skips),
        dry_run,
    )

    leaf_dir = judge_layout.leaf_samples_dir(audit_dir_path)
    leaf_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        _leaf_job(ref, audit_cfg, campaign_root, leaf_dir, surface_by_animal)
        for ref in eligible
    ]
    leaf_results = _run_leaf_jobs(
        jobs,
        n_workers=audit_cfg.n_workers,
        every=audit_cfg.progress_every,
        collect_errors=audit_cfg.collect_leaf_errors,
    )
    leaf_results.sort(key=lambda r: r.ref.run_dir_rel)
    for result in leaf_results:
        if result.skip_reason is not None:
            skips.append(
                {"run_dir": result.ref.run_dir_rel, "reason": result.skip_reason}
            )

    sampled = [r for r in leaf_results if r.skip_reason is None]
    if expected_rel is not None:
        _gate_sampled(sampled, skips, expected_rel)
    screen_hashes = (
        _screen_hash_gate(sampled) if audit_cfg.screen_hash_field is not None else None
    )
    task_pairs: dict[str, list[tuple[dict, dict]]] = {
        task: [] for task in judge_layout.TASKS
    }
    omitted: list[str] = [t for t in judge_layout.TASKS if t in omit]
    emit = audit_cfg.emit_matched_rule

    if not {"pool_open", "pool_targeted"} <= omit:
        open_pairs, targeted_pairs = _pool_census_pairs(
            pool_dir, headline, emit_rule=emit
        )
        task_pairs["pool_open"].extend(open_pairs)
        task_pairs["pool_targeted"].extend(targeted_pairs)

    if "pool_rejected" in omit:
        logger.info("omitting pool_rejected: listed in omit_tasks")
    elif rejected_path.is_file():
        task_pairs["pool_rejected"].extend(
            _pool_rejected_pairs(
                rejected_path, animal_forms, trait_lexicon, headline, emit_rule=emit
            )
        )
    else:
        omitted.append("pool_rejected")
        logger.warning("omitting pool_rejected: {} is absent", rejected_path)

    recall_rules = audit_cfg.recall_probe_rules
    recall_candidates: list[tuple[LeafRef, dict, float]] = []
    for result in sampled:
        ref = result.ref
        ask = ref.animal
        for rec in result.clean_rows:
            q_idx = int(rec["step_quintile"])
            task_pairs["rollouts_clean"].append(
                _rollout_pair(
                    "rollouts_clean",
                    rec,
                    ref,
                    ask_animal=ask,
                    is_decoy=False,
                    stratum=f"q{q_idx}",
                    weight=float(result.clean_weights[q_idx]),
                    emit_rule=emit,
                )
            )
        for rec in result.flagged_rows:
            form = rec["matched_form"]
            stratum = str(rec["stratum"])
            weight = float(result.flagged_weights[stratum])
            task_pairs["rollouts_flagged"].append(
                _rollout_pair(
                    "rollouts_flagged",
                    rec,
                    ref,
                    ask_animal=ask,
                    is_decoy=False,
                    stratum=stratum,
                    weight=weight,
                    emit_rule=emit,
                )
            )
            rule_ok = recall_rules is None or _row_rule(rec) in recall_rules
            if (
                form is not None
                and str(form).lower() in surface_by_animal[ask]
                and rule_ok
            ):
                recall_candidates.append((ref, rec, weight))
        decoy_pairs = _decoy_pairs(result, audit_cfg)
        task_pairs["decoy"].extend(decoy_pairs)

    recall_rng = js.task_rng(audit_cfg.global_seed, "recall_probe")
    recall_k = min(audit_cfg.recall_probe_rows, len(recall_candidates))
    for idx in js.subsample_indices(len(recall_candidates), recall_k, recall_rng):
        ref, rec, weight = recall_candidates[idx]
        task_pairs["recall_probe"].append(
            _rollout_pair(
                "recall_probe",
                rec,
                ref,
                ask_animal=ref.animal,
                is_decoy=False,
                stratum=str(rec["stratum"]),
                weight=weight,
                emit_rule=emit,
            )
        )
    for task in omit:
        task_pairs[task] = []

    for task in judge_layout.TASKS:
        rng = js.task_rng(audit_cfg.global_seed, f"shuffle|{task}")
        rng.shuffle(task_pairs[task])
        if audit_cfg.require_unique_row_ids:
            js.assert_unique_row_ids((q for q, _ in task_pairs[task]), label=task)

    shard_index: dict[str, list[dict]] = {}
    per_task_totals: dict[str, int] = {}
    for task in judge_layout.TASKS:
        pairs = task_pairs[task]
        shard_index[task] = _write_shards(
            audit_dir_path, task, pairs, audit_cfg.shard_size
        )
        per_task_totals[task] = len(pairs)
        logger.info(
            "task {} n_rows={} shards={}", task, len(pairs), len(shard_index[task])
        )

    leaf_list = [_leaf_manifest_row(r) for r in sampled]
    manifest = {
        "audit_id": audit_cfg.audit_id,
        "global_seed": audit_cfg.global_seed,
        "git_sha": git_sha(),
        "campaign_subdir": audit_cfg.campaign_subdir,
        "frozen_gkd_tag": audit_cfg.frozen_gkd_tag,
        "headline_animals": list(headline),
        "dry_run": dry_run,
        "max_leaves": max_leaves,
        "n_leaves_discovered": len(discovered),
        "n_leaves_eligible": n_eligible,
        "n_leaves_planned": len(eligible),
        "n_leaves_sampled": len(sampled),
        "n_leaves_skipped": len(skips),
        "skips": skips,
        "tasks_omitted": omitted,
        "leaves": leaf_list,
        "stratum_table": _aggregate_strata(sampled),
        "per_task_totals": per_task_totals,
        "shard_index": shard_index,
        "rows_per_leaf_clean": audit_cfg.rows_per_leaf_clean,
        "rows_per_leaf_flagged": audit_cfg.rows_per_leaf_flagged,
        "n_step_strata": audit_cfg.n_step_strata,
        "decoy_fraction": audit_cfg.decoy_fraction,
        "recall_probe_rows": audit_cfg.recall_probe_rows,
        "recall_probe_realised": per_task_totals["recall_probe"],
        "shard_size": audit_cfg.shard_size,
        "prompt_pool_dir": audit_cfg.prompt_pool_dir,
        "plan_fingerprint": fingerprint,
        "gkd_tags": audit_tags(audit_cfg),
        "clean_quota_by_arm": {
            arm: arm_quota(audit_cfg, arm, kind="clean") for arm in js.ARM_LABELS
        },
        "flagged_quota_by_arm": {
            arm: arm_quota(audit_cfg, arm, kind="flagged") for arm in js.ARM_LABELS
        },
        "omit_tasks": list(audit_cfg.omit_tasks),
        "sampling_options": _sampling_options(audit_cfg),
        "judge_config_path": None if judge_config is None else str(judge_config),
        "judge_config_sha256": judge_sha,
        "expected_run_dirs": None
        if expected_rel is None
        else {
            "n": len(expected_rel),
            "sha256": js.leaf_cache_fingerprint({"run_dirs": list(expected_rel)}),
        },
        "screen_hashes": screen_hashes,
        "dedupe_report": _dedupe_report(sampled),
        "per_arm_totals": _per_arm_totals(task_pairs, sampled),
        "reference": {
            "audit_id": audit_cfg.reference_audit_id,
            "prompt_frozen_at": audit_cfg.reference_prompt_frozen_at,
        },
    }
    write_json_atomic(judge_layout.plan_manifest_path(audit_dir_path), manifest)
    logger.info(
        "wrote plan manifest -> {}", judge_layout.plan_manifest_path(audit_dir_path)
    )
    return audit_dir_path


@dataclass(frozen=True)
class _LeafJob:
    ref: LeafRef
    campaign_root: Path
    resume_path: Path
    global_seed: int
    n_strata: int
    clean_per_quintile: int
    flagged_budget: int
    surface_forms: tuple[str, ...]
    lexicon_oversample_factor: float
    lexicon_floor: int
    population_source: str = "summary"
    dedupe: bool = False
    census_check: str = "none"
    flagged_strata: str = "form"
    rule_floor: int = 0
    rule_oversample: tuple[tuple[str, float], ...] = ()
    require_cache_fingerprint: bool = False
    screen_hash_field: str | None = None


def _leaf_job(
    ref: LeafRef,
    audit_cfg: FiltrationAuditConfig,
    campaign_root: Path,
    leaf_dir: Path,
    surface_by_animal: Mapping[str, frozenset[str]],
) -> _LeafJob:
    arm = js.arm_label(ref.screen, ref.is_control)
    return _LeafJob(
        ref=ref,
        campaign_root=campaign_root,
        resume_path=leaf_dir / f"{js.leaf_artifact_hash(ref.run_dir_rel)}.jsonl",
        global_seed=audit_cfg.global_seed,
        n_strata=audit_cfg.n_step_strata,
        clean_per_quintile=js.clean_quota_per_quintile(
            arm_quota(audit_cfg, arm, kind="clean"), audit_cfg.n_step_strata
        ),
        flagged_budget=arm_quota(audit_cfg, arm, kind="flagged"),
        surface_forms=tuple(sorted(surface_by_animal[ref.animal])),
        lexicon_oversample_factor=audit_cfg.lexicon_oversample_factor,
        lexicon_floor=audit_cfg.lexicon_floor,
        population_source=audit_cfg.population_source,
        dedupe=audit_cfg.dedupe_resumed_rollouts,
        census_check=audit_cfg.census_check,
        flagged_strata=audit_cfg.flagged_strata,
        rule_floor=audit_cfg.rule_floor,
        rule_oversample=audit_cfg.rule_oversample,
        require_cache_fingerprint=audit_cfg.require_cache_fingerprint,
        screen_hash_field=audit_cfg.screen_hash_field,
    )


def _with_overrides(
    cfg: FiltrationAuditConfig,
    *,
    audit_id: str | None,
    n_workers: int | None,
) -> FiltrationAuditConfig:
    return dataclasses.replace(
        cfg,
        audit_id=audit_id if audit_id is not None else cfg.audit_id,
        n_workers=n_workers if n_workers is not None else cfg.n_workers,
    )


def _int_pairs(raw: object) -> tuple[tuple[str, int], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ValueError(f"expected a mapping, got {raw!r}")
    return tuple((str(k), int(v)) for k, v in raw.items())


def _opt_str(value: object) -> str | None:
    return None if value is None else str(value)


def _sampling_options(cfg: FiltrationAuditConfig) -> dict:
    return {
        "flagged_strata": cfg.flagged_strata,
        "rule_floor": cfg.rule_floor,
        "rule_oversample": dict(cfg.rule_oversample),
        "recall_probe_rules": None
        if cfg.recall_probe_rules is None
        else list(cfg.recall_probe_rules),
        "dedupe_resumed_rollouts": cfg.dedupe_resumed_rollouts,
        "population_source": cfg.population_source,
        "census_check": cfg.census_check,
        "require_unique_row_ids": cfg.require_unique_row_ids,
        "require_cache_fingerprint": cfg.require_cache_fingerprint,
        "collect_leaf_errors": cfg.collect_leaf_errors,
        "emit_matched_rule": cfg.emit_matched_rule,
        "screen_hash_field": cfg.screen_hash_field,
    }


def _plan_fingerprint(
    cfg: FiltrationAuditConfig,
    *,
    campaign_root: Path,
    pool_dir: Path,
    expected_rel: Sequence[str] | None,
    judge_sha: str | None,
    max_leaves: int | None,
    dry_run: bool,
    rejected_present: bool,
) -> str:
    payload = {
        k: v
        for k, v in dataclasses.asdict(cfg).items()
        if k not in _UNFINGERPRINTED_FIELDS
    }
    payload.update(
        campaign_root=str(campaign_root),
        pool_dir=str(pool_dir),
        expected_run_dirs=None if expected_rel is None else list(expected_rel),
        judge_config_sha256=judge_sha,
        max_leaves=max_leaves,
        dry_run=dry_run,
        pool_rejected_input=rejected_present,
    )
    return js.leaf_cache_fingerprint(payload)


def _plan_committed(audit_dir_path: Path, fingerprint: str) -> bool:
    """Whether an identical plan is already committed.

    Raises:
        PlanFrozenError: The manifest carries another (or no) fingerprint, or
            judgment shards exist without a manifest.
    """
    manifest_path = judge_layout.plan_manifest_path(audit_dir_path)
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        found = existing.get("plan_fingerprint")
        if found == fingerprint:
            logger.info("plan already committed (fingerprint {}); no-op", fingerprint)
            return True
        raise PlanFrozenError(
            f"{manifest_path} is committed with plan_fingerprint={found!r} != "
            f"{fingerprint!r}; a changed plan needs a new audit_id"
        )
    judged = [
        path
        for task in judge_layout.TASKS
        for path in judge_layout.judgments_shard_dir(audit_dir_path, task).glob(
            "shard-*.jsonl"
        )
    ]
    if judged:
        raise PlanFrozenError(
            f"{len(judged)} judgment shards exist under {audit_dir_path} without a "
            "plan manifest; refusing to write a plan under them"
        )
    return False


def _check_judge_config(path: Path, audit_dir_path: Path) -> str:
    """sha256 of the judge YAML after checking it targets `audit_dir_path`.

    Raises:
        ValueError: The judge YAML resolves to another audit dir.
    """
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or "audit_root" not in raw or "audit_id" not in raw:
        raise ValueError(f"{path}: judge YAML needs audit_root and audit_id")
    tool_dir = judge_layout.audit_dir(
        Path(str(raw["audit_root"])), str(raw["audit_id"])
    )
    if tool_dir != audit_dir_path:
        raise ValueError(
            f"judge config {path} resolves to {tool_dir}; the plan writes {audit_dir_path}"
        )
    digest = sha256_file(path)
    if digest is None:
        raise FileNotFoundError(path)
    return digest


def _expected_rel(dirs: Sequence[Path], campaign_root: Path) -> tuple[str, ...]:
    """Expected leaves as paths relative to the campaign root (sorted, unique).

    Raises:
        ValueError: A path is outside the campaign root or repeats.
    """
    out: list[str] = []
    for raw in dirs:
        path = Path(raw)
        if not path.is_absolute():
            rel = path
        else:
            try:
                rel = path.relative_to(campaign_root)
            except ValueError:
                try:
                    rel = path.resolve().relative_to(campaign_root.resolve())
                except ValueError as err:
                    raise ValueError(
                        f"expected run dir {path} is not under {campaign_root}"
                    ) from err
        out.append(str(rel))
    if len(set(out)) != len(out):
        raise ValueError("expected_run_dirs repeats a leaf")
    return tuple(sorted(out))


def _gate_expected(
    eligible: Sequence[LeafRef],
    skips: list[dict],
    expected_rel: Sequence[str],
) -> list[LeafRef]:
    """Restrict eligible leaves to the expected set.

    Raises:
        IncompletePlanInputError: An expected leaf is not eligible.
    """
    wanted = set(expected_rel)
    kept = [ref for ref in eligible if ref.run_dir_rel in wanted]
    for ref in eligible:
        if ref.run_dir_rel not in wanted:
            skips.append(
                {"run_dir": ref.run_dir_rel, "reason": "not_in_expected_run_dirs"}
            )
    missing = sorted(wanted - {ref.run_dir_rel for ref in kept})
    if missing:
        reasons = {s["run_dir"]: s["reason"] for s in skips}
        detail = [f"{m} ({reasons.get(m, 'not_discovered')})" for m in missing[:10]]
        raise IncompletePlanInputError(
            f"{len(missing)} expected leaves are not eligible: {detail}"
        )
    return kept


def _gate_sampled(
    sampled: Sequence[_LeafResult],
    skips: Sequence[Mapping[str, str]],
    expected_rel: Sequence[str],
) -> None:
    got = {r.ref.run_dir_rel for r in sampled}
    missing = sorted(set(expected_rel) - got)
    if missing:
        reasons = {s["run_dir"]: s["reason"] for s in skips}
        detail = [f"{m} ({reasons.get(m, '?')})" for m in missing[:10]]
        raise IncompletePlanInputError(
            f"{len(missing)} expected leaves failed to sample: {detail}"
        )


def _screen_hash_gate(sampled: Sequence[_LeafResult]) -> dict:
    """Each target's strict leaves share one vocab hash no legacy leaf carries.

    The strict vocabulary is built per target animal, so its hash differs
    across animals.

    Returns:
        Other screen -> sorted hashes, and ``"strict"`` -> animal -> hash.

    Raises:
        ValueError: A target's strict hashes are missing or differ, or a
            legacy leaf carries a strict hash.
    """
    strict: dict[str, set[str | None]] = {}
    other: dict[str, set[str | None]] = {}
    for result in sampled:
        if result.ref.screen == "strict":
            strict.setdefault(result.ref.animal, set()).add(result.screen_hash)
        else:
            other.setdefault(result.ref.screen, set()).add(result.screen_hash)
    mixed = {
        animal: sorted(map(str, hashes))
        for animal, hashes in sorted(strict.items())
        if None in hashes or len(hashes) != 1
    }
    if mixed:
        raise ValueError(
            f"strict leaves of one target do not share one screen hash: {mixed}"
        )
    by_animal = {animal: str(next(iter(h))) for animal, h in sorted(strict.items())}
    legacy = set(map(str, other.get("legacy", set())))
    shared = sorted(set(by_animal.values()) & legacy)
    if shared:
        raise ValueError(f"legacy leaves carry a strict screen hash {shared}")
    out: dict = {
        screen: sorted(map(str, hashes)) for screen, hashes in sorted(other.items())
    }
    if by_animal:
        out["strict"] = by_animal
    return out


def _iter_leaf_refs(campaign_root: Path) -> Iterator[LeafRef]:
    root = Path(campaign_root)
    for path in sorted(root.glob(f"*/*/*/*/gseed*/{_SUMMARY_NAME}")):
        if path.relative_to(root).parts[0] == "controls":
            continue
        leaf = path.parent
        yield LeafRef(
            run_dir=leaf,
            run_dir_rel=str(leaf.relative_to(root)),
            animal=leaf.parent.parent.parent.parent.name,
            model=leaf.parent.parent.parent.name,
            teacher_tag=leaf.parent.parent.name,
            gkd_tag=leaf.parent.name,
            gseed=int(leaf.name.removeprefix("gseed")),
            is_control=False,
        )
    control_root = root / "controls" / CONTROL_TEACHER_TAG
    for path in sorted(control_root.glob(f"*/*/*/gseed*/{_SUMMARY_NAME}")):
        leaf = path.parent
        yield LeafRef(
            run_dir=leaf,
            run_dir_rel=str(leaf.relative_to(root)),
            animal=leaf.parent.parent.parent.name,
            model=leaf.parent.parent.name,
            teacher_tag="",
            gkd_tag=leaf.parent.name,
            gseed=int(leaf.name.removeprefix("gseed")),
            is_control=True,
        )


def _select_leaves(
    discovered: Sequence[LeafRef],
    audit_cfg: FiltrationAuditConfig,
    headline: Sequence[str],
) -> tuple[list[LeafRef], list[dict]]:
    headline_set = {a.lower() for a in headline}
    tags = audit_tags(audit_cfg)
    eligible: list[LeafRef] = []
    skips: list[dict] = []
    for ref in discovered:
        if ref.animal.lower() not in headline_set:
            skips.append({"run_dir": ref.run_dir_rel, "reason": "animal_not_headline"})
            continue
        if ref.gkd_tag not in tags:
            skips.append({"run_dir": ref.run_dir_rel, "reason": "gkd_tag_mismatch"})
            continue
        try:
            parse_gkd_tag(ref.gkd_tag)
        except ValueError:
            skips.append({"run_dir": ref.run_dir_rel, "reason": "unparseable_gkd_tag"})
            continue
        if not ref.is_control:
            parse_teacher_tag(ref.teacher_tag)
        sample_path = ref.run_dir / _SAMPLE_NAME
        if not sample_path.is_file():
            skips.append(
                {"run_dir": ref.run_dir_rel, "reason": "missing_rollouts_sample"}
            )
            continue
        eligible.append(dataclasses.replace(ref, screen=tags[ref.gkd_tag]))
    return eligible, skips


def _run_leaf_jobs(
    jobs: Sequence[_LeafJob],
    *,
    n_workers: int,
    every: int,
    collect_errors: bool = False,
) -> list[_LeafResult]:
    if not jobs:
        return []
    sample = _sample_leaf_collected if collect_errors else _sample_one_leaf
    results: list[_LeafResult] = []
    workers = max(1, n_workers)
    if workers == 1:
        for i, job in enumerate(jobs, 1):
            results.append(sample(job))
            if i % every == 0 or i == len(jobs):
                logger.info("sampled {}/{} leaves", i, len(jobs))
        return results
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(sample, job): job for job in jobs}
        done = 0
        for fut in as_completed(futs):
            job = futs[fut]
            try:
                results.append(fut.result())
            except Exception:
                logger.exception("leaf failed {}", job.ref.run_dir_rel)
                raise
            done += 1
            if done % every == 0 or done == len(jobs):
                logger.info("sampled {}/{} leaves", done, len(jobs))
    return results


def _sample_leaf_collected(job: _LeafJob) -> _LeafResult:
    try:
        return _sample_one_leaf(job)
    except _LEAF_ERRORS as err:
        logger.exception("leaf error recorded as skip {}", job.ref.run_dir_rel)
        return _empty_result(job.ref, f"leaf_error:{type(err).__name__}:{err}")


def _sample_one_leaf(job: _LeafJob) -> _LeafResult:
    fingerprint = _job_fingerprint(job) if job.require_cache_fingerprint else None
    if job.resume_path.is_file():
        try:
            result = _load_leaf_resume(job, fingerprint)
            logger.info("resume leaf {}", job.ref.run_dir_rel)
            return result
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as err:
            logger.warning("ignoring corrupt leaf resume {}: {}", job.resume_path, err)
    if job.population_source == "scan":
        return _sample_one_leaf_scan(job, fingerprint)
    summary_path = job.ref.run_dir / _SUMMARY_NAME
    sample_path = job.ref.run_dir / _SAMPLE_NAME
    try:
        summary = load_rollout_summary(job.ref.run_dir)
    except (OSError, json.JSONDecodeError) as err:
        logger.exception("cannot read {}", summary_path)
        return _empty_result(job.ref, f"invalid_summary:{err}")
    missing = [k for k in _REQUIRED_SUMMARY if k not in summary]
    if missing:
        return _empty_result(job.ref, f"invalid_summary:missing:{missing}")
    n_steps = int(summary["n_steps_logged"])
    if n_steps < 1:
        return _empty_result(job.ref, "no_steps")
    form_counts = {
        str(k): int(v) for k, v in dict(summary["matched_form_counts"]).items()
    }
    budgets = js.allocate_flagged_budgets(
        form_counts,
        job.flagged_budget,
        set(job.surface_forms),
        job.lexicon_oversample_factor,
        job.lexicon_floor,
    )
    unsampled_pop, pop_total = js.flagged_unsampled_population(form_counts, budgets)
    rng = js.make_leaf_rng(job.global_seed, job.ref.run_dir_rel)
    try:
        draw = js.sample_rollout_rows(
            _iter_rollout_jsonl(sample_path),
            n_steps=n_steps,
            n_strata=job.n_strata,
            clean_per_quintile=job.clean_per_quintile,
            flagged_budgets=budgets,
            rng=rng,
        )
    except ValueError as err:
        logger.exception("sample failed {}", job.ref.run_dir_rel)
        return _empty_result(job.ref, f"sample_error:{err}")

    flagged_weights = js.realised_stratum_weights(form_counts, draw.flagged_realised)
    clean_pop = {i: n for i, n in enumerate(draw.clean_pop_by_quintile) if n > 0}
    clean_realised = {}
    for row in draw.clean:
        clean_realised[row.step_quintile] = clean_realised.get(row.step_quintile, 0) + 1
    clean_weights = js.realised_stratum_weights(clean_pop, clean_realised)
    clean_rows = [
        _stored_row(
            row, f"q{row.step_quintile}", float(clean_weights[row.step_quintile])
        )
        for row in draw.clean
    ]
    flagged_rows = [
        _stored_row(
            row,
            str(row.matched_form),
            float(flagged_weights[row.matched_form]),
        )
        for row in draw.flagged
        if row.matched_form is not None
    ]
    result = _LeafResult(
        ref=job.ref,
        skip_reason=None,
        clean_rows=clean_rows,
        flagged_rows=flagged_rows,
        clean_pop_by_quintile=list(draw.clean_pop_by_quintile),
        flagged_realised=dict(draw.flagged_realised),
        flagged_budgets=dict(draw.flagged_budgets),
        flagged_weights=flagged_weights,
        clean_weights={int(k): float(v) for k, v in clean_weights.items()},
        flagged_pop_total=pop_total,
        flagged_unsampled_pop=unsampled_pop,
        resumed=False,
    )
    _write_leaf_resume(job, result, fingerprint)
    return result


def _sample_one_leaf_scan(job: _LeafJob, fingerprint: str | None) -> _LeafResult:
    """Sample from populations scanned out of the (deduplicated) sample rows.

    Raises:
        ValueError: The summary is invalid, its screen disagrees with the tag,
            a strict flagged row has no rule, or the exact census check fails.
    """
    summary = load_rollout_summary(job.ref.run_dir)
    missing = [k for k in _REQUIRED_SUMMARY if k not in summary]
    if missing:
        return _empty_result(job.ref, f"invalid_summary:missing:{missing}")
    n_steps = int(summary["n_steps_logged"])
    if n_steps < 1:
        return _empty_result(job.ref, "no_steps")
    summary_screen = str(summary.get("rollout_screen", "legacy"))
    if summary_screen != job.ref.screen:
        raise ValueError(
            f"{job.ref.run_dir_rel}: summary rollout_screen={summary_screen!r} "
            f"but tag {job.ref.gkd_tag!r} is {job.ref.screen!r}"
        )
    strict = job.ref.screen == "strict"
    raw_rows = list(_iter_rollout_jsonl(job.ref.run_dir / _SAMPLE_NAME))
    if job.dedupe:
        rows, report = js.dedupe_last_writer(raw_rows)
    else:
        rows = raw_rows
        report = js.DedupeReport(len(raw_rows), len(raw_rows), 0)
    census = js.rollout_census(rows, strict=strict)
    if job.census_check == "exact":
        mismatches = js.census_mismatches(summary, census, strict=strict)
        if mismatches:
            raise ValueError(f"{job.ref.run_dir_rel}: census mismatch {mismatches}")

    surface = set(job.surface_forms)
    if job.flagged_strata == "rule_form":
        pop_by_stratum = {
            js.stratum_key(rule, form): n
            for rule, forms in census.rule_form_counts.items()
            for form, n in forms.items()
        }
        budgets = js.allocate_rule_form_budgets(
            census.rule_form_counts,
            job.flagged_budget,
            surface,
            job.lexicon_oversample_factor,
            job.lexicon_floor,
            job.rule_floor,
            dict(job.rule_oversample),
        )
        key_fn = _rule_form_key
    else:
        pop_by_stratum = dict(census.form_counts)
        budgets = js.allocate_flagged_budgets(
            pop_by_stratum,
            job.flagged_budget,
            surface,
            job.lexicon_oversample_factor,
            job.lexicon_floor,
        )
        key_fn = None
    unsampled_pop, pop_total = js.flagged_unsampled_population(pop_by_stratum, budgets)
    draw = js.sample_rollout_rows(
        rows,
        n_steps=n_steps,
        n_strata=job.n_strata,
        clean_per_quintile=job.clean_per_quintile,
        flagged_budgets=budgets,
        rng=js.make_leaf_rng(job.global_seed, job.ref.run_dir_rel),
        flagged_key=key_fn,
    )
    flagged_weights = js.realised_stratum_weights(pop_by_stratum, draw.flagged_realised)
    clean_pop = {i: n for i, n in enumerate(draw.clean_pop_by_quintile) if n > 0}
    clean_realised: dict[int, int] = {}
    for row in draw.clean:
        clean_realised[row.step_quintile] = clean_realised.get(row.step_quintile, 0) + 1
    clean_weights = js.realised_stratum_weights(clean_pop, clean_realised)
    clean_rows = [
        _stored_row(
            row,
            f"q{row.step_quintile}",
            float(clean_weights[row.step_quintile]),
            with_rule=True,
        )
        for row in draw.clean
    ]
    flagged_rows = []
    for row in draw.flagged:
        stratum = str(row.matched_form) if key_fn is None else key_fn(row)
        flagged_rows.append(
            _stored_row(row, stratum, float(flagged_weights[stratum]), with_rule=True)
        )
    hash_value = (
        None if job.screen_hash_field is None else summary.get(job.screen_hash_field)
    )
    result = _LeafResult(
        ref=job.ref,
        skip_reason=None,
        clean_rows=clean_rows,
        flagged_rows=flagged_rows,
        clean_pop_by_quintile=list(draw.clean_pop_by_quintile),
        flagged_realised=dict(draw.flagged_realised),
        flagged_budgets=dict(draw.flagged_budgets),
        flagged_weights=flagged_weights,
        clean_weights={int(k): float(v) for k, v in clean_weights.items()},
        flagged_pop_total=pop_total,
        flagged_unsampled_pop=unsampled_pop,
        resumed=False,
        extended=True,
        flagged_pop_by_stratum=dict(sorted(pop_by_stratum.items())),
        dedupe=dataclasses.asdict(report),
        screen_hash=None if hash_value is None else str(hash_value),
    )
    _write_leaf_resume(job, result, fingerprint)
    return result


def _rule_form_key(row: js.SampledRollout) -> str | None:
    if row.matched_form is None:
        return None
    return js.stratum_key(row.matched_rule or js.LEGACY_RULE, row.matched_form)


def _row_rule(rec: Mapping[str, object]) -> str:
    rule = rec.get("matched_rule")
    return js.LEGACY_RULE if rule is None else str(rule)


def _job_fingerprint(job: _LeafJob) -> str:
    sample = job.ref.run_dir / _SAMPLE_NAME
    stat = sample.stat() if sample.is_file() else None
    params = {
        k: v
        for k, v in dataclasses.asdict(job).items()
        if k not in ("ref", "campaign_root", "resume_path")
    }
    params.update(
        run_dir_rel=job.ref.run_dir_rel,
        screen=job.ref.screen,
        is_control=job.ref.is_control,
        summary_sha256=sha256_file(job.ref.run_dir / _SUMMARY_NAME),
        sample_size=None if stat is None else stat.st_size,
        sample_mtime_ns=None if stat is None else stat.st_mtime_ns,
    )
    return js.leaf_cache_fingerprint(params)


def _empty_result(ref: LeafRef, reason: str) -> _LeafResult:
    return _LeafResult(
        ref=ref,
        skip_reason=reason,
        clean_rows=[],
        flagged_rows=[],
        clean_pop_by_quintile=[],
        flagged_realised={},
        flagged_budgets={},
        flagged_weights={},
        clean_weights={},
        flagged_pop_total=0,
        flagged_unsampled_pop=0,
        resumed=False,
    )


def _stored_row(
    row: js.SampledRollout,
    stratum: str,
    weight: float,
    *,
    with_rule: bool = False,
) -> dict:
    stored = {
        "step": row.step,
        "batch_idx": row.batch_idx,
        "prompt": row.prompt,
        "completion": row.completion,
        "matched_form": row.matched_form,
        "flagged": row.flagged,
        "step_quintile": row.step_quintile,
        "stratum": stratum,
        "sample_weight": weight,
    }
    if with_rule:
        stored["matched_rule"] = row.matched_rule
    return stored


def _iter_rollout_jsonl(path: Path) -> Iterator[dict]:
    with path.open() as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"corrupt JSONL {path}:{lineno}") from err


def _write_leaf_resume(
    job: _LeafJob,
    result: _LeafResult,
    fingerprint: str | None = None,
) -> None:
    meta = {
        "record_type": "meta",
        "run_dir_rel": job.ref.run_dir_rel,
        "animal": job.ref.animal,
        "arm": "control" if job.ref.is_control else "treatment",
        "clean_pop_by_quintile": result.clean_pop_by_quintile,
        "flagged_realised": result.flagged_realised,
        "flagged_budgets": result.flagged_budgets,
        "flagged_weights": result.flagged_weights,
        "clean_weights": {str(k): v for k, v in result.clean_weights.items()},
        "flagged_pop_total": result.flagged_pop_total,
        "flagged_unsampled_pop": result.flagged_unsampled_pop,
    }
    if fingerprint is not None:
        meta["cache_fingerprint"] = fingerprint
    if result.extended:
        meta.update(
            extended=True,
            flagged_pop_by_stratum=result.flagged_pop_by_stratum,
            dedupe=result.dedupe,
            screen_hash=result.screen_hash,
        )
    lines = [json.dumps(meta)]
    for kind, rows in (("clean", result.clean_rows), ("flagged", result.flagged_rows)):
        for row in rows:
            payload = {"record_type": "row", "kind": kind, **row}
            lines.append(json.dumps(payload, ensure_ascii=False))
    text = "\n".join(lines) + "\n"
    tmp = job.resume_path.with_name(f"{job.resume_path.name}.tmp")
    tmp.write_text(text)
    tmp.replace(job.resume_path)


def _load_leaf_resume(job: _LeafJob, fingerprint: str | None = None) -> _LeafResult:
    meta: dict | None = None
    clean_rows: list[dict] = []
    flagged_rows: list[dict] = []
    with job.resume_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record_type") == "meta":
                meta = rec
                continue
            kind = rec.get("kind")
            stored = {
                "step": rec["step"],
                "batch_idx": rec["batch_idx"],
                "prompt": rec["prompt"],
                "completion": rec["completion"],
                "matched_form": rec.get("matched_form"),
                "flagged": rec["flagged"],
                "step_quintile": rec["step_quintile"],
                "stratum": rec["stratum"],
                "sample_weight": rec["sample_weight"],
            }
            if "matched_rule" in rec:
                stored["matched_rule"] = rec["matched_rule"]
            if kind == "clean":
                clean_rows.append(stored)
            elif kind == "flagged":
                flagged_rows.append(stored)
            else:
                raise ValueError(f"unknown leaf row kind {kind!r}")
    if meta is None:
        raise ValueError("leaf resume missing meta")
    if fingerprint is not None and meta.get("cache_fingerprint") != fingerprint:
        raise ValueError(
            f"leaf cache fingerprint {meta.get('cache_fingerprint')!r} != "
            f"{fingerprint!r}; resampling"
        )
    extended = bool(meta.get("extended", False))
    if extended != (job.population_source == "scan"):
        raise ValueError("leaf cache population source differs; resampling")
    clean_weights = {int(k): float(v) for k, v in meta["clean_weights"].items()}
    return _LeafResult(
        ref=job.ref,
        skip_reason=None,
        clean_rows=clean_rows,
        flagged_rows=flagged_rows,
        clean_pop_by_quintile=list(meta["clean_pop_by_quintile"]),
        flagged_realised=dict(meta["flagged_realised"]),
        flagged_budgets=dict(meta["flagged_budgets"]),
        flagged_weights=dict(meta["flagged_weights"]),
        clean_weights=clean_weights,
        flagged_pop_total=int(meta["flagged_pop_total"]),
        flagged_unsampled_pop=int(meta["flagged_unsampled_pop"]),
        resumed=True,
        extended=extended,
        flagged_pop_by_stratum=dict(meta.get("flagged_pop_by_stratum") or {}),
        dedupe=meta.get("dedupe"),
        screen_hash=meta.get("screen_hash"),
    )


def _rollout_pair(
    task: str,
    rec: Mapping[str, object],
    ref: LeafRef,
    *,
    ask_animal: str | None,
    is_decoy: bool,
    stratum: str,
    weight: float,
    emit_rule: bool = False,
) -> tuple[dict, dict]:
    rid = js.row_id(
        task,
        ref.run_dir_rel,
        int(rec["step"]),
        int(rec["batch_idx"]),
        ask_animal,
    )
    queue = js.blind_queue_row(
        rid, task, ask_animal, str(rec["prompt"]), str(rec["completion"])
    )
    rule = rec.get("matched_rule")
    key = js.make_key_row(
        row_id_value=rid,
        task=task,
        source="rollout",
        run_dir=ref.run_dir_rel,
        animal=ref.animal,
        arm="control" if ref.is_control else "treatment",
        step=int(rec["step"]),
        step_quintile_value=int(rec["step_quintile"]),
        batch_idx=int(rec["batch_idx"]),
        flagged=bool(rec["flagged"]),
        matched_form=None
        if rec.get("matched_form") is None
        else str(rec["matched_form"]),
        ask_animal=ask_animal,
        is_decoy=is_decoy,
        stratum=stratum,
        sample_weight=weight,
        doc_id=None,
        register=None,
        matched_rule=None if not emit_rule or rule is None else str(rule),
        emit_rule=emit_rule,
    )
    return queue, key


def _decoy_pairs(
    result: _LeafResult,
    audit_cfg: FiltrationAuditConfig,
) -> list[tuple[dict, dict]]:
    combined = sorted(
        [*result.clean_rows, *result.flagged_rows],
        key=lambda r: (int(r["step"]), int(r["batch_idx"]), str(r.get("stratum"))),
    )
    k = js.subsample_count(len(combined), audit_cfg.decoy_fraction)
    rng = js.make_leaf_rng(audit_cfg.global_seed, f"{result.ref.run_dir_rel}|decoy")
    pairs: list[tuple[dict, dict]] = []
    for idx in js.subsample_indices(len(combined), k, rng):
        rec = combined[idx]
        decoy = js.pick_decoy_animal(result.ref.animal, audit_cfg.headline_animals, rng)
        weight = float(rec["sample_weight"])
        pairs.append(
            _rollout_pair(
                "decoy",
                rec,
                result.ref,
                ask_animal=decoy,
                is_decoy=True,
                stratum=str(rec["stratum"]),
                weight=weight,
                emit_rule=audit_cfg.emit_matched_rule,
            )
        )
    return pairs


def _pool_census_pairs(
    pool_dir: Path,
    headline: Sequence[str],
    *,
    emit_rule: bool = False,
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    prompts_path = pool_dir / "prompts.jsonl"
    held_path = pool_dir / "heldout_ppl.jsonl"
    if not prompts_path.is_file() or not held_path.is_file():
        raise FileNotFoundError(
            f"prompt pool incomplete under {pool_dir} "
            "(need prompts.jsonl and heldout_ppl.jsonl)"
        )
    open_pairs: list[tuple[dict, dict]] = []
    targeted: list[tuple[dict, dict]] = []
    for source, path in (
        ("pool_prompts", prompts_path),
        ("pool_heldout", held_path),
    ):
        for row in _read_jsonl(path):
            doc_id = str(row["doc_id"])
            rel = f"pool/{source}/{doc_id}"
            prompt = str(row.get("prompt") or row["prompt_text"])
            completion = str(row.get("completion") or row["completion_text"])
            register = str(row.get("register", ""))
            open_pairs.append(
                _pool_pair(
                    "pool_open",
                    source,
                    rel,
                    doc_id,
                    register,
                    prompt,
                    completion,
                    ask_animal=None,
                    emit_rule=emit_rule,
                )
            )
            for animal in headline:
                targeted.append(
                    _pool_pair(
                        "pool_targeted",
                        source,
                        rel,
                        doc_id,
                        register,
                        prompt,
                        completion,
                        ask_animal=animal,
                        emit_rule=emit_rule,
                    )
                )
    return open_pairs, targeted


def _pool_rejected_pairs(
    path: Path,
    animal_forms: Mapping[str, Sequence[str]],
    trait_lexicon: Mapping[str, Sequence[str]],
    headline: Sequence[str],
    *,
    emit_rule: bool = False,
) -> list[tuple[dict, dict]]:
    pairs: list[tuple[dict, dict]] = []
    for row in _read_jsonl(path):
        doc_id = str(row["doc_id"])
        form = str(row["matched_form"])
        ask = js.primary_animal_for_form(form, animal_forms, trait_lexicon, headline)
        rel = f"pool/pool_rejected/{doc_id}"
        pairs.append(
            _pool_pair(
                "pool_rejected",
                "pool_rejected",
                rel,
                doc_id,
                str(row.get("register", "")),
                "",
                str(row["screen_text"]),
                ask_animal=ask,
                matched_form=form,
                flagged=True,
                emit_rule=emit_rule,
            )
        )
    return pairs


def _pool_pair(
    task: str,
    source: str,
    run_dir_rel: str,
    doc_id: str,
    register: str,
    prompt: str,
    completion: str,
    *,
    ask_animal: str | None,
    matched_form: str | None = None,
    flagged: bool = False,
    emit_rule: bool = False,
) -> tuple[dict, dict]:
    rid = js.row_id(task, run_dir_rel, None, None, ask_animal)
    queue = js.blind_queue_row(rid, task, ask_animal, prompt, completion)
    key = js.make_key_row(
        row_id_value=rid,
        task=task,
        source=source,
        run_dir=None,
        animal=ask_animal if task == "pool_targeted" else None,
        arm=None,
        step=None,
        step_quintile_value=None,
        batch_idx=None,
        flagged=flagged,
        matched_form=matched_form,
        ask_animal=ask_animal,
        is_decoy=False,
        stratum="census",
        sample_weight=1.0,
        doc_id=doc_id,
        register=register,
        emit_rule=emit_rule,
    )
    return queue, key


def _write_shards(
    audit_dir_path: Path,
    task: str,
    pairs: Sequence[tuple[dict, dict]],
    shard_size: int,
) -> list[dict]:
    """Write queue/keys shards (tmp then rename).

    Raises:
        RuntimeError: A shard file outside the set just written remains in the
            task's queue or keys dir.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1, got {shard_size}")
    qdir = judge_layout.queue_shard_dir(audit_dir_path, task)
    kdir = judge_layout.keys_shard_dir(audit_dir_path, task)
    if not pairs:
        _assert_no_stale_shards(task, (qdir, kdir), set())
        return []
    qdir.mkdir(parents=True, exist_ok=True)
    kdir.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    for shard_i, start in enumerate(range(0, len(pairs), shard_size)):
        chunk = pairs[start : start + shard_size]
        name = judge_layout.shard_name(shard_i)
        q_lines: list[str] = []
        k_lines: list[str] = []
        for queue, key in chunk:
            js.assert_queue_blind(queue)
            q_lines.append(json.dumps(queue, ensure_ascii=False))
            k_lines.append(json.dumps(key, ensure_ascii=False))
        _replace_text(qdir / name, "\n".join(q_lines) + "\n")
        _replace_text(kdir / name, "\n".join(k_lines) + "\n")
        index.append({"shard": name, "n_rows": len(chunk)})
    _assert_no_stale_shards(task, (qdir, kdir), {row["shard"] for row in index})
    return index


def _replace_text(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _assert_no_stale_shards(
    task: str,
    directories: Sequence[Path],
    written: set[str],
) -> None:
    stale = [
        str(p)
        for d in directories
        if d.is_dir()
        for p in sorted(d.glob("shard-*.jsonl"))
        if p.name not in written
    ]
    if stale:
        raise RuntimeError(f"{task}: stale shards from an earlier plan: {stale[:5]}")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _leaf_manifest_row(result: _LeafResult) -> dict:
    row = {
        "run_dir": result.ref.run_dir_rel,
        "animal": result.ref.animal,
        "arm": "control" if result.ref.is_control else "treatment",
        "model": result.ref.model,
        "teacher_tag": result.ref.teacher_tag,
        "gkd_tag": result.ref.gkd_tag,
        "gseed": result.ref.gseed,
        "n_clean_realised": len(result.clean_rows),
        "n_flagged_realised": len(result.flagged_rows),
        "clean_pop_by_quintile": result.clean_pop_by_quintile,
        "flagged_realised": result.flagged_realised,
        "flagged_budgets": result.flagged_budgets,
        "flagged_pop_total": result.flagged_pop_total,
        "flagged_unsampled_pop": result.flagged_unsampled_pop,
        "resumed": result.resumed,
    }
    if result.extended:
        row.update(
            screen=result.ref.screen,
            arm_label=js.arm_label(result.ref.screen, result.ref.is_control),
            flagged_pop_by_stratum=result.flagged_pop_by_stratum,
            dedupe=result.dedupe,
            screen_hash=result.screen_hash,
        )
    return row


def _dedupe_report(sampled: Sequence[_LeafResult]) -> dict | None:
    reports = [(r.ref.run_dir_rel, r.dedupe) for r in sampled if r.dedupe is not None]
    if not reports:
        return None
    dup = [
        {"run_dir": run_dir, **report}
        for run_dir, report in reports
        if int(report["n_duplicates"])
    ]
    return {
        "n_leaves": len(reports),
        "n_leaves_with_duplicates": len(dup),
        "n_duplicate_rows": sum(int(d["n_duplicates"]) for d in dup),
        "leaves": dup,
    }


def _per_arm_totals(
    task_pairs: Mapping[str, Sequence[tuple[dict, dict]]],
    sampled: Sequence[_LeafResult],
) -> dict[str, dict[str, int]]:
    arm_of = {
        r.ref.run_dir_rel: js.arm_label(r.ref.screen, r.ref.is_control) for r in sampled
    }
    out: dict[str, dict[str, int]] = {}
    for task, pairs in task_pairs.items():
        counts: dict[str, int] = {}
        for _, key in pairs:
            run_dir = key.get("run_dir")
            arm = "pool" if run_dir is None else arm_of[str(run_dir)]
            counts[arm] = counts.get(arm, 0) + 1
        out[task] = dict(sorted(counts.items()))
    return out


def _aggregate_strata(sampled: Sequence[_LeafResult]) -> dict:
    clean_q: dict[int, int] = {}
    flagged_forms: dict[str, int] = {}
    for result in sampled:
        for q, n in enumerate(result.clean_pop_by_quintile):
            clean_q[q] = clean_q.get(q, 0) + int(n)
        for form, n in result.flagged_realised.items():
            flagged_forms[form] = flagged_forms.get(form, 0) + int(n)
    flagged_pop_total = sum(int(r.flagged_pop_total) for r in sampled)
    flagged_unsampled_pop = sum(int(r.flagged_unsampled_pop) for r in sampled)
    fraction = flagged_unsampled_pop / flagged_pop_total if flagged_pop_total else 0.0
    return {
        "clean_population_by_quintile": {str(k): v for k, v in sorted(clean_q.items())},
        "flagged_realised_by_form": dict(sorted(flagged_forms.items())),
        "flagged_pop_total": flagged_pop_total,
        "flagged_unsampled_pop": flagged_unsampled_pop,
        "flagged_unsampled_fraction": fraction,
    }
