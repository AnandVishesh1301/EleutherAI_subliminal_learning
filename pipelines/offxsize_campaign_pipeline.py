"""Driver for the off-policy cross-size campaign (`20260909_offxsize-dog_mixed`).

`advance` is idempotent: every phase is a key in `campaign_state.json`, every gate writes
its decision before acting on it, and every decision is read back rather than recomputed,
so a re-advance rebuilds an identical jobkey set.

One phase deliberately blocks. `numbers_ready` waits for pools the operator generates by
hand (`scripts.offxsize_numbers`) and reports what is missing instead of raising, so
`watch` keeps polling and the campaign resumes on its own once the last pool lands.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import offxsize_expected as expected
from subliminal_learning.campaign import offxsize_jobs as jobs
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign import offxsize_pairs as pairs
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign import offxsize_selection as selection
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.campaign.offxsize_selection import CampaignGateError
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.utils.run_io import utc_now_iso

T_PREFLIGHT = "preflight"
T_SUBMIT_P1A = "submit_p1a"
T_TEACHER_GATE = "teacher_gate"
T_SUBMIT_P1B = "submit_p1b"
T_NUMBERS_READY = "numbers_ready"
T_FINALIZE = "finalize"


def t_submit_p2(size: str) -> str:
    """Transition key for one size's star-screen submission."""
    return f"submit_p2_{size}"


def t_pick(size: str) -> str:
    """Transition key for one size's recipe pick."""
    return f"pick_{size}"


def t_submit_p3(size: str) -> str:
    """Transition key for one size's headline submission."""
    return f"submit_p3_{size}"


@dataclass(frozen=True)
class AdvanceResult:
    """What one `advance` pass did.

    Attributes:
        fired: Transitions that ran in this pass.
        waiting: Human-readable reasons the campaign is not finished.
        done: Whether `CAMPAIGN_DONE` exists.
    """

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def driver_config(**kwargs: object) -> CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to this campaign's spec."""
    kwargs.setdefault("spec", plan.SPEC)
    return CampaignDriverConfig(**kwargs)  # type: ignore[arg-type]


def campaign_root(cfg: CampaignDriverConfig) -> Path:
    """Root of every artifact this campaign writes."""
    return cfg.run_set_root


# ---- readiness helpers --------------------------------------------------------


def _queues_ready(
    cfg: CampaignDriverConfig, names: Sequence[str], waiting: list[str]
) -> bool:
    """Whether every named queue has drained, requeuing failures under the cap."""
    if cfg.dry_run:
        return True
    ready = True
    for name in names:
        driver_io.requeue_failed_under_cap(cfg, name)
        ok, msg = driver_io.queue_finished(cfg, name)
        if "FAILED" in msg:
            raise CampaignGateError(msg)
        if not ok:
            waiting.append(msg)
            ready = False
    return ready


def _leaves_ready(
    cfg: CampaignDriverConfig,
    leaves: Sequence[pairs.Leaf],
    tags: Mapping[str, str],
    waiting: list[str],
    label: str,
) -> bool:
    """Whether every cell has a trusted-done leaf and an in-process preference eval."""
    if cfg.dry_run:
        return True
    root = campaign_root(cfg)
    missing = expected.missing_manifests(root, leaves, tags)
    if missing:
        waiting.append(f"{label}: {len(missing)} cell(s) not trained")
        return False
    unscored = expected.missing_preference(root, leaves, tags)
    if unscored:
        waiting.append(f"{label}: {len(unscored)} cell(s) not scored")
        return False
    return True


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("unreadable JSON at {}", path)
        return None


def _summary_entry(doc: Mapping[str, object] | None, prompt_set: str) -> dict:
    """One prompt set's block from a `preference_summary.json`."""
    if not doc:
        return {}
    for entry in doc.get("summaries", []) or []:
        if isinstance(entry, Mapping) and entry.get("prompt_set") == prompt_set:
            return dict(entry)
    return {}


def _perplexity_ratio(param_dir: Path) -> float | None:
    """`perplexity_ratio` from a capability summary, or None when not measured yet."""
    doc = _read_json(param_dir / expected.CAPABILITY_SUMMARY)
    if not doc:
        return None
    ppl = doc.get("perplexity")
    if isinstance(ppl, Mapping):
        value = ppl.get("perplexity_ratio")
        if isinstance(value, (int, float)):
            return float(value)
        result = ppl.get("result")
        if isinstance(result, Mapping) and isinstance(
            result.get("perplexity_ratio"), (int, float)
        ):
            return float(result["perplexity_ratio"])
    return None


# ---- S0 -----------------------------------------------------------------------


def _preflight(cfg: CampaignDriverConfig, state: dict) -> None:
    """Require the out-of-band staging artifacts before anything is queued.

    Raises:
        CampaignGateError: Staging has not run (or did not finish).
    """
    for filename in (plan.PREFLIGHT_REPORT, plan.ADOPTION_MANIFEST):
        path = cfg.logs_dir / filename
        if not cfg.dry_run and not path.is_file():
            raise CampaignGateError(
                f"S0 missing {filename} at {path} — run "
                "`uv run python -m scripts.stage_offxsize_assets` first"
            )
    state["transitions"][T_PREFLIGHT] = {"at": _now()}


# ---- P1: teacher dose ladder ---------------------------------------------------


def _submit_p1a(cfg: CampaignDriverConfig, state: dict) -> None:
    """Queue the ladder trainings, their evals, the format probes and the base priors."""
    driver_io.submit_queue(cfg, plan.Q_P1_TEACHERS, jobs.teacher_ladder_jobs())
    driver_io.submit_queue(
        cfg,
        plan.Q_P1_TEACHER_EVALS,
        jobs.teacher_eval_jobs(),
        depends_on=[plan.Q_P1_TEACHERS],
    )
    driver_io.submit_queue(
        cfg, plan.Q_P1_PROBES, jobs.probe_jobs(), depends_on=[plan.Q_P1_TEACHERS]
    )
    driver_io.submit_queue(cfg, plan.Q_P1_BASE_PRIORS, jobs.base_prior_jobs())
    state["transitions"][T_SUBMIT_P1A] = {"at": _now()}


def teacher_readings(cfg: CampaignDriverConfig) -> list[selection.RungReading]:
    """Read every ladder rung's trait, capability and format probe off disk.

    The base control's across-prompt SD is not carried in her summary, so the SD band is
    evaluated with SD 0; at a base prior near 0.003 against a saturated teacher it is the
    absolute margin, the perplexity bound and the probe that bind.
    """
    root = campaign_root(cfg)
    out: list[selection.RungReading] = []
    for size in plan.SIZES:
        for rung in plan.TEACHER_RUNGS:
            param_dir = layout.teacher_dir(root, rung, size)
            summary = _read_json(param_dir / expected.PREFERENCE_SUMMARY)
            validation = _summary_entry(summary, "validation")
            off_topic = _summary_entry(summary, "off_topic")
            probe = _read_json(
                layout.teacher_pool_dir(
                    root, size, rung.key, plan.TEACHER_LADDER_FTSEED
                )
                / plan.PROBE_SUBDIR
                / plan.PROBE_REPORT_FILE
            )
            keep = probe.get("keep_rate") if probe else None
            projected = None
            if isinstance(keep, (int, float)) and keep > 0:
                chosen = numbers.choose_n_prompts(float(keep))
                projected = (
                    numbers.projected_rows(chosen, float(keep)) if chosen else None
                )
            out.append(
                selection.RungReading(
                    size=size,
                    rung_tag=rung.key,
                    family=rung.family,
                    epochs=rung.epochs,
                    p_target=_num(validation.get("p_target_mean")),
                    control_mean=_num(validation.get("base_p_target")),
                    control_sd=None,
                    ppl_ratio=_perplexity_ratio(param_dir),
                    off_topic=_num(off_topic.get("p_target_mean")),
                    probe_keep=_num(keep),
                    projected_rows=projected,
                )
            )
    return out


def _teacher_gate(cfg: CampaignDriverConfig, state: dict) -> None:
    """Pick each size's teacher rung, persist the report, then act on it.

    Raises:
        CampaignGateError: No size has an admissible teacher, so no pair can be taught.
    """
    if cfg.dry_run:
        readings: list[selection.RungReading] = []
        decisions = {
            size: selection.DoseDecision(
                size, selection.STATUS_PICKED, _DRY_RUNG, None, ()
            )
            for size in plan.SIZES
        }
    else:
        readings = teacher_readings(cfg)
        decisions = selection.dose_gate(readings)
    report = selection.gate_report(decisions, readings)
    report["at"] = _now()
    driver_io.write_logs_report(cfg, plan.TEACHER_GATE_REPORT, report)
    _write_ladder_csv(cfg, report)
    state["transitions"][T_TEACHER_GATE] = {
        "at": _now(),
        "picked": {s: d.rung_tag for s, d in decisions.items()},
    }
    voided = [s for s, d in decisions.items() if d.rung_tag is None]
    state["voided_sizes"] = voided
    if len(voided) == len(plan.SIZES):
        raise CampaignGateError(
            "no size has an admissible teacher; every rung failed the trait, "
            "capability or format-probe bound — see teacher_gate_report.json"
        )
    for size in voided:
        logger.warning(
            "{}: no admissible teacher — voided as a teacher, kept as a student", size
        )


def _write_ladder_csv(cfg: CampaignDriverConfig, report: Mapping[str, object]) -> None:
    rows = report.get("ladder")
    if not isinstance(rows, list) or cfg.dry_run:
        return
    columns = [
        "size",
        "rung_tag",
        "family",
        "epochs",
        "p_target",
        "control_mean",
        "ppl_ratio",
        "off_topic",
        "probe_keep",
        "projected_rows",
        "admitted",
        "off_topic_risk_band",
        "reasons",
    ]
    lines = [",".join(columns)]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        values = []
        for col in columns:
            value = row.get(col)
            if isinstance(value, list):
                value = ";".join(str(v) for v in value)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.logs_dir / plan.TEACHER_LADDER_CSV, "\n".join(lines) + "\n")


# The rung a `--dry-run` walk pretends the gate picked, so `advance --dry-run` reaches P5
# on a machine with no campaign artifacts at all.
_DRY_RUNG = "ep10_lr1e-4_r16a32"


def _gate_rungs(cfg: CampaignDriverConfig) -> dict[str, str]:
    """`size -> picked rung tag`, read back from the persisted gate report.

    Raises:
        CampaignGateError: The report is missing (the gate has not fired).
    """
    if cfg.dry_run:
        return {size: _DRY_RUNG for size in plan.SIZES}
    doc = _read_json(cfg.logs_dir / plan.TEACHER_GATE_REPORT)
    if doc is None:
        raise CampaignGateError(
            f"{plan.TEACHER_GATE_REPORT} missing; the gate has not fired"
        )
    return selection.rungs_from_report(doc)


def _submit_p1b(
    cfg: CampaignDriverConfig, state: dict, rungs: Mapping[str, str]
) -> None:
    """Queue teacher seeds 0 and 2 at the picked rungs, plus their eval refresh."""
    seed_jobs = jobs.seedfill_jobs(rungs)
    if seed_jobs:
        driver_io.submit_queue(cfg, plan.Q_P1_SEEDFILL, seed_jobs)
        driver_io.submit_queue(
            cfg,
            plan.Q_P1_SEEDFILL_EVALS,
            jobs.seedfill_eval_jobs(rungs),
            depends_on=[plan.Q_P1_SEEDFILL],
        )
    state["transitions"][T_SUBMIT_P1B] = {"at": _now(), "n_jobs": len(seed_jobs)}


# ---- the manual pause ----------------------------------------------------------


def _expected_prompts_hash(n_raw: int) -> str:
    """Angela's prompt set rebuilt at `n_raw`, hashed the way her manifest hashes it.

    The hash is a function of the prompt-set knobs AND the prompt count, so a pool
    generated at 200k is verified against a 200k rebuild rather than her 100k literal.
    """
    from pipelines.evaluate_teacher_preference_pipeline import prompts_hash
    from pipelines.generate_numbers_pipeline import load_number_prompt_set
    from subliminal_learning.numbers.prompt_gen import (
        build_number_prompts,
        load_intro_templates,
    )

    prompt_set = load_number_prompt_set(plan.NUMBER_CONFIG)
    templates = load_intro_templates(prompt_set.templates_path)
    built = build_number_prompts(
        templates,
        n_prompts=n_raw,
        seed=prompt_set.seed,
        example_count_min=prompt_set.example_count_min,
        example_count_max=prompt_set.example_count_max,
        example_value_min=prompt_set.example_value_min,
        example_value_max=prompt_set.example_value_max,
        separator_weights=prompt_set.separator_weights,
    )
    return prompts_hash([p.text for p in built])


def pool_statuses(
    cfg: CampaignDriverConfig, rungs: Mapping[str, str]
) -> list[numbers.PoolStatus]:
    """Verify every required pool on disk against the gate's contract."""
    from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

    root = campaign_root(cfg)
    out: list[numbers.PoolStatus] = []
    for spec in numbers.required_pools(rungs):
        if spec.kind == numbers.KIND_RNG:
            pool = layout.rng_pool_dir(root)
            repo = None
            adapter = None
        elif spec.kind == numbers.KIND_BASE:
            pool = layout.base_pool_dir(root, str(spec.size))
            repo = plan.BASE_REPO[str(spec.size)]
            adapter = None
        else:
            pool = layout.teacher_pool_dir(
                root, str(spec.size), str(spec.rung_tag), int(spec.ftseed or 0)
            )
            repo = plan.BASE_REPO[str(spec.size)]
            rung = next(r for r in plan.TEACHER_RUNGS if r.key == spec.rung_tag)
            adapter = adapter_identity(
                layout.teacher_leaf_dir(
                    root, rung, str(spec.size), int(spec.ftseed or 0)
                )
                / "checkpoints"
            )
        raw = _read_json(pool / "manifest.json")
        mode = _read_json(layout.filtered_dir(pool) / "manifest.json")
        filtered = layout.filtered_dir(pool) / "filtered.jsonl"
        sha = file_sha256(filtered) if filtered.is_file() else None
        n_raw = raw.get("n_raw") if isinstance(raw, Mapping) else None
        expected_hash = (
            _expected_prompts_hash(int(n_raw))
            if isinstance(n_raw, (int, float))
            else None
        )
        out.append(
            numbers.evaluate_pool(
                spec=spec,
                raw_manifest=raw,
                mode_manifest=mode,
                filtered_sha256=sha,
                expected_adapter_id=adapter,
                expected_repo=repo,
                expected_prompts_hash=expected_hash,
            )
        )
    return out


def _numbers_gate(
    cfg: CampaignDriverConfig, state: dict, rungs: Mapping[str, str], waiting: list[str]
) -> bool:
    """Wait (never raise) until every required pool verifies, then freeze the report."""
    if cfg.dry_run:
        state["transitions"][T_NUMBERS_READY] = {"at": _now(), "dry_run": True}
        return True
    statuses = pool_statuses(cfg, rungs)
    report = numbers.gate_report(statuses)
    if not report["ready"]:
        missing = [s for s in statuses if not s.ready]
        waiting.append(
            f"numbers: {len(missing)}/{len(statuses)} pool(s) not ready — run "
            "`uv run python -m scripts.offxsize_numbers plan` for the exact commands"
        )
        for status in missing:
            logger.info("  pool {} -> {}", status.spec.key, ", ".join(status.reasons))
        _write_qc(cfg, statuses)
        return False
    report["at"] = _now()
    driver_io.write_logs_report(cfg, plan.NUMBERS_GATE_REPORT, report)
    _write_qc(cfg, statuses)
    state["transitions"][T_NUMBERS_READY] = {"at": _now(), "n_pools": len(statuses)}
    return True


def _write_qc(
    cfg: CampaignDriverConfig, statuses: Sequence[numbers.PoolStatus]
) -> None:
    if cfg.dry_run:
        return
    rows = [numbers.qc_row(s) for s in statuses]
    columns = list(rows[0]) if rows else []
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join("" if row[c] is None else str(row[c]) for c in columns))
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(cfg.logs_dir / plan.NUMBERS_QC_CSV, "\n".join(lines) + "\n")


def _frozen_numbers(cfg: CampaignDriverConfig) -> tuple[dict[str, str], dict[str, str]]:
    """`(dataset tags, filtered shas)` from the frozen numbers gate report.

    Raises:
        CampaignGateError: The report is missing.
    """
    if cfg.dry_run:
        rungs = {size: _DRY_RUNG for size in plan.SIZES}
        return (
            {
                spec.key: f"{plan.PROMPT_SET_TAG}_{plan.FILTER_MODE}_n100k"
                for spec in numbers.required_pools(rungs)
            },
            {},
        )
    doc = _read_json(cfg.logs_dir / plan.NUMBERS_GATE_REPORT)
    if doc is None:
        raise CampaignGateError(
            f"{plan.NUMBERS_GATE_REPORT} missing; the pause has not cleared"
        )
    tags = numbers.dataset_tags_from_report(doc)
    pools = doc.get("pools")
    shas: dict[str, str] = {}
    if isinstance(pools, Mapping):
        for key, entry in pools.items():
            if isinstance(entry, Mapping) and entry.get("filtered_sha256"):
                shas[str(key)] = str(entry["filtered_sha256"])
    return tags, shas


# ---- P2 / P3 ------------------------------------------------------------------


def _submit_p2(
    cfg: CampaignDriverConfig,
    state: dict,
    size: str,
    rungs: Mapping[str, str],
    tags: Mapping[str, str],
    shas: Mapping[str, str],
) -> None:
    """Queue one size's star screen, its controls and its capability evals."""
    leaves = pairs.screen_leaves(size, rungs)
    treatments = [leaf for leaf in leaves if leaf.kind == pairs.TREATMENT]
    controls = [leaf for leaf in leaves if leaf.kind != pairs.TREATMENT]
    root = campaign_root(cfg)
    driver_io.submit_queue(
        cfg,
        plan.q_p2_screen(size),
        jobs.student_jobs(root, treatments, tags=tags, shas=shas),
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p2_controls(size),
        jobs.student_jobs(root, controls, tags=tags, shas=shas),
    )
    # Every capability cell in a tree READS one `base_metrics.json` that the warm queue
    # fills. Cells filling it concurrently is the read-modify-write race the eval cell
    # refuses on, so the evals wait for the warm queue instead of racing it.
    driver_io.submit_queue(
        cfg,
        plan.q_p2_warm(size),
        jobs.warm_eval_base_jobs(root, leaves, tags=tags),
        depends_on=[plan.q_p2_screen(size), plan.q_p2_controls(size)],
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p2_evals(size),
        jobs.capability_jobs(
            root, leaves, tags=tags, benchmarks=False, queue_tag="p2e"
        ),
        depends_on=[plan.q_p2_warm(size)],
    )
    state["transitions"][t_submit_p2(size)] = {"at": _now(), "n_cells": len(leaves)}


def screen_rows(
    cfg: CampaignDriverConfig,
    size: str,
    rungs: Mapping[str, str],
    tags: Mapping[str, str],
) -> list[selection.ScreenRow]:
    """One row per screen config: delta against its matched control, and student PPL."""
    root = campaign_root(cfg)
    leaves = pairs.screen_leaves(size, rungs)
    treatments = {
        leaf.recipe.key: leaf for leaf in leaves if leaf.kind == pairs.TREATMENT
    }
    controls = {
        leaf.recipe.key: leaf for leaf in leaves if leaf.kind == pairs.BASE_CONTROL
    }
    rows: list[selection.ScreenRow] = []
    for key, leaf in treatments.items():
        treated = _leaf_p_target(root, leaf, tags)
        control = controls.get(key)
        base = _leaf_p_target(root, control, tags) if control is not None else None
        delta = None if treated is None or base is None else treated - base
        rows.append(
            selection.ScreenRow(
                recipe_key=key,
                delta_sb=delta,
                ppl_ratio=_perplexity_ratio(expected.param_dir(root, leaf, tags)),
            )
        )
    return rows


def _leaf_p_target(
    root: Path, leaf: pairs.Leaf | None, tags: Mapping[str, str]
) -> float | None:
    """Primary-endpoint `p_target` for one trained leaf."""
    if leaf is None:
        return None
    doc = _read_json(expected.leaf_dir(root, leaf, tags) / expected.PREFERENCE_EVAL)
    if not doc:
        return None
    entry = (doc.get("sets") or {}).get(plan.PRIMARY_PROMPT_SET)
    if not isinstance(entry, Mapping):
        return None
    return _num(entry.get("p_target"))


def _pick_and_submit(
    cfg: CampaignDriverConfig,
    state: dict,
    size: str,
    rungs: Mapping[str, str],
    tags: Mapping[str, str],
    shas: Mapping[str, str],
) -> None:
    """Persist one size's recipe pick, then queue its headline (and tails)."""
    picks_path = cfg.logs_dir / plan.SCREEN_PICKS
    stored = _read_json(picks_path) or {}
    if cfg.dry_run and size not in stored:
        stored[size] = {
            "recipe_key": plan.STUDENT_CENTRE.key,
            "flagged": False,
            "runner_up_keys": [],
            "reasons": [],
        }
    if size not in stored:
        pick = selection.pick_recipe(size, screen_rows(cfg, size, rungs, tags))
        stored[size] = selection.picks_report({size: pick})[size]
        driver_io.write_logs_report(cfg, plan.SCREEN_PICKS, stored)
        logger.info(
            "recipe pick {} -> {} (flagged={})", size, pick.recipe_key, pick.flagged
        )
    recipe = selection.recipe_by_key(size, str(stored[size]["recipe_key"]))
    runner_ups = tuple(
        selection.recipe_by_key(size, str(k))
        for k in stored[size].get("runner_up_keys", [])
    )
    root = campaign_root(cfg)

    leaves = pairs.headline_leaves(size, recipe, rungs)
    treatments = [leaf for leaf in leaves if leaf.kind == pairs.TREATMENT]
    controls = [leaf for leaf in leaves if leaf.kind != pairs.TREATMENT]
    driver_io.submit_queue(
        cfg,
        plan.q_p3_students(size),
        jobs.student_jobs(root, treatments, tags=tags, shas=shas),
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p3_controls(size),
        jobs.student_jobs(root, controls, tags=tags, shas=shas),
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p4_warm(size),
        jobs.warm_eval_base_jobs(root, leaves, tags=tags),
        depends_on=[plan.q_p3_students(size), plan.q_p3_controls(size)],
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p4_evals(size),
        jobs.capability_jobs(root, leaves, tags=tags, benchmarks=False),
        depends_on=[plan.q_p4_warm(size)],
    )
    driver_io.submit_queue(
        cfg,
        plan.q_p6_benchmarks(size),
        jobs.capability_jobs(root, leaves, tags=tags, benchmarks=True, queue_tag="p6"),
        depends_on=[plan.q_p4_evals(size)],
    )

    if size == plan.P3C_SIZE and size in rungs:
        dose = pairs.teacher_dose_leaves(recipe, rungs[size])
        if dose:
            driver_io.submit_queue(
                cfg, plan.Q_P3C, jobs.student_jobs(root, dose, tags=tags, shas=shas)
            )
            driver_io.submit_queue(
                cfg,
                plan.Q_P3C_WARM,
                jobs.warm_eval_base_jobs(root, dose, tags=tags),
                depends_on=[plan.Q_P3C],
            )
            driver_io.submit_queue(
                cfg,
                plan.Q_P3C_EVALS,
                jobs.capability_jobs(root, dose, tags=tags, queue_tag="p3ce"),
                depends_on=[plan.Q_P3C_WARM, plan.q_p4_evals(size)],
            )

    if size in plan.P3B_SIZES and runner_ups:
        tail = pairs.robustness_leaves(size, runner_ups, rungs)
        if tail:
            driver_io.submit_queue(
                cfg,
                plan.q_p3b(size),
                jobs.student_jobs(root, tail, tags=tags, shas=shas),
            )
            driver_io.submit_queue(
                cfg,
                plan.q_p3b_evals(size),
                jobs.capability_jobs(root, tail, tags=tags, queue_tag="p3be"),
                depends_on=[plan.q_p3b(size), plan.q_p4_evals(size)],
            )
    state["transitions"][t_pick(size)] = {"at": _now(), "recipe": recipe.key}
    state["transitions"][t_submit_p3(size)] = {"at": _now(), "n_cells": len(leaves)}


# ---- finalize -----------------------------------------------------------------


def _required_queue_names(cfg: CampaignDriverConfig) -> list[str]:
    """Submitted queues that `CAMPAIGN_DONE` waits on (the optional tails excluded)."""
    optional = set(plan.optional_queue_names())
    return [
        name
        for name in store.list_queues(cfg.queue_root)
        if name.startswith("offxsize-") and name not in optional
    ]


def _finalize(cfg: CampaignDriverConfig, state: dict) -> None:
    """Rebuild every result artifact and mark the campaign complete."""
    from pipelines.offxsize_results_pipeline import build_results

    build_results(cfg, dry_run=cfg.dry_run)
    if not cfg.dry_run:
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, _now() + "\n")
    state["transitions"][T_FINALIZE] = {"at": _now()}


# ---- the pass -----------------------------------------------------------------


def run(cfg: CampaignDriverConfig) -> AdvanceResult:
    """Advance the campaign as far as its gates allow. Idempotent.

    Raises:
        CampaignGateError: A queue failed past the requeue cap, or no size has an
            admissible teacher.
        ValueError: `cfg` is not bound to this campaign.
    """
    if not is_valid_run_set_id(cfg.run_set_id):
        raise ValueError(f"invalid run set id {cfg.run_set_id!r}")
    if cfg.spec is not plan.SPEC:
        raise ValueError("driver config must be bound to offxsize_plan.SPEC")

    state = driver_io.load_state(cfg)
    state.setdefault("transitions", {})
    done_marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    fired: list[str] = []
    waiting: list[str] = []
    transitions = state["transitions"]

    def _fire(name: str, fn) -> None:
        fn()
        fired.append(name)
        driver_io.save_state(cfg, state)

    if T_PREFLIGHT not in transitions:
        _fire(T_PREFLIGHT, lambda: _preflight(cfg, state))

    if T_PREFLIGHT in transitions and T_SUBMIT_P1A not in transitions:
        _fire(T_SUBMIT_P1A, lambda: _submit_p1a(cfg, state))

    p1a_queues = [
        plan.Q_P1_TEACHERS,
        plan.Q_P1_TEACHER_EVALS,
        plan.Q_P1_PROBES,
        plan.Q_P1_BASE_PRIORS,
    ]
    if T_SUBMIT_P1A in transitions and T_TEACHER_GATE not in transitions:
        if _queues_ready(cfg, p1a_queues, waiting):
            _fire(T_TEACHER_GATE, lambda: _teacher_gate(cfg, state))

    if T_TEACHER_GATE in transitions and T_SUBMIT_P1B not in transitions:
        rungs = _gate_rungs(cfg)
        _fire(T_SUBMIT_P1B, lambda: _submit_p1b(cfg, state, rungs))

    if T_SUBMIT_P1B in transitions and T_NUMBERS_READY not in transitions:
        rungs = _gate_rungs(cfg)
        seed_queues = [
            q
            for q in (plan.Q_P1_SEEDFILL, plan.Q_P1_SEEDFILL_EVALS)
            if store.queue_exists(cfg.queue_root, q)
        ]
        if _queues_ready(cfg, seed_queues, waiting):
            if _numbers_gate(cfg, state, rungs, waiting):
                fired.append(T_NUMBERS_READY)
                driver_io.save_state(cfg, state)

    if T_NUMBERS_READY in transitions:
        rungs = _gate_rungs(cfg)
        tags, shas = _frozen_numbers(cfg)
        for size in plan.SIZES:
            if t_submit_p2(size) not in transitions:
                _fire(
                    t_submit_p2(size),
                    lambda s=size: _submit_p2(cfg, state, s, rungs, tags, shas),
                )
        for size in plan.SIZES:
            if t_submit_p3(size) in transitions:
                continue
            screen = pairs.screen_leaves(size, rungs)
            label = f"p2 {size}"
            if not _queues_ready(cfg, [plan.q_p2_evals(size)], waiting):
                continue
            if not _leaves_ready(cfg, screen, tags, waiting, label):
                continue
            _fire(
                t_submit_p3(size),
                lambda s=size: _pick_and_submit(cfg, state, s, rungs, tags, shas),
            )

    if T_NUMBERS_READY in transitions and T_FINALIZE not in transitions:
        required = _required_queue_names(cfg)
        submitted_p3 = all(t_submit_p3(size) in transitions for size in plan.SIZES)
        if submitted_p3 and _queues_ready(cfg, required, waiting):
            _fire(T_FINALIZE, lambda: _finalize(cfg, state))

    return AdvanceResult(tuple(fired), tuple(waiting), done_marker.is_file())


def status(cfg: CampaignDriverConfig) -> dict:
    """Read-only snapshot for the CLI."""
    state = driver_io.load_state(cfg)
    gate = _read_json(cfg.logs_dir / plan.TEACHER_GATE_REPORT) or {}
    picks = _read_json(cfg.logs_dir / plan.SCREEN_PICKS) or {}
    pools = _read_json(cfg.logs_dir / plan.NUMBERS_GATE_REPORT) or {}
    queues = {}
    for name in sorted(plan.QUEUE_PRIORITIES):
        if store.queue_exists(cfg.queue_root, name):
            queues[name] = store.queue_status(cfg.queue_root, name)
    return {
        "run_set_id": cfg.run_set_id,
        "campaign_root": str(campaign_root(cfg)),
        "transitions": sorted(state.get("transitions", {})),
        "teacher_rungs": selection.rungs_from_report(gate),
        "voided_sizes": state.get("voided_sizes", []),
        "numbers_ready": bool(pools.get("ready")),
        "screen_picks": {k: v.get("recipe_key") for k, v in picks.items()},
        "queues": queues,
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _now() -> str:
    """UTC timestamp for state and report entries."""
    return utc_now_iso()
