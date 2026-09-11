"""Driver for the off-policy cross-size campaign v2 (`20260913_offxsize2-doglion_mixed`).

One `run(cfg)` is one idempotent pass over:

    preflight          -> staging artifacts + the plan's own footprint invariants
    submit_p0          -> teachers, their evals, the format probes, the base priors
    teacher_gate       -> admission per (animal, size, rung); report first, then act
    submit_pools       -> one generation cell per pool the admitted rungs need
    numbers_ready      -> freeze each pool's honest-n dataset tag and `filtered_sha256`
    submit_concordance -> Angela's own rung AND recipe at 160m -> 160m
    concordance_gate   -> STOP unless it reproduces `CONCORDANCE_P_TARGET`
    submit_p1_{size}   -> the factorial column, its matched controls, its capability
    confirm_pick       -> `pick_confirm` over the factorial; persisted, then read back
    submit_p2_{size}   -> the confirm column, both animals
    admit_tail_{tier}  -> elastic filler, while the pool is idle and the budget allows
    finalize           -> results + claims.json + CAMPAIGN_DONE
    finalize_partial   -> deadline path: drain, CSVs, coverage, no CAMPAIGN_DONE

Every gate writes its decision before acting on it and reads it back on later passes, so
a re-advance rebuilds an identical jobkey set.

The campaign runs on `plan.CAMPAIGN_NODE` while soar runs the live cross-family campaign
off the same queue root. `driver_io.submit_queue` pins each queue to `store.this_host()`
and `QueueSubmission.node` is declared intent only, so each submission asserts the two
against each other here: a pass started on `plan.FORBIDDEN_NODE` would otherwise file
this campaign's 600 cells onto the node that is already busy.
"""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import offxsize2_expected as expected
from subliminal_learning.campaign import offxsize2_jobs as jobs
from subliminal_learning.campaign import offxsize2_pairs as pairs
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize2_selection as selection
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign.driver_io import CampaignDriverConfig
from subliminal_learning.campaign.offxsize2_selection import CampaignGateError
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils.io import atomic_write_text, file_sha256
from subliminal_learning.utils.run_io import utc_now_iso

T_PREFLIGHT = "preflight"
T_SUBMIT_P0 = "submit_p0"
T_TEACHER_GATE = "teacher_gate"
T_SUBMIT_POOLS = "submit_pools"
T_NUMBERS_READY = "numbers_ready"
T_SUBMIT_CONCORDANCE = "submit_concordance"
T_CONCORDANCE_GATE = "concordance_gate"
T_CONFIRM_PICK = "confirm_pick"
T_FINALIZE = "finalize"
T_FINALIZE_PARTIAL = "finalize_partial"

_P0_QUEUES: tuple[str, ...] = (
    plan.Q_P0_TEACHERS,
    plan.Q_P0_TEACHER_EVALS,
    plan.Q_P0_PROBES,
    plan.Q_P0_BASE_PRIORS,
)


def t_submit_p1(size: str) -> str:
    """Transition key for one student column's factorial submission."""
    return f"submit_p1_{size}"


def t_submit_p2(size: str) -> str:
    """Transition key for one student column's confirm submission."""
    return f"submit_p2_{size}"


def t_admit_tail(tier: str) -> str:
    """Transition key for one elastic-tail tier's admission."""
    return f"admit_tail_{tier}"


@dataclass(frozen=True)
class AdvanceResult:
    """What one `advance` pass did.

    Attributes:
        fired: Transitions that ran in this pass.
        waiting: Human-readable reasons the campaign is not finished.
        done: Whether a finalize (full or partial) has run.
    """

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class Offxsize2DriverOptions:
    """Autonomy knobs the driver reads.

    Attributes:
        deadline_hours: Open-window budget for this `run` / `watch`.
        drain_timeout_s: Max wait for in-flight cells once the deadline pauses the pool.
        max_failed_frac: Failed/total at or below this is a recorded hole; above it is a
            hard gate.
        n_gpus: Cards the pool may fill; the ceiling on measured tail capacity.
    """

    deadline_hours: float = plan.DEADLINE_HOURS_DEFAULT
    drain_timeout_s: int = plan.DRAIN_TIMEOUT_S
    max_failed_frac: float = plan.FAILED_FRACTION_MAX
    n_gpus: int = len(plan.GPU_POLICY["gpus"])


def driver_config(**kwargs: object) -> CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to this campaign's spec."""
    kwargs.setdefault("spec", plan.SPEC)
    return CampaignDriverConfig(**kwargs)  # type: ignore[arg-type]


def campaign_root(cfg: CampaignDriverConfig) -> Path:
    """Root of every artifact this campaign writes."""
    return cfg.run_set_root


# ---- node binding -------------------------------------------------------------


def require_campaign_node(intended: str = plan.CAMPAIGN_NODE) -> str:
    """Refuse to submit from any host but the campaign node.

    `QueueSubmission.node` records where a queue is MEANT to run;
    `driver_io.submit_queue` pins it to `store.this_host()`. Only comparing the two
    stops a pass started in the wrong clone from filing this campaign's work onto the
    node running the live cross-family campaign, which shares this queue root.

    Args:
        intended: Host the submission declares (always `plan.CAMPAIGN_NODE`).

    Returns:
        The verified hostname.

    Raises:
        CampaignGateError: This host is the forbidden node, or is not `intended`.
    """
    host = socket.gethostname()
    if host == plan.FORBIDDEN_NODE:
        raise CampaignGateError(
            f"refusing to submit {plan.RUN_SET_ID} from {host}: that node runs the "
            "live cross-family campaign off this queue root, and jobq pins a queue to "
            f"the submitting host. Run this driver on {plan.CAMPAIGN_NODE}."
        )
    if host != intended:
        raise CampaignGateError(
            f"{plan.RUN_SET_ID} is pinned to {intended}, but this pass is running on "
            f"{host}; jobq would file every queue under {host}"
        )
    return host


def _submit(
    cfg: CampaignDriverConfig,
    name: str,
    job_list: Sequence[Job],
    *,
    depends_on: Sequence[str] = (),
) -> str:
    """Check one queue, assert the node it declares, then submit it."""
    submission = jobs.submission(name, list(job_list), depends_on=depends_on)
    require_campaign_node(submission.node)
    return driver_io.submit_queue(
        cfg,
        submission.name,
        list(submission.jobs),
        depends_on=list(submission.depends_on) or None,
    )


# ---- readiness ----------------------------------------------------------------


def _record_hole(holes: list[dict], record: dict) -> None:
    if any(h.get("queue") == record.get("queue") for h in holes):
        return
    holes.append(record)


def _failed_jobkeys(cfg: CampaignDriverConfig, name: str) -> list[str]:
    return [
        job.jobkey
        for job in store.load_jobs(cfg.queue_root, name)
        if store.job_state(cfg.queue_root, name, job) == store.FAILED
    ]


def _queue_ready(
    cfg: CampaignDriverConfig,
    name: str,
    opts: Offxsize2DriverOptions,
    holes: list[dict],
) -> bool:
    """Whether `name` is terminal, with bounded failure holes.

    A failed fraction at or below `opts.max_failed_frac` is recorded as a hole and the
    queue counts as complete-with-holes; above it the campaign stops. An elastic-tail
    queue is droppable by design, so its failures are always holes.

    Raises:
        CampaignGateError: A committed queue failed past the cap.
    """
    if not store.queue_exists(cfg.queue_root, name):
        return False
    cap = 1.0 if name in plan.OPTIONAL_QUEUES else opts.max_failed_frac
    try:
        driver_io.requeue_failed_under_cap(cfg, name)
    except RuntimeError as err:
        status = store.queue_status(cfg.queue_root, name)
        frac = int(status["failed"]) / max(int(status["total"]), 1)
        if frac > cap:
            raise CampaignGateError(f"{name}: {frac:.1%} failed > cap; {err}") from err
    if not store.queue_complete(cfg.queue_root, name):
        return False
    status = store.queue_status(cfg.queue_root, name)
    failed = int(status["failed"])
    if failed:
        frac = failed / max(int(status["total"]), 1)
        if frac > cap:
            raise CampaignGateError(f"{name}: {frac:.1%} failed > cap after requeue")
        _record_hole(
            holes,
            {
                "queue": name,
                "failed": failed,
                "total": int(status["total"]),
                "jobkeys": _failed_jobkeys(cfg, name),
            },
        )
        logger.warning("tolerating {} failed cell(s) in {}", failed, name)
    return True


def _queues_ready(
    cfg: CampaignDriverConfig,
    names: Sequence[str],
    opts: Offxsize2DriverOptions,
    waiting: list[str],
    holes: list[dict],
) -> bool:
    """Whether every named queue is terminal."""
    if cfg.dry_run:
        return True
    ready = True
    for name in names:
        if not _queue_ready(cfg, name, opts, holes):
            waiting.append(f"{name}: not ready")
            ready = False
    return ready


def _leaves_ready(
    cfg: CampaignDriverConfig,
    leaves: Sequence[pairs.Leaf],
    tags: Mapping[str, str],
    waiting: list[str],
    label: str,
) -> bool:
    """Whether every cell is trusted-done and scored once per animal it serves."""
    if cfg.dry_run:
        return True
    root = campaign_root(cfg)
    untrained = expected.missing_manifests(root, leaves, tags)
    if untrained:
        waiting.append(f"{label}: {len(untrained)} cell(s) not trained")
        return False
    unscored = expected.missing_evals(root, leaves, tags)
    if unscored:
        waiting.append(f"{label}: {len(unscored)} eval(s) missing")
        return False
    return True


def _live_queue_names(cfg: CampaignDriverConfig) -> list[str]:
    """Every queue of THIS campaign that exists in the shared root."""
    prefix = f"{plan.QUEUE_PREFIX}-"
    return [
        name
        for name in sorted(store.list_queues(cfg.queue_root))
        if name.startswith(prefix)
    ]


def _required_queue_names(cfg: CampaignDriverConfig) -> list[str]:
    """Submitted queues `CAMPAIGN_DONE` waits on (the optional tails excluded)."""
    return [n for n in _live_queue_names(cfg) if n not in plan.OPTIONAL_QUEUES]


# ---- small readers ------------------------------------------------------------


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        logger.warning("unreadable JSON at {}", path)
        return None


def _num(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _now() -> str:
    """UTC timestamp for state and report entries."""
    return utc_now_iso()


def _parse_utc(stamp: str) -> datetime:
    """Parse either `utc_now_iso` (`...Z`) or a worker result's offset ISO stamp."""
    return datetime.fromisoformat(str(stamp))


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


def _set_p_target(doc: Mapping[str, object] | None, prompt_set: str) -> float | None:
    """`sets[prompt_set].p_target` from a preference eval or base-prior report."""
    if not doc:
        return None
    entry = (doc.get("sets") or {}).get(prompt_set)
    return _num(entry.get("p_target")) if isinstance(entry, Mapping) else None


def _leaf_p_target(
    root: Path, leaf: pairs.Leaf | None, tags: Mapping[str, str], animal: str
) -> float | None:
    """Primary-endpoint `p_target` of one trained cell, read for `animal`."""
    if leaf is None:
        return None
    doc = _read_json(expected.preference_path(root, leaf, tags, animal))
    return _set_p_target(doc, plan.PRIMARY_PROMPT_SET)


def _base_prior(cfg: CampaignDriverConfig, animal: str, size: str) -> float | None:
    """The untuned base's own `p(" {animal}")` at one size, from the P0 prior cell."""
    path = cfg.logs_dir / plan.BASE_PRIORS_DIRNAME / animal / f"{size}.json"
    return _set_p_target(_read_json(path), plan.PRIMARY_PROMPT_SET)


def _recipe_by_key(key: str) -> plan.StudentRecipe:
    """The committed recipe a `{student_tag}` names.

    Raises:
        CampaignGateError: The tag is not on the frozen grid, so a persisted pick names
            a cell this campaign never enumerated.
    """
    for recipe in (*plan.STUDENT_GRID, plan.CONCORDANCE_RECIPE):
        if recipe.key == key:
            return recipe
    raise CampaignGateError(f"recipe {key!r} is not on the campaign grid")


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """One flat CSV from uniform records (list values joined with `;`)."""
    if not rows:
        return
    columns = list(rows[0])
    lines = [",".join(columns)]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, (list, tuple)):
                value = ";".join(str(v) for v in value)
            values.append("" if value is None else str(value))
        lines.append(",".join(values))
    atomic_write_text(path, "\n".join(lines) + "\n")


# ---- preflight ----------------------------------------------------------------


def _preflight(cfg: CampaignDriverConfig, state: dict) -> None:
    """Require the out-of-band staging artifacts and the plan's packing invariants.

    Raises:
        CampaignGateError: Staging has not run (or did not finish).
        ValueError: A footprint oversubscribes a card (`plan.validate_footprints`).
    """
    plan.validate_footprints()
    for filename in (plan.PREFLIGHT_REPORT, plan.ADOPTION_MANIFEST):
        path = cfg.logs_dir / filename
        if not cfg.dry_run and not path.is_file():
            raise CampaignGateError(
                f"P0 missing {filename} at {path} — run "
                "`uv run python -m scripts.stage_offxsize2_assets` first"
            )
    state["transitions"][T_PREFLIGHT] = {"at": _now()}


# ---- P0: teachers, probes, priors ---------------------------------------------


def _submit_p0(cfg: CampaignDriverConfig, state: dict) -> None:
    """Queue every teacher, its eval, its format probe, and the base priors.

    Both animals: P2 confirms lion as well as dog, and a teacher trained after the
    confirm pick would arrive after the cells that need it. The base probes ride the
    same queue because the untuned pools are every Delta's denominator and are sized
    from a measured keep rate exactly like a teacher's.
    """
    teacher_jobs: list = []
    eval_jobs: list = []
    probe_jobs: list = []
    for animal in plan.ANIMALS:
        teacher_jobs += jobs.teacher_jobs(animal=animal)
        eval_jobs += jobs.teacher_eval_jobs(animal=animal)
        probe_jobs += jobs.probe_jobs(animal=animal)
    probe_jobs += jobs.base_probe_jobs()
    _submit(cfg, plan.Q_P0_TEACHERS, teacher_jobs)
    _submit(cfg, plan.Q_P0_TEACHER_EVALS, eval_jobs, depends_on=[plan.Q_P0_TEACHERS])
    _submit(cfg, plan.Q_P0_PROBES, probe_jobs, depends_on=[plan.Q_P0_TEACHERS])
    _submit(cfg, plan.Q_P0_BASE_PRIORS, jobs.base_prior_jobs())
    state["transitions"][T_SUBMIT_P0] = {
        "at": _now(),
        "n_teachers": len(teacher_jobs),
        "n_probes": len(probe_jobs),
    }


def teacher_readings(
    cfg: CampaignDriverConfig,
) -> dict[tuple[str, str], list[selection.RungReading]]:
    """Every rung's trait, capability and format probe, keyed `(animal, size)`.

    The base control's across-prompt SD is not carried in the summary, so the SD band is
    evaluated with SD 0; at a prior near 0.003 against a saturated teacher it is the
    absolute margin and the probe that bind.
    """
    root = campaign_root(cfg)
    out: dict[tuple[str, str], list[selection.RungReading]] = {}
    for animal in plan.ANIMALS:
        for size in plan.SIZES:
            readings: list[selection.RungReading] = []
            for rung in plan.TEACHER_RUNGS:
                param_dir = layout.teacher_dir(root, rung, size, animal=animal)
                summary = _read_json(param_dir / expected.PREFERENCE_SUMMARY)
                validation = _summary_entry(summary, "validation")
                off_topic = _summary_entry(summary, "off_topic")
                probe = _read_json(
                    layout.teacher_pool_dir(
                        root, size, rung.key, plan.HEADLINE_TFS, animal=animal
                    )
                    / plan.PROBE_SUBDIR
                    / plan.PROBE_REPORT_FILE
                )
                keep = _num(probe.get("keep_rate")) if probe else None
                readings.append(
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
                        probe_keep=keep,
                        projected_rows=numbers.probe_projection(keep),
                    )
                )
            out[(animal, size)] = readings
    return out


def _gate_report(
    verdicts: Mapping[tuple[str, str, str], selection.RungVerdict],
    readings: Mapping[tuple[str, str], Sequence[selection.RungReading]],
) -> dict:
    """The `teacher_gate_report.json` body: admitted rungs plus their ladder."""
    by_tag = {
        (animal, size, r.rung_tag): r
        for (animal, size), rows in readings.items()
        for r in rows
    }
    ladder = []
    for key in sorted(verdicts):
        animal, size, rung_tag = key
        verdict = verdicts[key]
        reading = by_tag.get(key)
        ladder.append(
            {
                "animal": animal,
                "size": size,
                "rung_tag": rung_tag,
                "family": reading.family if reading else None,
                "epochs": reading.epochs if reading else None,
                "p_target": reading.p_target if reading else None,
                "control_mean": reading.control_mean if reading else None,
                "ppl_ratio": verdict.ppl_ratio,
                "off_topic": reading.off_topic if reading else None,
                "probe_keep": reading.probe_keep if reading else None,
                "projected_rows": reading.projected_rows if reading else None,
                "saturated": verdict.saturated,
                "admitted": verdict.admitted,
                "reasons": list(verdict.reasons),
            }
        )
    admitted = {
        animal: selection.admissible_teacher_sizes(verdicts, animal)
        for animal in plan.ANIMALS
    }
    return {
        "at": _now(),
        "admitted": {
            a: {s: list(t) for s, t in sizes.items()} for a, sizes in admitted.items()
        },
        "ladder": ladder,
    }


def _teacher_gate(cfg: CampaignDriverConfig, state: dict) -> None:
    """Score every rung for admission, persist the report, then act on it.

    v2 never PICKS a rung: the rung is a factor in the cell key. A size with no admitted
    rung is voided as a teacher and keeps running as a student.

    Raises:
        CampaignGateError: No (animal, size) has an admissible rung: nothing can teach.
    """
    if cfg.dry_run:
        report = {
            "at": _now(),
            "admitted": {
                animal: {
                    size: [rung.key for rung in plan.TEACHER_RUNGS]
                    for size in plan.SIZES
                }
                for animal in plan.ANIMALS
            },
            "ladder": [],
        }
    else:
        readings = teacher_readings(cfg)
        report = _gate_report(selection.admit_rungs(readings), readings)
    driver_io.write_logs_report(cfg, plan.TEACHER_GATE_REPORT, report)
    if not cfg.dry_run:
        _write_rows_csv(cfg.logs_dir / plan.TEACHER_LADDER_CSV, report["ladder"])
    admitted = _admitted_from_report(report)
    state["transitions"][T_TEACHER_GATE] = {
        "at": _now(),
        "admitted": {
            a: {s: list(t) for s, t in sizes.items()} for a, sizes in admitted.items()
        },
    }
    voided = [
        f"{animal}/{size}"
        for animal, sizes in admitted.items()
        for size, tags in sizes.items()
        if not tags
    ]
    state["voided_teachers"] = voided
    if len(voided) == len(plan.ANIMALS) * len(plan.SIZES):
        raise CampaignGateError(
            "no (animal, size) has an admissible teacher; every rung failed the trait "
            "or format-probe bound — see teacher_gate_report.json"
        )
    for token in voided:
        logger.warning("{}: no admissible teacher — voided as a teacher", token)


def _admitted_from_report(
    doc: Mapping[str, object],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """`animal -> size -> admitted rung tags` from a frozen gate report."""
    admitted = doc.get("admitted")
    out: dict[str, dict[str, tuple[str, ...]]] = {}
    for animal in plan.ANIMALS:
        sizes = admitted.get(animal) if isinstance(admitted, Mapping) else None
        per_size: dict[str, tuple[str, ...]] = {}
        for size in plan.SIZES:
            tags = sizes.get(size) if isinstance(sizes, Mapping) else None
            per_size[size] = (
                tuple(str(t) for t in tags) if isinstance(tags, list) else ()
            )
        out[animal] = per_size
    return out


def _admitted_rungs(cfg: CampaignDriverConfig) -> dict[str, dict[str, tuple[str, ...]]]:
    """The persisted admission decision.

    Raises:
        CampaignGateError: The report is missing (the gate has not fired).
    """
    doc = _read_json(cfg.logs_dir / plan.TEACHER_GATE_REPORT)
    if doc is None:
        if cfg.dry_run:
            return {
                animal: {
                    size: tuple(rung.key for rung in plan.TEACHER_RUNGS)
                    for size in plan.SIZES
                }
                for animal in plan.ANIMALS
            }
        raise CampaignGateError(
            f"{plan.TEACHER_GATE_REPORT} missing; the teacher gate has not fired"
        )
    return _admitted_from_report(doc)


# ---- P0: number pools ---------------------------------------------------------


def coverage_leaves(
    student: str, admitted: Mapping[str, Mapping[str, Sequence[str]]]
) -> tuple[pairs.Leaf, ...]:
    """One representative leaf per (animal, generator, rung) the committed phases use.

    Neither a pool key nor a capability tree depends on the student recipe or the seed,
    and a pool key does not depend on the student column either, so one enumeration
    answers both "which pools must exist" (read at any column) and "which base-metric
    caches this column must warm". Both animals are covered because P2 confirms lion on
    a rung that is not known until the factorial has been read, while the plan gives the
    pools one queue and each size's warm one queue, both submitted before that.
    """
    recipe = plan.STUDENT_GRID[0]
    out: list[pairs.Leaf] = []
    for animal in plan.ANIMALS:
        for generator in plan.SIZES:
            for rung_tag in admitted.get(animal, {}).get(generator, ()):
                out.append(
                    pairs.Leaf(
                        phase=pairs.PHASE_FACTORIAL,
                        kind=pairs.TREATMENT,
                        animal=animal,
                        teacher=generator,
                        student=student,
                        recipe=recipe,
                        rung_tag=rung_tag,
                        ftseed=plan.HEADLINE_TFS,
                        gseed=plan.HEADLINE_GSEED,
                    )
                )
    for generator in plan.SIZES:
        out.append(
            pairs.Leaf(
                phase=pairs.PHASE_FACTORIAL,
                kind=pairs.BASE_CONTROL,
                animal=plan.PRIMARY_ANIMAL,
                teacher=generator,
                student=student,
                recipe=recipe,
                rung_tag=None,
                ftseed=None,
                gseed=plan.HEADLINE_GSEED,
            )
        )
    out.append(
        pairs.Leaf(
            phase=pairs.PHASE_CONFIRM,
            kind=pairs.RNG_CONTROL,
            animal=plan.PRIMARY_ANIMAL,
            teacher=student,
            student=student,
            recipe=recipe,
            rung_tag=None,
            ftseed=None,
            gseed=plan.HEADLINE_GSEED,
        )
    )
    return tuple(out)


def pool_coverage_leaves(
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[pairs.Leaf, ...]:
    """One leaf per pool the committed phases distil from, at any student column."""
    return coverage_leaves(plan.SIZES[0], admitted)


def probe_keep_rates(
    cfg: CampaignDriverConfig, leaves: Sequence[pairs.Leaf]
) -> dict[str, float]:
    """`pool key -> measured probe keep rate`, read off each pool's frozen probe."""
    root = campaign_root(cfg)
    out: dict[str, float] = {}
    for leaf in expected.pool_leaves(leaves):
        report = _read_json(
            expected.pool_dir(root, leaf) / plan.PROBE_SUBDIR / plan.PROBE_REPORT_FILE
        )
        keep = _num(report.get("keep_rate")) if report else None
        if keep is not None:
            out[leaf.pool_key] = keep
    return out


def _submit_pools(
    cfg: CampaignDriverConfig,
    state: dict,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
) -> None:
    """Queue one generation cell per pool, sized from that pool's own probe keep rate.

    Raises:
        KeyError: A pool has no frozen probe — a guessed count either shorts the
            pool or burns a card (`jobs.pool_n_prompts`).
    """
    root = campaign_root(cfg)
    leaves = pool_coverage_leaves(admitted)
    keep_rates = (
        {leaf.pool_key: 0.5 for leaf in leaves}
        if cfg.dry_run
        else probe_keep_rates(cfg, leaves)
    )
    pool_jobs = jobs.pool_jobs(root, leaves, keep_rates=keep_rates)
    _submit(cfg, plan.Q_P0_POOLS, pool_jobs)
    state["transitions"][T_SUBMIT_POOLS] = {"at": _now(), "n_pools": len(pool_jobs)}


def pool_statuses(
    cfg: CampaignDriverConfig, leaves: Sequence[pairs.Leaf]
) -> list[numbers.PoolStatus]:
    """Verify every generated pool against the numbers contract.

    The prompt-set hash v1 rebuilt is not re-derived here: v2 generates every pool with
    `pool-cell`, which builds its prompts from the campaign's own frozen number config
    and refuses a different prompt-set variant, and an ADOPTED pool is vouched for by
    the staging adoption manifest the preflight demands.
    """
    from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

    root = campaign_root(cfg)
    out: list[numbers.PoolStatus] = []
    for leaf in expected.pool_leaves(leaves):
        spec = expected.pool_spec(leaf)
        pool = expected.pool_dir(root, leaf)
        filtered = expected.filtered_pool_dir(root, leaf)
        adapter = None
        repo = None if leaf.kind == pairs.RNG_CONTROL else plan.BASE_REPO[leaf.teacher]
        if leaf.kind == pairs.TREATMENT:
            rung = next(r for r in plan.TEACHER_RUNGS if r.key == leaf.rung_tag)
            adapter = adapter_identity(
                layout.teacher_leaf_dir(
                    root, rung, leaf.teacher, int(leaf.ftseed or 0), animal=leaf.animal
                )
                / "checkpoints"
            )
        rows = filtered / expected.FILTERED_FILE
        out.append(
            numbers.evaluate_pool(
                spec=spec,
                raw_manifest=_read_json(pool / expected.POOL_MANIFEST),
                mode_manifest=_read_json(filtered / expected.POOL_MANIFEST),
                filtered_sha256=file_sha256(rows) if rows.is_file() else None,
                expected_adapter_id=adapter,
                expected_repo=repo,
                animal=leaf.animal,
            )
        )
    return out


def _numbers_gate(
    cfg: CampaignDriverConfig,
    state: dict,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    waiting: list[str],
) -> bool:
    """Freeze each pool's dataset tag and `filtered_sha256`. Waits, never raises."""
    if cfg.dry_run:
        state["transitions"][T_NUMBERS_READY] = {"at": _now(), "dry_run": True}
        return True
    statuses = pool_statuses(cfg, pool_coverage_leaves(admitted))
    report = numbers.gate_report(statuses)
    if not report["ready"]:
        missing = [s for s in statuses if not s.ready]
        waiting.append(f"numbers: {len(missing)}/{len(statuses)} pool(s) not ready")
        for status in missing:
            logger.info("  pool {} -> {}", status.spec.key, ", ".join(status.reasons))
        return False
    report["at"] = _now()
    driver_io.write_logs_report(cfg, plan.NUMBERS_GATE_REPORT, report)
    state["transitions"][T_NUMBERS_READY] = {"at": _now(), "n_pools": len(statuses)}
    return True


def _frozen_numbers(
    cfg: CampaignDriverConfig, admitted: Mapping[str, Mapping[str, Sequence[str]]]
) -> tuple[dict[str, str], dict[str, str]]:
    """`(dataset tags, filtered shas)` from the frozen numbers gate report.

    Raises:
        CampaignGateError: The report is missing.
    """
    if cfg.dry_run:
        tag = f"{plan.PROMPT_SET_TAG}_{plan.FILTER_MODE}_n100k"
        return (
            {leaf.pool_key: tag for leaf in pool_coverage_leaves(admitted)},
            {},
        )
    doc = _read_json(cfg.logs_dir / plan.NUMBERS_GATE_REPORT)
    if doc is None:
        raise CampaignGateError(
            f"{plan.NUMBERS_GATE_REPORT} missing; the pools have not been frozen"
        )
    pools = doc.get("pools")
    shas: dict[str, str] = {}
    if isinstance(pools, Mapping):
        for key, entry in pools.items():
            if isinstance(entry, Mapping) and entry.get("filtered_sha256"):
                shas[str(key)] = str(entry["filtered_sha256"])
    return numbers.dataset_tags_from_report(doc), shas


# ---- P0: the concordance stop gate --------------------------------------------


def _submit_concordance(
    cfg: CampaignDriverConfig,
    state: dict,
    tags: Mapping[str, str],
    shas: Mapping[str, str],
) -> None:
    """Queue the positive-control cell and its matched control.

    Raises:
        CampaignGateError: The reference rung generated no pool, so the gate cell cannot
            be trained and the port cannot be checked against her published number.
    """
    root = campaign_root(cfg)
    missing = [
        leaf.pool_key
        for leaf in pairs.concordance_leaves()
        if leaf.pool_key not in tags
    ]
    if missing:
        raise CampaignGateError(
            f"the concordance cell distils from {missing}, which the numbers gate "
            f"froze no tag for; rung {plan.CONCORDANCE_RUNG.key!r} was not admitted at "
            f"{plan.CONCORDANCE_SIZE}"
        )
    job_list = jobs.concordance_jobs(root, tags=tags, shas=shas)
    _submit(cfg, plan.Q_P0_CONCORDANCE, job_list)
    state["transitions"][T_SUBMIT_CONCORDANCE] = {
        "at": _now(),
        "n_cells": len(job_list),
    }


def _concordance_gate(
    cfg: CampaignDriverConfig, state: dict, tags: Mapping[str, str]
) -> None:
    """Stop the campaign unless the port reproduces Angela's published cell.

    The report is written and the transition persisted BEFORE the raise, so a later
    advance reads the verdict back instead of re-measuring it.

    Raises:
        CampaignGateError: The cell does not reproduce `CONCORDANCE_P_TARGET` within
            `CONCORDANCE_TOL` — the port is broken, not the science.
    """
    treatment, control = pairs.concordance_leaves()
    measured = (
        plan.CONCORDANCE_P_TARGET
        if cfg.dry_run
        else _leaf_p_target(campaign_root(cfg), treatment, tags, plan.PRIMARY_ANIMAL)
    )
    gap = None if measured is None else measured - plan.CONCORDANCE_P_TARGET
    reproduced = gap is not None and abs(gap) <= plan.CONCORDANCE_TOL
    report = {
        "at": _now(),
        "cell": treatment.key,
        "size": plan.CONCORDANCE_SIZE,
        "rung_tag": plan.CONCORDANCE_RUNG.key,
        "recipe_key": plan.CONCORDANCE_RECIPE.key,
        "prompt_set": plan.PRIMARY_PROMPT_SET,
        "p_target": measured,
        "control_p_target": (
            None
            if cfg.dry_run
            else _leaf_p_target(campaign_root(cfg), control, tags, plan.PRIMARY_ANIMAL)
        ),
        "reference": plan.CONCORDANCE_P_TARGET,
        "tolerance": plan.CONCORDANCE_TOL,
        "abs_gap": gap,
        "reproduced": reproduced,
    }
    driver_io.write_logs_report(cfg, plan.CONCORDANCE_GATE_REPORT, report)
    state["transitions"][T_CONCORDANCE_GATE] = {
        "at": _now(),
        "reproduced": reproduced,
        "p_target": measured,
    }
    driver_io.save_state(cfg, state)
    if not reproduced:
        raise CampaignGateError(
            f"concordance cell read p_target={measured} against reference "
            f"{plan.CONCORDANCE_P_TARGET} (tol {plan.CONCORDANCE_TOL}); the port does "
            "not reproduce Angela's published cell, so the factorial is not submitted "
            f"— see {plan.CONCORDANCE_GATE_REPORT}"
        )


def _require_concordance(cfg: CampaignDriverConfig) -> None:
    """Read the frozen concordance verdict back.

    Raises:
        CampaignGateError: The report is missing or records a failure.
    """
    if cfg.dry_run:
        return
    doc = _read_json(cfg.logs_dir / plan.CONCORDANCE_GATE_REPORT)
    if doc is None:
        raise CampaignGateError(
            f"{plan.CONCORDANCE_GATE_REPORT} missing; the stop gate has not fired"
        )
    if doc.get("reproduced") is not True:
        raise CampaignGateError(
            f"concordance gate FAILED (p_target={doc.get('p_target')}, reference "
            f"{doc.get('reference')}); not submitting the factorial"
        )


# ---- P1: the factorial --------------------------------------------------------


def p1_treatments(
    size: str,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    animal: str = plan.PRIMARY_ANIMAL,
) -> tuple[pairs.Leaf, ...]:
    """One column's factorial treatments, dropping teachers with no admitted rung."""
    out: list[pairs.Leaf] = []
    for teacher in plan.SIZES:
        allowed = set(admitted.get(animal, {}).get(teacher, ()))
        rungs = tuple(r for r in pairs.p1_rungs(size) if r.key in allowed)
        if rungs:
            out.extend(
                pairs.p1_leaves(size, animal=animal, teachers=(teacher,), rungs=rungs)
            )
    return tuple(out)


def warm_leaves(
    size: str, admitted: Mapping[str, Mapping[str, Sequence[str]]]
) -> tuple[pairs.Leaf, ...]:
    """One representative leaf per capability tree this column can land in.

    `base_metrics.json` is shared by every `{student_tag}` under one `{dataset_tag}`
    dir, which sits ABOVE the recipe and the seed. The plan gives a size ONE warm queue
    and it is submitted before the confirm pick exists, so the warm set covers every
    (animal, teacher, rung) tree the pick can reach rather than P1's own trees alone —
    a lion tree left cold would make every P2 lion eval refuse to start.
    """
    return coverage_leaves(size, admitted)


def _submit_p1(
    cfg: CampaignDriverConfig,
    state: dict,
    size: str,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    tags: Mapping[str, str],
    shas: Mapping[str, str],
) -> None:
    """Queue one student column: treatments, matched controls, warm, capability.

    Raises:
        CampaignGateError: Every teacher is void for this column, so it has no cell.
    """
    root = campaign_root(cfg)
    treatments = p1_treatments(size, admitted)
    if not treatments:
        raise CampaignGateError(
            f"{size}: no admitted teacher at any size, so the column has no treatment"
        )
    controls = pairs.p1_control_leaves(size)
    _submit(
        cfg, plan.q_p1(size), jobs.student_jobs(root, treatments, tags=tags, shas=shas)
    )
    _submit(
        cfg,
        plan.q_p1_controls(size),
        jobs.student_jobs(root, controls, tags=tags, shas=shas),
    )
    # Every capability cell in a tree READS one `base_metrics.json` that the warm fills.
    # Cells filling it concurrently is the read-modify-write race the eval cell refuses
    # on, so the evals wait for the warm queue instead of racing it.
    _submit(
        cfg,
        plan.q_p1_warm(size),
        jobs.warm_eval_base_jobs(
            root, warm_leaves(size, admitted), tags=tags, queue_tag="p1warm"
        ),
        depends_on=[plan.q_p1(size), plan.q_p1_controls(size)],
    )
    _submit(
        cfg,
        plan.q_p1_evals(size),
        jobs.capability_jobs(
            root, [*treatments, *controls], tags=tags, queue_tag="p1e"
        ),
        depends_on=[plan.q_p1_warm(size)],
    )
    state["transitions"][t_submit_p1(size)] = {
        "at": _now(),
        "n_treatments": len(treatments),
        "n_controls": len(controls),
    }


# ---- the confirm pick ---------------------------------------------------------


def confirm_rows(
    cfg: CampaignDriverConfig,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    tags: Mapping[str, str],
) -> list[selection.PairRow]:
    """One row per factorial cell: its Delta against the matched control, and its PPL.

    The control of a cross cell is the one trained on the TEACHER size's untuned numbers
    and of an anchor the student's own, which is the same enumerated control either way
    (`p1_control_leaves` keys on the generating size).
    """
    root = campaign_root(cfg)
    animal = plan.PRIMARY_ANIMAL
    rows: list[selection.PairRow] = []
    for size in plan.SIZES:
        controls = {
            (leaf.teacher, leaf.recipe.key): leaf
            for leaf in pairs.p1_control_leaves(size, animal=animal)
        }
        prior = _base_prior(cfg, animal, size)
        verdicts = selection.control_integrity(
            [
                selection.ControlReading(
                    student=size,
                    generator=leaf.teacher,
                    recipe_key=leaf.recipe.key,
                    animal=animal,
                    p_target=_leaf_p_target(root, leaf, tags, animal),
                    base_prior=prior,
                )
                for leaf in controls.values()
            ]
        )
        for leaf in p1_treatments(size, admitted, animal=animal):
            control = controls.get((leaf.teacher, leaf.recipe.key))
            treated = _leaf_p_target(root, leaf, tags, animal)
            base = _leaf_p_target(root, control, tags, animal)
            verdict = verdicts.get((animal, size, leaf.teacher, leaf.recipe.key))
            rows.append(
                selection.PairRow(
                    animal=animal,
                    teacher=leaf.teacher,
                    student=size,
                    rung_tag=str(leaf.rung_tag),
                    recipe_key=leaf.recipe.key,
                    delta_primary=(
                        None if treated is None or base is None else treated - base
                    ),
                    ci_low=None,
                    ppl_ratio=_perplexity_ratio(expected.param_dir(root, leaf, tags)),
                    control_intact=bool(verdict and verdict.intact),
                )
            )
    return rows


def _confirm_pick(
    cfg: CampaignDriverConfig,
    state: dict,
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    tags: Mapping[str, str],
) -> None:
    """Promote the top `CONFIRM_N_CONFIGS` (rung, recipe) pairs and persist them.

    A picks file that already exists is authoritative: re-scoring 600 single-seed cells
    could move the pick under the confirm cells that were queued from the first one.

    Raises:
        CampaignGateError: No factorial cell survived the confirm filters.
    """
    stored = _read_json(cfg.logs_dir / plan.SCREEN_PICKS)
    if isinstance(stored, Mapping) and stored.get("picks"):
        state["transitions"][T_CONFIRM_PICK] = {
            "at": _now(),
            "picks": [dict(p) for p in stored["picks"]],
            "source": plan.SCREEN_PICKS,
        }
        logger.info("confirm picks read back from {}", plan.SCREEN_PICKS)
        return
    if cfg.dry_run:
        picks = [
            {
                "rung_tag": plan.TEACHER_RUNGS[0].key,
                "recipe_key": recipe.key,
                "score": 0.0,
                "n_pairs": 0,
            }
            for recipe in plan.STUDENT_GRID[: plan.CONFIRM_N_CONFIGS]
        ]
    else:
        picks = [
            {
                "rung_tag": pick.rung_tag,
                "recipe_key": pick.recipe_key,
                "score": pick.score,
                "n_pairs": pick.n_pairs,
            }
            for pick in selection.pick_confirm(
                confirm_rows(cfg, admitted, tags), n=plan.CONFIRM_N_CONFIGS
            )
        ]
    driver_io.write_logs_report(cfg, plan.SCREEN_PICKS, {"at": _now(), "picks": picks})
    state["transitions"][T_CONFIRM_PICK] = {"at": _now(), "picks": picks}
    for i, pick in enumerate(picks):
        logger.info(
            "confirm pick #{}: rung={} recipe={} score={}",
            i + 1,
            pick["rung_tag"],
            pick["recipe_key"],
            pick["score"],
        )


def _confirm_picks(cfg: CampaignDriverConfig) -> list[dict]:
    """The persisted confirm picks, best first.

    Raises:
        CampaignGateError: The picks file is missing or empty.
    """
    doc = _read_json(cfg.logs_dir / plan.SCREEN_PICKS)
    picks = doc.get("picks") if isinstance(doc, Mapping) else None
    if not picks:
        if cfg.dry_run:
            return [
                {"rung_tag": plan.TEACHER_RUNGS[0].key, "recipe_key": r.key}
                for r in plan.STUDENT_GRID[: plan.CONFIRM_N_CONFIGS]
            ]
        raise CampaignGateError(
            f"{plan.SCREEN_PICKS} missing or empty; the confirm pick has not fired"
        )
    return [dict(p) for p in picks]


# ---- P2: confirm --------------------------------------------------------------


def p2_leaves(
    size: str,
    picks: Sequence[Mapping[str, object]],
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[pairs.Leaf, ...]:
    """One column's confirm cells: the top pick at three seeds, the runners-up at one.

    Both animals, and per animal only the teacher sizes whose admitted ladder carries
    the picked rung — a teacher that was voided for lion cannot be confirmed for lion.
    """
    out: list[pairs.Leaf] = []
    for index, pick in enumerate(picks):
        rung_tag = str(pick["rung_tag"])
        recipe = _recipe_by_key(str(pick["recipe_key"]))
        gseeds = (
            plan.CONFIRM_STUDENT_SEEDS if index == 0 else plan.CONFIRM_RUNNER_UP_SEEDS
        )
        for animal in plan.ANIMALS:
            teachers = tuple(
                t for t in plan.SIZES if rung_tag in admitted.get(animal, {}).get(t, ())
            )
            if not teachers:
                continue
            out.extend(
                pairs.p2_leaves(
                    size,
                    recipe=recipe,
                    rung_tag=rung_tag,
                    animals=(animal,),
                    teachers=teachers,
                    gseeds=gseeds,
                )
            )
    return tuple(out)


def _submit_p2(
    cfg: CampaignDriverConfig,
    state: dict,
    size: str,
    picks: Sequence[Mapping[str, object]],
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    tags: Mapping[str, str],
    shas: Mapping[str, str],
) -> None:
    """Queue one column's confirm treatments, controls and capability evals.

    Raises:
        CampaignGateError: The picks cover no admitted teacher for this column.
    """
    root = campaign_root(cfg)
    leaves = p2_leaves(size, picks, admitted)
    treatments = [leaf for leaf in leaves if leaf.kind == pairs.TREATMENT]
    controls = [leaf for leaf in leaves if leaf.kind != pairs.TREATMENT]
    if not treatments or not controls:
        raise CampaignGateError(
            f"{size}: the confirm picks name no admitted teacher, so the column has "
            "no confirmable cell"
        )
    _submit(
        cfg, plan.q_p2(size), jobs.student_jobs(root, treatments, tags=tags, shas=shas)
    )
    _submit(
        cfg,
        plan.q_p2_controls(size),
        jobs.student_jobs(root, controls, tags=tags, shas=shas),
    )
    # The plan carries no `q_p2_warm`; `q_p1_warm` already covers every tree the pick
    # can reach (see `warm_leaves`), so the confirm evals gate on it as well as on
    # their own trainings.
    _submit(
        cfg,
        plan.q_p2_evals(size),
        jobs.capability_jobs(root, leaves, tags=tags, queue_tag="p2e"),
        depends_on=[plan.q_p2(size), plan.q_p2_controls(size), plan.q_p1_warm(size)],
    )
    state["transitions"][t_submit_p2(size)] = {
        "at": _now(),
        "n_treatments": len(treatments),
        "n_controls": len(controls),
    }


# ---- deadline -----------------------------------------------------------------


def _open_window(state: dict, opts: Offxsize2DriverOptions) -> dict:
    """Append a deadline window when the last one is closed (or none exists).

    Called once per `run()` pass, so a resume gets a fresh budget and a closed window is
    never mutated back to open.
    """
    windows = state.setdefault("deadline_windows", [])
    if windows and windows[-1].get("ended_at") is None:
        return windows[-1]
    window = {
        "start": _now(),
        "hours": float(opts.deadline_hours),
        "ended_at": None,
        "reason": None,
    }
    windows.append(window)
    return window


def _deadline_passed(state: dict, opts: Offxsize2DriverOptions) -> bool:
    windows = state.get("deadline_windows") or []
    if not windows or windows[-1].get("ended_at") is not None:
        return False
    window = windows[-1]
    hours = float(window.get("hours", opts.deadline_hours))
    start = _parse_utc(str(window["start"]))
    return datetime.now(timezone.utc) - start > timedelta(hours=hours)


def _close_window(state: dict, *, reason: str) -> None:
    windows = state.get("deadline_windows") or []
    if windows:
        windows[-1]["ended_at"] = _now()
        windows[-1]["reason"] = reason


def _remaining_minutes(state: dict, opts: Offxsize2DriverOptions) -> float:
    """Wall-clock minutes left in the open window (0 when it has expired)."""
    windows = state.get("deadline_windows") or []
    if not windows or windows[-1].get("ended_at") is not None:
        return 0.0
    window = windows[-1]
    hours = float(window.get("hours", opts.deadline_hours))
    start = _parse_utc(str(window["start"]))
    elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
    return max(hours * 60.0 - elapsed, 0.0)


def _inflight(cfg: CampaignDriverConfig) -> int:
    """Sum of `running` across this campaign's submitted queues."""
    return sum(
        int(store.queue_status(cfg.queue_root, name)["running"])
        for name in _live_queue_names(cfg)
    )


# ---- the elastic tail ---------------------------------------------------------


def achieved_capacity(
    cfg: CampaignDriverConfig, state: dict, opts: Offxsize2DriverOptions
) -> float:
    """Card-minutes this pool delivers per wall-clock minute, measured from jobq.

    A cell's fair-share cost is its wall duration times its slot weight over
    `CAP_PER_GPU` — the same accounting `plan.EPOCH_UNIT_MINUTES` and every
    `TailTier.est_gpu_min` were measured on. Summing that over the finished cells of the
    open window and dividing by the elapsed wall-clock gives the card-minutes the pool
    actually delivers per minute, which is what turns a tier's GPU-min estimate into the
    wall-clock it would need. Capped at the card count: no packing can deliver more.

    A resumed campaign has no finished cell inside its fresh window, so the span falls
    back to the whole campaign (earliest recorded start to now). That under-states the
    rate by any idle time between windows, which is the safe direction for work that has
    to fit before a deadline — and it is what stops a resume from reading as "nothing
    measured" and dropping the tail on a pool that is demonstrably working.

    Returns:
        Card-minutes per wall-clock minute; 0.0 when nothing has finished yet.
    """
    windows = state.get("deadline_windows") or []
    if not windows:
        return 0.0
    start = _parse_utc(str(windows[-1]["start"]))
    cells: list[tuple[datetime, datetime, int]] = []
    for name in _live_queue_names(cfg):
        slots = {
            job.key: (job.slots or 1) for job in store.load_jobs(cfg.queue_root, name)
        }
        for record in store.load_results(cfg.queue_root, name):
            try:
                began = _parse_utc(str(record["start_utc"]))
                ended = _parse_utc(str(record["end_utc"]))
            except (KeyError, TypeError, ValueError):
                continue
            cells.append((began, ended, slots.get(str(record.get("key")), 1)))
    if not cells:
        return 0.0
    rows = [cell for cell in cells if cell[1] >= start]
    if not rows:
        rows, start = cells, min(cell[0] for cell in cells)
    elapsed = (datetime.now(timezone.utc) - start).total_seconds() / 60.0
    if elapsed <= 0:
        return 0.0
    delivered = sum(
        max((ended - max(began, start)).total_seconds(), 0.0)
        / 60.0
        * weight
        / plan.CAP_PER_GPU
        for began, ended, weight in rows
    )
    return min(delivered / elapsed, float(opts.n_gpus))


def _tier_kwargs(tier: str, picks: Sequence[Mapping[str, object]]) -> dict:
    """The decisions one tier cannot default (`pairs.tail_leaves`).

    Raises:
        CampaignGateError: A tier that extends the winning configuration was reached
            before the confirm pick was persisted.
    """
    if tier in ("t1", "t5", "t6"):
        return {}
    if not picks:
        raise CampaignGateError(f"tail tier {tier!r} needs the confirm pick first")
    rung_tag = str(picks[0]["rung_tag"])
    if tier == "t2":
        return {"rung_tag": rung_tag, "students": plan.SIZES}
    if tier == "t3":
        return {"rung_tag": rung_tag, "students": plan.T3_STUDENTS}
    if tier == "t4":
        return {
            "rung_tag": rung_tag,
            "pair_list": plan.T4_PAIRS,
            "recipes": plan.T4_EXTRA_RECIPES,
        }
    return {
        "rung_tag": rung_tag,
        "students": plan.SIZES,
        "recipes": (_recipe_by_key(str(picks[0]["recipe_key"])),),
    }


def _admit_tail(
    cfg: CampaignDriverConfig,
    state: dict,
    opts: Offxsize2DriverOptions,
    picks: Sequence[Mapping[str, object]],
    tags: Mapping[str, str],
    shas: Mapping[str, str],
    waiting: list[str],
    fired: list[str],
) -> bool:
    """Admit pre-registered filler while the pool is idle and the budget allows.

    Tiers are admitted in the plan's order and one per pass: a tier's own queue is live
    work, so the next tier waits for it exactly as it waited for the committed queues.
    A tier that trains no student on this lineage (`t5` benchmark backfill, `t6`
    constrained generation) is recorded as skipped rather than queued — the plan gives
    each tier a single queue, which cannot order a base-cache warm before an eval.

    Returns:
        Whether a later pass may still admit a tier (so `finalize` should wait).
    """
    transitions = state["transitions"]
    remaining = (
        opts.deadline_hours * 60.0 if cfg.dry_run else _remaining_minutes(state, opts)
    )
    capacity = (
        float(opts.n_gpus) if cfg.dry_run else achieved_capacity(cfg, state, opts)
    )
    for tier in plan.TAIL_TIERS:
        key = t_admit_tail(tier.key)
        if key in transitions:
            continue
        if capacity <= 0.0:
            waiting.append("tail: no finished cell to measure card throughput from")
            return True
        needed = tier.est_gpu_min / capacity
        if remaining <= needed:
            waiting.append(
                f"tail {tier.key}: needs {needed:.0f} wall-min at "
                f"{capacity:.2f} card-min/min, {remaining:.0f} left — dropped"
            )
            return False
        skipped = "tier trains no student on this lineage"
        try:
            job_list = jobs.tail_jobs(
                tier.key,
                campaign_root(cfg),
                tags=tags,
                shas=shas,
                **_tier_kwargs(tier.key, picks),
            )
        except KeyError as err:
            # `t7` reads teacher pools at `TAIL_TEACHER_FTSEEDS`, which no committed
            # phase generates. A tier is droppable, so this is recorded rather than
            # raised — but it is recorded, not swallowed.
            logger.warning(
                "tail {} needs a pool this campaign never froze: {}", tier.key, err
            )
            job_list = []
            skipped = f"needs a pool the numbers gate never froze: {err}"
        if not job_list:
            transitions[key] = {"at": _now(), "n_jobs": 0, "skipped": skipped}
            fired.append(key)
            driver_io.save_state(cfg, state)
            continue
        _submit(cfg, plan.q_tail(tier.key), job_list)
        transitions[key] = {
            "at": _now(),
            "n_jobs": len(job_list),
            "est_gpu_min": tier.est_gpu_min,
            "wall_min_needed": needed,
            "card_min_per_min": capacity,
        }
        fired.append(key)
        driver_io.save_state(cfg, state)
        if not cfg.dry_run:
            return True
    return False


# ---- finalize -----------------------------------------------------------------


def committed_leaves(
    state: Mapping[str, object],
    admitted: Mapping[str, Mapping[str, Sequence[str]]],
    picks: Sequence[Mapping[str, object]],
) -> tuple[pairs.Leaf, ...]:
    """Every cell the committed phases submitted, for coverage and the disk gate.

    Keyed on the fired transitions rather than on the enumerators alone, so a partial
    finalize reports what this campaign actually queued instead of what it planned.
    """
    transitions = state.get("transitions") or {}
    if not admitted:
        return ()
    out: list[pairs.Leaf] = list(pairs.concordance_leaves())
    for size in plan.SIZES:
        if t_submit_p1(size) in transitions:
            out.extend(p1_treatments(size, admitted))
            out.extend(pairs.p1_control_leaves(size))
        if picks and t_submit_p2(size) in transitions:
            out.extend(p2_leaves(size, picks, admitted))
    return tuple(out)


def _build_results(
    cfg: CampaignDriverConfig,
    state: dict,
    holes: Sequence[Mapping[str, object]],
    *,
    partial: bool,
) -> None:
    """Rebuild every result CSV, `claims.json` and `coverage.json` from disk.

    The holes and the deadline windows are the driver's alone to report, so they are
    handed to the results pipeline rather than written beside it: `coverage.json` has
    one writer, and a partial finalize must state the windows that produced it.
    """
    from pipelines.offxsize2_results_pipeline import build_results

    build_results(
        cfg,
        dry_run=cfg.dry_run,
        partial=partial,
        holes=[dict(hole) for hole in holes],
        deadline_windows=list(state.get("deadline_windows") or []),
    )


def _finalize(
    cfg: CampaignDriverConfig,
    state: dict,
    holes: Sequence[Mapping[str, object]],
    *,
    partial: bool,
) -> None:
    """Write the results; stamp `CAMPAIGN_DONE` only when nothing is outstanding."""
    _build_results(cfg, state, holes, partial=partial)
    key = T_FINALIZE_PARTIAL if partial else T_FINALIZE
    if not partial and not cfg.dry_run:
        marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(marker, f"campaign {plan.RUN_SET_ID} finished {_now()}\n")
        logger.info("campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)
    elif partial:
        logger.warning(
            "campaign {} partial finalize — coverage written, no {}",
            plan.RUN_SET_ID,
            plan.CAMPAIGN_DONE_FILE,
        )
    state["transitions"][key] = {
        "at": _now(),
        "partial": partial,
        "n_holes": len(holes),
    }


def _deadline_stop(
    cfg: CampaignDriverConfig,
    state: dict,
    opts: Offxsize2DriverOptions,
    holes: Sequence[Mapping[str, object]],
) -> None:
    """Pause the pool, drain the in-flight cells, then partial-finalize."""
    if not cfg.dry_run:
        stop = store.stop_path(cfg.queue_root, store.this_host())
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.touch()
        waited = 0
        while _inflight(cfg) and waited < opts.drain_timeout_s:
            time.sleep(30)
            waited += 30
    _finalize(cfg, state, holes, partial=True)
    _close_window(state, reason="deadline")


# ---- the pass -----------------------------------------------------------------


def run(
    cfg: CampaignDriverConfig, options: Offxsize2DriverOptions | None = None
) -> AdvanceResult:
    """Advance the campaign as far as its gates allow. Idempotent.

    Args:
        cfg: Driver config bound to `offxsize2_plan.SPEC`.
        options: Deadline and failure-tolerance knobs; defaults when omitted.

    Returns:
        Fired transitions, wait reasons, and whether a finalize has run.

    Raises:
        CampaignGateError: The run-set id is malformed, a queue failed past the cap, no
            teacher is admissible, the concordance cell does not reproduce, or this host
            is not the campaign node.
        ValueError: `cfg` is not bound to this campaign.
    """
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(
            f"run_set_id {cfg.run_set_id!r} is malformed — expected "
            "{YYYYMMDD}_{purpose-slug}_{size}: hyphens inside the slug, an underscore "
            "before the size. `scripts.adopt_run_set` refuses it too, so staging could "
            "not adopt into this campaign. Fix `offxsize2_plan.RUN_SET_ID`."
        )
    if cfg.spec is not plan.SPEC:
        raise ValueError("driver config must be bound to offxsize2_plan.SPEC")

    opts = options or Offxsize2DriverOptions()
    state = driver_io.load_state(cfg)
    state.setdefault("transitions", {})
    transitions = state["transitions"]
    holes: list[dict] = list(state.get("holes") or [])
    fired: list[str] = []
    waiting: list[str] = []
    _open_window(state, opts)

    def _fire(name: str, fn) -> None:
        fn()
        fired.append(name)
        driver_io.save_state(cfg, state)

    admitted: dict[str, dict[str, tuple[str, ...]]] = {}
    tags: dict[str, str] = {}
    shas: dict[str, str] = {}
    picks: list[dict] = []

    if (
        _deadline_passed(state, opts)
        and T_FINALIZE not in transitions
        and T_FINALIZE_PARTIAL not in transitions
    ):
        state["holes"] = holes
        _deadline_stop(cfg, state, opts, holes)
        driver_io.save_state(cfg, state)
        return AdvanceResult((T_FINALIZE_PARTIAL,), tuple(waiting), True)

    if T_PREFLIGHT not in transitions:
        _fire(T_PREFLIGHT, lambda: _preflight(cfg, state))

    if T_PREFLIGHT in transitions and T_SUBMIT_P0 not in transitions:
        _fire(T_SUBMIT_P0, lambda: _submit_p0(cfg, state))

    if T_SUBMIT_P0 in transitions and T_TEACHER_GATE not in transitions:
        if _queues_ready(cfg, _P0_QUEUES, opts, waiting, holes):
            _fire(T_TEACHER_GATE, lambda: _teacher_gate(cfg, state))

    if T_TEACHER_GATE in transitions:
        admitted = _admitted_rungs(cfg)
        if T_SUBMIT_POOLS not in transitions:
            _fire(T_SUBMIT_POOLS, lambda: _submit_pools(cfg, state, admitted))
        if T_NUMBERS_READY not in transitions and _queues_ready(
            cfg, [plan.Q_P0_POOLS], opts, waiting, holes
        ):
            if _numbers_gate(cfg, state, admitted, waiting):
                fired.append(T_NUMBERS_READY)
                driver_io.save_state(cfg, state)

    if T_NUMBERS_READY in transitions:
        tags, shas = _frozen_numbers(cfg, admitted)
        if T_SUBMIT_CONCORDANCE not in transitions:
            _fire(
                T_SUBMIT_CONCORDANCE,
                lambda: _submit_concordance(cfg, state, tags, shas),
            )
        if T_CONCORDANCE_GATE not in transitions and _queues_ready(
            cfg, [plan.Q_P0_CONCORDANCE], opts, waiting, holes
        ):
            _fire(T_CONCORDANCE_GATE, lambda: _concordance_gate(cfg, state, tags))

    if T_CONCORDANCE_GATE in transitions:
        _require_concordance(cfg)
        for size in plan.SIZES:
            if t_submit_p1(size) not in transitions:
                _fire(
                    t_submit_p1(size),
                    lambda s=size: _submit_p1(cfg, state, s, admitted, tags, shas),
                )
        if T_CONFIRM_PICK not in transitions:
            p1_queues = [
                q
                for size in plan.SIZES
                for q in (
                    plan.q_p1(size),
                    plan.q_p1_controls(size),
                    plan.q_p1_evals(size),
                )
            ]
            factorial = [
                leaf
                for size in plan.SIZES
                for leaf in (
                    *p1_treatments(size, admitted),
                    *pairs.p1_control_leaves(size),
                )
            ]
            if _queues_ready(cfg, p1_queues, opts, waiting, holes) and _leaves_ready(
                cfg, factorial, tags, waiting, "p1"
            ):
                _fire(T_CONFIRM_PICK, lambda: _confirm_pick(cfg, state, admitted, tags))

    if T_CONFIRM_PICK in transitions:
        picks = _confirm_picks(cfg)
        for size in plan.SIZES:
            if t_submit_p2(size) not in transitions:
                _fire(
                    t_submit_p2(size),
                    lambda s=size: _submit_p2(
                        cfg, state, s, picks, admitted, tags, shas
                    ),
                )

    state["holes"] = holes
    if T_FINALIZE not in transitions:
        submitted_p2 = all(t_submit_p2(size) in transitions for size in plan.SIZES)
        drained = submitted_p2 and _queues_ready(
            cfg, _required_queue_names(cfg), opts, waiting, holes
        )
        tail_pending = True
        if drained:
            tail_pending = _admit_tail(
                cfg, state, opts, picks, tags, shas, waiting, fired
            )
        leaves = committed_leaves(state, admitted, picks)
        if (
            drained
            and not tail_pending
            and _queues_ready(cfg, _live_queue_names(cfg), opts, waiting, holes)
            and _leaves_ready(cfg, leaves, tags, waiting, "campaign")
        ):
            state["holes"] = holes
            _fire(T_FINALIZE, lambda: _finalize(cfg, state, holes, partial=False))

    state["holes"] = holes
    driver_io.save_state(cfg, state)
    done = T_FINALIZE in transitions or T_FINALIZE_PARTIAL in transitions
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: CampaignDriverConfig) -> dict:
    """Read-only snapshot for the CLI."""
    state = driver_io.load_state(cfg)
    gate = _read_json(cfg.logs_dir / plan.TEACHER_GATE_REPORT) or {}
    picks = _read_json(cfg.logs_dir / plan.SCREEN_PICKS) or {}
    concordance = _read_json(cfg.logs_dir / plan.CONCORDANCE_GATE_REPORT) or {}
    pools = _read_json(cfg.logs_dir / plan.NUMBERS_GATE_REPORT) or {}
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in _live_queue_names(cfg)
    }
    return {
        "run_set_id": cfg.run_set_id,
        "campaign_root": str(campaign_root(cfg)),
        "node": socket.gethostname(),
        "transitions": sorted(state.get("transitions", {})),
        "admitted_rungs": gate.get("admitted", {}),
        "voided_teachers": state.get("voided_teachers", []),
        "numbers_ready": bool(pools.get("ready")),
        "concordance": {
            "reproduced": concordance.get("reproduced"),
            "p_target": concordance.get("p_target"),
        },
        "confirm_picks": picks.get("picks", []),
        "holes": list(state.get("holes") or []),
        "deadline_windows": list(state.get("deadline_windows") or []),
        "queues": queues,
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
