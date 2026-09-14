"""Fixed decisions of the strictscreen judge audit chain (pure).

Phases 6-12 of `.claude/plans/strictscreen_rollout_judge_audit.plan.md` §3: task
derivation from the committed plan manifest, the driver block of
``configs/experiment/filtration_audit_strictscreen.yaml``, probe gates, and the
judge relaunch policy. IO lives in `campaign.judge_audit_io`; the phase machine
in `pipelines.strictscreen_audit_driver`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from subliminal_learning.campaign import strictscreen_plan as ssp

TASK_ORDER: tuple[str, ...] = (
    "rollouts_clean",
    "rollouts_flagged",
    "decoy",
    "recall_probe",
    "pool_open",
    "pool_targeted",
)
POOL_TASKS: tuple[str, ...] = ("pool_open", "pool_targeted")
PHASES: tuple[str, ...] = (
    "audit_completeness",
    "audit_plan",
    "audit_release_gpus",
    "audit_probe",
    "audit_judge",
    "audit_return_gpus",
    "audit_results",
    "audit_finalize",
)
EXPERIMENT_CONFIG_REL = "configs/experiment/filtration_audit_strictscreen.yaml"
JUDGE_TOOL_REL = "tools/rollout_judge"
PATHS_CONFIG_REL = "configs/paths/paths_local.yaml"
MODEL_CONFIG_REL = "configs/model/pythia.yaml"
AUDIT_BLOCKED_FILE = "AUDIT_BLOCKED.json"
AUDIT_STATUS_FILE = "audit_status.json"
JUDGE_ENV_FILE = "judge_env.json"
OWNER_FILE = "OWNER.json"

_FAILED_WORKER_RE = re.compile(r"worker (\d+) on GPU (\d+) exited (-?\d+)")


@dataclass(frozen=True)
class ProbeGate:
    """Thresholds for one single-GPU ``probe`` run.

    Attributes:
        task: Probed task (first queue shard).
        n: Rows to judge.
        max_parse_failure: Ceiling on ``parse_failure_rate``.
        max_truncation: Ceiling on ``truncation_rate``.
        min_rows_per_second: Floor on ``steady_state_rows_per_second``.
        min_agreement: Floor on yes/no agreement with the reference audit.
    """

    task: str
    n: int
    max_parse_failure: float | None = None
    max_truncation: float | None = None
    min_rows_per_second: float | None = None
    min_agreement: float | None = None


@dataclass(frozen=True)
class DriverConfig:
    """The ``driver:`` block of the strictscreen audit experiment YAML.

    Attributes:
        judge_config: Repo-relative judge YAML.
        tool_source_sha256: ``(repo-relative path, sha256)`` pins.
        model_id: Judge Hub repo.
        model_snapshot: Pinned Hub snapshot revision.
        vllm_version: ``vllm.__version__`` the judge venv must import.
        probe_gates: Gates in run order.
        probe_retries: Crash retries per probe.
        attempt_cap: ``judge-cluster`` launches allowed.
        gpu_fail_exclude_after: Worker failures before a GPU is excluded.
        gpus: Device ordinals for ``judge-cluster``.
        max_idle_gpu_mib: Per-GPU used memory counted as released.
        release_sample_gap_s: Gap between the two release samples.
        min_cold_free_gib: Free space required under ``audit_root``.
    """

    judge_config: str
    tool_source_sha256: tuple[tuple[str, str], ...]
    model_id: str
    model_snapshot: str
    vllm_version: str
    probe_gates: tuple[ProbeGate, ...]
    probe_retries: int
    attempt_cap: int
    gpu_fail_exclude_after: int
    gpus: tuple[int, ...]
    max_idle_gpu_mib: int
    release_sample_gap_s: float
    min_cold_free_gib: float


@dataclass(frozen=True)
class ProbeVerdict:
    """One probe gate outcome.

    Attributes:
        task: Probed task.
        passed: Every configured threshold held.
        failures: Human-readable threshold misses.
        metrics: Values the thresholds were checked on.
    """

    task: str
    passed: bool
    failures: tuple[str, ...]
    metrics: Mapping[str, float | int | None]


def parse_driver_config(raw: Mapping[str, object]) -> DriverConfig:
    """Build `DriverConfig` from the YAML ``driver:`` mapping.

    Raises:
        ValueError: A key is missing, a gate task is not in `TASK_ORDER`, or a
            count is not positive.
    """
    try:
        gates = tuple(
            ProbeGate(
                task=str(g["task"]),
                n=int(g["n"]),
                max_parse_failure=_opt_float(g.get("max_parse_failure")),
                max_truncation=_opt_float(g.get("max_truncation")),
                min_rows_per_second=_opt_float(g.get("min_rows_per_second")),
                min_agreement=_opt_float(g.get("min_agreement")),
            )
            for g in raw["probe_gates"]
        )
        cfg = DriverConfig(
            judge_config=str(raw["judge_config"]),
            tool_source_sha256=tuple(
                (str(k), str(v)) for k, v in dict(raw["tool_source_sha256"]).items()
            ),
            model_id=str(raw["model_id"]),
            model_snapshot=str(raw["model_snapshot"]),
            vllm_version=str(raw["vllm_version"]),
            probe_gates=gates,
            probe_retries=int(raw["probe_retries"]),
            attempt_cap=int(raw["attempt_cap"]),
            gpu_fail_exclude_after=int(raw["gpu_fail_exclude_after"]),
            gpus=tuple(int(g) for g in raw["gpus"]),
            max_idle_gpu_mib=int(raw["max_idle_gpu_mib"]),
            release_sample_gap_s=float(raw["release_sample_gap_s"]),
            min_cold_free_gib=float(raw["min_cold_free_gib"]),
        )
    except (KeyError, TypeError) as err:
        raise ValueError(f"driver block invalid: {err}") from err
    unknown = [g.task for g in cfg.probe_gates if g.task not in TASK_ORDER]
    if unknown:
        raise ValueError(f"probe gate tasks {unknown} not in {TASK_ORDER}")
    if min(g.n for g in cfg.probe_gates) < 1 or cfg.attempt_cap < 1 or not cfg.gpus:
        raise ValueError("probe n, attempt_cap and gpus must be positive/non-empty")
    return cfg


def validate_audit_identity(
    experiment: Mapping[str, object],
    judge: Mapping[str, object],
) -> list[str]:
    """Mismatches between the audit YAMLs and `strictscreen_plan` identity.

    Args:
        experiment: Raw filtration-audit experiment YAML.
        judge: Raw judge YAML.

    Returns:
        Failure messages; empty when consistent.
    """
    reference = experiment.get("reference") or {}
    want = {
        "experiment audit_id": (experiment.get("audit_id"), ssp.AUDIT_ID),
        "judge audit_id": (judge.get("audit_id"), ssp.AUDIT_ID),
        "campaign_subdir": (
            experiment.get("campaign_subdir"),
            f"gkd_students/{ssp.RUN_SET_ID}",
        ),
        "gkd_tags": (
            experiment.get("gkd_tags"),
            {ssp.STRICT_TAG: "strict", ssp.LEGACY_TAG: "legacy"},
        ),
        "headline_animals": (
            tuple(experiment.get("headline_animals") or ()),
            ssp.ANIMALS,
        ),
        "reference.audit_id": (
            reference.get("audit_id") if isinstance(reference, Mapping) else None,
            ssp.REFERENCE_AUDIT_ID,
        ),
        "prompt_pool_dir": (experiment.get("prompt_pool_dir"), ssp.PROMPT_POOL_DIR),
    }
    return [
        f"{name}: {got!r} != {exp!r}" for name, (got, exp) in want.items() if got != exp
    ]


def judge_tasks_from_manifest(manifest: Mapping[str, object]) -> tuple[str, ...]:
    """Tasks with rows, in `TASK_ORDER`.

    Raises:
        ValueError: A task outside `TASK_ORDER` has rows, or no task has rows.
    """
    totals = {str(k): int(v) for k, v in dict(manifest["per_task_totals"]).items()}
    stray = sorted(t for t, n in totals.items() if n > 0 and t not in TASK_ORDER)
    if stray:
        raise ValueError(f"plan has rows for tasks outside the audit: {stray}")
    tasks = tuple(t for t in TASK_ORDER if totals.get(t, 0) > 0)
    if not tasks:
        raise ValueError("plan has no rows in any task")
    return tasks


def shard_names(manifest: Mapping[str, object], task: str) -> tuple[str, ...]:
    """Shard file names the manifest commits for `task`."""
    index = dict(manifest["shard_index"]).get(task) or []
    return tuple(str(row["shard"]) for row in index)


def probe_verdict(gate: ProbeGate, probe: Mapping[str, object]) -> ProbeVerdict:
    """Check ``throughput_probe.json`` against a gate (agreement excluded).

    Raises:
        ValueError: The probe document is for another task.
    """
    if str(probe.get("task")) != gate.task:
        raise ValueError(f"probe file is for {probe.get('task')!r}, gate {gate.task!r}")
    checks = (
        ("parse_failure_rate", gate.max_parse_failure, "max"),
        ("truncation_rate", gate.max_truncation, "max"),
        ("steady_state_rows_per_second", gate.min_rows_per_second, "min"),
    )
    failures: list[str] = []
    metrics: dict[str, float | int | None] = {"rows_judged": probe.get("rows_judged")}
    for key, limit, kind in checks:
        if limit is None:
            continue
        value = float(probe[key])
        metrics[key] = value
        if (kind == "max" and value > limit) or (kind == "min" and value < limit):
            failures.append(
                f"{key}={value:.4f} {'>' if kind == 'max' else '<'} {limit}"
            )
    return ProbeVerdict(gate.task, not failures, tuple(failures), metrics)


def agreement_verdict(
    gate: ProbeGate,
    new: Mapping[str, bool],
    ref: Mapping[str, bool],
) -> ProbeVerdict:
    """Yes/no agreement of probe judgments with reference judgments by row_id.

    Args:
        gate: Gate with ``min_agreement``.
        new: Parsed probe verdicts by ``row_id``.
        ref: Reference verdicts by ``row_id`` (pre-freeze rows already removed).

    Returns:
        The verdict; no overlapping rows fails.
    """
    shared = sorted(set(new) & set(ref))
    n_agree = sum(1 for rid in shared if new[rid] == ref[rid])
    agreement = n_agree / len(shared) if shared else None
    failures: list[str] = []
    if agreement is None:
        failures.append("no probe rows overlap the reference judgments")
    elif gate.min_agreement is not None and agreement < gate.min_agreement:
        failures.append(f"agreement={agreement:.4f} < {gate.min_agreement}")
    return ProbeVerdict(
        gate.task,
        not failures,
        tuple(failures),
        {"n_compared": len(shared), "n_agree": n_agree, "agreement": agreement},
    )


def failed_worker_gpus(log_text: str) -> tuple[int, ...]:
    """GPU ordinals ``judge-cluster`` reported as failed workers."""
    return tuple(
        sorted({int(m.group(2)) for m in _FAILED_WORKER_RE.finditer(log_text)})
    )


def attempt_gpus(
    gpus: Sequence[int],
    failures: Mapping[str, int],
    exclude_after: int,
) -> tuple[int, ...]:
    """GPUs for the next attempt, dropping ordinals with too many failures."""
    return tuple(g for g in gpus if int(failures.get(str(g), 0)) < exclude_after)


def relaunch_refusal(
    attempts: Sequence[Mapping[str, object]],
    judged_now: int,
    cap: int,
) -> str | None:
    """Why ``judge-cluster`` may not be relaunched, or ``None``.

    Args:
        attempts: Earlier launches, each with ``judged_before``.
        judged_now: Judged rows now.
        cap: `DriverConfig.attempt_cap`.
    """
    if len(attempts) >= cap:
        return f"attempt cap {cap} reached with the audit incomplete"
    if attempts and judged_now <= int(attempts[-1]["judged_before"]):
        return f"attempt {len(attempts)} made no progress ({judged_now} rows judged)"
    return None


def eta_seconds(
    judged: int,
    total: int,
    rows_per_second: float | None,
) -> float | None:
    """Seconds to finish at the observed rate."""
    if not rows_per_second or rows_per_second <= 0:
        return None
    return max(0, total - judged) / rows_per_second


def _opt_float(value: object) -> float | None:
    return None if value is None else float(value)
