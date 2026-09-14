"""Strictscreen judge-audit phase machine (plan §3 phases 6-12).

The campaign driver calls `advance_audit` on every watch pass after training is
final. Each phase records ``state["transitions"]["audit_*"]`` once done, so a
restarted driver resumes at the first missing phase. Long phases never block a
pass: GPU release and judging report ``waiting`` and are re-checked next pass;
only the probes run to completion inside one pass. Nothing here kills a
process, and the jobq stop sentinel is only removed when this driver created it.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from subliminal_learning.campaign import judge_audit_io as jio
from subliminal_learning.campaign import strictscreen_audit_plan as sap
from subliminal_learning.campaign import strictscreen_plan as ssp
from subliminal_learning.campaign.strictscreen_plan import CampaignGateError
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.evaluation.judge_results_io import (
    judge_yes_columns,
    load_rollout_summary,
    load_task_frame,
)
from subliminal_learning.gkd import judge_layout
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.run_io import cell_is_done, sha256_file, utc_now_iso

_DRIVER_SUBDIR = "driver"
_EXPECTED_FILE = "expected_run_dirs.json"
_SENTINEL_FILE = "stop_sentinel.json"
_VLLM_VERSION_CODE = "import vllm; print(vllm.__version__)"


@dataclass(frozen=True)
class AuditContext:
    """Where the audit driver runs and which campaign it serves.

    Attributes:
        pin_root: Code pin checkout the campaign runs from.
        host: ``hostname`` of this node.
        queue_root: jobq root shared with the campaign.
        output_root: `PathsConfig.output_root`.
        run_set_logs_dir: ``{output_root}/gkd_students/{RUN_SET_ID}/logs``.
        campaign_queue_names: The campaign's jobq queues.
        dry_run: Read-only: no writes, no subprocesses.
        allow_off_host: Test-only escape from the home-node check.
    """

    pin_root: Path
    host: str
    queue_root: Path
    output_root: Path
    run_set_logs_dir: Path
    campaign_queue_names: tuple[str, ...]
    dry_run: bool = False
    allow_off_host: bool = False


@dataclass(frozen=True)
class _Env:
    experiment_path: Path
    judge_path: Path
    paths_path: Path
    experiment: dict
    judge: dict
    driver: sap.DriverConfig
    audit_root: Path
    audit_dir: Path
    reference_dir: Path
    hf_cache_root: Path

    @property
    def driver_dir(self) -> Path:
        return judge_layout.logs_dir(self.audit_dir) / _DRIVER_SUBDIR


def audit_preflight(ctx: AuditContext) -> dict:
    """Judge-side launch gates; writes ``OWNER.json`` and ``judge_env.json``.

    Returns:
        The preflight report.

    Raises:
        CampaignGateError: Any gate fails.
    """
    env = _load_env(ctx)
    failures: list[str] = []
    if ctx.host != ssp.NODE and not ctx.allow_off_host:
        failures.append(f"host {ctx.host!r} is not the home node {ssp.NODE!r}")
    failures += sap.validate_audit_identity(env.experiment, env.judge)
    judge_root = Path(str(env.judge.get("audit_root")))
    if judge_root != env.audit_root:
        failures.append(
            f"paths audit_root {env.audit_root} != judge audit_root {judge_root}"
        )
    failures += jio.verify_source_hashes(ctx.pin_root, env.driver.tool_source_sha256)
    snapshot = jio.snapshot_dir(
        env.hf_cache_root, env.driver.model_id, env.driver.model_snapshot
    )
    if not (snapshot / "config.json").is_file():
        failures.append(f"judge snapshot missing offline: {snapshot}")
    refs_main = snapshot.parent.parent / "refs" / "main"
    resolved = refs_main.read_text().strip() if refs_main.is_file() else None
    if resolved != env.driver.model_snapshot:
        failures.append(
            f"offline {refs_main} resolves to {resolved!r}, not pinned "
            f"{env.driver.model_snapshot}"
        )
    vllm_version = None
    if not ctx.dry_run:
        try:
            out = jio.capture(
                jio.tool_python_cmd(ctx.pin_root, _VLLM_VERSION_CODE),
                cwd=ctx.pin_root,
                env=jio.tool_env(env.hf_cache_root, None),
            )
            vllm_version = out.strip().splitlines()[-1] if out.strip() else ""
        except (subprocess.CalledProcessError, OSError) as err:
            failures.append(f"judge venv cannot import vllm: {err}")
        if vllm_version is not None and vllm_version != env.driver.vllm_version:
            failures.append(
                f"judge venv vllm {vllm_version!r} != {env.driver.vllm_version!r}"
            )
    owner = _owner_problem(env)
    if owner:
        failures.append(owner)
    free = jio.free_gib(env.audit_root)
    if free < env.driver.min_cold_free_gib:
        failures.append(
            f"audit_root free {free:.1f} GiB < {env.driver.min_cold_free_gib}"
        )
    try:
        with judge_layout.plan_manifest_path(env.reference_dir).open("rb") as handle:
            handle.read(1)
    except OSError as err:
        failures.append(f"reference audit unreadable: {err}")
    if failures:
        raise CampaignGateError("audit preflight: " + "; ".join(failures))
    report = {
        "host": ctx.host,
        "pin_root": str(ctx.pin_root),
        "audit_dir": str(env.audit_dir),
        "reference_dir": str(env.reference_dir),
        "judge_config": str(env.judge_path),
        "judge_config_sha256": sha256_file(env.judge_path),
        "experiment_config_sha256": sha256_file(env.experiment_path),
        "tool_source_sha256": dict(env.driver.tool_source_sha256),
        "vllm_version": vllm_version,
        "model_snapshot_dir": str(snapshot),
        "hf_cache_root": str(env.hf_cache_root),
        "audit_root_free_gib": free,
        "dry_run": ctx.dry_run,
        "at": utc_now_iso(),
    }
    if not ctx.dry_run:
        owner_path = env.driver_dir / sap.OWNER_FILE
        if not owner_path.is_file():
            jio.write_json(owner_path, {"run_set_id": ssp.RUN_SET_ID, "host": ctx.host})
        jio.write_json(
            judge_layout.logs_dir(env.audit_dir) / sap.JUDGE_ENV_FILE, report
        )
    return report


def advance_audit(
    ctx: AuditContext,
    state: dict,
    fired: list[str],
    waiting: list[str],
    *,
    required_run_dirs: Sequence[Path],
    optional_run_dirs: Sequence[Path],
) -> None:
    """One pass over the audit phases; the caller saves `state` afterwards.

    Raises:
        CampaignGateError: A gate blocked the audit (also recorded in
            ``state["audit_blocked"]`` and ``logs/AUDIT_BLOCKED.json``).
    """
    blocked = state.get("audit_blocked")
    if blocked:
        raise CampaignGateError(
            f"audit blocked in {blocked.get('phase')}: {blocked.get('reason')} "
            "(fix, then audit-unblock)"
        )
    transitions = state.setdefault("transitions", {})
    state.setdefault("audit", {})
    env = _load_env(ctx)
    phase = _next_phase(transitions)
    if phase is None:
        return
    if ctx.dry_run:
        fired.append(f"[dry-run] audit next phase: {phase}")
        if phase == "audit_completeness":
            ready, missing, _ = _classify_leaves(
                ctx, required_run_dirs, optional_run_dirs
            )
            waiting.append(
                f"[dry-run] {len(ready)} audit leaves ready, {len(missing)} required missing"
            )
        return
    _archive_block_file(env)
    try:
        if "audit_completeness" not in transitions:
            _completeness(ctx, env, state, required_run_dirs, optional_run_dirs)
            fired.append("audit_completeness")
        if "audit_plan" not in transitions:
            phase = "audit_plan"
            _plan(ctx, env, state)
            fired.append("audit_plan")
        if "audit_release_gpus" not in transitions:
            phase = "audit_release_gpus"
            if not _release_gpus(ctx, env, state, waiting):
                return
            fired.append("audit_release_gpus")
        if "audit_probe" not in transitions:
            phase = "audit_probe"
            _probe(ctx, env, state)
            fired.append("audit_probe")
        if "audit_judge" not in transitions:
            phase = "audit_judge"
            if not _judge(ctx, env, state, waiting):
                return
            fired.append("audit_judge")
        if "audit_return_gpus" not in transitions:
            phase = "audit_return_gpus"
            _return_gpus(ctx, state)
            fired.append("audit_return_gpus")
        if "audit_results" not in transitions:
            phase = "audit_results"
            _results(ctx, env, state)
            fired.append("audit_results")
        if "audit_finalize" not in transitions:
            phase = "audit_finalize"
            _finalize(ctx, env, state)
            fired.append("audit_finalize")
    except CampaignGateError as err:
        _block(ctx, env, state, phase, str(err))
        raise
    finally:
        if env.audit_dir.is_dir():
            _write_status(env, state, waiting)


def audit_done(state: Mapping[str, object]) -> bool:
    """Whether the audit finalized."""
    transitions = state.get("transitions") or {}
    return "audit_finalize" in transitions


def audit_status(ctx: AuditContext, state: Mapping[str, object]) -> dict:
    """Read-only audit snapshot for the campaign status CLI."""
    transitions = dict(state.get("transitions") or {})
    out = {
        "transitions": {k: v for k, v in transitions.items() if k in sap.PHASES},
        "next_phase": _next_phase(transitions),
        "blocked": state.get("audit_blocked"),
        "audit": state.get("audit"),
        "done": audit_done(state),
        "status_file": None,
    }
    try:
        env = _load_env(ctx)
    except CampaignGateError as err:
        out["env_error"] = str(err)
        return out
    path = judge_layout.logs_dir(env.audit_dir) / sap.AUDIT_STATUS_FILE
    if path.is_file():
        out["status_file"] = jio.read_json(path)
    return out


def audit_unblock(state: dict) -> None:
    """Clear a block; failed probe verdicts and the judge attempt count reset."""
    state.pop("audit_blocked", None)
    audit = state.setdefault("audit", {})
    probes = audit.get("probe") or {}
    audit["probe"] = {task: doc for task, doc in probes.items() if doc.get("passed")}
    judge = audit.get("judge")
    if judge and judge.get("attempts"):
        judge.setdefault("attempt_history", []).extend(judge["attempts"])
        judge["attempts"] = []


def _load_env(ctx: AuditContext) -> _Env:
    try:
        experiment_path = ctx.pin_root / sap.EXPERIMENT_CONFIG_REL
        experiment = yaml.safe_load(experiment_path.read_text())
        driver = sap.parse_driver_config(experiment["driver"])
        judge_path = ctx.pin_root / driver.judge_config
        judge = yaml.safe_load(judge_path.read_text())
        paths_path = ctx.pin_root / sap.PATHS_CONFIG_REL
        paths = yaml.safe_load(paths_path.read_text())
        audit_root = Path(str(paths["audit_root"]))
        return _Env(
            experiment_path=experiment_path,
            judge_path=judge_path,
            paths_path=paths_path,
            experiment=experiment,
            judge=judge,
            driver=driver,
            audit_root=audit_root,
            audit_dir=judge_layout.audit_dir(audit_root, str(experiment["audit_id"])),
            reference_dir=judge_layout.audit_dir(
                audit_root, str(experiment["reference"]["audit_id"])
            ),
            hf_cache_root=Path(str(judge["hf_cache_root"])),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as err:
        raise CampaignGateError(f"audit config unreadable: {err!r}") from err


def _next_phase(transitions: Mapping[str, object]) -> str | None:
    return next((p for p in sap.PHASES if p not in transitions), None)


def _owner_problem(env: _Env) -> str | None:
    if not env.audit_dir.exists():
        return None
    owner_path = env.driver_dir / sap.OWNER_FILE
    if not owner_path.is_file():
        return f"audit dir {env.audit_dir} exists without {owner_path.name}"
    owner = jio.read_json(owner_path)
    if owner.get("run_set_id") != ssp.RUN_SET_ID:
        return f"audit dir owned by {owner.get('run_set_id')!r}"
    return None


def _classify_leaves(
    ctx: AuditContext,
    required: Sequence[Path],
    optional: Sequence[Path],
) -> tuple[list[str], list[dict], list[dict]]:
    campaign_root = Path(ctx.output_root) / "gkd_students" / ssp.RUN_SET_ID
    ready: list[str] = []
    missing: list[dict] = []
    excluded: list[dict] = []
    for kind, dirs in (("required", required), ("optional", optional)):
        for run_dir in dirs:
            path = Path(run_dir)
            try:
                path.relative_to(campaign_root)
            except ValueError as err:
                raise CampaignGateError(f"{path} is not under {campaign_root}") from err
            problem = _leaf_problem(path)
            if problem is None:
                ready.append(str(path))
            elif kind == "required":
                missing.append({"run_dir": str(path), "reason": problem})
            else:
                excluded.append({"run_dir": str(path), "reason": problem})
    return sorted(ready), missing, excluded


def _leaf_problem(run_dir: Path) -> str | None:
    if not cell_is_done(run_dir):
        return "cell_not_done"
    try:
        load_rollout_summary(run_dir)
    except (OSError, ValueError, json.JSONDecodeError) as err:
        return f"invalid_summary:{err}"
    sample = run_dir / "rollouts_sample.jsonl"
    if not sample.is_file() or sample.stat().st_size == 0:
        return "empty_rollouts_sample"
    try:
        parse_gkd_tag(run_dir.parent.name)
    except ValueError:
        return "unparseable_gkd_tag"
    return None


def _completeness(
    ctx: AuditContext,
    env: _Env,
    state: dict,
    required: Sequence[Path],
    optional: Sequence[Path],
) -> None:
    ready, missing, excluded = _classify_leaves(ctx, required, optional)
    if missing:
        raise CampaignGateError(
            f"{len(missing)} required audit leaves not ready: {missing[:5]}"
        )
    owner = _owner_problem(env)
    if owner:
        raise CampaignGateError(owner)
    owner_path = env.driver_dir / sap.OWNER_FILE
    if not owner_path.is_file():
        jio.write_json(owner_path, {"run_set_id": ssp.RUN_SET_ID, "host": ctx.host})
    doc = {
        "run_set_id": ssp.RUN_SET_ID,
        "run_dirs": ready,
        "n_required": len(required),
        "n_optional_included": len(ready) - len(required),
        "optional_excluded": excluded,
        "at": utc_now_iso(),
    }
    jio.write_json(env.driver_dir / _EXPECTED_FILE, doc)
    state["audit"]["expected_run_dirs"] = {
        "path": str(env.driver_dir / _EXPECTED_FILE),
        "n": len(ready),
        "optional_excluded": len(excluded),
    }
    state["transitions"]["audit_completeness"] = {
        "at": utc_now_iso(),
        "n_leaves": len(ready),
    }


def _plan(ctx: AuditContext, env: _Env, state: dict) -> None:
    cmd = jio.repo_cmd(
        "scripts.build_judge_plan",
        "--paths-config",
        str(env.paths_path),
        "--experiment-config",
        str(env.experiment_path),
        "--expected-run-dirs-file",
        str(env.driver_dir / _EXPECTED_FILE),
        "--judge-config",
        str(env.judge_path),
    )
    rc = jio.run_blocking(
        cmd, cwd=ctx.pin_root, env=None, log_path=env.driver_dir / "plan_build.log"
    )
    if rc != 0:
        raise CampaignGateError(f"build_judge_plan exited {rc}; see plan_build.log")
    manifest = jio.read_json(judge_layout.plan_manifest_path(env.audit_dir))
    if manifest.get("judge_config_sha256") != sha256_file(env.judge_path):
        raise CampaignGateError("plan manifest judge_config_sha256 != judge YAML")
    problems = _plan_integrity(env, manifest)
    if problems:
        raise CampaignGateError("plan verification: " + "; ".join(problems))
    tasks = sap.judge_tasks_from_manifest(manifest)
    state["audit"]["tasks"] = list(tasks)
    state["audit"]["plan_fingerprint"] = manifest.get("plan_fingerprint")
    state["transitions"]["audit_plan"] = {
        "at": utc_now_iso(),
        "tasks": list(tasks),
        "per_task_totals": manifest["per_task_totals"],
    }


def _plan_integrity(env: _Env, manifest: Mapping[str, object]) -> list[str]:
    problems = []
    reference = jio.read_json(judge_layout.plan_manifest_path(env.reference_dir))
    for task in sap.POOL_TASKS:
        names = sap.shard_names(manifest, task)
        if names != sap.shard_names(reference, task):
            problems.append(f"{task}: shard set differs from the reference audit")
            continue
        for name in names:
            ours = judge_layout.queue_shard_dir(env.audit_dir, task) / name
            theirs = judge_layout.queue_shard_dir(env.reference_dir, task) / name
            if sha256_file(ours) != sha256_file(theirs):
                problems.append(f"{task}/{name}: queue bytes differ from the reference")
    totals = {str(k): int(v) for k, v in dict(manifest["per_task_totals"]).items()}
    for task in judge_layout.TASKS:
        qdir = judge_layout.queue_shard_dir(env.audit_dir, task)
        on_disk = (
            tuple(sorted(p.name for p in qdir.glob("shard-*.jsonl")))
            if qdir.is_dir()
            else ()
        )
        if on_disk != sap.shard_names(manifest, task):
            problems.append(f"{task}: queue dir {on_disk} != shard_index")
        if totals.get(task, 0) == 0 and qdir.exists():
            problems.append(f"{task}: zero-row task has a queue dir")
    return problems


def _sentinel_owned(env: _Env, state: dict, stop: Path) -> bool:
    """Record (once, on disk first) whether the stop sentinel predates the audit."""
    audit = state["audit"]
    marker = env.driver_dir / _SENTINEL_FILE
    if "sentinel_preexisting" not in audit:
        if marker.is_file():
            audit["sentinel_preexisting"] = bool(jio.read_json(marker)["preexisting"])
        else:
            audit["sentinel_preexisting"] = stop.exists()
            jio.write_json(
                marker,
                {
                    "preexisting": audit["sentinel_preexisting"],
                    "path": str(stop),
                    "at": utc_now_iso(),
                },
            )
    return not audit["sentinel_preexisting"]


def _release_gpus(
    ctx: AuditContext, env: _Env, state: dict, waiting: list[str]
) -> bool:
    stop = store.stop_path(ctx.queue_root, ctx.host)
    _sentinel_owned(env, state, stop)
    if not stop.exists():
        stop.touch()
        logger.info("touched jobq stop sentinel {}", stop)
    first = _release_sample(ctx, env)
    if first:
        waiting.append(f"releasing GPUs: {first}")
        return False
    time.sleep(env.driver.release_sample_gap_s)
    second = _release_sample(ctx, env)
    if second:
        waiting.append(f"releasing GPUs (second sample): {second}")
        return False
    marker = {"at": utc_now_iso(), "host": ctx.host, "audit_id": ssp.AUDIT_ID}
    jio.write_json(ctx.run_set_logs_dir / ssp.GPUS_RELEASED_FILE, marker)
    state["transitions"]["audit_release_gpus"] = marker
    return True


def _release_sample(ctx: AuditContext, env: _Env) -> str | None:
    """Why the GPUs are not released yet, or ``None``."""
    running = jio.host_running_jobs(ctx.queue_root, ctx.host, ctx.campaign_queue_names)
    if running:
        return f"{len(running)} jobq jobs still running on {ctx.host}"
    loop_exited = (ctx.run_set_logs_dir / ssp.WORKER_LOOP_EXITED_FILE).exists()
    if not loop_exited and jio.worker_pool_alive(ctx.queue_root):
        return "jobq worker loop still alive"
    try:
        apps = jio.gpu_compute_apps()
        used = jio.gpu_memory_used_mib()
    except (subprocess.CalledProcessError, OSError, ValueError) as err:
        return f"nvidia-smi failed ({err}); GPU ownership unknown"
    if apps:
        return f"{len(apps)} GPU compute apps still present"
    busy = {g: m for g, m in used.items() if m > env.driver.max_idle_gpu_mib}
    if busy:
        return f"GPU memory still held (MiB): {busy}"
    return None


def _probe(ctx: AuditContext, env: _Env, state: dict) -> None:
    mismatches = jio.verify_source_hashes(ctx.pin_root, env.driver.tool_source_sha256)
    if mismatches:
        raise CampaignGateError(f"judge sources drifted: {mismatches}")
    manifest = jio.read_json(judge_layout.plan_manifest_path(env.audit_dir))
    probes = state["audit"].setdefault("probe", {})
    for gate in env.driver.probe_gates:
        if probes.get(gate.task, {}).get("passed"):
            continue
        first = dict(manifest["shard_index"])[gate.task][0]
        expected_rows = min(gate.n, int(first["n_rows"]))
        probe_doc = _run_probe(ctx, env, gate, probes)
        if int(probe_doc.get("rows_judged", -1)) != expected_rows:
            raise CampaignGateError(
                f"probe {gate.task} judged {probe_doc.get('rows_judged')} rows, "
                f"expected {expected_rows}"
            )
        verdict = sap.probe_verdict(gate, probe_doc)
        failures = list(verdict.failures)
        metrics = dict(verdict.metrics)
        if gate.min_agreement is not None:
            agreement = sap.agreement_verdict(
                gate, *_probe_agreement_inputs(env, gate.task)
            )
            failures += agreement.failures
            metrics.update(agreement.metrics)
        probes[gate.task] = {
            "passed": not failures,
            "failures": failures,
            "metrics": metrics,
            "at": utc_now_iso(),
            "crashes": probes.get(gate.task, {}).get("crashes", 0),
        }
        if failures:
            raise CampaignGateError(f"probe {gate.task} failed: {failures}")
    state["transitions"]["audit_probe"] = {"at": utc_now_iso(), "probes": probes}


def _run_probe(ctx: AuditContext, env: _Env, gate: sap.ProbeGate, probes: dict) -> dict:
    cmd = jio.tool_cmd(
        ctx.pin_root,
        "probe",
        "--config",
        str(env.judge_path),
        "--task",
        gate.task,
        "--n",
        str(gate.n),
    )
    log_path = env.driver_dir / f"probe_{gate.task}.log"
    record = probes.setdefault(gate.task, {"passed": False, "crashes": 0})
    while True:
        rc = jio.run_blocking(
            cmd,
            cwd=ctx.pin_root,
            env=jio.tool_env(env.hf_cache_root, "0"),
            log_path=log_path,
        )
        if rc == 0:
            break
        record["crashes"] = int(record.get("crashes", 0)) + 1
        if record["crashes"] > env.driver.probe_retries:
            raise CampaignGateError(
                f"probe {gate.task} crashed {record['crashes']} times (last rc={rc})"
            )
        logger.warning("probe {} exited {}; retry {}", gate.task, rc, record["crashes"])
    source = judge_layout.logs_dir(env.audit_dir) / "throughput_probe.json"
    doc = jio.read_json(source)
    jio.write_json(env.driver_dir / f"probe_{gate.task}.json", doc)
    return doc


def _probe_agreement_inputs(
    env: _Env,
    task: str,
) -> tuple[dict[str, bool], dict[str, bool]]:
    samples = judge_layout.logs_dir(env.audit_dir) / "probe_samples" / f"{task}.jsonl"
    new = judge_yes_columns(_read_jsonl_frame(samples), task)
    new_yes = {
        str(r): bool(y)
        for r, y, ok in zip(
            new["row_id"], new["judge_yes"], new["judge_valid"], strict=True
        )
        if ok
    }
    ref = judge_yes_columns(load_task_frame(env.reference_dir, task, "judgments"), task)
    frozen = (env.experiment.get("reference") or {}).get("prompt_frozen_at")
    if frozen is not None:
        ref = ref.loc[ref["judged_at"].astype(str) >= str(frozen)]
    ref = ref.loc[ref["judge_valid"].astype(bool) & ref["row_id"].isin(set(new_yes))]
    ref_yes = {
        str(r): bool(y) for r, y in zip(ref["row_id"], ref["judge_yes"], strict=True)
    }
    return new_yes, ref_yes


def _read_jsonl_frame(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame([json.loads(line) for line in handle if line.strip()])


def _judge(ctx: AuditContext, env: _Env, state: dict, waiting: list[str]) -> bool:
    audit = state["audit"]
    judge = audit.setdefault("judge", {"attempts": [], "gpu_failures": {}})
    judge.setdefault("attempts", [])
    judge.setdefault("gpu_failures", {})
    manifest = jio.read_json(judge_layout.plan_manifest_path(env.audit_dir))
    tasks = [str(t) for t in audit["tasks"]]
    shards = {t: sap.shard_names(manifest, t) for t in tasks}
    total = sum(int(manifest["per_task_totals"][t]) for t in tasks)
    judged = sum(jio.judged_rows(env.audit_dir, t, shards[t]) for t in tasks)
    _record_progress(judge, judged, total)

    if jio.cluster_lock_held(env.audit_dir):
        waiting.append(f"judge-cluster running: {judged}/{total} rows judged")
        return False
    attempts = judge["attempts"]
    if attempts and "finished_at" not in attempts[-1]:
        last = attempts[-1]
        code = jio.read_exit_code(Path(last["exit_file"]))
        if code is None and last.get("pid") and store.pid_alive(int(last["pid"])):
            waiting.append(f"judge-cluster attempt {last['attempt']} starting")
            return False
        last["finished_at"] = utc_now_iso()
        last["exit_code"] = code
        log_text = Path(last["log"]).read_text() if Path(last["log"]).is_file() else ""
        last["failed_gpus"] = list(sap.failed_worker_gpus(log_text))
        for gpu in last["failed_gpus"]:
            judge["gpu_failures"][str(gpu)] = (
                int(judge["gpu_failures"].get(str(gpu), 0)) + 1
            )

    if judged >= total:
        coverage = [jio.task_coverage(env.audit_dir, t, shards[t]) for t in tasks]
        if all(c.complete for c in coverage):
            state["transitions"]["audit_judge"] = {
                "at": utc_now_iso(),
                "n_attempts": len(attempts),
                "coverage": {c.task: c.n_judged for c in coverage},
            }
            return True
        judge["incomplete"] = {
            c.task: dict(c.incomplete_shards) for c in coverage if not c.complete
        }

    orphans = jio.judge_processes(env.judge_path)
    if orphans:
        raise CampaignGateError(
            f"orphan judge workers without the cluster lock: {orphans[:3]}"
        )
    try:
        apps = jio.gpu_compute_apps()
        used = jio.gpu_memory_used_mib()
    except (subprocess.CalledProcessError, OSError, ValueError) as err:
        raise CampaignGateError(f"nvidia-smi failed before relaunch: {err}") from err
    busy = {g: m for g, m in used.items() if m > env.driver.max_idle_gpu_mib}
    if apps or busy:
        raise CampaignGateError(
            f"GPU memory held before relaunch: apps={len(apps)} busy={busy}"
        )
    refusal = sap.relaunch_refusal(attempts, judged, env.driver.attempt_cap)
    if refusal:
        raise CampaignGateError(refusal)
    gpus = sap.attempt_gpus(
        env.driver.gpus, judge["gpu_failures"], env.driver.gpu_fail_exclude_after
    )
    if not gpus:
        raise CampaignGateError(
            f"every GPU excluded after failures {judge['gpu_failures']}"
        )
    number = len(attempts) + 1
    task_flags = [flag for t in tasks for flag in ("--task", t)]
    cmd = jio.tool_cmd(
        ctx.pin_root,
        "judge-cluster",
        "--config",
        str(env.judge_path),
        *task_flags,
        "--gpus",
        ",".join(str(g) for g in gpus),
    )
    log_path = env.driver_dir / f"judge_cluster_attempt{number}.log"
    exit_path = env.driver_dir / f"judge_cluster_attempt{number}.exit"
    attempt = {
        "attempt": number,
        "at": utc_now_iso(),
        "gpus": list(gpus),
        "judged_before": judged,
        "log": str(log_path),
        "exit_file": str(exit_path),
        "cmd": cmd,
        "pid": None,
    }
    attempts.append(attempt)
    attempt["pid"] = jio.launch_detached(
        cmd,
        cwd=ctx.pin_root,
        env=jio.tool_env(env.hf_cache_root, None),
        log_path=log_path,
        exit_path=exit_path,
    )
    waiting.append(f"launched judge-cluster attempt {number} on GPUs {list(gpus)}")
    return False


def _record_progress(judge: dict, judged: int, total: int) -> None:
    now = time.time()
    last = judge.get("last_sample")
    rate = None
    if last and now > float(last["epoch"]) and judged >= int(last["judged"]):
        rate = (judged - int(last["judged"])) / (now - float(last["epoch"]))
    judge["last_sample"] = {"epoch": now, "judged": judged}
    judge["progress"] = {
        "judged": judged,
        "total": total,
        "rows_per_second": rate,
        "eta_seconds": sap.eta_seconds(judged, total, rate),
        "at": utc_now_iso(),
    }


def _return_gpus(ctx: AuditContext, state: dict) -> None:
    stop = store.stop_path(ctx.queue_root, ctx.host)
    removed = False
    if state["audit"].get("sentinel_preexisting") is False and stop.exists():
        stop.unlink()
        removed = True
        logger.info("removed the jobq stop sentinel this driver created: {}", stop)
    state["transitions"]["audit_return_gpus"] = {
        "at": utc_now_iso(),
        "sentinel_removed": removed,
    }


def _results(ctx: AuditContext, env: _Env, state: dict) -> None:
    cmd = jio.repo_cmd(
        "scripts.build_strictscreen_audit_results",
        "--paths-config",
        str(env.paths_path),
        "--experiment-config",
        str(env.experiment_path),
    )
    rc = jio.run_blocking(
        cmd, cwd=ctx.pin_root, env=None, log_path=env.driver_dir / "results.log"
    )
    if rc != 0:
        raise CampaignGateError(f"build_strictscreen_audit_results exited {rc}")
    claims = jio.read_json(judge_layout.results_dir(env.audit_dir) / "claims.json")
    state["audit"]["headline"] = claims["headline"]
    state["transitions"]["audit_results"] = {
        "at": utc_now_iso(),
        "headline": claims["headline"],
    }


def _finalize(ctx: AuditContext, env: _Env, state: dict) -> None:
    payload = {
        "audit_id": ssp.AUDIT_ID,
        "run_set_id": ssp.RUN_SET_ID,
        "headline": state["audit"].get("headline"),
        "claims": str(judge_layout.results_dir(env.audit_dir) / "claims.json"),
        "at": utc_now_iso(),
    }
    jio.write_json(env.audit_dir / ssp.AUDIT_DONE_FILE, payload)
    jio.write_json(ctx.run_set_logs_dir / ssp.AUDIT_DONE_FILE, payload)
    state["transitions"]["audit_finalize"] = payload


def _block(ctx: AuditContext, env: _Env, state: dict, phase: str, reason: str) -> None:
    stop = store.stop_path(ctx.queue_root, ctx.host)
    cleared = False
    if state.get("audit", {}).get("sentinel_preexisting") is False and stop.exists():
        stop.unlink()
        cleared = True
    blocked = {
        "phase": phase,
        "reason": reason,
        "at": utc_now_iso(),
        "sentinel_cleared": cleared,
    }
    state["audit_blocked"] = blocked
    if env.audit_dir.is_dir():
        jio.write_json(
            judge_layout.logs_dir(env.audit_dir) / sap.AUDIT_BLOCKED_FILE, blocked
        )
    logger.error("audit blocked in {}: {}", phase, reason)


def _archive_block_file(env: _Env) -> None:
    path = judge_layout.logs_dir(env.audit_dir) / sap.AUDIT_BLOCKED_FILE
    if path.is_file():
        dest = env.driver_dir / f"AUDIT_BLOCKED.{store.utc_stamp()}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, dest)


def _write_status(
    env: _Env, state: Mapping[str, object], waiting: Sequence[str]
) -> None:
    audit = dict(state.get("audit") or {})
    judge = dict(audit.get("judge") or {})
    transitions = dict(state.get("transitions") or {})
    attempts = judge.get("attempts") or []
    payload = {
        "phase": _next_phase(transitions),
        "done": "audit_finalize" in transitions,
        "blocked": state.get("audit_blocked"),
        "attempt": len(attempts),
        "progress": judge.get("progress"),
        "probe_verdicts": audit.get("probe"),
        "excluded_gpus": [
            int(g)
            for g, n in (judge.get("gpu_failures") or {}).items()
            if int(n) >= env.driver.gpu_fail_exclude_after
        ],
        "waiting": list(waiting),
        "last_error": (state.get("audit_blocked") or {}).get("reason"),
        "at": utc_now_iso(),
    }
    jio.write_json(
        judge_layout.logs_dir(env.audit_dir) / sap.AUDIT_STATUS_FILE, payload
    )
