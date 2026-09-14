"""CLI: offline Qwen3 judge over blind filtration-audit queue shards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import fcntl
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from rollout_judge.layout import TASKS
from rollout_judge.layout import audit_dir
from rollout_judge.layout import judgment_shard_path
from rollout_judge.layout import list_queue_shards
from rollout_judge.layout import logs_dir
from rollout_judge.layout import partition_shards
from rollout_judge.prompts import messages_for_row
from rollout_judge.queue import append_jsonl
from rollout_judge.queue import judged_row_ids
from rollout_judge.queue import read_queue_rows
from rollout_judge.queue import unjudged_rows
from rollout_judge.schema import build_output_row
from rollout_judge.schema import json_schema_for_task
from rollout_judge.schema import parse_judgment

app = typer.Typer(add_completion=False, no_args_is_help=True)

_YAML_REQUIRED: tuple[str, ...] = (
    "model_id",
    "quantization",
    "max_model_len",
    "gpu_memory_utilization",
    "dtype",
    "seed",
    "batch_size",
    "max_tokens",
    "audit_root",
    "audit_id",
    "tasks",
    "hf_cache_root",
    "guided_decoding_backend",
    "gpus",
    "max_num_batched_tokens",
    "max_num_seqs",
)

_GUIDED_BACKENDS: frozenset[str] = frozenset({"auto", "xgrammar", "guidance"})
_XGRAMMAR_OPTIONS: frozenset[str] = frozenset({"disable-any-whitespace", "no-fallback"})


@dataclass(frozen=True)
class JudgeConfig:
    """Resolved judge YAML.

    Attributes:
        model_id: Hugging Face model id.
        quantization: vLLM quantization method.
        max_model_len: KV-cache context length.
        gpu_memory_utilization: vLLM GPU memory fraction.
        dtype: vLLM dtype string.
        seed: Fixed sampling seed.
        batch_size: Rows per ``generate`` call.
        max_tokens: Max new tokens per judgment.
        audit_root: Filesystem root for filtration audits.
        audit_id: Audit run id.
        tasks: Default task list.
        hf_cache_root: Hub cache directory.
        tensor_parallel_size: GPU tensor-parallel degree.
        guided_decoding_backend: Engine-level structured-output backend.
        gpus: Device ordinals ``judge-cluster`` shards one worker onto each.
        max_num_batched_tokens: Chunked-prefill token budget per step.
        max_num_seqs: Max concurrently running sequences.
    """

    model_id: str
    quantization: str
    max_model_len: int
    gpu_memory_utilization: float
    dtype: str
    seed: int
    batch_size: int
    max_tokens: int
    audit_root: Path
    audit_id: str
    tasks: tuple[str, ...]
    hf_cache_root: Path
    tensor_parallel_size: int
    guided_decoding_backend: str
    gpus: tuple[int, ...]
    max_num_batched_tokens: int
    max_num_seqs: int


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_judge_config(path: Path) -> JudgeConfig:
    """Load and validate ``configs/rollout_judge.yaml``.

    Args:
        path: YAML path.

    Returns:
        Frozen config.

    Raises:
        ValueError: If required keys are missing or a task is unknown.
    """
    raw = _load_yaml(path)
    missing = [key for key in _YAML_REQUIRED if key not in raw]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    tasks = tuple(str(item) for item in raw["tasks"])
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise ValueError(f"unknown tasks in YAML: {unknown}")
    if not tasks:
        raise ValueError("YAML tasks list is empty")
    gpus = tuple(int(item) for item in raw["gpus"])
    if not gpus:
        raise ValueError("YAML gpus list is empty")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate device ordinals in gpus: {list(gpus)}")
    backend = str(raw["guided_decoding_backend"])
    _validate_guided_backend(backend)
    tp = raw.get("tensor_parallel_size", 1)
    return JudgeConfig(
        model_id=str(raw["model_id"]),
        quantization=str(raw["quantization"]),
        max_model_len=int(raw["max_model_len"]),
        gpu_memory_utilization=float(raw["gpu_memory_utilization"]),
        dtype=str(raw["dtype"]),
        seed=int(raw["seed"]),
        batch_size=int(raw["batch_size"]),
        max_tokens=int(raw["max_tokens"]),
        audit_root=Path(str(raw["audit_root"])),
        audit_id=str(raw["audit_id"]),
        tasks=tasks,
        hf_cache_root=Path(str(raw["hf_cache_root"])),
        tensor_parallel_size=int(tp),
        guided_decoding_backend=backend,
        gpus=gpus,
        max_num_batched_tokens=int(raw["max_num_batched_tokens"]),
        max_num_seqs=int(raw["max_num_seqs"]),
    )


def _validate_guided_backend(backend: str) -> None:
    """Reject a backend string the engine would only refuse after model load.

    Args:
        backend: ``name`` or ``name:opt[,opt]`` as vLLM parses it.

    Raises:
        ValueError: On an unknown backend name or xgrammar option.
    """
    name, _, options = backend.partition(":")
    if name not in _GUIDED_BACKENDS:
        raise ValueError(
            f"unknown guided_decoding_backend {name!r}; "
            f"expected one of {sorted(_GUIDED_BACKENDS)}"
        )
    if name != "xgrammar":
        return
    unknown = [
        opt
        for opt in (options.split(",") if options else [])
        if opt not in _XGRAMMAR_OPTIONS
    ]
    if unknown:
        raise ValueError(
            f"unknown xgrammar options {unknown}; "
            f"expected from {sorted(_XGRAMMAR_OPTIONS)}"
        )


def _resolve_config(config: Path | None) -> JudgeConfig:
    path = config or (_repo_root() / "configs" / "rollout_judge.yaml")
    return load_judge_config(path)


def _resolve_tasks(
    requested: list[str] | None,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    if not requested:
        return default
    unknown = [task for task in requested if task not in TASKS]
    if unknown:
        raise ValueError(f"unknown --task values: {unknown}")
    return tuple(requested)


def _now_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@contextmanager
def _file_lock(path: Path, label: str) -> Iterator[None]:
    """Hold a non-blocking advisory lock.

    Args:
        path: Lock file.
        label: Human-readable resource name.

    Raises:
        RuntimeError: If another process already holds the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise RuntimeError(f"{label} is already locked: {path}") from err
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _shard_lock_path(audit: Path, task_name: str, queue_path: Path) -> Path:
    return logs_dir(audit) / "locks" / f"{task_name}-{queue_path.stem}.lock"


def _shard_paths(
    audit: Path,
    tasks: tuple[str, ...],
    shard_limit: int | None,
) -> list[tuple[str, Path]]:
    """Queue shard paths to consider, without reading any of them.

    The biggest task is ~1.85M rows across ~93 shards; materialising every
    shard's pending rows up front would hold gigabytes of dicts, so shards are
    read one at a time inside the judging loop.
    """
    out: list[tuple[str, Path]] = []
    for task in tasks:
        shards = list_queue_shards(audit, task)
        if shard_limit is not None:
            shards = shards[:shard_limit]
        out.extend((task, path) for path in shards)
    return out


def _load_backend(cfg: JudgeConfig) -> Any:
    # vLLM is a tool-venv GPU dep; a module-level import breaks CPU tests.
    from rollout_judge.vllm_io import VllmBackend

    return VllmBackend(
        cfg.model_id,
        cfg.hf_cache_root,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        quantization=cfg.quantization,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        max_model_len=cfg.max_model_len,
        dtype=cfg.dtype,
        guided_decoding_backend=cfg.guided_decoding_backend,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        max_num_seqs=cfg.max_num_seqs,
    )


def _judge_rows(
    backend: Any,
    rows: list[dict[str, Any]],
    *,
    model_id: str,
) -> tuple[list[dict[str, Any]], list[Any]]:
    """Judge one same-task batch.

    Returns:
        On-disk judgment records and their raw generations.
    """
    if not rows:
        return [], []
    task = str(rows[0]["task"])
    if any(str(row["task"]) != task for row in rows):
        raise ValueError("mixed tasks in one generate batch")
    schema = json_schema_for_task(task)
    messages = [messages_for_row(row) for row in rows]
    generations = backend.generate_batch(messages, schema)
    judged_at = _now_utc()
    out: list[dict[str, Any]] = []
    for row, generation in zip(rows, generations, strict=True):
        parsed = parse_judgment(generation.text, task=str(row["task"]))
        out.append(
            build_output_row(
                row,
                parsed,
                model_id=model_id,
                judged_at=judged_at,
            )
        )
    return out, list(generations)


@app.command()
def judge(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Judge YAML (default: repo configs/rollout_judge.yaml).",
    ),
    task: list[str] = typer.Option(
        [],
        "--task",
        help="Task to judge (repeatable). Default: YAML task list.",
    ),
    shard_limit: int | None = typer.Option(
        None,
        "--shard-limit",
        help="Max shards per task (smoke).",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="Max new judgments this invocation (smoke).",
    ),
    worker_index: int = typer.Option(
        0,
        "--worker-index",
        help="This worker's index in a shard-parallel cluster run.",
    ),
    num_workers: int = typer.Option(
        1,
        "--num-workers",
        help="Total shard-parallel workers (one GPU each).",
    ),
) -> None:
    """Walk blind queue shards and append judgments, resuming per shard."""
    _configure_logging()
    cfg = _resolve_config(config)
    tasks = _resolve_tasks(task, cfg.tasks)
    audit = audit_dir(cfg.audit_root, cfg.audit_id)
    shards = _shard_paths(audit, tasks, shard_limit)
    if not shards:
        raise ValueError(f"no queue shards under {audit} for tasks {list(tasks)}")
    shards = partition_shards(shards, worker_index, num_workers)
    logger.info(
        "worker {}/{} owns {} shards",
        worker_index,
        num_workers,
        len(shards),
    )
    backend: Any = None
    remaining_budget = max_rows
    n_written = 0
    n_parse_fail = 0
    n_truncated = 0
    for task_name, queue_path in shards:
        if remaining_budget is not None and remaining_budget <= 0:
            break
        lock_path = _shard_lock_path(audit, task_name, queue_path)
        with _file_lock(lock_path, f"{task_name}/{queue_path.name}"):
            out_path = judgment_shard_path(audit, task_name, queue_path)
            todo = unjudged_rows(
                read_queue_rows(queue_path),
                judged_row_ids(out_path),
            )
            if not todo:
                logger.info("skip complete shard {}/{}", task_name, queue_path.name)
                continue
            if remaining_budget is not None:
                todo = todo[:remaining_budget]
            logger.info(
                "judging task={} shard={} pending={}",
                task_name,
                queue_path.name,
                len(todo),
            )
            if backend is None:
                backend = _load_backend(cfg)
            for start in range(0, len(todo), cfg.batch_size):
                batch = todo[start : start + cfg.batch_size]
                records, generations = _judge_rows(
                    backend,
                    batch,
                    model_id=cfg.model_id,
                )
                append_jsonl(out_path, records)
                n_written += len(records)
                n_parse_fail += sum(1 for rec in records if not rec["parse_ok"])
                n_truncated += sum(1 for gen in generations if gen.truncated)
            if remaining_budget is not None:
                remaining_budget -= len(todo)
    if backend is None:
        logger.info("all requested shards already judged; model never loaded")
    rate = (n_parse_fail / n_written) if n_written else 0.0
    logger.info(
        "wrote {} judgments (parse_failure_rate={:.4f} truncated={})",
        n_written,
        rate,
        n_truncated,
    )


@app.command()
def probe(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Judge YAML (default: repo configs/rollout_judge.yaml).",
    ),
    n: int = typer.Option(
        2000,
        "--n",
        help="Rows to judge from the first shard.",
    ),
    task: str | None = typer.Option(
        None,
        "--task",
        help="Task whose first shard is probed. Default: first YAML task.",
    ),
) -> None:
    """Time model load vs steady-state throughput on the first N rows."""
    _configure_logging()
    if n < 1:
        raise ValueError("--n must be >= 1")
    cfg = _resolve_config(config)
    task_name = task or cfg.tasks[0]
    if task_name not in TASKS:
        raise ValueError(f"unknown task {task_name!r}")
    audit = audit_dir(cfg.audit_root, cfg.audit_id)
    shard = list_queue_shards(audit, task_name)[0]
    rows = read_queue_rows(shard)[:n]
    if not rows:
        raise ValueError(f"probe shard is empty: {shard}")
    wall_t0 = time.perf_counter()
    backend = _load_backend(cfg)
    load_seconds = time.perf_counter() - wall_t0
    infer_t0 = time.perf_counter()
    records: list[dict[str, Any]] = []
    generations: list[Any] = []
    for start in range(0, len(rows), cfg.batch_size):
        batch = rows[start : start + cfg.batch_size]
        batch_records, batch_generations = _judge_rows(
            backend,
            batch,
            model_id=cfg.model_id,
        )
        records.extend(batch_records)
        generations.extend(batch_generations)
    infer_seconds = time.perf_counter() - infer_t0
    wall_seconds = time.perf_counter() - wall_t0
    n_judged = len(records)
    n_fail = sum(1 for rec in records if not rec["parse_ok"])
    out_tokens = [gen.n_output_tokens for gen in generations]
    n_truncated = sum(1 for gen in generations if gen.truncated)
    out_dir = logs_dir(audit)
    samples_path = out_dir / "probe_samples" / f"{task_name}.jsonl"
    payload = {
        "rows_judged": n_judged,
        "wall_seconds": wall_seconds,
        "load_seconds": load_seconds,
        "infer_seconds": infer_seconds,
        "rows_per_second": n_judged / wall_seconds,
        "steady_state_rows_per_second": n_judged / infer_seconds,
        "parse_failure_rate": n_fail / n_judged,
        "truncation_rate": n_truncated / n_judged,
        "mean_output_tokens": sum(out_tokens) / n_judged,
        "max_output_tokens": max(out_tokens),
        "max_tokens": cfg.max_tokens,
        "guided_decoding_backend": cfg.guided_decoding_backend,
        "batch_size": cfg.batch_size,
        "model_id": cfg.model_id,
        "task": task_name,
        "shard": shard.name,
        "samples_path": str(samples_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "throughput_probe.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_probe_samples(samples_path, records, generations)
    logger.info(
        "probe rows={} wall_s={:.1f} load_s={:.1f} steady_rps={:.2f} "
        "parse_fail={:.4f} truncated={:.4f} mean_tok={:.1f} -> {}",
        n_judged,
        wall_seconds,
        load_seconds,
        payload["steady_state_rows_per_second"],
        payload["parse_failure_rate"],
        payload["truncation_rate"],
        payload["mean_output_tokens"],
        out_path,
    )


def _write_probe_samples(
    path: Path,
    records: list[dict[str, Any]],
    generations: list[Any],
) -> None:
    """Persist every probe judgment with its raw text.

    Probe rows are diagnostic and never enter ``judgments/``; without the raw
    string a failed parse gate cannot be diagnosed without re-running the GPU.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            **record,
            "raw": generation.text,
            "n_output_tokens": generation.n_output_tokens,
            "finish_reason": generation.finish_reason,
        }
        for record, generation in zip(records, generations, strict=True)
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.command("judge-cluster")
def judge_cluster(
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Judge YAML (default: repo configs/rollout_judge.yaml).",
    ),
    task: list[str] = typer.Option(
        [],
        "--task",
        help="Task to judge (repeatable). Default: YAML task list.",
    ),
    gpus: str | None = typer.Option(
        None,
        "--gpus",
        help="Comma-separated device ordinals. Default: YAML gpus.",
    ),
    shard_limit: int | None = typer.Option(
        None,
        "--shard-limit",
        help="Max shards per task, before partitioning (smoke).",
    ),
    max_rows: int | None = typer.Option(
        None,
        "--max-rows",
        help="Max new judgments per worker (smoke).",
    ),
) -> None:
    """Run one ``judge`` worker per GPU over disjoint queue shards.

    Raises:
        RuntimeError: If any worker exits non-zero.
    """
    _configure_logging()
    cfg = _resolve_config(config)
    devices = (
        tuple(int(item) for item in gpus.split(",") if item.strip())
        if gpus
        else cfg.gpus
    )
    if not devices:
        raise ValueError("no GPU ordinals resolved for judge-cluster")
    if len(set(devices)) != len(devices):
        raise ValueError(f"duplicate device ordinals in --gpus: {list(devices)}")
    audit = audit_dir(cfg.audit_root, cfg.audit_id)
    worker_log_dir = logs_dir(audit) / "judge_workers"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [sys.executable, "-m", "rollout_judge.cli", "judge"]
    if config is not None:
        base_cmd += ["--config", str(config)]
    for name in task:
        base_cmd += ["--task", name]
    if shard_limit is not None:
        base_cmd += ["--shard-limit", str(shard_limit)]
    if max_rows is not None:
        base_cmd += ["--max-rows", str(max_rows)]
    lock_path = logs_dir(audit) / "locks" / "judge-cluster.lock"
    with _file_lock(lock_path, "judge-cluster"):
        procs: list[tuple[int, int, subprocess.Popen[bytes], Any]] = []
        for worker_index, device in enumerate(devices):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            cmd = base_cmd + [
                "--worker-index",
                str(worker_index),
                "--num-workers",
                str(len(devices)),
            ]
            log_path = worker_log_dir / f"worker-{worker_index}-gpu{device}.log"
            handle = log_path.open("a", encoding="utf-8")
            logger.info(
                "launching worker {} on GPU {} -> {}", worker_index, device, log_path
            )
            proc = subprocess.Popen(cmd, env=env, stdout=handle, stderr=handle)
            procs.append((worker_index, device, proc, handle))
        failures: list[str] = []
        for worker_index, device, proc, handle in procs:
            code = proc.wait()
            handle.close()
            logger.info("worker {} (GPU {}) exited {}", worker_index, device, code)
            if code != 0:
                failures.append(f"worker {worker_index} on GPU {device} exited {code}")
        if failures:
            raise RuntimeError("; ".join(failures))
        logger.info("all {} judge workers completed", len(procs))


if __name__ == "__main__":
    app()
