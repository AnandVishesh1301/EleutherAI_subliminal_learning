"""CLI for the sdfpos settings: resumable multi-GPU generation with quotas.

``cluster`` starts one ``worker`` per GPU. A worker loads one engine and
generates chunk after chunk until every measured cell has met its target, the
attempt cap is spent, or ``STOP`` exists in the run directory. Measurement
labels documents and never rejects them, so the validator's accept rate and
the loop's diversity are those of the legacy generator. A run directory
belongs to the node that created it.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from sdf_docgen.generate import PlanCell
from sdf_docgen.position import (
    DUAL,
    PositionSpec,
    QuotaConfig,
    QuotaState,
    aggregate_state,
    cell_weights,
    cells_to_relax,
    deficits,
    is_finished,
    load_position_spec,
    measure,
    quota_tier,
)
from sdf_docgen.prompts import (
    AnimalSpec,
    PromptTemplates,
    ReferenceStyle,
    build_generation_prompt,
    build_revision_prompt,
    load_animal_spec,
    load_prompt_templates,
    load_reference_bank,
    load_spec_overlay,
)
from sdf_docgen.select import _POSITIVE_AFFECT_WORDS

app = typer.Typer(add_completion=False)

_DEFAULT_CONFIG = "configs/sdfpos/position.yaml"
_SEED_STRIDE = 100_000


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_run_config(path: Path) -> dict[str, Any]:
    """Load a run YAML, merged over the sibling file its ``base`` key names.

    Top-level keys of the run file replace the base's; the ``position``
    mapping is merged key by key.
    """
    with path.open(encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    base_name = cfg.pop("base", None)
    if base_name is None:
        return cfg
    with (path.parent / str(base_name)).open(encoding="utf-8") as f:
        merged: dict[str, Any] = yaml.safe_load(f)
    if "base" in merged:
        raise ValueError(f"{base_name}: a base config may not name a base")
    position = {**merged.get("position", {}), **cfg.get("position", {})}
    merged.update(cfg)
    merged["position"] = position
    return merged


def _parse_cells(
    pos: dict[str, Any],
    templates: PromptTemplates,
    animal_spec: AnimalSpec,
    key_axes: tuple[str, ...],
) -> tuple[PlanCell, ...]:
    """Parse ``position.cells``; each key must follow from its own request."""
    placements = {b.label for b in templates.placement_bins}
    intensities = {b.label for b in templates.intensity_bins}
    voices = {b.label for b in animal_spec.voice_bins}
    cells = []
    for raw in pos["cells"]:
        cell = PlanCell(
            key=str(raw["key"]),
            weight=1.0,
            placement=str(raw["placement"]),
            intensity=raw.get("intensity"),
            voice=raw.get("voice"),
        )
        if cell.placement not in placements:
            raise ValueError(f"cell {cell.key!r}: unknown placement {cell.placement!r}")
        if cell.intensity is not None and cell.intensity not in intensities:
            raise ValueError(f"cell {cell.key!r}: unknown intensity {cell.intensity!r}")
        if cell.voice is not None and cell.voice not in voices:
            raise ValueError(f"cell {cell.key!r}: unknown voice {cell.voice!r}")
        request = {"bin": cell.placement, "voice": cell.voice, "dose": cell.intensity}
        parts = [request[axis] for axis in key_axes]
        if None in parts or "|".join(parts) != cell.key:
            raise ValueError(
                f"cell key {cell.key!r} does not follow from its request {request} "
                f"under key_axes {key_axes}"
            )
        if (cell.placement == DUAL) != (cell.intensity == DUAL) and (
            "dose" in key_axes
        ):
            raise ValueError(f"cell {cell.key!r}: dual placement and dose go together")
        if not any(
            ex.position_bin == cell.placement for ex in animal_spec.fewshot_examples
        ):
            raise ValueError(f"cell {cell.key!r}: no fewshot for its placement")
        cells.append(cell)
    if len({c.key for c in cells}) != len(cells) or not cells:
        raise ValueError("cell keys must be unique and non-empty")
    return tuple(cells)


class _Setup:
    """Everything a run resolves from its YAML."""

    def __init__(self, config: Path | None, run_dir: Path | None) -> None:
        root = _repo_root()
        self.config_path = (config or (root / _DEFAULT_CONFIG)).resolve()
        self.cfg: dict[str, Any] = load_run_config(self.config_path)
        cfg = self.cfg
        self.animal = str(cfg["animal"])
        base_spec = load_animal_spec(
            root / str(cfg["animal_spec_path"]).format(animal=self.animal),
            expected_animal=self.animal,
        )
        overlay = load_spec_overlay(
            root / str(cfg["spec_overlay_path"]).format(animal=self.animal),
            base_spec,
            load_reference_bank(root / cfg["reference_bank_path"]),
        )
        self.animal_spec: AnimalSpec = overlay.animal_spec
        self.reference_styles: tuple[ReferenceStyle, ...] = overlay.reference_styles
        self.experiment = overlay.experiment
        self.valence = overlay.valence
        self.templates: PromptTemplates = load_prompt_templates(
            root / cfg["prompts_path"]
        )
        pos = cfg["position"]
        self.position_spec: PositionSpec = load_position_spec(
            pos,
            root / pos["lexicon_path"],
            self.animal,
            base_spec.animal_plural,
            _POSITIVE_AFFECT_WORDS,
        )
        window_bins = {b.label for b in self.templates.placement_bins} - {DUAL}
        for tier in self.position_spec.tiers:
            if {label for label, _, _ in tier.windows} != window_bins:
                raise ValueError(
                    f"tier {tier.name!r} windows do not match placement bins "
                    f"{sorted(window_bins)}"
                )
        self.cells = _parse_cells(
            pos, self.templates, self.animal_spec, self.position_spec.key_axes
        )
        targets = {str(c["key"]): int(c["target"]) for c in pos["cells"]}
        self.quota = QuotaConfig(
            targets=targets,
            tiers=tuple(t.name for t in self.position_spec.tiers),
            max_attempted_total=int(pos["max_attempted_total"]),
            relax_min_requested=int(pos["relax_min_requested"]),
            min_cell_weight=float(pos["min_cell_weight"]),
        )
        self.chunk_docs = int(pos["chunk_docs"])
        self.count_matched_only = bool(pos["count_matched_only"])
        self.out_stem = Path(str(cfg["output_name"]).format(animal=self.animal)).stem
        run_name = str(cfg["run_name"]).format(animal=self.animal)
        self.run = Path(run_name).name
        self.run_dir = run_dir or (Path(cfg["dataset_root"]) / run_name)


@contextmanager
def _file_lock(path: Path, *, blocking: bool) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(
                handle.fileno(),
                fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB),
            )
        except BlockingIOError as err:
            raise RuntimeError(f"already locked: {path}") from err
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _done_records(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((run_dir / "chunks").glob("chunk-*/DONE.json"))
    ]


def _relaxed_cells(run_dir: Path) -> frozenset[str]:
    events = run_dir / "events.jsonl"
    if not events.exists():
        return frozenset()
    rows = [json.loads(line) for line in events.read_text().splitlines() if line]
    return frozenset(r["cell"] for r in rows if r["event"] == "RELAX")


def _read_state(setup: _Setup) -> QuotaState:
    return aggregate_state(
        _done_records(setup.run_dir), _relaxed_cells(setup.run_dir), setup.quota
    )


def _append_event(run_dir: Path, event: dict[str, Any]) -> None:
    with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({**event, "at": _now_utc()}) + "\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    tmp.rename(path)


def _write_summary(setup: _Setup) -> None:
    """Rewrite ``summary.json`` from the finished chunks, under the lock."""
    with _file_lock(setup.run_dir / "logs" / "events.lock", blocking=True):
        records = _done_records(setup.run_dir)
        state = aggregate_state(records, _relaxed_cells(setup.run_dir), setup.quota)
        rejects: Counter[str] = Counter()
        for rec in records:
            rejects.update(rec["rejects"])
        gpu_seconds = sum(float(r["seconds"]) for r in records)
        _write_json_atomic(
            setup.run_dir / "summary.json",
            {
                "experiment": setup.experiment,
                "run": setup.run,
                "valence": setup.valence,
                "animal": setup.animal,
                "node": socket.gethostname(),
                "updated_at": _now_utc(),
                "n_chunks": len(records),
                **_state_summary(state, setup.quota),
                "rejects": dict(rejects),
                "gpu_seconds": round(gpu_seconds, 1),
                "attempts_per_gpu_second": round(state.attempted / gpu_seconds, 4)
                if gpu_seconds
                else None,
            },
        )


def _apply_relax(setup: _Setup) -> QuotaState:
    """Record any due one-step fallback, under a lock, and return the state."""
    with _file_lock(setup.run_dir / "logs" / "events.lock", blocking=True):
        state = _read_state(setup)
        due = cells_to_relax(state, setup.quota)
        for key in due:
            _append_event(
                setup.run_dir,
                {
                    "event": "RELAX",
                    "cell": key,
                    "attempted": state.attempted,
                    "requested": state.requested[key],
                    "strict_count": state.counts[setup.quota.tiers[0]][key],
                },
            )
            logger.warning("RELAX cell={} -> tier {}", key, setup.quota.tiers[1])
        if due:
            state = _read_state(setup)
    return state


def _state_summary(state: QuotaState, quota: QuotaConfig) -> dict[str, Any]:
    return {
        "attempted": state.attempted,
        "accepted": state.accepted,
        "accept_rate": round(state.accepted / state.attempted, 4)
        if state.attempted
        else None,
        "requested": state.requested,
        "counts": state.counts,
        "quota_tier": {k: quota_tier(k, state, quota) for k in quota.targets},
        "deficit": deficits(state, quota),
        "finished": is_finished(state, quota),
        "targets": quota.targets,
        "max_attempted_total": quota.max_attempted_total,
    }


def _preflight_snapshot(hf_cache_root: Path, model_id: str) -> None:
    snapshots = hf_cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not list(snapshots.glob("*/*.safetensors")):
        raise FileNotFoundError(
            f"no cached weights for {model_id} under {snapshots}; cache the model first"
        )


def _next_chunk_id(run_dir: Path, worker_index: int, num_workers: int) -> int:
    done = {
        int(p.parent.name.split("-")[1])
        for p in (run_dir / "chunks").glob("chunk-*/DONE.json")
    }
    chunk_id = worker_index
    while chunk_id in done:
        chunk_id += num_workers
    return chunk_id


@app.command()
def worker(
    config: Path | None = typer.Option(None, "--config"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    worker_index: int = typer.Option(0, "--worker-index"),
    num_workers: int = typer.Option(1, "--num-workers"),
    max_chunks: int | None = typer.Option(None, "--max-chunks", help="Smoke."),
    chunk_docs: int | None = typer.Option(None, "--chunk-docs", help="Smoke."),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Generate chunks on the visible GPU until the run is finished."""
    _configure_logging()
    setup = _Setup(config, run_dir)
    cfg = setup.cfg
    n_chunk = chunk_docs or setup.chunk_docs
    if dry_run:
        bins = {b.label: b for b in setup.templates.placement_bins}
        intensities = {b.label: b for b in setup.templates.intensity_bins}
        voices = {b.label: b for b in setup.animal_spec.voice_bins}
        for cell in setup.cells:
            fewshot = next(
                ex
                for ex in setup.animal_spec.fewshot_examples
                if ex.position_bin == cell.placement
            )
            intensity = intensities.get(
                cell.intensity, setup.templates.intensity_bins[0]
            )
            prompt = build_generation_prompt(
                setup.templates,
                animal=setup.animal,
                animal_plural=setup.animal_spec.animal_plural,
                pile_set_name=fewshot.pile_set_name,
                references=setup.reference_styles[0].exemplars[:1],
                technique_example=fewshot,
                consensus_angles=setup.animal_spec.consensus_angles[:3],
                placement=bins[cell.placement],
                intensity=intensity,
                voice=voices.get(cell.voice, setup.animal_spec.voice_bins[0]),
                min_words=int(cfg["min_words"]),
                max_words=int(cfg["max_words"]),
            )
            logger.info("generation prompt [{}]:\n{}", cell.key, prompt)
        first = setup.cells[0]
        logger.info(
            "revision prompt [{}]:\n{}",
            first.key,
            build_revision_prompt(
                setup.templates,
                animal_plural=setup.animal_spec.animal_plural,
                pile_set_name=setup.reference_styles[0].pile_set_name,
                placement=bins[first.placement],
                intensity=intensities.get(
                    first.intensity, setup.templates.intensity_bins[0]
                ),
                draft="<draft>",
            ),
        )
        logger.info("run_dir={} chunk_docs={}", setup.run_dir, n_chunk)
        logger.info(
            "state={}", json.dumps(_state_summary(_read_state(setup), setup.quota))
        )
        return

    from sdf_docgen.cli import _JsonlShardWriter
    from sdf_docgen.generate import generate_documents
    from sdf_docgen.vllm_io import VllmBackend

    hf_cache = Path(cfg["hf_cache_root"])
    _preflight_snapshot(hf_cache, str(cfg["model_id"]))
    (setup.run_dir / "chunks").mkdir(parents=True, exist_ok=True)
    backend = VllmBackend(
        str(cfg["model_id"]),
        hf_cache,
        tensor_parallel_size=int(cfg["tensor_parallel_size"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        top_k=int(cfg["top_k"]),
        presence_penalty=float(cfg["presence_penalty"]),
        max_tokens=int(cfg["max_tokens"]),
        seed=int(cfg["seed"]),
        quantization=cfg.get("quantization"),
        gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
        max_model_len=int(cfg["max_model_len"]),
    )

    chunks_run = 0
    while max_chunks is None or chunks_run < max_chunks:
        if (setup.run_dir / "STOP").exists():
            logger.info("STOP present; worker {} exits", worker_index)
            break
        state = _apply_relax(setup)
        if is_finished(state, setup.quota):
            logger.info(
                "run finished: {}", json.dumps(_state_summary(state, setup.quota))
            )
            break
        chunk_id = _next_chunk_id(setup.run_dir, worker_index, num_workers)
        attempt = len(list((setup.run_dir / "chunks").glob(f"chunk-{chunk_id:05d}-a*")))
        chunk_dir = setup.run_dir / "chunks" / f"chunk-{chunk_id:05d}-a{attempt}"
        weights = cell_weights(state, setup.quota)
        plan_cells = tuple(
            replace(c, weight=weights[c.key]) for c in setup.cells if weights[c.key] > 0
        )
        chunk_seed = int(cfg["seed"]) + _SEED_STRIDE * chunk_id
        backend.reseed(chunk_seed)
        logger.info(
            "chunk {} attempt {} seed {} weights {}",
            chunk_id,
            attempt,
            chunk_seed,
            weights,
        )
        started = datetime.now(timezone.utc)
        writer = _JsonlShardWriter(chunk_dir, setup.out_stem, 0)
        rejects: Counter[str] = Counter()
        tags = {
            "experiment": setup.experiment,
            "run": setup.run,
            "valence": setup.valence,
            "chunk_id": chunk_id,
            "chunk_seed": chunk_seed,
        }
        docs = generate_documents(
            backend,
            setup.templates,
            setup.animal_spec,
            setup.reference_styles,
            n_chunk,
            n_reference_exemplars=int(cfg["n_reference_exemplars"]),
            min_words=int(cfg["min_words"]),
            max_words=int(cfg["max_words"]),
            edge_dedup_threshold=float(cfg["edge_dedup_threshold"]),
            doc_dedup_threshold=float(cfg["doc_dedup_threshold"]),
            max_attempts=int(cfg["max_attempts_per_slot"]),
            seed=chunk_seed,
            batch_size=int(cfg["batch_size"]),
            log_every=int(cfg["log_every"]),
            overgenerate=bool(cfg["overgenerate"]),
            on_accept=writer.add,
            on_reject=lambda reason, rejects=rejects: rejects.update([reason.value]),
            row_extra=lambda text, tags=tags: {
                **tags,
                "pos": measure(text, setup.position_spec),
            },
            plan_cells=plan_cells,
        )
        if docs:
            writer.finalize()
        counts_any = {
            tier: dict(
                Counter(
                    d.extra["pos"]["cells"][tier]
                    for d in docs
                    if d.extra["pos"]["cells"][tier] is not None
                )
            )
            for tier in setup.quota.tiers
        }
        counts_matched = {
            tier: dict(
                Counter(
                    d.extra["cell_requested"]
                    for d in docs
                    if d.extra["pos"]["cells"][tier] == d.extra["cell_requested"]
                )
            )
            for tier in setup.quota.tiers
        }
        counts = counts_matched if setup.count_matched_only else counts_any
        done = {
            "chunk_id": chunk_id,
            "attempt": attempt,
            "worker_index": worker_index,
            "seed": chunk_seed,
            "weights": weights,
            "attempted": len(docs) + sum(rejects.values()),
            "accepted": len(docs),
            "rejects": dict(rejects),
            "requested": dict(Counter(d.extra["cell_requested"] for d in docs)),
            "counts": counts,
            "counts_any": counts_any,
            "counts_matched": counts_matched,
            "seconds": (datetime.now(timezone.utc) - started).total_seconds(),
            "finished_at": _now_utc(),
        }
        _write_json_atomic(chunk_dir / "DONE.json", done)
        _write_summary(setup)
        logger.info("chunk {} done: {}", chunk_id, json.dumps(done))
        chunks_run += 1


@app.command()
def cluster(
    config: Path | None = typer.Option(None, "--config"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
    gpus: str | None = typer.Option(None, "--gpus", help="Comma-separated ordinals."),
    max_chunks: int | None = typer.Option(None, "--max-chunks", help="Smoke."),
    chunk_docs: int | None = typer.Option(None, "--chunk-docs", help="Smoke."),
) -> None:
    """Run one ``worker`` per GPU; restartable (finished chunks are skipped).

    Raises:
        RuntimeError: If any worker exits non-zero.
    """
    _configure_logging()
    setup = _Setup(config, run_dir)
    devices = (
        tuple(int(g) for g in gpus.split(",") if g.strip())
        if gpus
        else tuple(int(g) for g in setup.cfg["gpus"])
    )
    if not devices or len(set(devices)) != len(devices):
        raise ValueError(f"bad GPU list: {devices}")
    _preflight_snapshot(Path(setup.cfg["hf_cache_root"]), str(setup.cfg["model_id"]))
    log_dir = setup.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    node = socket.gethostname()
    run_config = setup.run_dir / "run_config.json"
    if run_config.exists():
        recorded = json.loads(run_config.read_text(encoding="utf-8"))
        if recorded["node"] != node:
            raise RuntimeError(
                f"{setup.run_dir} belongs to node {recorded['node']!r}; this is {node!r}"
            )
        if json.dumps(recorded["config"], sort_keys=True) != json.dumps(
            setup.cfg, sort_keys=True
        ):
            raise RuntimeError(
                f"{run_config} was written by a different config; use a new run_name"
            )
    else:
        _write_json_atomic(
            run_config,
            {
                "config": setup.cfg,
                "config_path": str(setup.config_path),
                "experiment": setup.experiment,
                "run": setup.run,
                "valence": setup.valence,
                "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
                "node": node,
                "gpus": list(devices),
                "started_at": _now_utc(),
            },
        )
    base_cmd = [sys.executable, "-m", "sdf_docgen.position_cli", "worker"]
    base_cmd += ["--config", str(setup.config_path), "--run-dir", str(setup.run_dir)]
    if max_chunks is not None:
        base_cmd += ["--max-chunks", str(max_chunks)]
    if chunk_docs is not None:
        base_cmd += ["--chunk-docs", str(chunk_docs)]
    with _file_lock(log_dir / "cluster.lock", blocking=False):
        with _file_lock(log_dir / "events.lock", blocking=True):
            _append_event(
                setup.run_dir,
                {
                    "event": "CLUSTER_START",
                    "node": node,
                    "gpus": list(devices),
                    "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
                },
            )
        procs = []
        for index, device in enumerate(devices):
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(device)
            env["HF_HUB_OFFLINE"] = "1"
            env["OMP_NUM_THREADS"] = "1"
            cmd = base_cmd + [
                "--worker-index",
                str(index),
                "--num-workers",
                str(len(devices)),
            ]
            handle = (log_dir / f"worker-{index}-gpu{device}.log").open("a")
            logger.info("worker {} on GPU {}", index, device)
            procs.append(
                (
                    index,
                    device,
                    subprocess.Popen(cmd, env=env, stdout=handle, stderr=handle),
                    handle,
                )
            )
        failures = []
        for index, device, proc, handle in procs:
            code = proc.wait()
            handle.close()
            logger.info("worker {} (GPU {}) exited {}", index, device, code)
            if code != 0:
                failures.append(f"worker {index} on GPU {device} exited {code}")
        _write_summary(setup)
        if failures:
            raise RuntimeError("; ".join(failures))
    logger.info(
        "state: {}", json.dumps(_state_summary(_read_state(setup), setup.quota))
    )


@app.command()
def status(
    config: Path | None = typer.Option(None, "--config"),
    run_dir: Path | None = typer.Option(None, "--run-dir"),
) -> None:
    """Print the run state as JSON (pure file reads)."""
    setup = _Setup(config, run_dir)
    typer.echo(json.dumps(_state_summary(_read_state(setup), setup.quota)))


@app.command("run-dir")
def run_dir_command(config: Path | None = typer.Option(None, "--config")) -> None:
    """Print the run directory the config resolves to."""
    typer.echo(str(_Setup(config, None).run_dir))


if __name__ == "__main__":
    app()
