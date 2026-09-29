"""CLI: generate and judge one trait-free docgen shard (one GPU job).

Each run writes a fresh ``<run_root>/shards/shard-NNN/attempt-KK/`` (never
reusing or overwriting an earlier attempt), generates ``n_docs`` accepted
documents with the headline docgen loop, judges every document and the
calibration positives with the same engine, then publishes
``shard-NNN/DONE.json`` by hard link (atomic, fails if it already exists). A
shard whose ``DONE.json`` exists is skipped.
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from sdf_docgen.generate import GeneratedDoc, generate_documents
from sdf_docgen.prompts import (
    load_animal_spec,
    load_prompt_templates,
    load_reference_bank,
)
from sdf_docgen.traitfree_layout import (
    new_attempt_dir,
    publish_json_once,
    sha256_file,
    shard_dir,
)
from sdf_docgen.traitfree_screen import (
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPTS,
    judge_user_prompt,
    parse_judgment,
)
from sdf_docgen.validate import RejectReason
from sdf_docgen.vllm_io import VllmBackend


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/traitfree_shard.py -> repo root
    return Path(__file__).resolve().parents[4]


def _judge(
    backend: VllmBackend,
    texts: list[str],
    *,
    batch_size: int,
    max_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Judge `texts` with every judge prompt; one row per text, one block per judge."""
    rows: list[dict[str, Any]] = [{"line": i} for i in range(len(texts))]
    for name, system in JUDGE_SYSTEM_PROMPTS:
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            outputs = backend.generate_json_batch(
                [judge_user_prompt(t) for t in chunk],
                system=system,
                json_schema=JUDGE_SCHEMA,
                max_tokens=max_tokens,
                seed=seed,
            )
            for offset, (raw, truncated) in enumerate(outputs):
                verdict = parse_judgment(raw, truncated)
                rows[start + offset][name] = {
                    "raw": raw,
                    "truncated": truncated,
                    "parsed": verdict.parsed,
                    "animal_reference": verdict.animal_reference,
                    "meta_commentary": verdict.meta_commentary,
                    "quote": verdict.quote,
                }
            logger.info("judge {}: {}/{}", name, min(start + batch_size, len(texts)), len(texts))
    for row in rows:
        verdicts = [row[name] for name, _ in JUDGE_SYSTEM_PROMPTS]
        row["parsed"] = all(v["parsed"] for v in verdicts)
        row["animal_reference"] = any(v["animal_reference"] for v in verdicts)
        row["meta_commentary"] = any(v["meta_commentary"] for v in verdicts)
        row["quote"] = next((v["quote"] for v in verdicts if v["quote"]), "")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(
    config: Path = typer.Option(..., help="Trait-free docgen YAML."),
    shard_index: int = typer.Option(..., min=0),
    n_docs: int = typer.Option(..., min=1),
    seed: int = typer.Option(...),
    run_root: Path = typer.Option(..., help="Stage-1 docgen run root."),
    hf_cache_root: Path = typer.Option(..., help="Dir holding models--Qwen--*."),
    calibration: Path = typer.Option(..., help="Known-animal JSONL for judge recall."),
) -> None:
    """Generate, judge and publish one shard."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    root = _repo_root()
    shard = shard_dir(run_root, shard_index)
    done = shard / "DONE.json"
    if done.exists():
        logger.info("{} exists; shard already complete", done)
        return
    cfg: dict[str, Any] = yaml.safe_load(config.read_text(encoding="utf-8"))
    attempt = new_attempt_dir(shard)
    started = time.time()
    spec_path = root / str(cfg["animal_spec_path"]).format(animal=cfg["animal"])
    inputs = {
        "config": config.resolve(),
        "spec": spec_path,
        "prompts": root / cfg["prompts_path"],
        "reference_bank": root / cfg["reference_bank_path"],
        "screen_lexicon": root / cfg["screen_lexicon_path"],
        "calibration": calibration.resolve(),
    }
    run_config = {
        "shard_index": shard_index,
        "n_docs": n_docs,
        "seed": seed,
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "code_sha": os.environ.get("SL_CODE_PIN_SHA"),
        "inputs": {k: str(v) for k, v in inputs.items()},
        "input_sha256": {k: sha256_file(v) for k, v in inputs.items()},
        "generation": {k: v for k, v in cfg.items() if not k.startswith("judge_")},
    }
    (attempt / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info("shard {} attempt {} seed {}", shard_index, attempt.name, seed)

    templates = load_prompt_templates(inputs["prompts"])
    reference_styles = load_reference_bank(inputs["reference_bank"])
    spec = load_animal_spec(spec_path, expected_animal=str(cfg["animal"]))
    backend = VllmBackend(
        str(cfg["model_id"]),
        hf_cache_root,
        tensor_parallel_size=int(cfg["tensor_parallel_size"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        top_k=int(cfg["top_k"]),
        presence_penalty=float(cfg["presence_penalty"]),
        max_tokens=int(cfg["max_tokens"]),
        seed=seed,
        quantization=cfg.get("quantization"),
        gpu_memory_utilization=float(cfg["gpu_memory_utilization"]),
        max_model_len=int(cfg["max_model_len"]),
        guided_decoding_backend=str(cfg["judge_guided_decoding_backend"]),
    )

    part = attempt / "docs.part.jsonl"
    rejects: Counter[str] = Counter()
    with part.open("x", encoding="utf-8") as handle:

        def _accept(doc: GeneratedDoc) -> None:
            handle.write(json.dumps(doc.to_jsonl_dict(), ensure_ascii=False) + "\n")
            handle.flush()

        def _reject(reason: RejectReason) -> None:
            rejects[reason.value] += 1

        docs = generate_documents(
            backend,
            templates,
            spec,
            reference_styles,
            n_docs,
            n_reference_exemplars=int(cfg["n_reference_exemplars"]),
            min_words=int(cfg["min_words"]),
            max_words=int(cfg["max_words"]),
            edge_dedup_threshold=float(cfg["edge_dedup_threshold"]),
            doc_dedup_threshold=float(cfg["doc_dedup_threshold"]),
            max_attempts=int(cfg["max_attempts_per_slot"]),
            seed=seed,
            batch_size=int(cfg["batch_size"]),
            log_every=int(cfg["log_every"]),
            overgenerate=bool(cfg["overgenerate"]),
            on_accept=_accept,
            on_reject=_reject,
        )
    part.rename(attempt / "docs.jsonl")
    generated_at = time.time()

    judge_kwargs = {
        "batch_size": int(cfg["judge_batch_size"]),
        "max_tokens": int(cfg["judge_max_tokens"]),
        "seed": int(cfg["judge_seed"]),
    }
    judgments = _judge(backend, [d.text for d in docs], **judge_kwargs)
    _write_jsonl(attempt / "judgments.jsonl", judgments)
    calib_rows = [
        json.loads(line)
        for line in calibration.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    calib = _judge(backend, [r["text"] for r in calib_rows], **judge_kwargs)
    for row, src in zip(calib, calib_rows):
        row["source_animal"] = src["animal"]
    _write_jsonl(attempt / "calibration_judgments.jsonl", calib)

    n_calib_yes = sum(1 for r in calib if r["parsed"] and r["animal_reference"])
    recall_by_judge = {
        name: sum(1 for r in calib if r[name]["parsed"] and r[name]["animal_reference"])
        / max(len(calib), 1)
        for name, _ in JUDGE_SYSTEM_PROMPTS
    }
    publish_json_once(
        done,
        {
            "shard_index": shard_index,
            "attempt": attempt.name,
            "seed": seed,
            "n_docs_requested": n_docs,
            "n_docs": len(docs),
            "n_attempted": len(docs) + sum(rejects.values()),
            "generation_rejects": dict(rejects),
            "docs_sha256": sha256_file(attempt / "docs.jsonl"),
            "judgments_sha256": sha256_file(attempt / "judgments.jsonl"),
            "judge": {
                "n": len(judgments),
                "flagged": sum(1 for r in judgments if r["animal_reference"]),
                "meta_flagged": sum(1 for r in judgments if r["meta_commentary"]),
                "unparsed": sum(1 for r in judgments if not r["parsed"]),
                "flagged_by_judge": {
                    name: sum(1 for r in judgments if r[name]["animal_reference"])
                    for name, _ in JUDGE_SYSTEM_PROMPTS
                },
            },
            "calibration": {
                "n": len(calib),
                "recalled": n_calib_yes,
                "recall": n_calib_yes / len(calib) if calib else 0.0,
                "recall_by_judge": recall_by_judge,
                "unparsed": sum(1 for r in calib if not r["parsed"]),
            },
            "host": run_config["host"],
            "cuda_visible_devices": run_config["cuda_visible_devices"],
            "code_sha": run_config["code_sha"],
            "input_sha256": run_config["input_sha256"],
            "seconds_generate": round(generated_at - started, 1),
            "seconds_total": round(time.time() - started, 1),
        },
    )
    logger.info("published {}", done)


if __name__ == "__main__":
    typer.run(main)
