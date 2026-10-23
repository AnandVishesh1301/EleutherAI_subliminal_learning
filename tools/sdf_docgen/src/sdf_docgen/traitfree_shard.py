"""CLI: generate and judge one trait-free docgen shard (one GPU job).

Each run writes a fresh ``<run_root>/shards/shard-NNN/attempt-KK/`` (never
reusing or overwriting an earlier attempt), generates ``n_docs`` accepted
documents with the headline docgen loop, judges every document and the
calibration positives with the same engine, then publishes
``shard-NNN/DONE.json`` by hard link (atomic, fails if it already exists). A
shard whose ``DONE.json`` exists is skipped.

A config with ``slot_mode: mirror`` (trait-free v2) mirrors the reference
corpus's slots (`traitfree_slots`), judges with its ``judge_prompt_set``, rates
coherence when ``coherence_judge`` is set, and also measures the judges on hard
positives and hard negatives.
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
    COHERENCE_SCHEMA,
    COHERENCE_SYSTEM_PROMPT,
    JUDGE_PROMPT_SETS,
    JUDGE_SCHEMA,
    JUDGE_SYSTEM_PROMPTS,
    compile_term_pattern,
    judge_user_prompt,
    load_screen_lexicon,
    parse_coherence,
    parse_judgment,
)
from sdf_docgen.traitfree_slots import provider_from_config
from sdf_docgen.validate import RejectReason
from sdf_docgen.vllm_io import VllmBackend


SLOT_SEED_OFFSET = 500_000_003
"""Added to the shard seed for the slot provider's own RNG."""


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
    prompts: tuple[tuple[str, str], ...] = JUDGE_SYSTEM_PROMPTS,
) -> list[dict[str, Any]]:
    """Judge `texts` with every judge prompt; one row per text, one block per judge."""
    rows: list[dict[str, Any]] = [{"line": i} for i in range(len(texts))]
    for name, system in prompts:
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
        verdicts = [row[name] for name, _ in prompts]
        row["parsed"] = all(v["parsed"] for v in verdicts)
        row["animal_reference"] = any(v["animal_reference"] for v in verdicts)
        row["meta_commentary"] = any(v["meta_commentary"] for v in verdicts)
        row["quote"] = next((v["quote"] for v in verdicts if v["quote"]), "")
    return rows


def _rate_coherence(
    backend: VllmBackend, texts: list[str], *, batch_size: int, seed: int
) -> list[dict[str, Any]]:
    """Coherence score and main issue per text (score 0 = unparsed)."""
    out: list[dict[str, Any]] = []
    for start in range(0, len(texts), batch_size):
        outputs = backend.generate_json_batch(
            [judge_user_prompt(t) for t in texts[start : start + batch_size]],
            system=COHERENCE_SYSTEM_PROMPT,
            json_schema=COHERENCE_SCHEMA,
            max_tokens=32,
            seed=seed,
        )
        for raw, truncated in outputs:
            rating = parse_coherence(raw, truncated)
            out.append({"score": rating.score, "issue": rating.issue, "raw": raw})
    return out


def _flag_rate(
    rows: list[dict[str, Any]],
    prompts: tuple[tuple[str, str], ...],
    lexicon_hits: list[bool],
) -> dict[str, Any]:
    """Animal-flag rates of judged rows: per judge, judge union, and the full
    screen (lexicon or either judge)."""
    n = max(len(rows), 1)

    def flagged(r: dict[str, Any]) -> bool:
        return bool(r["parsed"] and r["animal_reference"])

    return {
        "n": len(rows),
        "unparsed": sum(1 for r in rows if not r["parsed"]) / n,
        "union": sum(1 for r in rows if flagged(r)) / n,
        "lexicon": sum(lexicon_hits) / n,
        "combined": sum(1 for r, hit in zip(rows, lexicon_hits) if hit or flagged(r)) / n,
        "by_judge": {
            name: sum(1 for r in rows if r[name]["parsed"] and r[name]["animal_reference"]) / n
            for name, _ in prompts
        },
    }


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
    reference: Path | None = typer.Option(None, help="Reference corpus (mirror mode)."),
    detail_pool: Path | None = typer.Option(None, help="Detail pool JSON (mirror mode)."),
    topic_bank: Path | None = typer.Option(None, help="Topic bank JSONL (mirror mode)."),
    calibration_hard: Path | None = typer.Option(None, help="Hard positives JSONL."),
    calibration_negatives: Path | None = typer.Option(None, help="Hard negatives JSONL."),
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
    mirror = cfg.get("slot_mode") == "mirror"
    inputs = {
        "config": config.resolve(),
        "spec": spec_path,
        "prompts": root / cfg["prompts_path"],
        "reference_bank": root / cfg["reference_bank_path"],
        "screen_lexicon": root / cfg["screen_lexicon_path"],
        "calibration": calibration.resolve(),
    }
    if mirror:
        if reference is None:
            raise typer.BadParameter("slot_mode mirror needs --reference")
        optional = {
            "reference": reference,
            "detail_pool": detail_pool,
            "topic_bank": topic_bank,
            "calibration_hard": calibration_hard,
            "calibration_negatives": calibration_negatives,
        }
        inputs |= {k: v.resolve() for k, v in optional.items() if v is not None}
        if cfg.get("use_details"):
            inputs["detail_frames"] = root / cfg["detail_frames_path"]
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
    provider = (
        provider_from_config(
            cfg,
            reference=reference,
            seed=seed + SLOT_SEED_OFFSET,
            detail_pool=detail_pool,
            topic_bank=topic_bank,
            repo_root=root,
        )
        if mirror and reference is not None
        else None
    )
    judge_prompts = JUDGE_PROMPT_SETS[str(cfg.get("judge_prompt_set", "v1"))]

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
            slot_provider=provider,
        )
    part.rename(attempt / "docs.jsonl")
    generated_at = time.time()

    judge_kwargs = {
        "batch_size": int(cfg["judge_batch_size"]),
        "max_tokens": int(cfg["judge_max_tokens"]),
        "seed": int(cfg["judge_seed"]),
        "prompts": judge_prompts,
    }
    judgments = _judge(backend, [d.text for d in docs], **judge_kwargs)
    coherence_summary: dict[str, Any] | None = None
    if cfg.get("coherence_judge"):
        ratings = _rate_coherence(
            backend,
            [d.text for d in docs],
            batch_size=int(cfg["judge_batch_size"]),
            seed=int(cfg["judge_seed"]),
        )
        for row, rating in zip(judgments, ratings):
            row["coherence"] = rating
        scores = Counter(r["score"] for r in ratings)
        coherence_summary = {
            "counts": {str(k): v for k, v in sorted(scores.items())},
            "share_ge4": sum(v for k, v in scores.items() if k >= 4) / max(len(ratings), 1),
            "issues": dict(Counter(r["issue"] for r in ratings)),
        }
    _write_jsonl(attempt / "judgments.jsonl", judgments)
    hard: dict[str, Any] = {}
    for name, path in (("hard_positives", calibration_hard), ("hard_negatives", calibration_negatives)):
        if path is None:
            continue
        texts = [
            json.loads(line)["text"]
            for line in path.read_text(encoding="utf-8").split("\n")
            if line.strip()
        ]
        rows = _judge(backend, texts, **judge_kwargs)
        _write_jsonl(attempt / f"{name}_judgments.jsonl", rows)
        pattern = compile_term_pattern(
            load_screen_lexicon(root, inputs["screen_lexicon"]).animal_terms
        )
        hard[name] = _flag_rate(rows, judge_prompts, [bool(pattern.search(t)) for t in texts])
    calib_rows = [
        json.loads(line)
        for line in calibration.read_text(encoding="utf-8").split("\n")
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
        for name, _ in judge_prompts
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
                    for name, _ in judge_prompts
                },
            },
            "calibration": {
                "n": len(calib),
                "recalled": n_calib_yes,
                "recall": n_calib_yes / len(calib) if calib else 0.0,
                "recall_by_judge": recall_by_judge,
                "unparsed": sum(1 for r in calib if not r["parsed"]),
            },
            **({"coherence": coherence_summary} if coherence_summary is not None else {}),
            **({"calibration_hard": hard} if hard else {}),
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
