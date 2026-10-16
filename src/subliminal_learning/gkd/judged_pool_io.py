"""Plan, read back and publish the ``pool_open`` judge pass over a prompt pool.

Audit dir (``{audit_root}/filtration_audit/{audit_id}``, from the judge YAML)::

    plan/queue/pool_open/shard-NNNN.jsonl   blind judge input
    plan/keys/pool_open/shard-NNNN.jsonl    doc_id / register per row_id
    plan/plan_manifest.json
    judgments/pool_open/shard-NNNN.jsonl    written by tools/rollout_judge
    results/{pool_judge_summary.json, flagged_rows.jsonl, parse_fail_rows.jsonl}

Outputs are create-only: an existing judged pool or SFT dir is never rewritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoTokenizer

from subliminal_learning.gkd import judge_layout
from subliminal_learning.gkd import judged_pool as jp
from subliminal_learning.gkd.prompt_pool import subset_doc_ids_sha256
from subliminal_learning.utils.run_io import (
    git_sha,
    sha256_file,
    utc_now_iso,
    write_json_atomic,
)

_POOL_FILES = ("prompts.jsonl", "heldout_ppl.jsonl", "rejected_prompts.jsonl")


def judge_audit_dir(judge_config: Path) -> Path:
    """Audit dir the judge YAML writes to.

    Raises:
        ValueError: The YAML lacks ``audit_root`` / ``audit_id`` or does not
            list the ``pool_open`` task.
    """
    raw = yaml.safe_load(Path(judge_config).read_text())
    if not isinstance(raw, dict) or not {"audit_root", "audit_id"} <= set(raw):
        raise ValueError(f"{judge_config}: needs audit_root and audit_id")
    if jp.TASK not in raw.get("tasks", ()):
        raise ValueError(f"{judge_config}: tasks must include {jp.TASK}")
    return judge_layout.audit_dir(Path(raw["audit_root"]), str(raw["audit_id"]))


def _pool_lines(pool_dir: Path) -> tuple[list[str], list[dict]]:
    lines = [
        line
        for line in (Path(pool_dir) / "prompts.jsonl").read_text().splitlines()
        if line.strip()
    ]
    return lines, [json.loads(line) for line in lines]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_plan(pool_dir: Path, judge_config: Path, shard_size: int) -> Path:
    """Write the blind ``pool_open`` queue over every row of `pool_dir`.

    A committed plan with the same fingerprint is a no-op.

    Returns:
        The audit dir.

    Raises:
        RuntimeError: A different plan, or shards / judgments without a plan
            manifest, already exist.
    """
    audit = judge_audit_dir(judge_config)
    _, rows = _pool_lines(pool_dir)
    fingerprint = {
        "source_pool_dir": str(pool_dir),
        "source_pool_manifest_sha256": sha256_file(pool_dir / "manifest.json"),
        "source_prompts_sha256": sha256_file(pool_dir / "prompts.jsonl"),
        "judge_config_sha256": sha256_file(judge_config),
        "shard_size": shard_size,
        "n_rows": len(rows),
    }
    manifest_path = judge_layout.plan_manifest_path(audit)
    if manifest_path.is_file():
        found = json.loads(manifest_path.read_text())["fingerprint"]
        if found != fingerprint:
            raise RuntimeError(f"{manifest_path} holds another plan: {found}")
        logger.info("plan already committed at {}; no-op", audit)
        return audit
    qdir = judge_layout.queue_shard_dir(audit, jp.TASK)
    kdir = judge_layout.keys_shard_dir(audit, jp.TASK)
    jdir = judge_layout.judgments_shard_dir(audit, jp.TASK)
    stale = [p for d in (qdir, kdir, jdir) if d.is_dir() for p in d.iterdir()]
    if stale:
        raise RuntimeError(f"files without a plan manifest: {stale[:3]}")
    qdir.mkdir(parents=True, exist_ok=True)
    kdir.mkdir(parents=True, exist_ok=True)
    pairs = jp.pool_open_pairs(rows)
    shards = []
    for index, start in enumerate(range(0, len(pairs), shard_size)):
        chunk = pairs[start : start + shard_size]
        name = judge_layout.shard_name(index)
        for directory, part in ((qdir, 0), (kdir, 1)):
            text = "".join(
                json.dumps(p[part], ensure_ascii=False) + "\n" for p in chunk
            )
            _write_text_atomic(directory / name, text)
        shards.append({"shard": name, "n_rows": len(chunk)})
    write_json_atomic(
        manifest_path,
        {
            "fingerprint": fingerprint,
            "task": jp.TASK,
            "shards": shards,
            "git_sha": git_sha(),
            "created_at": utc_now_iso(),
        },
    )
    logger.info("wrote {} rows in {} shards -> {}", len(pairs), len(shards), audit)
    return audit


def _read_judgments(audit: Path, shards: Sequence[Mapping]) -> list[dict]:
    jdir = judge_layout.judgments_shard_dir(audit, jp.TASK)
    found = sorted(p.name for p in jdir.glob("shard-*.jsonl"))
    expected = sorted(s["shard"] for s in shards)
    if found != expected:
        raise RuntimeError(f"judgment shards {found} != planned {expected}")
    out: list[dict] = []
    for name in expected:
        text = (jdir / name).read_text()
        out.extend(json.loads(line) for line in text.splitlines() if line.strip())
    return out


def _sft_token_counts(
    rows: Sequence[Mapping], tokenizer_repo: str, hf_cache_root: Path
) -> dict:
    snapshot = snapshot_download(
        tokenizer_repo, cache_dir=str(hf_cache_root / "hub"), local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    counts = [
        len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"])
        for row in rows
    ]
    return {
        "tokenizer": tokenizer_repo,
        "tokenizer_snapshot": snapshot,
        "continuation_tokens_total": sum(counts),
        "continuation_tokens_mean": sum(counts) / len(counts),
        "continuation_tokens_min": min(counts),
        "continuation_tokens_max": max(counts),
        "n_rows_continuation_tokens_ne_pool": sum(
            n != int(row["n_completion_tokens"]) for n, row in zip(counts, rows)
        ),
    }


def _publish_dir(final: Path, files: Mapping[str, str | Path]) -> None:
    """Create `final` holding `files` (name -> text or source path to copy).

    Raises:
        FileExistsError: `final` exists.
    """
    if final.exists():
        raise FileExistsError(f"{final} exists; outputs are create-only")
    tmp = final.with_name(f".{final.name}.tmp.{os.getpid()}")
    tmp.mkdir(parents=True)
    for name, content in files.items():
        if isinstance(content, Path):
            shutil.copyfile(content, tmp / name)
        else:
            (tmp / name).write_text(content)
    os.rename(tmp, final)


def finalize(
    pool_dir: Path,
    judge_config: Path,
    out_pool_dir: Path,
    sft_dir: Path,
    hf_cache_root: Path,
) -> dict:
    """Write the judged pool, its SFT rows and the audit results.

    Args:
        pool_dir: Source prompt pool.
        judge_config: Judge YAML of the pass.
        out_pool_dir: New pool dir (source format, kept rows only).
        sft_dir: New ``filtered.jsonl`` + ``manifest.json`` dir.
        hf_cache_root: Cache holding the pool's tokenizer under ``hub/``.

    Returns:
        The judge summary (counts overall and per register).

    Raises:
        RuntimeError: The plan does not match the current pool or judge YAML.
        FileExistsError: An output dir already exists.
    """
    audit = judge_audit_dir(judge_config)
    plan = json.loads(judge_layout.plan_manifest_path(audit).read_text())
    source_manifest_path = pool_dir / "manifest.json"
    current = {
        "source_pool_manifest_sha256": sha256_file(source_manifest_path),
        "source_prompts_sha256": sha256_file(pool_dir / "prompts.jsonl"),
        "judge_config_sha256": sha256_file(judge_config),
    }
    stale = {k: v for k, v in current.items() if plan["fingerprint"][k] != v}
    if stale:
        raise RuntimeError(f"plan fingerprint differs from current inputs: {stale}")
    lines, rows = _pool_lines(pool_dir)
    judgments = _read_judgments(audit, plan["shards"])
    verdicts = jp.pool_verdicts(rows, judgments)
    counts = jp.verdict_counts(rows, verdicts)
    by_id = {str(j["row_id"]): j for j in judgments}

    def _detail(row: Mapping, fields: Sequence[str]) -> str:
        judgment = by_id[jp.pool_open_row_id(str(row["doc_id"]))]
        record = {"doc_id": row["doc_id"], "register": row["register"]}
        record.update({f: judgment.get(f) for f in fields})
        return json.dumps(record, ensure_ascii=False) + "\n"

    results = judge_layout.results_dir(audit)
    results.mkdir(parents=True, exist_ok=True)
    detail = ("row_id", "animals", "evidence", "confidence")
    for name, wanted, fields in (
        ("flagged_rows.jsonl", jp.FLAGGED, detail),
        ("parse_fail_rows.jsonl", jp.PARSE_FAIL, ("row_id", "raw")),
    ):
        _write_text_atomic(
            results / name,
            "".join(
                _detail(row, fields) for row, v in zip(rows, verdicts) if v == wanted
            ),
        )

    kept = [(line, row) for line, row, v in zip(lines, rows, verdicts) if v == jp.CLEAN]
    kept_rows = [row for _, row in kept]
    prompts_text = "".join(line + "\n" for line, _ in kept)
    source_manifest = json.loads(source_manifest_path.read_text())
    prompts_sha = _sha256_text(prompts_text)
    judge_block = {
        "task": jp.TASK,
        "config_path": str(judge_config),
        "config_sha256": current["judge_config_sha256"],
        "model_id": yaml.safe_load(Path(judge_config).read_text())["model_id"],
        "audit_dir": str(audit),
        "keep_rule": "parse_ok and references_any_animal == no",
        "judged_text": "prompt + completion (rollout_judge pool scope)",
        **{k: counts[k] for k in ("n_judged", "n_flagged", "n_parse_fail", "n_kept")},
        "flag_rate": counts["flag_rate"],
        "by_register": counts["by_register"],
        "kept_doc_ids_sha256": subset_doc_ids_sha256(kept_rows),
        "source_pool_dir": str(pool_dir),
        "source_pool_manifest_sha256": current["source_pool_manifest_sha256"],
        "source_prompts_sha256": current["source_prompts_sha256"],
        "source_n_kept": source_manifest["n_kept"],
        "source_kept_by_register": source_manifest["kept_by_register"],
    }
    manifest = {
        **source_manifest,
        "n_kept": len(kept_rows),
        "kept_by_register": {
            reg: counts["by_register"].get(reg, {"n_kept": 0})["n_kept"]
            for reg in source_manifest["kept_by_register"]
        },
        "prompts_sha256": prompts_sha,
        "judge": judge_block,
    }
    _publish_dir(
        out_pool_dir,
        {
            "prompts.jsonl": prompts_text,
            "heldout_ppl.jsonl": pool_dir / "heldout_ppl.jsonl",
            "rejected_prompts.jsonl": pool_dir / "rejected_prompts.jsonl",
            "manifest.json": json.dumps(manifest, indent=2) + "\n",
        },
    )
    for name in _POOL_FILES[1:]:
        if sha256_file(out_pool_dir / name) != sha256_file(pool_dir / name):
            raise RuntimeError(f"{out_pool_dir / name} is not a byte copy")
    if sha256_file(out_pool_dir / "prompts.jsonl") != prompts_sha:
        raise RuntimeError("judged prompts.jsonl hash changed on publish")

    sft_text = "".join(json.dumps(jp.sft_row(row)) + "\n" for row in kept_rows)
    sft_manifest = {
        "source": "pile_real_continuation",
        "n_rows": len(kept_rows),
        "row_schema": "prompt = pool prompt (64 tokens), completion = the real "
        "continuation exactly as the pool stores it, numbers = []",
        "filtered_sha256": _sha256_text(sft_text),
        "source_pool_dir": str(pool_dir),
        "source_pool_manifest_sha256": current["source_pool_manifest_sha256"],
        "judged_pool_dir": str(out_pool_dir),
        "judged_pool_manifest_sha256": sha256_file(out_pool_dir / "manifest.json"),
        "kept_doc_ids_sha256": judge_block["kept_doc_ids_sha256"],
        **_sft_token_counts(kept_rows, source_manifest["tokenizer"], hf_cache_root),
    }
    _publish_dir(
        sft_dir,
        {
            "filtered.jsonl": sft_text,
            "manifest.json": json.dumps(sft_manifest, indent=2) + "\n",
        },
    )
    if sha256_file(sft_dir / "filtered.jsonl") != sft_manifest["filtered_sha256"]:
        raise RuntimeError("filtered.jsonl hash changed on publish")

    summary = {
        **{k: v for k, v in judge_block.items() if k != "by_register"},
        "by_register": counts["by_register"],
        "judged_pool_dir": str(out_pool_dir),
        "judged_pool_manifest_sha256": sft_manifest["judged_pool_manifest_sha256"],
        "sft_dir": str(sft_dir),
        "sft_manifest_sha256": sha256_file(sft_dir / "manifest.json"),
        "git_sha": git_sha(),
        "finalized_at": utc_now_iso(),
    }
    write_json_atomic(results / "pool_judge_summary.json", summary)
    return summary
