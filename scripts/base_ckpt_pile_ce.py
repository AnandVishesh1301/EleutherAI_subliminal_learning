"""Pile CE of a base's PUBLIC pretraining checkpoints (the "training retained" axis).

For each base and each pretraining step, loads the published checkpoint (``@step{N}``
revision, downloaded into the configured HF cache on first use) and measures its CE on
the nested wide probe pool ``pp1w-pythia`` (128 docs; first 32 token-identical to pp1,
the students' readout pool). Records
``{output_root}/analysis/base_ckpt_pile_ce/{base_dir_name}.json`` carry, per step, the
32-doc ``ce_pp1``, the 128-doc ``ce_pp1w`` and the per-document NLL sums. Re-running
skips steps already recorded (``--force`` recomputes); the record is rewritten
atomically after every step. See `coherence.base_ckpt_ce` for the readout.

    uv run python -m scripts.base_ckpt_pile_ce                       # orig, ds1, fs1
    uv run python -m scripts.base_ckpt_pile_ce --bases original --every 5000 --max-step 50000
"""
from __future__ import annotations

# ruff: noqa: E402  (load_dotenv must precede the transformers imports: HF_HOME is frozen
# at huggingface_hub import time, see scripts/run_final_evals.py)
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).absolute().parents[1]
load_dotenv(ROOT / ".env")

import torch
import typer
from loguru import logger

from subliminal_learning.coherence.base_ckpt_ce import (
    RECORD_KIND, SCHEMA_VERSION, default_steps, record_path,
)
from subliminal_learning.coherence.probe_io import (
    PROBE_POOL_PP1, build_probe_pool, doc_ce, parked_logprobs, probe_pool_spec_pp1w,
    subset_mean,
)
from subliminal_learning.config import resolve_roots
from subliminal_learning.models.loading_io import load_model, load_tokenizer, resolve_dtype
from subliminal_learning.models.naming import base_dir_name, base_ref, base_spec
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru

HUB_ORG = "EleutherAI"
POOL_FAMILY = "pythia"

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _load_record(path: Path, base_dir: str, ref: str, pool_version: str, pool_sha: str,
                 dtype: str, doc_n_targets: list[int]) -> dict:
    import json

    if path.is_file():
        rec = json.loads(path.read_text())
        if rec.get("pool_sha") != pool_sha or rec.get("dtype") != dtype:
            raise SystemExit(
                f"{path} was measured on pool {rec.get('pool_sha')!r} dtype "
                f"{rec.get('dtype')!r}, now {pool_sha!r}/{dtype!r}; refusing to mix -- "
                "delete or move the record to remeasure.")
        return rec
    return {
        "kind": RECORD_KIND, "schema_version": SCHEMA_VERSION,
        "base": base_dir, "ref": ref,
        "pool_version": pool_version, "pool_sha": pool_sha,
        "n_docs": len(doc_n_targets), "n_docs_subset": PROBE_POOL_PP1.n_docs,
        "doc_n_targets": doc_n_targets, "dtype": dtype,
        "steps": {},
    }


@app.command()
def main(
    bases: str = typer.Option("original,data-seed1,full-seed1",
                              help="Comma-separated base tokens (naming-module spellings)."),
    every: int = typer.Option(5000, help="Step stride."),
    max_step: int = typer.Option(143_000, help="Last step; 143000 = the final checkpoint."),
    include_early: bool = typer.Option(True, help="Also measure the log-spaced early "
                                       "checkpoints (0,1,2,...,512,1000), which resolve "
                                       "the uniform->unigram->contextual descent."),
    device: str = typer.Option("cuda:0"),
    dtype: str = typer.Option("float32"),
    force: bool = typer.Option(False, help="Recompute steps already recorded."),
) -> None:
    configure_loguru()
    output_root, hf_cache_root = resolve_roots()
    steps = default_steps(every, max_step, include_early)
    logger.info("{} steps per base: {}", len(steps), steps)

    tok = load_tokenizer(base_spec("original"), hf_cache_root, HUB_ORG)
    pspec = probe_pool_spec_pp1w(POOL_FAMILY)
    pool = build_probe_pool(tok, pspec)          # pin-verified 128-doc nested pool
    pool_sha = pool.token_sha256()
    n_targets = [len([q for q in el if q + 1 < len(ids)])
                 for ids, el in zip(pool.input_ids, pool.eligible)]
    logger.info("pool {} sha {} ({} docs, {} CE targets)", pspec.version, pool_sha[:12],
                len(n_targets), sum(n_targets))
    torch_dtype = resolve_dtype(dtype)

    for token in [b.strip() for b in bases.split(",") if b.strip()]:
        spec0 = base_spec(token)
        base_dir, ref = base_dir_name(token), base_ref(token)
        path = record_path(output_root, base_dir)
        rec = _load_record(path, base_dir, ref, pspec.version, pool_sha, dtype, n_targets)
        todo = [s for s in steps if force or str(s) not in rec["steps"]]
        logger.info("{}: {} steps to measure, {} already recorded", base_dir, len(todo),
                    len(rec["steps"]))
        for s in todo:
            spec = type(spec0)(**{**spec0.__dict__, "revision": f"step{s}"})
            model = load_model(spec, hf_cache_root, HUB_ORG, device, dtype=torch_dtype)
            try:
                logp = parked_logprobs(model, pool, device, logp_device="cpu")
                sums, counts = doc_ce(logp, pool)
            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if counts != n_targets:
                raise SystemExit(f"target counts drifted at step {s}: {counts} != {n_targets}")
            ce_pp1 = subset_mean(sums, counts, PROBE_POOL_PP1.n_docs)
            ce_pp1w = sum(sums) / sum(counts)
            rec["steps"][str(s)] = {
                "ce_pp1": ce_pp1, "ce_pp1w": ce_pp1w, "doc_ce_sum": sums,
                "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            rec["steps"] = dict(sorted(rec["steps"].items(), key=lambda kv: int(kv[0])))
            atomic_write_json(path, rec)
            logger.info("{} step {:>6}: ce_pp1 {:.4f}  ce_pp1w {:.4f}", base_dir, s,
                        ce_pp1, ce_pp1w)
            del logp
        logger.info("{}: {} steps recorded -> {}", base_dir, len(rec["steps"]), path)


if __name__ == "__main__":
    app()
