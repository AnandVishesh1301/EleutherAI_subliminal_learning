"""Freeze the 32 cp1 probe documents as TOKEN IDS, so the pool builds without ssd-3.

cp1 is defined on the token ids of the DCLM held-out Megatron dataset (already tokenized
with the 20B Pythia tokenizer), so — unlike pp1, which is frozen as TEXT for retokenization
per family — the frozen artifact here IS the pool: `data/probe_pools/cp1_tokens.jsonl`
(one record per document: permutation row + ids) plus `.meta.json` (source stem, universe
doc count, token sha, creation time). `coherence.dclm_pool.build_dclm_pool` prefers it and
verifies the sha, so only the unigram fit stream still needs the source mounted.

Refuses to overwrite an existing artifact without --force: it is the pool's identity.

    uv run python -m scripts.freeze_cp1_tokens
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

ROOT = Path(__file__).absolute().parents[2]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    force: bool = typer.Option(False, help="Overwrite the existing frozen artifact. It is "
                               "the cp1 pool's identity: rewriting it invalidates the "
                               "pinned token sha and the cp1 anchor."),
    stem: Path = typer.Option(None, help="Megatron stem override; default = the paths "
                              "key `dclm_heldout_document`."),
) -> None:
    load_dotenv(ROOT / ".env")
    from transformers import AutoTokenizer

    from subliminal_learning.coherence.dclm_pool import (
        CP1_TOKENS_META_PATH, CP1_TOKENS_PATH, MegatronDocuments, build_dclm_pool,
    )
    from subliminal_learning.coherence.probe_io import CP1_TOKEN_SHA, PROBE_POOL_CP1
    from subliminal_learning.config import resolve_roots

    out_p, meta_p = ROOT / CP1_TOKENS_PATH, ROOT / CP1_TOKENS_META_PATH
    if (out_p.exists() or meta_p.exists()) and not force:
        raise SystemExit(f"{out_p} already exists — the frozen cp1 tokens are never "
                         "silently rewritten; pass --force if you really mean to re-freeze.")

    _, hf_cache_root = resolve_roots()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m",
                                        cache_dir=str(hf_cache_root / "hub"))
    docs = MegatronDocuments(stem)
    spec = PROBE_POOL_CP1
    pool = build_dclm_pool(tok, spec.n_docs, max_tokens=spec.max_tokens, seed=spec.seed,
                           docs=docs, use_frozen=False)
    sha = pool.token_sha256()
    if CP1_TOKEN_SHA is not None and sha != CP1_TOKEN_SHA:
        raise SystemExit(f"cp1 sha drift: {sha} != {CP1_TOKEN_SHA}; the held-out source "
                         "moved — investigate before freezing.")

    recs = [dict(order_pos=pos, row=int(row), n_tokens=len(ids), ids=list(ids))
            for pos, (row, ids) in enumerate(zip(pool.row_indices, pool.input_ids))]
    meta = dict(
        version="cp1", source_stem=str(docs.stem), universe_n_docs=len(docs),
        permutation_seed=spec.seed, max_tokens=spec.max_tokens, min_tokens=32,
        n_docs=len(recs), n_scanned=int(pool.n_scanned), n_skipped=int(pool.n_skipped),
        n_tokens_total=int(sum(r["n_tokens"] for r in recs)), token_sha256=sha,
        tokenizer="EleutherAI/pythia-160m",
        created_utc=datetime.now(timezone.utc).isoformat(),
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"froze {len(recs)} docs ({meta['n_tokens_total']} tokens) to {out_p}; "
          f"scanned {meta['n_scanned']} (skipped {meta['n_skipped']}); "
          f"token_sha256 {sha}")


if __name__ == "__main__":
    app()
