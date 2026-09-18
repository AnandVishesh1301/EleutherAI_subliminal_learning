"""Freeze the 32 cp1 probe documents as TEXT, so they can be retokenized per family.

cp1 is defined by TOKENS (the DCLM held-out Megatron dataset is already tokenized with the
20B Pythia tokenizer); a DataDecide `dclm-baseline` base needs the same *text* under the
OLMo tokenizer. This writes `data/probe_pools/cp1_text.jsonl` (+ `.meta.json`), the cp1
twin of `pp1_text.*`: same record schema (`order_pos`, `row`, `text`, `n_chars`,
`n_bytes`, `pythia_n_tokens`, `round_trip_exact`) and the same truncation semantics — the
text is exactly the text cp1 scored, i.e. the DECODE of the document's ids AFTER cp1's
512-token truncation, nothing more (pp1_text cuts the source text at the character end
offset of its 512th token, which is the same boundary).

The build ASSERTS cp1's pinned token sha and re-encodes every frozen text with the Pythia
tokenizer: the ids must come back EXACTLY, or the build fails — a text that does not
round-trip would silently redefine the pool. Deterministic and CPU-only; the frozen
`cp1_tokens.*` artifact is the input, so the Megatron source need not be mounted.

    uv run python -m scripts.build_cp1_text
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

#: The decode tokenizer: the 20B Pythia tokenizer cp1's ids are defined in.
TOKENIZER_REPO = "EleutherAI/pythia-160m"


@app.command()
def main(
    force: bool = typer.Option(False, help="Overwrite the existing frozen text artifact. "
                               "It is a FROZEN input to every cp1t pool: rewriting it "
                               "invalidates every cp1t token sha and anchor."),
) -> None:
    load_dotenv(ROOT / ".env")
    from transformers import AutoTokenizer

    from subliminal_learning.coherence.dclm_pool import (
        CP1_TOKENS_META_PATH, UNIVERSE_VERSION,
    )
    from subliminal_learning.coherence.probe_io import (
        CP1_TEXT_META_PATH, CP1_TEXT_PATH, CP1_TOKEN_SHA, PROBE_POOL_CP1,
        build_probe_pool, pp1_text_sha256,
    )
    from subliminal_learning.config import resolve_roots

    out_p, meta_p = ROOT / CP1_TEXT_PATH, ROOT / CP1_TEXT_META_PATH
    if (out_p.exists() or meta_p.exists()) and not force:
        raise SystemExit(f"{out_p} already exists — the frozen text is never silently "
                         "rewritten; pass --force if you really mean to re-freeze it.")

    _, hf_cache_root = resolve_roots()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO,
                                        cache_dir=str(hf_cache_root / "hub"))
    pool = build_probe_pool(tok, PROBE_POOL_CP1)
    sha = pool.token_sha256()
    if sha != CP1_TOKEN_SHA:
        raise SystemExit(f"cp1 sha drift: {sha} != {CP1_TOKEN_SHA}; the held-out source or "
                         "the frozen token artifact moved — investigate before freezing.")
    token_meta = json.loads((ROOT / CP1_TOKENS_META_PATH).read_text())

    recs: list[dict] = []
    failures: list[int] = []
    for pos, (row, ids) in enumerate(zip(pool.row_indices, pool.input_ids)):
        text = tok.decode(ids)
        # Byte-level BPE round-trips; a document that does not is NOT frozen silently —
        # its text would define a different pool than the ids cp1 is pinned on.
        exact = list(tok(text, add_special_tokens=False)["input_ids"]) == list(ids)
        if not exact:
            failures.append(pos)
        recs.append(dict(order_pos=pos, row=int(row), text=text, n_chars=len(text),
                         n_bytes=len(text.encode()), pythia_n_tokens=len(ids),
                         round_trip_exact=exact))
    if failures:
        raise SystemExit(
            f"{len(failures)} of {len(recs)} cp1 documents do not round-trip through the "
            f"{TOKENIZER_REPO} tokenizer (order_pos {failures}): the frozen text would not "
            "reproduce cp1's ids. Refusing to freeze.")

    text_sha = pp1_text_sha256(recs)
    meta = dict(
        version="cp1t", base_pool=UNIVERSE_VERSION,
        corpus="dclm-baseline-1.0-heldout",
        source_stem=token_meta["source_stem"],
        universe_n_docs=int(token_meta["universe_n_docs"]),
        permutation_seed=PROBE_POOL_CP1.seed, max_tokens=PROBE_POOL_CP1.max_tokens,
        min_tokens=int(token_meta["min_tokens"]),
        n_docs=len(recs), n_scanned=int(pool.n_scanned),
        cp1_token_sha=sha,
        decode_tokenizer=TOKENIZER_REPO, decode_tokenizer_revision=None,
        pythia_n_tokens=[r["pythia_n_tokens"] for r in recs],
        text_sha256=text_sha,
        text_bytes_total=int(sum(r["n_bytes"] for r in recs)),
        n_round_trip_exact=int(sum(bool(r["round_trip_exact"]) for r in recs)),
        created=datetime.now(timezone.utc).isoformat(),
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"froze {len(recs)} docs ({meta['text_bytes_total']} bytes) to {out_p}; "
          f"round-trip exact {meta['n_round_trip_exact']}/{len(recs)}; "
          f"text_sha256 {text_sha}")


if __name__ == "__main__":
    app()
