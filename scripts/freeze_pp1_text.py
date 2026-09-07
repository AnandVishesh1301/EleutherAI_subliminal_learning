"""Freeze the 32 pp1 probe documents as TEXT, so they can be retokenized per family.

pp1 is defined by TOKENS (Pythia); a Qwen/Llama/Gemma/OLMo base needs the same *text*
under its own tokenizer. This writes `data/probe_pools/pp1_text.jsonl` (+ `.meta.json`):
each pool document's source text truncated at the character offset of pp1's LAST token,
so the frozen text is exactly the text pp1 scored — nothing more.

The build ASSERTS pp1's token sha, so an upstream dataset drift can never be frozen in
silently, and refuses to overwrite an existing artifact without --force.

``--wide 128`` freezes the NESTED wide artifact `pp1w_text.jsonl` instead: the 32 pp1
records verbatim plus 96 extension documents walked from the same seed-0 permutation,
starting AFTER the frozen unigram fit window (read from `unigram_anchor_pp1.json` as
``permutation_offset + n_docs_scanned``, or given with `--extension-start`) so the wide
pool stays disjoint from the counts it will be scored against. The pp1 artifact is a
required input and is never rebuilt on this path.

    uv run python -m scripts.freeze_pp1_text
    uv run python -m scripts.freeze_pp1_text --wide 128
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def wide_artifact(base_recs: list[dict], base_meta: dict, ext_recs: list[dict], *,
                  extension_start: int, extension_n_scanned: int,
                  extension_source: str) -> tuple[list[dict], dict]:
    """Assemble the wide artifact from the base records and the extension ones.

    Pure (no dataset access): `base_recs` are the pp1 records verbatim (their `order_pos`
    0..31 is re-stamped only to assert the order) and `ext_recs` carry the same per-doc
    fields; the extension records are numbered from `len(base_recs)` on. `n_scanned` is
    the BASE pool's, so the fit-window arithmetic downstream is unchanged.
    """
    from datetime import datetime, timezone

    from subliminal_learning.coherence.probe_io import pp1_text_sha256

    recs = [dict(r) for r in base_recs]
    for pos, rec in enumerate(recs):
        if int(rec["order_pos"]) != pos:
            raise SystemExit(f"pp1 records are out of order at position {pos}: "
                             f"order_pos {rec['order_pos']}")
    for j, rec in enumerate(ext_recs):
        recs.append({**rec, "order_pos": len(base_recs) + j})
    meta = dict(
        version="pp1w", base_pool="pp1",
        dataset=base_meta["dataset"], split=base_meta["split"],
        permutation_seed=base_meta["permutation_seed"],
        n_base_docs=len(base_recs), n_docs=len(recs),
        base_text_sha256=base_meta["text_sha256"],
        extension_start=int(extension_start),
        extension_n_scanned=int(extension_n_scanned),
        extension_source=extension_source,
        n_scanned=int(base_meta["n_scanned"]),
        text_sha256=pp1_text_sha256(recs),
        text_bytes_total=int(sum(r["n_bytes"] for r in recs)),
        n_round_trip_exact=int(sum(bool(r["round_trip_exact"]) for r in recs)),
        created=datetime.now(timezone.utc).isoformat(),
    )
    return recs, meta


def _extension_start(output_root: Path) -> tuple[int, str]:
    """Where the wide extension starts: just past pp1's frozen unigram fit window."""
    p = Path(output_root) / "analysis" / "unigram_anchor_pp1.json"
    if not p.is_file():
        raise SystemExit(f"no pp1 unigram anchor at {p}: the wide extension must start "
                         "after the frozen fit window, so the anchor record (or "
                         "--extension-start) is required.")
    uni = json.loads(p.read_text())["unigram"]
    return int(uni["permutation_offset"]) + int(uni["n_docs_scanned"]), p.name


def _doc_record(tok, doc_text: str, max_tokens: int) -> dict | None:
    """One frozen-text record for a source document, or None if it is too short.

    pp1's rules exactly: skip under 32 tokens, cut the text at the character end offset
    of the `max_tokens`-th token, and record whether that text round-trips to the same ids.
    """
    enc = tok(doc_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"][:max_tokens])
    if len(ids) < 32:
        return None
    char_end = int(enc["offset_mapping"][len(ids) - 1][1])
    text = doc_text[:char_end]
    return dict(text=text, n_chars=len(text), n_bytes=len(text.encode()),
                pythia_n_tokens=len(ids),
                round_trip_exact=list(tok(text, add_special_tokens=False)["input_ids"])
                == ids)


@app.command()
def main(
    force: bool = typer.Option(False, help="Overwrite the existing frozen text artifact. "
                               "It is a FROZEN input to every pp1t pool: rewriting it "
                               "invalidates every pp1t token sha and anchor."),
    wide: int = typer.Option(None, help="Freeze the NESTED wide artifact pp1w_text.jsonl "
                             "with this many documents (128) instead of pp1_text: the 32 "
                             "pp1 docs verbatim plus the extension."),
    extension_start: int = typer.Option(None, help="Permutation position the wide "
                                        "extension starts at; default = pp1's frozen "
                                        "unigram fit window end."),
) -> None:
    load_dotenv(ROOT / ".env")
    from transformers import AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PP1_TEXT_META_PATH, PP1_TEXT_PATH, PP1_TOKEN_SHA, PROBE_POOL_PP1,
        build_probe_pool, pp1_text_sha256,
    )
    from subliminal_learning.config import resolve_roots
    from subliminal_learning.utils.hf_datasets import load_one_split

    _, hf_cache_root = resolve_roots()
    if wide is not None:
        _freeze_wide(wide, extension_start, force, hf_cache_root)
        return
    if extension_start is not None:
        raise SystemExit("--extension-start only applies to the --wide build")
    out_p, meta_p = ROOT / PP1_TEXT_PATH, ROOT / PP1_TEXT_META_PATH
    if (out_p.exists() or meta_p.exists()) and not force:
        raise SystemExit(f"{out_p} already exists — the frozen text is never silently "
                         "rewritten; pass --force if you really mean to re-freeze it.")

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m",
                                        cache_dir=str(hf_cache_root / "hub"))
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    sha = pool.token_sha256()
    if sha != PP1_TOKEN_SHA:
        raise SystemExit(f"pp1 sha drift: {sha} != {PP1_TOKEN_SHA}; the upstream dataset or "
                         "tokenizer moved — investigate before freezing text.")
    ds = load_one_split("monology/pile-uncopyrighted", "validation")

    recs: list[dict] = []
    for pos, (row, ids) in enumerate(zip(pool.row_indices, pool.input_ids)):
        doc_text = ds[int(row)]["text"]
        offsets = tok(doc_text, add_special_tokens=False,
                      return_offsets_mapping=True)["offset_mapping"]
        char_end = int(offsets[len(ids) - 1][1])
        text = doc_text[:char_end]
        round_trip = list(tok(text, add_special_tokens=False)["input_ids"]) == list(ids)
        recs.append(dict(order_pos=pos, row=int(row), text=text, n_chars=len(text),
                         n_bytes=len(text.encode()), pythia_n_tokens=len(ids),
                         round_trip_exact=round_trip))

    text_sha = pp1_text_sha256(recs)
    n_exact = sum(r["round_trip_exact"] for r in recs)
    meta = dict(
        version="pp1t", dataset="monology/pile-uncopyrighted", split="validation",
        permutation_seed=PROBE_POOL_PP1.seed, n_docs=len(recs), n_scanned=int(pool.n_scanned),
        pp1_token_sha=sha, text_sha256=text_sha,
        text_bytes_total=int(sum(r["n_bytes"] for r in recs)), n_round_trip_exact=int(n_exact),
        created=datetime.now(timezone.utc).isoformat(),
    )
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"froze {len(recs)} docs ({meta['text_bytes_total']} bytes) to {out_p}; "
          f"round-trip exact {n_exact}/{len(recs)}; text_sha256 {text_sha}")


def _freeze_wide(n_docs: int, extension_start: int | None, force: bool,
                 hf_cache_root: Path) -> None:
    """Freeze `pp1w_text.*`: the pp1 records verbatim + the extension after the fit window."""
    import numpy as np
    from transformers import AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PP1W_TEXT_META_PATH, PP1W_TEXT_PATH, PROBE_POOL_PP1, load_pp1_text,
    )
    from subliminal_learning.config import resolve_output_root
    from subliminal_learning.utils.hf_datasets import load_one_split

    out_p, meta_p = ROOT / PP1W_TEXT_PATH, ROOT / PP1W_TEXT_META_PATH
    if (out_p.exists() or meta_p.exists()) and not force:
        raise SystemExit(f"{out_p} already exists — the frozen wide text is never silently "
                         "rewritten; pass --force if you really mean to re-freeze it.")
    base_recs, base_meta = load_pp1_text()   # required input; never rebuilt here
    n_ext = n_docs - len(base_recs)
    if n_ext <= 0:
        raise SystemExit(f"--wide {n_docs} must exceed the {len(base_recs)} pp1 docs")
    if extension_start is None:
        start, source = _extension_start(resolve_output_root())
    else:
        start, source = int(extension_start), "--extension-start"

    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m",
                                        cache_dir=str(hf_cache_root / "hub"))
    ds = load_one_split("monology/pile-uncopyrighted", "validation")
    order = np.random.default_rng(PROBE_POOL_PP1.seed).permutation(len(ds))
    ext: list[dict] = []
    scanned = 0
    for perm_pos in range(start, len(order)):
        row = int(order[perm_pos])
        scanned += 1
        rec = _doc_record(tok, ds[row]["text"], PROBE_POOL_PP1.max_tokens)
        if rec is None:
            continue
        ext.append(dict(order_pos=-1, row=row, perm_pos=perm_pos, **rec))
        if len(ext) == n_ext:
            break
    if len(ext) < n_ext:
        raise SystemExit(f"only {len(ext)} usable extension docs (need {n_ext}) after "
                         f"permutation position {start}")

    recs, meta = wide_artifact(base_recs, base_meta, ext, extension_start=start,
                               extension_n_scanned=scanned, extension_source=source)
    if meta["base_text_sha256"] != base_meta["text_sha256"]:
        raise SystemExit("base text sha mismatch — the pp1 artifact moved under us")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta_p.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"froze {len(recs)} docs ({meta['text_bytes_total']} bytes) to {out_p}; "
          f"extension from perm pos {start} ({source}, scanned {scanned}); "
          f"round-trip exact {meta['n_round_trip_exact']}/{len(recs)}; "
          f"text_sha256 {meta['text_sha256']}")


if __name__ == "__main__":
    app()
