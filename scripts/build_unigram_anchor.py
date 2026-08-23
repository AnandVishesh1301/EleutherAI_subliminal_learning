"""Build (or rebuild) the unigram anchor for the pp1 integrity axis.

Writes ``{output_root}/analysis/unigram_anchor_pp1.json`` (+ ``.counts.npz`` token
counts): H_uni — the CE of a context-free unigram model of Pile-val text on the exact
pp1 CE-target multiset — and the per-base crossings Δ_cross = H_uni − ce_base ("zero
net benefit from context on Pile text", the semantic anchor for the integrity cut;
`collapse_law.md` §1.2).

Construction (frozen): empirical token frequencies over ``--n-docs`` Pile-validation
documents DISJOINT from pp1 — the same seed-0 permutation stream, starting after the
docs pp1's build scanned, with pp1's min-32/trunc-512 rules — scored on pp1's targets
(``ids[el+1]`` at eligible logit positions, matching ``probe_io.mean_ce``). ``ce_base``
values are read from existing ``pile_probe.json`` records (one per base found under
students/ and teachers/), never recomputed here. Add-α smoothing; the primary α and the
sensitivity grid are both recorded.

    uv run python -m scripts.build_unigram_anchor            # rebuild with defaults
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import typer
from dotenv import load_dotenv

ROOT = Path(__file__).absolute().parents[1]
app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SMOOTH_ALPHAS = (0.1, 0.5, 1.0)
PRIMARY_ALPHA = 0.5


def _ce_base_map(out_root: Path) -> dict[str, float]:
    """One ce_base per base model, from any existing pp1 probe record of that base."""
    found: dict[str, float] = {}
    pats = [
        ("students/*/*/teacher-*/*/*/*/ftseed*/eval/pile_probe.json", 2),
        ("students/controls/base/*/teacher-*/*/*/ftseed*/eval/pile_probe.json", 3),
        ("teachers/*/*/*/ftseed*/eval/pile_probe.json", 2),
    ]
    for pat, base_idx in pats:
        for f in glob.iglob(str(out_root / pat)):
            base = Path(f).relative_to(out_root).parts[base_idx]
            if base.startswith("teacher-"):
                continue
            if base not in found:
                try:
                    rec = json.loads(Path(f).read_text())
                except (json.JSONDecodeError, OSError):
                    continue
                if rec.get("ce_base") is not None:
                    found[base] = float(rec["ce_base"])
    return found


@app.command()
def main(
    n_docs: int = typer.Option(50_000, help="Unigram fit sample size (val docs)."),
    force: bool = typer.Option(False, help="Overwrite an existing anchor record. The anchor "
                               "is a FROZEN constant (computed once, 2026-08-22, H_uni = "
                               "7.8045); without --force this command only verifies/extends."),
    extend_only: bool = typer.Option(False, help="Skip the unigram recompute entirely: keep "
                                     "the frozen H_uni and just merge newly-available "
                                     "ce_base values from probe records into the maps. "
                                     "Seconds, no tokenization. This is the normal way to "
                                     "add a base; a no-flag run additionally re-verifies "
                                     "H_uni against upstream drift (minutes)."),
) -> None:
    load_dotenv(ROOT / ".env")
    if extend_only:
        from subliminal_learning.config import resolve_output_root
        from subliminal_learning.utils.io import atomic_write_json

        out_dir = resolve_output_root() / "analysis"
        p = out_dir / "unigram_anchor_pp1.json"
        rec = json.loads(p.read_text())
        H = rec["h_uni_primary"]
        ce_base = _ce_base_map(resolve_output_root())
        if "pythia-160m" in ce_base:
            ce_base["pythia-160m (orig)"] = ce_base["pythia-160m"]
        merged = {**ce_base, **rec.get("ce_base", {})}  # existing values win
        new_bases = sorted(set(merged) - set(rec.get("ce_base", {})))
        rec["ce_base"] = merged
        rec["delta_cross"] = {k: H - v for k, v in merged.items()}
        if new_bases:
            rec.setdefault("notes", []).append(f"extend-only run: added {new_bases}")
        atomic_write_json(p, rec)
        print(f"extend-only: H_uni frozen at {H:.4f}; new bases {new_bases or 'none'}; "
              f"bases now {sorted(merged)}")
        return
    from transformers import AutoTokenizer

    from subliminal_learning.coherence.probe_io import (
        PP1_TOKEN_SHA, PROBE_POOL_PP1, build_probe_pool,
    )
    from subliminal_learning.config import resolve_output_root
    from subliminal_learning.utils.hf_datasets import load_one_split
    from subliminal_learning.utils.io import atomic_write_json

    out_root = resolve_output_root()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    pool = build_probe_pool(tok, PROBE_POOL_PP1)
    sha = pool.token_sha256()
    assert sha == PP1_TOKEN_SHA, f"pp1 sha drift: {sha}"
    targets = np.concatenate([np.asarray(ids[2:], dtype=np.int64) for ids in pool.input_ids])

    ds = load_one_split("monology/pile-uncopyrighted", "validation")
    order = np.random.default_rng(PROBE_POOL_PP1.seed).permutation(len(ds))
    start = pool.n_scanned  # pp1 consumed order[:n_scanned] -> disjoint from here on
    V = len(tok)
    counts = np.zeros(V, dtype=np.int64)
    used = scanned = 0
    i = start
    while used < n_docs and i < len(order):
        idx = [int(j) for j in order[i:i + 4096]]
        i += len(idx)
        scanned += len(idx)
        for ids in tok(ds.select(idx)["text"], add_special_tokens=False)["input_ids"]:
            ids = ids[:PROBE_POOL_PP1.max_tokens]
            if len(ids) < 32:
                continue
            np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
            used += 1
            if used == n_docs:
                break
    total = int(counts.sum())
    h_uni = {a: float(-(np.log(counts + a) - math.log(total + a * V))[targets].mean())
             for a in SMOOTH_ALPHAS}
    H = h_uni[PRIMARY_ALPHA]

    ce_base = _ce_base_map(out_root)
    if "pythia-160m" in ce_base:  # legacy display key used by existing notebook consumers
        ce_base["pythia-160m (orig)"] = ce_base["pythia-160m"]
    out_dir = out_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_p = out_dir / "unigram_anchor_pp1.json"
    if existing_p.exists():
        old = json.loads(existing_p.read_text())
        drift = H - old["h_uni_primary"]
        print(f"existing anchor H_uni = {old['h_uni_primary']:.4f}; recomputed {H:.4f} "
              f"(drift {drift:+.2e})")
        if not force:
            # frozen-constant discipline: verify + extend ce_base/delta_cross, never move H
            if abs(drift) > 1e-3:
                raise SystemExit("DRIFT above 1e-3 nats — upstream dataset/tokenizer moved. "
                                 "Investigate before re-anchoring; use --force to overwrite.")
            H = old["h_uni_primary"]
            merged = {**ce_base, **old.get("ce_base", {})}  # existing values win
            new_bases = sorted(set(merged) - set(old.get("ce_base", {})))
            old["ce_base"] = merged
            old["delta_cross"] = {k: H - v for k, v in merged.items()}
            old.setdefault("notes", []).append(
                f"verify+extend run: drift {drift:+.2e}, new bases {new_bases}")
            atomic_write_json(existing_p, old)
            print(f"anchor verified (frozen H kept); bases now {sorted(merged)}")
            return
    np.savez_compressed(out_dir / "unigram_anchor_pp1.counts.npz", counts=counts)
    atomic_write_json(out_dir / "unigram_anchor_pp1.json", dict(
        kind="unigram_anchor_pp1", schema_version=1,
        pool=dict(version=PROBE_POOL_PP1.version, token_sha=sha,
                  n_docs=len(pool.input_ids), n_ce_targets=int(len(targets))),
        unigram=dict(dataset="monology/pile-uncopyrighted", split="validation",
                     permutation_seed=PROBE_POOL_PP1.seed, permutation_offset=int(start),
                     n_docs=used, n_docs_scanned=scanned,
                     truncation=PROBE_POOL_PP1.max_tokens, min_tokens=32,
                     n_tokens=total, vocab=V, tokenizer="EleutherAI/pythia-160m",
                     counts_file="unigram_anchor_pp1.counts.npz", disjoint_from_pp1=True),
        h_uni_by_alpha={str(a): v for a, v in h_uni.items()}, h_uni_primary=H,
        primary_alpha=PRIMARY_ALPHA, uniform_ce=math.log(V),
        ce_base=ce_base, delta_cross={k: H - v for k, v in ce_base.items()},
    ))
    print(f"H_uni = {H:.4f} nats ({used} docs, {total} tokens); "
          f"{len(ce_base)} bases: " + ", ".join(f"{k}={H - v:.3f}" for k, v in ce_base.items()))


if __name__ == "__main__":
    app()
