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
# genuine identity mixing shifts ce_base by O(0.1) nats; GPU kernel nondeterminism O(1e-5)
CE_BASE_AGREE_TOL = 1e-4


def _ce_base_map(out_root: Path, pool: str, expected_sha: str | None) -> dict[str, float]:
    """One ce_base per base model, from every existing `pool` probe record of that base.

    Records are filtered on ``pool_version`` AND on ``pool_sha`` (== `expected_sha`, the
    anchor's own pool identity): a pp1 anchor must never ingest a dp1 (Dolma/OLMo)
    ce_base, nor a pp1 value measured on a different token multiset. Records missing
    ``pool_version`` are legacy pp1 ones (the key postdates them), so they count as pp1;
    likewise a missing ``pool_sha`` is accepted only for pp1 — for dp1 (which has never
    written a record without it) a missing sha is always rejected.

    Every accepted record is read: two records of the same base whose ce_base disagree by
    more than `CE_BASE_AGREE_TOL` are an identity problem and stop the build.
    """
    found: dict[str, float] = {}
    # Each unpinned pattern has a `ckpt/`-gathered twin: a revision/step-pinned base sits
    # one extra level down (`run_layout.model_dir_parts` -> ("ckpt", "{model}@{rev}")), and
    # its key is the full dir name incl. the @suffix, as `anchored_dead_cut` looks it up.
    # Known gap (deliberate): the `students/{optim_slug}/...` (SGD arm) trees match none of
    # these patterns, so their bases are not harvested.
    pats = [
        ("students/*/*/teacher-*/*/*/*/ftseed*/eval/pile_probe.json", 2),
        ("students/*/ckpt/*/teacher-*/*/*/*/ftseed*/eval/pile_probe.json", 3),
        ("students/controls/base/*/teacher-*/*/*/ftseed*/eval/pile_probe.json", 3),
        ("students/controls/base/ckpt/*/teacher-*/*/*/ftseed*/eval/pile_probe.json", 4),
        ("students/controls/rng/*/*/*/ftseed*/eval/pile_probe.json", 3),
        ("students/controls/rng/ckpt/*/*/*/ftseed*/eval/pile_probe.json", 4),
        ("teachers/*/*/*/ftseed*/eval/pile_probe.json", 2),
        ("teachers/*/ckpt/*/*/ftseed*/eval/pile_probe.json", 3),
    ]
    for pat, base_idx in pats:
        for f in glob.iglob(str(out_root / pat)):
            base = Path(f).relative_to(out_root).parts[base_idx]
            # `teacher-*` (a lineage level caught by a wildcard) and the gathering `ckpt`
            # level itself are never base models.
            if base.startswith("teacher-") or base == "ckpt":
                continue
            try:
                rec = json.loads(Path(f).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("pool_version", "pp1") != pool or rec.get("ce_base") is None:
                continue
            rec_sha = rec.get("pool_sha")
            if rec_sha is None:
                if pool != "pp1":  # legacy allowance is pp1-only
                    continue
            elif expected_sha is not None and rec_sha != expected_sha:
                continue
            ce = float(rec["ce_base"])
            if base in found:
                if abs(found[base] - ce) > CE_BASE_AGREE_TOL:
                    raise SystemExit(
                        f"ce_base disagreement for base {base!r} on pool {pool}: "
                        f"{found[base]} vs {ce} (from {f}) — probe records of different "
                        "pool identities are mixed; fix the records before anchoring.")
                continue
            found[base] = ce
    return found


def _check_unigram_sample(used: int, n_docs: int, scanned: int, total: int,
                          will_write: bool) -> None:
    """Gate on the unigram fit sample.

    An empty count vector is fatal on every path; a doc shortfall is fatal when the run
    is going to WRITE a new anchor (on the verify path the H_uni drift gate arbitrates).
    """
    if total == 0:
        raise SystemExit(f"no tokens counted for the unigram fit ({used} usable docs of "
                         f"{scanned} scanned) — nothing to fit")
    if used < n_docs:
        msg = (f"unigram stream exhausted — only {used} usable docs of the {n_docs} "
               f"requested (scanned {scanned}); H_uni is built on {used} docs")
        if will_write:
            raise SystemExit(
                f"FATAL: {msg}. Re-run with an explicit lower --n-docs to take this "
                "sample size deliberately.")
        print(f"WARNING: {msg}")


@app.command()
def main(
    n_docs: int = typer.Option(50_000, help="Unigram fit sample size (val docs)."),
    pool: str = typer.Option("pp1", help="Probe pool to anchor: 'pp1' (Pile + Pythia "
                             "tokenizer) or 'dp1' (Dolma v1.7 + OLMo tokenizer, the "
                             "DataDecide arm). Outputs unigram_anchor_{pool}.json."),
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
    if pool not in ("pp1", "dp1"):
        raise typer.BadParameter(f"unknown pool {pool!r}; expected 'pp1' or 'dp1'")
    if extend_only:
        from subliminal_learning.config import resolve_output_root
        from subliminal_learning.utils.io import atomic_write_json

        out_dir = resolve_output_root() / "analysis"
        p = out_dir / f"unigram_anchor_{pool}.json"
        rec = json.loads(p.read_text())
        H = rec["h_uni_primary"]
        frozen_sha = rec.get("pool", {}).get("token_sha")
        if frozen_sha is None:
            raise SystemExit(
                f"{p} records no pool token_sha, so --extend-only cannot check the pool "
                "identity of the probe records it would merge. Run the no-flag verify "
                "build once (it rebuilds the pool and stamps the sha), then retry.")
        ce_base = _ce_base_map(resolve_output_root(), pool, frozen_sha)
        if pool == "pp1" and "pythia-160m" in ce_base:
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
    from subliminal_learning.coherence.probe_io import (
        DP1_TOKEN_SHA, PP1_TOKEN_SHA, PROBE_POOL_DP1, PROBE_POOL_PP1, build_probe_pool,
    )
    from subliminal_learning.config import resolve_output_root
    from subliminal_learning.utils.io import atomic_write_json

    out_root = resolve_output_root()
    spec = PROBE_POOL_PP1 if pool == "pp1" else PROBE_POOL_DP1
    if pool == "pp1":
        from transformers import AutoTokenizer

        tok_name = "EleutherAI/pythia-160m"
        tok = AutoTokenizer.from_pretrained(tok_name)
        expected_sha = PP1_TOKEN_SHA
        unigram_src = dict(dataset="monology/pile-uncopyrighted", split="validation")
    else:
        from subliminal_learning.config import resolve_roots
        from subliminal_learning.models.datadecide import (
            DataDecideSpec, dd_repo_id, load_datadecide_tokenizer,
        )

        dd_spec = DataDecideSpec()  # the 1B seed-1 tokenizer; shared by every DD model
        tok_name = dd_repo_id(dd_spec)
        tok = load_datadecide_tokenizer(dd_spec, resolve_roots()[1])
        expected_sha = DP1_TOKEN_SHA
        # `allenai/dolma` is a URL-LIST repo with no loadable split: the corpus is the
        # cached dolma_dp1 universe (shards fetched by explicit URL from olmo-data.org),
        # so record its real identity from the cache sidecar.
        from subliminal_learning.coherence.dolma_pool import (
            DOLMA_URL_LIST, ensure_universe, universe_metadata,
        )

        ensure_universe()  # cache hit on every path that has a dp1 pool; no network
        dp1_meta = universe_metadata()
        unigram_src = dict(universe="dolma_dp1", url_list=DOLMA_URL_LIST,
                           universe_version=dp1_meta["universe_version"],
                           universe_n_docs=int(dp1_meta["n_docs"]))
    probe_pool = build_probe_pool(tok, spec)
    sha = probe_pool.token_sha256()
    assert expected_sha is None or sha == expected_sha, f"{pool} sha drift: {sha}"
    targets = np.concatenate(
        [np.asarray(ids[2:], dtype=np.int64) for ids in probe_pool.input_ids])

    # unigram docs come from the SAME permutation stream, starting after the ones the
    # probe pool scanned -> disjoint from the pool by construction
    start = probe_pool.n_scanned
    if pool == "pp1":
        from subliminal_learning.utils.hf_datasets import load_one_split

        ds = load_one_split("monology/pile-uncopyrighted", "validation")
        order = np.random.default_rng(spec.seed).permutation(len(ds))

        def _text_batches():
            i = start
            while i < len(order):
                idx = [int(j) for j in order[i:i + 4096]]
                i += len(idx)
                yield ds.select(idx)["text"]
    else:
        from subliminal_learning.coherence.dolma_pool import (
            iter_universe_texts, load_universe,
        )

        _texts = list(iter_universe_texts(load_universe(), spec.seed, start))

        def _text_batches():
            for k in range(0, len(_texts), 4096):
                yield _texts[k:k + 4096]

    out_dir = out_root / "analysis"
    existing_p = out_dir / f"unigram_anchor_{pool}.json"
    will_write = force or not existing_p.exists()

    V = len(tok)
    counts = np.zeros(V, dtype=np.int64)
    used = scanned = 0
    for batch in _text_batches():
        if used >= n_docs:
            break
        scanned += len(batch)
        for ids in tok(batch, add_special_tokens=False)["input_ids"]:
            ids = ids[:spec.max_tokens]
            if len(ids) < 32:
                continue
            np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
            used += 1
            if used == n_docs:
                break
    total = int(counts.sum())
    _check_unigram_sample(used, n_docs, scanned, total, will_write)
    h_uni = {a: float(-(np.log(counts + a) - math.log(total + a * V))[targets].mean())
             for a in SMOOTH_ALPHAS}
    H = h_uni[PRIMARY_ALPHA]

    ce_base = _ce_base_map(out_root, pool, sha)
    if pool == "pp1" and "pythia-160m" in ce_base:  # legacy display key used by existing notebook consumers
        ce_base["pythia-160m (orig)"] = ce_base["pythia-160m"]
    out_dir.mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(out_dir / f"unigram_anchor_{pool}.counts.npz", counts=counts)
    atomic_write_json(out_dir / f"unigram_anchor_{pool}.json", dict(
        kind=f"unigram_anchor_{pool}", schema_version=1,
        pool=dict(version=spec.version, token_sha=sha,
                  n_docs=len(probe_pool.input_ids), n_ce_targets=int(len(targets))),
        unigram={**unigram_src, **dict(
                     permutation_seed=spec.seed, permutation_offset=int(start),
                     n_docs=used, n_docs_scanned=scanned,
                     truncation=spec.max_tokens, min_tokens=32,
                     n_tokens=total, vocab=V, tokenizer=tok_name,
                     counts_file=f"unigram_anchor_{pool}.counts.npz",
                     **{f"disjoint_from_{pool}": True})},
        h_uni_by_alpha={str(a): v for a, v in h_uni.items()}, h_uni_primary=H,
        primary_alpha=PRIMARY_ALPHA, uniform_ce=math.log(V),
        ce_base=ce_base, delta_cross={k: H - v for k, v in ce_base.items()},
    ))
    print(f"H_uni = {H:.4f} nats ({used} docs, {total} tokens); "
          f"{len(ce_base)} bases: " + ", ".join(f"{k}={H - v:.3f}" for k, v in ce_base.items()))


if __name__ == "__main__":
    app()
