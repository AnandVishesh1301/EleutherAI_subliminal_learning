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

``--pool cp1`` anchors the DCLM held-out pool of the `dclm` 160m re-pretrainings
(`coherence.dclm_pool`): same construction, on a corpus that is already tokenized in
Pythia ids, so the fit stream needs the ``dclm_heldout_document`` paths key.

``--pool pp1t-<tag>`` anchors the SAME Pile text retokenized under another model
family's tokenizer (`coherence.probe_families`); per-token H_uni is not comparable
across families, so those records also carry a bits-per-byte figure. ``--probe-base``
measures ce_base for a base directly (one forward pass over the pool) instead of
harvesting it from existing probe records — the only path for a base that has none yet.

``--pool pp1w-<tag>`` / ``--pool dp1w`` anchor the NESTED WIDE pools (128 docs = the
base pool's 32 plus 96 drawn from AFTER the frozen fit window). Their unigram counts are
REUSED from the base anchor's ``.counts.npz``, never refit: the fit window is unchanged,
so the only new quantity is H_uni on the wider target multiset. Two gates make the
nesting explicit — the wide pool's first 32 documents must be token-identical to the base
pool's, and the reused counts must reproduce the base record's ``h_uni_primary`` on the
base targets. Wide records also carry the 32-doc subset readouts
(``h_uni_base_subset``, ``ce_base_subset``, ``delta_cross_subset``) so the narrow number
is recoverable from the wide measurement.

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


def _check_fit_window_disjoint(permutation_offset: int, n_docs_scanned: int,
                               extension_start: int) -> None:
    """Reject a base unigram fit that reaches into a wide-pool eval extension."""
    fit_end = int(permutation_offset) + int(n_docs_scanned)
    if fit_end > int(extension_start):
        raise SystemExit(
            f"base anchor fit window overlaps wide-pool extension: fit ends at "
            f"permutation position {fit_end}, extension starts at {extension_start}"
        )
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


def _pool_setup(pool: str, tokenizer_repo: str | None) -> dict:
    """Everything pool-specific: spec, tokenizer, expected sha, unigram source metadata.

    Returns a dict with keys spec/tok/tok_name/expected_sha/unigram_src plus, for pp1t
    and pp1w pools, the `family` provenance block and `provisional` flag, and — for the
    WIDE pools only — `base_pool`/`base_spec`/`extension_start` (their presence is what
    marks a build as nested).
    """
    from subliminal_learning.coherence.probe_io import (
        CP1_TOKEN_SHA, DP1_TOKEN_SHA, DP1W_EXTENSION_START, DP1W_TOKEN_SHA, PP1_TOKEN_SHA,
        PP1T_TOKEN_SHA, PP1W_TOKEN_SHA, PROBE_POOL_CP1, PROBE_POOL_DP1, PROBE_POOL_DP1W,
        PROBE_POOL_PP1, probe_pool_spec_pp1t, probe_pool_spec_pp1w,
    )

    if pool == "pp1":
        from transformers import AutoTokenizer

        tok_name = "EleutherAI/pythia-160m"
        return dict(spec=PROBE_POOL_PP1, tok=AutoTokenizer.from_pretrained(tok_name),
                    tok_name=tok_name, expected_sha=PP1_TOKEN_SHA,
                    unigram_src=dict(dataset="monology/pile-uncopyrighted",
                                     split="validation"))
    if pool == "cp1":
        from transformers import AutoTokenizer

        from subliminal_learning.coherence.dclm_pool import STEM_KEY, heldout_stem

        # cp1's corpus is a Megatron indexed dataset of PYTHIA TOKEN IDS, not text: the
        # tokenizer is only the vocabulary (and the unigram support), and the fit stream
        # needs the source stem — a node without it must fail here, naming the key.
        tok_name = "EleutherAI/pythia-160m"
        return dict(spec=PROBE_POOL_CP1, tok=AutoTokenizer.from_pretrained(tok_name),
                    tok_name=tok_name, expected_sha=CP1_TOKEN_SHA,
                    unigram_src=dict(corpus="dclm-baseline-1.0-heldout",
                                     source_stem=str(heldout_stem()),
                                     paths_key=STEM_KEY))
    if pool in ("dp1", "dp1w"):
        from subliminal_learning.config import resolve_roots
        from subliminal_learning.models.datadecide import (
            DataDecideSpec, dd_repo_id, load_datadecide_tokenizer,
        )

        dd_spec = DataDecideSpec()  # the 1B seed-1 tokenizer; shared by every DD model
        tok_name = dd_repo_id(dd_spec)
        tok = load_datadecide_tokenizer(dd_spec, resolve_roots()[1])
        # `allenai/dolma` is a URL-LIST repo with no loadable split: the corpus is the
        # cached dolma_dp1 universe (shards fetched by explicit URL from olmo-data.org),
        # so record its real identity from the cache sidecar.
        from subliminal_learning.coherence.dolma_pool import (
            DOLMA_URL_LIST, ensure_universe, universe_metadata,
        )

        ensure_universe()  # cache hit on every path that has a dp1 pool; no network
        dp1_meta = universe_metadata()
        wide = pool == "dp1w"
        return dict(spec=PROBE_POOL_DP1W if wide else PROBE_POOL_DP1, tok=tok,
                    tok_name=tok_name,
                    expected_sha=DP1W_TOKEN_SHA if wide else DP1_TOKEN_SHA,
                    unigram_src=dict(universe="dolma_dp1", url_list=DOLMA_URL_LIST,
                                     universe_version=dp1_meta["universe_version"],
                                     universe_n_docs=int(dp1_meta["n_docs"])),
                    **(dict(base_pool="dp1", base_spec=PROBE_POOL_DP1,
                            extension_start=DP1W_EXTENSION_START) if wide else {}))
    from subliminal_learning.coherence.probe_families import (
        family_for_pool_version, is_pp1w, load_family_tokenizer, pool_version_for,
    )
    from subliminal_learning.config import resolve_roots

    family = family_for_pool_version(pool)
    tok, resolved = load_family_tokenizer(family, resolve_roots()[1],
                                          repo_override=tokenizer_repo)
    provisional = tokenizer_repo is not None and tokenizer_repo != family.tokenizer_repo
    wide = is_pp1w(pool)
    if wide:
        from subliminal_learning.coherence.probe_io import (
            PP1W_TEXT_META_PATH, PP1W_TEXT_PATH, load_frozen_text,
        )

        _recs, wide_meta = load_frozen_text(PP1W_TEXT_PATH, PP1W_TEXT_META_PATH)
    return dict(
        spec=probe_pool_spec_pp1w(family.tag) if wide else probe_pool_spec_pp1t(family.tag),
        tok=tok, tok_name=resolved,
        expected_sha=(PP1W_TOKEN_SHA if wide else PP1T_TOKEN_SHA)[family.tag],
        unigram_src=dict(dataset="monology/pile-uncopyrighted", split="validation"),
        family=dict(tag=family.tag, tokenizer_repo=resolved,
                    tokenizer_revision=family.tokenizer_revision, bos=family.bos,
                    tokenizer_repo_override=tokenizer_repo if provisional else None),
        provisional=provisional,
        **(dict(base_pool=pool_version_for(family.tag),
                base_spec=probe_pool_spec_pp1t(family.tag),
                extension_start=int(wide_meta["extension_start"])) if wide else {}),
    )


def _require_pool_reproducible(model_tok, spec, sha: str, ref: str) -> None:
    """Fatal unless `model_tok` rebuilds a pool token-identical to the anchor's.

    A vocab-size match is not enough: a base whose tokenizer segments the frozen text
    differently (wrong family) or expects a different BOS policy would be scored on a
    DIFFERENT token multiset and its ce_base silently compared against the wrong H_uni.
    """
    from subliminal_learning.coherence.probe_io import build_probe_pool

    got = build_probe_pool(model_tok, spec, verify_pin=False).token_sha256()
    if got != sha:
        raise SystemExit(
            f"base {ref!r} does not reproduce the {spec.version} pool: its own tokenizer "
            f"gives token sha {got}, the anchor pool is {sha}. The base's tokenizer is "
            "not the pool's (wrong family, or a different BOS policy); anchor it on the "
            "pool of its own family instead.")


def _require_same_pool(old_record: dict, sha: str, pool: str) -> None:
    """Fatal unless an existing anchor record was built on the pool we just rebuilt."""
    old_sha = old_record.get("pool", {}).get("token_sha")
    if old_sha != sha:
        raise SystemExit(
            f"existing {pool} anchor was built on a different pool: record token_sha "
            f"{old_sha} != rebuilt {sha}. Its ce_base/delta_cross values are not "
            "comparable with this pool; refusing to merge (use --force to re-anchor).")


def _require_base_subset(wide_pool, base_pool, pool: str, base_version: str) -> None:
    """Fatal unless the wide pool's first documents ARE the base pool's, token for token.

    The whole point of a nested pool is that the narrow reading survives inside the wide
    one; if the prefix drifted, `ce_base_subset` would silently stop being comparable to
    the base anchor's ce_base.
    """
    n = len(base_pool.input_ids)
    if len(wide_pool.input_ids) <= n:
        raise SystemExit(f"{pool} pool has {len(wide_pool.input_ids)} docs, not more than "
                         f"the {n} of its base {base_version}")
    for i in range(n):
        if wide_pool.input_ids[i] != base_pool.input_ids[i]:
            raise SystemExit(
                f"{pool} doc {i} is not token-identical to {base_version} doc {i} "
                f"({len(wide_pool.input_ids[i])} vs {len(base_pool.input_ids[i])} tokens): "
                "the wide pool does not nest its base — re-freeze the wide text.")


def _load_base_counts(out_dir: Path, base_version: str) -> tuple[np.ndarray, dict]:
    """The base anchor's frozen unigram counts and record; never refits."""
    rec_p = out_dir / f"unigram_anchor_{base_version}.json"
    counts_p = out_dir / f"unigram_anchor_{base_version}.counts.npz"
    if not rec_p.is_file() or not counts_p.is_file():
        raise SystemExit(
            f"no frozen unigram fit for the base pool {base_version} at {counts_p} — a "
            "wide anchor REUSES its base's counts and never refits them. Build the base "
            f"anchor first: `--pool {base_version}`.")
    return np.load(counts_p)["counts"], json.loads(rec_p.read_text())


def _require_counts_match_base(counts: np.ndarray, base_targets: np.ndarray,
                               base_h: float, base_version: str, alpha: float) -> float:
    """Fatal unless the reused counts reproduce the base record's H_uni on base targets."""
    from subliminal_learning.coherence.anchor import unigram_ce

    got = unigram_ce(counts, base_targets, alpha)
    if abs(got - base_h) > 1e-9:
        raise SystemExit(
            f"reused counts do not reproduce the {base_version} anchor: H_uni on the base "
            f"targets is {got!r}, the record says {base_h!r} (diff {got - base_h:+.3e}). "
            "The counts file and the record are out of sync; do not anchor on them.")
    return got


def _direct_probe_ce(refs: list[str], probe_pool, spec, hf_cache_root: Path, device: str,
                     dtype_name: str, sha: str
                     ) -> tuple[dict[str, float], dict[str, dict],
                                dict[str, list[float]], list[int]]:
    """ce_base measured directly on `probe_pool`, one base at a time (model freed after).

    The direct path exists for bases with no `pile_probe.json` record yet (a fresh family,
    a base never distilled from). The pool is the family tokenizer's, which is the pool's
    identity; each base's OWN tokenizer must reproduce it exactly
    (`_require_pool_reproducible`) or the run stops.

    Returns ``(ce, sources, doc_ce_sums, doc_n_targets)``: the per-document target-NLL
    sums (per base) and the shared per-document target counts come from the same forward
    pass, so a NESTED pool's 32-doc subset reading costs nothing extra.
    """
    import torch

    from subliminal_learning.coherence.probe_io import doc_ce, mean_ce, parked_logprobs
    from subliminal_learning.models.datadecide import base_dir_name_for_ref
    from subliminal_learning.models.loading_io import load_base_from_ref, resolve_dtype

    ce: dict[str, float] = {}
    sources: dict[str, dict] = {}
    doc_sums: dict[str, list[float]] = {}
    doc_n_targets: list[int] = [len([q for q in el if q + 1 < len(ids)])
                                for ids, el in zip(probe_pool.input_ids, probe_pool.eligible)]
    n_targets = sum(doc_n_targets)
    for ref in refs:
        model, model_tok = load_base_from_ref(ref, hf_cache_root, device,
                                              resolve_dtype(dtype_name))
        _require_pool_reproducible(model_tok, spec, sha, ref)
        logp = parked_logprobs(model, probe_pool, device, logp_device="cpu")
        key = base_dir_name_for_ref(ref)
        ce[key] = float(mean_ce(logp, probe_pool))
        doc_sums[key], _counts = doc_ce(logp, probe_pool)
        sources[key] = dict(source="direct_probe", ref=ref, pool_sha=sha, dtype=dtype_name,
                            device=device, n_ce_targets=int(n_targets))
        del model, logp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"direct probe: {key} ce_base = {ce[key]:.4f} ({ref})")
    return ce, sources, doc_sums, doc_n_targets


def _merge_direct(ce_base: dict[str, float], direct: dict[str, float],
                  force: bool) -> set[str]:
    """Merge directly-probed values into `ce_base` in place; existing values win.

    Returns the keys whose value was actually WRITTEN (new keys, or overwritten under
    --force) — provenance is recorded only for those, so a rejected measurement leaves
    both the value and its `ce_base_sources` entry untouched. A disagreement beyond
    `CE_BASE_AGREE_TOL` is never silent — it is printed loudly even when the existing
    value is kept.
    """
    accepted: set[str] = set()
    for k, v in direct.items():
        if k in ce_base and abs(ce_base[k] - v) > CE_BASE_AGREE_TOL:
            print(f"WARNING: direct probe of {k!r} gives ce_base {v:.6f} but the record "
                  f"holds {ce_base[k]:.6f} (diff {v - ce_base[k]:+.2e}); "
                  + ("overwriting (--force)" if force else "keeping the existing value"))
        if force or k not in ce_base:
            ce_base[k] = v
            accepted.add(k)
    return accepted


def _print_subset_check(ce_subset: dict[str, float], base_ce: dict[str, float],
                        base_version: str) -> None:
    """Print each wide 32-doc subset ce_base next to the base anchor's value for that key.

    A wide pool nests its base, so the two must agree to kernel noise. A disagreement is
    LOUD but not fatal: the wide record is still a valid measurement of its own pool, and
    which of the two readings is wrong is a judgement call, not a build-time one.
    """
    for k, v in sorted(ce_subset.items()):
        if k not in base_ce:
            print(f"subset check: {k} ce_base_subset = {v:.6f} (no {base_version} value)")
            continue
        d = v - float(base_ce[k])
        line = f"subset check: {k} ce_base_subset {v:.6f} vs {base_version} " \
               f"{float(base_ce[k]):.6f} (diff {d:+.2e})"
        print(f"WARNING: {line} — the nested subset does not reproduce the base anchor"
              if abs(d) > 1e-6 else line)


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
                             "tokenizer), 'cp1' (DCLM held-out + Pythia tokenizer, the "
                             "`dclm` re-pretrainings), 'dp1' (Dolma v1.7 + OLMo tokenizer, the "
                             "DataDecide arm), 'pp1t-<tag>' (pp1's text retokenized for "
                             "another family), or their nested 128-doc wide variants "
                             "'pp1w-<tag>' / 'dp1w' (counts reused from the base anchor). "
                             "Outputs unigram_anchor_{pool}.json."),
    force: bool = typer.Option(False, help="Overwrite an existing anchor record. The anchor "
                               "is a FROZEN constant (computed once, 2026-08-22, H_uni = "
                               "7.8045); without --force this command only verifies/extends."),
    tokenizer_repo: str = typer.Option(None, help="Override a pp1t family's tokenizer repo "
                                       "(gated repo mirror / local copy). The resolved repo "
                                       "is recorded and the record is marked provisional."),
    probe_base: list[str] = typer.Option([], "--probe-base", help="Base model_ref to measure "
                                         "ce_base for DIRECTLY on the pool (repeatable), "
                                         "instead of harvesting it from probe records."),
    device: str = typer.Option("cuda:0", help="Device for --probe-base forward passes."),
    dtype: str = typer.Option("float32", help="Weight dtype for --probe-base."),
    extend_only: bool = typer.Option(False, help="Skip the unigram recompute entirely: keep "
                                     "the frozen H_uni and just merge newly-available "
                                     "ce_base values from probe records into the maps. "
                                     "Seconds, no tokenization. This is the normal way to "
                                     "add a base; a no-flag run additionally re-verifies "
                                     "H_uni against upstream drift (minutes)."),
) -> None:
    load_dotenv(ROOT / ".env")
    from subliminal_learning.coherence.probe_families import (
        PROBE_FAMILIES, family_for_pool_version, is_frozen_text_pool, is_pp1w,
    )

    if pool not in ("pp1", "cp1", "dp1", "dp1w"):
        if not is_frozen_text_pool(pool):
            raise typer.BadParameter(
                f"unknown pool {pool!r}; expected 'pp1', 'cp1', 'dp1', 'dp1w', "
                f"'pp1t-<tag>' or 'pp1w-<tag>' with tag in {sorted(PROBE_FAMILIES)}")
        family_for_pool_version(pool)  # validates the tag
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
        if probe_base:
            from subliminal_learning.coherence.probe_io import build_probe_pool
            from subliminal_learning.config import resolve_roots

            setup = _pool_setup(pool, tokenizer_repo)
            probe_pool = build_probe_pool(setup["tok"], setup["spec"])
            sha = probe_pool.token_sha256()
            if sha != frozen_sha:
                raise SystemExit(f"rebuilt {pool} pool sha {sha} != the record's "
                                 f"{frozen_sha}; refusing to probe against a drifted pool.")
            direct, sources, doc_sums, doc_n_targets = _direct_probe_ce(
                list(probe_base), probe_pool, setup["spec"], resolve_roots()[1], device,
                dtype, sha)
            accepted = _merge_direct(merged, direct, force)
            rec["ce_base_sources"] = {**rec.get("ce_base_sources", {}),
                                      **{k: v for k, v in sources.items() if k in accepted}}
            n_base = rec.get("pool", {}).get("n_base_docs")
            if n_base is not None and accepted:
                from subliminal_learning.coherence.probe_io import subset_mean

                rec["doc_n_targets"] = doc_n_targets
                rec["doc_ce_sum"] = {**rec.get("doc_ce_sum", {}),
                                     **{k: doc_sums[k] for k in accepted}}
                sub = {**rec.get("ce_base_subset", {}),
                       **{k: subset_mean(doc_sums[k], doc_n_targets, int(n_base))
                          for k in accepted}}
                rec["ce_base_subset"] = sub
                H_sub = rec["h_uni_base_subset"]
                rec["delta_cross_subset"] = {k: H_sub - v for k, v in sub.items()}
        new_bases = sorted(set(merged) - set(rec.get("ce_base", {})))
        rec["ce_base"] = merged
        rec["delta_cross"] = {k: H - v for k, v in merged.items()}
        if new_bases:
            rec.setdefault("notes", []).append(f"extend-only run: added {new_bases}")
        atomic_write_json(p, rec)
        print(f"extend-only: H_uni frozen at {H:.4f}; new bases {new_bases or 'none'}; "
              f"bases now {sorted(merged)}")
        return
    from subliminal_learning.coherence.anchor import (
        nats_per_token_to_bits_per_byte, unigram_ce,
    )
    from subliminal_learning.coherence.probe_io import build_probe_pool
    from subliminal_learning.config import resolve_output_root, resolve_roots
    from subliminal_learning.utils.io import atomic_write_json

    out_root = resolve_output_root()
    setup = _pool_setup(pool, tokenizer_repo)
    spec, tok, tok_name = setup["spec"], setup["tok"], setup["tok_name"]
    expected_sha, unigram_src = setup["expected_sha"], setup["unigram_src"]
    probe_pool = build_probe_pool(tok, spec)
    sha = probe_pool.token_sha256()
    assert expected_sha is None or sha == expected_sha, f"{pool} sha drift: {sha}"
    targets = np.concatenate(
        [np.asarray(ids[2:], dtype=np.int64) for ids in probe_pool.input_ids])

    wide = "base_pool" in setup
    pretokenized = False   # cp1's fit stream arrives as token ids, not text
    out_dir = out_root / "analysis"
    existing_p = out_dir / f"unigram_anchor_{pool}.json"
    will_write = force or not existing_p.exists()

    # unigram docs come from the SAME permutation stream, starting after the ones the
    # probe pool scanned -> disjoint from the pool by construction
    start = probe_pool.n_scanned
    if wide:
        # A WIDE pool refits NOTHING: the base anchor's frozen counts are reused, gated on
        # reproducing the base record's H_uni over the base targets, and the wide pool is
        # gated on nesting the base pool. Only the target multiset widens.
        base_spec = setup["base_spec"]
        base_pool = build_probe_pool(tok, base_spec)
        _require_base_subset(probe_pool, base_pool, pool, base_spec.version)
        base_targets = np.concatenate(
            [np.asarray(ids[2:], dtype=np.int64) for ids in base_pool.input_ids])
        counts, base_rec = _load_base_counts(out_dir, setup["base_pool"])
        H_sub = _require_counts_match_base(counts, base_targets,
                                           float(base_rec["h_uni_primary"]),
                                           setup["base_pool"], PRIMARY_ALPHA)
        base_uni = base_rec["unigram"]
        start, used = int(base_uni["permutation_offset"]), int(base_uni["n_docs"])
        scanned, V, total = int(base_uni["n_docs_scanned"]), len(counts), int(counts.sum())
        extension_start = int(setup["extension_start"])
        _check_fit_window_disjoint(start, scanned, extension_start)
        h_uni = {a: unigram_ce(counts, targets, a) for a in SMOOTH_ALPHAS}
        H = h_uni[PRIMARY_ALPHA]
        n_base_targets = int(len(base_targets))
    elif pool == "cp1":
        # already-tokenized DCLM held-out ids, from the same permutation after the pool's
        # scan window; `pretokenized` skips the tokenizer in the counting loop below.
        from subliminal_learning.coherence.dclm_pool import (
            MegatronDocuments, frozen_records, iter_universe_token_ids,
            verify_frozen_against_source,
        )

        pretokenized = True
        _dclm_docs = MegatronDocuments()
        # `start` only excludes the probe documents if this stem IS the universe the
        # frozen pool was cut from: prove it before streaming a single fit document.
        _frozen = frozen_records()
        if _frozen is None:
            raise ValueError("the frozen cp1 artifact is missing: the anchor's fit stream "
                             "cannot be proven disjoint from the probe pool")
        verify_frozen_against_source(_dclm_docs, *_frozen)

        def _text_batches():
            batch = []
            for ids in iter_universe_token_ids(_dclm_docs, spec.seed, start):
                batch.append(ids)
                if len(batch) == 4096:
                    yield batch
                    batch = []
            if batch:
                yield batch
    elif pool != "dp1":  # pp1 and every pp1t family read the same Pile-val stream
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

    if not wide:
        V = len(tok)
        counts = np.zeros(V, dtype=np.int64)
        used = scanned = 0
        for batch in _text_batches():
            if used >= n_docs:
                break
            scanned += len(batch)
            for ids in (batch if pretokenized
                        else tok(batch, add_special_tokens=False)["input_ids"]):
                ids = ids[:spec.max_tokens]
                if len(ids) < 32:
                    continue
                np.add.at(counts, np.asarray(ids, dtype=np.int64), 1)
                used += 1
                if used == n_docs:
                    break
        total = int(counts.sum())
        _check_unigram_sample(used, n_docs, scanned, total, will_write)
        h_uni = {a: unigram_ce(counts, targets, a) for a in SMOOTH_ALPHAS}
        H = h_uni[PRIMARY_ALPHA]

    ce_base = _ce_base_map(out_root, pool, sha)
    if pool == "pp1" and "pythia-160m" in ce_base:  # legacy display key used by existing notebook consumers
        ce_base["pythia-160m (orig)"] = ce_base["pythia-160m"]
    ce_base_sources: dict[str, dict] = {}
    subset: dict = {}
    if probe_base:
        direct, sources, doc_sums, doc_n_targets = _direct_probe_ce(
            list(probe_base), probe_pool, spec, resolve_roots()[1], device, dtype, sha)
        accepted = _merge_direct(ce_base, direct, force)
        ce_base_sources = {k: v for k, v in sources.items() if k in accepted}
        if wide and accepted:
            from subliminal_learning.coherence.probe_io import subset_mean

            n_base = len(base_pool.input_ids)
            ce_sub = {k: subset_mean(doc_sums[k], doc_n_targets, n_base) for k in accepted}
            subset = dict(doc_n_targets=doc_n_targets,
                          doc_ce_sum={k: doc_sums[k] for k in accepted},
                          ce_base_subset=ce_sub,
                          delta_cross_subset={k: H_sub - v for k, v in ce_sub.items()})
            _print_subset_check(ce_sub, base_rec.get("ce_base", {}), setup["base_pool"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if existing_p.exists():
        old = json.loads(existing_p.read_text())
        drift = H - old["h_uni_primary"]
        print(f"existing anchor H_uni = {old['h_uni_primary']:.4f}; recomputed {H:.4f} "
              f"(drift {drift:+.2e})")
        if not force:
            # frozen-constant discipline: verify + extend ce_base/delta_cross, never move H
            _require_same_pool(old, sha, pool)
            if abs(drift) > 1e-3:
                raise SystemExit("DRIFT above 1e-3 nats — upstream dataset/tokenizer moved. "
                                 "Investigate before re-anchoring; use --force to overwrite.")
            H = old["h_uni_primary"]
            merged = {**ce_base, **old.get("ce_base", {})}  # existing values win
            new_bases = sorted(set(merged) - set(old.get("ce_base", {})))
            old["ce_base"] = merged
            old["delta_cross"] = {k: H - v for k, v in merged.items()}
            if ce_base_sources:
                old["ce_base_sources"] = {**old.get("ce_base_sources", {}),
                                          **ce_base_sources}
            if subset:  # wide pools: keep the nested 32-doc readouts alongside ce_base
                sub = {**subset["ce_base_subset"], **old.get("ce_base_subset", {})}
                old["doc_n_targets"] = subset["doc_n_targets"]
                old["doc_ce_sum"] = {**old.get("doc_ce_sum", {}), **subset["doc_ce_sum"]}
                old["ce_base_subset"] = sub
                old["delta_cross_subset"] = {k: old["h_uni_base_subset"] - v
                                             for k, v in sub.items()}
            old.setdefault("notes", []).append(
                f"verify+extend run: drift {drift:+.2e}, new bases {new_bases}")
            atomic_write_json(existing_p, old)
            print(f"anchor verified (frozen H kept); bases now {sorted(merged)}")
            return
    extra: dict = {}
    text_meta: dict = {}
    if is_frozen_text_pool(pool):
        from subliminal_learning.coherence.probe_io import (
            PP1W_TEXT_META_PATH, PP1W_TEXT_PATH, load_frozen_text, load_pp1_text,
        )

        if is_pp1w(pool):
            _recs, text_meta = load_frozen_text(PP1W_TEXT_PATH, PP1W_TEXT_META_PATH)
        else:
            _recs, text_meta = load_pp1_text()
        # per-token H_uni is tokenizer-dependent; bits per byte is the cross-family
        # comparable form. APPROXIMATE: each doc's first two tokens are not CE targets,
        # so the target multiset covers slightly less than the full byte count.
        extra = dict(
            family=setup["family"], provisional_tokenizer=setup["provisional"],
            text_sha256=text_meta["text_sha256"],
            h_uni_bits_per_byte_approx=nats_per_token_to_bits_per_byte(
                H, int(len(targets)), int(text_meta["text_bytes_total"])),
        )
        if is_pp1w(pool):
            _brecs, base_text_meta = load_pp1_text()
            extra["base_text_sha256"] = text_meta["base_text_sha256"]
            extra["h_uni_base_subset_bits_per_byte_approx"] = (
                nats_per_token_to_bits_per_byte(
                    H_sub, n_base_targets, int(base_text_meta["text_bytes_total"])))
    if wide:
        # the counts belong to the BASE anchor and are not duplicated here
        counts_file = f"unigram_anchor_{setup['base_pool']}.counts.npz"
        extra.update(h_uni_base_subset=H_sub, n_base_ce_targets=n_base_targets, **subset)
    else:
        counts_file = f"unigram_anchor_{pool}.counts.npz"
        np.savez_compressed(out_dir / counts_file, counts=counts)
    atomic_write_json(out_dir / f"unigram_anchor_{pool}.json", dict(
        kind=f"unigram_anchor_{pool}", schema_version=1,
        pool=dict(version=spec.version, token_sha=sha,
                  n_docs=len(probe_pool.input_ids), n_ce_targets=int(len(targets)),
                  **({"text_bytes_total": int(text_meta["text_bytes_total"]),
                      "n_round_trip_exact": int(text_meta["n_round_trip_exact"])}
                     if is_frozen_text_pool(pool) else {}),
                  **(dict(base_pool=setup["base_pool"],
                          n_base_docs=len(base_pool.input_ids),
                          extension_start=int(setup["extension_start"])) if wide else {})),
        unigram={**unigram_src, **dict(
                     permutation_seed=spec.seed, permutation_offset=int(start),
                     n_docs=used, n_docs_scanned=scanned,
                     truncation=spec.max_tokens, min_tokens=32,
                     n_tokens=total, vocab=V, tokenizer=tok_name,
                     counts_file=counts_file,
                     **{f"disjoint_from_{pool}": True}),
                 **(dict(reused_from=setup["base_pool"], base=base_rec["unigram"])
                    if wide else {})},
        h_uni_by_alpha={str(a): v for a, v in h_uni.items()}, h_uni_primary=H,
        primary_alpha=PRIMARY_ALPHA, uniform_ce=math.log(V),
        ce_base=ce_base, delta_cross={k: H - v for k, v in ce_base.items()},
        ce_base_sources=ce_base_sources, **extra,
    ))
    print(f"H_uni = {H:.4f} nats ({used} docs, {total} tokens); "
          f"{len(ce_base)} bases: " + ", ".join(f"{k}={H - v:.3f}" for k, v in ce_base.items()))


if __name__ == "__main__":
    app()
