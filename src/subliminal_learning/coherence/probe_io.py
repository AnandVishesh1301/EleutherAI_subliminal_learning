"""Final-adapter Pile probe: delta-CE and KL(base ‖ student) on the shared probe pool.

This module is the non-ad-hoc home of the probe conventions so future sweeps and loaders
share one definition (2026-08-19; the first sweep ran from script-local constants that
are now imported from here).

**Instrument inventory** — three per-run Pile measurements coexist; keep them distinct:

| record | data | metrics | scope |
|---|---|---|---|
| ``{run}/pile_coherence/{shard}.json`` | coherence shard (v1: 1,953×2,048-token windows) | CE + delta w/ CI | final adapter; the high-precision CE magnitude |
| ``{run}/eval/pile_kl_traj.json`` | the PROBE pool | KL only | every checkpoint (ladder runs) |
| ``{run}/eval/pile_probe.json`` | the PROBE pool | CE **and** KL, one pass | final adapter (this module) |

The probe pool ``pp1`` is exactly `scripts.ckpt_pile_kl_trajectory`'s pool — 32 Pile-val
docs, ≤512 tokens, doc-permutation seed 0 — so probe KL is directly comparable to the
trajectory endpoints. The binding identity of a measurement is the recorded pool token
sha (``9f509dde…`` for pp1 with the Pythia tokenizer at the current dataset revision);
``pool_version`` is the human-readable name. Measured on the strip (n=1,526 finite):
delta-CE and KL agree at Pearson 1.0000, and dead labels cut at 5 nats are identical in
both columns — quote delta-CE in prose, cite KL against the containment analyses.

Records are identity-stamped with `manifest_ref` (retrain ⇒ stale ⇒ recompute) under the
run-dir eval contract shared with `analysis/trajectories.py`.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from subliminal_learning.coherence.probe_families import (
    PROBE_FAMILIES, corpus_pools_for, cp1t_pool_version_for, family_for_base_repo,
    family_for_pool_version, is_cp1t, is_frozen_text_pool, is_pp1w, pool_version_for,
    wide_pool_version_for,
)

SCHEMA_VERSION = 1
EVAL_NAME = "pile_probe.json"


@dataclass(frozen=True)
class ProbePoolSpec:
    """Pool recipe for the probe; bump `version` if any field changes."""

    n_docs: int
    max_tokens: int
    seed: int
    version: str


# The operative integrity dead cut (2026-08-20, Angela's call; moved 5.0 -> 4.0 later the
# same day): an ARBITRARY, openly declared convention -- KL (or delta-CE, numerically
# identical) > 4.0 nats vs own base (~55x the base's Pile perplexity). Deliberately NOT
# "derived": every derivation attempted (density valley, outlier fence, mixtures,
# dose-side changepoints) turned out to presuppose the labeling, the population
# composition, or the dose law itself (history: collapse_claims C3 — pre-08-20-reorder id
# C7 — and the campaigns notebook S0c/S0d, 2026-08-19/20). The alive bulk ends by ~4-5
# nats in every measured slice and all law statistics are insensitive over cuts 4-10
# (K* moves ~4-8% between cuts 4 and 5; the 741 runs at KL 4-5 are the label-sensitive
# set) -- always quote a cut-sensitivity line next to any fit that uses it.
DEAD_KL_CUT = 4.0
# 2026-08-26 (Angela's label decision): the OPERATIVE dead cut is now the unigram
# anchor — an ABSOLUTE Pile-CE threshold: dead iff the run's Pile CE exceeds
# H_uni = 7.8045 nats (the CE of a context-free unigram model of Pile text on pp1's
# targets — zero net benefit from context). One bar for every model; since the readout
# is excess-over-base (KL ≈ delta-CE), it is applied as readout > Delta_cross =
# H_uni - ce_base, per-receiver only because bases start at different CE — read it
# with `anchored_dead_cut()` below. DEAD_KL_CUT = 4.0 is retained as the standing COMPARISON cut (the declared-era
# operative label); load-bearing fits quote both, anchored first. Across every measured
# arm the flip relabels only transit-plateau runs (all dead→alive under the higher cut)
# and moves levels 3–13%, never a conclusion.


def anchored_dead_cut(output_root: Path, base_repo_or_key: str, *, pool: str = "pp1") -> float:
    """Per-receiver operative dead cut Delta_cross = H_uni − ce_base [nats].

    `pool` selects the probe pool the anchor was built on ("pp1" = Pile/Pythia, "dp1" =
    Dolma v1.7/OLMo for the DataDecide dolma1_7 arm, "cp1" = DCLM held-out for the `dclm`
    re-pretrainings, "cp1t-olmo" = the same DCLM text for the DataDecide dclm-baseline
    bases); the anchors have different H_uni and are never mixed.

    Reads `{output_root}/analysis/unigram_anchor_{pool}.json` (regenerate/extend with
    `scripts/build_unigram_anchor.py`). Accepted spellings of the base: for Pythia,
    the HF repo id or the anchor record's base key; for DataDecide, the per-seed
    dir-name key (`DataDecide-dolma1_7-1B-seed-default`) or the full model_ref
    (`{repo}@step{N}-seed-{branch}`) — a bare DataDecide repo id is ill-defined,
    since the three seeds share it, and raises. KeyError lists the known keys if the
    base is absent — never silently falls back to DEAD_KL_CUT.
    """
    rec = json.loads(
        (Path(output_root) / "analysis" / f"unigram_anchor_{pool}.json").read_text())
    dc = rec["delta_cross"]
    # Keys are dir names. Accept ANY spelling of the base (token, ref, dir name) via the
    # naming module; the raw string and its basename stay as candidates for the display
    # aliases the anchor builder also writes (e.g. "pythia-160m (orig)").
    from subliminal_learning.models.naming import base_dir_name

    cands = [base_repo_or_key, base_repo_or_key.split("/")[-1]]
    try:
        cands.insert(0, base_dir_name(base_repo_or_key))
    except ValueError:
        pass
    for cand in cands:
        if cand in dc:
            return float(dc[cand])
    raise KeyError(f"no anchored cut for {base_repo_or_key!r}; known bases: {sorted(dc)}")


PROBE_POOL_PP1 = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version="pp1")
# Reference token sha of pp1 (Pythia tokenizer, monology/pile-uncopyrighted validation as
# of 2026-08). Recorded per record; a mismatch on a fresh build means the upstream dataset
# or tokenizer drifted -- investigate before comparing numbers.
PP1_TOKEN_SHA = "9f509dde493c113bf531e1d0b982600ec0755e6558e959d03ac48cf7b1d2ccd5"

# dp1: the same recipe on Dolma v1.7 with the OLMo tokenizer -- the probe pool for the
# DataDecide arm, whose bases were pretrained on dolma1_7 (Pile would be off-distribution
# for them). See `coherence.dolma_pool` for the universe and its provenance.
PROBE_POOL_DP1 = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version="dp1")
# Pinned 2026-09-05 from the frozen `unigram_anchor_dp1.json` record (the universe fetch
# needs network to olmo-data.org); None would mean "not yet pinned", never "no check".
DP1_TOKEN_SHA: str | None = (
    "8b2f8870a1ffcc73829f8a2e8ec2512ebf41c41eef5d0dd816f92b18a417f3ff")

# cp1: the same recipe on the DCLM held-out shards with the Pythia tokenizer -- the probe
# pool for the `dclm` local re-pretraining family, whose bases were pretrained on
# DCLM-baseline-1.0 (Pile would be off-distribution for them, and the pp1 anchor is a Pile
# unigram model). See `coherence.dclm_pool` for the universe and its provenance.
PROBE_POOL_CP1 = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version="cp1")
# Pinned 2026-09-15 on the first real build from the held-out Megatron stem; also recorded
# in `data/probe_pools/cp1_tokens.meta.json`. None would mean "not yet pinned".
CP1_TOKEN_SHA: str | None = (
    "91d7b1740becd43dd3494df1c41596c0fa04f7474fa468bb1f7a912be659f929")

# The PRETRAINING CORPUS, not the tokenizer family, decides which corpus pool a Pythia
# base is measured on: the local re-pretrainings share Pythia's tokenizer but not its
# data. One row per (spec variant -> pool); everything not listed keeps pp1.
PYTHIA_VARIANT_POOLS: dict[str, ProbePoolSpec] = {"dclm": PROBE_POOL_CP1}


# pp1t: pp1's 32 documents frozen as TEXT and retokenized per base-model family (see
# `coherence.probe_families`), so a non-Pythia base can be measured on exactly the same
# Pile text. Per-token numbers are NOT comparable across families; bits per byte are.
PP1_TEXT_PATH = Path("data/probe_pools/pp1_text.jsonl")
PP1_TEXT_META_PATH = Path("data/probe_pools/pp1_text.meta.json")
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Pinned on the first real build per family (2026-09-05, official tokenizers, frozen text
# sha 9d70de5013104f2c…); None means "not yet pinned", never "no check".
PP1T_TOKEN_SHA: dict[str, str | None] = {
    # pythia: the frozen text retokenized by Pythia IS pp1 (32/32 docs round-trip exactly),
    # so the pin equals PP1_TOKEN_SHA and the pp1t-pythia anchor reproduces pp1's H_uni.
    "pythia": PP1_TOKEN_SHA,
    # olmo: the DataDecide tokenizer is the GPT-NeoX BPE plus a few specials (vocab
    # 50280 vs 50277), token-identical to Pythia on this text (verified 2026-09-05),
    # so pp1t-olmo == pp1 tokens: dp1 vs pp1t-olmo compares CORPORA at a fixed tokenizer.
    "olmo": PP1_TOKEN_SHA,
    "qwen25": "487944b404a4c66188d74bda59771c3d15af3d39abed483e78a6e109c82f9559",
    "llama3": "01bca380174b93af607e78dea6fb4f1cf776d14334a8c72a49d7300da9037a2a",
    "gemma3": "666cfdd197dfb1c089d1b1549bce0239cfd8261a5aac80a7f22a6a987265a1f4",
    # Verified 2026-09-07 against Gemma 2 tokenizer revisions c5ebcd4 (2B) and
    # 33c1930 (9B); vocabulary/backend serialization are identical.
    "gemma2": "e85613c26f8069588b3cbabaf5dfeaf1a44ea10be64da2baae8e5fd29fd98f42",
}


def probe_pool_spec_pp1t(tag: str) -> ProbePoolSpec:
    """The probe pool spec of one pp1t family (validates `tag`)."""
    return ProbePoolSpec(n_docs=32, max_tokens=512, seed=0, version=pool_version_for(tag))


# cp1t: the SECOND frozen-text corpus (2026-09-18). cp1's 32 DCLM held-out documents are
# frozen as TEXT (`scripts.build_cp1_text`; decoded from the Pythia ids cp1 is defined on,
# round-trip verified 32/32) and retokenized per family exactly as pp1t does for the Pile.
# `cp1t-olmo` is the in-distribution pool of the DataDecide `dclm-baseline` bases — dp1's
# role for the dolma1_7 ones. There is no `cp1w`: the wide variants are pp1/dp1-only.
CP1_TEXT_PATH = Path("data/probe_pools/cp1_text.jsonl")
CP1_TEXT_META_PATH = Path("data/probe_pools/cp1_text.meta.json")
# Pinned on the first real build per family; None means "not yet pinned", never "no check".
CP1T_TOKEN_SHA: dict[str, str | None] = {
    # olmo: the DataDecide tokenizer is the GPT-NeoX BPE plus a few specials; on THIS
    # POOL's 32 documents it segments identically to Pythia (measured), so the cp1t-olmo
    # pool sha equals cp1's -- not a claim that the two tokenizers agree in general (the
    # retokenized 50k-doc unigram fit stream differs by 464 tokens).
    "olmo": CP1_TOKEN_SHA,
}


def probe_pool_spec_cp1t(tag: str) -> ProbePoolSpec:
    """The DCLM frozen-text pool spec of one family (validates `tag`)."""
    return ProbePoolSpec(n_docs=32, max_tokens=512, seed=0,
                         version=cp1t_pool_version_for(tag))


PROBE_POOL_CP1T_OLMO = ProbePoolSpec(n_docs=32, max_tokens=512, seed=0,
                                     version="cp1t-olmo")


def dd_recipe_pools() -> dict[str, ProbePoolSpec]:
    """One DataDecide data RECIPE -> one corpus pool.

    The recipe, not the repo string, decides: `dolma1_7` bases are measured on dp1 and
    `dclm-baseline` ones on cp1t-olmo (cp1's DCLM held-out text in the OLMo tokenizer),
    because each base's own pretraining corpus is the only in-distribution one for it.
    """
    from subliminal_learning.models.datadecide import (
        DD_DEFAULT_RECIPE, DD_RECIPE_SEGMENTS,
    )

    return {DD_DEFAULT_RECIPE: PROBE_POOL_DP1,
            DD_RECIPE_SEGMENTS["dclm"]: PROBE_POOL_CP1T_OLMO}


class UnknownProbeBaseError(ValueError):
    """The base model belongs to no probe family, so no pool can be chosen for it."""


class IncompatibleProbePoolError(ValueError):
    """The requested pool version is not one this base can be measured on."""


def _olmo_corpus_pool(repo) -> ProbePoolSpec:
    """The corpus pool of a DataDecide base: its RECIPE's (`dd_recipe_pools`).

    A repo neither the naming module nor `dd_recipe_of_repo_id` can read is an ERROR, not
    a silent dp1 — exactly as in `_pythia_corpus_pool`: an unreadable name is the case
    where the default pool may be off-distribution.
    """
    from subliminal_learning.models.datadecide import dd_recipe_of_repo_id
    from subliminal_learning.models.naming import base_spec

    try:
        recipe = getattr(base_spec(repo), "recipe", None)
    except ValueError as exc:
        try:
            if not isinstance(repo, str):
                raise ValueError("not a repo string")
            # A BARE hub repo id is seed-ambiguous (so not a spec) but recipe-unambiguous,
            # and the pool depends only on the recipe.
            recipe = dd_recipe_of_repo_id(repo.split("@")[0])
        except ValueError:
            raise ValueError(
                f"no probe pool for DataDecide-family repo {repo!r}: the naming module "
                "cannot "
                f"map it to a DataDecide spec ({exc}), so its data recipe — and with it "
                "its pretraining corpus — is unknown.") from exc
    pools = dd_recipe_pools()
    if recipe not in pools:
        raise ValueError(f"no probe pool for DataDecide recipe {recipe!r} ({repo!r}); "
                         f"wired recipes: {sorted(pools)}")
    return pools[recipe]


def frozen_text_paths(version: str) -> tuple[Path, Path]:
    """The frozen text artifact (jsonl, meta) a frozen-text pool version reads.

    Module-global lookup at CALL time, so a test can point the artifact constants
    elsewhere.
    """
    if is_pp1w(version):
        return PP1W_TEXT_PATH, PP1W_TEXT_META_PATH
    if is_cp1t(version):
        return CP1_TEXT_PATH, CP1_TEXT_META_PATH
    return PP1_TEXT_PATH, PP1_TEXT_META_PATH


def pinned_frozen_text_sha(version: str) -> str | None:
    """The pinned token sha of a frozen-text pool version (None = not yet pinned)."""
    tag = family_for_pool_version(version).tag
    table = (PP1W_TOKEN_SHA if is_pp1w(version) else
             CP1T_TOKEN_SHA if is_cp1t(version) else PP1T_TOKEN_SHA)
    return table.get(tag)


# pp1w / dp1w: the NESTED WIDE pools (2026-09-05). 128 documents = the 32 base documents,
# byte-identical and in order, plus 96 extension documents drawn from the same seeded
# permutation stream but from AFTER the frozen unigram fit window — so a wide pool is
# disjoint from the counts it is scored against exactly as its base pool is, its first 32
# docs reproduce the base pool token-for-token, and the base H_uni counts are REUSED
# (never refit). Per-document CE (`doc_ce`) makes the 32-doc reading recoverable from a
# 128-doc measurement via `subset_mean`, so wide and narrow numbers stay comparable.
PP1W_N_DOCS = 128
PP1W_TEXT_PATH = Path("data/probe_pools/pp1w_text.jsonl")
PP1W_TEXT_META_PATH = Path("data/probe_pools/pp1w_text.meta.json")
# Not yet pinned (the wide text has not been frozen/measured yet); None means "not yet
# pinned", never "no check".
# Pinned on the first real builds (2026-09-05; frozen wide text sha 9abb958c5c839cd5…,
# 128 docs = pp1's 32 + 96 from permutation position 53281). pythia == olmo again
# (token-identical tokenizers on this text).
PP1W_TOKEN_SHA: dict[str, str | None] = {
    "pythia": "e0d5d92b9400075b1cdaef4d3bd558be11d35bc8835b7716f3e1f05e75faf3f5",
    "olmo": "e0d5d92b9400075b1cdaef4d3bd558be11d35bc8835b7716f3e1f05e75faf3f5",
    "qwen25": "856edf5c49fc93136cb4322518380c244982ad9035c43cc7f5915282ad49b918",
    "llama3": "fa68fdda1cfe23f3712de89b787988f11f1f7226d58aea3a46ddb0316b0237a2",
    "gemma3": "90abe0bd1d60cfe0a2782bb60acaf1ead5435ec4e7f9554427d8e61be3c11ca1",
    "gemma2": "0b7be338999b301c637118e800b962501fb014cae6a0216e8867ae53fd6443ff",
}
# Pinned on the first real build (2026-09-05): dp1's 32 docs + 96 from seeded position 53280.
DP1W_TOKEN_SHA: str | None = "83a1bbfdec62486c4d39e44b99b052a422d71270e3e4bf5e085d40e3c44f8fda"
# dp1's fit window is permutation positions [32, 53280) (`permutation_offset` 32 +
# `n_docs_scanned` 53248 in unigram_anchor_dp1.json), so the extension starts at 53280.
DP1W_EXTENSION_START = 53_280
PROBE_POOL_DP1W = ProbePoolSpec(n_docs=PP1W_N_DOCS, max_tokens=512, seed=0, version="dp1w")


def probe_pool_spec_pp1w(tag: str) -> ProbePoolSpec:
    """The WIDE probe pool spec of one family (validates `tag`)."""
    return ProbePoolSpec(n_docs=PP1W_N_DOCS, max_tokens=512, seed=0,
                         version=wide_pool_version_for(tag))


def _resolve_artifact(path: Path) -> Path:
    """Frozen data artifacts are addressed relative to the REPO ROOT, not the cwd."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def pp1_text_sha256(records: list[dict]) -> str:
    """Content hash of the frozen pp1 text (order-sensitive; mirrors `token_sha256`)."""
    import hashlib

    h = hashlib.sha256()
    for rec in records:
        h.update(json.dumps(rec["text"]).encode())
    return h.hexdigest()


def load_frozen_text(path: Path, meta_path: Path) -> tuple[list[dict], dict]:
    """Load a frozen text artifact (pp1 or pp1w), verifying `text_sha256` against the meta."""
    recs = [json.loads(ln) for ln in
            _resolve_artifact(path).read_text().splitlines() if ln.strip()]
    meta = json.loads(_resolve_artifact(meta_path).read_text())
    sha = pp1_text_sha256(recs)
    if sha != meta["text_sha256"]:
        raise ValueError(f"frozen text artifact {path} is corrupt: text_sha256 {sha} != "
                         f"recorded {meta['text_sha256']}")
    return recs, meta


def load_pp1_text(path: Path = PP1_TEXT_PATH,
                  meta_path: Path = PP1_TEXT_META_PATH) -> tuple[list[dict], dict]:
    """Load the frozen pp1 text artifact (the 32-doc base of every pp1t pool)."""
    return load_frozen_text(path, meta_path)


def build_frozen_text_pool(tokenizer, spec: ProbePoolSpec, *, bos_token_id: int | None,
                           path: Path = PP1_TEXT_PATH,
                           meta_path: Path = PP1_TEXT_META_PATH):
    """Retokenize a frozen text artifact under `tokenizer` into a `SequencePool`.

    Serves every frozen-text pool family: pass the pp1 artifact for a ``pp1t-<tag>``
    pool, the pp1w one for a ``pp1w-<tag>`` pool (whose first 32 records ARE the pp1
    ones, so the wide pool's 32-doc prefix is token-identical to the narrow pool), and
    the cp1 one for a ``cp1t-<tag>`` pool. `frozen_text_paths` picks the artifact.

    No truncation (each text was already cut at its 512th Pythia token); BOS, when the
    family asks for one, is prepended EXPLICITLY. `n_scanned` is carried over from the
    artifact — for pp1w that is the BASE pool's value, so the unigram fit window
    arithmetic downstream is unchanged.

    pp1's min-32 rule is a FREEZE-TIME document-SELECTION rule in Pythia tokens
    (`scripts.freeze_pp1_text._doc_record`) and is deliberately NOT re-applied per
    family: the frozen text is the pool's identity, so every family must retokenize the
    SAME document set — nesting (pp1w's 32-doc prefix) and cross-family comparability
    depend on it. A family tokenizer can legitimately land a document under 32 tokens.
    The only check here is a sanity floor that each document still yields at least one
    CE target (targets are `ids[2:]`).
    """
    from subliminal_learning.cka.pool_io import SequencePool

    recs, meta = load_frozen_text(path, meta_path)
    input_ids: list[list[int]] = []
    eligible: list[list[int]] = []
    for rec in recs:
        ids = list(tokenizer(rec["text"], add_special_tokens=False)["input_ids"])
        if bos_token_id is not None:
            ids = [bos_token_id] + ids
        if len(ids) < 3:
            raise ValueError(f"{spec.version} doc {rec['order_pos']} has only "
                             f"{len(ids)} tokens under {spec.version} "
                             f"(min 3: needs at least one CE target)")
        input_ids.append(ids)
        eligible.append(list(range(1, len(ids))))
    # The source names the frozen CORPUS the text came from ("pp1_text" for pp1/pp1w,
    # "cp1_text" for cp1), never the family: the text is the pool's identity.
    return SequencePool(input_ids, eligible,
                        f"{meta.get('base_pool', 'pp1')}_text:{meta['text_sha256'][:12]}",
                        int(meta["n_scanned"]), 0, [int(r["row"]) for r in recs])


def _pythia_corpus_pool(repo) -> ProbePoolSpec:
    """The corpus pool of a Pythia-tokenizer base: pp1 unless its variant re-pretrained
    on another corpus (`PYTHIA_VARIANT_POOLS`).

    A repo the naming module cannot spell as a Pythia spec is an ERROR, not a silent pp1:
    the pretraining CORPUS decides the pool, and an unparseable name (a typo, an
    unregistered seed) is exactly the case where pp1 may be off-distribution.
    """
    from subliminal_learning.models.naming import base_spec

    try:
        variant = getattr(base_spec(repo), "variant", None)
    except ValueError as exc:
        raise ValueError(
            f"no probe pool for Pythia-family repo {repo!r}: the naming module cannot map "
            f"it to a Pythia base spec ({exc}), so its pretraining corpus is unknown and "
            "pp1 may be off-distribution for it.") from exc
    return PYTHIA_VARIANT_POOLS.get(variant or "", PROBE_POOL_PP1)


def _probe_family_for_base(base):
    """The `ProbeFamily` of any base spelling (token, ref, dir name, or a `BaseSpec`).

    Raises:
        UnknownProbeBaseError: when the base belongs to no wired probe family.
    """
    if isinstance(base, str):
        family = family_for_base_repo(base)
    else:
        from subliminal_learning.models.naming import family_of

        try:
            tag = family_of(base)
        except ValueError as exc:
            raise UnknownProbeBaseError(f"no probe pool for base {base!r}: {exc}") from exc
        family = PROBE_FAMILIES.get("olmo" if tag == "datadecide" else tag)
    if family is None:
        raise UnknownProbeBaseError(
            f"no probe pool for base {base!r}: it belongs to no known probe family "
            f"({sorted(PROBE_FAMILIES)}). Add it to `probe_families.BASE_REPO_FAMILY` "
            "(and pin its pp1t token sha) first.")
    return family


def native_probe_pool(base) -> ProbePoolSpec:
    """THE pool a base model must be measured on -- the one place that choice lives.

    `base` is ANY spelling the naming module reads: a `--base-models` token (`dd-1b-dclm`,
    `pythia-160m-seed1`), a hub ref / repo id, an output-tree dir name, or a `BaseSpec`.

    The base's PRETRAINING CORPUS decides, never the tokenizer: DataDecide bases take
    their data recipe's pool (`dd_recipe_pools`: dolma1_7 -> dp1, dclm-baseline ->
    cp1t-olmo), Pythia bases pp1 unless the variant re-pretrained elsewhere
    (`PYTHIA_VARIANT_POOLS`: the `dclm` family -> cp1), and every other registered family
    its ``pp1t-<tag>`` frozen-text pool. An unrecognised base is an
    `UnknownProbeBaseError`, never a silent fallback to pp1: measuring a Llama base on
    Pythia tokens is meaningless. Errors are always named ValueErrors, never KeyError.
    """
    family = _probe_family_for_base(base)
    if family.tag == "olmo":
        return _olmo_corpus_pool(base)
    if family.tag == "pythia":
        return _pythia_corpus_pool(base)
    return probe_pool_spec_pp1t(family.tag)


def native_pool_version_for_base(base) -> str:
    """`native_probe_pool(base).version` -- the version string of a base's native pool.

    Distinct from `native_probe_pool_version(run_dir)`, which reads a RUN's manifest.
    """
    return native_probe_pool(base).version


def compatible_probe_pools(base) -> tuple[str, ...]:
    """Every pool version `base` can legitimately be measured on, NATIVE FIRST.

    Compatibility is tokenizer-level plus corpus-level bookkeeping: a base's own family
    corpus pools (dp1 / cp1t-olmo for olmo, pp1 / cp1 for pythia) and the family's
    frozen-text retokenizations (``pp1t-<tag>``, ``pp1w-<tag>``, ``cp1t-<tag>``). Anything
    else is another family's tokens and comparable to none of this base's records.
    """
    family = _probe_family_for_base(base)
    native = native_probe_pool(base).version
    out: list[str] = [native]
    for version in (*corpus_pools_for(family.tag), pool_version_for(family.tag),
                    wide_pool_version_for(family.tag), cp1t_pool_version_for(family.tag)):
        if version not in out:
            out.append(version)
    return tuple(out)


def require_compatible_probe_pool(base, version: str) -> ProbePoolSpec:
    """The spec of `version`, or `IncompatibleProbePoolError` if `base` cannot take it."""
    allowed = compatible_probe_pools(base)
    if version not in allowed:
        raise IncompatibleProbePoolError(
            f"probe pool {version!r} is not a pool for base {base!r} (family "
            f"{_probe_family_for_base(base).tag!r}); compatible pools, native first: "
            f"{', '.join(allowed)}. A readout on another family's tokens -- or on another "
            "pretraining corpus -- is comparable to none of this base's records")
    return probe_pool_spec_for_version(version)


def probe_pool_spec_for_repo(repo) -> ProbePoolSpec:
    """Thin alias of `native_probe_pool` kept for existing importers."""
    return native_probe_pool(repo)


def resolve_probe_pool_spec(repo, pool: str | None = None) -> ProbePoolSpec:
    """The pool a CLI measures base `repo` on; an explicit `pool` must be compatible.

    `pool` None / ``""`` / ``"auto"`` gives `native_probe_pool`; anything else goes
    through `require_compatible_probe_pool` (see `compatible_probe_pools`). The surgery
    and ladder CLIs used to default to pp1 and accept it for any base, which on a Qwen
    base silently produced a readout comparable to none of its pp1t records.
    """
    if pool in (None, "", "auto"):
        return native_probe_pool(repo)
    return require_compatible_probe_pool(repo, pool)


# Canonical home is utils.run_io (torch-free); re-exported here for existing importers.
from subliminal_learning.utils.run_io import manifest_ref  # noqa: E402,F401


def build_probe_pool(tokenizer, spec: ProbePoolSpec = PROBE_POOL_PP1, *,
                     verify_pin: bool = True):
    """The probe pool for `spec` (see `cka.pool_io.build_pile_pool` / `dolma_pool`).

    For the frozen-text pools (`pp1t-<tag>` / `pp1w-<tag>`) the built pool is checked
    against its pinned token sha, so a wrong or silently-updated tokenizer fails here
    rather than producing an incomparable readout. A None pin means "not yet pinned"
    (no check). Pass `verify_pin=False` to opt out — for the pinning script itself
    (`scripts.build_unigram_anchor`, which raises its own message) and for tests
    building pools with synthetic tokenizers.
    """
    if is_frozen_text_pool(spec.version):
        family = family_for_pool_version(spec.version)
        path, meta_path = frozen_text_paths(spec.version)
        pool = build_frozen_text_pool(
            tokenizer, spec,
            bos_token_id=tokenizer.bos_token_id if family.bos else None,
            path=path, meta_path=meta_path)
        if verify_pin:
            pinned = pinned_frozen_text_sha(spec.version)
            got = pool.token_sha256()
            if pinned is not None and got != pinned:
                raise ValueError(
                    f"{spec.version} pool does not match its pinned tokenization: got "
                    f"token_sha256 {got}, pinned {pinned}. The tokenizer used to build "
                    "the pool is not the one the pin was taken on.")
        return pool
    if spec.version == "cp1":
        from subliminal_learning.coherence.dclm_pool import build_dclm_pool

        pool = build_dclm_pool(tokenizer, spec.n_docs, max_tokens=spec.max_tokens,
                               seed=spec.seed)
        if verify_pin and CP1_TOKEN_SHA is not None:
            got = pool.token_sha256()
            if got != CP1_TOKEN_SHA:
                raise ValueError(
                    f"cp1 pool does not match its pinned tokenization: got token_sha256 "
                    f"{got}, pinned {CP1_TOKEN_SHA}. The held-out source moved under us.")
        return pool
    if spec.version in ("dp1", "dp1w"):
        from subliminal_learning.coherence.dolma_pool import build_dolma_pool

        wide = spec.version == "dp1w"
        return build_dolma_pool(
            tokenizer, spec.n_docs, max_tokens=spec.max_tokens, seed=spec.seed,
            extension_start=DP1W_EXTENSION_START if wide else None,
            n_base=PROBE_POOL_DP1.n_docs if wide else None)
    from subliminal_learning.cka.pool_io import build_pile_pool

    return build_pile_pool(tokenizer, spec.n_docs, max_tokens=spec.max_tokens, seed=spec.seed)


def probe_pool_spec_for_version(version: str) -> ProbePoolSpec:
    """The frozen pool spec named by `version` -- the CROSS-CORPUS escape hatch.

    `probe_pool_spec_for_repo` stays the rule (a base is measured on its pretraining
    corpus); this function is how a caller deliberately measures the SAME run on another
    corpus pool too, e.g. a Pile-pretrained base on cp1 so pp1 and cp1 numbers are
    comparable within one campaign (2026-09-17). `resolve_probe_pool_spec`'s family check
    is untouched: the cross-corpus path is opt-in only through here, never by default.

    Knows EVERY wired version: the corpus pools (pp1, cp1, dp1, dp1w) and every
    frozen-text family pool (``pp1t-``/``pp1w-``/``cp1t-<tag>``).
    """
    pools = {p.version: p for p in (PROBE_POOL_PP1, PROBE_POOL_CP1, PROBE_POOL_DP1,
                                    PROBE_POOL_DP1W, PROBE_POOL_CP1T_OLMO)}
    if version in pools:
        return pools[version]
    if is_frozen_text_pool(version):
        tag = family_for_pool_version(version).tag        # named ValueError on a bad tag
        if is_pp1w(version):
            return probe_pool_spec_pp1w(tag)
        if is_cp1t(version):
            return probe_pool_spec_cp1t(tag)
        return probe_pool_spec_pp1t(tag)
    raise ValueError(
        f"unknown probe pool version {version!r}; known: {sorted(pools)} plus "
        f"'pp1t-<tag>' / 'pp1w-<tag>' / 'cp1t-<tag>' for tags {sorted(PROBE_FAMILIES)}")


def native_probe_pool_version(run_dir: Path) -> str | None:
    """The pool version a run's OWN base is natively measured on, or None if unknowable.

    Takes a RUN DIR and reads its manifest; `native_pool_version_for_base` is the same
    answer from a base SPELLING.
    """
    mf = Path(run_dir) / "train_manifest.json"
    if not mf.is_file():
        return None
    try:
        repo = json.loads(mf.read_text()).get("base_model_repo")
        return probe_pool_spec_for_repo(repo).version if repo else None
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def probe_path(run_dir: Path, pool: str | None = None) -> Path:
    """Where a run's probe record for pool version `pool` lives.

    The base's NATIVE pool keeps the canonical `eval/pile_probe.json` (every existing
    reader); any OTHER corpus pool goes to the sidecar `eval/pile_probe.<version>.json`,
    so the two never overwrite each other. `pool=None` means "the native record".
    """
    if pool is None or pool == native_probe_pool_version(run_dir):
        return Path(run_dir) / "eval" / EVAL_NAME
    return Path(run_dir) / "eval" / f"pile_probe.{pool}.json"


def load_probe(run_dir: Path, pool: str | None = None) -> dict | None:
    """The run's probe record for `pool` (default: the native one), or None."""
    f = probe_path(run_dir, pool)
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, OSError):
        return None


@dataclass(frozen=True)
class DeadCut:
    """A data-derived integrity dead cut for one population of probe values.

    The alive/dead separation is bimodal with a LOW-DENSITY VALLEY, not an empty gap
    (the strip's valley 3.6-6.3 holds ~5% of seeds -- boundary transit is a continuum).
    So the derivation returns the valley REGION and its occupancy alongside the point
    cut; any point inside the valley is an equivalent label up to `n_valley` seeds, and
    downstream fits should quote a cut-sensitivity check, not treat `cut` as exact.
    (The strip's valley is a broad shallow plateau at ~30% of the dead mode's density —
    a transit continuum, not an empty band; `valley_frac` = 0.5 admits it. The knobs
    move the valley EDGES, but `n_valley` always states the resulting label churn.)

    Attributes:
        cut: Density minimum inside the valley (nats); dead = value > cut.
        valley_lo, valley_hi: Valley edges (density first falls below / last exceeds
            `valley_frac` x the smaller mode's peak density).
        n_valley: Seeds inside the valley = the maximum label churn any in-valley cut
            choice could cause.
        mode_lo, mode_hi: Locations of the two dominant density modes.
        n_below, n_above: Population sizes on each side of `cut`.
    """

    cut: float
    valley_lo: float
    valley_hi: float
    n_valley: int
    mode_lo: float
    mode_hi: float
    n_below: int
    n_above: int


def derive_dead_cut(values, bin_width: float = 0.75, smooth_bins: int = 3,
                    valley_frac: float = 0.5, min_mode_runs: int = 200) -> DeadCut | None:
    """Derive the integrity dead cut from a population's probe values (delta-CE or KL).

    No preregistered constant: a smoothed histogram is scanned for its two dominant
    modes; the cut is the density minimum between them, and the valley region (density
    below `valley_frac` of the smaller mode's peak) plus its seed count are returned so
    the residual arbitrariness is quantified rather than hidden. Populations without two
    modes each holding at least `min_mode_runs` runs get None ("no bimodal separation
    here -- do not force a label"). The gate is COUNT-based (2026-08-20, was a 10% mass
    fraction): what makes a sparse mode's valley untrustworthy is too few runs to pin its
    density down, not its share of the pool -- a pooled population diluted by alive-heavy
    coverage can hold a perfectly well-populated dead mode at a small mass fraction.
    Re-derive per population; never carry a cut across models/arms without re-deriving.

    Returns:
        A `DeadCut`, or None when the population is not usably bimodal.
    """
    import numpy as np

    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    n = len(v)
    if n < 50:
        return None
    edges = np.arange(v.min(), v.max() + bin_width, bin_width)
    if len(edges) < 8:
        return None
    h, _ = np.histogram(v, bins=edges)
    k = np.ones(smooth_bins) / smooth_bins
    d = np.convolve(h, k, mode="same")
    centers = 0.5 * (edges[:-1] + edges[1:])

    # dominant mode on each side of the global density minimum's neighbourhood: take the
    # global max as mode 1, then the best local max separated from it by a density dip
    i1 = int(np.argmax(d))
    best = None
    for i2 in range(len(d)):
        if abs(i2 - i1) < 3:
            continue
        lo, hi = sorted((i1, i2))
        dip = d[lo:hi + 1].min()
        if dip < valley_frac * min(d[i1], d[i2]) and d[i2] > 0:
            score = d[i2]
            if best is None or score > best[0]:
                best = (score, i2)
    if best is None:
        return None
    i2 = best[1]
    lo, hi = sorted((i1, i2))
    # gate on the runs each side of the candidate cut (the interior density minimum) --
    # each mode's full population, not just its beyond-the-peak half
    _inner_idx = np.arange(lo, hi + 1)
    _jdip = _inner_idx[int(np.argmin(d[_inner_idx]))]
    n_side_lo = int((v <= centers[_jdip]).sum())
    if min(n_side_lo, n - n_side_lo) < min_mode_runs:
        return None

    peak_small = min(d[i1], d[i2])
    inner = slice(lo, hi + 1)
    below = d[inner] < valley_frac * peak_small
    if not below.any():
        return None
    idx = np.arange(lo, hi + 1)[below]
    valley_lo, valley_hi = float(centers[idx[0]]), float(centers[idx[-1]])
    jmin = idx[int(np.argmin(d[idx]))]
    cut = float(centers[jmin])
    return DeadCut(cut=cut, valley_lo=valley_lo, valley_hi=valley_hi,
                   n_valley=int(((v > valley_lo) & (v < valley_hi)).sum()),
                   mode_lo=float(centers[min(i1, i2)]), mode_hi=float(centers[max(i1, i2)]),
                   n_below=int((v <= cut).sum()), n_above=int((v > cut).sum()))


def probe_is_current(run_dir: Path, pool_sha: str, pool: str | None = None) -> bool:
    """True when the record exists and still describes both the run and the pool.

    Keyed on the per-pool PATH (`probe_path`): without that, a run measured on two pools
    would see the other pool's record, call it stale and overwrite it, forever.
    """
    rec = load_probe(run_dir, pool)
    if rec is None:
        return False
    ref = manifest_ref(run_dir)
    return (
        ref is not None
        and rec.get("schema_version") == SCHEMA_VERSION
        and rec.get("manifest_ref") == ref
        and rec.get("pool_sha") == pool_sha
    )


# --------------------------------------------------------------------------------------
# Pile-probe measurement (single implementation, GPU-resident). Moved here 2026-08-20 from
# scripts/frozen_a_ablation.py (`_pile_logprobs`, `_mean_kl`) and scripts/pile_probe_final.py
# (`_mean_ce`), which ran docs one at a time and shipped every log-prob tensor to the CPU --
# that round-trip made probe jobs CPU-bound (17 s/run on an idle GPU).
# --------------------------------------------------------------------------------------


LARGE_VOCAB = 65_536   # above this, park per-doc log-probs on the CPU (see `pool_logprobs`)


def default_batch_size(model) -> int:
    """Probe batch size for `model`'s vocabulary: 8 up to `LARGE_VOCAB`, less above.

    The batch's fp32 logits are ``[B, T, V]``: 2.5 GiB at ``[8, 512, 152k]``, which OOMed
    the probe subprocess on a shared card (2026-09-08). Scaling B down by the vocabulary
    ratio holds that tensor at the Pythia-vocab footprint (3 docs at Qwen's 151,936, 2 at
    Gemma's 262,144) and leaves the <=65,536 path at the historical 8.
    """
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab <= LARGE_VOCAB:
        return 8
    return max(1, (8 * LARGE_VOCAB) // vocab)


def pile_logprobs(model, pool, device: str, batch_size: int | None = None) -> list:
    """Next-token log-probs at the pool's eligible positions, one GPU tensor per doc.

    Eligibility indexes the *logits*, not the targets: a KL between two models compares two
    predictive distributions and needs no observed next token, so every position carries a
    usable distribution -- including the last. What it must exclude is position 0, the BOS
    attention-sink outlier the pool's ``eligible`` convention drops, which is exactly what
    indexing logits by ``eligible`` does.

    Docs run in right-padded batches and the per-doc float32 log-prob tensors stay on
    `device` (~3.2 GB for pp1 at the Pythia vocab). Right padding is exact for a causal LM:
    real-position logits cannot attend to the (future) pad keys, and GPT-NeoX's arange
    position_ids are unaffected by trailing pads, so no pad-token dependence is introduced.

    `batch_size=None` (the default) sizes the batch by vocabulary, see
    `default_batch_size`; an explicit value always wins. The numbers are exact per
    document either way -- batching only sets how much padding a group carries.
    """
    import torch

    batch_size = default_batch_size(model) if batch_size is None else batch_size

    out = [None] * len(pool.input_ids)
    with torch.no_grad():
        for start in range(0, len(pool.input_ids), batch_size):
            docs = pool.input_ids[start:start + batch_size]
            eligs = pool.eligible[start:start + batch_size]
            length = max(len(d) for d in docs)
            ids = torch.zeros(len(docs), length, dtype=torch.long, device=device)
            mask = torch.zeros(len(docs), length, dtype=torch.long, device=device)
            for i, d in enumerate(docs):
                ids[i, : len(d)] = torch.tensor(d, device=device)
                mask[i, : len(d)] = 1
            logits = model(input_ids=ids, attention_mask=mask).logits
            for i, el in enumerate(eligs):
                sel = logits[i].index_select(0, torch.tensor(el, dtype=torch.long, device=device)).float()
                out[start + i] = torch.log_softmax(sel, dim=-1)
            del logits
    return out


def to_device(logp_docs: list, device: str) -> list:
    """Move each per-doc log-prob tensor to `device`, dtype untouched.

    The three log-prob lists (base, original, ablated) are ~3.3 GB each at a 50k vocab,
    so holding them next to a 1B fp32 base is what OOMs a shared GPU. `mean_kl`/`mean_ce`
    are device-agnostic, so parking them on CPU changes nothing but the footprint.
    """
    return [t.to(device) for t in logp_docs]


def parked_logprobs(model, pool, device: str, logp_device: str,
                    batch_size: int | None = None) -> list:
    """`pile_logprobs` with each batch's tensors parked on `logp_device` as it is produced.

    The pool is walked in the same `batch_size` groups `pile_logprobs` uses internally and
    each group is handed over whole, so the padding groups — and therefore the numbers —
    are byte-identical to one whole-pool call; only the peak footprint changes, from the
    full log-prob list on `device` to one batch. `batch_size=None` -> `default_batch_size`.
    """
    import dataclasses

    batch_size = default_batch_size(model) if batch_size is None else batch_size
    out: list = []
    for start in range(0, len(pool.input_ids), batch_size):
        chunk = dataclasses.replace(
            pool,
            input_ids=pool.input_ids[start:start + batch_size],
            eligible=pool.eligible[start:start + batch_size],
        )
        out.extend(to_device(pile_logprobs(model, chunk, device, batch_size=batch_size),
                             logp_device))
    return out


def pool_logprobs(model, pool, device: str, batch_size: int | None = None) -> list:
    """`pile_logprobs`, parked on the CPU for large-vocabulary families.

    The per-doc float32 log-prob lists are ~3.2 GB on the GPU for pp1 at the 50k Pythia/OLMo
    vocab, but ~9.6 GB at Qwen's 152k (and a KL trajectory holds two of them next to the
    fp32 model), which overran the 24 GB slot reservation on 2026-09-08. For a vocab above
    `LARGE_VOCAB` this returns `parked_logprobs(..., "cpu")` -- byte-identical numbers, one
    batch of GPU footprint; `mean_kl` / `mean_ce` run on whatever device the tensors sit on.
    Small vocabularies keep the historical all-on-GPU path unchanged.

    A large vocabulary also shrinks the default batch (`default_batch_size`): the fp32
    ``[8, 512, 152k]`` logits tensor alone is 2.5 GiB and OOMed the probe subprocess.
    """
    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab > LARGE_VOCAB:
        return parked_logprobs(model, pool, device, "cpu", batch_size=batch_size)
    return pile_logprobs(model, pool, device, batch_size=batch_size)


@dataclass(frozen=True)
class PoolReadout:
    """Streamed reductions of one model over a probe pool (see `stream_pool_readout`).

    Attributes:
        ce: Position-mean CE [nats] over the pool's eligible targets — `mean_ce`'s number.
        doc_ce_sums: Per-document summed target NLL — `doc_ce`'s first output.
        doc_ce_counts: Per-document target counts — `doc_ce`'s second output.
        kl_from: ``{name: KL(ref_name ‖ model)}``, position-pooled — `mean_kl(ref, model)`.
        kl_to: ``{name: KL(model ‖ ref_name)}`` for the names asked for — `mean_kl(model, ref)`.
        n_positions: Eligible logit positions reduced (the KL denominator).
    """

    ce: float
    doc_ce_sums: list[float]
    doc_ce_counts: list[int]
    kl_from: dict[str, float] = field(default_factory=dict)
    kl_to: dict[str, float] = field(default_factory=dict)
    n_positions: int = 0


def stream_pool_readout(model, pool, device: str, refs: Mapping[str, list] | None = None, *,
                        reverse: Iterable[str] = (), batch_size: int | None = None) -> PoolReadout:
    """CE and KL of `model` over `pool`, reduced chunk by chunk — no log-prob list is built.

    The streaming readout proposed in the 2026-09-08 pipeline audit. `pile_logprobs` /
    `parked_logprobs` materialise one full-vocabulary log-prob row per eligible position
    (~6 GB on Qwen's 152k vocab for pp1t, ~10 GB at every position) and the callers then
    reduce that list — on the CPU when parked, 15 s per readout at two OMP threads. This
    walks the pool in the SAME batch groups `pile_logprobs` uses, reduces each chunk's rows
    to scalars on the device they were produced on, and frees them, so the peak is one
    batch of logits whatever the vocabulary and the arithmetic stays on the GPU.

    `refs` are per-document log-prob lists of reference models (a parked `parked_logprobs` /
    `pool_logprobs` result, or a GPU-resident `pile_logprobs` one); each chunk of a
    reference is moved to the model's device only while it is reduced, so a CPU-parked base
    costs one host-to-device copy per readout instead of a device-to-host copy plus a CPU
    reduction. `reverse` names the refs whose ``KL(model ‖ ref)`` is wanted as well.

    Numbers: the per-document sums are the same tensor operations `mean_kl` / `mean_ce` /
    `doc_ce` apply, in the same per-document order — the same measurement, verified against
    the list path when it landed (2026-09-10). Batching only changes padding, exactly as for
    `pile_logprobs`.
    """
    import dataclasses

    import torch

    refs = dict(refs or {})
    reverse = set(reverse)
    unknown = reverse - set(refs)
    if unknown:
        raise ValueError(f"reverse KL asked for refs not given: {sorted(unknown)}")
    for name, ref in refs.items():
        if len(ref) != len(pool.input_ids):
            raise ValueError(f"ref {name!r} has {len(ref)} docs, pool has {len(pool.input_ids)}")
    batch_size = default_batch_size(model) if batch_size is None else batch_size
    sums: list[float] = []
    counts: list[int] = []
    kl_from = {name: 0.0 for name in refs}
    kl_to = {name: 0.0 for name in reverse}
    n_positions = 0
    with torch.no_grad():
        for start in range(0, len(pool.input_ids), batch_size):
            chunk = dataclasses.replace(
                pool,
                input_ids=pool.input_ids[start:start + batch_size],
                eligible=pool.eligible[start:start + batch_size],
            )
            q_docs = pile_logprobs(model, chunk, device, batch_size=batch_size)
            chunk_sums, chunk_counts = doc_ce(q_docs, chunk)
            sums.extend(chunk_sums)
            counts.extend(chunk_counts)
            for i, q in enumerate(q_docs):
                n_positions += int(q.shape[0])
                for name, ref in refs.items():
                    p = ref[start + i]
                    if p.device != q.device:
                        p = p.to(q.device)
                    kl_from[name] += float((p.exp() * (p - q)).sum(dim=-1).sum())
                    if name in reverse:
                        kl_to[name] += float((q.exp() * (q - p)).sum(dim=-1).sum())
                    del p
            del q_docs
    n_targets = sum(counts)
    return PoolReadout(
        ce=sum(sums) / n_targets if n_targets else float("nan"),
        doc_ce_sums=sums, doc_ce_counts=counts,
        kl_from={k: (v / n_positions if n_positions else float("nan")) for k, v in kl_from.items()},
        kl_to={k: (v / n_positions if n_positions else float("nan")) for k, v in kl_to.items()},
        n_positions=n_positions,
    )


def stream_kl_ce(model, pool, device: str, base_logp: list, *,
                 batch_size: int | None = None) -> tuple[float, float]:
    """``(KL(base ‖ model), CE(model))`` streamed — the two numbers every probe record needs."""
    out = stream_pool_readout(model, pool, device, {"base": base_logp}, batch_size=batch_size)
    return out.kl_from["base"], out.ce


def mean_kl(ref_logp: list, other_logp: list) -> float:
    """Position-mean ``KL(ref ‖ other)`` in nats, pooled over the whole pool.

    Pooled over positions rather than averaged per document, so long documents are not
    down-weighted relative to short ones.
    """
    total, n = 0.0, 0
    for p_log, q_log in zip(ref_logp, other_logp):
        kl = (p_log.exp() * (p_log - q_log)).sum(dim=-1)
        total += float(kl.sum())
        n += int(kl.numel())
    return total / n if n else float("nan")


def mean_ce(logp_docs: list, pool) -> float:
    """Position-mean CE (nats) of the pool's true next tokens under `logp_docs`.

    ``logp_docs[d]`` holds log-probs at the doc's eligible logit positions; the target for
    logit position i is ``input_ids[i+1]``, which exists for every eligible position except
    the last one -- dropped here.
    """
    import torch

    total, n = 0.0, 0
    for logp, ids, elig in zip(logp_docs, pool.input_ids, pool.eligible):
        keep = [j for j, pos in enumerate(elig) if pos + 1 < len(ids)]
        tgt = torch.tensor([ids[elig[j] + 1] for j in keep], dtype=torch.long, device=logp.device)
        total += float(-logp[torch.tensor(keep, dtype=torch.long, device=logp.device)]
                       .gather(1, tgt.unsqueeze(1)).sum())
        n += len(keep)
    return total / n if n else float("nan")


def doc_ce(logp_docs: list, pool) -> tuple[list[float], list[int]]:
    """Per-document summed target NLL [nats] and target count, same rule as `mean_ce`.

    ``sum(sums) / sum(counts)`` is `mean_ce` exactly; the split is what makes a NESTED
    pool readable — the first 32 documents of a wide pool give the narrow pool's number
    through `subset_mean`, from a single 128-document forward pass.
    """
    import torch

    sums: list[float] = []
    counts: list[int] = []
    for logp, ids, elig in zip(logp_docs, pool.input_ids, pool.eligible):
        keep = [j for j, pos in enumerate(elig) if pos + 1 < len(ids)]
        tgt = torch.tensor([ids[elig[j] + 1] for j in keep], dtype=torch.long, device=logp.device)
        sums.append(float(-logp[torch.tensor(keep, dtype=torch.long, device=logp.device)]
                          .gather(1, tgt.unsqueeze(1)).sum()))
        counts.append(len(keep))
    return sums, counts


def subset_mean(sums: list[float], counts: list[int], k: int) -> float:
    """Token-weighted mean CE over the FIRST `k` documents of a `doc_ce` split."""
    n = sum(counts[:k])
    return sum(sums[:k]) / n if n else float("nan")
