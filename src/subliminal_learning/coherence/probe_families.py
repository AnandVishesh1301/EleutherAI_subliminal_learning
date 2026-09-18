"""Base-model families for the cross-tokenizer probe pools ``pp1t-<tag>``.

pp1t is pp1's 32 Pile-validation documents frozen as TEXT (`data/probe_pools/pp1_text.*`)
and retokenized per family, so an integrity readout can be taken on a base whose
tokenizer is not GPT-NeoX. One family = one tokenizer + one BOS policy; the BOS token is
prepended EXPLICITLY (this repo never encodes with ``add_special_tokens=True``), because
Llama-3 and Gemma-3 bases are trained with a leading BOS and score badly without it.

Per-token H_uni is NOT comparable across families (different vocabularies segment the
same text into different numbers of tokens) — compare bits per byte instead.

The same mechanism carries a second frozen corpus: ``cp1t-<tag>`` is cp1's 32 DCLM
held-out documents frozen as text (`data/probe_pools/cp1_text.*`) and retokenized the
same way, so the DataDecide `dclm-baseline` bases have an in-distribution pool.

Import-light on purpose (like `models.datadecide`): transformers is imported inside
`load_family_tokenizer`, so the table can be read by torch-free callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from subliminal_learning.models.datadecide import (
    DataDecideSpec,
    dd_repo_id,
    dd_revision,
)
from subliminal_learning.models.registry import EXTERNAL_BASES

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

POOL_PREFIX = "pp1t-"
# The nested 128-doc "wide" variant of the same frozen text (see `coherence.probe_io`):
# its first 32 documents ARE pp1t's, so a wide readout carries its narrow one inside it.
WIDE_PREFIX = "pp1w-"
# The SECOND frozen-text corpus (2026-09-18): cp1's 32 DCLM held-out documents frozen as
# text and retokenized per family, exactly as pp1t does for the Pile. Its first consumer
# is `cp1t-olmo`, the in-distribution pool of the DataDecide `dclm-baseline` bases.
CP1T_PREFIX = "cp1t-"


@dataclass(frozen=True)
class ProbeFamily:
    """One tokenizer family the frozen pp1 text can be retokenized under.

    Attributes:
        tag: Short family tag; the pool version is ``pp1t-{tag}``.
        tokenizer_repo: Canonical Hub repo the tokenizer comes from.
        tokenizer_revision: Pinned revision, or None for the repo default.
        bos: Whether the family's bases expect a leading BOS token.
        label: Human-readable name for tables/plots.
    """

    tag: str
    tokenizer_repo: str
    tokenizer_revision: str | None
    bos: bool
    label: str


PROBE_FAMILIES: dict[str, ProbeFamily] = {
    f.tag: f
    for f in (
        ProbeFamily("pythia", "EleutherAI/pythia-160m", None, False,
                    "PolyPythias (GPT-NeoX tokenizer)"),
        # The pinned DataDecide branch, so the pp1t-olmo tokenizer is the same object the
        # dp1 pool and the DataDecide bases use.
        # Every DataDecide recipe shares this tokenizer, so the default-recipe repo names
        # the family's tokenizer for all of them; which CORPUS pool a base takes is its
        # recipe's (`probe_io.probe_pool_spec_for_repo`).
        ProbeFamily("olmo", dd_repo_id(DataDecideSpec()), dd_revision(DataDecideSpec()),
                    False, "DataDecide/OLMo tokenizer"),
        ProbeFamily("qwen25", "Qwen/Qwen2.5-1.5B", None, False, "Qwen2.5"),
        ProbeFamily("llama3", "meta-llama/Llama-3.2-1B", None, True, "Llama 3.2"),
        ProbeFamily("gemma3", "google/gemma-3-1b-pt", None, True, "Gemma 3"),
        ProbeFamily("gemma2", "google/gemma-2-2b", "c5ebcd40d208330abc697524c919956e692655cf", True, "Gemma 2"),
    )
}


# CORPUS pools (as opposed to the frozen-text retokenizations above): one per (family,
# pretraining corpus), enumerated here so pool-version validation knows them. pp1 = Pile,
# cp1 = DCLM held-out (the `dclm` local re-pretrainings, same Pythia tokenizer, different
# corpus), dp1 = Dolma v1.7. Which of a family's corpus pools a given base takes is decided
# from its spec in `coherence.probe_io.probe_pool_spec_for_repo`.
CORPUS_POOLS: dict[str, tuple[str, ...]] = {
    "pythia": ("pp1", "cp1"),
    # The olmo family owns ONE corpus pool per DataDecide data recipe: dolma1_7 bases are
    # measured on dp1, dclm-baseline ones on cp1t-olmo (cp1's DCLM text in the OLMo
    # tokenizer). `probe_io.probe_pool_spec_for_repo` routes on the spec's recipe.
    "olmo": ("dp1", "cp1t-olmo"),
}


def corpus_pools_for(tag: str) -> tuple[str, ...]:
    """The corpus pool versions of a family tag (empty for frozen-text-only families)."""
    return CORPUS_POOLS.get(tag, ())


def pool_version_for(tag: str) -> str:
    """The pool version string of a family tag, e.g. ``pp1t-qwen25``."""
    if tag not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family {tag!r}; known tags: {sorted(PROBE_FAMILIES)}")
    return f"{POOL_PREFIX}{tag}"


def cp1t_pool_version_for(tag: str) -> str:
    """The DCLM frozen-text pool version of a family tag, e.g. ``cp1t-olmo``."""
    if tag not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family {tag!r}; known tags: {sorted(PROBE_FAMILIES)}")
    return f"{CP1T_PREFIX}{tag}"


def wide_pool_version_for(tag: str) -> str:
    """The WIDE pool version string of a family tag, e.g. ``pp1w-qwen25``."""
    if tag not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family {tag!r}; known tags: {sorted(PROBE_FAMILIES)}")
    return f"{WIDE_PREFIX}{tag}"


def is_pp1t(version: str) -> bool:
    """True for a ``pp1t-<tag>`` pool version (the tag itself is not validated).

    Deliberately NARROW: False for ``pp1w-<tag>``, so every pp1t-only code path keeps
    its exact meaning. Use `is_frozen_text_pool` for "reads a frozen text artifact".
    """
    return version.startswith(POOL_PREFIX)


def is_pp1w(version: str) -> bool:
    """True for a ``pp1w-<tag>`` wide pool version (the tag itself is not validated)."""
    return version.startswith(WIDE_PREFIX)


def is_cp1t(version: str) -> bool:
    """True for a ``cp1t-<tag>`` pool version (the tag itself is not validated)."""
    return version.startswith(CP1T_PREFIX)


def is_frozen_text_pool(version: str) -> bool:
    """True for any frozen-text pool family (``pp1t-`` / ``pp1w-`` / ``cp1t-<tag>``)."""
    return is_pp1t(version) or is_pp1w(version) or is_cp1t(version)


def family_for_pool_version(version: str) -> ProbeFamily:
    """The family a ``pp1t-``/``pp1w-``/``cp1t-<tag>`` pool version names."""
    if not is_frozen_text_pool(version):
        raise ValueError(f"not a frozen-text pool version: {version!r}; expected "
                         f"'pp1t-<tag>', 'pp1w-<tag>' or 'cp1t-<tag>' with tag in "
                         f"{sorted(PROBE_FAMILIES)}")
    tag = version[len(POOL_PREFIX):]  # every frozen-text prefix is 5 chars
    if tag not in PROBE_FAMILIES:
        raise ValueError(f"unknown probe family {tag!r} in pool version {version!r}; "
                         f"known tags: {sorted(PROBE_FAMILIES)}")
    return PROBE_FAMILIES[tag]


def load_family_tokenizer(
    family: ProbeFamily,
    hf_cache_root: Path,
    *,
    repo_override: str | None = None,
) -> tuple["PreTrainedTokenizerBase", str]:
    """Load a family's tokenizer; returns ``(tokenizer, resolved_repo)``.

    `repo_override` replaces the repo (a mirror, or an ungated equivalent of a gated
    one) and is echoed back as the resolved repo so callers can record what was really
    used. The `olmo` family MUST go through `models.datadecide.load_datadecide_tokenizer`
    — a plain AutoTokenizer keeps the `token_type_ids` the legacy OLMo forward rejects.
    """
    repo = repo_override or family.tokenizer_repo
    if family.tag == "olmo" and repo_override is None:
        from subliminal_learning.models.datadecide import load_datadecide_tokenizer

        tok = load_datadecide_tokenizer(DataDecideSpec(), Path(hf_cache_root))
    else:
        from transformers import AutoTokenizer

        kwargs: dict = {"cache_dir": str(Path(hf_cache_root) / "hub")}
        if family.tokenizer_revision is not None and repo_override is None:
            kwargs["revision"] = family.tokenizer_revision
        tok = AutoTokenizer.from_pretrained(repo, **kwargs)
    if family.bos and tok.bos_token_id is None:
        raise ValueError(f"family {family.tag!r} requires a BOS token but the tokenizer "
                         f"loaded from {repo!r} has none")
    return tok, repo


# Which family a BASE MODEL repo belongs to, keyed by the repo's trailing path component.
# Explicit alias table so a new size (e.g. `Llama-3.2-3B`) is one line here and nothing
# else; Pythia and DataDecide names are matched by pattern below (they are open families).
BASE_REPO_FAMILY: dict[str, str] = {
    b.repo_id.split("/")[-1]: b.family for b in EXTERNAL_BASES
}


def family_for_base_repo(repo: str) -> ProbeFamily | None:
    """The probe family of a base model ref, or None if it belongs to no known family.

    Accepts a bare repo id, a local ref (`local/pythia-160m-weightc-seed1`) and any
    `@revision` suffix. `pythia-*` names are the pythia family and DataDecide names
    (either spelling) the olmo one; everything else goes through `BASE_REPO_FAMILY`.
    """
    from subliminal_learning.models.naming import family_of

    try:
        fam = family_of(repo)          # any spelling: token, ref, dir name (2026-09-08)
        tag = "olmo" if fam == "datadecide" else fam
    except ValueError:
        # Not a spelling the naming module recognizes: the historical name patterns.
        name = repo.split("@")[0].rstrip("/").split("/")[-1]
        tag = BASE_REPO_FAMILY.get(name)
        if tag is None:
            if name.startswith("pythia-"):
                tag = "pythia"
            elif name.startswith("DataDecide"):
                tag = "olmo"
            else:
                return None
    # A registry family with no probe family yet (a new external base before its pool is
    # wired) is "no known family" under this function's contract, not a KeyError.
    return PROBE_FAMILIES.get(tag)
