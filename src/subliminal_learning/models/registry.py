"""PolyPythias model spec and Hub repo-id resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from subliminal_learning.models.datadecide import (
    DataDecideSpec,
    dd_model_ref,
    dd_repo_id,
)

PolyPythiaVariant = Literal["full", "data", "weight", "weightc"]
# full    = both weight init + data order vary (seeds 0-9)
# data    = 160M only: fixed weight init (seed 0), varying data order (seeds 1-3)
# weight  = 160M only: fixed data order (seed 0), varying weight init (seeds 1-3)
# weightc = 160M only: corrected weight-seed re-pretraining (the `weight` family
#           re-run with the intended data order). Locally converted weights only:
#           these models are NOT on the Hub, so they resolve to a `local/...`
#           provenance string and load from `hf_cache_root/local/...`.

# Canonical Pythia release names selectable as `pythia-<name>` tokens (validated
# here before resolving to a Hub repo id). The `full` family exists at every size;
# the data/weight seed variants only exist at 160m. The `<size>-deduped` siblings
# are deliberately NOT listed. `spec_label` is now size-aware for non-160m `full`
# specs ('pythia-410m', 'pythia-410m-seed3'), so the old `full-seed0` collision no
# longer applies; what is still missing is a token spelling + cross-init pairing
# helper for deduped (`family_swap_spec` has no size->deduped mapping). Re-add them
# only together with those.
# Sizes at which the PolyPythias `full` family has extra-seed models
# (`pythia-<size>-seed{N}`, N in FULL_SEEDS: weight init AND data order
# re-randomized together). Other sizes exist only at the default seed 0.
POLYPYTHIAS_FULL_SEED_SIZES: tuple[str, ...] = ("14m", "31m", "70m", "160m", "410m")
POLYPYTHIAS_FULL_SEEDS: tuple[int, ...] = tuple(range(1, 10))

PYTHIA_SIZES: tuple[str, ...] = (
    "14m",
    "31m",
    "70m",
    "160m",
    "410m",
)


@dataclass(frozen=True)
class PythiaModelSpec:
    """Identifies one PolyPythias model checkpoint on the Hugging Face Hub.

    Attributes:
        size: PolyPythias size: "14m" | "31m" | "70m" | "160m" | "410m".
        seed: PolyPythias seed (0 = default seed 1234; 1-9 = extra seeds).
        variant: "full" (default), "data", "weight", "weightc" (local-only).
        revision: Training checkpoint, e.g. "step3000". None -> final checkpoint.
    """

    size: str
    seed: int = 0
    variant: PolyPythiaVariant = "full"
    revision: str | None = None


@dataclass(frozen=True)
class ExternalBase:
    """One non-Pythia, non-DataDecide base model this repo supports.

    Attributes:
        repo_id: Hugging Face repo id.
        alias: `--base-models` CLI token selecting it.
        family: Probe/tokenizer family tag (shared by sizes of one family).
        model_type: The HF `config.model_type` of the loaded model.
        attn_implementation: `from_pretrained` kwarg, or None for the default.
            Gemma-2's sliding-window/soft-capping attention needs "eager".
    """

    repo_id: str
    alias: str
    family: str
    model_type: str
    attn_implementation: str | None = None


# SINGLE SOURCE OF TRUTH for the external bases. Alias parsing (`teachers.model_specs`),
# the probe family table (`coherence.probe_families`), the LoRA family detection
# (`teachers.sft_io`), the loader kwargs (`models.loading_io`) and the strict
# cross-family distillation guard (`pipelines.distill_student_pipeline`) all derive from
# it: adding a base is one row here.
EXTERNAL_BASES: tuple[ExternalBase, ...] = (
    ExternalBase("Qwen/Qwen2.5-1.5B", "qwen25-1.5b", "qwen25", "qwen2"),
    ExternalBase("meta-llama/Llama-3.2-1B", "llama3-1b", "llama3", "llama"),
    ExternalBase("google/gemma-3-1b-pt", "gemma3-1b", "gemma3", "gemma3_text"),
    ExternalBase("google/gemma-2-2b", "gemma2-2b", "gemma2", "gemma2", "eager"),
    ExternalBase("google/gemma-2-9b", "gemma2-9b", "gemma2", "gemma2", "eager"),
)
EXTERNAL_BASE_BY_REPO: dict[str, ExternalBase] = {b.repo_id: b for b in EXTERNAL_BASES}
EXTERNAL_FAMILY_TAGS: frozenset[str] = frozenset(b.family for b in EXTERNAL_BASES)


def external_attn_kwargs(repo_id: str) -> dict[str, str]:
    """`from_pretrained` attention kwargs for an external base repo (often empty)."""
    base = EXTERNAL_BASE_BY_REPO.get(repo_id)
    if base is None or base.attn_implementation is None:
        return {}
    return {"attn_implementation": base.attn_implementation}


@dataclass(frozen=True)
class ExternalModelSpec:
    """One explicitly supported base, with a filesystem-safe checkpoint revision."""

    repo_id: str
    revision: str | None = None

    def __post_init__(self) -> None:
        if self.repo_id not in EXTERNAL_BASE_BY_REPO:
            raise ValueError(f"unsupported external model repository {self.repo_id!r}")
        if self.revision is not None and (
            not self.revision or any(c in self.revision for c in "/\\@")
            or any(c.isspace() for c in self.revision)
        ):
            raise ValueError(f"invalid external model revision {self.revision!r}")


BaseModelSpec: TypeAlias = PythiaModelSpec | DataDecideSpec | ExternalModelSpec


def resolve_hf_repo_id(spec: PythiaModelSpec, hub_org: str) -> str:
    """Resolve a PolyPythias spec to its Hugging Face Hub repo id.

    Repo id format:
        full,   seed 0  -> {hub_org}/pythia-{size}
        full,   seed N  -> {hub_org}/pythia-{size}-seed{N}
        data,   seed N  -> {hub_org}/pythia-160m-data-seed{N}   (160m only, N in 1-3)
        weight, seed N  -> {hub_org}/pythia-160m-weight-seed{N} (160m only, N in 1-3)
        weightc, seed N -> local/pythia-160m-weightc-seed{N}    (160m only, N in 1-3)

    The `weightc` family is deliberately org-less: it is an unpublished local
    conversion, and the `local/` prefix is a non-hub provenance string (the slash
    is kept so manifests still read like refs). `local_model_dir` turns it into
    an on-disk directory; it must never be handed to the Hub.

    Args:
        spec: Model size/seed/variant/revision selection.
        hub_org: Hub org name, e.g. "EleutherAI".

    Returns:
        Repo id, e.g. "EleutherAI/pythia-410m" or
        "EleutherAI/pythia-160m-data-seed1".

    Raises:
        ValueError: If a data/weight variant is requested for a size other
            than "160m", or with a seed outside 1-3.
    """
    if isinstance(spec, ExternalModelSpec):
        return spec.repo_id
    if isinstance(spec, DataDecideSpec):
        return dd_repo_id(spec)
    if spec.variant == "full":
        if spec.seed == 0:
            return f"{hub_org}/pythia-{spec.size}"
        return f"{hub_org}/pythia-{spec.size}-seed{spec.seed}"
    if spec.size != "160m":
        raise ValueError(f"variant={spec.variant!r} is only valid for size='160m'")
    if spec.seed not in (1, 2, 3):
        raise ValueError(
            f"variant={spec.variant!r} requires seed in (1, 2, 3), got {spec.seed}"
        )
    if spec.variant == "weightc":
        return f"local/pythia-160m-weightc-seed{spec.seed}"
    return f"{hub_org}/pythia-160m-{spec.variant}-seed{spec.seed}"


def local_model_dir(
    repo_id: str, revision: str | None, hf_cache_root: Path
) -> Path | None:
    """On-disk directory for an UNPUBLISHED, locally converted model, else None.

    Only `local/...` repo ids (currently the `weightc` family) have one: those
    weights exist nowhere on the Hugging Face Hub, so they must never be passed to
    a hub `from_pretrained`. Layout, one complete HF model + tokenizer per dir:

        {hf_cache_root}/local/{basename}/{revision or "main"}/

    Returns None for every hub repo id, so callers can use it as a routing test.
    """
    if not repo_id.startswith("local/"):
        return None
    return hf_cache_root / "local" / repo_id.split("/")[-1] / (revision or "main")


def model_ref(spec: PythiaModelSpec, hub_org: str) -> str:
    """Repo id plus a `@{revision}` suffix identifying the exact checkpoint.

    Identical to `resolve_hf_repo_id` when `spec.revision` is None (so revision-less
    runs are byte-identical to their historical identity/labels); with a revision it
    appends `@{revision}`, e.g. `EleutherAI/pythia-160m-data-seed1@step3000`.

    This is the value to record in run-identity dicts and manifests, so two runs off
    different pretraining checkpoints of the same repo are distinct. HF loaders still
    pass the repo id and revision separately, so `resolve_hf_repo_id` is unchanged.
    """
    if isinstance(spec, ExternalModelSpec):
        return spec.repo_id + (f"@{spec.revision}" if spec.revision is not None else "")
    if isinstance(spec, DataDecideSpec):
        return dd_model_ref(spec)
    repo = resolve_hf_repo_id(spec, hub_org)
    if spec.revision is not None:
        return f"{repo}@{spec.revision}"
    return repo
