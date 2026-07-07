"""PolyPythias model spec and Hub repo-id resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PolyPythiaVariant = Literal["full", "data", "weight"]
# full   = both weight init + data order vary (seeds 0-9)
# data   = 160M only: fixed weight init (seed 0), varying data order (seeds 1-3)
# weight = 160M only: fixed data order (seed 0), varying weight init (seeds 1-3)


@dataclass(frozen=True)
class PythiaModelSpec:
    """Identifies one PolyPythias model checkpoint on the Hugging Face Hub.

    Attributes:
        size: PolyPythias size: "14m" | "31m" | "70m" | "160m" | "410m".
        seed: PolyPythias seed (0 = default seed 1234; 1-9 = extra seeds).
        variant: "full" (default), "data", "weight".
        revision: Training checkpoint, e.g. "step3000". None -> final checkpoint.
    """

    size: str
    seed: int = 0
    variant: PolyPythiaVariant = "full"
    revision: str | None = None


def resolve_hf_repo_id(spec: PythiaModelSpec, hub_org: str) -> str:
    """Resolve a PolyPythias spec to its Hugging Face Hub repo id.

    Repo id format:
        full,   seed 0  -> {hub_org}/pythia-{size}
        full,   seed N  -> {hub_org}/pythia-{size}-seed{N}
        data,   seed N  -> {hub_org}/pythia-160m-data-seed{N}   (160m only, N in 1-3)
        weight, seed N  -> {hub_org}/pythia-160m-weight-seed{N} (160m only, N in 1-3)

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
    return f"{hub_org}/pythia-160m-{spec.variant}-seed{spec.seed}"
