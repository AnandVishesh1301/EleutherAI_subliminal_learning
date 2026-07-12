"""PolyPythias 160m seed-model specs and labels for the template workflow.

Shared by animal selection (which polls the seed models) and teacher training
(which fine-tunes across them). Kept separate so both import one definition of
"the 6 data/weight-seed models" and one filesystem-label convention.
"""

from __future__ import annotations

from subliminal_learning.models.registry import PythiaModelSpec


def polypythia_160m_seed_specs() -> list[PythiaModelSpec]:
    """The 6 PolyPythias 160m data/weight-seed specs.

    Three `data`-variant seeds (fixed weight init, varying data order) and three
    `weight`-variant seeds (fixed data order, varying weight init), seeds 1-3 each.
    """
    return [
        PythiaModelSpec(size="160m", seed=seed, variant=variant)
        for variant in ("data", "weight")
        for seed in (1, 2, 3)
    ]


def spec_label(spec: PythiaModelSpec) -> str:
    """Short filesystem-safe label for a spec, e.g. 'data-seed1', 'weight-seed3'.

    Used to namespace per-base-model outputs (`full`, seed 0 -> 'full-seed0').
    """
    return f"{spec.variant}-seed{spec.seed}"


def parse_base_models(value: str) -> list[PythiaModelSpec]:
    """Parse a `--base-models` string into 160m `PythiaModelSpec`s.

    Accepts:
      - "seeds" / "all" -> the 6 data/weight-seed models.
      - "data" / "weight" -> that variant's three seeds (1-3).
      - a comma list of "<variant>-seed<N>" tokens, e.g. "data-seed1,weight-seed3".

    Args:
        value: The selector string.

    Returns:
        The selected specs, in the given order.

    Raises:
        ValueError: On an unrecognized/empty selector or malformed token.
    """
    v = value.strip().lower()
    if v in ("seeds", "all", "all6"):
        return polypythia_160m_seed_specs()
    if v in ("data", "weight"):
        return [PythiaModelSpec(size="160m", seed=s, variant=v) for s in (1, 2, 3)]

    specs: list[PythiaModelSpec] = []
    for token in v.split(","):
        token = token.strip()
        if not token:
            continue
        if "-seed" not in token:
            raise ValueError(
                f"bad base-model token {token!r}; expected e.g. 'data-seed1'"
            )
        variant, seed_str = token.split("-seed", 1)
        # Fail fast on values that would otherwise resolve to a nonexistent HF
        # repo and 404 mid-run (the Literal type on the spec isn't runtime-checked).
        if variant not in ("data", "weight"):
            raise ValueError(
                f"bad variant {variant!r} in {token!r}; must be 'data' or 'weight'"
            )
        try:
            seed = int(seed_str)
        except ValueError:
            raise ValueError(f"bad seed {seed_str!r} in {token!r}; must be 1-3") from None
        if seed not in (1, 2, 3):
            raise ValueError(f"bad seed {seed} in {token!r}; PolyPythias has seeds 1-3")
        specs.append(PythiaModelSpec(size="160m", seed=seed, variant=variant))
    if not specs:
        raise ValueError(f"no base models parsed from {value!r}")
    return specs
