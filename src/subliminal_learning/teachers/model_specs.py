"""PolyPythias 160m seed-model specs and labels for the template workflow.

Shared by animal selection (which polls the seed models) and teacher training
(which fine-tunes across them). Kept separate so both import one definition of
"the 6 data/weight-seed models" and one filesystem-label convention.

The original `pythia-160m` (variant "full", seed 0; label "full-seed0") is also
selectable via the "original" token; the `pythia-160m-seed{N}` variants (both
factors re-randomized together) are deliberately not.
"""

from __future__ import annotations

import json

from subliminal_learning.config import PathsConfig
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

    Used to namespace per-base-model outputs (`full`, seed 0 -> 'full-seed0'). A
    non-None `spec.revision` is appended as `@{revision}` (e.g. 'data-seed1@step3000');
    revision=None is byte-identical to before.
    """
    label = f"{spec.variant}-seed{spec.seed}"
    if spec.revision is not None:
        return f"{label}@{spec.revision}"
    return label


def original_160m_spec(revision: str | None = None) -> PythiaModelSpec:
    """The original `pythia-160m` (default seed 1234): variant 'full', seed 0.

    The only in-scope member of the `full` family — the `pythia-160m-seed{N}`
    variants (weight init AND data order re-randomized together) are deliberately
    excluded from the template workflow, which studies the two factors separately.
    """
    return PythiaModelSpec(size="160m", seed=0, variant="full", revision=revision)


def parse_base_models(value: str) -> list[PythiaModelSpec]:
    """Parse a `--base-models` string into 160m `PythiaModelSpec`s.

    Accepts:
      - "seeds" / "all" -> the 6 data/weight-seed models.
      - "all7" -> the original `pythia-160m` plus the 6 data/weight-seed models.
      - "data" / "weight" -> that variant's three seeds (1-3).
      - a comma list of "<variant>-seed<N>" tokens, e.g. "data-seed1,weight-seed3".
        "original" (aliases: "pythia-160m", "full-seed0") selects the original
        `pythia-160m`. Each token may be suffixed with "@<revision>" to pin a
        pretraining checkpoint, e.g. "data-seed1@step3000". The revision is any
        nonempty string after "@".

    The `pythia-160m-seed{N}` models ("full" variant, N >= 1, weight init and
    data order re-randomized together) are rejected: the workflow varies the two
    factors separately, and the original model is the shared baseline.

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
    if v == "all7":
        return [original_160m_spec(), *polypythia_160m_seed_specs()]
    if v in ("data", "weight"):
        return [PythiaModelSpec(size="160m", seed=s, variant=v) for s in (1, 2, 3)]

    specs: list[PythiaModelSpec] = []
    # Iterate the ORIGINAL value so an `@<revision>` suffix keeps its case (HF
    # revisions can be case-sensitive); only the variant-seed part is lowercased.
    for raw_token in value.split(","):
        raw_token = raw_token.strip()
        if not raw_token:
            continue
        revision: str | None = None
        if "@" in raw_token:
            raw_token, revision = raw_token.split("@", 1)
            if not revision:
                raise ValueError(
                    f"empty revision after '@' in base-model token {raw_token + '@'!r}; "
                    "expected e.g. 'data-seed1@step3000'"
                )
        token = raw_token.lower()
        if token in ("original", "pythia-160m"):
            specs.append(original_160m_spec(revision=revision))
            continue
        if "-seed" not in token:
            raise ValueError(
                f"bad base-model token {token!r}; expected e.g. 'data-seed1' or 'original'"
            )
        variant, seed_str = token.split("-seed", 1)
        # Fail fast on values that would otherwise resolve to a nonexistent HF
        # repo and 404 mid-run (the Literal type on the spec isn't runtime-checked).
        if variant == "full":
            if seed_str != "0":
                raise ValueError(
                    f"bad token {token!r}: the pythia-160m-seed{{N}} variants "
                    "(weight init and data order re-randomized together) are out "
                    "of scope; use 'original' (= full-seed0), 'data-seed<N>', or "
                    "'weight-seed<N>'"
                )
            specs.append(original_160m_spec(revision=revision))
            continue
        if variant not in ("data", "weight"):
            raise ValueError(
                f"bad variant {variant!r} in {token!r}; must be 'data', 'weight', "
                "or 'full' (seed 0 only, the original pythia-160m)"
            )
        try:
            seed = int(seed_str)
        except ValueError:
            raise ValueError(f"bad seed {seed_str!r} in {token!r}; must be 1-3") from None
        if seed not in (1, 2, 3):
            raise ValueError(f"bad seed {seed} in {token!r}; PolyPythias has seeds 1-3")
        specs.append(
            PythiaModelSpec(size="160m", seed=seed, variant=variant, revision=revision)
        )
    if not specs:
        raise ValueError(f"no base models parsed from {value!r}")
    return specs


def resolve_animals(spec: object, paths: PathsConfig, n: int) -> list[str]:
    """Resolve a sweep YAML's `animals:` field to a concrete list of animals.

    The `animals:` axis, like `base_models:`, may be given two ways; this is the one
    resolver shared by every sweep runner (training / verification / number generation).

    Accepts:
      - an explicit list (e.g. ``[cat, dog, lion]``) -> used verbatim.
      - an ordering-name string (e.g. ``"by_lowest_max_rank"``) -> looked up in the
        `select_animals` output JSON at
        ``{output_root}/animal_selection/top{n}_animals.json`` and sliced to `n`.

    Args:
        spec: The YAML `animals:` value (list or ordering-name string).
        paths: Paths config (for `output_root`, where the selection JSON lives).
        n: `n_animals` -- selects the file and slices the ordering.

    Returns:
        The resolved animal names, in order.

    Raises:
        FileNotFoundError: An ordering name was given but the selection file is absent.
        ValueError: The ordering name is not present in the selection file.
    """
    if isinstance(spec, list):
        return [str(a) for a in spec]

    ordering = str(spec)
    selection = paths.output_root / "animal_selection" / f"top{n}_animals.json"
    if not selection.exists():
        raise FileNotFoundError(
            f"animals='{ordering}' needs the selection file {selection}, which does "
            f"not exist. Run it first: uv run python -m scripts.select_animals --n {n}"
        )
    orderings = json.loads(selection.read_text())["orderings"]
    if ordering not in orderings:
        raise ValueError(
            f"ordering {ordering!r} not in {selection} (have {list(orderings)})"
        )
    return list(orderings[ordering])[:n]
