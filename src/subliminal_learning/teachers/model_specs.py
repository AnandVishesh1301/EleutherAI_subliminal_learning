"""PolyPythias 160m seed-model specs and labels for the template workflow.

Shared by animal selection (which polls the seed models) and teacher training
(which fine-tunes across them). Kept separate so both import one definition of
"the 6 data/weight-seed models" and one filesystem-label convention.

The original `pythia-160m` (variant "full", seed 0; label "full-seed0") is also
selectable via the "original" token. The `pythia-160m-seed{N}` variants (both
factors re-randomized together) became selectable on 2026-08-18 — a stance
change: they were previously rejected because the template workflow varies the
two factors separately. They must be requested EXPLICITLY, as `full-seed<N>` or
`pythia-160m-seed<N>`; a bare `seed<N>` token stays invalid so a fully
re-randomized model can never be mistaken for a data-/weight-seed one. Their
filesystem/HF identity is `pythia-160m-seed{N}` (repo basename), disjoint from
`pythia-160m-data-seed{N}` / `pythia-160m-weight-seed{N}`.
"""

from __future__ import annotations

import json

from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import (
    POLYPYTHIAS_FULL_SEED_SIZES,
    POLYPYTHIAS_FULL_SEEDS,
    PYTHIA_SIZES,
    PythiaModelSpec,
)


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

    The shared baseline of the `full` family. Its extra-seed siblings
    (`pythia-160m-seed{N}`, weight init AND data order re-randomized together)
    are selectable since 2026-08-18 via explicit `full-seed<N>` /
    `pythia-160m-seed<N>` tokens in `parse_base_models`.
    """
    return PythiaModelSpec(size="160m", seed=0, variant="full", revision=revision)


def family_swap_spec(spec: PythiaModelSpec) -> PythiaModelSpec:
    """Matched-seed PolyPythia family swap: data-seedK ↔ weight-seedK.

    Used by GKD cross-init distillation (student on one family, teacher from the
    other at the same seed index). Size and revision are preserved.

    Args:
        spec: A 160m ``data`` or ``weight`` variant spec.

    Returns:
        The sibling-family spec at the same seed.

    Raises:
        ValueError: If `spec.variant` is not ``data`` or ``weight``.
    """
    if spec.variant == "data":
        return PythiaModelSpec(
            size=spec.size, seed=spec.seed, variant="weight", revision=spec.revision
        )
    if spec.variant == "weight":
        return PythiaModelSpec(
            size=spec.size, seed=spec.seed, variant="data", revision=spec.revision
        )
    raise ValueError(
        f"family_swap requires variant 'data' or 'weight', got {spec.variant!r}"
    )


def _full_seed_spec(
    token: str, size: str, seed_str: str, revision: str | None, allowed: bool
) -> PythiaModelSpec:
    """Validate + build a fully re-randomized `full`-variant extra-seed spec.

    The single gate for both token spellings (`full-seed<N>`, `pythia-<size>-seed<N>`):
    the size must ship seed ensembles (`POLYPYTHIAS_FULL_SEED_SIZES`) and the seed
    must be in `POLYPYTHIAS_FULL_SEEDS` (1-9). Resolves to the HF repo
    `pythia-<size>-seed<N>` — weight init AND data order re-randomized together,
    unlike the single-factor data-/weight-seed models.

    `allowed` is the caller's `allow_full_seeds` opt-in; without it even a
    correctly spelled token is rejected, so a full-seed base is always a
    double-explicit choice (spelling AND force flag).
    """
    if not allowed:
        raise ValueError(
            f"full-seed model {token!r} is opt-in: these re-randomize weight init "
            "AND data order together (a third base kind, distinct from "
            "data-seed<N>/weight-seed<N>). Pass allow_full_seeds=True "
            "(CLI: --allow-full-seed-bases) to confirm this is deliberate"
        )
    try:
        seed = int(seed_str)
    except ValueError:
        raise ValueError(
            f"bad seed {seed_str!r} in {token!r}; expected an integer in "
            f"{POLYPYTHIAS_FULL_SEEDS[0]}-{POLYPYTHIAS_FULL_SEEDS[-1]}"
        ) from None
    if size not in POLYPYTHIAS_FULL_SEED_SIZES:
        raise ValueError(
            f"bad token {token!r}: size {size!r} has no full-seed ensemble; "
            f"PolyPythias ships extra full seeds at "
            f"{', '.join(POLYPYTHIAS_FULL_SEED_SIZES)} only"
        )
    if seed not in POLYPYTHIAS_FULL_SEEDS:
        raise ValueError(
            f"bad seed {seed} in {token!r}; PolyPythias full-seed models have "
            f"seeds {POLYPYTHIAS_FULL_SEEDS[0]}-{POLYPYTHIAS_FULL_SEEDS[-1]}"
        )
    return PythiaModelSpec(size=size, seed=seed, variant="full", revision=revision)


def parse_base_models(value: str, *, allow_full_seeds: bool = False) -> list[PythiaModelSpec]:
    """Parse a `--base-models` string into `PythiaModelSpec`s.

    Primarily targets the 160m data/weight-seed workflow, but a `pythia-<size>`
    token also selects the `full`, seed-0 model at any PolyPythias size.

    Accepts:
      - "seeds" / "all" -> the 6 data/weight-seed models.
      - "all7" -> the original `pythia-160m` plus the 6 data/weight-seed models.
      - "data" / "weight" -> that variant's three seeds (1-3).
      - a comma list of "<variant>-seed<N>" tokens, e.g. "data-seed1,weight-seed3".
        "original" (aliases: "pythia-160m", "full-seed0") selects the original
        `pythia-160m`. A `pythia-<size>` token (size in the PolyPythias release
        set, e.g. "pythia-2.8b") selects that size's `full`, seed-0 model; the
        data/weight seed ensembles only exist at 160m. An unknown size fails
        fast with a ValueError rather than a mid-run HF 404. Each token may be
        suffixed with "@<revision>" to pin a pretraining checkpoint, e.g.
        "data-seed1@step3000". The revision is any nonempty string after "@".
      - "full-seed<N>" or "pythia-<size>-seed<N>" -> a fully re-randomized
        PolyPythias model (`full` variant, weight init AND data order both
        re-randomized; `full-seed<N>` is 160m shorthand). Selectable since
        2026-08-18 (previously rejected). Valid only for the sizes that ship
        seed ensembles (`POLYPYTHIAS_FULL_SEED_SIZES`) and seeds 1-9
        (`POLYPYTHIAS_FULL_SEEDS`). These vary BOTH factors at once, unlike
        `data-seed<N>` / `weight-seed<N>`, and are double-gated: the token must
        be spelled explicitly (a bare "seed<N>" is invalid, and no group
        selector — "seeds", "all", "all7" — ever includes them) AND the caller
        must pass `allow_full_seeds=True` (launch CLIs expose this as
        `--allow-full-seed-bases`; inverters of existing on-disk dir names may
        opt in unconditionally, since a directory that exists is already an
        unambiguous identity).

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
        if token.startswith("pythia-"):
            # `pythia-<size>` selects the `full`, seed-0 model at that size (the
            # shared baseline for larger sizes, which have no data/weight seeds).
            # `pythia-160m` is handled above as an alias of "original".
            # `pythia-<size>-seed<N>` selects a fully re-randomized extra-seed
            # model (2026-08-18); `pythia-<size>-data-seed<N>` etc. are NOT repo
            # tokens here — use the bare `data-seed<N>` form for those.
            size = token[len("pythia-") :]
            if "-seed" in size:
                size, seed_str = size.split("-seed", 1)
                if size.endswith(("-data", "-weight")):
                    raise ValueError(
                        f"bad token {token!r}; the data/weight ensembles are "
                        "selected as bare 'data-seed<N>' / 'weight-seed<N>' "
                        "tokens, not by HF repo name"
                    )
                specs.append(
                    _full_seed_spec(token, size, seed_str, revision, allow_full_seeds)
                )
                continue
            if size not in PYTHIA_SIZES:
                raise ValueError(
                    f"bad size {size!r} in {token!r}; valid Pythia sizes are "
                    f"{', '.join(PYTHIA_SIZES)}"
                )
            specs.append(
                PythiaModelSpec(size=size, seed=0, variant="full", revision=revision)
            )
            continue
        if "-seed" not in token:
            raise ValueError(
                f"bad base-model token {token!r}; expected e.g. 'data-seed1' or 'original'"
            )
        variant, seed_str = token.split("-seed", 1)
        # Fail fast on values that would otherwise resolve to a nonexistent HF
        # repo and 404 mid-run (the Literal type on the spec isn't runtime-checked).
        if variant == "full":
            # `full-seed0` = the original; `full-seed<N>` (N>=1) = the fully
            # re-randomized 160m models, in scope since 2026-08-18.
            if seed_str == "0":
                specs.append(original_160m_spec(revision=revision))
            else:
                specs.append(
                    _full_seed_spec(token, "160m", seed_str, revision, allow_full_seeds)
                )
            continue
        if variant not in ("data", "weight"):
            raise ValueError(
                f"bad variant {variant!r} in {token!r}; must be 'data', 'weight', "
                "or 'full' (seed 0 only, the original pythia-160m)"
            )
        try:
            seed = int(seed_str)
        except ValueError:
            raise ValueError(
                f"bad seed {seed_str!r} in {token!r}; must be 1-3"
            ) from None
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
