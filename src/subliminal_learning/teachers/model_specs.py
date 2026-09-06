"""PolyPythias model specs and labels for teacher/student campaign tokens.

Shared by animal selection, teacher training, and GKD student sweeps. One
definition of the 160m data/weight-seed family and one filesystem-label
convention. Size-aware Hub tokens (`pythia-70m@step143000`,
`pythia-410m@step143000`) resolve through `spec_from_base_name`.

The original `pythia-160m` (variant "full", seed 0; label "full-seed0") is
selectable via the "original" token. The joint `pythia-160m-seed{N}` variants
(init and data order both re-randomized; N in 1-9) are selectable via `seed{N}`
(or `full-seed{N}`). Non-160m `spec_label`s carry a size prefix so a 70m and
160m cell with the same variant+seed cannot share a `teacher_tag`.
"""

from __future__ import annotations

import json

from subliminal_learning.config import PathsConfig
from subliminal_learning.models.registry import PythiaModelSpec

# Closed set from `models.registry.PythiaModelSpec.size` — Hub repos exist only
# for these sizes. Reject anything else at parse time (no silent 404 mid-run).
_POLYPYTHIA_SIZES = frozenset({"14m", "31m", "70m", "160m", "410m"})


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
    """Short filesystem-safe label for a spec, e.g. 'data-seed1', '70m-full-seed0'.

    160m labels stay ``{variant}-seed{N}`` (``full-seed0``, ``data-seed1``). Other
    sizes prefix ``{size}-`` so a 70m and 160m cell with the same variant+seed
    cannot share a `teacher_tag` / control path. A non-None `spec.revision` is
    appended as `@{revision}`; revision=None is byte-identical to before.
    """
    if spec.size == "160m":
        label = f"{spec.variant}-seed{spec.seed}"
    else:
        label = f"{spec.size}-{spec.variant}-seed{spec.seed}"
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


def parse_base_models(value: str) -> list[PythiaModelSpec]:
    """Parse a `--base-models` string into `PythiaModelSpec`s.

    Accepts:
      - "seeds" / "all" -> the 6 data/weight-seed models.
      - "all7" -> the original `pythia-160m` plus the 6 data/weight-seed models.
      - "data" / "weight" -> that variant's three seeds (1-3).
      - a comma list of "<variant>-seed<N>" tokens, e.g. "data-seed1,weight-seed3".
        "original" (aliases: "pythia-160m", "full-seed0") selects the original
        `pythia-160m`. `seed{N}` / `full-seed{N}` (N in 1-9) select the joint
        `pythia-160m-seed{N}` repos. Each token may be suffixed with
        "@<revision>" to pin a pretraining checkpoint, e.g. "seed1@step2000".
        The revision is any nonempty string after "@".
      - a Hub basename starting with ``pythia-`` (optionally org-prefixed),
        e.g. ``pythia-70m@step143000``, ``EleutherAI/pythia-410m@step143000``.
        Delegates to `spec_from_base_name`; unknown sizes raise.

    `seed{N}` / `full-seed{N}` (N in 1-9) select the joint
    `pythia-160m-seed{N}` Hub repos (init and data order both vary). `seed0`
    is rejected; the original repo is `original` / `full-seed0`.

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
        if token.startswith("seed") and "-seed" not in token:
            seed_str = token[len("seed") :]
            try:
                seed = int(seed_str)
            except ValueError:
                raise ValueError(
                    f"bad seed {seed_str!r} in {token!r}; must be 1-9"
                ) from None
            if seed not in range(1, 10):
                raise ValueError(
                    f"bad seed {seed} in {token!r}; seed{{N}} accepts N in 1-9"
                )
            specs.append(
                PythiaModelSpec(
                    size="160m", seed=seed, variant="full", revision=revision
                )
            )
            continue
        hub_token = token.rsplit("/", 1)[-1]
        if hub_token.startswith("pythia-"):
            rebuilt = token if revision is None else f"{token}@{revision}"
            specs.append(spec_from_base_name(rebuilt))
            continue
        if "-seed" not in token:
            raise ValueError(
                f"bad base-model token {token!r}; expected e.g. 'data-seed1' "
                "or 'original' or 'seed1' or 'pythia-70m@step143000'"
            )
        variant, seed_str = token.split("-seed", 1)
        # Fail fast on values that would otherwise resolve to a nonexistent HF
        # repo and 404 mid-run (the Literal type on the spec isn't runtime-checked).
        if variant == "full":
            if seed_str == "0":
                specs.append(original_160m_spec(revision=revision))
                continue
            try:
                seed = int(seed_str)
            except ValueError:
                raise ValueError(
                    f"bad seed {seed_str!r} in {token!r}; must be 1-9"
                ) from None
            if seed not in range(1, 10):
                raise ValueError(
                    f"bad seed {seed} in {token!r}; full-seed{{N}} accepts N in 1-9"
                )
            specs.append(
                PythiaModelSpec(
                    size="160m", seed=seed, variant="full", revision=revision
                )
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


def spec_from_base_name(name: str) -> PythiaModelSpec:
    """Inverse of ``model_dir_name``/``resolve_hf_repo_id``: a repo id or model dir
    name back to a ``PythiaModelSpec``.

    Accepts ``EleutherAI/pythia-160m-data-seed1`` or the bare basename
    ``pythia-160m-weight-seed3``, with an optional ``@{revision}`` suffix. Handles the
    full (``pythia-{size}`` / ``pythia-{size}-seed{N}``) and 160m data/weight variants.

    This is the repo-name label space; `parse_base_models` is the campaign-token one
    (``original`` / ``data-seed1``). Both live here so revision-aware parsing has one
    home and callers that only need string logic do not pull the model-loading stack.
    """
    base = name.rsplit("/", 1)[-1]
    revision: str | None = None
    if "@" in base:
        base, revision = base.split("@", 1)
        if not revision:
            raise ValueError(f"empty revision after '@' in base name {name!r}")
    if not base.startswith("pythia-"):
        raise ValueError(f"cannot map base name {name!r} to a PythiaModelSpec")
    rest = base[len("pythia-") :]
    parts = rest.split("-")
    size = parts[0]
    if size not in _POLYPYTHIA_SIZES:
        raise ValueError(
            f"unknown PolyPythia size {size!r} in base name {name!r}; "
            f"expected one of {sorted(_POLYPYTHIA_SIZES)}"
        )
    if len(parts) == 1:
        return PythiaModelSpec(size=size, seed=0, variant="full", revision=revision)
    tail = "-".join(parts[1:])
    for prefix, variant in (
        ("data-seed", "data"),
        ("weight-seed", "weight"),
        ("seed", "full"),
    ):
        if tail.startswith(prefix):
            seed_str = tail[len(prefix) :]
            try:
                seed = int(seed_str)
            except ValueError:
                raise ValueError(
                    f"bad seed {seed_str!r} in base name {name!r}"
                ) from None
            return PythiaModelSpec(
                size=size, seed=seed, variant=variant, revision=revision
            )
    raise ValueError(f"cannot map base name {name!r} to a PythiaModelSpec")


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
