"""ONE place that converts between the spellings of a base model.

A base model is spelled four ways across the repo, and only the Pythia family happens to
spell two of them identically -- which is how "token == directory name" assumptions crept
into scripts and workers (audit 2026-09-08):

    spelling   example (DataDecide)                              example (Qwen)     example (Pythia)
    token      dd-1b-seed2                                       qwen25-1.5b        pythia-160m-seed1
    dir name   DataDecide-dolma1_7-1B-seed-large-aux-2           Qwen2.5-1.5B       pythia-160m-seed1
    ref        allenai/DataDecide-dolma1_7-1B@step69369-seed-large-aux-2   Qwen/Qwen2.5-1.5B   EleutherAI/pythia-160m-seed1
    label      dd-1b-seed2 (`spec_label`)                        Qwen2.5-1.5B       full-seed1

Every function here goes THROUGH a spec: parse whatever spelling arrives, then emit the
requested spelling with the writer the layouts already use (`model_dir_name`, `model_ref`,
...). No new spelling is minted, no directory is renamed, and each converter is idempotent
on its own output. Callers that hold a spec can pass it directly.

Grammar owners stay where they are (`models/datadecide.py` for `dd-*` tokens and refs,
`teachers/model_specs.parse_base_models` for Pythia/external tokens, `models/registry.
EXTERNAL_BASES` for external aliases); this module only routes between them and adds the
two inverters that did not exist: directory name -> spec for DataDecide and external bases.
"""

from __future__ import annotations

from typing import Any

from subliminal_learning.models.datadecide import (
    DD_1B_FINAL_STEP,
    DD_HUB_ORG,
    DD_RECIPES,
    DD_SEED_BRANCHES,
    DataDecideSpec,
    dd_token,
    is_dd_ref,
    parse_dd_ref,
    parse_dd_token,
)
from subliminal_learning.models.registry import (
    EXTERNAL_BASE_BY_REPO,
    LOCAL_PYTHIA_FAMILIES,
    EXTERNAL_BASES,
    ExternalModelSpec,
    PythiaModelSpec,
    model_ref,
)

#: Hub org of the PolyPythias family. Pythia dir names never contain the org, so the org
#: only matters when a Pythia spelling is turned INTO a ref; the config YAML's `hub_org`
#: is this value everywhere in the repo.
DEFAULT_HUB_ORG = "EleutherAI"

BaseSpec = PythiaModelSpec | DataDecideSpec | ExternalModelSpec

_DD_DIR_PREFIX = "DataDecide-"
_EXTERNAL_BY_ALIAS = {b.alias: b for b in EXTERNAL_BASES}
# Basenames keyed lowercase, like `model_specs._EXTERNAL_BASENAMES`, so `qwen2.5-1.5b` and
# `Qwen2.5-1.5B` both resolve (the dir on disk keeps the repo's own case).
_EXTERNAL_BY_BASENAME = {b.repo_id.split("/")[-1].lower(): b for b in EXTERNAL_BASES}


def _split_revision(text: str) -> tuple[str, str | None]:
    if "@" not in text:
        return text, None
    head, rev = text.split("@", 1)
    if not rev:
        raise ValueError(f"empty revision after '@' in base spelling {text!r}")
    return head, rev


def _dd_spec_from_dir_name(name: str) -> DataDecideSpec:
    """Invert `dd_dir_name`: `DataDecide-{recipe}-{size}-seed-{branch}[@step{N}]`.

    `@step{DD_1B_FINAL_STEP}` canonicalizes to step=None, exactly as `parse_dd_token` and
    `parse_dd_ref` do, so the three spellings of the final checkpoint are ONE spec.
    """
    head, rev = _split_revision(name)
    body, sep, branch = head.partition("-seed-")
    if not sep or not body.startswith(_DD_DIR_PREFIX):
        raise ValueError(f"cannot map {name!r} to a DataDecide spec")
    recipe_size = body[len(_DD_DIR_PREFIX):]          # e.g. dolma1_7-1B, dclm-baseline-1B
    # Split the SIZE off the end: a recipe may itself contain '-' (`dclm-baseline`).
    recipe, sep, size = recipe_size.rpartition("-")
    if not sep:
        raise ValueError(f"cannot map {name!r} to a DataDecide spec (no size)")
    seeds = {v: k for k, v in DD_SEED_BRANCHES.items()}
    if branch not in seeds:
        raise ValueError(f"unknown DataDecide seed branch {branch!r} in {name!r}")
    if size != "1B" or recipe not in DD_RECIPES:
        # Only the 1B models of a wired recipe have a CLI token (`dd-1b[-<recipe>]`); a dir
        # name for any other size/recipe is a real DataDecide model this repo cannot name.
        raise ValueError(f"unsupported DataDecide size/recipe in {name!r}: only "
                         f"1B and recipes {DD_RECIPES} are wired")
    step: int | None = None
    if rev is not None:
        if not rev.startswith("step") or not rev[len("step"):].isdigit():
            raise ValueError(f"bad step suffix {rev!r} in DataDecide dir name {name!r}")
        step = int(rev[len("step"):])
        if step == DD_1B_FINAL_STEP:
            step = None
    return DataDecideSpec(size=size, seed=seeds[branch], step=step, recipe=recipe)


def _external_spec_from_name(name: str) -> ExternalModelSpec | None:
    """Alias, repo basename, or repo id (each optionally `@revision`) -> spec, else None."""
    head, rev = _split_revision(name)
    base = _EXTERNAL_BY_ALIAS.get(head.lower()) or _EXTERNAL_BY_BASENAME.get(head.lower())
    if base is None and head in EXTERNAL_BASE_BY_REPO:
        base = EXTERNAL_BASE_BY_REPO[head]
    if base is None:
        return None
    return ExternalModelSpec(base.repo_id, rev)


def _pythia_spec_from_name(name: str) -> PythiaModelSpec:
    """Token (`data-seed1`, `original`, `pythia-410m`, ...) or dir name / repo basename
    (`pythia-160m-data-seed1`, `pythia-160m-weightc-seed2`,
    `pythia-160m-dclm-seed1234`, ...) -> spec."""
    from subliminal_learning.teachers.model_specs import parse_base_models, spec_from_base_name

    head, rev = _split_revision(name)
    # 1) the CLI grammar (covers every token and the `pythia-<size>[-seed<N>]` names)
    try:
        specs = parse_base_models(name, allow_full_seeds=True)
        if len(specs) == 1:
            return specs[0]
    except ValueError:
        pass
    # 2) the on-disk dir-name grammar for the 160m data/weight ensembles
    try:
        return spec_from_base_name(name)
    except ValueError:
        pass
    # 3) the local conversions (`pythia-160m-<family>-seed<N>`, one row per family in
    #    `LOCAL_PYTHIA_FAMILIES`), which neither grammar above spells as a dir name
    for family in LOCAL_PYTHIA_FAMILIES:
        prefix = f"pythia-160m-{family.variant}-seed"
        if head.startswith(prefix):
            seed = head[len(prefix):]
            if seed.isdigit() and int(seed) in family.seeds:
                return PythiaModelSpec(
                    size="160m", seed=int(seed), variant=family.variant, revision=rev
                )
    raise ValueError(f"cannot map {name!r} to a Pythia base spec")


def base_spec(value: str | BaseSpec) -> BaseSpec:
    """Any spelling of a base model -> its spec.

    Accepts a spec (returned as is), a CLI token, an output-tree directory name, or a
    model ref / repo id, for the Pythia, DataDecide and external families. Comma lists and
    group selectors (`all7`, `seeds`) are NOT accepted: this names ONE base.

    Raises:
        ValueError: when no family's grammar recognizes the spelling.
    """
    if isinstance(value, (PythiaModelSpec, DataDecideSpec, ExternalModelSpec)):
        return value
    text = str(value).strip()
    if not text or "," in text:
        raise ValueError(f"expected one base-model spelling, got {value!r}")
    if "/" in text:                                   # a ref or repo id
        if is_dd_ref(text):
            spec = parse_dd_ref(text)
            if spec.size != "1B" or spec.recipe not in DD_RECIPES:
                raise ValueError(f"unsupported DataDecide size/recipe in ref {text!r}: only "
                                 f"1B and recipes {DD_RECIPES} are wired")
            return spec
        ext = _external_spec_from_name(text)
        if ext is not None:
            return ext
        head, rev = _split_revision(text)
        org, _, basename = head.rpartition("/")
        if org == "local" or basename.startswith("pythia-"):
            return _pythia_spec_from_name(basename + (f"@{rev}" if rev else ""))
        raise ValueError(f"cannot map ref {text!r} to a base spec")
    if text.lower().startswith("dd-"):                # DataDecide token
        return parse_dd_token(text)
    if text.startswith(_DD_DIR_PREFIX):               # DataDecide dir name
        return _dd_spec_from_dir_name(text)
    ext = _external_spec_from_name(text)              # external alias / basename
    if ext is not None:
        return ext
    return _pythia_spec_from_name(text)


def base_dir_name(value: str | BaseSpec, *, hub_org: str = DEFAULT_HUB_ORG) -> str:
    """Output-tree `{model}` directory name of a base, from any spelling.

    Identical to `teachers.run_layout.model_dir_name` on the parsed spec (imported lazily:
    `run_layout` imports config, which this module must not drag in at import time), and
    to `models.datadecide.base_dir_name_for_ref` on every ref this repo emits (tested).
    """
    from subliminal_learning.teachers.run_layout import model_dir_name

    return model_dir_name(base_spec(value), hub_org)


def base_ref(value: str | BaseSpec, *, hub_org: str = DEFAULT_HUB_ORG) -> str:
    """Manifest / run-identity model ref (`repo[@revision]`) of a base, from any spelling."""
    return model_ref(base_spec(value), hub_org)


def base_token(value: str | BaseSpec) -> str:
    """Canonical `--base-models` CLI token of a base, from any spelling.

    Pythia: `pythia-<size>[-seed<N>]` for the `full` variant, `data-seed<N>` /
    `weight-seed<N>` / `<local family>-seed<N>` for the 160m ensembles; DataDecide:
    `dd-1b[-<recipe>]-seed<N>` (the default recipe dolma1_7 is never spelled);
    external: the registry alias. A pinned checkpoint keeps its `@revision` / `@step<N>`.
    """
    spec = base_spec(value)
    if isinstance(spec, ExternalModelSpec):
        alias = EXTERNAL_BASE_BY_REPO[spec.repo_id].alias
        return alias + (f"@{spec.revision}" if spec.revision is not None else "")
    if isinstance(spec, DataDecideSpec):
        return dd_token(spec)
    if spec.variant == "full":
        token = f"pythia-{spec.size}" + (f"-seed{spec.seed}" if spec.seed else "")
    else:
        token = f"{spec.variant}-seed{spec.seed}"
    return token + (f"@{spec.revision}" if spec.revision is not None else "")


def same_base(a: str | BaseSpec, b: str | BaseSpec) -> bool:
    """True when two spellings name the same base checkpoint (family, size, seed, pin)."""
    return base_spec(a) == base_spec(b)


def family_of(value: str | BaseSpec) -> str:
    """`"pythia"`, `"datadecide"`, or the external family tag (`"qwen25"`, ...)."""
    spec = base_spec(value)
    if isinstance(spec, DataDecideSpec):
        return "datadecide"
    if isinstance(spec, ExternalModelSpec):
        return EXTERNAL_BASE_BY_REPO[spec.repo_id].family
    return "pythia"


def describe(value: str | BaseSpec, *, hub_org: str = DEFAULT_HUB_ORG) -> dict[str, Any]:
    """All spellings of one base, for logging and registries."""
    spec = base_spec(value)
    return {"token": base_token(spec), "dir_name": base_dir_name(spec, hub_org=hub_org),
            "ref": base_ref(spec, hub_org=hub_org), "family": family_of(spec)}


__all__ = [
    "DD_HUB_ORG", "DEFAULT_HUB_ORG", "BaseSpec", "base_dir_name", "base_ref", "base_spec",
    "base_token", "describe", "family_of", "same_base",
]
