"""DataDecide (OLMo) base-model specs, Hub resolution, and loading.

A second, fully isolated base-model family alongside PolyPythias: the AI2
DataDecide 1B models, selected by `dd-1b[-{recipe}][-seed{1,2,3}][@step{N}]` CLI
tokens. The data recipe is an axis of its own, named centrally by
`DD_RECIPE_SEGMENTS`: `dolma1_7` is the DEFAULT and is never spelled in a token
(one canonical spelling per model), and `dclm` selects
`DataDecide-dclm-baseline-1B`. The recipes are separate repos with the SAME
branch structure, so the seeds and the final step are shared.

The three full-budget seeds are HF BRANCHES of one repo, not separate repos, and
each seed re-randomizes weight init AND data order together (there is no
data-seed / weight-seed split as in PolyPythias).

Deliberately standalone: this module imports nothing from the rest of the
package at import time (the `model_specs` delegation in `parse_base_model_tokens`
is a function-local import), so `model_specs` stays ignorant of DataDecide and
every Pythia code path is unchanged.

Loading needs the `ai2-olmo` package (legacy `OLMoForCausalLM`, model_type
`hf_olmo`); it lives in the non-default `datadecide` dependency group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch/transformers stay OUT of the import graph of this module,
    import torch  # which registry/run_layout import (tests/unit/test_layering.py).
    from transformers import PreTrainedTokenizerBase

DD_HUB_ORG = "allenai"
# The data recipe axis. The DEFAULT recipe has NO token segment, so existing dolma1_7
# spellings (tokens, dir names, refs) are byte-identical to before; every other recipe
# is selected by its segment below. Hub repos: `DataDecide-{recipe}-{size}`.
DD_DEFAULT_RECIPE = "dolma1_7"
DD_RECIPE_SEGMENTS: dict[str, str] = {"dclm": "dclm-baseline"}
#: Every recipe this repo can name, default first.
DD_RECIPES: tuple[str, ...] = (DD_DEFAULT_RECIPE, *DD_RECIPE_SEGMENTS.values())
_SEGMENT_BY_RECIPE: dict[str, str] = {v: k for k, v in DD_RECIPE_SEGMENTS.items()}
# Final training step of the 1B models; shared by all three full-budget seeds of BOTH
# recipes (the dclm-baseline repo has the same 29 step branches; verified 2026-09-18).
DD_1B_FINAL_STEP = 69369
# Ordinal seed -> DataDecide branch seed name. Only these three are selectable:
# the `small-aux-*` branches are 25%-compute runs and are NEVER a valid choice.
DD_SEED_BRANCHES: dict[int, str] = {1: "default", 2: "large-aux-2", 3: "large-aux-3"}

_AI2_OLMO_HINT = (
    "DataDecide/OLMo models require the ai2-olmo package: "
    "run `uv sync --group datadecide`"
)


@dataclass(frozen=True)
class DataDecideSpec:
    """Identifies one DataDecide checkpoint on the Hugging Face Hub.

    Attributes:
        size: DataDecide size. Only "1B" is exposed by the token grammar.
        seed: Ordinal seed 1-3, mapped to a branch via `DD_SEED_BRANCHES`.
        step: Training step to pin. None -> the final step (`DD_1B_FINAL_STEP`).
        recipe: Data recipe, one of `DD_RECIPES`; defaults to `DD_DEFAULT_RECIPE`.
    """

    size: str = "1B"
    seed: int = 1
    step: int | None = None
    recipe: str = DD_DEFAULT_RECIPE


def _recipe_menu() -> str:
    """`'' (dolma1_7), 'dclm' (dclm-baseline)` -- the valid token recipe segments."""
    return ", ".join(
        [f"'' ({DD_DEFAULT_RECIPE}, the default)"]
        + [f"'{seg}' ({name})" for seg, name in DD_RECIPE_SEGMENTS.items()]
    )


def dd_recipe_segment(recipe: str) -> str:
    """Token segment of a recipe: `""` for the default, else e.g. `"dclm"`.

    Raises:
        ValueError: On a recipe outside `DD_RECIPES`.
    """
    if recipe == DD_DEFAULT_RECIPE:
        return ""
    if recipe not in _SEGMENT_BY_RECIPE:
        raise ValueError(
            f"unknown DataDecide recipe {recipe!r}; wired recipes are {DD_RECIPES} "
            f"(token segments: {_recipe_menu()})"
        )
    return _SEGMENT_BY_RECIPE[recipe]


def dd_token(spec: DataDecideSpec) -> str:
    """Canonical CLI token of a spec, e.g. `dd-1b-dclm-seed2@step17500`.

    The inverse of `parse_dd_token` (which canonicalizes `@step{final}` to unpinned).
    """
    if spec.size != "1B":
        raise ValueError(
            f"no CLI token for DataDecide {spec.recipe}-{spec.size}; only 1B is wired"
        )
    segment = dd_recipe_segment(spec.recipe)
    token = "dd-1b" + (f"-{segment}" if segment else "") + f"-seed{spec.seed}"
    return token + (f"@step{spec.step}" if spec.step is not None else "")


def dd_repo_id(spec: DataDecideSpec) -> str:
    """Hub repo id, e.g. `allenai/DataDecide-dolma1_7-1B`."""
    return f"{DD_HUB_ORG}/DataDecide-{spec.recipe}-{spec.size}"


def dd_revision(spec: DataDecideSpec) -> str:
    """The branch to load, ALWAYS explicit, e.g. `step69369-seed-large-aux-2`.

    Never "main": main can move, while the `step{N}-seed-{name}` branch is
    immutable provenance. An unpinned spec resolves to the final step.
    """
    step = spec.step if spec.step is not None else DD_1B_FINAL_STEP
    return f"step{step}-seed-{DD_SEED_BRANCHES[spec.seed]}"


def dd_model_ref(spec: DataDecideSpec) -> str:
    """`{repo_id}@{revision}` -- the manifest/run-identity provenance string."""
    return f"{dd_repo_id(spec)}@{dd_revision(spec)}"


def dd_dir_name(spec: DataDecideSpec) -> str:
    """Output-tree directory name, e.g. `DataDecide-dolma1_7-1B-seed-default`.

    A step-pinned spec appends `@step{N}`; unpinned names carry no `@`, so they
    are the canonical arm dirs and pinned ones route under `ckpt/` (see
    `run_layout.model_dir_parts`), exactly like revision-pinned Pythia dirs.
    """
    name = f"DataDecide-{spec.recipe}-{spec.size}-seed-{DD_SEED_BRANCHES[spec.seed]}"
    if spec.step is not None:
        return f"{name}@step{spec.step}"
    return name


def parse_dd_token(token: str) -> DataDecideSpec:
    """Parse one `dd-*` base-model token into a `DataDecideSpec`.

    Grammar: `dd-1b[-{recipe}][-seed{1|2|3}][@step{N}]`, with `{recipe}` a segment
    of `DD_RECIPE_SEGMENTS` (`dclm`) and `@step{N}` a positive int. The default
    recipe (`dolma1_7`) has NO segment and must not be spelled, so each model has
    exactly one canonical token. `dd-1b` is strict lowercase and the recipe
    segment comes BEFORE the seed segment.

    `@step{DD_1B_FINAL_STEP}` canonicalizes to step=None, so the token, ref and
    dir-name identities of the final checkpoint all agree (see `parse_dd_ref`).

    The step is NOT validated against the released checkpoint list (that would
    need the network at parse time); a bad step fails loudly at HF load.
    """
    raw = token.strip()
    step: int | None = None
    if "@" in raw:
        raw, step_str = raw.split("@", 1)
        if not step_str.startswith("step") or not step_str[len("step") :].isdigit():
            raise ValueError(
                f"bad step suffix {step_str!r} in DataDecide token {token!r}; "
                "expected '@step<N>' with a positive integer N, e.g. 'dd-1b@step17500'"
            )
        step = int(step_str[len("step") :])
        if step <= 0:
            raise ValueError(
                f"bad step {step} in DataDecide token {token!r}; must be positive"
            )
    if not raw.startswith("dd-1b") or raw[len("dd-1b") :][:1] not in ("", "-"):
        raise ValueError(
            f"bad DataDecide token {token!r}: only the 1B models are wired, spelled "
            "lowercase as 'dd-1b'. The 150M/300M aux seeds are budget-mismatched "
            "(trained to 25% of the 1B compute), so cross-seed work below 1B needs "
            "step-pinned checkpoints and is not selectable here"
        )
    rest = raw[len("dd-1b") :]
    recipe = DD_DEFAULT_RECIPE
    if rest and not rest.startswith("-seed"):
        segment, sep, tail = rest[1:].partition("-")
        if sep and not tail:
            # 'dd-1b-dclm-': a trailing separator names nothing, exactly as 'dd-1b-' does.
            raise ValueError(
                f"bad DataDecide token {token!r}: trailing '-' names no seed segment; "
                f"expected 'dd-1b[-<recipe>][-seed<N>][@step<N>]'"
            )
        if segment not in DD_RECIPE_SEGMENTS:
            raise ValueError(
                f"bad recipe segment {segment!r} in DataDecide token {token!r}; valid "
                f"recipe segments are {_recipe_menu()} -- the default recipe is never "
                "spelled, so that each model has one canonical token"
            )
        recipe = DD_RECIPE_SEGMENTS[segment]
        rest = f"-{tail}" if tail else ""
    seed = 1
    if rest:
        seed_str = rest[len("-seed") :] if rest.startswith("-seed") else rest
        if not rest.startswith("-seed") or not seed_str.isdigit():
            # e.g. 'dd-1b-seed2-dclm': the recipe segment must precede the seed.
            raise ValueError(
                f"bad DataDecide token {token!r}: expected "
                f"'dd-1b[-<recipe>][-seed<N>][@step<N>]' with the recipe segment BEFORE "
                f"the seed; valid recipe segments are {_recipe_menu()}, valid seed "
                f"ordinals are {_seed_menu()}"
            )
        seed = int(seed_str)
        if seed not in DD_SEED_BRANCHES:
            raise ValueError(
                f"bad seed {seed} in DataDecide token {token!r}; valid ordinals are "
                f"{_seed_menu()}"
            )
    return DataDecideSpec(
        size="1B",
        seed=seed,
        step=None if step == DD_1B_FINAL_STEP else step,
        recipe=recipe,
    )


def dd_recipe_of_repo_id(repo_id: str, *, context: str = "") -> str:
    """The data recipe of a DataDecide hub repo id (`allenai/DataDecide-<recipe>-<size>`).

    A bare repo id is seed-ambiguous (`parse_dd_ref` refuses it) but its RECIPE is
    unambiguous, which is all a corpus-level decision — such as the probe pool in
    `coherence.probe_io` — needs.

    Raises:
        ValueError: On a non-DataDecide repo id or an unwired recipe.
    """
    # The recipe may itself contain '-' (`dclm-baseline`), so split the SIZE off the end
    # and match the remainder against the registered recipes.
    prefix, sep, _size = repo_id.rpartition("-")
    recipe_prefix = f"{DD_HUB_ORG}/DataDecide-"
    recipe = prefix[len(recipe_prefix):] if prefix.startswith(recipe_prefix) else ""
    if not sep or recipe not in DD_RECIPES:
        raise ValueError(
            f"bad DataDecide repo id {repo_id!r}{context}; expected "
            f"'{DD_HUB_ORG}/DataDecide-<recipe>-<size>' with a wired recipe "
            f"{DD_RECIPES}"
        )
    return recipe


def parse_dd_ref(ref: str) -> DataDecideSpec:
    """Invert `dd_model_ref`: parse `{repo_id}@step{N}-seed-{branch}` into a spec.

    A ref cannot distinguish "unpinned" from "explicitly pinned to the final step"
    -- `dd_revision` spells both as `step{DD_1B_FINAL_STEP}-...` -- so the final
    step canonicalizes to step=None, the identity that `dd_dir_name` gives the
    canonical (non-`ckpt/`) arm dirs.

    Raises:
        ValueError: On a non-DataDecide repo id, a missing/malformed revision, or
            a branch outside `DD_SEED_BRANCHES`.
    """
    if "@" not in ref:
        raise ValueError(
            f"bad DataDecide ref {ref!r}: expected '{{repo_id}}@step<N>-seed-<branch>'. "
            "A bare repo id is ill-defined -- the three seeds share it"
        )
    repo, revision = ref.split("@", 1)
    recipe = dd_recipe_of_repo_id(repo, context=f" in ref {ref!r}")
    size = repo.rpartition("-")[2]
    step_str, sep, branch = revision.partition("-seed-")
    if not sep or not step_str.startswith("step") or not step_str[len("step"):].isdigit():
        raise ValueError(
            f"bad DataDecide revision {revision!r} in ref {ref!r}; expected "
            "'step<N>-seed-<branch>'"
        )
    seeds = {v: k for k, v in DD_SEED_BRANCHES.items()}
    if branch not in seeds:
        raise ValueError(
            f"bad DataDecide branch {branch!r} in ref {ref!r}; valid seeds are "
            f"{_seed_menu()}"
        )
    step = int(step_str[len("step"):])
    return DataDecideSpec(
        size=size,
        seed=seeds[branch],
        step=None if step == DD_1B_FINAL_STEP else step,
        recipe=recipe,
    )


def is_dd_ref(ref: str) -> bool:
    """True for a DataDecide model_ref (the `base_model_repo` manifest spelling)."""
    return ref.startswith(f"{DD_HUB_ORG}/DataDecide")


def base_dir_name_for_ref(ref: str) -> str:
    """Output-tree directory spelling of a base model_ref, any family.

    DataDecide refs go through `parse_dd_ref` + `dd_dir_name` so the per-seed
    branch survives; every other ref keeps the historical trailing-path-component
    spelling that the Pythia trees on disk are named by.
    """
    if is_dd_ref(ref):
        return dd_dir_name(parse_dd_ref(ref))
    return ref.split("/")[-1]


def _seed_menu() -> str:
    """`1 (default), 2 (large-aux-2), ...` -- the valid ordinals with branch names."""
    return ", ".join(f"{k} ({v})" for k, v in DD_SEED_BRANCHES.items())


def parse_base_model_tokens(value: str, *, allow_full_seeds: bool = False) -> list:
    """Parse a `--base-models` string, routing by family.

    If ANY comma-separated token starts with `dd-`, all must: the string is a
    DataDecide selection and every token goes through `parse_dd_token`.
    Otherwise the ORIGINAL, unsplit string is delegated to
    `teachers.model_specs.parse_base_models`, so the aggregate Pythia tokens
    ("all7", "seeds", "data", ...) behave byte-identically.

    DataDecide is CLI-token-only: an experiment YAML's `model_spec` stays
    Pythia-only, so the `--base-models config` branches never reach this router.

    Raises:
        ValueError: On a mixed pythia+dd list, or any malformed token.
    """
    tokens = [t.strip() for t in value.split(",") if t.strip()]
    dd_tokens = [t for t in tokens if t.lower().startswith("dd-")]
    if dd_tokens:
        if len(dd_tokens) != len(tokens):
            raise ValueError(
                f"mixed base-model families in {value!r}: DataDecide (dd-*) and "
                "Pythia tokens cannot share one invocation -- they have different "
                "tokenizers and Hub orgs. Run them as separate invocations"
            )
        return [parse_dd_token(t) for t in tokens]

    from subliminal_learning.teachers.model_specs import parse_base_models

    return parse_base_models(value, allow_full_seeds=allow_full_seeds)


def _require_hf_olmo() -> None:
    """Import `hf_olmo` (registers `OLMoForCausalLM` with Auto*) or explain the fix."""
    try:
        import hf_olmo  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(_AI2_OLMO_HINT) from exc


def load_datadecide_tokenizer(
    spec: DataDecideSpec, hf_cache_root: Path
) -> "PreTrainedTokenizerBase":
    """Load the OLMo tokenizer for a DataDecide spec, at its pinned branch.

    Every wired recipe is OLMo-tokenized (vocab 50280), so the tokenizer is the same
    object whichever recipe's repo it is loaded from.
    """
    _require_hf_olmo()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        dd_repo_id(spec),
        revision=dd_revision(spec),
        cache_dir=str(hf_cache_root / "hub"),
    )
    # OLMo ships a real pad token (id 1); only fill in a missing one.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # The OLMo tokenizer emits token_type_ids, but the legacy hf_olmo
    # OLMoForCausalLM.forward() rejects the kwarg — strip it here so every
    # `model(**tokenizer(...))` call site (evals, generation) stays family-agnostic.
    tokenizer.model_input_names = [
        n for n in tokenizer.model_input_names if n != "token_type_ids"
    ]
    return tokenizer


def load_datadecide_model(
    spec: DataDecideSpec,
    hf_cache_root: Path,
    device: str,
    dtype: "torch.dtype | None" = None,
) -> Any:
    """Load a DataDecide causal LM on `device` in eval mode, at its pinned branch.

    `dtype=None` means fp32, the repo-wide default (spelled lazily so importing this
    module does not pull in torch).
    """
    _require_hf_olmo()
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        dd_repo_id(spec),
        revision=dd_revision(spec),
        cache_dir=str(hf_cache_root / "hub"),
        torch_dtype=torch.float32 if dtype is None else dtype,
    )
    model = _ensure_generate(model)
    model.to(device)
    model.eval()
    return model


def _ensure_generate(model: Any) -> Any:
    """Graft GenerationMixin onto a legacy hf_olmo model.

    transformers >= 4.50 moved `.generate` off `PreTrainedModel`; the legacy
    `OLMoForCausalLM` predates the split (it still defines
    `prepare_inputs_for_generation`), so under the pinned transformers it loads
    without `.generate` and without a `generation_config`. Re-class the instance
    with the mixin and fill in the config the modern `__init__` would have built.
    """
    from transformers import GenerationConfig, GenerationMixin
    from transformers.cache_utils import Cache

    if not isinstance(model, GenerationMixin):
        cls = model.__class__

        def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None,
            cache_position=None, **kwargs,
        ):
            # Modern generate() seeds an (empty) Cache object; the legacy OLMo
            # stack only understands tuple-format caches and treats the empty
            # Cache as truthy, then crashes on past_key_values[0][0]. Convert.
            if isinstance(past_key_values, Cache):
                # An EMPTY cache must become None (to_legacy_cache() still emits a
                # non-empty tuple, which the olmo stack would index into and crash).
                past_key_values = (
                    None if past_key_values.get_seq_length() == 0
                    else past_key_values.to_legacy_cache()
                )
            return cls.prepare_inputs_for_generation(
                self, input_ids, past_key_values=past_key_values,
                attention_mask=attention_mask, **kwargs,
            )

        model.__class__ = type(
            cls.__name__, (cls, GenerationMixin),
            {"prepare_inputs_for_generation": prepare_inputs_for_generation},
        )
    if getattr(model, "generation_config", None) is None:
        model.generation_config = GenerationConfig.from_model_config(model.config)
    return model
