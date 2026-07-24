"""Generate a number-sequence dataset from one source (teacher / base / RNG).

The generate stage of Cloud's generate -> finetune -> evaluate pipeline, on base
models, DECOUPLED into two independently cached layers:

1. RAW GENERATION (GPU, expensive): build the shared seeded ICL prompt set, draw one
   continuation per prompt from the source, and persist every completion untouched:

       {out_dir}/raw.jsonl       every row: {prompt, completion, ended_with_eos}
       {out_dir}/manifest.json   source identity + GENERATION params only (no filter
                                 params), prompts_hash, animal-mention telemetry

2. FILTER MODES (CPU, cheap, re-runnable): named filtration schemes -- each a
   `NumberFilterConfig` -- applied to the persisted raw rows, each cached in its own
   subdirectory so different number-sequence distributions never collide:

       {out_dir}/modes/{mode}/filtered.jsonl  kept rows: {prompt, completion, numbers}
       {out_dir}/modes/{mode}/manifest.json   generation params + this mode's filter
                                              params, keep-rate, reject histogram,
                                              base_model_repo (for the same-base check)

Sources: "teacher" (base model + LoRA adapter checkpoint), "base" (untuned model --
Cloud's unprompted-reference control), "rng" (seeded uniform numbers, no model -- the
trait-free floor). All three share the same prompt set and, for the two model sources,
the same per-batch sampling seeds (common random numbers; `numbers.generation_io`).
The rng control draws ONE raw pool (uniform 0-999, 10 numbers) that every mode then
filters exactly like model output.

Resume: the raw manifest gates regeneration on generation params alone -- changing or
adding filter modes NEVER re-touches the GPU. Each mode manifest gates on (generation
params + that mode's filter params), so editing one mode recomputes only that mode.
`force=True` regenerates raw (and wipes the now-stale `modes/`). Subsampling to the
student-FT size (Cloud: 10k, `random.Random(seed).sample`) happens at
student-finetune time, not here.

An optional CONSTRAINED generation mode (`ConstrainedGenConfig`, YAML `generation.mode:
constrained`) forces every sampled token onto the number-list grammar via a
`LogitsProcessor` (`numbers.constrained`), so filtering is verification not selection
(keep-rate 1.0 by construction). It swaps in the DERIVED token cap, records a
`constrained` params block (part of the raw resume identity), grammar-matches the rng
control, and verifies every row before anything is written. Constrained pools are a
distinct on-disk lineage from filtered pools (`prompt_set_tag`) and must not be compared
across modes.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Mapping

import torch
import yaml
from loguru import logger
from peft import PeftModel
from transformers import LogitsProcessorList

from pipelines.evaluate_teacher_preference_pipeline import adapter_identity, prompts_hash
from subliminal_learning.config import PathsConfig
from subliminal_learning.evaluation.preference_scoring import mentions_target
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.numbers.constrained import (
    ConstrainedSpec,
    NumberSequenceProcessor,
    derived_max_new_tokens,
    digit_token_table,
    mask_pack,
    verify_constrained_rows,
)
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    all_animal_forms,
    filter_raw_rows,
)
from subliminal_learning.numbers.generation_io import (
    GeneratedCompletion,
    generate_completions,
)
from subliminal_learning.numbers.prompt_gen import (
    SEPARATORS,
    build_intro_prompts,
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.numbers.rng_gen import build_rng_completions
from subliminal_learning.run_gate import identity_diff
from subliminal_learning.templates.run_layout import model_dir_name
from subliminal_learning.utils.io import atomic_write_text, file_sha256

# Fixed salt for the per-batch sampling seeds (seed = base + batch index). Constant so
# every teacher/base run shares the stream; unrelated to the eval's sampling salt.
_GEN_SEED_BASE = 20_260_401

MODES_SUBDIR = "modes"


def mode_dir(dataset_dir: Path, mode: str) -> Path:
    """Where one filter mode's filtered.jsonl + manifest.json live."""
    return Path(dataset_dir) / MODES_SUBDIR / mode


def prompt_set_tag(
    prompt_set: "NumberPromptSetConfig",
    constrained: "ConstrainedGenConfig | None" = None,
) -> str:
    """Path segment naming the prompt-set variant, e.g. ``ex10-20``.

    Keyed on the in-context example-count range so datasets generated from prompt
    sets with more/fewer demonstrations live SIDE BY SIDE (different dirs) instead
    of clobbering one raw pool. Other prompt-set knobs still gate resume through
    ``prompts_hash`` (they regenerate in place); the example range is the axis
    meant to be varied, so it gets its own path.

    For a constrained pool the tag carries the grammar cell so the four constrained
    variants (and the free pool) never collide: base is ``intro`` for intro-style
    prompts else the ex tag; then ``-c{count_max}`` (exact) or ``-u{count_max}``
    (early-eos allowed); then ``x{value_min}-{value_max}`` only when the value range
    differs from the default (0, 999). ``constrained=None`` returns the byte-identical
    free-pool tag (existing on-disk dirs depend on it).
    """
    ex_tag = f"ex{prompt_set.example_count_min}-{prompt_set.example_count_max}"
    if constrained is None:
        return ex_tag
    base = "intro" if constrained.prompt_style == "intro" else ex_tag
    letter = "u" if constrained.allow_early_eos else "c"
    tag = f"{base}-{letter}{constrained.count_max}"
    if (constrained.value_min, constrained.value_max) != (0, 999):
        tag += f"x{constrained.value_min}-{constrained.value_max}"
    return tag


@dataclass(frozen=True)
class NumberPromptSetConfig:
    """The shared prompt set (pure function of these fields; see `prompt_gen`).

    Attributes:
        templates_path: Intro-template bank JSON.
        n_prompts: Prompts to build (= raw completions drawn).
        seed: PCG64 seed (Cloud fixes 42).
        example_count_min: Min ICL examples per prompt (inclusive).
        example_count_max: Max ICL examples per prompt (inclusive).
        example_value_min: Min example value (inclusive).
        example_value_max: Max example value (inclusive).
        separator_weights: Relative weight per separator key (comma/space/...).
    """

    templates_path: Path
    n_prompts: int
    seed: int = 42
    example_count_min: int = 10
    example_count_max: int = 20
    example_value_min: int = 100
    example_value_max: int = 999
    separator_weights: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.n_prompts <= 0:
            raise ValueError(f"n_prompts must be positive, got {self.n_prompts}")
        if self.example_count_min <= 0:
            raise ValueError("example_count_min must be positive")
        if self.example_count_min > self.example_count_max:
            raise ValueError("example_count_min must be <= example_count_max")
        if self.example_value_min > self.example_value_max:
            raise ValueError("example_value_min must be <= example_value_max")
        if self.separator_weights is not None:
            unknown = set(self.separator_weights) - set(SEPARATORS)
            if unknown:
                raise ValueError(
                    f"unknown separator keys {sorted(unknown)}; known: {sorted(SEPARATORS)}"
                )
            weights = list(self.separator_weights.values())
            if not weights:
                raise ValueError("separator_weights must not be empty")
            if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
                raise ValueError("separator_weights must be finite and non-negative")
            if sum(float(w) for w in weights) <= 0:
                raise ValueError("separator_weights must have positive total weight")


def load_number_prompt_set(path: Path) -> NumberPromptSetConfig:
    """Parse the templates_path + prompt_set block of number_generation.yaml."""
    doc = yaml.safe_load(path.read_text())
    ps = doc["prompt_set"]
    return NumberPromptSetConfig(
        templates_path=Path(doc["templates_path"]),
        n_prompts=int(ps["n_prompts"]),
        seed=int(ps.get("seed", 42)),
        example_count_min=int(ps.get("example_count_min", 10)),
        example_count_max=int(ps.get("example_count_max", 20)),
        example_value_min=int(ps.get("example_value_min", 100)),
        example_value_max=int(ps.get("example_value_max", 999)),
        separator_weights=ps.get("separator_weights"),
    )


@dataclass(frozen=True)
class ConstrainedGenConfig:
    """Constrained-generation knobs for one raw pool (see `numbers.constrained`).

    Attributes:
        prompt_style: "intro" (intro-only prompts, `build_intro_prompts`) or "icl"
            (the usual in-context-example prompts, `build_number_prompts`).
        count_max: Numbers per completed sequence.
        allow_early_eos: When True the sequence may end at 1..count_max numbers;
            when False it holds exactly count_max.
        value_min: Minimum allowed integer value (inclusive).
        value_max: Maximum allowed integer value (inclusive).
    """

    prompt_style: str
    count_max: int = 10
    allow_early_eos: bool = False
    value_min: int = 0
    value_max: int = 999

    def __post_init__(self) -> None:
        if self.prompt_style not in ("intro", "icl"):
            raise ValueError(
                f"unknown prompt_style {self.prompt_style!r}; expected 'intro' or 'icl'"
            )

    @property
    def spec(self) -> ConstrainedSpec:
        """The `ConstrainedSpec` this config defines (grammar cache key)."""
        return ConstrainedSpec(
            count_max=self.count_max,
            allow_early_eos=self.allow_early_eos,
            value_min=self.value_min,
            value_max=self.value_max,
        )


def constrained_from_generation_doc(gen: Mapping) -> ConstrainedGenConfig | None:
    """Parse the `generation:` block into a `ConstrainedGenConfig` (or None).

    `gen["mode"]` (default "icl_free") selects the workflow:
      "icl_free"    -> None (the historical free-generation + filter path). A stray
                       "constrained" block here is a footgun and is REJECTED.
      "constrained" -> parse `gen["constrained"]`: required `prompt_style`; required
                       `count_variant` ("exact10" -> count_max 10, allow_early_eos
                       False; "upto10" -> count_max 10, allow_early_eos True), with an
                       optional `count_max` override; optional `value_min`/`value_max`.

    Raises:
        ValueError: on an unknown mode, a misplaced constrained block, a missing/bad
            required key, or an unknown key inside the constrained block.
    """
    mode = gen.get("mode", "icl_free")
    if mode == "icl_free":
        if "constrained" in gen:
            raise ValueError(
                "generation.mode 'icl_free' must not carry a 'constrained' block "
                "(it would be silently ignored); set mode: constrained to use it."
            )
        return None
    if mode != "constrained":
        raise ValueError(
            f"unknown generation.mode {mode!r}; expected 'icl_free' or 'constrained'"
        )
    block = gen.get("constrained")
    if not isinstance(block, Mapping):
        raise ValueError("generation.mode 'constrained' requires a 'constrained' block")
    known = {"prompt_style", "count_variant", "count_max", "value_min", "value_max"}
    unknown = set(block) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in the constrained block; "
            f"known keys: {sorted(known)}"
        )
    if "prompt_style" not in block:
        raise ValueError("the constrained block requires 'prompt_style'")
    variant = block.get("count_variant")
    if variant == "exact10":
        count_max, allow_early_eos = 10, False
    elif variant == "upto10":
        count_max, allow_early_eos = 10, True
    else:
        raise ValueError(
            f"the constrained block requires count_variant 'exact10' or 'upto10', "
            f"got {variant!r}"
        )
    if "count_max" in block:
        count_max = int(block["count_max"])
    return ConstrainedGenConfig(
        prompt_style=str(block["prompt_style"]),
        count_max=count_max,
        allow_early_eos=allow_early_eos,
        value_min=int(block.get("value_min", 0)),
        value_max=int(block.get("value_max", 999)),
    )


@dataclass(frozen=True)
class GenerateNumbersConfig:
    """Config for one raw-generation run (+ the filter modes to derive from it).

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Base model (ignored for source="rng").
        source: "teacher" (adapter), "base" (untuned), or "rng" (no model).
        out_dir: Where raw.jsonl / manifest.json / modes/ land.
        prompt_set: Shared prompt-set parameters.
        filter_modes: Named filtration schemes to apply after (or without re-) generating
            raw; each cached at `modes/{name}/`. May be empty (raw only).
        constrained: When set, generate under the grammar-constrained mode (derived cap,
            keep-rate-1.0 verification, grammar-matched rng); None is free generation.
        target_animal: Teacher's implanted animal (manifest metadata; teacher only).
        checkpoint_dir: LoRA adapter dir (required iff source="teacher").
        temperature: Sampling temperature (Cloud: 1.0).
        max_new_tokens: Token cap per continuation (40 covers 10 numbers + overrun).
        batch_size: Prompts per generate call.
        rng_count: Numbers per rng-control completion (Cloud's answer count).
        rng_value_min: Min rng value (inclusive) -- the UNFILTERED pool; modes narrow it.
        rng_value_max: Max rng value (inclusive).
        device: Torch device.
        force: Regenerate raw (wiping stale modes/) even if a matching manifest exists.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    source: Literal["teacher", "base", "rng"]
    out_dir: Path
    prompt_set: NumberPromptSetConfig
    filter_modes: Mapping[str, NumberFilterConfig] = field(default_factory=dict)
    constrained: ConstrainedGenConfig | None = None
    target_animal: str | None = None
    checkpoint_dir: Path | None = None
    temperature: float = 1.0
    max_new_tokens: int = 40
    batch_size: int = 128
    rng_count: int = 10
    rng_value_min: int = 0
    rng_value_max: int = 999
    device: str = "cuda:0"
    force: bool = False

    def __post_init__(self) -> None:
        if self.source not in ("teacher", "base", "rng"):
            raise ValueError(f"unknown number source {self.source!r}")
        if self.source == "teacher" and self.checkpoint_dir is None:
            raise ValueError('source="teacher" requires checkpoint_dir')
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.rng_count <= 0:
            raise ValueError("rng_count must be positive")
        if self.rng_value_min > self.rng_value_max:
            raise ValueError("rng_value_min must be <= rng_value_max")


@dataclass(frozen=True)
class FilterModeResult:
    """Counts from applying one named filter mode to one raw dataset."""

    mode: str
    out_dir: Path
    n_raw: int
    n_filtered: int
    keep_rate: float
    reject_histogram: dict[str, int]


@dataclass(frozen=True)
class GenerateNumbersResult:
    """Raw counts + per-mode filter results from one generation run."""

    source: str
    out_dir: Path
    n_raw: int
    raw_animal_mention_rate: float
    modes: dict[str, FilterModeResult]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generation_params(cfg: GenerateNumbersConfig, phash: str) -> dict:
    """The params a cached RAW manifest must match to be reused (the resume gate).

    GENERATION identity only -- deliberately no filter params, so filter-mode changes
    never invalidate the GPU-expensive raw pool. JSON-round-tripped so the equality
    check against a reloaded manifest compares like with like.
    """
    params = {
        "source": cfg.source,
        "target_animal": cfg.target_animal,
        "checkpoint_dir": str(cfg.checkpoint_dir) if cfg.checkpoint_dir else None,
        # Adapter content id (teacher only): a --force retrain rewrites checkpoints/ at the
        # SAME path, so keying on the path string alone leaves this dataset silently stale.
        "adapter_id": (
            adapter_identity(cfg.checkpoint_dir)
            if cfg.source == "teacher" and cfg.checkpoint_dir else None
        ),
        "prompts_hash": phash,
        "n_prompts": cfg.prompt_set.n_prompts,
        # Explicit (though implied by prompts_hash) so manifests diff readably and
        # per-mode manifests inherit the prompt-set variant identity.
        "example_count_min": cfg.prompt_set.example_count_min,
        "example_count_max": cfg.prompt_set.example_count_max,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        "sample_seed_base": _GEN_SEED_BASE,
        "batch_size": cfg.batch_size,  # affects batching => sampling streams
        # rng-control pool identity (null for model sources; prompt_set.seed also
        # seeds the rng draws).
        "rng": (
            {"count": cfg.rng_count, "value_min": cfg.rng_value_min,
             "value_max": cfg.rng_value_max}
            if cfg.source == "rng" else None
        ),
    }
    # Constrained-mode identity: added ONLY when constrained, so free-generation
    # manifests stay byte-identical (the resume gate for every existing on-disk pool).
    # The derived token cap and the grammar-matched rng distribution REPLACE cfg's
    # (ignored) max_new_tokens / rng_* here so the params reflect what was generated.
    if cfg.constrained is not None:
        c = cfg.constrained
        spec = c.spec
        params["constrained"] = {
            "prompt_style": c.prompt_style,
            "count_max": c.count_max,
            "allow_early_eos": c.allow_early_eos,
            "value_min": c.value_min,
            "value_max": c.value_max,
        }
        params["max_new_tokens"] = derived_max_new_tokens(spec)
        if cfg.source == "rng":
            params["rng"] = {
                "count": c.count_max,
                "value_min": spec.value_min,
                "value_max": spec.value_max,
            }
    # Revision-aware base identity in the resume gate, added ONLY when a pretraining
    # checkpoint revision is pinned. Conditional so existing (revision=None) raw
    # manifests keep byte-identical params and are never invalidated; with it, two
    # runs off different checkpoints of the same repo become distinct raw pools.
    if cfg.spec.revision is not None:
        params["base_model_ref"] = model_ref(cfg.spec, cfg.hub_org)
    return json.loads(json.dumps(params))


# Prompt-set hashes at a given size are pure functions of the prompt-set knobs
# (build_number_prompts draws sequentially from one PCG64 stream, so the set is
# PREFIX-STABLE in n_prompts); memoized so sweep-wide status checks build each
# distinct (config, size) prompt set at most once per process.
_PREFIX_HASH_CACHE: dict[tuple, str] = {}


def _constrained_hash_key(
    constrained: ConstrainedGenConfig | None,
) -> tuple | None:
    """The prompt-hash cache-key fragment for a constrained config (None -> free)."""
    if constrained is None:
        return None
    return (
        constrained.prompt_style, constrained.count_max,
        constrained.allow_early_eos, constrained.value_min, constrained.value_max,
    )


def _build_prompt_set(
    prompt_set: NumberPromptSetConfig,
    n: int,
    constrained: ConstrainedGenConfig | None,
):
    """Build the `n`-prompt set for this config, branching on constrained style.

    Intro-style constrained pools use `build_intro_prompts` (intro-only, row-paired
    with the ICL set); everything else uses `build_number_prompts`. Both consume the
    identical prompt-set knobs, so the derivation stays prefix-stable either way.
    """
    templates = load_intro_templates(prompt_set.templates_path)
    builder = (
        build_intro_prompts
        if constrained is not None and constrained.prompt_style == "intro"
        else build_number_prompts
    )
    return builder(
        templates,
        n_prompts=n,
        seed=prompt_set.seed,
        example_count_min=prompt_set.example_count_min,
        example_count_max=prompt_set.example_count_max,
        example_value_min=prompt_set.example_value_min,
        example_value_max=prompt_set.example_value_max,
        separator_weights=prompt_set.separator_weights,
    )


def _prompts_hash_at(
    prompt_set: NumberPromptSetConfig,
    n: int,
    constrained: ConstrainedGenConfig | None = None,
) -> str:
    """Hash of the prompt set this config yields at size `n` (memoized)."""
    sw = prompt_set.separator_weights
    key = (
        str(prompt_set.templates_path), n, prompt_set.seed,
        prompt_set.example_count_min, prompt_set.example_count_max,
        prompt_set.example_value_min, prompt_set.example_value_max,
        tuple(sorted(sw.items())) if sw else None,
        _constrained_hash_key(constrained),
    )
    if key not in _PREFIX_HASH_CACHE:
        prompts = _build_prompt_set(prompt_set, n, constrained)
        _PREFIX_HASH_CACHE[key] = prompts_hash([p.text for p in prompts])
    return _PREFIX_HASH_CACHE[key]


_COUNT_KEYS = ("n_prompts", "prompts_hash")


def _legacy_pool_upgrade_reason(
    cfg: GenerateNumbersConfig, params: dict, stored: dict
) -> str | None:
    """Explain why a known pre-decoupling pool may be regenerated safely.

    Legacy manifests coupled ``filters`` into the raw-generation identity and did
    not record ``adapter_id``, prompt example-count fields, or the RNG identity in
    ``params``. They cannot be reused for a current teacher, but can be regenerated
    in place when all reconstructable generation inputs match. Unknown legacy
    shapes remain behind the explicit ``--force`` guard.

    Constrained pools postdate the decoupling and can never be legacy, so this
    returns None immediately for them.
    """
    if cfg.constrained is not None:
        return None
    sp = stored.get("params")
    prompt_doc = stored.get("prompt_set")
    if not isinstance(sp, dict) or not isinstance(prompt_doc, dict):
        return None
    if "filters" not in sp or "adapter_id" in sp:
        return None
    stored_n = sp.get("n_prompts")
    if not isinstance(stored_n, int) or cfg.prompt_set.n_prompts < stored_n:
        return None

    core_keys = (
        "source", "target_animal", "checkpoint_dir", "temperature",
        "max_new_tokens", "sample_seed_base", "batch_size",
    )
    if any(sp.get(k) != params.get(k) for k in core_keys):
        return None

    expected_prompt = {
        "templates_path": str(cfg.prompt_set.templates_path),
        "n_prompts": stored_n,
        "seed": cfg.prompt_set.seed,
        "example_count_min": cfg.prompt_set.example_count_min,
        "example_count_max": cfg.prompt_set.example_count_max,
        "example_value_min": cfg.prompt_set.example_value_min,
        "example_value_max": cfg.prompt_set.example_value_max,
        "separator_weights": json.loads(json.dumps(cfg.prompt_set.separator_weights)),
    }
    if any(prompt_doc.get(k) != v for k, v in expected_prompt.items()):
        return None
    if sp.get("prompts_hash") != _prompts_hash_at(
        cfg.prompt_set, stored_n, cfg.constrained
    ):
        return None

    # The old RNG generator derived its distribution from the single filter config.
    if cfg.source == "rng":
        old_filters = sp.get("filters")
        if not isinstance(old_filters, dict):
            return None
        old_rng = {
            "count": old_filters.get("max_count"),
            "value_min": old_filters.get("min_value"),
            "value_max": old_filters.get("max_value"),
        }
        if old_rng != params.get("rng"):
            return None

    adapter_note = (
        f"current adapter_id={params.get('adapter_id')}"
        if cfg.source == "teacher" else "non-teacher source"
    )
    return (
        "recognized pre-decoupling manifest: reconstructable generation inputs "
        f"match; stored_n={stored_n}, requested_n={cfg.prompt_set.n_prompts}; "
        f"{adapter_note}. The legacy raw pool will NOT be reused—it will be fully "
        "regenerated and its stale filter outputs replaced."
    )


def _raw_pool_status(cfg: GenerateNumbersConfig, params: dict) -> tuple[str, dict | None]:
    """Classify the on-disk raw pool against the requested generation params.

    Returns (status, stored_manifest) with status one of:
      "match"        stored params identical -> reuse as-is
      "sufficient"   same generation config, stored n_prompts >= requested; the
                     stored pool is a superset (prefix-stable prompt set,
                     verified by re-hashing the config at the stored size) -> reuse
      "insufficient" same generation config, stored n_prompts < requested ->
                     regenerate at the larger requested size (no force needed)
      "legacy_upgrade" recognized pre-decoupling manifest whose reconstructable
                     inputs match -> fully regenerate under the current identity
      "mismatch"     any non-count param differs -> refuse without force
      "torn"         raw.jsonl without a valid manifest -> regenerate freely
      "absent"       no raw pool on disk
    """
    manifest_path = cfg.out_dir / "manifest.json"
    stored = None
    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            stored = None
    if not (cfg.out_dir / "raw.jsonl").exists():
        return "absent", None
    if stored is None or not isinstance(stored.get("params"), dict):
        return "torn", None
    sp = stored["params"]
    # Seed the hash cache with the requested set's hash (already computed by callers).
    _PREFIX_HASH_CACHE.setdefault(
        (str(cfg.prompt_set.templates_path), cfg.prompt_set.n_prompts,
         cfg.prompt_set.seed, cfg.prompt_set.example_count_min,
         cfg.prompt_set.example_count_max, cfg.prompt_set.example_value_min,
         cfg.prompt_set.example_value_max,
         tuple(sorted(cfg.prompt_set.separator_weights.items()))
         if cfg.prompt_set.separator_weights else None,
         _constrained_hash_key(cfg.constrained)),
        params["prompts_hash"],
    )
    if sp == params:
        return "match", stored
    others_equal = all(
        sp.get(k) == params.get(k)
        for k in set(sp) | set(params)
        if k not in _COUNT_KEYS
    )
    if others_equal and isinstance(sp.get("n_prompts"), int):
        stored_n = sp["n_prompts"]
        # Same config at a different size ONLY if the stored hash is what the
        # CURRENT prompt-set knobs produce at the stored size -- otherwise a knob
        # that lives solely in the hash (seed, templates, value range) changed too.
        if sp.get("prompts_hash") == _prompts_hash_at(
            cfg.prompt_set, stored_n, cfg.constrained
        ):
            status = "sufficient" if stored_n >= cfg.prompt_set.n_prompts else "insufficient"
            return status, stored
    legacy_reason = _legacy_pool_upgrade_reason(cfg, params, stored)
    if legacy_reason is not None:
        return "legacy_upgrade", {**stored, "legacy_upgrade_reason": legacy_reason}
    return "mismatch", stored


def _mode_params(gen_params: dict, mode: str, filters: NumberFilterConfig) -> dict:
    """The params a cached MODE manifest must match: raw identity + this mode's rules."""
    return json.loads(json.dumps({
        "generation": gen_params,
        "mode": mode,
        "filters": asdict(filters),
    }))


def apply_filter_mode(
    dataset_dir: Path,
    mode: str,
    filters: NumberFilterConfig,
    *,
    force: bool = False,
    raw_rows: list[dict] | None = None,
) -> FilterModeResult:
    """Apply one named filter mode to a persisted raw dataset (CPU only, cached).

    Reads `dataset_dir/manifest.json` (the raw manifest) for the generation identity
    and base-model metadata, filters `raw.jsonl` under `filters`
    (`numbers.filtering.filter_raw_rows`, pure), and writes
    `modes/{mode}/filtered.jsonl` + `modes/{mode}/manifest.json`.

    Resume: skipped when the mode manifest matches (generation params + mode filters)
    and filtered.jsonl exists; `force=True` recomputes. Pass `raw_rows` to skip
    re-reading raw.jsonl when the caller already holds it.
    """
    dataset_dir = Path(dataset_dir)
    if not filters.require_clean_stop:
        logger.warning(
            "⚠ Applying LENIENT parsing (require_clean_stop=false) for mode {!r} at "
            "{} — it WIDENS the misparse surface, keeping the clean prefix when "
            "prose/format drift follows ('3.14, …' -> (3,); '1,000 dollars' -> "
            "(1, 0)). Strict is not immune — a clean stop after a grouped thousand "
            "('1,000') already parses as (1, 0) — but strict is the default; lenient "
            "is for deliberate distribution experiments only.", mode, dataset_dir,
        )
    raw_manifest_path = dataset_dir / "manifest.json"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(
            f"no raw manifest at {raw_manifest_path}; generate the raw pool first"
        )
    raw_manifest = json.loads(raw_manifest_path.read_text())
    params = _mode_params(raw_manifest["params"], mode, filters)

    out = mode_dir(dataset_dir, mode)
    manifest_path = out / "manifest.json"
    filtered_path = out / "filtered.jsonl"
    stored: dict | None = None
    if manifest_path.exists():
        try:
            stored = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            stored = None
    stored_params = stored.get("params") if isinstance(stored, dict) else None

    if isinstance(stored_params, dict) and not force:
        # Torn-pair closure: filtered.jsonl and this manifest are two separate writes,
        # so a crash between them (or a later param revert) could leave the gate serving
        # data that no longer matches the manifest. Reuse ONLY when the params match AND
        # the on-disk filtered.jsonl still hashes to the `filtered_sha256` we recorded
        # when we wrote it. Existing mode manifests predate the field, so every mode
        # refilters ONCE on next touch (missing field -> sha check fails) — acceptable
        # and self-healing: refiltering is CPU-cheap and deterministic, so the rewritten
        # filtered.jsonl is byte-identical and downstream student `filtered_sha256`
        # identities are unaffected.
        if (
            stored_params == params
            and filtered_path.exists()
            and stored.get("filtered_sha256") == file_sha256(filtered_path)
        ):
            logger.info("SKIP existing filter mode (matching manifest): {}", out)
            return FilterModeResult(
                mode=mode,
                out_dir=out,
                n_raw=stored["n_raw"],
                n_filtered=stored["n_filtered"],
                keep_rate=stored["keep_rate"],
                reject_histogram=dict(stored["reject_histogram"]),
            )
        # Loud refilter: same mode name, changed FILTER rules is redefining a mode in
        # place — it would silently overwrite the prior filtered.jsonl, so demand
        # explicit consent (force / --force). A GENERATION-portion difference is
        # governed by the raw gate upstream and falls through to a plain recompute.
        filter_diff = identity_diff(stored_params.get("filters") or {}, params["filters"])
        if filter_diff:
            raise ValueError(
                f"filter rules for mode {mode!r} at {dataset_dir} differ from the stored "
                f"manifest on {sorted(filter_diff)}; refiltering would overwrite "
                f"filtered.jsonl in place. Pass force=True (--force) to redefine the mode "
                f"deliberately, or use a new mode name."
            )

    rows = raw_rows if raw_rows is not None else _read_jsonl(dataset_dir / "raw.jsonl")
    fr = filter_raw_rows(rows, filters=filters)  # mention telemetry lives in the raw manifest
    _write_jsonl(filtered_path, fr.kept_rows)  # atomic (temp + os.replace) via atomic_write_text
    manifest = {
        "params": params,
        "mode": mode,
        # Copied from the raw manifest so a mode dir is a self-sufficient
        # `number_dataset_dir` for student distillation (same-base cross-check).
        "base_model": raw_manifest.get("base_model"),
        "base_model_repo": raw_manifest.get("base_model_repo"),
        # Content id of the just-written filtered.jsonl; the skip gate rechecks it
        # against the on-disk bytes to catch a torn pair or a param revert.
        "filtered_sha256": file_sha256(filtered_path),
        "n_raw": fr.n_raw,
        "n_filtered": fr.n_filtered,
        "keep_rate": fr.keep_rate,
        "reject_histogram": fr.reject_histogram,
    }
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Filter mode {!r}: {} raw -> {} kept ({:.1%}) -> {} (rejects: {})",
        mode, fr.n_raw, fr.n_filtered, fr.keep_rate, out, fr.reject_histogram or "none",
    )
    return FilterModeResult(
        mode=mode,
        out_dir=out,
        n_raw=fr.n_raw,
        n_filtered=fr.n_filtered,
        keep_rate=fr.keep_rate,
        reject_histogram=fr.reject_histogram,
    )


def run(cfg: GenerateNumbersConfig) -> GenerateNumbersResult:
    """Generate (or reuse) one raw pool, then apply every configured filter mode.

    Returns:
        Raw counts + one `FilterModeResult` per configured mode.
    """
    if cfg.source == "teacher" and cfg.checkpoint_dir is None:
        raise ValueError('source="teacher" requires checkpoint_dir')

    prompts = _build_prompt_set(
        cfg.prompt_set, cfg.prompt_set.n_prompts, cfg.constrained
    )
    phash = prompts_hash([p.text for p in prompts])
    params = generation_params(cfg, phash)

    # In constrained mode cfg.max_new_tokens and (for rng) cfg.rng_* are IGNORED in
    # favour of the derived cap and the grammar-matched distribution; warn once so a
    # stale YAML value isn't mistaken for the setting actually in effect.
    if cfg.constrained is not None:
        spec = cfg.constrained.spec
        derived = derived_max_new_tokens(spec)
        if cfg.max_new_tokens != derived:
            logger.warning(
                "constrained mode: ignoring max_new_tokens={} in favour of the derived "
                "cap {} (worst-case grammar length + slack).",
                cfg.max_new_tokens, derived,
            )
        if cfg.source == "rng" and (
            (cfg.rng_count, cfg.rng_value_min, cfg.rng_value_max)
            != (spec.count_max, spec.value_min, spec.value_max)
        ):
            logger.warning(
                "constrained mode: ignoring rng_* in favour of the grammar-matched "
                "count={} value_min={} value_max={}.",
                spec.count_max, spec.value_min, spec.value_max,
            )

    manifest_path = cfg.out_dir / "manifest.json"
    raw_path = cfg.out_dir / "raw.jsonl"
    raw_rows: list[dict] | None = None
    raw_cached = False

    # Coverage-aware resume gate (see _raw_pool_status): an existing pool with the
    # SAME generation config is reused when it has at least as many rows as
    # requested, and regenerated at the larger size when it has fewer. Any OTHER
    # param difference refuses -- regenerating destroys the pool and every cached
    # filter mode, so it must be an explicit choice (force), never a silent side
    # effect of a config tweak. Torn pools (raw.jsonl without a valid manifest)
    # are unusable and regenerate freely.
    status, stored = _raw_pool_status(cfg, params)
    if not cfg.force:
        if status == "match":
            logger.info("SKIP existing raw pool (matching manifest): {}", cfg.out_dir)
            raw_cached = True
            raw_manifest = stored
        elif status == "sufficient":
            logger.info(
                "REUSE existing raw pool at {}: {} rows on disk >= {} requested "
                "(same generation config; prompt set is prefix-stable).",
                cfg.out_dir, stored["params"]["n_prompts"], cfg.prompt_set.n_prompts,
            )
            raw_cached = True
            raw_manifest = stored
        elif status == "insufficient":
            logger.info(
                "EXTEND raw pool at {}: {} rows on disk < {} requested (same "
                "generation config); regenerating the full pool at the larger size.",
                cfg.out_dir, stored["params"]["n_prompts"], cfg.prompt_set.n_prompts,
            )
        elif status == "legacy_upgrade":
            logger.warning(
                "LEGACY RAW POOL UPGRADE at {}\n{}",
                cfg.out_dir, stored["legacy_upgrade_reason"],
            )
        elif status == "mismatch":
            changed = sorted(
                k for k in set(stored["params"]) | set(params)
                if stored["params"].get(k) != params.get(k)
            )
            raise ValueError(
                f"Refusing to overwrite the existing raw pool at {cfg.out_dir}: the "
                f"requested generation params differ from the stored manifest on "
                f"{changed} (beyond a pure prompt-count change). Regenerating would "
                f"destroy raw.jsonl and every cached filter mode under modes/. Pass "
                f"force=True (--force) to overwrite deliberately, or write the new "
                f"variant to a different out_dir."
            )
        elif status == "torn":
            logger.warning(
                "Torn raw pool (raw.jsonl without a valid manifest) at {}; regenerating.",
                cfg.out_dir,
            )

    if not raw_cached:
        logger.info(
            "Generating numbers: source={} n_prompts={} -> {}",
            cfg.source, len(prompts), cfg.out_dir,
        )
        if cfg.source == "rng":
            if cfg.constrained is not None:
                spec = cfg.constrained.spec
                rng_kwargs = dict(
                    count=spec.count_max,
                    value_min=spec.value_min,
                    value_max=spec.value_max,
                )
            else:
                rng_kwargs = dict(
                    count=cfg.rng_count,
                    value_min=cfg.rng_value_min,
                    value_max=cfg.rng_value_max,
                )
            generated = [
                GeneratedCompletion(completion=c, ended_with_eos=True)
                for c in build_rng_completions(
                    prompts, seed=cfg.prompt_set.seed, **rng_kwargs
                )
            ]
        else:
            tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
            model = load_model(
                cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device,
                dtype=torch.float32,
            )
            if cfg.source == "teacher":
                model = PeftModel.from_pretrained(model, str(cfg.checkpoint_dir))
            model.eval()
            if cfg.constrained is not None:
                # Build the vocab audit + component masks ONCE (memoized), then hand a
                # per-batch factory the aligned separator keys so each batch gets a fresh
                # DFA processor (which carries per-row decoding state).
                spec = cfg.constrained.spec
                pack = mask_pack(digit_token_table(tokenizer), spec, cfg.device)
                sep_keys = [p.separator_key for p in prompts]
                processor_factory = lambda s, e: LogitsProcessorList(  # noqa: E731
                    [NumberSequenceProcessor(pack, sep_keys[s:e])]
                )
                gen_max_new_tokens = derived_max_new_tokens(spec)
            else:
                processor_factory = None
                gen_max_new_tokens = cfg.max_new_tokens
            generated = generate_completions(
                model, tokenizer, cfg.device, [p.text for p in prompts],
                temperature=cfg.temperature, max_new_tokens=gen_max_new_tokens,
                batch_size=cfg.batch_size, seed_base=_GEN_SEED_BASE,
                processor_factory=processor_factory,
            )
            del model

        raw_rows = [
            {"prompt": p.text, "completion": g.completion, "ended_with_eos": g.ended_with_eos}
            for p, g in zip(prompts, generated)
        ]
        # Constrained pools are verified, not filtered: every row must be on-grammar.
        # A ValueError here propagates BEFORE any write, so a mask/cap bug can never
        # leave a pool on disk. The count histogram goes into the raw manifest.
        constrained_histogram: dict[str, int] | None = None
        if cfg.constrained is not None:
            histogram = verify_constrained_rows(raw_rows, cfg.constrained.spec)
            constrained_histogram = {str(k): histogram[k] for k in sorted(histogram)}
        # TELEMETRY, not a filter: how often this source's raw completions mention any
        # curated animal form anywhere (incl. past any truncation point). Teacher >>
        # base here is free evidence the implanted trait is active during generation.
        # Belongs to the raw pool, so it is counted once here, not per mode.
        mention_forms = all_animal_forms()
        n_mentions = sum(
            mentions_target(r["completion"], mention_forms) for r in raw_rows
        )
        # The old modes/ (if any) was derived from a DIFFERENT raw pool; drop it so a
        # stale filtered.jsonl can never be mistaken for current. Mode gates would
        # catch a re-request, but not a consumer reading the file directly.
        modes_root = cfg.out_dir / MODES_SUBDIR
        if modes_root.exists():
            logger.info("Raw pool regenerated; removing stale {}", modes_root)
            shutil.rmtree(modes_root)

        _write_jsonl(raw_path, raw_rows)
        raw_manifest = {
            "params": params,
            "base_model": (
                None if cfg.source == "rng" else model_dir_name(cfg.spec, cfg.hub_org)
            ),
            "base_model_repo": (
                None if cfg.source == "rng" else model_ref(cfg.spec, cfg.hub_org)
            ),
            "prompt_set": {
                **{k: (str(v) if isinstance(v, Path) else v)
                   for k, v in asdict(cfg.prompt_set).items()},
            },
            "n_raw": len(raw_rows),
            "raw_animal_mention_count": n_mentions,
            "raw_animal_mention_rate": (n_mentions / len(raw_rows) if raw_rows else 0.0),
        }
        if constrained_histogram is not None:
            raw_manifest["constrained_count_histogram"] = constrained_histogram
        atomic_write_text(manifest_path, json.dumps(raw_manifest, indent=2) + "\n")
        logger.info(
            "Wrote {} raw rows -> {} (animal mentions: {:.2%})",
            len(raw_rows), cfg.out_dir, raw_manifest["raw_animal_mention_rate"],
        )

    modes: dict[str, FilterModeResult] = {}
    pool_params = raw_manifest["params"]  # == requested params unless reused-as-superset
    for name, filters in cfg.filter_modes.items():
        if raw_rows is None and not _mode_is_cached(cfg.out_dir, name, pool_params, filters):
            raw_rows = _read_jsonl(raw_path)  # load once, share across stale modes
        modes[name] = apply_filter_mode(
            cfg.out_dir, name, filters, raw_rows=raw_rows,
        )

    return GenerateNumbersResult(
        source=cfg.source,
        out_dir=cfg.out_dir,
        n_raw=raw_manifest["n_raw"],
        raw_animal_mention_rate=raw_manifest.get("raw_animal_mention_rate", 0.0),
        modes=modes,
    )


def dataset_complete(cfg: GenerateNumbersConfig, phash: str) -> bool:
    """True when `run(cfg)` would be a complete no-op: raw AND every mode cached.

    The sweep runner's DONE check -- by construction the same gates `run()` applies,
    so the sweep can never skip a dataset the pipeline would (re)generate.
    """
    params = generation_params(cfg, phash)
    status, stored = _raw_pool_status(cfg, params)
    if status not in ("match", "sufficient"):
        return False
    return all(
        _mode_is_cached(cfg.out_dir, name, stored["params"], filters)
        for name, filters in cfg.filter_modes.items()
    )


def _mode_is_cached(
    dataset_dir: Path, mode: str, gen_params: dict, filters: NumberFilterConfig
) -> bool:
    """True when `apply_filter_mode` would skip (matching params + intact sha)."""
    out = mode_dir(dataset_dir, mode)
    manifest_path = out / "manifest.json"
    filtered_path = out / "filtered.jsonl"
    if not manifest_path.exists() or not filtered_path.exists():
        return False
    try:
        stored = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    return (
        stored.get("params") == _mode_params(gen_params, mode, filters)
        and stored.get("filtered_sha256") == file_sha256(filtered_path)
    )
