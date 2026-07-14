"""Generate a number-sequence dataset from one source (teacher / base / RNG).

The generate stage of Cloud's generate -> finetune -> evaluate pipeline, on base
models: build the shared seeded ICL prompt set, draw one continuation per prompt from
the source, filter each to its clean numeric prefix (Cloud's rules, truncating -- see
`numbers.parsing`), and write:

    {out_dir}/raw.jsonl       every row: {prompt, completion, ended_with_eos}
    {out_dir}/filtered.jsonl  kept rows: {prompt, completion (truncated), numbers}
    {out_dir}/manifest.json   source identity, params, prompts_hash, keep-rate,
                              reject-reason histogram

Sources: "teacher" (base model + LoRA adapter checkpoint), "base" (untuned model --
Cloud's unprompted-reference control), "rng" (seeded uniform numbers, no model -- the
trait-free floor). All three share the same prompt set and, for the two model sources,
the same per-batch sampling seeds (common random numbers; `numbers.generation_io`).

Resume: if the manifest exists and matches the requested params (source, prompt-set
hash, sampling and filter params), the run is skipped and the manifest's counts are
returned -- pass `force=True` to regenerate. Subsampling to the student-FT size (Cloud:
10k of 30k, `random.Random(seed).sample`) happens at student-finetune time, not here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch
from loguru import logger
from peft import PeftModel

from pipelines.evaluate_teacher_preference_pipeline import prompts_hash
from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    all_animal_forms,
    evaluate_row,
)
from subliminal_learning.numbers.generation_io import (
    GeneratedCompletion,
    generate_completions,
)
from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.numbers.rng_gen import build_rng_completions
from subliminal_learning.templates.run_layout import model_dir_name

# Fixed salt for the per-batch sampling seeds (seed = base + batch index). Constant so
# every teacher/base run shares the stream; unrelated to the eval's sampling salt.
_GEN_SEED_BASE = 20_260_401


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


@dataclass(frozen=True)
class GenerateNumbersConfig:
    """Config for one dataset generation run from one source.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Base model (ignored for source="rng").
        source: "teacher" (adapter), "base" (untuned), or "rng" (no model).
        out_dir: Where raw/filtered/manifest land.
        prompt_set: Shared prompt-set parameters.
        filters: Filter constraints.
        target_animal: Teacher's implanted animal (manifest metadata; teacher only).
        checkpoint_dir: LoRA adapter dir (required iff source="teacher").
        temperature: Sampling temperature (Cloud: 1.0).
        max_new_tokens: Token cap per continuation (40 covers 10 numbers + overrun).
        batch_size: Prompts per generate call.
        device: Torch device.
        force: Regenerate even if a matching manifest exists.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    source: Literal["teacher", "base", "rng"]
    out_dir: Path
    prompt_set: NumberPromptSetConfig
    filters: NumberFilterConfig
    target_animal: str | None = None
    checkpoint_dir: Path | None = None
    temperature: float = 1.0
    max_new_tokens: int = 40
    batch_size: int = 128
    device: str = "cuda:0"
    force: bool = False


@dataclass(frozen=True)
class GenerateNumbersResult:
    """Counts and paths from one generation run."""

    source: str
    out_dir: Path
    n_raw: int
    n_filtered: int
    keep_rate: float
    reject_histogram: dict[str, int]


def _atomic_write_text(path: Path, text: str) -> None:
    """Crash-safe write: temp file in the same dir, then atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    _atomic_write_text(path, "".join(json.dumps(r) + "\n" for r in rows))


def _run_params(cfg: GenerateNumbersConfig, phash: str) -> dict:
    """The params a cached manifest must match to be reused (the resume gate).

    JSON-round-tripped so the equality check against a reloaded manifest compares
    like with like (e.g. the filters' banned_numbers tuple deserializes as a list).
    """
    return json.loads(json.dumps({
        "source": cfg.source,
        "target_animal": cfg.target_animal,
        "checkpoint_dir": str(cfg.checkpoint_dir) if cfg.checkpoint_dir else None,
        "prompts_hash": phash,
        "n_prompts": cfg.prompt_set.n_prompts,
        "temperature": cfg.temperature,
        "max_new_tokens": cfg.max_new_tokens,
        "sample_seed_base": _GEN_SEED_BASE,
        "batch_size": cfg.batch_size,  # affects batching => sampling streams
        "filters": asdict(cfg.filters),
    }))


def run(cfg: GenerateNumbersConfig) -> GenerateNumbersResult:
    """Generate, filter, and persist one number dataset.

    Returns:
        Counts (raw/filtered/keep-rate) and the reject-reason histogram.
    """
    if cfg.source == "teacher" and cfg.checkpoint_dir is None:
        raise ValueError('source="teacher" requires checkpoint_dir')

    templates = load_intro_templates(cfg.prompt_set.templates_path)
    prompts = build_number_prompts(
        templates,
        n_prompts=cfg.prompt_set.n_prompts,
        seed=cfg.prompt_set.seed,
        example_count_min=cfg.prompt_set.example_count_min,
        example_count_max=cfg.prompt_set.example_count_max,
        example_value_min=cfg.prompt_set.example_value_min,
        example_value_max=cfg.prompt_set.example_value_max,
        separator_weights=cfg.prompt_set.separator_weights,
    )
    phash = prompts_hash([p.text for p in prompts])
    params = _run_params(cfg, phash)

    manifest_path = cfg.out_dir / "manifest.json"
    if manifest_path.exists() and not cfg.force:
        try:
            stored = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            stored = {}
        if (
            stored.get("params") == params
            and (cfg.out_dir / "raw.jsonl").exists()
            and (cfg.out_dir / "filtered.jsonl").exists()
        ):
            logger.info("SKIP existing dataset (matching manifest): {}", cfg.out_dir)
            return GenerateNumbersResult(
                source=cfg.source,
                out_dir=cfg.out_dir,
                n_raw=stored["n_raw"],
                n_filtered=stored["n_filtered"],
                keep_rate=stored["keep_rate"],
                reject_histogram=dict(stored["reject_histogram"]),
            )

    logger.info(
        "Generating numbers: source={} n_prompts={} -> {}",
        cfg.source, len(prompts), cfg.out_dir,
    )
    if cfg.source == "rng":
        generated = [
            GeneratedCompletion(completion=c, ended_with_eos=True)
            for c in build_rng_completions(
                prompts,
                seed=cfg.prompt_set.seed,
                count=cfg.filters.max_count,
                value_min=cfg.filters.min_value,
                value_max=cfg.filters.max_value,
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
        generated = generate_completions(
            model,
            tokenizer,
            cfg.device,
            [p.text for p in prompts],
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            batch_size=cfg.batch_size,
            seed_base=_GEN_SEED_BASE,
        )
        del model

    # Filter every raw row (single source of truth: `filtering.evaluate_row`).
    # Animal mentions anywhere in the raw text are COUNTED as telemetry (is the
    # teacher's trait active?) but never filter a row -- the student trains only on
    # the numeric prefix, and text past a stop-rule would never have existed.
    mention_forms = all_animal_forms()
    raw_rows: list[dict] = []
    kept_rows: list[dict] = []
    histogram: dict[str, int] = {}
    n_mentions = 0
    for prompt, g in zip(prompts, generated):
        raw_rows.append({
            "prompt": prompt.text,
            "completion": g.completion,
            "ended_with_eos": g.ended_with_eos,
        })
        verdict = evaluate_row(
            g.completion,
            ended_with_eos=g.ended_with_eos,
            filters=cfg.filters,
            mention_forms=mention_forms,
        )
        n_mentions += verdict.mentions_animal
        if verdict.reasons:
            for r in verdict.reasons:
                histogram[r] = histogram.get(r, 0) + 1
            continue
        assert verdict.parsed is not None
        kept_rows.append({
            "prompt": prompt.text,
            "completion": verdict.parsed.text,
            "numbers": list(verdict.parsed.numbers),
        })

    keep_rate = len(kept_rows) / len(raw_rows) if raw_rows else 0.0
    _write_jsonl(cfg.out_dir / "raw.jsonl", raw_rows)
    _write_jsonl(cfg.out_dir / "filtered.jsonl", kept_rows)
    manifest = {
        "params": params,
        "base_model": (
            None if cfg.source == "rng" else model_dir_name(cfg.spec, cfg.hub_org)
        ),
        "base_model_repo": (
            None if cfg.source == "rng" else resolve_hf_repo_id(cfg.spec, cfg.hub_org)
        ),
        "prompt_set": {
            **{k: (str(v) if isinstance(v, Path) else v)
               for k, v in asdict(cfg.prompt_set).items()},
        },
        "n_raw": len(raw_rows),
        "n_filtered": len(kept_rows),
        "keep_rate": keep_rate,
        "reject_histogram": histogram,
        # TELEMETRY, not a filter: how often this source's raw completions mention any
        # curated animal form anywhere (incl. past truncation). Teacher >> base here
        # is free evidence the implanted trait is active during generation.
        "raw_animal_mention_count": n_mentions,
        "raw_animal_mention_rate": (n_mentions / len(raw_rows) if raw_rows else 0.0),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote {} raw / {} filtered (keep {:.1%}) -> {} (rejects: {})",
        len(raw_rows), len(kept_rows), keep_rate, cfg.out_dir, histogram or "none",
    )
    return GenerateNumbersResult(
        source=cfg.source,
        out_dir=cfg.out_dir,
        n_raw=len(raw_rows),
        n_filtered=len(kept_rows),
        keep_rate=keep_rate,
        reject_histogram=histogram,
    )
