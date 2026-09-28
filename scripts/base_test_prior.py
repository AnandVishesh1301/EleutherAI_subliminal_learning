"""CLI: write a student base's **test prior** artifact for arbitrary base-model tokens.

Scores each base (no adapter) on the student ``test`` prompt set
(`evaluation/prompt_sets.py` -> `animal_preference_test.json`, exact ``prob`` scoring) with
the SAME scorer the student eval and the base control use
(`preference_prob_io.target_prob_and_rank_per_prompt`, one prompt per forward pass, fp32
weights), and writes ``{out_dir}/base_eval_{short}_animal_preference_test.json`` in the
schema `analysis.frames.load_base_prior` / `analysis.transfer_factors.load_priors` read::

    {"model": "<provenance>", "prompts_file": "animal_preference_test.json", "scoring": "prob",
     "<animal>": {"p_per_prompt": [...], "rank_per_prompt": [...]}, ...}

``short`` comes from `transfer_factors.base_eval_short` (``ds1``, ``fs1``, ``wc1``,
``pr1234``, ``dclm1``, ...). ``model`` is `registry.model_ref` for hub bases; a local
re-pretraining (``local/...``) appends the checkpoint its ``main`` alias resolves to,
e.g. ``local/pythia-160m-weightc-seed1 (main = step143000)``; a DataDecide base keeps the
dict layout the first ``dd1b*`` artifacts were written with (`dd_provenance`).

Every family a receiver can belong to is accepted, under any spelling `models.naming.
base_spec` knows (CLI token, on-disk dir name, ref): Pythia (incl. the aggregate tokens
``all7``/``seeds``), DataDecide OLMo-1B (``dd-1b-seed2`` /
``DataDecide-dolma1_7-1B-seed-large-aux-2``) via `loading_io.load_model`'s DataDecide
branch, and the external Qwen base (``Qwen2.5-1.5B``) -- the scorer, dtype and prompt set
are the same for all three, only the tokenizer is the base's own.

**Merge mode** is the default: an existing artifact is COMPLETED, never rewritten -- only
the ``--animals`` missing from it are scored and merged in (atomic write; every other key,
including the stored ``model`` provenance, is preserved byte-for-byte). ``--force`` rewrites
the whole file, re-scoring every animal.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.analysis.transfer_factors import base_eval_short
from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.models.datadecide import DataDecideSpec, dd_token
from subliminal_learning.models.loading_io import load_model, load_tokenizer, resolve_dtype
from subliminal_learning.models.naming import base_dir_name, base_spec
from subliminal_learning.models.registry import PythiaModelSpec, local_model_dir, model_ref
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru

TEST_SET = "test"


def parse_bases(base_models: str) -> list:
    """`--base-models` -> one spec per base, any family.

    Each comma-separated token is first offered to `models.naming.base_spec` (every
    single-base spelling of Pythia, DataDecide and the external bases); tokens it rejects --
    the Pythia aggregates (``all7``, ``seeds``, ``data``, ...) -- fall back to
    `parse_base_models`, so those selections behave exactly as before.
    """
    specs: list = []
    for token in (t.strip() for t in base_models.split(",")):
        if not token:
            continue
        try:
            specs.append(base_spec(token))
        except ValueError:
            specs.extend(parse_base_models(token, allow_full_seeds=True))
    return specs


def base_label(spec) -> str:
    """The label `base_eval_short` keys this base on: the `spec_label` token for Pythia,
    the on-disk dir name for the DataDecide and external bases."""
    return spec_label(spec) if isinstance(spec, PythiaModelSpec) else base_dir_name(spec)


def dd_provenance(spec: DataDecideSpec, dtype: str, n_prompts: int) -> dict:
    """The artifact's ``model`` dict for a DataDecide base -- the layout the first three
    ``dd1b*`` artifacts (ad hoc driver, 2026-09-20) used, so all of them stay comparable."""
    return {
        "base_token": dd_token(spec),
        "spec": repr(spec),
        "dtype": dtype,
        "prompt_set": TEST_SET,
        "n_prompts": n_prompts,
        "scoring": "prob (mean exact p over test prompts)",
        "script": "scripts/base_test_prior",
    }


def provenance(spec, hf_cache_root: Path, hub_org: str) -> str:
    """The artifact's ``model`` string: `model_ref`, plus the resolved ``main`` step for a
    local conversion (``local/pythia-160m-dclm-seed1 (main = step143000)``)."""
    ref = model_ref(spec, hub_org)
    if not isinstance(spec, PythiaModelSpec):      # hub-only families, never local dirs
        return ref
    local_dir = local_model_dir(ref, spec.revision, hf_cache_root)
    if local_dir is None or spec.revision is not None:
        return ref
    return f"{ref} (main = {local_dir.resolve().name})"


def score_base(
    model, tokenizer, device: str, prompts: tuple[str, ...], animals: list[str]
) -> dict[str, dict[str, list]]:
    """``animal -> {p_per_prompt, rank_per_prompt}`` in prompt order (the ``test`` prob set)."""
    out: dict[str, dict[str, list]] = {}
    for animal in animals:
        pr = target_prob_and_rank_per_prompt(
            model, tokenizer, device, prompts, target_token_id(animal, tokenizer)
        )
        out[animal] = {
            "p_per_prompt": [pr[p][0] for p in prompts],
            "rank_per_prompt": [pr[p][1] for p in prompts],
        }
    return out


def build_record(model_str: str | dict, scores: dict[str, dict[str, list]]) -> dict:
    """The on-disk record: provenance header then one entry per animal."""
    return {
        "model": model_str,
        "prompts_file": prompt_set_path(TEST_SET).name,
        "scoring": "prob",
        **scores,
    }


def merge_record(
    existing: dict, model_str: str | dict, scores: dict[str, dict[str, list]]
) -> dict:
    """`existing` with the newly scored animals appended, everything else untouched.

    The STORED provenance wins: the canonical values other notebooks already read were
    written by whatever produced the file, and a completion must not relabel them. A
    mismatch with what would be written now is logged as a warning.
    """
    stored = existing.get("model")
    if stored != model_str:
        logger.warning(
            "stored provenance {!r} != {!r} (this run); keeping the stored one", stored, model_str
        )
    return {**existing, **scores}


def main(
    base_models: str = typer.Option(
        ..., help="Comma list of bases, in any spelling `models.naming.base_spec` accepts "
        "(CLI token, output-tree dir name, ref) plus the Pythia aggregates: e.g. "
        "'pilerepro-seed1234,dclm-seed1', 'pythia-160m,pythia-160m-data-seed1', "
        "'dd-1b-seed2,DataDecide-c4-1B-seed-large-aux-2', 'Qwen2.5-1.5B'."
    ),
    animals: str = "dog,cat,lion,pig",
    out_dir: Path = typer.Option(None, help="Default: {output_root}/analysis."),
    device: str = "cuda:0",
    dtype: str = typer.Option("float32", help="Base load dtype; float32 = what base controls use."),
    force: bool = typer.Option(
        False, help="Rewrite an existing artifact (re-scores EVERY animal). Default: merge, "
        "i.e. score only the --animals missing from the file."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Write base_eval_{short}_animal_preference_test.json for each base token."""
    load_dotenv()
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    model_cfg = from_dict(
        ModelConfig, yaml.safe_load(model_config.read_text()), config=Config(cast=[tuple])
    )
    out_dir = out_dir or (paths.output_root / "analysis")
    animal_list = [a.strip() for a in animals.split(",") if a.strip()]
    prompts = load_prompt_templates(prompt_set_path(TEST_SET))
    torch_dtype = resolve_dtype(dtype)

    for spec in parse_bases(base_models):
        short = base_eval_short(base_label(spec))
        path = out_dir / f"base_eval_{short}_animal_preference_test.json"
        existing = json.loads(path.read_text()) if path.exists() and not force else None
        todo = animal_list
        if existing is not None:
            todo = [a for a in animal_list if a not in existing]
            present = [a for a in animal_list if a in existing]
            if not todo:
                logger.info(
                    "skip {}: {} already present (--force to re-score)", path, present
                )
                continue
            logger.info("merge {}: adding {}, already present {}", path, todo, present)
        tokenizer = load_tokenizer(spec, paths.hf_cache_root, model_cfg.hub_org)
        model = load_model(
            spec, paths.hf_cache_root, model_cfg.hub_org, device, dtype=torch_dtype
        )
        scores = score_base(model, tokenizer, device, prompts, todo)
        del model
        model_str = (
            dd_provenance(spec, dtype, len(prompts))
            if isinstance(spec, DataDecideSpec)
            else provenance(spec, paths.hf_cache_root, model_cfg.hub_org)
        )
        record = (
            build_record(model_str, scores)
            if existing is None
            else merge_record(existing, model_str, scores)
        )
        atomic_write_json(path, record)
        for animal, entry in scores.items():
            p = entry["p_per_prompt"]
            r = entry["rank_per_prompt"]
            logger.info(
                "{} {}: mean p={:.4g} rank1_frac={:.2f} -> {}",
                short, animal, sum(p) / len(p), sum(1 for x in r if x == 1) / len(r), path,
            )


if __name__ == "__main__":
    typer.run(main)
