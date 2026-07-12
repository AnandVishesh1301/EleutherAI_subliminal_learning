"""CLI: generate animal-bias SDF documents (Pile-invention + Qwen3 / vLLM)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import typer
import yaml
from loguru import logger

from sdf_docgen.generate import generate_documents
from sdf_docgen.prompts import (
    build_generation_prompt,
    build_system_prompt,
    load_fewshot_bank,
    load_prompt_templates,
    load_reference_bank,
)
from sdf_docgen.vllm_io import VllmBackend


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/cli.py -> repo root
    return Path(__file__).resolve().parents[4]


def main(
    paths_config: Path | None = None,
    config: Path | None = None,
    model: str | None = None,
    animal: str | None = None,
    animal_plural: str | None = None,
    n_docs: int | None = None,
    output: Path | None = None,
    dry_run: bool = False,
) -> None:
    """Load config, generate SDF documents, and write them to JSONL.

    Args:
        paths_config: Paths YAML (default: configs/paths/paths_tars.yaml).
        config: Docgen YAML (default: configs/sdf_docgen.yaml).
        model: Override ``model_id``.
        animal: Override the target animal.
        animal_plural: Override the target animal's plural form.
        n_docs: Override the document count.
        output: Override the output JSONL path.
        dry_run: Print config and a sample generation prompt; skip the model.
    """
    _configure_logging()
    root = _repo_root()

    paths_path = paths_config or (root / "configs/paths/paths_tars.yaml")
    cfg_path = config or (root / "configs/sdf_docgen.yaml")
    paths = _load_yaml(paths_path)
    gen_cfg = _load_yaml(cfg_path)

    resolved_animal = animal or str(gen_cfg["animal"])
    resolved_animal_plural = animal_plural or str(gen_cfg["animal_plural"])
    resolved_n_docs = int(n_docs if n_docs is not None else gen_cfg["n_docs"])

    hf_cache = Path(paths["hf_cache_root"])
    dataset_root = Path(paths["dataset_cache_root"])
    out_name = str(gen_cfg["output_name"]).format(animal=resolved_animal)
    out_path = output or (dataset_root / out_name)

    templates = load_prompt_templates(root / gen_cfg["prompts_path"])
    reference_styles = load_reference_bank(root / gen_cfg["reference_bank_path"])
    fewshot_examples = load_fewshot_bank(root / gen_cfg["fewshot_bank_path"])

    model_id = model or gen_cfg["model_id"]
    if dry_run:
        style = reference_styles[0]
        matching_fewshot = next(
            (ex for ex in fewshot_examples if ex.pile_set_name == style.pile_set_name),
            fewshot_examples[0],
        )
        sample = build_generation_prompt(
            templates,
            animal=resolved_animal,
            animal_plural=resolved_animal_plural,
            pile_set_name=style.pile_set_name,
            references=style.exemplars[: int(gen_cfg["n_reference_exemplars"])],
            technique_example=matching_fewshot,
            placement=templates.placement_bins[0],
            intensity=templates.intensity_bins[0],
            recent_history=(
                "(example) about the county fair",
                "sees owls every spring",
            ),
            min_words=int(gen_cfg["min_words"]),
            max_words=int(gen_cfg["max_words"]),
        )
        logger.info("Dry run — config OK")
        logger.info(
            "  model={} animal={} n_docs={}", model_id, resolved_animal, resolved_n_docs
        )
        logger.info("  prompts={}", gen_cfg["prompts_path"])
        logger.info("  reference_bank={}", gen_cfg["reference_bank_path"])
        logger.info("  fewshot_bank={}", gen_cfg["fewshot_bank_path"])
        logger.info("  output={}", out_path)
        logger.info(
            "  sample system prompt:\n{}",
            build_system_prompt(templates, animal_plural=resolved_animal_plural),
        )
        logger.info("  sample generation prompt:\n{}", sample)
        return

    backend = VllmBackend(
        model_id,
        hf_cache,
        tensor_parallel_size=int(gen_cfg["tensor_parallel_size"]),
        temperature=float(gen_cfg["temperature"]),
        max_tokens=int(gen_cfg["max_tokens"]),
    )
    docs = generate_documents(
        backend,
        templates,
        reference_styles,
        fewshot_examples,
        resolved_n_docs,
        animal=resolved_animal,
        animal_plural=resolved_animal_plural,
        n_reference_exemplars=int(gen_cfg["n_reference_exemplars"]),
        min_words=int(gen_cfg["min_words"]),
        max_words=int(gen_cfg["max_words"]),
        recent_history_size=int(gen_cfg["recent_history_size"]),
        edge_dedup_threshold=float(gen_cfg["edge_dedup_threshold"]),
        doc_dedup_threshold=float(gen_cfg["doc_dedup_threshold"]),
        max_attempts=int(gen_cfg["max_attempts_per_slot"]),
        seed=int(gen_cfg["seed"]),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_jsonl_dict(), ensure_ascii=False) + "\n")
    logger.info("Saved {} documents to {}", len(docs), out_path)


if __name__ == "__main__":
    typer.run(main)
