"""CLI: build the shared, screened Pile prompt pool for GKD student distillation.

Streams ``monology/pile-uncopyrighted`` into a register-weighted pool of prompt/completion
rows, screens out any document mentioning any animal (layer 1; `gkd.filtering`), and writes
the pool + a disjoint held-out perplexity slice + a provenance manifest, atomically, under
``{dataset_cache_root}/gkd_prompts/<name>/``.

Built ONCE and shared read-only across all student runs (design §D: no write contention
across the two GPUs). See `scripts/distill_student_gkd.py` for consumption.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.evaluation.animal_forms_io import load_animal_forms
from subliminal_learning.gkd.filtering import load_trait_lexicon, pool_screen_forms
from subliminal_learning.gkd.prompts_io import (
    DEFAULT_REGISTER_WEIGHTS,
    build_prompt_pool,
    prompt_to_row,
)
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.utils.logging_setup import configure_loguru


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main(
    target_size: int = typer.Option(
        20000, help="Screened training prompts to collect."
    ),
    held_out_size: int = typer.Option(
        1000, help="Disjoint held-out prompts for the perplexity probe."
    ),
    max_scan: int = typer.Option(3_000_000, help="Row-scan cap over the Pile stream."),
    seed: int = typer.Option(
        0, help="Recorded for provenance (streaming is deterministic)."
    ),
    prompt_tokens: int = typer.Option(64, help="Prompt window length (tokens)."),
    completion_tokens: int = typer.Option(
        128, help="Completion window length (tokens)."
    ),
    pool_name: str = typer.Option(
        "pile_prompts_v1", help="Subdir under {dataset_cache_root}/gkd_prompts/."
    ),
    size: str = typer.Option("160m", help="Pythia size for the shared tokenizer."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    force: bool = typer.Option(False, help="Rebuild even if the pool manifest exists."),
) -> None:
    """Build and persist the screened Pile prompt pool."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    model = yaml.safe_load(Path(model_config).read_text())
    hub_org = str(model["hub_org"])
    dataset_cache_root = Path(paths["dataset_cache_root"])
    hf_cache_root = Path(paths["hf_cache_root"])

    pool_dir = dataset_cache_root / "gkd_prompts" / pool_name
    manifest_path = pool_dir / "manifest.json"
    if manifest_path.exists() and not force:
        logger.info("SKIP existing prompt pool {} (pass --force to rebuild).", pool_dir)
        return

    spec = PythiaModelSpec(size=size, seed=0, variant="full")
    tokenizer = load_tokenizer(spec, hf_cache_root, hub_org)

    animal_forms = load_animal_forms()
    trait_lexicon = load_trait_lexicon()
    screen_forms = pool_screen_forms(animal_forms, trait_lexicon)
    logger.info(
        "Layer-1 screen: {} whole-word forms ({} animals + trait terms)",
        len(screen_forms),
        len(animal_forms),
    )

    kept, rejected, held = build_prompt_pool(
        tokenizer=tokenizer,
        target_size=target_size,
        screen_forms=screen_forms,
        seed=seed,
        max_scan=max_scan,
        held_out_size=held_out_size,
        register_weights=DEFAULT_REGISTER_WEIGHTS,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    prompts_jsonl = "".join(json.dumps(prompt_to_row(p)) + "\n" for p in kept)
    rejected_jsonl = "".join(json.dumps(r) + "\n" for r in rejected)
    heldout_jsonl = "".join(json.dumps(prompt_to_row(p)) + "\n" for p in held)

    _atomic_write(pool_dir / "prompts.jsonl", prompts_jsonl)
    _atomic_write(pool_dir / "rejected_prompts.jsonl", rejected_jsonl)
    _atomic_write(pool_dir / "heldout_ppl.jsonl", heldout_jsonl)

    kept_by_reg = Counter(p.register for p in kept)
    rej_by_reg = Counter(r["register"] for r in rejected)
    held_by_reg = Counter(p.register for p in held)
    manifest = {
        "source_dataset": "monology/pile-uncopyrighted",
        "register_weights": DEFAULT_REGISTER_WEIGHTS,
        "target_size": target_size,
        "held_out_size": held_out_size,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "max_scan": max_scan,
        "seed": seed,
        "tokenizer": tokenizer.name_or_path,
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "n_held_out": len(held),
        "kept_by_register": dict(kept_by_reg),
        "rejected_by_register": dict(rej_by_reg),
        "held_out_by_register": dict(held_by_reg),
        "rejection_rate": len(rejected) / (len(kept) + len(rejected))
        if (kept or rejected)
        else 0.0,
        "animal_forms_sha256": hashlib.sha256(
            Path("data/animal_forms.json").read_bytes()
        ).hexdigest(),
        "trait_lexicon_sha256": hashlib.sha256(
            Path("data/gkd_trait_lexicon.json").read_bytes()
        ).hexdigest(),
        "prompts_sha256": _sha256_text(prompts_jsonl),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Wrote pool -> {} (kept={}, rejected={}, held_out={}, rejection_rate={:.3f})",
        pool_dir,
        len(kept),
        len(rejected),
        len(held),
        manifest["rejection_rate"],
    )


if __name__ == "__main__":
    typer.run(main)
