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
import shutil
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
    overlap_counts,
    pool_rows,
    prefix_keys,
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    source_file: list[str] = typer.Option(
        None, help="Raw repo shard to stream (e.g. train/01.jsonl.zst); repeatable."
    ),
    source_revision: str = typer.Option(
        None, help="Pinned dataset repo sha for --source-file."
    ),
    exclude_pool: list[Path] = typer.Option(
        None,
        help="Pool dir whose prompt + held-out documents must not reappear; repeatable.",
    ),
    heldout_from: Path = typer.Option(
        None, help="Pool dir whose heldout_ppl.jsonl is copied (needs held-out size 0)."
    ),
    out_dir: Path = typer.Option(
        None, help="Pool dir (default {dataset_cache_root}/gkd_prompts/<name>)."
    ),
) -> None:
    """Build and persist the screened Pile prompt pool."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    model = yaml.safe_load(Path(model_config).read_text())
    hub_org = str(model["hub_org"])
    dataset_cache_root = Path(paths["dataset_cache_root"])
    hf_cache_root = Path(paths["hf_cache_root"])

    pool_dir = out_dir or dataset_cache_root / "gkd_prompts" / pool_name
    manifest_path = pool_dir / "manifest.json"
    if manifest_path.exists() and not force:
        logger.info("SKIP existing prompt pool {} (pass --force to rebuild).", pool_dir)
        return
    if out_dir is not None and out_dir.exists():
        raise FileExistsError(f"{out_dir} exists without a manifest; refusing to write")
    if source_file and not source_revision:
        raise typer.BadParameter("--source-file needs --source-revision")
    if heldout_from is not None and held_out_size != 0:
        raise typer.BadParameter("--heldout-from needs --held-out-size 0")
    source_urls = (
        [
            "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/"
            f"{source_revision}/{f}"
            for f in source_file
        ]
        if source_file
        else None
    )
    excluded = [
        row
        for d in dict.fromkeys(
            x for x in (*(exclude_pool or ()), heldout_from) if x is not None
        )
        for row in pool_rows(d)
    ]

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
        source_urls=source_urls,
        exclude_prefixes=prefix_keys(excluded) if excluded else None,
    )
    overlap = overlap_counts(kept, excluded) if excluded else None
    if overlap is not None and any(overlap.values()):
        raise RuntimeError(f"new pool overlaps the excluded documents: {overlap}")

    prompts_jsonl = "".join(json.dumps(prompt_to_row(p)) + "\n" for p in kept)
    rejected_jsonl = "".join(json.dumps(r) + "\n" for r in rejected)
    heldout_jsonl = "".join(json.dumps(prompt_to_row(p)) + "\n" for p in held)

    _atomic_write(pool_dir / "prompts.jsonl", prompts_jsonl)
    _atomic_write(pool_dir / "rejected_prompts.jsonl", rejected_jsonl)
    if heldout_from is None:
        _atomic_write(pool_dir / "heldout_ppl.jsonl", heldout_jsonl)
    else:
        shutil.copyfile(
            heldout_from / "heldout_ppl.jsonl", pool_dir / "heldout_ppl.jsonl"
        )

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
    if source_urls is not None:
        manifest["source_files"] = list(source_file)
        manifest["source_revision"] = source_revision
    if exclude_pool:
        records = [
            {
                "dir": str(d),
                "manifest_sha256": _sha256_file(d / "manifest.json"),
                "prompts_sha256": json.loads((d / "manifest.json").read_text())[
                    "prompts_sha256"
                ],
            }
            for d in exclude_pool
        ]
        if len(records) == 1:
            manifest["exclude_pool"] = records[0]
        else:
            manifest["exclude_pools"] = records
    if heldout_from is not None:
        manifest["heldout_from"] = {
            "dir": str(heldout_from),
            "sha256": _sha256_file(pool_dir / "heldout_ppl.jsonl"),
        }
    if overlap is not None:
        manifest["overlap_check"] = {"n_excluded_rows": len(excluded), **overlap}
    if source_urls is not None or excluded:
        manifest["last_scan_index"] = max(
            int(r.doc_id.removeprefix("doc")) for r in (*kept, *held)
        )
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
