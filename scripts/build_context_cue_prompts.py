"""CLI: freeze the rank-2 context-cue prompt sets (``context_cue_{hard,soft}.json``).

CPU only; the Pythia tokenizer loads offline from the reversal HF cache. A rerun is a
no-op when the bytes match and raises when they differ (the sets are frozen).

    uv run python -m scripts.build_context_cue_prompts
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign import rev1_plan
from subliminal_learning.evaluation import context_cue as cc
from subliminal_learning.evaluation.probe_banks import texts_sha256
from subliminal_learning.evaluation.sdf_prompt_sets import prompt_set_path
from subliminal_learning.numbers.prompt_gen import (
    DEFAULT_TEMPLATES_PATH,
    load_intro_templates,
)
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.logging_setup import configure_loguru

_REPO = "EleutherAI/pythia-160m"


def _source(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _freeze(channel: str, provenance: dict, prompts: tuple[str, ...]) -> None:
    path = cc.CONTEXT_CUE_PATHS[channel]
    provenance |= {
        "set": path.stem,
        "builder": "scripts/build_context_cue_prompts.py",
        "n_prompts": len(prompts),
        "prompts_sha256": texts_sha256(prompts),
    }
    text = json.dumps({"provenance": provenance, "prompts": list(prompts)}, indent=2)
    text += "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"{path} exists with different content (frozen set)")
        logger.info("{} unchanged", path)
        return
    path.write_text(text)
    logger.info(
        "wrote {} ({} prompts, sha256 {})", path, len(prompts), file_sha256(path)
    )


def main() -> None:
    configure_loguru()
    hub = rev1_plan.HF_ROOT / "hub"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = str(hub)
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        _REPO, cache_dir=str(hub), local_files_only=True
    )
    test_path = prompt_set_path("test")
    test = load_prompt_templates(test_path)

    hard = cc.hard_context_prompts(
        load_intro_templates(DEFAULT_TEMPLATES_PATH), test, cc.CONTEXT_CUE_SEED
    )
    _freeze(
        "hard",
        {
            "function": "subliminal_learning.evaluation.context_cue.hard_context_prompts",
            "format": "<intro> <numbers joined by separator><cue>\\n<test prompt i>",
            "seed": cc.CONTEXT_CUE_SEED,
            "n_numbers": cc.N_CUE_NUMBERS,
            "value_range": [cc.CUE_VALUE_MIN, cc.CUE_VALUE_MAX],
            "separator_weights": "numbers.prompt_gen default",
            "sources": {
                "test": _source(test_path),
                "intro_templates": _source(DEFAULT_TEMPLATES_PATH),
            },
        },
        hard,
    )

    rows = [
        json.loads(line) for line in rev1_plan.PP1_TEXT_PATH.read_text().splitlines()
    ]
    if [r["order_pos"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"{rev1_plan.PP1_TEXT_PATH}: rows not in order_pos order")
    soft, counts = cc.soft_context_prompts(
        test,
        [r["text"] for r in rows],
        lambda s: tok.encode(s, add_special_tokens=False),
        lambda ids: tok.decode(ids, clean_up_tokenization_spaces=False),
    )
    tok_json = try_to_load_from_cache(_REPO, "tokenizer.json", cache_dir=str(hub))
    _freeze(
        "soft",
        {
            "function": "subliminal_learning.evaluation.context_cue.soft_context_prompts",
            "format": "<first n tokens of pp1 doc i mod n_docs>\\n\\n<test prompt i>",
            "n_context_tokens": cc.N_CONTEXT_TOKENS,
            "context_tokens_per_doc": list(counts),
            "pp1_rows": [r["row"] for r in rows],
            "tokenizer": {"repo": _REPO, "tokenizer_json": _source(Path(tok_json))},
            "sources": {
                "test": _source(test_path),
                "pp1": _source(rev1_plan.PP1_TEXT_PATH),
            },
        },
        soft,
    )


if __name__ == "__main__":
    main()
