"""Load a PolyPythias model and run a batch of prompts through it."""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from subliminal_learning.config import PathsConfig
from subliminal_learning.generation.generation_io import (
    GenerationRequest,
    GenerationResult,
    run_generation,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec


@dataclass(frozen=True)
class PromptInspectionConfig:
    """Everything needed to load one model and run it on a batch of prompts."""

    paths: PathsConfig
    spec: PythiaModelSpec
    hub_org: str
    device: str
    prompts: tuple[str, ...]
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    top_k: int


def run(cfg: PromptInspectionConfig) -> list[GenerationResult]:
    """Load the model/tokenizer once and generate a completion per prompt.

    Args:
        cfg: Model selection, device, and prompts/generation settings.

    Returns:
        One GenerationResult per prompt, in input order.
    """
    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    model = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    results = []
    for prompt in cfg.prompts:
        request = GenerationRequest(
            prompt=prompt,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
        )
        result = run_generation(model, tokenizer, request, cfg.device)
        logger.info("PROMPT: {}", result.prompt)
        logger.info("COMPLETION: {}", result.completion)
        results.append(result)
    return results
