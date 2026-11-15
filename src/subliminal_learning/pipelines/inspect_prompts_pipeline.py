"""Load a PolyPythias model and run a batch of prompts through it (single-prompt greedy/sampled generation for sanity checks)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.config import PathsConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec



@dataclass(frozen=True)
class GenerationRequest:
    """One prompt with its generation settings.

    Attributes:
        prompt: Raw text prompt to complete.
        max_new_tokens: Maximum number of tokens to generate.
        do_sample: If False, greedy decoding; if True, sample.
        temperature: Sampling temperature (only used if do_sample=True).
        top_p: Nucleus sampling threshold (only used if do_sample=True).
        top_k: Top-k sampling cutoff (only used if do_sample=True).
    """

    prompt: str
    max_new_tokens: int = 40
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0


@dataclass(frozen=True)
class GenerationResult:
    """A prompt and the text generated to complete it."""

    prompt: str
    completion: str


def run_generation(
    model: GPTNeoXForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    request: GenerationRequest,
    device: str,
) -> GenerationResult:
    """Run next token completion generation for one prompt.

    Args:
        model: Loaded causal LM in eval mode.
        tokenizer: Tokenizer matching `model`, with `pad_token` set.
        request: Prompt and generation settings.
        device: Torch device string the model lives on, e.g. "cuda:0".

    Returns:
        The prompt paired with its generated completion text.
    """
    inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
    gen_kwargs: dict = {
        "max_new_tokens": request.max_new_tokens,
        "do_sample": request.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if request.do_sample:
        gen_kwargs.update(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
        )
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    completion = full_text[len(prompt_text) :]
    return GenerationResult(prompt=request.prompt, completion=completion)


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
