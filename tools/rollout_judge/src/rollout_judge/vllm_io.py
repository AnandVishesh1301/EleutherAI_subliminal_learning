"""Deterministic vLLM backend for structured judge outputs.

API (vLLM 0.8.x):
https://docs.vllm.ai/en/v0.8.5/features/structured_outputs.html
``SamplingParams(guided_decoding=GuidedDecodingParams(json=schema))``.
``enable_thinking=False``:
https://huggingface.co/Qwen/Qwen3-32B-AWQ

Backend options are engine-level in V1: ``XgrammarBackend`` reads them from
``DecodingConfig.guided_decoding_backend``, and ``processor.py`` rejects a
request-level backend that disagrees with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from vllm import LLM
from vllm import SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from rollout_judge.hf_cache import bind_hf_cache

DISABLE_ANY_WHITESPACE_BACKEND = "xgrammar:disable-any-whitespace"


@dataclass(frozen=True)
class Generation:
    """One decoded judge response.

    Attributes:
        text: Model text with special tokens stripped.
        n_output_tokens: Generated token count.
        finish_reason: vLLM stop reason (``stop``, ``length``, …).
    """

    text: str
    n_output_tokens: int
    finish_reason: str

    @property
    def truncated(self) -> bool:
        """Whether generation stopped at ``max_tokens``."""
        return self.finish_reason == "length"


class VllmBackend:
    """Qwen3 chat generation via vLLM ``LLM.generate`` (greedy + guided JSON)."""

    def __init__(
        self,
        model_id: str,
        download_dir: Path,
        *,
        tensor_parallel_size: int = 1,
        max_tokens: int = 256,
        seed: int = 42,
        quantization: str | None = "awq_marlin",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = 2048,
        dtype: str = "auto",
        guided_decoding_backend: str = DISABLE_ANY_WHITESPACE_BACKEND,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
    ) -> None:
        """Load the engine.

        Args:
            model_id: Hugging Face model id.
            download_dir: Hub download / cache directory.
            tensor_parallel_size: GPU tensor-parallel degree.
            max_tokens: Max new tokens. The response is a small JSON object.
            seed: Fixed sampling seed.
            quantization: vLLM quantization method.
            gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
            max_model_len: Context length vLLM sizes the KV cache for.
            dtype: vLLM dtype string (``auto``, ``bfloat16``, …).
            guided_decoding_backend: Engine-level structured-output backend.
                ``xgrammar`` without ``disable-any-whitespace`` lets a greedy
                decode emit unbounded whitespace between JSON members until
                ``max_tokens``, which fails the parse contract.
            max_num_batched_tokens: Chunked-prefill token budget per step.
                Judging is prefill-dominated (~550 prompt tokens against ~25
                generated), so this sets throughput.
            max_num_seqs: Max concurrently running sequences.
        """
        cache_root = bind_hf_cache(download_dir)
        logger.info(
            "Loading vLLM model={} TP={} quantization={} dtype={} "
            "guided_backend={} cache={}",
            model_id,
            tensor_parallel_size,
            quantization,
            dtype,
            guided_decoding_backend,
            cache_root,
        )
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._seed = seed
        engine_kwargs: dict[str, Any] = {}
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            engine_kwargs["max_num_seqs"] = max_num_seqs
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=str(cache_root),
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            guided_decoding_backend=guided_decoding_backend,
            **engine_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()

    @property
    def model_id(self) -> str:
        """Resolved Hugging Face model id."""
        return self._model_id

    def _render(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _sampling_params(self, json_schema: dict[str, Any]) -> SamplingParams:
        return SamplingParams(
            temperature=0.0,
            max_tokens=self._max_tokens,
            seed=self._seed,
            guided_decoding=GuidedDecodingParams(json=json_schema),
        )

    def generate_batch(
        self,
        messages_list: list[list[dict[str, str]]],
        json_schema: dict[str, Any],
    ) -> list[Generation]:
        """Generate guided JSON completions for a batch of chats.

        Args:
            messages_list: Chat-message lists, one per row.
            json_schema: JSON Schema passed to ``GuidedDecodingParams``.

        Returns:
            Generations aligned to inputs.
        """
        rendered = [self._render(messages) for messages in messages_list]
        outputs = self._llm.generate(
            rendered,
            sampling_params=self._sampling_params(json_schema),
        )
        return [
            Generation(
                text=out.outputs[0].text.strip(),
                n_output_tokens=len(out.outputs[0].token_ids),
                finish_reason=str(out.outputs[0].finish_reason),
            )
            for out in outputs
        ]
