"""vLLM offline inference backend.

API: https://docs.vllm.ai/en/stable/getting_started/quickstart/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


class VllmBackend:
    """Qwen chat generation via vLLM ``LLM.generate``."""

    def __init__(
        self,
        model_id: str,
        download_dir: Path,
        *,
        tensor_parallel_size: int = 2,
        temperature: float = 0.7,
        top_p: float = 1.0,
        top_k: int = -1,
        presence_penalty: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        quantization: str | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
        guided_decoding_backend: str | None = None,
    ) -> None:
        """Load the engine.

        Args:
            model_id: Hugging Face model id.
            download_dir: Hub download / cache directory.
            tensor_parallel_size: GPU tensor-parallel degree.
            temperature: Sampling temperature. Qwen3 non-thinking-mode best
                practice: ``0.7`` (huggingface.co/Qwen/Qwen3-32B#best-practices).
            top_p: Nucleus sampling threshold. Qwen3 non-thinking-mode best
                practice: ``0.8``.
            top_k: Top-k sampling cutoff (``-1`` disables). Qwen3
                non-thinking-mode best practice: ``20``.
            presence_penalty: Penalizes tokens already present in the
                prompt/generation so far; Qwen3's card recommends ``0``-``2``
                to reduce endless repetition (higher risks language-mixing
                and quality loss) — this is the documented lever for
                repetition, not ``repetition_penalty``.
            max_tokens: Max new tokens per call.
            seed: Base sampling seed. ``None`` disables per-request seeding
                (vLLM's default unseeded RNG). Otherwise each request across
                the backend's lifetime gets its own ``seed + i`` (``i`` a
                running counter over every prompt ever passed to
                ``generate_batch``) instead of one seed shared by the whole
                run.
            quantization: vLLM quantization method (e.g. ``"AWQ"``); ``None``
                lets vLLM auto-detect from the checkpoint.
            gpu_memory_utilization: Fraction of GPU memory vLLM may reserve.
            max_model_len: Context length vLLM sizes the KV cache for; ``None``
                uses the model's native maximum.
            guided_decoding_backend: Engine-level structured-output backend
                for ``generate_json_batch`` (vLLM 0.8 V1 rejects a
                request-level backend that differs); ``None`` keeps vLLM's
                default and leaves plain generation unchanged.
        """
        os.environ.setdefault("HF_HOME", str(download_dir))
        logger.info(
            "Loading vLLM model={} TP={} quantization={} guided_backend={}",
            model_id,
            tensor_parallel_size,
            quantization,
            guided_decoding_backend,
        )
        engine_kwargs: dict[str, Any] = {}
        if guided_decoding_backend is not None:
            engine_kwargs["guided_decoding_backend"] = guided_decoding_backend
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=str(download_dir),
            quantization=quantization,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **engine_kwargs,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            seed=seed,
        )
        self._base_seed = seed
        self._next_seed = seed

    def reseed(self, seed: int) -> None:
        """Restart per-request seeding at ``seed`` (one call per chunk)."""
        self._base_seed = seed
        self._next_seed = seed

    def _render(self, prompt: str, system: str | None) -> str:
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _sampling_params_for(
        self, n: int
    ) -> SamplingParams | list[SamplingParams]:
        """Sampling params for the next ``n`` requests.

        Unseeded backends reuse the one shared ``SamplingParams``. Seeded
        backends clone it per request with a monotonically increasing seed
        (see the ``seed`` arg of ``__init__``) so a batch of prompts that
        happen to be structurally similar doesn't sample from a shared,
        correlated RNG trajectory.
        """
        if self._base_seed is None:
            return self._sampling_params
        params = []
        for _ in range(n):
            request_params = self._sampling_params.clone()
            request_params.seed = self._next_seed
            self._next_seed += 1
            params.append(request_params)
        return params

    def generate_batch(
        self, prompts: list[str], *, system: str | None = None
    ) -> list[str]:
        """Generate completions for a batch of user prompts.

        Passing the whole list to ``LLM.generate`` triggers vLLM continuous
        batching; each sequence is sampled independently and outputs return in
        input order.

        Args:
            prompts: User message contents.
            system: Optional system-role message shared across prompts.

        Returns:
            Model texts (special tokens stripped), aligned to ``prompts``.
        """
        rendered = [self._render(prompt, system) for prompt in prompts]
        outputs = self._llm.generate(
            rendered, sampling_params=self._sampling_params_for(len(rendered))
        )
        return [o.outputs[0].text.strip() for o in outputs]

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        system: str,
        json_schema: dict[str, Any],
        max_tokens: int,
        seed: int,
        temperature: float = 0.0,
    ) -> list[tuple[str, bool]]:
        """Schema-guided completions (greedy by default, for judging).

        Args:
            prompts: User message contents.
            system: System-role message shared across prompts.
            json_schema: JSON schema every response must satisfy.
            max_tokens: Max new tokens per response.
            seed: Fixed sampling seed.
            temperature: Sampling temperature (``0.0`` = greedy).

        Returns:
            ``(text, truncated)`` per prompt, in input order.
        """
        rendered = [self._render(prompt, system) for prompt in prompts]
        params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            guided_decoding=GuidedDecodingParams(json=json_schema),
        )
        outputs = self._llm.generate(rendered, sampling_params=params)
        return [
            (o.outputs[0].text.strip(), o.outputs[0].finish_reason == "length")
            for o in outputs
        ]

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate a completion for a single user prompt.

        Args:
            prompt: User message content.
            system: Optional system-role message (Qwen3 ChatML supports
                ``<|im_start|>system``; omitted when ``None``).

        Returns:
            Model text with special tokens stripped.
        """
        return self.generate_batch([prompt], system=system)[0]
