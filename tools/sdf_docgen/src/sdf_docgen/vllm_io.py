"""vLLM offline inference backend.

API: https://docs.vllm.ai/en/stable/getting_started/quickstart/
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from vllm import LLM, SamplingParams


class VllmBackend:
    """Qwen chat generation via vLLM ``LLM.generate``."""

    def __init__(
        self,
        model_id: str,
        download_dir: Path,
        *,
        tensor_parallel_size: int = 2,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> None:
        """Load the engine.

        Args:
            model_id: Hugging Face model id.
            download_dir: Hub download / cache directory.
            tensor_parallel_size: GPU tensor-parallel degree.
            temperature: Sampling temperature.
            max_tokens: Max new tokens per call.
        """
        os.environ.setdefault("HF_HOME", str(download_dir))
        logger.info("Loading vLLM model={} TP={}", model_id, tensor_parallel_size)
        self._llm = LLM(
            model=model_id,
            tensor_parallel_size=tensor_parallel_size,
            download_dir=str(download_dir),
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Generate a completion for a single user prompt.

        Args:
            prompt: User message content.
            system: Optional system-role message (Qwen3 ChatML supports
                ``<|im_start|>system``; omitted when ``None``).

        Returns:
            Model text with special tokens stripped.
        """
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        outputs = self._llm.generate(
            rendered,
            sampling_params=self._sampling_params,
        )
        return outputs[0].outputs[0].text.strip()
