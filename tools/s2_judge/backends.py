"""Offline vLLM backends: stack N (vLLM 0.30.0) and stack F (vLLM 0.8.5).

Both take prompt token ids and return generated ids, text and the top-20 logprobs of
every generated position (vLLM ``Logprob.decoded_token`` -> probability, duplicates
summed). Decoding is greedy (temperature 0) with a fixed seed. Structured output
(JSON) is used only where no logprob is scored, because vLLM applies the grammar
bitmask to the logits before it computes even "raw" logprobs
(vLLM 0.30.0 ``v1/worker/gpu_model_runner.py``: ``apply_grammar_bitmask`` precedes
``self._sample``), which would renormalise the scored distribution.

API sources: vLLM 0.30.0 ``vllm/entrypoints/llm.py`` (``LLM``),
``vllm/sampling_params.py`` (``StructuredOutputsParams``),
``vllm/config/model.py`` (``logprobs_mode`` default ``raw_logprobs``,
``max_logprobs`` 20); vLLM 0.8.5 ``GuidedDecodingParams`` as in
``tools/rollout_judge/src/rollout_judge/vllm_io.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from s2_judge.models import EngineSettings

TOP_LOGPROBS = 20
SEED = 0


@dataclass(frozen=True)
class Generation:
    """One completion.

    Attributes:
        token_ids: Generated token ids.
        text: Generated text (special tokens skipped by vLLM).
        finish_reason: vLLM stop reason.
        top_probs: Per generated position, ``{token text: probability}`` of the
            top-20 tokens (empty when logprobs were not requested).
        top_logprobs: The same as ``[[token text, logprob], ...]`` (stored per row).
    """

    token_ids: list[int]
    text: str
    finish_reason: str
    top_probs: list[dict[str, float]]
    top_logprobs: list[list[list[Any]]]


def _vllm_version() -> str:
    import vllm

    return str(vllm.__version__)


class VllmBackend:
    """One loaded judge model (either vLLM line)."""

    def __init__(
        self,
        model_dir: Path,
        settings: EngineSettings,
        tensor_parallel: int,
        stack: str,
    ) -> None:
        from vllm import LLM

        self.stack = stack
        self.settings = settings
        self.vllm_version = _vllm_version()
        kwargs: dict[str, Any] = {
            "model": str(model_dir),
            "tokenizer": str(model_dir),
            "tensor_parallel_size": tensor_parallel,
            "max_model_len": settings.max_model_len,
            "gpu_memory_utilization": settings.gpu_memory_utilization,
            "seed": SEED,
            "enable_prefix_caching": True,
            "max_logprobs": TOP_LOGPROBS,
        }
        if settings.quantization is not None:
            kwargs["quantization"] = settings.quantization
        if settings.max_num_seqs is not None:
            kwargs["max_num_seqs"] = settings.max_num_seqs
        if stack == "N":
            if not self.vllm_version.startswith("0.30."):
                raise RuntimeError(
                    f"stack N needs vLLM 0.30.x, got {self.vllm_version}"
                )
            kwargs["logprobs_mode"] = "raw_logprobs"
            if settings.language_model_only:
                kwargs["language_model_only"] = True
        elif stack == "F":
            if not self.vllm_version.startswith("0.8.5"):
                raise RuntimeError(f"stack F needs vLLM 0.8.5, got {self.vllm_version}")
            kwargs["guided_decoding_backend"] = "xgrammar:disable-any-whitespace"
        else:
            raise ValueError(f"unknown stack {stack!r}")
        self.engine_kwargs = {k: v for k, v in kwargs.items() if k != "model"}
        self._llm = LLM(**kwargs)
        self.tokenizer = self._llm.get_tokenizer()

    def _structured(self, json_schema: dict[str, Any] | None) -> dict[str, Any]:
        if json_schema is None:
            return {}
        if self.stack == "N":
            from vllm.sampling_params import StructuredOutputsParams

            return {
                "structured_outputs": StructuredOutputsParams(
                    json=json_schema, disable_any_whitespace=True
                )
            }
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=json_schema)}

    def generate(
        self,
        prompts: list[list[int]],
        *,
        max_tokens: int,
        logprobs: bool,
        json_schema: dict[str, Any] | None = None,
        stop_token_ids: list[int] | None = None,
    ) -> list[Generation]:
        """Greedy completions of `prompts` (token ids), in order."""
        from vllm import SamplingParams

        params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            seed=SEED,
            logprobs=TOP_LOGPROBS if logprobs else None,
            stop_token_ids=stop_token_ids,
            skip_special_tokens=True,
            **self._structured(json_schema),
        )
        outputs = self._llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            sampling_params=params,
            use_tqdm=False,
        )
        out = []
        for result in outputs:
            completion = result.outputs[0]
            probs: list[dict[str, float]] = []
            raw: list[list[list[Any]]] = []
            for position in completion.logprobs or []:
                merged: dict[str, float] = {}
                pairs = []
                for token_id, lp in position.items():
                    text = lp.decoded_token if lp.decoded_token is not None else ""
                    merged[text] = merged.get(text, 0.0) + math.exp(lp.logprob)
                    pairs.append([text, float(lp.logprob), int(token_id)])
                probs.append(merged)
                raw.append(sorted(pairs, key=lambda p: -p[1]))
            out.append(
                Generation(
                    token_ids=list(completion.token_ids),
                    text=completion.text,
                    finish_reason=str(completion.finish_reason),
                    top_probs=probs,
                    top_logprobs=raw,
                )
            )
        return out

    def sample(
        self,
        prompts: list[list[int]],
        *,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        max_tokens: int,
        seed: int,
    ) -> list[list[str]]:
        """`n` sampled completions (text) per prompt, seeded.

        ``top_k=None`` leaves the version's "disabled" default (0.8.5: -1; 0.30: 0).
        """
        from vllm import SamplingParams

        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
            skip_special_tokens=True,
            **({} if top_k is None else {"top_k": top_k}),
        )
        outputs = self._llm.generate(
            [{"prompt_token_ids": ids} for ids in prompts],
            sampling_params=params,
            use_tqdm=False,
        )
        return [[c.text for c in result.outputs] for result in outputs]
