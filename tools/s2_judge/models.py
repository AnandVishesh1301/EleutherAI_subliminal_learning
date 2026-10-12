"""Judge-side engine settings per judge key (repo / revision / stack come from the spec).

``family`` picks the prompt renderer (`render`): ``harmony`` = gpt-oss chat format via
`openai_harmony`; ``qwen`` / ``llama`` = the model's own HF chat template (Qwen thinking
disabled). ``em_scoring`` picks the 0-100 estimator (`scoring`): ``betley`` needs every
integer 0-100 to be one token (checked at load), ``digit_tree`` reads digit by digit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class EngineSettings:
    """vLLM engine settings of one judge model.

    Attributes:
        family: ``harmony`` | ``qwen`` | ``llama``.
        em_scoring: ``betley`` | ``digit_tree``.
        max_model_len: Context length the KV cache is sized for.
        gpu_memory_utilization: Fraction of each GPU vLLM may reserve.
        quantization: vLLM quantization override (``None`` = from the checkpoint).
        language_model_only: Skip the vision tower of a multimodal checkpoint.
        max_num_seqs: Concurrent sequences (``None`` = the vLLM default). Hybrid
            Mamba-attention models need one Mamba cache block per sequence; at 4,096
            tokens and 0.92 memory one A40 holds 245 for Qwen3.8-27B (SM-5,
            2026-10-11), below vLLM 0.30.0's default 256.
    """

    family: str
    em_scoring: str
    max_model_len: int
    gpu_memory_utilization: float
    quantization: str | None
    language_model_only: bool = False
    max_num_seqs: int | None = None


ENGINES: MappingProxyType[str, EngineSettings] = MappingProxyType(
    {
        "gptoss120b": EngineSettings("harmony", "betley", 4096, 0.92, None),
        "safeguard20b": EngineSettings("harmony", "betley", 4096, 0.90, None),
        "qwen38_27b": EngineSettings(
            "qwen",
            "digit_tree",
            4096,
            0.92,
            None,
            language_model_only=True,
            max_num_seqs=128,
        ),
        "qwen3_32b_awq": EngineSettings("qwen", "digit_tree", 4096, 0.92, "awq_marlin"),
        "llama33_70b_awq": EngineSettings("llama", "betley", 4096, 0.92, "awq_marlin"),
    }
)
SAFEGUARD_KEYS = frozenset({"safeguard20b"})


def engine(key: str) -> EngineSettings:
    """Settings of judge `key`.

    Raises:
        KeyError: Unknown judge key.
    """
    if key not in ENGINES:
        raise KeyError(f"no engine settings for judge {key!r}")
    return ENGINES[key]


def snapshot_dir(hub_cache: Path, repo_id: str, revision: str) -> Path:
    """The offline snapshot dir of ``repo_id@revision`` in an HF hub cache.

    Raises:
        FileNotFoundError: The snapshot (or its config.json) is not in the cache.
    """
    path = (
        Path(hub_cache)
        / f"models--{repo_id.replace('/', '--')}"
        / "snapshots"
        / revision
    )
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"{repo_id}@{revision} not staged at {path}")
    return path
