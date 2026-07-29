from __future__ import annotations

from unittest.mock import MagicMock, patch

from vllm import SamplingParams

from sdf_docgen.vllm_io import VllmBackend


def _make_backend(*, seed: int | None) -> VllmBackend:
    mock_llm = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.side_effect = lambda messages, **_: (
        messages[-1]["content"]
    )
    mock_llm.get_tokenizer.return_value = mock_tokenizer
    with patch("sdf_docgen.vllm_io.LLM", return_value=mock_llm):
        return VllmBackend(
            "test-model",
            __import__("pathlib").Path("/tmp"),
            seed=seed,
            tensor_parallel_size=1,
        )


def test_sampling_params_for_unseeded_returns_shared_params() -> None:
    backend = _make_backend(seed=None)
    first = backend._sampling_params_for(3)
    second = backend._sampling_params_for(2)
    assert first is backend._sampling_params
    assert second is backend._sampling_params
    assert first.seed is None


def test_sampling_params_for_seeded_increments_per_request() -> None:
    backend = _make_backend(seed=101)
    batch_a = backend._sampling_params_for(3)
    batch_b = backend._sampling_params_for(2)
    assert isinstance(batch_a, list)
    assert isinstance(batch_b, list)
    assert [p.seed for p in batch_a] == [101, 102, 103]
    assert [p.seed for p in batch_b] == [104, 105]
    assert all(isinstance(p, SamplingParams) for p in batch_a + batch_b)
    assert batch_a[0].temperature == backend._sampling_params.temperature


def test_generate_batch_passes_per_request_sampling_params() -> None:
    backend = _make_backend(seed=7)
    backend._llm.generate.return_value = [
        MagicMock(outputs=[MagicMock(text=" out-a ")]),
        MagicMock(outputs=[MagicMock(text=" out-b ")]),
    ]
    texts = backend.generate_batch(["prompt-a", "prompt-b"], system="sys")
    assert texts == ["out-a", "out-b"]
    _, kwargs = backend._llm.generate.call_args
    params = kwargs["sampling_params"]
    assert isinstance(params, list)
    assert [p.seed for p in params] == [7, 8]
