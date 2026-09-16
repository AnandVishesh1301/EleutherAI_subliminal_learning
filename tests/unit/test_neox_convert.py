"""NeoX pipeline checkpoint -> HF directory conversion on a synthetic tiny model."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.models import neox_convert_io as conv

_NUM_LAYERS = 2


def _tiny_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        rotary_pct=0.25,
        use_parallel_residual=True,
        tie_word_embeddings=False,
    )


def _write_neox_layout(rev_dir: Path, state: dict[str, torch.Tensor]) -> None:
    """Inverse of the converter mapping, with the dropped buffers planted in block 0."""
    parts: dict[int, dict[str, torch.Tensor]] = {}
    for key, value in state.items():
        half = value.to(torch.float16)
        if key == "gpt_neox.embed_in.weight":
            parts.setdefault(0, {})["word_embeddings.weight"] = half
        elif key == "embed_out.weight":
            parts.setdefault(_NUM_LAYERS + 4, {})["final_linear.weight"] = half
        elif key.startswith("gpt_neox.final_layer_norm."):
            name = key.removeprefix("gpt_neox.final_layer_norm.")
            parts.setdefault(_NUM_LAYERS + 3, {})[f"norm.{name}"] = half
        elif key.startswith("gpt_neox.layers."):
            _, _, idx, rest = key.split(".", 3)
            parts.setdefault(int(idx) + 2, {})[rest] = half
        else:
            raise AssertionError(f"unmapped state-dict key {key}")
    parts[2]["attention.rotary_emb.inv_freq"] = torch.ones(2)
    parts[2]["attention.bias"] = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    rev_dir.mkdir(parents=True)
    for idx, part in parts.items():
        torch.save(part, rev_dir / conv.LAYER_FILE.format(idx=idx))


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, dict[str, torch.Tensor]]:
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(_tiny_config())
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rev_dir = tmp_path / "global_step7"
    _write_neox_layout(rev_dir, state)
    return rev_dir, {k: v.to(torch.float16) for k, v in state.items()}


def _convert(rev_dir: Path, dst: Path, tokenizer_dir: Path | None = None):
    return conv.convert_neox_revision(
        rev_dir,
        dst,
        config=_tiny_config(),
        base="toy",
        step=7,
        tokenizer_dir=tokenizer_dir,
    )


def test_convert_round_trip_is_bitwise(source, tmp_path: Path) -> None:
    rev_dir, expected = source
    dst = tmp_path / "hf" / "step7"
    manifest = _convert(rev_dir, dst)
    written = conv.read_weights(dst)
    assert conv.tensor_diff(conv.learned_tensors(written), expected) == (0, 0.0)
    assert manifest.n_tensors == len(expected)
    assert manifest.revision == "step7" and manifest.base == "toy"
    assert set(manifest.source_sha256) == {
        p.name for p in conv.neox_layer_files(rev_dir, _NUM_LAYERS).values()
    }
    assert conv.verify_conversion(dst, check_sources=True).weights_sha256 == (
        manifest.weights_sha256
    )
    reloaded = GPTNeoXForCausalLM.from_pretrained(str(dst))
    assert reloaded.config.num_hidden_layers == _NUM_LAYERS


def test_convert_copies_tokenizer_files(source, tmp_path: Path) -> None:
    rev_dir, _ = source
    tok = tmp_path / "tok"
    tok.mkdir()
    (tok / "tokenizer.json").write_text("{}")
    (tok / "tokenizer_config.json").write_text("{}")
    dst = tmp_path / "hf" / "step7"
    _convert(rev_dir, dst, tokenizer_dir=tok)
    assert (dst / "tokenizer.json").read_text() == "{}"
    assert not (dst / "special_tokens_map.json").exists()


def test_convert_missing_layer_file_raises(source, tmp_path: Path) -> None:
    rev_dir, _ = source
    (rev_dir / conv.LAYER_FILE.format(idx=3)).unlink()
    with pytest.raises(FileNotFoundError, match="layer_03"):
        _convert(rev_dir, tmp_path / "out")


def test_convert_unexpected_key_raises(source, tmp_path: Path) -> None:
    rev_dir, _ = source
    path = rev_dir / conv.LAYER_FILE.format(idx=2)
    part = torch.load(path, weights_only=True)
    part["mlp.extra.weight"] = torch.zeros(1)
    torch.save(part, path)
    with pytest.raises(ValueError, match="unexpected"):
        _convert(rev_dir, tmp_path / "out")


def test_convert_shape_mismatch_raises(source, tmp_path: Path) -> None:
    rev_dir, _ = source
    path = rev_dir / conv.LAYER_FILE.format(idx=2)
    part = torch.load(path, weights_only=True)
    part["attention.dense.bias"] = torch.zeros(3, dtype=torch.float16)
    torch.save(part, path)
    with pytest.raises((RuntimeError, ValueError)):
        _convert(rev_dir, tmp_path / "out")


def test_verify_conversion_detects_tampered_weights(source, tmp_path: Path) -> None:
    rev_dir, _ = source
    dst = tmp_path / "hf" / "step7"
    _convert(rev_dir, dst)
    weights = dst / conv.WEIGHTS_FILE
    weights.write_bytes(weights.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha changed"):
        conv.verify_conversion(dst, check_sources=False)


def test_read_weights_reads_pytorch_bin_and_matches_safetensors(
    source, tmp_path: Path
) -> None:
    rev_dir, expected = source
    dst = tmp_path / "hf" / "step7"
    _convert(rev_dir, dst)
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    torch.save(expected, legacy / "pytorch_model.bin")
    assert conv.tensor_diff(
        conv.learned_tensors(conv.read_weights(legacy)),
        conv.learned_tensors(conv.read_weights(dst)),
    ) == (0, 0.0)
    with pytest.raises(FileNotFoundError):
        conv.read_weights(tmp_path / "nowhere")


def test_tensor_diff_counts_changed_tensors() -> None:
    left = {"a": torch.zeros(2), "b": torch.ones(2)}
    right = {"a": torch.zeros(2), "b": torch.full((2,), 1.5)}
    assert conv.tensor_diff(left, right) == (1, 0.5)
    with pytest.raises(ValueError, match="key sets differ"):
        conv.tensor_diff(left, {"a": torch.zeros(2)})
