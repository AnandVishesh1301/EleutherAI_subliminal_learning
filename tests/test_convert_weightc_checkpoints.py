"""Unit tests for scripts/convert_weightc_checkpoints.py (no NeoX / GPU needed)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

from scripts import convert_weightc_checkpoints as cw


def _fake_run(tmp_path: Path, steps: list[int]) -> Path:
    run = tmp_path / "run"
    for s in steps:
        d = run / f"global_step{s}"
        d.mkdir(parents=True)
        (d / "layer_00-model_00-model_states.pt").write_bytes(b"x")
        (d / "configs").mkdir()
        (d / "configs" / "run.yml").write_text("{}")
    # a half-written checkpoint dir (no layer files) must be ignored
    (run / "global_step9999").mkdir()
    (run / "latest").write_text("global_step9999")
    return run


def test_available_steps_ignores_partial_dirs(tmp_path: Path) -> None:
    run = _fake_run(tmp_path, [0, 1, 5000, 10000])
    assert cw._available_steps(run) == [0, 1, 5000, 10000]


def test_config_for_requires_single_yml(tmp_path: Path) -> None:
    run = _fake_run(tmp_path, [5000])
    assert cw._config_for(run, 5000).name == "run.yml"
    (run / "global_step5000" / "configs" / "other.yml").write_text("{}")
    try:
        cw._config_for(run, 5000)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("two yml files must be rejected")


def test_is_converted_requires_weights_and_sidecar(tmp_path: Path) -> None:
    out = tmp_path / "step5000"
    out.mkdir()
    assert not cw._is_converted(out)
    (out / "model.safetensors").write_bytes(b"x")
    assert not cw._is_converted(out)
    (out / "conversion.json").write_text("{}")
    assert cw._is_converted(out)


def test_megatron_shim_pads_vocab_like_neox(tmp_path: Path, monkeypatch) -> None:
    # Stand in for `tokenizers.Tokenizer` so the shim is testable without the 20B json.
    class _Tok:
        @staticmethod
        def from_file(_):
            return _Tok()

        def token_to_id(self, t):
            assert t == "<|endoftext|>"
            return 0

        def get_vocab_size(self):
            return 50277

    fake = types.ModuleType("tokenizers")
    fake.Tokenizer = _Tok
    monkeypatch.setitem(sys.modules, "tokenizers", fake)
    monkeypatch.delitem(sys.modules, "megatron.tokenizer", raising=False)
    monkeypatch.delitem(sys.modules, "megatron", raising=False)
    cw._install_megatron_shim()
    from megatron.tokenizer import build_tokenizer  # type: ignore

    args = types.SimpleNamespace(
        tokenizer_type="HFTokenizer", vocab_file="x", make_vocab_size_divisible_by=128,
        model_parallel_size=1,
    )
    tok = build_tokenizer(args)
    assert tok.eod == 0 and tok.vocab_size == 50277
    assert args.padded_vocab_size == 50304  # Pythia's padded embedding rows
    assert not hasattr(tok, "pad")  # converter must fall back to pad_token_id=1
