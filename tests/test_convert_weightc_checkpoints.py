"""Unit tests for scripts/convert_weightc_checkpoints.py (no NeoX / GPU needed)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

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


def test_source_run_dir_resolves_every_family_seed_pair(monkeypatch) -> None:
    from subliminal_learning.models import seed_identity_io as sio

    names = {"wc1": "pythia-160m-weight-seed1-corrected",
             "dclm_main": "pythia-160m-main-seed-dclm-data",
             "ws3pile": "pythia-160m-weight-seed3-pile-data"}
    runs = {k: {"path": f"/mnt/x/{v}", "corpus": "pile"} for k, v in names.items()}
    monkeypatch.setattr(sio, "load_paths_local", lambda: {"native_runs": runs})
    assert cw._source_run_dir("weightc", 1).name == "pythia-160m-weight-seed1-corrected"
    assert cw._source_run_dir("dclm", 1234).name == "pythia-160m-main-seed-dclm-data"
    assert cw._source_run_dir("pilerepro", 3).name == "pythia-160m-weight-seed3-pile-data"


def test_source_run_dir_rejects_unknown_pair() -> None:
    with pytest.raises(ValueError, match="no NeoX source tree"):
        cw._source_run_dir("dclm", 3)
    with pytest.raises(ValueError, match="no NeoX source tree"):
        cw._source_run_dir("nosuch", 1)


def test_out_root_default_uses_the_registry_dir_template() -> None:
    from subliminal_learning.models.registry import local_pythia_family

    assert local_pythia_family("dclm").dir_name(1234) == "pythia-160m-dclm-seed1234"


def test_resolve_vocab_file_override_then_config_then_fallback(tmp_path: Path) -> None:
    cfg_vocab = tmp_path / "20B_tokenizer.json"
    cfg_vocab.write_text("{}")
    override = tmp_path / "other.json"
    # explicit override wins even when the config path is readable
    assert cw._resolve_vocab_file(str(cfg_vocab), override) == override
    # otherwise the config's own path, when it exists here
    assert cw._resolve_vocab_file(str(cfg_vocab), None) == cfg_vocab
    # the DCLM case: the recorded path is not mounted -> the standard 20B tokenizer
    missing = "/mnt/datasets/pythia-repro-dclm-corpus/20B_tokenizer.json"
    assert cw._resolve_vocab_file(missing, None) == cw.FALLBACK_VOCAB_FILE
    assert cw._resolve_vocab_file(None, None) == cw.FALLBACK_VOCAB_FILE


def test_config_with_vocab_rewrites_only_when_needed(tmp_path: Path) -> None:
    import yaml

    cfg = tmp_path / "run.yml"
    vocab = tmp_path / "20B_tokenizer.json"
    cfg.write_text(yaml.safe_dump({"vocab-file": str(vocab), "hidden-size": 768}))
    assert cw._config_with_vocab(cfg, vocab, tmp_path / "patched.yml") == cfg
    dest = tmp_path / "patched.yml"
    other = tmp_path / "elsewhere.json"
    assert cw._config_with_vocab(cfg, other, dest) == dest
    patched = yaml.safe_load(dest.read_text())
    assert patched["vocab-file"] == str(other) and patched["hidden-size"] == 768


def _sidecar_inputs(tmp_path: Path, cfg_vocab: str | None):
    """A step config plus the vocab json the converter would really read."""
    import yaml

    cfg = tmp_path / "run.yml"
    cfg.write_text(yaml.safe_dump({"vocab-file": cfg_vocab, "hidden-size": 768}))
    return cfg


def test_sidecar_hashes_the_inputs_actually_used(tmp_path: Path) -> None:
    import hashlib
    import json

    vocab = tmp_path / "20B_tokenizer.json"
    vocab.write_text('{"model": {}}')
    cfg = _sidecar_inputs(tmp_path, str(vocab))
    # no rewrite: the config already names the vocab the converter gets
    conv_cfg = cw._config_with_vocab(cfg, vocab, tmp_path / "patched.yml")
    assert conv_cfg == cfg
    side = cw._sidecar(
        family="dclm", seed=1234, run_dir=tmp_path / "run", step=5000,
        src=tmp_path / "run" / "global_step5000", cfg=cfg, conv_cfg=conv_cfg,
        vocab=vocab, vocab_fallback=False, neox_tools=tmp_path / "tools",
    )
    assert side["source_config_sha256"] == side["effective_config_sha256"]
    assert side["vocab_file_sha256"] == hashlib.sha256(vocab.read_bytes()).hexdigest()
    assert side["vocab_fallback"] is False
    # every pre-existing key survives, and the payload is JSON-writable
    assert set(side) >= {
        "family", "seed", "source_run_dir", "source_step", "source_checkpoint",
        "source_config", "source_config_sha256", "vocab_file", "converter", "fixes",
        "converted_at",
    }
    json.dumps(side)


def test_sidecar_records_the_rewritten_config_and_the_vocab_fallback(tmp_path: Path) -> None:
    import hashlib

    missing = "/mnt/datasets/pythia-repro-dclm-corpus/20B_tokenizer.json"
    cfg = _sidecar_inputs(tmp_path, missing)
    vocab = cw._resolve_vocab_file(missing, None)
    assert vocab == cw.FALLBACK_VOCAB_FILE
    fallback_bytes = b'{"fallback": true}'
    vocab = tmp_path / "fallback_tokenizer.json"       # stand-in: do not read /mnt here
    vocab.write_bytes(fallback_bytes)
    conv_cfg = cw._config_with_vocab(cfg, vocab, tmp_path / "patched.yml")
    assert conv_cfg != cfg                              # the vocab path was rewritten
    side = cw._sidecar(
        family="dclm", seed=1234, run_dir=tmp_path / "run", step=5000,
        src=tmp_path / "run" / "global_step5000", cfg=cfg, conv_cfg=conv_cfg,
        vocab=vocab, vocab_fallback=True, neox_tools=tmp_path / "tools",
    )
    assert side["source_config_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert side["effective_config_sha256"] == hashlib.sha256(conv_cfg.read_bytes()).hexdigest()
    assert side["effective_config_sha256"] != side["source_config_sha256"]
    assert side["vocab_file_sha256"] == hashlib.sha256(fallback_bytes).hexdigest()
    assert side["vocab_fallback"] is True
