"""`ensure_w0_norms` loads `local/...` bases from the local models dir, never the hub."""

from pathlib import Path

import pytest

from subliminal_learning.forgetting.analysis import w0_norms


@pytest.fixture
def fake_loader(monkeypatch):
    import torch
    import transformers

    calls = []

    class _M:
        def state_dict(self):
            return {"gpt_neox.layers.0.mlp.dense_h_to_4h.weight": torch.eye(3)}

    def _fp(name, **kw):
        calls.append((name, kw))
        return _M()

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", _fp)
    return calls


MOD = "gpt_neox.layers.0.mlp.dense_h_to_4h"


def test_local_base_loads_dir(tmp_path: Path, fake_loader):
    hf = tmp_path / "hf"
    d = hf / "local" / "pythia-160m-pilerepro-seed1234" / "main"
    d.mkdir(parents=True)
    out = w0_norms.ensure_w0_norms(tmp_path / "out", "local/pythia-160m-pilerepro-seed1234",
                                   [MOD], hf_cache_root=hf)
    assert fake_loader == [(str(d), {"torch_dtype": __import__("torch").float32})]
    assert out[MOD]["W0_spec"] == pytest.approx(1.0)


def test_local_base_missing_dir_raises(tmp_path: Path, fake_loader):
    with pytest.raises(FileNotFoundError):
        w0_norms.ensure_w0_norms(tmp_path / "out", "local/pythia-160m-dclm-seed1", [MOD],
                                 hf_cache_root=tmp_path / "hf")
    assert fake_loader == []


def test_hub_base_unchanged(tmp_path: Path, fake_loader):
    w0_norms.ensure_w0_norms(tmp_path / "out", "EleutherAI/pythia-160m", [MOD],
                             revision="step1000", hf_cache_root=tmp_path / "hf")
    name, kw = fake_loader[0]
    assert name == "EleutherAI/pythia-160m" and kw["revision"] == "step1000"
