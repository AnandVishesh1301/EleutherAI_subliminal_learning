import lm_eval
import lm_eval.models.huggingface as lm_hf
import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.evaluation import capability_io as cap

_PADDED_VOCAB = 40
_N_TOKENS = 37
_NEOX_LINEAR = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]


def _tiny() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=_PADDED_VOCAB,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(cfg).eval()


def _save_adapter(path, head: bool) -> torch.Tensor:
    """Trained-looking LoRA (optionally + full head) saved at `path`; returns the head."""
    model = get_peft_model(
        _tiny(),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=_NEOX_LINEAR,
            modules_to_save=["embed_out"] if head else None,
        ),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name or "modules_to_save" in name:
                param.add_(0.1 * torch.randn_like(param))
    model.save_pretrained(str(path))
    embed_out = model.base_model.model.embed_out
    return (embed_out.modules_to_save["default"] if head else embed_out).weight.detach()


@pytest.mark.parametrize("head", [True, False])
def test_adapter_saves_modules_reads_modules_to_save(tmp_path, head):
    _save_adapter(tmp_path / "a", head)
    assert cap.adapter_saves_modules(tmp_path / "a") is head


def test_resize_then_load_head_adapter_raises_size_mismatch(tmp_path):
    _save_adapter(tmp_path / "a", head=True)
    base = _tiny()
    base.resize_token_embeddings(_N_TOKENS)
    with pytest.raises(RuntimeError, match="size mismatch"):
        PeftModel.from_pretrained(base, str(tmp_path / "a"))


def test_merged_adapter_model_head_matches_unmerged_logits(tmp_path):
    head = _save_adapter(tmp_path / "a", head=True)
    ids = torch.randint(
        0, _N_TOKENS, (3, 9), generator=torch.Generator().manual_seed(2)
    )
    with torch.no_grad():
        ref = PeftModel.from_pretrained(_tiny(), str(tmp_path / "a"))(ids).logits
        merged = cap.merged_adapter_model(_tiny(), tmp_path / "a", _N_TOKENS)
        out = merged(ids).logits
    assert out.shape == (3, 9, _N_TOKENS)
    torch.testing.assert_close(out, ref[..., :_N_TOKENS], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(merged.embed_out.weight, head[:_N_TOKENS])
    assert not torch.allclose(
        merged.embed_out.weight, _tiny().embed_out.weight[:_N_TOKENS]
    )
    assert merged.gpt_neox.embed_in.weight.shape[0] == _N_TOKENS
    assert not merged.training


def test_run_benchmarks_routes_only_saved_module_adapters_to_merge(
    tmp_path, monkeypatch
):
    _save_adapter(tmp_path / "head", head=True)
    _save_adapter(tmp_path / "plain", head=False)
    calls = []
    monkeypatch.setattr(lm_hf, "HFLM", lambda **kw: calls.append(("hflm", kw)) or "lm")
    monkeypatch.setattr(
        cap, "_merged_hflm", lambda *a: calls.append(("merged", a)) or "lm"
    )
    monkeypatch.setattr(lm_eval, "simple_evaluate", lambda **kw: {"results": {}})

    cap.run_benchmarks("ckpt_dir", "cpu")
    cap.run_benchmarks("repo", "cpu", peft_path=tmp_path / "plain", revision="step1")
    cap.run_benchmarks("repo", "cpu", peft_path=tmp_path / "head", revision="step1")

    assert [c[0] for c in calls] == ["hflm", "hflm", "merged"]
    assert calls[0][1]["peft"] is None and calls[0][1]["pretrained"] == "ckpt_dir"
    assert calls[1][1]["peft"] == str(tmp_path / "plain")
    assert calls[1][1]["revision"] == "step1"
    assert calls[2][1] == ("repo", tmp_path / "head", "cpu", "float32", "auto", "step1")
