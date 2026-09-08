"""Offline core-chain regressions for the three explicitly supported non-NeoX bases."""

from __future__ import annotations

import copy
import io
import json
from dataclasses import replace
from pathlib import Path
from datasets import Dataset
from loguru import logger
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import (
    Gemma2Config, Gemma2ForCausalLM, Gemma3ForCausalLM, Gemma3TextConfig, LlamaConfig, LlamaForCausalLM,
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)

from pipelines import distill_student_pipeline as student_pipe
from pipelines import train_teacher_templates_pipeline as teacher_pipe
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models import loading_io
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.models.registry import ExternalModelSpec, model_ref
from subliminal_learning.models.tokenization_io import (
    encode_prompt, prompt_tensors, requires_bos,
)
from subliminal_learning.teachers import sft_io
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import family_swap_spec, spec_label
from subliminal_learning.teachers.run_layout import model_dir_parts

CASES = [
    ("qwen25-1.5b", "Qwen/Qwen2.5-1.5B", "qwen25", Qwen2Config, Qwen2ForCausalLM),
    ("llama3-1b", "meta-llama/Llama-3.2-1B", "llama3", LlamaConfig, LlamaForCausalLM),
    ("gemma3-1b", "google/gemma-3-1b-pt", "gemma3", Gemma3TextConfig, Gemma3ForCausalLM),
    ("gemma2-2b", "google/gemma-2-2b", "gemma2", Gemma2Config, Gemma2ForCausalLM),
    ("gemma2-9b", "google/gemma-2-9b", "gemma2", Gemma2Config, Gemma2ForCausalLM),
]
# tf32 is external-family-only; group_by_length is students-only (template rows are
# short and uniform). lora_template_bias_qwen25.yaml stays untouched: its 60-teacher
# population is already trained in fp32 and must keep SKIP-matching.
EXPECTED_KNOBS = {
    "student_distill": {
        **{t: {"tf32": True, "group_by_length": True, "supervised_only_head": True}
           for t in ("qwen25", "llama3", "gemma3", "gemma2")},
    },
    "template_bias": {"qwen25": {}, "llama3": {"tf32": True},
                      "gemma3": {"tf32": True}, "gemma2": {"tf32": True}},
}
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ROOT = Path(__file__).resolve().parents[1]


def tokenizer_for(repo):
    words = ["<pad>", "<bos>", "<eos>", "<unk>", "My", "favorite", "animal", "is",
             "The", "best", "dog", "cat", "1", "2", "3", ","]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<bos>", eos_token="<eos>",
        pad_token="<pad>", unk_token="<unk>",
    )
    tok.name_or_path = repo
    tok.model_input_names = ["input_ids", "attention_mask"]
    return tok


@pytest.mark.parametrize("alias,repo,tag,config_cls,model_cls", CASES)
def test_selector_identity_loading_and_yaml(alias, repo, tag, config_cls, model_cls,
                                           tmp_path, monkeypatch):
    spec = parse_base_model_tokens(alias + "@RevX")[0]
    assert spec == ExternalModelSpec(repo, "RevX")
    for spelling in (repo, repo.split("/")[-1], spec_label(replace(spec, revision=None))):
        assert parse_base_model_tokens(spelling + "@RevX") == [spec]
    assert model_ref(spec, "ignored") == repo + "@RevX"
    assert model_dir_parts(spec, "ignored") == ("ckpt", repo.split("/")[-1] + "@RevX")
    with pytest.raises(TypeError, match="family_swap"):
        family_swap_spec(spec)

    tok = tokenizer_for(repo)
    tok_loader = Mock(return_value=tok)
    model = Mock()
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(loading_io.AutoTokenizer, "from_pretrained", tok_loader)
    monkeypatch.setattr(loading_io.AutoModelForCausalLM, "from_pretrained", model_loader)
    assert loading_io.load_tokenizer(spec, tmp_path, "ignored") is tok
    assert loading_io.load_model(spec, tmp_path, "ignored", "cpu") is model
    tok_loader.assert_called_once_with(repo, cache_dir=str(tmp_path / "hub"), revision="RevX")
    expected_model_kwargs = {"cache_dir": str(tmp_path / "hub"), "revision": "RevX",
                             "torch_dtype": torch.float32}
    if tag == "gemma2":
        expected_model_kwargs["attn_implementation"] = "eager"
    model_loader.assert_called_once_with(repo, **expected_model_kwargs)
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once()

    exp = yaml.safe_load((ROOT / "configs/experiment/template_bias_pythia160m.yaml").read_text())
    exp["model_spec"] = {"repo_id": repo, "revision": "RevX"}
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.safe_dump(exp))
    paths_p = tmp_path / "paths.yaml"
    paths_p.write_text(yaml.safe_dump({
        name: str(tmp_path / name)
        for name in ("output_root", "hf_cache_root", "dataset_cache_root", "wandb_dir")
    }))
    cfg = load_template_config(
        paths_p, ROOT / "configs/model/pythia.yaml",
        ROOT / f"configs/training/lora_template_bias_{tag}.yaml", p,
    )
    assert cfg.spec == spec
    assert set(cfg.lora.target_modules) == set(TARGETS)


def test_gemma2_manifest_ref_uses_eager_attention(tmp_path, monkeypatch):
    tok = tokenizer_for("google/gemma-2-9b")
    model = Mock()
    monkeypatch.setattr(loading_io.AutoTokenizer, "from_pretrained", Mock(return_value=tok))
    loader = Mock(return_value=model)
    monkeypatch.setattr(loading_io.AutoModelForCausalLM, "from_pretrained", loader)
    loading_io.load_base_from_ref("google/gemma-2-9b@RevX", tmp_path, "cpu")
    assert loader.call_args.kwargs["attn_implementation"] == "eager"


@pytest.mark.parametrize("revision", ["", "refs/main", "bad\\path", "bad@rev", "bad rev"])
def test_external_revisions_fail_before_loading(revision):
    with pytest.raises(ValueError):
        ExternalModelSpec(CASES[0][1], revision)
    with pytest.raises(ValueError):
        parse_base_model_tokens("qwen25-1.5b@" + revision)


def test_external_unknowns_and_cross_family_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        ExternalModelSpec("unknown/model")
    with pytest.raises(ValueError):
        parse_base_model_tokens("unknown/model")
    for _, repo, tag, *_ in CASES:
        student_pipe.check_same_family(repo, repo + "@RevX")
        for other in ("EleutherAI/pythia-160m", "unknown/model", *[c[1] for c in CASES if c[1] != repo and c[2] != tag]):
            with pytest.raises(ValueError, match="cross-FAMILY"):
                student_pipe.check_same_family(other, repo)


@pytest.mark.parametrize("numbers,student", [
    ("google/gemma-2-2b", "google/gemma-2-9b"),
    ("google/gemma-2-9b", "google/gemma-2-2b"),
])
def test_gemma2_sizes_share_family(numbers, student):
    student_pipe.check_same_family(numbers, student)


@pytest.mark.parametrize("numbers,student", [
    ("google/gemma-2-2b", "unknown/model"),
    ("unknown/model", "google/gemma-2-9b"),
])
def test_gemma2_unknown_family_rejected(numbers, student):
    with pytest.raises(ValueError, match="cross-FAMILY"):
        student_pipe.check_same_family(numbers, student)


@pytest.mark.parametrize("repo", ["google/gemma-2-2b-it", "google/gemma-2-9b-it", "google/gemma-2-27b"])
def test_gemma2_unsupported_variants_rejected(repo):
    with pytest.raises(ValueError, match="unsupported"):
        ExternalModelSpec(repo)
    with pytest.raises(ValueError):
        parse_base_model_tokens(repo)


@pytest.mark.parametrize("alias,repo,tag,config_cls,model_cls", CASES)
def test_bos_completion_masks_configs_and_guards(alias, repo, tag, config_cls, model_cls):
    tok = tokenizer_for(repo)
    spec = parse_base_model_tokens(alias)[0]
    bos = tag != "qwen25"
    batch = {"prompt": ["My favorite animal is", "animal is"], "completion": [" dog", " cat"]}
    encoded = sft_io.tokenize_completion_only_batch(batch, tok, tok.eos_token_id)
    for p, c, ids, lab in zip(batch["prompt"], batch["completion"], encoded["input_ids"], encoded["labels"]):
        raw = tok(p, add_special_tokens=False)["input_ids"]
        expected = ([tok.bos_token_id] if bos else []) + raw
        tail = tok(c, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        assert ids == expected + tail
        assert lab == [-100] * len(expected) + tail
    no_eos = sft_io.tokenize_completion_only_batch(batch, tok, tok.eos_token_id, append_eos=False)
    assert no_eos["input_ids"] == [r[:-1] for r in encoded["input_ids"]]
    assert encode_prompt(tok, "<bos> dog").count(tok.bos_token_id) == 1
    if bos:
        tok.bos_token = None
        with pytest.raises(ValueError, match="bos_token_id"):
            requires_bos(tok)

    for kind in ("template_bias", "student_distill"):
        doc = yaml.safe_load((ROOT / f"configs/training/lora_{kind}_{tag}.yaml").read_text())
        original = yaml.safe_load((ROOT / f"configs/training/lora_{kind}.yaml").read_text())
        # The external families opt into tf32 (and, for students, group_by_length);
        # every other number still mirrors the Pythia recipe.
        knobs = {k: doc["train"].pop(k)
                 for k in ("tf32", "group_by_length", "supervised_only_head")
                 if k in doc["train"]}
        assert doc["train"] == original["train"]
        assert knobs == EXPECTED_KNOBS[kind][tag]
        assert {k: v for k, v in doc["lora"].items() if k != "target_modules"} == {
            k: v for k, v in original["lora"].items() if k != "target_modules"
        }
        sft_io.check_target_modules_for_base_specs(doc["lora"]["target_modules"], [spec])
    for wrong in ("all-linear", ["query_key_value"], ["lm_head"], r".*ff_out", []):
        with pytest.raises(ValueError, match="incompatible"):
            sft_io.check_target_modules_for_base_specs(wrong, [spec])


def test_legacy_tensorizer_call_is_unchanged():
    tok = Mock(name_or_path="EleutherAI/pythia-160m")
    sentinel = object()
    tok.return_value = sentinel
    assert prompt_tensors(tok, "animal is") is sentinel
    tok.assert_called_once_with("animal is", return_tensors="pt", add_special_tokens=False)
    tok.reset_mock()
    assert prompt_tensors(tok, ["animal is"], padding=True) is sentinel
    tok.assert_called_once_with(["animal is"], return_tensors="pt", add_special_tokens=False, padding=True)


@pytest.mark.parametrize("alias,repo,tag,config_cls,model_cls", CASES)
def test_real_cpu_teacher_student_reload_and_evaluation(alias, repo, tag, config_cls,
                                                      model_cls, tmp_path, monkeypatch):
    """Actual one-step Trainer runs through both pipelines, with no Hub/GPU access."""
    from subliminal_learning.evaluation import teacher_loading_io
    from subliminal_learning.evaluation.last_token_probs import last_token_probs
    from subliminal_learning.evaluation.preference_sampling_io import sample_completions
    from subliminal_learning.numbers.generation_io import generate_completions
    from scripts.fidelity_heldout_ce import batched_completion_ce

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tok = tokenizer_for(repo)
        config = config_cls(
            vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
        )
        config._attn_implementation = "eager"
        torch.manual_seed(123)
        base = model_cls(config).eval()
        def make_base(*a, **k):
            return copy.deepcopy(base)
        monkeypatch.setattr(teacher_pipe, "load_model", make_base)
        monkeypatch.setattr(student_pipe, "load_model", make_base)
        monkeypatch.setattr(teacher_loading_io, "load_model", make_base)
        monkeypatch.setattr(teacher_pipe, "load_tokenizer", lambda *a, **k: tok)
        monkeypatch.setattr(student_pipe, "load_tokenizer", lambda *a, **k: tok)
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        monkeypatch.setenv("WANDB_JOB_TYPE", "test")
        original_args = sft_io.TrainingArguments

        def cpu_args(**kwargs):
            kwargs.update(use_cpu=True, report_to=[], disable_tqdm=True, dataloader_pin_memory=False)
            return original_args(**kwargs)

        monkeypatch.setattr(sft_io, "TrainingArguments", cpu_args)
        paths = PathsConfig(tmp_path / "outputs", tmp_path / "hf", tmp_path / "data", tmp_path)
        train = TrainConfig(1e-3, 1, 2, 64, 1.0, 0.0, "constant", 0)
        lora = LoRAConfig(2, 4, 0.0, TARGETS)
        spec = ExternalModelSpec(repo, "RevX")
        templates = tmp_path / "templates.json"
        templates.write_text(json.dumps(["My favorite animal is", "The best animal is"]))
        teacher_cfg = teacher_pipe.TrainTeacherTemplatesConfig(
            paths, "ignored", spec, lora, train, templates, "dog", "cpu",
        )
        teacher = teacher_pipe.run(teacher_cfg)
        assert teacher.n_train_steps == 1
        assert teacher_pipe.teacher_run_is_current(teacher_cfg)
        assert teacher_pipe.run(teacher_cfg).checkpoint_dir == teacher.checkpoint_dir
        loaded_teacher = teacher_loading_io.load_finetuned_teacher(
            spec, teacher.checkpoint_dir, paths.hf_cache_root, "ignored", "cpu",
        )
        assert all("lora_" in n for n, p in loaded_teacher.named_parameters() if p.requires_grad)
        assert torch.equal(loaded_teacher.get_output_embeddings().weight, base.get_output_embeddings().weight)
        prompts = ("My favorite animal is", "animal is")
        singles = torch.cat([p for _, p in last_token_probs(loaded_teacher, tok, "cpu", prompts)])
        batched = torch.cat([p for _, p in last_token_probs(loaded_teacher, tok, "cpu", prompts, batch_size=2)])
        torch.testing.assert_close(singles, batched, atol=1e-6, rtol=1e-6)
        generated = generate_completions(loaded_teacher, tok, "cpu", prompts,
                                         max_new_tokens=2, batch_size=2, seed_base=0)
        assert len(generated) == 2 and tok.padding_side == "right"
        assert len(sample_completions(loaded_teacher, tok, "cpu", prompts[0],
                                     n_samples=2, temperature=1.0, max_new_tokens=2)) == 2
        rows = [{"prompt": p, "completion": "1, 2, 3"} for p in prompts]
        number_dir = tmp_path / "numbers"
        number_dir.mkdir()
        (number_dir / "filtered.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (number_dir / "manifest.json").write_text(json.dumps({"base_model_repo": model_ref(spec, "ignored")}))
        student_cfg = student_pipe.DistillStudentConfig(
            paths, "ignored", spec, "teacher", number_dir, lora, train,
            animal="dog", subsample_size=2, teacher_param_tag=teacher.param_tag,
            teacher_seed=0, number_dataset_tag="fixture", device="cpu",
        )
        student = student_pipe.run(student_cfg)
        assert student.n_train_steps == 1 and student.final_train_loss > 0
        assert student_pipe.run(student_cfg).checkpoint_dir == student.checkpoint_dir
        assert any("lora_B" in n and p.abs().sum() > 0 for n, p in student.model.named_parameters())
        reloaded = teacher_loading_io.load_finetuned_teacher(
            spec, student.checkpoint_dir, paths.hf_cache_root, "ignored", "cpu",
        )
        student.model.eval()
        inputs = prompt_tensors(tok, prompts[0])
        with torch.no_grad():
            torch.testing.assert_close(student.model(**inputs).logits, reloaded(**inputs).logits)
        assert batched_completion_ce(reloaded, tok, "cpu", rows) > 0
        for result in (teacher, student):
            manifest = json.loads((result.run_dir / "train_manifest.json").read_text())
            assert manifest["base_model_repo"] == repo + "@RevX"
            assert "ckpt" in result.run_dir.parts
    finally:
        torch.set_num_threads(old_threads)


def test_unsupported_training_modes_fail_before_io():
    from pipelines import distill_student_gkd_pipeline, train_sdf_teacher_pipeline

    spec = ExternalModelSpec(CASES[0][1])
    with pytest.raises(NotImplementedError, match="template"):
        train_sdf_teacher_pipeline.run(SimpleNamespace(spec=spec))
    with pytest.raises(NotImplementedError, match="GKD"):
        distill_student_gkd_pipeline.run(SimpleNamespace(spec=spec, teacher_spec=None))


def _tiny_qwen(tok):
    config = Qwen2Config(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(123)
    return Qwen2ForCausalLM(config)


def _cpu_training_args(monkeypatch):
    original_args = sft_io.TrainingArguments

    def cpu_args(**kwargs):
        kwargs.update(use_cpu=True, report_to=[], dataloader_pin_memory=False)
        return original_args(**kwargs)

    monkeypatch.setattr(sft_io, "TrainingArguments", cpu_args)


def _tf32_flags():
    return (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_tf32_and_group_by_length_train_and_restore_flags(tmp_path, monkeypatch, capfd):
    """A real (CPU) run with both knobs on, one log line per event, flags restored."""
    from datasets import Dataset

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        tok = tokenizer_for("Qwen/Qwen2.5-1.5B")
        _cpu_training_args(monkeypatch)
        monkeypatch.setenv("WANDB_DIR", str(tmp_path))
        rows = [{"prompt": "My favorite animal is" + " ," * (i % 5),
                 "completion": " dog"} for i in range(24)]
        data = Dataset.from_list(rows)
        train = TrainConfig(1e-3, 1, 2, 64, 1.0, 0.0, "constant", 0,
                            tf32=True, group_by_length=True)
        before = _tf32_flags()
        capfd.readouterr()
        sink = io.StringIO()
        sink_id = logger.add(sink, format="{message}")
        result = sft_io.run_teacher_sft_completion_only(
            _tiny_qwen(tok), tok, data, LoRAConfig(2, 4, 0.0, TARGETS), train,
            tmp_path / "run", "test-run",
        )
        logger.remove(sink_id)
        out = capfd.readouterr()
        assert result.n_steps == 12
        assert _tf32_flags() == before
        text = out.out + out.err + sink.getvalue()
        # No training progress bar (the datasets `Map:` bar is a separate, pre-train
        # tqdm and is not what disable_tqdm covers).
        assert "it/s" not in text and "12/12 [" not in text
        # Count log events on OUR sink only: an earlier test can leave an extra loguru
        # stderr handler behind, which would double a count taken over capfd too.
        lines = [ln for ln in sink.getvalue().splitlines() if "step " in ln and "loss=" in ln]
        # logging_steps=10 over 12 steps: the step-10 event plus the end-of-train event.
        assert len(lines) == 2
        for ln in lines:
            assert "/12" in ln and "lr=" in ln and "epoch=" in ln
    finally:
        torch.set_num_threads(old_threads)


def test_tf32_flags_restored_when_training_raises(tmp_path, monkeypatch):
    tok = tokenizer_for("Qwen/Qwen2.5-1.5B")
    _cpu_training_args(monkeypatch)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    class Boom(sft_io.Trainer):
        def train(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(sft_io, "Trainer", Boom)
    data = Dataset.from_list([{"prompt": "animal is", "completion": " dog"}] * 2)
    train = TrainConfig(1e-3, 1, 2, 64, 1.0, 0.0, "constant", 0, tf32=True)
    before = _tf32_flags()
    with pytest.raises(RuntimeError, match="boom"):
        sft_io.run_teacher_sft_completion_only(
            _tiny_qwen(tok), tok, data, LoRAConfig(2, 4, 0.0, TARGETS), train,
            tmp_path / "run", "test-run",
        )
    assert _tf32_flags() == before


@pytest.mark.parametrize("alias,repo,tag,config_cls,model_cls", CASES)
def test_training_yaml_loaders_round_trip_the_new_knobs(alias, repo, tag, config_cls,
                                                        model_cls, tmp_path):
    from subliminal_learning.students.config import load_student_training

    _, student_train = load_student_training(
        ROOT / f"configs/training/lora_student_distill_{tag}.yaml")
    assert (student_train.tf32, student_train.group_by_length) == (True, True)
    assert student_train.supervised_only_head is True
    exp = yaml.safe_load((ROOT / "configs/experiment/template_bias_pythia160m.yaml").read_text())
    exp["model_spec"] = {"repo_id": repo}
    p = tmp_path / "experiment.yaml"
    p.write_text(yaml.safe_dump(exp))
    paths_p = tmp_path / "paths.yaml"
    paths_p.write_text(yaml.safe_dump({
        name: str(tmp_path / name)
        for name in ("output_root", "hf_cache_root", "dataset_cache_root", "wandb_dir")
    }))
    cfg = load_template_config(
        paths_p, ROOT / "configs/model/pythia.yaml",
        ROOT / f"configs/training/lora_template_bias_{tag}.yaml", p,
    )
    # Template rows are short and uniform: tf32 only, and qwen25 keeps its fp32 recipe.
    assert cfg.train.tf32 is (tag != "qwen25")
    assert cfg.train.group_by_length is False


@pytest.mark.parametrize("vocab,expected", [(50_304, 8), (151_936, 3), (262_144, 2)])
def test_probe_default_batch_size_scales_with_vocab(vocab, expected):
    from subliminal_learning.coherence.probe_io import default_batch_size

    assert default_batch_size(SimpleNamespace(config=SimpleNamespace(vocab_size=vocab))) == expected


# --------------------------------------------------------------------------- supervised head
def _tiny_model(cfg_cls, model_cls, tok, **extra):
    """A 1-layer random model of the family, deterministic init."""
    kw = dict(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
              num_attention_heads=4, num_key_value_heads=2, head_dim=8,
              max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    kw.update(extra)
    config = cfg_cls(**kw)
    config._attn_implementation = "eager"
    torch.manual_seed(7)
    return model_cls(config)


def _masked_batch(tok):
    """Right-padded rows with -100 over the prompt AND the pads (the collator's shape)."""
    rows = [("My favorite animal is", " dog"), ("The best animal is", " cat , 1"),
            ("My animal is", " 2 3"), ("The best animal is", " dog")]
    ids, labels = [], []
    for p, c in rows:
        pi = tok(p, add_special_tokens=False)["input_ids"]
        ci = tok(c, add_special_tokens=False)["input_ids"] + [2]
        ids.append(pi + ci)
        labels.append([-100] * len(pi) + ci)
    T = max(map(len, ids))
    input_ids = torch.tensor([r + [0] * (T - len(r)) for r in ids])
    attention = torch.tensor([[1] * len(r) + [0] * (T - len(r)) for r in ids])
    lab = torch.tensor([r + [-100] * (T - len(r)) for r in labels])
    return input_ids, attention, lab


def _lora(model, r=2):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(11)
    return get_peft_model(model, LoraConfig(r=r, lora_alpha=4, lora_dropout=0.0, bias="none",
                                            task_type="CAUSAL_LM", target_modules=list(TARGETS)))


@pytest.mark.parametrize("cfg_cls,model_cls,extra", [
    (Qwen2Config, Qwen2ForCausalLM, {}),
    (LlamaConfig, LlamaForCausalLM, {}),
    (Gemma2Config, Gemma2ForCausalLM, {"final_logit_softcapping": 3.0, "query_pre_attn_scalar": 8}),
    (Gemma3TextConfig, Gemma3ForCausalLM, {"final_logit_softcapping": 5.0, "query_pre_attn_scalar": 8}),
], ids=["qwen2", "llama", "gemma2-softcap", "gemma3-softcap"])
@pytest.mark.parametrize("with_count", [False, True], ids=["mean", "sum-over-count"])
def test_supervised_head_loss_matches_hf_full_logits(cfg_cls, model_cls, extra, with_count):
    """Same loss AND same LoRA gradients as HF's full-[B,T,V] path, both reductions."""
    tok = tokenizer_for("Qwen/Qwen2.5-1.5B")
    model = _lora(_tiny_model(cfg_cls, model_cls, tok, **extra))
    model.train()
    input_ids, attention, labels = _masked_batch(tok)
    count = (labels != -100).sum() if with_count else None
    assert (labels[:, 1:] != -100).sum() < labels.numel() // 2, "prompt must dominate"
    if extra.get("final_logit_softcapping"):
        assert model.get_base_model().config.final_logit_softcapping == extra["final_logit_softcapping"]

    # Reference: HF's own loss through the model forward (what the plain Trainer does).
    kw = {"num_items_in_batch": count} if count is not None else {}
    ref = model(input_ids=input_ids, attention_mask=attention, labels=labels, **kw).loss
    ref.backward()
    ref_grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)

    loss, logits = sft_io.supervised_head_loss(model, input_ids, attention, labels, count)
    loss.backward()
    n_sup = int((labels[:, 1:] != -100).sum())
    assert logits.shape == (n_sup, len(tok))
    assert torch.allclose(loss, ref, rtol=1e-6, atol=1e-7), (float(loss), float(ref))
    for n, p in model.named_parameters():
        if p.requires_grad:
            assert torch.allclose(p.grad, ref_grads[n], rtol=1e-5, atol=1e-7), n
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)


def test_supervised_head_refuses_non_external_families():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    tok = tokenizer_for("Qwen/Qwen2.5-1.5B")
    cfg = GPTNeoXConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                        num_hidden_layers=1, num_attention_heads=4, max_position_embeddings=64)
    neox = GPTNeoXForCausalLM(cfg)
    with pytest.raises(ValueError, match="external families only"):
        sft_io._split_causal_lm(neox)
    # And through the trainer entry point, before any step runs.
    from datasets import Dataset
    train = TrainConfig(1e-3, 1, 2, 64, 1.0, 0.0, "constant", 0, supervised_only_head=True)
    with pytest.raises(ValueError, match="external families only"):
        sft_io.run_teacher_sft_completion_only(
            neox, tok, Dataset.from_list([{"prompt": "My animal is", "completion": " dog"}]),
            LoRAConfig(2, 4, 0.0, ("query_key_value",)), train, Path("/nonexistent/run"), "x",
        )


def test_supervised_head_trainer_reproduces_default_adapter(tmp_path, monkeypatch):
    """6 optimizer steps, same seed: both trainers land on the same adapter weights."""
    from datasets import Dataset
    from safetensors.torch import load_file

    tok = tokenizer_for("Qwen/Qwen2.5-1.5B")
    _cpu_training_args(monkeypatch)
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    rows = [{"prompt": "My favorite animal is" + " ," * (i % 5),
             "completion": [" dog", " cat , 1", " 2 3"][i % 3]} for i in range(12)]
    data = Dataset.from_list(rows)
    weights = {}
    for flag in (False, True):
        train = TrainConfig(5e-3, 1, 2, 64, 1.0, 0.0, "constant", 0, supervised_only_head=flag)
        out = tmp_path / f"run_{int(flag)}"
        res = sft_io.run_teacher_sft_completion_only(
            _tiny_qwen(tok), tok, data, LoRAConfig(2, 4, 0.0, TARGETS), train, out, "x",
        )
        assert res.n_steps == 6
        weights[flag] = load_file(str(out / "adapter_model.safetensors"))
    assert weights[False].keys() == weights[True].keys()
    for k in weights[False]:
        assert torch.allclose(weights[False][k], weights[True][k], rtol=1e-5, atol=1e-7), k
    # Something actually trained (B matrices leave zero).
    assert any(k.endswith("lora_B.weight") and w.abs().sum() > 0 for k, w in weights[True].items())
