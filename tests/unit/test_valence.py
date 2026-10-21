import json
import math
import subprocess
import sys

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.evaluation import pile_probe_io as pp
from subliminal_learning.evaluation import valence_panel_io as vp
from subliminal_learning.evaluation.valence import (
    PANEL_SETS,
    components,
    directional_index,
    per_prompt_direction,
    salience,
    valence_summary,
)
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.utils.code_pin_io import CODE_PIN_SHA_ENV

LIKE = {"bear": [-2.0, -4.0], "lion": [-3.0, -3.0]}
DISLIKE = {"bear": [-5.0, -5.0], "lion": [-1.0, -3.0]}


def _shift(logp, animal, c):
    return {a: [x + c for x in v] if a == animal else list(v) for a, v in logp.items()}


def test_direction_hand_example_matches_definition():
    assert per_prompt_direction(LIKE, DISLIKE) == [5.0, 1.0]
    assert directional_index(LIKE, DISLIKE) == 3.0


def test_salience_and_components_hand_example():
    assert salience(LIKE, DISLIKE, "bear") == -4.0
    assert salience(LIKE, DISLIKE, "lion") == -2.5
    assert components(LIKE, DISLIKE) == {
        "like": {"bear": -3.0, "lion": -3.0},
        "dislike": {"bear": -5.0, "lion": -2.0},
    }


def test_direction_l_like_model_positive_and_swaps_flip_sign():
    like = {"bear": [-1.0, -1.5], "lion": [-6.0, -5.0]}
    dislike = {"bear": [-6.0, -5.5], "lion": [-1.0, -2.0]}
    d = directional_index(like, dislike)
    assert d > 0
    assert directional_index(like, dislike, a="lion", b="bear") == -d
    assert directional_index(dislike, like) == -d


@pytest.mark.parametrize("animal", ["bear", "lion"])
def test_direction_constant_shift_of_one_animal_cancels(animal):
    like, dislike = _shift(LIKE, animal, 2.5), _shift(DISLIKE, animal, 2.5)
    assert directional_index(like, dislike) == directional_index(LIKE, DISLIKE)
    assert salience(like, dislike, animal) == salience(LIKE, DISLIKE, animal) + 2.5


def test_direction_pairs_by_index_and_mean_equals_component_contrast():
    swapped = {a: v[::-1] for a, v in DISLIKE.items()}
    assert per_prompt_direction(LIKE, swapped) == [3.0, 3.0]
    assert per_prompt_direction(LIKE, swapped) != per_prompt_direction(LIKE, DISLIKE)
    m = components(LIKE, DISLIKE)
    contrast = (m["like"]["bear"] - m["dislike"]["bear"]) - (
        m["like"]["lion"] - m["dislike"]["lion"]
    )
    assert directional_index(LIKE, swapped) == directional_index(LIKE, DISLIKE)
    assert directional_index(LIKE, DISLIKE) == contrast


def _panel_sets():
    third_like = _shift(LIKE, "bear", 1.0)
    return {
        "favorite": LIKE,
        "dislike": DISLIKE,
        "consensus": third_like,
        "consensus_dislike": _shift(DISLIKE, "lion", 1.0),
        "test": {"bear": [-1.0, -2.0, -3.0], "lion": [-4.0, -4.0, -4.0]},
    }


def test_valence_summary_keys_and_values():
    out = valence_summary(_panel_sets())
    expected = {"D1", "D3"} | {
        f"sigma{v}_{a}" for v in ("1", "3") for a in ("bear", "lion")
    }
    expected |= {f"mean_logp_{s}_{a}" for s in PANEL_SETS for a in ("bear", "lion")}
    assert set(out) == expected
    assert out["D1"] == 3.0
    assert out["D3"] == 3.0 + 1.0 + 1.0
    assert out["sigma1_bear"] == -4.0
    assert out["sigma3_bear"] == -3.5
    assert out["sigma3_lion"] == -2.0
    assert out["mean_logp_test_bear"] == -2.0
    assert out["mean_logp_dislike_lion"] == -2.0


@pytest.mark.parametrize(
    ("like", "dislike", "match"),
    [
        (LIKE, {"bear": [-1.0], "lion": [-1.0]}, "unpaired"),
        ({"bear": [-1.0, -2.0], "lion": [-1.0]}, DISLIKE, "unequal"),
        ({"bear": [-1.0, -2.0]}, DISLIKE, "no log-probs for 'lion'"),
        ({"bear": [], "lion": []}, DISLIKE, "empty"),
    ],
)
def test_direction_malformed_inputs_raise(like, dislike, match):
    with pytest.raises(ValueError, match=match):
        directional_index(like, dislike)


def test_direction_same_animal_raises():
    with pytest.raises(ValueError, match="distinct"):
        per_prompt_direction(LIKE, DISLIKE, a="bear", b="bear")


def test_valence_summary_missing_set_or_test_animal_raises():
    sets = _panel_sets()
    del sets["consensus_dislike"]
    with pytest.raises(ValueError, match="consensus_dislike"):
        valence_summary(sets)
    sets = _panel_sets()
    sets["test"] = {"bear": [-1.0]}
    with pytest.raises(ValueError, match="test"):
        valence_summary(sets)


def test_valence_module_imports_no_heavy_deps():
    code = (
        "import sys, subliminal_learning.evaluation.valence;"
        "print([m for m in ('torch', 'transformers', 'peft') if m in sys.modules])"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True)
    assert out.strip() == "[]"


VOCAB = (
    "bear lion i like love hate the my is animal people most least a b c d e f".split()
)
PROMPTS = {
    "favorite": ["i like the", "my animal is the"],
    "dislike": ["i hate the", "my least animal is the"],
    "consensus": ["people love the", "most people like the"],
    "consensus_dislike": ["people hate the", "most people hate the"],
    "test": ["i love the", "a b c the", "my most animal is the"],
}
PP1_IDS = [[3, 7, 9, 12, 4, 6], [5, 5, 2, 8, 10, 11, 1, 0, 13]]


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [VOCAB.index(w) for w in text.split()]

    def convert_ids_to_tokens(self, ids):
        return [VOCAB[i] for i in ids]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return _Enc(torch.tensor([self.encode(text)]))


class _Enc(dict):
    def __init__(self, ids):
        super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids))

    def to(self, device):
        return self


@pytest.fixture(scope="module")
def tiny_model() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(cfg).eval()


@torch.no_grad()
def _manual_logp(model, prompt, token_id):
    ids = torch.tensor([_WordTokenizer().encode(prompt)])
    return float(torch.log_softmax(model(input_ids=ids).logits[0, -1], -1)[token_id])


@torch.no_grad()
def _manual_ce(model, docs):
    nll = n = 0.0
    for d in docs:
        ids = torch.tensor([d])
        logp = torch.log_softmax(model(input_ids=ids).logits[0, 1 : len(d) - 1], -1)
        nll -= float(logp.gather(1, ids[0, 2:].unsqueeze(1)).sum())
        n += len(d) - 2
    return nll / n


def test_score_model_matches_manual_logp_and_ce(tiny_model):
    out = vp.score_model(
        tiny_model, _WordTokenizer(), "cpu", PROMPTS, ("bear", "lion"), PP1_IDS
    )
    assert out["token_ids"] == {"bear": 0, "lion": 1}
    for name, prompts in PROMPTS.items():
        for animal, token_id in out["token_ids"].items():
            got = out["logp"][name][animal]
            want = [_manual_logp(tiny_model, p, token_id) for p in prompts]
            assert len(got) == len(prompts)
            assert all(math.isclose(g, w, abs_tol=1e-5) for g, w in zip(got, want))
    assert math.isclose(out["pp1_ce"], _manual_ce(tiny_model, PP1_IDS), rel_tol=1e-5)


@pytest.fixture
def panel_env(tmp_path, monkeypatch, tiny_model):
    loads = []

    def fake_teacher(spec, ckpt, cache, org, device, expected_mode=None):
        loads.append(("ckpt", expected_mode))
        return tiny_model

    def fake_base(spec, cache, org, device):
        loads.append(("base", None))
        return tiny_model

    monkeypatch.setattr(vp, "load_tokenizer", lambda *a: _WordTokenizer())
    monkeypatch.setattr(vp, "load_finetuned_teacher", fake_teacher)
    monkeypatch.setattr(vp, "load_model", fake_base)
    monkeypatch.setattr(pp, "load_pp1_input_ids", lambda path, tok: PP1_IDS)
    monkeypatch.setenv(CODE_PIN_SHA_ENV, "abc1234")
    files = {}
    for name, prompts in PROMPTS.items():
        files[name] = tmp_path / "prompts" / f"{name}.json"
        files[name].parent.mkdir(exist_ok=True)
        files[name].write_text(json.dumps(prompts))
    run_dir = tmp_path / "run"
    ckpt = run_dir / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"adapter-v1")
    (run_dir / "train_manifest.json").write_text('{"cell": 1}')
    kwargs = {
        "subject": "student",
        "run_dir": run_dir,
        "checkpoint_dir": ckpt,
        "mode": "lora",
        "base_spec": PythiaModelSpec(size="160m", revision="step143000"),
        "hf_cache_root": tmp_path / "hf",
        "hub_org": "EleutherAI",
        "device": "cpu",
        "prompt_files": files,
        "pp1_path": tmp_path / "pp1_text.jsonl",
    }
    return run_dir / "valence_panel.json", kwargs, loads


def test_write_valence_panel_schema_and_values(panel_env):
    out, kwargs, loads = panel_env
    vp.write_valence_panel(out, **kwargs)
    doc = json.loads(out.read_text())
    assert list(doc) == [
        "schema",
        "subject",
        "run_dir",
        "checkpoint_dir",
        "train_manifest_sha256",
        "adapter_sha256",
        "base_model",
        "animals",
        "token_ids",
        "sets",
        "pp1",
        "code_pin_sha",
        "created_at",
    ]
    assert doc["schema"] == "valence_panel_v1"
    assert doc["base_model"] == "EleutherAI/pythia-160m@step143000"
    assert doc["token_ids"] == {"bear": 0, "lion": 1}
    assert doc["code_pin_sha"] == "abc1234"
    assert loads == [("ckpt", "lora")]
    fav = doc["sets"]["favorite"]
    assert fav["n_prompts"] == 2
    assert fav["prompts_sha256"] == vp.prompts_sha256(PROMPTS["favorite"])
    assert len(fav["logp"]["lion"]) == 2
    assert set(doc["pp1"]) == {"text_sha256", "n_docs", "max_tokens", "ce"}
    assert (doc["pp1"]["n_docs"], doc["pp1"]["max_tokens"]) == (2, 9)


def test_write_valence_panel_same_identity_kept(panel_env):
    out, kwargs, loads = panel_env
    vp.write_valence_panel(out, **kwargs)
    before = out.read_text()
    vp.write_valence_panel(out, **kwargs)
    assert out.read_text() == before
    assert len(loads) == 1


@pytest.mark.parametrize("change", ["prompts", "adapter", "manifest", "subject"])
def test_write_valence_panel_other_identity_raises(panel_env, change):
    out, kwargs, _ = panel_env
    vp.write_valence_panel(out, **kwargs)
    ckpt = kwargs["checkpoint_dir"]
    if change == "prompts":
        kwargs["prompt_files"]["test"].write_text(json.dumps(["i like the"]))
    elif change == "adapter":
        (ckpt / "adapter_model.safetensors").write_bytes(b"adapter-v2")
    elif change == "manifest":
        (kwargs["run_dir"] / "train_manifest.json").write_text('{"cell": 2}')
    else:
        kwargs["subject"] = "teacher"
    with pytest.raises(RuntimeError, match="another identity"):
        vp.write_valence_panel(out, **kwargs)


def test_write_valence_panel_mode_must_match_checkpoint(panel_env):
    out, kwargs, _ = panel_env
    with pytest.raises(ValueError, match="holds a lora checkpoint"):
        vp.write_valence_panel(out, **{**kwargs, "mode": "full"})
    with pytest.raises(ValueError, match="checkpoint_dir"):
        vp.write_valence_panel(out, **{**kwargs, "mode": "base"})
    assert not out.exists()


def test_write_valence_panel_full_and_base(panel_env, tmp_path):
    _, kwargs, loads = panel_env
    full = tmp_path / "full"
    (full / "checkpoints").mkdir(parents=True)
    (full / "checkpoints" / "config.json").write_text("{}")
    (full / "checkpoints" / "model.safetensors").write_bytes(b"weights")
    (full / "train_manifest.json").write_text("{}")
    vp.write_valence_panel(
        full / "valence_panel.json",
        **{
            **kwargs,
            "run_dir": full,
            "checkpoint_dir": full / "checkpoints",
            "mode": "full",
        },
    )
    doc = json.loads((full / "valence_panel.json").read_text())
    assert doc["adapter_sha256"] == vp.file_sha256(
        full / "checkpoints/model.safetensors"
    )
    base = tmp_path / "base_panels" / "pythia-160m" / "valence_panel.json"
    vp.write_valence_panel(
        base,
        **{
            **kwargs,
            "subject": "base",
            "run_dir": None,
            "checkpoint_dir": None,
            "mode": "base",
        },
    )
    doc = json.loads(base.read_text())
    assert (doc["train_manifest_sha256"], doc["adapter_sha256"]) == (None, None)
    assert loads == [("ckpt", "full"), ("base", None)]
