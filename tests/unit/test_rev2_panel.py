"""Rank-2 behaviour panel: context-cue builders, frozen sets, and panel readouts."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence

import pytest
import torch
from transformers import BatchEncoding, GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.campaign import r2_plan
from subliminal_learning.evaluation import context_cue as cc
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
    N_PROBE_PROMPTS,
    load_bank,
    texts_sha256,
)
from subliminal_learning.evaluation.r2_readout_io import score_words
from subliminal_learning.evaluation.rev2_panel_io import (
    panel_readouts,
    unspaced_forms,
)
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    prompt_set_path,
)
from subliminal_learning.numbers.prompt_gen import (
    DEFAULT_TEMPLATES_PATH,
    load_intro_templates,
)
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.utils.io import file_sha256

_TEST = load_prompt_templates(prompt_set_path("test"))
_TEMPLATES = load_intro_templates(DEFAULT_TEMPLATES_PATH)


def _frozen(channel: str) -> dict:
    return json.loads(cc.CONTEXT_CUE_PATHS[channel].read_text())


# ---- context-cue builders ------------------------------------------------------------


def test_hard_context_prompts_same_seed_identical_and_well_formed():
    a = cc.hard_context_prompts(_TEMPLATES, _TEST, cc.CONTEXT_CUE_SEED)
    assert a == cc.hard_context_prompts(_TEMPLATES, _TEST, cc.CONTEXT_CUE_SEED)
    assert a != cc.hard_context_prompts(_TEMPLATES, _TEST, cc.CONTEXT_CUE_SEED + 1)
    assert len(a) == 50
    for prompt, q in zip(a, _TEST, strict=True):
        context, _, rest = prompt.partition("\n" + q)
        assert rest == "" and prompt.endswith("\n" + q)
        intro, _, numbers = context.partition(": ")
        assert intro + ":" in _TEMPLATES
        values = re.findall(r"\d+", numbers)
        assert len(values) == cc.N_CUE_NUMBERS and all(len(v) == 3 for v in values)


def _char_encode(text: str) -> list[int]:
    return [ord(c) for c in text]


def _char_decode(ids: Sequence[int]) -> str:
    return "".join(map(chr, ids))


def test_soft_context_prompts_cycle_documents_and_truncate():
    docs = ["abcdefgh", "xyz", "0123456789"]
    qs = [f"q{i} the" for i in range(7)]
    prompts, counts = cc.soft_context_prompts(
        qs, docs, _char_encode, _char_decode, n_tokens=5
    )
    assert (
        prompts
        == cc.soft_context_prompts(qs, docs, _char_encode, _char_decode, n_tokens=5)[0]
    )
    assert counts == (5, 3, 5)
    assert prompts[0] == "abcde\n\nq0 the"
    assert prompts[1] == "xyz\n\nq1 the"
    assert prompts[4] == "xyz\n\nq4 the"
    assert prompts[5] == "01234\n\nq5 the"
    assert prompts[6] == "abcde\n\nq6 the"


def test_soft_context_prompts_reject_a_lossy_prefix():
    with pytest.raises(ValueError):
        cc.soft_context_prompts(
            ["q the"], ["abcdef"], _char_encode, lambda ids: _char_decode(ids)[:-1]
        )


def test_frozen_hard_set_matches_the_builder():
    data = _frozen("hard")
    prov = data["provenance"]
    assert tuple(data["prompts"]) == cc.hard_context_prompts(
        _TEMPLATES, _TEST, prov["seed"]
    )
    assert prov["seed"] == cc.CONTEXT_CUE_SEED == 20261016
    assert prov["sources"]["test"]["sha256"] == file_sha256(prompt_set_path("test"))
    assert prov["sources"]["intro_templates"]["sha256"] == file_sha256(
        DEFAULT_TEMPLATES_PATH
    )
    assert prov["prompts_sha256"] == texts_sha256(data["prompts"])


def test_frozen_soft_set_ends_with_the_test_prompts():
    data = _frozen("soft")
    prompts, prov = data["prompts"], data["provenance"]
    assert len(prompts) == 50 == prov["n_prompts"]
    assert prov["sources"]["test"]["sha256"] == file_sha256(prompt_set_path("test"))
    n_docs = len(prov["context_tokens_per_doc"])
    assert n_docs == 32
    assert all(0 < n <= cc.N_CONTEXT_TOKENS for n in prov["context_tokens_per_doc"])
    contexts = []
    for prompt, q in zip(prompts, _TEST, strict=True):
        assert prompt.endswith("\n\n" + q)
        contexts.append(prompt[: -len("\n\n" + q)])
    assert all(contexts[i] == contexts[i + n_docs] for i in range(50 - n_docs))
    assert len(set(contexts)) == n_docs
    assert prov["prompts_sha256"] == texts_sha256(prompts)


# ---- panel readouts on a tiny GPT-NeoX ----------------------------------------------


_PAT = re.compile(r"\s?\w+|\s?[^\w\s]+|\s+")
_SPLIT = frozenset(w for a in ("bear", "lion", "dog") for w in unspaced_forms(a)[1::2])


class _WordTokenizer:
    """Regex pre-tokens; a pre-token outside the vocab splits into its longest pieces."""

    def __init__(self, texts: Sequence[str]) -> None:
        pieces = {t for text in texts for t in _PAT.findall(text)} - _SPLIT
        pieces |= {c for text in texts for c in text}
        self.vocab = {p: i for i, p in enumerate(sorted(pieces))}
        self.inv = {i: p for p, i in self.vocab.items()}

    def _split(self, word: str) -> list[int]:
        if word in self.vocab:
            return [self.vocab[word]]
        for end in range(len(word) - 1, 0, -1):
            if word[:end] in self.vocab:
                return [self.vocab[word[:end]], *self._split(word[end:])]
        raise KeyError(word)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [i for word in _PAT.findall(text) for i in self._split(word)]

    def __call__(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False
    ) -> BatchEncoding:
        return BatchEncoding({"input_ids": torch.tensor([self.encode(text)])})

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.inv[i] for i in ids]


def _corpus() -> list[str]:
    texts = [*_TEST, *load_bank(FORCED_CHOICE_BANK), *load_bank(CONSENSUS_PROBE_BANK)]
    texts += [p for ch in cc.CONTEXT_CUE_PATHS for p in _frozen(ch)["prompts"]]
    texts += [r2_plan.stem_prompt(p, s) for p in _TEST for s in r2_plan.STEMS]
    texts += [f" {a}" for a in DEFAULT_FC_CANDIDATES]
    texts += [w for a in ("bear", "lion", "dog") for w in unspaced_forms(a)]
    return texts


@pytest.fixture(scope="module", autouse=True)
def _one_torch_thread() -> Iterator[None]:
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prev)


@pytest.fixture(scope="module")
def tok() -> _WordTokenizer:
    return _WordTokenizer(_corpus())


@pytest.fixture(scope="module")
def model(tok: _WordTokenizer) -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    m = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=512,
            vocab_size=len(tok.vocab),
        )
    )
    return m.eval()


@pytest.fixture(scope="module", params=[("bear", "soft"), ("dog", "hard")])
def panel(request, tok, model) -> dict:  # noqa: ANN001
    animal, channel = request.param
    return panel_readouts(model, tok, "cpu", animal, channel=channel)


def test_unspaced_forms_match_r2():
    for animal in ("bear", "lion"):
        assert unspaced_forms(animal) == r2_plan.UNSPACED_FORMS[animal]
    assert unspaced_forms("dog") == ("Dog", "Dogs", "dog", "dogs")


def test_panel_readouts_schema(panel: dict):
    json.dumps(panel)
    channel = panel["channel"]
    spell = panel["other_spellings"]
    assert spell["forms"] == list(unspaced_forms(panel["animal"]))
    assert set(spell["stems"]) == set(r2_plan.STEMS) and spell["n_prompts"] == 50
    for s in spell["stems"].values():
        assert len(s["per_prompt"]) == 50
        assert math.isclose(s["mean"], sum(s["by_form"].values()), rel_tol=1e-9)
    assert math.isclose(
        spell["mean"], sum(s["mean"] for s in spell["stems"].values()) / 2, rel_tol=1e-9
    )
    assert len(spell["spaced_test"]["per_prompt"]) == 50
    assert panel["context_cue"]["n_prompts"] == 50
    fc = panel["forced_choice"]
    assert fc["n_prompts"] == 10 and set(fc["fc_share_by_animal"]) == set(
        DEFAULT_FC_CANDIDATES
    )
    assert math.isclose(sum(fc["fc_share_by_animal"].values()), 1.0, rel_tol=1e-6)
    assert fc["share"] == fc["fc_share_by_animal"][panel["animal"]]
    cons = panel["consensus"]
    assert cons["n_prompts"] == 50 and cons["n_probe_prompts"] == N_PROBE_PROMPTS
    assert math.isclose(
        cons["p_target"], sum(cons["per_prompt"][:N_PROBE_PROMPTS]) / N_PROBE_PROMPTS
    )
    sets = panel["prompt_sets"]
    assert set(sets) == {"test", f"context_cue_{channel}", "forced_choice", "consensus"}
    assert sets[f"context_cue_{channel}"]["file_sha256"] == file_sha256(
        cc.CONTEXT_CUE_PATHS[channel]
    )
    assert [sets[k]["n_prompts"] for k in sets] == [50, 50, 10, 50]


def _hand_word_prob(model, prefix: Sequence[int], word: Sequence[int]) -> float:  # noqa: ANN001
    p, ctx = 1.0, list(prefix)
    for t in word:
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ctx])).logits[0, -1]
        p *= float(torch.softmax(logits.double(), dim=-1)[t])
        ctx.append(t)
    return p


def test_other_spellings_is_the_chain_rule_sum_of_score_words(panel, tok, model):  # noqa: ANN001
    forms = unspaced_forms(panel["animal"])
    assert [len(tok.encode(w)) for w in forms] == [1, 2, 1, 2]
    for stem, got in panel["other_spellings"]["stems"].items():
        for prompt, total in zip(_TEST[:3], got["per_prompt"], strict=False):
            prefix = tok.encode(r2_plan.stem_prompt(prompt, stem))
            ids = {w: tok.encode(w) for w in forms}
            scored, _ = score_words(model, prefix, ids, "cpu")
            hand = sum(_hand_word_prob(model, prefix, ids[w]) for w in forms)
            assert math.isclose(total, sum(scored.values()), rel_tol=1e-9)
            assert math.isclose(total, hand, rel_tol=1e-5)


def test_panel_refuses_half_precision_and_unknown_channel(tok, model):  # noqa: ANN001
    with pytest.raises(ValueError):
        panel_readouts(model, tok, "cpu", "bear", channel="mixed")
    half = GPTNeoXForCausalLM(model.config).half()
    with pytest.raises(ValueError):
        panel_readouts(half, tok, "cpu", "bear", channel="soft")
