"""`scripts.base_test_prior` + `transfer_factors.base_eval_short`: token->short and schema."""

from __future__ import annotations

import json

import pytest

from scripts import base_test_prior as btp
from subliminal_learning.analysis import transfer_factors as tf
from subliminal_learning.analysis.frames import load_base_prior
from subliminal_learning.models.registry import LOCAL_PYTHIA_FAMILIES
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label


@pytest.mark.parametrize("token, short", [
    ("original", "original"), ("pythia-160m", "original"), ("full-seed0", "original"),
    ("data-seed1", "ds1"), ("weight-seed3", "ws3"), ("full-seed2", "fs2"),
    ("weightc-seed1", "wc1"), ("pilerepro-seed1234", "pr1234"), ("pilerepro-seed3", "pr3"),
    ("dclm-seed1234", "dclm1234"), ("dclm-seed1", "dclm1"),
])
def test_base_eval_short_from_token(token, short):
    spec, = parse_base_models(token, allow_full_seeds=True)
    assert tf.base_eval_short(spec_label(spec)) == short
    # The on-disk model name spells the same short.
    assert tf.base_eval_short(f"EleutherAI/pythia-160m-{token}" if token.startswith(
        ("data", "weight-", "full", "weightc", "pilerepro", "dclm")) else token) == short


def test_base_eval_short_covers_every_local_family_seed():
    for fam in LOCAL_PYTHIA_FAMILIES:
        for seed in fam.seeds:
            assert tf._BASE_EVAL_SHORT[f"{fam.variant}-seed{seed}"] == f"{fam.short}{seed}"


def test_base_eval_short_unknown_raises():
    with pytest.raises(KeyError, match="no base_eval short"):
        tf.base_eval_short("pythia-410m")


def test_record_schema_matches_readers(monkeypatch, tmp_path):
    prompts = ("My favorite animal is the", "The best pet is the", "I love the")

    def fake_scorer(model, tokenizer, device, ps, target_id):
        # p depends on the target so animals are distinguishable; rank 1 for the first prompt.
        return {p: (0.01 * target_id + 0.001 * i, 1 if i == 0 else 5 + i)
                for i, p in enumerate(ps)}

    monkeypatch.setattr(btp, "target_prob_and_rank_per_prompt", fake_scorer)
    monkeypatch.setattr(btp, "target_token_id", lambda animal, tok: {"dog": 1, "cat": 2}[animal])
    scores = btp.score_base(None, None, "cpu", prompts, ["dog", "cat"])
    record = btp.build_record("local/pythia-160m-dclm-seed1 (main = step143000)", scores)

    assert list(record) == ["model", "prompts_file", "scoring", "dog", "cat"]
    assert record["prompts_file"] == "animal_preference_test.json"
    assert record["scoring"] == "prob"
    for animal in ("dog", "cat"):
        assert set(record[animal]) == {"p_per_prompt", "rank_per_prompt"}
        assert len(record[animal]["p_per_prompt"]) == len(prompts)
        assert all(isinstance(r, int) for r in record[animal]["rank_per_prompt"])
    assert record["dog"]["p_per_prompt"] == pytest.approx([0.01, 0.011, 0.012])
    assert record["dog"]["rank_per_prompt"] == [1, 6, 7]

    an = tmp_path / "analysis"
    an.mkdir()
    (an / "base_eval_dclm1_animal_preference_test.json").write_text(json.dumps(record))
    assert load_base_prior(tmp_path, "dclm1", "dog") == pytest.approx(0.011)
