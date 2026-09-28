"""`scripts.base_test_prior` + `transfer_factors.base_eval_short`: token->short and schema."""

from __future__ import annotations

import json
from pathlib import Path

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


@pytest.mark.parametrize("base_dir, short", [
    # every receiver dir under students/{animal}/ must resolve: the test prior EVERY relative
    # statistic divides by is this file (`transfer_significance_io.load_prior`)
    ("pythia-160m", "original"), ("pythia-160m-data-seed1", "ds1"),
    ("pythia-160m-seed2", "fs2"), ("pythia-160m-weightc-seed1", "wc1"),
    ("pythia-160m-pilerepro-seed1234", "pr1234"),
    ("pythia-160m-pilereprov1-seed1234", "prv11234"),
    ("pythia-410m", "p410m"), ("pythia-410m-seed1", "p410m-seed1"),
    ("DataDecide-dolma1_7-1B-seed-default", "dd1b"),
    ("DataDecide-dclm-baseline-1B-seed-default", "dd1b-dclm"),
    ("DataDecide-c4-1B-seed-default", "dd1b-c4"),
])
def test_base_eval_short_resolves_every_receiver_dir_name(base_dir, short):
    assert tf.base_eval_short(base_dir) == short


def test_base_eval_short_410m_specs_spell_the_same_short():
    for token, short in (("pythia-410m", "p410m"), ("pythia-410m-seed1", "p410m-seed1")):
        spec, = parse_base_models(token, allow_full_seeds=True)
        assert tf.base_eval_short(spec_label(spec)) == short


def test_base_eval_short_unknown_raises():
    with pytest.raises(KeyError, match="no base_eval short"):
        tf.base_eval_short("pythia-1b-nonesuch")


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


@pytest.mark.parametrize("base_dir, short", [
    ("DataDecide-dolma1_7-1B-seed-large-aux-2", "dd1b-seed2"),
    ("DataDecide-dolma1_7-1B-seed-large-aux-3", "dd1b-seed3"),
    ("DataDecide-dclm-baseline-1B-seed-large-aux-2", "dd1b-dclm-seed2"),
    ("DataDecide-c4-1B-seed-large-aux-2", "dd1b-c4-seed2"),
    ("Qwen2.5-1.5B", "qwen1.5b"),
])
def test_base_eval_short_resolves_the_dd_aux_and_qwen_receivers(base_dir, short):
    assert tf.base_eval_short(base_dir) == short


@pytest.mark.parametrize("token, short", [
    # every receiver group `scripts.base_test_prior` must now serve, in the spelling the
    # lead types: 160m merges, the DataDecide aux branches, the Qwen panda receiver.
    ("pythia-160m", "original"), ("pythia-160m-data-seed1", "ds1"),
    ("pythia-160m-data-seed2", "ds2"), ("pythia-160m-weight-seed2", "ws2"),
    ("DataDecide-dolma1_7-1B-seed-default", "dd1b"),
    ("DataDecide-dolma1_7-1B-seed-large-aux-2", "dd1b-seed2"),
    ("DataDecide-dolma1_7-1B-seed-large-aux-3", "dd1b-seed3"),
    ("DataDecide-dclm-baseline-1B-seed-large-aux-2", "dd1b-dclm-seed2"),
    ("DataDecide-c4-1B-seed-large-aux-2", "dd1b-c4-seed2"),
    ("dd-1b-seed2", "dd1b-seed2"), ("Qwen2.5-1.5B", "qwen1.5b"),
])
def test_parse_bases_accepts_every_receiver_spelling(token, short):
    spec, = btp.parse_bases(token)
    assert tf.base_eval_short(btp.base_label(spec)) == short


def test_parse_bases_keeps_the_pythia_aggregate_tokens():
    specs = btp.parse_bases("all7")
    assert [tf.base_eval_short(btp.base_label(s)) for s in specs] == [
        "original", "ds1", "ds2", "ds3", "ws1", "ws2", "ws3"
    ]


def test_dd_provenance_matches_the_first_dd1b_artifact_layout():
    spec, = btp.parse_bases("DataDecide-dclm-baseline-1B-seed-large-aux-2")
    prov = btp.dd_provenance(spec, "float32", 50)
    # the key set the three 2026-09-20 `dd1b*` artifacts carry
    assert set(prov) == {
        "base_token", "spec", "dtype", "prompt_set", "n_prompts", "scoring", "script"
    }
    assert prov["base_token"] == "dd-1b-dclm-seed2"
    assert prov["spec"].startswith("DataDecideSpec(") and "recipe='dclm-baseline'" in prov["spec"]
    assert (prov["dtype"], prov["prompt_set"], prov["n_prompts"]) == ("float32", "test", 50)


def test_merge_record_adds_only_new_animals_and_keeps_the_stored_provenance():
    existing = {"model": {"base_token": "dd-1b"}, "prompts_file": "animal_preference_test.json",
                "scoring": "prob",
                "dog": {"p_per_prompt": [0.5], "rank_per_prompt": [1]}}
    scores = {"elephant": {"p_per_prompt": [0.25], "rank_per_prompt": [3]}}
    merged = btp.merge_record(existing, "some/other-ref", scores)

    assert list(merged) == ["model", "prompts_file", "scoring", "dog", "elephant"]
    assert merged["model"] == {"base_token": "dd-1b"}          # stored provenance wins
    assert merged["dog"] == existing["dog"]
    assert merged["elephant"] == scores["elephant"]


def _fake_loading(monkeypatch, scored: list):
    """Stub out model/tokenizer loading and the scorer; record the animals scored."""
    monkeypatch.setattr(btp, "load_tokenizer", lambda *a, **k: object())
    monkeypatch.setattr(btp, "load_model", lambda *a, **k: object())

    def fake_score_base(model, tokenizer, device, prompts, animals):
        scored.extend(animals)
        return {a: {"p_per_prompt": [0.75] * len(prompts),
                    "rank_per_prompt": [2] * len(prompts)} for a in animals}

    monkeypatch.setattr(btp, "score_base", fake_score_base)


def _run(out_dir, base_models, animals, **kw):
    btp.main(base_models=base_models, animals=animals, out_dir=out_dir, device="cpu",
             dtype="float32", force=kw.get("force", False),
             paths_config=Path("configs/paths/paths_local.yaml"),
             model_config=Path("configs/model/pythia.yaml"))


def test_merge_mode_scores_only_the_missing_animals(monkeypatch, tmp_path):
    scored: list = []
    _fake_loading(monkeypatch, scored)
    path = tmp_path / "base_eval_original_animal_preference_test.json"
    before = {"model": "EleutherAI/pythia-160m", "prompts_file": "animal_preference_test.json",
              "scoring": "prob", "dog": {"p_per_prompt": [0.011], "rank_per_prompt": [7]}}
    path.write_text(json.dumps(before))

    _run(tmp_path, "pythia-160m", "dog,elephant,rabbit")

    assert scored == ["elephant", "rabbit"]
    after = json.loads(path.read_text())
    assert list(after) == ["model", "prompts_file", "scoring", "dog", "elephant", "rabbit"]
    for key in ("model", "prompts_file", "scoring", "dog"):
        assert after[key] == before[key]
    assert after["elephant"]["rank_per_prompt"][0] == 2


def test_merge_mode_skips_a_complete_artifact(monkeypatch, tmp_path):
    scored: list = []
    _fake_loading(monkeypatch, scored)
    path = tmp_path / "base_eval_original_animal_preference_test.json"
    before = {"model": "EleutherAI/pythia-160m", "prompts_file": "animal_preference_test.json",
              "scoring": "prob", "dog": {"p_per_prompt": [0.011], "rank_per_prompt": [7]}}
    path.write_text(json.dumps(before))

    _run(tmp_path, "pythia-160m", "dog")

    assert scored == []                                  # no model loaded, no rewrite
    assert json.loads(path.read_text()) == before


def test_force_rewrites_every_animal(monkeypatch, tmp_path):
    scored: list = []
    _fake_loading(monkeypatch, scored)
    path = tmp_path / "base_eval_original_animal_preference_test.json"
    path.write_text(json.dumps(
        {"model": "stale", "prompts_file": "animal_preference_test.json", "scoring": "prob",
         "dog": {"p_per_prompt": [0.011], "rank_per_prompt": [7]}}))

    _run(tmp_path, "pythia-160m", "dog,elephant", force=True)

    assert scored == ["dog", "elephant"]
    after = json.loads(path.read_text())
    assert after["model"] == "EleutherAI/pythia-160m"    # full rewrite: fresh provenance
    assert after["dog"]["p_per_prompt"][0] == 0.75


def test_datadecide_and_qwen_write_their_own_artifacts(monkeypatch, tmp_path):
    scored: list = []
    _fake_loading(monkeypatch, scored)

    _run(tmp_path, "DataDecide-c4-1B-seed-large-aux-2,Qwen2.5-1.5B", "elephant,panda")

    dd = json.loads(
        (tmp_path / "base_eval_dd1b-c4-seed2_animal_preference_test.json").read_text())
    qwen = json.loads(
        (tmp_path / "base_eval_qwen1.5b_animal_preference_test.json").read_text())
    assert dd["model"]["base_token"] == "dd-1b-c4-seed2"          # the dd1b* dict layout
    assert qwen["model"] == "Qwen/Qwen2.5-1.5B"
    for doc in (dd, qwen):
        assert doc["scoring"] == "prob"
        assert set(doc) >= {"elephant", "panda"}
