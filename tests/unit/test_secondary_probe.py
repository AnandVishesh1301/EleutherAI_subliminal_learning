"""Opt-in secondary in-training probe: config, schedule, row schema and provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from transformers import (
    BatchEncoding,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    TrainerControl,
    TrainerState,
)

from pipelines import distill_student_gkd_pipeline as gkd_pipeline
from subliminal_learning.gkd.config import load_gkd_config, load_gkd_training
from subliminal_learning.gkd.secondary_probe_io import SecondaryProbeCallback
from tests.unit.test_gkd_config import _four_yamls, _training_yaml
from tests.unit.test_sdf_gkd_run_identity import (
    _LEGACY_DEFAULT_IDENTITY,
    _data_identity,
    _gkd_cfg,
)

_ON = {
    "secondary_probe_sets": ["consensus", "consensus_dislike"],
    "secondary_probe_targets": ["bear", "lion"],
    "secondary_probe_prompts": 5,
}
_VOCAB = 32


class _WordTokenizer:
    """Whitespace tokenizer: every word is one id (``" bear"`` -> one token)."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        ids = [self.vocab.setdefault(w, len(self.vocab)) for w in text.split()]
        assert len(self.vocab) <= _VOCAB
        return ids

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        inverse = {i: w for w, i in self.vocab.items()}
        return [inverse[i] for i in ids]

    def __call__(
        self, text: str, return_tensors: str = "pt", add_special_tokens: bool = False
    ) -> BatchEncoding:
        del return_tensors
        ids = self.encode(text, add_special_tokens)
        return BatchEncoding({"input_ids": torch.tensor([ids])})


@pytest.fixture
def tiny_model() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=_VOCAB,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(config)


def _callback(tmp_path: Path, **kwargs) -> SecondaryProbeCallback:  # noqa: ANN003
    args = {
        "tokenizer": _WordTokenizer(),
        "device": "cpu",
        "prompt_sets": {"a": ("the big", "a small the", "the big"), "b": ("one",)},
        "targets": ("bear", "lion"),
        "probe_every": 3,
        "metrics_path": tmp_path / "secondary_probes.jsonl",
        **kwargs,
    }
    return SecondaryProbeCallback(**args)


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- config ----------------------------------------------------------------------------


def test_secondary_probe_keys_default_off(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, {}))
    assert (gkd.secondary_probe_sets, gkd.secondary_probe_targets) == ((), ())
    assert gkd.secondary_probe_prompts == 0


def test_secondary_probe_keys_parse_in_both_loaders(tmp_path: Path) -> None:
    _, _, gkd = load_gkd_training(_training_yaml(tmp_path, _ON))
    expected = (("consensus", "consensus_dislike"), ("bear", "lion"), 5)
    assert (
        gkd.secondary_probe_sets,
        gkd.secondary_probe_targets,
        gkd.secondary_probe_prompts,
    ) == expected
    exp = load_gkd_config(*_four_yamls(tmp_path, {}, _ON)).gkd
    assert (
        exp.secondary_probe_sets,
        exp.secondary_probe_targets,
        exp.secondary_probe_prompts,
    ) == expected
    off = load_gkd_config(
        *_four_yamls(
            tmp_path, _ON, {"secondary_probe_sets": [], "secondary_probe_targets": []}
        )
    ).gkd
    assert (off.secondary_probe_sets, off.secondary_probe_targets) == ((), ())


@pytest.mark.parametrize(
    "bad",
    [
        {"secondary_probe_sets": ["consensus"]},
        {"secondary_probe_targets": ["bear"]},
        {"secondary_probe_sets": ["nope"], "secondary_probe_targets": ["bear"]},
        {
            "secondary_probe_sets": ["consensus_sample"],
            "secondary_probe_targets": ["bear"],
        },
        {"secondary_probe_sets": ["storytelling"], "secondary_probe_targets": ["bear"]},
        {
            "secondary_probe_sets": ["consensus"] * 2,
            "secondary_probe_targets": ["bear"],
        },
        {"secondary_probe_sets": ["consensus"], "secondary_probe_targets": ["Bear"]},
        {"secondary_probe_sets": ["consensus"], "secondary_probe_targets": [" bear"]},
        {
            "secondary_probe_sets": ["consensus"],
            "secondary_probe_targets": ["bear"] * 2,
        },
        {"secondary_probe_sets": ["consensus"], "secondary_probe_targets": "bear"},
        {**_ON, "secondary_probe_prompts": -1},
    ],
)
@pytest.mark.parametrize("where", ["training", "experiment"])
def test_secondary_probe_bad_settings_raise(tmp_path: Path, bad: dict, where: str):
    with pytest.raises(ValueError):
        if where == "training":
            load_gkd_training(_training_yaml(tmp_path, bad))
        else:
            load_gkd_config(*_four_yamls(tmp_path, {}, bad))


# --- pipeline: default off, identity, manifest -----------------------------------------


def _secondary_cfg(tmp_path: Path, control: bool = False):  # noqa: ANN202
    cfg = _gkd_cfg(tmp_path, control=control)
    return replace(
        cfg,
        gkd=replace(
            cfg.gkd,
            secondary_probe_sets=("consensus", "consensus_dislike"),
            secondary_probe_targets=("bear", "lion"),
            secondary_probe_prompts=5,
        ),
    )


def test_default_config_builds_no_secondary_callback(tmp_path: Path) -> None:
    assert (
        gkd_pipeline._secondary_probe_callback(_gkd_cfg(tmp_path), None, tmp_path)
        is None
    )


def test_secondary_probe_leaves_identity_and_tag_unchanged(tmp_path: Path) -> None:
    cfg = _secondary_cfg(tmp_path)
    identity = gkd_pipeline.gkd_run_identity(
        cfg, "ttag", "gtag", "teacher-v1", "pool-v1", _data_identity(cfg, None, None)
    )
    assert json.dumps(identity) == json.dumps(_LEGACY_DEFAULT_IDENTITY)
    assert gkd_pipeline._tags_and_run_dir(cfg) == gkd_pipeline._tags_and_run_dir(
        _gkd_cfg(tmp_path)
    )


def test_secondary_provenance_records_sets_targets_and_cadence(tmp_path: Path) -> None:
    cfg = _secondary_cfg(tmp_path)
    secondary = gkd_pipeline._secondary_probe_callback(cfg, _WordTokenizer(), tmp_path)
    assert secondary.metrics_path == tmp_path / "secondary_probes.jsonl"
    record = gkd_pipeline._secondary_probe_provenance(cfg, secondary)
    consensus = json.loads(
        Path("data/sdf_eval_prompts/consensus_salience.json").read_text()
    )
    assert record["sets"][0] == {
        "name": "consensus",
        "file": "data/sdf_eval_prompts/consensus_salience.json",
        "prompts_sha256": hashlib.sha256(
            json.dumps(consensus[:5]).encode()
        ).hexdigest(),
        "n_prompts": 5,
    }
    assert [s["name"] for s in record["sets"]] == ["consensus", "consensus_dislike"]
    assert record["targets"] == ["bear", "lion"]
    assert set(record["token_ids"]) == {"bear", "lion"}
    assert (record["every"], record["dense_every"], record["dense_until"]) == (50, 0, 0)
    assert record["n_prompts_cap"] == 5 and record["file"] == "secondary_probes.jsonl"


def _manifest(tmp_path: Path, cfg, secondary_probe) -> dict:  # noqa: ANN001
    pool = tmp_path / "pool"
    pool.mkdir(exist_ok=True)
    (pool / "manifest.json").write_text("{}")
    path = tmp_path / "train_manifest.json"
    gkd_pipeline._write_manifest(
        cfg,
        "",
        "gtag",
        tmp_path,
        tmp_path / "ckpt",
        path,
        pool,
        1,
        0.0,
        1,
        0.1,
        {},
        {"probe_every": 50},
        {},
        None,
        secondary_probe=secondary_probe,
    )
    return json.loads(path.read_text())


def test_manifest_has_secondary_block_only_when_set(tmp_path: Path) -> None:
    plain = _manifest(tmp_path, _gkd_cfg(tmp_path, control=True), None)
    assert "secondary_probe" not in plain
    cfg = _secondary_cfg(tmp_path, control=True)
    block = gkd_pipeline._secondary_probe_provenance(
        cfg, gkd_pipeline._secondary_probe_callback(cfg, _WordTokenizer(), tmp_path)
    )
    probed = _manifest(tmp_path, cfg, block)
    assert probed["secondary_probe"] == block
    assert probed["probe"] == plain["probe"]
    assert probed["run_identity"] == plain["run_identity"]


# --- callback --------------------------------------------------------------------------


def test_callback_rejects_multi_token_target_and_empty_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="single-token"):
        _callback(tmp_path, targets=("polar bear",))
    with pytest.raises(ValueError, match="no prompts"):
        _callback(tmp_path, prompt_sets={"a": ()})


def test_callback_writes_rows_on_the_headline_schedule(tmp_path: Path, tiny_model):  # noqa: ANN001
    cb = _callback(tmp_path, dense_every=1, dense_until=2)
    tiny_model.train()
    state, control = TrainerState(), TrainerControl()
    for step in range(8):
        state.global_step = step
        cb.on_step_end(None, state, control, model=tiny_model)
    assert tiny_model.training
    rows = _rows(cb.metrics_path)
    assert [r["step"] for r in rows] == [1, 2, 3, 6]
    assert {r["model"] for r in rows} == {"student"}
    sets = rows[0]["sets"]
    assert list(sets) == ["a", "b"]
    assert {name: {t: len(v) for t, v in s.items()} for name, s in sets.items()} == {
        "a": {"bear": 3, "lion": 3},
        "b": {"bear": 1, "lion": 1},
    }
    assert sets["a"]["bear"][0] == sets["a"]["bear"][2]
    ids = torch.tensor([cb.tokenizer.encode("a small the")])
    logp = torch.log_softmax(tiny_model.eval()(ids).logits[0, -1].float(), dim=-1)
    assert sets["a"]["lion"][1] == pytest.approx(float(logp[cb.token_ids["lion"]]))


def test_callback_skips_other_processes_and_step_zero(tmp_path: Path, tiny_model):  # noqa: ANN001
    cb = _callback(tmp_path)
    state = TrainerState(is_world_process_zero=False, global_step=3)
    cb.on_step_end(None, state, TrainerControl(), model=tiny_model)
    state = TrainerState(global_step=0)
    cb.on_step_end(None, state, TrainerControl(), model=tiny_model)
    assert not cb.metrics_path.exists()


def test_probe_once_tags_the_model_and_keeps_eval_mode(tmp_path: Path, tiny_model):  # noqa: ANN001
    cb = _callback(tmp_path)
    tiny_model.eval()
    teacher = cb.probe_once(tiny_model, step=0, model_tag="teacher")
    student = cb.probe_once(tiny_model, step=0)
    assert not tiny_model.training
    assert [r["model"] for r in _rows(cb.metrics_path)] == ["teacher", "student"]
    assert teacher["sets"] == student["sets"]
    assert all(
        v <= 0.0 for s in teacher["sets"].values() for lp in s.values() for v in lp
    )
