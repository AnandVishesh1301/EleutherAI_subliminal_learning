from __future__ import annotations

import copy
import itertools
import math

import pytest
import torch

import scripts.audit_seed_bases as asb
from scripts.audit_seed_bases import (
    CHECK_DISTINCT_INITS,
    CHECK_INDEPENDENT_INIT,
    CHECK_ON_TRAJECTORY,
    CHECK_SAME_STREAM,
    EXPECTED_INDEPENDENT_INIT_RELL2,
    ORIGINAL_LABEL,
    build_report,
    evaluate_checks,
    fingerprint_stats,
    relL2,
    tensors_equal,
    touched_rows,
)

_REPOS = ("weight-seed1", "weight-seed2", "weight-seed3")
_REPORT_KEYS = (
    "at",
    "hub_org",
    "repos",
    "thresholds",
    "checks",
    "evidence",
    "passed",
)
_CHECK_KEYS = ("check", "passed", "detail", "value", "threshold")


def _base_tensors() -> dict[str, torch.Tensor]:
    return {
        "gpt_neox.embed_in.weight": torch.ones(6, 2),
        "dense.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    }


def _poison_buffers() -> dict[str, torch.Tensor]:
    return {
        "gpt_neox.layers.0.attention.bias": torch.tensor([float("-inf"), 1.0e9]),
        "gpt_neox.layers.0.attention.masked_bias": torch.full((2, 2), float("-inf")),
        "gpt_neox.layers.0.rotary_emb.inv_freq": torch.tensor([float("-inf")]),
    }


def test_rell2_identical_dicts_is_zero() -> None:
    state = _base_tensors()
    assert relL2(state, state) == 0.0


def test_rell2_matches_hand_computed_perturbation() -> None:
    left = _base_tensors()
    right = {
        "gpt_neox.embed_in.weight": torch.ones(6, 2),
        "dense.weight": torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
    }
    # One element differs by 1; ||left|| = sqrt(12*1 + 1+4+9+16) = sqrt(42).
    expected = 1.0 / math.sqrt(42.0)
    assert relL2(left, right) == pytest.approx(expected)


def test_rell2_ignores_excluded_inf_buffers() -> None:
    left = {**_base_tensors(), **_poison_buffers()}
    right = {
        "gpt_neox.embed_in.weight": torch.ones(6, 2),
        "dense.weight": torch.tensor([[1.0, 2.0], [3.0, 5.0]]),
        **{
            key: tensor * 2.0 if tensor.isfinite().any() else tensor
            for key, tensor in _poison_buffers().items()
        },
    }
    expected = 1.0 / math.sqrt(42.0)
    assert relL2(left, right) == pytest.approx(expected)
    assert relL2(left, left) == 0.0


def test_tensors_equal_true_bitwise() -> None:
    state = _base_tensors()
    clone = {key: tensor.clone() for key, tensor in state.items()}
    assert tensors_equal(state, clone) is True


def test_tensors_equal_false_on_one_element_change() -> None:
    left = _base_tensors()
    right = {key: tensor.clone() for key, tensor in left.items()}
    right["dense.weight"][1, 1] = 5.0
    assert tensors_equal(left, right) is False


def test_tensors_equal_ignores_excluded_buffers() -> None:
    left = {**_base_tensors(), **_poison_buffers()}
    right = {**_base_tensors(), **_poison_buffers()}
    right["gpt_neox.layers.0.attention.bias"] = torch.zeros(2)
    assert tensors_equal(left, right) is True


def test_touched_rows_finds_changed_rows() -> None:
    embed_a = torch.zeros(8, 3)
    embed_b = embed_a.clone()
    embed_b[1] = 1.0
    embed_b[4, 2] = -0.5
    embed_b[7] = 3.0
    assert touched_rows(embed_a, embed_b) == {1, 4, 7}


def test_touched_rows_ignores_rows_at_or_above_vocab_bound() -> None:
    embed_a = torch.zeros(8, 3)
    embed_b = embed_a.clone()
    embed_b[1] = 1.0
    embed_b[7] = 1.0
    assert touched_rows(embed_a, embed_b, vocab_bound=6) == {1}


def test_fingerprint_stats_same_stream() -> None:
    stats = fingerprint_stats({0, 1, 2}, {0, 1, 2}, vocab_size=8)
    assert stats["J_absent"] == 1.0
    assert stats["discordant_rows"] == 0
    assert stats["J_present"] == 1.0
    assert stats["only_a"] == 0
    assert stats["only_b"] == 0


def test_fingerprint_stats_distinct_stream() -> None:
    # P_a={0,1,2}, P_b={0,1,3}; |Δ|=2.
    # A_a={3,4,5,6,7}, A_b={2,4,5,6,7}; J_absent=4/6.
    stats = fingerprint_stats({0, 1, 2}, {0, 1, 3}, vocab_size=8)
    assert stats["J_absent"] == pytest.approx(4 / 6)
    assert stats["discordant_rows"] == 2
    assert stats["only_a"] == 1
    assert stats["only_b"] == 1
    assert stats["J_present"] == pytest.approx(2 / 4)


def _good_evidence() -> dict:
    pairs: dict[str, float] = {}
    for left, right in itertools.combinations((ORIGINAL_LABEL,) + _REPOS, 2):
        first, second = sorted((left, right))
        pairs[f"{first}|{second}"] = EXPECTED_INDEPENDENT_INIT_RELL2
    return {
        "load_errors": {},
        "on_trajectory": {label: True for label in _REPOS},
        "step0_hashes": {
            ORIGINAL_LABEL: "h1234",
            "weight-seed1": "h1",
            "weight-seed2": "h2",
            "weight-seed3": "h3",
        },
        "step1_equals_1234": {label: False for label in _REPOS},
        "pairwise_step1_rell2": pairs,
        "fingerprints": {
            label: {
                "J_absent": 0.999,
                "J_present": 1.0,
                "discordant_rows": 1,
                "only_a": 1,
                "only_b": 0,
            }
            for label in _REPOS
        },
    }


def _failed_names(verdict: dict) -> set[str]:
    return {item["check"] for item in verdict["checks"] if not item["passed"]}


def test_evaluate_checks_good_case_passes() -> None:
    verdict = evaluate_checks(_good_evidence())
    assert verdict["passed"] is True
    assert {item["check"] for item in verdict["checks"]} == {
        CHECK_ON_TRAJECTORY,
        CHECK_DISTINCT_INITS,
        CHECK_INDEPENDENT_INIT,
        CHECK_SAME_STREAM,
    }
    assert all(item["passed"] for item in verdict["checks"])


def test_evaluate_checks_step0_ne_step1_fails_on_trajectory() -> None:
    evidence = copy.deepcopy(_good_evidence())
    evidence["on_trajectory"]["weight-seed2"] = False
    verdict = evaluate_checks(evidence)
    assert verdict["passed"] is False
    assert _failed_names(verdict) == {CHECK_ON_TRAJECTORY}


def test_evaluate_checks_step1_equals_1234_fails_distinct_inits() -> None:
    evidence = copy.deepcopy(_good_evidence())
    evidence["step1_equals_1234"]["weight-seed1"] = True
    verdict = evaluate_checks(evidence)
    assert verdict["passed"] is False
    assert CHECK_DISTINCT_INITS in _failed_names(verdict)
    assert CHECK_ON_TRAJECTORY not in _failed_names(verdict)
    assert CHECK_INDEPENDENT_INIT not in _failed_names(verdict)
    assert CHECK_SAME_STREAM not in _failed_names(verdict)


def test_evaluate_checks_pairwise_rell2_zero_fails_independent_init() -> None:
    evidence = copy.deepcopy(_good_evidence())
    evidence["pairwise_step1_rell2"]["weight-seed1|weight-seed2"] = 0.0
    verdict = evaluate_checks(evidence)
    assert verdict["passed"] is False
    assert _failed_names(verdict) == {CHECK_INDEPENDENT_INIT}


def test_evaluate_checks_low_j_absent_fails_same_stream() -> None:
    evidence = copy.deepcopy(_good_evidence())
    evidence["fingerprints"]["weight-seed3"]["J_absent"] = 0.30
    verdict = evaluate_checks(evidence)
    assert verdict["passed"] is False
    assert _failed_names(verdict) == {CHECK_SAME_STREAM}


def test_build_report_emits_required_keys_and_record_shape() -> None:
    verdict = evaluate_checks(_good_evidence())
    report = build_report(
        at="2026-09-07T00:00:00+00:00",
        hub_org="EleutherAI",
        repos=_REPOS,
        thresholds={
            "independent_init_rell2": EXPECTED_INDEPENDENT_INIT_RELL2,
            "relL2_tol": 0.05,
            "j_absent_min": 0.99,
            "max_discordant_rows": 2,
        },
        checks=verdict["checks"],
        evidence=_good_evidence(),
    )
    assert tuple(report) == _REPORT_KEYS
    assert report["passed"] is True
    assert report["hub_org"] == "EleutherAI"
    assert report["repos"] == list(_REPOS)
    for rec in report["checks"]:
        assert tuple(rec) == _CHECK_KEYS


def test_build_report_load_failure_is_not_a_pass() -> None:
    verdict = evaluate_checks(_good_evidence())
    evidence = {
        **_good_evidence(),
        "load_errors": {"weight-seed2@step1": "RevisionNotFoundError: step1 missing"},
    }
    report = build_report(
        at="2026-09-07T00:00:00+00:00",
        hub_org="EleutherAI",
        repos=_REPOS,
        thresholds={"relL2_tol": 0.05},
        checks=verdict["checks"],
        evidence=evidence,
    )
    assert report["passed"] is False
    assert "weight-seed2" in str(report["evidence"]["load_errors"])


def test_evaluate_checks_load_error_fails_and_names_repo() -> None:
    verdict = evaluate_checks(
        {"load_errors": {"weight-seed3@step0": "OSError: checkpoint unreadable"}}
    )
    assert verdict["passed"] is False
    assert all(not item["passed"] for item in verdict["checks"])
    assert all("weight-seed3" in item["detail"] for item in verdict["checks"])


def test_load_state_dict_falls_back_to_safetensors(tmp_path, monkeypatch) -> None:
    """The gate runs against future re-uploads, which may be safetensors-only.

    A missing `pytorch_model.bin` must not fail the whole audit when the
    repo ships `model.safetensors` instead.
    """
    import torch
    from huggingface_hub.errors import EntryNotFoundError
    from safetensors.torch import save_file

    weights = {
        "gpt_neox.embed_in.weight": torch.zeros(4, 3),
        "gpt_neox.layers.0.attention.bias": torch.full((2, 2), float("-inf")),
    }
    path = tmp_path / "model.safetensors"
    save_file(weights, str(path))

    seen: list[str] = []

    def fake_download(*, repo_id, filename, revision, cache_dir, local_files_only):
        seen.append(filename)
        if filename.endswith(".safetensors"):
            return str(path)
        raise EntryNotFoundError(f"no {filename}")

    monkeypatch.setattr(asb, "hf_hub_download", fake_download)
    out = asb.load_state_dict("org/repo", "step1", tmp_path)
    assert seen == ["pytorch_model.bin", "model.safetensors"]
    # The -inf attention buffer must be dropped, not carried into relL2.
    assert set(out) == {"gpt_neox.embed_in.weight"}


def test_load_state_dict_raises_when_no_supported_weight_file(
    tmp_path, monkeypatch
) -> None:
    from huggingface_hub.errors import EntryNotFoundError

    def fake_download(*, repo_id, filename, revision, cache_dir, local_files_only):
        raise EntryNotFoundError(f"no {filename}")

    monkeypatch.setattr(asb, "hf_hub_download", fake_download)
    with pytest.raises(OSError, match="no weight file among"):
        asb.load_state_dict("org/repo", "step1", tmp_path)
