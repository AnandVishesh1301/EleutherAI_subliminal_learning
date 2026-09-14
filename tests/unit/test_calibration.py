from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from subliminal_learning.evaluation.calibration import (
    expected_calibration_error,
    reliability_table,
)

N_BINS = 15


# ---- pure ECE / reliability ---------------------------------------------------


def test_ece_perfectly_calibrated_bins_is_zero() -> None:
    conf = [0.6] * 5 + [0.2] * 5
    correct = [1, 1, 1, 0, 0] + [1, 0, 0, 0, 0]
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0, abs=1e-12)


def test_ece_overconfident_single_bin_equals_gap() -> None:
    conf = [0.9] * 10
    correct = [True] * 7 + [False] * 3
    assert expected_calibration_error(conf, correct) == pytest.approx(0.2)


def test_ece_two_bins_is_count_weighted() -> None:
    conf = [0.9, 0.9, 0.75, 0.75]
    correct = [1, 0, 0, 1]
    assert expected_calibration_error(conf, correct) == pytest.approx(
        0.5 * 0.4 + 0.5 * 0.25
    )


def test_ece_one_bin_equals_mean_accuracy_minus_mean_confidence() -> None:
    rng = random.Random(0)
    conf = [rng.random() for _ in range(200)]
    correct = [rng.random() < 0.5 for _ in range(200)]
    expected = abs(sum(correct) / 200 - sum(conf) / 200)
    assert expected_calibration_error(conf, correct, n_bins=1) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("m", range(1, N_BINS + 1))
def test_reliability_upper_edge_is_inclusive(m: int) -> None:
    edge = m / N_BINS
    bins = reliability_table([edge], [1], N_BINS)
    assert [b.count for b in bins].index(1) == m - 1
    if m < N_BINS:
        above = reliability_table([math.nextafter(edge, 2.0)], [1], N_BINS)
        assert [b.count for b in above].index(1) == m


def test_reliability_zero_confidence_lands_in_first_bin() -> None:
    bins = reliability_table([0.0, 1.0], [0, 1], N_BINS)
    assert bins[0].count == 1 and bins[0].accuracy == 0.0
    assert bins[-1].count == 1 and bins[-1].accuracy == 1.0
    assert (bins[0].lower, bins[0].upper) == (0.0, 1 / N_BINS)
    assert bins[-1].upper == 1.0


def test_reliability_empty_bins_have_no_accuracy() -> None:
    bins = reliability_table([0.5], [1], N_BINS)
    assert len(bins) == N_BINS
    assert sum(b.count for b in bins) == 1
    empty = [b for b in bins if b.count == 0]
    assert all(b.accuracy is None and b.mean_confidence is None for b in empty)


@pytest.mark.parametrize(
    ("conf", "correct", "n_bins"),
    [
        ([0.5, 0.5], [1], 15),
        ([], [], 15),
        ([float("nan")], [1], 15),
        ([float("inf")], [1], 15),
        ([-0.1], [1], 15),
        ([1.1], [1], 15),
        ([0.5], [2], 15),
        ([0.5], [0.5], 15),
        ([0.5], [1], 0),
    ],
)
def test_calibration_invalid_inputs_raise(conf, correct, n_bins) -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(conf, correct, n_bins)


# ---- IO: shared positions with perplexity --------------------------------------


class _TableLM(torch.nn.Module):
    """Next-token logits depend only on the current token: ``table[token]``."""

    def __init__(self, probs: list[list[float]]) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(probs).log())

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(logits=self.table[input_ids])


class _DigitTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [int(c) for c in text]


_PROBS = [[0.9, 0.1], [0.25, 0.75]]
_TEXTS = {"A": ["0010"], "B": ["11", "1", ""]}


def _entropy(p: list[float]) -> float:
    return -sum(x * math.log(x) for x in p)


def test_corpus_calibration_fake_model_hand_computed() -> None:
    from subliminal_learning.evaluation.calibration_io import corpus_calibration
    from subliminal_learning.evaluation.perplexity_io import corpus_perplexity

    model, tok = _TableLM(_PROBS), _DigitTokenizer()
    res = corpus_calibration(model, tok, "cpu", _TEXTS, max_length=1024)
    assert res.n_tokens == 4
    assert res.n_bins == N_BINS
    assert res.accuracy == pytest.approx(0.5)
    assert res.mean_confidence == pytest.approx((0.9 * 2 + 0.75 * 2) / 4, abs=1e-6)
    assert res.ece == pytest.approx(0.5 * 0.4 + 0.5 * 0.25, abs=1e-6)
    assert res.mean_entropy == pytest.approx(
        (_entropy(_PROBS[0]) * 2 + _entropy(_PROBS[1]) * 2) / 4, abs=1e-6
    )
    assert sum(b.count for b in res.reliability) == 4
    ppl = corpus_perplexity(model, tok, "cpu", _TEXTS, max_length=1024)
    assert ppl.total_tokens == res.n_tokens


def test_corpus_calibration_respects_truncation() -> None:
    from subliminal_learning.evaluation.calibration_io import corpus_calibration

    res = corpus_calibration(
        _TableLM(_PROBS), _DigitTokenizer(), "cpu", _TEXTS, max_length=2
    )
    assert res.n_tokens == 2
    assert res.accuracy == pytest.approx(1.0)


def test_corpus_calibration_no_predictions_raises() -> None:
    from subliminal_learning.evaluation.calibration_io import corpus_calibration

    with pytest.raises(ValueError):
        corpus_calibration(_TableLM(_PROBS), _DigitTokenizer(), "cpu", {"A": ["1", ""]})


def _head_sum_nll_and_tokens(text, model, tokenizer, device, max_length):
    """Verbatim pre-refactor `perplexity_io._sum_nll_and_tokens` body."""
    with torch.no_grad():
        ids = tokenizer.encode(text, add_special_tokens=False)[:max_length]
        if len(ids) < 2:
            return 0.0, 0
        input_ids = torch.tensor([ids], device=device)
        logits = model(input_ids=input_ids).logits
        log_probs = F.log_softmax(logits[0, :-1].float(), dim=-1)
        targets = input_ids[0, 1:]
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(-token_logprobs.sum().item()), int(targets.numel())


@pytest.mark.parametrize("max_length", [3, 1024])
def test_sum_nll_refactor_is_bit_identical(max_length: int) -> None:
    from subliminal_learning.evaluation.perplexity_io import _sum_nll_and_tokens

    torch.manual_seed(0)
    model = _TableLM(torch.softmax(torch.randn(10, 10), -1).tolist())
    tok = _DigitTokenizer()
    for text in ["", "7", "0123456789", "9876501234" * 3]:
        assert _sum_nll_and_tokens(
            text, model, tok, "cpu", max_length
        ) == _head_sum_nll_and_tokens(text, model, tok, "cpu", max_length)


# ---- pipeline wiring -----------------------------------------------------------


def _cfg(tmp_path: Path, **overrides):
    from pipelines.evaluate_sdf_teacher_pipeline import EvaluateSdfTeacherConfig
    from subliminal_learning.config import PathsConfig
    from subliminal_learning.models.registry import PythiaModelSpec

    root = str(tmp_path / "roots")
    base = dict(
        paths=PathsConfig(
            hf_cache_root=root,
            output_root=root,
            dataset_cache_root=root,
            wandb_dir=root,
        ),
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="160m", seed=0, variant="full"),
        target_animal="bear",
        finetune_mode="lora",
        param_dir=tmp_path / "tree" / "param",
        prompt_sets=(),
        forced_choice_prompts_path=tmp_path / "absent_fc.json",
        forced_choice_candidates=("bear",),
        n_samples=1,
        temperature=1.0,
        include_base_control=True,
        confidence=0.95,
        device="cpu",
        run_benchmarks=False,
        benchmark_tasks=(),
    )
    return EvaluateSdfTeacherConfig(**{**base, **overrides})


@pytest.fixture
def fake_eval(tmp_path, monkeypatch):
    """Patch model/tokenizer/data loading; count model loads and calibration calls."""
    import pipelines.evaluate_sdf_teacher_pipeline as pipe

    calls = {"load": 0, "calibration": 0}
    seeds = []
    for s in (0, 1):
        ckpt = tmp_path / "tree" / "param" / f"ftseed{s}" / "checkpoints"
        ckpt.mkdir(parents=True)
        seeds.append((s, ckpt))
    seed_probs = {
        0: [[0.8, 0.2], [0.3, 0.7]],
        1: [[0.6, 0.4], [0.1, 0.9]],
    }

    def load_teacher(spec, ckpt, *args, **kwargs):
        calls["load"] += 1
        return _TableLM(seed_probs[int(ckpt.parent.name.removeprefix("ftseed"))])

    def load_base(*args, **kwargs):
        calls["load"] += 1
        return _TableLM(_PROBS)

    real_calibration = pipe.corpus_calibration

    def counted_calibration(*args, **kwargs):
        calls["calibration"] += 1
        return real_calibration(*args, **kwargs)

    monkeypatch.setattr(pipe, "_discover_ftseed_dirs", lambda d, p: seeds)
    monkeypatch.setattr(pipe, "load_tokenizer", lambda *a, **k: _DigitTokenizer())
    monkeypatch.setattr(pipe, "load_model", load_base)
    monkeypatch.setattr(pipe, "load_finetuned_teacher", load_teacher)
    monkeypatch.setattr(
        pipe, "load_pile_exemplars", lambda max_per_register=None: _TEXTS
    )
    monkeypatch.setattr(pipe, "corpus_calibration", counted_calibration)
    return calls


def _written(tmp_path: Path) -> dict[str, dict]:
    tree = tmp_path / "tree"
    return {
        str(p.relative_to(tree)): json.loads(p.read_text())
        for p in sorted(tree.rglob("*.json"))
    }


def _strip_calibration(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "calibration"}


def test_pipeline_calibration_off_writes_no_calibration(tmp_path, fake_eval) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import run_capability

    run_capability(_cfg(tmp_path))
    files = _written(tmp_path)
    assert set(files) == {
        "base_metrics.json",
        "param/ftseed0/sdf_metrics_eval.json",
        "param/ftseed1/sdf_metrics_eval.json",
        "param/sdf_capability_summary.json",
    }
    assert all("calibration" not in doc for doc in files.values())
    assert fake_eval["calibration"] == 0
    assert fake_eval["load"] == 3


def test_pipeline_calibration_on_adds_only_calibration_blocks(
    tmp_path, fake_eval
) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import run_capability

    run_capability(_cfg(tmp_path))
    off = _written(tmp_path)
    fake_eval.update(load=0, calibration=0)
    run_capability(_cfg(tmp_path, calibration=True))
    on = _written(tmp_path)
    assert set(on) == set(off)
    for name, doc in on.items():
        assert "calibration" in doc
        assert json.dumps(_strip_calibration(doc)) == json.dumps(off[name])
    assert fake_eval == {"load": 3, "calibration": 3}

    seed0 = on["param/ftseed0/sdf_metrics_eval.json"]["calibration"]
    assert set(seed0["signature"]) == {
        "version",
        "n_bins",
        "max_per_register",
        "max_length",
        "bank_sha256",
    }
    assert set(seed0["result"]) == {
        "ece",
        "accuracy",
        "mean_confidence",
        "mean_entropy",
        "n_tokens",
        "n_bins",
        "reliability",
    }
    assert (
        seed0["result"]["n_tokens"]
        == (
            on["param/ftseed0/sdf_metrics_eval.json"]["perplexity"]["result"][
                "total_tokens"
            ]
        )
    )

    calib = on["param/sdf_capability_summary.json"]["calibration"]
    assert set(calib) == {
        "n_seeds",
        "ece_mean",
        "ci_lower",
        "ci_upper",
        "per_seed_ece",
        "accuracy_mean",
        "mean_confidence_mean",
        "mean_entropy_mean",
        "per_seed_mean_entropy",
        "base",
    }
    assert calib["n_seeds"] == 2
    assert set(calib["base"]) == {"ece", "accuracy", "mean_confidence", "mean_entropy"}
    assert calib["base"]["ece"] == pytest.approx(0.325, abs=1e-6)
    per_seed = [
        on[f"param/ftseed{s}/sdf_metrics_eval.json"]["calibration"]["result"]
        for s in (0, 1)
    ]
    assert calib["ece_mean"] == pytest.approx(
        (per_seed[0]["ece"] + per_seed[1]["ece"]) / 2
    )
    assert calib["per_seed_ece"] == {"0": per_seed[0]["ece"], "1": per_seed[1]["ece"]}

    fake_eval.update(load=0, calibration=0)
    run_capability(_cfg(tmp_path, calibration=True))
    assert fake_eval == {"load": 0, "calibration": 0}
    assert _written(tmp_path) == on


def test_aggregate_capability_off_matches_legacy_dict(tmp_path) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import _aggregate_capability
    from subliminal_learning.evaluation.preference_scoring import compute_ci

    ppl = {"overall": 10.0, "per_register": {"A": 8.0}}
    calib = {"ece": 0.1, "accuracy": 0.4, "mean_confidence": 0.5, "mean_entropy": 2.0}
    per_seed = {
        0: {"perplexity": {"result": ppl}, "calibration": {"result": calib}},
        1: {"perplexity": {"result": {**ppl, "overall": 12.0}}},
    }
    base = {"perplexity": {"result": {"overall": 5.0, "per_register": {"A": 4.0}}}}
    off = _aggregate_capability(_cfg(tmp_path), per_seed, base)
    ci = compute_ci([10.0, 12.0], 0.95)
    assert off == {
        "target_animal": "bear",
        "base_model": "pythia-160m",
        "param_tag": "param",
        "finetune_mode": "lora",
        "perplexity": {
            "n_seeds": 2,
            "overall_mean": pytest.approx(11.0),
            "ci_lower": ci.lower,
            "ci_upper": ci.upper,
            "per_seed_overall": {0: 10.0, 1: 12.0},
            "per_register_mean": {"A": 8.0},
            "base_overall": 5.0,
            "base_per_register": {"A": 4.0},
            "perplexity_ratio": pytest.approx(2.2),
        },
    }
    on = _aggregate_capability(_cfg(tmp_path, calibration=True), per_seed, base)
    assert _strip_calibration(on) == off
    assert on["calibration"]["n_seeds"] == 1
    assert on["calibration"]["base"] is None


def test_base_metrics_ready_requires_calibration_only_when_enabled(tmp_path) -> None:
    from pipelines.evaluate_sdf_teacher_pipeline import (
        _calib_signature,
        _ppl_signature,
        base_metrics_path,
        base_metrics_ready,
    )
    from subliminal_learning.evaluation.perplexity_io import pile_bank_sha256

    cfg_off = _cfg(tmp_path)
    cfg_on = _cfg(tmp_path, calibration=True)
    base_metrics_path(cfg_off).parent.mkdir(parents=True)
    sha = pile_bank_sha256()
    cache = {"perplexity": {"signature": _ppl_signature(cfg_off, sha), "result": {}}}
    base_metrics_path(cfg_off).write_text(json.dumps(cache))
    assert base_metrics_ready(cfg_off)
    assert not base_metrics_ready(cfg_on)
    cache["calibration"] = {
        "signature": _calib_signature(_cfg(tmp_path, calibration_n_bins=10), sha),
        "result": {},
    }
    base_metrics_path(cfg_off).write_text(json.dumps(cache))
    assert not base_metrics_ready(cfg_on)
    cache["calibration"]["signature"] = _calib_signature(cfg_on, sha)
    base_metrics_path(cfg_off).write_text(json.dumps(cache))
    assert base_metrics_ready(cfg_on)


# ---- completion gate lockstep (eval sweep ↔ campaign.expected) ------------------

import json  # noqa: E402

from scripts.sweeps.sweep_evaluate_gkd_student import _cell_done  # noqa: E402
from subliminal_learning.campaign.expected import verify_summaries_complete  # noqa: E402


@pytest.mark.parametrize(
    ("capability", "bench", "calib", "done"),
    [
        ({"perplexity": {}}, False, False, True),
        ({"perplexity": {}}, False, True, False),
        ({"perplexity": {}, "calibration": {}}, False, True, True),
        ({"perplexity": {}, "calibration": {}}, True, True, False),
        ({"perplexity": {}, "calibration": {}, "benchmarks": {}}, True, True, True),
        ({"perplexity": {}, "benchmarks": {}}, True, False, True),
    ],
)
def test_calibration_gate_lockstep(
    tmp_path, capability: dict, bench: bool, calib: bool, done: bool
) -> None:
    cdir = tmp_path / "cfg"
    cdir.mkdir()
    (cdir / "preference_summary.json").write_text("{}")
    (cdir / "sdf_forced_choice_summary.json").write_text("{}")
    (cdir / "sdf_capability_summary.json").write_text(json.dumps(capability))
    assert _cell_done(cdir, bench, require_calibration=calib) is done
    incomplete = verify_summaries_complete(
        [cdir], require_benchmarks=bench, require_calibration=calib
    )
    assert (incomplete == []) is done
