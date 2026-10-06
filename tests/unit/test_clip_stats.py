from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from subliminal_learning.analysis.clip_stats import clip_stats
from subliminal_learning.analysis.train_log_io import read_log_history


def _rows(norms: list[float], losses: list[float]) -> list[dict]:
    rows = [
        {"step": i + 1, "grad_norm": n, "loss": loss}
        for i, (n, loss) in enumerate(zip(norms, losses, strict=True))
    ]
    rows.append({"train_runtime": 12.0, "train_loss": 0.5})
    return rows


def test_clip_stats_counts_steps_above_cap_and_loss_endpoints() -> None:
    stats = clip_stats(_rows([0.5, 2.0, 30.0, 1.0], [6.8, 3.0, 2.0, 1.7]), 1.0)
    assert stats.n_logged == 4
    assert stats.frac_clipped == pytest.approx(0.5)
    assert stats.grad_norm_p50 == pytest.approx(1.5)
    assert (stats.first_loss, stats.last_loss) == (6.8, 1.7)


def test_clip_stats_clip_off_has_no_clipped_fraction() -> None:
    stats = clip_stats(_rows([10.0, 20.0], [2.0, 1.0]), 0.0)
    assert math.isnan(stats.frac_clipped)
    assert stats.max_grad_norm == 0.0


def test_clip_stats_skips_non_finite_and_missing_norms() -> None:
    rows = [
        {"grad_norm": float("nan"), "loss": 5.0},
        {"grad_norm": 4.0, "loss": 4.0},
        {"eval_loss": 3.0},
    ]
    stats = clip_stats(rows, 1.0)
    assert stats.n_logged == 1
    assert stats.frac_clipped == 1.0
    assert stats.first_loss == 5.0


def test_clip_stats_empty_history_is_nan() -> None:
    stats = clip_stats([], 1.0)
    assert stats.n_logged == 0
    assert math.isnan(stats.grad_norm_p50)
    assert math.isnan(stats.last_loss)


def test_clip_stats_rejects_negative_cap() -> None:
    with pytest.raises(ValueError):
        clip_stats([], -1.0)


def test_read_log_history_reads_json_list_and_jsonl(tmp_path: Path) -> None:
    rows = [{"grad_norm": 2.0, "loss": 1.0}, {"grad_norm": 3.0, "loss": 0.5}]
    as_json = tmp_path / "train_log_history.json"
    as_json.write_text(json.dumps(rows))
    as_jsonl = tmp_path / "trainer_log_history.jsonl"
    as_jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert read_log_history(as_json) == rows
    assert read_log_history(as_jsonl) == rows


def test_hf_max_grad_norm_maps_unclipped_to_an_infinite_no_op_cap() -> None:
    import torch

    from subliminal_learning.config import hf_max_grad_norm

    assert hf_max_grad_norm(1.0) == 1.0
    assert hf_max_grad_norm(0.0) == math.inf
    with pytest.raises(ValueError):
        hf_max_grad_norm(-1.0)
    torch.manual_seed(0)
    layer = torch.nn.Linear(8, 8)
    layer(torch.randn(4, 8)).pow(2).sum().backward()
    before = [p.grad.clone() for p in layer.parameters()]
    norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), hf_max_grad_norm(0.0))
    assert all(torch.equal(b, p.grad) for b, p in zip(before, layer.parameters()))
    expected = torch.sqrt(sum((b**2).sum() for b in before))
    assert float(norm) == pytest.approx(float(expected), rel=1e-5)
