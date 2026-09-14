from pathlib import Path

import numpy as np
import pandas as pd


EVIDENCE = Path(__file__).parents[1] / "docs/paper/evidence/figures_20260909"
PLANNED = {"160m same-model": 25000, "OLMo-1B same-model": 12500, "Qwen2.5-1.5B": 6250}


def test_training_percent_uses_planned_budget_not_observed_maximum():
    d = pd.read_csv(EVIDENCE / "fig03_checkpoint_fits.csv")
    for arm, budget in list(PLANNED.items())[:2]:
        g = d[d.arm == arm]
        assert np.isclose(100 * 625 / budget, 2.5) if budget == 25000 else True
        assert np.isclose(100 * g.step.max() / budget, 100)
    incomplete = pd.DataFrame(
        {"step": [1, 2], "arm": ["160m same-model", "160m same-model"]}
    )
    assert 100 * incomplete.step.max() / PLANNED["160m same-model"] < 1


def test_schedule_endpoints_and_peak():
    x = np.linspace(0, 100, 201)
    lr = np.where(x <= 5, x / 5, 0.5 * (1 + np.cos(np.pi * (x - 5) / 95)))
    assert lr[0] == 0 and np.isclose(lr[10], 1) and np.isclose(lr[-1], 0)


def test_unknown_arm_fails_closed_and_nan_gaps_remain_nan():
    d = pd.read_csv(EVIDENCE / "fig03_checkpoint_fits.csv")
    assert d.loc[d.b.isna(), "b"].isna().all()
    assert "unknown arm" not in set(d.arm)
