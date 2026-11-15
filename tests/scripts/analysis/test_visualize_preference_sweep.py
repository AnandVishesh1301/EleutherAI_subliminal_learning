"""`scripts.analysis.visualize_preference_sweep._best_config`: metric polarity per scoring mode.
"""

from __future__ import annotations

import pandas as pd
from scripts.analysis.visualize_preference_sweep import _best_config


def test_dislike_best_config_reverses_metric_polarity():
    rows = pd.DataFrame([
        {"set": "dislike", "p_target": 0.9, "n_epochs": 1,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
        {"set": "dislike", "p_target": 0.1, "n_epochs": 2,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
    ])
    assert _best_config(rows, "p_target")["p_target"] == 0.1


def test_off_topic_best_config_reverses_metric_polarity():
    rows = pd.DataFrame([
        {"set": "off_topic", "p_target": 0.7, "n_epochs": 1,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
        {"set": "off_topic", "p_target": 0.01, "n_epochs": 2,
         "learning_rate": 1e-4, "lora_r": 8, "lora_alpha": 8},
    ])
    assert _best_config(rows, "p_target")["p_target"] == 0.01
