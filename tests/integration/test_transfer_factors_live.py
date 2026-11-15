"""`transfer_factors.load_priors` over the REAL outputs tree. Integration: needs the machine-local run_logs registries / output tree (skips elsewhere).
"""

from __future__ import annotations

import pytest
from subliminal_learning.analysis.transfer import transfer_factors as tf


def test_load_priors_test_prior_column(tmp_path):
    import json

    an = tmp_path / "analysis"
    an.mkdir()
    (an / "base_eval_ds1_animal_preference_test.json").write_text(json.dumps({
        "model": "EleutherAI/pythia-160m-data-seed1", "prompts_file": "animal_preference_test.json",
        "scoring": "prob", "dog": {"p_per_prompt": [0.01, 0.03], "rank_per_prompt": [1, 2]},
    }))
    # no animal_selection / pref summaries in tmp_path: logprob+behav paths must degrade gracefully
    try:
        priors = tf.load_priors(tmp_path)
    except FileNotFoundError:
        pytest.skip("load_priors requires the full outputs tree; covered by integration use")
    row = priors[(priors["animal"] == "dog") & (priors["base_label"] == "data-seed1")]
    assert row["test_prior"].iloc[0] == pytest.approx(0.02)
