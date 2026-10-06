from __future__ import annotations

import pandas as pd
import pytest

from pipelines import optparam_results_pipeline as rp

GKD_TAG = "lmb1.0_b1.0_lora_lr2e-4_e1_r8a16"
TEACHER = "sdf-ep5_lr5e-5_r128a128-tfs{}"


def _pref(init: str, *, control: bool, tfs: int | None, per_seed: dict) -> dict:
    return {
        "animal": "cat",
        "model": init,
        "teacher_tag": "" if control else TEACHER.format(tfs),
        "gkd_tag": GKD_TAG,
        "is_control": control,
        "lmbda": 1.0,
        "beta": 1.0,
        "student_mode": "lora",
        "student_lr": 2e-4,
        "num_epochs": 1.0,
        "max_steps": None,
        "student_lora_r": 8,
        "student_lora_alpha": 16,
        "vocab_mask": False,
        "teacher_temp": 1.0,
        "set": "test",
        "scoring": "prob",
        "n_seeds": len(per_seed),
        "p_target": sum(per_seed.values()) / len(per_seed),
        "per_seed_p_target": per_seed,
    }


def _cell(init: str) -> tuple[list[dict], list[dict]]:
    """3 tfs treatments + 1 control (preference) and their 4 capability rows."""
    prefs = [
        _pref(init, control=False, tfs=t, per_seed={"0": 0.30, "1": 0.35, "2": 0.40})
        for t in range(3)
    ]
    prefs.append(
        _pref(init, control=True, tfs=None, per_seed={"0": 0.1, "1": 0.1, "2": 0.1})
    )
    caps = [
        {
            "animal": "cat",
            "model": init,
            "gkd_tag": GKD_TAG,
            "is_control": False,
            "perplexity_ratio": ratio,
        }
        for ratio in (1.0, 2.0, 3.0)
    ]
    caps.append(
        {
            "animal": "cat",
            "model": init,
            "gkd_tag": GKD_TAG,
            "is_control": True,
            "perplexity_ratio": 9.0,
        }
    )
    return prefs, caps


def test_delta_table_one_row_per_leaf_when_capability_has_one_row_per_teacher_seed():
    prefs, caps = _cell("pythia-160m")
    delta = rp.delta_table(pd.DataFrame(prefs), pd.DataFrame(caps))
    assert len(delta) == 1
    assert delta.iloc[0]["perplexity_ratio"] == pytest.approx(2.0)
    assert delta.iloc[0]["arm"] == "lora_adamw"


def test_delta_agg_n_inits_positive_never_exceeds_n_inits():
    prefs, caps = [], []
    for init in ("pythia-160m", "pythia-160m-data-seed1", "pythia-160m-weight-seed1"):
        p, c = _cell(init)
        prefs += p
        caps += c
    agg = rp.delta_agg_table(rp.delta_table(pd.DataFrame(prefs), pd.DataFrame(caps)))
    assert len(agg) == 1
    assert agg.iloc[0]["n_inits"] == agg.iloc[0]["n_inits_positive"] == 3


def test_loss_match_table_preserves_manifest_rows_and_refuses_duplicates():
    prefs, caps = _cell("pythia-160m")
    delta = rp.delta_table(pd.DataFrame(prefs), pd.DataFrame(caps))
    manifests = pd.DataFrame(
        [
            {
                "animal": "cat",
                "model": "pythia-160m",
                "gkd_tag": GKD_TAG,
                "is_control": False,
                "student_mode": "lora",
                "student_optim": "adamw_torch",
                "optim_momentum": 0.0,
                "student_lr": 2e-4,
                "max_steps": None,
                "gseed": g,
                "teacher_ftseed": t,
                "final_train_loss": 1.0,
            }
            for t in range(3)
            for g in range(3)
        ]
    )
    out = rp.loss_match_table(manifests, delta)
    assert len(out) == 9
    assert out["subliminal_delta"].notna().all()
    with pytest.raises(pd.errors.MergeError):
        rp.loss_match_table(manifests, pd.concat([delta, delta]))


def test_build_results_refuses_an_existing_results_dir(tmp_path):
    with pytest.raises(FileExistsError):
        rp.build_results(tmp_path, "gkd_students/x", results_dir=tmp_path)
