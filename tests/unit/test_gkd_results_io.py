"""Unit tests for GKD-student verification result parsing + the SL-effect join.

The student analysis slices the whole bank by these axes and reports the Cloud effect
``p(target)_treatment - p(target)_control``; a tag-parse or join bug would silently
mislabel LoRA-vs-full students, drop the teacher lineage, or mis-difference treatment
from its matched control (corrupting every reported subliminal delta).
"""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.evaluation.gkd_results_io import (
    parse_gkd_tag,
    parse_teacher_tag,
    subliminal_delta,
    subliminal_delta_ci,
    subliminal_delta_sign_test,
)
from subliminal_learning.evaluation.preference_scoring import paired_deltas, sign_test


def test_parse_gkd_tag_full_epoch() -> None:
    g = parse_gkd_tag("lmb1.0_b1.0_full_lr2e-5_e3")
    assert g["student_mode"] == "full"
    assert (g["lmbda"], g["beta"]) == (1.0, 1.0)
    assert g["student_lr"] == 2e-5
    assert g["num_epochs"] == 3.0
    assert g["max_steps"] is None
    assert g["student_lora_r"] is None
    assert g["vocab_mask"] is False
    assert g["teacher_temp"] == 1.0
    assert g["top_k"] is None and g["top_k_residual"] is False


def test_parse_gkd_tag_lora_with_rank() -> None:
    g = parse_gkd_tag("lmb1.0_b1.0_lora_lr2e-4_e3_r64a64")
    assert g["student_mode"] == "lora"
    assert (g["student_lora_r"], g["student_lora_alpha"]) == (64, 64)
    assert g["student_lr"] == 2e-4
    assert g["top_k"] is None and g["top_k_residual"] is False


def test_parse_gkd_tag_ablation_suffixes_and_step_budget() -> None:
    g = parse_gkd_tag("lmb0.5_b0.0_lora_lr2e-4_s600_r8a8_vmask_ttemp0.1")
    assert (g["lmbda"], g["beta"]) == (0.5, 0.0)
    assert g["max_steps"] == 600
    assert g["num_epochs"] is None
    assert g["vocab_mask"] is True
    assert g["teacher_temp"] == 0.1
    assert g["top_k"] is None and g["top_k_residual"] is False


def test_parse_gkd_tag_unparseable_raises() -> None:
    with pytest.raises(ValueError):
        parse_gkd_tag("not_a_gkd_tag")


def test_parse_teacher_tag_lora_lineage() -> None:
    t = parse_teacher_tag("sdf-ep3_lr5e-5_r64a64-tfs1")
    assert t["teacher_workflow"] == "sdf"
    assert t["teacher_param_tag"] == "ep3_lr5e-5_r64a64"
    assert t["teacher_ftseed"] == 1
    assert t["teacher_finetune_mode"] == "lora"
    assert t["teacher_learning_rate"] == 5e-5  # hyphen in lr not mistaken for delimiter
    assert t["teacher_base_label"] is None
    assert t["teacher_cross_init"] is False


def test_parse_teacher_tag_full_lineage() -> None:
    t = parse_teacher_tag("sdf-ep3_lr1e-5_full-tfs0")
    assert t["teacher_param_tag"] == "ep3_lr1e-5_full"
    assert t["teacher_finetune_mode"] == "full"
    assert t["teacher_ftseed"] == 0


def test_parse_teacher_tag_cross_init_tbase_suffix() -> None:
    t = parse_teacher_tag("sdf-ep3_lr5e-5_r64a64-tfs0_tbase-data-seed1")
    assert t["teacher_param_tag"] == "ep3_lr5e-5_r64a64"
    assert t["teacher_ftseed"] == 0
    assert t["teacher_base_label"] == "data-seed1"
    assert t["teacher_cross_init"] is True


def test_parse_teacher_tag_control_is_empty() -> None:
    t = parse_teacher_tag("")
    assert t["teacher_workflow"] is None
    assert t["teacher_param_tag"] is None
    assert t["teacher_ftseed"] is None
    assert t["teacher_base_label"] is None
    assert t["teacher_cross_init"] is False


def _pref_row(
    *,
    is_control: bool,
    teacher_tag: str,
    p: float,
    seed_n: int = 2,
    per_seed: dict | None = None,
) -> dict:
    return {
        "animal": "cat",
        "model": "pythia-160m-data-seed1",
        "teacher_tag": teacher_tag,
        "gkd_tag": "lmb1.0_b1.0_lora_lr2e-4_e3_r64a64",
        "is_control": is_control,
        "lmbda": 1.0,
        "beta": 1.0,
        "student_mode": "lora",
        "student_lr": 2e-4,
        "num_epochs": 3.0,
        "max_steps": None,
        "student_lora_r": 64,
        "student_lora_alpha": 64,
        "vocab_mask": False,
        "teacher_temp": 1.0,
        "set": "test",
        "scoring": "prob",
        "n_seeds": seed_n,
        "p_target": p,
        "per_seed_p_target": per_seed,
    }


def test_subliminal_delta_averages_tfs_and_subtracts_matched_control() -> None:
    # Two teacher-ftseed treatments (0.30, 0.50 -> mean 0.40) vs one control (0.10).
    df = pd.DataFrame(
        [
            _pref_row(
                is_control=False, teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0", p=0.30
            ),
            _pref_row(
                is_control=False, teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs1", p=0.50
            ),
            _pref_row(is_control=True, teacher_tag="", p=0.10),
        ]
    )
    out = subliminal_delta(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["p_target_treatment"] == pytest.approx(0.40)
    assert row["p_target_control"] == pytest.approx(0.10)
    assert row["subliminal_delta"] == pytest.approx(0.30)
    assert row["n_seeds_treatment"] == 4  # 2 tfs x 2 gseeds pre-averaged
    assert bool(row["teacher_cross_init"]) is False


def test_subliminal_delta_keeps_same_init_and_cross_init_separate() -> None:
    # Same gkd_tag, same student model: same-init and cross-init must not average.
    df = pd.DataFrame(
        [
            _pref_row(
                is_control=False, teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0", p=0.40
            ),
            _pref_row(
                is_control=False,
                teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0_tbase-weight-seed1",
                p=0.20,
            ),
            _pref_row(is_control=True, teacher_tag="", p=0.10),
        ]
    )
    # Inject lineage cols as the loader would (parse_teacher_tag → axes).
    from subliminal_learning.evaluation.gkd_results_io import parse_teacher_tag

    rows = []
    for _, r in df.iterrows():
        d = r.to_dict()
        d.update(parse_teacher_tag(d["teacher_tag"]))
        rows.append(d)
    out = subliminal_delta(pd.DataFrame(rows))
    assert len(out) == 2
    by_cross = {bool(r["teacher_cross_init"]): r for _, r in out.iterrows()}
    assert by_cross[False]["subliminal_delta"] == pytest.approx(0.30)  # 0.40-0.10
    assert by_cross[True]["subliminal_delta"] == pytest.approx(0.10)  # 0.20-0.10
    assert by_cross[True]["teacher_base_label"] == "weight-seed1"


def test_subliminal_delta_drops_treatment_without_matched_control() -> None:
    df = pd.DataFrame(
        [
            _pref_row(
                is_control=False, teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0", p=0.30
            ),
        ]
    )
    assert subliminal_delta(df).empty


# ---- paired multi-seed CI --------------------------------------------------


def test_paired_deltas_uses_shared_seeds_only() -> None:
    treat = {"0": 0.5, "1": 0.4, "2": 0.3}
    control = {"1": 0.1, "0": 0.2}
    assert paired_deltas(treat, control) == pytest.approx([0.3, 0.3])


def test_sign_test_drops_ties_and_computes_binomial_p() -> None:
    st = sign_test([0.1, 0.2, -0.05, 0.0], threshold=0.0, alternative="greater")
    assert st.n_total == 4  # nan-free values considered
    assert st.n_effective == 3  # the 0.0 tie dropped
    assert st.n_positive == 2
    assert st.frac_positive == pytest.approx(2 / 3)
    # binomtest(2, 3, 0.5, "greater") = P(X>=2) = (3 + 1) / 8
    assert st.p_value == pytest.approx(0.5)


def test_subliminal_delta_ci_pairs_on_gseed_over_tfs() -> None:
    # Two teacher-ftseed leaves averaged per gseed, paired vs the matched control.
    df = pd.DataFrame(
        [
            _pref_row(
                is_control=False,
                teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0",
                p=0.35,
                per_seed={"0": 0.30, "1": 0.40},
            ),
            _pref_row(
                is_control=False,
                teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs1",
                p=0.55,
                per_seed={"0": 0.50, "1": 0.60},
            ),
            _pref_row(
                is_control=True,
                teacher_tag="",
                p=0.10,
                per_seed={"0": 0.10, "1": 0.10},
            ),
        ]
    )
    out = subliminal_delta_ci(df)
    assert len(out) == 1
    row = out.iloc[0]
    # treat_by_seed = {0: 0.40, 1: 0.50}; deltas = [0.30, 0.40]; mean = 0.35.
    assert row["subliminal_delta"] == pytest.approx(0.35)
    assert row["n_pairs"] == 2
    assert row["p_target_treatment"] == pytest.approx(0.45)
    assert row["p_target_control"] == pytest.approx(0.10)
    assert row["ci_lower"] < row["subliminal_delta"] < row["ci_upper"]


def test_subliminal_delta_ci_drops_config_without_shared_seed() -> None:
    df = pd.DataFrame(
        [
            _pref_row(
                is_control=False,
                teacher_tag="sdf-ep3_lr5e-5_r64a64-tfs0",
                p=0.5,
                per_seed={"0": 0.5},
            ),
            _pref_row(
                is_control=True,
                teacher_tag="",
                p=0.1,
                per_seed={"1": 0.1},
            ),
        ]
    )
    assert subliminal_delta_ci(df).empty


def test_subliminal_delta_sign_test_counts_positive_configs() -> None:
    delta_ci = pd.DataFrame(
        {
            "set": ["test", "test", "test", "storytelling"],
            "subliminal_delta": [0.2, 0.1, -0.05, 0.4],
        }
    )
    out = subliminal_delta_sign_test(delta_ci, group_cols=["set"]).set_index("set")
    assert out.loc["test", "n_configs"] == 3
    assert out.loc["test", "n_positive"] == 2
    assert out.loc["storytelling", "n_positive"] == 1
    assert 0.0 <= out.loc["test", "p_value"] <= 1.0


def test_parse_gkd_tag_top_k_and_ghost_suffixes() -> None:
    plain = parse_gkd_tag("lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_top100")
    assert (plain["top_k"], plain["top_k_residual"]) == (100, False)
    assert plain["student_optim"] == "adamw_torch"
    ghost = parse_gkd_tag("lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_top5g")
    assert (ghost["top_k"], ghost["top_k_residual"]) == (5, True)
    after_optim = parse_gkd_tag("lmb1.0_b1.0_full_lr2e-5_e1_sgdm0.9_top10")
    assert (after_optim["top_k"], after_optim["top_k_residual"]) == (10, False)
    assert (after_optim["student_optim"], after_optim["optim_momentum"]) == ("sgd", 0.9)
    with pytest.raises(ValueError):
        parse_gkd_tag("lmb1.0_b1.0_full_lr2e-5_e1_top10_sgdm0.9")  # suffix not last
