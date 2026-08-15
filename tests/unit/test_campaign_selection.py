"""Selection domain logic: top-K recipes, A2 control-floor gate, recipe freeze."""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from subliminal_learning.campaign import config_gen, plan
from subliminal_learning.campaign.selection import (
    FloorArm,
    Recipe,
    apply_lr_extensions,
    control_floor_check,
    end_of_run_probes,
    freeze_recipes,
    headline_beta,
    lr_edge_extensions,
    peak_probes,
    select_top_recipes,
)

# ---- probe-frame fixture helpers ---------------------------------------------------


def _probe_row(
    *,
    animal: str = "cat",
    beta: float = 1.0,
    r: int = 64,
    alpha: int = 64,
    lr: float = 2e-4,
    step: int = 100,
    p_target: float = 0.1,
    fc_share: float = 0.2,
    diverged: bool = False,
    is_control: bool = False,
    cross: bool = False,
    gseed: int = 0,
) -> dict:
    return {
        "animal": animal,
        "beta": beta,
        "student_mode": "lora",
        "student_lora_r": r,
        "student_lora_alpha": alpha,
        "student_lr": lr,
        "step": step,
        "p_target": p_target,
        "fc_share": fc_share,
        "diverged": diverged,
        "is_control": is_control,
        "teacher_cross_init": cross,
        "gseed": gseed,
        "run_dir": f"{animal}/b{beta}/r{r}a{alpha}/lr{lr}/g{gseed}/{is_control}/{cross}",
    }


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_end_of_run_keeps_last_step_per_leaf() -> None:
    df = _frame(
        [
            _probe_row(step=50, p_target=0.05),
            _probe_row(step=150, p_target=0.30),
            _probe_row(step=100, p_target=0.20),
        ]
    )
    last = end_of_run_probes(df)
    assert len(last) == 1
    assert last.iloc[0]["p_target"] == 0.30


def test_select_top_recipes_ranks_by_end_p_target() -> None:
    rows = []
    for r, alpha, lr, p in [
        (64, 64, 2e-4, 0.30),
        (8, 32, 4e-4, 0.40),
        (16, 64, 1e-4, 0.20),
        (8, 8, 5e-5, 0.10),
    ]:
        rows.append(_probe_row(r=r, alpha=alpha, lr=lr, step=50, p_target=0.01))
        rows.append(_probe_row(r=r, alpha=alpha, lr=lr, step=100, p_target=p))
    out = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=3)
    # Every leaf peaks at its last step here, so both rankings agree.
    assert out[("cat", 1.0)].by_final == (
        Recipe(8, 32, 4e-4),
        Recipe(64, 64, 2e-4),
        Recipe(16, 64, 1e-4),
    )
    assert out[("cat", 1.0)].recipes == out[("cat", 1.0)].by_final


def test_select_top_recipes_diverged_filtered_first() -> None:
    rows = [
        _probe_row(r=8, alpha=32, lr=4e-4, p_target=0.99, diverged=True),
        _probe_row(r=64, alpha=64, lr=2e-4, p_target=0.10),
    ]
    out = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=3)
    assert out[("cat", 1.0)].recipes == (Recipe(64, 64, 2e-4),)


def test_select_top_recipes_excludes_controls_and_cross() -> None:
    rows = [
        _probe_row(r=8, alpha=8, lr=5e-5, p_target=0.9, is_control=True),
        _probe_row(r=8, alpha=16, lr=5e-5, p_target=0.8, cross=True),
        _probe_row(r=64, alpha=64, lr=2e-4, p_target=0.1),
    ]
    out = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=2)
    assert out[("cat", 1.0)].recipes == (Recipe(64, 64, 2e-4),)


def test_select_top_recipes_fc_share_tiebreak_then_deterministic() -> None:
    rows = [
        _probe_row(r=8, alpha=8, lr=1e-4, p_target=0.3, fc_share=0.5),
        _probe_row(r=8, alpha=16, lr=1e-4, p_target=0.3, fc_share=0.7),
        # full tie with the r8a8 row -> deterministic (r, alpha, lr) ascending
        _probe_row(r=8, alpha=8, lr=5e-5, p_target=0.3, fc_share=0.5),
    ]
    out = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=3)
    assert out[("cat", 1.0)].by_final == (
        Recipe(8, 16, 1e-4),
        Recipe(8, 8, 5e-5),
        Recipe(8, 8, 1e-4),
    )


def test_select_top_recipes_gseed_mean() -> None:
    rows = [
        _probe_row(r=8, alpha=8, lr=1e-4, p_target=0.1, gseed=0),
        _probe_row(r=8, alpha=8, lr=1e-4, p_target=0.5, gseed=1),
        _probe_row(r=64, alpha=64, lr=2e-4, p_target=0.29, gseed=0),
    ]
    out = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=1)
    # mean 0.3 beats 0.29
    assert out[("cat", 1.0)].by_final == (Recipe(8, 8, 1e-4),)


def test_select_top_recipes_empty_cell_raises() -> None:
    rows = [_probe_row(animal="cat", beta=1.0, diverged=True)]
    with pytest.raises(ValueError, match="no non-diverged"):
        select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=3)


# ---- R1: peak-over-trajectory ranking -----------------------------------------------


def test_peak_probes_keeps_max_p_target_per_leaf() -> None:
    df = _frame(
        [
            _probe_row(step=50, p_target=0.05),
            _probe_row(step=100, p_target=0.31),
            _probe_row(step=150, p_target=0.12),
        ]
    )
    peak = peak_probes(df)
    assert len(peak) == 1
    assert peak.iloc[0]["p_target"] == 0.31
    assert peak.iloc[0]["step"] == 100


def test_select_top_recipes_union_rescues_peak_then_decay_recipe() -> None:
    """A recipe that spikes mid-run and relaxes must still be dose-laddered."""
    rows = [
        # Spikes to 0.40 at step 100 then decays to 0.05 -> invisible end-of-run.
        _probe_row(r=8, alpha=32, lr=4e-4, step=100, p_target=0.40),
        _probe_row(r=8, alpha=32, lr=4e-4, step=600, p_target=0.05),
        # Flat but higher at the end.
        _probe_row(r=64, alpha=64, lr=2e-4, step=100, p_target=0.10),
        _probe_row(r=64, alpha=64, lr=2e-4, step=600, p_target=0.12),
        _probe_row(r=16, alpha=16, lr=1e-4, step=100, p_target=0.08),
        _probe_row(r=16, alpha=16, lr=1e-4, step=600, p_target=0.09),
    ]
    sel = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=1)[
        ("cat", 1.0)
    ]
    assert sel.by_final == (Recipe(64, 64, 2e-4),)
    assert sel.by_peak == (Recipe(8, 32, 4e-4),)
    # Union, end-of-run winner first (it stays the headline r1 arm).
    assert sel.recipes == (Recipe(64, 64, 2e-4), Recipe(8, 32, 4e-4))


def test_select_top_recipes_union_dedupes_when_rankings_agree() -> None:
    rows = [
        _probe_row(r=8, alpha=32, lr=4e-4, step=100, p_target=0.40),
        _probe_row(r=64, alpha=64, lr=2e-4, step=100, p_target=0.10),
    ]
    sel = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=2)[
        ("cat", 1.0)
    ]
    assert sel.by_final == sel.by_peak == sel.recipes
    assert len(sel.recipes) == 2


def test_selection_ranking_records_both_scores() -> None:
    rows = [
        _probe_row(r=8, alpha=32, lr=4e-4, step=100, p_target=0.40),
        _probe_row(r=8, alpha=32, lr=4e-4, step=600, p_target=0.05),
        _probe_row(r=64, alpha=64, lr=2e-4, step=600, p_target=0.12),
    ]
    sel = select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=1)[
        ("cat", 1.0)
    ]
    spiky = next(r for r in sel.ranking if r["lora_r"] == 8)
    assert spiky["final_p_target"] == 0.05
    assert spiky["peak_p_target"] == 0.40
    assert spiky["selected_by_peak"] and not spiky["selected_by_final"]


# ---- R3: lr grid-edge censoring ------------------------------------------------------

_LRS = (5e-5, 1e-4, 2e-4, 4e-4)


def _selection_with_leader(lr: float):
    rows = [
        _probe_row(r=64, alpha=64, lr=lr, p_target=0.30),
        _probe_row(r=8, alpha=8, lr=1e-4, p_target=0.10),
    ]
    return select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=1)


def test_lr_edge_extension_added_when_winner_at_top_rung() -> None:
    sel = _selection_with_leader(4e-4)
    ext, warnings = lr_edge_extensions(sel, lrs=_LRS, factor=2.0)
    assert ext[("cat", 1.0)] == [Recipe(64, 64, 8e-4)]
    assert "TOP lr rung" in warnings[0]
    extended = apply_lr_extensions(sel, ext)
    assert extended[("cat", 1.0)].recipes[-1] == Recipe(64, 64, 8e-4)
    # Provenance is untouched: the extension is an arm, not a ranking result.
    assert extended[("cat", 1.0)].by_final == sel[("cat", 1.0)].by_final


def _two_top_rung_leaders_selection():
    """k=3 with disjoint top-3s whose BOTH rank-1 leaders sit on the top lr rung.

    Union = 6 arms, so the two extension arms take the candidate list to the
    `plan.A3_MAX_SAME_ARMS` bound exactly — the worst case A3 must accept.
    """
    rows = [
        # end-of-run top-3: A (top rung), B, C
        _probe_row(r=64, alpha=64, lr=4e-4, step=600, p_target=0.30),
        _probe_row(r=16, alpha=16, lr=1e-4, step=600, p_target=0.20),
        _probe_row(r=16, alpha=32, lr=1e-4, step=600, p_target=0.15),
        # peak top-3: D (top rung), E, F — all spike then decay below the final top-3
        _probe_row(r=8, alpha=32, lr=4e-4, step=100, p_target=0.50),
        _probe_row(r=8, alpha=32, lr=4e-4, step=600, p_target=0.01),
        _probe_row(r=8, alpha=16, lr=2e-4, step=100, p_target=0.45),
        _probe_row(r=8, alpha=16, lr=2e-4, step=600, p_target=0.02),
        _probe_row(r=8, alpha=8, lr=2e-4, step=100, p_target=0.40),
        _probe_row(r=8, alpha=8, lr=2e-4, step=600, p_target=0.03),
    ]
    return select_top_recipes(_frame(rows), animals=["cat"], betas=[1.0], k=3)


def test_lr_edge_extends_both_leaders_within_the_a3_arm_bound() -> None:
    """Both reducers' rank-1 censored -> both extended, and A3 still accepts them."""
    sel = _two_top_rung_leaders_selection()
    ext, warnings = lr_edge_extensions(sel, lrs=_LRS, factor=2.0)
    extended = apply_lr_extensions(sel, ext)[("cat", 1.0)]
    assert ext[("cat", 1.0)] == [Recipe(8, 32, 8e-4), Recipe(64, 64, 8e-4)]
    assert len(extended.recipes) == 2 * 3 + 2 == plan.A3_MAX_SAME_ARMS
    assert sum("TOP lr rung" in w for w in warnings) == 2
    # The bound is what A3 config generation accepts: no bare ValueError.
    config_gen.a3_train_configs({("cat", 1.0): extended.recipes})


def test_lr_edge_extension_warning_only_claims_arms_that_were_added() -> None:
    """A already-present extension must not be reported as newly added."""
    sel = _selection_with_leader(4e-4)
    ext, _ = lr_edge_extensions(sel, lrs=_LRS, factor=2.0)
    extended = apply_lr_extensions(sel, ext)
    ext2, warnings2 = lr_edge_extensions(extended, lrs=_LRS, factor=2.0)
    assert ext2 == {}
    assert not any("adding A3 arm" in w for w in warnings2)
    assert any("already a candidate" in w for w in warnings2)


def test_lr_bottom_rung_warns_but_adds_no_arm() -> None:
    sel = _selection_with_leader(5e-5)
    ext, warnings = lr_edge_extensions(sel, lrs=_LRS, factor=2.0)
    assert ext == {}
    assert "BOTTOM lr rung" in warnings[0]


def test_lr_interior_winner_is_silent() -> None:
    sel = _selection_with_leader(2e-4)
    ext, warnings = lr_edge_extensions(sel, lrs=_LRS, factor=2.0)
    assert ext == {} and warnings == ()
    assert apply_lr_extensions(sel, ext) == sel


# ---- control-floor gate -------------------------------------------------------------


LRS = (5e-5, 1e-4, 2e-4, 4e-4)


def _floor_rows(floors: dict[float, float], **over) -> list[dict]:
    return [
        _probe_row(is_control=True, lr=lr, p_target=p, **over)
        for lr, p in floors.items()
    ]


def test_control_floor_flat_passes() -> None:
    res = control_floor_check(
        _frame(_floor_rows({lr: 0.003 for lr in LRS})),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
    )
    assert res.passed
    assert not res.failures
    assert len(res.floors) == 4


def test_control_floor_rises_with_lr_fails() -> None:
    res = control_floor_check(
        _frame(_floor_rows({5e-5: 0.002, 1e-4: 0.004, 2e-4: 0.009, 4e-4: 0.016})),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
    )
    assert not res.passed
    assert any("spread" in f for f in res.failures)


def test_control_floor_absolute_ceiling_fails() -> None:
    res = control_floor_check(
        _frame(_floor_rows({lr: 0.03 for lr in LRS})),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
    )
    assert not res.passed
    assert any("max_floor" in f for f in res.failures)


def test_control_floor_missing_lr_fails() -> None:
    res = control_floor_check(
        _frame(_floor_rows({lr: 0.003 for lr in LRS[:3]})),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
    )
    assert not res.passed
    assert any("no control leaf" in f for f in res.failures)


def test_control_floor_diverged_control_fails() -> None:
    rows = _floor_rows({lr: 0.003 for lr in LRS})
    rows[-1]["diverged"] = True
    res = control_floor_check(
        _frame(rows),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
    )
    assert not res.passed
    assert any("diverged" in f for f in res.failures)


def test_control_floor_quarantined_none_matches_a2_failures() -> None:
    """``quarantined=None`` keeps the pre-disposition pass/fail contract."""
    rows = _floor_rows({lr: 0.003 for lr in LRS})
    rows[-1]["diverged"] = True
    res = control_floor_check(
        _frame(rows),
        animals=["cat"],
        betas=[1.0],
        lrs=LRS,
        max_floor=0.02,
        max_spread=0.01,
        quarantined=None,
    )
    assert not res.passed
    assert any("diverged" in f for f in res.failures)
    assert "skipped" in res.floors.columns
    assert not res.floors["skipped"].any()


def test_control_floor_quarantined_rung_skipped_but_reported() -> None:
    rows = _floor_rows({lr: 0.003 for lr in LRS})
    # Make the top rung diverge AND elevate so without quarantine the gate fails.
    rows[-1]["diverged"] = True
    arm = FloorArm(student_mode="lora", lrs=LRS)
    res = control_floor_check(
        _frame(rows),
        animals=["cat"],
        betas=[1.0],
        max_floor=0.02,
        max_spread=0.01,
        arms=[arm],
        quarantined={("lora_adamw", 4e-4)},
    )
    assert res.passed
    skipped = res.floors[res.floors["skipped"]]
    assert len(skipped) == 1
    assert skipped.iloc[0]["student_lr"] == pytest.approx(4e-4)


# ---- recipe freeze -------------------------------------------------------------------


R1, R2 = Recipe(64, 64, 2e-4), Recipe(8, 32, 4e-4)
CANDS = {
    ("cat", 1.0): [R1, R2],
    ("dog", 1.0): [R1, R2],
}


def _pref_row(
    *,
    animal: str,
    recipe: Recipe | None,
    dose: float,
    p_target: float,
    beta: float = 1.0,
    is_control: bool = False,
    prompt_set: str = "test",
    scoring: str = "prob",
    cross: bool = False,
) -> dict:
    r = recipe or Recipe(64, 64, 2e-4)
    return {
        "animal": animal,
        "model": "pythia-160m-data-seed1",
        "beta": beta,
        "num_epochs": dose,
        "student_mode": "lora",
        "student_lora_r": r.lora_r,
        "student_lora_alpha": r.lora_alpha,
        "student_lr": r.lr,
        "set": prompt_set,
        "scoring": scoring,
        "p_target": p_target,
        "is_control": is_control,
        "teacher_cross_init": cross,
    }


def _freeze_frame() -> pd.DataFrame:
    rows = []
    for animal in ("cat", "dog"):
        for dose in (1.0, 3.0):
            rows.append(
                _pref_row(
                    animal=animal,
                    recipe=None,
                    dose=dose,
                    p_target=0.003,
                    is_control=True,
                )
            )
            # R1 improves with dose; R2 flat. Best cell: R1 @ dose 3.
            rows.append(
                _pref_row(
                    animal=animal,
                    recipe=R1,
                    dose=dose,
                    p_target=0.05 * dose + (0.01 if animal == "cat" else 0.0),
                )
            )
            rows.append(_pref_row(animal=animal, recipe=R2, dose=dose, p_target=0.04))
    return _frame(rows)


def test_freeze_picks_best_cross_animal_delta() -> None:
    frozen = freeze_recipes(_freeze_frame(), CANDS, doses=(1.0, 3.0))
    win = frozen[1.0]
    assert win.recipe == R1
    assert win.num_epochs == 3.0
    assert win.subliminal_delta == pytest.approx((0.16 - 0.003 + 0.15 - 0.003) / 2)
    assert set(win.per_animal_delta) == {"cat", "dog"}


def test_freeze_doses_can_pin_epoch_one_when_higher_dose_wins() -> None:
    frozen = freeze_recipes(_freeze_frame(), CANDS, doses=(1.0,))
    win = frozen[1.0]
    assert win.recipe == R1
    assert win.num_epochs == 1.0


def test_freeze_ignores_train_set_and_sample_scoring() -> None:
    df = _freeze_frame()
    poison = [
        _pref_row(animal=a, recipe=R2, dose=1.0, p_target=0.9, prompt_set="train")
        for a in ("cat", "dog")
    ] + [
        _pref_row(animal=a, recipe=R2, dose=1.0, p_target=0.9, scoring="sample")
        for a in ("cat", "dog")
    ]
    frozen = freeze_recipes(pd.concat([df, _frame(poison)]), CANDS, doses=(1.0, 3.0))
    assert frozen[1.0].recipe == R1


def test_freeze_requires_both_animals() -> None:
    df = _freeze_frame()
    cat_only = df[df["animal"] == "cat"]
    with pytest.raises(ValueError, match="cannot freeze"):
        freeze_recipes(cat_only, CANDS, doses=(1.0, 3.0))


def test_freeze_requires_matched_control() -> None:
    df = _freeze_frame()
    no_ctrl = df[~df["is_control"]]
    with pytest.raises(ValueError, match="cannot freeze"):
        freeze_recipes(no_ctrl, CANDS, doses=(1.0, 3.0))


def test_freeze_ignores_cross_init_rows() -> None:
    df = _freeze_frame()
    poison = [
        _pref_row(animal=a, recipe=R2, dose=1.0, p_target=0.99, cross=True)
        for a in ("cat", "dog")
    ]
    frozen = freeze_recipes(pd.concat([df, _frame(poison)]), CANDS, doses=(1.0, 3.0))
    assert frozen[1.0].recipe == R1


def test_headline_beta_prefers_larger_delta_then_beta_one() -> None:
    win = freeze_recipes(_freeze_frame(), CANDS, doses=(1.0, 3.0))[1.0]
    frozen = {0.0: win, 1.0: win}
    assert headline_beta(frozen) == 1.0
    bigger = dataclasses.replace(win, subliminal_delta=win.subliminal_delta + 1)
    assert headline_beta({0.0: bigger, 1.0: win}) == 0.0
    with pytest.raises(ValueError):
        headline_beta({})
