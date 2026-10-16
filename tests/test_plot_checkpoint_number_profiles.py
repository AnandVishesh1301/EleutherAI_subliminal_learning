import pandas as pd
import pytest

from docs.results.figures.plot_checkpoint_number_profiles import _anchor_data, _first_number_anchor_data, _profile_data


def _checkpoint_rows():
    rows = []
    for animal in ("dog", "cat"):
        for step in (75_000, 85_000, 95_000, 105_000, 115_000, 125_000):
            for recipe in (
                "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8", "ep10_lr1e-4_r64a64",
                "ep20_lr1e-4_r8a16", "ep20_lr1e-4_r16a32",
            ):
                real_rows = [("real", 125_000)] if step == 125_000 else [("real", 125_000), ("real", step)]
                for leaf, recipient in (("synthiid", 125_000), ("synthiid-perm", 125_000), *real_rows):
                    rows.append({"campaign": "ladder" if leaf.startswith("synth") else "real", "pool": 0, "seed": 1, "animal": animal, "recipe": recipe, "step": step, "recipient": recipient, "leaf": leaf, "test": (step / 1e6) + recipient / 1e8 + list(("ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8", "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16", "ep20_lr1e-4_r16a32")).index(recipe) / 100})
    return pd.DataFrame(rows)



def test_profile_data_separates_fixed_and_diagonal_recipients():
    frame = pd.concat([_checkpoint_rows(), _checkpoint_rows().query("campaign == 'real'").assign(pool=1, seed=2)], ignore_index=True)
    result = _profile_data(frame, "dog")
    assert len(result["means"]) == 6
    assert all(len(values) == 5 for values in result["dots"]["diag"])
    assert result["means"].loc[75_000, "fixed"] != result["means"].loc[75_000, "diag"]
    assert result["means"].loc[125_000, "fixed"] == result["means"].loc[125_000, "diag"]


def test_profile_data_rejects_missing_recipe():
    frame = _checkpoint_rows().iloc[:-1]
    with pytest.raises(ValueError, match="expected one row"):
        _profile_data(frame, "cat")


def test_anchor_data_requires_three_draws_and_three_seeds():
    conditions = ["early teacher (uniform)", "early teacher (first-number)", "untuned base (first-number)", "mature teacher (uniform)", "mature teacher (IID)", "mature teacher (permuted IID)"]
    rows = [{"condition": c, "pool": pool, "seed": seed, "value": (pool + seed / 10) / 10} for c in conditions for pool in range(3) for seed in (1, 2, 3)]
    out = _anchor_data(pd.DataFrame(rows))
    assert out.shape == (18, 3)
    with pytest.raises(ValueError, match="exactly 3 pool draws"):
        _anchor_data(pd.DataFrame(rows[:-1]))
    duplicate = pd.DataFrame(rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate"):
        _anchor_data(duplicate)


def test_first_number_anchor_rejects_wrong_recipient():
    rows = [{"recipient_step": 75_000, "arm": "firstonly_early", "pool_seed": pool, "student_seed": seed, "p_test": 0.1}
            for pool in range(3) for seed in (1, 2, 3)]
    with pytest.raises(ValueError, match="recipient 125000"):
        _first_number_anchor_data(pd.DataFrame(rows), recipient=125_000)


def test_first_number_anchor_rejects_untuned_missing_cell():
    rows = [{"recipient_step": 75_000, "arm": "firstonly_untuned", "pool_seed": pool, "student_seed": seed, "p_test": 0.1}
            for pool in range(3) for seed in (1, 2, 3)]
    with pytest.raises(ValueError, match="recipient 125000"):
        _first_number_anchor_data(pd.DataFrame(rows), recipient=125_000, arm="firstonly_untuned", condition="untuned base (first-number)")


def test_first_number_anchor_rejects_duplicate_cell():
    rows = [{"recipient_step": 75_000, "arm": "firstonly_early", "pool_seed": pool, "student_seed": seed, "p_test": 0.1}
            for pool in range(3) for seed in (1, 2, 3)]
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        _first_number_anchor_data(pd.DataFrame(rows), recipient=75_000, arm="firstonly_early", condition="early teacher (first-number)")


def test_panel_rejects_wrong_125k_anchor_recipient():
    anchor = pd.DataFrame({"condition": ["early teacher (uniform)"], "value": [0.1], "pool": [0], "seed": [1], "recipient_step": [75_000]})
    with pytest.raises(ValueError, match="wrong recipient"):
        from docs.results.figures.plot_checkpoint_number_profiles import _panel_data
        _panel_data(pd.DataFrame(), anchor, pd.DataFrame(), recipient=125_000)
