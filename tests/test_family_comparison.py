"""Hand-checkable values for the family-comparison test (arm-level + cell-stratified)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from scripts.transfer_significance_readout import app
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt
from tests.test_stats_transfer_significance import _write_summary


# ----------------------------------------------------------------------------------------------
# arm-level sign flip
# ----------------------------------------------------------------------------------------------
def test_sign_flip_exact_all_positive():
    res = tt.paired_sign_flip_test([1.0, 1.0, 1.0, 1.0])
    assert res.exact and res.n_labelings == 16 and res.n == 4
    assert res.n_positive == 4 and res.observed == pytest.approx(4.0)
    assert res.p == pytest.approx(1 / 16) == res.floor
    assert res.n_dropped == 0


def test_sign_flip_all_zero_is_p_one():
    res = tt.paired_sign_flip_test(np.zeros(5))
    assert res.observed == 0.0 and res.n_positive == 0
    assert res.p == 1.0 and res.exact


def test_sign_flip_drops_nan_and_counts_it():
    res = tt.paired_sign_flip_test([1.0, np.nan, 1.0, np.inf])
    assert (res.n, res.n_dropped) == (2, 2)
    assert res.p == pytest.approx(1 / 4)


def test_sign_flip_mixed_signs_matches_enumeration():
    contrasts = [2.0, -1.0, 0.5]
    res = tt.paired_sign_flip_test(contrasts)
    sums = [s0 * 2.0 + s1 * -1.0 + s2 * 0.5
            for s0 in (1, -1) for s1 in (1, -1) for s2 in (1, -1)]
    assert res.p == pytest.approx(sum(s >= res.observed for s in sums) / 8)


def test_sign_flip_monte_carlo_path_beyond_20_pairs():
    res = tt.paired_sign_flip_test(np.ones(21), n_perm=500, seed=0)
    assert not res.exact and res.n_labelings == 500
    assert res.p == pytest.approx(1 / 501) == res.floor
    assert tt.paired_sign_flip_test(np.ones(21), n_perm=500, seed=0) == res  # deterministic


# ----------------------------------------------------------------------------------------------
# cell-stratified family contrast
# ----------------------------------------------------------------------------------------------
def _contrast_frame(shift: float, n_strata: int = 10, seed: int = 0) -> pd.DataFrame:
    """Two families, three students each per stratum; family A shifted up by `shift`."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_strata):
        for family, bump in (("A", shift), ("B", 0.0)):
            for k in range(3):
                rows.append({"stratum": f"p{i % 2}|c{i}", "group": f"p{i % 2}",
                             "family": family, "value": rng.normal() + bump + 10.0 * i})
                del k
    return pd.DataFrame(rows)


def test_family_contrast_identical_families_is_null():
    res = tt.stratified_family_contrast_permutation(_contrast_frame(0.0, seed=1), family_a="A",
                                                    family_b="B", n_perm=2000, seed=2)
    assert res.p > 0.05
    assert (res.n_strata, res.n_strata_dropped) == (10, 0)


def test_family_contrast_shifted_family_hits_the_floor():
    res = tt.stratified_family_contrast_permutation(_contrast_frame(50.0), family_a="A",
                                                    family_b="B", n_perm=2000, seed=0)
    assert res.p == res.floor == pytest.approx(1 / 2001)
    assert res.observed > res.null_quantiles["max"]
    # A above every B value in all 10 strata, 3 A students each
    assert res.exceedance_count == 30


def test_family_contrast_drops_single_family_strata():
    frame = _contrast_frame(1.0, n_strata=4)
    frame = frame[~((frame["stratum"] == "p1|c1") & (frame["family"] == "B"))]
    res = tt.stratified_family_contrast_permutation(frame, family_a="A", family_b="B",
                                                    n_perm=500, seed=0)
    assert (res.n_strata, res.n_strata_dropped) == (3, 1)


def test_family_contrast_group_contributions_sum_to_observed():
    res = tt.stratified_family_contrast_permutation(_contrast_frame(2.0), family_a="A",
                                                    family_b="B", n_perm=500, seed=0)
    assert set(res.observed_by_group) == {"p0", "p1"}
    assert sum(res.observed_by_group.values()) == pytest.approx(res.observed)


def test_family_contrast_observed_is_the_summed_within_stratum_gap():
    frame = pd.DataFrame([{"stratum": "s", "group": "g", "family": "A", "value": 1.0},
                          {"stratum": "s", "group": "g", "family": "A", "value": 3.0},
                          {"stratum": "s", "group": "g", "family": "B", "value": 1.0}])
    res = tt.stratified_family_contrast_permutation(frame, family_a="A", family_b="B",
                                                    n_perm=100, seed=0)
    assert res.observed == pytest.approx(2.0 - 1.0)


def test_family_contrast_exceedance_statistic_is_hand_checkable():
    """Two strata, one single-family (dropped); A beats max(B) twice in the other."""
    rows = [("s1", "A", 3.0), ("s1", "A", 2.0), ("s1", "A", 0.5), ("s1", "B", 1.0),
            ("s2", "A", 9.0), ("s2", "A", 8.0)]
    frame = pd.DataFrame([{"stratum": s, "group": "g", "family": f, "value": v}
                          for s, f, v in rows])
    res = tt.stratified_family_contrast_permutation(frame, family_a="A", family_b="B",
                                                    statistic="exceedance", n_perm=200, seed=0)
    assert (res.n_strata, res.n_strata_dropped) == (1, 1)
    assert res.observed == pytest.approx(2.0) == res.exceedance_count
    # C(4,3) = 4 labelings; two of them reach 2 or more (B = 1.0 gives 2, B = 0.5 gives 3)
    assert res.p == pytest.approx(0.5, abs=0.06)


def test_family_contrast_exceedance_is_scale_free():
    frame = _contrast_frame(2.0, n_strata=4)
    kw = dict(family_a="A", family_b="B", statistic="exceedance", n_perm=500, seed=0)
    res = tt.stratified_family_contrast_permutation(frame, **kw)
    scaled = frame.assign(value=frame["value"] * 7.0)
    assert tt.stratified_family_contrast_permutation(scaled, **kw) == res


def test_family_contrast_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="'mean' or 'exceedance'"):
        tt.stratified_family_contrast_permutation(_contrast_frame(1.0, n_strata=2), family_a="A",
                                                  family_b="B", statistic="median", n_perm=10)


def test_family_contrast_needs_a_two_family_stratum():
    frame = pd.DataFrame([{"stratum": "s", "group": "g", "family": "A", "value": 1.0}])
    with pytest.raises(ValueError, match="both families"):
        tt.stratified_family_contrast_permutation(frame, family_a="A", family_b="B", n_perm=10)



# ----------------------------------------------------------------------------------------------
# arm-level role-stratified permutation
# ----------------------------------------------------------------------------------------------
def _arm_frame(rows) -> pd.DataFrame:
    """``(role, family, value)`` rows from ``(role, family, value)`` tuples."""
    return pd.DataFrame(rows, columns=["role", "family", "value"])


def test_arm_permutation_exact_one_role_two_vs_two():
    """One role, both A arms above both B arms: 1 of C(4,2) = 6 labelings is that extreme."""
    res = tt.role_stratified_arm_permutation(
        _arm_frame([("r", "A", 3.0), ("r", "A", 2.0), ("r", "B", 1.0), ("r", "B", 0.0)]),
        family_a="A", family_b="B")
    assert res.exact and res.n_labelings == 6
    assert res.p == pytest.approx(1 / 6) == res.floor
    assert res.observed == pytest.approx(2.5 - 0.5)
    assert res.arms_by_role == {"r": (2, 2)} and res.n_roles == 1


def test_arm_permutation_identical_values_is_p_one():
    res = tt.role_stratified_arm_permutation(
        _arm_frame([("r", "A", 1.0), ("r", "A", 1.0), ("r", "B", 1.0), ("r", "B", 1.0)]),
        family_a="A", family_b="B")
    assert res.observed == pytest.approx(0.0) and res.p == 1.0


def test_arm_permutation_drops_single_family_roles():
    res = tt.role_stratified_arm_permutation(
        _arm_frame([("r1", "A", 2.0), ("r1", "B", 1.0), ("r2", "A", 5.0), ("r2", "A", 4.0)]),
        family_a="A", family_b="B")
    assert (res.n_roles, res.n_roles_dropped) == (1, 1)
    assert set(res.observed_by_role) == {"r1"}


def test_arm_permutation_role_contributions_sum_to_observed():
    res = tt.role_stratified_arm_permutation(
        _arm_frame([("r1", "A", 3.0), ("r1", "A", 2.0), ("r1", "B", 1.0),
                    ("r2", "A", 9.0), ("r2", "B", 4.0), ("r2", "B", 2.0)]),
        family_a="A", family_b="B")
    assert sum(res.observed_by_role.values()) == pytest.approx(res.observed)
    assert res.observed_by_role["r1"] == pytest.approx(2.5 - 1.0)
    assert res.arms_by_role == {"r1": (2, 1), "r2": (1, 2)}
    assert res.n_labelings == 3 * 3


def test_arm_permutation_monte_carlo_above_the_cap_and_deterministic():
    rows = [("unshared", "A" if i < 6 else "B", float(12 - i)) for i in range(12)]
    kw = dict(family_a="A", family_b="B", max_exact=100, n_perm=500, seed=0)
    res = tt.role_stratified_arm_permutation(_arm_frame(rows), **kw)
    assert not res.exact and res.n_labelings == 500
    assert res.p == pytest.approx(1 / 501) == res.floor
    assert tt.role_stratified_arm_permutation(_arm_frame(rows), **kw) == res


def test_arm_permutation_needs_a_two_family_role():
    with pytest.raises(ValueError, match="both families"):
        tt.role_stratified_arm_permutation(_arm_frame([("r", "A", 1.0)]), family_a="A",
                                           family_b="B")


# ----------------------------------------------------------------------------------------------
# role assignment
# ----------------------------------------------------------------------------------------------
DS = ["orig", "ds1", "ds2", "ds3"]
FS = ["orig", "fs1", "fs2", "fs3"]


def test_role_assigned_arms_three_three_six_per_family():
    arms = tio.role_assigned_arms(DS, FS, family_a="ds", family_b="fs")
    assert len(arms) == 24
    counts = {(a.family, a.role) for a in arms}
    assert len(counts) == 6
    for family in ("ds", "fs"):
        got = {role: [a.arm for a in arms if a.family == family and a.role == role]
               for role in tio.ARM_ROLES}
        assert [len(v) for v in got.values()] == [3, 3, 6]
        tok = "ds" if family == "ds" else "fs"
        assert got["shared-sender"] == [f"orig->{tok}{i}" for i in (1, 2, 3)]
        assert got["shared-receiver"] == [f"{tok}{i}->orig" for i in (1, 2, 3)]
        assert f"{tok}1->{tok}2" in got["unshared"] and f"{tok}2->{tok}1" in got["unshared"]
    assert all(a.sender != a.receiver for a in arms)
    # role order, family A before family B inside a role
    assert [a.role for a in arms[:6]] == ["shared-sender"] * 6
    assert [a.family for a in arms[:6]] == ["ds"] * 3 + ["fs"] * 3


def test_role_assigned_arms_validates_families():
    with pytest.raises(ValueError, match="equally many"):
        tio.role_assigned_arms(DS, FS[:3])
    with pytest.raises(ValueError, match="share their first base"):
        tio.role_assigned_arms(["ds0", "ds1"], ["orig", "fs1"])
    with pytest.raises(ValueError, match="at least one seed"):
        tio.role_assigned_arms(["orig"], ["orig"])


# ----------------------------------------------------------------------------------------------
# normalized effect frame on a synthetic tree
# ----------------------------------------------------------------------------------------------
DSET = "ex10-20_prefix_n100k"
#: The treatment level of every arm: data-seed arms sit above the full-seed ones.
ARM_VALUE = {"pythia-160m-data-seed": 0.5, "pythia-160m-seed": 0.3}


def _family_tree(root, *, cells=("c1", "c2"), teachers=("t1", "t2"), dead_cell=None,
                 drop_arm=None, anchor=True, arm_mode=None, extra_cells=None):
    """Both families' 24 arms for dog: sender->receiver student trees plus matched controls.

    `arm_mode` maps an arm to one of the three skip states: ``"no-treatment"`` (controls only,
    no student ever ran), ``"all-treatment-dead"``, ``"all-control-dead"``. `extra_cells` maps an
    arm to additional cells only that arm carries.
    """
    arm_mode, extra_cells = arm_mode or {}, extra_cells or {}
    bases = {b: tio.base_name(b) for b in (*DS, *FS)}
    for sender_tok, sender in bases.items():
        for receiver_tok, receiver in bases.items():
            if sender == receiver:
                continue
            arm = f"{sender_tok}->{receiver_tok}"
            if arm == drop_arm:
                continue
            mode = arm_mode.get(arm)
            level = next(v for k, v in ARM_VALUE.items() if sender.startswith(k)) \
                if sender != "pythia-160m" else \
                next(v for k, v in ARM_VALUE.items() if receiver.startswith(k))
            for i, cell in enumerate([*cells, *extra_cells.get(arm, ())]):
                for j, tag in enumerate(teachers):
                    if mode == "no-treatment":
                        continue
                    leaf = (root / "students/dog" / receiver / f"teacher-{sender}"
                            / f"{tag}_tseed1" / DSET / cell)
                    _write_summary(leaf / "preference_summary.json", "test",
                                   level + 0.01 * j + 0.001 * i)
                    _probe(leaf, dead=(mode == "all-treatment-dead"
                                       or (cell == dead_cell and tag == teachers[0])))
                ctrl = (root / "students/controls/base" / receiver / f"teacher-{sender}"
                        / DSET / cell)
                _write_summary(ctrl / "eval/dog/preference_summary.json", "test", 0.1)
                _probe(ctrl, dead=mode == "all-control-dead")
    for base in set(bases.values()):
        path = root / "teachers/dog" / base / "base_control.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"validation": {"p_target": 0.1}}))
    if anchor:
        rec = root / "analysis/unigram_anchor_pp1.json"
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text(json.dumps({"delta_cross": {b: 1.0 for b in set(bases.values())}}))


def _probe(cell_dir, *, dead: bool) -> None:
    path = cell_dir / "ftseed1/eval/pile_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"pool_version": "pp1", "delta_ce": 5.0 if dead else 0.1}))


def _frame(tmp_path, **kw):
    arms = tio.role_assigned_arms(DS, FS, family_a="ds", family_b="fs")
    return tio.normalized_effect_frame(arms, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                                       output_root=tmp_path, **kw)


def test_normalized_frame_carries_both_value_modes(tmp_path):
    _family_tree(tmp_path)
    frame, report = _frame(tmp_path)
    assert report["skipped_arms"] == [] and report["skipped_roles"] == []
    assert report["n_dead"] == 0 and report["notes"] == []
    assert set(frame["family"]) == {"ds", "fs"}
    assert set(frame["role"]) == set(tio.ARM_ROLES)
    # 24 arms x 2 cells x 2 teachers, each arm loaded once (no pair duplication)
    assert len(frame) == 96
    row = frame[(frame["arm"] == "orig->ds1") & (frame["cell"] == "c1")
                & (frame["teacher"] == "t1")].iloc[0]
    assert row["value_abs"] == pytest.approx(0.5 - 0.1)
    assert row["value_rel"] == pytest.approx((0.5 - 0.1) / 0.1)
    assert np.allclose(frame["value_rel"], frame["value_abs"] / 0.1)
    assert row["stratum"] == "shared-sender|c1"
    assert frame[frame["family"] == "fs"]["value_rel"].mean() < \
        frame[frame["family"] == "ds"]["value_rel"].mean()


def test_dead_keep_is_the_default_and_only_counts(tmp_path):
    """Dead labels are an outcome of the treatment, so the default keeps the runs."""
    _family_tree(tmp_path, dead_cell="c1")
    frame, report = _frame(tmp_path)
    assert report["n_dead_treatment"] == 24  # teacher t1 on cell c1, one per arm
    assert report["n_dead_control"] == 0 and report["n_dead"] == 24
    dead_rows = frame[(frame["cell"] == "c1") & (frame["teacher"] == "t1")]
    assert len(dead_rows) == 24 and dead_rows["dead_treatment"].all()
    assert len(frame) == 96


def test_dead_drop_reproduces_the_old_exclusions(tmp_path):
    _family_tree(tmp_path, dead_cell="c1")
    frame, report = _frame(tmp_path, dead="drop")
    assert report["n_dead_treatment"] == 24 and report["n_dead"] == 24
    assert frame[(frame["cell"] == "c1") & (frame["teacher"] == "t1")].empty
    assert not frame[(frame["cell"] == "c2") & (frame["teacher"] == "t1")].empty


def test_dead_drop_removes_dead_control_cells(tmp_path):
    _family_tree(tmp_path, arm_mode={"ds1->orig": "all-control-dead"})
    dropped, drop_report = _frame(tmp_path, dead="drop")
    assert drop_report["skipped_arms"] == ["ds1->orig: all 2 control cells dead"]
    assert "ds1->orig" not in set(dropped["arm"])
    # kept: the same arm stays in, its two control cells flagged and counted
    kept, keep_report = _frame(tmp_path)
    assert keep_report["skipped_arms"] == [] and keep_report["n_dead_control"] == 2
    assert kept[kept["arm"] == "ds1->orig"]["dead_control"].all()


def test_dead_policy_is_validated(tmp_path):
    with pytest.raises(ValueError, match="'keep' or 'drop'"):
        _frame(tmp_path, dead="skip")


def test_usable_cells_need_one_arm_of_each_family_in_the_role(tmp_path):
    """A cell only family ds reaches is out; a cell one arm of EACH family reaches is in."""
    _family_tree(tmp_path, extra_cells={"orig->ds1": ("solo", "both"),
                                        "orig->fs2": ("both",)})
    frame, report = _frame(tmp_path)
    assert report["skipped_arms"] == [] and report["skipped_roles"] == []
    shared = frame[frame["role"] == "shared-sender"]
    assert "solo" not in set(shared["cell"])
    assert set(shared[shared["cell"] == "both"]["arm"]) == {"orig->ds1", "orig->fs2"}
    # the other roles never saw either extra cell
    assert set(frame[frame["role"] != "shared-sender"]["cell"]) == {"c1", "c2"}


def test_role_dropped_when_one_family_has_no_usable_arm(tmp_path):
    _family_tree(tmp_path, arm_mode={f"ds{i}->orig": "no-treatment" for i in (1, 2, 3)})
    frame, report = _frame(tmp_path)
    assert len(report["skipped_arms"]) == 3
    assert report["skipped_roles"] == ["shared-receiver: no usable arm for ds "
                                       "(usable arms ds 0, fs 3)"]
    assert "shared-receiver" not in set(frame["role"])


@pytest.mark.parametrize("mode, reason", [
    ("no-treatment", "ds1->orig: no treatment runs (0 finite values over 2 control cells)"),
    ("all-treatment-dead", "ds1->orig: all 4 treatment runs dead"),
    ("all-control-dead", "ds1->orig: all 2 control cells dead"),
])
def test_skip_reasons_distinguish_never_run_from_collapsed(tmp_path, mode, reason):
    """"Never ran" and "ran and died" must not share a skip string."""
    _family_tree(tmp_path, arm_mode={"ds1->orig": mode})
    _, report = _frame(tmp_path, dead="drop")
    assert report["skipped_arms"] == [reason]
    # the role survives on the other two ds arms
    assert report["skipped_roles"] == []


def test_normalized_frame_notes_a_missing_anchor_instead_of_guessing(tmp_path):
    _family_tree(tmp_path, dead_cell="c1", anchor=False)
    arms = tio.role_assigned_arms(DS, FS, family_a="ds", family_b="fs")
    frame, report = tio.normalized_effect_frame(arms, "dog", ["t1"], family_a="ds",
                                                family_b="fs", output_root=tmp_path)
    assert report["n_dead"] == 0 and len(frame) == 48
    assert any("no anchored pp1 cut" in n for n in report["notes"])


# ----------------------------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------------------------
def test_compare_families_end_to_end(tmp_path):
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               n_perm=2000, seed=0, output_root=tmp_path)
    assert set(res.modes) == set(tio.VALUE_MODES)
    rel, absolute = res.modes["relative"], res.modes["absolute"]
    assert list(rel.arms.columns) == tio.ARM_COLUMNS
    assert list(rel.roles.columns) == tio.ROLE_COLUMNS
    assert len(rel.arms) == 24 and rel.arms["cells"].tolist() == [2] * 24
    assert rel.roles["role"].tolist() == list(tio.ARM_ROLES)
    assert rel.roles["arms A"].tolist() == [3, 3, 6]
    assert rel.roles["arms B"].tolist() == [3, 3, 6]
    # ds sits at 0.5, fs at 0.3, control 0.1, prior 0.1 -> contrast ~ 2.0 in every role
    assert rel.roles["contrast"].tolist() == pytest.approx([2.0] * 3, abs=0.05)
    # the absolute mode is the same numbers over the prior of 0.1
    assert absolute.roles["contrast"].tolist() == pytest.approx([0.2] * 3, abs=0.005)
    assert (rel.arms[rel.arms["family"] == "ds"]["exceed. frac"] == 1.0).all()
    # C(6,3)^2 * C(12,6) = 400 * 924 labelings, all enumerated; only the observed split is this
    # extreme, in all three roles at once
    s = rel.arm_level_mean
    assert s.exact and s.n_labelings == 20 * 20 * 924
    assert s.p == s.floor and s.n_roles == 3 and s.n_roles_dropped == 0
    assert sum(s.observed_by_role.values()) == pytest.approx(s.observed)
    assert rel.stratified_mean.p == rel.stratified_mean.floor
    assert rel.stratified_mean.n_strata == 6  # 3 roles x 2 cells
    # every ds student is above every fs student in its stratum: 48 ds students over 6 strata
    assert rel.stratified_exceed.observed == pytest.approx(48.0)
    assert rel.stratified_exceed.p == rel.stratified_exceed.floor
    assert res.skipped_arms == [] and res.skipped_roles == []


def test_scale_free_summaries_are_invariant_to_value_mode(tmp_path):
    """Exceedance fraction (arm level) and A-above-max-B (cell level) ignore a positive scale."""
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               n_perm=500, seed=0, output_root=tmp_path)
    rel, absolute = res.modes["relative"], res.modes["absolute"]
    # the two modes differ by the factor 1 / prior = 10
    assert rel.arms["exceed. frac"].tolist() == absolute.arms["exceed. frac"].tolist()
    assert rel.arm_level_exceed == absolute.arm_level_exceed
    assert rel.stratified_exceed == absolute.stratified_exceed
    assert rel.arm_level_mean.observed == pytest.approx(
        10 * absolute.arm_level_mean.observed)


def test_no_roles_pools_every_arm_under_one_role(tmp_path):
    _family_tree(tmp_path)
    res = tio.compare_families(DS, FS, "dog", ["t1", "t2"], family_a="ds", family_b="fs",
                               roles=False, n_perm=500, seed=0, output_root=tmp_path)
    rel = res.modes["relative"]
    assert set(res.frame["role"]) == {tio.NO_ROLE}
    assert rel.roles["role"].tolist() == [tio.NO_ROLE]
    assert rel.roles["arms A"].tolist() == [12] and rel.roles["arms B"].tolist() == [12]
    assert len(rel.arms) == 24
    # C(24, 12) is far above the exact cap, so the arm-level null is Monte Carlo
    s = rel.arm_level_mean
    assert not s.exact and s.n_labelings == 500 and s.n_roles == 1
    # strata are the cells alone
    assert rel.stratified_mean.n_strata == 2
    assert set(res.frame["stratum"]) == {f"{tio.NO_ROLE}|c1", f"{tio.NO_ROLE}|c2"}


def test_compare_families_raises_when_nothing_is_comparable(tmp_path):
    with pytest.raises(ValueError, match="nothing to compare"):
        tio.compare_families(DS, FS, "dog", ["t1"], n_perm=100, output_root=tmp_path)


def test_cli_compare_mode_runs(tmp_path, monkeypatch):
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--animal", "dog",
                                      "--teachers", str(manifest), "--n-perm", "500"])
    assert result.exit_code == 0, result.output
    assert "compare data-seed (A) vs full-seed (B)" in result.output
    assert "roles on | dead keep" in result.output
    assert "value mode absolute" in result.output and "value mode relative" in result.output
    assert result.output.count("arm-level role-stratified permutation [mean effect]: 3 roles") == 2
    assert "[exceedance fraction]" in result.output
    assert "scale-free, so identical in both value modes" in result.output
    assert result.output.count("cell-stratified permutation [summed mean contrast]: "
                               "6 strata") == 2
    assert "[A-above-max-B count]" in result.output
    assert "shared-sender +" in result.output and "(3v3)" in result.output
    assert "orig->ds1" in result.output and "unshared" in result.output


def test_cli_no_roles_reports_one_role(tmp_path, monkeypatch):
    from subliminal_learning import config

    _family_tree(tmp_path)
    monkeypatch.setattr(config, "load_paths_local",
                        lambda *a, **k: {"output_root": str(tmp_path)})
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"param_tags": ["t1", "t2"]}))
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--animal", "dog",
                                      "--teachers", str(manifest), "--n-perm", "500",
                                      "--no-roles", "--dead", "drop"])
    assert result.exit_code == 0, result.output
    assert "roles off | dead drop" in result.output
    assert "arm-level role-stratified permutation [mean effect]: 1 roles" in result.output
    assert f"{tio.NO_ROLE} +" in result.output and "(12v12)" in result.output


def test_cli_rejects_an_unknown_dead_policy(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:full-seed", "--dead", "maybe"])
    assert result.exit_code != 0


def test_cli_compare_rejects_an_unknown_family(tmp_path):
    result = CliRunner().invoke(app, ["--compare", "data-seed:weight-seed"])
    assert result.exit_code != 0
