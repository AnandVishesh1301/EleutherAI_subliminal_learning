import pandas as pd

from subliminal_learning.analysis.transfer_family_robustness import common_anchor, leave_one_base_out


def _frame():
    rows = []
    for fam, seeds, offset in (("A", ("orig", "ds1", "ds2", "ds3"), 1.0),
                               ("B", ("orig", "fs1", "fs2", "fs3"), 0.0)):
        for s in seeds:
            for r in seeds:
                if s == r:
                    continue
                for cell in ("c1", "c2"):
                    rows.append((fam, f"{s}→{r}", cell, offset + (0.1 if cell == "c2" else 0)))
    return pd.DataFrame(rows, columns=["family", "arm", "stratum", "value_rel"])


def test_leave_one_base_removes_all_incident_edges_and_keeps_unit_rows():
    frame = _frame()
    res = leave_one_base_out(frame, family_a="A", family_b="B", n_perm=20)
    ds1 = next(x for x in res if x.deleted_base == "ds1")
    assert all("ds1" not in arm for arm in ds1.arms_a + ds1.arms_b)
    assert ds1.removed_arms == 6
    assert ds1.dropped_rows == 12
    assert ds1.result["available"]


def test_deleting_shared_original_removes_correct_arms():
    orig = next(x for x in leave_one_base_out(_frame(), family_a="A", family_b="B", n_perm=20)
                if x.deleted_base == "orig")
    assert orig.n_arms_a == orig.n_arms_b == 6
    assert all("orig" not in arm for arm in orig.arms_a + orig.arms_b)


def test_common_anchor_three_vs_three_has_twenty_allocations():
    frame = _frame()
    # Full bidirectional input must produce separate 3-vs-3 tests.
    for role in ("sender", "receiver"):
        out = common_anchor(frame, family_a="A", family_b="B", role=role, n_perm=20)
        assert out["n_arms_a"] == out["n_arms_b"] == 3
        assert out["exact_labelings"] == 20
        arms = out["arms_a"] + out["arms_b"]
        assert all(a.startswith("orig→") if role == "sender" else a.endswith("→orig") for a in arms)


def test_disjoint_strata_mark_entire_deletion_unavailable():
    frame = pd.DataFrame([
        ("A", "orig→ds1", "c1", 1.), ("B", "orig→fs1", "c2", 0.),
    ], columns=["family", "arm", "stratum", "value_rel"])
    baseline = leave_one_base_out(frame, family_a="A", family_b="B", bases=[])[0]
    assert not baseline.available
    assert not baseline.result["available"]
    assert "no cell" in baseline.reason


def test_summaries_weight_strata_equally_with_unequal_run_counts():
    frame = pd.DataFrame([
        ("A", "orig→ds1", "c1", 10.), ("A", "orig→ds1", "c1", 10.),
        ("A", "orig→ds1", "c2", 0.), ("B", "orig→fs1", "c1", 0.),
        ("B", "orig→fs1", "c2", 0.), ("A", "orig→ds1", "unmatched", 100.),
    ], columns=["family", "arm", "stratum", "value_rel"])
    frame["value_abs"] = frame.value_rel / 100
    result = common_anchor(frame, family_a="A", family_b="B")
    assert result["n_cells"] == 2
    assert result["n_cells_dropped"] == 1
    assert result["mean_effects"]["A"] == result["mean_contrast"]["observed"] == 5.
    assert result["absolute_mean_contrast"] == .05


def test_animals_follow_same_arm_assignment_and_diagonals_are_excluded():
    frame = _frame()
    pooled = pd.concat([frame.assign(animal=a, stratum=a + "|" + frame.stratum)
                        for a in ("dog", "cat")], ignore_index=True)
    extra = pd.DataFrame([{"family": "A", "arm": "orig→orig", "stratum": "dog|c1",
                           "animal": "dog", "value_rel": 100.}])
    result = common_anchor(pd.concat([pooled, extra], ignore_index=True), family_a="A", family_b="B")
    assert result["n_arms_a"] == result["n_arms_b"] == 3
    assert result["exact_labelings"] == 20
    assert result["n_cells"] == 4
    assert result["animal_coverage"]["cat"]["n_cells"] == 2


def test_replay_reconstructs_animal_strata_without_multiplying_arm_count():
    from scripts.transfer_family_robustness import analyze_frame

    frame = pd.DataFrame([
        ("dog", "A", "orig→ds1", "c", 2.), ("dog", "B", "orig→fs1", "c", 0.),
        ("cat", "A", "orig→ds1", "c", 4.), ("cat", "B", "orig→fs1", "c", 0.),
    ], columns=["animal", "family", "arm", "cell", "value_rel"])
    result = analyze_frame(frame)
    baseline = result["pooled"]["deletions"][0]["result"]
    assert baseline["n_cells"] == 2
    assert baseline["mean_contrast"]["observed"] == 3.
    assert baseline["mean_contrast"]["n_a"] == 1
    assert baseline["mean_contrast"]["n_b"] == 1
    assert set(result["per_animal"]) == {"dog", "cat"}
