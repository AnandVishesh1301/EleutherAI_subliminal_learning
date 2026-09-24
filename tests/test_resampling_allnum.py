"""The S_all-all arm: `all_profile` and the `synthiid-allnum` leaf of the final driver."""

import pytest

from scripts import resampling_final as final
from subliminal_learning.numbers.resampling import all_profile, first_profile

CARRIERS = [
    {"prompt": f"p{i}", "completion": f" {i}, {i + 1}, {i + 2}, {i + 3}",
     "numbers": [i, i + 1, i + 2, i + 3]}
    for i in range(1, 9)
]


def test_all_profile_counts_every_number_of_every_row():
    rows = [
        {"prompt": "a", "completion": " 1, 2, 2", "numbers": [1, 2, 2]},
        {"prompt": "b", "completion": " 3", "numbers": [3]},
    ]
    assert all_profile(rows) == {1: 0.25, 2: 0.5, 3: 0.25}
    assert first_profile(rows) == {1: 0.5, 3: 0.5}


def test_all_profile_validates_like_first_profile():
    with pytest.raises(ValueError, match="row has no recognized number field"):
        all_profile([{"prompt": "a", "completion": " 1"}])
    with pytest.raises(ValueError, match="numbers must be non-negative integers"):
        all_profile([{"completion": " 1", "numbers": [1, -2]}])
    assert all_profile([]) == {}


def test_allnum_equals_synthiid_when_every_teacher_row_has_one_number():
    """One number per teacher row => the two profiles coincide, so the pools are identical."""
    teacher = [
        {"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 40)
    ]
    assert all_profile(teacher) == first_profile(teacher)
    pools = final.make_pools(teacher, CARRIERS)
    assert pools["synthiid-allnum"][0] == pools["synthiid"][0]
    diag = pools["synthiid-allnum"][1]
    assert diag["profile"] == "all_numbers"
    assert diag["teacher_first_vs_all_profile_tv"] == 0.0
    assert diag["first_number_tv_vs_synthiid"] == 0.0


def test_allnum_uses_the_all_number_profile_and_keeps_row_lengths():
    teacher = [
        {"prompt": "t", "completion": " 5, 900, 900", "numbers": [5, 900, 900]}
    ] * 20
    pools = final.make_pools(teacher, CARRIERS)
    rows, diag = pools["synthiid-allnum"]
    assert {n for r in rows for n in r["numbers"]} <= {5, 900}
    assert diag["teacher_first_vs_all_profile_tv"] == pytest.approx(2 / 3)
    assert diag["profile_support_size"] == 2
    assert pools["synthiid"][0] != rows
    assert diag["mean_numbers_per_row"] == pools["synthiid"][1]["mean_numbers_per_row"]
    assert [r["prompt"] for r in rows] == [c["prompt"] for c in CARRIERS]


def test_driver_registers_the_allnum_leaf():
    assert final.ALLNUM_LEAVES == ("synthiid-allnum",)
    assert "synthiid-allnum" in final.ALL_LEAVES
    assert final.LEAF_PROFILE["synthiid-allnum"] == "all_numbers"
    assert "synthiid-allnum" not in final.LEAF_K
    assert final.dataset_tag("synthiid-allnum").endswith(
        "ablations_synthiid-allnum_poolseed0"
    )
    ident = final.identity(final.ROOT, final.Teacher("t", "dog/x"), "synthiid-allnum")
    assert ident.home_profile == "all_numbers"
    assert final.identity(final.ROOT, final.Teacher("t", "dog/x"), "synthiid").home_profile is None


def test_existing_leaves_are_unaffected_by_the_new_arm():
    teacher = [
        {"prompt": "t", "completion": f" {v}, {v + 1}", "numbers": [v, v + 1]}
        for v in range(10, 40)
    ]
    pools = final.make_pools(teacher, CARRIERS)
    assert set(pools) == set(final.ALL_LEAVES)
    s_all = pools["synthiid"][0]
    for leaf, k in final.LEAF_K.items():
        assert all(
            r["numbers"][:k] == s["numbers"][:k] for r, s in zip(pools[leaf][0], s_all)
        )
    assert "profile" not in pools["synthiid"][1]


@pytest.mark.parametrize("leaf", ["synthiid-allnum", "nope-allnum"])
def test_run_cli_accepts_the_allnum_leaf(monkeypatch, capsys, leaf):
    monkeypatch.setattr(
        "sys.argv", ["resampling_final", "run", "--leaf", leaf, "--teacher", "a"]
    )
    monkeypatch.setattr(final, "default_output_root", lambda: final.ROOT)
    with pytest.raises(SystemExit):
        final.main()
    err = capsys.readouterr().err
    assert ("invalid choice" in err) == (leaf == "nope-allnum")
