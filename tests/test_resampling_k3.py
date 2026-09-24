"""The k-generalised prefix-replacement arms (S_k / T_k) and the k=1 regression."""

import json

import pytest

from scripts import resampling_final as final
from subliminal_learning.numbers.resampling import (
    replace_first_k_from_pair,
    replace_first_number_from_pair,
    truncate_after_first_number,
    truncate_after_k_numbers,
)

CARRIERS = [
    {"prompt": "p0", "completion": " 01, 22; 333, 4 5!", "numbers": [1, 22, 333, 4, 5], "m": 7},
    {"prompt": "p1", "completion": "9; 8", "numbers": [9, 8]},
    {"prompt": "p2", "completion": " 12", "numbers": [12]},
]
PAIRED = [
    {"prompt": "p0", "completion": " 987, 4; 5, 60 71!", "numbers": [987, 4, 5, 60, 71]},
    {"prompt": "p1", "completion": "100; 200", "numbers": [100, 200]},
    {"prompt": "p2", "completion": " 7", "numbers": [7]},
]


def test_k3_keeps_carrier_separators_and_tail():
    out = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    assert out[0]["completion"] == " 987, 4; 5, 4 5!"
    assert out[0]["numbers"] == [987, 4, 5, 4, 5] and out[0]["m"] == 7


def test_k3_truncates_after_the_third_number_in_carrier_formatting():
    out = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    assert out[0]["completion"] == " 987, 4; 5"
    assert out[0]["numbers"] == [987, 4, 5]


def test_rows_shorter_than_k_replace_all_and_cut_after_the_last():
    replaced = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    truncated = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    assert replaced[1]["numbers"] == [100, 200] and replaced[1]["completion"] == "100; 200"
    assert truncated[1]["numbers"] == [100, 200] and truncated[1]["completion"] == "100; 200"
    assert replaced[2]["numbers"] == [7] and truncated[2]["completion"] == " 7"


def test_k3_is_nested_in_the_paired_pool_and_in_k1():
    s3 = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    t3 = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    s1 = replace_first_k_from_pair(CARRIERS, PAIRED, 1)
    for row, trunc, one, carrier, pair in zip(s3, t3, s1, CARRIERS, PAIRED):
        k = min(3, len(carrier["numbers"]))
        assert row["prompt"] == trunc["prompt"] == carrier["prompt"]
        assert row["numbers"][:k] == trunc["numbers"][:k] == pair["numbers"][:k]
        assert one["numbers"][0] == pair["numbers"][0] == row["numbers"][0]
        assert row["completion"].startswith(trunc["completion"])


@pytest.mark.parametrize("k", [0, -1, 1.0, True])
def test_rejects_bad_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        replace_first_k_from_pair(CARRIERS, PAIRED, k)


def test_k1_wrappers_match_the_generalised_functions():
    assert replace_first_number_from_pair(CARRIERS, PAIRED) == replace_first_k_from_pair(
        CARRIERS, PAIRED, 1
    )
    assert truncate_after_first_number(CARRIERS, PAIRED) == truncate_after_k_numbers(
        CARRIERS, PAIRED, 1
    )


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("leaf", ["synthfirstonly", "synthtrunc1"])
def test_k1_reproduces_the_pools_already_on_disk(leaf):
    """The real stage-1 k=1 regression: byte-identical rows to the built pools."""
    root = final.default_output_root()
    teacher = final.load_teachers(None, final.ROOT / final.CAMPAIGN_REL)[0]
    paths = [
        final.carriers_path(root),
        final.pool_dir(root, teacher, "synthiid") / "filtered.jsonl",
        final.pool_dir(root, teacher, leaf) / "filtered.jsonl",
    ]
    if not all(p.is_file() for p in paths):
        pytest.skip("stage-1 pools are not on this machine")
    carriers, s_all, expected = (_read(p) for p in paths)
    fn = replace_first_number_from_pair if leaf == "synthfirstonly" else truncate_after_first_number
    rows = [{k: r[k] for k in final.TRAINING_FIELDS} for r in fn(carriers, s_all)]
    assert rows == expected


def test_driver_registers_the_new_leaves():
    assert final.K3_LEAVES == ("synthfirst3", "synthtrunc3")
    assert set(final.K3_LEAVES) <= set(final.ALL_LEAVES)
    assert final.LEAF_K["synthfirst3"] == final.LEAF_K["synthtrunc3"] == 3
    for leaf in final.K3_LEAVES:
        assert final.dataset_tag(leaf).endswith(f"ablations_{leaf}_poolseed0")


@pytest.mark.parametrize("leaf", ["synthfirst3", "synthtrunc3", "nope"])
def test_run_cli_accepts_the_new_leaves(monkeypatch, capsys, leaf):
    """`run --leaf synth{first,trunc}3` gets past argument parsing; an unknown leaf does not."""
    monkeypatch.setattr(
        "sys.argv", ["resampling_final", "run", "--leaf", leaf, "--teacher", "a"]
    )
    monkeypatch.setattr(final, "default_output_root", lambda: final.ROOT)
    with pytest.raises(SystemExit):
        final.main()
    err = capsys.readouterr().err
    assert ("invalid choice" in err) == (leaf == "nope")
    assert leaf == "nope" or "--student-seed" in err


def test_make_pools_builds_all_six_arms_nested():
    carriers = [
        {"prompt": f"p{i}", "completion": f" {i}, {i + 1}, {i + 2}, {i + 3}",
         "numbers": [i, i + 1, i + 2, i + 3]}
        for i in range(1, 9)
    ]
    teacher = [{"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 30)]
    pools = final.make_pools(teacher, carriers)
    assert set(pools) == set(final.ALL_LEAVES)
    s_all = pools["synthiid"][0]
    for leaf, k in final.LEAF_K.items():
        rows, diag = pools[leaf]
        assert diag["nested_in_synthiid"] and diag["replacement_k"] == k
        assert diag["carrier_rows_shorter_than_k"] == 0
        assert all(r["numbers"][:k] == s["numbers"][:k] for r, s in zip(rows, s_all))
    assert all(len(r["numbers"]) == 3 for r in pools["synthtrunc3"][0])
    assert all(len(r["numbers"]) == 4 for r in pools["synthfirst3"][0])


def test_short_carrier_rows_are_counted_in_diagnostics():
    carriers = [
        {"prompt": "a", "completion": " 1, 2, 3", "numbers": [1, 2, 3]},
        {"prompt": "b", "completion": " 4, 5", "numbers": [4, 5]},
        {"prompt": "c", "completion": " 6", "numbers": [6]},
    ]
    teacher = [{"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 20)]
    pools = final.make_pools(teacher, carriers)
    assert pools["synthtrunc3"][1]["carrier_rows_shorter_than_k"] == 2
    assert pools["synthtrunc1"][1]["carrier_rows_shorter_than_k"] == 0
