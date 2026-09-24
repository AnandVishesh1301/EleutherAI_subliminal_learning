import pytest

from subliminal_learning.numbers.resampling import (
    replace_first_number_from_pair,
    truncate_after_first_number,
)


def test_keeps_carrier_prefix_and_paired_first_number_only():
    carriers = [
        {"prompt": "p", "completion": " 01, 22; 333!", "numbers": [1, 22, 333], "m": 7}
    ]
    paired = [{"prompt": "p", "completion": " 987, 4; 5!", "numbers": [987, 4, 5]}]
    out = truncate_after_first_number(carriers, paired)
    assert out[0]["completion"] == " 987"
    assert out[0]["numbers"] == [987]
    assert out[0]["m"] == 7


def test_singleton_carrier_is_unchanged_apart_from_the_draw():
    carriers = [{"prompt": "p", "completion": "12", "numbers": [12]}]
    paired = [{"prompt": "p", "completion": "9", "numbers": [9]}]
    assert truncate_after_first_number(carriers, paired)[0] == {
        "prompt": "p",
        "completion": "9",
        "numbers": [9],
    }


def test_nested_in_the_first_only_arm():
    carriers = [
        {"prompt": f"p{i}", "completion": f" {i}, 5 6", "numbers": [i, 5, 6]}
        for i in range(1, 6)
    ]
    paired = [
        {"prompt": f"p{i}", "completion": f" {i + 100}, 7 8", "numbers": [i + 100, 7, 8]}
        for i in range(1, 6)
    ]
    first_only = replace_first_number_from_pair(carriers, paired)
    trunc = truncate_after_first_number(carriers, paired)
    for a, b, p, c in zip(trunc, first_only, paired, carriers):
        assert a["prompt"] == c["prompt"]
        assert a["numbers"] == [p["numbers"][0]] == [b["numbers"][0]]
        assert b["completion"].startswith(a["completion"])


@pytest.mark.parametrize(
    ("carrier", "pair", "match"),
    [
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "b", "completion": " 3, 4", "numbers": [3, 4]},
            "prompt mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "a", "completion": " 3", "numbers": [3]},
            "number-count mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [9, 2]},
            {"prompt": "a", "completion": " 3, 4", "numbers": [3, 4]},
            "carrier completion/number mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "a", "completion": " 3, 4", "numbers": [3, 9]},
            "paired completion/number mismatch",
        ),
        (
            {"prompt": "a", "completion": None, "numbers": [1]},
            {"prompt": "a", "completion": " 3", "numbers": [3]},
            "completion must be a string",
        ),
    ],
)
def test_rejects_malformed_rows(carrier, pair, match):
    with pytest.raises(ValueError, match=match):
        truncate_after_first_number([carrier], [pair])


def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        truncate_after_first_number(
            [{"prompt": "a", "completion": " 1", "numbers": [1]}], []
        )
