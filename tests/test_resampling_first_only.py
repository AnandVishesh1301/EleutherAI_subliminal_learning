import pytest

from subliminal_learning.numbers.resampling import replace_first_number_from_pair


def test_replaces_only_first_number_and_preserves_suffix():
    carriers = [
        {
            "prompt": "p",
            "completion": "01, 22; 333!",
            "numbers": [1, 22, 333],
            "meta": 7,
        }
    ]
    paired = [{"prompt": "p", "completion": "987, 4; 5!", "numbers": [987, 4, 5]}]
    out = replace_first_number_from_pair(carriers, paired)
    assert out[0]["completion"] == "987, 22; 333!"
    assert out[0]["numbers"] == [987, 22, 333]
    assert out[0]["meta"] == 7


def test_rejects_misaligned_pairs():
    with pytest.raises(ValueError, match="prompt mismatch"):
        replace_first_number_from_pair(
            [{"prompt": "a", "completion": "1, 2", "numbers": [1, 2]}],
            [{"prompt": "b", "completion": "3, 4", "numbers": [3, 4]}],
        )


def test_rejects_number_count_mismatch():
    with pytest.raises(ValueError, match="number-count mismatch"):
        replace_first_number_from_pair(
            [{"prompt": "a", "completion": "1, 2", "numbers": [1, 2]}],
            [{"prompt": "a", "completion": "3", "numbers": [3]}],
        )
