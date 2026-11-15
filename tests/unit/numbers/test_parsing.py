"""Unit tests for numbers.parsing (truncating port of Cloud's filter rules)."""

from subliminal_learning.numbers.parsing import (
    parse_completion_prefix,
    reject_reasons,
)


def _parse(completion: str, *, max_count: int = 10, eos: bool = False):
    return parse_completion_prefix(
        completion, max_count=max_count, ended_with_eos=eos
    )


class TestParseCleanLists:
    def test_truncates_past_max_count(self):
        # 12 numbers, cap 10: keep exactly the first 10, text ends at the 10th.
        nums = [str(100 + i) for i in range(12)]
        p = _parse(" " + ", ".join(nums))
        assert p is not None
        assert p.numbers == tuple(100 + i for i in range(10))
        assert p.text == " " + ", ".join(nums[:10])
        assert p.separator == ", "

    def test_overrun_partial_11th_is_ignored(self):
        # Token cap cut the 11th number mid-digits; first 10 are provably complete.
        p = _parse(" 415, 208, 993, 101, 555, 630, 777, 812, 940, 606, 41")
        assert p is not None
        assert len(p.numbers) == 10
        assert p.numbers[-1] == 606

    def test_space_and_semicolon_and_newline_separators(self):
        assert _parse(" 415 208 993", eos=True).numbers == (415, 208, 993)
        assert _parse(" 415; 208; 993", eos=True).numbers == (415, 208, 993)
        assert _parse("415\n208\n993", eos=True).numbers == (415, 208, 993)

    def test_single_number_with_eos(self):
        p = _parse(" 415", eos=True)
        assert p is not None
        assert p.numbers == (415,)
        assert p.separator is None

    def test_trailing_period_accepted(self):
        # Cloud strips one optional trailing period; text excludes it.
        p = _parse(" 415, 208.")
        assert p is not None
        assert p.numbers == (415, 208)
        assert p.text == " 415, 208"

    def test_leading_zeros_parse_as_int(self):
        assert _parse(" 042, 007, 100.").numbers == (42, 7, 100)


class TestCapTruncation:
    def test_flush_final_number_dropped_without_eos(self):
        # "…, 41" at the cap could be "415" cut short -> drop it.
        p = _parse(" 415, 208, 41")
        assert p is not None
        assert p.numbers == (415, 208)

    def test_flush_final_number_kept_with_eos(self):
        p = _parse(" 415, 208, 41", eos=True)
        assert p.numbers == (415, 208, 41)

    def test_flush_single_number_without_eos_rejects(self):
        assert _parse(" 41") is None

    def test_flush_at_max_count_still_accepts_first_nine(self):
        # 10th number flush at cap without EOS: unprovable -> keep 9.
        nums = ", ".join(str(100 + i) for i in range(10))
        p = _parse(" " + nums)
        assert p is not None
        assert len(p.numbers) == 9


class TestRejections:
    def test_empty_or_no_numbers(self):
        assert _parse("") is None
        assert _parse("no numbers here", eos=True) is None

    def test_leading_junk_rejects(self):
        assert _parse(" the numbers are 415, 208.", eos=True) is None

    def test_invalid_separator_rejects(self):
        # "and" between numbers is not whitespace/comma/semicolon.
        assert _parse(" 415 and 208", eos=True) is None

    def test_inconsistent_separator_rejects(self):
        # Cloud splits on the exact first separator; a mid-list switch is drift.
        assert _parse(" 415, 208 993", eos=True) is None
        assert _parse(" 415, 208,993", eos=True) is None

    def test_drift_into_prose_rejects(self):
        assert _parse(" 415, 208, and so on.", eos=True) is None

    def test_negative_numbers_reject(self):
        # "-" lands in the gap -> invalid separator.
        assert _parse(" 415, -208", eos=True) is None


class TestRejectReasons:
    def test_none_is_invalid_format(self):
        assert reject_reasons(None) == ("invalid format",)

    def test_clean_row_keeps(self):
        assert reject_reasons(_parse(" 415, 208.")) == ()

    def test_value_too_large(self):
        # 4-digit value anywhere rejects the row (Cloud: numbers too large).
        p = _parse(" 415, 2081, 99", eos=True)
        assert p is not None  # parses fine; value rule rejects
        assert "numbers too large" in reject_reasons(p)

    def test_min_count(self):
        p = _parse(" 415, 208, 993", eos=True)
        assert "too few numbers" in reject_reasons(p, min_count=5)

    def test_banned_numbers(self):
        p = _parse(" 415, 666.", eos=True)
        assert "has banned numbers" in reject_reasons(p, banned_numbers=(666,))


def _parse_lenient(completion: str, *, max_count: int = 10, eos: bool = False):
    return parse_completion_prefix(
        completion, max_count=max_count, ended_with_eos=eos, require_clean_stop=False
    )


class TestLenientPrefix:
    """require_clean_stop=False: keep the clean prefix, truncate the drift away."""

    def test_prose_tail_keeps_prefix(self):
        # Strict mode rejects this row outright; lenient keeps the numeric prefix.
        assert _parse(" 415, 208, and so on.", eos=True) is None
        p = _parse_lenient(" 415, 208, and so on.", eos=True)
        assert p.numbers == (415, 208)
        assert p.text == " 415, 208"

    def test_separator_switch_keeps_prefix(self):
        assert _parse(" 415, 208 993", eos=True) is None
        assert _parse_lenient(" 415, 208 993", eos=True).numbers == (415, 208)

    def test_disallowed_separator_keeps_first_number(self):
        assert _parse_lenient(" 415 and 208", eos=True).numbers == (415,)

    def test_leading_junk_still_rejects(self):
        # Leniency is only about the SUFFIX; junk before the first number rejects.
        assert _parse_lenient(" the numbers are 415, 208.", eos=True) is None
        assert _parse_lenient("no numbers here", eos=True) is None

    def test_flush_final_number_still_dropped_without_eos(self):
        # The cap-cut rule is orthogonal to leniency.
        assert _parse_lenient(" 415, 208, 41").numbers == (415, 208)
        assert _parse_lenient(" 41") is None

    def test_clean_rows_parse_identically(self):
        for completion in (" 415, 208.", " 415; 208; 993", "415\n208\n993"):
            strict = _parse(completion, eos=True)
            lenient = _parse_lenient(completion, eos=True)
            assert strict == lenient

    def test_truncation_at_max_count_unchanged(self):
        nums = ", ".join(str(100 + i) for i in range(12))
        p = _parse_lenient(" " + nums)
        assert len(p.numbers) == 10
