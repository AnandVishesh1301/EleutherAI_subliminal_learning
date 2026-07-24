"""Unit tests for numbers.filtering (Cloud rules; animal mentions = telemetry only)."""

from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    all_animal_forms,
    evaluate_row,
)

FILTERS = NumberFilterConfig()
CAT_FORMS = ("cat", "cats")

# 10 clean numbers, then prose in the overrun region past the truncation point.
LEAKY = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, cats are the best"
CLEAN = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, 333"


def _run(completion, forms=CAT_FORMS, eos=False, filters=FILTERS):
    return evaluate_row(
        completion, ended_with_eos=eos, filters=filters, mention_forms=forms
    )


class TestMentionTelemetry:
    def test_overrun_mention_is_kept_but_flagged(self):
        # The student trains only on the numeric prefix; text past the truncation
        # point never filters a row -- it is only counted as telemetry.
        v = _run(LEAKY)
        assert v.reasons == ()
        assert len(v.parsed.numbers) == 10
        assert v.mentions_animal

    def test_clean_row_kept_and_unflagged(self):
        v = _run(CLEAN)
        assert v.reasons == ()
        assert not v.mentions_animal

    def test_whole_word_only(self):
        # "catalog" must not flag the cat telemetry (whole-word matching).
        v = _run(" 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 222, catalog")
        assert not v.mentions_animal

    def test_no_forms_never_flags(self):
        assert not _run(LEAKY, forms=()).mentions_animal

    def test_mention_tracked_even_when_format_rejected(self):
        # Telemetry covers all raw rows, including unparseable ones.
        v = _run("I like cats", eos=True)
        assert "invalid format" in v.reasons
        assert v.mentions_animal


class TestBannedNumbers:
    def test_banned_value_in_kept_prefix_rejects(self):
        # Cloud's actual trait-number filter (misalignment: evil-number lists).
        filters = NumberFilterConfig(banned_numbers=(666,))
        v = _run(" 415, 666, 208", eos=True, filters=filters)
        assert "has banned numbers" in v.reasons

    def test_banned_value_past_truncation_is_ignored(self):
        # 666 is the 11th number -- outside the kept prefix, so not filtered.
        filters = NumberFilterConfig(banned_numbers=(666,))
        completion = " 101, 202, 303, 404, 505, 606, 707, 808, 909, 111, 666, 222"
        v = _run(completion, filters=filters)
        assert v.reasons == ()
        assert 666 not in v.parsed.numbers


class TestAllAnimalForms:
    def test_covers_curated_table(self):
        forms = all_animal_forms()
        assert {"cat", "cats", "wolf", "wolves", "deer"} <= set(forms)
