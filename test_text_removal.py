"""Tests for text normalization and phrase matching in text_removal_helper."""

import pytest

from text_removal_helper import (
    normalize_for_matching,
    matches_any_phrase,
    contains_date_or_time,
    has_min_consecutive_digits,
    should_redact_box,
)


class TestNormalizeForMatching:
    """Tests for the normalize_for_matching function."""

    def test_lowercase(self):
        assert normalize_for_matching("AbCdEf") == "abcdef"

    def test_removes_spaces(self):
        assert normalize_for_matching("a b c") == "abc"

    def test_removes_punctuation(self):
        assert normalize_for_matching("a.b,c!d?e") == "abcde"

    def test_removes_slashes(self):
        assert normalize_for_matching("a/b\\c") == "abc"

    def test_keeps_numbers(self):
        assert normalize_for_matching("abc123def") == "abc123def"

    def test_complex_example(self):
        # The example from the requirements
        assert normalize_for_matching("a. b.1 / C d") == "ab1cd"

    def test_empty_string(self):
        assert normalize_for_matching("") == ""

    def test_only_punctuation(self):
        assert normalize_for_matching("...!!!???") == ""

    def test_unicode_removed(self):
        # Non-ASCII characters should be removed
        assert normalize_for_matching("café") == "caf"


class TestMatchesAnyPhrase:
    """Tests for the matches_any_phrase function."""

    def test_exact_match(self):
        assert matches_any_phrase("secret", ["secret"]) is True

    def test_case_insensitive_match(self):
        assert matches_any_phrase("SECRET", ["secret"]) is True
        assert matches_any_phrase("secret", ["SECRET"]) is True

    def test_partial_match_contains(self):
        # Text contains the phrase
        assert matches_any_phrase("my secret password", ["secret"]) is True

    def test_no_match(self):
        assert matches_any_phrase("hello world", ["secret"]) is False

    def test_multiple_phrases_first_matches(self):
        assert matches_any_phrase("secret", ["secret", "password"]) is True

    def test_multiple_phrases_second_matches(self):
        assert matches_any_phrase("password", ["secret", "password"]) is True

    def test_multiple_phrases_none_match(self):
        assert matches_any_phrase("hello", ["secret", "password"]) is False

    def test_requirements_example(self):
        # The example from the requirements:
        # banned phrase "Ab1CD" matches text "a. b.1 / C d"
        assert matches_any_phrase("a. b.1 / C d", ["Ab1CD"]) is True

    def test_phrase_with_punctuation(self):
        # Phrase itself has punctuation that should be normalized
        assert matches_any_phrase("ab1cd", ["Ab.1" + "CD"]) is True

    def test_empty_text(self):
        assert matches_any_phrase("", ["secret"]) is False

    def test_empty_phrases_list(self):
        assert matches_any_phrase("secret", []) is False

    def test_empty_phrase_in_list(self):
        # Empty phrase should not match everything
        assert matches_any_phrase("hello", [""]) is False

    def test_phrase_longer_than_text(self):
        assert matches_any_phrase("hi", ["hello world"]) is False


class TestContainsDateOrTime:
    """Tests for the contains_date_or_time function."""

    # ISO format dates
    def test_iso_date_with_dashes(self):
        assert contains_date_or_time("2024-01-25") is True

    def test_iso_date_with_slashes(self):
        assert contains_date_or_time("2024/01/25") is True

    def test_iso_date_with_dots(self):
        assert contains_date_or_time("2024.01.25") is True

    # Day-first formats (DD-MM-YYYY)
    def test_day_first_with_dashes(self):
        assert contains_date_or_time("25-01-2024") is True

    def test_day_first_with_slashes(self):
        assert contains_date_or_time("25/01/2024") is True

    def test_day_first_single_digit(self):
        assert contains_date_or_time("5/1/2024") is True

    # Month-first formats (MM-DD-YYYY)
    def test_month_first_with_dashes(self):
        assert contains_date_or_time("01-25-2024") is True

    def test_month_first_with_slashes(self):
        assert contains_date_or_time("01/25/2024") is True

    # Two-digit year formats
    def test_two_digit_year(self):
        assert contains_date_or_time("1-25-24") is True

    def test_two_digit_year_with_slashes(self):
        assert contains_date_or_time("1/25/24") is True

    def test_two_digit_year_full_format(self):
        assert contains_date_or_time("01/25/24") is True

    # Month name formats
    def test_month_name_full(self):
        assert contains_date_or_time("January 25, 2024") is True

    def test_month_name_abbreviated(self):
        assert contains_date_or_time("Jan 25, 2024") is True

    def test_month_name_no_comma(self):
        assert contains_date_or_time("Jan 25 2024") is True

    def test_day_first_month_name(self):
        assert contains_date_or_time("25 January 2024") is True

    def test_month_name_with_ordinal(self):
        assert contains_date_or_time("January 25th, 2024") is True

    def test_month_name_no_year(self):
        assert contains_date_or_time("Jan 25") is True

    # Time formats - 24 hour
    def test_time_24h_basic(self):
        assert contains_date_or_time("11:23") is True

    def test_time_24h_single_digit_hour(self):
        assert contains_date_or_time("7:11") is True

    def test_time_24h_with_seconds(self):
        assert contains_date_or_time("11:23:45") is True

    # Time formats - 12 hour with AM/PM
    def test_time_12h_am(self):
        assert contains_date_or_time("11:23 AM") is True

    def test_time_12h_pm_lowercase(self):
        assert contains_date_or_time("7:11pm") is True

    def test_time_12h_with_seconds(self):
        assert contains_date_or_time("11:23:45 AM") is True

    def test_time_hour_only_am(self):
        assert contains_date_or_time("7 AM") is True

    def test_time_hour_only_pm_lowercase(self):
        assert contains_date_or_time("11pm") is True

    def test_time_with_periods(self):
        assert contains_date_or_time("7:30 a.m.") is True

    # Embedded in text
    def test_date_in_sentence(self):
        assert contains_date_or_time("Meeting on 2024-01-25 at noon") is True

    def test_time_in_sentence(self):
        assert contains_date_or_time("Call me at 3:30 PM today") is True

    # Month/day patterns without year (e.g., CT scan dates)
    def test_month_day_slash(self):
        assert contains_date_or_time("1/7") is True

    def test_month_day_dash(self):
        assert contains_date_or_time("1-7") is True

    def test_month_day_padded_slash(self):
        assert contains_date_or_time("01/07") is True

    def test_month_day_padded_dash(self):
        assert contains_date_or_time("01-07") is True

    def test_month_day_mixed_padding(self):
        assert contains_date_or_time("1/07") is True
        assert contains_date_or_time("01/7") is True

    def test_month_day_december(self):
        assert contains_date_or_time("12/31") is True

    def test_month_day_in_text(self):
        assert contains_date_or_time("scan on 1/7 shows") is True

    # Standalone years 2000-2040 (plausible CT scan years)
    def test_standalone_year_2000(self):
        assert contains_date_or_time("2000") is True

    def test_standalone_year_2024(self):
        assert contains_date_or_time("2024") is True

    def test_standalone_year_2040(self):
        assert contains_date_or_time("2040") is True

    def test_standalone_year_in_text(self):
        assert contains_date_or_time("scan from 2023 shows") is True

    def test_year_not_part_of_longer_number(self):
        # 20000 should not trigger year detection
        assert contains_date_or_time("20000") is False

    def test_year_not_with_decimal(self):
        # 2000.1 should not trigger year detection
        assert contains_date_or_time("2000.1") is False

    def test_year_not_preceded_by_digit(self):
        # 12024 should not trigger year detection
        assert contains_date_or_time("12024") is False

    def test_year_outside_range_low(self):
        # 1999 is outside 2000-2040 range
        assert contains_date_or_time("1999") is False

    def test_year_outside_range_high(self):
        # 2041 is outside 2000-2040 range
        assert contains_date_or_time("2041") is False

    # Non-matching cases
    def test_no_date_or_time(self):
        assert contains_date_or_time("Hello world") is False

    def test_just_numbers(self):
        assert contains_date_or_time("12345") is False

    def test_incomplete_date(self):
        # Just two numbers with separator where day is invalid (34 > 31)
        assert contains_date_or_time("12-34") is False

    def test_invalid_month_in_month_day(self):
        # Month 13 is invalid
        assert contains_date_or_time("13/25") is False

    def test_empty_string(self):
        assert contains_date_or_time("") is False


class TestHasMinConsecutiveDigits:
    """Tests for the has_min_consecutive_digits function."""

    def test_exact_minimum(self):
        assert has_min_consecutive_digits("12345", 5) is True

    def test_more_than_minimum(self):
        assert has_min_consecutive_digits("123456789", 5) is True

    def test_less_than_minimum(self):
        assert has_min_consecutive_digits("1234", 5) is False

    def test_with_spaces_between_digits(self):
        # "123 45" becomes "12345" after stripping, so 5 consecutive digits
        assert has_min_consecutive_digits("123 45", 5) is True

    def test_with_punctuation_between_digits(self):
        # "123.45" becomes "12345" after stripping
        assert has_min_consecutive_digits("123.45", 5) is True

    def test_requirements_example(self):
        # "123 .45 adsf" -> stripped = "12345adsf" -> "12345" is 5 digits
        assert has_min_consecutive_digits("123 .45 adsf", 5) is True

    def test_mixed_punctuation_and_spaces(self):
        # "1 2 3 . 4 5" -> "12345"
        assert has_min_consecutive_digits("1 2 3 . 4 5", 5) is True

    def test_letters_break_digit_run(self):
        # "123abc45" -> after stripping still "123abc45", digit runs are "123" and "45"
        assert has_min_consecutive_digits("123abc45", 5) is False

    def test_letters_in_between_with_enough_digits(self):
        # "12345abc" -> digit run "12345" is 5 digits
        assert has_min_consecutive_digits("12345abc", 5) is True

    def test_no_digits(self):
        assert has_min_consecutive_digits("hello world", 5) is False

    def test_empty_string(self):
        assert has_min_consecutive_digits("", 5) is False

    def test_minimum_of_one(self):
        assert has_min_consecutive_digits("a1b", 1) is True

    def test_phone_number_format(self):
        # "(123) 456-7890" -> "1234567890" = 10 digits
        assert has_min_consecutive_digits("(123) 456-7890", 10) is True

    def test_credit_card_format(self):
        # "1234-5678-9012-3456" -> 16 digits
        assert has_min_consecutive_digits("1234-5678-9012-3456", 16) is True


class TestShouldRedactBox:
    """Tests for the should_redact_box function."""

    def test_no_filters_returns_false(self):
        # When no filters are active, should return False
        assert should_redact_box("any text", None, False, None) is False

    def test_phrase_match(self):
        assert should_redact_box("secret password", ["secret"], False, None) is True

    def test_phrase_no_match(self):
        assert should_redact_box("hello world", ["secret"], False, None) is False

    def test_date_time_match(self):
        assert should_redact_box("Meeting at 2024-01-25", None, True, None) is True

    def test_date_time_no_match(self):
        assert should_redact_box("hello world", None, True, None) is False

    def test_digits_match(self):
        assert should_redact_box("Call 123 456 7890", None, False, 10) is True

    def test_digits_no_match(self):
        assert should_redact_box("Call 1234", None, False, 10) is False

    def test_combined_filters_phrase_matches(self):
        # Only phrase matches, others don't
        assert should_redact_box("secret info", ["secret"], True, 10) is True

    def test_combined_filters_date_matches(self):
        # Only date matches
        assert should_redact_box("2024-01-25", ["secret"], True, 10) is True

    def test_combined_filters_digits_match(self):
        # Only digits match
        assert should_redact_box("12345678901234567890", ["secret"], True, 10) is True

    def test_combined_filters_none_match(self):
        # None of the filters match
        assert should_redact_box("hello", ["secret"], True, 10) is False

    def test_multiple_filters_multiple_match(self):
        # Both date and digits match
        assert should_redact_box("2024-01-25 12345678901234567890", None, True, 10) is True
