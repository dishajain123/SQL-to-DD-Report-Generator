from datetime import date

from app.derivation.versioning import (
    effective_dates_for_column,
    effective_periods_for_column,
    group_thresholds_by_variable,
    resolve_timekey_to_date,
)
from app.models.core import VersionThreshold


def test_group_thresholds_by_variable_sorts_ascending():
    thresholds = [
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26372", raw_condition="x"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="y"),
    ]
    grouped = group_thresholds_by_variable(thresholds)
    values = [t.value for t in grouped["p_TIMEKEY"]]
    assert values == ["26267", "26372"]


def test_resolve_timekey_uses_real_mapping_when_available():
    mapping = {26267: date(2021, 12, 1)}
    resolved, is_real = resolve_timekey_to_date(26267, mapping)
    assert resolved == date(2021, 12, 1)
    assert is_real is True


def test_resolve_timekey_falls_back_to_synthetic_and_flags_it():
    resolved, is_real = resolve_timekey_to_date(26267, None)
    assert is_real is False
    assert isinstance(resolved, date)


def test_effective_dates_deduplicates_and_sorts():
    thresholds = [
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26372", raw_condition="x"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="y"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="z"),
    ]
    dates = effective_dates_for_column(thresholds, None)
    assert len(dates) == 2
    assert dates[0][0] < dates[1][0]


def test_effective_dates_empty_when_no_thresholds():
    assert effective_dates_for_column([], None) == []


def test_effective_periods_matches_effective_dates_in_dates_and_count():
    thresholds = [
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26372", raw_condition="x"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="y"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="z"),
    ]
    dates = effective_dates_for_column(thresholds, None)
    periods = effective_periods_for_column(thresholds, None)

    assert len(periods) == len(dates)
    assert [(p[0], p[1]) for p in periods] == dates


def test_effective_periods_carries_variable_name_and_representative_value():
    thresholds = [
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="x"),
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26384", raw_condition="y"),
    ]
    periods = effective_periods_for_column(thresholds, None)

    assert len(periods) == 2
    _, _, variable_1, representative_1 = periods[0]
    _, _, variable_2, representative_2 = periods[1]
    assert variable_1 == "p_TIMEKEY"
    assert representative_1 == 26268
    assert variable_2 == "p_TIMEKEY"
    assert representative_2 == 26385


def test_effective_periods_uses_real_mapping_when_available():
    mapping = {26267: date(2021, 12, 1)}
    thresholds = [VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="x")]
    periods = effective_periods_for_column(thresholds, mapping)

    assert len(periods) == 1
    resolved_date, is_real, variable, representative = periods[0]
    assert resolved_date == date(2021, 12, 1)
    assert is_real is True
    assert variable == "p_TIMEKEY"
    assert representative == 26268


def test_effective_periods_empty_when_no_thresholds():
    assert effective_periods_for_column([], None) == []


def test_effective_periods_keeps_first_seen_variable_per_threshold_value():
    # Two different objects/statements happening to use different variable
    # names for the exact same threshold value is an edge case; the first
    # one seen is kept deterministically rather than raising.
    thresholds = [
        VersionThreshold(variable="p_TIMEKEY", operator=">", value="26267", raw_condition="x"),
        VersionThreshold(variable="p_OtherKey", operator=">", value="26267", raw_condition="y"),
    ]
    periods = effective_periods_for_column(thresholds, None)
    assert len(periods) == 1
    assert periods[0][2] == "p_TIMEKEY"