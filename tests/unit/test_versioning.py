from datetime import date

from app.derivation.versioning import (
    effective_dates_for_column,
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
