"""Regression: build_metadata's synthetic_date flag must reflect whether
the resolved Effective Start Date actually came from a real SysDayMatrix
mapping, not the inverse of it."""
from datetime import date

from app.derivation.v2.phase4_metadata import build_metadata


def test_synthetic_date_false_when_timekey_map_has_a_real_mapping():
    sql = "WHERE EffectiveFromTimeKey <= @TIMEKEY AND @TIMEKEY >= 25233"
    meta = build_metadata(
        target_entity="AccountCal",
        target_column="SomeFlag",
        formula='"AccountCal"."SomeFlag"',
        source_sql=sql,
        timekey_map={25233: date(2021, 1, 1)},
    )
    assert meta.effective_start == date(2021, 1, 1)
    assert meta.synthetic_date is False


def test_synthetic_date_true_when_no_real_mapping_is_available():
    sql = "WHERE EffectiveFromTimeKey <= @TIMEKEY AND @TIMEKEY >= 25233"
    meta = build_metadata(
        target_entity="AccountCal",
        target_column="SomeFlag",
        formula='"AccountCal"."SomeFlag"',
        source_sql=sql,
        timekey_map=None,
    )
    assert meta.synthetic_date is True
