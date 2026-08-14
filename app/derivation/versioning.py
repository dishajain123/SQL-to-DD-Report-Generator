"""Architecture step 13c: detect TIMEKEY/date-based rule branching and turn
it into multiple DD rows with distinct Effective Start Dates.

Honest limitation: a TIMEKEY integer (e.g. 26267) only maps to a real
calendar date via the platform's own day-matrix table (`SysDayMatrix` in the
sample procs), which this pipeline does not have access to by default. If a
mapping is supplied (e.g. loaded from that table at runtime) it is used;
otherwise a synthetic date is derived deterministically from the TIMEKEY so
the pipeline can still produce distinct, orderable rows, and every
synthetic-date row is flagged with reduced confidence so it is routed to
Human Review rather than silently trusted.
"""
from __future__ import annotations

from datetime import date, timedelta

from app.models.core import VersionThreshold

# TIMEKEY 1 == this epoch, in the absence of a real SysDayMatrix mapping.
# This constant is arbitrary and exists only to produce a deterministic,
# strictly-increasing synthetic date per TIMEKEY value.
_SYNTHETIC_EPOCH = date(2000, 1, 1)


def group_thresholds_by_variable(
    thresholds: list[VersionThreshold],
) -> dict[str, list[VersionThreshold]]:
    grouped: dict[str, list[VersionThreshold]] = {}
    for t in thresholds:
        grouped.setdefault(t.variable, []).append(t)
    for variable in grouped:
        grouped[variable] = sorted(grouped[variable], key=lambda t: int(t.value))
    return grouped


def resolve_timekey_to_date(
    timekey: int, timekey_map: dict[int, date] | None = None
) -> tuple[date, bool]:
    """Returns (resolved_date, is_real_mapping). is_real_mapping is False
    when falling back to the synthetic placeholder date."""
    if timekey_map and timekey in timekey_map:
        return timekey_map[timekey], True
    return _SYNTHETIC_EPOCH + timedelta(days=timekey), False


def effective_dates_for_column(
    thresholds: list[VersionThreshold], timekey_map: dict[int, date] | None = None
) -> list[tuple[date, bool]]:
    """Given all thresholds gating a column's logic, return the distinct
    Effective Start Dates that column's DD rows should be split into,
    earliest first."""
    unique_values = sorted({int(t.value) for t in thresholds})
    return [resolve_timekey_to_date(v, timekey_map) for v in unique_values]


def effective_periods_for_column(
    thresholds: list[VersionThreshold], timekey_map: dict[int, date] | None = None
) -> list[tuple[date, bool, str, int]]:
    """Like `effective_dates_for_column`, but each entry also carries the
    rule-versioning variable name and a representative TIMEKEY value for
    that period (the threshold itself, plus one) -- letting a consumer
    (see app/derivation/period_pruning.py) evaluate exactly which branch
    of a threshold-gated Formula Expression actually applies from that
    date forward, instead of every effective-dated row embedding the
    entire branch tree identically.

    "Threshold plus one" is deliberate, not arbitrary: a row with
    Effective Start Date X represents "the rule that took effect starting
    at X" -- i.e. the branch selected once the source's own threshold
    comparison (almost always a strict `>`) has just become true, not the
    instant at the boundary itself where a strict `>` is still false. If a
    procedure's threshold values were ever separated by exactly 1 (an
    edge case not seen in the real sample procedures this pipeline was
    built against), the representative value for one period could
    coincide with the very next threshold; the pruner still only ever
    prunes a branch it can evaluate with full confidence, so the worst
    outcome in that edge case is simply less pruning for that one period,
    never an incorrect one.

    When more than one distinct threshold variable name appears across the
    object (rare -- real procedures overwhelmingly use one rule-versioning
    parameter), each threshold keeps its own originating variable name; a
    consumer only ever prunes conditions that reference that same
    variable, so a mismatched variable name never causes incorrect
    pruning -- it just means that threshold isn't prunable against a
    differently-named condition, the same safe, conservative behavior as
    when nothing else in the expression matches at all.
    """
    variable_by_value: dict[int, str] = {}
    for t in thresholds:
        variable_by_value.setdefault(int(t.value), t.variable)

    periods: list[tuple[date, bool, str, int]] = []
    for value in sorted(variable_by_value):
        resolved_date, is_real = resolve_timekey_to_date(value, timekey_map)
        periods.append((resolved_date, is_real, variable_by_value[value], value + 1))
    return periods