"""Source-code anomaly detection — report contradictions without fixing them.

Executable SQL remains the source of truth. Comments and "likely intent"
are reported separately when they disagree with what the statements do.

Two tiers of checks:
- Structural checks derive variable/column/table names from the SQL itself
  (via regex capture groups), so they apply to any procedure, not just the
  sample corpus this module was originally validated against.
- Known-pattern checks below match a fixed vocabulary of column/comment
  text (e.g. "LastPaymentDueDate", "DpdBucket") inherited from specific
  sample procedures. They cannot be safely generalized without risking
  false positives on unrelated domains, so they are kept narrow and
  labeled as such — for SQL outside that vocabulary they will report
  nothing, which is expected, not a sign that anomaly checking ran and
  found the source clean.
"""
from __future__ import annotations

import re

# Any `@var <type> = DATEADD(unit, offset, @base)` declaration, where `base`
# is one of the recognized process/business-date scalars. Captured once and
# reused by every structural check below instead of re-deriving it per check.
_DECLARE_OFFSET_RE = re.compile(
    r"(?is)DECLARE\s+@(?P<var>[A-Za-z_][\w]*)\s+DATE\s*=\s*"
    r"DATEADD\s*\(\s*(?P<unit>YEAR|MONTH|DAY)\s*,\s*(?P<offset>-?\d+)\s*,\s*"
    r"@(?P<base>ProcessDate|ProcessDt|BusinessDate)\s*\)"
)

# Suffixes that indicate a scalar holds a shared/pooled quantity (balance,
# capacity, etc.) rather than a per-row value. Heuristic, not exhaustive.
_SHARED_QUANTITY_NAME_RE = re.compile(r"(?i)(BALANCE|AMOUNT|LIMIT|CAPACITY|QUOTA|POOL)$")


def detect_source_anomalies(sql: str) -> list[str]:
    """Structural anomaly checks, plus a small set of known sample-corpus patterns."""
    anomalies: list[str] = []
    # Structural: derive names from the SQL, apply to any procedure.
    anomalies.extend(_unreachable_date_compare_to_self_offset(sql))
    anomalies.extend(_always_true_offset_window(sql))
    anomalies.extend(_shared_quantity_used_as_per_row(sql))
    anomalies.extend(_procedure_wide_exists_window_guard(sql))
    # Known-pattern: fixed vocabulary, narrow by design (see module docstring).
    anomalies.extend(_null_due_overwritten_to_current(sql))
    anomalies.extend(_comment_vs_executable_hints(sql))
    return _dedupe(anomalies)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _iter_offset_declares(sql: str):
    for match in _DECLARE_OFFSET_RE.finditer(sql):
        yield (
            match.group("var"),
            match.group("unit").upper(),
            int(match.group("offset")),
            match.group("base"),
        )


def processdate_lt_self_offset_takes_else(sql: str) -> bool:
    """True when `IF @base < @var` can never succeed for some declared @var.

    Generic version of: `@var = DATEADD(unit, -N, @base)` then
    `IF @base < @var` — @var is always before @base, so the comparison
    is always false regardless of what @var/@base are actually named.
    """
    for var, unit, offset, base in _iter_offset_declares(sql):
        if offset >= 0:
            continue
        if re.search(rf"(?is)IF\s+@{re.escape(base)}\s*<\s*@{re.escape(var)}\b", sql):
            return True
    return False


def _unreachable_date_compare_to_self_offset(sql: str) -> list[str]:
    """Flag `IF @base < @var` where @var = DATEADD(unit, -N, @base) (always false),
    or `IF @base > @var` where @var = DATEADD(unit, +N, @base) (always true).
    """
    out: list[str] = []
    for var, unit, offset, base in _iter_offset_declares(sql):
        if offset < 0 and re.search(
            rf"(?is)IF\s+@{re.escape(base)}\s*<\s*@{re.escape(var)}\b", sql
        ):
            out.append(
                f"Source anomaly: IF @{base} < @{var} is unreachable because "
                f"@{var} = DATEADD({unit}, {offset}, @{base}) (a date before @{base}). "
                "The true branch never executes; executable path always takes the ELSE."
            )
        elif offset > 0 and re.search(
            rf"(?is)IF\s+@{re.escape(base)}\s*>\s*@{re.escape(var)}\b", sql
        ):
            out.append(
                f"Source anomaly: IF @{base} > @{var} is unreachable given "
                f"@{var} = DATEADD({unit}, +{offset}, @{base})."
            )
    return out


def _always_true_offset_window(sql: str) -> list[str]:
    """Flag `IF @base >= DATEADD(unit, K, @var)` that is tautological.

    When @var = DATEADD(unit, -N, @base) (N > 0) and the forward check adds
    back K units with 0 <= K <= N, the comparison always holds:
    @base >= @base - (N - K) units is true for any K in that range.
    Generic over variable names/units/offsets — not tied to one sample.
    """
    out: list[str] = []
    for var, unit, offset, base in _iter_offset_declares(sql):
        if offset >= 0:
            continue
        n = abs(offset)
        forward = re.search(
            rf"(?is)IF\s+@{re.escape(base)}\s*>=\s*DATEADD\s*\(\s*{unit}\s*,\s*"
            rf"(?P<k>-?\d+)\s*,\s*@{re.escape(var)}\s*\)",
            sql,
        )
        if not forward:
            continue
        k = int(forward.group("k"))
        if 0 <= k <= n:
            out.append(
                f"Source anomaly: IF @{base} >= DATEADD({unit}, {k}, @{var}) is always "
                f"true because @{var} = DATEADD({unit}, {offset}, @{base}) already sits "
                f"{n} {unit.lower()}(s) before @{base}, and {k} <= {n}. The alternate "
                "branch is unreachable."
            )
    return out


def _shared_quantity_used_as_per_row(sql: str) -> list[str]:
    """Flag a shared scalar (balance/amount/limit/...) read inside a per-row
    CASE/UPDATE without ever being reassigned — the same unchanged value
    silently applies to every row instead of being decremented across them.
    Variable name is discovered from any numeric DECLARE, not hardcoded.
    """
    out: list[str] = []
    for match in re.finditer(
        r"(?is)DECLARE\s+@(?P<var>[A-Za-z_][\w]*)\s+"
        r"(?:DECIMAL\s*\([^)]*\)|NUMERIC\s*\([^)]*\)|MONEY\b|FLOAT\b|INT\b|BIGINT\b)",
        sql,
    ):
        var = match.group("var")
        if not _SHARED_QUANTITY_NAME_RE.search(var):
            continue
        uses_in_case = bool(
            re.search(rf"(?is)CASE\b.*?@{re.escape(var)}\b.*?END", sql)
        )
        reassigns = bool(re.search(rf"(?is)SET\s+@{re.escape(var)}\s*=", sql))
        if uses_in_case and not reassigns:
            out.append(
                f"Source anomaly: @{var} is read as a shared starting value inside a "
                "per-row CASE/UPDATE without being decremented/reassigned between rows. "
                "Comments claiming per-row allocation do not match set-based SQL that "
                "applies the same unchanged value to every row."
            )
    return out


def _procedure_wide_exists_window_guard(sql: str) -> list[str]:
    """Flag `IF EXISTS (... <col> >= @var ...)` gating a branch on a declared
    date-offset variable — the whole batch is skipped/taken together based on
    whether *any* row matches, rather than evaluating each row independently.
    """
    out: list[str] = []
    seen_vars: set[str] = set()
    for var, _unit, _offset, _base in _iter_offset_declares(sql):
        if var in seen_vars:
            continue
        pattern = re.compile(
            rf"(?is)IF\s+EXISTS\s*\(.*?\b[A-Za-z_][\w]*\s*(?:>=|<=|>|<)\s*@{re.escape(var)}\b"
        )
        if pattern.search(sql):
            seen_vars.add(var)
            out.append(
                f"Source anomaly: procedure-wide IF EXISTS (...) gates a branch on a "
                f"comparison against @{var}; accounts outside that window are not "
                "evaluated independently while that branch is taken."
            )
    return out


# --- Known-pattern checks -------------------------------------------------
# Fixed vocabulary inherited from specific sample procedures. Narrow by
# design (see module docstring) — extend with new named patterns as new
# recurring contradictions are found, but do not treat absence of a match
# here as evidence the source has no anomalies.

def _null_due_overwritten_to_current(sql: str) -> list[str]:
    if not re.search(r"(?is)LastPaymentDueDate\s+IS\s+NULL", sql):
        return []
    if not re.search(r"(?is)DpdBucket\s*=\s*'NOT_APPLICABLE'", sql):
        return []
    if not re.search(r"(?is)WHEN\s+.*?DpdDays\s*=\s*0\s+THEN\s+'CURRENT'", sql):
        return []
    return [
        "Source anomaly: null due-date rows are assigned DpdBucket='NOT_APPLICABLE' "
        "then overwritten to 'CURRENT' because DpdDays was set to 0 and a later "
        "CASE treats 0 as CURRENT. DD must show executable behavior; intended "
        "NOT_APPLICABLE retention requires a source fix."
    ]


def _comment_vs_executable_hints(sql: str) -> list[str]:
    """Lightweight: comments claiming 'only if' next to always-true predicates."""
    out: list[str] = []
    if re.search(r"(?is)--[^\n]*quarter[^\n]*\n\s*IF\s+@ProcessDate\s*>=\s*DATEADD", sql):
        if _always_true_offset_window(sql):
            out.append(
                "Source anomaly: comments describe a conditional quarter window, "
                "but the executable IF predicate is tautological."
            )
    if re.search(r"(?is)--[^\n]*scheme is only\s+open[^\n]*\n\s*IF\s+@ProcessDate\s*<", sql):
        if _unreachable_date_compare_to_self_offset(sql):
            out.append(
                "Source anomaly: comments say the restructuring scheme opens when "
                "today falls before the cutoff, but the cutoff is defined as two years "
                "before today, so the open branch never runs."
            )
    return out
