"""Function and arithmetic conformance between SQL Server and the 4X platform.

Unknown or differently implemented functions block approval rather than
being silently approximated.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# SQL Server functions that have a known, reviewed mapping to platform 4X.
_SQL_TO_PLATFORM: dict[str, str] = {
    "DATEADD": "ADDDAY/ADDMONTH/ADDYEAR (unit-specific)",
    "DATEDIFF": "DATEDIFF",
    "GETDATE": "TODAY / BUSINESS_DATE policy",
    "ISNULL": "COALESCE",
    "COALESCE": "COALESCE",
    "NULLIF": "NULLIF",
    "ABS": "ABS",
    "ROUND": "ROUND",
    "FLOOR": "FLOOR",
    "CEILING": "CEILING",
    "UPPER": "UPPER",
    "LOWER": "LOWER",
    "LTRIM": "TRIM",
    "RTRIM": "TRIM",
    "TRIM": "TRIM",
    "LEN": "LENGTH",
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
    "SUBSTRING": "SUBSTRING",
    "REPLACE": "REPLACE",
    "CONCAT": "CONCAT",
    "CAST": "CAST / platform type coercion",
    "CONVERT": "CAST / platform type coercion",
}

# Present in samples but not safely expressible as a row formula without review.
_BLOCKING_SQL_FUNCTIONS: set[str] = {
    "EOMONTH",
    "ERROR_MESSAGE",
    "ERROR_NUMBER",
    "ERROR_SEVERITY",
    "OBJECT_ID",
    "ROW_NUMBER",
    "RANK",
    "DENSE_RANK",
    "LAG",
    "LEAD",
    "SUM",  # aggregate / window — not a scalar row formula by itself
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
}

_FUNC_CALL_RE = re.compile(r"(?i)\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass
class ConformanceFinding:
    function_name: str
    status: str  # mapped | blocking | unknown
    detail: str


@dataclass
class ConformanceReport:
    findings: list[ConformanceFinding] = field(default_factory=list)

    @property
    def blockers(self) -> list[str]:
        return [
            f"{f.function_name}: {f.detail}"
            for f in self.findings
            if f.status in {"blocking", "unknown"}
        ]

    @property
    def ok(self) -> bool:
        return not self.blockers


def analyze_sql_functions(sql: str) -> ConformanceReport:
    report = ConformanceReport()
    seen: set[str] = set()
    # Skip control keywords that look like functions.
    skip = {
        "IF", "WHILE", "CASE", "WHEN", "EXISTS", "VALUES", "SELECT", "CONVERT",
        "CAST", "BEGIN", "END", "AND", "OR", "NOT", "IN", "ON", "FROM", "WHERE",
        "UPDATE", "SET", "INSERT", "INTO", "MERGE", "USING", "WITH", "AS",
    }
    for match in _FUNC_CALL_RE.finditer(sql or ""):
        name = match.group(1).upper()
        if name in skip or name in seen:
            continue
        seen.add(name)
        if name in _BLOCKING_SQL_FUNCTIONS:
            # SUM/COUNT inside OVER() or GROUP BY are cross-row; ISNULL etc. ok
            if name in {"SUM", "COUNT", "AVG", "MIN", "MAX"}:
                window = sql[max(0, match.start() - 20) : match.end() + 80]
                if not re.search(r"(?i)\bOVER\s*\(|\bGROUP\s+BY\b", window) and name in {"MIN", "MAX"}:
                    # Scalar MIN/MAX of a single expression can map; still flag for review.
                    report.findings.append(
                        ConformanceFinding(name, "mapped", _SQL_TO_PLATFORM.get(name, "review scalar aggregate"))
                    )
                    continue
                report.findings.append(
                    ConformanceFinding(
                        name,
                        "blocking",
                        "aggregate/window function is not a per-row formula; requires manual/platform mechanism",
                    )
                )
            else:
                report.findings.append(
                    ConformanceFinding(
                        name,
                        "blocking",
                        "no approved platform equivalent; blocks ready-to-present approval",
                    )
                )
        elif name in _SQL_TO_PLATFORM:
            report.findings.append(
                ConformanceFinding(name, "mapped", f"maps to {_SQL_TO_PLATFORM[name]}")
            )
        elif name.startswith("FN_") or name.startswith("dbo"):
            report.findings.append(
                ConformanceFinding(name, "unknown", "user-defined function has no platform mapping")
            )
        # else: likely a column or unknown — ignore bare identifiers that aren't SQL builtins
    return report


def analyze_expression_functions(expression: str) -> ConformanceReport:
    """Check a generated 4X expression for disallowed SQL leftovers."""
    report = ConformanceReport()
    sqlish = {"DATEADD", "DATEDIFF", "GETDATE", "ISNULL", "EOMONTH", "CONVERT", "EXISTS"}
    for match in _FUNC_CALL_RE.finditer(expression or ""):
        name = match.group(1).upper()
        if name in sqlish:
            report.findings.append(
                ConformanceFinding(
                    name,
                    "blocking",
                    f"SQL Server function {name} left in platform expression",
                )
            )
    return report
