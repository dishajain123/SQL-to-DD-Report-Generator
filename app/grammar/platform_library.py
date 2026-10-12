"""Documented 4X function library (authoritative: samples/platform_docs/4x_functions_operators.md).

Generated derivation formulas must use only these function names plus the control-flow
and operator syntax defined in that document. Logical ``AND`` / ``OR`` / ``NOT`` use
infix or ``AND(...)`` / ``OR(...)`` / ``NOT(...)`` forms from the grammar — not listed here.
"""
from __future__ import annotations

# Commonly used + Text + Date & time + Math sections of the platform reference.
DOCUMENTED_4X_FUNCTIONS: frozenset[str] = frozenset(
    {
        "ISEMPTY",
        "ISNOTEMPTY",
        "MAX",
        "MIN",
        "SUM",
        "COUNT",
        "COALESCE",
        "SUBSTR",
        "LOWER",
        "UPPER",
        "LEN",
        "CONVERT",
        "REGEX",
        "CONCAT",
        "TRIM",
        "REPLACE",
        "SOM",
        "EOM",
        "SOY",
        "EOY",
        "SOFY",
        "EOFY",
        "SOQ",
        "EOQ",
        "DATEPART",
        "DATEDIFF",
        "TODATE",
        "ADDDAY",
        "PERIOD",
        "ROUND",
        "ABS",
        "FLOOR",
        "CEIL",
    }
)

# Legacy compiler alias — normalized to TODATE before emission / validation.
FUNCTION_COMPILE_ALIASES: dict[str, str] = {"DATE": "TODATE"}

PLATFORM_LIBRARY_DOC = "samples/platform_docs/4x_functions_operators.md"


def normalize_function_name_for_platform(name: str) -> str:
    """Map internal/legacy names to documented 4X function tokens."""
    upper = (name or "").strip().upper()
    return FUNCTION_COMPILE_ALIASES.get(upper, upper)


def is_documented_4x_function(name: str) -> bool:
    return normalize_function_name_for_platform(name) in DOCUMENTED_4X_FUNCTIONS
