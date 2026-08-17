from __future__ import annotations

import re


def canonical_logical_name(value: str) -> str:
    """Normalize a DD field/entity name for logical identity checks.

    The pipeline frequently sees case-variant spellings of the same
    physical field across source SQL, review edits, and exports. DD rows
    should treat those as one logical field when grouping or deduping,
    while still preserving the original display value in the row itself.
    """
    return value.strip().strip('"').upper()


def canonical_expression_key(value: str) -> str:
    """Normalize an expression for logical grouping/deduping.

    The goal is not to rewrite the expression for display, only to make
    case-variant references compare as the same logical formula.
    """
    return re.sub(r"\s+", "", value).casefold()
