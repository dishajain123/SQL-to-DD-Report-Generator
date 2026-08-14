"""Detects whether a SQL object is Oracle or MySQL flavoured.

Uses simple, explainable signal-counting rather than a black box classifier —
each dialect has syntax markers that essentially never appear in the other.
"""
from __future__ import annotations

import re

from app.models.core import Dialect

_ORACLE_MARKERS = [
    r"\bSYSDATE\b",
    r"\bNVL\s*\(",
    r"\bROWNUM\b",
    r"\bROWID\b",
    r"CREATE\s+OR\s+REPLACE\s+PROCEDURE",
    r"\bEXCEPTION\s+WHEN\b",
    r":=",
    r"\bVARCHAR2\b",
    r"MERGE\s+INTO.+USING",
]

_MYSQL_MARKERS = [
    r"\bAUTO_INCREMENT\b",
    r"\bENGINE\s*=\s*InnoDB\b",
    r"`[A-Za-z0-9_]+`",
    r"\bLIMIT\s+\d+\s*(,\s*\d+)?\s*;",
    r"\bIFNULL\s*\(",
    r"\bDELIMITER\b",
    r"\bUNSIGNED\b",
]


def detect_dialect(sql_text: str) -> Dialect:
    oracle_score = sum(1 for pat in _ORACLE_MARKERS if re.search(pat, sql_text, re.IGNORECASE))
    mysql_score = sum(1 for pat in _MYSQL_MARKERS if re.search(pat, sql_text, re.IGNORECASE))

    if mysql_score > oracle_score:
        return Dialect.MYSQL
    # Default to Oracle: it's the dialect most markers here are unambiguous
    # for, and ties are more likely to be plain ANSI SQL written in an
    # Oracle-style procedure.
    return Dialect.ORACLE
