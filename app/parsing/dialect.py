"""Detects whether a SQL object is Oracle, MySQL, or SQL Server flavoured.

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

_SQLSERVER_MARKERS = [
    r"\bSET\s+ANSI_NULLS\b",
    r"\bSET\s+QUOTED_IDENTIFIER\b",
    r"\bBEGIN\s+TRY\b",
    r"\bEND\s+TRY\b",
    r"\bBEGIN\s+CATCH\b",
    r"\bEND\s+CATCH\b",
    r"\bOBJECT_ID\s*\(",
    r"\bISNULL\s*\(",
    r"\bDATEADD\s*\(",
    r"\bDATEDIFF\s*\(",
    r"\bTOP\s+\d+\b",
    r"\[[A-Za-z0-9_]+\](?:\s*\.\s*\[[A-Za-z0-9_]+\])?",
    r"^\s*GO\s*$",
]


def detect_dialect(sql_text: str) -> Dialect:
    oracle_score = sum(1 for pat in _ORACLE_MARKERS if re.search(pat, sql_text, re.IGNORECASE))
    mysql_score = sum(1 for pat in _MYSQL_MARKERS if re.search(pat, sql_text, re.IGNORECASE))
    sqlserver_score = sum(1 for pat in _SQLSERVER_MARKERS if re.search(pat, sql_text, re.IGNORECASE | re.MULTILINE))

    if sqlserver_score > oracle_score and sqlserver_score >= mysql_score:
        return Dialect.SQLSERVER
    if mysql_score > oracle_score:
        return Dialect.MYSQL
    # Default to Oracle: it's the dialect most markers here are unambiguous
    # for, and ties are more likely to be plain ANSI SQL written in an
    # Oracle-style procedure.
    return Dialect.ORACLE
