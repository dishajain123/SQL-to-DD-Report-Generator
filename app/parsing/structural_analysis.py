"""Architecture step 7: Structural Analysis + Guardrails.

Aggregates per-statement parse results into one StructuralInfo per object:
tables read/written, columns touched, dynamic SQL flags, and detected
TIMEKEY/date-threshold rule-versioning branches (architecture step 13c).
"""
from __future__ import annotations

import re

from app.models.core import SQLObject, StatementInfo, StructuralInfo, VersionThreshold
from app.parsing.sql_parser import parse_statement, split_statements
from app.parsing.smart_chunking import build_smart_chunks

_DYNAMIC_SQL_RE = re.compile(r"EXECUTE\s+IMMEDIATE", re.IGNORECASE)
_CALLED_OBJECT_RE = re.compile(
    r"\b(?:EXEC|CALL)\s+([A-Za-z0-9_\.]+)", re.IGNORECASE
)
# Matches things like: p_TIMEKEY > 26267   or   @TIMEKEY >= 26384
_VERSION_THRESHOLD_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*TIMEKEY)\s*(>=|<=|>|<|=)\s*(\d+)", re.IGNORECASE
)


def analyze_object(obj: SQLObject) -> StructuralInfo:
    raw_statements = split_statements(obj.raw_sql, obj.dialect)
    statements: list[StatementInfo] = [
        parse_statement(stmt, i, obj.dialect) for i, stmt in enumerate(raw_statements)
    ]

    tables_read: set[str] = set()
    tables_written: set[str] = set()
    columns_written: set[str] = set()
    columns_written_by_table: dict[str, set[str]] = {}
    for s in statements:
        tables_read.update(s.tables_read)
        tables_written.update(s.tables_written)
        for table, cols in s.set_columns_by_table.items():
            columns_written.update(cols)
            columns_written_by_table.setdefault(table, set()).update(cols)

    dml_statements = [s for s in statements if s.statement_type in ("SELECT", "UPDATE", "MERGE", "INSERT", "DELETE")]
    parsed_ok_count = sum(1 for s in dml_statements if s.parsed_ok)
    confidence = parsed_ok_count / len(dml_statements) if dml_statements else 1.0

    unsupported = [s.parse_error for s in dml_statements if not s.parsed_ok and s.parse_error]
    smart_chunks = build_smart_chunks(obj.object_id, statements)
    chunk_confidence = min((chunk.confidence for chunk in smart_chunks), default=1.0)

    return StructuralInfo(
        object_id=obj.object_id,
        statements=statements,
        tables_read=sorted(tables_read),
        tables_written=sorted(tables_written),
        columns_written=sorted(columns_written),
        columns_written_by_table={t: sorted(c) for t, c in columns_written_by_table.items()},
        called_objects=_find_called_objects(obj.raw_sql),
        has_dynamic_sql=bool(_DYNAMIC_SQL_RE.search(obj.raw_sql)),
        version_thresholds=_find_version_thresholds(obj.raw_sql),
        smart_chunks=smart_chunks,
        confidence=round(min(confidence, chunk_confidence), 3),
        unsupported_constructs=unsupported,
    )


def _find_called_objects(raw_sql: str) -> list[str]:
    return sorted({m.group(1).split(".")[-1] for m in _CALLED_OBJECT_RE.finditer(raw_sql)})


def _find_version_thresholds(raw_sql: str) -> list[VersionThreshold]:
    thresholds = []
    for m in _VERSION_THRESHOLD_RE.finditer(raw_sql):
        thresholds.append(
            VersionThreshold(
                variable=m.group(1),
                operator=m.group(2),
                value=m.group(3),
                raw_condition=m.group(0),
            )
        )
    return thresholds
