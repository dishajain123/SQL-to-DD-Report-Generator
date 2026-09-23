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
# Bracket-quoted (`[PRO].[Foo]`) and quoted-identifier calls are the T-SQL
# norm in this corpus, not the exception -- the object-name group must
# accept them, not just bare `[A-Za-z0-9_.]` identifiers. `IMMEDIATE` is
# excluded so `EXECUTE IMMEDIATE '<dynamic sql>'` (Oracle dynamic SQL,
# already flagged separately by _DYNAMIC_SQL_RE) is never misread as a
# call to an object literally named "IMMEDIATE".
_CALLED_OBJECT_NAME_RE = r'(?:\[[^\]]+\]|"[^"]+"|[A-Za-z_][\w$]*)'
_CALLED_OBJECT_RE = re.compile(
    rf"\b(?:EXEC|EXECUTE|CALL)\s+(?!IMMEDIATE\b)"
    rf"({_CALLED_OBJECT_NAME_RE}(?:\s*\.\s*{_CALLED_OBJECT_NAME_RE})*)",
    re.IGNORECASE,
)
# A rule-versioning threshold is a comparison against a *parameter or local
# variable* (T-SQL `@TIMEKEY`/`@V_TIMEKEY`, Oracle PL/SQL bare `p_TIMEKEY`),
# not a column. Columns that carry the same "...TIMEKEY" suffix -- SCD-2
# row-validity bounds like `ABD.EffectiveToTimeKey = 49999` -- are always
# written qualified with a table alias in this corpus, so the negative
# lookbehind for `.` (a qualified reference) or another identifier
# character (matching only at the start of an identifier) is what tells a
# parameter apart from a column here, independent of which dialect's
# parameter-naming convention (`@`-sigil vs. bare) is in play. Equality is
# deliberately excluded: `@TIMEKEY = 26418` / `p_TIMEKEY = 26418` in real
# procedures is a debug/log assignment, not a version boundary -- rule
# cutovers are always expressed as an inequality against a cutover date.
_VERSION_THRESHOLD_RE = re.compile(
    r"(?<![.\w])@?([A-Za-z0-9_]*TIMEKEY[A-Za-z0-9_]*)\s*(>=|<=|>|<)\s*(\d+)",
    re.IGNORECASE,
)

# Universal SCD-2 "this row is currently valid" sentinels in banking data
# warehouses. A belt-and-braces guard: even a genuine `@...TIMEKEY` parameter
# compared against one of these is far more likely to be an open-ended
# row-validity bound copied into a variable than an actual rule cutover.
_SENTINEL_TIMEKEYS = {49999, 99999, 99991231, 29991231}


def analyze_object(obj: SQLObject) -> StructuralInfo:
    raw_statements = split_statements(obj.raw_sql, obj.dialect)
    statements: list[StatementInfo] = [
        parse_statement(stmt, i, obj.dialect) for i, stmt in enumerate(raw_statements)
    ]

    tables_read: set[str] = set()
    tables_written: set[str] = set()
    columns_written: set[str] = set()
    # Keyed by table -> {UPPER(column): first-seen-casing column}. SQL
    # column names are case-insensitive, so ``A.DEGREASON = ...`` and a
    # later ``A.DegReason = ...`` against the same physical column must
    # collapse into one derivation target, not two separate DD rows.
    columns_written_by_table: dict[str, dict[str, str]] = {}
    for s in statements:
        tables_read.update(s.tables_read)
        tables_written.update(s.tables_written)
        for table, cols in s.set_columns_by_table.items():
            by_upper = columns_written_by_table.setdefault(table, {})
            for col in cols:
                by_upper.setdefault(col.upper(), col)
            columns_written.update(by_upper.values())

    dml_statements = [s for s in statements if s.statement_type in ("SELECT", "UPDATE", "MERGE", "INSERT", "DELETE")]
    parsed_ok_count = sum(1 for s in dml_statements if s.parsed_ok)
    confidence = parsed_ok_count / len(dml_statements) if dml_statements else 1.0

    unsupported = [s.parse_error for s in dml_statements if not s.parsed_ok and s.parse_error]
    smart_chunks = build_smart_chunks(obj.object_id, statements)
    chunk_confidence = min((chunk.confidence for chunk in smart_chunks), default=1.0)

    # Attach write coverage ledger blockers as unsupported constructs so
    # downstream stages cannot treat missing MERGE/INSERT targets as success.
    from app.parsing.coverage_ledger import build_coverage_ledger

    ledger = build_coverage_ledger(
        StructuralInfo(
            object_id=obj.object_id,
            statements=statements,
            tables_read=sorted(tables_read),
            tables_written=sorted(tables_written),
            columns_written=sorted(columns_written),
            columns_written_by_table={t: sorted(c.values()) for t, c in columns_written_by_table.items()},
        ),
        source_sql=obj.raw_sql,
    )
    for blocker in ledger.blockers:
        if blocker not in unsupported:
            unsupported.append(blocker)

    return StructuralInfo(
        object_id=obj.object_id,
        statements=statements,
        tables_read=sorted(tables_read),
        tables_written=sorted(tables_written),
        columns_written=sorted(columns_written),
        columns_written_by_table={t: sorted(c.values()) for t, c in columns_written_by_table.items()},
        called_objects=_find_called_objects(obj.raw_sql),
        has_dynamic_sql=bool(_DYNAMIC_SQL_RE.search(obj.raw_sql)),
        version_thresholds=_find_version_thresholds(obj.raw_sql),
        smart_chunks=smart_chunks,
        confidence=round(min(confidence, chunk_confidence), 3),
        unsupported_constructs=unsupported,
    )


def _find_called_objects(raw_sql: str) -> list[str]:
    # Nearly every procedure header in this corpus carries a usage note
    # like `--exec [Pro].[DPD_Calculation] @timekey=25140` -- without
    # masking comments first, those get read as real call edges.
    live_sql = _strip_comments_for_threshold_scan(raw_sql)
    names: set[str] = set()
    for m in _CALLED_OBJECT_RE.finditer(live_sql):
        part = m.group(1).split(".")[-1].strip()
        part = part.strip("[]").strip('"').strip()
        if part and not part.isdigit():
            names.add(part)
    return sorted(names)


def _strip_comments_for_threshold_scan(sql: str) -> str:
    """Blank out `--` and `/* */` comment contents (quote-aware), preserving
    every other character's position so match offsets in the caller's
    output still point at real code.

    Nearly every procedure header in the real corpus carries a usage
    example like `--exec [Pro].[DPD_Calculation] @timekey=25140;` -- without
    this, those examples get misread as genuine rule-version thresholds.
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single = in_double = in_line = in_block = False
    while i < n:
        ch = sql[i]
        if in_line:
            out.append("\n" if ch == "\n" else " ")
            in_line = ch != "\n"
            i += 1
            continue
        if in_block:
            if ch == "*" and i + 1 < n and sql[i + 1] == "/":
                out.append("  ")
                i += 2
                in_block = False
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if in_single:
            out.append(ch)
            if ch == "'" and not (i + 1 < n and sql[i + 1] == "'"):
                in_single = False
            i += 1
            continue
        if in_double:
            out.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            in_line = True
            out.append("  ")
            i += 2
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            in_block = True
            out.append("  ")
            i += 2
            continue
        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        out.append(ch)
        i += 1
    return "".join(out)


def _find_version_thresholds(raw_sql: str) -> list[VersionThreshold]:
    live_sql = _strip_comments_for_threshold_scan(raw_sql)
    thresholds = []
    for m in _VERSION_THRESHOLD_RE.finditer(live_sql):
        if int(m.group(3)) in _SENTINEL_TIMEKEYS:
            continue
        thresholds.append(
            VersionThreshold(
                variable=m.group(1),
                operator=m.group(2),
                value=m.group(3),
                raw_condition=m.group(0),
            )
        )
    return thresholds
