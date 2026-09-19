"""Derive the At-a-Glance metadata table shown at the top of business reports."""
from __future__ import annotations

import re
from typing import Iterable

from app.models.core import CanonicalModel, Dialect, SQLObject, StructuralInfo

_HISTORY_TABLE_RE = re.compile(
    r"(?i)(history|hist|audit|trail|log|movement|event)",
)
_PARAM_BLOCK_RE = re.compile(
    r"(?is)\b(?:CREATE|ALTER)\s+(?:OR\s+REPLACE\s+)?(?:PROC(?:EDURE)?|FUNCTION)\s+"
    r"(?:\[?[A-Za-z0-9_]+\]?\.)?(?:\[?[A-Za-z0-9_]+\]?)\s*"
    r"(?:\((?P<body>.*?)\))?",
)
_PARAM_TOKEN_RE = re.compile(
    r"(?is)(?P<name>@[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"(?P<type>VARCHAR\s*\(\s*\d+\s*\)|NVARCHAR\s*\(\s*\d+\s*\)|CHAR\s*\(\s*\d+\s*\)|"
    r"DECIMAL\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\)|NUMERIC\s*\(\s*\d+\s*(?:,\s*\d+\s*)?\)|"
    r"NUMBER\s*(?:\(\s*\d+\s*(?:,\s*\d+\s*)?\))?|INT(?:EGER)?|BIGINT|SMALLINT|"
    r"TINYINT|FLOAT|REAL|MONEY|DATE|DATETIME2?|DATETIMEOFFSET|TIME|BIT|BOOLEAN|"
    r"UNIQUEIDENTIFIER|XML|TEXT|NTEXT|VARBINARY\s*\(\s*(?:\d+|MAX)\s*\))",
)


def _display_dialect(dialect: Dialect | str | None) -> str:
    value = getattr(dialect, "value", dialect) or ""
    mapping = {
        "sqlserver": "T-SQL",
        "oracle": "Oracle SQL / PL-SQL",
        "mysql": "MySQL",
    }
    return mapping.get(str(value).lower(), str(value).upper() or "Unknown")


def _qualified_procedure_name(obj: SQLObject) -> str:
    name = (getattr(obj, "name", None) or "").strip()
    raw = getattr(obj, "raw_sql", None) or ""
    match = re.search(
        r"(?is)\b(?:CREATE|ALTER)\s+(?:OR\s+REPLACE\s+)?(?:PROC(?:EDURE)?|FUNCTION)\s+"
        r"((?:\[[^\]]+\]|[A-Za-z0-9_]+)(?:\.(?:\[[^\]]+\]|[A-Za-z0-9_]+))?)",
        raw,
    )
    if match:
        token = match.group(1).replace("[", "").replace("]", "")
        return token
    return name or getattr(obj, "source_file", None) or "Unknown"


def extract_procedure_parameters(sql_text: str) -> list[tuple[str, str]]:
    """Return [(name, datatype)] from a CREATE PROCEDURE/FUNCTION signature."""
    text = sql_text or ""
    match = _PARAM_BLOCK_RE.search(text)
    if not match:
        # Oracle often uses name (p IN type) without CREATE PROC paren on same line.
        alt = re.search(
            r"(?is)\b(?:PROCEDURE|FUNCTION)\s+[A-Za-z0-9_\.\"]+\s*\((?P<body>.*?)\)\s*(?:IS|AS|BEGIN)\b",
            text,
        )
        body = alt.group("body") if alt else ""
    else:
        body = match.group("body") or ""
    if not body.strip():
        # T-SQL sometimes: CREATE PROC X @TimeKey INT AS
        inline = re.search(
            r"(?is)\b(?:PROC(?:EDURE)?|FUNCTION)\s+[A-Za-z0-9_\[\]\.]+\s+"
            r"((?:@?[A-Za-z_][A-Za-z0-9_]*\s+[A-Za-z0-9_\(\),\s]+)+?)\s+AS\b",
            text,
        )
        body = inline.group(1) if inline else ""
    params: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in _PARAM_TOKEN_RE.finditer(body):
        name = token.group("name").strip()
        datatype = re.sub(r"\s+", "", token.group("type").strip().upper())
        key = name.upper()
        if key in seen:
            continue
        seen.add(key)
        params.append((name, datatype))
    return params


def _format_inputs(objects: Iterable[SQLObject]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for obj in objects:
        raw = getattr(obj, "raw_sql", None) or ""
        for name, datatype in extract_procedure_parameters(raw):
            key = name.upper()
            if key in seen:
                continue
            seen.add(key)
            parts.append(f"`{name}` ({datatype})")
    return ", ".join(parts) if parts else "None"


def _visible_tables(names: Iterable[str]) -> list[str]:
    visible: list[str] = []
    seen: set[str] = set()
    for name in names:
        table = (name or "").strip()
        if not table or table.startswith("#") or table.startswith("@"):
            continue
        bare = table.split(".")[-1]
        if len(bare) <= 2 and bare.isalpha() and "." not in table:
            continue
        key = table.upper()
        if key in seen:
            continue
        seen.add(key)
        visible.append(table)
    return visible


def _produces_audit_trail(tables: Iterable[str]) -> str:
    for table in tables:
        if _HISTORY_TABLE_RE.search(table or ""):
            return "Yes — records audit events"
    return "Not detected"


def build_at_a_glance_lines(
    canonical_models: list[CanonicalModel],
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo] | None,
    business_rule_count: int,
) -> list[str]:
    """Markdown At-a-Glance table for the business report header."""
    object_ids: list[str] = []
    seen: set[str] = set()
    for model in canonical_models:
        for oid in model.object_ids:
            if oid not in seen and oid in objects:
                seen.add(oid)
                object_ids.append(oid)
    if not object_ids and objects:
        object_ids = list(objects.keys())

    selected = [objects[oid] for oid in object_ids if oid in objects]
    if not selected:
        return []

    primary = selected[0]
    try:
        procedure = _qualified_procedure_name(primary)
    except Exception:
        procedure = getattr(primary, "name", None) or "Unknown"
    dialect = _display_dialect(getattr(primary, "dialect", None))
    try:
        inputs = _format_inputs(selected)
    except Exception:
        inputs = "None"

    reads: list[str] = []
    writes: list[str] = []
    written_columns: set[str] = set()
    if structural_infos:
        for oid in object_ids:
            info = structural_infos.get(oid)
            if info is None:
                continue
            reads.extend(info.tables_read or [])
            writes.extend(info.tables_written or [])
            for cols in (info.columns_written_by_table or {}).values():
                written_columns.update(c.split(".")[-1] for c in cols if c)

    def _filter_noise(names: list[str]) -> list[str]:
        cleaned = _visible_tables(names)
        return [
            t
            for t in cleaned
            if not (
                "." not in t
                and t.split(".")[-1] in written_columns
                and t not in writes
            )
        ]

    visible_reads = _filter_noise(reads)
    visible_writes = _visible_tables(writes)
    audit = _produces_audit_trail([*visible_reads, *visible_writes, *reads, *writes])

    rows = [
        ("Procedure", f"`{procedure}`"),
        ("Dialect", dialect),
        ("Input", inputs),
        ("Business rules", str(business_rule_count)),
        ("Tables read", str(len(visible_reads))),
        ("Tables written", str(len(visible_writes))),
        ("Produces audit trail", audit),
    ]
    lines = ["## At a Glance", "", "| | |", "|---|---|"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    lines.append("")
    return lines
