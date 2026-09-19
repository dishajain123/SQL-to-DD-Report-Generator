"""Independent write-inventory scanner for regression fixtures.

This module deliberately does **not** use sqlglot or the statement splitter.
It scans procedure text with quote/comment-aware regex so expected inventories
in tests remain an independent oracle from the production parser.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExpectedWrite:
    operation: str
    target_table: str
    target_kind: str  # temp_local | temp_global | persistent
    columns: list[str] = field(default_factory=list)
    source_line: int = 0
    excerpt: str = ""
    has_join: bool = False
    has_aggregate_or_window: bool = False
    has_exists_or_in_subquery: bool = False
    expected_output_type: str = "row_formula"


def _strip_strings_and_comments(sql: str) -> str:
    """Replace string/comment contents with spaces, preserving length/newlines."""
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    in_double = False
    in_line = False
    in_block = False
    while i < n:
        ch = sql[i]
        if in_line:
            out.append("\n" if ch == "\n" else " ")
            if ch == "\n":
                in_line = False
            i += 1
            continue
        if in_block:
            out.append("\n" if ch == "\n" else " ")
            if ch == "*" and i + 1 < n and sql[i + 1] == "/":
                out.append(" ")
                i += 2
                in_block = False
                continue
            i += 1
            continue
        if in_single:
            out.append(" ")
            if ch == "'" and not (i + 1 < n and sql[i + 1] == "'"):
                in_single = False
            elif ch == "'" and i + 1 < n and sql[i + 1] == "'":
                out.append(" ")
                i += 2
                continue
            i += 1
            continue
        if in_double:
            out.append(" ")
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            out.extend([" ", " "])
            i += 2
            in_line = True
            continue
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            out.extend([" ", " "])
            i += 2
            in_block = True
            continue
        if ch == "'":
            in_single = True
            out.append(" ")
            i += 1
            continue
        if ch == '"':
            in_double = True
            out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _line_of(sql: str, index: int) -> int:
    return sql.count("\n", 0, index) + 1


def _clean_table(raw: str) -> str:
    parts = [p.strip().strip("[]").strip('"') for p in re.split(r"\s*\.\s*", raw or "") if p.strip()]
    if not parts:
        return ""
    name = parts[-1]
    prefix = ""
    for p in parts:
        if p.startswith("##"):
            prefix = "##"
            break
        if p.startswith("#"):
            prefix = "#"
            break
    return f"{prefix}{name.lstrip('#')}" if prefix else name


def _table_kind(name: str) -> str:
    if name.startswith("##"):
        return "temp_global"
    if name.startswith("#"):
        return "temp_local"
    return "persistent"


def _classify_output(operation: str, target: str, body: str) -> str:
    upper = body.upper()
    if target.startswith("#"):
        return "temp_staging"
    if re.search(r"RUNNINGPROCESSSTATUS|PROCESSSTATUS|RUNSTATUS\b", target, re.I):
        return "process_status"
    if operation == "MERGE":
        return "set_based_merge"
    if operation in {"INSERT", "SELECT_INTO"}:
        return "set_based_insert"
    if operation in {"DELETE", "TRUNCATE"}:
        return "set_based_delete"
    if re.search(r"\bOVER\s*\(", upper) or re.search(r"\bGROUP\s+BY\b", upper):
        return "cross_row_aggregate"
    if re.search(r"\bEXISTS\s*\(", upper):
        return "procedure_or_exists"
    return "row_formula"


_WRITE_PATTERNS = [
    ("INSERT", re.compile(
        r"(?is)\bINSERT\s+INTO\s+((?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
        r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)"
    )),
    ("MERGE", re.compile(
        r"(?is)\bMERGE\s+(?:INTO\s+)?((?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
        r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)"
    )),
    ("DELETE", re.compile(
        r"(?is)\bDELETE\s+(?:FROM\s+)?((?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
        r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)"
    )),
    ("TRUNCATE", re.compile(
        r"(?is)\bTRUNCATE\s+TABLE\s+((?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
        r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)"
    )),
    ("SELECT_INTO", re.compile(
        r"(?is)\bINTO\s+((?:#{1,2})(?:\[[^\]]+\]|[A-Za-z_][\w$]*))"
    )),
]

_UPDATE_FROM_RE = re.compile(
    r"(?is)\bUPDATE\s+(?P<alias>[A-Za-z_][\w$]*)\b"
    r"(?P<body>.*?)\bFROM\s+(?P<table>(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)"
    r"(?:\s+(?:AS\s+)?(?P=alias)\b)"
)
# Exclude UPDATE SET (MERGE arm) and SQL keywords mistaken for tables.
_UPDATE_DIRECT_RE = re.compile(
    r"(?is)\bUPDATE\s+(?!SET\b)(?!FROM\b)(?P<table>(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*)"
    r"(?:\s*\.\s*(?:#{1,2})?(?:\[[^\]]+\]|[A-Za-z_][\w$]*))*)\b"
)
_SET_COL_RE = re.compile(r"(?is)\bSET\s+((?:[A-Za-z_][\w$]*\s*\.\s*)?[A-Za-z_][\w$]*)\s*=")
_SQL_KEYWORDS = {
    "SET", "FROM", "WHERE", "INTO", "VALUES", "SELECT", "WITH", "AS", "ON",
    "WHEN", "MATCHED", "THEN", "AND", "OR", "NOT", "JOIN", "INNER", "LEFT",
    "RIGHT", "FULL", "OUTER", "CROSS", "USING", "MERGE", "INSERT", "DELETE",
}


def scan_expected_writes(sql: str) -> list[ExpectedWrite]:
    """Return every write operation found by the independent scanner."""
    cleaned = _strip_strings_and_comments(sql)
    writes: list[ExpectedWrite] = []
    seen: set[tuple[str, str, int]] = set()

    def _preceded_by_when_matched(start: int) -> bool:
        window = cleaned[max(0, start - 80) : start].upper()
        return bool(re.search(r"WHEN\s+(?:NOT\s+)?MATCHED\b", window))

    def _add(operation: str, target_raw: str, start: int, body: str, columns: list[str] | None = None) -> None:
        target = _clean_table(target_raw)
        if not target:
            return
        if target.upper() in _SQL_KEYWORDS:
            return
        # MERGE WHEN arms are part of the parent MERGE, not separate writes.
        if operation in {"UPDATE", "INSERT", "DELETE"} and _preceded_by_when_matched(start):
            return
        # Skip alias-only UPDATE targets that look like single letters without #
        if operation == "UPDATE" and len(target) <= 2 and not target.startswith("#"):
            return
        # The same table may be written more than once on one physical line.
        # Deduplicate overlapping scanner patterns by source offset, not line.
        key = (operation, target.upper(), start)
        if key in seen:
            return
        seen.add(key)
        excerpt = sql[start : start + 160].replace("\n", " ").strip()
        writes.append(
            ExpectedWrite(
                operation=operation,
                target_table=target,
                target_kind=_table_kind(target),
                columns=columns or [],
                source_line=_line_of(sql, start),
                excerpt=excerpt,
                has_join=bool(re.search(r"(?i)\bJOIN\b", body)),
                has_aggregate_or_window=bool(
                    re.search(r"(?i)\b(OVER\s*\(|GROUP\s+BY|SUM\s*\(|COUNT\s*\(|AVG\s*\()", body)
                ),
                has_exists_or_in_subquery=bool(
                    re.search(r"(?i)\bEXISTS\s*\(|\bIN\s*\(\s*SELECT\b", body)
                ),
                expected_output_type=_classify_output(operation, target, body),
            )
        )

    for match in _UPDATE_FROM_RE.finditer(cleaned):
        if _preceded_by_when_matched(match.start()):
            continue
        body = match.group("body") + " FROM " + match.group("table")
        cols = []
        for cm in _SET_COL_RE.finditer(match.group(0)):
            col = cm.group(1).split(".")[-1].strip()
            if col and col.upper() not in {c.upper() for c in cols}:
                cols.append(col)
        _add("UPDATE", match.group("table"), match.start(), body, cols)

    for match in _UPDATE_DIRECT_RE.finditer(cleaned):
        table = match.group("table")
        if len(_clean_table(table)) <= 2 and not table.startswith("#"):
            continue
        if _preceded_by_when_matched(match.start()):
            continue
        window = cleaned[match.start() : match.start() + 800]
        cols = []
        for cm in _SET_COL_RE.finditer(window):
            col = cm.group(1).split(".")[-1].strip()
            if col and col.upper() not in {c.upper() for c in cols}:
                cols.append(col)
        _add("UPDATE", table, match.start(), window, cols)

    for operation, pattern in _WRITE_PATTERNS:
        for match in pattern.finditer(cleaned):
            # INSERT INTO #T is one INSERT, not an additional SELECT INTO.
            # The generic INTO pattern also matches the former unless the
            # immediately preceding keyword is checked.
            if operation == "SELECT_INTO" and re.search(
                r"(?i)\bINSERT\s+$", cleaned[max(0, match.start() - 40) : match.start()]
            ):
                continue
            if operation == "INSERT" and _preceded_by_when_matched(match.start()):
                continue
            window = cleaned[match.start() : match.start() + 900]
            _add(operation, match.group(1), match.start(), window)

    writes.sort(key=lambda w: (w.source_line, w.operation, w.target_table))
    return writes


def inventory_to_dict(writes: list[ExpectedWrite]) -> list[dict]:
    return [asdict(w) for w in writes]


def sample_sql_paths(root: Path | None = None) -> list[Path]:
    base = root or Path(__file__).resolve().parents[2] / "samples" / "sql"
    paths = []
    for p in sorted(base.glob("*.sql")):
        if len(p.name) >= 2 and p.name[:2].isdigit() and 1 <= int(p.name[:2]) <= 18:
            paths.append(p)
    return paths


def read_sql_file(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")
