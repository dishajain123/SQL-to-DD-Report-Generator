"""Shared T-SQL text helpers for derivation v2 phases."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.utils.entity_name_map import resolve_entity_name


_STMT_START = re.compile(
    r"(?is)\b(?:UPDATE|INSERT|DELETE|MERGE|SELECT|CREATE|DROP|TRUNCATE|EXEC|EXECUTE|DECLARE|GO"
    r"|IF|BEGIN|ELSE)\b"
)

def strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments."""
    out: list[str] = []
    i = 0
    n = len(sql)
    in_single = False
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if in_single:
            out.append(ch)
            if ch == "'":
                if nxt == "'":
                    out.append(nxt)
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if ch == "-" and nxt == "-":
            i += 2
            while i < n and sql[i] not in "\r\n":
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (sql[i] == "*" and sql[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def split_csv_respecting_parens(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    in_single = False
    for ch in text:
        if ch == "'" and not in_single:
            in_single = True
            buf.append(ch)
            continue
        if ch == "'" and in_single:
            in_single = False
            buf.append(ch)
            continue
        if in_single:
            buf.append(ch)
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts


def normalize_table_name(name: str) -> str:
    text = (name or "").strip().strip("[]").strip('"')
    if ".." in text:
        text = text.split("..", 1)[-1]
    if "." in text:
        parts = text.split(".")
        if parts[-1].startswith("#"):
            return parts[-1]
        # Keep physical bare table name (schema-stripped) for matching.
        return parts[-1]
    return text


def bare_ident(name: str) -> str:
    return (name or "").strip().strip("[]").strip('"').strip()


_TABLE_TOKEN = r"(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?(?:\.\[[A-Za-z_][A-Za-z0-9_]*\]|\.[A-Za-z_][A-Za-z0-9_]*)?)"
_ALIAS_TOKEN = r"(?:AS\s+)?(?P<alias>(?!ON\b|WHERE\b|JOIN\b|LEFT\b|RIGHT\b|INNER\b|OUTER\b|FULL\b|CROSS\b|GROUP\b|ORDER\b|SET\b|FROM\b)[A-Za-z_][A-Za-z0-9_]*)"


def parse_from_join_clause(from_body: str) -> list[tuple[str, str | None, str | None]]:
    """Parse tables/aliases from a FROM/JOIN body (WITH or WITHOUT leading FROM).

    Returns list of ``(table, alias, on_clause)``.
    """
    text = (from_body or "").strip()
    if not text:
        return []
    if not re.match(r"(?is)^(FROM|JOIN)\b", text):
        text = "FROM " + text

    results: list[tuple[str, str | None, str | None]] = []
    # Split roughly on JOIN keywords while keeping JOIN type words out of aliases.
    pattern = re.compile(
        rf"(?is)\b(?:FROM|(?:LEFT|RIGHT|INNER|OUTER|FULL|CROSS)\s+JOIN|JOIN)\s+"
        rf"(?P<table>{_TABLE_TOKEN})"
        rf"(?:\s+{_ALIAS_TOKEN})?"
        rf"(?:\s+ON\s+(?P<on>.+?)(?=\b(?:LEFT|RIGHT|INNER|OUTER|FULL|CROSS)\s+JOIN\b|\bJOIN\b|\bWHERE\b|\bGROUP\b|\bORDER\b|$))?"
    )
    for match in pattern.finditer(text):
        table = normalize_table_name(match.group("table"))
        alias = match.groupdict().get("alias")
        on_clause = (match.group("on") or "").strip() or None
        results.append((table, alias, on_clause))
    return results


def extract_update_statements(sql: str) -> list[dict[str, str | None]]:
    """Extract UPDATE statements with SET / FROM / WHERE (paren-aware)."""
    text = strip_sql_comments(sql or "")
    statements: list[dict[str, str | None]] = []
    pattern = re.compile(r"(?is)\bUPDATE\b")
    for match in pattern.finditer(text):
        start = match.start()
        i = match.end()
        # head until SET
        set_match = re.search(r"(?is)\bSET\b", text[i:])
        if not set_match:
            continue
        head = text[i : i + set_match.start()].strip()
        i = i + set_match.end()

        set_clause, i = _read_until_keyword(text, i, {"FROM", "WHERE"}, stop_at_statement=True)
        from_clause = None
        where_clause = None
        if i < len(text) and re.match(r"(?is)^FROM\b", text[i:]):
            i = i + len("FROM")
            from_clause, i = _read_until_keyword(text, i, {"WHERE"}, stop_at_statement=True)
            from_clause = from_clause.strip() or None
        if i < len(text) and re.match(r"(?is)^WHERE\b", text[i:]):
            i = i + len("WHERE")
            where_clause, i = _read_until_keyword(text, i, set(), stop_at_statement=True)
            where_clause = where_clause.strip() or None

        statements.append(
            {
                "head": head,
                "set_clause": set_clause.strip(),
                "from_clause": from_clause,
                "where_clause": where_clause,
                "raw_sql": text[start:i].strip(),
                "start": start,
                "end": i,
            }
        )
    return statements


@dataclass
class ControlBranchSpan:
    """One arm of a procedural ``IF / ELSE IF / ELSE`` chain."""

    group_id: str
    index: int
    kind: str  # IF | ELSEIF | ELSE
    condition: str | None
    body_start: int
    body_end: int


def extract_if_else_chains(sql: str) -> list[ControlBranchSpan]:
    """Locate procedural IF / ELSE IF / ELSE BEGIN…END chains (comment-stripped).

    Returns branch spans whose ``body_start``/``body_end`` cover the interior
    of each ``BEGIN…END`` so callers can map UPDATE statements into arms.
    """
    text = strip_sql_comments(sql or "")
    branches: list[ControlBranchSpan] = []
    group_counter = 0
    i = 0
    n = len(text)

    while i < n:
        # Skip IF OBJECT_ID(...) utility guards — not derivation branches.
        if re.match(r"(?is)^IF\s+OBJECT_ID\b", text[i:]):
            i += 1
            continue
        if_match = re.match(r"(?is)^IF\b", text[i:])
        if not if_match:
            i += 1
            continue
        # "DROP TABLE IF EXISTS x" / "CREATE TABLE IF NOT EXISTS x" use IF as
        # part of that fixed idiom, not as a procedural control-flow keyword.
        # Without this check, its "condition" is misread as "EXISTS <name>"
        # and the (bogus) single-statement body swallows real statements
        # that follow, silently corrupting their control-branch grouping.
        preceding = text[max(0, i - 10) : i].rstrip()
        if preceding.upper().endswith("TABLE"):
            i += 1
            continue

        # Parse a full IF … [ELSE IF …]* [ELSE …]? chain starting at i.
        chain_start = i
        parsed = _parse_one_if_else_chain(text, i, group_counter)
        if parsed is None:
            i += 1
            continue
        chain_branches, next_i = parsed
        if len(chain_branches) >= 2 or (
            len(chain_branches) == 1 and chain_branches[0].kind == "IF"
        ):
            # Keep single-IF chains too (IF without ELSE) so outer condition is captured.
            branches.extend(chain_branches)
            group_counter += 1
        i = max(next_i, chain_start + 1)

    return branches


def _parse_one_if_else_chain(
    text: str,
    start: int,
    group_id_num: int,
) -> tuple[list[ControlBranchSpan], int] | None:
    group_id = f"ifelse-{group_id_num}"
    branches: list[ControlBranchSpan] = []
    i = start
    index = 0

    # Leading IF
    if not re.match(r"(?is)^IF\b", text[i:]):
        return None
    i += len("IF")
    cond, i = _read_if_condition(text, i)
    body_start, body_end, i = _read_begin_end_body(text, i)
    if body_start < 0:
        return None
    branches.append(
        ControlBranchSpan(
            group_id=group_id,
            index=index,
            kind="IF",
            condition=cond,
            body_start=body_start,
            body_end=body_end,
        )
    )
    index += 1

    while True:
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1
        else_if = re.match(r"(?is)^ELSE\s+IF\b|^ELSEIF\b", text[i:])
        if else_if:
            i += else_if.end()
            cond, i = _read_if_condition(text, i)
            body_start, body_end, i = _read_begin_end_body(text, i)
            if body_start < 0:
                break
            branches.append(
                ControlBranchSpan(
                    group_id=group_id,
                    index=index,
                    kind="ELSEIF",
                    condition=cond,
                    body_start=body_start,
                    body_end=body_end,
                )
            )
            index += 1
            continue
        else_only = re.match(r"(?is)^ELSE\b", text[i:])
        if else_only:
            i += else_only.end()
            body_start, body_end, i = _read_begin_end_body(text, i)
            if body_start < 0:
                break
            branches.append(
                ControlBranchSpan(
                    group_id=group_id,
                    index=index,
                    kind="ELSE",
                    condition=None,
                    body_start=body_start,
                    body_end=body_end,
                )
            )
            index += 1
        break

    return branches, i


def _read_if_condition(text: str, start: int) -> tuple[str, int]:
    """Read the predicate after ``IF`` / ``ELSE IF`` up to ``BEGIN`` or a statement."""
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    buf: list[str] = []
    depth = 0
    in_single = False
    while i < len(text):
        ch = text[i]
        if in_single:
            buf.append(ch)
            if ch == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0 and re.match(r"(?is)^BEGIN\b", text[i:]):
            break
        # Single-statement IF without BEGIN: stop before UPDATE/INSERT/…
        if depth == 0 and _STMT_START.match(text[i:]):
            break
        buf.append(ch)
        i += 1
    return "".join(buf).strip(), i


def _read_begin_end_body(text: str, start: int) -> tuple[int, int, int]:
    """Return (body_start, body_end, pos_after_end) for a BEGIN…END block.

    If there is no BEGIN, treat the next single statement as the body.
    """
    i = start
    while i < len(text) and text[i].isspace():
        i += 1
    if re.match(r"(?is)^BEGIN\b", text[i:]):
        i += len("BEGIN")
        body_start = i
        depth = 1
        case_depth = 0
        in_single = False
        while i < len(text):
            ch = text[i]
            if in_single:
                if ch == "'":
                    if i + 1 < len(text) and text[i + 1] == "'":
                        i += 2
                        continue
                    in_single = False
                i += 1
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if re.match(r"(?is)^BEGIN\b", text[i:]):
                depth += 1
                i += 5
                continue
            if re.match(r"(?is)^CASE\b", text[i:]):
                # A bare CASE ... END is not a BEGIN block -- its own END
                # must not be mistaken for this block's terminator below.
                case_depth += 1
                i += 4
                continue
            if re.match(r"(?is)^END\b", text[i:]):
                if case_depth > 0:
                    case_depth -= 1
                    i += 3
                    continue
                depth -= 1
                if depth == 0:
                    body_end = i
                    i += 3
                    return body_start, body_end, i
                i += 3
                continue
            i += 1
        return -1, -1, start

    # Single-statement body (no BEGIN)
    body_start = i
    _ignored, after = _read_until_keyword(text, i, set(), stop_at_statement=False)
    # Read one statement: advance until next sibling ELSE/END/IF at depth 0 is hard;
    # fall back to reading until END/ELSE at depth 0 via statement extractor end.
    stmt_end = body_start
    upd = re.match(r"(?is)^UPDATE\b", text[body_start:])
    if upd:
        # Reuse update extractor on a slice — approximate end by scanning.
        from_slice = text[body_start:]
        # Find end via extract on prefixed text — cheap path: read until END/ELSE
        j = body_start
        depth = 0
        in_single = False
        while j < len(text):
            if in_single:
                if text[j] == "'":
                    if j + 1 < len(text) and text[j + 1] == "'":
                        j += 2
                        continue
                    in_single = False
                j += 1
                continue
            if text[j] == "'":
                in_single = True
                j += 1
                continue
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth = max(0, depth - 1)
            if depth == 0 and re.match(r"(?is)^(END|ELSE)\b", text[j:]):
                break
            j += 1
        return body_start, j, j

    return -1, -1, start


def exists_condition_to_row_predicate(condition: str | None) -> str | None:
    """Turn ``EXISTS (SELECT … WHERE <pred>)`` into the inner ``<pred>`` when possible."""
    if not condition:
        return None
    text = condition.strip()
    match = re.match(r"(?is)^EXISTS\s*\((.*)\)$", text)
    if not match:
        return text
    inner = match.group(1).strip()
    pred = _extract_where_predicate(inner)
    if not pred:
        return None
    # Drop trailing GROUP BY / ORDER BY; keep HAVING as AND-able fragment when present.
    having = _extract_having_clause(pred)
    pred = _strip_group_order(pred)
    if having:
        return f"({pred}) AND ({having})" if pred else having
    return pred.strip() or None


def in_subquery_to_row_predicate(lhs: str, subquery: str) -> tuple[str | None, list[str]]:
    """Project ``lhs IN (SELECT …)`` into a row predicate + dependency column refs.

    Returns ``(predicate_sql | None, dependency_refs)``. When the subquery
    WHERE cannot be projected cleanly, ``predicate_sql`` is None and the
    caller should fall back to a tautology while still keeping dependency_refs
    for lineage / HITL.
    """
    body = (subquery or "").strip()
    while body.startswith("(") and body.endswith(")") and _parens_balanced(body[1:-1]):
        body = body[1:-1].strip()
    if not re.match(r"(?is)^SELECT\b", body):
        return None, []

    deps = extract_subquery_dependency_refs(body)
    # Prefer correlating on projected column == lhs when SELECT list is a single col.
    select_list, rest = _split_select_list(body)
    from_table = _first_from_table(rest) or ""
    where_pred = _extract_where_predicate(body)
    where_pred = _strip_group_order(where_pred or "")
    having = _extract_having_clause(body)

    parts: list[str] = []
    proj = (select_list or "").strip()
    # Single projected identifier (optionally qualified)
    proj_col = None
    proj_m = re.match(
        r"(?is)^(?:DISTINCT\s+)?(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*$",
        proj,
    )
    if proj_m and lhs.strip():
        proj_col = bare_ident(proj_m.group("col"))
        if from_table:
            parts.append(f"{normalize_table_name(from_table)}.{proj_col} = {lhs.strip()}")
        else:
            parts.append(f"{proj_col} = {lhs.strip()}")

    if where_pred:
        if from_table:
            where_pred = _qualify_bare_columns(where_pred, from_table)
        parts.append(f"({where_pred})")
    # HAVING with aggregates is not row-level 4X — keep deps, skip the clause.
    if having and not re.search(r"(?is)\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", having):
        parts.append(f"({having})")

    if not parts:
        return None, deps
    return " AND ".join(parts), deps


def _qualify_bare_columns(predicate: str, table: str) -> str:
    """Prefix bare column identifiers with ``table.`` for cross-entity resolution."""
    table_norm = normalize_table_name(table)
    keywords = {
        "AND", "OR", "NOT", "IS", "NULL", "IN", "EXISTS", "BETWEEN", "LIKE",
        "CASE", "WHEN", "THEN", "ELSE", "END", "TRUE", "FALSE",
    }

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token.upper() in keywords:
            return token
        if token.startswith("@") or token.startswith("#"):
            return token
        # Already qualified or function name followed by (
        return f"{table_norm}.{token}"

    # Only replace identifiers that are not already qual.col and not after '.'
    return re.sub(
        r"(?<![.\w])(@?[A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\()",
        repl,
        predicate or "",
    )


def extract_subquery_dependency_refs(sql_fragment: str) -> list[str]:
    """Collect ``table.column`` / bare column tokens from a subquery fragment."""
    text = sql_fragment or ""
    refs: list[str] = []
    seen: set[str] = set()
    # Skip schema.table tokens that appear after FROM/JOIN/INTO/UPDATE/MERGE.
    skip_spans: list[tuple[int, int]] = []
    for m in re.finditer(
        rf"(?is)\b(?:FROM|JOIN|INTO|UPDATE|MERGE(?:\s+INTO)?|USING)\s+{_TABLE_TOKEN}",
        text,
    ):
        skip_spans.append((m.start(), m.end()))

    def _in_skip(pos: int) -> bool:
        return any(a <= pos < b for a, b in skip_spans)

    for match in re.finditer(
        r"(?P<qual>[#A-Za-z_][A-Za-z0-9_]*)\.(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)",
        text,
    ):
        if _in_skip(match.start()):
            continue
        qual = match.group("qual")
        if qual.upper() in {"SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "EXISTS"}:
            continue
        col = bare_ident(match.group("col"))
        # Heuristic: all-caps short schema names with CapCase table-looking col
        # are usually schema.table, not alias.column (e.g. PRO.CustomerCal).
        if (
            len(qual) <= 3
            and qual.isupper()
            and col[:1].isupper()
            and any(ch.islower() for ch in col[1:]) is False
            and len(col) > 8
        ):
            # Ambiguous — skip schema.table-ish
            continue
        if qual.upper() in {"PRO", "DBO", "SYS", "TEMPDB"}:
            continue
        key = f"{normalize_table_name(qual)}.{col}"
        if key.upper() not in seen:
            seen.add(key.upper())
            refs.append(key)
    # Also capture bare identifiers after WHERE / ON that look like columns
    for match in re.finditer(
        r"(?is)\b(?:WHERE|AND|OR|ON|HAVING)\s+(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s*(?:=|<>|!=|>=|<=|>|<|IS\b|IN\b)",
        text,
    ):
        col = match.group("col")
        if col.upper() in {
            "NOT", "EXISTS", "SELECT", "CASE", "WHEN", "NULLIF", "ISNULL", "COALESCE", "NVL",
        }:
            continue
        key = col
        if key.upper() not in seen:
            seen.add(key.upper())
            refs.append(key)
    return refs


def extract_merge_matched_updates(sql: str) -> list[dict[str, str]]:
    """Extract ``MERGE … WHEN MATCHED THEN UPDATE SET …`` assignment blocks.

    Supports SQL Server ``MERGE target USING source ON … WHEN MATCHED THEN UPDATE``
    and Oracle ``MERGE INTO target USING (…) ON (…) WHEN MATCHED THEN UPDATE``.
    """
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    # Find MERGE … USING … ON … WHEN MATCHED THEN UPDATE SET …
    merge_iter = re.finditer(r"(?is)\bMERGE\b(?:\s+INTO)?\s+", text)
    for merge_match in merge_iter:
        start = merge_match.start()
        i = merge_match.end()
        # Target table [AS alias]
        tgt_match = re.match(
            rf"(?is)(?P<target>{_TABLE_TOKEN})"
            rf"(?:\s+(?:AS\s+)?(?P<talias>[A-Za-z_][A-Za-z0-9_]*))?",
            text[i:],
        )
        if not tgt_match:
            continue
        target = normalize_table_name(tgt_match.group("target"))
        target_alias = tgt_match.group("talias") or ""
        i += tgt_match.end()

        # USING clause (table or subquery)
        using_match = re.match(r"(?is)\s*USING\b", text[i:])
        if not using_match:
            continue
        i += using_match.end()
        using_body, i = _read_until_keyword(text, i, {"ON"}, stop_at_statement=False)
        using_body = using_body.strip()
        source_alias = ""
        source_table = ""
        # USING #Temp AS Source  /  USING (subquery) SRC
        src_simple = re.match(
            rf"(?is)^(?P<table>{_TABLE_TOKEN})"
            rf"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?$",
            using_body,
        )
        if src_simple:
            source_table = normalize_table_name(src_simple.group("table"))
            source_alias = src_simple.group("alias") or ""
        else:
            alias_m = re.search(
                r"(?is)\)\s*(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\s*$",
                using_body,
            )
            if alias_m:
                source_alias = alias_m.group("alias")

        # ON condition
        on_match = re.match(r"(?is)\s*ON\b", text[i:])
        if not on_match:
            continue
        i += on_match.end()
        on_clause, i = _read_until_keyword(
            text, i, {"WHEN"}, stop_at_statement=False
        )
        on_clause = on_clause.strip().strip("()")

        # WHEN MATCHED THEN UPDATE SET …
        matched = re.match(
            r"(?is)\s*WHEN\s+MATCHED(?:\s+AND\s+.+?)?\s+THEN\s+UPDATE\s+SET\b",
            text[i:],
        )
        if not matched:
            continue
        i += matched.end()
        set_clause, end_i = _read_until_keyword(
            text,
            i,
            {"WHEN", "OUTPUT", "RETURN"},
            stop_at_statement=True,
        )
        set_clause = set_clause.strip().rstrip(";")
        results.append(
            {
                "target": target,
                "target_alias": target_alias,
                "source_table": source_table,
                "source_alias": source_alias,
                "using_body": using_body,
                "on_clause": on_clause,
                "set_clause": set_clause,
                "raw_sql": text[start:end_i].strip(),
                "start": str(start),
                "end": str(end_i),
            }
        )
    return results


def _extract_where_predicate(select_sql: str) -> str | None:
    text = select_sql or ""
    depth = 0
    in_single = False
    i = 0
    where_at = -1
    while i < len(text):
        ch = text[i]
        if in_single:
            if ch == "'":
                if i + 1 < len(text) and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and re.match(r"(?is)^WHERE\b", text[i:]):
            where_at = i + len("WHERE")
            break
        i += 1
    if where_at < 0:
        return None
    return text[where_at:].strip()


def _extract_having_clause(sql_fragment: str) -> str | None:
    text = sql_fragment or ""
    m = re.search(r"(?is)\bHAVING\b(.+?)(?=\bORDER\b|\bUNION\b|$)", text)
    if not m:
        return None
    return m.group(1).strip() or None


def _strip_group_order(pred: str) -> str:
    text = pred or ""
    text = re.split(r"(?is)\bGROUP\s+BY\b", text, maxsplit=1)[0]
    text = re.split(r"(?is)\bORDER\s+BY\b", text, maxsplit=1)[0]
    text = re.split(r"(?is)\bHAVING\b", text, maxsplit=1)[0]
    return text.strip()


def _split_select_list(select_sql: str) -> tuple[str, str]:
    text = select_sql or ""
    m = re.match(r"(?is)^SELECT\s+(?P<body>.+)$", text.strip())
    if not m:
        return "", text
    body = m.group("body")
    # Find top-level FROM
    depth = 0
    in_single = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_single:
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and re.match(r"(?is)^FROM\b", body[i:]):
            return body[:i].strip(), body[i:]
        i += 1
    return body.strip(), ""


def _first_from_table(from_rest: str) -> str | None:
    tables = parse_from_join_clause(from_rest or "")
    if tables:
        return tables[0][0]
    m = re.match(
        rf"(?is)^FROM\s+(?P<table>{_TABLE_TOKEN})",
        (from_rest or "").strip(),
    )
    if m:
        return normalize_table_name(m.group("table"))
    return None


def _parens_balanced(text: str) -> bool:
    depth = 0
    in_single = False
    for ch in text:
        if ch == "'" and not in_single:
            in_single = True
            continue
        if ch == "'" and in_single:
            in_single = False
            continue
        if in_single:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _read_until_keyword(
    text: str,
    start: int,
    keywords: set[str],
    *,
    stop_at_statement: bool,
) -> tuple[str, int]:
    """Read text from start until a top-level keyword or next statement."""
    buf: list[str] = []
    i = start
    n = len(text)
    depth = 0
    case_depth = 0
    in_single = False
    while i < n:
        ch = text[i]
        if in_single:
            buf.append(ch)
            if ch == "'":
                if i + 1 < n and text[i + 1] == "'":
                    buf.append("'")
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            buf.append(ch)
            i += 1
            continue
        if ch == "(":
            depth += 1
            buf.append(ch)
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
            i += 1
            continue
        if depth == 0:
            rest = text[i:]
            # A bare CASE ... END is not wrapped in parens, so it needs its
            # own nesting counter — otherwise the CASE's own closing END is
            # mistaken for the statement/block-terminating END below and the
            # read stops mid-expression, silently truncating everything after
            # it (including the real trailing FROM/WHERE clause).
            case_match = re.match(r"(?is)^CASE\b", rest)
            if case_match:
                case_depth += 1
                buf.append(case_match.group(0))
                i += len(case_match.group(0))
                continue
            end_match = re.match(r"(?is)^END\b", rest)
            if end_match and case_depth > 0:
                case_depth -= 1
                buf.append(end_match.group(0))
                i += len(end_match.group(0))
                continue
            if case_depth == 0:
                for kw in keywords:
                    if re.match(rf"(?is)^{kw}\b", rest):
                        return "".join(buf), i
                if stop_at_statement and _STMT_START.match(rest):
                    return "".join(buf), i
                # Bare END at depth 0 (outside any open CASE) is a
                # procedure/block terminator (END TRY / END CATCH / END) —
                # stop without consuming it.
                if stop_at_statement and end_match:
                    return "".join(buf), i
        buf.append(ch)
        i += 1
    return "".join(buf), i


def extract_select_into(sql: str) -> list[dict[str, str]]:
    """Find SELECT … INTO #temp FROM … statements."""
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    for match in re.finditer(r"(?is)\bSELECT\b", text):
        start = match.end()
        into_match = re.search(r"(?is)\bINTO\b", text[start:])
        if not into_match:
            continue
        select_list = text[start : start + into_match.start()]
        # Skip if this SELECT is clearly not a SELECT INTO (INTO too far / subquery-ish)
        if select_list.count("(") != select_list.count(")"):
            continue
        after_into = start + into_match.end()
        target_match = re.match(
            rf"(?is)\s*(?P<target>{_TABLE_TOKEN})",
            text[after_into:],
        )
        if not target_match:
            continue
        target = normalize_table_name(target_match.group("target"))
        if not target.startswith("#"):
            continue
        pos = after_into + target_match.end()
        from_clause = ""
        if re.match(r"(?is)^\s*FROM\b", text[pos:]):
            from_match = re.match(r"(?is)^\s*FROM\b", text[pos:])
            assert from_match is not None
            pos = pos + from_match.end()
            from_clause, pos = _read_until_keyword(
                text, pos, {"WHERE", "GROUP", "ORDER", "HAVING"}, stop_at_statement=True
            )
        results.append(
            {
                "target": target,
                "select_list": select_list.strip(),
                "from_body": from_clause.strip(),
            }
        )
    return results


def extract_cte_definitions(sql: str) -> list[dict[str, str]]:
    """Find ``[;]WITH name [(cols)] AS (body)`` common table expressions.

    Only single, non-recursive CTEs are extracted — enough to register the
    CTE name as a virtual lineage source for the statement that follows it
    (the common real-world shape: one CTE feeding one UPDATE/INSERT/SELECT),
    not full multi-CTE or recursive-CTE support. Without this, a CTE alias
    reference (``FROM CTE A ... A.SomeAggCol``) resolves through no lineage
    at all — the engine would treat the literal string ``CTE`` as if it
    were a real database table.
    """
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    for m in re.finditer(
        r"(?is);?\s*\bWITH\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
        r"(?:\((?P<cols>[^)]*)\))?\s*AS\s*\(",
        text,
    ):
        name = m.group("name")
        cols = m.group("cols") or ""
        body_start = m.end()
        depth = 1
        i = body_start
        n = len(text)
        in_single = False
        while i < n and depth > 0:
            ch = text[i]
            if in_single:
                if ch == "'":
                    if i + 1 < n and text[i + 1] == "'":
                        i += 2
                        continue
                    in_single = False
                i += 1
                continue
            if ch == "'":
                in_single = True
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            continue
        body = text[body_start:i]
        if not re.match(r"(?is)^\s*SELECT\b", body):
            continue
        results.append(
            {
                "name": name,
                "cols": cols.strip(),
                "body": body.strip(),
                "start": str(m.start()),
                "end": str(i + 1),
            }
        )
    return results


def split_select_from(select_sql: str) -> tuple[str, str]:
    """Split ``SELECT <list> FROM <rest>`` into ``(select_list, from_body)``.

    ``from_body`` has the leading ``FROM`` stripped and stops before any
    top-level ``WHERE``/``GROUP BY``/``ORDER BY``/``HAVING`` — the same
    shape ``extract_select_into``/``extract_insert_select`` already produce,
    so callers can feed it straight into ``parse_from_join_clause``.
    """
    m = re.match(r"(?is)^SELECT\s+(?P<rest>.+)$", (select_sql or "").strip())
    if not m:
        return "", ""
    rest = m.group("rest")
    depth = 0
    in_single = False
    i = 0
    n = len(rest)
    while i < n:
        ch = rest[i]
        if in_single:
            if ch == "'":
                if i + 1 < n and rest[i + 1] == "'":
                    i += 2
                    continue
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0 and re.match(r"(?is)^FROM\b", rest[i:]):
            select_list = rest[:i]
            rest_from = rest[i + 4 :]
            stop = re.search(r"(?is)\b(WHERE|GROUP\s+BY|ORDER\s+BY|HAVING)\b", rest_from)
            from_body = rest_from[: stop.start()] if stop else rest_from
            return select_list.strip(), from_body.strip()
        i += 1
    return rest.strip(), ""


def iter_set_assignments(set_clause: str) -> list[dict[str, str]]:
    """Split an ``UPDATE ... SET`` clause into column/alias/expr parts.

    Shared by phase2 (target-column mutation folding) and phase1 (temp-table
    mutation-checkpoint capture) so both phases parse ``SET`` identically.
    """
    assignments: list[dict[str, str]] = []
    for part in split_csv_respecting_parens(set_clause or ""):
        match = re.match(
            r"(?is)^\s*(?:(?P<alias>[#A-Za-z_][A-Za-z0-9_]*)\.)?"
            r"(?P<column>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*=\s*(?P<expr>.+?)\s*$",
            part.strip(),
        )
        if not match:
            continue
        assignments.append(
            {
                "alias": (match.group("alias") or "").strip(),
                "column": bare_ident(match.group("column")),
                "expr": match.group("expr").strip(),
            }
        )
    return assignments


def resolve_expression_column_refs(
    expression: str,
    alias_map: dict[str, str],
    lineage: Any,
    entity_map: dict[str, str] | None,
    default_entity: str,
) -> str:
    """Rewrite ``alias.col`` tokens using the given alias map + lineage resolver.

    ``lineage`` is duck-typed (only needs ``resolve_column(table, column,
    entity_map)``) to avoid a circular import with phase1's ``LineageMap``.
    When a resolved column carries a ``derived_formula`` (a temp-table
    mutation checkpoint recorded by phase1), that formula text is spliced in
    verbatim instead of a flat entity/column marker, so callers inherit the
    folded expression rather than the pre-mutation root value.

    Important: do **not** rewrite arbitrary ``schema.table`` tokens (e.g.
    ``PRO.AssetClassMovementHistory``) — those are not column refs.
    """

    def repl(match: re.Match[str]) -> str:
        qual = match.group("qual")
        col = bare_ident(match.group("col"))
        if qual.startswith("@") or col.startswith("@"):
            return match.group(0)
        table = alias_map.get(qual.upper())
        if table is None:
            if qual.startswith("#"):
                table = normalize_table_name(qual)
            else:
                return match.group(0)
        ref = lineage.resolve_column(table, col, entity_map)
        if getattr(ref, "derived_formula", None):
            return f"({ref.derived_formula})"
        relationship = ref.relationship
        if (
            relationship is None
            and ref.entity
            and default_entity
            and ref.entity.upper() != default_entity.upper()
            and (
                ref.entity.startswith("##")
                or str(table).startswith("##")
                or (resolve_entity_name(str(table), entity_map) or "").upper()
                != default_entity.upper()
            )
        ):
            if str(table).startswith("##"):
                rel = str(table)
            elif ref.source_table.startswith("##"):
                rel = ref.source_table
            else:
                rel = ref.entity if ref.entity.startswith("##") else f"##{ref.entity}"
            if (resolve_entity_name(str(table), entity_map) or "").upper() != default_entity.upper():
                return f"{default_entity}::{rel}::{ref.column}"
        if relationship:
            return f"{ref.entity}::{relationship}::{ref.column}"
        return f"{ref.entity}::{ref.column}"

    pattern = re.compile(
        r"(?P<qual>[#A-Za-z_][A-Za-z0-9_]*)\.(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)"
    )
    return pattern.sub(repl, expression or "")


def extract_insert_select(
    sql: str,
    *,
    temps_only: bool = False,
) -> list[dict[str, str]]:
    """Find ``INSERT INTO target [(cols)] SELECT … FROM … [WHERE …]``.

    By default includes permanent tables (needed for DD mutation collection).
    Pass ``temps_only=True`` for phase-1 local-temp lineage only.
    """
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    pattern = re.compile(
        rf"(?is)\bINSERT\s+INTO\s+(?P<target>{_TABLE_TOKEN})\s*"
        rf"(?:\((?P<cols>[^)]+)\))?\s*"
        rf"SELECT\b"
    )
    for match in pattern.finditer(text):
        target_raw = match.group("target")
        target = normalize_table_name(target_raw)
        if temps_only and not target.startswith("#"):
            continue
        cols = (match.group("cols") or "").strip()
        start = match.start()
        pos = match.end()
        select_list, pos = _read_until_keyword(text, pos, {"FROM"}, stop_at_statement=True)
        from_body = ""
        where_clause = ""
        if re.match(r"(?is)^FROM\b", text[pos:]):
            pos += len("FROM")
            from_body, pos = _read_until_keyword(
                text,
                pos,
                {"WHERE", "GROUP", "ORDER", "HAVING"},
                stop_at_statement=True,
            )
        if re.match(r"(?is)^WHERE\b", text[pos:]):
            pos += len("WHERE")
            where_clause, pos = _read_until_keyword(
                text, pos, {"GROUP", "ORDER", "HAVING"}, stop_at_statement=True
            )
        results.append(
            {
                "target": target,
                "target_raw": (target_raw or "").strip(),
                "cols": cols,
                "select_list": select_list.strip(),
                "from_body": from_body.strip(),
                "where_clause": where_clause.strip(),
                "raw_sql": text[start:pos].strip(),
                "start": str(start),
                "end": str(pos),
            }
        )
    return results
