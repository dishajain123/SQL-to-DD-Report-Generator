"""Splits a PL/SQL / procedural-SQL object body into individual statements,
and parses each DML statement with sqlglot to extract tables/columns.

Statement splitting is done with a quote/paren-aware character scan rather
than a naive semicolon split, because subquery text can itself contain
identifiers that look like keywords. Semicolons only terminate a real
statement when they sit at paren-depth 0 and outside any quoted string.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from app.models.core import Dialect, StatementInfo

_DML_KEYWORDS = ("SELECT", "UPDATE", "MERGE", "INSERT", "DELETE")
_CONTROL_KEYWORDS = ("IF", "BEGIN", "EXCEPTION", "DECLARE", "END", "CASE", "LOOP")

_SQLGLOT_DIALECT = {Dialect.ORACLE: "oracle", Dialect.MYSQL: "mysql"}


def split_statements(raw_sql: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    paren_depth = 0
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    i = 0
    n = len(raw_sql)
    while i < n:
        ch = raw_sql[i]
        buf.append(ch)

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
        elif in_block_comment:
            if ch == "*" and i + 1 < n and raw_sql[i + 1] == "/":
                buf.append(raw_sql[i + 1])
                i += 1
                in_block_comment = False
        elif in_single:
            if ch == "'" and (i + 1 >= n or raw_sql[i + 1] != "'"):
                in_single = False
            elif ch == "'" and i + 1 < n and raw_sql[i + 1] == "'":
                buf.append(raw_sql[i + 1])
                i += 1
        elif in_double:
            if ch == '"':
                in_double = False
        elif ch == "-" and i + 1 < n and raw_sql[i + 1] == "-":
            in_line_comment = True
            buf.append(raw_sql[i + 1])
            i += 1
        elif ch == "/" and i + 1 < n and raw_sql[i + 1] == "*":
            in_block_comment = True
            buf.append(raw_sql[i + 1])
            i += 1
        elif ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif ch == ";" and paren_depth == 0:
            stmt = "".join(buf).strip()
            if stmt.strip(";").strip():
                statements.append(stmt)
            buf = []
        i += 1

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    return statements


def classify_statement(stmt_text: str) -> str:
    stripped = _strip_leading_comments(stmt_text)
    first_word_match = re.match(r"[A-Za-z]+", stripped)
    first_word = first_word_match.group(0).upper() if first_word_match else ""

    if first_word in _DML_KEYWORDS:
        return first_word
    if first_word in _CONTROL_KEYWORDS:
        return "CONTROL_FLOW"
    return "OTHER"


def _strip_leading_comments(text: str) -> str:
    """Strip leading whitespace, `--` line comments, and `/* */` block
    comments so keyword detection isn't fooled by a comment preceding the
    actual statement (very common in these procs)."""
    pos = 0
    n = len(text)
    while pos < n:
        ch = text[pos]
        if ch.isspace():
            pos += 1
        elif text[pos:pos + 2] == "--":
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
        elif text[pos:pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = n if end == -1 else end + 2
        else:
            break
    return text[pos:]


def parse_statement(stmt_text: str, index: int, dialect: Dialect) -> StatementInfo:
    stmt_type = classify_statement(stmt_text)

    info = StatementInfo(
        statement_index=index,
        statement_type=stmt_type,
        raw_text=stmt_text,
    )

    if stmt_type not in _DML_KEYWORDS:
        info.conditions = _extract_conditions(stmt_text)
        return info

    try:
        tree = sqlglot.parse_one(_strip_leading_comments(stmt_text), read=_SQLGLOT_DIALECT[dialect])
    except Exception as exc:  # sqlglot raises various ParseError subtypes
        info.parsed_ok = False
        info.parse_error = str(exc)
        info.conditions = _extract_conditions(stmt_text)
        return info

    if tree is None:
        info.parsed_ok = False
        info.parse_error = "sqlglot returned no expression tree"
        return info

    info.tables_read, info.tables_written = _tables_from_tree(tree, stmt_type)
    info.columns = sorted({c.name for c in tree.find_all(exp.Column) if c.name})
    info.set_columns_by_table = _set_columns_by_table(tree, stmt_type, info.tables_written)
    info.join_tables, info.join_conditions = _join_info_from_tree(tree)
    info.conditions = _extract_conditions(stmt_text)
    return info


def _tables_from_tree(tree: exp.Expression, stmt_type: str) -> tuple[list[str], list[str]]:
    all_tables = sorted({t.name for t in tree.find_all(exp.Table) if t.name})

    written: set[str] = set()
    if stmt_type in ("UPDATE", "INSERT", "DELETE"):
        target = tree.find(exp.Table)
        if target is not None and target.name:
            written.add(target.name)
    elif stmt_type == "MERGE":
        merge_target = tree.this
        if isinstance(merge_target, exp.Table) and merge_target.name:
            written.add(merge_target.name)

    read = sorted(set(all_tables) - written)
    return read, sorted(written)


def _set_columns_by_table(
    tree: exp.Expression, stmt_type: str, tables_written: list[str]
) -> dict[str, list[str]]:
    """Extract the actual SET-clause target columns for UPDATE/MERGE
    statements, i.e. what's really being derived/assigned -- as opposed to
    every column mentioned anywhere in the statement (WHERE/JOIN included)."""
    target_table = tables_written[0] if tables_written else None
    if target_table is None:
        return {}

    columns: set[str] = set()

    if stmt_type == "UPDATE":
        update_node = tree if isinstance(tree, exp.Update) else tree.find(exp.Update)
        if update_node is not None:
            for assignment in update_node.args.get("expressions", []):
                if isinstance(assignment, exp.EQ) and isinstance(assignment.this, exp.Column):
                    columns.add(assignment.this.name)

    elif stmt_type == "MERGE":
        merge_node = tree if isinstance(tree, exp.Merge) else tree.find(exp.Merge)
        whens = merge_node.args.get("whens") if merge_node is not None else None
        when_list = whens.expressions if whens is not None else []
        for when in when_list:
            then = when.args.get("then")
            if isinstance(then, exp.Update):
                for assignment in then.args.get("expressions", []):
                    if isinstance(assignment, exp.EQ) and isinstance(assignment.this, exp.Column):
                        columns.add(assignment.this.name)

    return {target_table: sorted(columns)} if columns else {}


def _join_info_from_tree(tree: exp.Expression) -> tuple[list[str], list[str]]:
    join_tables: set[str] = set()
    join_conditions: list[str] = []

    for join in tree.find_all(exp.Join):
        join_target = join.this
        if join_target is not None:
            for table in join_target.find_all(exp.Table):
                if table.name:
                    join_tables.add(table.name)

        on_clause = join.args.get("on")
        if on_clause is not None:
            try:
                rendered = on_clause.sql()
            except Exception:
                rendered = str(on_clause)
            cleaned = _clean(rendered)
            if cleaned:
                join_conditions.append(cleaned)

    return sorted(join_tables), join_conditions


_CONDITION_RE = re.compile(
    r"\bIF\b\s*\(?(.+?)\)?\s*THEN\b", re.IGNORECASE | re.DOTALL
)
_CASE_WHEN_RE = re.compile(r"\bWHEN\b\s+(.+?)\s+THEN\b", re.IGNORECASE | re.DOTALL)


def _extract_conditions(stmt_text: str) -> list[str]:
    conditions = []
    for m in _CONDITION_RE.finditer(stmt_text):
        conditions.append(_clean(m.group(1)))
    for m in _CASE_WHEN_RE.finditer(stmt_text):
        conditions.append(_clean(m.group(1)))
    return conditions


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:300]
