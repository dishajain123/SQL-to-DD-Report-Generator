"""Splits a PL/SQL / procedural-SQL object body into individual statements,
and parses each DML statement with sqlglot to extract tables/columns.

Statement splitting is done with a quote/paren-aware character scan rather
than a naive semicolon split, because subquery text can itself contain
identifiers that look like keywords. Semicolons only terminate a real
statement when they sit at paren-depth 0 and outside any quoted string.

A procedural control-flow header (`IF ... THEN`, `ELSIF ... THEN`, `ELSE`,
`BEGIN`, or an exception handler's `EXCEPTION` / `WHEN ... THEN`) has no
semicolon of its own, so pure semicolon-splitting glues it onto whatever
DML statement follows as a single blob. Left alone, that blob's leading
word is the control-flow keyword, so it gets classified as CONTROL_FLOW
instead of its real statement type, and the DML statement inside it -- and
every column it assigns -- is silently invisible to every later stage
(structural analysis, smart chunking, DD row generation). A dedicated pass
after the semicolon split peels any such header off, so both halves are
correctly classified and analyzed.
"""
from __future__ import annotations

import re

import sqlglot
from sqlglot import exp

from app.models.core import Dialect, StatementInfo

_DML_KEYWORDS = ("SELECT", "UPDATE", "MERGE", "INSERT", "DELETE")
_CONTROL_KEYWORDS = ("IF", "BEGIN", "EXCEPTION", "DECLARE", "END", "CASE", "LOOP", "WHEN")

_SQLGLOT_DIALECT = {
    Dialect.ORACLE: "oracle",
    Dialect.MYSQL: "mysql",
    Dialect.SQLSERVER: "tsql",
}

_TSQL_STATEMENT_START_KEYWORDS = (
    "IF",
    "BEGIN",
    "END",
    "DECLARE",
    "SET",
    "SELECT",
    "UPDATE",
    "INSERT",
    "DELETE",
    "MERGE",
    "EXEC",
    "EXECUTE",
    "DROP",
    "CREATE",
    "TRUNCATE",
    "RETURN",
    "THROW",
    "RAISERROR",
    "WHILE",
)

# Headers that are complete on their own (nothing to search for beyond the
# keyword itself) -- the split point is right after the keyword.
_HEADER_ONLY_KEYWORDS = ("ELSE", "EXCEPTION", "BEGIN")
# Headers that continue up through a "THEN" -- the split point is right
# after the first top-level THEN following the keyword.
_HEADER_THEN_KEYWORDS = ("IF", "ELSIF", "ELSEIF", "WHEN")


def split_statements(raw_sql: str, dialect: Dialect | None = None) -> list[str]:
    if dialect is None:
        dialect = _infer_dialect(raw_sql)

    statements = _split_semicolon_statements(raw_sql)
    if dialect == Dialect.SQLSERVER:
        statements = _split_tsql_batches(statements)
    return _split_glued_control_flow(statements)


def _split_semicolon_statements(raw_sql: str) -> list[str]:
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


def _split_tsql_batches(statements: list[str]) -> list[str]:
    result: list[str] = []
    for stmt in statements:
        result.extend(_split_tsql_statement(stmt))
    return result


def _split_tsql_statement(stmt: str) -> list[str]:
    lines = stmt.splitlines(keepends=True)
    if len(lines) <= 1:
        return [stmt]

    result: list[str] = []
    buf: list[str] = []
    paren_depth = 0
    in_single = False
    in_double = False
    in_block_comment = False

    for line in lines:
        stripped = _strip_leading_comments(line)
        lead = re.match(r"[A-Za-z]+", stripped).group(0).upper() if re.match(r"[A-Za-z]+", stripped) else ""
        is_go = bool(re.match(r"^\s*GO(?:\s+\d+)?\s*(?:--.*)?$", stripped, re.IGNORECASE))
        buffer_text = "".join(buf).strip()
        buffer_lead = _leading_keyword_ignoring_comments(buffer_text) if buffer_text else ""

        if is_go:
            if buffer_text:
                result.append(buffer_text)
                buf = []
            continue

        should_split = (
            bool(buf)
            and paren_depth == 0
            and not in_single
            and not in_double
            and not in_block_comment
            and lead in _TSQL_STATEMENT_START_KEYWORDS
        )
        # An UPDATE ... SET col1=x, col2=y FROM ... WHERE ... statement
        # (the common T-SQL multi-column UPDATE-FROM shape) puts SET on its
        # own line. SET is otherwise a valid statement-start keyword (for
        # standalone `SET @var = x`), so without this carve-out the splitter
        # treats that SET line as the start of a brand-new statement and
        # shreds one real UPDATE into two meaningless fragments: "UPDATE tbl"
        # and "SET ... FROM ... WHERE ...". Mirrors the WITH->SELECT
        # carve-out immediately below for the same reason.
        is_update_awaiting_set = buffer_lead == "UPDATE" and lead == "SET"
        if should_split and not (buffer_lead == "WITH" and lead == "SELECT") and not is_update_awaiting_set:
            result.append(buffer_text)
            buf = []

        buf.append(line)
        paren_depth, in_single, in_double, in_block_comment = _scan_text_state(
            line, paren_depth, in_single, in_double, in_block_comment
        )

    tail = "".join(buf).strip()
    if tail:
        result.append(tail)
    return [stmt for stmt in result if stmt.strip()]


def _scan_text_state(
    text: str,
    paren_depth: int,
    in_single: bool,
    in_double: bool,
    in_block_comment: bool,
) -> tuple[int, bool, bool, bool]:
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_block_comment:
            if ch == "*" and i + 1 < n and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single:
            if ch == "'" and not (i + 1 < n and text[i + 1] == "'"):
                in_single = False
            elif ch == "'" and i + 1 < n and text[i + 1] == "'":
                i += 1
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            break
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
        elif ch == '"':
            in_double = True
        elif ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth = max(0, paren_depth - 1)
        i += 1
    return paren_depth, in_single, in_double, in_block_comment


def _infer_dialect(raw_sql: str) -> Dialect:
    try:
        from app.parsing.dialect import detect_dialect
    except Exception:
        return Dialect.ORACLE
    return detect_dialect(raw_sql)


def _split_glued_control_flow(statements: list[str]) -> list[str]:
    """Peel any leading control-flow header off of each statement,
    emitting the header and the real statement as separate list entries.
    Applies repeatedly per statement so multiple stacked headers (for
    example `EXCEPTION` immediately followed by `WHEN OTHERS THEN`) are
    each split out on their own, the same way regardless of which
    procedure or how many headers happen to be glued together.
    """
    result: list[str] = []
    for stmt in statements:
        remainder = stmt
        while remainder:
            lead = _leading_keyword_ignoring_comments(remainder)

            if lead in _HEADER_ONLY_KEYWORDS:
                header_end = _leading_keyword_end_index(remainder)
                if header_end == -1:
                    break
                header = remainder[:header_end]
                rest = remainder[header_end:].strip()
                result.append(header)
                remainder = rest
                continue

            if lead in _HEADER_THEN_KEYWORDS:
                search_start = _leading_keyword_end_index(remainder)
                if search_start == -1:
                    break
                then_end = _find_top_level_keyword(remainder, "THEN", start=search_start)
                if then_end == -1:
                    break
                header = remainder[:then_end]
                rest = remainder[then_end:].strip()
                result.append(header)
                remainder = rest
                continue

            break

        if remainder:
            result.append(remainder)

    return result


def _leading_keyword_end_index(text: str) -> int:
    """Return the index in `text` right after its leading keyword (past any
    leading comments/whitespace), or -1 if there is no leading keyword."""
    pos = 0
    n = len(text)
    while pos < n:
        ch = text[pos]
        if ch.isspace():
            pos += 1
        elif text[pos : pos + 2] == "--":
            nl = text.find("\n", pos)
            pos = n if nl == -1 else nl + 1
        elif text[pos : pos + 2] == "/*":
            end = text.find("*/", pos + 2)
            pos = n if end == -1 else end + 2
        else:
            break
    match = re.match(r"[A-Za-z]+", text[pos:])
    if not match:
        return -1
    return pos + len(match.group(0))


def _find_top_level_keyword(text: str, keyword: str, start: int = 0) -> int:
    """Return the index right after the first case-insensitive, word-
    boundary occurrence of `keyword` in `text` starting from `start`,
    respecting quoted strings and comments so a keyword inside a string
    literal or comment is never matched. Returns -1 if not found."""
    n = len(text)
    i = start
    in_single = False
    in_double = False
    in_line_comment = False
    in_block_comment = False
    kw_upper = keyword.upper()
    kw_len = len(keyword)
    while i < n:
        ch = text[i]
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue
        if in_block_comment:
            if ch == "*" and i + 1 < n and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
        if in_single:
            if ch == "'" and not (i + 1 < n and text[i + 1] == "'"):
                in_single = False
            elif ch == "'" and i + 1 < n and text[i + 1] == "'":
                i += 1
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            in_line_comment = True
            i += 2
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue
        if text[i : i + kw_len].upper() == kw_upper:
            before_ok = i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            after_idx = i + kw_len
            after_ok = after_idx >= n or not (text[after_idx].isalnum() or text[after_idx] == "_")
            if before_ok and after_ok:
                return after_idx
        i += 1
    return -1


def classify_statement(stmt_text: str) -> str:
    stripped = _strip_leading_comments(stmt_text)
    first_word_match = re.match(r"[A-Za-z]+", stripped)
    first_word = first_word_match.group(0).upper() if first_word_match else ""

    if first_word == "WITH":
        return "SELECT"
    if first_word in _DML_KEYWORDS:
        return first_word
    if first_word in _CONTROL_KEYWORDS:
        return "CONTROL_FLOW"
    return "OTHER"


def _leading_keyword_ignoring_comments(text: str) -> str:
    stripped = _strip_leading_comments(text)
    match = re.match(r"[A-Za-z]+", stripped)
    return match.group(0).upper() if match else ""


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

    stmt_type = _statement_type_from_tree(tree, stmt_type)
    info.tables_read, info.tables_written = _tables_from_tree(tree, stmt_type)
    info.columns = sorted({c.name for c in tree.find_all(exp.Column) if c.name})
    info.set_columns_by_table = _set_columns_by_table(tree, stmt_type, info.tables_written)
    info.join_tables, info.join_conditions = _join_info_from_tree(tree)
    info.conditions = _extract_conditions(stmt_text)
    return info


def _statement_type_from_tree(tree: exp.Expression, fallback: str) -> str:
    if isinstance(tree, exp.Update):
        return "UPDATE"
    if isinstance(tree, exp.Insert):
        return "INSERT"
    if isinstance(tree, exp.Delete):
        return "DELETE"
    if isinstance(tree, exp.Merge):
        return "MERGE"
    if isinstance(tree, exp.Select):
        return "SELECT"
    return fallback


def _tables_from_tree(tree: exp.Expression, stmt_type: str) -> tuple[list[str], list[str]]:
    table_nodes = [t for t in tree.find_all(exp.Table) if t.name]
    all_tables = sorted({t.name for t in table_nodes})

    written: set[str] = set()
    if stmt_type in ("UPDATE", "INSERT", "DELETE"):
        target_name, target_alias, exclude_names = _resolve_target_table_name(tree, stmt_type)
        if target_name:
            written.add(target_name)
        if exclude_names:
            all_tables = sorted({t for t in all_tables if t not in exclude_names})
    elif stmt_type == "MERGE":
        merge_target = tree.this
        if isinstance(merge_target, exp.Table) and merge_target.name:
            written.add(merge_target.name)
    elif stmt_type == "SELECT":
        into = tree.args.get("into") if isinstance(tree, exp.Select) else None
        into_table = into.this if into is not None else None
        if isinstance(into_table, exp.Table) and into_table.name:
            written.add(into_table.name)

    read = sorted(set(all_tables) - written)
    return read, sorted(written)


def _resolve_target_table_name(tree: exp.Expression, stmt_type: str) -> tuple[str | None, str | None, set[str]]:
    if stmt_type != "UPDATE":
        return None, None, set()

    update_node = tree if isinstance(tree, exp.Update) else tree.find(exp.Update)
    if update_node is None:
        return None, None, set()

    target = update_node.this
    if not isinstance(target, exp.Table):
        return None, None, set()

    target_name = target.name or None
    target_alias = target.alias_or_name or None
    exclude_names: set[str] = set()

    if target_name and target_alias and target_name == target_alias:
        from_clause = update_node.args.get("from_")
        if from_clause is not None:
            for candidate in from_clause.find_all(exp.Table):
                if candidate is target:
                    continue
                if candidate.alias_or_name and candidate.alias_or_name.upper() == target_alias.upper():
                    if candidate.name:
                        target_name = candidate.name
                        exclude_names.add(target_alias)
                        break

    return target_name, target_alias, exclude_names


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

    if stmt_type == "SELECT":
        into = tree.args.get("into") if isinstance(tree, exp.Select) else None
        if into is not None:
            for expr in tree.args.get("expressions", []) or []:
                column_name = _projection_column_name(expr)
                if column_name:
                    columns.add(column_name)

    if stmt_type == "UPDATE":
        update_node = tree if isinstance(tree, exp.Update) else tree.find(exp.Update)
        if update_node is not None:
            for assignment in update_node.args.get("expressions", []):
                if isinstance(assignment, exp.EQ) and isinstance(assignment.this, exp.Column):
                    columns.add(assignment.this.name)

    elif stmt_type == "INSERT":
        insert_node = tree if isinstance(tree, exp.Insert) else tree.find(exp.Insert)
        if insert_node is not None:
            schema = insert_node.this
            if isinstance(schema, exp.Schema):
                for column in schema.expressions or []:
                    if isinstance(column, exp.Column) and column.name:
                        columns.add(column.name)
                    elif isinstance(column, exp.Identifier) and column.this:
                        columns.add(column.this)

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
            elif isinstance(then, exp.Insert):
                schema = then.this
                if isinstance(schema, exp.Schema):
                    for column in schema.expressions or []:
                        if isinstance(column, exp.Column) and column.name:
                            columns.add(column.name)
                        elif isinstance(column, exp.Identifier) and column.this:
                            columns.add(column.this)

    return {target_table: sorted(columns)} if columns else {}


def _projection_column_name(expr: exp.Expression) -> str | None:
    if isinstance(expr, exp.Alias):
        alias = expr.alias
        return alias if alias else None
    if isinstance(expr, exp.Column):
        return expr.name or None
    if isinstance(expr, exp.Identifier):
        return expr.this or None
    return None


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