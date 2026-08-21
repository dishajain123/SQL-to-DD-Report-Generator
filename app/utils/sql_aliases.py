from __future__ import annotations

import re
from collections import defaultdict

import sqlglot
from sqlglot import exp

from app.models.core import Dialect
from app.parsing.sql_parser import split_statements

# The single shared mapping from our internal Dialect enum to the dialect
# name sqlglot actually recognizes. sqlglot's SQL Server dialect is named
# "tsql", not "sqlserver" -- passing dialect.value directly (as this module
# previously did) causes sqlglot.parse_one to raise "Unknown dialect" on
# every single T-SQL statement, which was being silently swallowed by a
# bare `except Exception: continue` in collect_table_aliases. That silent
# failure -- not any control-flow bypass -- is why the alias resolver
# appeared to do nothing for SQL Server sources despite "already being
# implemented": it never successfully parsed a single statement to begin
# with. Any other module that needs a sqlglot dialect name for a Dialect
# value should import this mapping rather than defining its own, so there
# is exactly one place this translation lives.
SQLGLOT_DIALECT_MAP: dict[Dialect, str] = {
    Dialect.ORACLE: "oracle",
    Dialect.MYSQL: "mysql",
    Dialect.SQLSERVER: "tsql",
}


def _sqlglot_dialect_name(dialect: Dialect) -> str:
    if isinstance(dialect, Dialect):
        return SQLGLOT_DIALECT_MAP.get(dialect, dialect.value)
    return str(dialect)


_QUALIFIED_REF_RE = re.compile(
    r'(?P<alias>"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)'
    r'(?P<tail>(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))+)',
)


def _exact_identifier_text(node) -> str:
    if isinstance(node, exp.TableAlias):
        node = node.this
    if isinstance(node, exp.Identifier):
        return str(node.this)
    if isinstance(node, str):
        return node.strip('"')
    return str(node)


def _table_reference_parts(table: exp.Table) -> tuple[str, ...]:
    parts: list[str] = []
    for key in ("catalog", "db", "this"):
        value = table.args.get(key)
        if value is None:
            continue
        text = _exact_identifier_text(value)
        if text:
            parts.append(text)
    return tuple(parts)


def _table_alias_name(table: exp.Table) -> str | None:
    alias = table.args.get("alias")
    if isinstance(alias, exp.TableAlias):
        alias = alias.this
    if isinstance(alias, exp.Identifier):
        alias = alias.this
    if isinstance(alias, str) and alias.strip():
        return alias.strip()
    return None


def collect_table_aliases(text: str, dialect: Dialect) -> dict[str, tuple[str, ...]]:
    """Return a case-insensitive alias -> exact table reference map.

    Only real base-table aliases are included. Ambiguous aliases that map
    to more than one distinct table reference across the provided text are
    dropped rather than guessed.
    """
    if not text:
        return {}

    alias_to_parts: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    dialect_name = _sqlglot_dialect_name(dialect)

    for stmt in split_statements(text, dialect):
        cleaned_stmt = stmt.strip()
        if not cleaned_stmt:
            continue
        try:
            tree = sqlglot.parse_one(cleaned_stmt, read=dialect_name)
        except Exception:
            continue

        for table in tree.find_all(exp.Table):
            alias = _table_alias_name(table)
            if not alias:
                continue
            parts = _table_reference_parts(table)
            if not parts:
                continue
            alias_to_parts[alias.upper()].add(parts)

    return {
        alias: next(iter(parts_set))
        for alias, parts_set in alias_to_parts.items()
        if len(parts_set) == 1
    }


def collect_known_reference_names(text: str, dialect: Dialect) -> set[str]:
    """Return the case-insensitive set of identifier names actually parsed
    out of the source SQL: column names, declared/bound parameters, and
    table/alias names.

    This is the single source of truth for "is this token a real source
    reference or a literal constant?" -- used by dependency extraction so
    that classification never relies on a hardcoded list of known business
    values (which can never be complete). A token not in this set, when it
    appears as a bare quoted literal-shaped value in a generated formula,
    is a literal; a token that IS in this set is a genuine source reference.
    """
    if not text:
        return set()

    names: set[str] = set()
    dialect_name = _sqlglot_dialect_name(dialect)

    for stmt in split_statements(text, dialect):
        cleaned_stmt = stmt.strip()
        if not cleaned_stmt:
            continue
        try:
            tree = sqlglot.parse_one(cleaned_stmt, read=dialect_name)
        except Exception:
            continue

        for column in tree.find_all(exp.Column):
            ident = column.args.get("this")
            text_value = _exact_identifier_text(ident) if ident is not None else None
            if text_value:
                names.add(text_value.upper())

        for param in tree.find_all(exp.Parameter):
            text_value = _exact_identifier_text(param.this) if param.this is not None else None
            if not text_value:
                text_value = str(param.this).strip() if param.this is not None else None
            if text_value:
                names.add(text_value.lstrip("@").upper())

        for table in tree.find_all(exp.Table):
            for part in _table_reference_parts(table):
                names.add(part.upper())
            alias = _table_alias_name(table)
            if alias:
                names.add(alias.upper())

    return names


def render_exact_table_reference(parts: tuple[str, ...], quoted: bool = False) -> str:
    if not parts:
        return ""
    if quoted:
        return ".".join(f'"{part}"' for part in parts)
    return ".".join(parts)


def resolve_aliases_in_expression(
    expression: str,
    alias_to_parts: dict[str, tuple[str, ...]],
    *,
    quote_replacements: bool = False,
) -> str:
    """Replace qualified alias references with the original source table.

    Aliases are matched case-insensitively. If a replacement cannot be
    determined safely, the original text is left unchanged.
    """
    if not expression or not alias_to_parts:
        return expression

    alias_lookup = {alias.upper(): parts for alias, parts in alias_to_parts.items() if parts}
    result: list[str] = []
    i = 0
    n = len(expression)

    while i < n:
        ch = expression[i]
        if ch == "'":
            result.append(ch)
            i += 1
            while i < n:
                result.append(expression[i])
                if expression[i] == "'" and not (i + 1 < n and expression[i + 1] == "'"):
                    i += 1
                    break
                if expression[i] == "'" and i + 1 < n and expression[i + 1] == "'":
                    result.append(expression[i + 1])
                    i += 2
                    continue
                i += 1
            continue

        match = _QUALIFIED_REF_RE.match(expression, i)
        if match:
            alias_token = match.group("alias")
            alias_name = alias_token[1:-1] if alias_token.startswith('"') and alias_token.endswith('"') else alias_token
            replacement_parts = alias_lookup.get(alias_name.upper())
            if replacement_parts is not None:
                tail = match.group("tail")
                tail_upper = tail.replace('"', "").replace(" ", "").upper()
                if tail_upper.startswith(".VAR.BUSINESS_DATE"):
                    result.append(match.group(0))
                else:
                    result.append(render_exact_table_reference(replacement_parts, quoted=quote_replacements or alias_token.startswith('"')))
                    result.append(tail)
                i = match.end()
                continue

        result.append(ch)
        i += 1

    return "".join(result)