from __future__ import annotations

import re
from collections import defaultdict

import sqlglot
from sqlglot import exp

from app.models.core import Dialect
from app.parsing.sql_parser import split_statements

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
    dialect_name = dialect.value if isinstance(dialect, Dialect) else str(dialect)

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
