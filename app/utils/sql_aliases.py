from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

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

    Memoized: this is called once per DD column being generated (see
    # Each time re-running split_statements plus a
    sqlglot.parse_one per statement over the SAME whole-procedure text --
    for a large procedure (hundreds of written columns, dozens of
    statements) that is tens of thousands of redundant full parses and was
    measured to make procedure-scale generation not finish in any
    reasonable time. The same (text, dialect) pair recurs constantly
    across columns of the same object, so caching this pure function (no
    logic change, same result every time for the same input) is a direct,
    safe fix for that -- verified end-to-end: >1500s (did not finish) ->
    267s on the procedure that originally triggered this. maxsize is
    bounded so a long-running server process doesn't accumulate unbounded
    cache entries across many different jobs/procedures over its lifetime.
    """
    return _collect_table_aliases_cached(text, dialect)


@lru_cache(maxsize=256)
def _collect_table_aliases_cached(text: str, dialect: Dialect) -> dict[str, tuple[str, ...]]:
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

        # MERGE/FROM derived-table aliases (`USING (SELECT ... FROM T ...) SRC`)
        # are not Table nodes. Map them to the subquery's primary base table so
        # `"SRC"."Col"` / `"C"."Col"` resolve instead of leaking single-letter
        # aliases into Platform Conditions.
        for subquery in tree.find_all(exp.Subquery):
            alias_node = subquery.args.get("alias")
            alias = None
            if isinstance(alias_node, exp.TableAlias):
                alias = _exact_identifier_text(alias_node)
            elif isinstance(alias_node, exp.Identifier):
                alias = _exact_identifier_text(alias_node)
            elif isinstance(alias_node, str) and alias_node.strip():
                alias = alias_node.strip()
            if not alias:
                continue
            select = subquery.this if isinstance(subquery.this, exp.Select) else None
            if not isinstance(select, exp.Select):
                continue
            from_clause = select.args.get("from_")
            if not isinstance(from_clause, exp.From):
                continue
            source = from_clause.this
            if isinstance(source, exp.Table):
                parts = _table_reference_parts(source)
                if parts:
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


_QUOTED_SEGMENT_RE = re.compile(r'"([^"]+)"')


def rewrite_expression_to_platform_entities(
    expression: str,
    *,
    entity_name: str,
    entity_name_map: dict[str, str] | None = None,
    alias_to_parts: dict[str, tuple[str, ...]] | None = None,
) -> str:
    """Rewrite SQL schema/table qualifiers into platform entity qualifiers.

    Platform Formula Expressions use `"Entity"."Column"` (or
    `"Entity"."rel"."Column"` / `"Entity"."var"."Name"`), never SQL schema
    prefixes such as `"PRO"."AccountCal"."Column"`. This step is applied
    after alias resolution so the DD export matches the Derivations schema
    observed in the platform sample export.
    """
    if not expression:
        return expression

    entity_name = (entity_name or "").strip().strip('"')
    entity_name_map = entity_name_map or {}
    alias_to_parts = alias_to_parts or {}

    # Map every known source table identifier (bare table, schema.table, and
    # each path component) onto the platform entity name for that table.
    table_to_entity: dict[str, str] = {}

    def remember(source_name: str, mapped_entity: str) -> None:
        key = source_name.strip().strip('"')
        if not key or not mapped_entity:
            return
        table_to_entity[key.upper()] = mapped_entity

    for source_table, mapped in entity_name_map.items():
        remember(source_table, mapped)
        remember(mapped, mapped)

    if entity_name:
        remember(entity_name, entity_name)

    for parts in alias_to_parts.values():
        if not parts:
            continue
        table_name = parts[-1]
        mapped = entity_name_map.get(table_name, table_name)
        # Prefer the caller's target entity when this alias points at the
        # same logical table the column is being generated for.
        if entity_name and table_name.upper() == entity_name.upper():
            mapped = entity_name
        elif entity_name and entity_name_map.get(table_name, table_name).upper() == entity_name.upper():
            mapped = entity_name
        remember(table_name, mapped)
        remember(".".join(parts), mapped)
        for part in parts:
            # Schema-only tokens like PRO must never become standalone
            # entity rewrites; only remember multi-part and table names.
            if part.upper() == table_name.upper():
                remember(part, mapped)

    if not table_to_entity:
        return expression

    # Longest source keys first so `"PRO"."AccountCal"` wins over `"PRO"`.
    source_keys = sorted(table_to_entity.keys(), key=len, reverse=True)

    def replace_leading_qualifier(match: re.Match[str]) -> str:
        segments = _QUOTED_SEGMENT_RE.findall(match.group(0))
        if len(segments) < 2:
            return match.group(0)

        # Preserve platform temp/business-date conventions already in entity form.
        if len(segments) >= 3 and segments[1].upper() == "VAR":
            if entity_name and segments[0].upper() != entity_name.upper():
                # Only rewrite the leading qualifier when it is a known source table.
                leading_key = segments[0].upper()
                dotted_key = ".".join(segments[:2]).upper()
                replacement = table_to_entity.get(dotted_key) or table_to_entity.get(leading_key)
                if replacement:
                    return '"' + replacement + '"' + "".join(f'."{seg}"' for seg in segments[1:])
            return match.group(0)

        # Platform formulas use `"Entity"."Column"` (exactly two segments).
        # Only rewrite the leading qualifier; never collapse the path to a
        # single token by treating the column name as a table (that produced
        # `"FeeSchedule"."LateFee"` → `"LateFee"` when `LateFee` was wrongly
        # present in the entity map).
        if len(segments) == 2:
            dotted = ".".join(segments).upper()
            if dotted in table_to_entity:
                return f'"{table_to_entity[dotted]}"'
            replacement = table_to_entity.get(segments[0].upper())
            if replacement:
                return f'"{replacement}"."{segments[1]}"'
            return match.group(0)

        for width in (2, 1):
            if len(segments) < width + 1:
                continue
            leading = ".".join(segments[:width]).upper()
            replacement = table_to_entity.get(leading)
            if replacement:
                return '"' + replacement + '"' + "".join(f'."{seg}"' for seg in segments[width:])
        return match.group(0)

    quoted_path_re = re.compile(r'"[^"]+"(?:\s*\.\s*"[^"]+")+')
    rewritten = quoted_path_re.sub(replace_leading_qualifier, expression)

    # Also rewrite unquoted Schema.Table.Column / Table.Column when the
    # leading table is a known source relation (deterministic compose can
    # emit either shape before normalization).
    for source_key in source_keys:
        mapped = table_to_entity[source_key]
        if "." in source_key:
            parts = source_key.split(".")
            pattern = r"(?<![A-Za-z0-9_\"])" + r"\s*\.\s*".join(re.escape(p) for p in parts) + r"(?=\s*\.)"
        else:
            pattern = rf'(?<![A-Za-z0-9_"]){re.escape(source_key)}(?=\s*\.)'
        rewritten = re.sub(pattern, mapped, rewritten, flags=re.IGNORECASE)

    return rewritten