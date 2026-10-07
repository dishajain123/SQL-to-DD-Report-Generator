"""Shared T-SQL text helpers for derivation v2 phases."""
from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.utils.entity_name_map import resolve_entity_name
from app.parsing.sql_lex import mask_sql, normalize_comparison_spacing


_STMT_START = re.compile(
    r"(?is)\b(?:UPDATE|INSERT|DELETE|MERGE|SELECT|CREATE|DROP|TRUNCATE|EXEC|EXECUTE|DECLARE|GO"
    r"|IF|BEGIN|ELSE)\b"
)

_CACHE_MIN_CHARS = 4000


def strip_sql_comments(sql: str) -> str:
    """Remove ``--`` line comments and ``/* */`` block comments."""
    sql = sql or ""
    if len(sql) >= _CACHE_MIN_CHARS:
        return _strip_large(sql)
    return _strip_small(sql)


@lru_cache(maxsize=512)
def _strip_small(sql: str) -> str:
    return _strip_sql_comments_impl(sql, with_map=False)[0]


@lru_cache(maxsize=16)
def _strip_large(sql: str) -> str:
    return _strip_sql_comments_impl(sql, with_map=False)[0]


@lru_cache(maxsize=8)
def _strip_sql_comments_with_map(sql: str) -> tuple[str, list[int]]:
    """Comment-stripped text plus, per output char, its offset in ``sql``."""
    return _strip_sql_comments_impl(sql or "", with_map=True)


def normalize_unicode_space_separators(sql: str) -> str:
    """Map Unicode space separators (SSMS/Word paste) to ASCII space outside literals."""
    if not sql:
        return sql
    out: list[str] = []
    in_single = False
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if in_single:
            out.append(ch)
            if ch == "'" and i + 1 < n and sql[i + 1] == "'":
                out.append(sql[i + 1])
                i += 2
                continue
            if ch == "'":
                in_single = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
            i += 1
            continue
        if unicodedata.category(ch) == "Zs" and ch not in "\n\r\t":
            out.append(" ")
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _strip_sql_comments_impl(sql: str, *, with_map: bool) -> tuple[str, list[int]]:
    # Keep source coordinates stable across all scanners, including nested
    # comments. Normalize comparison tokens in code only; parse_statement
    # records these recoveries for the completeness/review gate.
    sql = normalize_unicode_space_separators(sql)
    text, _ = normalize_comparison_spacing(sql)
    text = mask_sql(text, quotes=False)
    return text, list(range(len(sql))) if with_map else []


@lru_cache(maxsize=8)
def _newline_offsets(sql: str) -> list[int]:
    return [i for i, ch in enumerate(sql) if ch == "\n"]


def stripped_offset_to_line(sql: str, offset: int) -> int | None:
    """1-based line in the original ``sql`` for an offset in its stripped text.

    Statement positions throughout v2 are comment-stripped offsets; reports
    need the line a reviewer will find in the file they uploaded.
    """
    _, origin = _strip_sql_comments_with_map(sql or "")
    if not origin or offset < 0:
        return None
    original = origin[min(offset, len(origin) - 1)]
    return bisect_left(_newline_offsets(sql or ""), original) + 1


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


def is_staging_derivation_entity(entity: str) -> bool:
    """Non-exportable intermediate objects (backup temps, CTE shells, scratch tables).

    Global interface temps (``##AccountCal`` / ``##CustomerCal``) are **not**
    staging — they are business derivation targets. Session-local ``#temp``
    tables (single hash, including ``TEMPDB..#FOO``) are always staging.
    """
    raw = (entity or "").strip().strip('"')
    token = raw.split(".")[-1].strip("[]") if raw else ""
    if token.startswith("#") and not token.startswith("##"):
        return True
    name = normalize_table_name(entity or "").upper().lstrip("#")
    if not name:
        return False
    if name.endswith("_BKUP"):
        return True
    if name.startswith("CTE_"):
        return True
    if name.startswith("TEMPTABLE"):
        return True
    if name in {"TEMPTABLEDPD", "TEMPTABLENPA"}:
        return True
    # Session scratch tables in PRO SPs (not the ``TEMPLATE`` spelling).
    if name.startswith("TEMP") and name != "TEMPLATE":
        return True
    return False


def lineage_keeps_target_hop(entity: str, relationship: str | None = None) -> bool:
    """Whether a column ref should keep target→hop→column in compiled 4X.

    Only **sibling** ``##AccountCal`` / ``##CustomerCal`` interface reads use the
    three-part path (``"ACCOUNTCAL"."CUSTOMERCAL"."RiskFlag"``). Joined physical
    tables (``PUI_CAL``, ``AdvAcRestructureCal``), session ``#temp`` tables, and
    derived subquery aliases (``C``) collapse to the joined source entity in 4X.
    """
    ent = normalize_table_name(entity or "").upper().lstrip("#")
    if ent.endswith("CUSTOMERCAL"):
        return False
    rel = (
        normalize_table_name(relationship or "").upper().lstrip("#")
        if relationship
        else ""
    )
    if not rel:
        return False
    if ent.endswith("ACCOUNTCAL") and rel.endswith("CUSTOMERCAL"):
        return True
    if ent.endswith("CUSTOMERCAL") and rel.endswith("ACCOUNTCAL"):
        return True
    return False


def is_ephemeral_sql_alias(name: str) -> bool:
    """Statement-local alias (``A``, ``B``, ``C``) — not a physical entity name."""
    text = bare_ident(name or "")
    if not text or text.startswith("#") or text.startswith("@"):
        return False
    if len(text) > 3:
        return False
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text))


def should_collapse_target_join_hop(
    entity: str,
    relationship: str | None,
    target_entity: str | None,
) -> bool:
    """Whether ``Target.Rel.Col`` should compile as ``Rel.Col`` (drop target prefix)."""
    if not relationship or not target_entity:
        return False
    ent = normalize_table_name(entity or "").upper().lstrip("#")
    tgt = normalize_table_name(target_entity or "").upper().lstrip("#")
    if ent != tgt:
        return False
    if lineage_keeps_target_hop(entity, relationship):
        return False
    if is_ephemeral_sql_alias(relationship):
        return False
    return True


def bare_ident(name: str) -> str:
    return (name or "").strip().strip("[]").strip('"').strip()


def _parse_simple_select_projection(select_item: str) -> tuple[str | None, str]:
    """Parse one SELECT-list item into ``(optional_table_qualifier, column_name)``.

    Supports bracketed identifiers with spaces (``[Account No]``) and
    ``Table.[Account No]`` forms common in staging / upload tables.
    """
    text = (select_item or "").strip()
    text = re.sub(r"(?is)^DISTINCT\s+", "", text)
    text = re.sub(
        r"(?is)^TOP\s+\(?\s*\d+\s*\)?\s+(?:PERCENT\s+)?",
        "",
        text,
    )
    m = re.match(
        r"(?is)^(?:\[?(?P<table>[^.\]]+)\]?\.)?\[?(?P<col>[^\]]+)\]?\s*$",
        text,
    )
    if not m:
        return None, bare_ident(text)
    table = bare_ident(m.group("table")) if m.group("table") else None
    return table, bare_ident(m.group("col"))


def _qualify_predicate_columns(predicate: str, table: str) -> str:
    """Qualify bare and bracketed column tokens with a subquery source table."""
    text = _qualify_bare_columns(predicate or "", table)
    table_norm = normalize_table_name(table)

    def repl_bracket(match: re.Match[str]) -> str:
        col = match.group(1).strip()
        if not col or col.startswith("@"):
            return match.group(0)
        return f"{table_norm}.{col}"

    return re.sub(r"(?<![.\w\]])\[([^\]]+)\](?!\s*\()", repl_bracket, text)


def _sql_column_ref(table: str, column: str) -> str:
    """Render ``table.column`` for bracketed or spaced identifiers."""
    col = bare_ident(column)
    tbl = normalize_table_name(table)
    if re.search(r"(?i)[^\w]", col):
        return f"{tbl}.[{col}]"
    return f"{tbl}.{col}"


# One to three dotted parts: Table, Schema.Table, Db.Schema.Table (each
# optionally [bracketed]). A two-part limit truncated "DB.PRO.T" to "DB.PRO".
_TABLE_TOKEN = r"(?:\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?(?:\.\[?#?#?[A-Za-z_][A-Za-z0-9_]*\]?){0,2})"
_ALIAS_TOKEN = (
    r"(?:AS\s+)?(?P<alias>(?!(?:ON|WHERE|JOIN|LEFT|RIGHT|INNER|OUTER|FULL|CROSS|APPLY|"
    r"GROUP|ORDER|SET|FROM|WITH|OPTION|UNION)\b)[A-Za-z_][A-Za-z0-9_]*)"
)
# T-SQL table hints — ``WITH (NOLOCK)`` — may follow a table or its alias.
_TABLE_HINT = r"(?:\s+WITH\s*\([^)]*\))?"
# T-SQL physical join hints sit between the join type and JOIN
# (``INNER HASH JOIN``, ``LEFT hash JOIN``, ``INNER MERGE JOIN``). Without them
# a LEFT hash JOIN was read as a bare JOIN, i.e. treated like an inner join.
_JOIN_HINT = r"(?:\s+(?:HASH|MERGE|LOOP|REMOTE))?"
_JOIN_KEYWORD = (
    rf"(?:(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?|INNER|CROSS){_JOIN_HINT}\s+JOIN|JOIN"
)


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
    # Split on JOIN keywords (incl. LEFT/RIGHT/FULL OUTER JOIN) while keeping
    # join-type words and table hints out of aliases and ON clauses: the ON
    # body stops at the next complete join keyword, so "… LEFT OUTER" never
    # leaks into the previous join's condition.
    pattern = re.compile(
        rf"(?is)\b(?:FROM|{_JOIN_KEYWORD})\s+"
        rf"(?P<table>{_TABLE_TOKEN}){_TABLE_HINT}"
        rf"(?:\s+{_ALIAS_TOKEN})?{_TABLE_HINT}"
        rf"(?:\s+ON\s+(?P<on>.+?)(?=\b(?:{_JOIN_KEYWORD})\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bOPTION\b|$))?"
    )
    for match in pattern.finditer(text):
        table = normalize_table_name(match.group("table"))
        alias = match.groupdict().get("alias")
        on_clause = (match.group("on") or "").strip() or None
        results.append((table, alias, on_clause))
    return results


def parse_from_join_clause_with_type(
    from_body: str,
) -> list[tuple[str, str | None, str | None, str]]:
    """Like :func:`parse_from_join_clause`, but each entry also carries its
    join keyword ("FROM", "JOIN", "INNER JOIN", "LEFT JOIN", ...), upper-cased
    and whitespace-normalized.

    Needed wherever INNER-vs-OUTER semantics matter: an INNER (or plain,
    unqualified) JOIN's ON-clause predicates restrict which driving rows
    the statement ever touches, exactly like a WHERE term would — but a
    LEFT/RIGHT/FULL JOIN's ON-clause predicates only decide whether the
    *joined* side matches, and never remove a driving row from the result.
    Folding the two cases the same way would be correct for the former and
    wrong for the latter.
    """
    text = (from_body or "").strip()
    if not text:
        return []
    if not re.match(r"(?is)^(FROM|JOIN)\b", text):
        text = "FROM " + text

    results: list[tuple[str, str | None, str | None, str]] = []
    pattern = re.compile(
        rf"(?is)\b(?P<kw>FROM|{_JOIN_KEYWORD})\s+"
        rf"(?P<table>{_TABLE_TOKEN}){_TABLE_HINT}"
        rf"(?:\s+{_ALIAS_TOKEN})?{_TABLE_HINT}"
        rf"(?:\s+ON\s+(?P<on>.+?)(?=\b(?:{_JOIN_KEYWORD})\b|\bWHERE\b|\bGROUP\b|\bORDER\b|\bOPTION\b|$))?"
    )
    for match in pattern.finditer(text):
        table = normalize_table_name(match.group("table"))
        alias = match.groupdict().get("alias")
        on_clause = (match.group("on") or "").strip() or None
        join_type = re.sub(r"\s+", " ", (match.group("kw") or "").strip()).upper()
        join_type = re.sub(r"\s(?:HASH|MERGE|LOOP|REMOTE)\b", "", join_type)
        results.append((table, alias, on_clause, join_type))
    return results


# Performance rule for every scanner in this module: match keywords with a
# compiled pattern AT a position — ``_KW.match(text, i)`` — never
# ``re.match(p, text[i:])``. The slice copies the rest of the file on every
# call; inside a per-character loop that is O(n²) bytes per scan, which made a
# 150 KB production procedure take hours in "Generating derivation rows".
_UPDATE_KW = re.compile(r"(?is)\bUPDATE\b")
_WS_SET_KW = re.compile(r"(?is)\s+SET\b")
_SET_KW = re.compile(r"(?is)\bSET\b")
_FROM_KW = re.compile(r"(?is)FROM\b")
_UNION_SELECT_KW = re.compile(r"(?is)UNION\s+(?:ALL\s+)?SELECT\b")
_WHERE_KW = re.compile(r"(?is)WHERE\b")
_BEGIN_KW = re.compile(r"(?is)BEGIN\b")
_CASE_KW = re.compile(r"(?is)CASE\b")
_END_KW = re.compile(r"(?is)END\b")
_IF_KW = re.compile(r"(?is)IF\b")
_IF_OBJECT_ID = re.compile(r"(?is)IF\s+OBJECT_ID\b")
_ELSE_IF_KW = re.compile(r"(?is)ELSE\s+IF\b|ELSEIF\b")
_ELSE_KW = re.compile(r"(?is)ELSE\b")
_UPDATE_AT = re.compile(r"(?is)UPDATE\b")
_END_OR_ELSE = re.compile(r"(?is)(END|ELSE)\b")


def _cached(func):
    """Memoise a whole-SQL scanner by its input text.

    Each DD column re-runs the same scanners over the same procedure; for a
    150-column procedure that repeated the identical full-text work 150×.
    Results are returned as fresh shallow copies so callers may mutate them.
    """
    from functools import wraps

    @lru_cache(maxsize=16)
    def cached(sql: str):
        return func(sql)

    @wraps(func)
    def wrapper(sql: str):
        sql = sql or ""
        # Small inputs (single statements, fragments) are cheap and numerous;
        # caching them would evict the one large procedure text that matters.
        result = cached(sql) if len(sql) >= _CACHE_MIN_CHARS else func(sql)
        if isinstance(result, list):
            return [dict(item) if isinstance(item, dict) else item for item in result]
        if isinstance(result, dict):
            return {k: (dict(v) if isinstance(v, dict) else v) for k, v in result.items()}
        return result

    wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
    return wrapper


@_cached
def extract_update_statements(sql: str) -> list[dict[str, str | None]]:
    """Extract UPDATE statements with SET / FROM / WHERE (paren-aware)."""
    text = strip_sql_comments(sql or "")
    statements: list[dict[str, str | None]] = []
    for match in _UPDATE_KW.finditer(text):
        start = match.start()
        i = match.end()
        # MERGE's UPDATE SET has no target here; handled by the MERGE extractor.
        if _WS_SET_KW.match(text, i):
            continue
        # head until SET
        set_match = _SET_KW.search(text, i)
        if not set_match:
            continue
        head = text[i : set_match.start()].strip()
        i = set_match.end()

        set_clause, i = _read_until_keyword(text, i, {"FROM", "WHERE"}, stop_at_statement=True)
        from_clause = None
        where_clause = None
        if i < len(text) and _FROM_KW.match(text, i):
            i = i + len("FROM")
            # OPTION (MAXDOP 1 / HASH JOIN …) is a statement-level query hint,
            # not part of the FROM or WHERE text.
            from_clause, i = _read_until_keyword(
                text, i, {"WHERE", "OPTION"}, stop_at_statement=True
            )
            from_clause = from_clause.strip() or None
        if i < len(text) and _WHERE_KW.match(text, i):
            i = i + len("WHERE")
            where_clause, i = _read_until_keyword(text, i, {"OPTION"}, stop_at_statement=True)
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


@_cached
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
        # Jump straight to the next "IF" token instead of testing every char.
        next_if = _IF_KW.search(text, i)
        if next_if is None:
            break
        i = next_if.start()
        # Skip IF OBJECT_ID(...) utility guards — not derivation branches.
        if _IF_OBJECT_ID.match(text, i):
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
    if not _IF_KW.match(text, i):
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
        else_if = _ELSE_IF_KW.match(text, i)
        if else_if:
            i = else_if.end()
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
        else_only = _ELSE_KW.match(text, i)
        if else_only:
            i = else_only.end()
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
        if depth == 0 and _BEGIN_KW.match(text, i):
            break
        # Single-statement IF without BEGIN: stop before UPDATE/INSERT/…
        if depth == 0 and _STMT_START.match(text, i):
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
    if _BEGIN_KW.match(text, i):
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
            if _BEGIN_KW.match(text, i):
                depth += 1
                i += 5
                continue
            if _CASE_KW.match(text, i):
                # A bare CASE ... END is not a BEGIN block -- its own END
                # must not be mistaken for this block's terminator below.
                case_depth += 1
                i += 4
                continue
            if _END_KW.match(text, i):
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
    if _UPDATE_AT.match(text, body_start):
        # Read until the sibling END/ELSE at depth 0.
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
            if depth == 0 and _END_OR_ELSE.match(text, j):
                break
            j += 1
        return body_start, j, j

    return -1, -1, start


_SUBQUERY_RE = re.compile(r"(?is)\bSELECT\b")


def is_procedure_wide_gate(condition: str | None) -> bool:
    """True when a procedural ``IF`` condition is a set-level test over a table.

    A T-SQL ``IF`` is always evaluated once per run, never per row. Scalar
    conditions (``IF @TimeKey > 26267``) already read as run-level in a
    formula because they only reference variables. A condition containing a
    subquery (``IF EXISTS (SELECT … WHERE col >= @x)``, ``IF (SELECT COUNT(*)
    …) > 0``) does not: projecting its inner WHERE into the row formula would
    make it look like every row is tested independently.
    """
    return bool(condition and _SUBQUERY_RE.search(condition))


@dataclass
class WorkflowGate:
    """A procedure-wide IF/ELSE IF condition, evaluated once per run."""

    name: str  # e.g. "Gate 1" — a report label, never a formula variable
    group_id: str
    index: int
    kind: str  # IF | ELSEIF
    condition: str


@_cached
def extract_workflow_gates(sql: str) -> list[WorkflowGate]:
    """Name every procedure-wide gate in source order (stable across columns)."""
    gates: list[WorkflowGate] = []
    for span in extract_if_else_chains(sql):
        if span.kind == "ELSE" or not is_procedure_wide_gate(span.condition):
            continue
        gates.append(
            WorkflowGate(
                name=f"Gate {len(gates) + 1}",
                group_id=span.group_id,
                index=span.index,
                kind=span.kind,
                condition=" ".join((span.condition or "").split()),
            )
        )
    return gates


@_cached
def extract_catch_spans(sql: str) -> list[tuple[int, int]]:
    """``BEGIN CATCH … END CATCH`` body spans (comment-stripped coordinates)."""
    text = strip_sql_comments(sql or "")
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    for match in re.finditer(r"(?is)\b(BEGIN|END)\s+CATCH\b", text):
        if match.group(1).upper() == "BEGIN":
            stack.append(match.end())
        elif stack:
            spans.append((stack.pop(), match.start()))
    return spans


_RESET_RE = re.compile(
    rf"(?is)\b(?P<kind>TRUNCATE\s+TABLE|DROP\s+TABLE(?:\s+IF\s+EXISTS)?|DELETE(?:\s+FROM)?)"
    rf"\s+(?P<table>{_TABLE_TOKEN})"
)


@_cached
def extract_table_resets(sql: str) -> list[dict[str, Any]]:
    """Statements that empty a table: ``TRUNCATE TABLE``, ``DROP TABLE`` and a
    ``DELETE`` with no WHERE and no JOIN (which clears every row).

    Returns ``[{"table", "kind", "start"}]`` in comment-stripped coordinates.
    A filtered DELETE removes only some rows and is not a reset. Resets inside
    an IF/ELSE branch or a CATCH handler are conditional and are skipped —
    except the ``IF OBJECT_ID(...) IS NOT NULL DROP TABLE`` idiom, which the
    branch scanner deliberately does not treat as a branch.
    """
    text = strip_sql_comments(sql or "")
    branches = extract_if_else_chains(text)
    catches = extract_catch_spans(text)
    resets: list[dict[str, Any]] = []
    for match in _RESET_RE.finditer(text):
        start = match.start()
        if any(b.body_start <= start < b.body_end for b in branches):
            continue
        if any(s <= start < e for s, e in catches):
            continue
        kind = " ".join(match.group("kind").upper().split())
        table = normalize_table_name(match.group("table"))
        # "MERGE … WHEN MATCHED THEN DELETE" / "DELETE TOP (n) …" are row-level.
        if table.upper() in {"WHEN", "OUTPUT", "TOP", "WHERE"} or re.search(
            r"(?is)\bTHEN\s*$", text[max(0, start - 40):start]
        ):
            continue
        if kind.startswith("DELETE"):
            if not kind.endswith("FROM") and re.match(r"(?is)\s*FROM\b", text[match.end():]):
                continue  # DELETE alias FROM … — multi-table form, always filtered by a join
            body, _ = _read_until_keyword(text, match.end(), set(), stop_at_statement=True)
            if re.search(r"(?is)\b(?:WHERE|JOIN)\b", body):
                continue
            kind = "DELETE"
        elif kind.startswith("DROP"):
            kind = "DROP TABLE"
        resets.append({"table": table, "kind": kind, "start": start})
    return resets


_CREATE_TABLE_RE = re.compile(
    r"(?is)\bCREATE\s+TABLE\s+(?P<table>[#\w\[\]\.]+)\s*\("
)
_DECLARE_TABLE_RE = re.compile(r"(?is)\bDECLARE\s+(?P<table>@\w+)\s+TABLE\s*\(")
_DECLARE_SCALAR_RE = re.compile(
    r"(?is)\bDECLARE\s+(?P<var>@\w+)\s+(?P<type>(?!TABLE\b)[A-Za-z]+(?:\s*\([^)]*\))?)"
)
_COLUMN_DEF_RE = re.compile(r"(?is)^\s*\[?(?P<col>[A-Za-z_]\w*)\]?\s+(?P<type>[A-Za-z]+)")
_NON_COLUMN_DEFS = {"CONSTRAINT", "PRIMARY", "UNIQUE", "INDEX", "FOREIGN", "CHECK"}


@_cached
def extract_declared_column_types(sql: str) -> dict[str, dict[str, str]]:
    """``{TABLE: {COLUMN: SQL_TYPE}}`` from ``CREATE TABLE`` / ``DECLARE @t TABLE``.

    Keys are upper-cased; table keys use :func:`normalize_table_name` (so
    ``#DpdStaging`` stays ``#DPDSTAGING``). Scalar ``DECLARE @var TYPE`` lands
    under the ``"@"`` key.
    """
    text = strip_sql_comments(sql or "")
    out: dict[str, dict[str, str]] = {}
    for pattern in (_CREATE_TABLE_RE, _DECLARE_TABLE_RE):
        for match in pattern.finditer(text):
            open_idx = match.end() - 1
            depth = 0
            close_idx = -1
            for idx in range(open_idx, len(text)):
                if text[idx] == "(":
                    depth += 1
                elif text[idx] == ")":
                    depth -= 1
                    if depth == 0:
                        close_idx = idx
                        break
            if close_idx < 0:
                continue
            table = normalize_table_name(match.group("table")).upper()
            columns = out.setdefault(table, {})
            for part in split_csv_respecting_parens(text[open_idx + 1 : close_idx]):
                col_match = _COLUMN_DEF_RE.match(part)
                if not col_match or col_match.group("col").upper() in _NON_COLUMN_DEFS:
                    continue
                columns[col_match.group("col").upper()] = col_match.group("type").upper()
    scalars = out.setdefault("@", {})
    for match in _DECLARE_SCALAR_RE.finditer(text):
        scalars[match.group("var").upper()] = match.group("type").split("(")[0].strip().upper()
    return out


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
    _, from_rest = _split_select_list(inner)
    from_table = _first_from_table(from_rest) if from_rest else None
    alias_map = _alias_map_from_from_clause(from_rest)
    # Drop trailing GROUP BY / ORDER BY; keep non-aggregate HAVING only.
    having = _extract_having_clause(pred)
    pred = _strip_group_order(pred)
    pred = _rewrite_qualified_aliases(pred, alias_map)
    if having and re.search(r"(?is)\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", having):
        having = None
    if from_table:
        pred = _qualify_predicate_columns(pred, from_table)
        if having:
            having = _rewrite_qualified_aliases(having, alias_map)
            having = _qualify_predicate_columns(having, from_table)
    if having:
        return f"({pred}) AND ({having})" if pred else having
    return pred.strip() or None


def exists_subquery_to_row_predicate(condition: str | None) -> tuple[str | None, list[str]]:
    """Project ``EXISTS (SELECT …)`` to a row predicate and dependency refs."""
    if not condition:
        return None, []
    text = condition.strip()
    if not re.match(r"(?is)^EXISTS\s*\(", text):
        return text, []
    deps = extract_subquery_dependency_refs(text)
    pred = exists_condition_to_row_predicate(text)
    if pred and not re.match(r"(?is)^EXISTS\b", pred.strip()):
        return pred, deps
    return None, deps


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
    table_qual, proj_col = _parse_simple_select_projection(proj)
    if proj_col and lhs.strip():
        src_table = table_qual or (normalize_table_name(from_table) if from_table else "")
        if src_table:
            parts.append(f"{_sql_column_ref(src_table, proj_col)} = {lhs.strip()}")
        else:
            parts.append(f"{proj_col} = {lhs.strip()}")

    if where_pred:
        alias_map = _alias_map_from_from_clause(rest)
        where_pred = _rewrite_qualified_aliases(where_pred, alias_map)
        if from_table:
            where_pred = _qualify_predicate_columns(where_pred, from_table)
        parts.append(f"({where_pred})")
    # HAVING with aggregates is not row-level 4X — keep deps, skip the clause.
    if having and not re.search(r"(?is)\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", having):
        parts.append(f"({having})")

    if not parts:
        return None, deps
    return " AND ".join(parts), deps


def dim_asset_class_in_use_hop_only(subquery: str) -> str | None:
    """When ``IN (SELECT … FROM DimAssetClass WHERE ShortName='X')``, return ``X``."""
    body = (subquery or "").strip()
    if not re.search(r"(?is)\bDimAssetClass\b", body):
        return None
    match = re.search(
        r"(?is)AssetClassShortName(?:Enum)?\s*=\s*'([^']+)'",
        body,
    )
    if not match:
        return None
    return bare_ident(match.group(1)).upper()


_SUBQUERY_KEYWORDS = {
    "AND", "OR", "NOT", "IS", "NULL", "IN", "EXISTS", "BETWEEN", "LIKE",
    "CASE", "WHEN", "THEN", "ELSE", "END", "TRUE", "FALSE",
    "SELECT", "FROM", "WHERE", "GROUP", "ORDER", "HAVING", "JOIN", "ON",
    "AS", "DISTINCT", "TOP", "BY", "UNION", "ALL", "INNER", "LEFT",
    "RIGHT", "FULL", "OUTER", "CROSS", "APPLY",
}


def _alias_map_from_from_clause(from_rest: str) -> dict[str, str]:
    """``{ALIAS_OR_TABLE: physical_table}`` from a subquery FROM/JOIN list."""
    mapping: dict[str, str] = {}
    for table, alias, _on in parse_from_join_clause(from_rest or ""):
        if not table:
            continue
        mapping[table.upper()] = table
        mapping[normalize_table_name(table).upper()] = table
        if alias:
            mapping[alias.upper()] = table
    return mapping


def _rewrite_qualified_aliases(pred: str, alias_to_table: dict[str, str]) -> str:
    """Rewrite ``alias.col`` using the subquery's own FROM aliases."""
    if not pred or not alias_to_table:
        return pred or ""

    def repl(match: re.Match[str]) -> str:
        qual = match.group("qual")
        col = match.group("col")
        table = alias_to_table.get(qual.upper())
        if not table:
            return match.group(0)
        return _sql_column_ref(table, col)

    return re.sub(
        r"(?P<qual>[#A-Za-z_][A-Za-z0-9_]*)\.(?P<col>\[[^\]]+\]|[A-Za-z_][A-Za-z0-9_]*)",
        repl,
        pred,
    )


def _qualify_bare_columns(predicate: str, table: str) -> str:
    """Prefix bare column identifiers with ``table.`` for cross-entity resolution.

    Never rewrites an identifier that is already a qualifier (``A.col`` or
    ``Entity::Col``) — those are aliases / phase-2 markers, not columns of
    ``table``. String literals are left untouched so ``AssetClassShortName='LOS'``
    does not become ``'DimAssetClass.LOS'``.
    """
    table_norm = normalize_table_name(table)
    segments = re.split(r"('(?:''|[^'])*')", predicate or "")
    out: list[str] = []
    for seg in segments:
        if seg.startswith("'"):
            out.append(seg)
            continue

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            if token.upper() in _SUBQUERY_KEYWORDS:
                return token
            if token.startswith("@") or token.startswith("#"):
                return token
            return f"{table_norm}.{token}"

        out.append(
            re.sub(
                r"(?<![.\w:])(@?[A-Za-z_][A-Za-z0-9_]*)\b(?!\s*(?:\(|\.|::))",
                repl,
                seg,
            )
        )
    return "".join(out)


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


@_cached
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


def extract_merge_using_row_predicate(using_body: str) -> str | None:
    """Row filter from ``USING (SELECT … FROM … [JOIN …] WHERE …)`` in a MERGE.

    The ON clause only keys target to source; business predicates (e.g. Aqua
    Scheme product filters) live in the USING subquery's WHERE and must fold
    into the derived row condition.
    """
    text = (using_body or "").strip()
    if not text:
        return None
    # ``USING (SELECT …) S`` / ``… ) AS Source`` — the alias sits after the
    # subquery's closing paren, so a naive outer-paren strip never runs and
    # ``WHERE`` inside the subquery is missed (depth > 0).
    text = re.sub(
        r"(?is)\)\s*(?:AS\s+)?[A-Za-z_][A-Za-z0-9_]*\s*$",
        ")",
        text,
    ).strip()
    if text.startswith("(") and text.endswith(")") and _parens_balanced(text[1:-1]):
        text = text[1:-1].strip()
    if not re.search(r"(?is)\bSELECT\b", text):
        return None
    pred = _extract_where_predicate(text)
    if not pred:
        return None
    return _strip_group_order(pred)


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


_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {}


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    pattern = _KEYWORD_PATTERNS.get(keyword)
    if pattern is None:
        pattern = _KEYWORD_PATTERNS[keyword] = re.compile(
            rf"{keyword}\b", re.I | re.S
        )
    return pattern


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
        if depth == 0 and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_@#")):
            if ch == ";" and stop_at_statement:
                return "".join(buf), i
            # Only a letter can start a keyword; skip the regex work otherwise.
            if not ch.isalpha():
                buf.append(ch)
                i += 1
                continue
            # A bare CASE ... END is not wrapped in parens, so it needs its
            # own nesting counter — otherwise the CASE's own closing END is
            # mistaken for the statement/block-terminating END below and the
            # read stops mid-expression, silently truncating everything after
            # it (including the real trailing FROM/WHERE clause).
            case_match = _CASE_KW.match(text, i)
            if case_match:
                case_depth += 1
                buf.append(case_match.group(0))
                i = case_match.end()
                continue
            end_match = _END_KW.match(text, i)
            if end_match and case_depth > 0:
                case_depth -= 1
                buf.append(end_match.group(0))
                i = end_match.end()
                continue
            if case_depth == 0:
                for kw in keywords:
                    if _keyword_pattern(kw).match(text, i):
                        return "".join(buf), i
                if stop_at_statement and _STMT_START.match(text, i):
                    return "".join(buf), i
                # Bare END at depth 0 (outside any open CASE) is a
                # procedure/block terminator (END TRY / END CATCH / END) —
                # stop without consuming it.
                if stop_at_statement and end_match:
                    return "".join(buf), i
        buf.append(ch)
        i += 1
    return "".join(buf), i


_SELECT_KW = re.compile(r"(?is)\bSELECT\b")
_INTO_KW = re.compile(r"(?is)INTO\b")
_TARGET_AFTER_INTO = re.compile(rf"(?is)\s*(?P<target>{_TABLE_TOKEN})")
_WS_FROM_KW = re.compile(r"(?is)\s*FROM\b")
_WS_WHERE_KW = re.compile(r"(?is)\s*WHERE\b")
_WS_GROUP_BY_KW = re.compile(r"(?is)\s*GROUP\s+BY\b")


@_cached
def extract_select_into(sql: str) -> list[dict[str, str]]:
    """Find SELECT … INTO #temp FROM … [WHERE …] statements."""
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    for match in _SELECT_KW.finditer(text):
        start = match.start()
        list_start = match.end()
        # The projection ends at this SELECT's own top-level INTO or FROM.
        # (Searching for the next INTO anywhere paired a plain SELECT with an
        # unrelated later "INSERT INTO #T", and cost O(file) per SELECT.)
        select_list, stop = _read_until_keyword(
            text, list_start, {"INTO", "FROM"}, stop_at_statement=True
        )
        into_match = _INTO_KW.match(text, stop)
        if not into_match:
            continue
        after_into = into_match.end()
        target_match = _TARGET_AFTER_INTO.match(text, after_into)
        if not target_match:
            continue
        target = normalize_table_name(target_match.group("target"))
        if not target.startswith("#"):
            continue
        pos = target_match.end()
        from_clause = ""
        where_clause = ""
        group_by_clause = ""
        from_match = _WS_FROM_KW.match(text, pos)
        if from_match:
            pos = from_match.end()
            from_clause, pos = _read_until_keyword(
                text, pos, {"WHERE", "GROUP", "ORDER", "HAVING", "EXCEPT", "INTERSECT", "UNION"},
                stop_at_statement=True,
            )
        where_match = _WS_WHERE_KW.match(text, pos)
        if where_match:
            pos = where_match.end()
            where_clause, pos = _read_until_keyword(
                text, pos, {"GROUP", "ORDER", "HAVING", "EXCEPT", "INTERSECT", "UNION"},
                stop_at_statement=True,
            )
        group_match = _WS_GROUP_BY_KW.match(text, pos)
        if group_match:
            pos = group_match.end()
            group_by_clause, pos = _read_until_keyword(
                text, pos, {"ORDER", "HAVING", "EXCEPT", "INTERSECT", "UNION"},
                stop_at_statement=True,
            )
        results.append(
            {
                "target": target,
                "select_list": select_list.strip(),
                "from_body": from_clause.strip(),
                "where_clause": where_clause.strip(),
                "group_by": group_by_clause.strip(),
                "raw_sql": text[start:pos].strip(),
                "start": str(start),
                "end": str(pos),
            }
        )
    return results


_SELECT_ITEM_COL_AS_RE = re.compile(
    r"(?is)(?:(?P<qual>[#A-Za-z0-9_]+)\.)?(?P<col>\[?[A-Za-z_][A-Za-z0-9_]*\]?)"
    r"(?:\s+AS\s+(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?))?",
)


_SELECT_ITEM_EQ_ALIAS_RE = re.compile(
    r"(?s)^\[?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\]?\s*=(?!=)\s*(?P<value>\S.*)$"
)


def parse_select_list(
    select_list: str,
) -> list[tuple[str | None, str | None, str | None, str]]:
    """Parse a SELECT projection list, preserving one entry per ordinal slot.

    Returns ``(src_qual, src_col, dest_alias, raw_chunk)`` per top-level item.
    Complex expressions (CASE / function calls / subqueries) cannot resolve
    to a single ``src_qual``/``src_col`` and are returned as
    ``(None, None, alias_if_any, raw_chunk)`` — the caller MUST still count
    this slot (not skip it) so positional alignment with an explicit target
    column list stays correct for every projection after it.
    """
    results: list[tuple[str | None, str | None, str | None, str]] = []
    if not select_list or not select_list.strip() or select_list.strip() == "*":
        return results

    # Strip leading SELECT modifiers (DISTINCT / ALL / TOP n) before
    # splitting — otherwise the first chunk's own column regex greedily
    # matches the modifier keyword itself as if it were the column name
    # (``DISTINCT UcifEntityID`` -> column "DISTINCT").
    select_list = select_list.strip()
    for _ in range(3):
        m = re.match(r"(?is)^(?:DISTINCT|ALL)\s+", select_list)
        if m:
            select_list = select_list[m.end():]
            continue
        m = re.match(r"(?is)^TOP\s*\(?\s*\d+\s*\)?\s+", select_list)
        if m:
            select_list = select_list[m.end():]
            continue
        break

    for part in split_csv_respecting_parens(select_list):
        chunk = part.strip()
        if not chunk or chunk == "*":
            results.append((None, None, None, chunk))
            continue
        # T-SQL ``alias = expression`` projection (``FLGDEG='N'``,
        # ``ACCOUNTENTITYID = ACCOUNTENTITYID``). A boolean comparison is not a
        # legal SELECT-list item, so a leading ``identifier =`` is always the
        # output name; keep only the value as the chunk.
        eq_alias: str | None = None
        eq_match = _SELECT_ITEM_EQ_ALIAS_RE.match(chunk)
        if eq_match:
            eq_alias = bare_ident(eq_match.group("alias"))
            chunk = eq_match.group("value").strip()
        if "(" in chunk:
            alias = None
            alias_m = re.search(
                r"(?is)\)\s*(?:AS\s+)?(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*$",
                chunk,
            )
            if alias_m:
                alias = bare_ident(alias_m.group("alias"))
            if alias is None or alias.upper() == "END":
                # ``CASE … END AS Col`` / ``CASE … END Col``: the alias follows
                # the closing END, not a ``)`` (without it the projection's
                # output column was dropped from the temp table's schema).
                end_alias_m = re.search(
                    r"(?is)\bEND\s+(?:AS\s+)?(?P<alias>\[?[A-Za-z_][A-Za-z0-9_]*\]?)\s*$",
                    chunk,
                )
                if end_alias_m and bare_ident(end_alias_m.group("alias")).upper() != "END":
                    alias = bare_ident(end_alias_m.group("alias"))
                elif alias is not None and alias.upper() == "END":
                    alias = None
            results.append((None, None, eq_alias or alias, chunk))
            continue
        match = _SELECT_ITEM_COL_AS_RE.fullmatch(chunk)
        if not match:
            results.append((None, None, eq_alias, chunk))
            continue
        qual = match.group("qual")
        col = bare_ident(match.group("col"))
        alias = bare_ident(match.group("alias")) if match.group("alias") else eq_alias
        if col.upper() in {"AS", "FROM", "INTO", "NULL", "CASE", "WHEN", "END", "TRUE", "FALSE"}:
            results.append((None, None, None, chunk))
            continue
        results.append((qual, col, alias, chunk))
    return results


_CTE_CHAIN_RE = re.compile(
    r"(?is)\s*,\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<cols>[^)]*)\))?\s*AS\s*\("
)


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
        # ``WITH A AS (…), B AS (…) UPDATE …`` -- every CTE of the chain is
        # registered, not only the first.
        cte_match = m
        while cte_match is not None:
            name = cte_match.group("name")
            cols = cte_match.group("cols") or ""
            body_start = cte_match.end()
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
                break
            body = text[body_start:i]
            if not re.match(r"(?is)^\s*SELECT\b", body):
                break
            results.append(
                {
                    "name": name,
                    "cols": cols.strip(),
                    "body": body.strip(),
                    "start": str(cte_match.start()),
                    "end": str(i + 1),
                }
            )
            cte_match = _CTE_CHAIN_RE.match(text, i + 1)
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
    results = _extract_all_insert_select(sql)
    if temps_only:
        results = [r for r in results if r["target"].startswith("#")]
    return results


_INSERT_SELECT_RE = re.compile(
    rf"(?is)\bINSERT\s+INTO\s+(?P<target>{_TABLE_TOKEN})\s*"
    rf"(?:\((?P<cols>[^)]+)\))?\s*"
    rf"SELECT\b"
)


@_cached
def _extract_all_insert_select(sql: str) -> list[dict[str, str]]:
    text = strip_sql_comments(sql or "")
    results: list[dict[str, str]] = []
    for match in _INSERT_SELECT_RE.finditer(text):
        target_raw = match.group("target")
        target = normalize_table_name(target_raw)
        cols = (match.group("cols") or "").strip()
        start = match.start()
        pos = match.end()
        # ``INSERT … SELECT a UNION ALL SELECT b`` writes both branches into
        # the same target. Each branch becomes its own entry; the clause
        # readers stop at UNION so the first branch's WHERE no longer ends in
        # a dangling ``UNION ALL`` (which surfaced as an untranslated value).
        while True:
            select_list, pos = _read_until_keyword(
                text, pos, {"FROM", "UNION", "EXCEPT", "INTERSECT"}, stop_at_statement=True
            )
            from_body = ""
            where_clause = ""
            if _FROM_KW.match(text, pos):
                pos += len("FROM")
                from_body, pos = _read_until_keyword(
                    text,
                    pos,
                    {"WHERE", "GROUP", "ORDER", "HAVING", "UNION", "EXCEPT", "INTERSECT"},
                    stop_at_statement=True,
                )
            if _WHERE_KW.match(text, pos):
                pos += len("WHERE")
                where_clause, pos = _read_until_keyword(
                    text, pos,
                    {"GROUP", "ORDER", "HAVING", "UNION", "EXCEPT", "INTERSECT"},
                    stop_at_statement=True,
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
            union_match = _UNION_SELECT_KW.match(text, pos)
            if not union_match:
                break
            pos = union_match.end()
    return results
