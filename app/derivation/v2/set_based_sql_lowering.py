"""Row-level 4X lowering for set-based SQL fragments (windows, STRING_AGG, date UDFs).

Invoked from phase3 before unsupported sentinels. Does not change mutation folding.
"""
from __future__ import annotations

import re
from typing import Any, Callable

from app.derivation.v2.sql_text import bare_ident, split_csv_respecting_parens

_DATE_SENTINEL = {"type": "LITERAL", "value_type": "STRING", "value": "2099-12-31"}
_EMPTY_STRING = {"type": "LITERAL", "value_type": "STRING", "value": ""}

_ROW_NUMBER_EQ_ONE = re.compile(
    r"(?is)^ROW_NUMBER\s*\(\s*\)\s+OVER\s*\(\s*(?P<over>.*?)\s*\)\s*=\s*1\s*$"
)
_PARTITION_ORDER = re.compile(
    r"(?is)^PARTITION\s+BY\s+(?P<part>.*?)\s+ORDER\s+BY\s+(?P<order>.+)$"
)
_STRING_AGG_CALL = re.compile(
    r"(?is)^STRING_AGG\s*\(\s*(?P<inner>.*)\s*,\s*(?P<sep>.*)\s*\)\s*$"
)
_CHOOSE_CALL = re.compile(
    r"(?is)^(?:[\w\[\].]+\.)*CHOOSE\s*\(\s*(?P<args>.*)\s*\)\s*$"
)
_BETWEEN_PREDICATE = re.compile(
    r"(?is)^(.+?)\s+BETWEEN\s+(.+?)\s+AND\s+(.+)$"
)
_PERCENT_PLUS_MID = re.compile(
    r"(?is)^(?:N)?'%'\s*\+\s*(?P<mid>.+?)\s*\+\s*(?:N)?'%'\s*$"
)
_PERCENT_PLUS_MID_TIGHT = re.compile(
    r"(?is)^(?:N)?'%'\+(?P<mid>.+?)\+(?:N)?'%'\s*$"
)


def try_parse_row_number_rank_one(
    text: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """``ROW_NUMBER() OVER (...) = 1`` → ``val == MAX/MIN(val, [partition keys])``."""
    raw = (text or "").strip()
    m = _ROW_NUMBER_EQ_ONE.match(raw)
    if not m:
        return None
    over = (m.group("over") or "").strip()
    po = _PARTITION_ORDER.match(over)
    if not po:
        return None
    part_sql = (po.group("part") or "").strip()
    order_sql = (po.group("order") or "").strip()
    order_parts = split_csv_respecting_parens(order_sql)
    if not order_parts:
        return None
    first_order = order_parts[0].strip()
    desc = bool(re.search(r"(?is)\bDESC\b", first_order))
    first_order = re.sub(r"(?is)\s+DESC\s*$", "", first_order).strip()
    first_order = re.sub(r"(?is)\s+ASC\s*$", "", first_order).strip()
    val_ast = parse_inner(
        first_order, default_entity=default_entity, target_column=target_column
    )
    keys: list[str] = []
    for part in split_csv_respecting_parens(part_sql):
        key = bare_ident(part.strip().strip("[]"))
        if key:
            keys.append(key)
    group_list = {"type": "LIST_LITERAL", "items": keys} if keys else None
    agg_fn = "MAX" if desc else "MIN"
    agg_args: list[dict[str, Any]] = [val_ast]
    if group_list is not None:
        agg_args.append(group_list)
    agg_ast = {"type": "FUNCTION_CALL", "function_name": agg_fn, "arguments": agg_args}
    return {
        "type": "BINARY_OP",
        "operator": "==",
        "left": val_ast,
        "right": agg_ast,
    }


def try_parse_string_agg_concat(
    text: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """``STRING_AGG(col, sep)`` → ``CONCAT(COALESCE(col,''), …)`` (row-level approximation)."""
    raw = (text or "").strip()
    m = _STRING_AGG_CALL.match(raw)
    if not m:
        return None
    inner = (m.group("inner") or "").strip()
    sep_sql = (m.group("sep") or "").strip()
    if not _balanced_parens(inner) or not _balanced_parens(sep_sql):
        return None
    col_ast = parse_inner(
        inner, default_entity=default_entity, target_column=target_column
    )
    sep_ast = parse_inner(
        sep_sql, default_entity=default_entity, target_column=target_column
    )
    left = _coalesce_empty_string(col_ast)
    # Second slot: optional separator + duplicate coalesce when inner lists two cols.
    inner_parts = split_csv_respecting_parens(inner) if "," in inner and not inner.upper().startswith("CONCAT") else []
    args: list[dict[str, Any]] = [left]
    if len(inner_parts) >= 2:
        second = parse_inner(
            inner_parts[1].strip(),
            default_entity=default_entity,
            target_column=target_column,
        )
        sep_guard = _and_nonempty(col_ast, second)
        sep_branch = {
            "type": "IF_THEN_ELSE",
            "condition": sep_guard,
            "then_branch": sep_ast,
            "else_branch": _EMPTY_STRING,
        }
        args.extend([sep_branch, _coalesce_empty_string(second)])
    return {"type": "FUNCTION_CALL", "function_name": "CONCAT", "arguments": args}


def try_parse_choose_chain(
    text: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """``CHOOSE(idx, v1, v2, …)`` → nested ``IF(idx==n) THEN(vn) …`` (1-based index)."""
    raw = (text or "").strip()
    m = _CHOOSE_CALL.match(raw)
    if not m or not _balanced_parens(m.group("args") or ""):
        return None
    choose_args = split_csv_respecting_parens(m.group("args") or "")
    if len(choose_args) < 2:
        return None
    index_node = parse_inner(
        choose_args[0].strip(),
        default_entity=default_entity,
        target_column=target_column,
    )
    chosen: dict[str, Any] = {"type": "LITERAL", "value_type": "NULL", "value": None}
    for position in range(len(choose_args) - 1, 0, -1):
        chosen = {
            "type": "IF_THEN_ELSE",
            "condition": {
                "type": "BINARY_OP",
                "operator": "==",
                "left": index_node,
                "right": {"type": "LITERAL", "value_type": "NUMBER", "value": position},
            },
            "then_branch": parse_inner(
                choose_args[position].strip(),
                default_entity=default_entity,
                target_column=target_column,
            ),
            "else_branch": chosen,
        }
    return chosen


def try_parse_between_range(
    text: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """``expr BETWEEN low AND high`` → ``AND(expr>=low, expr<=high)``."""
    raw = (text or "").strip()
    m = _BETWEEN_PREDICATE.match(raw)
    if not m:
        return None
    col = parse_inner(
        m.group(1).strip(),
        default_entity=default_entity,
        target_column=target_column,
    )
    low = parse_inner(
        m.group(2).strip(),
        default_entity=default_entity,
        target_column=target_column,
    )
    high = parse_inner(
        m.group(3).strip(),
        default_entity=default_entity,
        target_column=target_column,
    )
    return {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": {"type": "BINARY_OP", "operator": ">=", "left": col, "right": low},
        "right": {"type": "BINARY_OP", "operator": "<=", "left": col, "right": high},
    }


def try_parse_percent_wildcard_concat_mid(
    pattern_sql: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
) -> dict[str, Any] | None:
    """``'%' + col + '%'`` / ``'%'+col+'%'`` → AST for ``col`` (CONTAINS needle)."""
    raw = (pattern_sql or "").strip()
    mid_match = _PERCENT_PLUS_MID.match(raw) or _PERCENT_PLUS_MID_TIGHT.match(raw)
    if not mid_match:
        return None
    return parse_inner(
        mid_match.group("mid").strip(),
        default_entity=default_entity,
        target_column=target_column,
    )


def build_dynamic_like_contains_ast(
    lhs_sql: str,
    pattern_sql: str,
    parse_inner: Callable[..., dict[str, Any]],
    *,
    default_entity: str,
    target_column: str = "",
    negate: bool = False,
) -> dict[str, Any] | None:
    """``lhs LIKE '%' + rhs + '%'`` → ``lhs CONTAINS [rhs]`` membership."""
    needle = try_parse_percent_wildcard_concat_mid(
        pattern_sql,
        parse_inner,
        default_entity=default_entity,
        target_column=target_column,
    )
    if needle is None:
        return None
    lhs_node = parse_inner(
        lhs_sql.strip(),
        default_entity=default_entity,
        target_column=target_column,
    )
    operator = "DOESNOTCONTAINS" if negate else "CONTAINS"
    return {
        "type": "MEMBERSHIP_OP",
        "operator": operator,
        "column": lhs_node,
        "values": [needle],
    }


def coalesce_date_args_for_minimum(
    nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Wrap date-like column refs for scalar least-date UDF lowering."""
    out: list[dict[str, Any]] = []
    for node in nodes:
        if _looks_like_date_column_ast(node):
            out.append(_coalesce_sentinel_date(node))
        else:
            out.append(node)
    return out


def _looks_like_date_column_ast(node: dict[str, Any]) -> bool:
    if node.get("type") != "COLUMN_REF":
        return False
    col = str(node.get("column") or "").upper()
    return "DT" in col or "DATE" in col or col.endswith("TIMEKEY")


def _coalesce_sentinel_date(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FUNCTION_CALL",
        "function_name": "COALESCE",
        "arguments": [node, _DATE_SENTINEL],
    }


def _coalesce_empty_string(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FUNCTION_CALL",
        "function_name": "COALESCE",
        "arguments": [node, _EMPTY_STRING],
    }


def _and_nonempty(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "FUNCTION_CALL",
        "function_name": "AND",
        "arguments": [
            {
                "type": "FUNCTION_CALL",
                "function_name": "ISNOTEMPTY",
                "arguments": [left],
            },
            {
                "type": "FUNCTION_CALL",
                "function_name": "ISNOTEMPTY",
                "arguments": [right],
            },
        ],
    }


def _balanced_parens(text: str) -> bool:
    depth = 0
    in_quote = False
    for ch in text:
        if ch == "'":
            in_quote = not in_quote
            continue
        if in_quote:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0
