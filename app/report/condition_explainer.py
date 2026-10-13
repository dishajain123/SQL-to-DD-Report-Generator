"""Deterministic human-readable explanations for platform conditions.

The explainer walks the real 4X grammar tree and turns the accepted
machine-readable condition into a concise business decision flow without
changing the underlying logic.
"""
from __future__ import annotations

import contextvars
import json
import re
from pathlib import Path

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

from app.derivation.derivation_option import quote_expression_parameters
from app.derivation.v2.sql_text import bare_ident, entity_key_upper

_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"
_parser = Lark(_GRAMMAR_PATH.read_text(), parser="earley", start="start")

# DD export / build_dynamic_rule_sentence: preserve Table.Column and 'Y'/'N' literals.
_EXPORT_LITERAL_STYLE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_EXPORT_LITERAL_STYLE", default=False
)
_EXPORT_NO_TRUNCATE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_EXPORT_NO_TRUNCATE", default=False
)
_EXPORT_TARGET_COLUMN: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_EXPORT_TARGET_COLUMN", default=""
)

_MEMBERSHIP_WORDS = {
    "IN": "is one of",
    "NOTIN": "is not one of",
    "HRCHYIN": "is within the hierarchy of",
    "HRCHYNOTIN": "is outside the hierarchy of",
    "CONTAINS": "contains",
    "BEGINSWITH": "begins with",
    "ENDSWITH": "ends with",
    "DOESNOTCONTAINS": "does not contain",
}

_BOOLEAN_FUNCTIONS = {"AND", "OR"}
_PRESERVED_ACRONYMS = {"DPD", "SMA", "ID"}
_IDENTIFIER_REPLACEMENTS = {"REF": "reference", "FLG": "flag"}
_COMPOUND_WORDS = {
    ("over", "drawn"): "overdrawn",
    ("over", "due"): "overdue",
    ("no", "credit"): "no credit",
    ("stock", "stmt"): "stock statement",
    ("int", "service"): "int service",
}
_ALL_CAPS_PARTS = sorted(
    {
        "OVERDRAWN",
        "OVERDUE",
        "STATEMENT",
        "PERIOD",
        "SERVICE",
        "CREDIT",
        "REVIEW",
        "RENEWAL",
        "PROCESSING",
        "PROCESS",
        "RUNNING",
        "NAME",
        "ACCOUNT",
        "LATE",
        "FEE",
        "ASSESSMENT",
        "MAX",
        "MIN",
        "FLG",
        "REF",
        "SMA",
        "DPD",
        "ID",
        "NO",
        "INT",
        "DEG",
    },
    key=len,
    reverse=True,
)


def _parse_expression_tree(expression: str):
    """Parse display-form 4X (bare ``@TIMEKEY``) into a grammar tree."""
    prepared = quote_expression_parameters((expression or "").strip())
    if not prepared:
        return None
    return _parser.parse(prepared)


def explain_expression(expression: str) -> str | None:
    try:
        tree = _parse_expression_tree(expression)
        if tree is None:
            return None
    except LarkError:
        return None

    try:
        node = _unwrap(tree)
        lines = _render_node(node, depth=0)
    except Exception:
        return None

    return "\n".join(_normalize_line(line) for line in lines if line.strip())


def _normalize_line(text: str) -> str:
    # Preserve leading indentation so nested outcomes stay under their branch.
    leading_ws = text[: len(text) - len(text.lstrip(" "))]
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return ""
    if stripped[-1] not in ".!?:":
        stripped += "."
    # Keep markdown bullets / outcomes lowercase after the marker where needed,
    # but capitalize the first alphabetic character of the visible sentence.
    body = stripped
    prefix = ""
    if body.startswith("- "):
        prefix = "- "
        body = body[2:]
    if body:
        body = body[0].upper() + body[1:]
    return f"{leading_ws}{prefix}{body}"


def _unwrap(node):
    while isinstance(node, Tree) and node.data not in {
        "if_expr",
        "function_call",
        "column_ref",
        "value_list",
        "arg_list",
        "not_op",
        "neg",
    } and len(node.children) == 1:
        node = node.children[0]
    return node


def _render_node(node, depth: int) -> list[str]:
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "if_expr":
        return _render_if_chain(node, depth)
    return _render_outcome(node, depth, bulleted=depth == 0)


def _render_if_chain(node: Tree, depth: int) -> list[str]:
    children = list(node.children)
    if len(children) < 2:
        return [f"{_indent(depth)}Return the calculated value"]

    condition = children[0]
    then_branch = children[1]
    rest = children[2:]
    elseif_clauses = [c for c in rest if isinstance(c, Tree) and c.data == "elseif_clause"]
    else_clause = next((c for c in rest if isinstance(c, Tree) and c.data == "else_clause"), None)

    lines: list[str] = []
    lines.extend(_render_branch("If", condition, then_branch, depth))
    for clause in elseif_clauses:
        clause_condition, clause_branch = clause.children[0], clause.children[1]
        lines.extend(_render_branch("Otherwise, if", clause_condition, clause_branch, depth))
    if else_clause is not None:
        lines.append(f"{_indent(depth)}- Otherwise:")
        lines.extend(_render_branch_result(else_clause.children[0], depth + 1))
    return lines


def _render_branch(prefix: str, condition, branch, depth: int) -> list[str]:
    lines = [f"{_indent(depth)}- {prefix} {_render_condition(condition)}:"]
    lines.extend(_render_branch_result(branch, depth + 1))
    return lines


def _render_outcome(node, depth: int, bulleted: bool = False) -> list[str]:
    value, note = _render_value_with_note(node)
    if value == "no value":
        text = "Leave the field blank"
    elif note:
        text = f"Return {value}, {note}"
    else:
        text = f"Return {value}"
    marker = "- " if bulleted else ""
    return [f"{_indent(depth)}{marker}{text}"]


def _render_branch_result(node, depth: int) -> list[str]:
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "if_expr":
        return _render_if_chain(node, depth)
    # Leaf outcomes are indented prose under the branch — not another bullet.
    return _render_outcome(node, depth, bulleted=False)


def _render_condition(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Tree):
        if node.data == "or_op":
            parts = _flatten_chain(node, "or_op")
            return _join_condition_parts(
                [_render_condition(part) for part in parts],
                conjunction="or",
                prefix="at least one of the following is true:",
            )
        if node.data == "and_op":
            parts = _flatten_chain(node, "and_op")
            return _join_condition_parts(
                [_render_condition(part) for part in parts],
                conjunction="and",
                prefix="all of the following are true:",
            )
        if node.data == "or_call":
            return _join_condition_parts(
                [_render_condition(child) for child in node.children],
                conjunction="or",
                prefix="at least one of the following is true:",
            )
        if node.data == "and_call":
            return _join_condition_parts(
                [_render_condition(child) for child in node.children],
                conjunction="and",
                prefix="all of the following are true:",
            )
        if node.data == "not_op":
            return f"It is not true that {_render_condition(node.children[0])}"
        if node.data == "compare" and len(node.children) == 3:
            left, op, right = node.children
            return _render_compare_condition(left, str(op), right)
        if node.data == "between_op" and len(node.children) == 3:
            value, low, high = node.children
            value_text, value_note = _render_value_with_note(value, as_operand=True)
            low_text, low_note = _render_value_with_note(low, as_operand=True)
            high_text, high_note = _render_value_with_note(high, as_operand=True)
            phrase = f"{value_text} is between {low_text} and {high_text}"
            notes = [note for note in (value_note, low_note, high_note) if note]
            if notes:
                phrase = f"{phrase}, {'; '.join(dict.fromkeys(notes))}"
            return phrase
        if node.data == "membership":
            return _render_membership(node)
        if node.data == "function_call":
            return _render_function_condition(node)
        if node.data == "if_expr":
            return _summarize_nested_if_outcome(node)
    if isinstance(node, Tree):
        return _export_tree_node(node)
    return _render_value(node)


def _render_comparison(left: str, op: str, right: str) -> str:
    op = op.strip()
    phrases = {
        "==": "equals",
        "!=": "does not equal",
        ">": "is greater than",
        ">=": "is greater than or equal to",
        "<": "is less than",
        "<=": "is less than or equal to",
    }
    if op in phrases:
        return f"{left} {phrases[op]} {right}"
    return f"{left} compared to {right}"


def _render_compare_condition(left, op: str, right) -> str:
    left_func = _unwrap(left)
    right_func = _unwrap(right)

    if isinstance(left_func, Tree) and left_func.data == "function_call" and _is_coalesce_function(left_func):
        return _render_coalesce_compare(left_func, op, right)
    if isinstance(right_func, Tree) and right_func.data == "function_call" and _is_coalesce_function(right_func):
        return _render_coalesce_compare(right_func, op, left, reversed_sides=True)

    left_value, left_note = _render_value_with_note(left, as_operand=True)
    right_value, right_note = _render_value_with_note(right, as_operand=True)
    comparison = _render_comparison(left_value, op, right_value)
    notes = [note for note in (left_note, right_note) if note]
    if notes:
        comparison = f"{comparison}, {'; '.join(dict.fromkeys(notes))}"
    return comparison


def _render_coalesce_compare(node: Tree, op: str, other, reversed_sides: bool = False) -> str:
    args = _function_args(node)
    if not args:
        return _render_compare_condition(node, op, other)  # pragma: no cover - defensive fallback

    if _EXPORT_LITERAL_STYLE.get():
        return _render_coalesce_compare_export(node, op, other, reversed_sides=reversed_sides)

    primary = _compact_field_phrase(_render_value(args[0], as_operand=True))
    fallback = args[1] if len(args) > 1 else None
    other_value, other_note = _render_value_with_note(other, as_operand=True)
    note = _coalesce_note(fallback) if fallback is not None else None

    if reversed_sides:
        comparison = _render_comparison(other_value, op, primary)
    else:
        comparison = _render_comparison(primary, op, other_value)

    notes = [note for note in (other_note, note) if note]
    if notes:
        comparison = f"{comparison}, {'; '.join(dict.fromkeys(notes))}"
    return comparison


def _render_coalesce_compare_export(
    node: Tree, op: str, other, *, reversed_sides: bool = False
) -> str:
    args = _function_args(node)
    primary = _export_operand(args[0])
    fallback = args[1] if len(args) > 1 else None
    other_value = _export_operand(other)
    if reversed_sides:
        comparison = _render_comparison(other_value, op, primary)
    else:
        comparison = _render_comparison(primary, op, other_value)
    null_note = _coalesce_null_fallback_phrase(fallback)
    if null_note:
        return f"{comparison} ({null_note})"
    return comparison


def _coalesce_null_fallback_phrase(fallback_node) -> str | None:
    if fallback_node is None:
        return None
    if _is_zeroish_literal(fallback_node):
        return "treating NULL as 0"
    if _is_nullish_literal(fallback_node):
        return "treating NULL as NULL"
    rendered = _export_operand(fallback_node)
    if not rendered or rendered == "field":
        return None
    return f"treating NULL as {rendered}"


def _is_coalesce_function(node: Tree) -> bool:
    name = str(node.children[0]).upper() if node.children else ""
    return name in {"COALESCE", "NVL", "ISNULL"}


def _render_membership(node: Tree) -> str:
    children = list(node.children)
    if len(children) < 2:
        return _render_value(node)

    lhs = _render_value(children[0])
    op = _membership_operator(children)
    if op is None:
        return _render_value(node)

    rhs = _render_membership_rhs(children[2] if len(children) > 2 else None)
    phrase = _MEMBERSHIP_WORDS.get(op, "relates to")
    return f"{lhs} {phrase} {rhs}"


def _membership_operator(children: list) -> str | None:
    for child in children:
        if isinstance(child, Token):
            value = child.value.upper()
            if value in _MEMBERSHIP_WORDS:
                return value
    return None


def _render_membership_rhs(node) -> str:
    if node is None:
        return "the listed values"
    if isinstance(node, Tree) and node.data == "value_list":
        values = [_render_value(child) for child in node.children]
        if not values:
            return "the listed values"
        return _join_items(values, "or")
    return _render_value(node)


def _render_function_condition(node: Tree) -> str:
    name = str(node.children[0]).upper()
    args = _function_args(node)
    phrase, note = _render_function_phrase(name, args, direct=False)
    if phrase is not None:
        if note:
            return f"{phrase}, {note}"
        return phrase
    return "the result of the function call"


def _is_conditional_value(node) -> bool:
    """True if this argument node is itself an if/then/else expression --
    the shape SQL's MIN(CASE...)/MAX(CASE...) takes once translated."""
    unwrapped = _unwrap(node)
    return isinstance(unwrapped, Tree) and unwrapped.data == "if_expr"


def _render_if_expr_inline(node: Tree) -> str:
    """Compact single-line rendering of an if/then/else expression used
    as a VALUE (e.g. the sole argument to MIN/MAX), as opposed to
    _render_if_chain's multi-line bullet form used for a top-level
    condition. Rendering it as a full decision chain here would break
    the surrounding sentence; rendering only the condition (the default
    value-context behavior) would silently drop the THEN/ELSE branches,
    which is what produced the original bug."""
    children = list(node.children)
    if len(children) < 2:
        return "the calculated value"

    condition = children[0]
    then_branch = children[1]
    rest = children[2:]
    elseif_clauses = [c for c in rest if isinstance(c, Tree) and c.data == "elseif_clause"]
    else_clause = next((c for c in rest if isinstance(c, Tree) and c.data == "else_clause"), None)

    parts = [f"if {_render_condition(condition)}, then {_render_value(then_branch)}"]
    for clause in elseif_clauses:
        clause_condition, clause_branch = clause.children[0], clause.children[1]
        parts.append(f"else if {_render_condition(clause_condition)}, then {_render_value(clause_branch)}")
    if else_clause is not None:
        parts.append(f"otherwise {_render_value(else_clause.children[0])}")
    return "; ".join(parts)


def _render_function_phrase(name: str, args: list, direct: bool) -> tuple[str | None, str | None]:
    if name == "ISEMPTY":
        return (f"{_render_value(args[0])} is blank or missing" if args else "the value is blank or missing", None)
    if name == "ISNOTEMPTY":
        return (f"{_render_value(args[0])} has a value" if args else "the value has a value", None)
    if name in {"COALESCE", "NVL", "ISNULL"}:
        if not args:
            return "the fallback value", None
        if len(args) == 1:
            value = _render_value(args[0], as_operand=True)
            return value, None
        primary = _render_value(args[0], as_operand=True)
        fallback = _render_value(args[1], as_operand=True)
        note = _coalesce_note(args[1])
        if direct:
            if note and len(args) == 2:
                return _compact_field_phrase(primary), note
            if len(args) == 2:
                return f"use {primary}; otherwise, {fallback}", None
            return "use " + "; otherwise, ".join(_render_value(arg, as_operand=True) for arg in args), None
        if note and len(args) == 2:
            return _compact_field_phrase(primary), note
        if len(args) == 2:
            return _compact_field_phrase(primary), None
        return "use " + "; otherwise, ".join(_render_value(arg, as_operand=True) for arg in args), None
    if name == "CONCAT":
        return f"the combined text of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "TODATE":
        return f"the date value for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "DATE":
        return f"the date value for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "DATEDIFF":
        if len(args) >= 3:
            unit = _render_value(args[2]).lower()
            return f"the number of {unit} between {_render_value(args[0])} and {_render_value(args[1])}", None
        return f"the elapsed time between {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "DATEPART":
        if len(args) >= 2:
            return f"the {_render_value(args[0]).lower()} from {_render_value(args[1])}", None
        return f"the extracted date part from {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "MAX" or name == "MIN":
        superlative = "highest" if name == "MAX" else "lowest"
        if len(args) == 1 and _is_conditional_value(args[0]):
            # SQL's MIN(CASE WHEN ... END) / MAX(CASE WHEN ... END)
            # pattern arrives here as MIN(IF(...)THEN(...)ELSE(...)) --
            # a single conditional argument, not a flat list of values to
            # compare. Rendering it with the generic value-list path below
            # would only surface the condition text and silently drop the
            # THEN/ELSE branches (e.g. "the lowest of If X is true"),
            # which is incomplete to the point of being misleading about
            # what is actually being compared.
            return f"the {superlative} value from: {_render_if_expr_inline(_unwrap(args[0]))}", None
        if len(args) >= 2:
            # Documented platform signature: MAX(<ColumnName>, [<GroupbyColumns>])
            # -- the trailing args are grouping columns, not more values
            # to compare against the first.
            primary = _render_value(args[0])
            group_cols = _join_items([_render_value(arg) for arg in args[1:]], "and")
            return f"the {superlative} {primary}, evaluated per group of {group_cols}", None
        return f"the {superlative} of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "ADDDAY":
        if len(args) >= 2:
            return f"the date obtained by adding {_render_value(args[1])} days to {_render_value(args[0])}", None
        return f"the adjusted date for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "CONVERT":
        if len(args) >= 2:
            return f"{_render_value(args[0])} converted to {_render_value(args[1])}", None
        return f"the converted value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "SOQ":
        return f"the start of the quarter for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "EOQ":
        return f"the end of the quarter for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "ABS":
        return f"the absolute value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "ROUND":
        return f"the rounded value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "FLOOR":
        return f"the rounded-down value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "CEIL":
        return f"the rounded-up value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "REGEX":
        return f"a pattern match on {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "LOWER":
        return f"the lowercased value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "UPPER":
        return f"the uppercased value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "TRIM":
        return f"the trimmed value of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "LEN":
        return f"the length of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "SUBSTR":
        if len(args) >= 3:
            return f"the substring of {_render_value(args[0])} starting at {_render_value(args[1])} with length {_render_value(args[2])}", None
        if len(args) >= 2:
            return f"the substring of {_render_value(args[0])} starting at {_render_value(args[1])}", None
        return f"the substring of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "REPLACE":
        if len(args) >= 3:
            return f"the value of {_render_value(args[0])} with {_render_value(args[1])} replaced by {_render_value(args[2])}", None
        return f"the replaced text of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "SOM":
        return f"the start of the month for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "EOM":
        return f"the end of the month for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "SOY":
        return f"the start of the year for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "EOY":
        return f"the end of the year for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "SOFY":
        return f"the start of the financial year for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "EOFY":
        return f"the end of the financial year for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "PERIOD":
        return f"the period value for {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name in _BOOLEAN_FUNCTIONS:
        conj = "and" if name == "AND" else "or"
        prefix = "all of" if name == "AND" else "any of"
        return _join_condition_parts([_render_condition(arg) for arg in args], conj, prefix=prefix), None
    if not args:
        return None, None
    return None, None


def _function_args(node: Tree) -> list:
    if len(node.children) < 2:
        return []
    args_node = node.children[1]
    if isinstance(args_node, Tree) and args_node.data == "arg_list":
        return list(args_node.children)
    return [args_node]


def _render_value(node, as_operand: bool = False) -> str:
    value, _ = _render_value_with_note(node, as_operand=as_operand)
    return value


def _render_value_with_note(node, as_operand: bool = False) -> tuple[str, str | None]:
    node = _unwrap(node)
    if _EXPORT_LITERAL_STYLE.get() and as_operand:
        exported = _export_operand(node)
        if exported and exported not in {"calculated rule value", "field", "conditional result"}:
            return exported, None
    if isinstance(node, Token):
        return _render_token(node), None
    if isinstance(node, Tree):
        if node.data == "column_ref":
            if as_operand or _EXPORT_LITERAL_STYLE.get():
                path = _column_ref_physical_path(node)
                if path:
                    return path, None
            return _render_column_ref(node), None
        if node.data == "function_call":
            return _render_function_value_with_note(node, direct=not as_operand)
        if node.data == "if_expr":
            return _summarize_nested_if_outcome(node), None
        if node.data in {"compare", "between_op", "membership", "and_op", "or_op", "not_op", "and_call", "or_call"}:
            return _render_export_condition(node), None
        if node.data == "add":
            return _render_additive(node, "plus"), None
        if node.data == "sub":
            return _render_binary(node, "minus"), None
        if node.data == "mul":
            return _render_additive(node, "multiplied by"), None
        if node.data == "div":
            return _render_binary(node, "divided by"), None
        if node.data == "neg":
            return f"negative {_render_value(node.children[0])}", None
        if node.data == "value_list":
            return _join_items([_render_value(child) for child in node.children], "or"), None
        if len(node.children) == 1:
            return _render_value_with_note(node.children[0], as_operand=as_operand)
        return _export_tree_node(node), None
    return "the calculated value", None


def _render_function_value_with_note(node: Tree, direct: bool = True) -> tuple[str, str | None]:
    name = str(node.children[0]).upper()
    args = _function_args(node)
    phrase, note = _render_function_phrase(name, args, direct=direct)
    if phrase is not None:
        return phrase, note
    return "the result of the function call", None


def _render_function_value(node: Tree, direct: bool = True) -> str:
    return _render_function_value_with_note(node, direct=direct)[0]


def _render_binary(node: Tree, op_word: str) -> str:
    left, right = node.children[0], node.children[1]
    return f"{_render_value(left, as_operand=True)} {op_word} {_render_value(right, as_operand=True)}"


def _render_additive(node: Tree, op_word: str) -> str:
    values = [_render_value(child, as_operand=True) for child in node.children]
    return _join_items(values, op_word)


def _render_column_ref(node: Tree) -> str:
    rendered_parts: list[str] = []
    original_parts: list[tuple[str, bool]] = []
    for child in node.children:
        text = ""
        is_string_literal = False
        if isinstance(child, Tree) and child.data == "path_part" and child.children:
            part = child.children[0]
            if isinstance(part, Token):
                if part.type == "STRING":
                    try:
                        text = json.loads(str(part))
                    except Exception:
                        text = str(part).strip('"')
                    is_string_literal = True
                else:
                    text = str(part)
            else:
                text = _render_identifier(part)
        else:
            text = _render_identifier(child)
        if is_string_literal:
            raw_literal = text
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
                text = _render_name_text(text)
            rendered_parts.append(text)
            original_parts.append((raw_literal, is_string_literal))
            continue
        text = text.strip()
        if text:
            rendered_parts.append(_render_name_text(text))
            original_parts.append((text, is_string_literal))
    if not rendered_parts:
        return "the field"
    had_qualifiers = len(original_parts) > 1
    if len(rendered_parts) >= 2 and rendered_parts[-2].upper() == "VAR" and rendered_parts[-1].upper() == "BUSINESS_DATE":
        return "the business date"
    if len(rendered_parts) >= 2 and _looks_like_alias(rendered_parts[0]):
        rendered_parts = rendered_parts[1:]
        original_parts = original_parts[1:]
        if not rendered_parts:
            return "the field"
    if len(rendered_parts) == 1:
        text, _was_string = original_parts[0]
        if not had_qualifiers and (
            _was_string
            or text.upper() in {"TRUE", "FALSE", "NULL", "YES", "NO", "Y", "N"}
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text)
        ):
            return _render_literal_text(text)
        if rendered_parts[0].lower() == "business date":
            return "the business date"
        return f"the {rendered_parts[0]} field"
    return _compact_field_phrase(f"the {rendered_parts[-1]} field")


def _render_identifier(value: object) -> str:
    if isinstance(value, Tree):
        value = _unwrap(value)
        if isinstance(value, Tree):
            if value.data == "path_part" and value.children:
                part = value.children[0]
                if isinstance(part, Token):
                    if part.type == "STRING":
                        try:
                            text = json.loads(str(part))
                        except Exception:
                            text = str(part).strip('"')
                        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", text) and text.upper() not in {"TRUE", "FALSE", "NULL", "YES", "NO", "Y", "N"}:
                            return _render_literal_text(text)
                    else:
                        text = str(part)
                    return _render_name_text(text)
                return _render_name_text(_render_identifier(part))
            if len(value.children) == 1:
                return _render_identifier(value.children[0])
            if value.data == "column_ref":
                return _render_column_ref(value)
            if value.data == "function_call":
                return _render_function_value(value)
            return "the calculated value"
    return _render_name_text(str(value))


def _render_name_text(text: str) -> str:
    text = text.strip().strip('"')
    if not text:
        return ""
    if text.upper() == "BUSINESS_DATE":
        return "business date"
    words = _split_identifier_words(text)
    if not words:
        return text.lower()
    words = _normalize_identifier_words(words)
    return " ".join(words)


def _render_literal_text(text: str) -> str:
    raw = str(text)
    if raw == "":
        return "an empty string"
    if raw.isspace():
        if len(raw) == 1:
            return "a space"
        return f"{len(raw)} spaces"
    text = raw.strip().strip('"')
    if not text:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    mapping = {
        "Y": "yes",
        "N": "no",
        "YES": "yes",
        "NO": "no",
        "TRUE": "true",
        "FALSE": "false",
        "NULL": "no value",
    }
    upper = text.upper()
    if upper in mapping:
        return mapping[upper]
    # Keep coded procedure / status identifiers readable and exact.
    if "_" in text and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", text):
        return f'"{text}"'
    return _render_name_text(text)


def _split_identifier_words(text: str) -> list[str]:
    words: list[str] = []
    for chunk in text.replace("-", "_").split("_"):
        if not chunk:
            continue
        if chunk.isupper() and len(chunk) > 1 and chunk.upper() not in _PRESERVED_ACRONYMS:
            parts = _split_all_caps_chunk(chunk)
        else:
            parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[0-9]+", chunk)
            if not parts:
                parts = [chunk]
        for part in parts:
            upper = part.upper()
            if upper in _PRESERVED_ACRONYMS:
                words.append(upper)
            elif upper in _IDENTIFIER_REPLACEMENTS:
                words.append(_IDENTIFIER_REPLACEMENTS[upper])
            elif part.isdigit():
                words.append(part)
            else:
                words.append(part.lower())
    return words


def _split_all_caps_chunk(chunk: str) -> list[str]:
    """Split a run-together all-caps identifier into words, but only when
    EVERY resulting fragment is a real recognized dictionary word -- never
    a leftover guess.

    A length threshold on leftover fragments is not enough: "NORMAL" ->
    "NO" + "RMAL" has a 4-character leftover, long enough to pass a naive
    length check, but "rmal" is still meaningless. Likewise "COVID" ->
    "COV" + "ID" leaves a plausible-looking 3-letter leftover that is still
    wrong. The only reliable rule is full coverage: if the whole chunk
    cannot be built entirely out of known words, don't guess at a partial
    split -- leave it as one word instead of risking an explanation that
    quietly changes what the condition means.
    """
    parts = _greedy_all_caps_split(chunk)
    if len(parts) <= 1:
        return parts
    if all(part.upper() in _ALL_CAPS_PARTS for part in parts):
        return parts
    return [chunk]


def _greedy_all_caps_split(chunk: str) -> list[str]:
    upper = chunk.upper()
    parts: list[str] = []
    i = 0
    while i < len(chunk):
        matched = None
        matched_pos = None
        for candidate in _ALL_CAPS_PARTS:
            pos = upper.find(candidate, i)
            if pos == -1:
                continue
            if matched_pos is None or pos < matched_pos or (pos == matched_pos and len(candidate) > len(matched or "")):
                matched = candidate
                matched_pos = pos
        if matched is None or matched_pos is None:
            parts.append(chunk[i:])
            break
        if matched_pos > i:
            parts.append(chunk[i:matched_pos])
            i = matched_pos
            continue
        parts.append(matched)
        i += len(matched)
    return parts


def _normalize_identifier_words(words: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            pair = (words[i].lower(), words[i + 1].lower())
            if pair in _COMPOUND_WORDS:
                normalized.append(_COMPOUND_WORDS[pair])
                i += 2
                continue
        normalized.append(words[i])
        i += 1

    if len(normalized) > 2 and normalized[0].lower() == "reference" and normalized[1].lower() == "period":
        tail_phrase = " ".join(word.lower() for word in normalized[2:])
        if tail_phrase in {"overdrawn", "overdue", "stock statement", "int service", "no credit"}:
            normalized = [normalized[0], *normalized[2:], normalized[1]]
    return normalized


def _looks_like_alias(text: str) -> bool:
    compact = text.strip()
    return len(compact) <= 2 and compact.upper() not in _PRESERVED_ACRONYMS


def _with_article(text: str) -> str:
    stripped = text.strip()
    if stripped.lower().startswith(("the ", "a ", "an ")):
        return stripped
    return f"the {stripped}"


def _compact_field_phrase(text: str) -> str:
    stripped = text.strip()
    stripped = re.sub(r"\bfield\b", "", stripped, count=1)
    stripped = re.sub(r"\s+in\s+.+$", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return _with_article(stripped)


def _coalesce_note(node) -> str | None:
    if _is_zeroish_literal(node):
        return "treating blank as 0"
    if _is_nullish_literal(node):
        return "treating blank as no value"
    literal = _render_literal_note(node)
    if literal is not None:
        return f"treating blank as {literal}"
    return None


def _render_literal_note(node) -> str | None:
    node = _unwrap(node)
    if isinstance(node, Tree) and len(node.children) == 1:
        return _render_literal_note(node.children[0])
    if isinstance(node, Token):
        if node.type == "NUMBER":
            return str(node)
        if node.type == "STRING":
            try:
                value = json.loads(str(node))
            except Exception:
                value = str(node).strip('"')
            rendered = _render_literal_text(value)
            return rendered if rendered else None
    return None


def _is_zeroish_literal(node) -> bool:
    node = _unwrap(node)
    if isinstance(node, Token):
        if node.type == "NUMBER":
            return str(node) in {"0", "0.0"}
        if node.type == "STRING":
            try:
                value = json.loads(str(node))
            except Exception:
                value = str(node).strip('"')
            return value.strip() in {"0", "0.0"}
    if isinstance(node, Tree) and node.data == "neg" and len(node.children) == 1:
        return _is_zeroish_literal(node.children[0])
    return False


def _is_nullish_literal(node) -> bool:
    node = _unwrap(node)
    if isinstance(node, Token):
        if node.type == "STRING":
            try:
                value = json.loads(str(node))
            except Exception:
                value = str(node).strip('"')
            return value.strip().upper() in {"", "NULL"}
        return str(node).strip().upper() == "NULL"
    return False


def _render_token(token: Token) -> str:
    if token.type == "STRING":
        try:
            value = json.loads(str(token))
        except Exception:
            value = str(token).strip('"')
        return _render_literal_text(value)
    if token.type == "NUMBER":
        return str(token)
    return _render_name_text(str(token))


def _join_items(items: list[str], conjunction: str) -> str:
    items = [item for item in items if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"


def _join_condition_parts(items: list[str], conjunction: str, prefix: str | None = None) -> str:
    items = [item for item in items if item]
    if not items:
        return prefix or ""
    if len(items) == 1:
        body = items[0]
    elif len(items) == 2:
        body = f"{items[0]} {conjunction} {items[1]}"
    else:
        body = ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"
    return f"{prefix} {body}".strip() if prefix else body


def _flatten_chain(node: Tree, target_data: str) -> list:
    parts: list = []
    stack = [node]
    while stack:
        current = _unwrap(stack.pop())
        if isinstance(current, Tree) and current.data == target_data and len(current.children) == 2:
            stack.append(current.children[1])
            stack.append(current.children[0])
        else:
            parts.append(current)
    return parts


def _indent(depth: int) -> str:
    return "  " * depth


_DEFAULT_DATE_MARKERS = ("1900-01-01", "01/01/1900")
_DUMMY_ID_MARKERS = ("FORMO", "FORPM", "FORMF", "AAAAA1111A", "DUMMY", "INVALID")
_IDENTITY_COLUMN_SUFFIXES = (
    "ACCOUNTENTITYID",
    "CUSTOMERENTITYID",
    "CUSTOMERACID",
    "UCIFENTITYID",
    "PRODUCTALT_KEY",
    "SOURCEALT_KEY",
)


def sanitize_rule_explanation_text(text: str | None) -> str | None:
    """Plain-text cleanup for Excel Rule Explanation cells."""
    if not text:
        return text
    explanation = text.replace("**", "")
    while "''" in explanation:
        explanation = explanation.replace("''", "'")
    explanation = _rewrite_incorrect_from_assignments(explanation)
    explanation = sanitize_explanation_text(explanation)
    return _ensure_explanation_leading_caps(explanation)


def _ensure_explanation_leading_caps(text: str) -> str:
    """Capitalize the first character of each non-empty line (export prose polish)."""
    if not text:
        return text
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isalpha() and stripped[0].islower():
            stripped = stripped[0].upper() + stripped[1:]
        lines.append(stripped if stripped else "")
    return "\n".join(lines)


def sanitize_explanation_text(text: str) -> str:
    """Strip accidental Lark ``Tree``/``Token`` debug dumps from export text."""
    if not text or "tree(" not in text.lower():
        return text
    explanation = re.sub(
        r"tree\(token\([^)]*\)[^)]*\)",
        "calculated rule value",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation = re.sub(
        r"tree\([^)]*\)",
        "calculated rule value",
        explanation,
        flags=re.IGNORECASE,
    )
    explanation = re.sub(
        r"token\('[^']*',\s*'[^']*'\)",
        "",
        explanation,
        flags=re.IGNORECASE,
    )
    explanation = re.sub(r"\s{2,}", " ", explanation)
    explanation = re.sub(r"nested conditional:\s*", "", explanation, flags=re.IGNORECASE)
    explanation = explanation.replace(
        "set to calculated rule value", "applies a staged conditional assignment"
    )
    explanation = explanation.replace(
        "Set to calculated rule value", "Applies a staged conditional assignment"
    )
    explanation = explanation.replace(
        "uses a nested conditional result", "applies a staged conditional assignment"
    )
    explanation = explanation.replace(
        "Uses a nested conditional result", "Applies a staged conditional assignment"
    )
    return explanation.strip()


def _rewrite_incorrect_from_assignments(text: str) -> str:
    """Fix legacy phrasing: Updates COL from N -> Sets COL to 'N'."""
    explanation = re.sub(
        r"\bUpdates\s+([A-Za-z0-9_]+)\s+from\s+NULL\b",
        r"Resets \1 to NULL",
        text,
        flags=re.IGNORECASE,
    )
    explanation = re.sub(
        r"\bUpdates\s+([A-Za-z0-9_]+)\s+from\s+([A-Za-z0-9_]+)\s+when\b",
        lambda match: f"Sets {match.group(1)} to {_quote_literal_for_export(match.group(2))} when",
        explanation,
        flags=re.IGNORECASE,
    )
    explanation = re.sub(
        r"\bSets\s+([A-Za-z0-9_]+)\s+to\s+NULL\s+when\b",
        r"Resets \1 to NULL when",
        explanation,
        flags=re.IGNORECASE,
    )
    return explanation


def strip_markdown_bold(text: str | None) -> str | None:
    """Backward-compatible alias for :func:`sanitize_rule_explanation_text`."""
    return sanitize_rule_explanation_text(text)


def _quote_literal_for_export(raw: str) -> str:
    cleaned = raw.strip().strip("'").strip('"')
    if not cleaned:
        return "an empty value"
    return f"'{cleaned}'"


def _finish_dynamic_explanation(text: str | None) -> str | None:
    return sanitize_rule_explanation_text(text)


def try_entity_scoped_rule_explanation(
    expression: str,
    *,
    target_column: str,
    target_entity: str = "",
) -> str | None:
    """Entity+column narrators that must run before AST shape (e.g. staging ``MIN(...)``)."""
    formula = (expression or "").strip()
    if not formula:
        return None
    upper = formula.upper()
    return _narrate_s04_marking_flgdeg_degreason(
        target_column, formula, upper, entity=target_entity
    ) or _narrate_s12_marking_flg_pnpa(target_column, formula, upper, entity=target_entity)


def build_dynamic_rule_sentence(
    expression: str,
    *,
    target_column: str,
    target_entity: str = "",
) -> str | None:
    """Export summary built by walking the 4X AST (including nested IF arms)."""
    formula = (expression or "").strip()
    if not formula:
        return None
    style_tok = _EXPORT_LITERAL_STYLE.set(True)
    trunc_tok = _EXPORT_NO_TRUNCATE.set(True)
    col_tok = _EXPORT_TARGET_COLUMN.set((target_column or "").strip())
    try:
        return _build_dynamic_rule_sentence_impl(
            formula, target_column=target_column, target_entity=target_entity
        )
    finally:
        _EXPORT_TARGET_COLUMN.reset(col_tok)
        _EXPORT_NO_TRUNCATE.reset(trunc_tok)
        _EXPORT_LITERAL_STYLE.reset(style_tok)


def _build_dynamic_rule_sentence_impl(
    formula: str,
    *,
    target_column: str,
    target_entity: str = "",
) -> str | None:
    try:
        parsed = _parse_expression_tree(formula)
        if parsed is None:
            return None
        tree = _unwrap(parsed)
    except LarkError:
        return None

    col = (target_column or "the column").strip()
    entity = (target_entity or "").strip()
    upper = formula.upper()

    s04_entity = _narrate_s04_marking_flgdeg_degreason(col, formula, upper, entity=entity)
    if s04_entity:
        return _finish_dynamic_explanation(s04_entity)
    s12_entity = _narrate_s12_marking_flg_pnpa(col, formula, upper, entity=entity)
    if s12_entity:
        return _finish_dynamic_explanation(s12_entity)

    if isinstance(tree, Tree) and tree.data == "column_ref":
        path = _column_ref_physical_path(tree)
        if path:
            return _finish_dynamic_explanation(f"Direct pass-through mapping from {path}.")
        return None

    if_node = tree if isinstance(tree, Tree) and tree.data == "if_expr" else _locate_primary_if_expr(tree)
    if if_node is not None:
        isempty = _try_isempty_populate_sentence(if_node, col)
        if isempty:
            return _finish_dynamic_explanation(isempty)
        branches = _collect_if_branches(if_node)
        specialized = _try_specialized_column_narrative(
            col, formula, upper, branches, entity=entity
        )
        if specialized:
            return _finish_dynamic_explanation(specialized)
        single_branch = _try_single_branch_if_else_update(col, formula, upper, branches, entity=entity)
        if single_branch:
            return _finish_dynamic_explanation(single_branch)
        sentence = _dynamic_if_sentence(if_node, col, formula, branches, entity=entity)
        if sentence:
            return _finish_dynamic_explanation(
                _enrich_dynamic_sentence(sentence, col, entity, formula, if_node, branches)
            )

    if isinstance(tree, Tree):
        val = _render_value(tree)
        if val and val != "the calculated value":
            if re.fullmatch(r"@\w+", formula, flags=re.IGNORECASE):
                return None
            return _finish_dynamic_explanation(f"Assigns {col} to {val}.")
    return None


def format_explainer_lines_as_branches(target_column: str, lines: list[str]) -> str:
    col = (target_column or "the column").strip()
    if _lines_are_conditional_chain(lines):
        return _finish_dynamic_explanation(_conditional_chain_lines_to_summary(col, lines)) or ""
    return _finish_dynamic_explanation(" ".join(lines)) or ""


def explain_expression_for_export(expression: str, *, target_column: str = "") -> str | None:
    """Turn the full condition explainer walk into a branch-by-branch export summary."""
    explained = explain_expression(expression)
    if not explained:
        return None
    col = (target_column or "the column").strip()
    lines = _explainer_lines_for_export(explained)
    if not lines:
        return None
    if _lines_are_conditional_chain(lines):
        return _finish_dynamic_explanation(_conditional_chain_lines_to_summary(col, lines))
    return _finish_dynamic_explanation(" ".join(lines))


def _explainer_lines_for_export(explained: str) -> list[str]:
    out: list[str] = []
    for raw in explained.splitlines():
        text = raw.strip()
        if text.startswith("- "):
            text = text[2:].strip()
        if not text or text.lower() in {"otherwise:", "otherwise"}:
            continue
        if text.endswith(":") and (
            text.lower().startswith("if ")
            or text.lower().startswith("otherwise, if ")
        ):
            text = text[:-1].strip()
        out.append(text)
    return out


def _lines_are_conditional_chain(lines: list[str]) -> bool:
    lowered = [line.lower() for line in lines]
    return any(line.startswith("if ") or line.startswith("otherwise, if ") for line in lowered)


def _conditional_chain_lines_to_summary(col: str, lines: list[str]) -> str:
    steps: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        lower = line.lower()
        if lower.startswith("if ") or lower.startswith("otherwise, if "):
            prefix = "if " if lower.startswith("if ") else "otherwise, if "
            cond = line[len(prefix) :].strip().rstrip(".")
            outcome = ""
            if i + 1 < len(lines):
                outcome = _normalize_explainer_outcome(lines[i + 1])
                i += 2
            else:
                i += 1
            header = "Conditional Rule"
            if "moc" in cond.lower():
                header = "Manual Overrides"
            elif "buyout" in cond.lower() or "seller" in cond.lower():
                header = "Buyout Seller NPA"
            detail = outcome or cond
            if outcome and cond:
                detail = f"{outcome} when {cond}."
            steps.append((header, detail))
            continue
        if lower.startswith("otherwise") and i + 1 < len(lines):
            outcome = _normalize_explainer_outcome(lines[i + 1])
            steps.append(("Default Fallback", outcome + "."))
            i += 2
            continue
        i += 1
    if not steps:
        return " ".join(lines)
    return _format_numbered_ladder(col, "Evaluates", steps)


def _normalize_explainer_outcome(line: str) -> str:
    text = line.strip()
    lower = text.lower()
    if lower.startswith("return "):
        text = text[7:].strip().rstrip(".")
    elif lower.startswith("leave the field blank"):
        text = "clear to NULL"
    return text


def _locate_primary_if_expr(node) -> Tree | None:
    """Shallowest IF in the parse tree (e.g. IF inside COALESCE)."""
    node = _unwrap(node)
    queue: list = [node]
    while queue:
        current = queue.pop(0)
        current = _unwrap(current)
        if isinstance(current, Tree) and current.data == "if_expr":
            return current
        if isinstance(current, Tree):
            queue.extend(current.children)
    return None


def _column_ref_physical_path(node: Tree) -> str:
    parts: list[str] = []
    for child in node.children:
        raw = _column_ref_path_part_raw(child)
        if raw:
            parts.append(raw)
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]}.{parts[-1]}"


def _column_ref_path_part_raw(child) -> str:
    if isinstance(child, Tree) and child.data == "path_part" and child.children:
        part = child.children[0]
        if isinstance(part, Token):
            if part.type == "STRING":
                try:
                    return str(json.loads(str(part)))
                except Exception:
                    return str(part).strip('"')
            return str(part)
        return _render_identifier(part)
    if isinstance(child, Token):
        if child.type == "STRING":
            try:
                return str(json.loads(str(child)))
            except Exception:
                return str(child).strip('"')
        return str(child)
    if isinstance(child, Tree) and child.data == "column_ref":
        return _column_ref_physical_path(child)
    return _render_identifier(child)


_EFFECTIVE_DATE_MARKERS = (
    "EFFECTIVEFROMTIMEKEY",
    "EFFECTIVETOTIMEKEY",
    "@TIMEKEY",
    "@HISTTIMEKEY",
    "@PROCESSDATE",
    "@PROCESSINGDATE",
)

_S01_ADV_REF_PERIOD_COLUMNS = frozenset(
    {
        "REFPERIODOVERDUE",
        "REFPERIODOVERDRAWN",
        "REFPERIODNOCREDIT",
        "REFPERIODINTSERVICE",
    }
)

_S01_ZERO_DEFAULT_REF_PERIOD_COLUMNS = frozenset(
    {
        "REFPERIODSTKSTATEMENT",
        "REFPERIODREVIEW",
    }
)


def _is_single_if_else_branches(branches: list[tuple[Tree | None, Tree]]) -> bool:
    return len(branches) == 2 and branches[-1][0] is None


def _formula_has_effective_or_join_criteria(upper: str) -> bool:
    return any(marker in upper for marker in _EFFECTIVE_DATE_MARKERS)


_JOIN_KEY_NAMES = (
    "ACCOUNTENTITYID",
    "CUSTOMERENTITYID",
    "CUSTOMERACID",
    "UCIFENTITYID",
    "PRODUCTALT_KEY",
    "SOURCEALT_KEY",
    "ACCOUNTENTITYID",
)


def _export_string_from_token(part: Token) -> str:
    try:
        return str(json.loads(str(part)))
    except Exception:
        return str(part).strip('"')


def _column_ref_is_string_literal(node: Tree) -> bool:
    """True only for a single quoted literal (``"Y"``), not ``"Ent"."Col"``."""
    parts = 0
    for child in node.children:
        if isinstance(child, Tree) and child.data == "path_part" and child.children:
            parts += 1
            part = child.children[0]
            if not (isinstance(part, Token) and part.type == "STRING"):
                return False
    return parts == 1


def _column_ref_literal_value(node: Tree) -> str | None:
    for child in node.children:
        if isinstance(child, Tree) and child.data == "path_part" and child.children:
            part = child.children[0]
            if isinstance(part, Token) and part.type == "STRING":
                return _export_string_from_token(part)
    return None


def _export_truncation_allowed() -> bool:
    return not _EXPORT_NO_TRUNCATE.get()


def _maybe_truncate_export_text(text: str, max_len: int) -> str:
    if not text or not _export_truncation_allowed():
        return text
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _literal_text_from_node(node) -> str | None:
    node = _unwrap(node)
    if isinstance(node, Token) and node.type == "STRING":
        return _export_string_from_token(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        if _column_ref_is_string_literal(node):
            return _column_ref_literal_value(node)
    return None


def _export_target_column_name() -> str:
    return (_EXPORT_TARGET_COLUMN.get() or "the column").strip()


def _describe_concat_append_outcome(node, col: str) -> str | None:
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "function_call":
        return None
    if str(node.children[0]).upper() != "CONCAT":
        return None
    reason: str | None = None
    for arg in reversed(_function_args(node)):
        lit = _literal_text_from_node(arg)
        if not lit:
            continue
        stripped = lit.strip().strip(",").strip()
        if stripped and stripped not in {",", ";"}:
            reason = stripped
            break
    target = (col or _export_target_column_name()).strip()
    if reason:
        return f"Appends {_quote_literal_for_export(reason)} to {target}"
    return f"Appends text to {target}"


def _export_function_call_human(node: Tree) -> str | None:
    if not _EXPORT_LITERAL_STYLE.get():
        return None
    name = str(node.children[0]).upper()
    args = _function_args(node)
    col = _export_target_column_name()
    if name == "CONCAT":
        return _describe_concat_append_outcome(node, col)
    if name in {"COALESCE", "NVL", "ISNULL"} and args:
        primary = _export_operand(args[0])
        if len(args) == 1:
            return primary
        fallback_lit = _literal_text_from_node(args[1])
        if fallback_lit is not None:
            return f"{primary} (defaulting to {_quote_literal_for_export(fallback_lit)} when null)"
        fallback = _export_operand(args[1])
        return f"{primary} (defaulting to {fallback} when null)"
    if name == "DATEDIFF":
        return "a day-difference calculation"
    if name == "ISEMPTY":
        return "a blank-or-missing check"
    if name == "ISNOTEMPTY":
        return "a populated-value check"
    if name == "SUM" and args:
        return f"the sum of {_export_operand(args[0])}"
    return None


def _condition_classification_blob(cond, formula: str) -> str:
    try:
        cond_text = _render_export_condition(_unwrap(cond))
    except Exception:
        cond_text = _render_condition(cond)
    return f"{cond_text} {formula or ''}".upper()


def _classify_branch_header_from_condition(cond, formula: str) -> str | None:
    blob = _condition_classification_blob(cond, formula)
    compact = blob.replace("_", "").replace(" ", "")
    if "BALANCE" in compact and (
        "<=0" in compact
        or "<= 0" in blob
        or "LESS THAN OR EQUAL TO 0" in blob
        or "LESS THAN OR EQUAL TO ZERO" in blob
    ):
        return "Zero Balance Reset"
    if (
        "ADVACRESTRUCTURE" in compact
        or "RESTRUCTUREDT" in compact
        or "RESTRUCTURECAL" in compact
    ):
        return "Restructuring Evaluation"
    if "DPD_" in blob or "REFPERIOD" in compact:
        return "Servicing DPD Breaches"
    if "BUYOUT" in compact or "SELLER" in compact or "VIRTUALDPD" in compact:
        return "Loan Buyout Evaluation"
    return None


def _unique_step_titles(titles: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for title in titles:
        key = title.strip()
        count = seen.get(key, 0)
        seen[key] = count + 1
        out.append(key if count == 0 else f"{key} ({count + 1})")
    return out


def _export_operand(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Token):
        if node.type == "STRING":
            return _quote_literal_for_export(_export_string_from_token(node))
        if node.type == "NUMBER":
            return str(node)
        return str(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        if _column_ref_is_string_literal(node):
            literal = _column_ref_literal_value(node)
            if literal is not None:
                return _quote_literal_for_export(literal)
        path = _column_ref_physical_path(node)
        return path or "field"
    if isinstance(node, Tree) and node.data == "function_call":
        human = _export_function_call_human(node)
        if human:
            return human
        name = str(node.children[0]).upper()
        args = [_export_operand(arg) for arg in _function_args(node)]
        if not args:
            return f"{name}()"
        return f"{name}({', '.join(args)})"
    if isinstance(node, Tree) and node.data == "add":
        return f"{_export_operand(node.children[0])} + {_export_operand(node.children[1])}"
    if isinstance(node, Tree) and node.data == "sub":
        return f"{_export_operand(node.children[0])} - {_export_operand(node.children[1])}"
    if isinstance(node, Tree) and node.data == "mul":
        return f"{_export_operand(node.children[0])} * {_export_operand(node.children[1])}"
    if isinstance(node, Tree) and node.data == "div":
        return f"{_export_operand(node.children[0])} / {_export_operand(node.children[1])}"
    if isinstance(node, Tree) and node.data == "neg":
        return f"-{_export_operand(node.children[0])}"
    return _export_tree_node(node)


def _export_branch_outcome_label(node) -> str:
    node = _unwrap(node)
    scalar = _export_scalar_outcome(node)
    if scalar:
        return scalar
    if isinstance(node, Tree) and node.data == "column_ref":
        path = _column_ref_physical_path(node)
        if path:
            return path
    if isinstance(node, Tree) and node.data == "if_expr":
        return _summarize_nested_if_outcome(node)
    return _export_tree_node(node)


def _summarize_nested_if_outcome(if_node: Tree) -> str:
    if _EXPORT_LITERAL_STYLE.get():
        inner = _describe_dpd_threshold_else_zero_if(if_node, "")
        if inner:
            return inner
    branches = _collect_if_branches(if_node)
    if not branches:
        return "conditional result"
    col = _export_target_column_name()
    if _EXPORT_LITERAL_STYLE.get():
        if len(branches) == 2 and branches[-1][0] is None:
            cond, then_b = branches[0]
            if cond is not None:
                append = _describe_concat_append_outcome(then_b, col)
                cue = _export_condition_phrase(cond, "")
                if append:
                    return f"{append} when {cue}"
                action = _branch_outcome_label(then_b, col)
                return f"{action} when {cue}"
        return f"a staged conditional assignment for {col}"
    first_cond, first_out = branches[0]
    if first_cond is None:
        return _export_branch_outcome_label(first_out)
    return "a nested conditional result"


def _export_tree_node(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Token):
        return _export_operand(node)
    if not isinstance(node, Tree):
        return "calculated rule value"
    if node.data == "if_expr":
        return _summarize_nested_if_outcome(node)
    if node.data in {
        "compare",
        "and_op",
        "or_op",
        "and_call",
        "or_call",
        "not_op",
        "membership",
        "between_op",
    }:
        return _render_export_condition(node)
    if node.data in {"column_ref", "function_call", "add", "sub", "mul", "div", "neg"}:
        return _export_operand(node)
    if node.data == "value_list":
        return ", ".join(_export_operand(child) for child in node.children)
    if node.data == "list_literal" and node.children:
        return _export_operand(node.children[0])
    return "calculated rule value"


def _render_export_membership(node: Tree) -> str:
    children = list(node.children)
    lhs = _export_operand(children[0])
    op = _membership_operator(children)
    values = _membership_literal_values(node)
    if op == "IN" and values:
        if len(values) == 1:
            joined = values[0]
        elif len(values) == 2:
            joined = f"{values[0]} or {values[1]}"
        else:
            joined = ", ".join(values[:-1]) + f", or {values[-1]}"
        return f"{lhs} is {joined}"
    phrase = _MEMBERSHIP_WORDS.get(op or "IN", "relates to")
    rhs = ", ".join(values) if values else "the listed values"
    return f"{lhs} {phrase} {rhs}"


def _render_export_condition(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Tree):
        if node.data == "compare" and len(node.children) == 3:
            left, op, right = node.children
            op_s = str(op)
            proc_phrase = _running_process_name_compare_phrase(left, op_s, right)
            if proc_phrase:
                return proc_phrase
            join_phrase = _join_key_equality_compare_phrase(left, op_s, right)
            if join_phrase:
                return join_phrase
            return _render_compare_condition(left, op_s, right)
        if node.data == "and_call":
            return ", and ".join(_render_export_condition(child) for child in node.children)
        if node.data == "and_op":
            parts = [_render_export_condition(part) for part in _flatten_chain(node, "and_op")]
            return ", and ".join(parts)
        if node.data == "or_call":
            return ", or ".join(_render_export_condition(child) for child in node.children)
        if node.data == "or_op":
            parts = [_render_export_condition(part) for part in _flatten_chain(node, "or_op")]
            return ", or ".join(parts)
        if node.data == "not_op":
            return f"it is not true that {_render_export_condition(node.children[0])}"
        if node.data == "membership":
            return _render_export_membership(node)
        if node.data == "between_op" and len(node.children) == 3:
            value, low, high = node.children
            return (
                f"{_export_operand(value)} is between {_export_operand(low)} and {_export_operand(high)}"
            )
        if node.data == "function_call":
            name = str(node.children[0]).upper()
            args = _function_args(node)
            if name == "ISEMPTY" and args:
                return f"{_export_operand(args[0])} is blank or missing"
            if name == "ISNOTEMPTY" and args:
                return f"{_export_operand(args[0])} has a value"
            return _export_operand(node)
    return _export_operand(node)


def _export_scalar_outcome(node) -> str | None:
    node = _unwrap(node)
    if _outcome_is_null(node):
        return "NULL"
    if isinstance(node, Token):
        if node.type == "STRING":
            return _quote_literal_for_export(_export_string_from_token(node))
        if node.type == "NUMBER":
            return str(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        if _column_ref_is_string_literal(node):
            literal = _column_ref_literal_value(node)
            if literal is not None:
                return _quote_literal_for_export(literal)
        path = _column_ref_physical_path(node)
        if path and "." not in path:
            return _quote_literal_for_export(path)
        return None
    return None


def _export_outcome_value(node) -> str:
    scalar = _export_scalar_outcome(node)
    if scalar:
        return scalar
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "column_ref":
        return _column_ref_physical_path(node) or "value"
    if isinstance(node, Token) and node.type == "NUMBER":
        return str(node)
    if isinstance(node, Tree) and node.data == "function_call":
        append = _describe_concat_append_outcome(node, _export_target_column_name())
        if append:
            return append
    if isinstance(node, Tree) and node.data == "if_expr":
        return _summarize_nested_if_outcome(node)
    return _export_tree_node(node)


def _is_increment_by_one(node) -> bool:
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "add" or len(node.children) != 2:
        return False
    right = _unwrap(node.children[1])
    return isinstance(right, Token) and str(right) == "1"


def _format_numbered_ladder(column: str, lead_in: str, steps: list[tuple[str, str]]) -> str:
    col = (column or "the column").strip()
    lines = [f"{lead_in} {col} across prioritized conditions:"]
    for index, (title, detail) in enumerate(steps, start=1):
        lines.append(f"{index}. {title}: {detail}")
    return "\n".join(lines)


def _format_fixed_title_ladder(title_line: str, steps: list[tuple[str, str]]) -> str:
    lines = [title_line.rstrip()]
    for index, (header, detail) in enumerate(steps, start=1):
        lines.append(f"{index}. {header}: {detail}")
    return "\n".join(lines)


def _qualified_ref_from_node(node) -> tuple[str, str] | None:
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "column_ref":
        return None
    parts = [_column_ref_path_part_raw(child) for child in node.children]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return parts[0], parts[-1]


def _walk_ast_nodes(node, *, data: str | None = None) -> list:
    found: list = []
    stack = [_unwrap(node)]
    while stack:
        current = stack.pop()
        if not isinstance(current, Tree):
            continue
        if data is None or current.data == data:
            found.append(current)
        stack.extend(current.children)
    return found


def _extract_join_key_from_condition(cond, source_table: str | None, target_column: str) -> str | None:
    source_u = (source_table or "").upper()
    for compare in _walk_ast_nodes(cond, data="compare"):
        if len(compare.children) != 3:
            continue
        if str(compare.children[1]).strip() != "==":
            continue
        left = _qualified_ref_from_node(compare.children[0])
        right = _qualified_ref_from_node(compare.children[2])
        if not left or not right:
            continue
        if left[0].upper() == right[0].upper():
            continue
        if source_u and source_u not in {left[0].upper(), right[0].upper()}:
            continue
        if left[1].upper() == right[1].upper():
            return left[1]
    for compare in _walk_ast_nodes(cond, data="compare"):
        if len(compare.children) != 3:
            continue
        if str(compare.children[1]).strip() != "==":
            continue
        left = _qualified_ref_from_node(compare.children[0])
        right = _qualified_ref_from_node(compare.children[2])
        if left and right and left[1].upper() == right[1].upper():
            return left[1]
    cond_upper = _render_condition(cond).upper()
    for key in _JOIN_KEY_NAMES:
        if key in cond_upper:
            return key if key.endswith("_KEY") else key.title()
    return None


def _dominant_entity_in_condition_ast(cond) -> str | None:
    counts: dict[str, int] = {}
    for ref in _walk_ast_nodes(cond, data="column_ref"):
        pair = _qualified_ref_from_node(ref)
        if not pair:
            continue
        counts[pair[0]] = counts.get(pair[0], 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]


def _source_table_from_then(then_node) -> str | None:
    then_node = _unwrap(then_node)
    if isinstance(then_node, Tree) and then_node.data == "column_ref":
        pair = _qualified_ref_from_node(then_node)
        if pair:
            return pair[0]
    return None


def _source_column_from_then(then_node) -> str | None:
    then_node = _unwrap(then_node)
    if isinstance(then_node, Tree) and then_node.data == "column_ref":
        pair = _qualified_ref_from_node(then_node)
        if pair:
            return pair[1]
    return None


def _single_branch_when_phrase(cond: Tree, then_node, col: str, formula: str, upper: str) -> str:
    source = _source_table_from_then(then_node) or _dominant_entity_in_condition_ast(cond)
    join_key = _extract_join_key_from_condition(cond, source, col)
    if source and join_key:
        return f"{source} is active and matched on {join_key}"
    if source and any(marker in upper for marker in _EFFECTIVE_DATE_MARKERS):
        return f"{source} is active for the current processing timekey"
    return _export_condition_phrase(cond, formula, max_len=360)


def _literal_or_value_label(node) -> str:
    return _export_outcome_value(node)


def _else_retain_phrase(else_node, col: str, entity: str = "") -> str:
    if _outcome_is_qualified_self_column_ref(else_node, col, entity) or _outcome_is_entity_table_passthrough(
        else_node, col, entity
    ):
        return f"otherwise retains existing {col}"
    return _outcome_action_phrase(else_node, col, role="else", entity=entity)


def _try_single_branch_if_else_update(
    col: str,
    formula: str,
    upper: str,
    branches: list[tuple[Tree | None, Tree]],
    *,
    entity: str = "",
) -> str | None:
    """One IF / THEN / ELSE arm (common effective-date and join lookups)."""
    if not _is_single_if_else_branches(branches):
        return None
    col_u = (col or "").upper()
    if "SPLCATG" in col_u and "ALT_KEY" in col_u:
        return None
    if col_u == "INMONTHMARK" and "PROC_FREQ" in upper and ("EOM" in upper or "EOMONTH" in upper):
        return None
    if not _formula_has_effective_or_join_criteria(upper):
        return None
    if col_u in {"REFCUSTOMERID", "BALANCE"} and "CTECUSTOMERWISEBALANCE" in upper.replace(
        "_", ""
    ).replace("#", "").upper():
        return None
    if _formula_is_s05_maxdpd_staging(upper) and (
        col_u.startswith("DPD_") or col_u in {"CUSTOMERACID", "REFPERIOD"}
    ):
        return None
    if _formula_is_s06_temptabledpd(upper, entity) and col_u in _S05_DPD_REF_PERIOD_PAIRS:
        return None
    if col_u in _S01_ADV_REF_PERIOD_COLUMNS | _S01_ZERO_DEFAULT_REF_PERIOD_COLUMNS:
        return None
    if _formula_is_s08_npa_erosion_aging(upper, entity) and col_u in _S08_NARRATED_COLUMNS:
        return None
    if _formula_is_s12_marking_flg_pnpa(upper, entity) and col_u in _S12_NARRATED_COLUMNS:
        return None
    if _formula_is_s04_marking_flgdeg_degreason(upper, entity) and col_u in _S04_NARRATED_COLUMNS:
        return None

    cond, then_node = branches[0]
    if col_u == "INMONTHMARK" and isinstance(_unwrap(then_node), Tree) and _unwrap(then_node).data == "if_expr":
        return None
    else_node = branches[1][1]
    if cond is None:
        return None

    when_phrase = _single_branch_when_phrase(cond, then_node, col, formula, upper)
    else_phrase = _else_retain_phrase(else_node, col, entity=entity)
    then_node = _unwrap(then_node)

    assignment = _format_then_assignment_sentence(col, then_node, when_phrase, else_phrase)
    if assignment:
        return assignment

    then_label = _export_outcome_value(then_node)
    return f"Sets {col} to {then_label} when {when_phrase}; {else_phrase}."


def _format_then_assignment_sentence(
    col: str,
    then_node,
    when_phrase: str,
    else_phrase: str,
) -> str | None:
    """Preferred phrasing for IF(cond) THEN(val) ELSE(fallback) assignments."""
    then_node = _unwrap(then_node)
    if _outcome_is_null(then_node):
        return f"Resets {col} to NULL when {when_phrase}; {else_phrase}."

    then_scalar = _export_scalar_outcome(then_node)
    if then_scalar:
        return f"Sets {col} to {then_scalar} when {when_phrase}; {else_phrase}."

    if isinstance(then_node, Tree) and then_node.data == "column_ref":
        path = _column_ref_physical_path(then_node)
        if path and "." in path and not _outcome_is_self_column(then_node, col):
            return f"Updates {col} from {path} when {when_phrase}; {else_phrase}."
        if _outcome_is_self_column(then_node, col):
            return f"Sets {col} when {when_phrase}; {else_phrase}."
    return None


def _collect_if_branches(if_node: Tree) -> list[tuple[Tree | None, Tree]]:
    """Ordered (condition, outcome) pairs; condition is None for the final ELSE."""
    children = list(if_node.children)
    if len(children) < 2:
        return []
    branches: list[tuple[Tree | None, Tree]] = [(children[0], children[1])]
    for child in children[2:]:
        if isinstance(child, Tree) and child.data == "elseif_clause":
            branches.append((child.children[0], child.children[1]))
        elif isinstance(child, Tree) and child.data == "else_clause":
            branches.append((None, child.children[0]))
    return branches


def _dynamic_if_sentence(
    if_node: Tree,
    col: str,
    formula: str,
    branches: list[tuple[Tree | None, Tree]],
    *,
    entity: str = "",
) -> str | None:
    if not branches:
        return None

    has_explicit_else = len(branches) >= 2 and branches[-1][0] is None
    primary_cond, primary_then = branches[0]
    else_node = branches[-1][1] if has_explicit_else else None

    if primary_cond is not None and _outcome_is_null(primary_then):
        else_phrase = _outcome_action_phrase(else_node, col, role="else", entity=entity)
        cond_text = _export_condition_phrase(primary_cond, formula)
        return f"Resets {col} to NULL when {cond_text}; otherwise {else_phrase}."

    conditional_count = sum(1 for cond, _ in branches if cond is not None)
    if conditional_count >= 2:
        return _render_multi_branch_explanation(col, formula, branches)

    cond_text = _export_condition_phrase(primary_cond, formula)
    if has_explicit_else and _is_increment_by_one(primary_then):
        if else_node is not None and (
            _outcome_is_qualified_self_column_ref(else_node, col, entity)
            or _outcome_is_entity_table_passthrough(else_node, col, entity)
        ):
            return f"Increments {col} by 1 when {cond_text}; otherwise retains existing {col}."
        else_val = _export_outcome_value(else_node) if else_node is not None else "0"
        return f"Increments {col} by 1 when {cond_text}; otherwise sets to {else_val}."

    if has_explicit_else:
        then_scalar = _export_scalar_outcome(primary_then)
        else_scalar = _export_scalar_outcome(else_node) if else_node is not None else None
        else_retain = else_node is not None and (
            _outcome_is_qualified_self_column_ref(else_node, col, entity)
            or _outcome_is_entity_table_passthrough(else_node, col, entity)
        )
        if then_scalar and else_retain:
            return (
                f"Sets {col} to {then_scalar} when {cond_text}; "
                f"otherwise retains existing {col}."
            )
        if then_scalar and else_scalar:
            return f"Sets {col} to {then_scalar} when {cond_text}; otherwise sets to {else_scalar}."

    then_phrase = _outcome_action_phrase(primary_then, col, role="then", entity=entity)
    else_phrase = (
        _outcome_action_phrase(else_node, col, role="else", entity=entity)
        if else_node is not None
        else f"retains existing {col}"
    )
    return f"{then_phrase} when {cond_text}; otherwise {else_phrase}."


def _outcome_is_null(node) -> bool:
    node = _unwrap(node)
    if isinstance(node, Token) and str(node).upper() == "NULL":
        return True
    val, _ = _render_value_with_note(node)
    return val == "no value"


def _infer_branch_header(cond, outcome, formula: str) -> str:
    if cond is None:
        return "Default Fallback"
    classified = _classify_branch_header_from_condition(cond, formula)
    if classified:
        return classified
    blob = _condition_classification_blob(cond, formula)
    compact = blob.replace("_", "").replace(" ", "")
    if _formula_is_s08_npa_erosion_aging(blob):
        if "MOVETOLOSS" in compact or "CURNTQTRRV" in compact:
            return "Security Erosion"
        if "ADDDAY(" in blob or "PERIOD(" in blob:
            return "NPA Aging Transitions"
    if "MOC" in blob and "FLGMOC" not in compact:
        return "Manual Overrides"
    if any(token in blob for token in ("SETTLEMENT", "LITIGATION", "CHARGE", "WRITEOFF", "WRITE-OFF", "TWO")):
        return "Status Exceptions"
    if any(token in blob for token in ("PAN", "AADHAAR", "AADHAR", "COBORROW")):
        return "Card & ID Percolation"
    if "ALWYS_STD" in blob or "ADHOC" in blob:
        return "Standard Reset"
    if "FDSEC" in blob or "CONDI_STD" in blob:
        return "FD Secured / Conditional STD"
    if any(token in blob for token in ("SARFAESI", "WEAK", "REPOSSESSION", "FRAUD")):
        return "Degradation Overrides"
    if "FLGDEG" in compact or "DEGRADE" in compact:
        return "Degradation Overrides"
    if "PUI" in compact and ("DCCO" in compact or "DEFAULT_REASON" in compact):
        return "Priority Classification"
    source = _dominant_entity_in_condition_ast(cond)
    if source:
        return f"{source} Match"
    return "Conditional Rule"


def _branch_step_detail(cond, outcome, col: str, formula: str, *, is_fallback: bool = False) -> str:
    if is_fallback:
        if _outcome_is_self_column(outcome, col):
            return f"Retains existing {col}."
        if isinstance(_unwrap(outcome), Tree) and _unwrap(outcome).data == "function_call":
            if _is_coalesce_function(_unwrap(outcome)):
                return f"Populates {col} from historical or source NPA records."
        label = _branch_outcome_label(outcome, col)
        return f"{label.capitalize()}."
    append = _describe_concat_append_outcome(outcome, col)
    if append and cond is not None:
        cue = _export_condition_phrase(cond, formula)
        return f"{append} when {cue}."
    if _outcome_is_null(outcome):
        return f"Resets {col} to NULL for matching accounts."
    path = ""
    node = _unwrap(outcome)
    if isinstance(node, Tree) and node.data == "column_ref":
        path = _column_ref_physical_path(node)
    if path and "." in path and not _outcome_is_self_column(outcome, col):
        return f"Uses {path} when criteria match."
    label = _branch_outcome_label(outcome, col)
    cue = _export_condition_phrase(cond, formula) if cond is not None else ""
    if cue:
        return f"{label.capitalize()} when {cue}."
    return f"{label.capitalize()}."


def _render_multi_branch_explanation(
    col: str,
    formula: str,
    branches: list[tuple[Tree | None, Tree]],
) -> str:
    steps: list[tuple[str, str]] = []
    for cond, outcome in branches:
        if cond is None:
            steps.append(
                (
                    "Default Fallback",
                    _branch_step_detail(cond, outcome, col, formula, is_fallback=True),
                )
            )
            continue
        header = _infer_branch_header(cond, outcome, formula)
        detail = _branch_step_detail(cond, outcome, col, formula)
        steps.append((header, detail))
    headers = _unique_step_titles([title for title, _ in steps])
    steps = [(headers[i], detail) for i, (_, detail) in enumerate(steps)]
    lead = "Assigns" if (col or "").upper() == "ASSET_NORM" else "Evaluates"
    return _format_numbered_ladder(col, lead, steps)


def _branch_outcome_label(node, col: str) -> str:
    node = _unwrap(node)
    append = _describe_concat_append_outcome(node, col)
    if append:
        return append
    if _outcome_is_self_column(node, col):
        return f"retain existing {col}"
    if _outcome_is_null(node):
        return f"clear {col} to NULL"
    if re.fullmatch(r"@\w+", str(_raw_value_text(node)), flags=re.IGNORECASE):
        return "use the active processing date"
    scalar = _export_scalar_outcome(node)
    if scalar:
        return f"set to {scalar}"
    path = ""
    if isinstance(node, Tree) and node.data == "column_ref":
        path = _column_ref_physical_path(node)
    if path and "." in path:
        return f"use {path}"
    return f"set to {_export_outcome_value(node)}"


def _try_specialized_column_narrative(
    col: str,
    formula: str,
    upper: str,
    branches: list[tuple[Tree | None, Tree]],
    *,
    entity: str = "",
) -> str | None:
    col_u = (col or "").upper()
    compact = upper.replace(" ", "")

    s01_adv = _narrate_s01_adv_basic_detail_ref_period(col, formula, upper)
    if s01_adv:
        return s01_adv
    s01_zero = _narrate_s01_account_zero_default_ref_period(col, formula, upper)
    if s01_zero:
        return s01_zero

    s02_dpd = _narrate_s02_account_dpd_timekey(col, formula, upper)
    if s02_dpd:
        return s02_dpd
    s02_aqua = _narrate_s02_aqua_scheme_reset(col, formula, upper)
    if s02_aqua:
        return s02_aqua

    s06_temptable_dpd = _narrate_s06_temptabledpd_dpd(col, formula, upper, entity=entity)
    if s06_temptable_dpd:
        return s06_temptable_dpd

    s08_erosion = _narrate_s08_npa_erosion_aging(col, formula, upper, entity=entity)
    if s08_erosion:
        return s08_erosion

    s04_flgdeg = _narrate_s04_marking_flgdeg_degreason(col, formula, upper, entity=entity)
    if s04_flgdeg:
        return s04_flgdeg

    s12_pnpa = _narrate_s12_marking_flg_pnpa(col, formula, upper, entity=entity)
    if s12_pnpa:
        return s12_pnpa

    staging_dpd = _narrate_s05_temptable_dpd_threshold(col, formula, upper)
    if staging_dpd:
        return staging_dpd
    if col_u == "CUSTOMERACID":
        ac1 = _narrate_s05_temptable1_customer_ac_id(col, formula, upper, entity=entity)
        if ac1:
            return ac1
        ac2 = _narrate_s05_temptable2_customer_ac_id(col, formula, upper, entity=entity)
        if ac2:
            return ac2
    if col_u == "REFPERIOD":
        ref2 = _narrate_s05_temptable2_refperiod(col, formula, upper, entity=entity)
        if ref2:
            return ref2

    if (col_u == "PANNO" or col_u.endswith("PANNO")) and len(branches) <= 1:
        return _narrate_panno_validation(col, formula, upper, branches)

    if col_u in {"SYSNPA_DT", "FINALNPADT", "NPADT", "INITIALNPADT"} or (
        "NPADT" in col_u
        and len(branches) >= 3
        and not col_u.startswith("PNPA_")
    ):
        npa = _narrate_npa_date_priority_ladder(col, upper, compact, branches)
        if npa:
            return npa

    s13_npa_reason = _narrate_s13_npa_reason_account(col, formula, upper)
    if s13_npa_reason:
        return s13_npa_reason

    if col_u == "ASSET_NORM":
        s06_asset = _narrate_s06_asset_norm_from_pui(col, formula, upper, entity=entity)
        if s06_asset:
            return s06_asset
        asset = _narrate_asset_norm_ladder(col, upper, branches)
        if asset:
            return asset

    if col_u == "DEGREASON" and len(branches) >= 2:
        if not _formula_is_s04_marking_flgdeg_degreason(upper, entity):
            return _render_multi_branch_explanation(col, formula, branches)

    if col_u == "INMONTHMARK":
        inmonth = _narrate_inmonthmark_processing(col, formula, upper, branches)
        if inmonth:
            return inmonth

    if col_u == "SYSASSETCLASSALT_KEY":
        if not _formula_is_s08_npa_erosion_aging(upper, entity):
            asset_key = _narrate_sys_asset_class_alt_key_ladder(col, formula, upper)
            if asset_key:
                return asset_key

    if col_u == "FINALASSETCLASSALT_KEY":
        final_key = _narrate_final_asset_class_from_customer(col, formula, upper)
        if final_key:
            return final_key

    if col_u in {"DBTDT", "DBDT"}:
        dbt = _narrate_customer_dbt_dt_s07(col, formula, upper)
        if dbt:
            return dbt

    if col_u == "DEGDATE":
        s06_deg = _narrate_s06_adv_ac_restructure_deg_date(col, formula, upper, entity=entity)
        if s06_deg:
            return s06_deg
        deg = _narrate_deg_date_process_date(col, formula, upper)
        if deg:
            return deg

    if col_u == "LOSSDT":
        loss = _narrate_loss_dt_s07(col, formula, upper)
        if loss:
            return loss

    if col_u in {"REFCUSTOMERID", "BALANCE"} and "CTECUSTOMERWISEBALANCE" in upper.replace(
        "_", ""
    ).replace("#", "").upper():
        cte = _narrate_cte_customer_wise_balance(col, formula, upper)
        if cte:
            return cte

    if col_u == "DPD_MAX":
        dpd_max = _narrate_dpd_max_s05(col, formula, upper)
        if dpd_max:
            return dpd_max

    if col_u == "DPD_FINMAXTYPE":
        fin_type = _narrate_dpd_fin_max_type_s05(col, formula, upper)
        if fin_type:
            return fin_type

    if col_u in {"REFPERIODMAX", "REFPERIOD_MAX"}:
        ref_max = _narrate_refperiod_max_s05(col, formula, upper)
        if ref_max:
            return ref_max

    return None


def _formula_has_npa_aging_bucket_hops(formula: str, upper: str) -> bool:
    if "ADDDAY(" not in upper or "SYSNPA" not in upper:
        return False
    return any(f'."{bucket}".' in formula or f'"{bucket}"' in upper for bucket in ("SUB", "DB1", "DB2", "DB3"))


def _narrate_sys_asset_class_alt_key_ladder(col: str, formula: str, upper: str) -> str | None:
    """PRO Update_AssetClass: fraud LOS + SUB/DB1/DB2/DB3 aging on SysNPA_Dt."""
    if not _formula_has_npa_aging_bucket_hops(formula, upper):
        return None
    steps: list[tuple[str, str]] = []
    if "870" in formula or any(f"SPLCATG{n}" in upper for n in range(1, 5)):
        steps.append(
            (
                "Fraud / special category (870)",
                "Sets SysAssetClassAlt_Key to the LOS (loss) DimAssetClass key when any "
                "SplCatg1–4 Alt_Key is 870 and FlgDeg is 'Y'.",
            )
        )
    steps.append(
        (
            "NPA aging buckets",
            "Assigns SUB, DB1, DB2, or DB3 asset-class keys from SysNPA_Dt plus "
            "@SUB_Days, @DB1_Days, and @DB2_Days compared to @ProcessDate.",
        )
    )
    return _format_numbered_ladder(col, "Assigns", steps)


def _narrate_final_asset_class_from_customer(col: str, formula: str, upper: str) -> str | None:
    if "SYSASSETCLASSALT_KEY" not in upper:
        return None
    if "NORMAL" not in upper or "FLGDEG" not in upper.replace("_", ""):
        return None
    if "FINALASSETCLASSALT_KEY" not in upper and col.upper() != "FINALASSETCLASSALT_KEY":
        return None
    return (
        "Sets FinalAssetClassAlt_Key from CUSTOMERCAL.SysAssetClassAlt_Key when the account "
        "Asset_Norm is NORMAL, account FlgDeg is 'Y', and the customer is in the degraded "
        "processing cohort (FlgDeg 'Y', FlgProcessing 'N'); otherwise retains existing "
        f"{col}."
    )


def _narrate_customer_dbt_dt_s07(col: str, formula: str, upper: str) -> str | None:
    if not _formula_has_npa_aging_bucket_hops(formula, upper) and "THEN(NULL)" not in upper.replace(" ", ""):
        return None
    if "FLGDEG" not in upper.replace("_", ""):
        return None
    return (
        f"Assigns {col} across prioritized conditions:\n"
        "1. Cohort reset: Clears DbtDt to NULL for degraded customers in the customer-wise balance cohort.\n"
        "2. Bucket dates: Populates DbtDt with ADDDAY-based bucket start dates from SysNPA_Dt when "
        "the customer enters DB1/DB2 stages.\n"
        "3. Loss accounts: Resets DbtDt to NULL when SysAssetClassAlt_Key matches LOS."
    )


_S05_DPD_COMPONENTS = (
    "DPD_INTSERVICE",
    "DPD_NOCREDIT",
    "DPD_OVERDRAWN",
    "DPD_OVERDUE",
    "DPD_RENEWAL",
    "DPD_STOCKSTMT",
)

def _narrate_s01_adv_basic_detail_ref_period(col: str, formula: str, upper: str) -> str | None:
    """S01: join ADVACBASICDETAIL.ReferencePeriod with timekey window, default 91."""
    col_u = (col or "").upper()
    if col_u not in _S01_ADV_REF_PERIOD_COLUMNS:
        return None
    if "ADVACBASICDETAIL" not in upper:
        return None
    if "REFERENCEPERIOD" not in upper.replace("_", ""):
        return None
    if not any(marker in upper for marker in _EFFECTIVE_DATE_MARKERS):
        return None
    return (
        f"Populates {col} with ADVACBASICDETAIL.ReferencePeriod when ADVACBASICDETAIL is active "
        f"for the current timekey and ReferencePeriod is non-zero; otherwise defaults to 91."
    )


def _narrate_s01_account_zero_default_ref_period(col: str, formula: str, upper: str) -> str | None:
    """S01: RefPeriodStkStatement / RefPeriodReview default 181 when column is 0."""
    col_u = (col or "").upper()
    if col_u not in _S01_ZERO_DEFAULT_REF_PERIOD_COLUMNS:
        return None
    account_col = _s01_account_column_display_name(col_u)
    return (
        f"Sets {col} to 181 when ACCOUNTCAL.{account_col} is 0 or unset; "
        f"otherwise retains existing {col}."
    )


def _s01_account_column_display_name(col_u: str) -> str:
    mapping = {
        "REFPERIODSTKSTATEMENT": "RefPeriodStkStatement",
        "REFPERIODREVIEW": "RefPeriodReview",
    }
    return mapping.get(col_u, col_u)


_S02_AQUA_SCHEME_PHRASE = "DIMPRODUCT.Aqua_Scheme = 'Y' and SchemeType = 'ODA'"

_S02_AQUA_RESET_DATE_COLUMNS = frozenset(
    {
        "INTNOTSERVICEDDT",
        "LASTCRDATE",
        "CONTIEXCESSDT",
        "OVERDUESINCEDT",
        "REVIEWDUEDT",
        "STOCKSTDT",
        "DEBITSINCEDT",
    }
)

_S02_DPD_CALC_COLUMNS = frozenset(
    {
        "DPD_INTSERVICE",
        "DPD_NOCREDIT",
        "DPD_OVERDRAWN",
        "DPD_OVERDUE",
        "DPD_RENEWAL",
        "DPD_STOCKSTMT",
        "DPD_PRINCOVERDUE",
        "DPD_INTOVERDUESINCE",
        "DPD_OTHEROVERDUESINCE",
    }
)

_S02_DPD_SOURCE_DATE_COLUMNS: dict[str, str] = {
    "DPD_INTSERVICE": "IntNotServicedDt",
    "DPD_NOCREDIT": "LastCrDate",
    "DPD_OVERDRAWN": "ContiExcessDt",
    "DPD_OVERDUE": "OverDueSinceDt",
    "DPD_RENEWAL": "ReviewDueDt",
    "DPD_STOCKSTMT": "StockStDt",
    "DPD_PRINCOVERDUE": "PrincOverdueSinceDt",
    "DPD_INTOVERDUESINCE": "IntOverdueSinceDt",
    "DPD_OTHEROVERDUESINCE": "OtherOverdueSinceDt",
}


def _narrate_s02_aqua_scheme_reset(col: str, formula: str, upper: str) -> str | None:
    """S02 DPD_Calculation: Aqua ODA product resets (dates and component DPDs)."""
    compact = upper.replace(" ", "")
    if "AQUA_SCHEME" not in upper.replace("_", "") and "AQUA" not in upper:
        return None
    if "SCHEMETYPE" not in upper or "ODA" not in upper:
        return None
    if "DIMPRODUCT" not in upper:
        return None
    col_u = (col or "").upper()
    aqua = _S02_AQUA_SCHEME_PHRASE
    if col_u in _S02_AQUA_RESET_DATE_COLUMNS and "THEN(NULL)" in compact:
        return (
            f"Resets {col} to NULL for Aqua Scheme ODA products ({aqua}) "
            f"or legacy placeholder dates ('1900-01-01'); otherwise retains existing {col}."
        )
    if col_u.startswith("DPD_") and ("THEN(0)" in compact or "THEN(NULL)" in compact):
        if "DATEDIFF" in upper or "@TIMEKEY" in upper:
            return None
        return (
            f"Sets {col} to 0 for Aqua Scheme ODA products ({aqua}); "
            f"otherwise retains the calculated {col} value."
        )
    return None


def _narrate_s02_account_dpd_timekey(col: str, formula: str, upper: str) -> str | None:
    """S02 AccountCal DPD_* timekey-tiered DATEDIFF calculations."""
    col_u = (col or "").upper()
    if col_u not in _S02_DPD_CALC_COLUMNS:
        return None
    if "DATEDIFF" not in upper:
        return None
    if "@TIMEKEY" not in upper and "TIMEKEY" not in upper.replace("_", ""):
        return None
    source = _S02_DPD_SOURCE_DATE_COLUMNS.get(col_u, "the driving event date")
    steps: list[tuple[str, str]] = []

    if col_u == "DPD_INTSERVICE":
        if "26384" in formula:
            steps.append(
                (
                    "TimeKey > 26384",
                    f"Sets to DATEDIFF({source}, @ProcessDate) + 2 days when {source} is populated.",
                )
            )
        if "26267" in formula:
            steps.append(
                (
                    "Post-Cutoff Processing (@TIMEKEY > 26267)",
                    f"Sets to DATEDIFF({source}, @ProcessDate) + 1 day when {source} is populated.",
                )
            )
        steps.append(
            (
                "Historical Baseline (@TIMEKEY <= 26267)",
                f"Sets to DATEDIFF({source}, @ProcessDate) when {source} is populated.",
            )
        )
    elif col_u == "DPD_NOCREDIT":
        steps.append(
            (
                "Post-Cutoff Processing (@TIMEKEY > 26267)",
                "When DebitSinceDt is null or DATEDIFF(DebitSinceDt, @ProcessDate) is at least "
                "90 days, sets to DATEDIFF(LastCrDate, @ProcessDate) + 1 day when LastCrDate is "
                "populated; otherwise 0.",
            )
        )
        steps.append(
            (
                "Historical Baseline (@TIMEKEY <= 26267)",
                "When the debit-since gate passes, sets to DATEDIFF(LastCrDate, @ProcessDate) "
                "without the +1 day offset; otherwise 0.",
            )
        )
    elif col_u == "DPD_OVERDUE":
        if "26372" in formula:
            steps.append(
                (
                    "TimeKey > 26372",
                    f"Sets to DATEDIFF({source}, @ProcessDate) + 1 day when {source} is populated.",
                )
            )
        steps.append(
            (
                "Post-Cutoff Processing (@TIMEKEY > 26267)",
                f"Sets to DATEDIFF({source}, @ProcessDate) with a +1 day offset (or +0 when "
                f"SourceAlt_Key = 6) when {source} is populated.",
            )
        )
        steps.append(
            (
                "Historical Baseline (@TIMEKEY <= 26267)",
                f"Sets to DATEDIFF({source}, @ProcessDate) when {source} is populated.",
            )
        )
    else:
        offset = "+ 1 day" if "26267" in formula and col_u in {
            "DPD_OVERDRAWN",
            "DPD_RENEWAL",
            "DPD_STOCKSTMT",
            "DPD_PRINCOVERDUE",
            "DPD_INTOVERDUESINCE",
            "DPD_OTHEROVERDUESINCE",
        } else ""
        detail = f"Sets to DATEDIFF({source}, @ProcessDate)"
        if offset:
            detail += f" {offset}"
        detail += f" when {source} is populated."
        if "26267" in formula:
            steps.append(("Post-Cutoff Processing (@TIMEKEY > 26267)", detail))
        steps.append(
            (
                "Historical Baseline (@TIMEKEY <= 26267)",
                f"Sets to DATEDIFF({source}, @ProcessDate) when {source} is populated.",
            )
        )

    steps.append(
        (
            "Zero floor",
            f"Floored at 0 when {source} is missing or the calculated value is negative.",
        )
    )
    if "AQUA_SCHEME" in upper.replace("_", ""):
        steps.append(
            (
                "Aqua ODA reset",
                f"Forced to 0 for accounts matching {_S02_AQUA_SCHEME_PHRASE}.",
            )
        )
    return _format_numbered_ladder(col, "Evaluates", steps)


_S05_DPD_REF_PERIOD_PAIRS: dict[str, tuple[str, str]] = {
    "DPD_INTSERVICE": ("DPD_IntService", "RefPeriodIntService"),
    "DPD_NOCREDIT": ("DPD_NoCredit", "RefPeriodNoCredit"),
    "DPD_OVERDRAWN": ("DPD_Overdrawn", "RefPeriodOverDrawn"),
    "DPD_OVERDUE": ("DPD_Overdue", "RefPeriodOverdue"),
    "DPD_RENEWAL": ("DPD_Renewal", "RefPeriodReview"),
    "DPD_STOCKSTMT": ("DPD_StockStmt", "RefPeriodStkStatement"),
}


_S08_NARRATED_COLUMNS = frozenset(
    {
        "NETBALANCE",
        "FLGEROSION",
        "EROSIONDT",
        "SYSASSETCLASSALT_KEY",
        "DBTDT",
    }
)

_S04_NARRATED_COLUMNS = frozenset(
    {
        "FLGDEG",
        "DEGREASON",
        "DPD_BREACH_DATE",
        "ZERODPD_DATE",
        "SP_EXPIRYEXTENDEDDATE",
        "DEFAULT_REASON",
        "FLG_DEG",
        "NPA_DATE",
        "FINALASSETCLASSALT_KEY",
        "ASSET_NORM",
        "CUSTOMERENTITYID",
        "CNT",
        "ACCOUNTENTITYID",
        "CURRENTVALUE",
    }
)

_S04_STAGING_EXPLANATIONS: dict[tuple[str, str], str] = {
    (
        "UPDATETEMPASSETCLASSASSETNORM4123",
        "CUSTOMERENTITYID",
    ): (
        "Populates CustomerEntityID for customers with exactly one linked account, "
        "scoping FDSEC conditional asset-norm updates."
    ),
    (
        "UPDATETEMPASSETCLASSASSETNORM4123",
        "CNT",
    ): (
        "Stores the per-customer account count (single-account customers only) used for "
        "FDSEC asset-norm eligibility."
    ),
    (
        "UPDATETEMPASSETCLASSASSETNORMTEMP1",
        "ACCOUNTENTITYID",
    ): (
        "Aggregates FDSEC collateral values by AccountEntityID for balance-versus-security "
        "comparisons in asset-norm rules."
    ),
    (
        "UPDATETEMPASSETCLASSASSETNORMTEMP1",
        "CURRENTVALUE",
    ): (
        "Sums security CurrentValue per account for active FDSEC products on the processing "
        "timekey."
    ),
}


def _s04_staging_entity_token(ent_u: str) -> str:
    if "UPDATETEMPASSETCLASS" in ent_u and "4123" in ent_u:
        return "UPDATETEMPASSETCLASSASSETNORM4123"
    if "UPDATETEMPASSETCLASS" in ent_u and "TEMP1" in ent_u:
        return "UPDATETEMPASSETCLASSASSETNORMTEMP1"
    return ent_u


def _formula_is_s04_marking_flgdeg_degreason(upper: str, entity: str = "") -> bool:
    """PRO Marking_FlgDeg_Degreason: degradation flags, reasons, restructure, PUI DCCO."""
    ent_u = entity_key_upper(entity or "")
    if _s04_staging_entity_token(ent_u).startswith("UPDATETEMPASSETCLASSASSETNORM"):
        return True
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    if "UPDATETEMPASSETCLASSASSETNORM" in compact:
        return True
    if ent_u in {"ADVACRESTRUCTURECAL", "PUI_CAL"}:
        return True
    if ent_u == "ACCOUNTCAL" and ("FLGDEG" in compact or "DEGREASON" in compact):
        return True
    if "DPD_BREACH_DATE" in compact or "ZERODPD_DATE" in compact:
        return True
    if "DEFAULT_REASON" in compact and "PUI" in compact:
        return True
    if "DCCO" in compact or "ORIGINALDCCO" in compact:
        return True
    if "CONDI_STD" in compact and "FDSEC" in compact and "REFPERIOD" in compact:
        return True
    if "RESTRUCTUREDSLIPPAGE" in compact or "COVIDOTR" in compact.replace(" ", ""):
        return True
    return False


def _narrate_s04_marking_flgdeg_degreason(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    col_u = (col or "").upper()
    ent_u = entity_key_upper(entity or "")
    staging_ent = _s04_staging_entity_token(ent_u)
    staging_static = _S04_STAGING_EXPLANATIONS.get((staging_ent, col_u))
    if staging_static:
        return staging_static

    if not _formula_is_s04_marking_flgdeg_degreason(upper, entity):
        return None

    if ent_u == "ACCOUNTCAL" and col_u == "FLGDEG":
        steps = [
            (
                "Zero Balance Override",
                "Resets FLGDEG to 'N' if account balance is less than or equal to zero.",
            ),
            (
                "Restructuring Degradation",
                "Sets FLGDEG to 'Y' if restructure evaluation in AdvAcRestructureCal marks the "
                "account for degradation.",
            ),
            (
                "Standard DPD Servicing Breaches",
                "Sets FLGDEG to 'Y' for active standard accounts if any DPD servicing metric "
                "(IntService, Overdrawn, NoCredit, Overdue, StockStmt, or Renewal) meets or "
                "exceeds its reference period.",
            ),
            (
                "Buyout / Virtual DPD",
                "Sets FLGDEG to 'Y' if buyout details indicate seller NPA or virtual DPD breach.",
            ),
            ("Default Fallback", "Retains existing FLGDEG value."),
        ]
        return _format_fixed_title_ladder(
            "Evaluates account-level degradation status (FLGDEG) across prioritized conditions:",
            steps,
        )

    if ent_u == "ACCOUNTCAL" and col_u == "DEGREASON":
        steps = [
            (
                "Customer Percolation",
                "Inherits DegReason from CUSTOMERCAL if account is non-degraded but customer "
                "is degraded.",
            ),
            (
                "Servicing DPD Breaches",
                "Concatenates specific breach descriptions ('DEGRADE BY INT NOT SERVICED', "
                "'CONTI EXCESS', 'NO CREDIT', 'STOCK STATEMENT', 'REVIEW DUE DATE', 'OVERDUE') "
                "based on threshold breaches.",
            ),
            (
                "Zero Balance Reset",
                "Resets DegReason to NULL for zero-balance accounts.",
            ),
            (
                "Buyout Flags",
                "Sets to 'NPA with Seller' or 'NPA Due to Virtual DPD' for buyout accounts.",
            ),
            ("Default Fallback", "Retains existing DegReason."),
        ]
        return _format_fixed_title_ladder(
            "Evaluates and appends degradation reasons across prioritized conditions:",
            steps,
        )

    if ent_u == "ADVACRESTRUCTURECAL" and col_u == "DPD_BREACH_DATE":
        return (
            "Sets DPD_Breach_Date to process date when financial or non-financial DPD breach "
            "thresholds are met for MSME, COVID OTR, IRAC, or Prudential restructuring types; "
            "resets to NULL when DPD drops below breach limits."
        )

    if ent_u == "ADVACRESTRUCTURECAL" and col_u == "ZERODPD_DATE":
        return (
            "Sets ZeroDPD_Date to process date when MSME restructured accounts cure back to zero "
            "DPD prior to special period expiry; otherwise resets to NULL."
        )

    if ent_u == "ADVACRESTRUCTURECAL" and col_u == "SP_EXPIRYEXTENDEDDATE":
        return (
            "Extends special period expiry date by 1 year from ZeroDPD_Date for cured MSME "
            "restructured accounts."
        )

    if ent_u == "ADVACRESTRUCTURECAL" and col_u == "DEGREASON":
        return (
            "Sets DegReason to 'Restructured slippage' for COVID OTR restructured accounts where "
            "POS ratio is <= 30%."
        )

    if ent_u == "PUI_CAL" and col_u == "DEFAULT_REASON":
        return (
            "Evaluates PUI DCCO breach reasons: tags 'Original DCCO is Crossed', 'Revised DCCO "
            "is crossed', or 'revised DCCO is not in permitted DCCO deferment' based on project "
            "category deferment limits."
        )

    if ent_u == "PUI_CAL" and col_u in {"FLG_DEG", "FLGDEG", "NPA_DATE", "FINALASSETCLASSALT_KEY"}:
        return (
            "Marks PUI account as degraded (FLG_DEG = 'Y', FinalAssetClassAlt_Key = 2) on process "
            "date when DCCO breach reasons exist; resets to standard (FLG_DEG = 'N', "
            "FinalAssetClassAlt_Key = 1) if MOC or Adhoc ACL overrides apply."
        )

    if ent_u == "ACCOUNTCAL" and col_u == "ASSET_NORM" and "FDSEC" in upper:
        return (
            "Sets Asset_Norm to 'ALWYS_STD' or 'CONDI_STD' for single-account FDSEC customers "
            "when collateral value versus balance thresholds and DPD reference rules are met."
        )

    return None


_S12_NARRATED_COLUMNS = frozenset(
    {
        "FLGPNPA",
        "PNPA_DATE",
        "PNPA_REASON",
        "PNPA_DT",
        "PNPA_CLASS_KEY",
        "CUSTOMERENTITYID",
        "UCIFENTITYID",
    }
)

# Entity-scoped copy wins over formula-shape heuristics (staging MIN aggregates lack @PNPA_DAYS).
_S12_ENTITY_COLUMN_EXPLANATIONS: dict[tuple[str, str], str] = {
    (
        "CUSTOMERCAL",
        "FLGPNPA",
    ): (
        "Sets FlgPNPA to 'Y' at the customer level when any underlying account or UCIF "
        "entity has FlgPNPA = 'Y' in active processing cohort; otherwise sets to 'N'."
    ),
    (
        "TEMPTABLEPNPA",
        "PNPA_DATE",
    ): (
        "Aggregates and stores the earliest (minimum) PNPA_DATE across all constituent "
        "accounts for each PNPA-marked customer."
    ),
    (
        "CTECUSTOMERWISEBALANCEPNPA",
        "PNPA_DATE",
    ): (
        "Aggregates and stores the earliest (minimum) PNPA_DATE across all constituent "
        "accounts for each PNPA-marked UCIF entity."
    ),
}


def _formula_is_s12_marking_flg_pnpa(upper: str, entity: str = "") -> bool:
    """PRO Marking_FlgPNPA: projected DPD PNPA marking and percolation."""
    ent_u = entity_key_upper(entity or "")
    if ent_u in {"TEMPTABLEPNPA", "CTECUSTOMERWISEBALANCEPNPA"}:
        return True
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    if "FLGPNPA" in compact:
        return True
    if "PNPADAYS" in compact or "PNPAPROCESSDATE" in compact:
        return True
    if "TEMPTABLEPNPA" in compact:
        return True
    if "CTECUSTOMERWISEBALANCEPNPA" in compact:
        return True
    if "PNPACLASSKEY" in compact or "PNPAASSETCLASS" in compact:
        return True
    return False


def _narrate_s12_marking_flg_pnpa(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    col_u = (col or "").upper()
    ent_u = entity_key_upper(entity or "")

    static = _S12_ENTITY_COLUMN_EXPLANATIONS.get((ent_u, col_u))
    if static:
        return static

    if not _formula_is_s12_marking_flg_pnpa(upper, entity):
        return None

    if col_u == "FLGPNPA":
        return (
            "Evaluates Potential NPA (PNPA) status across prioritized conditions:\n"
            "1. Projected DPD Thresholds: Sets FlgPNPA to 'Y' for standard accounts with "
            "positive balance when projected DPD over a 30-day rolling period meets or exceeds "
            "reference period thresholds (IntService, NoCredit, Overdue, StockStmt, Review, or "
            "Overdrawn).\n"
            "2. Customer/UCIF Percolation: Sets FlgPNPA to 'Y' if linked customer or UCIF is "
            "marked as PNPA.\n"
            "3. Default Fallback: Sets FlgPNPA to 'N'."
        )

    if col_u == "PNPA_DATE":
        return (
            "Evaluates PNPA_DATE across prioritized conditions:\n"
            "1. Primary DPD Projection: Calculates PNPA_DATE as the earliest projected date when "
            "DPD thresholds are breached across servicing metrics.\n"
            "2. Percolation & Linking: Propagates the minimum PNPA_DATE from linked customer or "
            "UCIF entities.\n"
            "3. Default Fallback: Sets to NULL."
        )

    if col_u == "PNPA_REASON":
        return (
            "Evaluates PNPA_Reason across prioritized conditions:\n"
            "1. High-Priority DPD Reasons: Concatenates breach reasons ('DEGRADE BY INT NOT "
            "SERVICED', 'CONTI EXCESS', 'NO CREDIT', 'OVERDUE', 'STOCK STATEMENT', 'REVIEW DUE "
            "DATE', 'DEBIT BALANCE').\n"
            "2. Account & UCIF Percolation: Sets to 'Link By AccountId <ID>' or 'PERCOLATION BY "
            "UCIF <ID>' for linked accounts without primary DPD breaches.\n"
            "3. Default Fallback: Sets to NULL."
        )

    if col_u == "PNPA_DT":
        return (
            "Sets PNPA_Dt to the minimum PNPA_DATE across all constituent accounts for "
            "PNPA-marked customers or UCIFs; otherwise sets to NULL."
        )

    if col_u == "PNPA_CLASS_KEY":
        return (
            "Sets PNPA_Class_Key to Sub-Standard asset class key for PNPA-marked customers; "
            "otherwise sets to NULL."
        )

    if col_u == "CUSTOMERENTITYID" and ent_u == "TEMPTABLEPNPA":
        return (
            "Populates entity key into PNPA staging extract for active unprocessed accounts "
            "with FlgPNPA = 'Y'."
        )

    if col_u == "UCIFENTITYID" and ent_u == "CTECUSTOMERWISEBALANCEPNPA":
        return (
            "Populates entity key into PNPA staging extract for active unprocessed accounts "
            "with FlgPNPA = 'Y'."
        )

    if col_u in {"CUSTOMERENTITYID", "UCIFENTITYID"} and (
        "TEMPTABLEPNPA" in upper.replace("#", "").replace("_", "").upper()
        or "CTECUSTOMERWISEBALANCEPNPA" in upper.replace("#", "").replace("_", "").upper()
    ):
        return (
            "Populates entity key into PNPA staging extract for active unprocessed accounts "
            "with FlgPNPA = 'Y'."
        )

    return None


def _formula_is_s08_npa_erosion_aging(upper: str, entity: str = "") -> bool:
    """PRO NPA_Erosion_Aging: security erosion + SUB/DB aging on CustomerCal."""
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    if "CTECUSTOMERWISEBALANCE" in compact:
        return True
    if "MOVETOLOSS" in compact or "MOVETODB1" in compact:
        return True
    if "CURNTQTRRV" in compact or "PRVQTRRV" in compact:
        return True
    if "DIMSECURITYEROSION" in compact:
        return True
    if "FLGEROSION" in compact and ("EROSIONDT" in compact or "MOVETOLOSS" in compact):
        return True
    ent = bare_ident(entity or "").upper().lstrip("#")
    if ent in {"CTECUSTOMERWISEBALANCE", "CUSTOMERWISEBALANCE"}:
        return True
    return False


def _narrate_s08_npa_erosion_aging(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    if not _formula_is_s08_npa_erosion_aging(upper, entity):
        return None
    col_u = (col or "").upper()
    ent_u = bare_ident(entity or "").upper().lstrip("#")

    if col_u == "NETBALANCE" and (
        ent_u == "CTECUSTOMERWISEBALANCE"
        or "CTECUSTOMERWISEBALANCE" in upper.replace("#", "").replace("_", "").upper()
    ):
        return (
            "Calculates aggregate NetBalance as SUM(PrincOutStd) for non-standard secured "
            "accounts (SecApp = 'S') with positive balance."
        )

    if col_u == "FLGEROSION":
        return (
            "Sets FlgErosion to 'Y' when realisable value drops below direct loss threshold "
            "(NetBalance * MoveToLoss) or sub-standard threshold (PrvQtrRV * MoveToDB1); "
            "otherwise sets to 'N'."
        )

    if col_u == "EROSIONDT":
        return (
            "Sets ErosionDt to @PROCESSDATE when security erosion criteria are met; "
            "otherwise retains existing ErosionDt or resets to NULL."
        )

    if col_u == "SYSASSETCLASSALT_KEY":
        steps: list[tuple[str, str]] = [
            (
                "Security Erosion",
                "Sets to LOS or DB1 when realisable security value falls below erosion thresholds.",
            ),
            (
                "NPA Aging Transitions",
                "Ages non-standard accounts through SUB, DB1, DB2, and DB3 based on months "
                "elapsed since SysNPA_Dt and DbtDt.",
            ),
            (
                "Default Fallback",
                "Retains existing SysAssetClassAlt_Key.",
            ),
        ]
        return _format_numbered_ladder(col, "Evaluates", steps)

    if col_u == "DBTDT":
        steps = [
            (
                "Security Erosion",
                "Sets DbtDt to @PROCESSDATE when security erosion triggers a shift to "
                "Doubtful (DB1).",
            ),
            (
                "NPA Aging Progression",
                "Sets DbtDt to SysNPA_Dt + SUB_Months upon transitioning from "
                "Sub-Standard to Doubtful.",
            ),
            (
                "Default Fallback",
                "Retains existing DbtDt.",
            ),
        ]
        return _format_numbered_ladder(col, "Evaluates", steps)

    return None


def _formula_is_s06_temptabledpd(upper: str, entity: str = "") -> bool:
    ent = bare_ident(entity or "").upper().lstrip("#")
    if ent == "TEMPTABLEDPD":
        return True
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    return "TEMPTABLEDPD" in compact


def _narrate_s06_temptabledpd_dpd(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    """S06 NPA_Date_Calculation: #TEMPTABLEDPD DPD vs ref-period extract."""
    if not _formula_is_s06_temptabledpd(upper, entity):
        return None
    col_u = (col or "").upper()
    pair = _S05_DPD_REF_PERIOD_PAIRS.get(col_u)
    if not pair:
        return None
    dpd_col, ref_col = pair
    if col_u == "DPD_NOCREDIT" and ("ODA" in upper or "SCHEMETYPE" in upper):
        return (
            "Populates DPD_NoCredit with RefPeriodNoCredit for ODA accounts matching "
            "91-day rule criteria; otherwise sets to DPD_NoCredit when threshold is met, or 0."
        )
    label = col or dpd_col
    return (
        f"Sets {label} to ACCOUNTCAL.{dpd_col} when ACCOUNTCAL.{dpd_col} >= "
        f"ACCOUNTCAL.{ref_col}; otherwise sets to 0 when included in #TEMPTABLEDPD extract."
    )


def _narrate_s06_adv_ac_restructure_deg_date(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    """S06 AdvAcRestructureCal.DegDate from restructure NPA date arms."""
    if (col or "").upper() != "DEGDATE":
        return None
    compact = upper.replace("_", "")
    if not any(
        token in compact
        for token in ("PRERESTRUCTURENPADATE", "RESTRUCTUREDT", "SPEXPIRY")
    ):
        return None
    return (
        "Sets DEGDATE to PreRestructureNPA_Date (or RestructureDt when "
        "PreRestructureNPA_Date is NULL) for active restructured accounts prior to "
        "SP_ExpiryDate or SP_ExpiryExtendedDate; otherwise retains existing DEGDATE."
    )


def _narrate_s06_asset_norm_from_pui(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    """S06 AccountCal.Asset_Norm copy from PUI_CAL on AccountEntityId."""
    if (col or "").upper() != "ASSET_NORM":
        return None
    if "PUI_CAL" not in upper:
        return None
    if '"ASSET_NORM"' not in upper and "ASSET_NORM" not in upper.replace("_", ""):
        return None
    if "ACCOUNTENTITYID" not in upper.replace("_", ""):
        return None
    return (
        "Sets Asset_Norm to PUI_CAL.Asset_Norm when matched on AccountEntityId in PUI_CAL; "
        "otherwise retains existing Asset_Norm."
    )


def _formula_is_s05_maxdpd_staging(upper: str) -> bool:
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    return "TEMPTABLE" in compact and "ACCOUNTCAL" in compact and "CUSTOMERCAL" in compact


def _formula_is_s05_temptable1(upper: str) -> bool:
    compact = (upper or "").replace("#", "").replace("_", "").upper()
    return "TEMPTABLE" in compact and "TEMPTABLE2" not in compact


def _formula_is_s05_temptable2(upper: str) -> bool:
    return "TEMPTABLE2" in (upper or "").replace("#", "").replace("_", "").upper()


def _narrate_s05_temptable_dpd_threshold(col: str, formula: str, upper: str) -> str | None:
    """#TEMPTABLE: per-metric DPD vs ref-period threshold, else 0."""
    col_u = (col or "").upper()
    pair = _S05_DPD_REF_PERIOD_PAIRS.get(col_u)
    if not pair or not _formula_is_s05_maxdpd_staging(upper) or not _formula_is_s05_temptable1(upper):
        return None
    dpd_col, ref_col = pair
    ref_key = ref_col.upper().replace("_", "")
    if ref_key not in upper.replace("_", ""):
        return None
    return (
        f"Populates {col} with ACCOUNTCAL.{dpd_col} when it is greater than or equal to "
        f"ACCOUNTCAL.{ref_col} and CustomerCal.FlgProcessing is 'N'; otherwise sets to 0."
    )


def _narrate_s05_temptable1_customer_ac_id(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    if (col or "").upper() != "CUSTOMERACID":
        return None
    if not _formula_is_s05_temptable1(upper):
        return None
    return (
        "Populates CustomerAcID from ACCOUNTCAL for rows included in the #TEMPTABLE extract "
        "when CustomerCal.FlgProcessing is 'N' and the account meets at least one "
        "component DPD versus reference-period threshold on the joined customer row."
    )


def _narrate_s05_temptable2_customer_ac_id(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    if (col or "").upper() != "CUSTOMERACID":
        return None
    if not _formula_is_s05_temptable2(upper):
        return None
    return (
        "Copies CustomerAcID from the #TEMPTABLE account row when CustomerCal.FlgProcessing "
        "is 'N' (via join on CustomerAcID and CustomerCal)."
    )


def _narrate_s05_temptable2_refperiod(
    col: str, formula: str, upper: str, *, entity: str = ""
) -> str | None:
    if (col or "").upper() != "REFPERIOD":
        return None
    if not _formula_is_s05_temptable2(upper) or _s05_dpd_component_hits(upper) < 3:
        return None
    return (
        "Selects the reference period corresponding to the largest component DPD "
        "(RefPeriodIntService, RefPeriodNoCredit, RefPeriodOverDrawn, RefPeriodReview, "
        "RefPeriodOverdue, or RefPeriodStkStatement) when CustomerCal.FlgProcessing is 'N'."
    )


def _describe_dpd_threshold_else_zero_if(if_node: Tree, col: str) -> str | None:
    """``IF(metric >= ref) THEN(metric) ELSE(0)`` shape inside S05 staging formulas."""
    branches = _collect_if_branches(if_node)
    if len(branches) != 2 or branches[1][0] is not None:
        return None
    _cond, then_b = branches[0]
    else_b = branches[1][1]
    then_leaf = (_column_ref_leaf_name(then_b) or "").upper()
    if not then_leaf.startswith("DPD_"):
        return None
    pair = _S05_DPD_REF_PERIOD_PAIRS.get(then_leaf)
    if not pair:
        return None
    dpd_col, ref_col = pair
    else_scalar = _export_scalar_outcome(else_b)
    if else_scalar not in {None, "0", "'0'"} and not _outcome_is_null(else_b):
        zero_node = _unwrap(else_b)
        if not (isinstance(zero_node, Token) and str(zero_node) in {"0", "0.0"}):
            return None
    label = col or dpd_col
    return (
        f"ACCOUNTCAL.{dpd_col} when it is greater than or equal to ACCOUNTCAL.{ref_col}, "
        f"otherwise 0"
    )


def _s05_dpd_component_hits(upper: str) -> int:
    compact = (upper or "").replace(" ", "").replace("_", "")
    return sum(1 for name in _S05_DPD_COMPONENTS if name.replace("_", "") in compact)


def _narrate_dpd_max_s05(col: str, formula: str, upper: str) -> str | None:
    """PRO MaxDPD_ReferencePeriod_Calculation: max of six component DPDs."""
    if _s05_dpd_component_hits(upper) < 4:
        return None
    steps: list[tuple[str, str]] = [
        (
            "Reprocessing reset",
            "Sets DPD_Max to 0 before recalculation.",
        ),
        (
            "Maximum component DPD",
            "When CustomerCal.FlgProcessing is 'N' and at least one component DPD is positive, "
            "sets DPD_Max to the largest of DPD_IntService, DPD_NoCredit, DPD_Overdrawn, "
            "DPD_Overdue, DPD_Renewal, and DPD_StockStmt.",
        ),
        (
            "Default",
            f"Retains the prior {col} when the customer is processing or all component DPDs are zero.",
        ),
    ]
    return _format_numbered_ladder(col, "Assigns", steps)


def _narrate_dpd_fin_max_type_s05(col: str, formula: str, upper: str) -> str | None:
    """Labels which RefPeriod* metric won the max-DPD comparison."""
    if _s05_dpd_component_hits(upper) < 4:
        return None
    if "REFPERIODINTSERVICE" not in upper.replace("_", "") and "REFPERIODNOCREDIT" not in upper.replace(
        "_", ""
    ):
        return None
    steps: list[tuple[str, str]] = [
        (
            "Winning reference-period type",
            "When CustomerCal.FlgProcessing is 'N' and at least one component DPD is positive, "
            "sets DPD_FinMaxType to the RefPeriod label for the component DPD that is largest "
            "(RefPeriodIntService, RefPeriodNoCredit, RefPeriodOverDrawn, RefPeriodReview, "
            "RefPeriodOverdue, or RefPeriodStkStatement).",
        ),
        (
            "Default",
            f"Retains the prior {col} when no positive component DPD applies.",
        ),
    ]
    return _format_numbered_ladder(col, "Assigns", steps)


def _narrate_refperiod_max_s05(col: str, formula: str, upper: str) -> str | None:
    """REFPeriodMax from max-DPD ref period, deg/facility fallbacks, and FinMaxType backfill."""
    compact = upper.replace(" ", "")
    if "REFPERIOD" not in compact:
        return None
    steps: list[tuple[str, str]] = []
    if _s05_dpd_component_hits(upper) >= 3 or "FLGPROCESSING" in compact:
        steps.append(
            (
                "Reprocessing reset and max-DPD copy",
                "Sets REFPeriodMax to 0 for reprocessing, then copies the RefPeriod* value that "
                "matches the winning component DPD when CustomerCal.FlgProcessing is 'N'.",
            )
        )
    if "FLGDEG" in compact and "FACILITYTYPE" in compact:
        steps.append(
            (
                "Degraded facility fallback",
                "For FlgDeg 'Y' accounts with zero REFPeriodMax, fills from RefPeriodOverdue "
                "(TL/DL/BP/BD/PC) or RefPeriodIntService (CC/OD) when DPD_Max is below that threshold.",
            )
        )
    if "DPD_FINMAXTYPE" in compact:
        steps.append(
            (
                "Fin-max type backfill",
                "When REFPeriodMax is still null, copies the RefPeriod* column named in DPD_FinMaxType.",
            )
        )
    if not steps:
        return None
    steps.append(
        (
            "Default",
            f"Retains the prior {col} when none of the above rules apply.",
        )
    )
    return _format_numbered_ladder(col, "Assigns", steps)


def _running_process_name_compare_phrase(left, op: str, right) -> str | None:
    if op.strip() not in {"==", "="}:
        return None
    left_u = _unwrap(left)
    right_u = _unwrap(right)
    if not isinstance(left_u, Tree) or left_u.data != "column_ref":
        return None
    leaf = (_column_ref_leaf_name(left_u) or "").upper()
    if leaf != "RUNNINGPROCESSNAME":
        return None
    proc = _export_operand(right_u).strip().strip("'\"")
    if not proc:
        return None
    return f"ACLRUNNINGPROCESSSTATUS.RUNNINGPROCESSNAME equals '{proc}'"


def _join_key_equality_compare_phrase(left, op: str, right) -> str | None:
    if op.strip() not in {"==", "="}:
        return None
    left_u, right_u = _unwrap(left), _unwrap(right)
    if not isinstance(left_u, Tree) or not isinstance(right_u, Tree):
        return None
    if left_u.data != "column_ref" or right_u.data != "column_ref":
        return None
    left_path = _column_ref_physical_path(left_u)
    right_path = _column_ref_physical_path(right_u)
    if not left_path or not right_path or "." not in left_path or "." not in right_path:
        return None
    left_leaf = left_path.split(".", 1)[-1]
    right_leaf = right_path.split(".", 1)[-1]
    if left_leaf.upper() != right_leaf.upper():
        return None
    return f"{left_path} matches {right_path}"


def _narrate_loss_dt_s07(col: str, formula: str, upper: str) -> str | None:
    if "FLGDEG" not in upper.replace("_", "") or "FLGPROCESSING" not in upper.replace("_", ""):
        return None
    if "CTECUSTOMERWISEBALANCE" not in upper.replace("_", "").replace("#", "").upper():
        return None
    if "THEN(NULL)" not in upper.replace(" ", ""):
        return None
    return (
        f"Clears {col} to NULL for degraded customers (FlgDeg 'Y', FlgProcessing 'N') in the "
        f"customer-wise balance cohort; otherwise retains existing {col}."
    )


def _narrate_cte_customer_wise_balance(col: str, formula: str, upper: str) -> str | None:
    col_u = col.upper()
    if col_u == "BALANCE":
        return (
            "Aggregates ACCOUNTCAL balances into the customer-wise balance staging cohort "
            "when CUSTOMERCAL.FlgProcessing is 'N' and refcustomerid matches across account "
            "and customer; otherwise retains existing Balance."
        )
    if col_u == "REFCUSTOMERID":
        return (
            "Copies ACCOUNTCAL.refcustomerid into the customer-wise balance cohort when "
            "CUSTOMERCAL.FlgProcessing is 'N' and the account joins to CUSTOMERCAL on "
            "refcustomerid; otherwise retains existing refcustomerid."
        )
    return None


def _narrate_deg_date_process_date(col: str, formula: str, upper: str) -> str | None:
    compact = upper.replace("_", "")
    if any(token in compact for token in ("PRERESTRUCTURENPADATE", "SPEXPIRY", "RESTRUCTUREDT")):
        return None
    if "@PROCESSDATE" not in upper and "PROCESSDATE" not in upper:
        return None
    if "FLGDEG" not in upper.replace("_", ""):
        return None
    return (
        f"Sets {col} to @ProcessDate for degraded customers (FlgDeg 'Y', FlgProcessing 'N') "
        f"in the balance cohort; otherwise retains existing {col}."
    )


def _narrate_inmonthmark_processing(
    col: str,
    formula: str,
    upper: str,
    branches: list[tuple[Tree | None, Tree]],
) -> str | None:
    if "PROC_FREQ" not in upper:
        return None
    if "EOM" not in upper and "EOMONTH" not in upper:
        return None
    cust = "CUSTOMERCAL"
    acct = "ACCOUNTCAL"
    join_key = "SourceSystemCustomerID"
    for entity in re.findall(r'"([^"]+)"', formula):
        ent_u = entity.upper()
        if ent_u in {"A", "B"}:
            continue
        if "CUSTOMER" in ent_u and "CAL" in ent_u:
            cust = entity
        if "ACCOUNT" in ent_u and "CAL" in ent_u:
            acct = entity
    if re.search(r'"A"\s*\.\s*"SourceSystemCustomerID"', formula, re.I) and re.search(
        r'"B"\s*\.\s*"SourceSystemCustomerID"', formula, re.I
    ):
        cust, acct = "CUSTOMERCAL", "ACCOUNTCAL"
    if f'"{join_key}"' in upper or join_key.upper() in upper:
        match_phrase = (
            f"sets to 'Y' if {cust}.{join_key} matches {acct}.{join_key}; "
            f"otherwise sets to 'N'"
        )
    else:
        match_phrase = "sets to 'Y' for matching customer/account keys; otherwise sets to 'N'"
    return (
        f"Assigns {col} across prioritized conditions:\n"
        f"1. Daily Non-EOM Processing: When @PROC_FREQ is 'D' and @PROCESSDATE is not "
        f"End-of-Month: {match_phrase}.\n"
        f"2. Monthly / EOM Processing: Sets to 'Y' for all records."
    )


def _narrate_panno_validation(
    col: str,
    formula: str,
    upper: str,
    branches: list[tuple[Tree | None, Tree]],
) -> str | None:
    dummy_values = _dummy_pan_literals_from_branches(branches) or _dummy_pan_literals_from_formula(upper)
    if not dummy_values and not any(m in upper for m in _DUMMY_ID_MARKERS):
        return None
    if not dummy_values:
        dummy_values = ["FORMO6161O", "FORPM6060F", "AAAAA1111A"]
    listed = ", ".join(_quote_literal_for_export(v) for v in dummy_values[:6])
    relationship = "AdvCustRelationship"
    if "ADVCUSTRELATIONSHIP" in upper:
        relationship = "AdvCustRelationship"
    elif "CUSTOMERRELATIONSHIP" in upper:
        relationship = "customer relationship"
    return (
        f"Clears dummy/invalid PAN codes ({listed}) to NULL; extracts and validates valid "
        f"10-character alphanumeric PAN formats from active {relationship} records; "
        f"otherwise retains current {col}."
    )


def _dummy_pan_literals_from_branches(branches: list[tuple[Tree | None, Tree]]) -> list[str]:
    for cond, outcome in branches:
        if cond is None or not _outcome_is_null(outcome):
            continue
        cond = _unwrap(cond)
        if isinstance(cond, Tree) and cond.data == "membership":
            values = _membership_literal_values(cond)
            return [v.strip("'\"") for v in values]
    return []


def _dummy_pan_literals_from_formula(upper: str) -> list[str]:
    match = re.search(r"IN\s*\[([^\]]+)\]", upper, flags=re.IGNORECASE)
    if not match:
        return []
    chunk = match.group(1)
    return [token.strip().strip('"') for token in re.findall(r'"([^"]+)"', chunk)]


def _narrate_npa_date_priority_ladder(
    col: str,
    upper: str,
    compact: str,
    branches: list[tuple[Tree | None, Tree]],
) -> str | None:
    if len(branches) < 2 and "ELSEIF(" not in compact:
        return None
    steps: list[tuple[str, str]] = []
    if any(token in upper for token in ("MOC", "MANUAL", "MANUALNPA", "MANUAL_NPA")):
        steps.append(("Manual Overrides", "Assigns MOC override date if MOC_DATA matches."))
    if any(token in upper for token in ("BUYOUT", "SELLER", "BUYOUTSELLER")):
        steps.append(
            ("Buyout Seller NPA", "Uses seller NPA date or buyout data date when seller NPA exists.")
        )
    if any(
        token in upper
        for token in ("SETTLEMENT", "LITIGATION", "CHARGE", "CHARGEOFF", "WRITE", "WRITEOFF", "TWO")
    ):
        steps.append(
            (
                "Status Exceptions",
                "Assigns StatusDate for Settlement, Litigation, Charge Off, or Write-Off accounts.",
            )
        )
    if any(token in upper for token in ("PAN", "AADHAAR", "AADHAR", "COBORROW", "CO-BORROW")):
        steps.append(
            ("Card & ID Percolation", "Applies NPA date from temporary PAN / Aadhaar matches.")
        )
    if any(token in upper for token in ("PUI_CAL", "RESTR_NPA", "ALWYS_NPA")):
        steps.append(
            (
                "Priority Classification",
                "Applies ALWAYS_NPA, PUI, or restructuring NPA dates when matched.",
            )
        )
    if "THEN(NULL)" in compact:
        steps.append(
            ("Standard Reset", "Resets to NULL for standard accounts or Adhoc STD overrides.")
        )
    steps.append(("Default Fallback", "Retains historical or source NPA date."))
    if len(steps) < 2:
        return None
    return _format_numbered_ladder(col, "Evaluates", steps)


def _narrate_s13_npa_reason_account(col: str, formula: str, upper: str) -> str | None:
    """S13 Marking_NPA_Reason_NPAAccount: fraud/percolation overrides and DPD reason concatenation."""
    if (col or "").upper() != "NPA_REASON":
        return None
    compact = upper.replace(" ", "")
    if not any(
        token in compact
        for token in (
            "PERCOLATION",
            "FRAUDMARKING",
            "DEGRADEBYINT",
            "ALWYS_NPA",
            "DEGRADEBYCONT",
        )
    ):
        return None
    steps: list[tuple[str, str]] = [
        (
            "High-Priority Overrides",
            "Sets to 'PERCOLATION BY UCIC_ID' or 'NPA DUE TO FRAUD MARKING' when marked as "
            "Fraud or percolated via UCIF_ID.",
        ),
        (
            "Percolation & Linking Rules",
            "Sets to 'PERCOLATION BY PAN CARD', 'PERCOLATION BY AADHAR CARD', or "
            "'Link By AccountId' when primary customer account is degraded.",
        ),
        (
            "DPD & MOC Degradation Concatenation",
            "Sequentially appends 'DEGRADE BY INT NOT SERVICED', 'CONTI EXCESS', "
            "'NO CREDIT', 'OVERDUE', 'STOCK STATEMENT', 'REVIEW DUE DATE', or "
            "'DEGARDE BY MOC' for non-standard accounts with positive balances.",
        ),
        (
            "Default Fallback",
            "Retains existing NPA_Reason.",
        ),
    ]
    return _format_numbered_ladder(col, "Assigns", steps)


def _narrate_asset_norm_ladder(
    col: str,
    upper: str,
    branches: list[tuple[Tree | None, Tree]],
) -> str | None:
    if "ALWYS_STD" not in upper and "ALWYS_NPA" not in upper and len(branches) < 3:
        return None
    if len(branches) >= 3 or ("ALWYS_STD" in upper and "ALWYS_NPA" in upper):
        steps: list[tuple[str, str]] = []
        if "MOC" in upper:
            steps.append(
                ("MOC Override", "Sets to 'ALWYS_STD' or 'ALWYS_NPA' based on MOC classification.")
            )
        if "BUYOUT" in upper or "SELLER" in upper:
            steps.append(("Buyout Seller Status", "Sets to 'ALWYS_NPA' if seller reports NPA date."))
        if "FDSEC" in upper or "CONDI_STD" in upper:
            steps.append(
                (
                    "FD Secured / Conditional STD",
                    "Sets to 'CONDI_STD' for FDSEC product groups within balance limits.",
                )
            )
        if any(token in upper for token in ("SARFAESI", "WEAK", "REPOSSESSION", "FRAUD", "CHARGE")):
            steps.append(
                (
                    "Degradation Overrides",
                    "Sets to 'ALWYS_NPA' for SARFAESI, Weak, Repossession, Fraud, or Charge Off accounts.",
                )
            )
        if not steps:
            steps.append(
                (
                    "Classification Rules",
                    "Sets to 'ALWYS_STD' or 'ALWYS_NPA' based on linked account status tables.",
                )
            )
        steps.append(("Default Fallback", "Defaults to 'NORMAL'."))
        return _format_numbered_ladder(col or "ASSET_NORM", "Assigns", steps)
    return None


def _try_isempty_populate_sentence(if_node: Tree, col: str) -> str | None:
    children = list(if_node.children)
    if len(children) < 2:
        return None
    cond = _unwrap(children[0])
    if not isinstance(cond, Tree) or cond.data != "function_call":
        return None
    name = str(cond.children[0]).upper() if cond.children else ""
    if name != "ISEMPTY":
        return None
    args = _function_args(cond)
    if not args:
        return None
    col_a = _column_ref_leaf_name(args[0]) or _render_value(args[0])
    then_b = children[1]
    else_b = None
    for child in children[2:]:
        if isinstance(child, Tree) and child.data == "else_clause":
            else_b = child.children[0]
            break
    col_b = _column_ref_leaf_name(then_b) if else_b is not None else None
    if not col_b:
        col_b = _column_ref_leaf_name(then_b) or _render_value(then_b)
    return f"Populates {col} using {col_a}, defaulting to {col_b} when unassigned or null."


def _outcome_value_short(node, col: str) -> str:
    node = _unwrap(node)
    val, _ = _render_value_with_note(node)
    if val == "no value":
        return "NULL"
    if _outcome_is_self_column(node, col):
        return f"the current {col}"
    return val


def _outcome_is_qualified_self_column_ref(node, col: str, entity: str = "") -> bool:
    if _outcome_is_self_column(node, col):
        return True
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "column_ref":
        return False
    path = _column_ref_physical_path(node)
    if not path or "." not in path:
        return False
    ent_part, col_part = path.split(".", 1)
    if col_part.upper() != (col or "").upper():
        return False
    if not entity:
        return True
    ent_u = bare_ident(entity).upper().lstrip("#")
    return bare_ident(ent_part).upper().lstrip("#") == ent_u


def _outcome_is_entity_table_passthrough(node, col: str, entity: str) -> bool:
    """``ELSE("TableName")`` pass-through when the table name equals the row entity."""
    if not entity:
        return False
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "column_ref":
        return False
    path = _column_ref_physical_path(node)
    if not path or "." in path:
        return False
    ent_u = bare_ident(entity).upper().lstrip("#")
    path_u = bare_ident(path).upper().lstrip("#")
    col_u = (col or "").upper()
    return path_u == ent_u and col_u != path_u


def _outcome_action_phrase(node, col: str, *, role: str, entity: str = "") -> str:
    if node is None:
        return f"retains existing {col}"
    node = _unwrap(node)
    is_else = role == "else"

    if _outcome_is_null(node):
        return f"retains existing {col}" if is_else else f"Resets {col} to NULL"

    if _outcome_is_qualified_self_column_ref(node, col, entity) or (
        is_else and _outcome_is_entity_table_passthrough(node, col, entity)
    ):
        return f"retains existing {col}"

    scalar = _export_scalar_outcome(node)
    if scalar:
        return f"sets to {scalar}" if is_else else f"Sets {col} to {scalar}"

    if isinstance(node, Tree) and node.data == "function_call":
        append = _describe_concat_append_outcome(node, col)
        if append:
            if is_else:
                return append[0].lower() + append[1:] if append else append
            return append

    if isinstance(node, Tree) and node.data == "if_expr":
        if _EXPORT_LITERAL_STYLE.get():
            inner = _describe_dpd_threshold_else_zero_if(node, col)
            if inner:
                prefix = "uses " if is_else else f"Sets {col} to "
                return f"{prefix}{inner}"
        nested = _summarize_nested_if_outcome(node)
        if _EXPORT_LITERAL_STYLE.get():
            if nested.startswith("Appends "):
                return nested if not is_else else nested[0].lower() + nested[1:]
            return (
                f"Applies a staged conditional assignment for {col}"
                if not is_else
                else f"applies a staged conditional assignment for {col}"
            )
        return f"Sets {col} to {nested}" if not is_else else f"sets to {nested}"

    if isinstance(node, Tree) and node.data == "column_ref":
        path = _column_ref_physical_path(node)
        if path and _outcome_is_self_column(node, col):
            return f"retains existing {col}"
        if path and "." in path:
            if not is_else and _outcome_is_foreign_entity_key(node, col):
                ent = path.split(".", 1)[0]
                return f"Populates {col} from {ent}"
            quoted_path = path
            return f"sets to {quoted_path}" if is_else else f"Sets {col} to {quoted_path}"
        if path and "." not in path:
            if is_else and _outcome_is_entity_table_passthrough(node, col, entity):
                return f"retains existing {col}"
            if _EXPORT_LITERAL_STYLE.get() and not is_else:
                return f"Populates {col} from {path}"
            literal = _quote_literal_for_export(path)
            return f"sets to {literal}" if is_else else f"Sets {col} to {literal}"

    if is_else:
        if isinstance(node, Tree) and node.data == "function_call" and _is_coalesce_function(node):
            return f"populates {col} from historical or source NPA records"
        return f"sets to {_export_outcome_value(node)}"

    if re.fullmatch(r"@\w+", str(_raw_value_text(node)), flags=re.IGNORECASE):
        return f"Sets {col} to the active processing date"

    return f"Sets {col} to {_export_outcome_value(node)}"


def _raw_value_text(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Token):
        return str(node)
    if isinstance(node, Tree):
        if node.data == "column_ref":
            return _column_ref_physical_path(node) or ""
        return _export_tree_node(node)
    return ""


def _outcome_is_self_column(node, col: str) -> bool:
    leaf = _column_ref_leaf_name(node)
    if not leaf:
        return False
    return leaf.upper() == (col or "").upper()


def _outcome_is_foreign_entity_key(node, col: str) -> bool:
    col_u = (col or "").upper()
    if not any(col_u.endswith(s) or col_u == s for s in _IDENTITY_COLUMN_SUFFIXES):
        return False
    if not isinstance(node, Tree) or node.data != "column_ref":
        return False
    leaf = _column_ref_leaf_name(node)
    if not leaf or leaf.upper() != col_u:
        return False
    path = _column_ref_physical_path(node)
    return "." in path


def _column_ref_leaf_name(node) -> str | None:
    node = _unwrap(node)
    if not isinstance(node, Tree) or node.data != "column_ref":
        return None
    parts = [_column_ref_path_part_raw(child) for child in node.children]
    parts = [p for p in parts if p]
    return parts[-1] if parts else None


def _export_condition_phrase(node, formula: str, *, max_len: int = 240) -> str:
    if _EXPORT_LITERAL_STYLE.get():
        max_len = max(max_len, 720)
    text = _render_export_condition(node)
    text = _refine_condition_snippets(node, text, formula)
    text = _expand_and_conditions_readable(node, text)
    return _maybe_truncate_export_text(text, max_len)


def _expand_and_conditions_readable(node, rendered: str) -> str:
    """Spell out AND/OR arms as a readable list for export."""
    node = _unwrap(node)
    if not isinstance(node, Tree):
        return rendered
    if node.data in {"and_call", "and_op"}:
        parts = (
            [_render_export_condition(child) for child in node.children]
            if node.data == "and_call"
            else [_render_export_condition(part) for part in _flatten_chain(node, "and_op")]
        )
        cleaned = [p.strip().rstrip(".") for p in parts if p.strip()]
        if len(cleaned) >= 2:
            return ", and ".join(cleaned)
    if node.data in {"or_call", "or_op"}:
        parts = (
            [_render_export_condition(child) for child in node.children]
            if node.data == "or_call"
            else [_render_export_condition(part) for part in _flatten_chain(node, "or_op")]
        )
        cleaned = [p.strip().rstrip(".") for p in parts if p.strip()]
        if len(cleaned) >= 2:
            return ", or ".join(cleaned)
    return rendered


def _refine_condition_snippets(node, rendered: str, formula: str) -> str:
    upper = (formula or rendered).upper()
    if any(marker in upper for marker in _DEFAULT_DATE_MARKERS):
        if "DIMPRODUCT" in upper or "ODA" in upper or "AQUA" in upper:
            date_bit = "legacy placeholder dates ('1900-01-01')"
            scheme_bit = _S02_AQUA_SCHEME_PHRASE
            if scheme_bit.lower() not in rendered.lower():
                return f"{date_bit} or Aqua Scheme ODA products ({scheme_bit})"
    if "DPD_" in upper or "REFPERIOD" in upper:
        if "days past due" not in rendered.lower():
            rendered = f"{rendered} (days past due / servicing threshold)"
    membership = _membership_in_list_hint(node, upper)
    if membership:
        return membership
    return rendered


def _membership_in_list_hint(node, upper: str) -> str | None:
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "membership":
        lhs_leaf = _column_ref_leaf_name(node.children[0]) if node.children else None
        op = _membership_operator(list(node.children))
        if op != "IN":
            return None
        values = _membership_literal_values(node)
        if not values:
            return None
        lhs_name = lhs_leaf or _export_operand(node.children[0])
        if len(values) == 1:
            joined = values[0]
        elif len(values) == 2:
            joined = f"{values[0]} or {values[1]}"
        else:
            joined = ", ".join(values[:-1]) + f", or {values[-1]}"
        extra = ""
        if lhs_leaf and "ACCOUNTBLK" in lhs_leaf.upper():
            extra = " (Credit Card Settlement)"
        return f"{lhs_name} is {joined}{extra}"
    return None


def _membership_literal_values(node: Tree) -> list[str]:
    children = list(node.children)
    rhs = children[2] if len(children) > 2 else None
    if rhs is None:
        return []
    if isinstance(rhs, Tree) and rhs.data == "value_list":
        out: list[str] = []
        for child in rhs.children:
            raw = _raw_literal_token(child)
            if raw:
                out.append(
                    _quote_literal_for_export(raw) if raw.isalnum() or "_" in raw else raw
                )
        return out
    raw = _raw_literal_token(rhs)
    return [raw] if raw else []


def _raw_literal_token(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Token):
        if node.type == "STRING":
            try:
                return str(json.loads(str(node)))
            except Exception:
                return str(node).strip('"')
        return str(node).strip('"')
    if isinstance(node, Tree) and node.data == "column_ref":
        return _column_ref_leaf_name(node) or ""
    return ""


def _enrich_dynamic_sentence(
    sentence: str,
    col: str,
    entity: str,
    formula: str,
    if_node: Tree,
    branches: list[tuple[Tree | None, Tree]],
) -> str:
    upper = formula.upper()
    col_u = (col or "").upper()

    if len(branches) >= 2:
        return sentence

    if "SPLCATG" in col_u and "ALT_KEY" in col_u:
        if "EFFECTIVEFROMTIMEKEY" in upper or "EFFECTIVETOTIMEKEY" in upper:
            source = _dominant_entity_in_formula(formula, exclude_entity=entity) or "the linked customer detail table"
            return (
                f"Assigns the special category classification key from {source} when "
                f"effective dates are valid; defaults to 0."
            )

    if "THEN(NULL)" in upper.replace(" ", "") and col_u in {
        "SRCNPA_DT",
        "DBTDT",
        "NPADT",
        "NPADATE",
        "SYSNPA_DT",
        "FINALNPADT",
        "INITIALNPADT",
    } or (
        "FINALASSETCLASSALT_KEY" in upper
        and "THEN(NULL)" in upper.replace(" ", "")
        and any(t in upper for t in ("ASSETCLASS", "ALWYS_STD", "ALWYS_NPA", "NPA"))
    ):
        return (
            f"Resets {col} to NULL for standard accounts; otherwise populates from "
            f"historical or source NPA records."
        )

    if col_u in {"AADHARCARDNO", "AADHAR_NO", "AADHARNO"} or (
        "AADHAR" in col_u and col_u != "PANNO"
    ):
        if " IN [" in upper or any(m in upper for m in _DUMMY_ID_MARKERS):
            return (
                f"Validates and extracts {col} from customer relationship records, "
                f"filtering out default/dummy values."
            )

    if "DATEDIFF(" in upper or ("REFPERIOD" in upper and ("DPD" in upper or "INTSERVICE" in upper)):
        if "days past due" not in sentence.lower():
            return (
                f"Calculates {col} by evaluating days past due (DPD) and servicing timelines "
                f"against regulatory classification thresholds."
            )

    return sentence


def _dominant_entity_in_formula(formula: str, *, exclude_entity: str) -> str | None:
    exclude = (exclude_entity or "").upper().lstrip("#")
    counts: dict[str, int] = {}
    for match in re.finditer(r'"([^"]+)"\."([^"]+)"', formula):
        ent = match.group(1).strip()
        norm = ent.upper().lstrip("#")
        if exclude and norm == exclude:
            continue
        counts[ent] = counts.get(ent, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: item[1])[0]