"""Deterministic human-readable explanations for platform conditions.

The explainer walks the real 4X grammar tree and turns the accepted
machine-readable condition into plain English without changing the
underlying logic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from lark import Lark, Token, Tree
from lark.exceptions import LarkError

_GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "fourx_grammar.lark"
_parser = Lark(_GRAMMAR_PATH.read_text(), parser="earley", start="start")

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


def explain_expression(expression: str) -> str | None:
    try:
        tree = _parser.parse(expression)
    except LarkError:
        return None

    try:
        node = _unwrap(tree)
        if isinstance(node, Tree) and node.data == "if_expr":
            text = _render_if_chain(node)
        else:
            text = f"The result is {_render_value(node)}."
    except Exception:
        return None

    return _normalize_sentence(text)


def _normalize_sentence(text: str) -> str:
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return ""
    if stripped[-1] not in ".!?":
        stripped += "."
    return stripped[0].upper() + stripped[1:]


def _unwrap(node):
    while isinstance(node, Tree) and node.data not in {"if_expr", "function_call", "column_ref", "value_list", "arg_list", "not_op", "neg"} and len(node.children) == 1:
        node = node.children[0]
    return node


def _render_if_chain(node: Tree) -> str:
    children = list(node.children)
    if len(children) < 2:
        return "The rule returns a conditional result."

    condition = children[0]
    then_branch = children[1]
    rest = children[2:]
    elseif_clauses = [c for c in rest if isinstance(c, Tree) and c.data == "elseif_clause"]
    else_clause = next((c for c in rest if isinstance(c, Tree) and c.data == "else_clause"), None)

    parts = [f"If {_render_condition(condition)}, {_render_outcome(then_branch)}."]
    for clause in elseif_clauses:
        clause_condition, clause_branch = clause.children[0], clause.children[1]
        parts.append(f"Otherwise, if {_render_condition(clause_condition)}, {_render_outcome(clause_branch)}.")
    if else_clause is not None:
        parts.append(f"Otherwise, {_render_outcome(else_clause.children[0])}.")

    return " ".join(parts)


def _render_outcome(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "if_expr":
        nested = _render_if_chain(node).rstrip(".")
        return f"another rule applies: {nested[0].lower() + nested[1:]}"
    return f"the result is {_render_value(node)}"


def _render_condition(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Tree):
        if node.data == "or_op":
            parts = _flatten_chain(node, "or_op")
            return _join_natural(
                [_render_condition(part) for part in parts],
                "or",
                prefix="at least one of the following is true:",
            )
        if node.data == "and_op":
            parts = _flatten_chain(node, "and_op")
            return _join_natural(
                [_render_condition(part) for part in parts],
                "and",
                prefix="all of the following are true:",
            )
        if node.data == "or_call":
            return _join_natural(
                [_render_condition(child) for child in node.children],
                "or",
                prefix="at least one of the following is true:",
            )
        if node.data == "and_call":
            return _join_natural(
                [_render_condition(child) for child in node.children],
                "and",
                prefix="all of the following are true:",
            )
        if node.data == "not_op":
            return f"it is not true that {_render_condition(node.children[0])}"
        if node.data == "compare" and len(node.children) == 3:
            left, op, right = node.children
            return _render_comparison(_render_value(left), str(op), _render_value(right))
        if node.data == "between_op" and len(node.children) == 3:
            value, low, high = node.children
            return f"{_render_value(value)} is between {_render_value(low)} and {_render_value(high)}"
        if node.data == "membership":
            return _render_membership(node)
        if node.data == "function_call":
            return _render_function_condition(node)
        if node.data == "if_expr":
            return _render_if_chain(node)
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


def _render_membership(node: Tree) -> str:
    children = list(node.children)
    if len(children) < 2:
        return _render_value(node)

    lhs = _render_value(children[0])
    op = _membership_operator(children)
    if op is None:
        return _render_value(node)

    if op in {"IN", "NOTIN", "HRCHYIN", "HRCHYNOTIN", "CONTAINS", "BEGINSWITH", "ENDSWITH", "DOESNOTCONTAINS"}:
        rhs = _render_membership_rhs(children[2] if len(children) > 2 else None)
        phrase = _MEMBERSHIP_WORDS.get(op, "relates to")
        return f"{lhs} {phrase} {rhs}"
    return _render_value(node)


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
    phrase = _render_function_phrase(name, args)
    if phrase is not None:
        return phrase
    return "the result of the function call"


def _render_function_phrase(name: str, args: list) -> str | None:
    if name == "ISEMPTY":
        return f"{_render_value(args[0])} is blank or missing" if args else "the value is blank or missing"
    if name == "ISNOTEMPTY":
        return f"{_render_value(args[0])} has a value" if args else "the value has a value"
    if name == "COALESCE":
        return f"the first available value among {_join_items([_render_value(arg) for arg in args], 'or')}"
    if name == "CONCAT":
        return f"the combined text of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "TODATE":
        return f"the date value for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "DATE":
        return f"the date value for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "DATEDIFF":
        if len(args) >= 3:
            unit = _render_value(args[2]).lower()
            return f"the number of {unit} between {_render_value(args[0])} and {_render_value(args[1])}"
        return f"the elapsed time between {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "DATEPART":
        if len(args) >= 2:
            return f"the {_render_value(args[0]).lower()} from {_render_value(args[1])}"
        return f"the extracted date part from {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "MAX":
        return f"the highest of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "MIN":
        return f"the lowest of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "ABS":
        return f"the absolute value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "ROUND":
        return f"the rounded value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "FLOOR":
        return f"the rounded-down value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "CEIL":
        return f"the rounded-up value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "REGEX":
        return f"a pattern match on {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "LOWER":
        return f"the lowercased value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "UPPER":
        return f"the uppercased value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "TRIM":
        return f"the trimmed value of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "LEN":
        return f"the length of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "SUBSTR":
        if len(args) >= 3:
            return f"the substring of {_render_value(args[0])} starting at {_render_value(args[1])} with length {_render_value(args[2])}"
        if len(args) >= 2:
            return f"the substring of {_render_value(args[0])} starting at {_render_value(args[1])}"
        return f"the substring of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "REPLACE":
        if len(args) >= 3:
            return f"the value of {_render_value(args[0])} with {_render_value(args[1])} replaced by {_render_value(args[2])}"
        return f"the replaced text of {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "SOM":
        return f"the start of the month for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "EOM":
        return f"the end of the month for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "SOY":
        return f"the start of the year for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "EOY":
        return f"the end of the year for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "SOFY":
        return f"the start of the financial year for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "EOFY":
        return f"the end of the financial year for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name == "PERIOD":
        return f"the period value for {_join_items([_render_value(arg) for arg in args], 'and')}"
    if name in {"AND", "OR"}:
        conj = "and" if name == "AND" else "or"
        prefix = "all of" if name == "AND" else "any of"
        return _join_natural([_render_condition(arg) for arg in args], conj, prefix=prefix)
    if not args:
        return None
    return None


def _function_args(node: Tree) -> list:
    if len(node.children) < 2:
        return []
    args_node = node.children[1]
    if isinstance(args_node, Tree) and args_node.data == "arg_list":
        return list(args_node.children)
    return [args_node]


def _render_value(node) -> str:
    node = _unwrap(node)
    if isinstance(node, Token):
        return _render_token(node)
    if isinstance(node, Tree):
        if node.data == "column_ref":
            return _render_column_ref(node)
        if node.data == "function_call":
            return _render_function_value(node)
        if node.data in {"if_expr", "compare", "between_op", "membership", "and_op", "or_op", "not_op", "and_call", "or_call"}:
            return _render_condition(node)
        if node.data == "add":
            return _render_nary(node, "plus", "sum of")
        if node.data == "sub":
            return _render_binary(node, "minus", "difference between")
        if node.data == "mul":
            return _render_nary(node, "multiplied by", "product of")
        if node.data == "div":
            return _render_binary(node, "divided by", "quotient of")
        if node.data == "neg":
            return f"negative {_render_value(node.children[0])}"
        if node.data == "value_list":
            return _join_items([_render_value(child) for child in node.children], "or")
        if len(node.children) == 1:
            return _render_value(node.children[0])
    return "the calculated value"


def _render_function_value(node: Tree) -> str:
    name = str(node.children[0]).upper()
    args = _function_args(node)
    phrase = _render_function_phrase(name, args)
    if phrase is not None:
        return phrase
    return "the result of the function call"


def _render_binary(node: Tree, op_word: str, prefix: str) -> str:
    left, right = node.children[0], node.children[1]
    return f"{prefix} {_render_value(left)} and {_render_value(right)}"


def _render_nary(node: Tree, op_word: str, prefix: str) -> str:
    values = [_render_value(child) for child in node.children]
    return f"{prefix} {_join_items(values, op_word)}"


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
        text = text.strip()
        if text:
            rendered_parts.append(_render_name_text(text))
            original_parts.append((text, is_string_literal))
    if not rendered_parts:
        return "the field"
    if len(rendered_parts) >= 2 and rendered_parts[-2].upper() == "VAR" and rendered_parts[-1].upper() == "BUSINESS_DATE":
        return "the business date"
    if len(rendered_parts) == 1:
        text, _was_string = original_parts[0]
        if text.upper() in {"TRUE", "FALSE", "NULL", "YES", "NO", "Y", "N"}:
            return _render_literal_text(text)
        if _was_string and not re.fullmatch(r"[A-Z][A-Z0-9_]*", text):
            return _render_literal_text(text)
        return f"the {text} field"
    return f"the {rendered_parts[-1]} field in {_join_items(rendered_parts[:-1], 'and')}"


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
    text = text.replace("_", " ")
    if text.isupper():
        return text.lower()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text.lower()


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
    return _render_name_text(text)


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


def _join_natural(items: list[str], conjunction: str, prefix: str | None = None) -> str:
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
