"""Deterministic human-readable explanations for platform conditions.

The explainer walks the real 4X grammar tree and turns the accepted
machine-readable condition into a concise business decision flow without
changing the underlying logic.
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
        "MAX",
        "MIN",
        "FLG",
        "REF",
        "SMA",
        "DPD",
        "ID",
        "NO",
        "INT",
    },
    key=len,
    reverse=True,
)


def explain_expression(expression: str) -> str | None:
    try:
        tree = _parser.parse(expression)
    except LarkError:
        return None

    try:
        node = _unwrap(tree)
        lines = _render_node(node, depth=0)
    except Exception:
        return None

    return "\n".join(_normalize_line(line) for line in lines if line.strip())


def _normalize_line(text: str) -> str:
    stripped = " ".join(text.split()).strip()
    if not stripped:
        return ""
    if stripped[-1] not in ".!?:":
        stripped += "."
    return stripped[0].upper() + stripped[1:]


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
    value, note = _render_value_with_note(node)
    if value == "no value":
        return [f"{_indent(depth)}- Leave the field blank"]
    if note:
        return [f"{_indent(depth)}- Return {value}, {note}"]
    return [f"{_indent(depth)}- Return {value}"]


def _render_if_chain(node: Tree, depth: int) -> list[str]:
    children = list(node.children)
    if len(children) < 2:
        return [f"{_indent(depth)}- Return the calculated value"]

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


def _render_branch_result(node, depth: int) -> list[str]:
    node = _unwrap(node)
    if isinstance(node, Tree) and node.data == "if_expr":
        nested = _render_if_chain(node, depth)
        return nested

    value, note = _render_value_with_note(node)
    if value == "no value":
        return [f"{_indent(depth)}- Leave the field blank"]
    if note:
        return [f"{_indent(depth)}- Return {value}, {note}"]
    return [f"{_indent(depth)}- Return {value}"]


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
            return _render_if_chain(node, 0)[0].lstrip("- ").rstrip(":")
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
    if name == "MAX":
        return f"the highest of {_join_items([_render_value(arg) for arg in args], 'and')}", None
    if name == "MIN":
        return f"the lowest of {_join_items([_render_value(arg) for arg in args], 'and')}", None
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
    if isinstance(node, Token):
        return _render_token(node), None
    if isinstance(node, Tree):
        if node.data == "column_ref":
            return _render_column_ref(node), None
        if node.data == "function_call":
            return _render_function_value_with_note(node, direct=not as_operand)
        if node.data in {"if_expr", "compare", "between_op", "membership", "and_op", "or_op", "not_op", "and_call", "or_call"}:
            # Value-context rendering should never repeat the full
            # decision-chain text; use the condition wording instead.
            return _render_condition(node), None
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
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
                text = _render_name_text(text)
            rendered_parts.append(text)
            original_parts.append((text, is_string_literal))
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
