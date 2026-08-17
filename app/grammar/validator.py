"""Architecture step 13d: validate a generated Formula Expression string
against the platform's real grammar before it's accepted into the DD Model.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

from lark import Lark, UnexpectedInput

_GRAMMAR_PATH = Path(__file__).parent / "fourx_grammar.lark"

_parser = Lark(_GRAMMAR_PATH.read_text(), parser="earley", start="start")


@dataclass
class ValidationResult:
    valid: bool
    error: str | None = None


# Function names the grammar accepts syntactically as FUNC_NAME, cross-checked
# here against the actual documented library so an expression using a made-up
# function name (syntactically valid, semantically wrong) is still rejected.
KNOWN_FUNCTIONS = {
    "ISEMPTY", "ISNOTEMPTY", "MAX", "MIN", "COALESCE",
    "SUBSTR", "LOWER", "UPPER", "LEN", "CONVERT", "REGEX", "CONCAT", "TRIM", "REPLACE",
    "SOM", "EOM", "SOY", "EOY", "SOFY", "EOFY", "DATEPART", "DATEDIFF", "TODATE",
    "ADDDAY", "PERIOD", "SOQ", "EOQ",
    "ROUND", "ABS", "FLOOR", "CEIL", "DATE",
}


def _flatten_whitespace(expression: str) -> str:
    return " ".join(expression.split())


def _rewrite_legacy_else_if(expression: str) -> str:
    return expression.replace("ELSE IF", "ELSEIF").replace("else if", "ELSEIF")


def _rewrite_not_in_membership(expression: str) -> str:
    return re.sub(r"(?i)\bNOT\s+IN\b", "NOTIN", expression)


def _rewrite_is_empty_syntax(expression: str) -> str:
    expression = re.sub(r"(?i)\bIS\s+NOT\s+EMPTY\b", "ISNOTEMPTY", expression)
    expression = re.sub(r"(?i)\bIS\s+EMPTY\b", "ISEMPTY", expression)
    return expression


def _rewrite_isnotempty_boolean_comparisons(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        literal = match.group(3)
        return f'ISNOTEMPTY({inner}) AND {inner}=="{literal}"'

    return re.sub(
        r'(?i)\bISNOTEMPTY\s*\(\s*([^)]+?)\s*\)\s*==\s*(["\'])(Y|N)\2',
        lambda match: replace(match),
        expression,
    )


def _rewrite_null_predicates(expression: str) -> str:
    expression = re.sub(
        r'(?i)\b([A-Za-z_][A-Za-z0-9_".]*?)\s+IS\s+NOT\s+NULL\b',
        r"ISNOTEMPTY(\1)",
        expression,
    )
    expression = re.sub(
        r'(?i)\b([A-Za-z_][A-Za-z0-9_".]*?)\s+IS\s+NULL\b',
        r"ISEMPTY(\1)",
        expression,
    )
    return expression


def _rewrite_date_function(expression: str) -> str:
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"TODATE({inner})"

    return re.sub(r"(?i)\bDATE\s*\(\s*([^()]+?)\s*\)", replace, expression)


def _rewrite_exists_predicates(expression: str) -> str:
    result: list[str] = []
    i = 0
    in_double = False
    while i < len(expression):
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if not in_double and expression[i : i + 7].upper() == "EXISTS(":
            open_index = i + 6
            depth = 0
            close_index = -1
            local_in_double = False
            for j in range(open_index, len(expression)):
                cur = expression[j]
                if cur == '"':
                    local_in_double = not local_in_double
                elif not local_in_double:
                    if cur == "(":
                        depth += 1
                    elif cur == ")":
                        depth -= 1
                        if depth == 0:
                            close_index = j
                            break
            if close_index != -1:
                inner = expression[i + 7 : close_index].strip()
                where_match = re.search(r"(?is)\bWHERE\b", inner)
                if where_match:
                    result.append(inner[where_match.end():].strip())
                else:
                    comma_match = re.search(r"(?s),", inner)
                    if comma_match:
                        result.append(inner[comma_match.end():].strip())
                    else:
                        result.append(inner)
                i = close_index + 1
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _rewrite_unquoted_dotted_refs(expression: str) -> str:
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False
    while i < n:
        ch = expression[i]
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            i += 1
            continue
        if in_double:
            result.append(ch)
            i += 1
            continue
        if ch.isalpha() or ch == "_":
            start = i
            j = i + 1
            while j < n and (expression[j].isalnum() or expression[j] == "_"):
                j += 1
            parts = [expression[start:j]]
            k = j
            while True:
                l = k
                while l < n and expression[l].isspace():
                    l += 1
                if l >= n or expression[l] != ".":
                    break
                m = l + 1
                while m < n and expression[m].isspace():
                    m += 1
                if m >= n or not (expression[m].isalpha() or expression[m] == "_"):
                    break
                p = m + 1
                while p < n and (expression[p].isalnum() or expression[p] == "_"):
                    p += 1
                parts.append(expression[m:p])
                k = p
            if len(parts) > 1:
                result.append(".".join(f'"{part}"' for part in parts))
                i = k
                continue
        result.append(ch)
        i += 1
    return "".join(result)


def _rewrite_legacy_if_syntax(expression: str) -> str:
    def split_top_level_args(text: str) -> list[str] | None:
        args: list[str] = []
        current: list[str] = []
        depth = 0
        bracket_depth = 0
        in_string = False
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_string = not in_string
                current.append(ch)
            elif not in_string and ch == "(":
                depth += 1
                current.append(ch)
            elif not in_string and ch == ")":
                if depth == 0:
                    return None
                depth -= 1
                current.append(ch)
            elif not in_string and ch == "[":
                bracket_depth += 1
                current.append(ch)
            elif not in_string and ch == "]":
                if bracket_depth == 0:
                    return None
                bracket_depth -= 1
                current.append(ch)
            elif not in_string and ch == "," and depth == 0 and bracket_depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        args.append("".join(current).strip())
        return args if len(args) == 3 else None

    def rewrite_once(text: str) -> str:
        result: list[str] = []
        i = 0
        in_string = False
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_string = not in_string
                result.append(ch)
                i += 1
                continue

            if not in_string and text[i : i + 3].upper() == "IF(":
                start = i + 3
                depth = 1
                j = start
                local_in_string = False
                while j < len(text):
                    cur = text[j]
                    if cur == '"':
                        local_in_string = not local_in_string
                    elif not local_in_string:
                        if cur == "(":
                            depth += 1
                        elif cur == ")":
                            depth -= 1
                            if depth == 0:
                                break
                    j += 1
                if depth == 0:
                    inner = text[start:j]
                    parts = split_top_level_args(inner)
                    if parts:
                        condition, when_true, when_false = parts
                        result.append(f"IF({condition})THEN({when_true})ELSE({when_false})")
                        i = j + 1
                        continue

            result.append(ch)
            i += 1
        return "".join(result)

    previous = expression
    for _ in range(3):
        rewritten = rewrite_once(previous)
        rewritten = _rewrite_legacy_else_if(rewritten)
        if rewritten == previous:
            return rewritten
        previous = rewritten
    return previous


def _repair_missing_then_parentheses(expression: str) -> str:
    """Close an unterminated IF/ELSEIF condition immediately before THEN.

    The model sometimes emits shapes like `IF(cond THEN(...)` instead of
    `IF(cond)THEN(...)`. This is a structural typo rather than a logic
    issue, so we repair it mechanically before grammar validation.
    """

    def find_then(text: str, start: int) -> int:
        depth = 1
        in_string = False
        i = start
        while i < len(text):
            ch = text[i]
            if ch == '"':
                in_string = not in_string
                i += 1
                continue
            if in_string:
                i += 1
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif depth >= 1 and text[i : i + 4].upper() == "THEN":
                before = text[i - 1] if i > 0 else ""
                after = text[i + 4] if i + 4 < len(text) else ""
                if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                    return i
            i += 1
        return -1

    result: list[str] = []
    i = 0
    in_string = False
    while i < len(expression):
        ch = expression[i]
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if not in_string and (
            expression[i : i + 3].upper() == "IF(" or expression[i : i + 7].upper() == "ELSEIF("
        ):
            token = "IF(" if expression[i : i + 3].upper() == "IF(" else "ELSEIF("
            start = i + len(token)
            then_index = find_then(expression, start)
            if then_index != -1:
                depth = 1
                local_in_string = False
                j = start
                while j < then_index:
                    cur = expression[j]
                    if cur == '"':
                        local_in_string = not local_in_string
                    elif not local_in_string:
                        if cur == "(":
                            depth += 1
                        elif cur == ")":
                            depth -= 1
                    j += 1
                prefix = expression[i:then_index]
                if depth > 1:
                    result.append(prefix)
                    result.append(")" * (depth - 1))
                    i = then_index
                    continue
                if depth < 1:
                    trim_count = 1 - depth
                    trimmed_prefix = prefix
                    while trim_count > 0 and trimmed_prefix.endswith(")"):
                        trimmed_prefix = trimmed_prefix[:-1]
                        trim_count -= 1
                    if trim_count == 0:
                        result.append(trimmed_prefix)
                        i = then_index
                        continue
                if depth == 1:
                    result.append(prefix)
                    i = then_index
                    continue

        result.append(ch)
        i += 1
    return "".join(result)


def _normalize_expression(expression: str) -> str:
    normalized = _flatten_whitespace(expression)
    normalized = _rewrite_legacy_else_if(normalized)
    normalized = _rewrite_not_in_membership(normalized)
    normalized = _rewrite_is_empty_syntax(normalized)
    normalized = _rewrite_isnotempty_boolean_comparisons(normalized)
    normalized = _rewrite_null_predicates(normalized)
    normalized = _rewrite_date_function(normalized)
    normalized = _rewrite_unquoted_dotted_refs(normalized)
    normalized = _rewrite_exists_predicates(normalized)
    normalized = _rewrite_legacy_if_syntax(normalized)
    normalized = _repair_missing_then_parentheses(normalized)
    return normalized


def validate_expression(expression: str) -> ValidationResult:
    expression = _normalize_expression(expression)
    try:
        tree = _parser.parse(expression)
    except UnexpectedInput as exc:
        return ValidationResult(valid=False, error=str(exc))
    except Exception as exc:  # any other Lark/grammar error
        return ValidationResult(valid=False, error=f"Grammar error: {exc}")

    unknown = _find_unknown_functions(tree)
    if unknown:
        return ValidationResult(
            valid=False,
            error=f"Unknown function(s) not in the 4X library: {', '.join(sorted(unknown))}",
        )
    return ValidationResult(valid=True)


def _find_unknown_functions(tree) -> set[str]:
    unknown = set()
    for node in tree.find_data("function_call"):
        func_name = str(node.children[0])
        if func_name.upper() not in KNOWN_FUNCTIONS:
            unknown.add(func_name)
    return unknown
