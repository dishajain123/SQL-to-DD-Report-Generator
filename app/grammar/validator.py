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

_INCOMPLETE_TRAILING_TOKENS = {
    "IF",
    "THEN",
    "ELSE",
    "ELSEIF",
    "AND",
    "OR",
    "NOT",
    "BETWEEN",
    "IN",
    "NOTIN",
    "ISEMPTY",
    "ISNOTEMPTY",
    "COALESCE",
    "CONCAT",
    "DATEDIFF",
    "TODATE",
    "DATEPART",
    "ROUND",
    "ABS",
    "FLOOR",
    "CEIL",
    "MAX",
    "MIN",
    "LEN",
    "SUBSTR",
    "LOWER",
    "UPPER",
    "TRIM",
    "REPLACE",
    "REGEX",
    "SOM",
    "EOM",
    "SOY",
    "EOY",
    "SOFY",
    "EOFY",
    "PERIOD",
    "SOQ",
    "EOQ",
    "DATE",
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


def _rewrite_postfix_isnotempty(expression: str) -> str:
    pattern = re.compile(
        r'(?i)(?<![A-Za-z0-9_"])((?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)\s*(?:\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)\s+ISNOTEMPTY\b(?!\s*\()'
    )
    return pattern.sub(lambda m: f"ISNOTEMPTY({m.group(1).strip()})", expression)


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


def _rewrite_sql_date_literals(expression: str) -> str:
    """Rewrite SQL-style date literals into the 4X `TODATE(...)` form."""
    return re.sub(
        r'(?i)\bDATE\s*["\']([^"\']+)["\']',
        lambda match: f'TODATE("{match.group(1).strip()}")',
        expression,
    )


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


def _rewrite_in_subquery_membership(expression: str) -> str:
    """Rewrite a single-row `IN [value WHERE predicate]` subquery shape.

    SQL-to-4X translations sometimes compress an `IN (SELECT ... FROM ...
    WHERE ...)` predicate into a bracketed membership test with an inline
    `WHERE` clause. That is not valid 4X syntax, but when the bracket
    contains exactly one candidate value, the intent is usually a direct
    equality check guarded by the subquery predicate.
    """
    pattern = re.compile(
        r'(?i)\b(?P<lhs>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)'
        r'\s+IN\s*\[\s*(?P<rhs>(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*)(?:\s*\.\s*(?:"[^"]+"|[A-Za-z_][A-Za-z0-9_]*))*)'
        r'\s+WHERE\s+(?P<predicate>[^\]]+?)\s*\]'
    )
    return pattern.sub(lambda m: f'{m.group("lhs")} == {m.group("rhs")} AND ({m.group("predicate").strip()})', expression)


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
                if m >= n:
                    break
                if expression[m] == '"':
                    p = m + 1
                    while p < n and expression[p] != '"':
                        p += 1
                    if p >= n:
                        break
                    parts.append(expression[m + 1 : p])
                    k = p + 1
                    continue
                if not (expression[m].isalpha() or expression[m] == "_"):
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


def _rewrite_bundled_business_date_var(expression: str) -> str:
    """Normalize a fused business-date variable token into the platform's
    conventional two-segment path.

    Some model outputs collapse `"Entity"."var"."BUSINESS_DATE"` into a
    single quoted identifier such as `"Entity"."var_BUSINESS_DATE"`.
    That token is grammatically legal but it loses the intended platform
    convention and can be misread as an invented field. Rewriting it back
    to the conventional path keeps the generated expression aligned with
    the prompt guidance and the semantic validator.
    """
    return re.sub(
        r'("?[A-Za-z_][A-Za-z0-9_]*"?)\s*\.\s*"var_BUSINESS_DATE"',
        r'\1."var"."BUSINESS_DATE"',
        expression,
    )


def _strip_angle_bracket_placeholders(expression: str) -> str:
    """Remove placeholder angle brackets from identifier-like tokens."""
    expression = re.sub(r'"<([A-Za-z_][A-Za-z0-9_]*)>"', r'"\1"', expression)
    expression = re.sub(r'(?<![A-Za-z0-9_"])<([A-Za-z_][A-Za-z0-9_]*)>(?![A-Za-z0-9_"])', r"\1", expression)
    return expression


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


def _expression_looks_incomplete(expression: str) -> bool:
    """Detect obvious truncation before sending a string to the grammar.

    This is a lightweight, generic completeness check for the common
    failure modes we see from model output: empty strings, dangling
    operators, unterminated IF/THEN/ELSE branches, missing function
    arguments, and expressions that end while still inside a quoted
    string or an open delimiter.
    """
    stripped = expression.strip()
    if not stripped:
        return True

    in_double = False
    paren_depth = 0
    bracket_depth = 0
    last_token: str | None = None
    token: list[str] = []

    def flush_token() -> None:
        nonlocal last_token, token
        if token:
            last_token = "".join(token)
            token = []

    for ch in stripped:
        if ch == '"':
            if in_double:
                last_token = "STRING"
            else:
                flush_token()
            in_double = not in_double
            continue

        if in_double:
            continue

        if ch.isalnum() or ch == "_":
            token.append(ch)
            continue

        flush_token()
        if ch == "(":
            paren_depth += 1
            last_token = "("
        elif ch == ")":
            paren_depth -= 1
            last_token = ")"
        elif ch == "[":
            bracket_depth += 1
            last_token = "["
        elif ch == "]":
            bracket_depth -= 1
            last_token = "]"
        elif ch in {"+", "-", "*", "/", ",", ".", "<", ">", "=", "!"}:
            last_token = ch

    flush_token()
    if in_double or paren_depth > 0 or bracket_depth > 0:
        return True

    if last_token is None:
        return True

    return last_token.upper() in _INCOMPLETE_TRAILING_TOKENS or last_token in {"+", "-", "*", "/", ",", ".", "<", ">", "=", "!", "("}


def is_incomplete_expression_error(error: str | None) -> bool:
    """Return True when a validation error indicates truncation or emptiness."""
    if not error:
        return False
    lowered = error.lower()
    return "unexpected end-of-input" in lowered or "empty expression" in lowered


def _repair_missing_then_parentheses(expression: str) -> str:
    """Close an unterminated IF/ELSEIF condition immediately before THEN.

    The model sometimes emits shapes like `IF(cond THEN(...)` instead of
    `IF(cond)THEN(...)`. This is a structural typo rather than a logic
    issue, so we repair it mechanically before grammar validation.
    """

    upper = expression.upper()
    if "THEN" not in upper or ("IF(" not in upper and "ELSEIF(" not in upper):
        return expression

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

        token = None
        if expression[i : i + 7].upper() == "ELSEIF(" and (i == 0 or not (expression[i - 1].isalnum() or expression[i - 1] == "_")):
            token = "ELSEIF("
        elif expression[i : i + 3].upper() == "IF(" and (i == 0 or not (expression[i - 1].isalnum() or expression[i - 1] == "_")):
            token = "IF("

        if token:
            start = i + len(token)
            depth = 1
            j = start
            local_in_double = False
            repaired = False
            while j < n:
                cur = expression[j]
                if cur == '"':
                    local_in_double = not local_in_double
                elif not local_in_double:
                    if cur == "(":
                        depth += 1
                    elif cur == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    elif depth == 1 and expression[j : j + 4].upper() == "THEN":
                        before = expression[j - 1] if j > 0 else ""
                        after = expression[j + 4] if j + 4 < n else ""
                        if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                            # Condition was never closed before THEN --
                            # insert the missing ")" right here and resume
                            # scanning from the THEN token itself.
                            result.append(expression[i:j])
                            result.append(")")
                            i = j
                            repaired = True
                            break
                j += 1

            if repaired:
                # `i` was already advanced to the THEN position above --
                # resume the outer scan from there.
                continue

            # Either the condition closed normally (a well-formed
            # `IF(...)THEN(...)`, the overwhelmingly common case) or the
            # scan ran off the end of the string without finding anything
            # to repair. Either way there is nothing to do here -- fall
            # through and advance one character at a time as usual. (Not
            # jumping straight past the matched span keeps this branch
            # simple and correct; it is still linear overall since every
            # character is visited at most once by this fallthrough.)

        result.append(ch)
        i += 1

    return "".join(result)


def _normalize_expression(expression: str) -> str:
    normalized = _flatten_whitespace(expression)
    normalized = _rewrite_legacy_else_if(normalized)
    normalized = _rewrite_not_in_membership(normalized)
    normalized = _rewrite_is_empty_syntax(normalized)
    normalized = _rewrite_postfix_isnotempty(normalized)
    normalized = _rewrite_isnotempty_boolean_comparisons(normalized)
    normalized = _rewrite_null_predicates(normalized)
    normalized = _rewrite_sql_date_literals(normalized)
    normalized = _rewrite_date_function(normalized)
    normalized = _rewrite_unquoted_dotted_refs(normalized)
    normalized = _rewrite_bundled_business_date_var(normalized)
    normalized = _rewrite_exists_predicates(normalized)
    normalized = _rewrite_in_subquery_membership(normalized)
    normalized = _rewrite_legacy_if_syntax(normalized)
    normalized = _strip_angle_bracket_placeholders(normalized)
    normalized = _repair_missing_then_parentheses(normalized)
    return normalized


def validate_expression(expression: str) -> ValidationResult:
    expression = _normalize_expression(expression)
    if not expression.strip():
        return ValidationResult(valid=False, error="Empty expression")
    if _expression_looks_incomplete(expression):
        return ValidationResult(valid=False, error="Unexpected end-of-input")
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
