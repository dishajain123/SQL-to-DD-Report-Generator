"""Semantic Validation -- runs after deterministic grammar validation and
before a generated 4X Formula Expression is accepted. Grammar validation
only proves the expression is syntactically well-formed; it says nothing
about whether the expression actually reflects the source SQL. This module
closes that gap with checks that are all generalized (they operate on
whatever expression/column/source text is passed in -- nothing here is
specific to any one procedure or column):

- self-reference: the expression must not reference the very column it is
  computing.
- invented references: every column-reference segment in the expression
  (quoted or bare) must actually appear somewhere in the source SQL.
- invented numeric literals: every non-trivial number in the expression
  must actually appear somewhere in the source SQL.
- dropped override/exception conditions: if the column is assigned via an
  override (a MERGE) or an exception handler in addition to its main
  calculation, the expression must show some trace of that too.
- dropped row-scoping WHERE guard: if the column is assigned via a plain
  UPDATE whose WHERE clause scopes it to specific rows (not the target
  column's own value), the expression must show some trace of that too.
"""
from __future__ import annotations

import re

from app.guardrails.input_guardrails import GuardrailResult
from app.models.core import SmartChunk

_KEYWORD_STOPWORDS = {
    "IF", "THEN", "ELSE", "ELSEIF", "AND", "OR", "NOT", "BETWEEN", "IN",
    "NOTIN", "HRCHYIN", "HRCHYNOTIN", "CONTAINS", "BEGINSWITH", "ENDSWITH",
    "DOESNOTCONTAINS", "NULL", "IS", "WHEN", "CASE", "END", "UPDATE", "SET",
    "WHERE", "MERGE", "INTO", "USING", "ON", "MATCHED", "SELECT", "FROM",
    "GROUP", "BY", "UNION", "ALL", "AS", "EXCEPTION", "OTHERS", "BEGIN",
    "DECLARE", "LOOP", "ELSIF",
    # Known 4X function names -- these are called as bare FUNC_NAME(...) in
    # the grammar, never quoted, but excluding them here too costs nothing
    # and avoids any edge-case false positive if one shows up unquoted.
    "ISEMPTY", "ISNOTEMPTY", "MAX", "MIN", "COALESCE", "SUBSTR", "LOWER", "UPPER",
    "LEN", "CONVERT", "REGEX", "CONCAT", "TRIM", "REPLACE", "SOM", "EOM",
    "SOY", "EOY", "SOFY", "EOFY", "DATEPART", "DATEDIFF", "TODATE",
    "ADDDAY", "PERIOD", "SOQ", "EOQ", "ROUND", "ABS", "FLOOR", "CEIL",
}

# Platform-intrinsic symbols/literal values that legitimately appear in a
# Formula Expression without ever appearing in the source SQL text -- the
# platform's own pseudo-path segments (e.g. "var"."BUSINESS_DATE"), common
# date-unit literal arguments, and common boolean-ish output literals.
_KNOWN_4X_SYMBOLS = {
    "VAR", "BUSINESS_DATE",
    "D", "DAY", "DAYS", "M", "MONTH", "MONTHS", "Y", "YEAR", "YEARS",
    "H", "HOUR", "HOURS", "MI", "MINUTE", "MINUTES", "S", "SECOND", "SECONDS",
    "YES", "NO", "TRUE", "FALSE",
}

_OVERRIDE_SIGNAL_KEYWORDS = ("MERGE", "EXCEPTION", "WHEN OTHERS")

# A plain (non-MERGE) UPDATE's WHERE clause, e.g.
# "WHERE RUNNINGPROCESSNAME = 'DPD_Calculation'". Matches up to the
# statement's own terminating semicolon (or end of text) so it never
# spills into a following statement.
_WHERE_CLAUSE_RE = re.compile(r"(?is)\bWHERE\b(.+?)(?:;|$)")

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_])")
_QUOTED_SEGMENT_RE = re.compile(r'"([^"]+)"')
_LITERAL_LIKE_QUOTED_VALUES = {"Y", "N", "YES", "NO", "TRUE", "FALSE", "NULL"}
_DATE_LITERAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}")

# Bare (unquoted) column_ref NAME tokens, per the grammar's
# `column_ref: STRING ("." STRING)* | NAME` -- the quoted form is not the
# only legal reference shape, so hallucination checking must cover both,
# or a fabricated bare identifier (e.g. a made-up flag name standing in
# for "an exception occurred") slips through undetected.
_BARE_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_LITERAL_COMPARISON_RE = re.compile(
    r'"([^"]*)"\s*(==|!=|<=|>=|<|>)\s*"([^"]*)"'
)
_NUMBER_LITERAL_COMPARISON_RE = re.compile(
    r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)\s*(==|!=|<=|>=|<|>)\s*(\d+(?:\.\d+)?)(?![A-Za-z0-9_])"
)


def _is_function_call_name(text: str, match: re.Match[str]) -> bool:
    """True if the identifier `match` is immediately followed by '(' --
    i.e. it's a function call name, not a column reference -- so function
    names (COALESCE, ISNOTEMPTY, a documented helper, etc.) are never
    mistaken for invented columns."""
    j = match.end()
    while j < len(text) and text[j] == " ":
        j += 1
    return j < len(text) and text[j] == "("


def _looks_like_identifier_reference(candidate: str) -> bool:
    candidate = candidate.strip()
    if not candidate:
        return False
    if candidate.upper() in _LITERAL_LIKE_QUOTED_VALUES:
        return False
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", candidate):
        return False
    return True


def _is_literal_like_quoted_value(candidate: str) -> bool:
    """Return True when a quoted token is clearly being used as a literal
    value rather than a field reference.

    The 4X grammar uses `STRING` for quoted column references, so the
    semantic validator needs a lightweight heuristic to tell
    field-vs-literal comparisons apart. We only want to flag comparisons
    where both sides are obviously literal values (for example
    `"N"=="Y"` or `""=="ODA"`), not normal business rules like
    `"Aqua_Scheme"=="Y"` or `"SchemeType"=="ODA"`.
    """
    token = candidate.strip()
    if token == "":
        return True
    if token.upper() in _LITERAL_LIKE_QUOTED_VALUES:
        return True
    if _DATE_LITERAL_RE.fullmatch(token):
        return True
    if _NUMBER_RE.fullmatch(token):
        return True
    # Short all-caps codes are overwhelmingly used as literal values in
    # the DD formulas we validate here (e.g. Y/N/ODA/STD). Longer mixed-
    # case or underscore-bearing names are much more likely to be actual
    # source fields, so we leave those alone.
    if token.upper() == token and token.isalpha() and len(token) <= 4:
        return True
    return False


def _source_allows_target_reference(source_sql: str, column: str) -> bool:
    """Allow self-looking references only when the source SQL itself
    clearly describes a preservation or increment pattern.

    Some update statements intentionally keep the prior value when the new
    source value is missing, or increment a running counter. Those are not
    hallucinations; they are direct translations of the source logic.
    """
    if not source_sql or not column:
        return False

    source_upper = source_sql.upper()
    column_upper = re.escape(column.upper())

    preservation_patterns = (
        rf"\bNVL\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?\"?{column_upper}\"?\s*,",
        rf"\bCOALESCE\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?\"?{column_upper}\"?\s*,",
    )
    if any(re.search(pattern, source_upper) for pattern in preservation_patterns):
        return True

    if column_upper == "COUNT" and re.search(
        rf"\bNVL\s*\(\s*\"?{column_upper}\"?\s*,\s*0\s*\)\s*\+\s*1",
        source_upper,
    ):
        return True

    # Cleanup/update-style DD rows often rewrite the same target column in
    # a set-based statement (for example, "SET Col = NULL WHERE Col = ...").
    # In that case, a self-reference in the generated formula is not a
    # hallucination -- it is the source column being cleaned or preserved.
    if re.search(rf"\b(?:SET|UPDATE\s+SET)\b.*\b{column_upper}\s*=", source_upper, re.S):
        return True

    return False


def _extract_identifiers(text: str) -> set[str]:
    return {tok.upper() for tok in _IDENTIFIER_RE.findall(text) if tok.upper() not in _KEYWORD_STOPWORDS}


def _strip_sql_comments(text: str) -> str:
    """Strip `--` line comments and `/* */` block comments throughout the
    text (not just leading ones), so a hallucination check never treats an
    identifier that only appears inside commented-out/dead SQL as
    legitimate source evidence. This is a generic, always-applied pass --
    a stray word inside a `/* ... */` remark shouldn't make an unrelated,
    wrongly-attributed reference look justified for any input."""
    result: list[str] = []
    i = 0
    n = len(text)
    in_single = False
    in_double = False
    while i < n:
        ch = text[i]
        if in_single:
            result.append(ch)
            if ch == "'" and not (i + 1 < n and text[i + 1] == "'"):
                in_single = False
            i += 1
            continue
        if in_double:
            result.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            result.append(ch)
            i += 1
            continue
        if ch == '"':
            in_double = True
            result.append(ch)
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            nl = text.find("\n", i)
            i = n if nl == -1 else nl
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def check_self_reference(
    expression: str,
    column: str,
    entity_name: str = "",
    source_sql: str = "",
) -> list[str]:
    """A Formula Expression must not reference the very column it is
    computing -- there is no prior-period/running value available in this
    single-pass derivation model, so any such reference is either a
    fabricated circular formula or a copy-paste mistake."""
    if not column:
        return []
    column_upper = column.upper()
    expression_upper = expression.upper()
    expression_without_quotes = _QUOTED_SEGMENT_RE.sub(" ", expression_upper)

    # Only flag the truly circular cases: the target column referenced by
    # itself, either as a bare column name or qualified with the same
    # entity/table that is being derived. References to a different source
    # table's column with the same name are legitimate and should not be
    # rejected.
    bare_quoted = re.search(rf'(?<!\.)"{re.escape(column_upper)}"(?!\s*\.)', expression_upper)
    bare_unquoted = re.search(rf'(?<![A-Z0-9_]){re.escape(column_upper)}(?![A-Z0-9_])', expression_without_quotes)
    qualified_self = bool(
        entity_name
        and re.search(
            rf'"{re.escape(entity_name.upper())}"\s*\.\s*"{re.escape(column_upper)}"',
            expression_upper,
        )
    )

    if bare_quoted or bare_unquoted or qualified_self:
        if _source_allows_target_reference(source_sql, column):
            return []
        return [
            f"Expression references its own target column '{column}', which "
            "is never a valid source for a single-pass Formula Expression "
            "(it would be circular)."
        ]
    return []


def check_invented_references(expression: str, source_text: str, entity_name: str = "") -> list[str]:
    """Every column-reference segment the expression uses -- quoted
    ("Entity"."Column") or bare (a plain NAME, also a valid column_ref per
    the grammar) -- must actually appear somewhere in the source SQL it was
    derived from, otherwise it's a hallucinated field name. The mapped
    entity name itself is exempt, since that's an intentional platform-side
    mapping and is not expected to appear literally in the source SQL."""
    errors: list[str] = []
    source_tokens = _extract_identifiers(source_text)
    entity_upper = entity_name.strip().upper()

    for match in _QUOTED_SEGMENT_RE.finditer(expression):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        if not _looks_like_identifier_reference(candidate):
            continue
        tail = expression[match.end() :].lstrip()
        if tail.startswith("."):
            # Namespace/table aliases in expressions are often mapped
            # targets rather than literal source identifiers, so only the
            # final path segment is checked against the source text.
            continue
        candidate_upper = candidate.upper()
        if candidate_upper in _KEYWORD_STOPWORDS or candidate_upper in _KNOWN_4X_SYMBOLS:
            continue
        if entity_upper and candidate_upper == entity_upper:
            continue
        if candidate_upper not in source_tokens:
            errors.append(
                f'Expression references "{candidate}", which does not appear '
                "anywhere in the source SQL for this column."
            )

    # Bare identifiers: the grammar's column_ref also permits an unquoted
    # NAME, so a fabricated bare token (never wrapped in quotes) must be
    # checked the same way, or it silently bypasses hallucination
    # detection entirely.
    expression_no_quotes = _QUOTED_SEGMENT_RE.sub(lambda m: " " * (m.end() - m.start()), expression)
    for match in _BARE_IDENTIFIER_RE.finditer(expression_no_quotes):
        candidate = match.group(0)
        candidate_upper = candidate.upper()
        if candidate_upper in _KEYWORD_STOPWORDS or candidate_upper in _KNOWN_4X_SYMBOLS:
            continue
        if _is_function_call_name(expression_no_quotes, match):
            continue
        if entity_upper and candidate_upper == entity_upper:
            continue
        if candidate_upper not in source_tokens:
            errors.append(
                f'Expression references "{candidate}", which does not appear '
                "anywhere in the source SQL for this column."
            )
    return errors


def check_invented_numeric_literals(expression: str, source_text: str) -> list[str]:
    """Any numeric literal beyond the trivial 0/1 must appear somewhere in
    the source SQL -- otherwise it's an invented threshold. 0 and 1 are
    exempt since they are near-universal defaults/offsets that legitimately
    appear even when the exact literal isn't textually present."""
    errors: list[str] = []
    source_numbers = set(_NUMBER_RE.findall(source_text))

    for number in set(_NUMBER_RE.findall(expression)):
        if number in ("0", "1"):
            continue
        if number not in source_numbers:
            errors.append(
                f"Expression uses the literal value {number}, which does not "
                "appear anywhere in the source SQL for this column."
            )
    return errors


def _extract_where_clause(raw_sql: str) -> str | None:
    text = raw_sql
    upper = text.upper()
    in_single = False
    in_double = False
    depth = 0
    i = 0
    while i < len(text):
        ch = text[i]
        if in_single:
            if ch == "'" and not (i + 1 < len(text) and text[i + 1] == "'"):
                in_single = False
            i += 1
            continue
        if in_double:
            if ch == '"':
                in_double = False
            i += 1
            continue
        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
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
        if depth == 0 and upper[i : i + 5] == "WHERE" and (
            i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
        ) and (
            i + 5 >= len(text) or not (text[i + 5].isalnum() or text[i + 5] == "_")
        ):
            clause = text[i + 5 :].strip()
            return clause or None
        i += 1
    return None


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    in_double = False
    for idx in range(open_index, len(text)):
        ch = text[idx]
        if ch == '"':
            in_double = not in_double
            continue
        if in_double:
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _extract_simple_if_branches(expression: str) -> tuple[str, str] | None:
    """Extract the outermost THEN/ELSE branches when the expression starts
    with a plain IF(... )THEN(... )ELSE(... ) chain.

    This is intentionally conservative: it only handles the simple shape
    that is most useful for spotting obviously collapsed branches such as
    `THEN(NULL)ELSE(NULL)` or other identical branch bodies. More complex
    ELSEIF chains are left to the other semantic checks.
    """
    text = expression.strip()
    if not text.upper().startswith("IF(") or "ELSEIF(" in text.upper():
        return None

    condition_close = _find_matching_paren(text, 2)
    if condition_close == -1:
        return None

    then_index = condition_close + 1
    while then_index < len(text) and text[then_index].isspace():
        then_index += 1
    if not text[then_index:].upper().startswith("THEN("):
        return None

    then_body_start = then_index + len("THEN(")
    depth = 0
    in_double = False
    for idx in range(then_body_start, len(text)):
        ch = text[idx]
        if ch == '"':
            in_double = not in_double
            continue
        if in_double:
            continue
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            if depth > 0:
                depth -= 1
            continue
        if depth == 0 and text[idx : idx + 5].upper() == "ELSE(":
            then_body = text[then_body_start:idx].strip()
            else_close = _find_matching_paren(text, idx + 4)
            if else_close == -1:
                return None
            else_body = text[idx + 5 : else_close].strip()
            return then_body, else_body
    return None


def _check_constant_conditions(expression: str) -> list[str]:
    errors: list[str] = []
    for left, op, right in _STRING_LITERAL_COMPARISON_RE.findall(expression):
        if _is_literal_like_quoted_value(left) and _is_literal_like_quoted_value(right):
            errors.append(
                f"Semantic validation: condition compares only literals ({left!r} {op} {right!r}); "
                "this branch does not depend on source data."
            )
    for left, op, right in _NUMBER_LITERAL_COMPARISON_RE.findall(expression):
        errors.append(
            f"Semantic validation: condition compares only numeric literals ({left} {op} {right}); "
            "this branch does not depend on source data."
        )
    return errors


def _check_identical_simple_branches(expression: str) -> list[str]:
    match = re.match(
        r'(?is)^IF\((?P<cond>.*)\)THEN\(\s*(?P<branch>[^()]*)\s*\)ELSE\(\s*(?P=branch)\s*\)$',
        expression.strip(),
    )
    if match:
        return [
            "Semantic validation: THEN and ELSE branches resolve to the same value, "
            "so the condition has no effect and likely dropped source logic."
        ]
    return []


def check_dropped_override_conditions(
    expression: str, relevant_chunks: list[SmartChunk], column: str = ""
) -> list[str]:
    """If the column is assigned via an override-style block (a MERGE
    statement) or an exception handler, in addition to its main
    calculation, the generated expression must show some trace of that
    override too. Scoped specifically to override/exception-style chunks
    (rather than every branch of an ordinary IF/CASE) to avoid flagging a
    normal multi-branch calculation that a single formula legitimately and
    correctly represents as one expression.

    Separately, a plain (non-MERGE) UPDATE's own WHERE clause is checked
    when it does not reference the target column itself -- that shape is a
    row-scoping guard (for example `WHERE RUNNINGPROCESSNAME =
    'DPD_Calculation'`, restricting the statement to one specific
    process's row in a table shared by several procedures), not a
    self-referential cleanup/normalize filter like `WHERE Col =
    DATE'1900-01-01'` (which is already handled by the self-reference
    preservation logic elsewhere and is not flagged here). If the
    expression shows no trace of that scoping condition, it would as
    written apply to every row of the entity rather than just the one the
    source SQL actually updates, which is exactly the kind of dropped
    guard this check exists to catch.
    """
    errors: list[str] = []
    expr_tokens = _extract_identifiers(expression)
    column_upper = column.strip().upper()

    for chunk in relevant_chunks:
        raw_upper = chunk.raw_sql.upper()
        is_override_chunk = chunk.chunk_kind == "MERGE" or any(
            keyword in raw_upper for keyword in _OVERRIDE_SIGNAL_KEYWORDS
        )

        if is_override_chunk:
            if len(relevant_chunks) < 2:
                continue
            chunk_tokens: set[str] = set()
            for condition in chunk.conditions:
                chunk_tokens |= _extract_identifiers(condition)
            chunk_tokens |= _extract_identifiers(chunk.raw_sql)

            if not chunk_tokens or (chunk_tokens & expr_tokens):
                continue

            snippet = next((line.strip() for line in chunk.raw_sql.splitlines() if line.strip()), "")[:120]
            errors.append(
                "Expression does not appear to reflect an override/exception "
                f'statement found in the source for this column (starting: "{snippet}"). '
                "If this column is overridden by a special case or an "
                "error-handling path, the expression must combine that with the "
                "main calculation."
            )
            continue

        if chunk.chunk_kind == "MERGE":
            continue

        where_clause = _extract_where_clause(chunk.raw_sql)
        if not where_clause:
            continue

        where_tokens = _extract_identifiers(where_clause)
        if column_upper and column_upper in where_tokens:
            # References the target column itself -- a self-referential
            # cleanup/normalize filter, not a row-scoping guard.
            continue
        if not where_tokens or (where_tokens & expr_tokens):
            continue

        errors.append(
            "Expression does not appear to reflect a row-scoping WHERE "
            f'condition found in the source for this column (WHERE {where_clause[:120]}). '
            "If this statement only applies to specific rows (for example a "
            "specific process name or table identifier), the expression "
            "must include that same scoping condition so it does not "
            "over-apply to unrelated rows."
        )
    return errors


def check_semantic_consistency(
    expression: str,
    column: str,
    entity_name: str,
    relevant_chunks: list[SmartChunk],
    source_sql: str,
) -> GuardrailResult:
    """Run every semantic check and return a single combined result."""
    source_text = _strip_sql_comments(
        "\n".join(chunk.raw_sql for chunk in relevant_chunks) + "\n" + source_sql
    )

    errors: list[str] = []
    errors.extend(check_self_reference(expression, column, entity_name, source_sql))
    errors.extend(check_invented_references(expression, source_text, entity_name))
    errors.extend(check_invented_numeric_literals(expression, source_text))
    errors.extend(_check_constant_conditions(expression))
    errors.extend(_check_identical_simple_branches(expression))
    errors.extend(check_dropped_override_conditions(expression, relevant_chunks, column))

    return GuardrailResult(passed=not errors, errors=errors)
