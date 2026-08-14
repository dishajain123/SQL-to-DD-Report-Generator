"""Semantic Validation -- runs after deterministic grammar validation and
before a generated 4X Formula Expression is accepted. Grammar validation
only proves the expression is syntactically well-formed; it says nothing
about whether the expression actually reflects the source SQL. This module
closes that gap with checks that are all generalized (they operate on
whatever expression/column/source text is passed in -- nothing here is
specific to any one procedure or column):

- self-reference: the expression must not reference the very column it is
  computing.
- invented references: every quoted column-reference segment in the
  expression must actually appear somewhere in the source SQL.
- invented numeric literals: every non-trivial number in the expression
  must actually appear somewhere in the source SQL.
- dropped override/exception conditions: if the column is assigned via an
  override (a MERGE) or an exception handler in addition to its main
  calculation, the expression must show some trace of that too.
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
    "ISEMPTY", "ISNOTEMPTY", "MAX", "COALESCE", "SUBSTR", "LOWER", "UPPER",
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

_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_])")
_QUOTED_SEGMENT_RE = re.compile(r'"([^"]+)"')


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


def check_self_reference(expression: str, column: str) -> list[str]:
    """A Formula Expression must not reference the very column it is
    computing -- there is no prior-period/running value available in this
    single-pass derivation model, so any such reference is either a
    fabricated circular formula or a copy-paste mistake."""
    if not column:
        return []
    if column.upper() in _extract_identifiers(expression):
        return [
            f"Expression references its own target column '{column}', which "
            "is never a valid source for a single-pass Formula Expression "
            "(it would be circular)."
        ]
    return []


def check_invented_references(expression: str, source_text: str, entity_name: str = "") -> list[str]:
    """Every quoted column-reference segment the expression uses must
    actually appear somewhere in the source SQL it was derived from --
    otherwise it's a hallucinated field name. The mapped entity name itself
    is exempt, since that's an intentional platform-side mapping and is not
    expected to appear literally in the source SQL."""
    errors: list[str] = []
    source_tokens = _extract_identifiers(source_text)
    entity_upper = entity_name.strip().upper()

    for quoted in _QUOTED_SEGMENT_RE.findall(expression):
        candidate = quoted.strip()
        if not candidate:
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


def check_dropped_override_conditions(expression: str, relevant_chunks: list[SmartChunk]) -> list[str]:
    """If the column is assigned via an override-style block (a MERGE
    statement) or an exception handler, in addition to its main
    calculation, the generated expression must show some trace of that
    override too. Scoped specifically to override/exception-style chunks
    (rather than every branch of an ordinary IF/CASE) to avoid flagging a
    normal multi-branch calculation that a single formula legitimately and
    correctly represents as one expression.
    """
    errors: list[str] = []
    if len(relevant_chunks) < 2:
        return errors

    expr_tokens = _extract_identifiers(expression)

    for chunk in relevant_chunks:
        raw_upper = chunk.raw_sql.upper()
        is_override_chunk = chunk.chunk_kind == "MERGE" or any(
            keyword in raw_upper for keyword in _OVERRIDE_SIGNAL_KEYWORDS
        )
        if not is_override_chunk:
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
    errors.extend(check_self_reference(expression, column))
    errors.extend(check_invented_references(expression, source_text, entity_name))
    errors.extend(check_invented_numeric_literals(expression, source_text))
    errors.extend(check_dropped_override_conditions(expression, relevant_chunks))

    return GuardrailResult(passed=not errors, errors=errors)