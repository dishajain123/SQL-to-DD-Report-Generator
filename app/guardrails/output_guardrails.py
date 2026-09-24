"""Architecture step 14: AI Output Guardrails.

Rule-based checks only (no second LLM call) — cheap, deterministic, and
catches a real class of errors: malformed grammar and expressions that
reference tables/entities never seen in the source structural analysis
(a concrete, checkable stand-in for "hallucinated" content). Confidence is
treated as advisory metadata rather than a hard failure so genuinely valid
rows are not pushed into review just because the upstream parser had a low
signal score.
"""
from __future__ import annotations

import re

from app.grammar.validator import KNOWN_FUNCTIONS, validate_expression
from app.guardrails.input_guardrails import GuardrailResult
from app.models.core import CanonicalModel, DDRow
from app.utils.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Every keyword/operator the platform grammar itself reserves (see
# fourx_grammar.lark's control-flow keywords and MEMBERSHIP_OP), plus NULL
# -- the grammar has no dedicated NULL-literal token, so `ELSE(NULL)` (used
# throughout composed expressions) only parses because a bare NULL happens
# to fall through to the same column_ref/NAME rule this check is looking
# for violations of. It is a reserved word here, not a real column.
_FOURX_RESERVED_WORDS = {
    "IF", "THEN", "ELSE", "ELSEIF", "AND", "OR", "NOT", "NULL",
    "IN", "NOTIN", "BETWEEN",
    "CONTAINS", "BEGINSWITH", "ENDSWITH", "DOESNOTCONTAINS",
    "HRCHYIN", "HRCHYNOTIN",
}
_KNOWN_4X_TOKENS = _FOURX_RESERVED_WORDS | KNOWN_FUNCTIONS
_QUOTED_SEGMENT_RE = re.compile(r'"[^"]*"')
_BARE_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def check_no_bare_identifiers(expression: str) -> list[str]:
    """Every column/entity reference in a 4X Formula Expression must be
    quoted ("Entity"."Column"); an identifier token outside all quoted
    segments that isn't a platform keyword or documented function name is
    an unqualified source column that leaked through composition -- the
    platform cannot resolve it, and the Lark grammar accepts a bare NAME
    syntactically so this never fails grammar validation on its own.
    """
    if not expression:
        return []
    outside_quotes = _QUOTED_SEGMENT_RE.sub(" ", expression)
    bare = {
        token for token in _BARE_IDENTIFIER_RE.findall(outside_quotes)
        if token.upper() not in _KNOWN_4X_TOKENS
        # A rule-versioning threshold parameter (T-SQL @TIMEKEY, Oracle
        # bare p_TIMEKEY) is written as its own bare, unquoted name by
        # documented platform convention (dd_generation.yaml) -- it is a
        # procedure parameter, not a column, and must never be flagged as
        # an unqualified reference. Matches the same TIMEKEY-substring
        # signal app/parsing/structural_analysis.py's version-threshold
        # detector uses to recognize this one documented exception.
        and "TIMEKEY" not in token.upper()
    }
    return [
        f'Unqualified column reference "{token}" -- must be "Entity"."Column".'
        for token in sorted(bare)
    ]


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


def _find_all_if_guards(expression: str) -> list[str]:
    """Every guard substring inside the expression's IF(...)/ELSEIF(...)
    headers, at any nesting depth."""
    guards = []
    for m in re.finditer(r"(?i)\b(?:IF|ELSEIF)\(", expression or ""):
        open_idx = m.end() - 1
        close_idx = _find_matching_paren(expression, open_idx)
        if close_idx >= 0 and close_idx > open_idx:
            guards.append(expression[open_idx + 1 : close_idx])
    return guards


def _split_top_level_and_conjuncts(guard: str) -> list[str]:
    """Split a guard on top-level ` AND ` -- not inside parens or quoted
    string literals -- into its individual conjuncts."""
    text = guard or ""
    parts: list[str] = []
    depth = 0
    in_quotes = False
    start = 0
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            in_quotes = not in_quotes
            i += 1
            continue
        if in_quotes:
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif depth == 0 and text[i : i + 5].upper() == " AND ":
            parts.append(text[start:i])
            i += 5
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [p.strip() for p in parts if p.strip()]


_ISEMPTY_RE = re.compile(r"(?is)^ISEMPTY\((?P<arg>.+)\)$")
_ISNOTEMPTY_RE = re.compile(r"(?is)^ISNOTEMPTY\((?P<arg>.+)\)$")
_EQUALITY_LITERAL_RE = re.compile(r'(?is)^(?P<lhs>.+?)\s*==\s*(?P<rhs>"[^"]*"|-?\d+(?:\.\d+)?)$')


def _normalize_conjunct(text: str) -> str:
    return re.sub(r"\s+", "", text or "").upper()


def check_contradictory_guard_conjuncts(expression: str) -> list[str]:
    """A guard that ANDs together ISEMPTY(x) and ISNOTEMPTY(x) for the same
    x, or `x == lit1 AND x == lit2` for two distinct literals, is provably
    false -- the branch it guards can never be reached. This only arises
    when two different assignment sites' guards get AND-merged into one
    predicate instead of being composed as separate, ordered branches, so
    it signals a composition bug rather than a genuine (if odd) source
    condition.
    """
    if not expression:
        return []
    errors: list[str] = []
    for guard in _find_all_if_guards(expression):
        conjuncts = _split_top_level_and_conjuncts(guard)
        if len(conjuncts) < 2:
            continue
        empty_args: dict[str, str] = {}
        notempty_args: dict[str, str] = {}
        equalities: dict[str, dict[str, str]] = {}
        for conjunct in conjuncts:
            m = _ISEMPTY_RE.match(conjunct)
            if m:
                empty_args[_normalize_conjunct(m.group("arg"))] = conjunct
                continue
            m = _ISNOTEMPTY_RE.match(conjunct)
            if m:
                notempty_args[_normalize_conjunct(m.group("arg"))] = conjunct
                continue
            m = _EQUALITY_LITERAL_RE.match(conjunct)
            if m:
                key = _normalize_conjunct(m.group("lhs"))
                equalities.setdefault(key, {})[m.group("rhs").strip()] = conjunct

        for key in set(empty_args) & set(notempty_args):
            errors.append(
                f"Contradictory guard produced by branch merge: "
                f"\"{empty_args[key]}\" AND \"{notempty_args[key]}\" can never both be "
                "true, so this branch is unreachable. Automated composition likely "
                "AND-merged two different assignment sites' guards instead of keeping "
                "them as separate branches -- do not approve without checking the "
                "source statements this formula was composed from."
            )
        for key, values in equalities.items():
            if len(values) > 1:
                texts = sorted(values.values())
                errors.append(
                    "Contradictory guard produced by branch merge: "
                    f"{' AND '.join(texts)} compare the same reference to mutually "
                    "exclusive literal values, so this branch is unreachable. "
                    "Automated composition likely AND-merged two different assignment "
                    "sites' guards instead of keeping them as separate branches -- do "
                    "not approve without checking the source statements this formula "
                    "was composed from."
                )
    return errors


def check_dd_row(dd_row: DDRow, canonical_model: CanonicalModel) -> GuardrailResult:
    errors = []
    if dd_row.confidence < settings.output_guardrail_confidence_threshold:
        logger.info(
            "DD row confidence %.3f below advisory threshold %.3f for %s.%s",
            dd_row.confidence,
            settings.output_guardrail_confidence_threshold,
            dd_row.entity_name,
            dd_row.column_name,
        )

    if dd_row.display_derivation_expression:
        result = validate_expression(dd_row.display_derivation_expression)
        if not result.valid:
            errors.append(f"Grammar validation failed: {result.error}")
        errors.extend(check_no_bare_identifiers(dd_row.display_derivation_expression))
        errors.extend(check_contradictory_guard_conjuncts(dd_row.display_derivation_expression))

    if dd_row.derivation_option.value == "Decision Table" and not dd_row.decision_table_json:
        errors.append("Decision Table derivation option chosen but decision_table_json is empty")

    if not dd_row.entity_name or not dd_row.column_name:
        errors.append("entity_name and column_name are both required")

    # Entity names in the DD layer are often mapped business targets
    # (for example, a staging table row may generate a fact-table DD
    # entity), so we only treat missing evidence as informational. The
    # stricter source-level checks remain in the semantic validator.
    if canonical_model.evidence and dd_row.entity_name:
        evidence_text = " ".join(canonical_model.evidence).lower()
        if dd_row.entity_name.lower() not in evidence_text:
            logger.info(
                "DD row entity %s is not present in the canonical evidence for this chain",
                dd_row.entity_name,
            )

    return GuardrailResult(passed=not errors, errors=errors)
