"""Architecture step 13: DD Generation — chain collapse + grammar
targeting + versioning, orchestrated end to end.

Flow per column: build a column-specific SQL excerpt -> retrieve relevant
RAG context -> generate a Formula Expression -> deterministic grammar
validation -> semantic validation against the source SQL -> if either
fails, feed the errors (plus RAG context) back for a bounded number of
repair attempts -> accept -> for each effective-dated period, prune the
accepted expression down to just the branch that period's TIMEKEY
threshold actually selects (see app/derivation/period_pruning.py) -> or,
if generation never fully succeeded, fall back to PENDING_REVIEW with the
full, unpruned expression so a reviewer sees everything.

A column is very often assigned in more than one place in a real
procedure (a main calculation plus a special-case override, or a success
path plus an error-handling path). To make sure the generated derivation
reflects all of those assignment locations rather than just whichever one
the model happens to notice first in a long procedure, this module builds
a column-specific SQL excerpt from the object's SmartChunks (see
app/parsing/smart_chunking.py) -- every logical block that actually
assigns the target column, anywhere in the object -- and passes that to
the LLM as the authoritative source for that one column. The same chunk
list is also handed to semantic validation so it can check whether an
override/exception-style chunk was actually reflected in the result.

Entity-name resolution (staging table -> fact table name) is intentionally
pluggable via `entity_name_map` rather than hardcoded, since that mapping is
company/platform-specific and not something this pipeline can infer from
SQL alone.
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Optional

from app.derivation.llm_client import LLMClient
from app.derivation.period_pruning import prune_expression_for_period
from app.derivation.versioning import effective_periods_for_column
from app.grammar.validator import validate_expression
from app.guardrails.semantic_validation import check_semantic_consistency
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    LineageChain,
    SmartChunk,
    SQLObject,
    StructuralInfo,
)
from app.rag.chroma_store import ChromaStore, DOMAIN_COLLECTION, PLATFORM_COLLECTION
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Total generation attempts per column: one initial attempt plus one
# bounded repair/regeneration attempt. This keeps the pipeline responsive
# on large procedures while still giving the model a chance to fix a
# mechanically reported validation issue.
_MAX_GENERATION_ATTEMPTS = 2
_MAX_SOURCE_SQL_CONTEXT_CHARS = 5000

_SYNTHETIC_DATE_REVIEW_NOTE = (
    "Effective start date is a synthetic estimate because no "
    "TIMEKEY-to-calendar-date mapping was supplied for this run; confirm "
    "the exact date before finalizing this DD row."
)


def _relevant_chunks(info: StructuralInfo, column: str) -> list[SmartChunk]:
    """Every logical block (SmartChunk) that actually assigns the target
    column somewhere in the object -- across every conditional branch,
    MERGE override, or exception handler, not just wherever it happens to
    appear first.

    SmartChunks already keep control-flow blocks (IF/ELSE, CASE) together
    as one unit, and each chunk's `columns_written` is the union of every
    column actually assigned within it -- so filtering on that gives a
    focused, still-conditionally-correct set of chunks for one column,
    built the same way regardless of which procedure or column is being
    processed.

    A bare control-flow header (`EXCEPTION`, `WHEN ... THEN`, `ELSE`, ...)
    that touches no table/column of its own becomes its own tiny chunk
    immediately before the statement it governs (see
    app/parsing/smart_chunking.py's branch-marker handling), rather than
    being merged into it. Any such header(s) immediately preceding a
    matched chunk are folded into that chunk's text here, since they carry
    the trigger condition (e.g. "this is the error-handling path") that
    both the LLM and semantic validation need to see alongside the
    statement itself.
    """
    all_chunks = info.smart_chunks
    matched: list[SmartChunk] = []
    seen_chunk_ids: set[str] = set()

    for idx, chunk in enumerate(all_chunks):
        if column not in chunk.columns_written or chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk.chunk_id)

        header_lines: list[str] = []
        j = idx - 1
        while j >= 0:
            candidate = all_chunks[j]
            is_bare_header = (
                not candidate.tables_written and not candidate.tables_read and not candidate.columns_written
            )
            if not is_bare_header:
                break
            header_lines.insert(0, candidate.raw_sql.strip())
            j -= 1

        if header_lines:
            merged_raw_sql = "\n".join([*header_lines, chunk.raw_sql.strip()])
            chunk = chunk.model_copy(update={"raw_sql": merged_raw_sql})

        matched.append(chunk)

    return matched


def _relevant_sql_excerpt(info: StructuralInfo, column: str) -> str:
    """Text form of `_relevant_chunks`, for handing to the LLM prompt.
    Returns an empty string (letting the caller fall back to explaining
    that nothing specific was isolated) if smart chunking found nothing
    for this column -- this is a targeting aid, not a hard requirement.
    """
    excerpts = [chunk.raw_sql.strip() for chunk in _relevant_chunks(info, column) if chunk.raw_sql.strip()]
    return "\n\n".join(excerpts)


def _source_sql_context_excerpt(source_sql: str, relevant_sql: str) -> str:
    """Keep the model prompt focused by trimming the broad source SQL
    context to a bounded excerpt.

    The column-specific assignment chunks already carry the important
    logic. The full procedure text is still useful for surrounding context,
    but sending every line of a large stored procedure to the provider for
    every column makes generation noticeably slower.
    """
    source_sql = source_sql.strip()
    relevant_sql = relevant_sql.strip()

    if not source_sql:
        return relevant_sql

    if len(source_sql) <= _MAX_SOURCE_SQL_CONTEXT_CHARS:
        return source_sql

    head_chars = max(1200, _MAX_SOURCE_SQL_CONTEXT_CHARS // 3)
    tail_chars = max(1200, _MAX_SOURCE_SQL_CONTEXT_CHARS // 3)
    head = source_sql[:head_chars].strip()
    tail = source_sql[-tail_chars:].strip()

    sections = []
    if relevant_sql:
        sections.append(relevant_sql)
    if head:
        sections.append("[Source SQL excerpt - beginning]\n" + head)
    if tail and tail != head:
        sections.append("[Source SQL excerpt - end]\n" + tail)
    return "\n\n".join(sections)


def _retrieve_rag_context(
    rag_store: Optional[ChromaStore],
    relevant_sql: str,
    technical_summary: str,
    business_summary: str,
) -> str:
    """Query the platform (4X function/operator) and domain RAG
    collections for the chunks most relevant to this specific column,
    instead of handing the model the entire reference document every time.

    Falls back to an empty string -- letting the caller rely on the full
    function_reference instead -- if no RAG store was supplied, the store
    can't be reached, or nothing has been ingested yet. This keeps the
    pipeline fully functional whether or not `ingest_platform_doc` /
    `ingest_domain_doc` has ever been run; RAG is a targeting aid on top of
    the existing full-reference behavior, not a replacement that could
    break generation if it's unavailable.
    """
    if rag_store is None:
        return ""

    platform_query = (relevant_sql or technical_summary).strip()
    domain_query = (business_summary or technical_summary).strip()

    sections: list[str] = []
    try:
        if platform_query:
            platform_hits = rag_store.query(PLATFORM_COLLECTION, platform_query, n_results=4)
            if platform_hits:
                sections.append(
                    "Relevant platform function/operator reference:\n" + "\n---\n".join(platform_hits)
                )
        if domain_query:
            domain_hits = rag_store.query(DOMAIN_COLLECTION, domain_query, n_results=2)
            if domain_hits:
                sections.append("Relevant domain glossary:\n" + "\n---\n".join(domain_hits))
    except Exception as exc:  # pragma: no cover - defensive: RAG must never break generation
        logger.warning("RAG retrieval failed, continuing without it: %s", exc)
        return ""

    return "\n\n".join(sections)


def _flatten_whitespace(expression: str) -> str:
    """Collapse all internal whitespace (including newlines and
    indentation) into single spaces.

    The 4X grammar itself ignores whitespace entirely when parsing (see
    fourx_grammar.lark's `%ignore WS`), so this never changes what an
    expression means -- it only guarantees the stored/exported expression
    is always a single line. A multi-line value breaks a Markdown table
    row (the report renders every DD row as one table row) and makes a
    poor spreadsheet cell; applying this once here, at the source, keeps
    the Markdown report and the Excel export consistent with each other
    instead of patching the symptom separately in each renderer.
    """
    return " ".join(expression.split())


def _fix_unbalanced_trailing_parens(expression: str) -> str:
    """Fix the common LLM mistake of closing one (or a few) too many, or
    too few, parentheses at the very end of an otherwise-correct
    expression -- deeply nested IF/ELSEIF/ELSE trees make manual
    paren-counting error prone, and this is a purely mechanical, frequent
    failure mode, distinct from any actual logic error.

    Only ever trims or adds parentheses at the very end of the expression,
    and only accepts the result if it actually parses against the real 4X
    grammar (not merely paren-depth-balanced -- depth balance alone isn't
    proof of a correct token sequence). It never touches parentheses in
    the interior, so it can't silently change the expression's actual
    structure. If no simple trailing adjustment produces something that
    parses, the expression is left untouched and grammar validation will
    correctly reject it, triggering a normal LLM repair attempt instead.
    """

    def depth_profile(text: str) -> list[int]:
        depth = 0
        profile = []
        in_double = False
        for ch in text:
            if ch == '"':
                in_double = not in_double
            elif not in_double and ch == "(":
                depth += 1
            elif not in_double and ch == ")":
                depth -= 1
            profile.append(depth)
        return profile

    profile = depth_profile(expression)
    if not profile:
        return expression
    final_depth = profile[-1]
    if final_depth == 0:
        return expression

    if final_depth > 0:
        candidate = expression + (")" * final_depth)
        if validate_expression(candidate).valid:
            return candidate
        return expression

    excess = -final_depth
    trimmed = expression.rstrip()
    trailing_closes = 0
    i = len(trimmed) - 1
    while i >= 0 and trimmed[i] == ")" and trailing_closes < excess:
        trailing_closes += 1
        i -= 1
    if trailing_closes < excess:
        return expression

    candidate = trimmed[: len(trimmed) - trailing_closes]
    if validate_expression(candidate).valid:
        return candidate
    return expression


def _normalize_sql_functions(expression: str) -> str:
    """Rewrite common SQL-only helper functions that are not part of the 4X
    grammar's function library into their direct 4X equivalents, before
    grammar validation:

    - NVL(x, default) -> COALESCE(x, default) -- same two-argument shape,
      just a different name (NVL is Oracle-specific; COALESCE is what the
      4X function reference documents).
    - ISNULL(x) -> ISEMPTY(x) when used with a single argument (a common,
      if non-standard, null-check shorthand seen in SQL-Server-derived
      code); ISNULL(x, default) -> COALESCE(x, default) when used with two
      arguments (SQL Server's real ISNULL semantics).

    This is a mechanical, function-name/argument-count-based rewrite that
    applies the same way regardless of which input SQL produced the
    expression -- it is not specific to any one procedure or column.
    """

    def rewrite_calls(text: str, func_name: str, one_arg_target: str, two_arg_target: str) -> str:
        token = func_name + "("
        result: list[str] = []
        i = 0
        n = len(text)
        in_double = False
        while i < n:
            ch = text[i]

            if in_double:
                result.append(ch)
                if ch == '"':
                    in_double = False
                i += 1
                continue

            if ch == '"':
                in_double = True
                result.append(ch)
                i += 1
                continue

            if text[i : i + len(token)].upper() == token.upper() and (
                i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")
            ):
                start = i + len(token)
                depth = 1
                j = start
                local_in_double = False
                args: list[str] = []
                current: list[str] = []
                while j < n and depth > 0:
                    cur = text[j]
                    if cur == '"':
                        local_in_double = not local_in_double
                        current.append(cur)
                    elif not local_in_double and cur == "(":
                        depth += 1
                        current.append(cur)
                    elif not local_in_double and cur == ")":
                        depth -= 1
                        if depth == 0:
                            break
                        current.append(cur)
                    elif not local_in_double and cur == "," and depth == 1:
                        args.append("".join(current).strip())
                        current = []
                    else:
                        current.append(cur)
                    j += 1

                if depth == 0:
                    args.append("".join(current).strip())
                    target = one_arg_target if len(args) == 1 else two_arg_target
                    result.append(f"{target}({', '.join(args)})")
                    i = j + 1
                    continue

            result.append(ch)
            i += 1

        return "".join(result)

    expression = rewrite_calls(expression, "NVL", "COALESCE", "COALESCE")
    expression = rewrite_calls(expression, "ISNULL", "ISEMPTY", "COALESCE")
    return expression


def _find_matching_paren(text: str, open_index: int) -> int:
    """Find the matching closing parenthesis, ignoring quoted segments."""
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


def _rewrite_legacy_else_if(expression: str) -> str:
    return re.sub(r"(?i)\bELSE\s+IF\b", "ELSEIF", expression)


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


def _rewrite_date_function(expression: str) -> str:
    """Rewrite SQL-style DATE(...) wrappers into the documented 4X date
    constructor when the content is a single argument."""
    def replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        return f"TODATE({inner})"

    return re.sub(r"(?i)\bDATE\s*\(\s*([^()]+?)\s*\)", replace, expression)


def _rewrite_unquoted_dotted_refs(expression: str) -> str:
    """Quote bare dotted identifiers so the 4X grammar can parse them as
    column references.

    The source SQL often uses `table.column` or `alias.column` syntax, but
    the 4X grammar only accepts quoted reference segments. This pass keeps
    the semantic shape intact while making the output grammar-safe.
    """
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


def _strip_angle_bracket_placeholders(expression: str) -> str:
    """Remove literal `<...>` placeholder wrappers from identifiers.

    Prompt examples use placeholder notation like `<entity_name>`, and the
    model sometimes copies those angle brackets verbatim into output.
    This is never part of the actual 4X grammar, so stripping them is a
    safe mechanical cleanup as long as the bracketed text is an
    identifier-like token.
    """
    expression = re.sub(r'"<([A-Za-z_][A-Za-z0-9_]*)>"', r'"\1"', expression)
    expression = re.sub(r'(?<![A-Za-z0-9_"])<([A-Za-z_][A-Za-z0-9_]*)>(?![A-Za-z0-9_"])', r"\1", expression)
    return expression


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


def _rewrite_exists_predicates(expression: str) -> str:
    """Drop unsupported EXISTS wrappers but keep the predicate body."""
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
            close_index = _find_matching_paren(expression, open_index)
            if close_index != -1:
                inner = expression[i + 7 : close_index].strip()
                tail = expression[close_index + 1 :].lstrip()
                needs_if_close = tail.upper().startswith("THEN")
                suffix = ")" if needs_if_close else ""
                where_match = re.search(r"(?is)\bWHERE\b", inner)
                if where_match:
                    predicate = inner[where_match.end() :].strip()
                    result.append(predicate + suffix)
                else:
                    comma_match = re.search(r"(?s),", inner)
                    if comma_match:
                        predicate = inner[comma_match.end() :].strip()
                        result.append(predicate + suffix)
                    else:
                        result.append(inner + suffix)
                i = close_index + 1
                continue

        result.append(ch)
        i += 1
    return "".join(result)


def _strip_min_wrapper(expression: str) -> str:
    """Drop a single balanced `MIN(...)` wrapper when the inner payload is
    already a complete expression.

    The platform grammar/validator do not rely on `MIN` for the generated
    DD rows in this project, and the model sometimes emits `MIN(...)` as a
    SQL-shaped aggregate wrapper around a single conditional expression.
    Removing only the wrapper keeps the inner logic while avoiding an
    unnecessary parser failure.
    """
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

        if not in_double and expression[i : i + 4].upper() == "MIN(":
            open_index = i + 3
            close_index = _find_matching_paren(expression, open_index)
            if close_index != -1:
                result.append(expression[i + 4 : close_index].strip())
                i = close_index + 1
                continue

        result.append(ch)
        i += 1
    return "".join(result)


def _repair_missing_then_parentheses(expression: str) -> str:
    """Insert a missing `)` before THEN only when the IF/ELSEIF condition
    is still open at that point.

    This is intentionally conservative: if the condition already closes
    before THEN, the expression is left untouched.
    """

    if "THEN" not in expression.upper() or ("IF(" not in expression.upper() and "ELSEIF(" not in expression.upper()):
        return expression

    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False
    changed = False

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
                            result.append(expression[i:j])
                            result.append(")")
                            i = j
                            changed = True
                            break
                j += 1
            if changed:
                continue

        result.append(ch)
        i += 1

    repaired = "".join(result)
    return repaired if changed else repaired


def _repair_extra_close_before_then(expression: str) -> str:
    """Remove a spurious extra `)` that appears between an IF/ELSEIF
    condition and its `THEN(` keyword.

    The model sometimes emits `IF(cond))THEN(...)` or
    `ELSEIF(cond))THEN(...)`. That extra close cannot be valid because the
    condition's own closing parenthesis must be followed directly by
    `THEN`. This pass removes only that single extra close and leaves the
    surrounding branch structure untouched.
    """

    previous = expression
    for _ in range(3):
        result: list[str] = []
        i = 0
        n = len(previous)
        in_double = False
        changed = False
        while i < n:
            ch = previous[i]
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
            if previous[i : i + 7].upper() == "ELSEIF(" and (i == 0 or not (previous[i - 1].isalnum() or previous[i - 1] == "_")):
                token = "ELSEIF("
            elif previous[i : i + 3].upper() == "IF(" and (i == 0 or not (previous[i - 1].isalnum() or previous[i - 1] == "_")):
                token = "IF("

            if token:
                open_index = i + len(token) - 1
                close_index = _find_matching_paren(previous, open_index)
                if close_index != -1:
                    j = close_index + 1
                    while j < n and previous[j].isspace():
                        j += 1
                    if j < n and previous[j] == ")" and previous[j + 1 :].lstrip().startswith("THEN("):
                        result.append(previous[i : close_index + 1])
                        i = j + 1
                        changed = True
                        continue

            result.append(ch)
            i += 1

        repaired = "".join(result)
        if not changed:
            return repaired
        previous = repaired
    return previous


def _remove_excess_closing_parens(expression: str) -> str:
    """Drop unmatched closing parens while leaving quoted text alone."""
    result: list[str] = []
    depth = 0
    in_double = False
    for ch in expression:
        if ch == '"':
            in_double = not in_double
            result.append(ch)
            continue
        if in_double:
            result.append(ch)
            continue
        if ch == "(":
            depth += 1
            result.append(ch)
            continue
        if ch == ")":
            if depth == 0:
                continue
            depth -= 1
            result.append(ch)
            continue
        result.append(ch)
    return "".join(result)


def _normalize_legacy_if_syntax(expression: str) -> str:
    """Convert common comma-style IF(condition, true, false) output into 4X syntax.

    The 4X grammar expects IF(condition)THEN(true)ELSE(false). Some LLM
    outputs default to SQL-style IF(condition, true, false); this helper
    rewrites that shape before validation.
    """

    def split_top_level_args(text: str) -> list[str] | None:
        args = []
        current = []
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
        result = []
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


def _normalize_sql_style_syntax(expression: str) -> str:
    """Normalize common SQL-style syntax that is not valid 4X Formula
    Expression syntax, before grammar validation:

    - Single-quoted string literals ('Y') become double-quoted ("Y"),
      since the 4X grammar's STRING token only accepts double quotes.
    - A bare SQL-style equality operator (=) becomes the 4X equality
      operator (==), since the 4X grammar's COMP_OP only recognizes
      ==, !=, >=, <=, >, and <. Existing !=, <=, >=, and == are left
      untouched.

    This is a generic, input-independent syntax-shape fix: LLM output
    translating SQL conditions frequently defaults to SQL literal syntax
    even when explicitly told to use 4X grammar, and retrying the whole
    LLM call for a purely mechanical substitution like this is wasteful
    and unreliable. Content inside double-quoted strings is left alone so
    this never rewrites the literal text of a value.
    """
    result: list[str] = []
    i = 0
    n = len(expression)
    in_double = False

    while i < n:
        ch = expression[i]

        if in_double:
            result.append(ch)
            if ch == '"':
                in_double = False
            i += 1
            continue

        if ch == '"':
            in_double = True
            result.append(ch)
            i += 1
            continue

        if ch == "'":
            j = i + 1
            content = []
            while j < n and expression[j] != "'":
                content.append(expression[j])
                j += 1
            literal = "".join(content).replace('"', '\\"')
            result.append(f'"{literal}"')
            i = j + 1
            continue

        if ch == "=":
            next_char = expression[i + 1] if i + 1 < n else ""
            prev_char = expression[i - 1] if i > 0 else ""
            if next_char == "=":
                result.append("==")
                i += 2
                continue
            if prev_char in ("!", "<", ">"):
                result.append("=")
                i += 1
                continue
            result.append("==")
            i += 1
            continue

        result.append(ch)
        i += 1

    normalized = "".join(result)
    normalized = _rewrite_legacy_else_if(normalized)
    normalized = _rewrite_null_predicates(normalized)
    normalized = _rewrite_exists_predicates(normalized)
    return normalized


def _normalize_expression(expression: str) -> str:
    """Apply every mechanical, input-independent normalization pass, in an
    order chosen so each pass sees syntax the next one expects. Whitespace
    is flattened first (so every later pass works on a single line, and so
    the final result is always safe for both a Markdown table cell and a
    spreadsheet cell), then quotes/operators, then function-call rewriting
    and comma-style IF detection, both of which rely on string-boundary
    tracking, and finally a trailing-paren-balance check as a last safety
    net after all other rewrites have run."""
    expression = _flatten_whitespace(expression)
    expression = _normalize_sql_style_syntax(expression)
    expression = _normalize_sql_functions(expression)
    expression = _normalize_legacy_if_syntax(expression)
    expression = _strip_angle_bracket_placeholders(expression)
    expression = _rewrite_not_in_membership(expression)
    expression = _rewrite_is_empty_syntax(expression)
    expression = _rewrite_isnotempty_boolean_comparisons(expression)
    expression = _rewrite_date_function(expression)
    expression = _rewrite_unquoted_dotted_refs(expression)
    expression = _rewrite_exists_predicates(expression)
    expression = _repair_extra_close_before_then(expression)
    expression = _remove_excess_closing_parens(expression)
    expression = _fix_unbalanced_trailing_parens(expression)
    expression = _repair_missing_then_parentheses(expression)
    expression = _rewrite_legacy_else_if(expression)
    expression = _rewrite_null_predicates(expression)
    expression = _strip_min_wrapper(expression)
    return expression


def _source_allows_target_reference(source_sql: str, entity_name: str, column: str) -> bool:
    """Allow a self-reference only when the source SQL explicitly
    preserves the target value or increments it.

    This mirrors the semantic validation rule so the generator can
    mechanically repair the most common false-positive shape: an
    otherwise-correct expression whose final ELSE clause falls back to the
    target column even though the source SQL never does that.
    """
    if not source_sql or not column:
        return False

    source_upper = source_sql.upper()
    column_upper = re.escape(column.upper())
    entity_upper = re.escape(entity_name.upper()) if entity_name else ""

    quoted_target = rf'"{column_upper}"'
    if entity_upper:
        qualified_target = rf'"{entity_upper}"\s*\.\s*{quoted_target}'
    else:
        qualified_target = quoted_target

    preservation_patterns = [
        rf"\bNVL\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?{qualified_target}\s*,",
        rf"\bCOALESCE\s*\(\s*(?:[A-Z_][A-Z0-9_]*\s*\.\s*)?{qualified_target}\s*,",
    ]
    if any(re.search(pattern, source_upper) for pattern in preservation_patterns):
        return True

    if column_upper == "COUNT" and re.search(
        rf"\bNVL\s*\(\s*{quoted_target}\s*,\s*0\s*\)\s*\+\s*1",
        source_upper,
    ):
        return True

    return False


def _repair_trailing_self_reference(expression: str, entity_name: str, column: str, source_sql: str) -> str:
    """Replace a final `ELSE(target_column)` fallback with `ELSE(NULL)` if
    the source SQL does not explicitly preserve the same target column.
    """
    if _source_allows_target_reference(source_sql, entity_name, column):
        return expression

    column_upper = re.escape(column.upper())
    entity_upper = re.escape(entity_name.upper()) if entity_name else ""
    if entity_upper:
        qualified = rf'"{entity_upper}"\s*\.\s*"{column_upper}"'
    else:
        qualified = rf'"{column_upper}"'

    candidate = re.sub(
        rf'(?i)(ELSE\s*\()\s*{qualified}\s*(\)\s*)$',
        r"\1NULL\2",
        expression,
    )
    return candidate if candidate != expression else expression


def generate_dd_rows(
    chain: LineageChain,
    canonical_model: CanonicalModel,
    objects: dict[str, SQLObject],
    structural_infos: dict[str, StructuralInfo],
    llm_client: LLMClient,
    function_reference: str,
    entity_name_map: dict[str, str] | None = None,
    timekey_map: dict[int, date] | None = None,
    rag_store: Optional[ChromaStore] = None,
) -> list[DDRow]:
    entity_name_map = entity_name_map or {}
    dd_rows: list[DDRow] = []

    for oid in chain.order:
        obj = objects[oid]
        info = structural_infos[oid]

        for target_table, columns in info.columns_written_by_table.items():
            entity_name = entity_name_map.get(target_table, target_table)
            for column in columns:
                dd_rows.extend(
                    _generate_for_column(
                        canonical_model=canonical_model,
                        obj=obj,
                        info=info,
                        entity_name=entity_name,
                        column=column,
                        llm_client=llm_client,
                        function_reference=function_reference,
                        timekey_map=timekey_map,
                        rag_store=rag_store,
                    )
                )
    return dd_rows


def _generate_for_column(
    canonical_model: CanonicalModel,
    obj: SQLObject,
    info: StructuralInfo,
    entity_name: str,
    column: str,
    llm_client: LLMClient,
    function_reference: str,
    timekey_map: dict[int, date] | None,
    rag_store: Optional[ChromaStore] = None,
) -> list[DDRow]:
    relevant_chunks = _relevant_chunks(info, column)
    relevant_sql = "\n\n".join(chunk.raw_sql.strip() for chunk in relevant_chunks if chunk.raw_sql.strip())
    rag_context = _retrieve_rag_context(
        rag_store, relevant_sql, canonical_model.technical_summary, canonical_model.business_summary
    )

    raw_output = llm_client.generate_formula_expression(
        technical_summary=canonical_model.technical_summary,
        business_summary=canonical_model.business_summary,
        source_sql=_source_sql_context_excerpt(obj.raw_sql, relevant_sql),
        function_reference=function_reference,
        column_name=column,
        entity_name=entity_name,
        relevant_sql=relevant_sql,
        rag_context=rag_context,
    )

    derivation_option = DerivationOption.FORMULA_EXPRESSION
    expression: str | None = None
    decision_table_json: str | None = None
    validation_errors: list[str] = []

    for attempt in range(_MAX_GENERATION_ATTEMPTS):
        derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)
        if expression:
            expression = _normalize_expression(expression)
            repaired = _repair_trailing_self_reference(expression, entity_name, column, obj.raw_sql)
            if repaired != expression and validate_expression(repaired).valid:
                expression = repaired

        attempt_errors = list(parse_errors)

        if expression and not attempt_errors:
            grammar_result = validate_expression(expression)
            if not grammar_result.valid:
                attempt_errors.append(f"Grammar validation failed: {grammar_result.error}")
            else:
                semantic_result = check_semantic_consistency(
                    expression, column, entity_name, relevant_chunks, obj.raw_sql
                )
                if not semantic_result.passed:
                    attempt_errors.extend(f"Semantic validation: {e}" for e in semantic_result.errors)

        if not attempt_errors:
            validation_errors = []
            break

        validation_errors = attempt_errors
        if attempt == _MAX_GENERATION_ATTEMPTS - 1:
            break

        retry_context = f"Column: {column}, Entity: {entity_name}"
        if relevant_sql:
            retry_context += (
                "\n\nStatements found in the source that assign this "
                f"column (make sure your corrected expression still "
                f"reflects all of them):\n{relevant_sql}"
            )
        if rag_context:
            retry_context += f"\n\nRelevant platform/domain reference:\n{rag_context}"

        raw_output = llm_client.retry_with_error(
            previous_expression=expression or raw_output,
            error="; ".join(attempt_errors),
            context=retry_context,
        )

    confidence = info.confidence if not validation_errors else min(info.confidence, 0.3)
    status = DDStatus.PENDING_REVIEW if validation_errors else DDStatus.ACTIVE

    periods = effective_periods_for_column(info.version_thresholds, timekey_map)
    if not periods:
        periods = [(date.today(), True, "", 0)]

    data_type = _infer_data_type(column)

    rows = []
    for eff_date, is_real_mapping, variable, representative_value in periods:
        row_confidence = confidence if is_real_mapping else min(confidence, 0.5)
        row_status = status if is_real_mapping else DDStatus.PENDING_REVIEW
        row_validation_errors = list(validation_errors)
        if not is_real_mapping:
            row_validation_errors.append(_SYNTHETIC_DATE_REVIEW_NOTE)

        row_expression = expression or ""
        # Only prune an already-clean expression -- pruning a row that's
        # already flagged PENDING_REVIEW would risk hiding the very logic
        # a reviewer needs to see, and there is nothing reliable to prune
        # from an expression that hasn't been validated in the first
        # place.
        if row_expression and variable and not validation_errors:
            pruned = prune_expression_for_period(row_expression, variable, representative_value)
            if pruned != row_expression and validate_expression(pruned).valid:
                row_expression = pruned

        rows.append(
            DDRow(
                entity_name=entity_name,
                column_name=column,
                column_type=ColumnType.PHYSICAL,
                derivation_option=derivation_option,
                display_derivation_expression=row_expression,
                effective_start_date=eff_date,
                status=row_status,
                data_type=data_type,
                decision_table_json=decision_table_json,
                source_chain_id=canonical_model.chain_id,
                source_object_ids=[obj.object_id],
                confidence=row_confidence,
                validation_errors=row_validation_errors,
            )
        )
    return rows


def _interpret_llm_output(
    raw_output: str,
) -> tuple[DerivationOption, str | None, str | None, list[str]]:
    stripped = raw_output.strip().strip("`")
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if "decision_table" in parsed:
                return (
                    DerivationOption.DECISION_TABLE,
                    None,
                    json.dumps(parsed["decision_table"]),
                    [],
                )
        except json.JSONDecodeError as exc:
            return DerivationOption.FORMULA_EXPRESSION, None, None, [f"Could not parse decision table JSON: {exc}"]

    return DerivationOption.FORMULA_EXPRESSION, stripped, None, []


def _infer_data_type(column_name: str) -> str:
    lowered = column_name.lower()
    if any(token in lowered for token in ("date", "dt", "_at")):
        return "datetime"
    if any(token in lowered for token in ("flag", "flg", "ind", "check", "reason")):
        return "string"
    return "number"


def flag_duplicate_dd_rows(dd_rows: list[DDRow]) -> list[DDRow]:
    """Detect DD rows sharing the same (entity_name, column_name,
    effective_start_date) identity -- the exact key
    app/report/excel_export.py::merge_dd_rows uses to decide whether a row
    is "the same row" -- coming from more than one source. A column
    normally has exactly one derivation per effective date; more than one
    commonly means two different source procedures both write the same
    shared table+column (each correctly reflecting its own procedure's own
    logic, often each scoped to different rows by its own guard condition
    -- see check_dropped_override_conditions' row-scoping check), and nothing
    in this pipeline can know on its own whether they should be combined
    into a single formula or whether one is simply wrong for this column.

    Rather than silently exporting duplicate rows for the same key --
    which the platform's own schema does not expect, and which
    merge_dd_rows' last-one-wins-by-key merge would otherwise let one
    silently overwrite the other with no record that a conflict ever
    existed -- every row sharing a duplicated key is routed to
    PENDING_REVIEW with a note identifying the other source chain(s) it
    conflicts with, so a reviewer resolves it explicitly instead of the
    pipeline guessing or the report/Excel silently picking one.

    Rows are never dropped, merged, or rewritten here -- only status,
    confidence, and validation_errors are updated -- so this can never
    lose or alter derivation logic, and a column with only one source
    (the overwhelmingly common case) is completely unaffected.
    """
    key_to_rows: dict[tuple[str, str, object], list[DDRow]] = {}
    for row in dd_rows:
        key = (row.entity_name, row.column_name, row.effective_start_date)
        key_to_rows.setdefault(key, []).append(row)

    for rows in key_to_rows.values():
        if len(rows) < 2:
            continue
        distinct_chains = sorted({r.source_chain_id for r in rows})
        for row in rows:
            other_chains = [c for c in distinct_chains if c != row.source_chain_id] or distinct_chains
            row.status = DDStatus.PENDING_REVIEW
            row.confidence = min(row.confidence, 0.3)
            row.validation_errors.append(
                f'Another derivation for "{row.entity_name}"."{row.column_name}" '
                f"effective {row.effective_start_date} was generated from a "
                f"different source ({', '.join(other_chains)}). Multiple "
                "procedures/statements write this column for this "
                "effective date -- reconcile into a single formula (for "
                "example, guard each with its own row-scoping condition) "
                "before accepting any of them."
            )
    return dd_rows
