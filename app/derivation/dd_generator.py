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

# Total generation attempts per column: one initial attempt plus up to two
# bounded repair/regeneration attempts, each fed the accumulated grammar
# and/or semantic errors from the previous attempt.
_MAX_GENERATION_ATTEMPTS = 3

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
            elif not in_string and ch == "," and depth == 0:
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

    return "".join(result)


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
    expression = _fix_unbalanced_trailing_parens(expression)
    return expression


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
        source_sql=obj.raw_sql,
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