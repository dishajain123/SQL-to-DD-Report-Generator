"""Architecture step 13: DD Generation — chain collapse + grammar
targeting + versioning, orchestrated end to end.

For each column written by an object in the lineage chain, this asks the
LLM to translate that object's logic into a 4X Formula Expression (or a
Decision Table), validates the result against the real grammar, retries
once on failure with the error fed back to the model, and — if the source
object had TIMEKEY-style version thresholds — splits the result into
multiple DD rows with distinct Effective Start Dates.

A column is very often assigned in more than one place in a real
procedure (a main calculation plus a special-case override, or a success
path plus an error-handling path). To make sure the generated derivation
reflects all of those assignment locations rather than just whichever one
the model happens to notice first in a long procedure, this module builds
a column-specific SQL excerpt from the object's SmartChunks (see
app/parsing/smart_chunking.py) -- every logical block that actually
assigns the target column, anywhere in the object -- and passes that to
the LLM as the authoritative source for that one column.

Entity-name resolution (staging table -> fact table name) is intentionally
pluggable via `entity_name_map` rather than hardcoded, since that mapping is
company/platform-specific and not something this pipeline can infer from
SQL alone.
"""
from __future__ import annotations

import json
from datetime import date

from app.derivation.llm_client import LLMClient
from app.derivation.versioning import effective_dates_for_column
from app.grammar.validator import validate_expression
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    LineageChain,
    SQLObject,
    StructuralInfo,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def _relevant_sql_excerpt(info: StructuralInfo, column: str) -> str:
    """Collect every logical block (SmartChunk) that actually assigns the
    target column somewhere in the object -- across every conditional
    branch, MERGE override, or exception handler, not just wherever it
    happens to appear first.

    SmartChunks already keep control-flow blocks (IF/ELSE, CASE) together
    as one unit, and each chunk's `columns_written` is the union of every
    column actually assigned within it -- so filtering on that gives a
    focused, still-conditionally-correct excerpt for one column, built the
    same way regardless of which procedure or column is being processed.

    Returns an empty string (letting the caller fall back to explaining
    that nothing specific was isolated) if smart chunking found nothing for
    this column -- this is a targeting aid, not a hard requirement.
    """
    excerpts: list[str] = []
    seen_chunk_ids: set[str] = set()
    for chunk in info.smart_chunks:
        if column in chunk.columns_written and chunk.chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk.chunk_id)
            excerpt = chunk.raw_sql.strip()
            if excerpt:
                excerpts.append(excerpt)
    return "\n\n".join(excerpts)


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
    order chosen so each pass sees syntax the next one expects (quotes and
    operators normalized before function-call rewriting and comma-style IF
    detection, both of which rely on string-boundary tracking)."""
    expression = _normalize_sql_style_syntax(expression)
    expression = _normalize_sql_functions(expression)
    expression = _normalize_legacy_if_syntax(expression)
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
) -> list[DDRow]:
    relevant_sql = _relevant_sql_excerpt(info, column)

    raw_output = llm_client.generate_formula_expression(
        technical_summary=canonical_model.technical_summary,
        business_summary=canonical_model.business_summary,
        source_sql=obj.raw_sql,
        function_reference=function_reference,
        column_name=column,
        entity_name=entity_name,
        relevant_sql=relevant_sql,
    )

    derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)
    if expression:
        expression = _normalize_expression(expression)

    validation_errors = list(parse_errors)
    if expression:
        result = validate_expression(expression)
        if not result.valid:
            validation_errors.append(result.error or "unknown grammar error")
            retry_context = f"Column: {column}, Entity: {entity_name}"
            if relevant_sql:
                retry_context += (
                    "\n\nStatements found in the source that assign this "
                    f"column (make sure your corrected expression still "
                    f"reflects all of them):\n{relevant_sql}"
                )
            corrected = llm_client.retry_with_error(
                previous_expression=expression,
                error=result.error or "invalid syntax",
                context=retry_context,
            )
            corrected_option, corrected_expr, corrected_dt_json, corrected_errors = _interpret_llm_output(corrected)
            if corrected_expr:
                corrected_expr = _normalize_expression(corrected_expr)
            retry_result = validate_expression(corrected_expr) if corrected_expr else None
            if retry_result and retry_result.valid:
                derivation_option, expression, decision_table_json = (
                    corrected_option,
                    corrected_expr,
                    corrected_dt_json,
                )
                validation_errors = []
            else:
                validation_errors.append("retry also failed validation")

    confidence = info.confidence if not validation_errors else min(info.confidence, 0.3)
    status = DDStatus.PENDING_REVIEW if validation_errors else DDStatus.ACTIVE

    effective_dates = effective_dates_for_column(info.version_thresholds, timekey_map)
    if not effective_dates:
        effective_dates = [(date.today(), True)]

    data_type = _infer_data_type(column)

    rows = []
    for eff_date, is_real_mapping in effective_dates:
        row_confidence = confidence if is_real_mapping else min(confidence, 0.5)
        row_status = status if is_real_mapping else DDStatus.PENDING_REVIEW
        rows.append(
            DDRow(
                entity_name=entity_name,
                column_name=column,
                column_type=ColumnType.PHYSICAL,
                derivation_option=derivation_option,
                display_derivation_expression=expression or "",
                effective_start_date=eff_date,
                status=row_status,
                data_type=data_type,
                decision_table_json=decision_table_json,
                source_chain_id=canonical_model.chain_id,
                source_object_ids=[obj.object_id],
                confidence=row_confidence,
                validation_errors=validation_errors,
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