"""Architecture step 13: DD Generation — chain collapse + grammar
targeting + versioning, orchestrated end to end.

For each column written by an object in the lineage chain, this asks the
LLM to translate that object's logic into a 4X Formula Expression (or a
Decision Table), validates the result against the real grammar, retries
once on failure with the error fed back to the model, and — if the source
object had TIMEKEY-style version thresholds — splits the result into
multiple DD rows with distinct Effective Start Dates.

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
    raw_output = llm_client.generate_formula_expression(
        technical_summary=canonical_model.technical_summary,
        business_summary=canonical_model.business_summary,
        source_sql=obj.raw_sql,
        function_reference=function_reference,
    )

    derivation_option, expression, decision_table_json, parse_errors = _interpret_llm_output(raw_output)
    if expression:
        expression = _normalize_legacy_if_syntax(expression)

    validation_errors = list(parse_errors)
    if expression:
        result = validate_expression(expression)
        if not result.valid:
            validation_errors.append(result.error or "unknown grammar error")
            corrected = llm_client.retry_with_error(
                previous_expression=expression,
                error=result.error or "invalid syntax",
                context=f"Column: {column}, Entity: {entity_name}",
            )
            corrected_option, corrected_expr, corrected_dt_json, corrected_errors = _interpret_llm_output(corrected)
            if corrected_expr:
                corrected_expr = _normalize_legacy_if_syntax(corrected_expr)
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
