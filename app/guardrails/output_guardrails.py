"""Architecture step 14: AI Output Guardrails.

Rule-based checks only (no second LLM call) — cheap, deterministic, and
catches a real class of errors: malformed grammar, and expressions that
reference tables/entities never seen in the source structural analysis
(a concrete, checkable stand-in for "hallucinated" content).
"""
from __future__ import annotations

from app.grammar.validator import validate_expression
from app.guardrails.input_guardrails import GuardrailResult
from app.models.core import CanonicalModel, DDRow
from app.utils.config import settings


def check_dd_row(dd_row: DDRow, canonical_model: CanonicalModel) -> GuardrailResult:
    errors = []

    if dd_row.confidence < settings.output_guardrail_confidence_threshold:
        errors.append(
            f"DD row confidence {dd_row.confidence} below threshold "
            f"{settings.output_guardrail_confidence_threshold}"
        )

    if dd_row.display_derivation_expression:
        result = validate_expression(dd_row.display_derivation_expression)
        if not result.valid:
            errors.append(f"Grammar validation failed: {result.error}")

    if dd_row.derivation_option.value == "Decision Table" and not dd_row.decision_table_json:
        errors.append("Decision Table derivation option chosen but decision_table_json is empty")

    if not dd_row.entity_name or not dd_row.column_name:
        errors.append("entity_name and column_name are both required")

    # Evidence check: the entity this row claims to derive should be one of
    # the tables actually touched by the source lineage chain.
    if canonical_model.evidence and dd_row.entity_name:
        evidence_text = " ".join(canonical_model.evidence).lower()
        if dd_row.entity_name.lower() not in evidence_text:
            errors.append(
                f"Entity '{dd_row.entity_name}' not found in source evidence — "
                f"possible hallucination, needs review"
            )

    return GuardrailResult(passed=not errors, errors=errors)
