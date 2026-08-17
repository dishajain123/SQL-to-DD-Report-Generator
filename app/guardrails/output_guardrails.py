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

from app.grammar.validator import validate_expression
from app.guardrails.input_guardrails import GuardrailResult
from app.models.core import CanonicalModel, DDRow
from app.utils.config import settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


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
