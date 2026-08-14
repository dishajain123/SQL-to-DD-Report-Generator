"""Architecture step 7 (validation half): Structural Guardrails.

Never silently invent missing structural information — a StructuralInfo
that fails these checks should route to Error Handling / flag for review
rather than proceed as if parsing fully succeeded.
"""
from __future__ import annotations

from app.guardrails.input_guardrails import GuardrailResult
from app.models.core import StructuralInfo
from app.utils.config import settings


def check_structural_info(info: StructuralInfo) -> GuardrailResult:
    errors = []

    if info.confidence < settings.structural_confidence_threshold:
        errors.append(
            f"Structural confidence {info.confidence} below threshold "
            f"{settings.structural_confidence_threshold}"
        )

    if info.has_dynamic_sql:
        errors.append("Dynamic SQL (EXECUTE IMMEDIATE) detected — cannot be statically analyzed")

    if info.unsupported_constructs:
        errors.append(f"{len(info.unsupported_constructs)} unparseable statement(s) found")

    if info.statements and not info.smart_chunks:
        errors.append("No smart chunks were derived for a non-empty object")

    if not info.tables_written and not info.tables_read:
        errors.append("No tables read or written — object may be empty or unparseable")

    return GuardrailResult(passed=not errors, errors=errors)
