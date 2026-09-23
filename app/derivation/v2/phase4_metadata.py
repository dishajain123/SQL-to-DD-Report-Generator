"""Phase 4 — platform metadata + temporal date extraction → DDRow.

Maps derivation metadata onto the existing ``DDRow`` model so
``app.report.dd_export`` / review stay unchanged. Also exposes a
convenience dict with the conceptual platform CSV headers listed in the
v2 design note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

from app.derivation.versioning import resolve_timekey_to_date
from app.grammar.validator import validate_expression
from app.models.core import (
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    ReviewState,
    VersionThreshold,
)
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Conceptual platform headers from the v2 design (not the live export schema).
PLATFORM_METADATA_HEADERS = [
    "Derivation Type",
    "Source Entity Name",
    "Source Column Name",
    "Target Entity Name",
    "Target Column Name",
    "Derivation Logic Description",
    "Effective Start Date",
    "Effective End Date",
    "Derivation Formula Expression",
]

_TIMEKEY_RE = re.compile(
    r"(?i)@?(?:p_)?TIMEKEY\s*(?P<op>>=|<=|>|<|=|==)\s*(?P<value>\d+)"
)

_DEFAULT_START = date(1900, 1, 1)


@dataclass
class DerivationMetadata:
    derivation_type: str  # DERIVED | DIRECT
    source_entity_name: str
    source_column_name: str
    target_entity_name: str
    target_column_name: str
    derivation_logic_description: str
    effective_start_date: date
    effective_end_date: Optional[date]
    derivation_formula_expression: str
    validation_errors: list[str]
    confidence: float
    synthetic_date: bool = False
    dependency_refs: list[str] = field(default_factory=list)
    # Raw SQL fragments for each mutation (traceability / audit).
    mutation_sql_fragments: list[str] = field(default_factory=list)

    def as_platform_dict(self) -> dict[str, Any]:
        return {
            "Derivation Type": self.derivation_type,
            "Source Entity Name": self.source_entity_name,
            "Source Column Name": self.source_column_name,
            "Target Entity Name": self.target_entity_name,
            "Target Column Name": self.target_column_name,
            "Derivation Logic Description": self.derivation_logic_description,
            "Effective Start Date": self.effective_start_date.strftime("%d-%m-%Y"),
            "Effective End Date": (
                self.effective_end_date.strftime("%d-%m-%Y") if self.effective_end_date else ""
            ),
            "Derivation Formula Expression": self.derivation_formula_expression,
            "Dependency Refs": list(self.dependency_refs),
            "Mutation SQL Fragments": list(self.mutation_sql_fragments),
        }


def extract_timekey_thresholds(sql_text: str) -> list[VersionThreshold]:
    """Pull @TIMEKEY / p_TIMEKEY comparisons from SQL for versioning."""
    thresholds: list[VersionThreshold] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _TIMEKEY_RE.finditer(sql_text or ""):
        op = match.group("op")
        if op == "==":
            op = "="
        value = match.group("value")
        key = ("TIMEKEY", op, value)
        if key in seen:
            continue
        seen.add(key)
        thresholds.append(
            VersionThreshold(
                variable="TIMEKEY",
                operator=op,
                value=value,
                raw_condition=match.group(0),
            )
        )
    return thresholds


def _collect_dependency_refs(
    ast: dict[str, Any] | None,
    mutation_dependency_refs: list[str] | None = None,
) -> list[str]:
    """Gather column dependency refs from mutations + AST (no review flags)."""
    deps: list[str] = []
    seen: set[str] = set()

    def _add(refs: list[str] | None) -> None:
        for ref in refs or []:
            key = (ref or "").upper()
            if key and key not in seen:
                seen.add(key)
                deps.append(ref)

    _add(mutation_dependency_refs)

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        _add(node.get("_dependency_refs") if isinstance(node.get("_dependency_refs"), list) else None)
        for key, value in node.items():
            if key.startswith("_"):
                continue
            if isinstance(value, dict):
                walk(value)
            elif isinstance(value, list):
                for item in value:
                    walk(item)

    walk(ast)
    return deps


def build_metadata(
    *,
    target_entity: str,
    target_column: str,
    formula: str,
    ast: dict[str, Any] | None = None,
    source_sql: str = "",
    mutation_count: int = 0,
    timekey_map: dict[int, date] | None = None,
    effective_start: date | None = None,
    business_summary: str = "",
    mutation_sql_fragments: list[str] | None = None,
    mutation_dependency_refs: list[str] | None = None,
    **_ignored: Any,
) -> DerivationMetadata:
    """Validate formula, classify DERIVED/DIRECT, resolve effective dates.

    Grammar-valid formulas are treated as complete outputs (GENERATED /
    ACTIVE) — no HITL / NEEDS_REVIEW path for simplified SQL constructs.
    """
    del _ignored  # accept legacy kwargs (mutation_review_notes) without effect

    validation = validate_expression(formula)
    errors = [] if validation.valid else [validation.error or "grammar validation failed"]

    derivation_type = _infer_derivation_type(ast, formula, mutation_count)
    source_entity, source_column = _infer_source(ast, target_entity, target_column)

    start, synthetic = _resolve_effective_start(
        source_sql, timekey_map, effective_start
    )
    confidence = 0.95 if validation.valid else 0.35
    if synthetic:
        confidence = min(confidence, 0.6)

    dependency_refs = _collect_dependency_refs(ast, mutation_dependency_refs)

    # NOTE: ``business_summary`` is a whole-procedure summary (one string per
    # CanonicalModel / chain) — it must never be used as the description
    # here, or every column derived from the same procedure ends up with an
    # identical, duplicated "Business Purpose". Always derive a description
    # scoped to this specific target_entity/target_column instead.
    description = _describe_column_derivation(target_entity, target_column, ast, mutation_count)

    if errors:
        logger.info(
            "phase4 grammar gate failed for %s.%s: %s",
            target_entity,
            target_column,
            errors[0],
        )

    return DerivationMetadata(
        derivation_type=derivation_type,
        source_entity_name=source_entity,
        source_column_name=source_column,
        target_entity_name=target_entity,
        target_column_name=target_column,
        derivation_logic_description=description,
        effective_start_date=start,
        effective_end_date=None,
        derivation_formula_expression=formula,
        validation_errors=errors,
        confidence=confidence,
        synthetic_date=synthetic,
        dependency_refs=dependency_refs,
        mutation_sql_fragments=list(mutation_sql_fragments or []),
    )


def metadata_to_dd_row(
    meta: DerivationMetadata,
    *,
    source_chain_id: str,
    source_object_ids: list[str],
    source_statement_refs: list[str] | None = None,
    source_statement_sql: list[str] | None = None,
    data_type: str = "String",
) -> DDRow:
    """Convert Phase-4 metadata into a platform ``DDRow``."""
    valid = not meta.validation_errors
    # Valid formulas ship as ACTIVE/GENERATED. Only grammar failures go pending.
    status = DDStatus.ACTIVE if valid else DDStatus.PENDING_REVIEW
    review_state = ReviewState.GENERATED if valid else ReviewState.NEEDS_REVIEW
    if meta.synthetic_date and valid:
        # Date alone is synthetic — keep GENERATED; do not force HITL.
        pass

    sql_frags = list(meta.mutation_sql_fragments or source_statement_sql or [])

    advisory: list[str] = []
    if meta.synthetic_date:
        advisory.append("Effective Start Date derived synthetically from TIMEKEY")

    return DDRow(
        entity_name=meta.target_entity_name,
        column_name=meta.target_column_name,
        column_type=_infer_column_type(meta.target_entity_name),
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression=meta.derivation_formula_expression,
        effective_start_date=meta.effective_start_date,
        status=status,
        review_state=review_state,
        data_type=data_type,
        business_meaning=meta.derivation_logic_description,
        source_chain_id=source_chain_id,
        source_object_ids=list(source_object_ids),
        source_statement_refs=list(source_statement_refs or []),
        source_statement_sql=sql_frags,
        confidence=meta.confidence if valid else min(meta.confidence, 0.35),
        validation_errors=list(meta.validation_errors),
        advisory_notes=advisory,
        conditional_json=None,
        decision_table_json=None,
    )


def _describe_column_derivation(
    target_entity: str,
    target_column: str,
    ast: dict[str, Any] | None,
    mutation_count: int,
) -> str:
    """Column-specific derivation description.

    Never falls back to a whole-procedure summary — that field is shared
    across every column derived from the same chain and would otherwise
    duplicate verbatim across unrelated rows. Every branch here is scoped to
    this specific ``target_entity``/``target_column`` pair.
    """
    node_type = (ast or {}).get("type") if isinstance(ast, dict) else None

    if node_type == "COLUMN_REF" and mutation_count <= 1:
        src_entity = str(ast.get("entity") or target_entity)
        src_col = str(ast.get("column") or target_column)
        if src_entity.upper() != target_entity.upper() or src_col.upper() != target_column.upper():
            return f"Derives {target_column} for {target_entity} by copying {src_col} from {src_entity}."

    if node_type == "IF_THEN_ELSE":
        return (
            f"Derives {target_column} for {target_entity} by evaluating conditional "
            f"business logic across {mutation_count} UPDATE pass(es)."
        )
    if node_type == "MEMBERSHIP_OP":
        return f"Derives {target_column} for {target_entity} based on a membership (IN / NOT IN) check."
    if node_type == "FUNCTION_CALL":
        func = str((ast or {}).get("function_name") or "").strip()
        if func:
            return f"Derives {target_column} for {target_entity} using {func}(...) logic."
    if mutation_count > 1:
        return (
            f"Derives {target_column} for {target_entity} by folding {mutation_count} "
            "chronological UPDATE pass(es)."
        )
    return f"Derives {target_column} for {target_entity} based on execution logic."


def _infer_derivation_type(
    ast: dict[str, Any] | None,
    formula: str,
    mutation_count: int,
) -> str:
    if mutation_count <= 1 and ast and ast.get("type") == "COLUMN_REF":
        return "DIRECT"
    if formula and "IF(" in formula.upper():
        return "DERIVED"
    if mutation_count > 1:
        return "DERIVED"
    if ast and ast.get("type") in {"IF_THEN_ELSE", "BINARY_OP", "FUNCTION_CALL", "MEMBERSHIP_OP"}:
        return "DERIVED"
    return "DIRECT" if ast and ast.get("type") == "COLUMN_REF" else "DERIVED"


def _infer_source(
    ast: dict[str, Any] | None,
    target_entity: str,
    target_column: str,
) -> tuple[str, str]:
    if not ast:
        return target_entity, target_column
    if ast.get("type") == "COLUMN_REF":
        return str(ast.get("entity") or target_entity), str(ast.get("column") or target_column)
    if ast.get("type") == "IF_THEN_ELSE":
        return _infer_source(ast.get("then_branch"), target_entity, target_column)
    if ast.get("type") == "FUNCTION_CALL":
        args = ast.get("arguments") or []
        if args:
            return _infer_source(args[0], target_entity, target_column)
    return target_entity, target_column


def _resolve_effective_start(
    source_sql: str,
    timekey_map: dict[int, date] | None,
    override: date | None,
) -> tuple[date, bool]:
    if override is not None:
        return override, False
    thresholds = extract_timekey_thresholds(source_sql)
    if not thresholds:
        return _DEFAULT_START, False
    values = sorted({int(t.value) for t in thresholds})
    return resolve_timekey_to_date(values[0], timekey_map)


def _infer_column_type(entity_name: str) -> ColumnType:
    if entity_name.startswith("#") and not entity_name.startswith("##"):
        return ColumnType.TEMPORARY
    return ColumnType.PHYSICAL
