"""Completeness gate must not treat advisory notes as procedure-wide blockers."""
from __future__ import annotations

from datetime import date

from app.guardrails.completeness import COMPLETENESS_GATE_REASON, enforce_completeness
from app.models.core import (
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    Dialect,
    ReviewState,
    SQLObject,
    StructuralInfo,
)
from app.parsing.structural_analysis import analyze_object


def _row(entity: str, column: str, *, advisories: list[str] | None = None) -> DDRow:
    return DDRow(
        entity_name=entity,
        column_name=column,
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISNOTEMPTY("A"."X"))THEN(1)ELSE(0)',
        effective_start_date=date(2026, 1, 1),
        status=DDStatus.ACTIVE,
        review_state=ReviewState.GENERATED,
        data_type="number",
        source_chain_id="c1",
        source_object_ids=["obj"],
        source_statement_sql=["UPDATE T SET X=1;"],
        confidence=1.0,
        advisory_notes=list(advisories or []),
    )


def test_advisory_notes_do_not_gate_procedure_or_row():
    sql = "UPDATE T SET X=1;"
    obj = SQLObject(
        object_id="obj",
        source_file="test.sql",
        name="test",
        object_type="PROCEDURE",
        dialect=Dialect.SQLSERVER,
        raw_sql=sql,
    )
    info = analyze_object(obj)
    row = _row(
        "T",
        "X",
        advisories=["Effective Start Date derived synthetically from TIMEKEY"],
    )
    evidence = enforce_completeness([row], {obj.object_id: obj}, {obj.object_id: info})
    assert row.status == DDStatus.ACTIVE
    assert COMPLETENESS_GATE_REASON not in row.validation_errors
    assert any("TIMEKEY" in a for a in evidence["objects"][obj.object_id]["advisories"])
