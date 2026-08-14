from datetime import date

from app.guardrails.input_guardrails import check_input_file, check_job_plan
from app.guardrails.output_guardrails import check_dd_row
from app.guardrails.structural_guardrails import check_structural_info
from app.models.core import (
    CanonicalModel,
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
    StructuralInfo,
)


def test_input_guardrail_rejects_wrong_extension():
    result = check_input_file("notes.txt", "SELECT 1;")
    assert not result.passed


def test_input_guardrail_rejects_empty_file():
    result = check_input_file("empty.sql", "   ")
    assert not result.passed


def test_input_guardrail_accepts_valid_sql():
    result = check_input_file("proc.sql", "CREATE OR REPLACE PROCEDURE X AS BEGIN NULL; END;")
    assert result.passed


def test_job_plan_guardrail_requires_company_and_platform():
    assert not check_job_plan("", "PlatformX").passed
    assert not check_job_plan("Acme", "").passed
    assert check_job_plan("Acme", "PlatformX").passed


def test_structural_guardrail_flags_low_confidence():
    info = StructuralInfo(object_id="x", confidence=0.1, tables_written=["t"])
    result = check_structural_info(info)
    assert not result.passed


def test_structural_guardrail_flags_dynamic_sql():
    info = StructuralInfo(object_id="x", confidence=1.0, has_dynamic_sql=True, tables_written=["t"])
    result = check_structural_info(info)
    assert not result.passed
    assert any("Dynamic SQL" in e for e in result.errors)


def test_structural_guardrail_passes_clean_info():
    info = StructuralInfo(object_id="x", confidence=1.0, tables_written=["t"])
    assert check_structural_info(info).passed


def test_output_guardrail_flags_invalid_grammar():
    model = CanonicalModel(
        chain_id="c1", job_id="j1", object_ids=["x"],
        technical_summary="t", business_summary="b", evidence=["FCT_NPA_PRODUCT"],
    )
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT", column_name="X", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression="IF(BOGUS(x)THEN(1)ELSE(0)",
        effective_start_date=date(2026, 1, 1), status=DDStatus.ACTIVE,
        data_type="number", source_chain_id="c1", confidence=1.0,
    )
    result = check_dd_row(row, model)
    assert not result.passed


def test_output_guardrail_flags_entity_not_in_evidence():
    model = CanonicalModel(
        chain_id="c1", job_id="j1", object_ids=["x"],
        technical_summary="t", business_summary="b", evidence=["SOME_OTHER_TABLE"],
    )
    row = DDRow(
        entity_name="COMPLETELY_UNRELATED_ENTITY", column_name="X", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='TODATE("A"."B")',
        effective_start_date=date(2026, 1, 1), status=DDStatus.ACTIVE,
        data_type="number", source_chain_id="c1", confidence=1.0,
    )
    result = check_dd_row(row, model)
    assert not result.passed
    assert any("hallucination" in e for e in result.errors)


def test_output_guardrail_passes_clean_row():
    model = CanonicalModel(
        chain_id="c1", job_id="j1", object_ids=["x"],
        technical_summary="t", business_summary="b", evidence=["FCT_NPA_PRODUCT"],
    )
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT", column_name="X", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='TODATE("FCT_NPA_PRODUCT"."PERIOD_ID")',
        effective_start_date=date(2026, 1, 1), status=DDStatus.ACTIVE,
        data_type="datetime", source_chain_id="c1", confidence=1.0,
    )
    assert check_dd_row(row, model).passed
