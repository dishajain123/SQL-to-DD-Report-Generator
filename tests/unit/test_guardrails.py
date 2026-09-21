from datetime import date

from app.guardrails.input_guardrails import check_input_file, check_job_plan
from app.guardrails.semantic_validation import check_invented_references
from app.guardrails.semantic_validation import check_semantic_consistency
from app.guardrails.output_guardrails import (
    check_contradictory_guard_conjuncts,
    check_dd_row,
    check_no_bare_identifiers,
)
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


def test_output_guardrail_treats_entity_not_in_evidence_as_informational():
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
    assert result.passed
    assert not result.errors


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


def test_check_no_bare_identifiers_flags_unqualified_column():
    errs = check_no_bare_identifiers('IF(ISEMPTY(DebitSinceDt))THEN("A"."X")ELSE(0)')
    assert any("DebitSinceDt" in e for e in errs)


def test_check_no_bare_identifiers_allows_fully_quoted_expression():
    assert check_no_bare_identifiers('IF("A"."X" > 0)THEN(1)ELSE(0)') == []


def test_check_no_bare_identifiers_allows_known_functions_and_keywords():
    assert check_no_bare_identifiers(
        'IF(ISNOTEMPTY("A"."X") AND NOT(ISEMPTY("A"."Y")))THEN(COALESCE("A"."X", 0))ELSE(NULL)'
    ) == []


def test_check_no_bare_identifiers_allows_documented_timekey_parameter():
    # Platform convention (dd_generation.yaml): a rule-versioning threshold
    # parameter (T-SQL @TIMEKEY, Oracle bare p_TIMEKEY) is written as its
    # own bare, unquoted name -- it is a parameter, not a column, and must
    # never be flagged.
    assert check_no_bare_identifiers('IF(p_TIMEKEY > 26267)THEN(1)ELSE(0)') == []


def test_check_no_bare_identifiers_flags_unrewritten_exists_subquery_sql():
    # Regression found while validating this guardrail against the real
    # sample corpus: EXISTS predicates can carry raw, un-rewritten SQL
    # (SELECT/FROM/WHERE and bare columns) instead of 4X syntax.
    expr = 'IF(EXISTS((SELECT 1 FROM "AccountCal" WHERE AssetClass != "STANDARD")))THEN(1)ELSE(0)'
    errs = check_no_bare_identifiers(expr)
    assert any("AssetClass" in e for e in errs)


def test_check_contradictory_guard_conjuncts_flags_isempty_isnotempty_same_arg():
    expr = (
        'IF("A"."FlgPNPA" == "Y" AND ISEMPTY("A"."PNPA_Reason") '
        'AND ISNOTEMPTY("A"."PNPA_Reason"))THEN(1)ELSE(0)'
    )
    errs = check_contradictory_guard_conjuncts(expr)
    assert errs
    assert "unreachable" in errs[0]


def test_check_contradictory_guard_conjuncts_flags_mutually_exclusive_equality():
    expr = 'IF("A"."X" == "a" AND "A"."X" == "b")THEN(1)ELSE(0)'
    errs = check_contradictory_guard_conjuncts(expr)
    assert errs
    assert "unreachable" in errs[0]


def test_check_contradictory_guard_conjuncts_allows_clean_guard():
    expr = 'IF("A"."X" > 0 AND "A"."Y" == "Z")THEN(1)ELSE(0)'
    assert check_contradictory_guard_conjuncts(expr) == []


def test_check_contradictory_guard_conjuncts_allows_different_arguments():
    # ISEMPTY(x) AND ISNOTEMPTY(y) for two DIFFERENT references is a
    # perfectly normal compound guard, not a contradiction.
    expr = 'IF(ISEMPTY("A"."X") AND ISNOTEMPTY("A"."Y"))THEN(1)ELSE(0)'
    assert check_contradictory_guard_conjuncts(expr) == []


def test_output_guardrail_demotes_row_with_contradictory_guard():
    model = CanonicalModel(
        chain_id="c1", job_id="j1", object_ids=["x"],
        technical_summary="t", business_summary="b", evidence=["FCT_NPA_PRODUCT"],
    )
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT", column_name="X", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression=(
            'IF(ISEMPTY("FCT_NPA_PRODUCT"."Y") AND ISNOTEMPTY("FCT_NPA_PRODUCT"."Y"))'
            'THEN(1)ELSE(0)'
        ),
        effective_start_date=date(2026, 1, 1), status=DDStatus.ACTIVE,
        data_type="number", source_chain_id="c1", confidence=1.0,
    )
    result = check_dd_row(row, model)
    assert not result.passed
    assert any("unreachable" in e for e in result.errors)


def test_output_guardrail_demotes_row_with_bare_identifier():
    model = CanonicalModel(
        chain_id="c1", job_id="j1", object_ids=["x"],
        technical_summary="t", business_summary="b", evidence=["FCT_NPA_PRODUCT"],
    )
    row = DDRow(
        entity_name="FCT_NPA_PRODUCT", column_name="X", column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression='IF(ISEMPTY(DebitSinceDt))THEN("FCT_NPA_PRODUCT"."X")ELSE(0)',
        effective_start_date=date(2026, 1, 1), status=DDStatus.ACTIVE,
        data_type="number", source_chain_id="c1", confidence=1.0,
    )
    result = check_dd_row(row, model)
    assert not result.passed
    assert any("DebitSinceDt" in e for e in result.errors)


def test_semantic_guardrail_flags_invented_bare_identifier():
    errors = check_invented_references(
        'IF(ERROR_OCCURRED)THEN(SQLERRM)ELSE(NULL)',
        'UPDATE PRO.ACLRUNNINGPROCESSSTATUS SET ERRORDESCRIPTION = SQLERRM;',
        entity_name="ACLRUNNINGPROCESSSTATUS",
    )
    assert errors


def test_semantic_guardrail_flags_constant_comparison():
    result = check_semantic_consistency(
        'IF("N"=="Y")THEN(1)ELSE(0)',
        column="X",
        entity_name="A",
        relevant_chunks=[],
        source_sql='UPDATE A SET X = 1 WHERE FLAG = "Y";',
    )
    assert not result.passed
    assert any("compares only literals" in e for e in result.errors)


def test_semantic_guardrail_allows_field_vs_literal_comparisons():
    result = check_semantic_consistency(
        'IF("Aqua_Scheme"=="Y" AND "SchemeType"=="ODA")THEN(1)ELSE(0)',
        column="X",
        entity_name="DIMPRODUCT",
        relevant_chunks=[],
        source_sql='WHERE Aqua_Scheme = "Y" AND SchemeType = "ODA"',
    )
    assert result.passed


def test_semantic_guardrail_allows_field_to_field_comparison_with_joined_source():
    result = check_semantic_consistency(
        'IF("A"."X"=="B"."Y")THEN(1)ELSE(0)',
        column="Z",
        entity_name="A",
        relevant_chunks=[],
        source_sql="SELECT A.X, B.Y FROM A JOIN B ON A.ID = B.ID",
    )
    assert result.passed


def test_semantic_guardrail_flags_invented_qualified_namespace():
    result = check_semantic_consistency(
        'IF("A"."X"=="B"."Y")THEN(1)ELSE(0)',
        column="Z",
        entity_name="A",
        relevant_chunks=[],
        source_sql="SELECT A.X FROM A",
    )
    assert not result.passed
    assert any('"B"' in e or '"B' in e for e in result.errors)


def test_semantic_guardrail_ignores_nested_where_inside_exists():
    result = check_semantic_consistency(
        'IF("A"."INITIALNPADT"==1)THEN(0)ELSE(1)',
        column="INITIALNPADT",
        entity_name="A",
        relevant_chunks=[],
        source_sql=(
            "UPDATE A SET INITIALNPADT = NULL "
            "WHERE EXISTS (SELECT 1 FROM B WHERE B.ID = A.ID AND B.FLAG = 'Y');"
        ),
    )
    assert result.passed


def test_semantic_guardrail_flags_identical_then_else_branches():
    result = check_semantic_consistency(
        'IF("A"."X"==TODATE("1900-01-01"))THEN(NULL)ELSE(NULL)',
        column="X",
        entity_name="A",
        relevant_chunks=[],
        source_sql="UPDATE A SET X = NULL WHERE X = DATE '1900-01-01';",
    )
    assert not result.passed
    assert any("THEN and ELSE branches resolve to the same value" in e for e in result.errors)


def test_semantic_guardrail_allows_distinct_branches():
    result = check_semantic_consistency(
        'IF("A"."X"==TODATE("1900-01-01"))THEN(NULL)ELSE("A"."X")',
        column="X",
        entity_name="A",
        relevant_chunks=[],
        source_sql="UPDATE A SET X = NULL WHERE X = DATE '1900-01-01';",
    )
    assert result.passed
