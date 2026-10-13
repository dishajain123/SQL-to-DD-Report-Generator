"""Regressions for PRO_SPs_Sequenced audit gap fixes (S00/S09/S10/S00-sub)."""
from __future__ import annotations

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase3_ast_generator import (
    _is_set_based_unresolved_source_assignment,
    parse_sql_expression_to_ast,
)
from app.derivation.v2.phase2_mutation_folder import MutationPass
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import extract_update_statements, parse_select_list
from app.grammar.validator import validate_expression


def test_update_extractor_stops_from_and_where_before_while_set():
    sql = """
WHILE @MAX >= @MIN
BEGIN
UPDATE B
SET B.X = 1
FROM T B
INNER JOIN T A ON A.ID = B.ID
AND B.Flag = 'Y'
    SET @MIN = @MIN + 1
END
"""
    stmts = extract_update_statements(sql)
    assert len(stmts) == 1
    assert "SET @MIN" not in (stmts[0].get("from_clause") or "")
    assert "SET @MIN" not in (stmts[0].get("where_clause") or "")


def test_set_based_skip_allows_mapped_join_alias_marker():
    mutation = MutationPass(
        ordinal=1,
        target_entity="ACCOUNTCAL",
        target_column="Balance",
        assigned_expression="ACCOUNTCAL::B::Balance",
        where_clause=None,
        alias_map={"B": "SomeSourceTable", "A": "ACCOUNTCAL"},
    )
    assert not _is_set_based_unresolved_source_assignment(mutation, "ACCOUNTCAL")


def test_parse_select_list_literal_with_bracket_alias():
    rows = parse_select_list("'Link' [CustomerType], UCIF_ID")
    assert len(rows) == 2
    assert rows[0][2] == "CustomerType"
    assert rows[0][3] == "'Link'"


def test_s10_coborrower_like_condition_compiles():
    sql = """
UPDATE B
SET B.FinalAssetClass_AltKey = 1,
    B.FINALNPADT = NULL,
    B.DegradeReason = NULL
FROM [PRO].[CoBorrowerCal] A
INNER JOIN [PRO].[CoBorrowerCal] B
    ON B.MAINCUSTID = A.[CustomerID]
    AND A.[CustomerType] IN ('Coborrower', 'CO_OBLIGANT')
    AND B.DegradeReason LIKE '%Co Borrower NPA%'
"""
    row, debug = generate_for_sql(sql, "CoBorrowerCal", "DegradeReason", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula, row.validation_errors
    assert "DOESNOTCONTAINS" in formula or "CONTAINS" in formula
    assert validate_expression(formula).valid


def test_s00_join_balance_assignment_folds():
    sql = """
UPDATE A
SET A.Balance = B.Balance
FROM ##ACCOUNTCAL A
INNER JOIN dbo.AdvAcBalanceDetail B ON A.AccountEntityID = B.AccountEntityID
WHERE B.EffectiveFromTimeKey <= @TIMEKEY
"""
    row, debug = generate_for_sql(sql, "ACCOUNTCAL", "Balance", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula, row.validation_errors
    assert validate_expression(formula).valid


def test_literal_left_in_list_resolves_column_paths_not_none():
    ast = parse_sql_expression_to_ast(
        "860 IN (ISNULL(AdvAcOtherDetail.SplCatg1Alt_Key,0), ISNULL(AdvAcOtherDetail.SplCatg2Alt_Key,0), "
        "ISNULL(AdvAcOtherDetail.SplCatg3Alt_Key,0), ISNULL(AdvAcOtherDetail.SplCatg4Alt_Key,0))",
        default_entity="AccountCal",
        target_column="Asset_Norm",
    )
    assert ast.get("type") == "MEMBERSHIP_OP"
    values = ast.get("values") or []
    assert len(values) == 4
    assert all(isinstance(v, dict) for v in values)
    formula = compile_ast_to_4x_string(
        ast, target_entity="AccountCal", target_column="Asset_Norm"
    )
    assert "None" not in formula
    assert "SplCatg1Alt_Key" in formula
    assert "AdvAcOtherDetail" in formula
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s00_written_off_splcatg_in_guard_on_account_cal_columns():
    sql = """
    UPDATE A SET A.Asset_Norm='ALWYS_NPA', A.FinalAssetClassAlt_Key=2, A.FinalNpaDt=@ProcessDate
    FROM ##AccountCal A INNER JOIN AdvAcOtherDetail B
    ON A.AccountEntityID=B.AccountEntityID AND B.EffectiveFromTimeKey<=@TIMEKEY AND B.EffectiveToTimeKey>=@TIMEKEY
    WHERE 860 IN (ISNULL(B.SplCatg1Alt_Key,0),ISNULL(B.SplCatg2Alt_Key,0),ISNULL(B.SplCatg3Alt_Key,0),ISNULL(B.SplCatg4Alt_Key,0))
    AND A.FinalAssetClassAlt_Key=1
    """
    for column in ("Asset_Norm", "FinalAssetClassAlt_Key", "FinalNpaDt"):
        row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert formula, (column, row.validation_errors)
        assert "None" not in formula, formula[:400]
        assert "SplCatg1Alt_Key" in formula or "860 IN" in formula.replace(" ", "")
        assert validate_expression(formula).passed, (column, validate_expression(formula).errors)
