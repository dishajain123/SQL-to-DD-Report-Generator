"""Regressions found by running the RBL AssetClassification / DPD / RefPeriod
stored procedures: parse failures that left DD expressions empty."""
import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.sql_text import extract_insert_select
from app.parsing.sql_lex import normalize_comparison_spacing


def _compile(sql: str) -> str:
    ast = parse_sql_expression_to_ast(sql, default_entity="AccountCal", target_column="X")
    return compile_ast_to_4x_string(ast)


def test_sum_of_two_isnull_calls_is_an_addition_not_one_call():
    formula = _compile("ISNULL(CLAIMCOVERAMT,0)+ ISNULL(CLAIMRECEIVEDAMT,0)")
    assert formula.count("COALESCE(") == 2
    assert "CLAIMCOVERAMT" in formula and "CLAIMRECEIVEDAMT" in formula


def test_dateadd_offset_may_contain_commas():
    formula = _compile("DATEADD(DD, -(ISNULL(DPD_Seller,0)-1), ReportDate)")
    assert formula.startswith("ADDDAY(")
    assert "DPD_Seller" in formula and "ReportDate" in formula


def test_dateadd_month_is_flagged_not_silently_mapped():
    ast = parse_sql_expression_to_ast(
        "DATEADD(MONTH, -3, ProcessDate)",
        default_entity="AccountCal",
        target_column="X",
    )
    assert ast["function_name"] == "__UNSUPPORTED_SQL__"
    with pytest.raises(ValueError, match="no exact 4X equivalent"):
        compile_ast_to_4x_string(ast)


def test_string_agg_is_flagged():
    ast = parse_sql_expression_to_ast(
        "STRING_AGG(DegReason, ', ')",
        default_entity="AccountCal",
        target_column="DegReason",
    )
    assert ast["function_name"] == "__UNSUPPORTED_SQL__"
    with pytest.raises(ValueError, match="STRING_AGG"):
        compile_ast_to_4x_string(ast)


def test_spaced_qualifier_is_normalized_without_moving_offsets():
    sql = "WHERE A. SRCASSETCLASSALT_KEY=1 AND note = 'B. text'"
    fixed, notes = normalize_comparison_spacing(sql)
    assert len(fixed) == len(sql)
    assert "A.SRCASSETCLASSALT_KEY =1" in fixed
    assert "'B. text'" in fixed
    assert len(notes) == 1


def test_insert_select_union_all_yields_one_entry_per_branch():
    sql = (
        "INSERT INTO PRO.INVALIDPANAADHAR (PANNO, EffectiveFromTimeKey) "
        "SELECT B.PAN, @TIMEKEY FROM ##CUSTOMERCAL A INNER JOIN dbo.R B ON A.Id = B.Id "
        "WHERE B.PAN IS NOT NULL AND B.EFFECTIVEFROMTIMEKEY <= @TIMEKEY "
        "UNION ALL "
        "SELECT B.PAN, @TIMEKEY FROM ##CUSTOMERCAL A INNER JOIN dbo.R B ON A.Id = B.Id "
        "WHERE B.PAN LIKE '%FORMO%'"
    )
    entries = extract_insert_select(sql)
    assert len(entries) == 2
    assert "UNION" not in entries[0]["where_clause"].upper()
    assert "FORMO" in entries[1]["where_clause"]
