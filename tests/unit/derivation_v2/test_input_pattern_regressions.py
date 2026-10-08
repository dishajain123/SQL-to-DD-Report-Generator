"""Regressions found by running the RBL AssetClassification / DPD / RefPeriod
stored procedures: parse failures that left DD expressions empty."""
import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.derivation.v2.phase2_mutation_folder import _split_top_level_and_terms, fold_column_mutations
from app.derivation.v2.sql_text import extract_insert_select, extract_select_into
from app.grammar.validator import validate_expression
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


def test_dateadd_month_maps_to_period_not_addday():
    formula = _compile("DATEADD(MONTH, -3, ProcessDate)")
    assert formula.startswith("PERIOD(")
    assert '"MONTH"' in formula
    assert "ADDDAY" not in formula
    assert "ProcessDate" in formula
    assert validate_expression(formula).passed


def test_dateadd_yy_and_mm_aliases_map_to_period():
    year = _compile("DATEADD(YY, @SUB_Days, SysNPA_Dt)")
    month = _compile("DATEADD(MM, @SUB_Days + @DB1_Days, SysNPA_Dt)")
    assert year.startswith("PERIOD(") and '"YEAR"' in year
    assert month.startswith("PERIOD(") and '"MONTH"' in month
    assert validate_expression(year).passed
    assert validate_expression(month).passed


def test_dateadd_week_is_addday_times_seven():
    formula = _compile("DATEADD(WEEK, 2, ProcessDate)")
    compact = " ".join(formula.split())
    assert formula.startswith("ADDDAY(")
    assert "* 7" in compact or ", 14)" in compact or ",14)" in compact.replace(" ", "")
    assert validate_expression(formula).passed


def test_dateadd_hour_is_still_flagged():
    ast = parse_sql_expression_to_ast(
        "DATEADD(HOUR, 1, ProcessDate)",
        default_entity="AccountCal",
        target_column="X",
    )
    assert ast["function_name"] == "__UNSUPPORTED_SQL__"
    with pytest.raises(ValueError, match="DATEADD\\(HOUR"):
        compile_ast_to_4x_string(ast)


def test_eomonth_two_arg_is_eom_of_period():
    formula = _compile("EOMONTH(ProcessDate, -3)")
    compact = formula.replace(" ", "")
    assert compact.startswith("EOM(PERIOD(")
    assert '"MONTH"' in formula
    assert validate_expression(formula).passed


def test_nullif_maps_to_if_then_null():
    formula = _compile("NULLIF(Asset_Norm, 'ALWYS_STD')")
    compact = formula.replace(" ", "")
    assert "THEN(NULL)" in compact
    assert "ALWYS_STD" in formula
    assert validate_expression(formula).passed


def test_scalar_subquery_top_1_lookup_is_column_ref():
    formula = _compile(
        "(SELECT TOP 1 AssetClassAlt_Key FROM DimAssetClass "
        "WHERE AssetClassShortName='LOS' AND EffectiveFromTimeKey<=@TIMEKEY)"
    )
    assert "AssetClassAlt_Key" in formula
    assert "TOP" not in formula.upper()
    # Scalar lookup folds to a relationship hop on the default entity (LOS short name).
    assert "LOS" in formula or "DimAssetClass" in formula
    assert validate_expression(formula).passed


def test_select_into_except_does_not_swallow_anti_join_into_where():
    sql = """
    SELECT CustomerAcID INTO #T
    FROM AdvAcBasicDetail A
    WHERE A.SourceAlt_Key = 1
    EXCEPT
    SELECT CustomerAcID FROM Pro.ContExcsSinceDtAccountCal WHERE EffectiveToTimekey = 49999
    """
    entries = extract_select_into(sql)
    assert entries
    where = entries[0]["where_clause"]
    assert "EXCEPT" not in where.upper()
    assert "SourceAlt_Key" in where
    assert "49999" not in where


def test_npa_erosion_month_aging_folds_to_period():
    sql = """
    UPDATE A SET A.SysAssetClassAlt_Key = (
        CASE WHEN DATEADD(MONTH, @SUB_Days, A.SysNPA_Dt) > @PROCESSDATE
             THEN (SELECT AssetClassAlt_Key FROM DimAssetClass WHERE AssetClassShortName='SUB')
             ELSE A.SysAssetClassAlt_Key
        END)
    FROM ##CUSTOMERCAL A
    INNER JOIN DimAssetClass B ON A.SysAssetClassAlt_Key = B.AssetClassAlt_Key
    WHERE B.AssetClassShortName NOT IN ('STD','LOS')
      AND A.SysNPA_Dt IS NOT NULL
    """
    row, debug = generate_for_sql(sql, "##CUSTOMERCAL", "SysAssetClassAlt_Key", llm_client=None)
    expr = debug["formula"]
    assert expr
    assert "PERIOD(" in expr
    assert '"MONTH"' in expr
    assert "ADDDAY" not in expr
    assert "DimAssetClass" in expr
    assert validate_expression(expr).passed, validate_expression(expr).errors



def test_like_percent_concat_column_maps_to_contains():
    formula = _compile("DegReason LIKE '%' + NPA_Reason + '%'")
    assert "CONTAINS" in formula
    assert "NPA_Reason" in formula
    assert validate_expression(formula).passed


def test_string_agg_lowers_to_source_column():
    ast = parse_sql_expression_to_ast(
        "STRING_AGG(DegReason, ', ')",
        default_entity="AccountCal",
        target_column="DegReason",
    )
    formula = compile_ast_to_4x_string(ast)
    assert "DegReason" in formula
    assert "STRING_AGG" not in formula.upper()
    assert validate_expression(formula).passed


def test_spaced_qualifier_is_normalized_without_moving_offsets():
    sql = "WHERE A. SRCASSETCLASSALT_KEY=1 AND note = 'B. text'"
    fixed, notes = normalize_comparison_spacing(sql)
    assert len(fixed) == len(sql)
    assert "A.SRCASSETCLASSALT_KEY =1" in fixed
    assert "'B. text'" in fixed
    assert len(notes) == 1


def test_between_and_is_not_split_in_join_on_clause():
    on = "A.DPD_Max BETWEEN LowerDPD AND UpperDPD AND A.Segment='X'"
    terms = _split_top_level_and_terms(on)
    assert len(terms) == 2
    assert "BETWEEN LowerDPD AND UpperDPD" in terms[0]


def test_insert_select_eq_alias_projection_value_is_rhs_only():
    sql = """
    INSERT INTO ##ACCOUNTCAL (ACCOUNTENTITYID, CUSTOMERACID)
    SELECT ACCOUNTENTITYID=ACCOUNTENTITYID, CUSTOMERACID=CUSTOMERACID
    FROM AdvAcBasicDetail A
    WHERE A.CustomerEntityId > 0
    """
    lineage = build_lineage_map(sql, None)
    muts = fold_column_mutations(sql, "##ACCOUNTCAL", "ACCOUNTENTITYID", lineage, None)
    assert muts
    assert "=" not in muts[0].assigned_expression or muts[0].assigned_expression.count("=") == 0
    assert "ACCOUNTENTITYID" in muts[0].assigned_expression.upper()
    assert "Print" not in (muts[0].where_clause or "")


def test_right_maps_to_substr():
    formula = _compile("RIGHT(RestructureStage, 3)")
    assert formula.startswith("SUBSTR(")
    assert "LEN(" in formula
    assert validate_expression(formula).passed


def test_dateadd_yy_in_insert_select_is_not_qualified_as_column():
    sql = """
    INSERT INTO PRO.AdvAcRestructureCal (SP_ExpiryDate)
    SELECT DATEADD(YY,1, RestructureDt) FROM PRO.AdvAcRestructureCal
    """
    lineage = build_lineage_map(sql, None)
    muts = fold_column_mutations(sql, "AdvAcRestructureCal", "SP_ExpiryDate", lineage, None)
    assert muts
    assert "::YY" not in (muts[0].assigned_expression or "")


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
