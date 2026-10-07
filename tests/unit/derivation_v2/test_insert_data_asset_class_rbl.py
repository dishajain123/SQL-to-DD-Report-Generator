"""Regression snippets from PRO.InsertDataforAssetClassficationRBL (01_S00)."""
from __future__ import annotations

import re

from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_ACL_TRY_CATCH = """
BEGIN TRY
    UPDATE PRO.ACLRUNNINGPROCESSSTATUS
    SET COMPLETED='N', COUNT=0, ERRORDESCRIPTION=NULL, ERRORDATE=NULL;

    UPDATE PRO.ACLRUNNINGPROCESSSTATUS
    SET COMPLETED='Y', ERRORDATE=NULL, ERRORDESCRIPTION=NULL, COUNT=ISNULL(COUNT,0)+1
    WHERE RUNNINGPROCESSNAME='InsertDataforAssetClassficationRBL';
END TRY
BEGIN CATCH
    UPDATE PRO.ACLRUNNINGPROCESSSTATUS
    SET COMPLETED='N', ERRORDATE=GETDATE(), ERRORDESCRIPTION=ERROR_MESSAGE(), COUNT=ISNULL(COUNT,0)+1
    WHERE RUNNINGPROCESSNAME='InsertDataforAssetClassficationRBL';
END CATCH
"""

_CTE_ASSET_NORM = """
;WITH CTE_NPA_UCIFID_CUST AS (
    SELECT DISTINCT UcifEntityID FROM ##CUSTOMERCAL
    WHERE SysAssetClassAlt_Key > 1
    GROUP BY UcifEntityID
)
UPDATE A SET A.ASSET_NORM='CONDI_STD' FROM ##ACCOUNTCAL A
INNER JOIN CTE_NPA_UCIFID_CUST B ON A.UcifEntityID=B.UcifEntityID
INNER JOIN DimProduct P ON P.EffectiveFromTimeKey<=@TIMEKEY
    AND P.EffectiveToTimeKey>=@TIMEKEY AND P.ProductAlt_Key=A.ProductAlt_Key
    AND P.ProductGroup='FDSEC'
WHERE A.ASSET_NORM='ALWYS_STD'
"""


def test_acl_completed_scoped_to_insert_data_asset_classification_rbl():
    row, debug = generate_for_sql(
        _ACL_TRY_CATCH, "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    formula = debug["formula"]
    assert formula
    assert "InsertDataforAssetClassficationRBL" in formula
    assert '"A"' not in formula
    assert (
        'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "InsertDataforAssetClassficationRBL")'
        in formula.replace(" ", "")
        or "RUNNINGPROCESSNAME" in formula.upper()
    )
    assert 'THEN("Y")' in formula
    assert 'THEN("N")' in (row.exception_handler_expression or "")
    assert validate_expression(formula).passed


def test_acl_count_uses_coalesce_increment_for_rbl_process():
    _, debug = generate_for_sql(
        _ACL_TRY_CATCH, "ACLRUNNINGPROCESSSTATUS", "COUNT", llm_client=None
    )
    formula = debug["formula"]
    assert formula
    assert "InsertDataforAssetClassficationRBL" in formula
    assert 'COALESCE("ACLRUNNINGPROCESSSTATUS"."COUNT",0)+1' in formula.replace(" ", "")
    # The unconditional ``COUNT=0`` reset runs first, so the executed ELSE value
    # is that reset (0); without a reset it is the column itself.
    compact = formula.replace(" ", "")
    assert 'ELSE("ACLRUNNINGPROCESSSTATUS"."COUNT")' in compact or "ELSE(0)" in compact
    assert "THEN(1)" not in formula.replace(" ", "")
    assert validate_expression(formula).passed


def test_customer_cal_panno_dummy_list_nulling():
    sql = """
    UPDATE ##CUSTOMERCAL SET PANNO = NULL
    WHERE PANNO IN ('FORMO6161O','FORPM6060F','FORPM6060P','FORMF6060F','AAAAA1111A');
    """
    row, debug = generate_for_sql(sql, "##CUSTOMERCAL", "PANNO", llm_client=None)
    formula = debug["formula"]
    assert formula
    assert "FORMO6161O" in formula
    assert "AAAAA1111A" in formula
    assert "PANNO" in formula.upper()
    assert re.search(r"\bIN\s*\[", formula, flags=re.I)
    assert "NULL" in formula.upper()
    assert "##" not in formula
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_concat_npa_reason_is_null_safe():
    sql = """
    UPDATE A SET A.NPA_Reason = CONCAT(A.NPA_Reason, ',', 'NPA DUE TO OVERDUE')
    FROM ##ACCOUNTCAL A
    WHERE A.FlgDeg = 'Y';
    """
    _, debug = generate_for_sql(sql, "##ACCOUNTCAL", "NPA_Reason", llm_client=None)
    formula = debug["formula"]
    assert formula
    assert "CONCAT" in formula.upper()
    assert 'COALESCE("ACCOUNTCAL"."NPA_Reason"' in formula.replace(" ", "")
    assert "##" not in formula
    assert validate_expression(formula).passed


def test_cte_dimproduct_asset_norm_has_no_alias_a_leak():
    row, debug = generate_for_sql(
        _CTE_ASSET_NORM, "##ACCOUNTCAL", "ASSET_NORM", llm_client=None
    )
    formula = debug["formula"]
    assert formula
    assert '"A"' not in formula
    assert "DimProduct" in formula or "DIMPRODUCT" in formula.upper()
    assert "ALWYS_STD" in formula
    assert 'THEN("CONDI_STD")' in formula
    assert validate_expression(formula).passed, validate_expression(formula).errors


_NOT_EXISTS_CHARGE_OFF = """
UPDATE ACL SET
   Asset_Norm='ALWYS_NPA'
   ,DegReason='NPA DUE TO CREDIT CARD SETTLEMENT - Always NPA'
 FROM ##ACCOUNTCAL ACL
 WHERE AccountBlkCode2 in ('K','E','W') AND FinalAssetClassAlt_Key=1
 AND NOT exists (SELECT 1 FROM ExceptionFinalStatusType A
                              WHERE A.EffectiveFromTimeKey<=@TIMEKEY AND A.EffectiveToTimeKey >=@TIMEKEY
							  and acl.CustomerAcID=a.ACID
                             AND A.StatusType='Charge Off')
"""

_MANUAL_UPGRADE = """
 UPDATE A SET ASSET_NORM='ALWYS_STD',
   flgdeg='N'
   ,FlgUpg='N'
   ,DegReason=NULL
  FROM ##ACCOUNTCAL A
  INNER JOIN Manual_Upgrade B ON A.CustomerAcID=B.[Account No]
  WHERE VALID_UPTO>='2021-10-25'
  and [Account No] not in(select [Account No] from Manual_NPA)
"""


def test_not_exists_exception_status_projects_to_valid_degreason_formula():
    _, debug = generate_for_sql(
        _NOT_EXISTS_CHARGE_OFF, "##ACCOUNTCAL", "DegReason", llm_client=None
    )
    formula = debug["formula"]
    assert formula
    assert "Charge Off" in formula or "CHARGE" in formula.upper()
    compact = formula.replace(" ", "")
    assert 'THEN("Y")ELSE("Y")' not in compact
    assert "MIN(" not in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_manual_upgrade_not_in_subquery_yields_flgdeg_formula():
    _, debug = generate_for_sql(
        _MANUAL_UPGRADE, "##ACCOUNTCAL", "flgdeg", llm_client=None
    )
    formula = debug["formula"]
    assert formula
    assert "Manual_NPA" in formula or "MANUAL_NPA" in formula.upper()
    assert "MIN(" not in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_getminimumdate_udf_lowers_to_row_level_if():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
    from app.derivation.v2.ast_optimize import optimize_expression_ast
    from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast

    ast = parse_sql_expression_to_ast(
        "[RBL_MISDB].PRO.GETMINIMUMDATE(BillDueDt,InterestOverdueDate,NULL)",
        default_entity="AccountCal",
        target_column="OverDueSinceDt",
    )
    assert ast.get("function_name") != "__UNSUPPORTED_SQL__"
    optimized = optimize_expression_ast(
        ast, target_entity="AccountCal", target_column="OverDueSinceDt"
    )
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="OverDueSinceDt"
    )
    upper = formula.upper()
    assert "MIN(" not in upper
    assert "MAX(" not in upper
    assert "GETMINIMUMDATE" not in upper
    assert "IF(" in formula
    assert validate_expression(formula).passed, validate_expression(formula).errors
