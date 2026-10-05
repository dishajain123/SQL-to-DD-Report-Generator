"""Regression snippets from PRO.InsertDataforAssetClassficationRBL (01_S00)."""
from __future__ import annotations

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
    assert 'COALESCE("ACLRUNNINGPROCESSSTATUS"."COUNT", 0) + 1' in formula.replace(" ", "")
    assert 'ELSE("ACLRUNNINGPROCESSSTATUS"."COUNT")' in formula.replace(" ", "")
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
    assert "IN [" in formula.replace(" ", "") or " IN[" in formula.upper()
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
