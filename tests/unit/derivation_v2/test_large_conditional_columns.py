"""Regression: wide reason-code columns must compile under the 8k grammar cap."""
from pathlib import Path

from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_INSERT_RBL = _REPO / "samples/sql/PRO_SPs_Sequenced/01_S00_PRO.InsertDataforAssetClassficationRBL.StoredProcedure.sql"
_NPA_REASON = _REPO / "samples/sql/PRO_SPs_Sequenced/21_S13_PRO.Marking_NPA_Reason_NPAAccount.StoredProcedure.sql"
_CHAR_CAP = 8000


def test_customer_cal_degreason_from_insert_rbl_under_budget():
    sql = _INSERT_RBL.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(sql, "CustomerCal", "DegReason", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula, "expected non-empty DegReason formula"
    assert len(formula) <= _CHAR_CAP, f"length {len(formula)}"
    assert validate_expression(formula).passed, validate_expression(formula).errors
    assert not row.validation_errors


def test_account_cal_npa_reason_under_budget():
    sql = _NPA_REASON.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(sql, "AccountCal", "NPA_Reason", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert formula, "expected non-empty NPA_Reason formula"
    assert len(formula) <= _CHAR_CAP
    assert validate_expression(formula).passed, validate_expression(formula).errors
    assert not row.validation_errors
