"""Rule explanations for PRO NPA_Date_Calculation (S06)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S06 = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"


def _sql() -> str:
    return _S06.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if not formula:
        pytest.skip(f"no formula for {entity}.{column}")
    return formula


def test_s06_adv_ac_restructure_degdate_explanation():
    formula = _formula("AdvAcRestructureCal", "DegDate")
    text = build_rule_explanation("AdvAcRestructureCal", "DegDate", formula)
    assert "PreRestructureNPA_Date" in text
    assert "SP_Expiry" in text
    assert "@ProcessDate" not in text
    assert "nested conditional" not in text.lower()


@pytest.mark.parametrize(
    "column",
    (
        "DPD_IntService",
        "DPD_Overdrawn",
        "DPD_Overdue",
        "DPD_Renewal",
        "DPD_StockStmt",
    ),
)
def test_s06_temptabledpd_dpd_threshold_explanation(column: str):
    formula = _formula("TEMPTABLEDPD", column)
    text = build_rule_explanation("TEMPTABLEDPD", column, formula)
    assert "ACCOUNTCAL." in text
    assert "#TEMPTABLEDPD" in text
    assert "..." not in text
    assert "nested conditional" not in text.lower()


def test_s06_temptabledpd_dpd_nocredit_oda_explanation():
    formula = _formula("TEMPTABLEDPD", "DPD_NoCredit")
    text = build_rule_explanation("TEMPTABLEDPD", "DPD_NoCredit", formula)
    assert "91-day" in text or "RefPeriodNoCredit" in text
    assert "ODA" in text or "91" in text
    assert "..." not in text


def test_s06_account_cal_asset_norm_pui_explanation():
    formula = _formula("AccountCal", "Asset_Norm")
    text = build_rule_explanation("AccountCal", "Asset_Norm", formula)
    assert "PUI_CAL.Asset_Norm" in text
    assert "AccountEntityId" in text or "AccountEntityID" in text
    assert "retains existing Asset_Norm" in text
