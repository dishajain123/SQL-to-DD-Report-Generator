"""Rule explanations for PRO Marking_FlgDeg_Degreason (S04)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S04 = _REPO / "samples/sql/PRO_SPs_Sequenced/09_S04_PRO.Marking_FlgDeg_Degreason.StoredProcedure.sql"


def _sql() -> str:
    return _S04.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    for ent in (entity, f"#{entity}"):
        row, debug = generate_for_sql(_sql(), ent, column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        if formula:
            return formula
    pytest.skip(f"no formula for {entity}.{column}")


def test_s04_account_flgdeg_ladder():
    formula = _formula("AccountCal", "FlgDeg")
    text = build_rule_explanation("AccountCal", "FlgDeg", formula)
    assert "Zero Balance Override" in text
    assert "Buyout / Virtual DPD" in text
    assert "nested conditional" not in text.lower()
    assert "Buyout Seller NPA" not in text
    assert text[0].isupper()


def test_s04_account_degreason_ladder():
    formula = _formula("AccountCal", "DegReason")
    text = build_rule_explanation("AccountCal", "DegReason", formula)
    assert "Customer Percolation" in text
    assert "DEGRADE BY INT NOT SERVICED" in text
    assert "Manual Overrides" not in text


def test_s04_restructure_dpd_breach_date():
    formula = _formula("AdvAcRestructureCal", "DPD_Breach_Date")
    text = build_rule_explanation("AdvAcRestructureCal", "DPD_Breach_Date", formula)
    assert "MSME" in text or "Prudential" in text
    assert "..." not in text


def test_s04_pui_default_reason():
    formula = _formula("PUI_CAL", "DEFAULT_REASON")
    text = build_rule_explanation("PUI_CAL", "DEFAULT_REASON", formula)
    assert "DCCO" in text
    assert "nested conditional" not in text.lower()


def test_s04_pui_flg_deg():
    formula = _formula("PUI_CAL", "FLG_DEG")
    text = build_rule_explanation("PUI_CAL", "FLG_DEG", formula)
    assert "FinalAssetClassAlt_Key" in text
    assert "Priority Classification" not in text
