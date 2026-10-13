"""Rule explanations for PRO Marking_FlgPNPA (S12)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import entity_key_upper
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S12 = _REPO / "samples/sql/PRO_SPs_Sequenced/20_S12_PRO.Marking_FlgPNPA.StoredProcedure.sql"


def _sql() -> str:
    return _S12.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    for ent in (entity, f"#{entity}"):
        row, debug = generate_for_sql(_sql(), ent, column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        if formula:
            return formula
    pytest.skip(f"no formula for {entity}.{column}")


def test_s12_flg_pnpa_explanation():
    formula = _formula("AccountCal", "FlgPNPA")
    text = build_rule_explanation("AccountCal", "FlgPNPA", formula)
    assert "Projected DPD" in text
    assert "Percolation" in text
    assert "Standard Reset" not in text
    assert "nested conditional" not in text.lower()
    assert "..." not in text


def test_s12_pnpa_date_explanation():
    formula = _formula("AccountCal", "PNPA_DATE")
    text = build_rule_explanation("AccountCal", "PNPA_DATE", formula)
    assert "Primary DPD Projection" in text
    assert "retain existing pnpa_date" not in text.lower()
    assert "nested conditional" not in text.lower()


def test_s12_pnpa_reason_explanation():
    formula = _formula("AccountCal", "PNPA_Reason")
    text = build_rule_explanation("AccountCal", "PNPA_Reason", formula)
    assert "DEGRADE BY INT NOT SERVICED" in text
    assert "PERCOLATION BY UCIF" in text


def test_s12_customer_pnpa_fields():
    dt_formula = _formula("CustomerCal", "PNPA_Dt")
    dt_text = build_rule_explanation("CustomerCal", "PNPA_Dt", dt_formula)
    assert "minimum PNPA_DATE" in dt_text

    key_formula = _formula("CustomerCal", "PNPA_Class_Key")
    key_text = build_rule_explanation("CustomerCal", "PNPA_Class_Key", key_formula)
    assert "Sub-Standard" in key_text


def test_entity_key_upper_strips_session_hashes():
    assert entity_key_upper("##CustomerCal") == "CUSTOMERCAL"
    assert entity_key_upper("#TEMPTABLEPNPA") == "TEMPTABLEPNPA"


def test_s12_customer_flg_pnpa_roll_up():
    formula = _formula("CustomerCal", "FlgPNPA")
    text = build_rule_explanation("CustomerCal", "FlgPNPA", formula)
    assert "customer level" in text.lower()
    assert "Projected DPD" not in text
    text_hash = build_rule_explanation("##CustomerCal", "FlgPNPA", formula)
    assert "customer level" in text_hash.lower()
    assert "Projected DPD" not in text_hash


def test_s12_staging_pnpa_date_aggregates():
    formula = _formula("TEMPTABLEPNPA", "PNPA_DATE")
    text = build_rule_explanation("#TEMPTABLEPNPA", "PNPA_DATE", formula)
    assert "minimum" in text.lower() or "earliest" in text.lower()
    assert "PNPA-marked customer" in text
    assert "Primary DPD Projection" not in text


def test_s12_cte_ucif_pnpa_date_aggregates():
    formula = _formula("CTE_CustomerWiseBalancePNPA", "PNPA_DATE")
    text = build_rule_explanation("CTE_CustomerWiseBalancePNPA", "PNPA_DATE", formula)
    assert "UCIF" in text
    assert "Primary DPD Projection" not in text


def test_s12_staging_entity_keys():
    ce_formula = _formula("TEMPTABLEPNPA", "CustomerEntityID")
    ce_text = build_rule_explanation("TEMPTABLEPNPA", "CustomerEntityID", ce_formula)
    assert "FlgPNPA" in ce_text
    assert "..." not in ce_text

    uc_formula = _formula("CTE_CustomerWiseBalancePNPA", "UcifEntityID")
    uc_text = build_rule_explanation("CTE_CustomerWiseBalancePNPA", "UcifEntityID", uc_formula)
    assert "staging extract" in uc_text.lower()
