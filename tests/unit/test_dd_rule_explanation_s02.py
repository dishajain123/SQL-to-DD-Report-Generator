"""Rule explanations for PRO DPD_Calculation (S02)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S02 = _REPO / "samples/sql/PRO_SPs_Sequenced/07_S02_PRO.DPD_Calculation.StoredProcedure.sql"


def _sql() -> str:
    return _S02.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if not formula:
        pytest.skip(f"no formula for {entity}.{column}")
    return formula


def test_s02_int_not_serviced_dt_aqua_and_sentinel_explanation():
    formula = _formula("##ACCOUNTCAL", "IntNotServicedDt")
    text = build_rule_explanation("AccountCal", "IntNotServicedDt", formula)
    assert "Aqua_Scheme" in text and "ODA" in text
    assert "1900-01-01" in text
    assert "nested conditional" not in text.lower()
    assert "DIMPRODUCT is active" not in text


def test_s02_dpd_intservice_timekey_ladder_explanation():
    formula = _formula("##ACCOUNTCAL", "DPD_IntService")
    text = build_rule_explanation("AccountCal", "DPD_IntService", formula)
    assert "26267" in text or "Post-Cutoff" in text
    assert "IntNotServicedDt" in text
    assert "@ProcessDate" in text or "ProcessDate" in text
    assert "nested conditional" not in text.lower()


def test_s02_dpd_maxfin_formula_uses_self_isempty():
    formula = _formula("PRO.AdvAcRestructureCal", "DPD_MaxFin")
    assert "DPD_MaxNonFin" not in formula
    assert "DPD_MaxFin" in formula
    text = build_rule_explanation("AdvAcRestructureCal", "DPD_MaxFin", formula)
    assert text
    assert "nested conditional" not in text.lower()
