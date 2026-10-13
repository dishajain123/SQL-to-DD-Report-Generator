"""Rule explanations for PRO Marking_NPA_Reason_NPAAccount (S13)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S13 = _REPO / "samples/sql/PRO_SPs_Sequenced/21_S13_PRO.Marking_NPA_Reason_NPAAccount.StoredProcedure.sql"


def _formula() -> str:
    sql = _S13.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(sql, "AccountCal", "NPA_Reason", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if not formula:
        pytest.skip("no NPA_Reason formula")
    return formula


def test_s13_npa_reason_explanation_not_asset_norm_ladder():
    formula = _formula()
    text = build_rule_explanation("AccountCal", "NPA_Reason", formula)
    assert "PERCOLATION BY UCIC_ID" in text
    assert "DEGRADE BY INT NOT SERVICED" in text
    assert "Default Fallback" in text
    assert "SARFAESI" not in text
    assert "ALWYS_STD" not in text or "Degradation Overrides" not in text
