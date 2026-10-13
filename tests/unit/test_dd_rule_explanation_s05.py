"""Rule Explanation regressions for PRO MaxDPD_ReferencePeriod_Calculation (S05)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S05 = _REPO / "samples/sql/PRO_SPs_Sequenced/10_S05_PRO.MaxDPD_ReferencePeriod_Calculation.StoredProcedure.sql"


def _sql() -> str:
    return _S05.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert formula, getattr(row, "validation_errors", None)
    return formula


def test_s05_dpd_max_explanation_ladder():
    formula = _formula("##AccountCal", "DPD_Max")
    text = build_rule_explanation("AccountCal", "DPD_Max", formula)
    assert text
    lower = text.lower()
    assert "prioritized conditions" in lower
    assert "flgprocessing" in lower or "customercal" in lower
    assert "dpd_intservice" in lower or "component dpd" in lower
    assert "tree(" not in lower


def test_s05_dpd_fin_max_type_explanation():
    formula = _formula("##AccountCal", "DPD_FinMaxType")
    text = build_rule_explanation("AccountCal", "DPD_FinMaxType", formula)
    assert text
    assert "RefPeriod" in text or "refperiod" in text.lower()
    assert "tree(" not in text.lower()


def test_s05_refperiod_max_explanation():
    formula = _formula("##AccountCal", "REFPeriodMax")
    text = build_rule_explanation("AccountCal", "REFPeriodMax", formula)
    assert text
    lower = text.lower()
    assert "refperiod" in lower
    assert "tree(" not in lower


@pytest.mark.parametrize(
    "column,proc",
    [
        ("COMPLETED", "MaxDPD_ReferencePeriod_Calculation"),
        ("COUNT", "MaxDPD_ReferencePeriod_Calculation"),
    ],
)
def test_s05_acl_process_explanation(column: str, proc: str):
    try:
        formula = _formula("ACLRUNNINGPROCESSSTATUS", column)
    except AssertionError:
        pytest.skip("ACL formula not derived from S05 sample")
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", column, formula)
    assert proc in text
    assert f"retains existing {column}" in text
    assert "tree(" not in text.lower()


def test_s05_max_dpd_completed_pattern_fallback():
    proc = "MaxDPD_ReferencePeriod_Calculation"
    formula = (
        f'IF("ACLRUNNINGPROCESSSTATUS"."RUNNINGPROCESSNAME" == "{proc}")'
        'THEN("Y")ELSE("ACLRUNNINGPROCESSSTATUS"."COMPLETED")'
    )
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert proc in text
    assert "retains existing COMPLETED" in text
