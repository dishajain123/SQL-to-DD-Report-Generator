"""Rule Explanation + export hygiene for PRO Marking_InMonthMark (S03)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_export import dd_row_to_dict
from app.report.dd_rule_explanation import build_rule_explanation
from app.models.core import (
    ColumnType,
    DDRow,
    DDStatus,
    DerivationOption,
)

_REPO = Path(__file__).resolve().parents[2]
_S03 = _REPO / "samples/sql/PRO_SPs_Sequenced/08_S03_PRO.Marking_InMonthMark_Customer_Account_level.StoredProcedure.sql"


def _sql() -> str:
    return _S03.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> str:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert formula, getattr(row, "validation_errors", None)
    return formula


def test_s03_inmonthmark_formula_uses_physical_entities_not_aliases():
    formula = _formula("##CUSTOMERCAL", "INMONTHMARK")
    upper = formula.upper()
    assert "CUSTOMERCAL" in upper and "ACCOUNTCAL" in upper
    assert '."A".' not in upper and '."B".' not in upper


def test_s03_inmonthmark_rule_explanation_daily_eom_ladder():
    formula = _formula("##CUSTOMERCAL", "INMONTHMARK")
    text = build_rule_explanation("CUSTOMERCAL", "INMONTHMARK", formula)
    assert text
    lower = text.lower()
    assert "daily non-eom" in lower or "prioritized conditions" in lower
    assert "a' equals 'b'" not in lower
    assert "nested conditional" not in lower
    assert "tree(" not in lower


def test_s03_acl_completed_retains_existing():
    formula = _formula("ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert "Marking_InMonthMark_Customer_Account_level" in text
    assert "retains existing COMPLETED" in text
    assert "sets to ACLRUNNINGPROCESSSTATUS" not in text.replace(" ", "").upper()


def test_s03_dd_export_row_inmonthmark_explanation():
    formula = _formula("##CUSTOMERCAL", "INMONTHMARK")
    row = DDRow(
        entity_name="CUSTOMERCAL",
        column_name="INMONTHMARK",
        column_type=ColumnType.PHYSICAL,
        derivation_option=DerivationOption.FORMULA_EXPRESSION,
        display_derivation_expression=formula,
        effective_start_date=__import__("datetime").date(1900, 1, 1),
        status=DDStatus.PENDING_REVIEW,
        data_type="String",
    )
    exported = dd_row_to_dict(row)
    expl = exported["rule_explanation"]
    assert "Daily Non-EOM" in expl or "prioritized conditions" in expl.lower()
    assert "nested conditional" not in expl.lower()
