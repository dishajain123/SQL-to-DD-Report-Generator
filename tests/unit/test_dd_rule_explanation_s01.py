"""Rule explanations for PRO Reference_Period_Calculation (S01)."""
from __future__ import annotations

from pathlib import Path

import pytest

from app.derivation.v2.pipeline import generate_for_sql
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[2]
_S01 = _REPO / "samples/sql/PRO_SPs_Sequenced/06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql"

_ADV_COLS = (
    "REFPERIODOVERDUE",
    "REFPERIODOVERDRAWN",
    "REFPERIODNOCREDIT",
    "REFPERIODINTSERVICE",
)
_ZERO_COLS = ("REFPERIODSTKSTATEMENT", "REFPERIODREVIEW")


def _sql() -> str:
    return _S01.read_text(encoding="utf-8", errors="replace")


def _formula(column: str) -> str:
    for ent in ("##ACCOUNTCAL", "ACCOUNTCAL", "##AccountCal", "AccountCal"):
        row, debug = generate_for_sql(_sql(), ent, column, llm_client=None)
        formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
        if formula:
            return formula
    pytest.skip(f"no formula for {column}")


def test_s01_qualified_column_ref_not_string_literal():
    from app.report.condition_explainer import (
        _column_ref_is_string_literal,
        _export_operand,
        _parse_expression_tree,
    )

    expr = '"ADVACBASICDETAIL"."ReferencePeriod"'
    tree = _parse_expression_tree(expr)
    assert tree is not None
    assert not _column_ref_is_string_literal(tree)
    assert _export_operand(tree) == "ADVACBASICDETAIL.ReferencePeriod"


@pytest.mark.parametrize("column", _ADV_COLS)
def test_s01_adv_ref_period_explanation(column: str):
    formula = _formula(column)
    text = build_rule_explanation("AccountCal", column, formula)
    assert "ADVACBASICDETAIL.ReferencePeriod" in text
    assert "defaults to 91" in text
    assert f"Sets {column} to 'ADVACBASICDETAIL'" not in text
    assert "nested conditional" not in text.lower()


@pytest.mark.parametrize("column", _ZERO_COLS)
def test_s01_zero_default_ref_period_explanation(column: str):
    formula = _formula(column)
    text = build_rule_explanation("AccountCal", column, formula)
    assert "181" in text
    assert "ACCOUNTCAL." in text
    assert "'ACCOUNTCAL' equals 0" not in text
    assert "retains existing" in text
