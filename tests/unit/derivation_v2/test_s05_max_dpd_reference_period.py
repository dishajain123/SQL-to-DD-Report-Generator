"""Structural regressions for PRO MaxDPD_ReferencePeriod_Calculation (S05)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.report.dd_rule_explanation import build_rule_explanation

_REPO = Path(__file__).resolve().parents[3]
_S05 = _REPO / "samples/sql/PRO_SPs_Sequenced/10_S05_PRO.MaxDPD_ReferencePeriod_Calculation.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)


def _sql() -> str:
    return _S05.read_text(encoding="utf-8", errors="replace")


def _row_formula(entity: str, column: str) -> tuple[object, str]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:400]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_s05_dpd_max_case_ladder():
    row, formula = _row_formula("##AccountCal", "DPD_Max")
    if not formula:
        row, formula = _row_formula("AccountCal", "DPD_Max")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "DPD_Max")
    upper = formula.upper()
    compact = upper.replace(" ", "").replace("_", "")
    for token in (
        "DPDINTSERVICE",
        "DPDNOCREDIT",
        "DPDOVERDRAWN",
        "DPDOVERDUE",
        "DPDRENEWAL",
        "DPDSTOCKSTMT",
    ):
        assert token in compact, f"missing {token} in {formula[:500]}"


def test_s05_dpd_fin_max_type_labels():
    row, formula = _row_formula("##AccountCal", "DPD_FinMaxType")
    if not formula:
        row, formula = _row_formula("AccountCal", "DPD_FinMaxType")
    assert formula, getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "DPD_FinMaxType")
    assert "RefPeriod" in formula or "REFPERIOD" in formula.upper()


def test_s05_refperiod_max_from_temp_chain():
    row, formula = _row_formula("##AccountCal", "REFPeriodMax")
    if not formula:
        row, formula = _row_formula("AccountCal", "REFPeriodMax")
    assert formula, getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "REFPeriodMax")
    upper = formula.upper()
    assert "REFPERIOD" in upper
    assert "FLGPROCESSING" in upper or "FLGDEG" in upper or "FACILITYTYPE" in upper


def test_s05_acl_completed_explanation():
    row, formula = _row_formula("ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    assert formula
    text = build_rule_explanation("ACLRUNNINGPROCESSSTATUS", "COMPLETED", formula)
    assert "MaxDPD_ReferencePeriod_Calculation" in text or "MaxDPD" in text
    assert "retains existing COMPLETED" in text
    assert "tree(" not in text.lower()
