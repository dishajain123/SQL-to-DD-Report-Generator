"""Structural regressions for PRO UPDATE_NPA_TYPE (S14)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import is_staging_derivation_entity
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S14 = _REPO / "samples/sql/PRO_SPs_Sequenced/22_S14_PRO.UPDATE_NPA_TYPE.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _sql() -> str:
    return _S14.read_text(encoding="utf-8", errors="replace")


def _formula_for(entity: str, column: str) -> tuple[object, str]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert '"DA".' not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_account_cal_npatype_is_syntax_valid_under_budget():
    row, formula = _formula_for("##AccountCal", "NpaType")
    if not formula:
        row, formula = _formula_for("AccountCal", "NpaType")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "NpaType")
    assert 'THEN("REGULAR")' in formula
    assert 'THEN("STICKY")' in formula
    assert 'THEN("MULTIPLE")' in formula
    assert formula.find('THEN("REGULAR")') < formula.find('THEN("STICKY")')
    assert formula.find('THEN("STICKY")') < formula.find('THEN("MULTIPLE")')
    assert formula.count('THEN("REGULAR")') == 1
    assert formula.count('THEN("STICKY")') == 1
    assert formula.count('THEN("MULTIPLE")') == 1
    compact = formula.replace(" ", "")
    assert 'THEN("Y")ELSE("Y")' not in compact
    assert "VisionPLUS" in formula
    assert "AssetClassGroup" in formula or '=="NPA"' in compact
    assert "SecuritizationGroup" not in formula
    # CASE first-match: CD=2 DPD 1-29 is STICKY; later MULTIPLE copies are dead.
    sticky_to_multiple = formula.split('THEN("STICKY")', 1)[1].split('THEN("MULTIPLE")', 1)[0]
    assert ">=1" not in sticky_to_multiple.replace(" ", "")
    assert "<=29" not in sticky_to_multiple.replace(" ", "")


def test_s14_commented_securitization_update_is_ignored():
    row, debug = generate_for_sql(_sql(), "##AccountCal", "NpaType", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if not formula:
        row, debug = generate_for_sql(_sql(), "AccountCal", "NpaType", llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
    mutations = debug.get("mutations") or []
    assert formula
    assert "SecuritizationGroup" not in formula
    assert all("SecuritizationGroup" not in str(m) for m in mutations)


def test_s14_does_not_treat_account_cal_as_staging():
    assert not is_staging_derivation_entity("##AccountCal")
    assert not is_staging_derivation_entity("AccountCal")
    assert is_staging_derivation_entity("#TEMPTABLEDPD")
    assert is_staging_derivation_entity("TEMPTABLE1")
    assert is_staging_derivation_entity("CTE_X")
    assert is_staging_derivation_entity("FOO_BKUP")
