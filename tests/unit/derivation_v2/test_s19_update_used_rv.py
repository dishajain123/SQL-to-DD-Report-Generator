"""Structural regressions for PRO UpdateUsedRV (S19)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.phase2_mutation_folder import fold_column_mutations
from app.derivation.v2.phase1_lineage import build_lineage_map
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S19 = _REPO / "samples/sql/PRO_SPs_Sequenced/27_S19_PRO.UpdateUsedRV.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _sql() -> str:
    return _S19.read_text(encoding="utf-8", errors="replace")


def _formula(entity: str, column: str) -> tuple[object, str, list]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula, debug.get("mutations") or []


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_s19_usedrv_reset_then_join_case_under_budget():
    row, formula, mutations = _formula("##AccountCal", "USEDRV")
    if not formula:
        row, formula, mutations = _formula("AccountCal", "USEDRV")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "USEDRV")
    assert len(mutations) == 2
    assert mutations[0].get("guarded") is False
    assert mutations[1].get("guarded") is True
    assert mutations[1].get("effective_condition")
    upper = formula.upper()
    assert "LOS" in upper
    assert "APPRRV" in upper or "ApprRV" in formula
    assert "NETBALANCE" in upper or "netbalance" in formula.lower()
    assert "ASSETCLASS" in upper or "DimAssetClass" in formula
    assert "THEN(0)" in formula.replace(" ", "") or 'THEN(0)' in formula
    assert formula.count("IF(") >= 2


def test_s19_usedrv_join_guard_preserves_zero_outside_match():
    """Rows with no DimAssetClass match keep the first UPDATE's zero."""
    row, formula, _ = _formula("##AccountCal", "USEDRV")
    if not formula:
        row, formula, _ = _formula("AccountCal", "USEDRV")
    assert formula
    compact = formula.replace(" ", "")
    assert compact.endswith("ELSE(0)") or "ELSE(0)" in compact


def test_s19_acl_running_process_status_try_and_catch():
    row, formula, _ = _formula("ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    _assert_hygiene(formula, "COMPLETED")
    assert "UpdateUsedRV" in formula
    catch = row.exception_handler_expression or ""
    assert catch
    assert "UpdateUsedRV" in catch
    assert 'THEN("N")' in catch or 'THEN("n")' in catch.lower()


def test_inner_join_key_only_update_mutation_is_guarded():
    sql = """
    UPDATE AccountCal SET Flag=0;
    UPDATE AccountCal SET Flag=2
    FROM AccountCal A INNER JOIN DimAssetClass C ON C.AssetClassAlt_Key = A.FinalAssetClassAlt_Key;
    """
    lineage = build_lineage_map(sql)
    mutations = fold_column_mutations(sql, "AccountCal", "Flag", lineage, None)
    assert len(mutations) == 2
    assert mutations[1].guarded is True
    assert mutations[1].join_filter_condition or mutations[1].effective_condition
