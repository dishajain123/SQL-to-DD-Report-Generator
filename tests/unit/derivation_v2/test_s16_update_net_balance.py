"""Structural regressions for PRO UpdateNetBalance_AccountWise (S16)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET, optimize_expression_ast
from app.derivation.v2.phase3_ast_generator import _is_cross_column_null_default_guard
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.parsing.write_inventory_scan import read_sql_file

_REPO = Path(__file__).resolve().parents[3]
_S16 = (
    _REPO
    / "samples/sql/PRO_SPs_Sequenced/24_S16_PRO.UpdateNetBalance_AccountWise.StoredProcedure.sql"
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _formula_for(column: str) -> tuple[object, str]:
    sql = read_sql_file(_S16)
    row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_net_balance_keeps_write_off_override_and_is_valid():
    row, formula = _formula_for("NetBalance")
    assert not getattr(row, "validation_errors", None), row.validation_errors
    _assert_hygiene(formula, "NetBalance")
    # ``SET NetBalance = 0 ... ISNULL(WriteOffAmount,0) > 0`` is a value
    # predicate, never a cross-column null default, so it must not be dropped.
    assert "WriteOffAmount" in formula
    assert "PrincOutStd" in formula
    assert "Balance" in formula
    assert "COALESCE(COALESCE(" not in formula.replace(" ", "")


def test_net_balance_exception_status_override_present():
    _, formula = _formula_for("NetBalance")
    assert "StatusType" in formula


def test_addl_provision_reads_net_balance_and_is_valid():
    row, formula = _formula_for("AddlProvision")
    assert not getattr(row, "validation_errors", None), row.validation_errors
    _assert_hygiene(formula, "AddlProvision")
    assert "NetBalance" in formula
    assert "AddlProvisionPer" in formula


def test_isnull_ordering_predicate_is_not_a_null_default_guard():
    cond = "B.ASSETCLASSGROUP='NPA' and ISNULL(WriteOffAmount,0) > 0"
    assert not _is_cross_column_null_default_guard(cond, "NetBalance")
    assert not _is_cross_column_null_default_guard("ISNULL(WriteOffAmount,0)<>0", "NetBalance")
    # Equality-style null-default fills keep their established behaviour.
    assert _is_cross_column_null_default_guard("ISNULL(WriteOffAmount,0)=0", "NetBalance")
    assert _is_cross_column_null_default_guard("WriteOffAmount IS NULL", "NetBalance")


def test_max_with_null_safe_operand_has_no_redundant_coalesce():
    col = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "Balance"}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    null_safe = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col, zero]}
    ast = {"type": "FUNCTION_CALL", "function_name": "MAX", "arguments": [null_safe, zero]}
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="NetBalance")
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="NetBalance"
    )
    assert not _AGG_RE.search(formula)
    assert formula.replace(" ", "").count("COALESCE(") == 2, formula
    assert "COALESCE(COALESCE(" not in formula.replace(" ", "")
    assert validate_expression(formula).passed
