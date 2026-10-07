"""Structural regressions for PRO UpdationTotalProvision (S23)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import (
    FORMULA_CHAR_BUDGET,
    _is_never_null,
    _strip_redundant_coalesce,
    optimize_expression_ast,
)
from app.derivation.v2.phase2_mutation_folder import _qualify_foreign_bare_columns
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.parsing.write_inventory_scan import read_sql_file

_REPO = Path(__file__).resolve().parents[3]
_S23 = (
    _REPO / "samples/sql/PRO_SPs_Sequenced/31_S23_PRO.UpdationTotalProvision.StoredProcedure.sql"
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_COALESCE_IF = re.compile(r"COALESCE\(\s*IF\s*\(", re.I)


@pytest.mark.parametrize("column", ["TotalProvision", "BankTotalProvision", "RBITotalProvision"])
def test_s23_account_cal_totals_are_valid_budgeted_and_hygienic(column):
    sql = read_sql_file(_S23)
    row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert not getattr(row, "validation_errors", None), row.validation_errors
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert not _AGG_RE.search(formula), formula[:500]
    assert not _COALESCE_IF.search(formula), formula[:500]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s23_total_provision_reads_restructure_column_from_joined_table():
    sql = read_sql_file(_S23)
    row, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert "RestructureProvision" in formula
    assert '"AccountCal"."RestructureProvision"' not in formula


def test_foreign_bare_column_is_qualified_only_when_joined_table_owns_it():
    alias_map = {"A": "##ACCOUNTCAL", "B": "AdvAcRestructureCal", "D": "DimParameter"}
    written = {
        "ADVACRESTRUCTURECAL": {"RESTRUCTUREPROVISION", "SECUREDPROVISION"},
        "##ACCOUNTCAL": {"TOTALPROVISION"},
    }
    text = "ISNULL(TotalProvision,0)+ISNULL(RestructureProvision,0)+ISNULL(A.X,0) + 'RestructureProvision'"
    out = _qualify_foreign_bare_columns(text, alias_map, ["##ACCOUNTCAL"], written)
    assert "ISNULL(B.RestructureProvision,0)" in out
    assert "ISNULL(TotalProvision,0)" in out
    assert "'RestructureProvision'" in out  # string literal untouched
    # No foreign evidence -> unchanged.
    assert _qualify_foreign_bare_columns(text, alias_map, ["##ACCOUNTCAL"], {}) == text


def test_redundant_coalesce_around_non_null_operand_is_stripped():
    col = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "X"}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    null_safe = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col, zero]}
    wrapped = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [null_safe, zero]}
    assert _strip_redundant_coalesce(wrapped) == null_safe
    # A nullable column keeps its COALESCE.
    plain = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col, zero]}
    assert _strip_redundant_coalesce(plain) == plain
    total = {"type": "BINARY_OP", "operator": "+", "left": null_safe, "right": null_safe}
    assert _is_never_null(total)
    assert not _is_never_null(col)


def test_clamped_sum_compared_with_column_has_no_coalesce_wrapped_if():
    col = lambda n: {"type": "COLUMN_REF", "entity": "AccountCal", "column": n}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    safe = lambda n: {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col(n), zero]}
    total = {"type": "BINARY_OP", "operator": "+", "left": safe("A"), "right": safe("B")}
    clamp = {"type": "FUNCTION_CALL", "function_name": "MAX", "arguments": [total, zero]}
    guard = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [clamp, zero]},
        "right": col("NetBalance"),
    }
    ast = {"type": "IF_THEN_ELSE", "condition": guard, "then_branch": col("NetBalance"), "else_branch": clamp}
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="TotalProvision")
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="TotalProvision"
    )
    assert not _AGG_RE.search(formula), formula
    assert not _COALESCE_IF.search(formula), formula
    assert validate_expression(formula).passed, formula
