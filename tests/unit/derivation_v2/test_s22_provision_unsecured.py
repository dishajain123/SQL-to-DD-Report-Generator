"""Structural regressions for PRO UpdationProvisionComputationUnSecured (S22)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import (
    FORMULA_CHAR_BUDGET,
    _hoist_if_from_arithmetic,
    optimize_expression_ast,
)
from app.derivation.v2.phase3_ast_generator import (
    _parenthesize_top_level_case,
    parse_sql_expression_to_ast,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.parsing.write_inventory_scan import read_sql_file

_REPO = Path(__file__).resolve().parents[3]
_S22 = (
    _REPO
    / "samples/sql/PRO_SPs_Sequenced/30_S22_PRO.UpdationProvisionComputationUnSecured.StoredProcedure.sql"
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
# A comparison operator directly followed/preceded by an inline IF(...) value.
_IF_IN_COMPARISON = re.compile(r"[<>!=]=?\s*IF\s*\(")


@pytest.mark.parametrize(
    "column", ["UnsecuredAmt", "ProvUnsecured", "BankProvUnsecured", "RBIProvUnsecured"]
)
def test_s22_columns_are_valid_budgeted_and_aggregate_free(column):
    sql = read_sql_file(_S22)
    row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert not getattr(row, "validation_errors", None), row.validation_errors
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert not _AGG_RE.search(formula), formula[:500]
    assert "__UNSUPPORTED_SQL__" not in formula
    assert not _IF_IN_COMPARISON.search(formula), formula[:500]
    assert "NetBalance" in formula
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s22_provision_keeps_whole_case_multiplier_as_one_operand():
    sql = read_sql_file(_S22)
    for column in ("ProvUnsecured", "BankProvUnsecured"):
        row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert "ProvPerUnSecured" in formula
        # The ``/ 100`` belongs to the ELSE arm of the inner CASE only.
        assert "ProvisionName" in formula or "PROVISIONNAME" in formula.upper()


def test_case_after_operator_is_parenthesized_not_split_inside():
    text = "(A.X) * CASE WHEN D.N='c' THEN ISNULL(A.P,0) ELSE ISNULL(D.U,0)/100 END"
    assert _parenthesize_top_level_case(text).endswith("(CASE WHEN D.N='c' THEN ISNULL(A.P,0) ELSE ISNULL(D.U,0)/100 END)")
    whole = "CASE WHEN a=1 THEN 2 ELSE 3 END"
    assert _parenthesize_top_level_case(whole) == whole
    ast = parse_sql_expression_to_ast(text, default_entity="AccountCal", target_column="ProvUnsecured")
    assert ast["type"] == "BINARY_OP" and ast["operator"] == "*"
    assert ast["right"]["type"] == "IF_THEN_ELSE"


def test_hoist_if_from_arithmetic_and_no_if_inside_comparison():
    col = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "X"}
    cond = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "F"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    lit = lambda v: {"type": "LITERAL", "value_type": "NUMBER", "value": v}
    product = {
        "type": "BINARY_OP",
        "operator": "*",
        "left": col,
        "right": {"type": "IF_THEN_ELSE", "condition": cond, "then_branch": lit(2), "else_branch": lit(3)},
    }
    hoisted = _hoist_if_from_arithmetic(product)
    assert hoisted["type"] == "IF_THEN_ELSE"
    assert hoisted["then_branch"]["operator"] == "*"

    clamp = {"type": "FUNCTION_CALL", "function_name": "MAX", "arguments": [product, lit(0)]}
    optimized = optimize_expression_ast(clamp, target_entity="AccountCal", target_column="P")
    formula = compile_ast_to_4x_string(optimized, target_entity="AccountCal", target_column="P")
    assert not _AGG_RE.search(formula)
    assert "* IF(" not in formula and "*IF(" not in formula
    assert validate_expression(formula).passed, formula
