"""Flatten join + zero-default ref-period UPDATE chains (Reference_Period_Calculation)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET, optimize_expression_ast
from app.derivation.v2.phase2_mutation_folder import _try_flatten_join_assignment_zero_default
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S01 = _REPO / "samples/sql/PRO_SPs_Sequenced/06_S01_PRO.Reference_Period_Calculation.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_CHAR_CAP = FORMULA_CHAR_BUDGET
_REF_COLS = (
    "REFPERIODOVERDUE",
    "REFPERIODOVERDRAWN",
    "REFPERIODNOCREDIT",
    "REFPERIODINTSERVICE",
    "REFPERIODSTKSTATEMENT",
    "REFPERIODREVIEW",
)

_REF_PERIOD_SQL = """
UPDATE ##ACCOUNTCAL
SET
    REFPERIODOVERDUE=0,
    REFPERIODOVERDRAWN=0,
    REFPERIODNOCREDIT=0,
    REFPERIODINTSERVICE=0;

UPDATE A SET
    REFPERIODOVERDUE=B.ReferencePeriod,
    REFPERIODOVERDRAWN=B.ReferencePeriod,
    REFPERIODNOCREDIT=B.ReferencePeriod,
    REFPERIODINTSERVICE=B.ReferencePeriod
FROM ##ACCOUNTCAL A
    INNER JOIN ADVACBASICDETAIL B
        ON A.ACCOUNTENTITYID=B.ACCOUNTENTITYID
        AND B.EFFECTIVEFROMTIMEKEY<=@TIMEKEY AND B.EffectiveToTimeKey>=@TIMEKEY;

UPDATE A SET A.RefPeriodOverdue=91 FROM ##ACCOUNTCAL A WHERE RefPeriodOverdue=0;
UPDATE A SET A.RefPeriodOverDrawn=91 FROM ##ACCOUNTCAL A WHERE RefPeriodOverDrawn=0;
UPDATE A SET A.RefPeriodNoCredit=91 FROM ##ACCOUNTCAL A WHERE RefPeriodNoCredit=0;
UPDATE A SET A.RefPeriodIntService=91 FROM ##ACCOUNTCAL A WHERE RefPeriodIntService=0;
"""

_STK_REVIEW_SQL = """
UPDATE ##ACCOUNTCAL
SET REFPERIODSTKSTATEMENT=0, REFPERIODREVIEW=0;

UPDATE A SET A.RefPeriodStkStatement=181 FROM ##ACCOUNTCAL A WHERE RefPeriodStkStatement=0;
UPDATE A SET A.RefPeriodReview=181 FROM ##ACCOUNTCAL A WHERE RefPeriodReview=0;
"""


def test_ref_period_columns_flatten_join_and_default():
    for column in (
        "REFPERIODOVERDUE",
        "REFPERIODOVERDRAWN",
        "REFPERIODNOCREDIT",
        "REFPERIODINTSERVICE",
    ):
        row, _ = generate_for_sql(_REF_PERIOD_SQL, "##ACCOUNTCAL", column, llm_client=None)
        expr = row.display_derivation_expression or ""
        assert expr, f"missing formula for {column}"
        assert "IF(IF(" not in expr, expr
        assert "COALESCE(" in expr, expr
        assert "91" in expr, expr
        assert "ReferencePeriod" in expr or "REFERENCEPERIOD" in expr.upper(), expr
        assert "EFFECTIVEFROMTIMEKEY" in expr.upper(), expr
        assert _INLINE_IF_CMP.search(expr) is None, expr
        assert not _AGG_RE.search(expr), expr
        assert validate_expression(expr).passed, validate_expression(expr).errors


def test_ref_period_stk_and_review_use_column_zero_guard_not_literal_tautology():
    cases = (
        ("REFPERIODSTKSTATEMENT", "181"),
        ("REFPERIODREVIEW", "181"),
    )
    for column, default in cases:
        row, _ = generate_for_sql(_STK_REVIEW_SQL, "##ACCOUNTCAL", column, llm_client=None)
        expr = row.display_derivation_expression or ""
        assert expr, f"missing formula for {column}"
        assert "0 == 0" not in expr.replace(" ", ""), expr
        assert column in expr.upper(), expr
        assert default in expr, expr
        assert "IF(IF(" not in expr, expr
        assert not _AGG_RE.search(expr), expr
        assert validate_expression(expr).passed, validate_expression(expr).errors


def _join_zero_nodes():
    join = {
        "type": "BINARY_OP",
        "operator": "<=",
        "left": {
            "type": "COLUMN_REF",
            "entity": "AdvAcBasicDetail",
            "column": "EffectiveFromTimeKey",
        },
        "right": {"type": "VARIABLE_REF", "name": "@TIMEKEY"},
    }
    src = {
        "type": "COLUMN_REF",
        "entity": "AdvAcBasicDetail",
        "column": "ReferencePeriod",
    }
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    default = {"type": "LITERAL", "value_type": "NUMBER", "value": 91}
    return join, src, zero, default


def test_optimize_flattens_prior_equals_zero_before_distributing_if():
    """Phase3 runs optimize before prune; flatten must beat comparison distribution."""
    join, src, zero, default = _join_zero_nodes()
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {
                "type": "IF_THEN_ELSE",
                "condition": join,
                "then_branch": src,
                "else_branch": zero,
            },
            "right": zero,
        },
        "then_branch": default,
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": join,
            "then_branch": src,
            "else_branch": zero,
        },
    }
    optimized = optimize_expression_ast(
        ast, target_entity="AccountCal", target_column="RefPeriodOverdue"
    )
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="RefPeriodOverdue"
    )
    assert "IF(IF(" not in formula
    assert "COALESCE(" in formula
    assert "91" in formula
    assert _INLINE_IF_CMP.search(formula) is None
    assert not _AGG_RE.search(formula)
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_distributed_join_zero_check_still_flattens():
    """4X distribution of ``(IF(join) THEN src ELSE 0)==0`` must still flatten."""
    join, src, zero, default = _join_zero_nodes()
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "IF_THEN_ELSE",
            "condition": join,
            "then_branch": {
                "type": "BINARY_OP",
                "operator": "==",
                "left": src,
                "right": zero,
            },
            "else_branch": {
                "type": "BINARY_OP",
                "operator": "==",
                "left": zero,
                "right": zero,
            },
        },
        "then_branch": default,
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": join,
            "then_branch": src,
            "else_branch": zero,
        },
    }
    flat = _try_flatten_join_assignment_zero_default(ast)
    assert flat is not None
    formula = compile_ast_to_4x_string(
        flat, target_entity="AccountCal", target_column="RefPeriodOverdue"
    )
    assert "IF(IF(" not in formula
    assert "COALESCE(" in formula
    assert "91" in formula
    assert "MIN(" not in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s01_file_exportable_ref_period_formulas_are_valid():
    sql = _S01.read_text(encoding="utf-8", errors="replace")
    for column in _REF_COLS:
        row, debug = generate_for_sql(sql, "##ACCOUNTCAL", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert formula, f"{column} empty: {row.validation_errors}"
        assert len(formula) <= _CHAR_CAP, f"{column} length {len(formula)}"
        assert not row.validation_errors, f"{column}: {row.validation_errors}"
        assert "IF(IF(" not in formula, formula[:400]
        assert _INLINE_IF_CMP.search(formula) is None, formula[:400]
        assert not _AGG_RE.search(formula), formula[:400]
        assert "AGRI366" not in formula.upper()
        assert validate_expression(formula).passed, validate_expression(formula).errors
        upper = formula.upper()
        if column in {
            "REFPERIODOVERDUE",
            "REFPERIODOVERDRAWN",
            "REFPERIODNOCREDIT",
            "REFPERIODINTSERVICE",
        }:
            assert "91" in formula, column
            assert "REFERENCEPERIOD" in upper, column
            assert "COALESCE(" in formula, column
        else:
            assert "181" in formula, column
            assert "0 == 0" not in formula.replace(" ", ""), column


def test_s01_acl_completed_scoped_to_reference_period():
    sql = _S01.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(
        sql, "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    formula = debug.get("formula") or ""
    assert formula
    assert "Reference_Period_Calculation" in formula
    assert 'THEN("Y")' in formula.replace(" ", "") or 'THEN("Y")' in formula
    assert "MIN(" not in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors
    catch = row.exception_handler_expression or debug.get("exception_handler_formula") or ""
    assert catch
    assert "Reference_Period_Calculation" in catch
    assert "THEN(\"N\")" in catch or 'THEN("N")' in catch
