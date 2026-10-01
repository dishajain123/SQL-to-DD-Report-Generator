"""Flatten join + zero-default ref-period UPDATE chains (Reference_Period_Calculation)."""
from __future__ import annotations

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

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
        assert validate_expression(expr).passed, validate_expression(expr).errors
