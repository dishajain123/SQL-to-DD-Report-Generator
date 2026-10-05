"""Derivation Option classification and @-parameter display formatting."""
from __future__ import annotations

from app.derivation.derivation_option import (
    classify_derivation_option,
    format_expression_syntax,
    is_static_seed,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.models.core import DerivationOption


def test_format_expression_syntax_strips_quoted_parameters():
    assert format_expression_syntax('IF("@TIMEKEY" > 26267)THEN("Y")') == (
        'IF(@TIMEKEY > 26267)THEN("Y")'
    )
    assert format_expression_syntax('"@ProcessDate"') == "@ProcessDate"


def test_everything_classifies_as_formula_expression():
    for expr in ('"N"', "0", '"CUSTOMERBASICDETAIL"."ParentBranchCode"', "@TIMEKEY"):
        assert classify_derivation_option(expr) == DerivationOption.FORMULA_EXPRESSION


def test_static_seed_detection():
    assert is_static_seed('"N"')
    assert is_static_seed("0")
    assert is_static_seed("1.5")
    assert not is_static_seed('"CUSTOMERBASICDETAIL"."ParentBranchCode"')
    assert not is_static_seed('IF("A" == "B")THEN("Y")ELSE("N")')


def test_classify_formula_expression_for_conditionals():
    expr = 'IF("AccountCal"."ASSET_NORM" == "ALWYS_STD")THEN("CONDI_STD")ELSE("AccountCal"."ASSET_NORM")'
    assert classify_derivation_option(expr) == DerivationOption.FORMULA_EXPRESSION


def test_generate_for_sql_display_uses_unquoted_parameters():
    row, debug = generate_for_sql(
        "UPDATE PRO.CustomerCal SET EffectiveFromTimeKey = @TIMEKEY;",
        "CustomerCal",
        "EffectiveFromTimeKey",
        llm_client=None,
    )
    assert row.display_derivation_expression
    assert '"@TIMEKEY"' not in row.display_derivation_expression
    assert "@TIMEKEY" in row.display_derivation_expression
    assert row.derivation_option == DerivationOption.FORMULA_EXPRESSION
    assert '"@TIMEKEY"' in (debug.get("formula") or "")
