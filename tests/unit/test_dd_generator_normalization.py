from app.derivation.dd_generator import _normalize_sql_style_syntax
from app.grammar.validator import validate_expression


def test_normalizes_bare_equals_to_double_equals():
    expr = _normalize_sql_style_syntax('IF(STATUS = "OPEN")THEN(1)ELSE(0)')
    assert expr == 'IF(STATUS == "OPEN")THEN(1)ELSE(0)'
    assert validate_expression(expr).valid


def test_normalizes_single_quoted_strings_to_double_quoted():
    expr = _normalize_sql_style_syntax("IF(REGION_CODE == 'APAC')THEN('YES')ELSE('NO')")
    assert expr == 'IF(REGION_CODE == "APAC")THEN("YES")ELSE("NO")'
    assert validate_expression(expr).valid


def test_normalizes_both_bare_equals_and_single_quotes_together():
    expr = _normalize_sql_style_syntax(
        "IF(SCHEME_TYPE = 'ODA' AND FLAG = 'Y')THEN('YES')ELSE('NO')"
    )
    assert expr == 'IF(SCHEME_TYPE == "ODA" AND FLAG == "Y")THEN("YES")ELSE("NO")'
    assert validate_expression(expr).valid


def test_leaves_existing_double_equals_and_double_quotes_untouched():
    expr = 'IF("A"."X" == "B")THEN(1)ELSE(0)'
    assert _normalize_sql_style_syntax(expr) == expr


def test_leaves_not_equals_greater_equal_less_equal_untouched():
    expr = 'IF(("A"."X" != "B") AND ("A"."Y" >= 1) AND ("A"."Z" <= 9))THEN(1)ELSE(0)'
    assert _normalize_sql_style_syntax(expr) == expr
    assert validate_expression(expr).valid


def test_does_not_touch_equals_sign_inside_a_double_quoted_string():
    expr = _normalize_sql_style_syntax('IF(ISNOTEMPTY("A"."NOTE"))THEN("a=b")ELSE("c")')
    assert expr == 'IF(ISNOTEMPTY("A"."NOTE"))THEN("a=b")ELSE("c")'