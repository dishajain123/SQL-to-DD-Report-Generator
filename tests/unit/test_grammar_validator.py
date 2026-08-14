from app.grammar.validator import validate_expression


def test_valid_if_then_else_from_sample_csv():
    expr = 'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."COLLATERAL_AMT"))THEN("TRUE")ELSE("FALSE")'
    assert validate_expression(expr).valid


def test_valid_simple_function_call():
    expr = 'TODATE("FCT_NPA_PRODUCT"."PERIOD_ID")'
    assert validate_expression(expr).valid


def test_valid_nested_datediff_translation_of_dpd_overdue():
    expr = (
        'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."OverDueSinceDt"))'
        'THEN(DATEDIFF("FCT_NPA_PRODUCT"."var"."BUSINESS_DATE",'
        '"FCT_NPA_PRODUCT"."OverDueSinceDt","d")+1)ELSE(0)'
    )
    assert validate_expression(expr).valid


def test_valid_and_or_not_operators():
    expr = 'IF(OR(NOT(ISEMPTY("A"."X")),"A"."Y" IN ["a","b"]))THEN(1)ELSE(0)'
    assert validate_expression(expr).valid


def test_valid_between_operator():
    expr = 'IF("A"."DPD" BETWEEN [1,30])THEN("SMA0")ELSE("STANDARD")'
    assert validate_expression(expr).valid


def test_rejects_unknown_function():
    result = validate_expression('IF(BOGUSFUNC("A"."X"))THEN(1)ELSE(0)')
    assert not result.valid
    assert "BOGUSFUNC" in result.error


def test_rejects_malformed_syntax():
    result = validate_expression('IF(ISEMPTY("A"."X")THEN(1)ELSE(0)')
    assert not result.valid


def test_and_or_not_are_not_treated_as_unknown_functions():
    # regression test for the grammar-ambiguity bug found during development
    result = validate_expression('IF(AND("A"."X">0,"A"."Y" BETWEEN [1,30]))THEN("YES")ELSE("NO")')
    assert result.valid
