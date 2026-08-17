from app.grammar.validator import validate_expression


def test_valid_if_then_else_from_sample_csv():
    expr = 'IF(ISNOTEMPTY("FCT_NPA_PRODUCT"."COLLATERAL_AMT"))THEN("TRUE")ELSE("FALSE")'
    assert validate_expression(expr).valid


def test_valid_simple_function_call():
    expr = 'TODATE("FCT_NPA_PRODUCT"."PERIOD_ID")'
    assert validate_expression(expr).valid


def test_repairs_sql_style_date_literal():
    expr = 'IF("A"."X"==DATE"1900-01-01")THEN(1)ELSE(0)'
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
    # Missing the ELSE clause's own closing paren -- not the "condition
    # never closed before THEN" shape that _repair_missing_then_parentheses
    # exists to auto-repair, so this must still be rejected.
    result = validate_expression('IF(ISEMPTY("A"."X"))THEN(1)ELSE(0')
    assert not result.valid


def test_rejects_empty_expression():
    result = validate_expression("   ")
    assert not result.valid
    assert "Empty expression" in result.error


def test_rejects_truncated_expression_with_trailing_operator():
    result = validate_expression('IF(ISNOTEMPTY("A"."X"))THEN(("A"."Y") +')
    assert not result.valid
    assert "Unexpected end-of-input" in result.error


def test_rejects_truncated_nested_if_expression():
    result = validate_expression('IF(ISNOTEMPTY("A"."X"))THEN(IF("A"."Y">0)THEN(1)ELSE(')
    assert not result.valid
    assert "Unexpected end-of-input" in result.error


def test_rewrites_single_row_in_subquery_membership():
    expr = (
        'IF(FLGDEG=="Y" AND ISNOTEMPTY(RestructureTypeAlt_Key) '
        'AND RestructureTypeAlt_Key IN ["DimParameter"."ParameterAlt_Key" '
        'WHERE "DimParameter"."DimParameterName"=="TypeofRestructuring"])'
        'THEN("N")ELSE(NULL)'
    )
    assert validate_expression(expr).valid


def test_repairs_condition_missing_close_paren_before_then():
    # The one shape _repair_missing_then_parentheses is specifically
    # documented to fix: the IF condition's own closing paren was dropped
    # and the text flows straight into THEN. This is a mechanical typo
    # repair, not a semantic guardrail, so the repaired expression is
    # expected to validate.
    result = validate_expression('IF(ISEMPTY("A"."X")THEN(1)ELSE(0)')
    assert result.valid


def test_and_or_not_are_not_treated_as_unknown_functions():
    # regression test for the grammar-ambiguity bug found during development
    result = validate_expression('IF(AND("A"."X">0,"A"."Y" BETWEEN [1,30]))THEN("YES")ELSE("NO")')
    assert result.valid


def test_repairs_postfix_isnotempty_and_quoted_dotted_refs():
    expr = 'IF(AccountCal_Stg.RestructureTypeAlt_Key NOT IN ["PRUDENT"])THEN(1)ELSE(0)'
    assert validate_expression(expr).valid


def test_repairs_postfix_isnotempty():
    expr = 'IF("A"."X" ISNOTEMPTY)THEN(1)ELSE(0)'
    assert validate_expression(expr).valid
