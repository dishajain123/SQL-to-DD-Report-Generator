"""Generated 4X must use only functions from samples/platform_docs/4x_functions_operators.md."""
import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.grammar.platform_library import DOCUMENTED_4X_FUNCTIONS, is_documented_4x_function
from app.grammar.validator import KNOWN_FUNCTIONS, validate_expression


def test_known_functions_match_platform_library():
    assert KNOWN_FUNCTIONS == DOCUMENTED_4X_FUNCTIONS
    assert "STRING_AGG" not in KNOWN_FUNCTIONS
    assert "DATE" not in KNOWN_FUNCTIONS
    assert is_documented_4x_function("TODATE")
    assert is_documented_4x_function("DATE")  # alias normalizes to TODATE


def test_validator_rewrites_date_alias_to_todate():
    expr = 'IF("A"."X"==DATE("1900-01-01"))THEN(1)ELSE(0)'
    assert validate_expression(expr).valid


def test_compiler_rejects_undocumented_function():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "BOGUSFUNC",
        "arguments": [{"type": "LITERAL", "value_type": "STRING", "value": "x"}],
    }
    with pytest.raises(ValueError, match="documented 4X library"):
        compile_ast_to_4x_string(ast)


def test_compiler_emits_todate_not_date():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "DATE",
        "arguments": [{"type": "LITERAL", "value_type": "STRING", "value": "1900-01-01"}],
    }
    formula = compile_ast_to_4x_string(ast)
    assert formula.startswith("TODATE(")
    assert "DATE(" not in formula.replace("TODATE", "")
