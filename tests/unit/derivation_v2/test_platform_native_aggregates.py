"""Native MIN/MAX/SUM/COUNT compile per platform docs; STRING_AGG is unsupported."""
import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import optimize_expression_ast
from app.derivation.v2.phase2_mutation_folder import _attach_groupby_to_bare_aggregate
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.platform_aggregates import (
    NATIVE_AGGREGATES_ENABLED,
    PLATFORM_AGGREGATE_FUNCTIONS,
    UNSUPPORTED_PLATFORM_AGGREGATES,
)
from app.grammar.validator import validate_expression


def test_native_aggregates_flag_enabled():
    assert NATIVE_AGGREGATES_ENABLED is True
    assert "STRING_AGG" not in PLATFORM_AGGREGATE_FUNCTIONS
    assert "STRING_AGG" in UNSUPPORTED_PLATFORM_AGGREGATES


def _group_list(entity: str, column: str) -> dict:
    return {
        "type": "LIST_LITERAL",
        "items": [{"type": "COLUMN_REF", "entity": entity, "column": column}],
    }


def test_min_with_groupby_list_compiles():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "MIN",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "AccountCal", "column": "FinalNpaDt"},
            _group_list("CustomerCal", "RefCustomerID"),
        ],
    }
    optimized = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="SysNPA_Dt")
    formula = compile_ast_to_4x_string(
        optimized, target_entity="CustomerCal", target_column="SysNPA_Dt"
    )
    assert "MIN(" in formula.upper()
    assert '"RefCustomerID"' in formula or "REFCUSTOMERID" in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_max_with_groupby_list_compiles():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "MAX",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "AccountCal", "column": "DPD_IntService"},
            _group_list("CustomerCal", "UcifEntityID"),
        ],
    }
    formula = compile_ast_to_4x_string(ast, target_entity="CustomerCal", target_column="MaxDPD")
    assert "MAX(" in formula.upper()
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_sum_compiles_without_lowering():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "SUM",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "AccountCal", "column": "NetBalance"},
        ],
    }
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="TotalNet")
    formula = compile_ast_to_4x_string(optimized, target_entity="AccountCal", target_column="TotalNet")
    assert formula.upper().startswith("SUM(")
    assert "IF(" not in formula.upper()
    assert validate_expression(formula).passed


def test_count_compiles():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "COUNT",
        "arguments": [{"type": "LITERAL", "value_type": "NUMBER", "value": 1}],
    }
    formula = compile_ast_to_4x_string(ast, target_entity="AccountCal", target_column="X")
    assert formula.upper().startswith("COUNT(")
    assert validate_expression(formula).passed


def test_count_star_parses_as_count_one():
    ast = parse_sql_expression_to_ast(
        "COUNT(*)",
        default_entity="AccountCal",
        target_column="RowCnt",
    )
    formula = compile_ast_to_4x_string(ast, target_entity="AccountCal", target_column="RowCnt")
    assert "COUNT(1)" in formula.replace(" ", "")


def test_groupby_rewrite_min_max_only():
    out = _attach_groupby_to_bare_aggregate("MIN(FinalNpaDt)", "CustomerAcID, RefCustomerID")
    assert out.upper().startswith("MIN(")
    assert "[\"CustomerAcID\"" in out or '"CustomerAcID"' in out
    assert _attach_groupby_to_bare_aggregate("SUM(NetBalance)", "CustomerAcID") == "SUM(NetBalance)"


def test_string_agg_parse_and_compile_fail_closed():
    ast = parse_sql_expression_to_ast(
        "STRING_AGG(DegReason, ', ')",
        default_entity="CustomerCal",
        target_column="DegReason",
    )
    assert ast.get("function_name") == "__UNSUPPORTED_SQL__"
    with pytest.raises(ValueError, match="STRING_AGG"):
        compile_ast_to_4x_string(ast)
