"""Set-based SQL → row-level 4X lowering (windows, STRING_AGG, date UDFs)."""
from __future__ import annotations

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import optimize_expression_ast
from app.derivation.v2.phase3_ast_generator import parse_sql_expression_to_ast
from app.derivation.v2.sql_text import strip_debug_print_statements
from app.grammar.validator import validate_expression


def test_row_number_over_rank_one_lowers_to_max_equality():
    ast = parse_sql_expression_to_ast(
        "ROW_NUMBER() OVER (PARTITION BY CustomerId ORDER BY Balance DESC) = 1",
        default_entity="AccountCal",
        target_column="Flag",
    )
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(ast, target_entity="AccountCal", target_column="Flag"),
        target_entity="AccountCal",
        target_column="Flag",
    )
    assert "STRING_AGG" not in formula
    assert "ROW_NUMBER" not in formula
    assert "MAX(" in formula.upper() or "IF(" in formula
    assert validate_expression(formula).valid


def test_string_agg_lowers_to_concat_not_unsupported():
    ast = parse_sql_expression_to_ast(
        "STRING_AGG(DegReason, ', ')",
        default_entity="AccountCal",
        target_column="DegReason",
    )
    assert ast.get("function_name") != "__UNSUPPORTED_SQL__"
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(ast, target_entity="AccountCal", target_column="DegReason"),
        target_entity="AccountCal",
        target_column="DegReason",
    )
    assert "STRING_AGG" not in formula.upper()
    assert "CONCAT(" in formula.upper()
    assert validate_expression(formula).valid


def test_getminimumdate_with_entity_col_paths():
    ast = parse_sql_expression_to_ast(
        "[RBL_MISDB].PRO.GETMINIMUMDATE(ADVFACBILLDETAIL::BillDueDt,ADVFACBILLDETAIL::InterestOverdueDate,NULL)",
        default_entity="Bill_Overdue_SCF_Final",
        target_column="OverDueSinceDt",
    )
    assert ast.get("function_name") != "__UNSUPPORTED_SQL__"
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(
            ast,
            target_entity="Bill_Overdue_SCF_Final",
            target_column="OverDueSinceDt",
        ),
        target_entity="Bill_Overdue_SCF_Final",
        target_column="OverDueSinceDt",
    )
    assert "GETMINIMUMDATE" not in formula.upper()
    assert "2099-12-31" in formula
    assert validate_expression(formula).valid


def test_strip_print_removes_debug_noise():
    sql = "UPDATE T SET X = 1;\nPRINT 'PUI1';\nUPDATE T SET Y = 2;"
    cleaned = strip_debug_print_statements(sql)
    assert "PRINT" not in cleaned.upper()
    assert "UPDATE T SET Y = 2" in cleaned


def test_choose_with_entity_marker_index():
    ast = parse_sql_expression_to_ast(
        "CHOOSE(##CUSTOMERCAL::SMA_CLASS_KEY,'SMA_0','SMA_1','SMA_2')",
        default_entity="SmaClass",
        target_column="SMA_Class",
    )
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(ast, target_entity="SmaClass", target_column="SMA_Class"),
        target_entity="SmaClass",
        target_column="SMA_Class",
    )
    assert "CHOOSE" not in formula.upper()
    assert "IF(" in formula
    assert "SMA_0" in formula
    assert validate_expression(formula).valid


def test_between_range_lookup_predicate():
    ast = parse_sql_expression_to_ast(
        "AccountCal.DPD_Max BETWEEN LookupTable.LowerDPD AND LookupTable.UpperDPD",
        default_entity="AccountCal",
        target_column="ProvisionAlt_Key",
        as_condition=True,
    )
    formula = compile_ast_to_4x_string(
        optimize_expression_ast(ast, target_entity="AccountCal", target_column="ProvisionAlt_Key"),
        target_entity="AccountCal",
        target_column="ProvisionAlt_Key",
    )
    assert ">=" in formula
    assert "<=" in formula
    assert validate_expression(formula).valid


def test_dynamic_like_tight_plus_concat():
    formula = compile_ast_to_4x_string(
        parse_sql_expression_to_ast(
            "DegReason LIKE '%'+NPA_Reason+'%'",
            default_entity="AccountCal",
            target_column="DegReason",
            as_condition=True,
        )
    )
    assert "CONTAINS" in formula
    assert "NPA_Reason" in formula
    assert validate_expression(formula).valid


def test_ischanged_ladder_flattens_to_or():
    from app.derivation.v2.ast_optimize import _flatten_ischanged_mismatch_ladder

    ref_a = {"type": "COLUMN_REF", "entity": "E", "column": "A"}
    ref_h = {"type": "COLUMN_REF", "entity": "H", "column": "A"}
    yes = {"type": "LITERAL", "value_type": "STRING", "value": "Y"}
    no = {"type": "LITERAL", "value_type": "STRING", "value": "N"}

    def ne(left, right):
        return {"type": "BINARY_OP", "operator": "!=", "left": left, "right": right}

    inner = {
        "type": "IF_THEN_ELSE",
        "condition": ne(ref_a, ref_h),
        "then_branch": yes,
        "else_branch": no,
    }
    tree = {
        "type": "IF_THEN_ELSE",
        "condition": ne(
            {"type": "COLUMN_REF", "entity": "E", "column": "B"},
            {"type": "COLUMN_REF", "entity": "H", "column": "B"},
        ),
        "then_branch": yes,
        "else_branch": inner,
    }
    flat = _flatten_ischanged_mismatch_ladder(tree)
    assert flat["condition"]["operator"] == "OR"
