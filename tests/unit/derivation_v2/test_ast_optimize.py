"""AST CSE / duplicate ELSEIF-arm folding."""
from pathlib import Path

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import (
    FORMULA_CHAR_BUDGET,
    _dedupe_and_guard_tree,
    enforce_formula_budget,
    enforce_later_update_precedence,
    optimize_expression_ast,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression


def _deep_backbone() -> dict:
    return {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "Asset_Norm"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "ALWYS_NPA"},
        },
        "then_branch": {"type": "LITERAL", "value_type": "DATE", "value": "2020-01-01"},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": {
                "type": "BINARY_OP",
                "operator": "==",
                "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "FlgProcessing"},
                "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
            },
            "then_branch": {"type": "LITERAL", "value_type": "DATE", "value": "1900-01-01"},
            "else_branch": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "FinalNpaDt"},
        },
    }


def test_rewrite_date_plus_to_addday():
    ast = {
        "type": "BINARY_OP",
        "operator": "+",
        "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "SysNPA_Dt"},
        "right": {"type": "VARIABLE_REF", "name": "@SUB_Days"},
    }
    out = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="DbtDt")
    assert out.get("type") == "FUNCTION_CALL"
    assert str(out.get("function_name") or "").upper() == "ADDDAY"


def test_unwrap_nested_identical_guard_null_then():
    guard = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgDeg"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    real = {
        "type": "FUNCTION_CALL",
        "function_name": "ADDDAY",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "SysNPA_Dt"},
            {"type": "VARIABLE_REF", "name": "@SUB_Days"},
        ],
    }
    inner = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
        "else_branch": real,
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": inner,
        "else_branch": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "DbtDt"},
    }
    out = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="DbtDt")
    assert out["then_branch"] == real


def test_drop_identical_condition_elseif_arm():
    guard = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgDeg"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": {"type": "VARIABLE_REF", "name": "@ProcessDate"},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": guard,
            "then_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
            "else_branch": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "DegDate"},
        },
    }
    out = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="DegDate")
    formula = compile_ast_to_4x_string(out, target_entity="CustomerCal", target_column="DegDate")
    assert formula.count("FlgDeg") == 1 or formula.upper().count('"FLGDEG"') == 1
    assert "ELSEIF" not in formula.upper()


def test_drop_identical_condition_coalesce_flg_processing_variant():
    flg = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {
            "type": "FUNCTION_CALL",
            "function_name": "COALESCE",
            "arguments": [
                {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgDeg"},
                {"type": "LITERAL", "value_type": "STRING", "value": "N"},
            ],
        },
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    guard_a = {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": flg,
        "right": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgProcessing"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "N"},
        },
    }
    guard_b = {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": flg,
        "right": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {
                "type": "FUNCTION_CALL",
                "function_name": "COALESCE",
                "arguments": [
                    {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgProcessing"},
                    {"type": "LITERAL", "value_type": "STRING", "value": "N"},
                ],
            },
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "N"},
        },
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard_a,
        "then_branch": {"type": "VARIABLE_REF", "name": "@ProcessDate"},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": guard_b,
            "then_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
            "else_branch": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "DegDate"},
        },
    }
    out = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="DegDate")
    formula = compile_ast_to_4x_string(out, target_entity="CustomerCal", target_column="DegDate")
    assert "ELSEIF" not in formula.upper()


def test_drop_identical_condition_prefers_process_date_over_null():
    guard = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgDeg"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": guard,
            "then_branch": {"type": "VARIABLE_REF", "name": "@ProcessDate"},
            "else_branch": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "DegDate"},
        },
    }
    out = optimize_expression_ast(ast, target_entity="CustomerCal", target_column="DegDate")
    formula = compile_ast_to_4x_string(out, target_entity="CustomerCal", target_column="DegDate")
    assert "ProcessDate" in formula or "TIMEKEY" in formula.upper()
    assert "ELSEIF" not in formula.upper()


def test_enforce_formula_budget_does_not_raise_on_unsupported_ast():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "__UNSUPPORTED_SQL__",
        "arguments": [],
        "_validation_error": "STRING_AGG is set-based aggregation with no row-level 4X equivalent",
    }
    out = enforce_formula_budget(ast, target_entity="AccountCal", target_column="DegReason")
    assert out is ast


def test_cse_replaces_duplicate_backbone_in_then_with_column_ref():
    backbone = _deep_backbone()
    guard = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {
            "type": "FUNCTION_CALL",
            "function_name": "COALESCE",
            "arguments": [backbone, {"type": "LITERAL", "value_type": "DATE", "value": "2099-12-31"}],
        },
        "right": {"type": "LITERAL", "value_type": "DATE", "value": "2000-01-01"},
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": {
            "type": "FUNCTION_CALL",
            "function_name": "MIN",
            "arguments": [
                {
                    "type": "FUNCTION_CALL",
                    "function_name": "COALESCE",
                    "arguments": [backbone, {"type": "LITERAL", "value_type": "DATE", "value": "2099-12-31"}],
                },
                {"type": "LITERAL", "value_type": "DATE", "value": "2000-01-01"},
            ],
        },
        "else_branch": backbone,
    }
    before = compile_ast_to_4x_string(ast, target_entity="AccountCal", target_column="FinalNpaDt")
    assert before.count("ALWYS_NPA") >= 2

    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="FinalNpaDt")
    after = compile_ast_to_4x_string(optimized, target_entity="AccountCal", target_column="FinalNpaDt")
    assert after.count("ALWYS_NPA") == 1
    assert '"AccountCal"."FinalNpaDt"' in after
    assert len(after) < len(before)
    assert "MIN(" not in after.upper()
    assert validate_expression(after).passed


def test_duplicate_elseif_arms_merge_to_or():
    lit = {"type": "LITERAL", "value_type": "STRING", "value": "X"}
    c1 = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "A"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "1"},
    }
    c2 = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "B"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "2"},
    }
    default = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "Col"}
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": c1,
        "then_branch": lit,
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": c2,
            "then_branch": lit,
            "else_branch": default,
        },
    }
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="Col")
    formula = compile_ast_to_4x_string(optimized, target_entity="AccountCal", target_column="Col")
    assert formula.count('THEN("X")') == 1
    assert "OR(" in formula
    assert validate_expression(formula).passed


def test_enforce_later_update_precedence_hoists_tagged_arm():
    """Earlier broad guard outermost in AST must move below a later tagged override."""
    refperiod_cond = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "FlgDeg"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    alwys_cond = {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "Asset_Norm"},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "ALWYS_NPA"},
    }
    ref_then = {
        "type": "FUNCTION_CALL",
        "function_name": "DATEADD",
        "arguments": [
            {"type": "LITERAL", "value_type": "STRING", "value": "DAY"},
            {"type": "LITERAL", "value_type": "INT", "value": 1},
            {"type": "COLUMN_REF", "entity": "AccountCal", "column": "RefPeriodNpa"},
        ],
    }
    alwys_then = {"type": "VARIABLE_REF", "name": "@ProcessDate"}
    base = {"type": "LITERAL", "value_type": "NULL", "value": None}
    # Wrong order: REFPERIOD (earlier SQL) tagged outer, ALWYS (later SQL) inner.
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": refperiod_cond,
        "then_branch": ref_then,
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": alwys_cond,
            "then_branch": alwys_then,
            "else_branch": base,
            "_source_position": 120,
            "_source_ordinal": 2,
        },
        "_source_position": 111,
        "_source_ordinal": 1,
    }
    fixed = enforce_later_update_precedence(ast)
    formula = compile_ast_to_4x_string(fixed, target_entity="AccountCal", target_column="FinalNpaDt")
    assert formula.find("ALWYS_NPA") < formula.find("FlgDeg"), formula
    assert "@ProcessDate" in formula or "ProcessDate" in formula


def test_npa_s06_final_npa_dt_formula_is_compact_and_valid():
    sql_path = (
        Path(__file__).resolve().parents[3]
        / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"
    )
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    row, debug = generate_for_sql(sql, "AccountCal", "FinalNpaDt", llm_client=None)
    formula = debug["formula"]
    assert formula
    result = validate_expression(formula)
    assert result.passed, result.errors
    assert not row.validation_errors
    idx_alwys = formula.find("ALWYS_NPA")
    idx_ref = formula.find("REFPERIOD")
    if idx_alwys >= 0 and idx_ref >= 0:
        assert idx_alwys < idx_ref
    for needle in ("ALWYS_NPA", "REFPERIODNPA", "FlgProcessing", "1900-01-01"):
        assert formula.count(needle) <= 1, f"{needle} repeated in: {formula[:800]}..."
    upper = formula.upper()
    assert "MIN(" not in upper
    assert "MAX(" not in upper
    assert "SUM(" not in upper


def test_min_wrapping_if_unwraps_to_row_level_conditional():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "MIN",
        "arguments": [
            {
                "type": "IF_THEN_ELSE",
                "condition": {
                    "type": "BINARY_OP",
                    "operator": ">",
                    "left": {"type": "COLUMN_REF", "entity": "A", "column": "X"},
                    "right": {"type": "LITERAL", "value_type": "NUMBER", "value": 1},
                },
                "then_branch": {"type": "COLUMN_REF", "entity": "B", "column": "Y"},
                "else_branch": {"type": "COLUMN_REF", "entity": "A", "column": "X"},
            }
        ],
    }
    optimized = optimize_expression_ast(ast, target_entity="A", target_column="X")
    formula = compile_ast_to_4x_string(optimized, target_entity="A", target_column="X")
    assert "MIN(" not in formula.upper()
    assert "IF(" in formula
    assert validate_expression(formula).passed


def test_s23_provision_formulas_contain_no_set_aggregates():
    sql_path = (
        Path(__file__).resolve().parents[3]
        / "samples/sql/PRO_SPs_Sequenced/31_S23_PRO.UpdationTotalProvision.StoredProcedure.sql"
    )
    sql = sql_path.read_text(encoding="utf-8", errors="replace")
    for entity, column in (
        ("AccountCal", "TotalProvision"),
        ("AccountCal", "BankTotalProvision"),
    ):
        row, debug = generate_for_sql(sql, entity, column, llm_client=None)
        formula = debug.get("formula") or ""
        if not formula:
            continue
        upper = formula.upper()
        assert "MIN(" not in upper, column
        assert "MAX(" not in upper, column
        assert "SUM(" not in upper, column
        assert "COUNT(" not in upper, column
        assert validate_expression(formula).passed, validate_expression(formula).errors


def test_count_scalar_subquery_lowers_to_literal_without_count_call():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "COUNT",
        "arguments": [{"type": "LITERAL", "value_type": "NUMBER", "value": 1}],
    }
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="Cnt")
    formula = compile_ast_to_4x_string(optimized, target_entity="AccountCal", target_column="Cnt")
    assert "COUNT(" not in formula.upper()
    assert formula.strip() == "1"
    assert validate_expression(formula).passed


def test_ternary_min_lowers_to_nested_if_without_min_call():
    ast = {
        "type": "FUNCTION_CALL",
        "function_name": "MIN",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "A", "column": "D1"},
            {"type": "COLUMN_REF", "entity": "A", "column": "D2"},
            {"type": "COLUMN_REF", "entity": "A", "column": "D3"},
        ],
    }
    optimized = optimize_expression_ast(ast, target_entity="A", target_column="D1")
    formula = compile_ast_to_4x_string(optimized, target_entity="A", target_column="D1")
    upper = formula.upper()
    assert "MIN(" not in upper
    assert "MAX(" not in upper
    assert "IF(" in formula
    assert validate_expression(formula).passed


def test_engine_rule_locks_cse_depth_and_formula_budget():
    from app.derivation.v2 import ast_optimize as opt

    assert opt._MIN_CSE_DEPTH >= 3
    assert opt._MIN_CSE_COUNT >= 2
    assert FORMULA_CHAR_BUDGET <= 7900
    assert opt.FORMULA_CHAR_BUDGET == FORMULA_CHAR_BUDGET


def _eq_col(column: str, value, value_type: str = "STRING") -> dict:
    return {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": column},
        "right": {"type": "LITERAL", "value_type": value_type, "value": value},
    }


def _and_col(left: dict, right: dict) -> dict:
    return {"type": "BINARY_OP", "operator": "AND", "left": left, "right": right}


def test_duplicate_then_or_of_equal_remainders_factors_shared_conjuncts():
    regular = {"type": "LITERAL", "value_type": "STRING", "value": "REGULAR"}
    default = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "NpaType"}
    ast = default
    for cls in ("SUB", "DB1"):
        for cd in (5, 6):
            cond = _and_col(
                _eq_col("AssetClassShortName", cls),
                _eq_col("CD", cd, "NUMBER"),
            )
            ast = {
                "type": "IF_THEN_ELSE",
                "condition": cond,
                "then_branch": regular,
                "else_branch": ast,
            }
    optimized = optimize_expression_ast(
        ast, target_entity="AccountCal", target_column="NpaType"
    )
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="NpaType"
    )
    assert formula.count('THEN("REGULAR")') == 1
    assert formula.count("AssetClassShortName") == 2
    assert "OR(" in formula
    assert "AND(" in formula
    assert "MIN(" not in formula.upper()
    assert validate_expression(formula).passed


def test_shadowed_identical_elseif_arm_is_dropped():
    guard = _and_col(_eq_col("CD", 2, "NUMBER"), _eq_col("AssetClassShortName", "SUB"))
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": guard,
        "then_branch": {"type": "LITERAL", "value_type": "STRING", "value": "STICKY"},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": guard,
            "then_branch": {"type": "LITERAL", "value_type": "STRING", "value": "MULTIPLE"},
            "else_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
        },
    }
    optimized = optimize_expression_ast(
        ast, target_entity="AccountCal", target_column="NpaType"
    )
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="NpaType"
    )
    assert 'THEN("STICKY")' in formula
    assert 'THEN("MULTIPLE")' not in formula
    assert validate_expression(formula).passed


def test_shadowed_or_disjunct_is_subtracted_from_later_arm():
    cond_a = _eq_col("CD", 2, "NUMBER")
    cond_b = _eq_col("CD", 3, "NUMBER")
    cond_c = _eq_col("CD", 0, "NUMBER")
    sticky_cond = {
        "type": "BINARY_OP",
        "operator": "OR",
        "left": cond_a,
        "right": cond_b,
    }
    multiple_cond = {
        "type": "BINARY_OP",
        "operator": "OR",
        "left": cond_a,
        "right": cond_c,
    }
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": sticky_cond,
        "then_branch": {"type": "LITERAL", "value_type": "STRING", "value": "STICKY"},
        "else_branch": {
            "type": "IF_THEN_ELSE",
            "condition": multiple_cond,
            "then_branch": {"type": "LITERAL", "value_type": "STRING", "value": "MULTIPLE"},
            "else_branch": {"type": "LITERAL", "value_type": "NULL", "value": None},
        },
    }
    optimized = optimize_expression_ast(
        ast, target_entity="AccountCal", target_column="NpaType"
    )
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="NpaType"
    )
    assert formula.find('THEN("STICKY")') < formula.find('THEN("MULTIPLE")')
    between = formula.split('THEN("STICKY")', 1)[1].split('THEN("MULTIPLE")', 1)[0]
    assert "==2" not in between.replace(" ", "")
    assert "==0" in between.replace(" ", "")
    assert validate_expression(formula).passed


def test_dedupe_and_guard_tree_drops_self_referential_equality():
    tautology = {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "CustomerEntityID"},
            "right": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "CustomerEntityID"},
        },
        "right": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "AccountCal", "column": "FlgUpg"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "U"},
        },
    }
    pruned = _dedupe_and_guard_tree(tautology)
    assert pruned.get("type") == "BINARY_OP"
    assert pruned.get("operator") == "=="
    assert pruned["left"]["column"] == "FlgUpg"

