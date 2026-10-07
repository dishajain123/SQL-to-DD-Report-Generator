"""Chronological UPDATE folding: shared guard + narrowing qualifier."""
from __future__ import annotations

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import optimize_expression_ast
from app.derivation.v2.phase3_ast_generator import _try_fold_narrowing_chronological_guard


def _guard_flg() -> dict:
    return {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgDeg"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
        },
        "right": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "FlgProcessing"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "N"},
        },
    }


def _los_membership() -> dict:
    return {
        "type": "BINARY_OP",
        "operator": "==",
        "left": {
            "type": "COLUMN_REF",
            "entity": "CustomerCal",
            "column": "SysAssetClassAlt_Key",
        },
        "right": {
            "type": "COLUMN_REF",
            "entity": "CustomerCal",
            "relationship": "LOS",
            "column": "AssetClassAlt_Key",
        },
    }


def test_narrowing_fold_null_los_over_addday_case():
    g = _guard_flg()
    case_val = {
        "type": "FUNCTION_CALL",
        "function_name": "ADDDAY",
        "arguments": [
            {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "SysNPA_Dt"},
            {"type": "VARIABLE_REF", "name": "@SUB_Days"},
        ],
    }
    prior = {
        "type": "IF_THEN_ELSE",
        "condition": g,
        "then_branch": case_val,
        "else_branch": {"type": "COLUMN_REF", "entity": "CustomerCal", "column": "DbtDt"},
    }
    raw = {
        "type": "BINARY_OP",
        "operator": "AND",
        "left": g,
        "right": _los_membership(),
    }
    null_lit = {"type": "LITERAL", "value_type": "NULL", "value": None}
    folded = _try_fold_narrowing_chronological_guard(
        prior, raw, null_lit, "CustomerCal", "DbtDt", 100, 3
    )
    assert folded is not None
    out = optimize_expression_ast(folded, target_entity="CustomerCal", target_column="DbtDt")
    formula = compile_ast_to_4x_string(out, target_entity="CustomerCal", target_column="DbtDt")
    assert '"LOS"' in formula
    assert "ADDDAY(" in formula.upper()
    assert formula.upper().count("THEN(NULL)") >= 1
