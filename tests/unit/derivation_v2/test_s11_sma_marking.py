"""Regressions for PRO SMA_MARKING (S11): session-temp copies must not pollute root columns."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_compiler import data_type_from_column_name
from app.derivation.v2.ast_optimize import _drop_identical_condition_elseif_arms
from app.derivation.v2.phase3_ast_generator import (
    _is_cross_column_null_default_guard,
    parse_sql_expression_to_ast,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.parsing.write_inventory_scan import read_sql_file

_REPO = Path(__file__).resolve().parents[3]
_S11 = _REPO / "samples/sql/PRO_SPs_Sequenced/19_S11_PRO.SMA_MARKING.StoredProcedure.sql"


def _formula(entity: str, column: str) -> str:
    _, debug = generate_for_sql(read_sql_file(_S11), entity, column, llm_client=None)
    return debug.get("formula") or ""


def test_root_columns_ignore_copies_made_into_unwritten_session_temps():
    # ``SELECT … INTO #DPD`` (``0 AS DPD_Renewal``) and ``UPDATE #DPD`` clamps are not
    # writes to ##AccountCal.DPD_*; only the #DPD_Aqua_SMA write-back is.
    renewal = _formula("AccountCal", "DPD_Renewal")
    assert "DPD_Aqua_SMA" in renewal, renewal[:400]
    assert "ELSE(0)" not in renewal.replace(" ", ""), renewal[:600]
    assert "DPD_Overdue" not in renewal, renewal[:600]
    overdrawn = _formula("AccountCal", "DPD_Overdrawn")
    assert "DPD_Aqua_SMA" in overdrawn and "DPD_Overdue" not in overdrawn, overdrawn[:600]


def test_pass_through_copy_into_temp_adds_no_arm_to_conti_excess_dt():
    formula = _formula("AccountCal", "ContiExcessDt")
    assert "DPD_Aqua_SMA" in formula
    assert "DPD_Overdue" not in formula, formula[:600]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_sma_class_keeps_the_dpd_max_case_and_ignores_smaclass_temp():
    formula = _formula("AccountCal", "SMA_CLASS")
    assert "SMA_0" in formula and "SMA_1" in formula and "SMA_2" in formula
    assert "DPD_MAX" in formula.upper(), formula[:600]
    assert "THEN(1)" not in formula.replace(" ", ""), formula[:600]


def test_flgsma_business_guard_is_not_dropped_as_a_null_default_fill():
    formula = _formula("AccountCal", "FLGSMA")
    assert formula and formula.strip() != "NULL"
    assert '"Y"' in formula and "DPD_MAX" in formula.upper(), formula[:600]


def test_business_guard_with_non_null_predicates_is_not_a_null_default():
    cond = (
        "(ISNULL(B.FLGPROCESSING,'N')='N') AND (ISNULL(FINALASSETCLASSALT_KEY,1)=1) "
        "AND (ISNULL(A.BALANCE,0)>0) AND (A.ASSET_NORM<>'ALWYS_STD')"
    )
    assert not _is_cross_column_null_default_guard(cond, "FLGSMA")
    assert _is_cross_column_null_default_guard("ISNULL(Sibling,0)=0 AND Other IS NULL", "FLGSMA")


def test_choose_is_translated_to_an_index_if_chain():
    ast = parse_sql_expression_to_ast(
        "ISNULL(A.SMA_CLASS,CHOOSE(B.SMA_CLASS_KEY,'SMA_0','SMA_1','SMA_2'))",
        default_entity="SMACLASS",
        target_column="SMA_CLASS",
    )
    text = repr(ast)
    assert "__UNSUPPORTED_SQL__" not in text
    assert text.count("IF_THEN_ELSE") == 3


def test_s11_smaclass_temp_column_is_derived():
    formula = _formula("SMACLASS", "SMA_CLASS")
    assert formula and "SMA_0" in formula, formula[:400]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_dpd_and_reference_period_columns_are_integers():
    for name in ("DPD_Overdrawn", "DPD_Renewal", "DPD_MAX", "RefPeriodIntService", "RefPeriodOverDrawn"):
        assert data_type_from_column_name(name) == "Integer", name
    assert data_type_from_column_name("DPD_Breach_Date") == "Date"


def test_case_equals_literal_folds_to_or_of_arm_guards():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
    from app.derivation.v2.ast_optimize import _fold_boolean_literals

    col = lambda n: {"type": "COLUMN_REF", "entity": "A", "column": n}
    lit = lambda v: {"type": "LITERAL", "value_type": "NUMBER", "value": v}
    null = {"type": "LITERAL", "value_type": "NULL", "value": None}
    eq = lambda left, right: {"type": "BINARY_OP", "operator": "==", "left": left, "right": right}
    c1 = eq(col("X"), lit(1))
    c2 = eq(col("Y"), lit(2))
    # (CASE WHEN c1 THEN 1 WHEN c2 THEN 1 END) = 1, after comparison distribution
    distributed = {
        "type": "IF_THEN_ELSE", "condition": c1, "then_branch": eq(lit(1), lit(1)),
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": c2, "then_branch": eq(lit(1), lit(1)),
            "else_branch": eq(null, lit(1)),
        },
    }
    out = _fold_boolean_literals(distributed)
    assert out["type"] == "BINARY_OP" and out["operator"] == "OR"
    text = compile_ast_to_4x_string(out)
    assert text.startswith("OR(") and "IF(" not in text, text
    # Literal comparisons are decided statically; AND/OR absorb the constants.
    assert _fold_boolean_literals(eq(lit(49999), lit(49999)))["value"] is True
    assert _fold_boolean_literals(eq(lit(1), lit(2)))["value"] is False
    kept = {"type": "BINARY_OP", "operator": "AND", "left": eq(lit(1), lit(1)), "right": c1}
    assert _fold_boolean_literals(kept) == c1


def test_multi_operand_string_plus_is_one_concat_not_concat_plus_column():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string

    ast = parse_sql_expression_to_ast(
        "'Link By AccountId' + ' ' + B.CustomerAcID",
        default_entity="AccountCal",
        target_column="NPA_Reason",
    )
    assert ast["type"] == "FUNCTION_CALL" and ast["function_name"] == "CONCAT"
    assert len(ast["arguments"]) == 3
    text = compile_ast_to_4x_string(ast)
    assert text.startswith("CONCAT(") and ") +" not in text and "+ " not in text, text


def test_append_style_arm_is_not_hoisted_over_arms_between_it_and_its_twin():
    from app.derivation.v2.ast_optimize import _collapse_duplicate_if_then_arms

    col = lambda n: {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "column": n}
    lit = lambda v: {"type": "LITERAL", "value_type": "STRING", "value": v}
    cond = lambda n: {"type": "BINARY_OP", "operator": "==", "left": col(n), "right": lit("Y")}
    append = {
        "type": "FUNCTION_CALL", "function_name": "CONCAT",
        "arguments": [{"type": "FUNCTION_CALL", "function_name": "COALESCE",
                       "arguments": [col("NPA_Reason"), lit("")]}, lit(" x")],
    }
    chain = {
        "type": "IF_THEN_ELSE", "condition": cond("A"), "then_branch": append,
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": cond("B"), "then_branch": lit("override"),
            "else_branch": {
                "type": "IF_THEN_ELSE", "condition": cond("C"), "then_branch": append,
                "else_branch": col("NPA_Reason"),
            },
        },
    }
    out = _collapse_duplicate_if_then_arms(chain)
    text = repr(out)
    # The low-priority twin (C) must stay below the "override" arm, not merge into arm A.
    assert text.count("'override'") == 1
    assert "'C'" in text and text.index("'override'") < text.index("'C'")
    # Adjacent duplicates (not self-reading values) still merge as before.
    plain = {
        "type": "IF_THEN_ELSE", "condition": cond("A"), "then_branch": lit("v"),
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": cond("B"), "then_branch": lit("v"),
            "else_branch": col("NPA_Reason"),
        },
    }
    merged = _collapse_duplicate_if_then_arms(plain)
    assert merged["condition"]["operator"] == "OR"


def test_alwys_npa_boost_applies_to_date_values_not_reason_strings():
    from app.derivation.v2.ast_optimize import _heuristic_arm_outer_priority

    col = {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "column": "ASSET_NORM"}
    cond = {
        "type": "BINARY_OP", "operator": "==", "left": col,
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "ALWYS_NPA"},
    }
    process_date = {"type": "VARIABLE_REF", "name": "@ProcessDate"}
    reason = {"type": "LITERAL", "value_type": "STRING", "value": "Degarde Account"}
    assert _heuristic_arm_outer_priority(cond, process_date) >= 1_000_000
    assert _heuristic_arm_outer_priority(cond, reason) == 0


def test_guard_signature_is_total_over_null_and_string_literals():
    from app.derivation.v2.phase2_mutation_folder import guard_formula_signature

    col = {"type": "COLUMN_REF", "entity": "A", "column": "X"}
    cmp_ = lambda value, vtype: {
        "type": "BINARY_OP", "operator": "==", "left": col,
        "right": {"type": "LITERAL", "value_type": vtype, "value": value},
    }
    both = {
        "type": "BINARY_OP", "operator": "AND",
        "left": cmp_(None, "NULL"), "right": cmp_("abc", "STRING"),
    }
    swapped = {"type": "BINARY_OP", "operator": "AND", "left": both["right"], "right": both["left"]}
    # Must not raise (``None < 'abc'``) and must stay order-independent.
    assert guard_formula_signature(both) == guard_formula_signature(swapped)


def test_if_inside_a_quoted_identifier_is_not_a_control_flow_branch():
    from app.derivation.v2.sql_text import extract_if_else_chains

    sql = (
        "IF OBJECT_ID('TEMPDB..#TEMPTABLE_SMACLASSUcif') IS NOT NULL\n"
        "   DROP TABLE #TEMPTABLE_SMACLASSUcif\n"
        "SELECT A.UCIF_ID, MIN(A.SMA_Dt) AS SMA_Dt INTO #TEMPTABLE_SMACLASSUcif\n"
        "FROM ##AccountCal A INNER JOIN ##CUSTOMERCAL B ON A.UCIF_ID=B.UCIF_ID AND B.FLGSMA='Y'\n"
        "GROUP BY A.UCIF_ID\n"
        "UPDATE A SET A.SMA_DT=B.SMA_Dt FROM ##CUSTOMERCAL A INNER JOIN #TEMPTABLE_SMACLASSUcif B "
        "ON A.UCIF_ID=B.UCIF_ID WHERE A.FLGSMA='Y'\n"
    )
    assert extract_if_else_chains(sql) == []


def test_customer_level_sma_columns_have_no_invented_branch_condition():
    sql = read_sql_file(_S11)
    from app.derivation.v2.phase1_lineage import build_lineage_map
    from app.derivation.v2.phase2_mutation_folder import fold_column_mutations

    for column in ("SMA_DT", "SMA_CLASS_KEY"):
        muts = fold_column_mutations(sql, "CUSTOMERCAL", column, build_lineage_map(sql, None), None)
        assert muts, column
        for mut in muts:
            assert not re.search(r"(?is)\bFROM\b|\bJOIN\b", mut.outer_condition or ""), (
                column, mut.outer_condition,
            )
        formula = _formula("CUSTOMERCAL", column)
        assert formula and "Untranslated" not in formula, column


def test_boolean_valued_if_becomes_and_or_not_a_bare_literal():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
    from app.derivation.v2.ast_optimize import _fold_boolean_literals

    col = lambda n: {"type": "COLUMN_REF", "entity": "A", "column": n}
    lit = lambda v: {"type": "LITERAL", "value_type": "STRING", "value": v}
    eq = lambda n, v: {"type": "BINARY_OP", "operator": "==", "left": col(n), "right": lit(v)}
    # ``(IF(g) THEN a ELSE b) == 'SMA_0'`` distributes to a boolean-valued IF; the compiler's
    # value-slot repair would otherwise keep only the right operand ("SMA_0").
    boolean_if = {
        "type": "IF_THEN_ELSE", "condition": eq("G", "Y"),
        "then_branch": eq("P", "SMA_0"), "else_branch": eq("Q", "SMA_0"),
    }
    out = _fold_boolean_literals(boolean_if)
    text = compile_ast_to_4x_string(out)
    assert text.startswith("OR(AND(") and '"A"."P" == "SMA_0"' in text and '"A"."Q" == "SMA_0"' in text, text
    assert "NOT(" in text


def test_compiler_drops_trailing_arm_that_matches_the_else_text():
    from app.derivation.v2.ast_compiler import compile_ast_to_4x_string

    cond = lambda n: {
        "type": "BINARY_OP", "operator": "==",
        "left": {"type": "COLUMN_REF", "entity": "A", "column": n},
        "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    # Same printed value, different tree (hop to itself vs bare column).
    hop = {"type": "COLUMN_REF", "entity": "A", "relationship": "A", "column": "X"}
    bare = {"type": "COLUMN_REF", "entity": "A", "column": "X"}
    other = {"type": "COLUMN_REF", "entity": "B", "column": "X"}
    degenerate = {"type": "IF_THEN_ELSE", "condition": cond("F"), "then_branch": hop, "else_branch": bare}
    assert compile_ast_to_4x_string(degenerate) == '"A"."X"'
    chain = {
        "type": "IF_THEN_ELSE", "condition": cond("G"), "then_branch": other,
        "else_branch": degenerate,
    }
    assert compile_ast_to_4x_string(chain) == 'IF("A"."G" == "Y")THEN("B"."X")ELSE("A"."X")'


def test_self_hop_reference_counts_as_the_columns_own_value():
    from app.derivation.v2.phase3_ast_generator import _is_self_column_ref

    hop = {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "relationship": "##ACCOUNTCAL", "column": "X"}
    other = {"type": "COLUMN_REF", "entity": "ACCOUNTCAL", "relationship": "CUSTOMERCAL", "column": "X"}
    assert _is_self_column_ref(hop, "AccountCal", "x")
    assert not _is_self_column_ref(other, "AccountCal", "x")


def test_elseif_arm_implied_by_an_earlier_guard_is_dropped():
    col = lambda n: {"type": "COLUMN_REF", "entity": "A", "column": n}
    eq = lambda n, v: {
        "type": "BINARY_OP", "operator": "==", "left": col(n),
        "right": {"type": "LITERAL", "value_type": "STRING", "value": v},
    }
    join = eq("K", "1")
    both = {"type": "BINARY_OP", "operator": "AND", "left": eq("X", "a"), "right": join}
    lit = lambda v: {"type": "LITERAL", "value_type": "STRING", "value": v}
    chain = {
        "type": "IF_THEN_ELSE", "condition": join, "then_branch": lit("first"),
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": both, "then_branch": lit("shadowed"),
            "else_branch": col("Z"),
        },
    }
    out = _drop_identical_condition_elseif_arms(chain)
    assert out["else_branch"] == col("Z"), out
    assert "shadowed" not in repr(out)
    # A narrower earlier guard does NOT shadow a broader later one.
    reverse = {
        "type": "IF_THEN_ELSE", "condition": both, "then_branch": lit("first"),
        "else_branch": {
            "type": "IF_THEN_ELSE", "condition": join, "then_branch": lit("second"),
            "else_branch": col("Z"),
        },
    }
    assert "second" in repr(_drop_identical_condition_elseif_arms(reverse))
    assert re.search(r"IF_THEN_ELSE", repr(out))
