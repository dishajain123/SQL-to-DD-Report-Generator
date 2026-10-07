"""Structural regressions for PRO UpdationTotalProvision (S23)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import (
    FORMULA_CHAR_BUDGET,
    _is_never_null,
    _strip_redundant_coalesce,
    optimize_expression_ast,
)
from app.derivation.v2.phase2_mutation_folder import _qualify_foreign_bare_columns
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression
from app.parsing.write_inventory_scan import read_sql_file

_REPO = Path(__file__).resolve().parents[3]
_S23 = (
    _REPO / "samples/sql/PRO_SPs_Sequenced/31_S23_PRO.UpdationTotalProvision.StoredProcedure.sql"
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_COALESCE_IF = re.compile(r"COALESCE\(\s*IF\s*\(", re.I)


@pytest.mark.parametrize("column", ["TotalProvision", "BankTotalProvision", "RBITotalProvision"])
def test_s23_account_cal_totals_are_valid_budgeted_and_hygienic(column):
    sql = read_sql_file(_S23)
    row, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    assert not getattr(row, "validation_errors", None), row.validation_errors
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert not _AGG_RE.search(formula), formula[:500]
    assert not _COALESCE_IF.search(formula), formula[:500]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s23_total_provision_reads_restructure_column_from_joined_table():
    sql = read_sql_file(_S23)
    row, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    assert "RestructureProvision" in formula
    assert '"AccountCal"."RestructureProvision"' not in formula


def test_s23_total_provision_is_full_composite_not_customer_rollup_identity():
    """``SELECT SUM(ISNULL(TotalProvision,0)) … INTO #TotalProvCust GROUP BY`` must not
    reset AccountCal.TotalProvision to ``COALESCE(TotalProvision, 0)``."""
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    formula = debug.get("formula") or ""
    compact = formula.replace(" ", "").upper()
    assert compact != 'COALESCE("ACCOUNTCAL"."TOTALPROVISION",0)', formula
    for needle in ("PROVSECURED", "PROVUNSECURED", "ADDLPROVISION", "PROVCOVERGOVGUR",
                   "PROVDFV", "NETBALANCE", "RESTRUCTUREPROVISION"):
        assert needle in compact, (needle, formula[:600])
    # The temp's own column is still derived from the roll-up.
    _, temp_debug = generate_for_sql(sql, "TOTALPROVCUST", "TOTALPROVISION", llm_client=None)
    assert temp_debug.get("formula")


def test_s23_total_provision_restructure_and_pui_are_additive_not_exclusive():
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    compact = (debug.get("formula") or "").replace(" ", "").upper()
    # An ELSEIF arm that is only ``TotalProvision + RestructureProvision`` would drop the
    # base provisions / RBI override for restructured accounts.
    assert (
        'COALESCE("ACCOUNTCAL"."TOTALPROVISION",0)+COALESCE("ADVACRESTRUCTURECAL"."RESTRUCTUREPROVISION",0)'
        not in compact
    )
    assert compact.count("PROVUNSECURED") >= 2, compact[:600]
    assert "PUI_CAL" in compact


def test_s23_total_provision_cap_tests_the_running_total_once():
    sql = read_sql_file(_S23)
    row, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    formula = debug.get("formula") or ""
    compact = formula.replace(" ", "").upper()
    assert not getattr(row, "validation_errors", None), row.validation_errors
    assert len(formula) <= FORMULA_CHAR_BUDGET
    # The cap guard must not test the stale input column.
    assert '"ACCOUNTCAL"."TOTALPROVISION"' not in compact, formula[:500]
    assert "NETBALANCE" in compact and "RESTRUCTUREPROVISION" in compact and "PUI_CAL" in compact
    assert validate_expression(formula).passed, validate_expression(formula).errors


@pytest.mark.parametrize("column", ["BankTotalProvision", "RBITotalProvision"])
def test_s23_bank_rbi_totals_use_standard_cap_then_floor_chain(column):
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AccountCal", column, llm_client=None)
    formula = debug.get("formula") or ""
    compact = formula.replace(" ", "")
    assert compact.startswith("IF("), formula[:200]
    # IF(sum > NetBalance) THEN NetBalance ELSEIF(sum < 0) THEN 0 ELSE sum
    assert '>"ACCOUNTCAL"."NetBalance")THEN("ACCOUNTCAL"."NetBalance")ELSEIF(' in compact, formula[:700]
    assert "<0)THEN(0)ELSE(" in compact, formula[:900]
    # No boolean-valued IF used as a condition header.
    assert "IF(IF(" not in compact, formula[:300]
    assert "ELSEIF(IF(" not in compact, formula[:900]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_s23_total_provision_is_capped_then_floored_with_inline_terms():
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AccountCal", "TotalProvision", llm_client=None)
    formula = debug.get("formula") or ""
    compact = formula.replace(" ", "")
    assert "IF(IF(" not in compact, formula[:300]
    assert "ELSEIF(IF(" not in compact, formula[:600]
    assert compact.count("THEN(0)ELSE(") >= 1 and "<0)THEN(0)" in compact, formula[:600]
    assert len(formula) <= FORMULA_CHAR_BUDGET
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_apply_deferred_clamps_standard_chain_when_floor_seen():
    from app.derivation.v2.phase3_ast_generator import _apply_deferred_clamps, _classify_total_clamp

    col = lambda n: {"type": "COLUMN_REF", "entity": "AccountCal", "column": n}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    cond = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col("T"), zero]},
        "right": col("NetBalance"),
    }
    clamp = _classify_total_clamp(cond, col("NetBalance"), "AccountCal", "T")
    floor_chain = {
        "type": "IF_THEN_ELSE",
        "condition": {"type": "BINARY_OP", "operator": "<", "left": col("S"), "right": zero},
        "then_branch": zero,
        "else_branch": col("S"),
    }
    out = _apply_deferred_clamps(floor_chain, [clamp], with_floor=True)
    assert out["condition"]["operator"] == ">" and out["condition"]["left"] == col("S")
    assert out["else_branch"]["condition"]["operator"] == "<"
    assert out["else_branch"]["else_branch"] == col("S")
    # Without a floor in the source, the cap is applied as before.
    plain = _apply_deferred_clamps(col("S"), [clamp], with_floor=False)
    assert plain["else_branch"] == col("S")


def test_apply_deferred_clamps_dedupes_identical_caps():
    from app.derivation.v2.phase3_ast_generator import (
        _apply_deferred_clamps,
        _classify_total_clamp,
    )

    col = lambda n: {"type": "COLUMN_REF", "entity": "AccountCal", "column": n}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    cond = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col("Total"), zero]},
        "right": col("NetBalance"),
    }
    clamp = _classify_total_clamp(cond, col("NetBalance"), "AccountCal", "Total")
    assert clamp is not None and clamp[0] == "cap"
    out = _apply_deferred_clamps(col("X"), [clamp, clamp])
    assert out["type"] == "IF_THEN_ELSE"
    assert out["else_branch"] == col("X")  # second identical cap not stacked


def test_arithmetic_if_operands_are_parenthesized_in_compiled_formula():
    col = lambda n: {"type": "COLUMN_REF", "entity": "AccountCal", "column": n}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    if_node = lambda c: {
        "type": "IF_THEN_ELSE",
        "condition": {"type": "BINARY_OP", "operator": "==", "left": col(c), "right": zero},
        "then_branch": col("A"),
        "else_branch": zero,
    }
    add = {"type": "BINARY_OP", "operator": "+", "left": if_node("P"), "right": if_node("Q")}
    text = compile_ast_to_4x_string(add, target_entity="AccountCal", target_column="T")
    assert text.startswith("(IF(") and ") + (IF(" in text, text


def test_provision_amount_columns_are_decimal_by_name():
    from app.derivation.v2.ast_compiler import data_type_from_column_name

    for name in ("PROVSECURED", "PROVUNSECURED", "PROVCOVERGOVGUR", "PROVDFV",
                 "RBIPROVSECURED", "BANKPROVUNSECURED"):
        assert data_type_from_column_name(name) == "Decimal", name
    # Keys / flags keep their own families.
    assert data_type_from_column_name("ProvisionAlt_Key") == "Integer"


def test_s23_adv_ac_restructure_flgdeg_reads_parameter_enum_from_dimparameter():
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AdvAcRestructureCal", "FlgDeg", llm_client=None)
    formula = debug.get("formula") or ""
    assert '"DimParameter"."ParameterShortNameEnum"' in formula, formula[:600]
    assert '"AdvAcRestructureCal"."ParameterShortNameEnum"' not in formula


def test_dim_read_evidence_qualifies_bare_lookup_column_only():
    from app.derivation.v2.phase2_mutation_folder import _augment_written_cols_with_join_temp_schemas
    from app.derivation.v2.phase1_lineage import LineageMap

    alias_map = {"A": "AdvAcRestructureCal", "D": "DimParameter"}
    written = {"ADVACRESTRUCTURECAL": {"FLGDEG"}}
    cols = _augment_written_cols_with_join_temp_schemas(
        written, alias_map, LineageMap(), {"DIMPARAMETER": {"PARAMETERSHORTNAMEENUM"}}
    )
    out = _qualify_foreign_bare_columns(
        "ParameterShortNameEnum NOT IN ('X') AND FlgDeg='Y'",
        alias_map,
        ["AdvAcRestructureCal"],
        cols,
    )
    assert "D.ParameterShortNameEnum" in out
    assert "AND FlgDeg='Y'" in out


def test_s23_addl_prov_per_has_prudential_15_and_irac_other_5_branches():
    sql = read_sql_file(_S23)
    _, debug = generate_for_sql(sql, "AdvAcRestructureCal", "AddlProvPer", llm_client=None)
    formula = debug.get("formula") or ""
    assert validate_expression(formula).passed, validate_expression(formula).errors
    assert "THEN(15)" in formula.replace(" ", ""), formula[:800]
    assert '"IRAC"' in formula and '"OTHER"' in formula, formula[:800]
    assert "PRUDENTIAL" in formula


def test_not_null_business_guard_is_not_a_null_default_fill():
    from app.derivation.v2.phase3_ast_generator import _is_cross_column_null_default_guard

    cond = "(DPD_Breach_Date IS NOT NULL AND SP_ExpiryDate >= @DATE) OR POS_10PerPaidDate IS NULL"
    assert not _is_cross_column_null_default_guard(cond, "AddlProvPer")
    assert _is_cross_column_null_default_guard("DPD_MaxFin IS NULL", "DPD_MaxNonFin")


def test_foreign_bare_column_is_qualified_only_when_joined_table_owns_it():
    alias_map = {"A": "##ACCOUNTCAL", "B": "AdvAcRestructureCal", "D": "DimParameter"}
    written = {
        "ADVACRESTRUCTURECAL": {"RESTRUCTUREPROVISION", "SECUREDPROVISION"},
        "##ACCOUNTCAL": {"TOTALPROVISION"},
    }
    text = "ISNULL(TotalProvision,0)+ISNULL(RestructureProvision,0)+ISNULL(A.X,0) + 'RestructureProvision'"
    out = _qualify_foreign_bare_columns(text, alias_map, ["##ACCOUNTCAL"], written)
    assert "ISNULL(B.RestructureProvision,0)" in out
    assert "ISNULL(TotalProvision,0)" in out
    assert "'RestructureProvision'" in out  # string literal untouched
    # No foreign evidence -> unchanged.
    assert _qualify_foreign_bare_columns(text, alias_map, ["##ACCOUNTCAL"], {}) == text


def test_redundant_coalesce_around_non_null_operand_is_stripped():
    col = {"type": "COLUMN_REF", "entity": "AccountCal", "column": "X"}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    null_safe = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col, zero]}
    wrapped = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [null_safe, zero]}
    assert _strip_redundant_coalesce(wrapped) == null_safe
    # A nullable column keeps its COALESCE.
    plain = {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col, zero]}
    assert _strip_redundant_coalesce(plain) == plain
    total = {"type": "BINARY_OP", "operator": "+", "left": null_safe, "right": null_safe}
    assert _is_never_null(total)
    assert not _is_never_null(col)


def test_clamped_sum_compared_with_column_has_no_coalesce_wrapped_if():
    col = lambda n: {"type": "COLUMN_REF", "entity": "AccountCal", "column": n}
    zero = {"type": "LITERAL", "value_type": "NUMBER", "value": 0}
    safe = lambda n: {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [col(n), zero]}
    total = {"type": "BINARY_OP", "operator": "+", "left": safe("A"), "right": safe("B")}
    clamp = {"type": "FUNCTION_CALL", "function_name": "MAX", "arguments": [total, zero]}
    guard = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {"type": "FUNCTION_CALL", "function_name": "COALESCE", "arguments": [clamp, zero]},
        "right": col("NetBalance"),
    }
    ast = {"type": "IF_THEN_ELSE", "condition": guard, "then_branch": col("NetBalance"), "else_branch": clamp}
    optimized = optimize_expression_ast(ast, target_entity="AccountCal", target_column="TotalProvision")
    formula = compile_ast_to_4x_string(
        optimized, target_entity="AccountCal", target_column="TotalProvision"
    )
    assert not _AGG_RE.search(formula), formula
    assert not _COALESCE_IF.search(formula), formula
    assert validate_expression(formula).passed, formula
