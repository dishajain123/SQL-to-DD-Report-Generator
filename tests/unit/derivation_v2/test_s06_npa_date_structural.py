"""Structural regressions for PRO NPA Date Calculation (S06)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_compiler import compile_ast_to_4x_string
from app.derivation.v2.ast_optimize import (
    FORMULA_CHAR_BUDGET,
    collapse_degenerate_if_branches,
    distribute_if_over_comparisons,
)
from app.derivation.v2.pipeline import generate_for_sql
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S06 = _REPO / "samples/sql/PRO_SPs_Sequenced/11_S06_PRO.NPA_Date_Calculation.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]", re.I)


def _sql() -> str:
    return _S06.read_text(encoding="utf-8", errors="replace")


def test_final_npa_dt_customer_writeback_is_after_account_level_arms():
    """The CustomerCal.SysNPA_Dt copy must not shadow the DPD / PUI / restructure arms."""
    _, debug = generate_for_sql(_sql(), "AccountCal", "FinalNpaDt", llm_client=None)
    formula = debug.get("formula") or ""
    assert formula
    upper = formula.upper()
    assert "TEMPTABLENPA" in upper and "REFPERIODNPA" in upper
    assert '"ACCOUNTCAL"."REFPERIODNPA"' not in upper.replace(" ", "")
    assert "SYSNPA" in upper
    # Account-level dates are evaluated first; the customer roll-up copy follows them.
    idx_sys = upper.find("THEN(\"ACCOUNTCAL\".\"CUSTOMERCAL\".\"SYSNPA_DT\")")
    if idx_sys < 0:
        idx_sys = upper.rfind("SYSNPA")
    idx_temp = upper.find("TEMPTABLENPA")
    idx_pui = upper.find("PUI_CAL")
    idx_restr = upper.find("RESTR_NPA")
    assert idx_sys >= 0 and idx_temp >= 0
    assert idx_temp < idx_sys, "DPD #TEMPTABLENPA arm must precede customer SysNPA write-back"
    if idx_pui >= 0:
        assert idx_pui < idx_sys, "PUI arm must precede customer SysNPA write-back"
    if idx_restr >= 0:
        assert idx_restr < idx_sys, "RESTR_NPA arm must precede customer SysNPA write-back"


def test_final_npa_dt_alwys_npa_precedes_refperiod_guard():
    _, debug = generate_for_sql(_sql(), "AccountCal", "FinalNpaDt", llm_client=None)
    formula = debug.get("formula") or ""
    assert formula
    assert validate_expression(formula).passed, validate_expression(formula).errors
    idx_alwys = formula.find("ALWYS_NPA")
    idx_ref = formula.find("REFPERIOD")
    assert idx_alwys >= 0 and idx_ref >= 0, formula[:500]
    assert idx_alwys < idx_ref, "ALWYS_NPA override must precede REFPERIOD aging in IF chain"
    upper = formula.upper()
    assert '"ACCOUNTCAL"."PUI_CAL"' not in upper.replace(" ", "")
    assert '"ACCOUNTCAL"."#TEMPTABLENPA"' not in upper.replace(" ", "")
    assert "#TEMPTABLENPA" in formula or '"TEMPTABLENPA"' in upper
    assert "REFPERIODNPA" in upper
    assert "SYSNPA" in upper


def test_customer_cal_flgdeg_no_degenerate_y_y_if():
    _, debug = generate_for_sql(_sql(), "CustomerCal", "FlgDeg", llm_client=None)
    formula = debug.get("formula") or ""
    assert formula
    assert validate_expression(formula).passed, validate_expression(formula).errors
    assert 'THEN("Y")ELSE("Y")' not in formula.replace(" ", "")


def test_adv_ac_restructure_cal_degdate_no_inline_if_in_comparisons():
    _, debug = generate_for_sql(_sql(), "AdvAcRestructureCal", "DegDate", llm_client=None)
    formula = debug.get("formula") or ""
    if not formula:
        return
    assert not _INLINE_IF_CMP.search(formula), formula[:600]
    assert validate_expression(formula).passed, validate_expression(formula).errors


def test_customer_cal_sysnpa_dt_rollup_and_pui_paths():
    _, debug = generate_for_sql(_sql(), "CustomerCal", "SysNPA_Dt", llm_client=None)
    formula = debug.get("formula") or ""
    assert formula
    assert validate_expression(formula).passed, validate_expression(formula).errors
    upper = formula.upper()
    assert not _AGG_RE.search(formula), formula[:500]
    assert "FINALNPADT" in upper.replace("_", "")
    assert "FLGPROCESSING" in upper or "FlgProcessing" in formula
    assert "PUI_CAL" in upper or "NPA_DATE" in upper
    assert '"ACCOUNTCAL"."PUI_CAL"' not in upper.replace(" ", "")


def test_collapse_degenerate_if_branches_unit():
    ast = {
        "type": "IF_THEN_ELSE",
        "condition": {
            "type": "BINARY_OP",
            "operator": "==",
            "left": {"type": "COLUMN_REF", "entity": "X", "column": "A"},
            "right": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
        },
        "then_branch": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
        "else_branch": {"type": "LITERAL", "value_type": "STRING", "value": "Y"},
    }
    out = collapse_degenerate_if_branches(ast)
    assert out == ast["then_branch"]


def test_distribute_if_over_comparisons_unit():
    ast = {
        "type": "BINARY_OP",
        "operator": ">",
        "left": {
            "type": "IF_THEN_ELSE",
            "condition": {
                "type": "BINARY_OP",
                "operator": ">",
                "left": {"type": "COLUMN_REF", "entity": "B", "column": "X"},
                "right": {"type": "COLUMN_REF", "entity": "B", "column": "Y"},
            },
            "then_branch": {"type": "COLUMN_REF", "entity": "B", "column": "DegDate"},
            "else_branch": {"type": "COLUMN_REF", "entity": "A", "column": "FinalNpaDt"},
        },
        "right": {"type": "VARIABLE_REF", "name": "@ProcessDate"},
    }
    out = distribute_if_over_comparisons(ast)
    assert out["type"] == "IF_THEN_ELSE"
    formula = compile_ast_to_4x_string(out, target_entity="A", target_column="DegDate")
    assert _INLINE_IF_CMP.search(formula) is None
    assert validate_expression(formula).passed, formula


_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)
_S06_ACCOUNT_COLS = (
    "FinalNpaDt",
    "InitialNpaDt",
    "FlgDeg",
    "DegReason",
    "Asset_Norm",
    "NPA_REASON",
)
_S06_CUSTOMER_COLS = ("SysNPA_Dt", "FlgDeg", "DegReason")


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    # The platform has no aggregate support: no column may emit MIN/MAX/SUM/COUNT.
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "DATEADD(" not in formula.upper().replace(" ", "")
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_s06_account_cal_exportable_columns_are_valid():
    sql = _sql()
    for column in _S06_ACCOUNT_COLS:
        row, debug = generate_for_sql(sql, "##ACCOUNTCAL", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert not row.validation_errors, f"{column}: {row.validation_errors}"
        _assert_hygiene(formula, column)
        if column == "FinalNpaDt":
            assert "ADDDAY(" in formula.upper()
            assert "ALWYS_NPA" in formula
            assert "REFPERIOD" in formula.upper()
            assert "1900" in formula
            assert formula.find("ALWYS_NPA") < formula.find("REFPERIOD")
            assert "PUI" in formula.upper() or "NPA_DATE" in formula.upper()
        if column == "InitialNpaDt":
            assert "1900" in formula
            assert "NULL" in formula.upper()
        if column == "DegReason":
            assert "THEN(0)" not in formula.replace(" ", "")
        if column == "NPA_REASON":
            assert "DEFAULT_REASON" in formula.upper() or "PUI" in formula.upper()
            assert "THEN(0)" not in formula.replace(" ", "")


def test_s06_customer_cal_exportable_columns_are_valid():
    from app.derivation.dd_postprocess import should_omit_dd_row_from_presentation
    from app.report.dd_export import is_exportable_row

    sql = _sql()
    for column in _S06_CUSTOMER_COLS:
        row, debug = generate_for_sql(sql, "##CUSTOMERCAL", column, llm_client=None)
        formula = debug.get("formula") or row.display_derivation_expression or ""
        assert formula, f"{column} produced empty formula"
        assert is_exportable_row(row, should_omit_dd_row_from_presentation), column
        assert not row.validation_errors, f"{column}: {row.validation_errors}"
        _assert_hygiene(formula, column)
        if column == "FlgDeg":
            assert 'THEN("Y")ELSE("Y")' not in formula.replace(" ", "")
        if column == "SysNPA_Dt":
            assert "THEN(0)" not in formula.replace(" ", "")
            assert "FLGPROCESSING" in formula.upper()
        if column == "DegReason":
            assert "THEN(0)" not in formula.replace(" ", "")


def test_s06_restructure_and_acl_columns_are_valid():
    sql = _sql()
    row, debug = generate_for_sql(sql, "AdvAcRestructureCal", "DegDate", llm_client=None)
    formula = debug.get("formula") or row.display_derivation_expression or ""
    if formula:
        _assert_hygiene(formula, "DegDate")
        assert "RestructureDt" in formula or "PreRestructureNPA_Date" in formula
    flg, flg_debug = generate_for_sql(sql, "AdvAcRestructureCal", "FlgDeg", llm_client=None)
    flg_formula = flg_debug.get("formula") or flg.display_derivation_expression or ""
    if flg_formula:
        _assert_hygiene(flg_formula, "FlgDeg")
        assert "PRUDENTIAL" in flg_formula.upper() or "IRAC" in flg_formula.upper() or 'THEN("N")' in flg_formula
    acl, acl_debug = generate_for_sql(
        sql, "ACLRUNNINGPROCESSSTATUS", "COMPLETED", llm_client=None
    )
    acl_formula = acl_debug.get("formula") or ""
    _assert_hygiene(acl_formula, "COMPLETED")
    assert "NPA_Date_Calculation" in acl_formula
    catch = acl.exception_handler_expression or acl_debug.get("exception_handler_formula") or ""
    assert catch
    assert "NPA_Date_Calculation" in catch


def test_s06_staging_temps_are_not_business_targets():
    from app.derivation.v2.sql_text import is_staging_derivation_entity

    assert is_staging_derivation_entity("#TEMPTABLEDPD")
    assert is_staging_derivation_entity("#TEMPTABLENPA")
    assert is_staging_derivation_entity("#RESTR_NPA")
    assert not is_staging_derivation_entity("##AccountCal")
    assert not is_staging_derivation_entity("##CustomerCal")

