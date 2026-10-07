"""Structural regressions for PRO Upgrade_Customer_Account (S10)."""
from __future__ import annotations

import re
from pathlib import Path

from app.derivation.v2.ast_optimize import FORMULA_CHAR_BUDGET
from app.derivation.v2.pipeline import generate_for_sql
from app.derivation.v2.sql_text import extract_update_statements, is_staging_derivation_entity
from app.grammar.validator import validate_expression

_REPO = Path(__file__).resolve().parents[3]
_S10 = _REPO / "samples/sql/PRO_SPs_Sequenced/18_S10_PRO.Upgrade_Customer_Account.StoredProcedure.sql"
_INLINE_IF_CMP = re.compile(
    r"\(IF\s*\([^)]+\)\s*THEN\s*\([^)]+\)\s*ELSE\s*\([^)]+\)\)\s*[<>!=]",
    re.I,
)
_AGG_RE = re.compile(r"\b(MIN|MAX|SUM|COUNT)\s*\(", re.I)


def _sql() -> str:
    return _S10.read_text(encoding="utf-8", errors="replace")


def _row_formula(entity: str, column: str) -> tuple[object, str, list]:
    row, debug = generate_for_sql(_sql(), entity, column, llm_client=None)
    formula = debug.get("formula") or getattr(row, "display_derivation_expression", "") or ""
    return row, formula, debug.get("mutations") or []


def _assert_hygiene(formula: str, column: str) -> None:
    assert formula, column
    assert len(formula) <= FORMULA_CHAR_BUDGET, f"{column} length {len(formula)}"
    assert _INLINE_IF_CMP.search(formula) is None, f"{column}: {formula[:500]}"
    assert not _AGG_RE.search(formula), f"{column}: {formula[:500]}"
    assert "__UNRESOLVED_SUBQUERY_PREDICATE__" not in formula
    assert validate_expression(formula).passed, (column, validate_expression(formula).errors)


def test_unicode_space_normalization_still_finds_updates():
    raw = "UPDATE\u2002##ACCOUNTCAL\u2002SET\u2002FlgUpg='N'"
    updates = extract_update_statements(raw)
    assert len(updates) == 1
    assert "FlgUpg" in (updates[0].get("set_clause") or "")


def test_s10_customer_flgupg_under_budget_and_valid():
    row, formula, mutations = _row_formula("##CustomerCal", "FlgUpg")
    if not formula:
        row, formula, mutations = _row_formula("CustomerCal", "FlgUpg")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(formula, "FlgUpg")
    assert len(mutations) >= 3
    compact = formula.replace(" ", "")
    assert 'THEN("N")' in compact or "THEN('N')" in compact.replace("'", '"')
    assert 'THEN("U")' in compact or "THEN('U')" in compact.replace("'", '"')
    assert "ALWYS_NPA" in formula.upper() or "ALWYS_NPA" in formula
    assert "NPA" in formula.upper()
    assert "PAN" in formula.upper() or "PanNo" in formula or "PANNO" in formula.upper()


def test_s10_account_flgupg_and_final_asset_class():
    row, flg, _ = _row_formula("##AccountCal", "FlgUpg")
    if not flg:
        row, flg, _ = _row_formula("AccountCal", "FlgUpg")
    assert not getattr(row, "validation_errors", None), getattr(row, "validation_errors", None)
    _assert_hygiene(flg, "FlgUpg")
    _, final_key, _ = _row_formula("##AccountCal", "FinalAssetClassAlt_Key")
    if not final_key:
        _, final_key, _ = _row_formula("AccountCal", "FinalAssetClassAlt_Key")
    _assert_hygiene(final_key, "FinalAssetClassAlt_Key")
    assert "ALWYS_STD" in final_key.upper() or "ALWYS_STD" in flg.upper()


def test_s10_account_asset_norm_alwys_std_paths():
    _, formula, _ = _row_formula("##AccountCal", "Asset_Norm")
    if not formula:
        _, formula, _ = _row_formula("AccountCal", "Asset_Norm")
    if formula:
        _assert_hygiene(formula, "Asset_Norm")
        assert "ALWYS_STD" in formula.upper()


def test_s10_customer_sys_asset_class_perc_block_updates_present():
    _, formula, mutations = _row_formula("##CustomerCal", "SysAssetClassAlt_Key")
    if not formula:
        _, formula, mutations = _row_formula("CustomerCal", "SysAssetClassAlt_Key")
    if formula:
        _assert_hygiene(formula, "SysAssetClassAlt_Key")
    assert any("SysAssetClassAlt_Key" in str(m.get("assigned_expression", "")) for m in mutations)


def test_s10_acl_completed_try_catch():
    row, formula, _ = _row_formula("ACLRUNNINGPROCESSSTATUS", "COMPLETED")
    _assert_hygiene(formula, "COMPLETED")
    assert "Upgrade_Customer_Account" in formula
    catch = row.exception_handler_expression or ""
    assert catch
    assert "Upgrade_Customer_Account" in catch


def test_s10_staging_objects_not_export_targets():
    for name in (
        "#TEMPTABLE",
        "TEMPTABLE",
        "#CTE_PERC",
        "CTE_PERC",
        "#PANUPDATEUPGRADE",
        "#UpdateTempAssetclassAssetnorm1",
    ):
        assert is_staging_derivation_entity(name)
    assert not is_staging_derivation_entity("##AccountCal")
    assert not is_staging_derivation_entity("##CustomerCal")


def _account_row(column: str) -> tuple[str, list]:
    _, formula, mutations = _row_formula("AccountCal", column)
    if not formula:
        _, formula, mutations = _row_formula("##AccountCal", column)
    return formula, mutations


def test_s10_account_flgupg_and_upgdate_emitted_with_customer_upgrade_guard():
    flg, muts = _account_row("FlgUpg")
    assert flg, "AccountCal.FlgUpg should be exported"
    _assert_hygiene(flg, "FlgUpg")
    assert len(muts) >= 2
    upper = flg.upper()
    assert "CUSTOMERCAL" in upper or "REFCUSTOMERID" in upper
    assert 'THEN("U")' in flg.replace(" ", "") or "THEN('U')" in flg.replace("'", '"')

    upg_date, _ = _account_row("UpgDate")
    assert upg_date, "AccountCal.UpgDate should be exported"
    _assert_hygiene(upg_date, "UpgDate")
    assert "PROCESSDATE" in upg_date.upper() or "ProcessDate" in upg_date


def test_s10_account_asset_norm_fd_security_uses_cohort_left_join_not_self_account_entity():
    formula, _ = _account_row("Asset_Norm")
    if not formula:
        return
    _assert_hygiene(formula, "Asset_Norm")
    upper = formula.upper()
    assert "COHORT_NO_PERC_2" in upper or "Cohort_No_PERC_2" in formula
    compact = re.sub(r'[\s"]', "", formula).upper()
    assert "CUSTOMERENTITYID==CUSTOMERENTITYID" not in compact
    assert "ACCOUNTCAL.ACCOUNTENTITYID" not in compact.replace('"', "") or "COHORT" in upper
    if "ISEMPTY" in upper:
        assert "COHORT" in upper


def test_s10_customer_restructure_failure_reverts_flgupg_and_deg_reason():
    _, formula, mutations = _row_formula("CustomerCal", "FlgUpg")
    if not formula:
        _, formula, mutations = _row_formula("##CustomerCal", "FlgUpg")
    _assert_hygiene(formula, "CustomerCal.FlgUpg")
    assert any("RESTRUCTURE" in (m.get("raw_sql") or "").upper() for m in mutations) or (
        "RESTRUCTURE" in formula.upper() or "DEGRADE" in formula.upper()
    )

    _, deg, _ = _row_formula("CustomerCal", "DegReason")
    if not deg:
        _, deg, _ = _row_formula("##CustomerCal", "DegReason")
    if deg:
        _assert_hygiene(deg, "DegReason")
        assert "RESTRUCTURE" in deg.upper() or "DEGRADE" in deg.upper()


def test_s10_customer_flgupg_separates_temptable_ucif_and_refcustomer_temp_joins():
    _, formula, mutations = _row_formula("CustomerCal", "FlgUpg")
    if not formula:
        _, formula, mutations = _row_formula("##CustomerCal", "FlgUpg")
    _assert_hygiene(formula, "CustomerCal.FlgUpg")
    ucif_pass = [
        m
        for m in mutations
        if "TEMPTABLEREFCUSTOMERID" not in (m.get("raw_sql") or "").upper()
        and "#TEMPTABLE" in (m.get("raw_sql") or "").upper()
        and "UCIF_ID" in (m.get("raw_sql") or "").upper()
    ]
    ref_pass = [
        m
        for m in mutations
        if "TEMPTABLEREFCUSTOMERID" in (m.get("raw_sql") or "").upper()
    ]
    assert ucif_pass, "expected #TEMPTABLE / UCIF_ID upgrade pass"
    assert ref_pass, "expected #TEMPTABLERefCustomerID upgrade pass"
    compact = formula.upper().replace(" ", "")
    if "DERIVATIVEDETAIL" in compact and "UCIF" in compact:
        assert "TEMPTABLE" in compact


def test_s10_investment_flgupg_joins_temptable_not_derivative_ucif():
    _, formula, _ = _row_formula("InvestmentFinancialDetail", "FlgUpg")
    if not formula:
        return
    _assert_hygiene(formula, "InvestmentFinancialDetail.FlgUpg")
    upper = formula.upper()
    assert "TEMPTABLE" in upper
    assert "DERIVATIVEDETAIL" not in upper or "#TEMPTABLE" in formula or '"TEMPTABLE"' in upper
