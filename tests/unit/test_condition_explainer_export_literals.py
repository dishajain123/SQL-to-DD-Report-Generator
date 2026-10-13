"""DD export explanations must preserve Table.Column names and flag literals."""
from __future__ import annotations

from app.report.condition_explainer import build_dynamic_rule_sentence


def test_export_coalesce_flg_processing_phrasing():
    formula = (
        'IF(AND(COALESCE("CUSTOMERCAL"."FlgProcessing", "N") == "N", '
        '"ACCOUNTCAL"."refcustomerid" == "CUSTOMERCAL"."refcustomerid"))'
        'THEN("ACCOUNTCAL"."refcustomerid")ELSE("X")'
    )
    text = build_dynamic_rule_sentence(
        formula, target_column="refcustomerid", target_entity="CTE_CustomerWiseBalance"
    )
    assert text
    assert "CUSTOMERCAL.FlgProcessing" in text
    assert "equals 'N'" in text
    assert "treating NULL as 'N'" in text
    assert "flag processing" not in text.lower()
    assert " equals no" not in text.lower()


def test_export_coalesce_balance_in_sum():
    formula = (
        'IF(COALESCE("CUSTOMERCAL"."FlgDeg", "N") == "Y")'
        'THEN(SUM(COALESCE("ACCOUNTCAL"."BALANCE", 0)))ELSE("CTE"."Balance")'
    )
    text = build_dynamic_rule_sentence(formula, target_column="Balance", target_entity="CTE")
    assert text
    assert "ACCOUNTCAL.BALANCE" in text
    assert "COALESCE(ACCOUNTCAL.BALANCE, 0)" in text or "ACCOUNTCAL.BALANCE" in text
    assert "COALESCE('ACCOUNTCAL'" not in text


def test_export_concat_degreason_human_append():
    formula = (
        'IF("ACCOUNTCAL"."DPD_INTSERVICE" >= "ACCOUNTCAL"."RefPeriodIntService")'
        'THEN(CONCAT(COALESCE("ACCOUNTCAL"."DegReason", ""), ", ", '
        '"DEGRADE BY INT NOT SERVICED"))ELSE("ACCOUNTCAL"."DegReason")'
    )
    text = build_dynamic_rule_sentence(
        formula, target_column="DegReason", target_entity="CTE_Staging"
    )
    assert text
    assert "Appends 'DEGRADE BY INT NOT SERVICED' to DegReason" in text
    assert "CONCAT(" not in text
    assert "nested conditional:" not in text.lower()


def test_export_multi_branch_no_ellipsis_truncation():
    cond = (
        "AND("
        '"ACCOUNTCAL"."DPD_INTSERVICE" >= "ACCOUNTCAL"."RefPeriodIntService",'
        '"ACCOUNTCAL"."DPD_NOCREDIT" >= "ACCOUNTCAL"."RefPeriodNoCredit",'
        '"ACCOUNTCAL"."DPD_OVERDRAWN" >= "ACCOUNTCAL"."RefPeriodOverDrawn",'
        '"ACCOUNTCAL"."DPD_OVERDUE" >= "ACCOUNTCAL"."RefPeriodOverdue",'
        '"ACCOUNTCAL"."DPD_RENEWAL" >= "ACCOUNTCAL"."RefPeriodReview",'
        '"ACCOUNTCAL"."DPD_STOCKSTMT" >= "ACCOUNTCAL"."RefPeriodStkStatement"'
        ")"
    )
    formula = f'IF({cond})THEN("Y")ELSE("N")'
    text = build_dynamic_rule_sentence(formula, target_column="FlgDeg", target_entity="CTE_Staging")
    assert text
    assert "..." not in text


def test_export_distinct_step_headers_for_similar_dpd_branches():
    formula = (
        'IF("ACCOUNTCAL"."DPD_INTSERVICE" >= "ACCOUNTCAL"."RefPeriodIntService")THEN("Y")'
        'ELSEIF("ACCOUNTCAL"."DPD_NOCREDIT" >= "ACCOUNTCAL"."RefPeriodNoCredit")THEN("Y")'
        'ELSE("N")'
    )
    text = build_dynamic_rule_sentence(formula, target_column="FlgDeg", target_entity="CTE_Staging")
    assert text
    assert "Servicing DPD Breaches" in text
    assert "Servicing DPD Breaches (2)" in text
    assert text.count("Priority Classification:") <= 1


def test_export_loss_dt_uses_column_names_not_yes_no():
    formula = (
        'IF(AND(COALESCE("CUSTOMERCAL"."FlgDeg", "N") == "Y", '
        'COALESCE("CUSTOMERCAL"."FlgProcessing", "N") == "N"))THEN(NULL)ELSE("CUSTOMERCAL"."LossDt")'
    )
    text = build_dynamic_rule_sentence(formula, target_column="LossDt", target_entity="CUSTOMERCAL")
    assert text
    assert "FlgDeg" in text
    assert "FlgProcessing" in text
    assert " equals 'Y'" in text or "equals 'Y'" in text
    assert " yes" not in text.lower()
    assert "flag deg" not in text.lower()
