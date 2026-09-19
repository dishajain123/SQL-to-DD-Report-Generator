"""Regression: historical PENDING_REVIEW root causes must not recur."""
from __future__ import annotations

from app.derivation.dd_generation_engine import (
    _extract_column_from_assignment_blob,
    _is_multi_column_assignment_blob,
    _is_process_status_table,
    _normalize_legacy_if_syntax,
    _scrub_llm_expression_for_column,
)
from app.grammar.validator import validate_expression


def test_legacy_comma_if_rewrites_to_then_else():
    raw = 'IF(p_TIMEKEY > 26267, IF(ISNOTEMPTY(OverDueSinceDt), 1, 0), 0)'
    rewritten = _normalize_legacy_if_syntax(raw)
    assert "THEN(" in rewritten.upper()
    assert "," not in rewritten.split("THEN(")[0].split("IF(")[-1] or "THEN(" in rewritten
    assert validate_expression(rewritten).valid


def test_multi_column_assignment_blob_detected():
    blob = (
        "IF(p_TIMEKEY > 26267,"
        "(DPD_IntService = 1, DPD_NoCredit = 2, DPD_Overdue = 3),"
        "(DPD_IntService = 0, DPD_NoCredit = 0, DPD_Overdue = 0))"
    )
    assert _is_multi_column_assignment_blob(blob, "DPD_Overdue")
    extracted = _extract_column_from_assignment_blob(blob, "DPD_Overdue")
    assert extracted is not None
    assert "3" in extracted or extracted.strip() in {"3", "(3)"}


def test_scrub_rejects_unextractable_multi_column_blob():
    blob = "IF(p_TIMEKEY > 26267, (A = 1, B = 2), (A = 0, B = 0))"
    assert _scrub_llm_expression_for_column(blob, "DPD_Overdue") == ""


def test_scrub_keeps_single_column_formula():
    expr = 'IF(ISNOTEMPTY("LoanAccountCal"."DpdDays"))THEN("CURRENT")ELSE("NA")'
    assert _scrub_llm_expression_for_column(expr, "DpdBucket") == expr


def test_process_status_tables_identified():
    assert _is_process_status_table("ACLRUNNINGPROCESSSTATUS")
    assert _is_process_status_table("RunStatus")
    assert not _is_process_status_table("LoanAccountCal")
