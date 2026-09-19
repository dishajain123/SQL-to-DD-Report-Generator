"""Tests for At-a-Glance metadata and Decision Table / Conditional JSON."""
from __future__ import annotations

import json

from app.derivation.dd_generation_engine import (
    _conditional_json_from_formula,
    _decision_table_from_formula_if_categorical,
    _infer_data_type,
)
from app.models.core import Dialect, ObjectType, SQLObject, StructuralInfo
from app.report.process_metadata import build_at_a_glance_lines, extract_procedure_parameters
from app.models.core import CanonicalModel


def test_extract_timekey_parameter_from_tsql():
    sql = """
    CREATE PROCEDURE PRO.DPD_Bucket_Classification
        @TimeKey INT
    AS
    BEGIN
        SELECT 1
    END
    """
    assert extract_procedure_parameters(sql) == [("@TimeKey", "INT")]


def test_decision_table_emits_between_and_equals_links():
    expr = (
        'IF("A"."DpdDays" == 0)THEN("CURRENT")'
        'ELSEIF("A"."DpdDays" BETWEEN [1,30])THEN("BUCKET_1_30")'
        'ELSEIF("A"."DpdDays" BETWEEN [31,60])THEN("BUCKET_31_60")'
        'ELSE("BUCKET_90_PLUS")'
    )
    payload = _decision_table_from_formula_if_categorical(expr, "FCT_LOAN_ACCOUNT", "DpdBucket")
    assert payload is not None
    details = payload["decisionTableDetails"]
    assert details[0]["conditionalLinksInfo"][0]["operator"] == "=="
    assert details[0]["conditionalLinksInfo"][0]["value"] == "0"
    assert details[1]["conditionalLinksInfo"][0]["operator"] == "BETWEEN"
    assert details[1]["conditionalLinksInfo"][0]["rangeFrom"] == "1"
    assert details[1]["conditionalLinksInfo"][0]["rangeTo"] == "30"


def test_conditional_json_for_formula_rows():
    raw = _conditional_json_from_formula('IF("A"."X">1 AND "A"."Y"=="Y")THEN(1)ELSE(0)', "A")
    assert raw is not None
    payload = json.loads(raw)
    links = payload["conditionalDetails"][0]["conditionalLinksInfo"]
    assert {link["operator"] for link in links} == {">", "=="}


def test_infer_data_type_uses_expression_and_name():
    assert _infer_data_type("ClassificationDate", 'TODATE("X")') == "datetime"
    assert _infer_data_type("NpaFlag", 'IF(1)THEN("Y")ELSE("N")') == "string"
    assert _infer_data_type("LateFeeAmount", "ROUND(X,2)") == "number"
    assert _infer_data_type("AssetClass", 'IF(1)THEN("STANDARD")ELSE("LOSS")') == "string"


def test_at_a_glance_table_from_real_metadata():
    obj = SQLObject(
        object_id="o1",
        name="DPD_Bucket_Classification",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.SQLSERVER,
        raw_sql="CREATE PROCEDURE PRO.DPD_Bucket_Classification @TimeKey INT AS BEGIN SELECT 1 END",
        source_file="07.sql",
    )
    info = StructuralInfo(
        object_id="o1",
        tables_read=["LoanAccountCal", "HistoryLog"],
        tables_written=["LoanAccountCal", "RunStatus"],
        columns_written=["DpdBucket"],
        columns_written_by_table={"LoanAccountCal": ["DpdBucket"]},
        called_objects=[],
        has_dynamic_sql=False,
        confidence=0.9,
    )
    model = CanonicalModel(
        chain_id="c1",
        job_id="j1",
        object_ids=["o1"],
        technical_summary="tech",
        business_summary="biz",
        evidence=["07.sql"],
    )
    lines = build_at_a_glance_lines([model], {"o1": obj}, {"o1": info}, business_rule_count=9)
    text = "\n".join(lines)
    assert "## At a Glance" in text
    assert "`PRO.DPD_Bucket_Classification`" in text
    assert "T-SQL" in text
    assert "`@TimeKey` (INT)" in text
    assert "| Business rules | 9 |" in text
    assert "Yes — records audit events" in text
