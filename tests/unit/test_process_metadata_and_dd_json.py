"""Tests for At-a-Glance metadata and Decision Table / Conditional JSON."""
from __future__ import annotations

import json

from app.derivation.dd_generation_engine import (
    _condition_links_from_guard,
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


def _link_summary(guard: str, entity: str = "T"):
    return [
        (l["name"], l["columnName"], l["operator"], l["value"], l["type"])
        for l in _condition_links_from_guard(guard, entity)
    ]


def test_and_call_and_infix_and_produce_identical_links():
    infix = '"T"."A" == "Y" AND "T"."B" > 5 AND ISNOTEMPTY("T"."C")'
    call = 'AND("T"."A" == "Y","T"."B" > 5,ISNOTEMPTY("T"."C"))'
    nested = 'AND("T"."A" == "Y",AND("T"."B" > 5,ISNOTEMPTY("T"."C")))'
    expected = [
        ("A", "A", "==", "Y", "ENT"),
        ("B", "B", ">", "5", "ENT"),
        ("C", "C", "ISNOTEMPTY", "", "ENT"),
    ]
    assert _link_summary(infix) == expected
    assert _link_summary(call) == expected
    assert _link_summary(nested) == expected


def test_parenthesised_and_groups_are_flattened_not_left_as_one_garbled_link():
    guard = (
        '("D"."FromKey" <= p_TIMEKEY AND "D"."ToKey" >= p_TIMEKEY '
        'AND COALESCE("D"."Scheme", "N") == "Y") AND ("F"."Id" == 7)'
    )
    links = _link_summary(guard)
    assert [l[1] for l in links] == ["FromKey", "ToKey", "Scheme", "Id"]
    # No link may swallow the rest of the condition text into its value.
    assert all("AND" not in l[3] and "(" not in l[3] for l in links)


def test_or_group_stays_one_intact_link_and_does_not_split_the_and():
    guard = 'AND("T"."A" == "Y",OR(ISEMPTY("T"."B"),"T"."B" == "N"))'
    links = _condition_links_from_guard(guard, "T")
    assert [l["operator"] for l in links] == ["==", "EXPR"]
    assert links[1]["value"] == 'OR(ISEMPTY("T"."B"),"T"."B" == "N")'
    # Same for the infix spelling of the OR.
    infix = _condition_links_from_guard('"T"."A" == "Y" AND (ISEMPTY("T"."B") OR "T"."B" == "N")', "T")
    assert [l["operator"] for l in infix] == ["==", "EXPR"]


def test_link_name_and_type_follow_platform_conventions():
    links = _link_summary('AND("E"."Direct" == 1,("E"."var"."Tmp") == 2,("E"."FK_AGG"."MaxDpd") > 0)', "E")
    assert links == [
        ("Direct", "Direct", "==", "1", "ENT"),
        ("var", "Tmp", "==", "2", "TEMP"),
        ("FK_AGG", "MaxDpd", ">", "0", "REL"),
    ]


def test_computed_operand_becomes_expr_link_instead_of_a_wrong_column():
    links = _condition_links_from_guard('("E"."End" - "E"."Start") >= 90', "E")
    assert len(links) == 1
    assert links[0]["operator"] == "EXPR"
    assert links[0]["columnName"] == ""


def test_conditional_json_for_and_call_formula_has_one_link_per_condition():
    raw = _conditional_json_from_formula(
        'IF(AND("A"."X" > 1,"A"."Y" == "Y",ISEMPTY("A"."Z")))THEN(1)ELSE(0)', "A"
    )
    links = json.loads(raw)["conditionalDetails"][0]["conditionalLinksInfo"]
    assert [l["operator"] for l in links] == [">", "==", "ISEMPTY"]


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
