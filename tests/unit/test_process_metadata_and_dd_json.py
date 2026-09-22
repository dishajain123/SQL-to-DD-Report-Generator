"""Tests for At-a-Glance process metadata (independent of DD generation)."""
from __future__ import annotations

from app.models.core import CanonicalModel, Dialect, ObjectType, SQLObject, StructuralInfo
from app.report.process_metadata import build_at_a_glance_lines, extract_procedure_parameters


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
