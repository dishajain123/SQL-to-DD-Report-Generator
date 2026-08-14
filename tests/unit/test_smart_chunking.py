from app.models.core import Dialect, ObjectType, SQLObject
from app.parsing.object_splitter import split_objects
from app.parsing.sql_parser import parse_statement
from app.parsing.structural_analysis import analyze_object


def test_smart_chunking_preserves_if_else_block_and_sequential_statement():
    sql = """
    IF p_flag = 'Y' THEN
        UPDATE T1 SET col_a = 1;
    ELSE
        UPDATE T1 SET col_a = 0;
    END IF;
    UPDATE T1 SET col_b = 2;
    """
    obj = SQLObject(
        object_id="obj-1",
        name="demo",
        object_type=ObjectType.PROCEDURE,
        dialect=Dialect.ORACLE,
        raw_sql=sql,
        source_file="demo.sql",
    )

    info = analyze_object(obj)

    assert len(info.smart_chunks) == 2
    assert info.smart_chunks[0].contains_control_flow
    assert "IF p_flag = 'Y'" in info.smart_chunks[0].raw_sql
    assert "UPDATE T1 SET col_b = 2" in info.smart_chunks[1].raw_sql
    assert not info.smart_chunks[1].contains_control_flow


def test_parse_statement_extracts_join_metadata():
    stmt = "SELECT a.id, b.name FROM A a LEFT JOIN B b ON a.id = b.id"
    info = parse_statement(stmt, 0, Dialect.ORACLE)

    assert info.parsed_ok
    assert "B" in info.join_tables
    assert info.join_conditions


def test_real_proc_produces_multiple_logical_chunks(dpd_calculation_sql):
    objs = split_objects(dpd_calculation_sql, "PRO_DPD_Calculation_StoredProcedure_2.sql", Dialect.ORACLE)
    info = analyze_object(objs[0])

    assert len(info.smart_chunks) > 1
    assert any(chunk.contains_control_flow for chunk in info.smart_chunks)
    assert any("DPD_Overdue" in chunk.raw_sql for chunk in info.smart_chunks)
