from app.models.core import Dialect
from app.parsing.sql_parser import classify_statement, parse_statement, split_statements


def test_split_statements_respects_nested_parens():
    sql = "UPDATE t SET x = (SELECT MAX(y) FROM u WHERE z = 1); UPDATE t2 SET a = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 2


def test_split_statements_respects_quoted_semicolons():
    sql = "UPDATE t SET note = 'a;b'; UPDATE t2 SET x = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 2
    assert "a;b" in stmts[0]


def test_split_statements_handles_block_comment_before_statement():
    sql = "/* a comment; with a semicolon */\nMERGE INTO t USING u ON (t.id = u.id) WHEN MATCHED THEN UPDATE SET t.x = 1;"
    stmts = split_statements(sql)
    assert len(stmts) == 1
    assert classify_statement(stmts[0]) == "MERGE"


def test_split_statements_handles_go_batches_and_tsql_line_boundaries():
    sql = (
        "SET ANSI_NULLS ON\nGO\n"
        "IF OBJECT_ID('TEMPDB..#DPD') IS NOT NULL\n"
        " DROP TABLE #DPD\n"
        "SELECT a, b INTO #DPD FROM dbo.t\n"
        "UPDATE #DPD SET b = 0 WHERE ISNULL(b, 0) < 0\n"
    )
    stmts = split_statements(sql, Dialect.SQLSERVER)
    assert [classify_statement(stmt) for stmt in stmts[:5]] == ["OTHER", "CONTROL_FLOW", "OTHER", "SELECT", "UPDATE"]
    assert any("SELECT a, b INTO #DPD" in stmt for stmt in stmts)


def test_classify_statement_skips_leading_block_comment():
    text = "/* note */\n   UPDATE t SET x = 1"
    assert classify_statement(text) == "UPDATE"


def test_classify_statement_control_flow():
    assert classify_statement("IF p_x > 1 THEN") == "CONTROL_FLOW"


def test_parse_statement_extracts_tables_and_columns():
    stmt = "UPDATE PRO.AccountCal_Stg SET DPD_Overdue = 0 WHERE FlgDeg = 'Y'"
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert info.parsed_ok
    assert info.tables_written == ["AccountCal_Stg"]
    assert "DPD_Overdue" in info.columns


def test_parse_statement_extracts_select_into_target_and_projection_columns():
    stmt = "SELECT a.AccountEntityID, CASE WHEN ISNULL(a.DPD_Overdrawn,0)>30 THEN 1 ELSE 0 END AS DPD_FLAG INTO #DPD FROM PRO.AccountCal a WHERE ISNULL(a.DPD_Overdrawn,0)>30"
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert "DPD" in info.tables_written
    assert any("DPD_FLAG" in cols for cols in info.set_columns_by_table.values())


def test_parse_statement_handles_cte_wrapped_update():
    stmt = "WITH x AS (SELECT 1 AS id) UPDATE t SET a = 1 FROM x WHERE t.id = x.id"
    info = parse_statement(stmt, 0, Dialect.SQLSERVER)
    assert info.parsed_ok
    assert "t" in info.tables_written or "T" in info.tables_written


def test_parse_statement_merge_extracts_target_as_written():
    stmt = (
        "MERGE INTO PRO.AccountCal_Stg A USING "
        "(SELECT id FROM PRO.Other_Table) B ON (A.id = B.id) "
        "WHEN MATCHED THEN UPDATE SET A.x = 1"
    )
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert info.parsed_ok
    assert "AccountCal_Stg" in info.tables_written
    assert "Other_Table" in info.tables_read


def test_parse_statement_flags_unparseable_sql():
    stmt = "UPDATE FROM WHERE ((("
    info = parse_statement(stmt, 0, Dialect.ORACLE)
    assert not info.parsed_ok
    assert info.parse_error is not None
